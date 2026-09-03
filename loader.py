"""Model loading, download, and ComfyUI/AIMDO memory management for Breeze TTS 2."""

from __future__ import annotations

import atexit
import contextlib
import gc
import importlib.metadata
import importlib.util
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import int8
from . import native
from .vendor.codec_config import Qwen3TTSTokenizerV2Config
from .vendor.codec_model import Qwen3TTSTokenizerV2Model

logger = logging.getLogger("BreezeTTS2")

MODEL_FOLDER_NAME = "breezetts2"
MIRROR_REPO_ID = "drbaph/Breeze-TTS-2-comfyui"
HF_ENDPOINT = "https://huggingface.co"

BF16_LABEL = "bf16 (best quality)"
INT8_LABEL = "int8 (smallest, slower)"
HYBRID_LABEL = "int8 hybrid (recommended)"
TE_INT8_LABEL = "int8 text encoder only"
REPO_CHOICES: dict[str, tuple[str, str]] = {
    HYBRID_LABEL: (MIRROR_REPO_ID, "Breeze-TTS-2-int8-hybrid.safetensors"),
    BF16_LABEL: (MIRROR_REPO_ID, "Breeze-TTS-2-bf16.safetensors"),
    INT8_LABEL: (MIRROR_REPO_ID, "Breeze-TTS-2-int8-convrot.safetensors"),
    TE_INT8_LABEL: (MIRROR_REPO_ID, "Breeze-TTS-2-int8-text-encoder.safetensors"),
}

DTYPE_OPTIONS = ["auto", "bf16", "fp32"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu"]
ATTENTION_OPTIONS = ["auto", "eager", "sdpa", "flash_attention", "sageattention"]
DECODE_MODE_OPTIONS = ["eager", "cuda_graphs"]

SMALL_FILE_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "audio_tokenizer/*",
]

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    import comfy.model_management as mm
    import comfy.model_patcher as model_patcher

    _ComfyCorePatcher = getattr(model_patcher, "CoreModelPatcher", None)
except Exception:
    mm = None
    model_patcher = None
    _ComfyCorePatcher = None


if model_patcher is not None:
    class _BreezeModelPatcher(model_patcher.ModelPatcher):
        # ModelPatcher.__del__ touches module globals (CallbacksMP) that are
        # already torn down when our long-lived patchers get collected during
        # interpreter shutdown. Detach happens eagerly on unload, so __del__
        # is only a fallback and must never print unraisable teardown noise.
        def __del__(self):
            try:
                super().__del__()
            except Exception:
                pass

    _DynamicBase = getattr(model_patcher, "ModelPatcherDynamic", None)
    if _DynamicBase is not None:
        class _BreezeModelPatcherDynamic(_DynamicBase):
            def __del__(self):
                try:
                    super().__del__()
                except Exception:
                    pass
    else:
        _BreezeModelPatcherDynamic = None
else:
    _BreezeModelPatcher = None
    _BreezeModelPatcherDynamic = None


def _safe_repo_name(repo_id: str) -> str:
    for ch in "/\\:":
        repo_id = repo_id.replace(ch, "_")
    return repo_id


def model_dirs() -> list[Path]:
    dirs: list[Path] = []
    if folder_paths is not None:
        primary = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
        for extra in folder_paths.folder_names_and_paths.get(MODEL_FOLDER_NAME, ([], set()))[0]:
            candidate = Path(extra)
            if candidate not in dirs:
                dirs.append(candidate)
        if primary not in dirs:
            dirs.insert(0, primary)
    else:
        dirs.append(Path(__file__).resolve().parent / "models" / MODEL_FOLDER_NAME)
    return dirs


def register_model_folder() -> None:
    if folder_paths is None:
        return
    for base in model_dirs():
        base.mkdir(parents=True, exist_ok=True)
        registered = folder_paths.folder_names_and_paths.get(MODEL_FOLDER_NAME)
        if registered is None or str(base) not in [str(p) for p in registered[0]]:
            folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(base))


def _has_component_files(model_dir: Path, weights_name: str) -> bool:
    if not (model_dir / "config.json").is_file():
        return False
    if not (model_dir / "audio_tokenizer" / "model.safetensors").is_file():
        return False
    return (model_dir / weights_name).is_file()


def _download_model_files(repo_id: str, weights_name: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    allow = list(SMALL_FILE_PATTERNS) + [weights_name]
    logger.info("Downloading %s (%s) to %s", repo_id, weights_name, dest)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=allow,
        endpoint=HF_ENDPOINT,
    )


def resolve_model_dir(repo_choice: str, download_if_missing: bool) -> tuple[Path, str]:
    repo_id, weights_name = REPO_CHOICES.get(repo_choice, (None, None))
    if repo_id is None:
        raise ValueError(f"Unknown model choice: {repo_choice!r}")
    safe_name = _safe_repo_name(repo_id)
    for base in model_dirs():
        candidate = base / safe_name
        if _has_component_files(candidate, weights_name):
            return candidate, weights_name
    dest = model_dirs()[0] / safe_name
    if not download_if_missing:
        raise FileNotFoundError(
            f"Breeze TTS 2 model files not found in {[str(d / safe_name) for d in model_dirs()]}. "
            "Enable download_if_missing on the Breeze TTS 2 Load Model node or place the checkpoint there."
        )
    _download_model_files(repo_id, weights_name, dest)
    return dest, weights_name


def weights_path(model_dir: Path, weights_name: str) -> Path:
    return model_dir / weights_name


# --------------------------------------------------------------------------- #
# Device / dtype / attention resolution
# --------------------------------------------------------------------------- #
def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if mm is not None:
            return torch.device(mm.get_torch_device())
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but torch.cuda.is_available() is False.")
        # comfy_aimdo's get_devctx(int(index)) needs an explicit index
        return torch.device("cuda", torch.cuda.current_device())
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device(device_name)


def resolve_dtype_mode(dtype_name: str, device: torch.device) -> str:
    if device.type == "cpu":
        if dtype_name == "bf16":
            logger.warning("bf16 requested on CPU; falling back to fp32.")
        return "fp32"
    if dtype_name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp32"
    if dtype_name == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this GPU does not support it. Use dtype=auto.")
        return "bf16"
    return "fp32"


def flash_attn_installed() -> bool:
    """True only when flash_attn is a real, importable, installed distribution.

    find_spec alone is not enough: other node packs (e.g. SeedVR2) inject a
    stub flash_attn module with a valid __spec__ into sys.modules when the real
    package is missing, so find_spec reports a false positive. transformers
    resolves the version through importlib.metadata, which then raises
    PackageNotFoundError deep inside model construction.
    """
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        importlib.metadata.version("flash_attn")
    except importlib.metadata.PackageNotFoundError:
        logger.warning(
            "flash_attn is present in sys.modules but has no package metadata "
            "(a stub installed by another custom node); treating it as unavailable."
        )
        return False
    return True


def resolve_attention(attention: str, device: torch.device, dtype_mode: str) -> str:
    """Returns the transformers attention implementation wired into all submodels.

    sageattention is applied at generation time by patching SDPA (see
    native.attention_runtime), so it maps to the sdpa implementation here.
    """
    flash_usable = (
        flash_attn_installed()
        and device.type == "cuda"
        and dtype_mode != "fp32"
    )
    if attention == "auto":
        return "flash_attention_2" if flash_usable else "sdpa"
    if attention == "eager":
        return "eager"
    if attention == "sdpa":
        return "sdpa"
    if attention == "flash_attention":
        if not flash_attn_installed():
            raise ImportError("flash_attention selected but the flash_attn package is not installed.")
        if dtype_mode == "fp32":
            logger.warning("flash_attention does not support fp32; falling back to sdpa.")
            return "sdpa"
        return "flash_attention_2"
    if attention == "sageattention":
        if importlib.util.find_spec("sageattention") is None:
            raise ImportError("sageattention selected but the sageattention package is not installed.")
        return "sdpa"
    raise ValueError(f"Unknown attention backend: {attention!r}")


# --------------------------------------------------------------------------- #
# ComfyUI / AIMDO memory management
# --------------------------------------------------------------------------- #
def dynamic_vram_active(device: torch.device) -> bool:
    if device.type == "cpu":
        return False
    try:
        import comfy.memory_management

        if not bool(comfy.memory_management.aimdo_enabled):
            return False
        import comfy_aimdo.control
        import comfy_aimdo.host_buffer
        import comfy_aimdo.model_vbar

        return (
            comfy_aimdo.control.lib is not None
            and comfy_aimdo.host_buffer.lib is not None
            and comfy_aimdo.model_vbar.lib is not None
        )
    except Exception:
        return False


def _ensure_writable_device_property(module: nn.Module) -> None:
    cls = type(module)
    device_attr = getattr(cls, "device", None)
    if device_attr is None or not isinstance(device_attr, property) or device_attr.fset is not None:
        return
    if getattr(cls, "_breeze_writable_device", False):
        return

    def get_device(self):
        stored = self.__dict__.get("_breeze_runtime_device")
        if stored is not None:
            return stored
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def set_device(self, value):
        self.__dict__["_breeze_runtime_device"] = (
            value if isinstance(value, torch.device) else torch.device(value)
        )

    new_cls = type(
        cls.__name__,
        (cls,),
        {
            "__module__": cls.__module__,
            "device": property(get_device, set_device),
            "_breeze_writable_device": True,
        },
    )
    module.__class__ = new_cls


def _register_many_with_comfy(patchers: list) -> None:
    if mm is None:
        return
    to_load = []
    already = {id(loaded.model) for loaded in mm.current_loaded_models}
    for patcher in patchers:
        if patcher is None or id(patcher.model) in already:
            continue
        to_load.append(patcher)
    if to_load:
        mm.load_models_gpu(to_load)
        logger.debug("Loaded %d module(s) through ComfyUI memory management.", len(to_load))


def register_runtime_module(module: nn.Module, device: torch.device, *, dynamic: bool | None = None):
    module._breeze_runtime_device = torch.device(device)
    _ensure_writable_device_property(module)
    if _ComfyCorePatcher is None or device.type == "cpu":
        module.to(device)
        return None
    use_dynamic = dynamic_vram_active(device) and dynamic is not False
    if use_dynamic and _BreezeModelPatcherDynamic is not None:
        patcher_class = _BreezeModelPatcherDynamic
    else:
        patcher_class = _BreezeModelPatcher
    patcher = patcher_class(module, load_device=device, offload_device=torch.device("cpu"))
    module.model_loaded_weight_memory = 0
    _register_many_with_comfy([patcher])
    if not patcher.is_dynamic():
        module.device = torch.device(device)
    return patcher


def _unregister_from_comfy(patcher) -> None:
    if patcher is None or mm is None:
        return
    for loaded in list(mm.current_loaded_models):
        if id(loaded.model) == id(patcher.model):
            if getattr(loaded, "model_finalizer", None) is not None:
                loaded.model_finalizer.detach()
            if getattr(loaded, "_patcher_finalizer", None) is not None:
                loaded._patcher_finalizer.detach()
            mm.current_loaded_models.remove(loaded)
    patcher.detach()


def _empty_accelerator_cache() -> None:
    if mm is not None:
        try:
            mm.soft_empty_cache()
            return
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def install_comfy_unload_hook() -> None:
    if mm is None or getattr(mm, "_breeze_tts2_unload_hook_installed", False):
        return
    original_unload_all = mm.unload_all_models

    def unload_all_with_breeze(*args, **kwargs):
        bundle = _ACTIVE_BUNDLE
        if bundle is not None:
            unload_breeze_bundle(bundle, reason="ComfyUI unload_all_models")
        return original_unload_all(*args, **kwargs)

    mm.unload_all_models = unload_all_with_breeze
    if hasattr(mm, "unload_model_and_clones"):
        original_unload_clones = mm.unload_model_and_clones

        def unload_clones_with_breeze(*args, **kwargs):
            bundle = _ACTIVE_BUNDLE
            if bundle is not None:
                unload_breeze_bundle(bundle, reason="ComfyUI unload_model_and_clones")
            return original_unload_clones(*args, **kwargs)

        mm.unload_model_and_clones = unload_clones_with_breeze
    mm._breeze_tts2_unload_hook_installed = True


# --------------------------------------------------------------------------- #
# Bundle lifecycle
# --------------------------------------------------------------------------- #
@dataclass
class BreezeBundle:
    model: nn.Module
    codec: nn.Module
    tokenizer: Any
    model_dir: Path
    weights_name: str
    device: torch.device
    dtype_name: str
    attention: str
    decode_mode: str = "eager"
    quantized: bool = False
    patchers: list = field(default_factory=list)


_ACTIVE_BUNDLE: BreezeBundle | None = None
_ACTIVE_LOAD_KEY: tuple[Any, ...] | None = None


def _dtype_policy(dtype_mode: str):
    def policy(name: str):
        if name.endswith("weight_scale"):
            return None
        if name in ("lm_head.weight", "depth_decoder.codebooks_head.weight"):
            return torch.float32
        return torch.bfloat16 if dtype_mode == "bf16" else torch.float32

    return policy


def _build_codec(model_dir: Path) -> Qwen3TTSTokenizerV2Model:
    import json

    codec_path = model_dir / "audio_tokenizer" / "config.json"
    codec_cfg = json.loads(codec_path.read_text(encoding="utf-8"))
    for junk in ("architectures", "transformers_version"):
        codec_cfg.pop(junk, None)
    config = Qwen3TTSTokenizerV2Config(**codec_cfg)
    import accelerate

    with accelerate.init_empty_weights():
        codec = Qwen3TTSTokenizerV2Model(config)
    for name, tensor in native.iter_checkpoint_items(model_dir / "audio_tokenizer"):
        native._set_tensor(codec, name, tensor, None)
    native.materialize_meta_buffers(codec)
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    return codec


def load_breeze_bundle(
    repo_choice: str,
    dtype_name: str,
    device_name: str,
    attention_choice: str,
    download_if_missing: bool,
    decode_mode: str = "eager",
) -> BreezeBundle:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY

    if decode_mode not in DECODE_MODE_OPTIONS:
        raise ValueError(f"Unknown decode_mode: {decode_mode!r} (expected one of {DECODE_MODE_OPTIONS}).")

    model_dir, weights_name = resolve_model_dir(repo_choice, download_if_missing)
    wfile = weights_path(model_dir, weights_name)
    device = resolve_device(device_name)
    dtype_mode = resolve_dtype_mode(dtype_name, device)
    attn_impl = resolve_attention(attention_choice, device, dtype_mode)
    load_key = (str(model_dir), weights_name, wfile.stat().st_mtime_ns, str(device), dtype_mode, attention_choice, decode_mode)

    if _ACTIVE_BUNDLE is not None and _ACTIVE_LOAD_KEY == load_key:
        resume_bundle_to_device(_ACTIVE_BUNDLE)
        return _ACTIVE_BUNDLE
    if _ACTIVE_BUNDLE is not None:
        unload_breeze_bundle(_ACTIVE_BUNDLE, reason="load settings changed")

    from transformers import AutoTokenizer

    quant_map = int8.scan_checkpoint_quantization(wfile)
    config_dict = native.read_config(model_dir)
    model = native.build_breeze_model(config_dict, attn_impl)
    if quant_map:
        int8.replace_quantized_linears(model, quant_map)
    native.load_breeze_weights(model, wfile, dtype_policy=_dtype_policy(dtype_mode))
    native.convert_modules_for_comfy(model)
    if dtype_mode == "bf16":
        native.set_runtime_dtype(model.text_encoder, torch.bfloat16)
        native.set_runtime_dtype(model.backbone_model, torch.bfloat16)
        native.set_runtime_dtype(model.embed_text_tokens, torch.bfloat16)
        native.set_runtime_dtype(model.text_encoder_proj, torch.bfloat16)
        native.set_runtime_dtype(model.depth_decoder.model, torch.bfloat16)
        native.set_runtime_dtype(model.depth_decoder.codebooks_head, torch.float32)
        native.set_runtime_dtype(model.lm_head, torch.float32)
    else:
        native.set_runtime_dtype(model, torch.float32)

    codec = _build_codec(model_dir)
    native.convert_modules_for_comfy(codec)
    native.set_runtime_dtype(codec, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    bundle = BreezeBundle(
        model=model,
        codec=codec,
        tokenizer=tokenizer,
        model_dir=Path(model_dir),
        weights_name=weights_name,
        device=device,
        dtype_name=dtype_mode,
        attention=attention_choice,
        decode_mode=decode_mode,
        quantized=bool(quant_map),
    )

    patchers = []
    try:
        # cuda_graphs captures weight addresses at compile time, so the model
        # must stay resident: register it non-dynamic (AIMDO paging off). The
        # codec is pinned too: at ~280MB, dynamic paging just re-staged it on
        # every decode, spamming a "prepared for dynamic VRAM" line per
        # multi-speaker turn and adding latency.
        patcher = register_runtime_module(model, device, dynamic=False if decode_mode == "cuda_graphs" else None)
        if patcher is not None:
            patchers.append(patcher)
        patcher = register_runtime_module(codec, device, dynamic=False)
        if patcher is not None:
            patchers.append(patcher)
    except Exception:
        for created in patchers:
            _unregister_from_comfy(created)
        unload_breeze_bundle(bundle, reason="registration failed", hard=True)
        raise
    bundle.patchers = patchers

    if bundle.quantized:
        int8.log_int8_banner(model, device)

    _ACTIVE_BUNDLE = bundle
    _ACTIVE_LOAD_KEY = load_key
    install_comfy_unload_hook()
    _empty_accelerator_cache()
    return bundle


def resume_bundle_to_device(bundle: BreezeBundle) -> None:
    _register_many_with_comfy(bundle.patchers)


_POOL_WARN_LINES = (
    b"uncaptured free of a captured allocation",
    b"This is technically allowed, but may indicate you are losing",
)


@contextlib.contextmanager
def _silence_captured_pool_warnings():
    """Park fd 2 while CUDA-graph pools are released and re-emit everything else.

    Under the cudaMallocAsync backend the allocator prints two-line C++
    warnings to stderr for every graph-pool tensor freed outside capture.
    No log level or Python filter reaches them, so stderr is parked in a
    temp file for the release and replayed minus those two lines; output
    from other threads is preserved.
    """
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return
    with tempfile.TemporaryFile() as park:
        os.dup2(park.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            park.seek(0)
            for line in park.read().splitlines(keepends=True):
                if any(marker in line for marker in _POOL_WARN_LINES):
                    continue
                os.write(2, line)


def release_depth_graphs(model: nn.Module | None) -> None:
    """Drop captured CUDA graphs and their pool tensors without warning spam."""
    runners = getattr(model, "_breeze_depth_runners", None) or {}
    if not any(r._graph_prefill is not None for r in runners.values()):
        model._breeze_depth_runners = runners
        return
    with _silence_captured_pool_warnings():
        for runner in runners.values():
            runner._graph_prefill = None
            runner._graph_steps = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    model._breeze_depth_runners = runners


def _release_active_graphs_at_exit() -> None:
    if _ACTIVE_BUNDLE is not None and _ACTIVE_BUNDLE.model is not None:
        try:
            release_depth_graphs(_ACTIVE_BUNDLE.model)
        except Exception:
            pass


atexit.register(_release_active_graphs_at_exit)


def unload_breeze_bundle(bundle: BreezeBundle | None, reason: str = "unload", hard: bool = True) -> None:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY
    if bundle is None:
        return
    logger.info("Unloading Breeze TTS 2 bundle (%s).", reason)
    release_depth_graphs(bundle.model)
    for patcher in bundle.patchers:
        try:
            _unregister_from_comfy(patcher)
        except Exception:
            pass
    bundle.patchers = []
    if hard:
        for module in (bundle.model, bundle.codec):
            if module is None:
                continue
            try:
                if hasattr(module, "dynamic_vbars"):
                    module.dynamic_vbars = {}
                module.to_empty(device=torch.device("meta"))
            except Exception:
                pass
    bundle.model = None
    bundle.codec = None
    bundle.tokenizer = None
    if _ACTIVE_BUNDLE is bundle:
        _ACTIVE_BUNDLE = None
        _ACTIVE_LOAD_KEY = None
    gc.collect()
    _empty_accelerator_cache()

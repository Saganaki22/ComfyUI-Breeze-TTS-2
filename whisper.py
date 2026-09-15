"""Whisper ASR node for producing reference transcripts for Breeze TTS 2 voice cloning."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

from . import loader

logger = logging.getLogger("BreezeTTS2")

CATEGORY = "Breeze TTS 2"

POPULAR_WHISPER_MODELS = {
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    "whisper-large-v3": "openai/whisper-large-v3",
    "whisper-medium": "openai/whisper-medium",
    "whisper-small": "openai/whisper-small",
    "whisper-base": "openai/whisper-base",
    "whisper-tiny": "openai/whisper-tiny",
}

LANGUAGE_CHOICES = [
    "auto", "english", "chinese", "german", "spanish", "russian", "korean",
    "french", "japanese", "portuguese", "turkish", "polish", "catalan",
]

DTYPE_OPTIONS = ["auto", "bf16", "fp32"]

WHISPER_IGNORE_PATTERNS = ["*.msgpack", "*.h5", "tf_model*", "flax_model*"]

_PIPELINE_CACHE: dict = {}

try:
    import folder_paths
except Exception:
    folder_paths = None


WHISPER_FOLDER_NAME = "audio_encoders"


def _whisper_roots() -> list[Path]:
    """Every folder searched for Whisper models, in priority order.

    Same layering as loader.model_dirs(): the running install's models tree
    first, then any path registered for "audio_encoders" (an
    extra_model_paths.yaml section can map it directly), then audio_encoders
    folders inside models trees mapped through standard folder names only.
    """
    dirs: list[Path] = []
    seen: set[str] = set()

    def add(candidate: Path) -> None:
        folded = os.path.normcase(str(candidate))
        if folded not in seen:
            seen.add(folded)
            dirs.append(candidate)

    if folder_paths is not None:
        add(Path(folder_paths.models_dir) / WHISPER_FOLDER_NAME)
        for extra in folder_paths.folder_names_and_paths.get(WHISPER_FOLDER_NAME, ([], set()))[0]:
            add(Path(extra))
        for root in loader.mapped_models_roots():
            candidate = root / WHISPER_FOLDER_NAME
            if candidate.is_dir():
                add(candidate)
    else:
        dirs.append(Path(__file__).resolve().parent / "models" / WHISPER_FOLDER_NAME)
    return dirs


def whisper_model_choices() -> list[str]:
    choices = list(POPULAR_WHISPER_MODELS.keys())
    for root in _whisper_roots():
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "config.json").is_file() and child.name not in choices:
                choices.append(child.name)
    return choices


def _resolve_whisper_dir(model_name: str, download_if_missing: bool) -> Path:
    roots = _whisper_roots()
    if model_name in POPULAR_WHISPER_MODELS:
        repo_id = POPULAR_WHISPER_MODELS[model_name]
        folder_name = repo_id.replace("/", "_")
        for root in roots:
            local = root / folder_name
            if (local / "config.json").is_file():
                return local
        local = roots[0] / folder_name
        if not download_if_missing:
            raise FileNotFoundError(
                f"Whisper model '{model_name}' not found (looked in {[str(r / folder_name) for r in roots]}). "
                "Enable download_if_missing."
            )
        from huggingface_hub import snapshot_download

        logger.info("Downloading Whisper model %s to %s", repo_id, local)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local),
            ignore_patterns=WHISPER_IGNORE_PATTERNS,
            endpoint=loader.HF_ENDPOINT,
        )
        return local
    for root in roots:
        local = root / model_name
        if (local / "config.json").is_file():
            return local
    raise FileNotFoundError(f"Whisper model folder '{model_name}' not found under {[str(r) for r in roots]}.")


def _resolve_whisper_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "fp32" or device.type == "cpu":
        return torch.float32
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float16 if device.type == "cuda" else torch.float32


def _get_whisper_pipeline(model_name: str, dtype_name: str, device: torch.device, download_if_missing: bool):
    key = (model_name, dtype_name, str(device))
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        patcher = getattr(cached, "_breeze_tts2_patcher", None)
        if patcher is not None:
            loader._register_many_with_comfy([patcher])
        return cached

    from transformers import pipeline as hf_pipeline

    model_dir = _resolve_whisper_dir(model_name, download_if_missing)
    torch_dtype = _resolve_whisper_dtype(dtype_name, device)
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=str(model_dir),
        torch_dtype=torch_dtype,
        device=device,
    )
    patcher = None
    if device.type != "cpu":
        # A plain (non-paging) patcher keeps ComfyUI in charge of offload while
        # the pipeline's declared device stays consistent with the weights.
        from . import native

        native.convert_modules_for_comfy(pipe.model)
        native.set_runtime_dtype(pipe.model, torch_dtype)
        patcher = loader.register_runtime_module(pipe.model, device, dynamic=False)
        pipe._breeze_tts2_patcher = patcher
    _PIPELINE_CACHE[key] = pipe
    return pipe


class BreezeWhisperTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio to transcribe (e.g. the reference clip used for cloning)."}),
                "model": (
                    whisper_model_choices(),
                    {"default": "whisper-large-v3-turbo", "tooltip": "Whisper model used for transcription."},
                ),
                "dtype": (DTYPE_OPTIONS, {"default": "auto", "tooltip": "Compute dtype for Whisper."}),
                "language": (
                    LANGUAGE_CHOICES,
                    {"default": "auto", "tooltip": "Spoken language hint; auto detects it."},
                ),
                "task": (
                    ["transcribe", "translate"],
                    {"default": "transcribe", "tooltip": "transcribe keeps the source language; translate writes English."},
                ),
                "chunk_length_s": (
                    "INT",
                    {"default": 30, "min": 0, "max": 120, "tooltip": "Chunk length in seconds for long inputs (0 uses the pipeline default)."},
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Download the Whisper model from Hugging Face when missing locally."},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "transcript")
    FUNCTION = "transcribe"
    CATEGORY = CATEGORY
    DESCRIPTION = "Transcribe reference audio with Whisper to get the exact transcript voice cloning needs."

    def transcribe(self, audio, model, dtype, language, task, chunk_length_s, download_if_missing):
        device = loader.resolve_device("auto")
        waveform = audio["waveform"][0].detach().float().cpu()
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        audio_np = waveform.numpy()

        pipe = _get_whisper_pipeline(model, dtype, device, bool(download_if_missing))
        generate_kwargs = {"task": task}
        if language != "auto":
            generate_kwargs["language"] = language
        call_kwargs = {}
        if int(chunk_length_s) > 0:
            call_kwargs["chunk_length_s"] = int(chunk_length_s)
        result = pipe(
            {"array": audio_np, "sampling_rate": int(audio["sample_rate"])},
            generate_kwargs=generate_kwargs,
            **call_kwargs,
        )
        transcript = result["text"].strip()
        return (audio, transcript)


NODE_CLASS_MAPPINGS = {
    "BreezeWhisperTranscribe": BreezeWhisperTranscribe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BreezeWhisperTranscribe": "Breeze TTS 2 Whisper Transcribe",
}

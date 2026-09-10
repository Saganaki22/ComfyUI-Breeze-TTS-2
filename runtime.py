"""Eager inference runtime for Breeze TTS 2.

Reimplements the official eager streaming path (breeze-tts
FastBreezeStreamingRuntime with all fast stages disabled) using KV caches and
the ComfyUI progress/interrupt hooks. Sampling order, CFG formulas, token
layout, and EOS/pad handling follow the upstream implementation exactly.
"""

from __future__ import annotations

import gc
import inspect
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.masking_utils import create_causal_mask

from .native import _CAUSAL_MASK_EMBED_KEY

logger = logging.getLogger("BreezeTTS2")

SAMPLE_RATE = 24_000
FRAMES_PER_SECOND = 12.5
MAX_SEQ_LEN = 2048
# One codec frame = one backbone position, so reference audio spends prompt
# budget at 12.5 tokens/s. 60 s = 750 tokens, leaving >=1200 for speech.
MAX_REFERENCE_SECONDS = 60.0
AUDIO_TAG = "<|AUDIO|>"
AUDIO_EOS = "<|audio_eos|>"
INSTRUCTION_BOS = "<ins_bos>"
INSTRUCTION_EOS = "<ins_eos>"
DEFAULT_INSTRUCTION = "Speak clearly and naturally."


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fix_seed(seed: int) -> None:
    if seed and seed > 0:
        set_all_seeds(seed)


# --------------------------------------------------------------------------- #
# Prompt templates (from breeze_infer/templates.py)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BranchInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    text_ids_mask: torch.Tensor
    text_ids_len: torch.Tensor
    input_values: torch.Tensor | None


def _speaker_prefix(speaker: str) -> str:
    if speaker in (None, ""):
        return ""
    if speaker.startswith("[") and speaker.endswith("]"):
        return speaker
    return f"[{speaker}]"


def design_segments(text: str, instruction: str, speaker: str = "S0") -> list[dict[str, Any]]:
    prefix = _speaker_prefix(speaker)
    return [
        {
            "type": "text",
            "text": f"{prefix}{INSTRUCTION_BOS}{instruction}{INSTRUCTION_EOS}{text}",
        }
    ]


def design_negative_segments(text: str, speaker: str = "S0") -> list[dict[str, Any]]:
    prefix = _speaker_prefix(speaker)
    return [{"type": "text", "text": f"{prefix}{text}"}]


def ref_segments(
    ref_text: str, text: str, instruction: str, speaker: str = "S0", *, with_instruction: bool = True
) -> list[dict[str, Any]]:
    prefix = _speaker_prefix(speaker)
    segments: list[dict[str, Any]] = [{"type": "text", "text": f"{prefix}{ref_text}"}]
    segments.append({"type": "audio", "append_eos": True})
    if with_instruction:
        segments.append(
            {
                "type": "text",
                "text": f"{prefix}{INSTRUCTION_BOS}{instruction}{INSTRUCTION_EOS}{text}",
            }
        )
    else:
        segments.append({"type": "text", "text": f"{prefix}{text}"})
    return segments


def _prepare_one(tokenizer, model_config, segments, ref_codes: torch.Tensor | None) -> dict[str, torch.Tensor]:
    rendered_segments: list[dict[str, str]] = []
    for segment in segments:
        if segment["type"] == "text":
            encoded = tokenizer(segment["text"], add_special_tokens=True)
            rendered = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
            rendered_segments.append({"type": "text", "value": rendered})
        elif segment["type"] == "audio":
            codes = ref_codes
            if codes is None:
                raise ValueError("Audio segment provided but no reference codes were given.")
            placeholders = AUDIO_TAG * codes.shape[0]
            if segment.get("append_eos", False):
                placeholders += AUDIO_EOS
            rendered_segments.append({"type": "audio", "value": placeholders})
        else:
            raise ValueError(f"Unknown segment type: {segment['type']}")

    final_text = "".join(segment["value"] for segment in rendered_segments)
    encoded = tokenizer(final_text, add_special_tokens=False, return_tensors="pt")

    text_ids_mask: list[bool] = []
    text_ids_len: list[int] = []
    for segment in rendered_segments:
        segment_len = len(tokenizer(segment["value"], add_special_tokens=False)["input_ids"])
        if segment["type"] == "text":
            text_ids_mask.extend([True] * segment_len)
            text_ids_len.append(segment_len)
        else:
            text_ids_mask.extend([False] * segment_len)

    num_codebooks = getattr(model_config, "num_codebooks", 16)
    if ref_codes is not None:
        audio_tokens = ref_codes.unsqueeze(0)
    else:
        audio_tokens = torch.zeros((1, 0, num_codebooks), dtype=torch.int16)

    encoded["audio_tokens"] = audio_tokens
    encoded["text_ids_mask"] = torch.tensor([text_ids_mask], dtype=torch.bool)
    encoded["text_ids_len"] = torch.tensor(text_ids_len, dtype=torch.long)
    return encoded


def _to_branch(tokenizer, model_config, device, segments, ref_codes) -> BranchInputs:
    prepared = _prepare_one(tokenizer, model_config, segments, ref_codes)
    input_values = prepared["audio_tokens"]
    return BranchInputs(
        input_ids=prepared["input_ids"].to(device),
        attention_mask=prepared["attention_mask"].to(device),
        text_ids_mask=prepared["text_ids_mask"].to(device),
        text_ids_len=prepared["text_ids_len"].to(device),
        input_values=input_values.to(device) if input_values.shape[1] > 0 else None,
    )


def _left_pad(tensor: torch.Tensor, target_len: int, value: float) -> torch.Tensor:
    pad_len = target_len - tensor.shape[1]
    if pad_len <= 0:
        return tensor
    pad_shape = (tensor.shape[0], pad_len, *tensor.shape[2:])
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([pad, tensor], dim=1)


def build_generation_batch(
    model,
    tokenizer,
    *,
    cond_segments,
    negative_segments,
    ref_codes,
    cfg_scale: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build the (possibly CFG-paired) inputs_embeds batch for generation.

    Returns (inputs_embeds [B, L, H], attention_mask [B, L], base_positions [B],
    prefill_len) where B is 2 with CFG and 1 without.
    """
    model_config = model.config
    cond = _to_branch(tokenizer, model_config, device, cond_segments, ref_codes)
    branches = [cond]
    if cfg_scale != 1.0:
        if negative_segments is None:
            raise ValueError(f"cfg_scale={cfg_scale} requires a negative prompt branch.")
        branches.append(_to_branch(tokenizer, model_config, device, negative_segments, ref_codes))

    max_len = max(branch.input_ids.shape[1] for branch in branches)
    input_ids = torch.cat(
        [_left_pad(branch.input_ids, max_len, tokenizer.pad_token_id) for branch in branches], dim=0
    )
    attention_mask = torch.cat(
        [_left_pad(branch.attention_mask, max_len, 0) for branch in branches], dim=0
    )
    text_ids_mask = torch.cat(
        [_left_pad(branch.text_ids_mask, max_len, False) for branch in branches], dim=0
    )
    text_ids_len = torch.cat([branch.text_ids_len for branch in branches], dim=0)
    if all(branch.input_values is not None for branch in branches):
        input_values = torch.cat([branch.input_values for branch in branches], dim=0)
    elif all(branch.input_values is None for branch in branches):
        input_values = None
    else:
        raise RuntimeError("Cond and negative branches disagree on reference audio presence.")

    inputs_embeds = model.merge_prompt(input_ids, text_ids_mask, text_ids_len, input_values)
    base_positions = attention_mask.sum(dim=1).long()
    return inputs_embeds.contiguous(), attention_mask.contiguous(), base_positions, max_len


# --------------------------------------------------------------------------- #
# Sampling (from models/cudagraph/sampling.py)
# --------------------------------------------------------------------------- #
def apply_repetition_penalty(logits, token_history, repetition_penalty):
    if repetition_penalty == 1.0 or token_history.numel() == 0:
        return logits
    unique_toks = token_history.unique()
    tok_logits = logits[..., unique_toks]
    logits[..., unique_toks] = torch.where(
        tok_logits > 0, tok_logits / repetition_penalty, tok_logits * repetition_penalty
    )
    return logits


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    token_history: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
    suppress_tokens=None,
) -> torch.Tensor:
    """HF-compatible order: penalty -> suppress -> temperature -> top_k -> top_p -> softmax -> sample."""
    logits = logits.clone().float()
    if token_history is not None:
        logits = apply_repetition_penalty(logits, token_history, repetition_penalty)
    if suppress_tokens:
        logits[..., list(suppress_tokens)] = float("-inf")
    if not do_sample:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k)
        threshold = topk_vals[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = F.pad(sorted_indices_to_remove[..., :-1], (1, 0), value=False)
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_indices, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


@dataclass
class GenerationParams:
    max_new_tokens: int = 1500
    max_seq_len: int = 2048
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    depth_top_p: float = 1.0


def _check_interrupted() -> None:
    try:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# CUDA RNG poison healing (ComfyUI 0.35.0+ "Offset increment" crash, issue #10)
# --------------------------------------------------------------------------- #
# Every torch.cuda.graph capture registers the *default* CUDA generator and
# sets its capturing_ flag (capture_prologue). If a capture fails after that
# point but before capture_end (mempool conflicts, nested captures, invalid
# ops), the flag stays stuck: every later CUDA RNG call anywhere in the
# process — including torch.multinomial in sample_logits — raises
# "RuntimeError: Offset increment outside graph capture encountered
# unexpectedly." The flag is shared process state, so the crash surfaces in
# whichever node samples first, not necessarily in the one that broke the
# capture. ComfyUI 0.35.0's model compiler captures CUDA graphs around its own
# model runs, which makes such stuck captures much more likely.
_RNG_POISON_MESSAGE = "Offset increment outside graph capture"


def is_rng_poison_error(exc: BaseException | None) -> bool:
    return isinstance(exc, RuntimeError) and _RNG_POISON_MESSAGE in str(exc)


def heal_cuda_rng() -> bool:
    """Un-poison the shared default CUDA generator after an aborted capture.

    Two layers, because a failed capture can leave two kinds of damage:
      1. The generator's capturing_ flag stuck true (fixed by swapping in a
         fresh generator state via graphsafe_set_state).
      2. A capture invalidated at the driver level with the current stream
         parked on the capture stream (fixed by restoring the default stream,
         then synchronizing and flushing the allocator).
    Returns True when a probe RNG call succeeds afterwards.
    """
    if not torch.cuda.is_available():
        return False
    try:
        device = torch.cuda.current_device()
        torch.rand(1, device=device)
        return True  # not poisoned; nothing to do
    except RuntimeError:
        pass
    try:
        generator = torch.cuda.default_generators[torch.cuda.current_device()]
        generator.graphsafe_set_state(torch.Generator(device="cuda"))
        torch.rand(1, device="cuda")
        logger.warning(
            "cuda_graphs: CUDA RNG state was left poisoned by an aborted graph "
            "capture; reset the generator state."
        )
        return True
    except RuntimeError:
        pass
    try:
        torch.cuda.set_stream(torch.cuda.default_stream())
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.rand(1, device="cuda")
        logger.warning(
            "cuda_graphs: CUDA RNG state was left poisoned by an invalidated "
            "graph capture; restored the default stream and reset the "
            "generator state."
        )
        return True
    except Exception:
        logger.warning(
            "cuda_graphs: failed to heal the poisoned CUDA RNG state; "
            "generation cannot sample. Restarting ComfyUI clears it.",
            exc_info=True,
        )
        return False


def ensure_cuda_rng_usable(device=None) -> bool:
    """Probe the CUDA RNG and heal it if a stuck capture poisoned it."""
    if device is not None and device.type != "cuda":
        return True
    try:
        torch.rand(1, device="cuda" if device is None else device)
        return True
    except RuntimeError as exc:
        if not is_rng_poison_error(exc):
            raise
    logger.warning(
        "cuda_graphs: CUDA RNG state was left poisoned by an aborted graph "
        "capture elsewhere in the process; healing before generation."
    )
    return heal_cuda_rng()


def _new_static_cache(config, max_cache_len: int, batch: int, device, dtype):
    # transformers renamed StaticCache's batch argument between 4.x and 5.x.
    params = inspect.signature(StaticCache.__init__).parameters
    kwargs = {"config": config, "max_cache_len": max_cache_len}
    if "max_batch_size" in params:
        kwargs["max_batch_size"] = batch
    else:
        kwargs["batch_size"] = batch
    if "device" in params:
        kwargs["device"] = device
    if "dtype" in params:
        kwargs["dtype"] = dtype
    return StaticCache(**kwargs)


class _GraphFallback(Exception):
    pass


class _DepthRunner:
    """Depth decoder hot loop with everything precomputable precomputed.

    The official CUDA-graph path precomputed rope, per-position attention
    masks, cache positions, and a static KV cache once; this does the same for
    the eager path. With use_graph the per-step forwards are additionally
    captured into CUDA graphs by hand (inductor's cudagraph trees cannot
    record under ComfyUI's default cudaMallocAsync allocator). Graph replay
    pins the weight addresses it captured, so the loader registers the model
    non-dynamic (AIMDO paging off) in that mode. Runners are stored on the
    model keyed by batch size so captured graphs persist across generations.
    """

    def __init__(self, model, params: GenerationParams, cfg_scale: float, use_graph: bool = False):
        self.model = model
        self.depth = model.depth_decoder.model
        self.head = model.depth_decoder.codebooks_head
        self.vocab = self.depth.vocab_size
        self.codebook_size = model.config.codec_codebook_size
        self.num_decode = model.config.num_codebooks - 1
        self.params = params
        self.cfg_scale = cfg_scale
        self.use_graph = use_graph
        self._graph_prefill = None
        self._graph_steps = None
        self._capture_after_frame = False
        self._ready = False

    def _prepare(self, batch: int, device, dtype):
        depth = self.depth
        length = 2 + self.num_decode
        dummy = torch.zeros(1, 1, 1, device=device, dtype=dtype)
        pos = torch.arange(length, device=device)
        cos, sin = depth.rotary_emb(dummy, pos.unsqueeze(0))
        self._cos, self._sin = cos[0], sin[0]
        self.cache = _new_static_cache(depth.config, length, batch, device, dtype)
        min_val = torch.finfo(dtype).min
        # Masks must carry the true batch size: flash-attention's varlen path
        # derives cu_seqlens from the mask shape, so a broadcast batch-1 mask
        # aborts the kernel under CFG (batch=2).
        prefill_mask = torch.full((batch, 1, 2, length), min_val, device=device, dtype=dtype)
        prefill_mask[:, 0, 0, 0] = 0.0
        prefill_mask[:, 0, 1, :2] = 0.0
        self._prefill_mask = prefill_mask
        decode_masks = []
        for t in range(2, length):
            mask = torch.full((batch, 1, 1, length), min_val, device=device, dtype=dtype)
            mask[..., : t + 1] = 0.0
            decode_masks.append(mask)
        self._decode_masks = decode_masks
        self._positions = [torch.tensor([t], device=device) for t in range(2, length)]
        self._offsets = [i * self.vocab for i in range(self.num_decode)]
        self._prefill_pos = torch.arange(2, device=device)
        self._prefill_rope = (self._cos[:2].unsqueeze(0), self._sin[:2].unsqueeze(0))
        self._rope_steps = [
            (self._cos[t : t + 1].unsqueeze(0), self._sin[t : t + 1].unsqueeze(0))
            for t in range(2, length)
        ]
        self._ready = True

    def _prefill_forward(self, embeds: torch.Tensor) -> torch.Tensor:
        hidden = embeds
        for layer in self.depth.layers:
            hidden = layer(
                hidden,
                attention_mask=self._prefill_mask,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=self._prefill_pos,
                position_embeddings=self._prefill_rope,
            )
        hidden = self.depth.norm(hidden)
        return self.head(hidden[:, 1:, :].float(), cache_position=self._prefill_pos[1:])

    def _step_forward(self, emb, mask, cos, sin, cache_position) -> torch.Tensor:
        hidden = emb
        for layer in self.depth.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
            )
        hidden = self.depth.norm(hidden)
        return self.head(hidden.float(), cache_position=cache_position)

    def _record(self, fn, static_in: torch.Tensor, *consts):
        # torch.compile's inductor cudagraphs cannot capture under ComfyUI's
        # default cudaMallocAsync allocator (triton rejects pool pointers), so
        # capture the eager steps manually: only static_in varies per call,
        # the precomputed masks/rope/positions are captured by reference.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):

                self.cache.reset()

                fn(static_in, *consts)
        torch.cuda.current_stream().wait_stream(stream)

        self.cache.reset()

        graph = torch.cuda.CUDAGraph()
        # thread_local: ComfyUI's server thread (and monitor custom nodes)
        # poll CUDA stats from other threads; process-global capture treats
        # any such call during capture as fatal. Capture is per-thread here.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            static_out = fn(static_in, *consts)
        return graph, static_in, static_out

    def _maybe_capture(self, prefill_shape, step_shape, device, dtype) -> None:
        if not self.use_graph or self._graph_prefill is not None:
            return
        try:
            prefill_in = torch.zeros(prefill_shape, device=device, dtype=dtype)
            self._graph_prefill = self._record(self._prefill_forward, prefill_in)
            steps = []
            for cb_idx in range(1, self.num_decode):
                step_in = torch.zeros(step_shape, device=device, dtype=dtype)
                cos, sin = self._rope_steps[cb_idx - 1]
                consts = (self._decode_masks[cb_idx - 1], cos, sin, self._positions[cb_idx - 1])
                steps.append(self._record(self._step_forward, step_in, *consts))
            self._graph_steps = steps
        except Exception:
            logger.warning("cuda_graphs: capture failed; depth decode falls back to eager.", exc_info=True)
            self.use_graph = False
            self._graph_prefill = None
            self._graph_steps = None
            # A failed capture can leave the shared default generator's
            # capture flag stuck; heal it so the eager fallback can sample.
            heal_cuda_rng()

    def _call(self, entry, eager, data, *args) -> torch.Tensor:
        if entry is None:
            return eager(data, *args)
        graph, static_in, static_out = entry
        try:
            static_in.copy_(data)
            graph.replay()
            # Clone: the static output buffer is overwritten by the next replay.
            return static_out.clone()
        except Exception:
            logger.warning("cuda_graphs: replay failed; depth decode falls back to eager.", exc_info=True)
            self.use_graph = False
            self._graph_prefill = None
            self._graph_steps = None
            raise _GraphFallback()

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        half = logits.shape[0] // 2
        if logits.shape[0] >= 2:
            cond = logits[:half, 0, :]
            uncond = logits[half:, 0, :]
            cfg = uncond + self.cfg_scale * (cond - uncond)
        else:
            cfg = logits[:, 0, :]
        cfg[..., self.codebook_size : self.vocab] = float("-inf")
        return sample_logits(
            cfg,
            temperature=self.params.depth_temperature,
            top_k=self.params.depth_top_k,
            top_p=self.params.depth_top_p,
            do_sample=True,
        ).clamp_(0, self.vocab - 1)

    def run(self, backbone_hidden: torch.Tensor, first_token: torch.Tensor) -> torch.Tensor:
        """Returns the 15 remaining codebook tokens for one frame."""
        while True:
            try:
                return self._run_frame(backbone_hidden, first_token)
            except _GraphFallback:
                continue  # graphs are disabled by _call; the retry is eager

    def _run_frame(self, backbone_hidden: torch.Tensor, first_token: torch.Tensor) -> torch.Tensor:
        depth = self.depth
        batch = backbone_hidden.shape[0]
        device = backbone_hidden.device
        dtype = backbone_hidden.dtype
        if not self._ready:
            self._prepare(batch, device, dtype)
        self.cache.reset()

        backbone_h = backbone_hidden
        if depth.backbone_hidden_state_projector is not None:
            backbone_h = depth.backbone_hidden_state_projector(backbone_h)
        ids = torch.zeros(batch, 2, dtype=torch.long, device=device)
        ids[:, 1] = first_token
        embeds = depth.embed_tokens(ids)
        embeds[:, 0] = backbone_h
        embeds = depth.inputs_embeds_projector(embeds)

        if self.use_graph and self._graph_prefill is None and not self._capture_after_frame:
            # First frame stays eager: it lazily allocates the static cache
            # tensors and gives us the real input shapes to capture with.
            self._capture_after_frame = True
            self._capture_shapes = (embeds.shape, device, dtype)
        logits = self._call(self._graph_prefill, self._prefill_forward, embeds)
        token = self._sample(logits)
        tokens = [token]

        step_shape = None
        for cb_idx in range(1, self.num_decode):
            offset_tok = (token + self._offsets[cb_idx]).repeat(batch)
            emb = depth.embed_tokens(offset_tok.unsqueeze(1).clamp_(0, self.model.config.num_codebooks * self.vocab - 1))
            emb = depth.inputs_embeds_projector(emb)
            if step_shape is None:
                step_shape = emb.shape
            if self._graph_steps is not None:
                logits = self._call(self._graph_steps[cb_idx - 1], self._step_forward, emb)
            else:
                cos, sin = self._rope_steps[cb_idx - 1]
                logits = self._step_forward(emb, self._decode_masks[cb_idx - 1], cos, sin, self._positions[cb_idx - 1])
            token = self._sample(logits)
            tokens.append(token)

        if self._capture_after_frame:
            self._capture_after_frame = False
            prefill_shape, cap_device, cap_dtype = self._capture_shapes
            self._maybe_capture(prefill_shape, step_shape, cap_device, cap_dtype)
        return torch.cat(tokens, dim=0)


def get_depth_runner(model, params: GenerationParams, cfg_scale: float, batch: int, use_graph: bool) -> _DepthRunner:
    runners = getattr(model, "_breeze_depth_runners", None)
    if runners is None:
        runners = model._breeze_depth_runners = {}
    runner = runners.get(batch)
    if runner is None:
        runner = _DepthRunner(model, params, cfg_scale, use_graph=use_graph)
        runners[batch] = runner
    else:
        runner.params = params
        runner.cfg_scale = cfg_scale
    return runner


@torch.inference_mode()
def _generate_codes_impl(
    model,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    base_positions: torch.Tensor,
    prefill_len: int,
    cfg_scale: float,
    params: GenerationParams,
    progress_callback: Callable[[int], None] | None = None,
    decode_mode: str = "eager",
) -> torch.Tensor:
    """Run the eager backbone + depth-decoder loop; returns codec codes [num_frames, 16]."""
    device = inputs_embeds.device
    config = model.config
    vocab = config.vocab_size
    num_codebooks = config.num_codebooks
    reserved = tuple(range(int(config.codec_codebook_size), vocab))
    use_graph = decode_mode == "cuda_graphs"
    if use_graph and device.type != "cuda":
        logger.warning("cuda_graphs decode mode requires a CUDA device; using eager decode.")
        use_graph = False

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    past = DynamicCache()
    out = model.backbone_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past,
        use_cache=True,
    )
    hidden = out.last_hidden_state
    logits = model.lm_head(hidden[:, -1, :].float())
    if logits.shape[0] >= 2:
        cond_logits = logits[:1]
        uncond_logits = logits[1:]
        logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
    else:
        logits = logits[:1]
    token = sample_logits(
        logits,
        temperature=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
        do_sample=True,
        suppress_tokens=reserved,
    ).view(1)

    batch = inputs_embeds.shape[0]
    depth = get_depth_runner(model, params, cfg_scale, batch, use_graph)
    # Preallocated attention mask and token history, like the official graph
    # path: step t slices a view instead of concatenating growing tensors.
    mask_buffer = torch.ones(batch, prefill_len + params.max_new_tokens + 1, dtype=attention_mask.dtype, device=device)
    mask_buffer[:, :prefill_len] = attention_mask
    token_history = torch.empty(params.max_new_tokens, dtype=torch.long, device=device)
    cache_positions = [torch.tensor([prefill_len + i], device=device) for i in range(params.max_new_tokens)]

    frames: list[torch.Tensor] = []
    for step_idx in range(params.max_new_tokens):
        _check_interrupted()
        if int(token.item()) == config.vocab_size:
            break

        if batch >= 2:
            token_batch = token.repeat(2)
            depth_hidden = hidden[:, -1, :]
        else:
            token_batch = token
            depth_hidden = hidden[:1, -1, :]

        depth_tokens = depth.run(depth_hidden, token_batch)
        frame = torch.cat([token.view(1), depth_tokens], dim=0)
        if not bool((frame == config.codebook_pad_token_id).all().item()):
            frames.append(frame.detach())

        if progress_callback is not None:
            progress_callback(step_idx + 1)

        token_history[step_idx] = token[0]

        frame_embeds = model.backbone_model.embed_tokens(frame.view(1, 1, num_codebooks))
        if batch >= 2:
            frame_embeds = frame_embeds.repeat(2, 1, 1)
        step_mask = mask_buffer[:, : prefill_len + step_idx + 1]
        step_position_ids = (base_positions + step_idx).unsqueeze(-1)
        out = model.backbone_model(
            inputs_embeds=frame_embeds,
            attention_mask=step_mask,
            position_ids=step_position_ids,
            past_key_values=past,
            cache_position=cache_positions[step_idx],
            use_cache=True,
        )
        hidden = out.last_hidden_state
        logits = model.lm_head(hidden[:, -1, :].float())
        if logits.shape[0] >= 2:
            cond_logits = logits[:1]
            uncond_logits = logits[1:]
            logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
        else:
            logits = logits[:1]
        token = sample_logits(
            logits,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            do_sample=True,
            token_history=token_history[: step_idx + 1],
            repetition_penalty=params.repetition_penalty,
            suppress_tokens=reserved,
        ).view(1)

    if not frames:
        raise RuntimeError("Breeze TTS 2 produced no audio frames.")
    return torch.stack(frames, dim=0)


def generate_codes(
    model,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    base_positions: torch.Tensor,
    prefill_len: int,
    cfg_scale: float,
    params: GenerationParams,
    progress_callback: Callable[[int], None] | None = None,
    decode_mode: str = "eager",
) -> torch.Tensor:
    """Generate codec codes, healing a poisoned CUDA RNG state if needed.

    Sampling anywhere in the process raises "Offset increment outside graph
    capture encountered unexpectedly" once some other CUDA graph capture has
    left the shared default generator stuck (issue #10, seen since ComfyUI
    0.35.0). Probe up front, and retry once after healing if it strikes
    mid-generation.
    """
    ensure_cuda_rng_usable(inputs_embeds.device)
    try:
        return _generate_codes_impl(
            model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            base_positions=base_positions,
            prefill_len=prefill_len,
            cfg_scale=cfg_scale,
            params=params,
            progress_callback=progress_callback,
            decode_mode=decode_mode,
        )
    except RuntimeError as exc:
        if not is_rng_poison_error(exc):
            raise
    logger.warning(
        "cuda_graphs: sampling hit a CUDA RNG state poisoned by an aborted "
        "graph capture; healing and retrying the generation once."
    )
    if not heal_cuda_rng():
        raise RuntimeError(
            "Breeze TTS 2 cannot sample: the CUDA RNG state was left stuck by "
            "an aborted CUDA graph capture and could not be healed. Restart "
            "ComfyUI to clear it."
        )
    with torch.inference_mode():
        return _generate_codes_impl(
            model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            base_positions=base_positions,
            prefill_len=prefill_len,
            cfg_scale=cfg_scale,
            params=params,
            progress_callback=progress_callback,
            decode_mode=decode_mode,
        )


# --------------------------------------------------------------------------- #
# Speech-length prediction for progress display
# --------------------------------------------------------------------------- #
# Calibrated on the official checkpoint: en ~= 4.1 frames/text-token,
# zh ~= 3.5 frames/text-token, each inline vocal event ~= 5 extra frames
# (12.5 frames per second of speech).
FRAMES_PER_TOKEN_EN = 4.1
FRAMES_PER_TOKEN_ZH = 3.5
EVENT_FRAMES = 5.0

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_EN_EVENT_RE = re.compile(
    r"\((?:laugh|laughs|laughing|cough|coughs|clears\s+throat|sigh|sighs|sniff|sneeze|groan|gasp|hum)\)",
    re.IGNORECASE,
)
_ZH_EVENT_RE = re.compile(r"\[(?:笑|笑声|咳嗽|清嗓子|叹气|叹息|抽泣|哭|喘息|呼气)\]")


def estimate_speech_frames(tokenizer, text: str) -> int:
    """Estimate how many codec frames the text will speak, for progress display."""
    events = len(_EN_EVENT_RE.findall(text)) + len(_ZH_EVENT_RE.findall(text))
    stripped = _EN_EVENT_RE.sub(" ", _ZH_EVENT_RE.sub(" ", text))
    stripped = stripped.strip()
    if not stripped:
        return int(events * EVENT_FRAMES) + 8
    ids = tokenizer(stripped, add_special_tokens=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    cjk = sum(1 for tok in tokens if _CJK_RE.search(tok))
    other = len(tokens) - cjk
    est = cjk * FRAMES_PER_TOKEN_ZH + other * FRAMES_PER_TOKEN_EN + events * EVENT_FRAMES + 4.0
    return max(16, int(round(est)))


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def comfy_audio_to_tensor(audio: dict) -> tuple[torch.Tensor, int]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    wav = waveform[0].detach().float().cpu()
    if wav.dim() > 1:
        wav = wav.mean(dim=0)
    return wav, sample_rate


def tensor_audio_to_comfy(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> dict:
    audio = audio.detach().float().cpu().clamp(-1.0, 1.0)
    return {"waveform": audio.view(1, 1, -1).contiguous(), "sample_rate": int(sample_rate)}


def encode_reference_audio(codec_model, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Encode a float waveform into codec codes [frames, 16] (int, cpu)."""
    device = next(codec_model.parameters()).device
    dtype = next(codec_model.parameters()).dtype
    if sample_rate != int(codec_model.config.input_sample_rate):
        import torchaudio.functional as AF

        wav = AF.resample(wav, sample_rate, int(codec_model.config.input_sample_rate))
    values = wav.to(device=device, dtype=dtype).unsqueeze(0)
    mask = torch.ones_like(values, dtype=torch.long)
    encoded = codec_model.encode(values, mask, return_dict=True)
    codes = encoded.audio_codes[0]
    return codes.detach().to(torch.int16).cpu().contiguous()


def decode_codes(codec_model, codes: torch.Tensor) -> torch.Tensor:
    """Decode codec codes [frames, 16] into a float waveform at 24 kHz."""
    device = next(codec_model.parameters()).device
    codes = codes.to(device=device, dtype=torch.long).unsqueeze(0)
    decoded = codec_model.decode(codes, return_dict=True)
    audio = decoded.audio_values[0]
    while audio.dim() > 1:
        audio = audio[0]
    return audio.detach().float().cpu()

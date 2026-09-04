"""ComfyUI nodes for Breeze TTS 2."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import torch

from . import loader
from . import native
from . import runtime
from .loader import ATTENTION_OPTIONS, DECODE_MODE_OPTIONS, DEVICE_OPTIONS, DTYPE_OPTIONS, REPO_CHOICES

logger = logging.getLogger("BreezeTTS2")

CATEGORY = "Breeze TTS 2"
PROGRESS_UNITS_PER_GENERATION = 1000

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


def get_model_choices() -> list[str]:
    return list(REPO_CHOICES.keys())


def _text_input(default: str, tooltip: str) -> tuple:
    return ("STRING", {"default": default, "multiline": True, "tooltip": tooltip})


def _generation_controls() -> dict:
    return {
        "max_new_tokens": (
            "INT",
            {
                "default": 1500,
                "min": 64,
                "max": 3000,
                "step": 8,
                "tooltip": "Maximum audio frames to generate (12.5 frames per second of speech; the model stops at EOS by itself).",
            },
        ),
        "temperature": (
            "FLOAT",
            {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Backbone sampling temperature."},
        ),
        "top_k": ("INT", {"default": 50, "min": 0, "max": 1024, "tooltip": "Backbone top-k (0 disables)."}),
        "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Backbone top-p (1.0 disables)."}),
        "repetition_penalty": (
            "FLOAT",
            {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "HF-style repetition penalty on generated backbone tokens."},
        ),
        "depth_temperature": (
            "FLOAT",
            {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Depth decoder (codebook 1-15) sampling temperature."},
        ),
        "depth_top_k": ("INT", {"default": 50, "min": 0, "max": 1024, "tooltip": "Depth decoder top-k (0 disables)."}),
        "depth_top_p": (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Depth decoder top-p (1.0 disables)."},
        ),
        "seed": (
            "INT",
            {"default": 42, "min": 0, "max": 2**31 - 1, "tooltip": "0 uses the current random state. A positive value is repeatable."},
        ),
    }


def _cfg_input(default: float, tooltip: str) -> tuple:
    return (
        "FLOAT",
        {"default": default, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": tooltip},
    )


class BreezeTTS2LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    get_model_choices(),
                    {
                        "default": loader.HYBRID_LABEL,
                        "tooltip": (
                            "int8 hybrid: backbone + text encoder INT8, depth decoder bf16 — bf16 speed at 5.5 GiB (recommended).\n"
                            "bf16: no quantization — best quality, 7.5 GiB.\n"
                            "int8: all transformer linears INT8 — smallest at 5.2 GiB, ~60% slower decode.\n"
                            "int8 text encoder only: decode path stays bf16 — 6.8 GiB."
                        ),
                    },
                ),
                "dtype": (
                    DTYPE_OPTIONS,
                    {"default": "auto", "tooltip": "auto picks bf16 on supporting GPUs, fp32 otherwise."},
                ),
                "device": (DEVICE_OPTIONS, {"default": "auto", "tooltip": "auto uses ComfyUI's active torch device."}),
                "attention": (
                    ATTENTION_OPTIONS,
                    {"default": "auto", "tooltip": "auto uses flash_attention_2 when flash_attn is installed, else sdpa."},
                ),
                "decode_mode": (
                    DECODE_MODE_OPTIONS,
                    {
                        "default": "eager",
                        "tooltip": (
                            "cuda_graphs captures the depth decode loop into CUDA graphs: much faster, but the "
                            "model weights stay fully resident in VRAM (AIMDO paging is bypassed for it) and the "
                            "first generation spends a couple of seconds capturing the graphs."
                        ),
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Download the selected checkpoint from Hugging Face when missing locally."},
                ),
            }
        }

    RETURN_TYPES = ("BREEZE_TTS2_MODEL",)
    RETURN_NAMES = ("breeze_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load a Breeze TTS 2 checkpoint with dtype, device, and attention selection."

    def load(self, model, dtype, device, attention, decode_mode, download_if_missing):
        bundle = loader.load_breeze_bundle(model, dtype, device, attention, bool(download_if_missing), decode_mode)
        return (bundle,)


def _generate_audio(
    bundle,
    *,
    text: str,
    instruction: str,
    ref_audio: dict | None,
    ref_text: str | None,
    cfg_scale: float,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    depth_temperature: float,
    depth_top_k: int,
    depth_top_p: float,
    seed: int,
    ref_codes: torch.Tensor | None = None,
    progress_callback=None,
    progress_label: str | None = None,
) -> dict:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")
    bundle = loader.ensure_bundle_ready(bundle)
    device = bundle.device

    if ref_codes is None and ref_audio is not None:
        if not ref_text or not ref_text.strip():
            raise ValueError("Reference audio requires its exact transcript (ref_text).")
        wav, sample_rate = runtime.comfy_audio_to_tensor(ref_audio)
        if wav.numel() == 0:
            raise ValueError("Reference audio is empty.")
        ref_codes = runtime.encode_reference_audio(bundle.codec, wav, sample_rate)
        ref_seconds = ref_codes.shape[0] / runtime.FRAMES_PER_SECOND
        if ref_seconds > runtime.MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"Reference audio is {ref_seconds:.0f}s; the maximum is {runtime.MAX_REFERENCE_SECONDS:.0f}s. "
                "Breeze spends prompt budget at 12.5 tokens per second of reference, so a long clip "
                "leaves no room to speak. Trim the clip and provide its exact transcript."
            )

    if ref_codes is not None:
        cond = runtime.ref_segments(ref_text.strip(), text, instruction)
        negative = runtime.ref_segments(ref_text.strip(), text, instruction, with_instruction=False)
    else:
        cond = runtime.design_segments(text, instruction)
        negative = runtime.design_negative_segments(text)

    runtime.fix_seed(int(seed))
    inputs_embeds, attention_mask, base_positions, prefill_len = runtime.build_generation_batch(
        bundle.model,
        bundle.tokenizer,
        cond_segments=cond,
        negative_segments=negative if cfg_scale != 1.0 else None,
        ref_codes=ref_codes,
        cfg_scale=float(cfg_scale),
        device=device,
    )
    max_frames = min(int(max_new_tokens), runtime.MAX_SEQ_LEN - 1 - prefill_len)
    if max_frames < 64:
        raise ValueError(
            f"The prompt alone is {prefill_len} tokens of the {runtime.MAX_SEQ_LEN}-token context, "
            f"leaving only {max_frames} audio frames. Use a shorter reference clip or shorter text."
        )
    params = runtime.GenerationParams(
        max_new_tokens=max_frames,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        depth_temperature=float(depth_temperature),
        depth_top_k=int(depth_top_k),
        depth_top_p=float(depth_top_p),
    )

    est_frames = min(runtime.estimate_speech_frames(bundle.tokenizer, text), max_frames)
    logger.info(
        "Prompt: %d tokens | approx %ds of speech (%d frames, cap %d) | cfg %.1f",
        prefill_len, est_frames / runtime.FRAMES_PER_SECOND, est_frames, max_frames, float(cfg_scale),
    )

    pbar = ProgressBar(max_frames) if (ProgressBar is not None and progress_callback is None) else None
    show_cli = tqdm is not None and (progress_callback is None or progress_label)
    cli_pbar = (
        tqdm(
            total=est_frames,
            desc=progress_label or "Breeze TTS 2 (~%.0fs)" % (est_frames / runtime.FRAMES_PER_SECOND),
            unit="frame", dynamic_ncols=True, leave=True,
        )
        if show_cli else None
    )

    def on_frame_progress(current: int) -> None:
        if pbar is not None:
            pbar.update_absolute(min(current, max_frames), max_frames)
        if cli_pbar is not None and current > cli_pbar.n:
            cli_pbar.update(current - cli_pbar.n)
        if progress_callback is not None:
            progress_callback(current, max_frames)

    try:
        with torch.inference_mode(), native.attention_runtime(bundle.attention):
            codes = runtime.generate_codes(
                bundle.model,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                base_positions=base_positions,
                prefill_len=prefill_len,
                cfg_scale=float(cfg_scale),
                params=params,
                progress_callback=on_frame_progress,
                decode_mode=bundle.decode_mode,
            )
        wav = runtime.decode_codes(bundle.codec, codes)
    finally:
        if cli_pbar is not None:
            cli_pbar.total = cli_pbar.n
            cli_pbar.close()

    if not bool(torch.isfinite(wav).all()):
        raise RuntimeError("Breeze TTS 2 generated non-finite audio samples.")
    if wav.numel() == 0:
        raise RuntimeError("Breeze TTS 2 produced no audio.")
    return runtime.tensor_audio_to_comfy(wav)


def _stitch_reference(audio: dict, ref_audio: dict, mode: str) -> dict:
    if mode == "none":
        return audio
    ref_wav, ref_sr = runtime.comfy_audio_to_tensor(ref_audio)
    if ref_wav.numel() == 0:
        return audio
    sample_rate = int(audio["sample_rate"])
    if ref_sr != sample_rate:
        import torchaudio.functional as AF

        ref_wav = AF.resample(ref_wav, ref_sr, sample_rate)
    generated = audio["waveform"].view(-1)
    parts = (ref_wav, generated) if mode == "before" else (generated, ref_wav)
    stitched = torch.cat(parts).clamp(-1.0, 1.0)
    return {"waveform": stitched.view(1, 1, -1).contiguous(), "sample_rate": sample_rate}


class BreezeTTS2VoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "breeze_model": ("BREEZE_TTS2_MODEL",),
            "text": _text_input(
                "(sigh) It is good to hear your voice again after all this time.",
                "Text to speak. Vocal events like (laugh) (sigh) (cough) (clears throat) work inline; use [笑] [叹气] etc. in Chinese.",
            ),
            "reference_audio": ("AUDIO", {"tooltip": "Clean reference speech to clone timbre, rhythm, and style from."}),
            "reference_text": _text_input(
                "This is the exact transcript of the reference audio.",
                "The exact transcript of the reference audio.",
            ),
            "cfg_scale": _cfg_input(1.0, "Guidance scale. 1.0 clones the reference as-is; raise it to push away from the reference."),
        }
        required.update(_generation_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = CATEGORY
    DESCRIPTION = "Clone a speaker from clean reference audio and its exact transcript."

    def clone(self, breeze_model, text, reference_audio, reference_text, cfg_scale, **controls):
        audio = _generate_audio(
            breeze_model,
            text=text,
            instruction=runtime.DEFAULT_INSTRUCTION,
            ref_audio=reference_audio,
            ref_text=reference_text,
            cfg_scale=cfg_scale,
            **controls,
        )
        return (audio,)


class BreezeTTS2VoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "breeze_model": ("BREEZE_TTS2_MODEL",),
            "text": _text_input(
                "(sigh) Welcome aboard. Your journey begins now.",
                "Text to speak. Vocal events like (laugh) (sigh) (cough) (clears throat) work inline; use [笑] [叹气] etc. in Chinese.",
            ),
            "instruction": _text_input(
                "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery.",
                "Natural-language description of the voice to create. Match the instruction language to the text language.",
            ),
            "cfg_scale": _cfg_input(4.0, "Guidance scale. 4 is recommended for instruction-following."),
        }
        required.update(_generation_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "design"
    CATEGORY = CATEGORY
    DESCRIPTION = "Create a voice from a natural-language description, without reference audio."

    def design(self, breeze_model, text, instruction, cfg_scale, **controls):
        audio = _generate_audio(
            breeze_model,
            text=text,
            instruction=instruction,
            ref_audio=None,
            ref_text=None,
            cfg_scale=cfg_scale,
            **controls,
        )
        return (audio,)


class BreezeTTS2VoiceDirection:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "breeze_model": ("BREEZE_TTS2_MODEL",),
            "text": _text_input(
                "(clears throat) We need to discuss what happened last night.",
                "Text to speak. Vocal events like (laugh) (sigh) (cough) (clears throat) work inline; use [笑] [叹气] etc. in Chinese.",
            ),
            "reference_audio": ("AUDIO", {"tooltip": "Reference speech whose speaker identity is kept."}),
            "reference_text": _text_input(
                "This is the exact transcript of the reference audio.",
                "The exact transcript of the reference audio.",
            ),
            "instruction": _text_input(
                "Speak slowly with a restrained, serious tone.",
                "Direction for tone, emotion, pace, and delivery applied on top of the cloned voice.",
            ),
            "cfg_scale": _cfg_input(4.0, "Guidance scale. 4 is recommended for instruction-following."),
            "stitch_reference": (
                ["none", "before", "after"],
                {
                    "default": "none",
                    "tooltip": (
                        "Stitch the original reference clip into the output audio: "
                        "'none' returns the generated speech only, 'before' plays the reference clip first, "
                        "'after' appends it at the end. Purely an output edit; generation is unchanged."
                    ),
                },
            ),
        }
        required.update(_generation_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "direct"
    CATEGORY = CATEGORY
    DESCRIPTION = "Clone a voice from reference audio while steering tone, emotion, pace, and delivery."

    def direct(self, breeze_model, text, reference_audio, reference_text, instruction, cfg_scale, stitch_reference="none", **controls):
        audio = _generate_audio(
            breeze_model,
            text=text,
            instruction=instruction,
            ref_audio=reference_audio,
            ref_text=reference_text,
            cfg_scale=cfg_scale,
            **controls,
        )
        return (_stitch_reference(audio, reference_audio, stitch_reference),)


# --------------------------------------------------------------------------- #
# Multi-speaker dialogue
# --------------------------------------------------------------------------- #
MAX_SPEAKERS = 8
NO_AUDIO = "none"

MULTI_SPEAKER_DEFAULT_TEXT = (
    "Alice: Hey Bob, did you finish recording the demo?\n"
    "Bob: (sigh) Almost. I just need one more take.\n"
    "Alice: (laugh) That is what you said an hour ago!"
)

_SPEAKER_LINE_RE = re.compile(r"^\s*(?:\[\s*([^\[\]:]{1,64}?)\s*\]|([^\[\]:]{1,64}?))\s*:\s*(.*)$")
_BOLD_SPEAKER_RE = re.compile(r"^\s*\*\*\s*([^*]{1,64}?)\s*:?\s*\*\*\s*:?\s*(.*)$")
_SPEAKER_KEYS = ("speaker", "name", "character", "role")
_TEXT_KEYS = ("text", "line", "content", "message", "dialogue")
_WRAPPER_KEYS = ("script", "dialogue", "dialog", "turns", "lines", "conversation")


def _normalize_speaker_name(name: Any) -> str:
    return " ".join(str(name).strip().strip("*_").split()).lower()


def _compact_speaker_key(name: Any) -> str:
    """Aggressive match key: lowercase letters/digits only, so 'Ali G' == 'alig' == 'Ali-G'."""
    return re.sub(r"[\W_]+", "", str(name).lower())


def _parse_script_plain(text: str) -> list[tuple[str, str]]:
    turns: list[list] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        # LLM-written scripts often bold the name: **Bob:** ... / **Bob**: ...
        bold = _BOLD_SPEAKER_RE.match(line)
        if bold:
            name, content = bold.group(1), (bold.group(2) or "")
        else:
            match = _SPEAKER_LINE_RE.match(line)
            if not match:
                if line.strip():
                    if not turns:
                        raise ValueError("Script must start with a speaker line like 'Alice: your text'.")
                    turns[-1][1].append(line.strip())
                continue
            name = match.group(1) or match.group(2)
            content = match.group(3)
        name = name.strip().strip("*_").strip()
        content = content.strip()
        turns.append([name, [content] if content else []])
    return [(name, " ".join(parts).strip()) for name, parts in turns if " ".join(parts).strip()]


def _parse_script_json(payload: str) -> list[tuple[str, str]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Script looks like JSON but failed to parse ({exc}). "
            'Expected a list like [{"speaker": "Alice", "text": "Hi!"}] or plain Name: lines.'
        ) from exc
    if isinstance(data, dict):
        for key in _WRAPPER_KEYS:
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError('JSON script must be a list of {"speaker": ..., "text": ...} entries.')
    turns: list[tuple[str, str]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"JSON turn {index + 1} must be an object with speaker and text fields.")
        name = next((item[k] for k in _SPEAKER_KEYS if item.get(k) is not None), None)
        content = next((item[k] for k in _TEXT_KEYS if item.get(k) is not None), None)
        if name is None or content is None:
            raise ValueError(
                f"JSON turn {index + 1} needs a speaker and a text field; got keys: {sorted(item.keys())}."
            )
        name, content = str(name).strip(), str(content).strip()
        if name and content:
            turns.append((name, content))
    return turns


def _parse_script(text: str) -> list[tuple[str, str]]:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Script cannot be empty.")
    looks_like_bracket_name = re.match(r"^\s*\[[^\]]+\]\s*:", stripped)
    if stripped[0] in "[{" and not looks_like_bracket_name:
        turns = _parse_script_json(stripped)
    else:
        turns = _parse_script_plain(stripped)
    if not turns:
        raise ValueError(
            "No dialogue turns found. Use 'Name: line' per line or a JSON list of "
            '{"speaker": "Alice", "text": "Hi!"} entries.'
        )
    return turns


def _speaker_audio_options() -> list[str]:
    try:
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio"])
        return [NO_AUDIO] + sorted(files)
    except Exception:
        return [NO_AUDIO]


def _load_audio_file(path: str) -> dict:
    """Decode an audio file from the input folder into a ComfyUI AUDIO dict."""
    errors: list[str] = []
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T.copy())
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return {"waveform": wav.unsqueeze(0).contiguous(), "sample_rate": int(sample_rate)}
    except Exception as exc:
        errors.append(f"soundfile: {exc}")
    try:
        import torchaudio

        wav, sample_rate = torchaudio.load(path)
        wav = wav.detach().float()
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return {"waveform": wav.unsqueeze(0).contiguous(), "sample_rate": int(sample_rate)}
    except Exception as exc:
        errors.append(f"torchaudio: {exc}")
    try:
        import av

        container = av.open(path)
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("no audio stream in file")
        frames: list[torch.Tensor] = []
        sample_rate = 0
        for frame in container.decode(stream):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.dtype == torch.int16:
                buf = buf.float() / 32768.0
            elif buf.dtype == torch.int32:
                buf = buf.float() / 2147483648.0
            elif buf.dtype == torch.uint8:
                buf = (buf.float() - 128.0) / 128.0
            else:
                buf = buf.float()
            n_channels = stream.channels
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)
            sample_rate = int(frame.sample_rate or sample_rate)
        container.close()
        if not frames:
            raise ValueError("no audio frames decoded")
        wav = torch.cat(frames, dim=1)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return {"waveform": wav.unsqueeze(0).contiguous(), "sample_rate": int(sample_rate)}
    except Exception as exc:
        errors.append(f"av: {exc}")
    raise RuntimeError(f"Could not decode audio file '{path}'. " + "; ".join(errors))


def _concat_audio_segments(segments: list[dict], pause_seconds: float) -> dict:
    if not segments:
        raise RuntimeError("No audio segments were generated.")
    sample_rate = int(segments[0]["sample_rate"])
    pause_samples = int(max(0.0, float(pause_seconds)) * sample_rate)
    silence = torch.zeros((1, 1, pause_samples), dtype=torch.float32) if pause_samples > 0 else None
    parts: list[torch.Tensor] = []
    for segment in segments:
        if int(segment["sample_rate"]) != sample_rate:
            raise RuntimeError("Generated turns have mismatched sample rates.")
        waveform = segment["waveform"]
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        if parts and silence is not None:
            parts.append(silence)
        parts.append(waveform.detach().float().cpu())
    return {"waveform": torch.cat(parts, dim=-1).contiguous(), "sample_rate": sample_rate}


class BreezeTTS2Speaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": (
                    "STRING",
                    {"default": "Alice", "tooltip": "Speaker name the script uses ('Alice: hello'). Case-insensitive."},
                ),
                "audio": (
                    "COMBO",
                    {
                        "multiselect": False,
                        "options": _speaker_audio_options(),
                        "audio_upload": True,
                        "tooltip": (
                            "Reference clip from the ComfyUI input folder — click to browse or drag-drop a file onto the node. "
                            "Select 'none' to design the voice from the instruction instead."
                        ),
                    },
                ),
                "reference_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Exact transcript of the reference clip. Leave empty to auto-transcribe with Whisper "
                            "(auto_transcribe_if_empty). Can be converted to an input to wire the Whisper Transcribe node."
                        ),
                    },
                ),
                "auto_transcribe_if_empty": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Transcribe the reference clip with whisper-large-v3-turbo when reference_text is empty. "
                            "Check the console log: a wrong transcript hurts cloning."
                        ),
                    },
                ),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Voice description. Required when no reference audio is selected (designs the voice). "
                            "With reference audio it steers tone, emotion, pace, and delivery (voice direction)."
                        ),
                    },
                ),
                "cfg_scale": _cfg_input(
                    1.0,
                    "Guidance for this speaker. 1.0 keeps a cloned voice as-is; around 4.0 is recommended for "
                    "designed voices and strong directions.",
                ),
            },
            "optional": {
                "reference_audio": (
                    "AUDIO",
                    {"tooltip": "Optional wired audio (e.g. from a Load Audio node). Overrides the file dropdown when connected."},
                ),
            },
        }

    RETURN_TYPES = ("BREEZE_SPEAKER", "AUDIO", "STRING")
    RETURN_NAMES = ("speaker", "audio", "transcript")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Define one named speaker for the Multi-Speaker node: clone the reference clip, or design the voice "
        "from the instruction when no clip is selected."
    )

    def prepare(self, name, audio, reference_text, auto_transcribe_if_empty, instruction, cfg_scale, reference_audio=None):
        name = (name or "").strip()
        if not name:
            raise ValueError("Speaker name cannot be empty; the script uses it as 'Name: line'.")

        audio_dict = None
        if reference_audio is not None:
            audio_dict = reference_audio
        elif audio and audio != NO_AUDIO:
            import folder_paths

            path = folder_paths.get_annotated_filepath(audio)
            audio_dict = _load_audio_file(path)
            logger.info("Speaker '%s': loaded reference audio '%s'.", name, audio)

        transcript = (reference_text or "").strip()
        auto_used = False
        if audio_dict is not None and not transcript and bool(auto_transcribe_if_empty):
            from . import whisper as whisper_asr

            _passthrough, transcript = whisper_asr.BreezeWhisperTranscribe().transcribe(
                audio_dict, "whisper-large-v3-turbo", "auto", "auto", "transcribe", 30, True
            )
            transcript = transcript.strip()
            auto_used = bool(transcript)
            logger.info("Speaker '%s' auto-transcript (whisper-large-v3-turbo): %s", name, transcript)

        if audio_dict is not None and not transcript:
            raise ValueError(
                f"Speaker '{name}' has reference audio but no transcript. "
                "Type the exact transcript or enable auto_transcribe_if_empty."
            )
        if audio_dict is None and not (instruction or "").strip():
            raise ValueError(
                f"Speaker '{name}' has no reference audio, so the instruction designs the voice — it cannot be empty."
            )

        speaker = {
            "name": name,
            "ref_audio": audio_dict,
            "ref_text": transcript,
            "instruction": (instruction or "").strip(),
            "cfg_scale": float(cfg_scale),
        }
        preview = audio_dict
        if preview is None:
            preview = {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": runtime.SAMPLE_RATE}

        if audio_dict is None:
            ui_text = f"{name}: designed voice (instruction only, no reference audio)"
        else:
            label = "auto-transcript (whisper)" if auto_used else "transcript"
            ui_text = f"{name} — {label}:\n{transcript}"
        # Widgets can't be written to from execution, so the transcript shows
        # as read-only display text on the node (and stays on the output pin).
        return {"ui": {"text": [ui_text]}, "result": (speaker, preview, transcript)}

    @classmethod
    def VALIDATE_INPUTS(cls, audio, **kwargs):
        if audio in (None, "", NO_AUDIO):
            return True
        import folder_paths

        if not folder_paths.exists_annotated_filepath(audio):
            return f"Invalid audio file: {audio}"
        return True

    @classmethod
    def IS_CHANGED(cls, audio, reference_audio=None, **kwargs):
        if reference_audio is not None or audio in (None, "", NO_AUDIO):
            return ""
        import folder_paths

        path = folder_paths.get_annotated_filepath(audio)
        if not os.path.isfile(path):
            return ""
        stat = os.stat(path)
        return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"


def _multi_speaker_audio(
    bundle,
    *,
    text: str,
    speakers: list,
    pause_between_speakers: float,
    seed: int,
    controls: dict,
) -> dict:
    bundle = loader.ensure_bundle_ready(bundle)
    turns = _parse_script(text)

    cast: dict[str, tuple[int, dict]] = {}
    cast_compact: dict[str, str | None] = {}
    for slot, speaker in enumerate(speakers):
        if speaker is None:
            continue
        key = _normalize_speaker_name(speaker.get("name", ""))
        if not key:
            raise ValueError(f"Speaker in slot {slot + 1} has an empty name.")
        if key in cast:
            raise ValueError(
                f"Duplicate speaker name '{speaker.get('name')}' (slots {cast[key][0] + 1} and {slot + 1})."
            )
        cast[key] = (slot, speaker)
        compact = _compact_speaker_key(key)
        if compact:
            if compact in cast_compact:
                if cast_compact[compact] != key:
                    cast_compact[compact] = None  # two cast members share this key; never guess
            else:
                cast_compact[compact] = key
    if not cast:
        raise ValueError("No speakers connected. Wire at least one Breeze TTS 2 Speaker node.")

    def resolve_speaker(name: str) -> str | None:
        key = _normalize_speaker_name(name)
        if key in cast:
            return key
        return cast_compact.get(_compact_speaker_key(name)) or None

    unknown = sorted({name for name, _line in turns if resolve_speaker(name) is None})
    if unknown:
        available = ", ".join(speaker["name"] for _slot, speaker in cast.values())
        raise ValueError(f"Script uses unknown speaker(s): {', '.join(unknown)}. Cast: {available}.")

    turn_keys = [resolve_speaker(name) for name, _line in turns]
    for (name, _line), key in zip(turns, turn_keys):
        if key != _normalize_speaker_name(name):
            logger.info("Script speaker '%s' matched to cast member '%s'.", name, cast[key][1]["name"])
    used = set(turn_keys)
    cast_summary = ", ".join(
        f"{speaker['name']} ({'clone' if speaker['ref_audio'] is not None else 'design'})"
        for _slot, speaker in cast.values()
        if _normalize_speaker_name(speaker["name"]) in used
    )
    turn_estimates = [runtime.estimate_speech_frames(bundle.tokenizer, line) for _name, line in turns]
    logger.info(
        "Multi-speaker script: %d speaker(s), %d turn(s), ~%.0fs of speech total | cast: %s",
        len(used),
        len(turns),
        sum(turn_estimates) / runtime.FRAMES_PER_SECOND,
        cast_summary,
    )

    ref_codes: dict[str, torch.Tensor] = {}
    for key in used:
        _slot, speaker = cast[key]
        if speaker["ref_audio"] is None:
            continue
        wav, sample_rate = runtime.comfy_audio_to_tensor(speaker["ref_audio"])
        if wav.numel() == 0:
            raise ValueError(f"Speaker '{speaker['name']}' reference audio is empty.")
        codes = runtime.encode_reference_audio(bundle.codec, wav, sample_rate)
        ref_seconds = codes.shape[0] / runtime.FRAMES_PER_SECOND
        if ref_seconds > runtime.MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"Speaker '{speaker['name']}' reference clip is {ref_seconds:.0f}s; the maximum is "
                f"{runtime.MAX_REFERENCE_SECONDS:.0f}s. Trim the clip."
            )
        ref_codes[key] = codes

    total_units = len(turns) * PROGRESS_UNITS_PER_GENERATION
    pbar = ProgressBar(total_units) if ProgressBar is not None else None
    segments: list[dict] = []
    for index, (name, line) in enumerate(turns):
        key = turn_keys[index]
        slot, speaker = cast[key]
        instruction = speaker["instruction"] or runtime.DEFAULT_INSTRUCTION
        # Seed is offset per speaker slot (not per turn) so a designed voice
        # stays anchored to the same seed every time that speaker talks.
        turn_seed = seed + slot if seed else 0
        progress_label = (
            f"[{index + 1}/{len(turns)}] {speaker['name']} "
            f"(~{turn_estimates[index] / runtime.FRAMES_PER_SECOND:.0f}s)"
        )

        def update_turn(current: int, total: int, turn_index: int = index) -> None:
            if pbar is None:
                return
            fraction = min(1.0, max(0.0, float(current) / max(1, int(total))))
            pbar.update_absolute(
                turn_index * PROGRESS_UNITS_PER_GENERATION + round(fraction * PROGRESS_UNITS_PER_GENERATION),
                total_units,
            )

        segments.append(
            _generate_audio(
                bundle,
                text=line,
                instruction=instruction,
                ref_audio=None,
                ref_text=speaker["ref_text"],
                cfg_scale=speaker["cfg_scale"],
                seed=turn_seed,
                ref_codes=ref_codes.get(key),
                progress_callback=update_turn,
                progress_label=progress_label,
                **controls,
            )
        )
        if pbar is not None:
            pbar.update_absolute((index + 1) * PROGRESS_UNITS_PER_GENERATION, total_units)

    return _concat_audio_segments(segments, pause_between_speakers if len(segments) > 1 else 0.0)


class BreezeTTS2MultiSpeaker:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "breeze_model": ("BREEZE_TTS2_MODEL",),
            "text": _text_input(
                MULTI_SPEAKER_DEFAULT_TEXT,
                "Dialogue script: one 'Name: line' per line (a line without a name continues the previous speaker; "
                "'[Name]:' also works), or paste a JSON list like [{\"speaker\": \"Alice\", \"text\": \"Hi!\"}] — "
                "handy for LLM-written dialogue. Inline vocal events like (laugh) (sigh) (cough) or [笑] work in both "
                "formats. The widget can be converted to an input to wire a string node.",
            ),
            "pause_between_speakers": (
                "FLOAT",
                {"default": 0.3, "min": 0.0, "max": 3.0, "step": 0.05, "tooltip": "Seconds of silence between dialogue turns."},
            ),
        }
        required.update(_generation_controls())
        optional = {}
        for index in range(1, MAX_SPEAKERS + 1):
            optional[f"speaker_{index}"] = (
                "BREEZE_SPEAKER",
                {"tooltip": f"Cast slot {index}. Wire a Breeze TTS 2 Speaker node here."},
            )
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Generate multi-speaker dialogue. Each script turn is spoken by its named speaker; "
        "speakers can be cloned voices, designed voices, or a mix."
    )

    def generate(
        self,
        breeze_model,
        text,
        pause_between_speakers,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        depth_temperature,
        depth_top_k,
        depth_top_p,
        seed,
        **kwargs,
    ):
        controls = {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "depth_temperature": float(depth_temperature),
            "depth_top_k": int(depth_top_k),
            "depth_top_p": float(depth_top_p),
        }
        speakers = [kwargs.get(f"speaker_{index}") for index in range(1, MAX_SPEAKERS + 1)]
        audio = _multi_speaker_audio(
            breeze_model,
            text=text,
            speakers=speakers,
            pause_between_speakers=float(pause_between_speakers),
            seed=int(seed),
            controls=controls,
        )
        return (audio,)


NODE_CLASS_MAPPINGS = {
    "BreezeTTS2LoadModel": BreezeTTS2LoadModel,
    "BreezeTTS2VoiceClone": BreezeTTS2VoiceClone,
    "BreezeTTS2VoiceDesign": BreezeTTS2VoiceDesign,
    "BreezeTTS2VoiceDirection": BreezeTTS2VoiceDirection,
    "BreezeTTS2Speaker": BreezeTTS2Speaker,
    "BreezeTTS2MultiSpeaker": BreezeTTS2MultiSpeaker,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BreezeTTS2LoadModel": "Breeze TTS 2 Load Model",
    "BreezeTTS2VoiceClone": "Breeze TTS 2 Voice Clone",
    "BreezeTTS2VoiceDesign": "Breeze TTS 2 Voice Design",
    "BreezeTTS2VoiceDirection": "Breeze TTS 2 Voice Direction",
    "BreezeTTS2Speaker": "Breeze TTS 2 Speaker",
    "BreezeTTS2MultiSpeaker": "Breeze TTS 2 Multi-Speaker",
}

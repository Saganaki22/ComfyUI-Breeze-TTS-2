from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn


def test_codec_load_does_not_force_backbone_resident(loader, monkeypatch):
    codec = nn.Linear(2, 2)
    backbone_patcher = SimpleNamespace(model=nn.Linear(2, 2))
    codec_patcher = SimpleNamespace(model=codec)
    bundle = SimpleNamespace(codec=codec, patchers=[backbone_patcher, codec_patcher])
    load = Mock()
    monkeypatch.setattr(loader.mm, "load_models_gpu", load)

    assert loader.resume_codec_to_device(bundle)
    load.assert_called_once_with([codec_patcher], force_full_load=True)


def test_codec_load_failure_is_not_hidden(loader, monkeypatch):
    codec = nn.Linear(2, 2)
    bundle = SimpleNamespace(codec=codec, patchers=[SimpleNamespace(model=codec)])
    monkeypatch.setattr(loader.mm, "load_models_gpu", Mock(side_effect=RuntimeError("not enough memory")))

    with pytest.raises(RuntimeError, match="not enough memory"):
        loader.resume_codec_to_device(bundle)


def test_runtime_uses_codec_execution_device(runtime):
    class Codec(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(1, device="meta"))
            self.device = torch.device("cpu")
            self.config = SimpleNamespace(input_sample_rate=24000)

        def encode(self, values, mask, return_dict):
            assert values.device == self.device
            return SimpleNamespace(audio_codes=torch.zeros(1, 2, 16, dtype=torch.long))

        def decode(self, codes, return_dict):
            assert codes.device == self.device
            return SimpleNamespace(audio_values=torch.ones(1, 1, 240))

    codec = Codec()
    assert runtime.encode_reference_audio(codec, torch.zeros(240), 24000).shape == (2, 16)
    assert runtime.decode_codes(codec, torch.zeros(2, 16, dtype=torch.long)).shape == (240,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for partial offload")
def test_real_comfy_partial_offload_then_codec_encode_decode(loader, native, runtime):
    class Codec(nn.Module):
        def __init__(self):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(4))
            self.register_buffer("offset", torch.zeros(1, 4, 1), persistent=False)
            self.conv = nn.Conv1d(1, 4, 3, padding=1)
            self.linear = native._ComfyLinear(4, 4)
            self.config = SimpleNamespace(input_sample_rate=24000)

        def features(self, values):
            hidden = self.conv(values)
            hidden = self.linear(hidden.transpose(1, 2)).transpose(1, 2)
            return hidden * self.gamma[None, :, None] + self.offset

        def encode(self, values, mask, return_dict):
            features = self.features(values.unsqueeze(1))
            codes = (features.mean(1) > 0).long().unsqueeze(-1).expand(-1, -1, 16)
            return SimpleNamespace(audio_codes=codes)

        def decode(self, codes, return_dict):
            values = codes[:, :, 0].float().unsqueeze(1)
            return SimpleNamespace(audio_values=self.features(values).mean(1, keepdim=True))

    codec = Codec().eval()
    patcher = loader.register_runtime_module(codec, torch.device("cuda:0"), dynamic=False)
    bundle = SimpleNamespace(codec=codec, patchers=[patcher])
    try:
        freed = patcher.partially_unload(torch.device("cpu"), memory_to_free=patcher.model_size())
        assert freed > 0
        assert codec.conv.weight.device.type == "cpu"
        assert codec.gamma.device.type == "cuda"
        with pytest.raises(RuntimeError):
            runtime.encode_reference_audio(codec, torch.zeros(240), 24000)

        assert loader.resume_codec_to_device(bundle)
        assert all(p.device.type == "cuda" for p in codec.parameters())
        assert codec.offset.device.type == "cuda"
        codes = runtime.encode_reference_audio(codec, torch.zeros(240), 24000)
        audio = runtime.decode_codes(codec, codes)
        assert codes.shape == (240, 16)
        assert audio.shape == (240,)
        assert audio.device.type == "cpu"
        assert torch.isfinite(audio).all()
    finally:
        loader._unregister_from_comfy(patcher)


def test_reference_encode_resumes_speech_model(breeze, monkeypatch):
    nodes = breeze.nodes
    bundle = SimpleNamespace(codec=object(), model=object(), device=torch.device("cpu"))
    events = []
    monkeypatch.setattr(nodes.loader, "ensure_bundle_ready", lambda bundle: bundle)
    monkeypatch.setattr(nodes.loader, "resume_codec_to_device", lambda bundle: events.append("codec"))
    monkeypatch.setattr(nodes.loader, "resume_bundle_to_device", lambda bundle: events.append("backbone"))
    monkeypatch.setattr(nodes.runtime, "encode_reference_audio", lambda *args: torch.zeros(2, 16))

    def stop_before_generation(*args, **kwargs):
        events.append("prompt")
        raise RuntimeError("stop before generation")

    monkeypatch.setattr(nodes.runtime, "ref_segments", stop_before_generation)
    with pytest.raises(RuntimeError, match="stop before generation"):
        nodes._generate_audio(
            bundle, text="Hello", instruction="", ref_audio={"waveform": torch.zeros(1, 1, 240), "sample_rate": 24000},
            ref_text="Hello", cfg_scale=1, max_new_tokens=64, temperature=0.9, top_k=50,
            top_p=1, repetition_penalty=1.1, depth_temperature=0.9, depth_top_k=50,
            depth_top_p=1, seed=1,
        )
    assert events == ["codec", "backbone", "prompt"]


@pytest.mark.parametrize("uses_reference", [False, True])
def test_multi_speaker_only_prepares_codec_for_used_references(breeze, monkeypatch, uses_reference):
    nodes = breeze.nodes
    bundle = SimpleNamespace(codec=object(), tokenizer=object())
    audio = {"waveform": torch.zeros(1, 1, 240), "sample_rate": 24000}
    speakers = [
        dict(name="Alice", ref_audio=audio if uses_reference else None, ref_text="Hello", instruction="", cfg_scale=1),
        dict(name="Unused", ref_audio=audio, ref_text="Hello", instruction="", cfg_scale=1),
    ]
    events = []
    monkeypatch.setattr(nodes.loader, "ensure_bundle_ready", lambda bundle: bundle)
    monkeypatch.setattr(nodes.loader, "resume_codec_to_device", lambda bundle: events.append("codec"))
    monkeypatch.setattr(nodes.runtime, "estimate_speech_frames", lambda *args: 2)

    def encode(*args):
        events.append("encode")
        return torch.zeros(2, 16)

    def generate(*args, **kwargs):
        assert (kwargs["ref_codes"] is not None) == uses_reference
        events.append("generate")
        return audio

    monkeypatch.setattr(nodes.runtime, "encode_reference_audio", encode)
    monkeypatch.setattr(nodes, "_generate_audio", generate)
    nodes._multi_speaker_audio(bundle, text="Alice: Hello", speakers=speakers,
                               pause_between_speakers=0, seed=1, controls={})
    assert events == (["codec", "encode", "generate"] if uses_reference else ["generate"])

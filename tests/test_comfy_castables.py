"""Exercise Breeze shims against ComfyUI's cast and DynamicVRAM paths."""

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def make_module(native, kind):
    if kind == "linear":
        return nn.Linear(8, 12), torch.randn(2, 3, 8)
    if kind == "embedding":
        return nn.Embedding(16, 8), torch.tensor([[0, 5, 15], [1, 3, 9]])
    if kind == "breeze_norm":
        return native.BreezeRMSNorm(8), torch.randn(2, 3, 8)
    if kind == "qwen_norm":
        return native.Qwen3RMSNorm(8), torch.randn(2, 3, 8)
    from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2RMSNorm

    return T5Gemma2RMSNorm(8), torch.randn(2, 3, 8)


KINDS = ["linear", "embedding", "breeze_norm", "qwen_norm", "gemma_norm"]


@pytest.mark.parametrize("kind", KINDS)
def test_conversion_preserves_parameters_and_output(native, kind):
    module, inputs = make_module(native, kind)
    expected = module(inputs)
    parameters = dict(module.named_parameters())
    keys = list(module.state_dict())
    native.convert_modules_for_comfy(module)
    native.convert_modules_for_comfy(module)
    assert list(module.state_dict()) == keys
    assert all(dict(module.named_parameters())[name] is value for name, value in parameters.items())
    torch.testing.assert_close(module(inputs), expected)


@pytest.mark.parametrize("kind", KINDS)
def test_vbar_cache_miss_casts_cpu_weights_for_cuda_inputs(native, monkeypatch, kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise cross-device VBAR casting")
    import comfy.ops as ops

    module, inputs = make_module(native, kind)
    inputs = inputs.cuda()
    expected = copy.deepcopy(module).cuda()(inputs)
    native.convert_modules_for_comfy(module)
    # Emulate a VBAR cache miss with synchronous transfers; ComfyUI still
    # performs the real host-to-CUDA transfer, weight casting and operation.
    module._v = object()
    module._v_signature = None
    module._pin_state = {"fast_disk": False}
    monkeypatch.setattr(ops.comfy_aimdo.model_vbar, "vbar_fault", lambda _: None)
    monkeypatch.setattr(ops.comfy_aimdo.model_vbar, "vbar_signature_compare", lambda *_: False)
    monkeypatch.setattr(ops.comfy.model_management, "DISABLE_PINNED_MEMORY", True)
    monkeypatch.setattr(ops.comfy.model_management, "get_offload_stream", lambda _: None)
    torch.testing.assert_close(module(inputs), expected)
    torch.cuda.synchronize()
    assert module.weight.device.type == "cpu"
    assert not hasattr(module, "_prefetch")


def test_audio_offsets_follow_cuda_inputs(native):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for stranded-buffer coverage")
    config = SimpleNamespace(hidden_size=8, num_codebooks=2, vocab_size=16)
    module = native.BreezeBackboneModelEmbeddings(config).cuda()
    inputs = torch.tensor([[[0, 1], [3, 5]]], device="cuda")
    expected = module(inputs)
    module.audio_tokens_offsets = module.audio_tokens_offsets.cpu()

    torch.testing.assert_close(module(inputs), expected)
    assert module.audio_tokens_offsets.device.type == "cpu"


def test_convnext_gamma_follows_cuda_activations(breeze):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for stranded-parameter coverage")
    codec = importlib.import_module(f"{breeze.__name__}.vendor.codec_model")
    block = codec.Qwen3TTSTokenizerV2ConvNeXtBlock(8).cuda()
    inputs = torch.randn(1, 8, 16, device="cuda")
    expected = block(inputs)
    block.gamma = nn.Parameter(block.gamma.detach().cpu())

    torch.testing.assert_close(block(inputs), expected)
    assert block.gamma.device.type == "cpu"

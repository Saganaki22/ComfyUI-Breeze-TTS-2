import importlib

import pytest
import torch


@pytest.fixture
def codec_module(breeze):
    return importlib.import_module(f"{breeze.__name__}.vendor.codec_model")


@pytest.fixture
def tiny_mimi_config(codec_module):
    return codec_module.MimiConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_filters=2,
        upsampling_ratios=[2, 2],
        sampling_rate=80,
        frame_rate=10,
        upsample_groups=16,
        codebook_dim=4,
        vector_quantization_hidden_dimension=4,
        codebook_size=16,
        num_quantizers=4,
        num_semantic_quantizers=1,
        sliding_window=8,
        use_cache=False,
    )


def test_encoder_keeps_flat_checkpoint_layout(codec_module, tiny_mimi_config):
    encoder = codec_module.Qwen3TTSTokenizerV2Encoder._from_config(tiny_mimi_config)
    keys = set(encoder.state_dict())
    assert "encoder.layers.0.conv.weight" in keys
    assert "encoder_transformer.layers.0.self_attn.q_proj.weight" in keys
    assert "downsample.conv.weight" in keys
    assert "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum" in keys
    assert not any(key.startswith(("decoder.", "upsample.", "encoder.encoder_seanet.")) for key in keys)


def test_encoder_matches_upstream_mimi_codes(codec_module, tiny_mimi_config):
    upstream = importlib.import_module("transformers.models.mimi.modeling_mimi")
    reference = upstream.MimiModel._from_config(tiny_mimi_config).eval()
    encoder = codec_module.Qwen3TTSTokenizerV2Encoder._from_config(tiny_mimi_config).eval()
    # Random codebook vectors make nearest-neighbor selection nontrivial.
    for module in reference.modules():
        if isinstance(module, upstream.MimiEuclideanCodebook):
            module.embed_sum.normal_()
    reference_state = {}
    for key, value in reference.state_dict().items():
        if key.startswith(("encoder.", "encoder_transformer.", "downsample.", "quantizer.")):
            key = key.replace("encoder.encoder_seanet.", "encoder.")
            key = key.replace("encoder.encoder_transformer.", "encoder_transformer.")
            key = key.replace("encoder.downsample.", "downsample.")
            reference_state[key] = value
    encoder.load_state_dict(reference_state, strict=True)
    audio = torch.linspace(-0.5, 0.5, 80).reshape(1, 1, -1)
    expected = reference.encode(audio, return_dict=True).audio_codes
    actual = encoder.encode(audio, return_dict=True).audio_codes
    assert actual.shape == (1, 4, 10)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(encoder.encode(audio, return_dict=False)[0], expected)

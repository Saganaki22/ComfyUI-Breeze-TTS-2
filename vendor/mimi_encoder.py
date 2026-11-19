"""Self-contained flat-layout Mimi encoder for the Breeze TTS 2 audio tokenizer.

The checkpoint (`audio_tokenizer/model.safetensors`) stores the encoder half of
the tokenizer with the classic flat Mimi key layout:

    encoder.encoder.layers.*                    SEANet encoder
    encoder.encoder_transformer.layers.*        transformer
    encoder.downsample.conv.weight              latent downsampler
    encoder.quantizer.semantic_residual_vector_quantizer.*
    encoder.quantizer.acoustic_residual_vector_quantizer.*

`transformers.MimiModel` used to build exactly that tree, so the tokenizer
encoder could simply subclass it. Transformers 5.17 split Mimi into
`MimiEncoderModel` / `MimiDecoderModel` halves behind read-only `upsample` /
`downsample` properties, which broke that subclass twice over: assigning the
unused decoder halves raised
``AttributeError: property 'upsample' ... has no setter``, and the new module
tree (`encoder_seanet`, `encoder_transformer` under a nested model, ...) no
longer matched any checkpoint key.

So the encoder is vendored here instead, keeping the flat key layout and
touching only stable transformers surface (ACT2FN, PreTrainedModel,
ModelOutput). Two deliberate deviations from upstream, both to stop this file
from chasing transformers internals:

* RoPE inverse frequencies are computed locally (upstream reads
  `config.rope_parameters`, `config.rope_theta` or a `ROPE_INIT_FUNCTIONS`
  table depending on the release) for the `default` rope type only, which is
  all this config uses.
* The sliding-window causal mask is built here instead of via
  `masking_utils.create_sliding_window_causal_mask`, whose signature and
  accepted kwargs keep changing (see the compat dance in `codec_model.py`),
  and attention is always eager: the encoder runs fp32 on <= a few thousand
  12 Hz frames, where the difference is noise and reproducibility matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import MimiConfig
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

logger = logging.get_logger(__name__)


@dataclass
class MimiEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, codes_length)`):
        Discrete code embeddings computed by the encoder + quantizer stack.
    """

    audio_codes: Optional[torch.LongTensor] = None


class MimiConv1d(nn.Module):
    """Conv1d with asymmetric (causal) padding, as used by the Mimi SEANet."""

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        pad_mode: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode if pad_mode is None else pad_mode
        self.in_channels = in_channels

        if stride > 1 and dilation > 1:
            logger.warning(
                "MimiConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias
        )

        # Plain python ints rather than buffers: derived-only, never saved, and
        # this keeps them off the meta device under init_empty_weights.
        conv_kernel = self.conv.kernel_size[0]
        self.stride = self.conv.stride[0]
        self.kernel_size = (conv_kernel - 1) * self.conv.dilation[0] + 1
        self.padding_total = self.kernel_size - self.stride
        self.padding_right = self.padding_total // 2
        self.padding_left = self.padding_total - self.padding_right

    def _get_extra_padding_for_conv1d(self, hidden_states: torch.Tensor) -> int:
        length = hidden_states.shape[-1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + self.kernel_size - self.padding_total
        return ideal_length - length

    @staticmethod
    def _pad1d(hidden_states: torch.Tensor, paddings: tuple, mode: str = "constant", value: float = 0.0):
        """Like F.pad, but tolerates reflect padding on inputs shorter than the pad."""
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if mode != "reflect":
            return nn.functional.pad(hidden_states, paddings, mode, value)

        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
        padded = nn.functional.pad(hidden_states, paddings, mode, value)
        return padded[..., : padded.shape[-1] - extra_pad]

    def _get_output_length(self, input_length):
        n_frames = (input_length - self.kernel_size + self.padding_total) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + self.kernel_size - self.padding_total
        extra_padding = ideal_length - input_length

        if self.causal:
            padding_left = self.padding_total
            padding_right = extra_padding
        else:
            padding_left = self.padding_left
            padding_right = self.padding_right + extra_padding

        input_length = input_length + padding_left + padding_right
        conv = self.conv
        return (input_length + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        extra_padding = self._get_extra_padding_for_conv1d(hidden_states)

        if self.causal:
            hidden_states = self._pad1d(hidden_states, (self.padding_total, extra_padding), mode=self.pad_mode)
        else:
            hidden_states = self._pad1d(
                hidden_states, (self.padding_left, self.padding_right + extra_padding), mode=self.pad_mode
            )

        return self.conv(hidden_states)


class MimiResnetBlock(nn.Module):
    """Residual block from the SEANet encoder used by Mimi."""

    def __init__(self, config, dim: int, dilations: list):
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU(), MimiConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)]
        self.block = nn.ModuleList(block)

        self.shortcut = (
            MimiConv1d(config, dim, dim, kernel_size=1) if config.use_conv_shortcut else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        for layer in self.block:
            hidden_states = layer(hidden_states)
        return self.shortcut(residual) + hidden_states


class MimiEncoder(nn.Module):
    """SEANet encoder as used by Mimi (downsamples by `upsampling_ratios`)."""

    def __init__(self, config):
        super().__init__()
        model = [MimiConv1d(config, config.audio_channels, config.num_filters, config.kernel_size)]
        scaling = 1

        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            for j in range(config.num_residual_layers):
                model += [MimiResnetBlock(config, current_scale, [config.dilation_growth_rate**j, 1])]
            model += [nn.ELU(), MimiConv1d(config, current_scale, current_scale * 2, kernel_size=ratio * 2, stride=ratio)]
            scaling *= 2

        model += [nn.ELU(), MimiConv1d(config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size)]
        self.layers = nn.ModuleList(model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MimiLayerScale(nn.Module):
    """Per-channel learnable residual rescale (Touvron et al. 2021)."""

    def __init__(self, config):
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((config.hidden_size,), config.layer_scale_initial_scale, requires_grad=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


def _rope_inv_freq(config, device=None) -> torch.Tensor:
    """Inverse rotary frequencies for the default RoPE type.

    Reads whichever spelling of the rope setting this transformers release
    leaves on the config (`rope_parameters` dict, `rope_theta`, or neither).
    """
    rope_params = getattr(config, "rope_parameters", None)
    base = None
    if isinstance(rope_params, dict):
        base = rope_params.get("rope_theta")
        rope_type = rope_params.get("rope_type", "default")
    else:
        rope_type = getattr(rope_params, "rope_type", "default")
    if base is None:
        base = getattr(config, "rope_theta", 10000.0)
    if rope_type not in (None, "default"):
        raise ValueError(
            f"Vendored Mimi encoder only supports the default rope type, got {rope_type!r}."
        )

    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class MimiRotaryEmbedding(nn.Module):
    """RoPE cos/sin tables, computed per call from the config (no saved state)."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = _rope_inv_freq(self.config, device=x.device)
        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].to(dtype=torch.float)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MimiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Additive mask allowing `kv <= q` and `kv > q - window`. Mirrors
# `masking_utils.sliding_window_causal_mask_function` from transformers, which
# is what the installed Mimi (5.3+) applies in its encoder transformer.
def _sliding_window_causal_mask(
    n_query: int, n_key: int, window: int, device, dtype
) -> torch.Tensor:
    q = torch.arange(n_query, device=device)[:, None]
    kv = torch.arange(n_key, device=device)[None, :]
    allowed = (kv <= q) & (kv > q - window)
    return torch.zeros(n_query, n_key, device=device, dtype=dtype).masked_fill_(~allowed, torch.finfo(dtype).min)


class MimiAttention(nn.Module):
    """Grouped-query sliding-window causal attention, eager."""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.scaling = 1 / math.sqrt(config.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query_states = (query_states * cos) + (_rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (_rotate_half(key_states) * sin)

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class MimiTransformerLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = MimiAttention(config=config, layer_idx=layer_idx)
        self.mlp = MimiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn_layer_scale = MimiLayerScale(config)
        self.mlp_layer_scale = MimiLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + self.mlp_layer_scale(hidden_states)


class MimiTransformerModel(nn.Module):
    """Encoder transformer: pre-norm layers, no final norm (matches the checkpoint)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [MimiTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = MimiRotaryEmbedding(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = _sliding_window_causal_mask(
            seq_len, seq_len, self.config.sliding_window, hidden_states.device, hidden_states.dtype
        )
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )
        return hidden_states


class MimiEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config, epsilon: float = 1e-5):
        super().__init__()
        self.register_buffer("initialized", torch.tensor([True], dtype=torch.float32))
        self.register_buffer("cluster_usage", torch.ones(config.codebook_size))
        self.register_buffer("embed_sum", torch.zeros(config.codebook_size, config.codebook_dim))
        self.epsilon = epsilon

    @property
    def embed(self) -> torch.Tensor:
        # Derived from the two saved buffers, so it survives weight loading and
        # device moves without a stale cache.
        return self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        dists = torch.cdist(hidden_states[None].float(), self.embed[None].float(), p=2)[0]
        return dists.argmin(dim=-1).view(*shape[:-1])

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(embed_ind, self.embed)


class MimiVectorQuantization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codebook = MimiEuclideanCodebook(config)

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.codebook.encode(hidden_states.permute(0, 2, 1))

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return self.codebook.decode(embed_ind).permute(0, 2, 1)


class MimiResidualVectorQuantizer(nn.Module):
    def __init__(self, config, num_quantizers: Optional[int] = None):
        super().__init__()
        self.num_quantizers = num_quantizers if num_quantizers is not None else config.num_quantizers
        self.layers = nn.ModuleList([MimiVectorQuantization(config) for _ in range(self.num_quantizers)])

        self.input_proj = None
        self.output_proj = None
        if config.vector_quantization_hidden_dimension != config.hidden_size:
            self.input_proj = nn.Conv1d(config.hidden_size, config.vector_quantization_hidden_dimension, 1, bias=False)
            self.output_proj = nn.Conv1d(config.vector_quantization_hidden_dimension, config.hidden_size, 1, bias=False)

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None) -> torch.Tensor:
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)

        num_quantizers = self.num_quantizers if num_quantizers is None else num_quantizers

        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            residual = residual - layer.decode(indices)
            all_indices.append(indices)
        return torch.stack(all_indices)

class MimiSplitResidualVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_num_quantizers = config.num_quantizers
        self.num_semantic_quantizers = config.num_semantic_quantizers
        self.num_acoustic_quantizers = config.num_quantizers - config.num_semantic_quantizers

        self.semantic_residual_vector_quantizer = MimiResidualVectorQuantizer(config, self.num_semantic_quantizers)
        self.acoustic_residual_vector_quantizer = MimiResidualVectorQuantizer(config, self.num_acoustic_quantizers)

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None) -> torch.Tensor:
        num_quantizers = self.max_num_quantizers if num_quantizers is None else num_quantizers
        if num_quantizers > self.max_num_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be lower than the total number of quantizers {self.max_num_quantizers}, but is currently {num_quantizers}."
            )
        if num_quantizers < self.num_semantic_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be higher than the number of semantic quantizers {self.num_semantic_quantizers}, but is currently {num_quantizers}."
            )

        # codes is [K, B, T], with T frames, K nb of codebooks.
        codes = self.semantic_residual_vector_quantizer.encode(embeddings)
        if num_quantizers > self.num_semantic_quantizers:
            acoustic_codes = self.acoustic_residual_vector_quantizer.encode(
                embeddings, num_quantizers=num_quantizers - self.num_semantic_quantizers
            )
            codes = torch.cat([codes, acoustic_codes], dim=0)
        return codes


class MimiPreTrainedModel(PreTrainedModel):
    """Minimal pretrained base: the encoder is loaded from the flat checkpoint keys.

    `MimiConfig` is only used as the config schema here (never instantiated), so
    importing it is safe across the releases this pack supports.
    """

    config_class = MimiConfig
    base_model_prefix = "mimi"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MimiTransformerLayer"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _can_compile_fullgraph = False
    _supports_attention_backend = False


class MimiEncoderModel(MimiPreTrainedModel):
    """Flat-layout Mimi encoder half: SEANet -> transformer -> downsample -> quantizer.

    Module names mirror the `encoder.*` keys of the audio_tokenizer checkpoint:
    `encoder`, `encoder_transformer`, `downsample`, `quantizer`.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = MimiEncoder(config)
        self.encoder_transformer = MimiTransformerModel(config)

        self.downsample = None
        if config.frame_rate != config.encodec_frame_rate:
            self.downsample = MimiConv1d(
                config,
                config.hidden_size,
                config.hidden_size,
                kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
                stride=2,
                bias=False,
                pad_mode="replicate",
            )

        self.quantizer = MimiSplitResidualVectorQuantizer(config)

        bits_per_codebook = int(math.log2(config.codebook_size))
        if 2**bits_per_codebook != config.codebook_size:
            raise ValueError("The codebook_size must be a power of 2.")

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(
        self,
        input_values: torch.Tensor,
        num_quantizers: Optional[int] = None,
        return_dict: Optional[bool] = None,
    ):
        """Encode `(B, 1, samples)` audio into codes of shape `(B, num_quantizers, frames)`."""
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        if self.downsample is None:
            raise ValueError("This encoder config has no latent downsampler; cannot encode codes.")

        hidden_states = self.encoder(input_values)
        hidden_states = self.encoder_transformer(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.downsample(hidden_states)
        audio_codes = self.quantizer.encode(hidden_states, num_quantizers).transpose(0, 1)

        if not return_dict:
            return (audio_codes,)
        return MimiEncoderOutput(audio_codes=audio_codes)


__all__ = [
    "MimiConv1d",
    "MimiEncoder",
    "MimiEncoderModel",
    "MimiEncoderOutput",
    "MimiSplitResidualVectorQuantizer",
    "MimiTransformerModel",
]

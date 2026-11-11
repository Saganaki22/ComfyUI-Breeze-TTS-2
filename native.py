"""Vendored Breeze TTS 2 inference model.

Model code adapted from breezeblue-ai/breeze-tts (Apache-2.0); config classes
originally (c) Sesame and The HuggingFace Inc. team. Trimmed to the inference
path: the training forward, loss handling, the unused in-checkpoint Mimi codec,
gradient checkpointing, and the HF generation mixin are removed.

The backbone uses native transformers Qwen3 layers; the text encoder is the
native transformers T5Gemma2 encoder (keys verified identical to the
checkpoint). The depth decoder is the vendored Breeze implementation.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.auto import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg
from safetensors import safe_open

# transformers >=5 ships a native T5Gemma2 encoder; older versions fall back to
# the vendored upstream compat implementation. The native one is flagged
# _supports_flash_attn=False, but register_te_flash_attention() below wires it
# into transformers' FA2 path under a custom name.
try:
    from transformers.models.t5gemma2.modeling_t5gemma2 import (
        T5Gemma2RotaryEmbedding,
        T5Gemma2TextEncoder,
        T5Gemma2TextScaledWordEmbedding,
    )
    from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextConfig

    _TE_NATIVE = True
except ImportError:
    from .vendor.t5gemma2_compat import (
        T5Gemma2RotaryEmbedding,
        T5Gemma2TextConfig,
        T5Gemma2TextEncoder,
        T5Gemma2TextScaledWordEmbedding,
    )

    _TE_NATIVE = False

# create_causal_mask renamed its embedding kwarg from input_embeds (<5) to
# inputs_embeds (>=5); pick whichever this installation accepts.
_CAUSAL_MASK_EMBED_KEY = (
    "inputs_embeds"
    if "inputs_embeds" in inspect.signature(create_causal_mask).parameters
    else "input_embeds"
)

# transformers >=5.9 removed the cache_position kwarg from create_causal_mask;
# pass it only when this installation accepts it.
_CAUSAL_MASK_ACCEPTS_CACHE_POSITION = "cache_position" in inspect.signature(create_causal_mask).parameters


# Deliberately avoids the substring "flash_attention_2": transformers
# substring-matches it in get_correct_attn_implementation and would run the
# _supports_flash_attn gate that blocks T5Gemma2.
_TE_FA2_NAME = "breeze_te_fa2"


def register_te_flash_attention() -> None:
    """Register FA2 for the native T5Gemma2 encoder under a custom name.

    transformers blocks flash-attention on T5Gemma2 with
    ``_supports_flash_attn = False`` (a softcapping concern), but this
    checkpoint has ``attn_logit_softcapping = None`` and flash-attn handles
    its sliding-window / GQA / bidirectional layers natively. Dispatch goes
    through ``ALL_ATTENTION_FUNCTIONS.get_interface``, which never consults
    that flag, so a registered alias is enough. The matching mask factory
    hands FA2 the 2D padding mask it unpads internally.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _TE_FA2_NAME in ALL_ATTENTION_FUNCTIONS.keys():
        return
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    def breeze_te_fa2_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, **kwargs):
        # The T5Gemma2 encoder is bidirectional: causal stays off and the
        # sliding window is symmetric. transformers' own FA2 wrapper corrupts
        # right-padded batches for this model, so pad/unpad by hand.
        q = query.transpose(1, 2)  # [B, S, H, D]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        bsz, seqlen, heads, head_dim = q.shape
        window = (-1, -1)
        if sliding_window is not None and seqlen > sliding_window:
            window = (sliding_window - 1, sliding_window - 1)
        if attention_mask is None:
            out = flash_attn_func(q, k, v, dropout_p=dropout, softmax_scale=scaling, causal=False, window_size=window)
            return out, None
        mask = attention_mask.reshape(bsz, seqlen).bool()
        lengths = mask.sum(dim=1, dtype=torch.int32)
        cu_seqlens = torch.zeros(bsz + 1, dtype=torch.int32, device=q.device)
        cu_seqlens[1:] = lengths.cumsum(0)
        idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
        max_len = int(lengths.max())
        out = q.new_zeros(bsz * seqlen, heads, head_dim)
        out[idx] = flash_attn_varlen_func(
            q.reshape(bsz * seqlen, heads, head_dim)[idx],
            k.reshape(bsz * seqlen, *k.shape[2:])[idx],
            v.reshape(bsz * seqlen, *v.shape[2:])[idx],
            cu_seqlens, cu_seqlens,
            max_len, max_len,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=False,
            window_size=window,
        )
        return out.view(bsz, seqlen, heads, head_dim), None

    ALL_ATTENTION_FUNCTIONS.register(_TE_FA2_NAME, breeze_te_fa2_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(_TE_FA2_NAME, flash_attention_mask)

try:
    import comfy.ops as _comfy_ops

    _cast_bias_weight = _comfy_ops.cast_bias_weight
    _uncast_bias_weight = _comfy_ops.uncast_bias_weight
except Exception:

    def _cast_bias_weight(module, x, *args, **kwargs):
        return module.weight, module.bias, None

    def _uncast_bias_weight(module, weight, bias, stream=None):
        return None


logger = logging.getLogger("BreezeTTS2")


def _default_rope_parameters(config, device=None):
    """transformers <5 default RoPE; removed from ROPE_INIT_FUNCTIONS in 5.x."""
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    return inv_freq, 1.0


# --------------------------------------------------------------------------- #
# Configs (trimmed from breeze_base_config.py)
# --------------------------------------------------------------------------- #
class BreezeDepthDecoderConfig(PretrainedConfig):
    model_type = "breeze_depth_decoder_model"
    base_config_key = "depth_decoder_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_codebooks=32,
        backbone_hidden_size=2048,
        vocab_size=2051,
        hidden_size=1024,
        intermediate_size=8192,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=33,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        rope_theta=500000,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        audio_embed_size=None,
        **kwargs,
    ):
        kwargs.pop("tie_word_embeddings", False)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=False,
            **kwargs,
        )
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.backbone_hidden_size = backbone_hidden_size
        self.audio_embed_size = audio_embed_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads


class BreezeConfig(PretrainedConfig):
    model_type = "breeze"
    base_config_key = "breeze_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_codebooks=32,
        vocab_size=2051,
        text_vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=128002,
        codebook_pad_token_id=2050,
        codebook_eos_token_id=0,
        bos_token_id=128000,
        eos_token_id=None,
        audio_token_id=128002,
        audio_eos_token_id=128003,
        rope_theta=500000,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        tie_codebooks_embeddings=True,
        depth_decoder_config=None,
        **kwargs,
    ):
        kwargs.pop("tie_word_embeddings", False)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=False,
            **kwargs,
        )
        if isinstance(depth_decoder_config, dict):
            self.depth_decoder_config = BreezeDepthDecoderConfig(**depth_decoder_config)
        elif isinstance(depth_decoder_config, BreezeDepthDecoderConfig):
            self.depth_decoder_config = depth_decoder_config
        else:
            self.depth_decoder_config = BreezeDepthDecoderConfig()
        self.text_vocab_size = text_vocab_size
        self.num_codebooks = num_codebooks
        self.audio_token_id = audio_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.codebook_pad_token_id = codebook_pad_token_id
        self.codebook_eos_token_id = codebook_eos_token_id
        self.tie_codebooks_embeddings = tie_codebooks_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads


# Register the custom config types once at module import so AutoConfig (and
# AutoTokenizer, which resolves the config internally) can handle
# model_type "breeze". Without this, transformers >=5.16 falls back to a
# generic PreTrainedConfig whose RoPE standardization crashes on
# max_position_embeddings (PR #2, P2).
AutoConfig.register("breeze_depth_decoder_model", BreezeDepthDecoderConfig, exist_ok=True)
AutoConfig.register("breeze", BreezeConfig, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model blocks (from breeze.py)
# --------------------------------------------------------------------------- #
class BreezeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class BreezeRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: PretrainedConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type, _default_rope_parameters)
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _recompute(self, device=None):
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.inv_freq = inv_freq
        self.original_inv_freq = inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BreezeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class BreezeAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class BreezeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BreezeAttention(config=config, layer_idx=layer_idx)
        self.mlp = BreezeMLP(config)
        self.input_layernorm = BreezeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BreezeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class BreezeDepthDecoderModel(nn.Module):
    def __init__(self, config: BreezeDepthDecoderConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if getattr(config, "audio_embed_size", None):
            self.audio_embed_size = config.audio_embed_size
        else:
            self.audio_embed_size = config.backbone_hidden_size
        self.embed_tokens = nn.Embedding(config.num_codebooks * config.vocab_size, self.audio_embed_size)
        self.layers = nn.ModuleList(
            [BreezeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BreezeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BreezeRotaryEmbedding(config=config)
        self.inputs_embeds_projector = nn.Linear(self.audio_embed_size, config.hidden_size, bias=False)
        if config.backbone_hidden_size != self.audio_embed_size:
            self.backbone_hidden_state_projector = nn.Linear(
                config.backbone_hidden_size, self.audio_embed_size, bias=False
            )
        else:
            self.backbone_hidden_state_projector = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        backbone_last_hidden_state: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            inputs_seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_seq_length, device=device)
        if inputs_embeds is None:
            codebook_idxs = torch.clamp(cache_position - 1, min=0)
            offset = codebook_idxs * self.vocab_size
            inputs_embeds = self.embed_tokens(input_ids + offset)
            if backbone_last_hidden_state is not None:
                if self.backbone_hidden_state_projector is not None:
                    inputs_embeds[:, 0] = self.backbone_hidden_state_projector(backbone_last_hidden_state)
                else:
                    inputs_embeds[:, 0] = backbone_last_hidden_state
        inputs_embeds = self.inputs_embeds_projector(inputs_embeds)
        causal_mask = create_causal_mask(
            config=self.config,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **({"cache_position": cache_position} if _CAUSAL_MASK_ACCEPTS_CACHE_POSITION else {}),
            **{_CAUSAL_MASK_EMBED_KEY: inputs_embeds},
        )
        hidden_states = inputs_embeds
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class BreezeCodebooksHead(nn.Module):
    def __init__(self, hidden_size, num_codebooks, vocab_size):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.weight = nn.Parameter(torch.empty(self.num_codebooks - 1, hidden_size, vocab_size))

    def forward(self, hidden_states, cache_position=None):
        if cache_position is None:
            seq_length = hidden_states.shape[1]
            codebook_weight = self.weight[torch.arange(seq_length)]
        else:
            codebook_idxs = cache_position - 1
            codebook_weight = self.weight[codebook_idxs]
        hidden_states = [
            nn.functional.linear(
                hidden_states[:, codebook_idx, :], codebook_weight[codebook_idx].T
            )
            for codebook_idx in range(codebook_weight.shape[0])
        ]
        hidden_states = torch.stack(hidden_states, dim=1)
        return hidden_states

    def extra_repr(self):
        return f"weight_shape={list(self.weight.shape)}, num_codebooks={self.num_codebooks}"


class BreezeDepthDecoder(nn.Module):
    """Inference wrapper matching the checkpoint's depth_decoder.* key layout."""

    def __init__(self, config: BreezeDepthDecoderConfig):
        super().__init__()
        self.model = BreezeDepthDecoderModel(config)
        self.codebooks_head = BreezeCodebooksHead(config.hidden_size, config.num_codebooks, config.vocab_size)


class BreezeBackboneModelEmbeddings(nn.Module):
    def __init__(self, config: BreezeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        if getattr(config, "audio_embed_size", None):
            self.audio_embed_size = config.audio_embed_size
        else:
            self.audio_embed_size = self.hidden_size
        self.embed_audio_tokens = nn.Embedding(config.num_codebooks * config.vocab_size, self.hidden_size)
        if self.audio_embed_size != self.hidden_size:
            self.audio_embeds_projector = nn.Linear(self.audio_embed_size, self.hidden_size, bias=False)
        else:
            self.audio_embeds_projector = None
        self.register_buffer(
            "audio_tokens_offsets",
            torch.arange(config.num_codebooks) * config.vocab_size,
            persistent=False,
        )

    def forward(self, input_ids):
        input_embeds = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        if self.audio_embeds_projector:
            input_embeds = self.audio_embeds_projector(input_embeds)
        input_embeds = input_embeds.sum(dim=2)
        return input_embeds


class BreezeBackboneAdapter(nn.Module):
    """External Qwen3 backbone with Breeze audio-token embeddings (inference only)."""

    def __init__(self, config: BreezeConfig, layers, norm, rotary_emb):
        super().__init__()
        self.config = config
        if not hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
        self.embed_tokens = BreezeBackboneModelEmbeddings(config)
        self.layers = layers
        self.norm = norm
        self.rotary_emb = rotary_emb

    @classmethod
    def create_from_config(cls, config: BreezeConfig):
        backbone_config = dict(getattr(config, "backbone_config", None) or {})
        for junk in ("architectures", "torch_dtype", "transformers_version"):
            backbone_config.pop(junk, None)
        llm_config = AutoConfig.for_model(**backbone_config)
        llm_config._attn_implementation = getattr(config, "_attn_implementation", "eager")
        layers = nn.ModuleList([Qwen3DecoderLayer(llm_config, idx) for idx in range(llm_config.num_hidden_layers)])
        norm = Qwen3RMSNorm(llm_config.hidden_size, eps=llm_config.rms_norm_eps)
        rotary_emb = Qwen3RotaryEmbedding(config=llm_config)
        return cls(config, layers, norm, rotary_emb)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            **({"cache_position": cache_position} if _CAUSAL_MASK_ACCEPTS_CACHE_POSITION else {}),
            **{_CAUSAL_MASK_EMBED_KEY: inputs_embeds},
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class BreezeInferenceModel(nn.Module):
    """Breeze TTS 2 model trimmed to the inference path (no codec_model)."""

    def __init__(self, config: BreezeConfig, text_encoder: nn.Module):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.backbone_eos_token_id = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size + 1, bias=False)
        self.embed_text_tokens = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.backbone_model = BreezeBackboneAdapter.create_from_config(config)
        self.depth_decoder = BreezeDepthDecoder(config.depth_decoder_config)
        self.text_encoder = text_encoder
        self.text_encoder_proj = nn.Linear(
            config.text_encoder_config.hidden_size, config.hidden_size, bias=False
        )
        self.text_encoder_feature_layer_idx = (-1,)

    def _batched_text_encoder_forward(self, segments, output_hidden_states=False):
        if not segments:
            return [], []
        device = segments[0].device
        lengths = [s.shape[0] for s in segments]
        sorted_indices = sorted(range(len(segments)), key=lambda i: lengths[i])
        buckets = []
        current_bucket = []
        current_min_len = None
        for idx in sorted_indices:
            length = lengths[idx]
            if not current_bucket:
                current_bucket = [idx]
                current_min_len = length
            elif length / max(current_min_len, 1) <= 2:
                current_bucket.append(idx)
            else:
                buckets.append(current_bucket)
                current_bucket = [idx]
                current_min_len = length
        if current_bucket:
            buckets.append(current_bucket)
        hidden_states = [None] * len(segments)
        layer_hidden_states = [None] * len(segments) if output_hidden_states else None
        for bucket_indices in buckets:
            bucket_lengths = [lengths[i] for i in bucket_indices]
            max_len = max(bucket_lengths)
            padded_ids = torch.zeros(len(bucket_indices), max_len, dtype=segments[0].dtype, device=device)
            attn_mask = torch.zeros(len(bucket_indices), max_len, dtype=torch.long, device=device)
            pos_ids = torch.zeros(len(bucket_indices), max_len, dtype=torch.long, device=device)
            for bucket_pos, idx in enumerate(bucket_indices):
                length = lengths[idx]
                padded_ids[bucket_pos, :length] = segments[idx]
                attn_mask[bucket_pos, :length] = 1
                pos_ids[bucket_pos, :length] = torch.arange(length, device=device)
            output = self.text_encoder(
                input_ids=padded_ids,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                output_hidden_states=output_hidden_states,
            )
            full_hs = output.last_hidden_state
            for bucket_pos, idx in enumerate(bucket_indices):
                hidden_states[idx] = full_hs[bucket_pos, : lengths[idx]]
            if output_hidden_states is not None and getattr(output, "hidden_states", None) is not None:
                all_layer_hs = list(output.hidden_states)
                if layer_hidden_states is None:
                    layer_hidden_states = [None] * len(segments)
                for bucket_pos, idx in enumerate(bucket_indices):
                    layer_hidden_states[idx] = [lhs[bucket_pos, : lengths[idx]] for lhs in all_layer_hs]
        return hidden_states, layer_hidden_states

    def convert_input_ids_to_embeds(self, input_ids, text_ids_mask, text_ids_len):
        total_text_tokens = int(text_ids_mask.sum().item())
        assert total_text_tokens == int(text_ids_len.sum().item()), (
            f"text_ids_mask sum {total_text_tokens} does not match text_ids_len sum {int(text_ids_len.sum().item())}"
        )
        text_ids_len_list = [int(x) for x in text_ids_len.reshape(-1).tolist()]
        text_ids_len_idx = 0
        all_seg_token_ids = []
        for batch_idx in range(input_ids.shape[0]):
            mask_row = text_ids_mask[batch_idx]
            sample_text_tokens = int(mask_row.sum().item())
            if sample_text_tokens == 0:
                continue
            text_ids = input_ids[batch_idx][mask_row]
            segment_lengths = []
            running = 0
            while running < sample_text_tokens:
                if text_ids_len_idx >= len(text_ids_len_list):
                    raise ValueError("text_ids_len exhausted before covering all text tokens in the batch")
                seg_len = text_ids_len_list[text_ids_len_idx]
                text_ids_len_idx += 1
                if seg_len <= 0:
                    continue
                segment_lengths.append(seg_len)
                running += seg_len
            if running != sample_text_tokens:
                raise ValueError(
                    f"text_ids_len segments sum {running} does not match text_ids_mask sum {sample_text_tokens} "
                    f"for batch index {batch_idx}"
                )
            all_seg_token_ids.extend(text_ids.split(segment_lengths, dim=0))
        if text_ids_len_idx != len(text_ids_len_list):
            raise ValueError(
                f"Unused text_ids_len entries detected: consumed {text_ids_len_idx} of {len(text_ids_len_list)}"
            )
        if all_seg_token_ids:
            seg_hidden_states, _ = self._batched_text_encoder_forward(all_seg_token_ids)
            projected = [self.text_encoder_proj(hs.unsqueeze(0)).squeeze(0) for hs in seg_hidden_states]
            text_embeds = torch.cat(projected, dim=0)
        else:
            text_embeds = torch.empty(
                (0, self.config.hidden_size),
                device=input_ids.device,
                dtype=self.embed_text_tokens.weight.dtype,
            )
        inputs_embeds = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.config.hidden_size),
            dtype=text_embeds.dtype,
            device=text_embeds.device,
        )
        inputs_embeds[text_ids_mask] = text_embeds
        return inputs_embeds

    def merge_prompt(self, input_ids, text_ids_mask, text_ids_len, input_values):
        """Merge text-token embeddings with pre-encoded audio codes into one inputs_embeds tensor."""
        inputs_embeds = self.convert_input_ids_to_embeds(input_ids, text_ids_mask, text_ids_len)
        if input_values is not None:
            audio_token_mask = input_ids == self.config.audio_token_id
            num_audio_tokens = audio_token_mask.sum().item()
            if num_audio_tokens == 0:
                raise RuntimeError(
                    f"Audio token mismatch: expected audio_token_id={self.config.audio_token_id} in input_ids, "
                    f"but found 0 matches. input_values shape: {tuple(input_values.shape)}."
                )
            audio_embeds = self.backbone_model.embed_tokens(input_values)
            audio_embeds_flat = audio_embeds.reshape(-1, audio_embeds.shape[-1]).to(inputs_embeds.dtype)
            inputs_embeds[audio_token_mask] = audio_embeds_flat
            audio_eos_frame_ids = (
                torch.ones((1, 1, self.config.num_codebooks), device=input_ids.device, dtype=torch.long)
                * self.config.codebook_eos_token_id
            )
            audio_eos_embeds = self.backbone_model.embed_tokens(audio_eos_frame_ids).squeeze(1)
            audio_eos_embeds = audio_eos_embeds.to(inputs_embeds.dtype)
            audio_eos_token_mask = input_ids == self.config.audio_eos_token_id
            inputs_embeds[audio_eos_token_mask] = audio_eos_embeds.repeat(audio_eos_token_mask.sum(), 1)
        return inputs_embeds


# --------------------------------------------------------------------------- #
# ComfyUI-castable module shims (same approach as the FireRedTTS3 / Raon packs)
# --------------------------------------------------------------------------- #
class _ComfyLinear(nn.Linear):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []
    # ComfyUI 2026-11 combined-linear support expects this on custom castables.
    _combined_linear = None

    def forward(self, x):
        if not hasattr(self, "_v") and self.weight.device == x.device:
            return F.linear(x, self.weight, self.bias)
        weight, bias, stream = _cast_bias_weight(self, x, offloadable=True)
        try:
            return F.linear(x, weight, bias)
        finally:
            _uncast_bias_weight(self, weight, bias, stream)


class _ComfyEmbedding(nn.Embedding):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []
    bias = None
    _combined_linear = None

    def _weight_dtype(self):
        return getattr(self, "weight_comfy_model_dtype", None) or self.weight.dtype

    def forward(self, input):
        if not hasattr(self, "_v") and self.weight.device == input.device:
            return F.embedding(
                input, self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse,
            )
        weight, bias, stream = _cast_bias_weight(
            self, dtype=self._weight_dtype(), device=input.device, offloadable=True
        )
        try:
            return F.embedding(
                input, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse,
            )
        finally:
            _uncast_bias_weight(self, weight, bias, stream)


def _comfy_rmsnorm_forward(self, hidden_states):
    # BreezeRMSNorm / Qwen3RMSNorm math: fp32 variance, weight * normed
    input_dtype = hidden_states.dtype
    if not hasattr(self, "_v") and self.weight.device == hidden_states.device:
        weight = self.weight
        stream = None
        bias = None
    else:
        weight, bias, stream = _cast_bias_weight(self, hidden_states, offloadable=True)
    try:
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return weight * hidden_states.to(input_dtype)
    finally:
        _uncast_bias_weight(self, weight, bias, stream)


def _comfy_t5gemma2_rmsnorm_forward(self, hidden_states):
    # T5Gemma2RMSNorm math: fp32 norm, (1 + weight) gemma-style scaling
    if not hasattr(self, "_v") and self.weight.device == hidden_states.device:
        weight = self.weight
        stream = None
        bias = None
    else:
        weight, bias, stream = _cast_bias_weight(self, hidden_states, offloadable=True)
    try:
        output = hidden_states.float() * torch.rsqrt(
            hidden_states.float().pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        )
        output = output * (1.0 + weight.float())
        return output.type_as(hidden_states)
    finally:
        _uncast_bias_weight(self, weight, bias, stream)


def _patch_rmsnorm(module: nn.Module, forward_fn) -> None:
    if getattr(module, "_breeze_comfy_cast", False):
        return
    if not hasattr(module, "variance_epsilon"):
        module.variance_epsilon = module.eps
    module.bias = None
    module.comfy_cast_weights = True
    module.weight_function = []
    module.bias_function = []
    module._combined_linear = None
    module.forward = forward_fn.__get__(module, module.__class__)
    module._breeze_comfy_cast = True


def convert_modules_for_comfy(model: nn.Module) -> None:
    """Patch castable modules in-place so DynamicVRAM can page their weights."""
    for module in model.modules():
        if isinstance(module, (_ComfyLinear, _ComfyEmbedding)):
            continue
        if isinstance(module, nn.Linear):
            module.__class__ = _ComfyLinear
        elif type(module) is nn.Embedding:
            module.__class__ = _ComfyEmbedding
        elif isinstance(module, BreezeRMSNorm) or module.__class__.__name__ == "Qwen3RMSNorm":
            _patch_rmsnorm(module, _comfy_rmsnorm_forward)
        elif module.__class__.__name__ == "T5Gemma2RMSNorm":
            _patch_rmsnorm(module, _comfy_t5gemma2_rmsnorm_forward)


def set_runtime_dtype(module: nn.Module, dtype: torch.dtype) -> None:
    """Tag floating tensors with the dtype Comfy/AIMDO should materialize them in.

    INT8 ConvRot weights are never tagged (not floating) and per-row weight
    scales stay fp32 so the quantized kernels receive exact scales.
    """
    for sub in module.modules():
        for name, value in sub.named_parameters(recurse=False):
            if (
                value is not None
                and value.is_floating_point()
                and not name.endswith("inv_freq")
                and not name.endswith("weight_scale")
            ):
                setattr(sub, f"{name}_comfy_model_dtype", dtype)
        for name, value in sub.named_buffers(recurse=False):
            if (
                value is not None
                and value.is_floating_point()
                and not name.endswith("inv_freq")
                and not name.endswith("weight_scale")
            ):
                setattr(sub, f"{name}_comfy_model_dtype", dtype)


# --------------------------------------------------------------------------- #
# Weight loading
# --------------------------------------------------------------------------- #
def read_config(model_dir: Path) -> dict[str, Any]:
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {model_dir}.")
    return json.loads(config_path.read_text(encoding="utf-8"))


def iter_checkpoint_items(path: Path) -> Iterable[tuple[str, torch.Tensor]]:
    path = Path(path)
    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            shards = sorted(set(weight_map.values()))
        elif (path / "model.safetensors").is_file():
            shards = ["model.safetensors"]
        else:
            raise FileNotFoundError(f"No model.safetensors or index in {path}.")
        files = [path / shard for shard in shards]
    else:
        files = [path]
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def _set_tensor(module: nn.Module, name: str, tensor: torch.Tensor, dtype: torch.dtype | None) -> None:
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    try:
        from accelerate.utils.modeling import set_module_tensor_to_device

        # dtype is passed explicitly: with dtype=None accelerate would re-cast
        # the value to the (fp32) dtype the meta parameter was declared with.
        set_module_tensor_to_device(module, name, device="cpu", value=tensor.contiguous(), dtype=tensor.dtype)
        return
    except ImportError:
        pass
    target = dict(module.named_parameters(remove_duplicate=False)).get(name)
    if target is None:
        target = dict(module.named_buffers(remove_duplicate=False)).get(name)
    if target is None:
        raise KeyError(name)
    if target.shape != tensor.shape:
        raise ValueError(
            f"Shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
        )
    target.data = tensor.contiguous()


def build_breeze_model(config_dict: dict[str, Any], attn_implementation: str) -> BreezeInferenceModel:
    """Build the inference model on the meta device from a config.json dict."""
    cfg = dict(config_dict)
    codec_codebook_size = int((cfg.get("codec_config") or {}).get("codebook_size", 2048))
    cfg["codec_codebook_size"] = codec_codebook_size
    te_dict = dict(cfg.pop("text_encoder_config", None) or {})
    cfg.pop("codec_config", None)
    cfg.pop("codec_model", None)
    for junk in ("architectures", "transformers_version", "dtype", "torch_dtype"):
        te_dict.pop(junk, None)
    backbone_dict = dict(cfg.pop("backbone_config", None) or {})
    for junk in ("architectures", "torch_dtype", "transformers_version"):
        backbone_dict.pop(junk, None)

    config = BreezeConfig(backbone_config=backbone_dict, **cfg)

    te_config = T5Gemma2TextConfig(**te_dict)
    if not _TE_NATIVE:
        # 4.x warns about FA2 without a declared dtype; weights are cast at load anyway
        te_config.torch_dtype = torch.bfloat16
    if _TE_NATIVE and attn_implementation == "flash_attention_2":
        # transformers flags the native T5Gemma2 encoder as no-FA2; the custom
        # registration above routes it through transformers' own FA2 path.
        register_te_flash_attention()
        te_config._attn_implementation = _TE_FA2_NAME
    else:
        te_config._attn_implementation = attn_implementation
    config._attn_implementation = attn_implementation
    # The depth decoder attends with custom per-position masks over a static
    # cache; transformers' FA2 varlen path cannot consume those. It runs M<=2
    # steps where the backend is irrelevant, so it always uses sdpa.
    config.depth_decoder_config._attn_implementation = "sdpa"
    config.text_encoder_config = te_config

    import accelerate

    with accelerate.init_empty_weights():
        text_encoder = T5Gemma2TextEncoder(te_config)
        text_encoder.eval()
        model = BreezeInferenceModel(config, text_encoder)
    return model


def load_breeze_weights(
    model: BreezeInferenceModel,
    checkpoint: Path,
    dtype_policy=None,
) -> None:
    """Stream every tensor from checkpoint into model, casting floats per dtype_policy.

    codec_model.* tensors from the official checkpoint are not part of the
    inference model and are skipped. The backbone audio-token embedding is tied
    to the depth decoder embedding and loaded once.
    """
    param_names = set(dict(model.named_parameters(remove_duplicate=False)))
    buffer_names = set(dict(model.named_buffers(remove_duplicate=False)))
    loaded: set[str] = set()
    unexpected: list[str] = []
    quant_meta = 0
    for name, tensor in iter_checkpoint_items(checkpoint):
        if name.endswith(".comfy_quant"):
            quant_meta += 1
            continue
        if name.startswith("codec_model."):
            continue
        if name not in param_names and name not in buffer_names:
            unexpected.append(name)
            continue
        target_dtype = dtype_policy(name) if dtype_policy is not None else None
        _set_tensor(model, name, tensor, target_dtype)
        loaded.add(name)

    tied = "backbone_model.embed_tokens.embed_audio_tokens.weight"
    if model.config.tie_codebooks_embeddings and tied not in loaded:
        model.backbone_model.embed_tokens.embed_audio_tokens.weight = (
            model.depth_decoder.model.embed_tokens.weight
        )
        loaded.add(tied)

    missing = [name for name in param_names if name not in loaded]
    if missing:
        raise RuntimeError(
            f"Weights missing from {checkpoint}: {len(missing)} tensor(s), first: {missing[:8]}"
        )
    if unexpected:
        logger.debug(
            "Ignored %d unexpected tensor(s) from %s, first: %s",
            len(unexpected), checkpoint, unexpected[:8],
        )
    if quant_meta:
        logger.debug("Consumed %d comfy_quant metadata entries from %s.", quant_meta, checkpoint)
    materialize_meta_buffers(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def materialize_meta_buffers(model: nn.Module) -> None:
    """Recompute deterministic non-persistent buffers left on meta by init_empty_weights."""
    for module in model.modules():
        if isinstance(module, BreezeRotaryEmbedding):
            module._recompute(torch.device("cpu"))
        elif isinstance(module, (Qwen3RotaryEmbedding, T5Gemma2RotaryEmbedding)):
            fresh = type(module)(config=module.config)
            for bname, buf in fresh.named_buffers():
                setattr(module, bname, buf.to("cpu"))
            if hasattr(fresh, "attention_scaling"):
                module.attention_scaling = fresh.attention_scaling
        elif isinstance(module, T5Gemma2TextScaledWordEmbedding):
            module.embed_scale = torch.tensor(module.scalar_embed_scale)
    for name, buf in model.named_buffers(remove_duplicate=False):
        if buf is not None and buf.is_meta:
            raise RuntimeError(f"Buffer {name} is still on the meta device after weight loading.")


# --------------------------------------------------------------------------- #
# SageAttention runtime (monkeypatches SDPA like the FireRedTTS3 / Raon packs)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def attention_runtime(attention: str):
    if attention != "sageattention":
        yield
        return
    from sageattention import sageattn

    original_sdpa = F.scaled_dot_product_attention

    def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if (
            attn_mask is not None
            or dropout_p not in (0, 0.0)
            or query.device.type != "cuda"
            or query.dtype not in (torch.float16, torch.bfloat16)
        ):
            return original_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, **kwargs,
            )
        try:
            output = sageattn(query, key, value, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)
            return output[0] if isinstance(output, tuple) else output
        except Exception:
            return original_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, **kwargs,
            )

    F.scaled_dot_product_attention = sage_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa

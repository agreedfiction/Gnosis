# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig

from ...models.qwen3.feature_extractors import (
    AttnFeatureExtractorLite_D3,
    ConfFeatureExtractorLite,
    HiddenFeatureExtractorLite,
    CorrectnessHeadLite,
)
import torch.nn.functional as F

def _safe_dtype_param(module: nn.Module) -> torch.dtype:
    """Return the dtype of the first parameter in module, defaulting to bfloat16."""
    for p in module.parameters():
        return p.dtype
    return torch.bfloat16


logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
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


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
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
    attention_mask: Optional[torch.Tensor],
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


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
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


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
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


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ---- feature toggles (simple booleans) ----
        self.use_stop_attn = bool(getattr(config, "use_stop_attn", True))
        self.use_stop_conf = bool(getattr(config, "use_stop_conf", False))
        self.use_stop_hid  = bool(getattr(config, "use_stop_hid",  True))

        # ---- dims (safe defaults) ----
        D_ATT  = int(getattr(config, "stop_att_dim", 256))
        D_CONF = int(getattr(config, "stop_conf_dim", 128))
        D_HID  = int(getattr(config, "stop_hid_dim", 256))

        k_conf = int(getattr(config, "stop_k_conf", 192))
        k_hid  = int(getattr(config, "stop_k_hid", 192))

        max_layers = int(getattr(config, "num_hidden_layers", 32))
        max_heads  = int(getattr(config, "num_attention_heads", 32))

        self._custom_head_names = []

        if self.use_stop_attn:
            self.attn_extractor = AttnFeatureExtractorLite_D3(
                D_ATT=D_ATT,
                d_grid=192,
                cnn_channels=(32, 64, 128),
                grid_conv_layers=6,
                K=8,
                pdrop=0.10,
                max_layers=max_layers,
                max_heads=max_heads,
                feature_mode="both",
                stats_groups=("all",),
                spec_radii=(0.15, 0.35, 0.60),
                band_widths=(None, None),
            )
            self._custom_head_names.append("attn_extractor")
        else:
            self.attn_extractor = None

        if self.use_stop_conf:
            self.conf_extractor = ConfFeatureExtractorLite(
                D_CONF=D_CONF,
                d_tok=128,
                k_conf=k_conf,
                base_c=64,
                K=3,
                sab_layers=2,
                sab_heads=4,
                pdrop=0.10,
            )
            self._custom_head_names.append("conf_extractor")
        else:
            self.conf_extractor = None

        if self.use_stop_hid:
            self.hid_extractor = HiddenFeatureExtractorLite(
                D_model=config.hidden_size,
                D_HID=D_HID,
                d_tok=192,
                k_hid=k_hid,
                groups=8,
                K=3,
                sab_layers=3,
                sab_heads=8,
                pdrop=0.10,
            )
            self._custom_head_names.append("hid_extractor")
        else:
            self.hid_extractor = None

        self.stop_head = CorrectnessHeadLite(
            D_ATT=D_ATT, D_CONF=D_CONF, D_HID=D_HID, pdrop=0.10,
            use_attn=self.use_stop_attn, use_conf=self.use_stop_conf, use_hid=self.use_stop_hid,
        )
        self._custom_head_names.append("stop_head")

        # Initialize weights and apply final processing
        self.post_init()

        # freeze base; train only aux modules that exist
        trainable_prefixes = tuple(self._custom_head_names)
        for n, p in self.named_parameters():
            if n.startswith(trainable_prefixes):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

    # ----------------------------------------------------------------------------------
    # Aux correctness scorer (returns sequence-level probability; shape (B,1))
    # ----------------------------------------------------------------------------------
    def _should_stop(
        self,
        last_hidden: torch.Tensor,
        attn_stack: list[torch.Tensor] | None,
        token_probs: torch.Tensor,
        mask: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = last_hidden.shape
        out_dtype = _safe_dtype_param(self.stop_head)

        token_probs_s = torch.clamp(torch.nan_to_num(token_probs, nan=1.0, posinf=1.0, neginf=1e-8), 1e-8, 1.0)
        last_hidden_s = torch.nan_to_num(last_hidden, nan=0.0, posinf=0.0, neginf=0.0)

        A = None
        if self.use_stop_attn:
            if attn_stack is None or len(attn_stack) == 0:
                raise RuntimeError("use_stop_attn=True but no reduced attentions were provided.")
            with torch.no_grad():
                A = torch.stack(attn_stack, dim=1).detach()
                if A.dim() != 5 or A.shape[-1] != A.shape[-2]:
                    raise RuntimeError(f"Expected reduced attn (B,L,H,k,k), got {tuple(A.shape)}")
                A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            z_att = None
            z_conf = None
            z_hid = None
            if self.use_stop_attn:
                z_att = self.attn_extractor(A.to(_safe_dtype_param(self.attn_extractor)))
            if self.use_stop_conf:
                z_conf = self.conf_extractor(token_probs_s.to(_safe_dtype_param(self.conf_extractor)))
            if self.use_stop_hid:
                z_hid  = self.hid_extractor(last_hidden_s.to(_safe_dtype_param(self.hid_extractor)))

            if z_att  is not None: z_att  = torch.nan_to_num(z_att,  nan=0.0, posinf=0.0, neginf=0.0)
            if z_conf is not None: z_conf = torch.nan_to_num(z_conf, nan=0.0, posinf=0.0, neginf=0.0)
            if z_hid  is not None: z_hid  = torch.nan_to_num(z_hid,  nan=0.0, posinf=0.0, neginf=0.0)

            logits = self.stop_head(
                z_att.to(out_dtype)  if z_att  is not None else None,
                z_conf.to(out_dtype) if z_conf is not None else None,
                z_hid.to(out_dtype)  if z_hid  is not None else None,
            )

        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.amp.autocast("cuda", enabled=False):
            probs = torch.sigmoid(logits.to(torch.float32))
        probs = torch.clamp(torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0), 1e-6, 1.0 - 1e-6).to(out_dtype)
        return probs

    # Correctness BCE loss
    def compute_stop_loss(
        self,
        probs_seq: torch.Tensor,
        correctness_label: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if probs_seq.dim() == 2 and probs_seq.size(-1) == 1:
            probs_seq = probs_seq.squeeze(-1)
        probs = torch.nan_to_num(probs_seq.float(), nan=0.5)
        probs = probs.clamp(min=eps, max=1.0 - eps)

        labels = correctness_label
        if labels.dim() == 2 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
        labels = torch.nan_to_num(labels.float(), nan=0.0)

        keep = labels.ne(-1.0)
        if not torch.any(keep):
            return probs.sum() * 0.0

        y = labels[keep].clamp_(0.0, 1.0)
        p = probs[keep]

        loss = F.binary_cross_entropy(p, y, reduction="mean")
        if not torch.isfinite(loss):
            loss = probs.sum() * 0.0
        return loss

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        apply_budget: Optional[bool] = None,
        correctness_label: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        CausalLMOutputWithPast with an extra field `stop_prob` (B,1) when computed.
        """
        output_attentions = (
            True if (self.training or apply_budget)
            else (self.config.output_attentions if output_attentions is None else output_attentions)
        )
        if getattr(self, "use_stop_attn", False) != True:
            output_attentions = False
        output_hidden_states = False

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_full = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_full[:, slice_indices, :])

        loss = None
        if labels is not None and correctness_label is None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        hidden_for_head = hidden_full[:, :-1, :]
        B, S_hid, _ = hidden_for_head.shape
        if attention_mask is not None:
            mask = attention_mask[:, :S_hid].float()
        else:
            mask = torch.ones((B, S_hid), device=hidden_for_head.device, dtype=torch.float)

        stop_prob = None
        if self.training and correctness_label is not None:
            logits_step = self.lm_head(hidden_full)
            logits_step = logits_step[:, :-1, :] if logits_step.size(1) == labels.size(1) + 1 else logits_step
            logp = torch.log_softmax(logits_step.float(), dim=-1)
            tgt = labels.clamp(min=0)
            log_tok_p = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            log_tok_p = torch.where(labels.eq(-100), torch.zeros_like(log_tok_p), log_tok_p)
            token_probs_so_far = torch.clamp(log_tok_p.exp(), min=1e-8).to(logits_step.dtype).detach()
            
            attn_stack = outputs.attentions

            stop_prob = self._should_stop(
                last_hidden=hidden_for_head,
                attn_stack=attn_stack,
                token_probs=token_probs_so_far[:, :-1] if token_probs_so_far.size(1) > S_hid else token_probs_so_far,
                mask=mask,
                input_ids=input_ids[:, :S_hid] if input_ids is not None else None,
            )

            loss = self.compute_stop_loss(stop_prob, correctness_label)

        if apply_budget and not self.training:
            attn_stack = outputs.attentions
            # In inference without labels, pass dummy token probs 
            dummy_probs = torch.ones((B, S_hid), device=hidden_for_head.device, dtype=hidden_for_head.dtype)
            stop_prob = self._should_stop(
                last_hidden=hidden_for_head,
                attn_stack=attn_stack,
                token_probs=dummy_probs,
                mask=mask,
                input_ids=input_ids[:, :S_hid] if input_ids is not None else None,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        if stop_prob is not None:
            output.stop_prob = stop_prob.detach() if not self.training else stop_prob
            
        return output


class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]

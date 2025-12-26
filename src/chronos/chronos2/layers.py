# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import ModelOutput

from .config import Chronos2CoreConfig


class RoPE(nn.Module):
    """Applies rotary position embeddings (RoPE) to input tensors.

    Implementation adapted from:
    https://github.com/huggingface/transformers/blob/965cf677695dd363285831afca8cf479cf0c600c/src/transformers/models/llama/modeling_llama.py#L95
    """

    def __init__(self, dim: int, base: float = 10000):
        super().__init__()

        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.inv_freq: torch.Tensor  # type hint for type checker
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
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
        q_embed = (q * cos) + (RoPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (RoPE.rotate_half(k) * sin)
        return q_embed, k_embed


class Chronos2LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# This is how transformers keeps track of LayerNorm classes ¯\_(ツ)_/¯
ALL_LAYERNORM_LAYERS.append(Chronos2LayerNorm)  # type: ignore


class MLP(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()

        assert not config.is_gated_act, "gated activations are unsupported"
        self.mlp: nn.Module = MLP(config)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


@dataclass
class AttentionOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


class MHA(nn.Module):
    """Multi-head Attention Layer"""

    def __init__(self, config: Chronos2CoreConfig, use_rope: bool = True):
        super().__init__()
        self.d_model: int = config.d_model
        self.kv_proj_dim: int = config.d_kv
        self.n_heads: int = config.num_heads
        self.dropout: float = config.dropout_rate
        self.inner_dim: int = self.n_heads * self.kv_proj_dim
        self.config = config

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = RoPE(dim=self.kv_proj_dim, base=config.rope_theta)

    def _eager_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager attention implementation using manual matmul.

        Args:
            query_states: [batch, n_heads, seq_len, kv_proj_dim]
            key_states: [batch, n_heads, seq_len, kv_proj_dim]
            value_states: [batch, n_heads, seq_len, kv_proj_dim]
            mask: [batch, n_heads, q_len, kv_len]

        Returns:
            attn_output: [batch, n_heads, seq_len, kv_proj_dim]
            attn_weights: [batch, n_heads, q_len, kv_len]
        """
        # Compute attention weights (no scaling - this is the original Chronos-2 implementation)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))  # "bnqd,bnkd->bnqk"
        scores += mask
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, attn_weights

    def _sdpa_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """SDPA attention implementation using torch.nn.functional.scaled_dot_product_attention.

        Args:
            query_states: [batch, n_heads, seq_len, kv_proj_dim]
            key_states: [batch, n_heads, seq_len, kv_proj_dim]
            value_states: [batch, n_heads, seq_len, kv_proj_dim]
            mask: [batch, n_heads, q_len, kv_len] - additive mask (0 for valid, -inf for invalid)

        Returns:
            attn_output: [batch, n_heads, seq_len, kv_proj_dim]
            attn_weights: None (SDPA doesn't return weights)
        """
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,  # Match eager implementation (no scaling)
        )

        return attn_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        """Multi-head attention forward pass.

        Args:
            hidden_states : Input tensor of shape [batch_size, seq_len, d_model]
            mask : Attention mask tensor of shape [batch_size, num_heads, q_len, kv_len]
            encoder_states : Encoder states for cross-attention. Defaults to None.
            position_ids : Position IDs for RoPE. Defaults to None.
            output_attentions : Whether to return attention weights. Defaults to False.

        Returns:
            AttentionOutput: Contains:
                - hidden_states : Output tensor of shape [batch_size, seq_len, d_model]
                - attn_weights : Attention weights if output_attentions=True
        """
        if self.use_rope:
            assert position_ids is not None, "position_ids must be provided when self.use_rope=True"

        # Force eager attention if output_attentions is True (only eager returns weights)
        attn_implementation = self.config._attn_implementation
        if output_attentions:
            attn_implementation = "eager"

        seq_length = hidden_states.shape[1]

        def shape(states: torch.Tensor) -> torch.Tensor:
            """(batch, seq_len, inner_dim) -> (batch, n_heads, seq_len, kv_proj_dim)"""
            return rearrange(states, "b s (h d) -> b h s d", h=self.n_heads, s=seq_length, d=self.kv_proj_dim)

        def unshape(states: torch.Tensor) -> torch.Tensor:
            """(batch, n_heads, seq_len, kv_proj_dim) -> (batch, seq_len, inner_dim)"""
            return rearrange(states, "b h s d -> b s (h d)", h=self.n_heads, s=seq_length, d=self.kv_proj_dim)

        # Construct query states
        query_states = shape(self.q(hidden_states))
        is_cross_attention = encoder_states is not None

        # Construct key/value states
        if is_cross_attention:
            key_states = shape(self.k(encoder_states))
            value_states = shape(self.v(encoder_states))
        else:
            key_states = shape(self.k(hidden_states))
            value_states = shape(self.v(hidden_states))
            if self.use_rope:
                cos, sin = self.rope_embed(value_states, position_ids)
                query_states, key_states = RoPE.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if attn_implementation == "sdpa":
            attn_output, attn_weights = self._sdpa_attention(query_states, key_states, value_states, mask)
        else:  # eager
            attn_output, attn_weights = self._eager_attention(query_states, key_states, value_states, mask)

        # Project attention output
        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)

        return AttentionOutput(hidden_states=attn_output, attn_weights=attn_weights if output_attentions else None)


class TimeSelfAttention(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.self_attention = MHA(config, use_rope=True)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states, position_ids=position_ids, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


class TimeCrossAttention(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.cross_attention = MHA(config, use_rope=False)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.cross_attention(
            normed_hidden_states,
            mask=attention_mask,
            encoder_states=encoder_states,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


class GroupSelfAttention(nn.Module):
    """Self-attention applied along the batch axis masked by the group attention mask"""

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        # we don't use RoPE here because there's no natural ordering along the batch axis
        self.self_attention = MHA(config, use_rope=False)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = False
    ) -> AttentionOutput:
        # flip time and batch axes because attention operates along dim=-2
        hidden_states = rearrange(hidden_states, "batch time d -> time batch d")
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # flip time and batch axes back to their original position
        hidden_states = rearrange(hidden_states, "time batch d -> batch time d")

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


class CrossGroupAttention(nn.Module):
    """
    Selective Cross-Group Attention with Gated Fusion.
    
    This layer enables information flow between different groups (e.g., electricity prices
    in Germany and France) by:
    1. Creating group-level summary representations via mean pooling
    2. Selectively attending to relevant groups using one of:
       - Top-k attention: attend only to k most similar groups
       - Similarity threshold: attend only to groups above cosine similarity threshold
       - Sparse routing: learned sparse distribution over groups
    3. Using a learnable gate to control how much cross-group information flows back
    
    The selective mechanisms reduce negative transfer on heterogeneous datasets (dominick, tourism)
    while preserving gains on correlated datasets (exchange_rate, electricity).
    
    Config flags:
    - cross_group_top_k: int | None - If set, attend only to top-k groups (+ self if always_include_self)
    - cross_group_similarity_threshold: float | None - If set, mask attention below this cosine sim
    - cross_group_always_include_self: bool - Always include self-attention in top-k/threshold
    - cross_group_use_sparse_routing: bool - Use learned sparse routing
    - cross_group_routing_temperature: float - Temperature for routing softmax
    """

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.num_heads
        self.d_kv = config.d_kv
        self.inner_dim = self.n_heads * self.d_kv
        
        # Selective attention config
        self.top_k = getattr(config, 'cross_group_top_k', None)
        self.similarity_threshold = getattr(config, 'cross_group_similarity_threshold', None)
        self.always_include_self = getattr(config, 'cross_group_always_include_self', True)
        self.use_sparse_routing = getattr(config, 'cross_group_use_sparse_routing', False)
        self.routing_temperature = getattr(config, 'cross_group_routing_temperature', 1.0)
        # Dynamic routing: skip cross-group attention if groups are dissimilar
        self.dynamic_routing = getattr(config, 'cross_group_dynamic_routing', False)
        self.dynamic_threshold = getattr(config, 'cross_group_dynamic_threshold', 0.5)
        # Margin gate: require clear separation (top1 - median > delta) to reduce false positives
        self.margin_gate = getattr(config, 'cross_group_margin_gate', False)
        self.margin_delta = getattr(config, 'cross_group_margin_delta', 0.1)
        
        # Tracking statistics (updated during forward pass)
        self._last_avg_similarity = None
        self._last_margin = None  # top1 - median similarity
        self._last_cross_group_applied = None
        self._stats_history = []  # List of (avg_sim, margin, was_applied, skip_reason) tuples
        
        # Layer norms
        self.layer_norm_summary = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.layer_norm_cross = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
        # Cross-group attention projections
        # Shape: (d_model,) -> (d_model,) for Q, K, V
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Sparse routing network (if enabled): maps summary to routing logits
        # Input: (d_model,) -> Output: scalar logit per group (applied dynamically)
        if self.use_sparse_routing:
            self.router = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 4),
                nn.ReLU(),
                nn.Linear(config.d_model // 4, 1)  # Single logit per group
            )
        
        # Gating mechanism: learns how much cross-group info to incorporate
        # Input: (2 * d_model,) -> Output: (d_model,)
        self.gate_proj = nn.Linear(config.d_model * 2, config.d_model, bias=True)
        
        # Projection to broadcast cross-group info back to all time steps
        self.broadcast_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.scale = config.d_model ** -0.5

    def _create_group_summaries(
        self, 
        hidden_states: torch.Tensor,  # Shape: (B, T, D) where B=batch, T=time, D=d_model
        group_ids: torch.Tensor,      # Shape: (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """
        Create a summary representation for each unique group using mean pooling.
        
        Returns:
            group_summaries: Shape (G, D) where G=num_groups, D=d_model
            unique_groups: Shape (G,) - unique group IDs
            group_to_batch_idx: list of length G, each element is tensor of batch indices
        """
        batch_size, seq_len, d_model = hidden_states.shape  # (B, T, D)
        unique_groups = torch.unique(group_ids)  # (G,)
        num_groups = len(unique_groups)
        
        group_summaries = []
        group_to_batch_idx = []
        
        for g_id in unique_groups:
            # Find all batch elements belonging to this group
            mask = (group_ids == g_id)  # (B,) boolean
            batch_indices = torch.where(mask)[0]  # (group_size,)
            group_to_batch_idx.append(batch_indices)
            
            # Get hidden states for this group: (group_size, T, D)
            group_hidden = hidden_states[mask]
            
            # Normalize and mean pool over both batch items and time
            normed_hidden = self.layer_norm_summary(group_hidden)  # (group_size, T, D)
            
            # Mean pool: (group_size, T, D) -> (D,)
            summary = normed_hidden.mean(dim=(0, 1))
            group_summaries.append(summary)
        
        # Stack: list of (D,) -> (G, D)
        group_summaries = torch.stack(group_summaries, dim=0)
        
        return group_summaries, unique_groups, group_to_batch_idx

    def _compute_cosine_similarity(
        self,
        group_summaries: torch.Tensor,  # Shape: (G, D)
    ) -> torch.Tensor:
        """
        Compute pairwise cosine similarity between group summaries.
        
        Returns:
            cosine_sim: Shape (G, G) - cosine similarity matrix
        """
        # Normalize summaries to unit vectors: (G, D)
        normalized = torch.nn.functional.normalize(group_summaries, p=2, dim=-1)
        # Cosine similarity: (G, G)
        cosine_sim = torch.matmul(normalized, normalized.transpose(-1, -2))
        return cosine_sim

    def _apply_top_k_mask(
        self,
        attn_scores: torch.Tensor,  # Shape: (G, G)
        k: int,
    ) -> torch.Tensor:
        """
        Apply top-k masking to attention scores.
        For each query group, keep only top-k keys (+ self if always_include_self).
        
        Returns:
            masked_scores: Shape (G, G) with -inf for masked positions
        """
        G = attn_scores.shape[0]
        device = attn_scores.device
        dtype = attn_scores.dtype
        
        # Clamp k to valid range
        effective_k = min(k, G)
        
        # Find top-k indices per row: (G, k)
        _, top_k_indices = torch.topk(attn_scores, effective_k, dim=-1)
        
        # Create mask: (G, G) - True means KEEP
        mask = torch.zeros(G, G, dtype=torch.bool, device=device)
        row_indices = torch.arange(G, device=device).unsqueeze(1).expand(-1, effective_k)
        mask[row_indices, top_k_indices] = True
        
        # Always include self-attention on diagonal if configured
        if self.always_include_self:
            mask.fill_diagonal_(True)
        
        # Apply mask: set non-top-k to -inf
        masked_scores = attn_scores.clone()
        masked_scores[~mask] = torch.finfo(dtype).min
        
        return masked_scores

    def _apply_similarity_threshold_mask(
        self,
        attn_scores: torch.Tensor,  # Shape: (G, G)
        group_summaries: torch.Tensor,  # Shape: (G, D)
        threshold: float,
    ) -> torch.Tensor:
        """
        Apply similarity threshold masking.
        Mask attention where cosine similarity is below threshold.
        
        Returns:
            masked_scores: Shape (G, G) with -inf for masked positions
        """
        G = attn_scores.shape[0]
        device = attn_scores.device
        dtype = attn_scores.dtype
        
        # Compute cosine similarity: (G, G)
        cosine_sim = self._compute_cosine_similarity(group_summaries)
        
        # Create mask: True means KEEP (similarity >= threshold)
        mask = cosine_sim >= threshold
        
        # Always include self-attention on diagonal if configured
        if self.always_include_self:
            mask.fill_diagonal_(True)
        
        # Apply mask
        masked_scores = attn_scores.clone()
        masked_scores[~mask] = torch.finfo(dtype).min
        
        return masked_scores

    def _apply_sparse_routing(
        self,
        group_summaries: torch.Tensor,  # Shape: (G, D)
    ) -> torch.Tensor:
        """
        Apply learned sparse routing using a small router network.
        The router outputs logits that are used to create a sparse attention distribution.
        
        Returns:
            routing_weights: Shape (G, G) - sparse routing weights
        """
        G, D = group_summaries.shape
        device = group_summaries.device
        
        # Compute routing logits for each group: (G, 1)
        routing_logits = self.router(group_summaries)  # (G, 1)
        
        # Create pairwise routing scores: (G, G)
        # Score[i,j] = how much group i should attend to group j
        # Use outer sum of logits (symmetric) + learned asymmetry from Q/K
        routing_scores = routing_logits + routing_logits.transpose(-1, -2)  # (G, G)
        
        # Apply temperature scaling
        routing_scores = routing_scores / self.routing_temperature
        
        # Sparsemax-like: keep top-k per row where k is adaptive
        # For simplicity, use softmax with temperature (entmax would require extra dependency)
        # Low temperature -> more sparse
        routing_weights = torch.softmax(routing_scores.squeeze(-1), dim=-1)  # (G, G)
        
        return routing_weights

    def _cross_group_attention(
        self,
        group_summaries: torch.Tensor,  # Shape: (G, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply selective cross-group attention.
        
        Attention mechanism:
        1. Project summaries to Q, K, V
        2. Compute attention scores
        3. Apply selective masking (top-k OR threshold OR sparse routing)
        4. Softmax and apply to values
        
        Returns:
            cross_group_info: Shape (G, D)
            attn_weights: Shape (G, G)
        """
        G, D = group_summaries.shape  # (num_groups, d_model)
        
        # Normalize summaries
        normed_summaries = self.layer_norm_cross(group_summaries)  # (G, D)
        
        # Project to Q, K, V: all (G, D)
        Q = self.q_proj(normed_summaries)
        K = self.k_proj(normed_summaries)
        V = self.v_proj(normed_summaries)
        
        # Compute attention scores: (G, G)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        
        # Apply selective masking based on config (mutually exclusive strategies)
        if self.use_sparse_routing:
            # Sparse routing: use router network to compute weights directly
            attn_weights = self._apply_sparse_routing(group_summaries)
        else:
            # Standard attention with optional masking
            if self.top_k is not None:
                # Top-k masking: attend only to k most relevant groups
                attn_scores = self._apply_top_k_mask(attn_scores, self.top_k)
            elif self.similarity_threshold is not None:
                # Similarity threshold: mask low-similarity groups
                attn_scores = self._apply_similarity_threshold_mask(
                    attn_scores, group_summaries, self.similarity_threshold
                )
            # else: no masking (original full attention)
            
            # Softmax to get attention weights: (G, G)
            attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: (G, G) @ (G, D) -> (G, D)
        attn_output = torch.matmul(attn_weights, V)
        
        # Project output: (G, D)
        cross_group_info = self.o_proj(attn_output)
        
        return cross_group_info, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        group_ids: torch.Tensor,      # Shape: (B,)
        output_attentions: bool = False,
    ) -> AttentionOutput:
        """
        Apply selective cross-group attention with gated fusion.
        
        Pipeline:
        1. Create summary for each group via mean pooling: (B, T, D) -> (G, D)
        2. (Optional) Dynamic routing: check if groups are similar enough
        3. Selective cross-group attention: (G, D) -> (G, D)
        4. Broadcast back to all tokens with gating: (G, D) -> (B, T, D)
        
        Complexity:
        - Summary creation: O(B * T * D)
        - Cross-group attention: O(G² * D) where G = num_groups
        - Broadcast fusion: O(B * T * D)
        """
        B, T, D = hidden_states.shape  # batch, time, d_model
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Handle edge case: only one group (no cross-group attention needed)
        unique_groups = torch.unique(group_ids)
        G = len(unique_groups)
        if G <= 1:
            self._last_avg_similarity = 1.0
            self._last_cross_group_applied = False
            return AttentionOutput(hidden_states=hidden_states, attn_weights=None)
        
        # Step 1: Create group summaries - (B, T, D) -> (G, D)
        group_summaries, unique_groups, group_to_batch_idx = self._create_group_summaries(
            hidden_states, group_ids
        )
        
        # Step 2: Dynamic routing check - compute avg similarity and decide
        if self.dynamic_routing or self.margin_gate:
            cosine_sim = self._compute_cosine_similarity(group_summaries)  # (G, G)
            # Off-diagonal similarities (exclude self-similarity on diagonal)
            mask = ~torch.eye(G, dtype=torch.bool, device=device)
            off_diag_sims = cosine_sim[mask]
            avg_similarity = off_diag_sims.mean().item() if G > 1 else 1.0
            
            # Compute margin: for each group, get max similarity to other groups
            # Then compute margin = top1 - median across all off-diagonal
            if G > 2:
                top1_sim = off_diag_sims.max().item()
                median_sim = off_diag_sims.median().item()
                margin = top1_sim - median_sim
            else:
                margin = 0.0  # Can't compute margin with only 2 groups
            
            self._last_avg_similarity = avg_similarity
            self._last_margin = margin
            
            # Gate 1: Average similarity threshold
            if self.dynamic_routing and avg_similarity < self.dynamic_threshold:
                self._last_cross_group_applied = False
                self._stats_history.append((avg_similarity, margin, False, "low_avg_sim"))
                return AttentionOutput(hidden_states=hidden_states, attn_weights=None)
            
            # Gate 2: Margin gate - require clear separation to avoid false positives
            # Skip if everything is "vaguely similar" (high avg but low margin)
            if self.margin_gate and margin < self.margin_delta:
                self._last_cross_group_applied = False
                self._stats_history.append((avg_similarity, margin, False, "low_margin"))
                return AttentionOutput(hidden_states=hidden_states, attn_weights=None)
            
            self._last_cross_group_applied = True
            self._stats_history.append((avg_similarity, margin, True, "applied"))
        else:
            self._last_avg_similarity = None
            self._last_margin = None
            self._last_cross_group_applied = True
        
        # Step 3: Selective cross-group attention - (G, D) -> (G, D)
        cross_group_info, attn_weights = self._cross_group_attention(group_summaries)
        
        # Step 3: Broadcast cross-group info back to each batch element with gating
        output_hidden_states = hidden_states.clone()
        
        for g_idx, batch_indices in enumerate(group_to_batch_idx):
            group_size = len(batch_indices)
            
            # Get cross-group info for this group: (D,)
            cross_info = cross_group_info[g_idx]
            
            # Broadcast projection: (D,) -> (D,)
            cross_info_broadcast = self.broadcast_proj(cross_info)
            
            # Expand to all time steps for this group: (D,) -> (group_size, T, D)
            cross_info_expanded = cross_info_broadcast.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            cross_info_expanded = cross_info_expanded.expand(group_size, T, -1)   # (group_size, T, D)
            
            # Get original hidden states for this group: (group_size, T, D)
            group_hidden = hidden_states[batch_indices]
            
            # Compute gate: sigmoid(W * [original; cross_info])
            # Input: (group_size, T, 2*D) -> Output: (group_size, T, D)
            gate_input = torch.cat([group_hidden, cross_info_expanded], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))
            
            # Apply gated fusion: output = original + gate * dropout(cross_info)
            gated_cross_info = gate * self.dropout(cross_info_expanded)
            output_hidden_states[batch_indices] = group_hidden + gated_cross_info
        
        return AttentionOutput(
            hidden_states=output_hidden_states,
            attn_weights=attn_weights if output_attentions else None
        )


class ResidualBlock(nn.Module):
    """A generic residual block which can be used for input and output embedding layers"""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = Chronos2LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out

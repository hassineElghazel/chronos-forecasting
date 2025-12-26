"""
Series Memory Bank for Retrieval-Augmented Time Series Forecasting

This module implements a memory bank that stores per-series representations
and supports retrieval-based cross-attention, avoiding the pitfalls of
random batch-based summaries.

Key features:
- Per-series summaries (not batch summaries)
- Top-K retrieval based on cosine similarity
- Gated fusion with confidence-based weighting
- Memory bank can be built from training data or accumulated during inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class MemoryBankConfig:
    """Configuration for the Series Memory Bank."""
    d_model: int = 512
    max_memory_size: int = 10000  # Maximum number of series to store
    top_k: int = 5  # Number of memories to retrieve
    similarity_threshold: float = 0.7  # Minimum similarity for retrieval
    use_gating: bool = True  # Whether to use confidence-based gating
    gate_temperature: float = 1.0  # Temperature for gating sigmoid
    normalize_memories: bool = True  # L2 normalize stored memories


class SeriesMemoryBank(nn.Module):
    """
    A memory bank that stores per-series representations for retrieval.
    
    Unlike batch-based approaches, this stores individual series representations
    and retrieves similar ones based on cosine similarity.
    
    Memory Flow:
    1. Series are encoded → hidden states
    2. Hidden states are mean-pooled → series representation (d_model)
    3. Series representations are stored in memory bank
    4. At inference, query the bank to retrieve similar series
    5. Cross-attend to retrieved memories with gated fusion
    """
    
    def __init__(self, config: MemoryBankConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_size = config.max_memory_size
        self.top_k = config.top_k
        self.threshold = config.similarity_threshold
        self.use_gating = config.use_gating
        self.gate_temp = config.gate_temperature
        self.normalize = config.normalize_memories
        
        # Memory storage (not a parameter, updated manually)
        self.register_buffer(
            'memory_bank',
            torch.zeros(config.max_memory_size, config.d_model),
            persistent=False
        )
        self.register_buffer(
            'memory_count',
            torch.tensor(0, dtype=torch.long),
            persistent=False
        )
        
        # Optional: metadata storage for interpretability
        self.memory_metadata: List[dict] = []
        
    def reset(self):
        """Clear the memory bank."""
        self.memory_bank.zero_()
        self.memory_count.zero_()
        self.memory_metadata.clear()
        
    def add_memories(
        self,
        series_representations: torch.Tensor,
        metadata: Optional[List[dict]] = None
    ):
        """
        Add series representations to the memory bank.
        
        Args:
            series_representations: (num_series, d_model) tensor
            metadata: Optional list of dicts with series info (dataset, idx, etc.)
        """
        num_series = series_representations.size(0)
        
        # Normalize if configured
        if self.normalize:
            series_representations = F.normalize(series_representations, p=2, dim=-1)
        
        # Get current count and available space
        current_count = self.memory_count.item()
        available_space = self.max_size - current_count
        
        if available_space >= num_series:
            # Enough space, just append
            self.memory_bank[current_count:current_count + num_series] = series_representations
            self.memory_count += num_series
        else:
            # FIFO replacement: shift old memories out, add new ones
            if num_series >= self.max_size:
                # New memories exceed max size, keep only the last max_size
                self.memory_bank[:] = series_representations[-self.max_size:]
                self.memory_count.fill_(self.max_size)
            else:
                # Shift old memories and add new ones
                shift_amount = num_series - available_space
                self.memory_bank[:-shift_amount] = self.memory_bank[shift_amount:].clone()
                self.memory_bank[-num_series:] = series_representations
                self.memory_count.fill_(self.max_size)
        
        # Store metadata
        if metadata:
            self.memory_metadata.extend(metadata)
            # Trim metadata to match memory bank size
            if len(self.memory_metadata) > self.max_size:
                self.memory_metadata = self.memory_metadata[-self.max_size:]
    
    def retrieve(
        self,
        query_representations: torch.Tensor,
        exclude_self: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve top-K similar memories for each query.
        
        Args:
            query_representations: (batch_size, d_model) query vectors
            exclude_self: If True, exclude exact matches (for eval without leakage)
            
        Returns:
            retrieved_memories: (batch_size, top_k, d_model)
            similarities: (batch_size, top_k) cosine similarities
            mask: (batch_size, top_k) boolean mask (True = valid memory)
        """
        batch_size = query_representations.size(0)
        current_count = self.memory_count.item()
        
        if current_count == 0:
            # Empty memory bank - return zeros
            device = query_representations.device
            return (
                torch.zeros(batch_size, self.top_k, self.d_model, device=device),
                torch.zeros(batch_size, self.top_k, device=device),
                torch.zeros(batch_size, self.top_k, dtype=torch.bool, device=device)
            )
        
        # Normalize query
        if self.normalize:
            query_representations = F.normalize(query_representations, p=2, dim=-1)
        
        # Get active memories
        active_memories = self.memory_bank[:current_count]  # (M, d_model)
        
        # Compute cosine similarities
        # (batch_size, d_model) @ (d_model, M) -> (batch_size, M)
        similarities = torch.matmul(query_representations, active_memories.T)
        
        # Exclude exact matches if needed (similarity > 0.999)
        if exclude_self:
            similarities = torch.where(
                similarities > 0.999,
                torch.tensor(-float('inf'), device=similarities.device),
                similarities
            )
        
        # Apply threshold mask
        valid_mask = similarities >= self.threshold
        similarities = torch.where(valid_mask, similarities, torch.tensor(-float('inf'), device=similarities.device))
        
        # Get top-K
        k = min(self.top_k, current_count)
        top_sims, top_indices = torch.topk(similarities, k=k, dim=-1)
        
        # Gather memories
        # top_indices: (batch_size, k)
        retrieved = active_memories[top_indices]  # (batch_size, k, d_model)
        
        # Create valid mask (not -inf)
        mask = top_sims > -float('inf')
        
        # Pad if k < top_k
        if k < self.top_k:
            pad_size = self.top_k - k
            retrieved = F.pad(retrieved, (0, 0, 0, pad_size), value=0)
            top_sims = F.pad(top_sims, (0, pad_size), value=0)
            mask = F.pad(mask, (0, pad_size), value=False)
        
        return retrieved, top_sims, mask
    
    def get_stats(self) -> dict:
        """Get memory bank statistics."""
        return {
            'memory_count': self.memory_count.item(),
            'max_size': self.max_size,
            'fill_ratio': self.memory_count.item() / self.max_size,
        }


class MemoryAugmentedAttention(nn.Module):
    """
    Cross-attention layer that attends to retrieved memories with gated fusion.
    
    This replaces the batch-based CGA with a retrieval-based approach:
    1. Query: current series representation
    2. Keys/Values: retrieved memories from the bank
    3. Fusion: gated based on similarity confidence
    """
    
    def __init__(self, config: MemoryBankConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        
        # Cross-attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Gating mechanism
        if config.use_gating:
            self.gate_proj = nn.Linear(d_model * 2, 1)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Scaling factor
        self.scale = d_model ** -0.5
        
    def forward(
        self,
        query: torch.Tensor,
        retrieved_memories: torch.Tensor,
        similarities: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply memory-augmented attention.
        
        Args:
            query: (batch_size, d_model) current series representations
            retrieved_memories: (batch_size, top_k, d_model) retrieved memories
            similarities: (batch_size, top_k) similarity scores
            mask: (batch_size, top_k) valid memory mask
            
        Returns:
            output: (batch_size, d_model) memory-augmented representations
        """
        # Check if any memories are valid
        if not mask.any():
            return query
        
        # Project query and memories
        Q = self.q_proj(query).unsqueeze(1)  # (B, 1, d)
        K = self.k_proj(retrieved_memories)   # (B, K, d)
        V = self.v_proj(retrieved_memories)   # (B, K, d)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, 1, K)
        attn_scores = attn_scores.squeeze(1)  # (B, K)
        
        # Apply mask (set invalid positions to -inf)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, K)
        attn_weights = attn_weights.masked_fill(~mask, 0)  # Zero out invalid
        
        # Weighted sum of values
        attn_weights = attn_weights.unsqueeze(-1)  # (B, K, 1)
        memory_output = (attn_weights * V).sum(dim=1)  # (B, d)
        
        # Project output
        memory_output = self.out_proj(memory_output)
        
        # Gated fusion
        if self.config.use_gating:
            # Gate based on similarity confidence
            max_sim = similarities.max(dim=-1, keepdim=True)[0]  # (B, 1)
            gate_input = torch.cat([query, memory_output], dim=-1)  # (B, 2d)
            gate = torch.sigmoid(self.gate_proj(gate_input))  # (B, 1)
            
            # Scale gate by similarity (higher similarity = more confident)
            confidence = torch.sigmoid(
                self.config.gate_temperature * (max_sim - self.config.similarity_threshold)
            )
            gate = gate * confidence
            
            # Fuse
            output = query + gate * memory_output
        else:
            output = query + memory_output
        
        # Layer norm
        output = self.layer_norm(output)
        
        return output


class MemoryAugmentedForecaster(nn.Module):
    """
    Wrapper that adds memory-augmented attention to a base encoder.
    
    Usage:
    1. Build memory bank from training/reference series
    2. At inference, encode series → retrieve → fuse → decode
    """
    
    def __init__(self, config: MemoryBankConfig):
        super().__init__()
        self.memory_bank = SeriesMemoryBank(config)
        self.memory_attention = MemoryAugmentedAttention(config)
        self.config = config
        
        # Statistics tracking
        self._stats = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'avg_top_similarity': 0.0,
            'retrievals_above_threshold': 0,
        }
    
    def build_memory_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None
    ):
        """
        Build memory bank from encoder hidden states.
        
        Args:
            hidden_states: (batch_size, seq_len, d_model) encoder outputs
            group_ids: Optional (batch_size,) series identifiers
        """
        # Mean-pool over sequence dimension to get series representations
        series_reps = hidden_states.mean(dim=1)  # (batch_size, d_model)
        
        # Create metadata if group_ids provided
        metadata = None
        if group_ids is not None:
            metadata = [{'group_id': gid.item()} for gid in group_ids]
        
        self.memory_bank.add_memories(series_reps, metadata)
    
    def augment_representations(
        self,
        hidden_states: torch.Tensor,
        exclude_self: bool = True
    ) -> torch.Tensor:
        """
        Augment series representations with retrieved memories.
        
        Args:
            hidden_states: (batch_size, seq_len, d_model) encoder outputs
            exclude_self: Exclude exact matches in retrieval
            
        Returns:
            augmented: (batch_size, seq_len, d_model) memory-augmented outputs
        """
        _, seq_len, _ = hidden_states.shape
        
        # Get series-level representations
        series_reps = hidden_states.mean(dim=1)  # (batch_size, d_model)
        
        # Retrieve similar memories
        retrieved, similarities, mask = self.memory_bank.retrieve(
            series_reps, exclude_self=exclude_self
        )
        
        # Update statistics
        self._stats['total_queries'] += batch_size
        valid_retrievals = mask.any(dim=-1).sum().item()
        self._stats['successful_retrievals'] += valid_retrievals
        if mask.any():
            valid_sims = similarities[mask]
            if valid_sims.numel() > 0:
                self._stats['avg_top_similarity'] = (
                    self._stats['avg_top_similarity'] * 0.9 + 
                    valid_sims.max().item() * 0.1
                )
        
        # Apply memory attention to get augmented series representation
        augmented_series = self.memory_attention(
            series_reps, retrieved, similarities, mask
        )  # (batch_size, d_model)
        
        # Broadcast augmentation to all tokens via residual
        # Option 1: Add directly (simple)
        # Option 2: Use cross-attention (more expressive)
        # Here we use a simple broadcast with learned scaling
        augmentation = augmented_series - series_reps  # Delta from original
        augmentation = augmentation.unsqueeze(1).expand(-1, seq_len, -1)
        
        augmented_hidden = hidden_states + augmentation
        
        return augmented_hidden
    
    def get_stats(self) -> dict:
        """Get retrieval statistics."""
        stats = self._stats.copy()
        stats.update(self.memory_bank.get_stats())
        if stats['total_queries'] > 0:
            stats['retrieval_rate'] = stats['successful_retrievals'] / stats['total_queries']
        else:
            stats['retrieval_rate'] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset retrieval statistics."""
        self._stats = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'avg_top_similarity': 0.0,
            'retrievals_above_threshold': 0,
        }

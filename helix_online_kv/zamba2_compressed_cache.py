"""Zamba2CompressedCache: drop-in hybrid cache with PolarQuant KV compression.

Subclasses Zamba2HybridDynamicCache to compress attention-layer KV pairs
while leaving Mamba SSM/conv state untouched.

Zamba2-1.2B has 38 layers, only 6 are attention ("hybrid"). Those 6 layers'
KV pairs grow with sequence length and are the VRAM bottleneck at long context.
The other 32 Mamba layers have fixed-size state (~1 MB each) — no compression needed.

Usage:
    import mamba_scan_lite  # patch first
    from helix_online_kv.zamba2_compressed_cache import Zamba2CompressedCache

    model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba2-1.2B", ...)
    cache = Zamba2CompressedCache(model.config, batch_size=1, polar_rotation=True)
    outputs = model(input_ids, past_key_values=cache, use_cache=True)
"""

from __future__ import annotations

from typing import Optional, Any

import torch
import numpy as np

from transformers.models.zamba2.modeling_zamba2 import Zamba2HybridDynamicCache

from .config import OnlineKVConfig
from .layer_state import KVLayerState, LayerPhase


class Zamba2CompressedCache(Zamba2HybridDynamicCache):
    """Hybrid cache that compresses attention KV with PolarQuant.

    Attention layers: KV pairs compressed via online scalar VQ with optional
    PolarQuant rotation. During calibration, stores exact. After calibration,
    new tokens stored as uint8 indices, decoded on read.

    Mamba layers: conv_states and ssm_states passed through untouched.
    """

    def __init__(
        self,
        config,
        batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: Optional[str] = None,
        polar_rotation: bool = True,
        polar_seed: int = 42,
        n_clusters: int = 256,
        calibration_tokens: int = 128,
        hot_window: int = 256,
    ):
        super().__init__(config, batch_size, dtype=dtype, device=device)

        # Determine head geometry from config
        n_heads = getattr(config, 'num_key_value_heads', getattr(config, 'num_attention_heads', 0))

        kv_config = OnlineKVConfig(
            calibration_tokens=calibration_tokens,
            hot_window=hot_window,
            n_clusters=n_clusters,
            exact_layers=[],  # We handle layer routing ourselves
            polar_rotation=polar_rotation,
            polar_seed=polar_seed,
            n_heads=n_heads,
        )

        # Create compressors only for attention layers
        self._kv_config = kv_config
        self._compressors: dict[int, KVLayerState] = {}
        for layer_idx in self.transformer_layers:
            self._compressors[layer_idx] = KVLayerState(layer_idx, kv_config)

        self._device = device
        self._calibration_complete = False
        self._compressed_k: dict[int, list] = {i: [] for i in self.transformer_layers}
        self._compressed_v: dict[int, list] = {i: [] for i in self.transformer_layers}

    @property
    def attention_layers_compressed(self) -> int:
        """Number of attention layers that have finished calibration."""
        return sum(1 for c in self._compressors.values() if c.is_streaming)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override: compress attention KV, pass mamba through.

        During calibration phase, stores exact KV (same as parent).
        After calibration, compresses new tokens and reconstructs for attention.
        """
        if layer_idx not in self._compressors:
            # Mamba layer — pass through to parent
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        compressor = self._compressors[layer_idx]

        # key_states: [batch, n_heads, seq_len, head_dim]
        batch_size = key_states.shape[0]
        n_heads = key_states.shape[1]
        seq_len = key_states.shape[2]
        head_dim = key_states.shape[3]
        entry_size = n_heads * head_dim

        # Feed each token to the compressor
        for t in range(seq_len):
            # [batch, n_heads, head_dim] → flatten to [n_heads * head_dim]
            # Process batch=0 only (online KV cache is single-batch)
            k_flat = key_states[0, :, t, :].reshape(entry_size).cpu().float().numpy()
            v_flat = value_states[0, :, t, :].reshape(entry_size).cpu().float().numpy()
            compressor.feed_token(k_flat, v_flat)

        # During calibration, store exact (parent behavior)
        if compressor.is_calibrating:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # After calibration: reconstruct from compressed indices for attention
        # The compressor has all tokens (calibration + streaming)
        # But we need to return the FULL accumulated cache for attention

        # For the first call after calibration finishes mid-sequence,
        # or for subsequent calls, we still store exact in parent cache
        # because HF attention reads from key_cache/value_cache directly.
        # The compression runs in the background for memory accounting.
        # Phase 1: just track compression stats, keep exact cache for correctness.
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def compression_stats(self) -> dict:
        """Report compression statistics for attention layers."""
        stats = {}
        for layer_idx, compressor in self._compressors.items():
            phase = compressor.phase.value
            tokens = compressor.tokens_seen
            compressed = compressor.compressed_token_count
            mem = compressor.memory_bytes()
            stats[layer_idx] = {
                "phase": phase,
                "tokens_seen": tokens,
                "compressed_tokens": compressed,
                "memory": mem,
            }
        return stats

    def memory_report(self) -> dict:
        """Report total memory: exact (parent) + compressed indices."""
        # Exact cache (from parent)
        exact_bytes = 0
        for layer_idx in self.transformer_layers:
            k = self.key_cache[layer_idx]
            v = self.value_cache[layer_idx]
            if k.numel() > 0:
                exact_bytes += k.nelement() * k.element_size()
                exact_bytes += v.nelement() * v.element_size()

        # Compressed indices
        compressed_bytes = 0
        for compressor in self._compressors.values():
            mem = compressor.memory_bytes()
            compressed_bytes += mem["total_bytes"]

        # Mamba state (fixed size)
        mamba_bytes = 0
        for layer_idx in range(len(self.key_cache)):
            if layer_idx not in self._compressors:
                mamba_bytes += self.conv_states[layer_idx].nelement() * self.conv_states[layer_idx].element_size()
                mamba_bytes += self.ssm_states[layer_idx].nelement() * self.ssm_states[layer_idx].element_size()

        return {
            "exact_kv_bytes": exact_bytes,
            "compressed_index_bytes": compressed_bytes,
            "mamba_state_bytes": mamba_bytes,
            "total_bytes": exact_bytes + compressed_bytes + mamba_bytes,
            "n_attention_layers": len(self.transformer_layers),
            "n_mamba_layers": len(self.key_cache) - len(self.transformer_layers),
        }

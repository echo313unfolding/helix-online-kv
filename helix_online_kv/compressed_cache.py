"""CompressedKVCache: drop-in DynamicCache replacement with online compression.

This module requires torch and transformers. Import guard ensures clean error
messages when running numpy-only experiments.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    from transformers.cache_utils import DynamicCache
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    DynamicCache = object  # Fallback so class definition works

import numpy as np

from .config import OnlineKVConfig
from .layer_state import KVLayerState, LayerPhase
from .aging_policy import AgingPolicy


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "CompressedKVCache requires torch and transformers. "
            "Install with: pip install helix-online-kv[torch]"
        )


class CompressedKVCache(DynamicCache):
    """Online compressed KV cache — subclasses DynamicCache for full HF compat.

    Lifecycle:
        1. Calibration (first N tokens): Accumulates KV values, fits per-layer codebooks.
        2. Streaming (subsequent tokens): New entries assigned via nearest centroid.
        3. Aging: Tokens beyond hot_window demoted from Tier 0 (exact) to Tier 1 (compressed).

    All DynamicCache methods (get_mask_sizes, get_seq_length, __getitem__, etc.)
    are inherited. Only update() is overridden to add compression side-effects.
    """

    def __init__(self, config: OnlineKVConfig, n_layers: int):
        _require_torch()
        super().__init__()
        self.config = config
        self.n_layers_expected = n_layers
        self.aging = AgingPolicy(hot_window=config.hot_window)

        # Per-layer compression state
        self.layer_states: list[KVLayerState] = [
            KVLayerState(i, config) for i in range(n_layers)
        ]

        self._total_tokens = 0
        self._calibration_stats: list[dict] = []

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def calibration_complete(self) -> bool:
        """True when all non-exact layers have finalized codebooks."""
        return all(
            ls.is_streaming or ls.is_exact
            for ls in self.layer_states
        )

    def update(
        self,
        key_states: "torch.Tensor",
        value_states: "torch.Tensor",
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Override DynamicCache.update() to add compression side-effects.

        The parent handles all storage and retrieval. We piggyback on the
        call to feed data into OnlineCodebook for calibration / assignment.
        """
        # Parent handles exact storage + returns full accumulated KV
        k_out, v_out = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Track tokens (only on layer 0 to avoid double-counting)
        if layer_idx == 0:
            self._total_tokens += key_states.shape[2]

        ls = self.layer_states[layer_idx]
        if ls.is_exact:
            return k_out, v_out

        # Feed new tokens into codebook for calibration / streaming assignment
        seq_len = key_states.shape[2]
        for t in range(seq_len):
            k_np = key_states[:, :, t, :].detach().cpu().float().numpy().ravel()
            v_np = value_states[:, :, t, :].detach().cpu().float().numpy().ravel()
            stats = ls.feed_token(k_np, v_np)
            if stats is not None:
                self._calibration_stats.append(stats)

        return k_out, v_out

    def memory_report(self) -> dict:
        """Report memory usage breakdown."""
        exact_bytes = 0
        compressed_bytes = 0
        for ls in self.layer_states:
            mem = ls.memory_bytes()
            compressed_bytes += mem["total_bytes"]

        for layer_idx in range(len(self)):
            try:
                k, v = self[layer_idx]
                exact_bytes += k.element_size() * k.nelement()
                exact_bytes += v.element_size() * v.nelement()
            except (IndexError, AttributeError):
                pass

        return {
            "exact_bytes": exact_bytes,
            "compressed_bytes": compressed_bytes,
            "total_bytes": exact_bytes + compressed_bytes,
            "n_layers": self.n_layers_expected,
            "total_tokens": self._total_tokens,
            "calibration_complete": self.calibration_complete,
        }

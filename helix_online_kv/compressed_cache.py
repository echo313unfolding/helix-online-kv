"""CompressedKVCache: drop-in DynamicCache replacement with online compression.

This module requires torch and transformers. Import guard ensures clean error
messages when running numpy-only experiments.

GPU-native path (2026-03-28): When KV tensors arrive on GPU, calibration and
assignment stay on device via TorchCodebook. No .cpu().numpy() round-trips.

Session persistence (2026-03-28): save_session() / load_session() serialize
the full compressed state (codebooks + indices + DynamicCache KV tensors)
to a single file. Resume a conversation from compressed state without
recomputing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

try:
    import torch
    from transformers.cache_utils import DynamicCache
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    DynamicCache = object  # Fallback so class definition works

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
    """Online compressed KV cache -- subclasses DynamicCache for full HF compat.

    Lifecycle:
        1. Calibration (first N tokens): Accumulates KV values, fits per-layer codebooks.
        2. Streaming (subsequent tokens): New entries assigned via nearest centroid.
        3. Aging: Tokens beyond hot_window demoted from Tier 0 (exact) to Tier 1 (compressed).

    All DynamicCache methods (get_mask_sizes, get_seq_length, __getitem__, etc.)
    are inherited. Only update() is overridden to add compression side-effects.

    When device is specified (e.g. 'cuda'), all codebook operations stay on that
    device. Otherwise falls back to numpy path on CPU.
    """

    def __init__(self, config: OnlineKVConfig, n_layers: int,
                 device: str | "torch.device" | None = None):
        _require_torch()
        super().__init__()
        self.config = config
        self.n_layers_expected = n_layers
        self.aging = AgingPolicy(hot_window=config.hot_window)
        self._device = device

        # Per-layer compression state (device-aware)
        self.layer_states: list[KVLayerState] = [
            KVLayerState(i, config, device=device) for i in range(n_layers)
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
        call to feed data into the codebook for calibration / assignment.

        GPU-native: when layer_states use TorchCodebook, tensors stay on
        device -- no .cpu().numpy() round-trips.
        """
        # Parent handles exact storage + returns full accumulated KV
        k_out, v_out = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Track tokens (only on layer 0 to avoid double-counting)
        if layer_idx == 0:
            self._total_tokens += key_states.shape[2]

        ls = self.layer_states[layer_idx]
        if ls.is_exact:
            return k_out, v_out

        # Feed new tokens into codebook for calibration / streaming assignment.
        # key_states shape: [batch, n_heads, seq_len, head_dim]
        seq_len = key_states.shape[2]

        if ls._use_torch:
            # GPU-native path: flatten [batch, n_heads, head_dim] → [entry_size]
            # and pass torch tensors directly. No CPU round-trip.
            for t in range(seq_len):
                k_flat = key_states[:, :, t, :].detach().float().reshape(-1)
                v_flat = value_states[:, :, t, :].detach().float().reshape(-1)
                stats = ls.feed_token(k_flat, v_flat)
                if stats is not None:
                    self._calibration_stats.append(stats)
        else:
            # Legacy numpy path (CPU models)
            import numpy as np
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

    def save_session(self, path: Union[str, Path]) -> dict:
        """Save full compressed KV cache state to a file.

        Serializes:
            - Config (for reconstruction)
            - Per-layer codebooks (centroids, phase, token counts)
            - Compressed indices (uint8, the bulk of the data)
            - DynamicCache KV tensors (Tier 0 exact tokens)
            - Calibration stats

        The saved file can be loaded with CompressedKVCache.load_session()
        to resume a conversation without recomputing.

        Args:
            path: File path to save to (.pt format).

        Returns:
            Dict with save metadata (n_layers, total_tokens, file_size_bytes).
        """
        path = Path(path)

        # Collect DynamicCache state (the parent's KV tensor storage).
        # HF transformers 4.57+: DynamicCache stores layers in self.layers[]
        # with .keys and .values attributes (DynamicLayer objects).
        kv_tensors = {}
        for layer_idx in range(len(self)):
            try:
                k, v = self[layer_idx]
                kv_tensors[f"key_{layer_idx}"] = k.cpu()
                kv_tensors[f"value_{layer_idx}"] = v.cpu()
            except (IndexError, AttributeError):
                pass

        # Collect layer states
        layer_dicts = []
        for ls in self.layer_states:
            ld = ls.state_dict()
            # Move torch tensor indices to CPU for serialization
            if ls._use_torch and ld.get("k_indices"):
                ld["k_indices"] = [idx.cpu() for idx in ld["k_indices"]]
                ld["v_indices"] = [idx.cpu() for idx in ld["v_indices"]]
            # Move codebook centroids to CPU
            if ls._use_torch and ld.get("k_codebook"):
                k_cb = ld["k_codebook"]
                if k_cb.get("centroids") is not None:
                    k_cb["centroids"] = k_cb["centroids"].cpu()
                v_cb = ld["v_codebook"]
                if v_cb.get("centroids") is not None:
                    v_cb["centroids"] = v_cb["centroids"].cpu()
            layer_dicts.append(ld)

        state = {
            "version": 1,
            "config": {
                "calibration_tokens": self.config.calibration_tokens,
                "hot_window": self.config.hot_window,
                "n_clusters": self.config.n_clusters,
                "max_kmeans_iters": self.config.max_kmeans_iters,
                "rtol": self.config.rtol,
                "drift_threshold": self.config.drift_threshold,
                "drift_ema_alpha": self.config.drift_ema_alpha,
                "exact_layers": list(self.config.exact_layers),
            },
            "n_layers": self.n_layers_expected,
            "total_tokens": self._total_tokens,
            "calibration_stats": self._calibration_stats,
            "layer_states": layer_dicts,
            "kv_tensors": kv_tensors,
        }

        torch.save(state, path)

        file_size = path.stat().st_size
        return {
            "n_layers": self.n_layers_expected,
            "total_tokens": self._total_tokens,
            "calibration_complete": self.calibration_complete,
            "file_size_bytes": file_size,
            "path": str(path),
        }

    @classmethod
    def load_session(cls, path: Union[str, Path],
                     device: str | "torch.device" | None = None,
                     ) -> "CompressedKVCache":
        """Load a saved compressed KV cache session.

        Reconstructs the full CompressedKVCache state including codebooks,
        compressed indices, and DynamicCache KV tensors.

        Args:
            path: File path to load from (.pt format).
            device: Device to place tensors on. If None, uses CPU.

        Returns:
            Reconstructed CompressedKVCache ready for continued generation.
        """
        _require_torch()
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=False)

        if state.get("version", 0) != 1:
            raise ValueError(f"Unknown session version: {state.get('version')}")

        # Reconstruct config
        cfg_dict = state["config"]
        config = OnlineKVConfig(**cfg_dict)

        n_layers = state["n_layers"]
        cache = cls(config, n_layers, device=device)
        cache._total_tokens = state["total_tokens"]
        cache._calibration_stats = state.get("calibration_stats", [])

        # Restore layer states
        for i, ld in enumerate(state["layer_states"]):
            ls = cache.layer_states[i]
            ls._tokens_seen = ld["tokens_seen"]
            ls.phase = LayerPhase(ld["phase"])

            # Restore codebooks
            if ld.get("k_codebook") and ls.k_codebook is not None:
                if hasattr(ls.k_codebook, "load_state_dict"):
                    ls.k_codebook.load_state_dict(ld["k_codebook"])
                    ls.v_codebook.load_state_dict(ld["v_codebook"])
                else:
                    # Numpy OnlineCodebook — restore centroids directly
                    k_cb = ld["k_codebook"]
                    if k_cb.get("centroids") is not None:
                        centroids = k_cb["centroids"]
                        if isinstance(centroids, torch.Tensor):
                            centroids = centroids.numpy()
                        ls.k_codebook._centroids = centroids
                        ls.k_codebook._finalized = k_cb["finalized"]
                    v_cb = ld["v_codebook"]
                    if v_cb.get("centroids") is not None:
                        centroids = v_cb["centroids"]
                        if isinstance(centroids, torch.Tensor):
                            centroids = centroids.numpy()
                        ls.v_codebook._centroids = centroids
                        ls.v_codebook._finalized = v_cb["finalized"]

            # Restore compressed indices
            if ld.get("k_indices"):
                if ls._use_torch:
                    ls._k_indices = [idx.to(device) for idx in ld["k_indices"]]
                    ls._v_indices = [idx.to(device) for idx in ld["v_indices"]]
                else:
                    import numpy as np
                    ls._k_indices = [
                        idx.numpy() if isinstance(idx, torch.Tensor) else idx
                        for idx in ld["k_indices"]
                    ]
                    ls._v_indices = [
                        idx.numpy() if isinstance(idx, torch.Tensor) else idx
                        for idx in ld["v_indices"]
                    ]

        # Restore DynamicCache KV tensors via the parent's update() method.
        # This properly initializes internal DynamicLayer objects.
        kv_tensors = state.get("kv_tensors", {})
        target_device = device if device is not None else "cpu"
        for layer_idx in range(n_layers):
            k_key = f"key_{layer_idx}"
            v_key = f"value_{layer_idx}"
            if k_key in kv_tensors and v_key in kv_tensors:
                k_tensor = kv_tensors[k_key].to(target_device)
                v_tensor = kv_tensors[v_key].to(target_device)
                # Use DynamicCache.update() (grandparent) to restore tensors.
                # Skip our override to avoid re-feeding into codebooks.
                DynamicCache.update(cache, k_tensor, v_tensor, layer_idx)

        return cache

"""Per-layer KV compression state: codebooks + index buffers.

Supports both numpy (CPU-only) and torch (GPU-native) paths.
When a torch device is specified, uses TorchCodebook and stores
indices as torch.Tensor on device -- no CPU round-trips.
"""

from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np

from .codebook import OnlineCodebook
from .config import OnlineKVConfig

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class LayerPhase(Enum):
    CALIBRATING = "calibrating"
    STREAMING = "streaming"
    EXACT = "exact"  # Layer excluded from compression


class KVLayerState:
    """Per-layer state for online KV cache compression.

    Holds separate codebooks for K and V, tracks calibration phase,
    and manages compressed index buffers.

    When device is a torch device (e.g. 'cuda'), uses TorchCodebook and
    stores indices as torch.Tensor on that device. Otherwise uses
    OnlineCodebook with numpy arrays (original behavior).
    """

    def __init__(self, layer_idx: int, config: OnlineKVConfig,
                 device: Union[str, "torch.device", None] = None):
        self.layer_idx = layer_idx
        self.config = config
        self._use_torch = device is not None and _HAS_TORCH
        self._device = device

        # Determine if this layer is exact (never compressed)
        if layer_idx in config.exact_layers:
            self.phase = LayerPhase.EXACT
            self.k_codebook = None
            self.v_codebook = None
        elif self._use_torch:
            from .torch_codebook import TorchCodebook
            self.phase = LayerPhase.CALIBRATING
            self.k_codebook = TorchCodebook(
                n_clusters=config.n_clusters,
                max_iters=config.max_kmeans_iters,
                rtol=config.rtol,
                drift_threshold=config.drift_threshold,
                drift_ema_alpha=config.drift_ema_alpha,
                device=device,
            )
            self.v_codebook = TorchCodebook(
                n_clusters=config.n_clusters,
                max_iters=config.max_kmeans_iters,
                rtol=config.rtol,
                drift_threshold=config.drift_threshold,
                drift_ema_alpha=config.drift_ema_alpha,
                device=device,
            )
        else:
            self.phase = LayerPhase.CALIBRATING
            self.k_codebook = OnlineCodebook(
                n_clusters=config.n_clusters,
                max_iters=config.max_kmeans_iters,
                rtol=config.rtol,
                drift_threshold=config.drift_threshold,
                drift_ema_alpha=config.drift_ema_alpha,
            )
            self.v_codebook = OnlineCodebook(
                n_clusters=config.n_clusters,
                max_iters=config.max_kmeans_iters,
                rtol=config.rtol,
                drift_threshold=config.drift_threshold,
                drift_ema_alpha=config.drift_ema_alpha,
            )

        # Compressed index buffers (populated after calibration)
        self._k_indices: list = []
        self._v_indices: list = []
        self._tokens_seen = 0

    @property
    def tokens_seen(self) -> int:
        return self._tokens_seen

    @property
    def is_exact(self) -> bool:
        return self.phase == LayerPhase.EXACT

    @property
    def is_calibrating(self) -> bool:
        return self.phase == LayerPhase.CALIBRATING

    @property
    def is_streaming(self) -> bool:
        return self.phase == LayerPhase.STREAMING

    @property
    def compressed_token_count(self) -> int:
        """Number of tokens stored as compressed indices."""
        return len(self._k_indices)

    def feed_token(self, k_values, v_values) -> dict | None:
        """Process one token's KV values.

        During calibration: accumulates into codebook.
        After calibration finalize: assigns to nearest centroid.

        Args:
            k_values: Flat K values for this token (numpy array or torch tensor).
            v_values: Flat V values for this token (numpy array or torch tensor).

        Returns:
            Calibration stats dict when finalization happens, None otherwise.
        """
        self._tokens_seen += 1

        if self.phase == LayerPhase.EXACT:
            return None

        if self.phase == LayerPhase.CALIBRATING:
            self.k_codebook.feed_calibration(k_values)
            self.v_codebook.feed_calibration(v_values)

            # Check if we've accumulated enough for calibration
            if self._tokens_seen >= self.config.calibration_tokens:
                k_stats = self.k_codebook.finalize_calibration()
                v_stats = self.v_codebook.finalize_calibration()
                self.phase = LayerPhase.STREAMING
                return {
                    "layer": self.layer_idx,
                    "k_calibration": k_stats,
                    "v_calibration": v_stats,
                }
            return None

        # STREAMING phase: assign to nearest centroid
        k_idx = self.k_codebook.assign(k_values)
        v_idx = self.v_codebook.assign(v_values)
        self._k_indices.append(k_idx)
        self._v_indices.append(v_idx)
        return None

    def feed_tokens_batch(self, k_batch, v_batch) -> list[dict]:
        """Process a batch of tokens' KV values at once.

        More efficient than calling feed_token() in a loop — avoids
        per-token Python overhead during prefill.

        Args:
            k_batch: [seq_len, entry_size] K values (numpy or torch).
            v_batch: [seq_len, entry_size] V values (numpy or torch).

        Returns:
            List of calibration stats dicts (one per finalization event).
        """
        if self._use_torch:
            seq_len = k_batch.shape[0]
        else:
            seq_len = k_batch.shape[0] if hasattr(k_batch, 'shape') else len(k_batch)

        stats_list = []
        for t in range(seq_len):
            stats = self.feed_token(k_batch[t], v_batch[t])
            if stats is not None:
                stats_list.append(stats)
        return stats_list

    def get_compressed_kv(self, token_offset: int):
        """Decode compressed K/V for a specific token.

        Returns:
            (k_decoded, v_decoded) as float32 (numpy or torch depending on mode).
        """
        if self.phase == LayerPhase.EXACT:
            raise RuntimeError(f"Layer {self.layer_idx} is exact -- no compressed data")
        if not self.k_codebook.finalized:
            raise RuntimeError("Codebook not finalized yet")
        k = self.k_codebook.decode(self._k_indices[token_offset])
        v = self.v_codebook.decode(self._v_indices[token_offset])
        return k, v

    def get_all_compressed_k(self):
        """Decode all compressed K values.

        Returns:
            2D float32 array/tensor [n_compressed_tokens, entry_size] or None.
        """
        if not self._k_indices:
            return None
        if self._use_torch:
            indices = torch.stack(self._k_indices)
            return self.k_codebook.decode(indices)
        else:
            indices = np.stack(self._k_indices)
            return self.k_codebook.decode(indices)

    def get_all_compressed_v(self):
        """Decode all compressed V values.

        Returns:
            2D float32 array/tensor [n_compressed_tokens, entry_size] or None.
        """
        if not self._v_indices:
            return None
        if self._use_torch:
            indices = torch.stack(self._v_indices)
            return self.v_codebook.decode(indices)
        else:
            indices = np.stack(self._v_indices)
            return self.v_codebook.decode(indices)

    def memory_bytes(self) -> dict:
        """Report memory usage for this layer's compressed state."""
        if self.phase == LayerPhase.EXACT:
            return {"codebook_bytes": 0, "index_bytes": 0, "total_bytes": 0}

        codebook_bytes = 0
        if self._use_torch:
            if self.k_codebook and self.k_codebook.centroids is not None:
                codebook_bytes += self.k_codebook.centroids.nelement() * self.k_codebook.centroids.element_size()
            if self.v_codebook and self.v_codebook.centroids is not None:
                codebook_bytes += self.v_codebook.centroids.nelement() * self.v_codebook.centroids.element_size()
            index_bytes = sum(idx.nelement() * idx.element_size() for idx in self._k_indices)
            index_bytes += sum(idx.nelement() * idx.element_size() for idx in self._v_indices)
        else:
            if self.k_codebook and self.k_codebook.centroids is not None:
                codebook_bytes += self.k_codebook.centroids.nbytes
            if self.v_codebook and self.v_codebook.centroids is not None:
                codebook_bytes += self.v_codebook.centroids.nbytes
            index_bytes = sum(idx.nbytes for idx in self._k_indices)
            index_bytes += sum(idx.nbytes for idx in self._v_indices)

        return {
            "codebook_bytes": codebook_bytes,
            "index_bytes": index_bytes,
            "total_bytes": codebook_bytes + index_bytes,
        }

    def state_dict(self) -> dict:
        """Export full layer state for serialization (Gap 3: session save/load)."""
        state = {
            "layer_idx": self.layer_idx,
            "phase": self.phase.value,
            "tokens_seen": self._tokens_seen,
            "use_torch": self._use_torch,
        }
        if self.k_codebook is not None and hasattr(self.k_codebook, "state_dict"):
            state["k_codebook"] = self.k_codebook.state_dict()
            state["v_codebook"] = self.v_codebook.state_dict()
        if self._use_torch:
            state["k_indices"] = self._k_indices  # list of torch.Tensor
            state["v_indices"] = self._v_indices
        else:
            state["k_indices"] = self._k_indices  # list of np.ndarray
            state["v_indices"] = self._v_indices
        return state

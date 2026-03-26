"""Per-layer KV compression state: codebooks + index buffers."""

from __future__ import annotations

from enum import Enum

import numpy as np

from .codebook import OnlineCodebook
from .config import OnlineKVConfig


class LayerPhase(Enum):
    CALIBRATING = "calibrating"
    STREAMING = "streaming"
    EXACT = "exact"  # Layer excluded from compression


class KVLayerState:
    """Per-layer state for online KV cache compression.

    Holds separate codebooks for K and V, tracks calibration phase,
    and manages compressed index buffers.
    """

    def __init__(self, layer_idx: int, config: OnlineKVConfig):
        self.layer_idx = layer_idx
        self.config = config

        # Determine if this layer is exact (never compressed)
        if layer_idx in config.exact_layers:
            self.phase = LayerPhase.EXACT
            self.k_codebook = None
            self.v_codebook = None
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
        self._k_indices: list[np.ndarray] = []
        self._v_indices: list[np.ndarray] = []
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

    def feed_token(self, k_values: np.ndarray, v_values: np.ndarray) -> dict | None:
        """Process one token's KV values.

        During calibration: accumulates into codebook.
        After calibration finalize: assigns to nearest centroid.

        Args:
            k_values: Flat float32 K values for this token (n_heads * head_dim).
            v_values: Flat float32 V values for this token.

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

                # Compress all calibration tokens retroactively
                # (They were fed as calibration data, now assign them indices)
                # Note: calibration data was consumed by finalize, so we can't
                # retroactively compress those tokens. They stay in Tier 0 (exact).

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

    def get_compressed_kv(self, token_offset: int) -> tuple[np.ndarray, np.ndarray]:
        """Decode compressed K/V for a specific token.

        Args:
            token_offset: Index into the compressed buffer (0 = first compressed token).

        Returns:
            (k_decoded, v_decoded) as float32 arrays.
        """
        if self.phase == LayerPhase.EXACT:
            raise RuntimeError(f"Layer {self.layer_idx} is exact — no compressed data")
        if not self.k_codebook.finalized:
            raise RuntimeError("Codebook not finalized yet")
        k = self.k_codebook.decode(self._k_indices[token_offset])
        v = self.v_codebook.decode(self._v_indices[token_offset])
        return k, v

    def get_all_compressed_k(self) -> np.ndarray | None:
        """Decode all compressed K values.

        Returns:
            2D float32 array [n_compressed_tokens, entry_size] or None if empty.
        """
        if not self._k_indices:
            return None
        indices = np.stack(self._k_indices)  # [n_tokens, entry_size]
        return self.k_codebook.decode(indices)

    def get_all_compressed_v(self) -> np.ndarray | None:
        """Decode all compressed V values.

        Returns:
            2D float32 array [n_compressed_tokens, entry_size] or None if empty.
        """
        if not self._v_indices:
            return None
        indices = np.stack(self._v_indices)
        return self.v_codebook.decode(indices)

    def memory_bytes(self) -> dict:
        """Report memory usage for this layer's compressed state."""
        if self.phase == LayerPhase.EXACT:
            return {"codebook_bytes": 0, "index_bytes": 0, "total_bytes": 0}

        codebook_bytes = 0
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

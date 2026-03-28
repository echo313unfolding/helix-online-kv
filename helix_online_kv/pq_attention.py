"""PQ Attention State: manages per-layer PQ codebooks and indices for CDC-03.

Used by EchoModel to accelerate the K-side scoring step in hybrid attention.
During calibration, accumulates K vectors per head. After calibration, fits
PQ codebooks and maintains indices incrementally for each new K token.

Compute savings: O(M * sub_clusters + M * seq_len) per query instead of
O(seq_len * head_dim). For M=8, head_dim=64: 8x reduction on scoring step.
"""

from __future__ import annotations

from typing import Optional

import torch

from .torch_product_codebook import TorchProductCodebook, batched_pq_scores


class PQAttentionState:
    """Manages per-layer, per-head PQ codebooks and indices for CDC-03.

    Lifecycle:
        1. Calibration: accumulate() called per layer per step.
        2. Fitting: fit_layer() called when calibration completes.
        3. Streaming: update_indices() called per layer per step.
        4. Scoring: get_scores() returns PQ approximate scores for a query.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        n_subspaces: int = 8,
        sub_clusters: int = 256,
        exact_layers: Optional[set] = None,
        device: str | torch.device = "cpu",
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_subspaces = n_subspaces
        self.sub_clusters = sub_clusters
        self.exact_layers = exact_layers or set()
        self.device = device

        # Per-layer, per-head PQ codebooks (fitted after calibration)
        # {layer_idx: list of n_heads TorchProductCodebook}
        self._codebooks: dict[int, list[TorchProductCodebook]] = {}

        # Packed centroids for batched scoring
        # {layer_idx: [n_heads, M, sub_clusters, sub_dim]}
        self._centroids_packed: dict[int, torch.Tensor] = {}

        # Per-layer PQ indices for K cache
        # {layer_idx: [n_heads, n_cached, M] uint8}
        self._k_indices: dict[int, torch.Tensor] = {}

        # Calibration data (accumulated during calibration phase)
        # {layer_idx: list of [n_heads, n_tokens, head_dim] tensors}
        self._cal_buffers: dict[int, list[torch.Tensor]] = {}

    @property
    def fitted_layers(self) -> set:
        """Set of layer indices with fitted PQ codebooks."""
        return set(self._centroids_packed.keys())

    def accumulate(
        self, key_states: torch.Tensor, layer_idx: int
    ) -> None:
        """Accumulate K vectors during calibration phase.

        Args:
            key_states: [bsz, n_heads, seq_len, head_dim] K tensor
                (after GQA repeat_kv expansion).
            layer_idx: Layer index.
        """
        if layer_idx in self.exact_layers:
            return

        if layer_idx not in self._cal_buffers:
            self._cal_buffers[layer_idx] = []

        # Store K vectors per head: [n_heads, seq_len, head_dim]
        # Take batch 0 (bsz=1 during generation)
        k = key_states[0].detach()  # [n_heads, seq_len, head_dim]
        self._cal_buffers[layer_idx].append(k)

    def fit_layer(
        self,
        layer_idx: int,
        key_states: Optional[torch.Tensor] = None,
    ) -> dict:
        """Fit PQ codebooks for a layer.

        Uses accumulated calibration data. If key_states is provided,
        also computes initial PQ indices for those K tokens.

        Args:
            layer_idx: Layer index.
            key_states: Optional [bsz, n_heads, seq_len, head_dim] current K cache.

        Returns:
            Dict with fitting stats.
        """
        if layer_idx in self.exact_layers:
            return {"layer": layer_idx, "skipped": True}

        # Concatenate calibration data: [n_heads, total_tokens, head_dim]
        if layer_idx in self._cal_buffers and self._cal_buffers[layer_idx]:
            cal_data = torch.cat(
                self._cal_buffers[layer_idx], dim=1
            )  # [n_heads, total_tokens, head_dim]
            del self._cal_buffers[layer_idx]
        elif key_states is not None:
            # No buffered data — use current K cache
            cal_data = key_states[0].detach()  # [n_heads, seq_len, head_dim]
        else:
            return {"layer": layer_idx, "error": "no calibration data"}

        n_heads = cal_data.shape[0]
        codebooks = []
        stats_list = []

        for h in range(n_heads):
            cb = TorchProductCodebook(
                n_subspaces=self.n_subspaces,
                sub_clusters=self.sub_clusters,
                head_dim=self.head_dim,
                device=str(self.device),
            )
            # Feed calibration data for this head
            head_data = cal_data[h]  # [total_tokens, head_dim]
            cb.feed_calibration(head_data)
            stats = cb.finalize_calibration()
            codebooks.append(cb)
            stats_list.append(stats)

        self._codebooks[layer_idx] = codebooks

        # Pack centroids for batched scoring
        packed = torch.stack(
            [cb.pack_centroids() for cb in codebooks]
        )  # [n_heads, M, sub_clusters, sub_dim]
        self._centroids_packed[layer_idx] = packed

        # Compute initial PQ indices for current K cache
        if key_states is not None:
            self._compute_all_indices(key_states, layer_idx)

        return {
            "layer": layer_idx,
            "n_heads": n_heads,
            "n_subspaces": self.n_subspaces,
            "avg_mse": sum(
                s["calibration_mse_total"] for s in stats_list
            ) / len(stats_list),
        }

    def update_indices(
        self, key_states: torch.Tensor, layer_idx: int
    ) -> None:
        """Compute PQ indices for new K tokens (incremental).

        Compares current seq_len with cached index count and computes
        PQ indices for any new tokens.

        Args:
            key_states: [bsz, n_heads, seq_len, head_dim] full K cache.
            layer_idx: Layer index.
        """
        if layer_idx not in self._codebooks:
            return

        seq_len = key_states.shape[2]
        n_cached = 0
        if layer_idx in self._k_indices:
            n_cached = self._k_indices[layer_idx].shape[1]

        if n_cached >= seq_len:
            return  # Already up to date

        if n_cached == 0:
            # First time — compute all
            self._compute_all_indices(key_states, layer_idx)
            return

        # Incremental: compute PQ indices for new tokens only
        new_k = key_states[0, :, n_cached:, :]  # [n_heads, n_new, head_dim]
        codebooks = self._codebooks[layer_idx]
        new_indices_list = []

        for h in range(self.n_heads):
            idx = codebooks[h].assign(new_k[h])  # [n_new, M]
            new_indices_list.append(idx)

        new_indices = torch.stack(new_indices_list)  # [n_heads, n_new, M]
        self._k_indices[layer_idx] = torch.cat(
            [self._k_indices[layer_idx], new_indices], dim=1
        )

    def get_state(
        self, layer_idx: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get PQ centroids and indices for a layer.

        Returns:
            (centroids_packed, pq_k_indices) or (None, None) if not fitted.
            centroids_packed: [n_heads, M, sub_clusters, sub_dim]
            pq_k_indices: [n_heads, n_cached, M] uint8
        """
        centroids = self._centroids_packed.get(layer_idx)
        indices = self._k_indices.get(layer_idx)
        if centroids is None or indices is None:
            return None, None
        return centroids, indices

    def _compute_all_indices(
        self, key_states: torch.Tensor, layer_idx: int
    ) -> None:
        """Compute PQ indices for all K tokens in the cache."""
        codebooks = self._codebooks[layer_idx]
        k_data = key_states[0]  # [n_heads, seq_len, head_dim]
        indices_list = []

        for h in range(self.n_heads):
            idx = codebooks[h].assign(k_data[h])  # [seq_len, M]
            indices_list.append(idx)

        self._k_indices[layer_idx] = torch.stack(
            indices_list
        )  # [n_heads, seq_len, M]

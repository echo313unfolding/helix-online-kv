"""TorchProductCodebook: GPU-native product quantization for compressed-domain attention.

Same algorithm as ProductCodebook but entirely in torch. M independent
TorchVectorCodebook sub-codebooks, each over sub_dim = head_dim/M dimensions.

Key attention primitives (batched for multi-head, multi-query):
    precompute_distance_tables(q) -> [Q, M, sub_clusters] or [M, sub_clusters]
    gather_pq_scores(tables, indices) -> [Q, seq_len] or [seq_len]

Multi-head batched scoring (for integration with _hybrid_attention_torch):
    batched_pq_scores(q, centroids_packed, pq_indices, scale)
        q: [n_heads, q_len, head_dim]
        centroids_packed: [n_heads, M, sub_clusters, sub_dim]
        pq_indices: [n_heads, seq_len, M]
        -> [n_heads, q_len, seq_len]
"""

from __future__ import annotations

import torch

from .torch_vector_codebook import TorchVectorCodebook


class TorchProductCodebook:
    """Product quantization: M independent sub-codebooks over head_dim, in torch."""

    def __init__(
        self,
        n_subspaces: int = 8,
        sub_clusters: int = 256,
        head_dim: int = 64,
        max_iters: int = 30,
        rtol: float = 1e-4,
        device: str | torch.device = "cpu",
    ):
        if head_dim % n_subspaces != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by "
                f"n_subspaces ({n_subspaces})"
            )

        self.n_subspaces = n_subspaces
        self.sub_clusters = sub_clusters
        self.head_dim = head_dim
        self.sub_dim = head_dim // n_subspaces
        self.max_iters = max_iters
        self.rtol = rtol
        self.device = torch.device(device)

        self._codebooks: list[TorchVectorCodebook] = [
            TorchVectorCodebook(
                n_clusters=sub_clusters,
                dim=self.sub_dim,
                max_iters=max_iters,
                rtol=rtol,
                device=device,
            )
            for _ in range(n_subspaces)
        ]
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized

    def feed_calibration(self, vectors: torch.Tensor) -> None:
        """Accumulate [N, head_dim] vectors for calibration.

        Splits each vector into M subspaces and feeds each sub-codebook.
        """
        if self._finalized:
            raise RuntimeError("TorchProductCodebook already finalized")
        vectors = vectors.detach().float()
        if vectors.ndim != 2 or vectors.shape[1] != self.head_dim:
            raise ValueError(
                f"Expected [N, {self.head_dim}], got {vectors.shape}"
            )

        for s in range(self.n_subspaces):
            start = s * self.sub_dim
            end = start + self.sub_dim
            self._codebooks[s].feed_calibration(vectors[:, start:end])

    def finalize_calibration(self) -> dict:
        """Fit per-subspace k-means on accumulated calibration data.

        Returns:
            Dict with per-subspace calibration stats.
        """
        if self._finalized:
            raise RuntimeError("TorchProductCodebook already finalized")
        if not self._codebooks[0]._calibration_buffer:
            raise RuntimeError("No calibration data fed")

        sub_stats = []
        for s in range(self.n_subspaces):
            stats = self._codebooks[s].finalize_calibration()
            sub_stats.append(stats)

        self._finalized = True

        return {
            "n_subspaces": self.n_subspaces,
            "sub_dim": self.sub_dim,
            "head_dim": self.head_dim,
            "sub_clusters": [s["n_clusters"] for s in sub_stats],
            "calibration_mse_total": float(
                torch.tensor(
                    [s["calibration_mse"] for s in sub_stats]
                ).mean().item()
            ),
        }

    def assign(self, vectors: torch.Tensor) -> torch.Tensor:
        """Assign vectors to nearest PQ centroid per subspace.

        Args:
            vectors: [N, head_dim] float32 tensor.

        Returns:
            [N, M] uint8 index tensor (one index per subspace).
        """
        if not self._finalized:
            raise RuntimeError(
                "Not finalized -- call finalize_calibration() first"
            )
        vectors = vectors.detach().float().to(self.device)
        if vectors.ndim != 2 or vectors.shape[1] != self.head_dim:
            raise ValueError(
                f"Expected [N, {self.head_dim}], got {vectors.shape}"
            )

        N = vectors.shape[0]
        indices = torch.zeros(
            N, self.n_subspaces, dtype=torch.uint8, device=self.device
        )

        for s in range(self.n_subspaces):
            start = s * self.sub_dim
            end = start + self.sub_dim
            indices[:, s] = self._codebooks[s].assign(vectors[:, start:end])

        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode [N, M] uint8 indices back to [N, head_dim] float32 vectors.

        Args:
            indices: [N, M] uint8 tensor.

        Returns:
            [N, head_dim] float32 reconstructed vectors.
        """
        if not self._finalized:
            raise RuntimeError("Not finalized")

        N = indices.shape[0]
        result = torch.zeros(N, self.head_dim, device=self.device)

        for s in range(self.n_subspaces):
            start = s * self.sub_dim
            end = start + self.sub_dim
            result[:, start:end] = self._codebooks[s].decode(indices[:, s])

        return result

    # -- Compressed-domain attention primitives --

    def precompute_distance_tables(self, q: torch.Tensor) -> torch.Tensor:
        """Precompute dot-product tables: q_sub @ centroids_s.T per subspace.

        Args:
            q: [head_dim] or [Q, head_dim] query vector(s).

        Returns:
            [M, sub_clusters] or [Q, M, sub_clusters] distance tables.
        """
        if not self._finalized:
            raise RuntimeError("Not finalized")

        squeeze = False
        if q.ndim == 1:
            q = q.unsqueeze(0)
            squeeze = True

        q = q.float().to(self.device)
        Q_len = q.shape[0]
        max_k = max(
            len(cb.centroids) for cb in self._codebooks
        )
        tables = torch.zeros(
            Q_len, self.n_subspaces, max_k, device=self.device
        )

        for s in range(self.n_subspaces):
            start = s * self.sub_dim
            end = start + self.sub_dim
            q_sub = q[:, start:end]                          # [Q, sub_dim]
            cents = self._codebooks[s].centroids              # [k_s, sub_dim]
            k_s = cents.shape[0]
            tables[:, s, :k_s] = q_sub @ cents.T             # [Q, k_s]

        if squeeze:
            return tables.squeeze(0)
        return tables

    def gather_pq_scores(
        self, tables: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Gather and sum PQ scores from precomputed tables.

        For each token t: score_t = sum_s tables[s, indices[t, s]]

        Args:
            tables: [M, sub_clusters] or [Q, M, sub_clusters]
            indices: [seq_len, M] uint8 PQ indices.

        Returns:
            [seq_len] or [Q, seq_len] approximate attention scores (pre-scale).
        """
        if tables.ndim == 2:
            # Single query: tables [M, sub_clusters], indices [S, M]
            seq_len = indices.shape[0]
            scores = torch.zeros(seq_len, device=tables.device)
            for s in range(self.n_subspaces):
                scores += tables[s][indices[:, s].long()]
            return scores
        else:
            # Batched queries: tables [Q, M, sub_clusters], indices [S, M]
            Q_len = tables.shape[0]
            seq_len = indices.shape[0]
            scores = torch.zeros(Q_len, seq_len, device=tables.device)
            for s in range(self.n_subspaces):
                tables_s = tables[:, s, :]           # [Q, sub_clusters]
                idx_s = indices[:, s].long()         # [S]
                idx_exp = idx_s.unsqueeze(0).expand(Q_len, -1)  # [Q, S]
                scores += torch.gather(tables_s, 1, idx_exp)
            return scores

    # -- Quality metrics --

    def cosine_similarity(
        self, vectors: torch.Tensor, indices: torch.Tensor
    ) -> float:
        """Cosine similarity between original vectors and PQ reconstruction."""
        orig = vectors.detach().float().reshape(-1).to(self.device)
        recon = self.decode(indices).reshape(-1)
        dot = float(torch.dot(orig, recon).item())
        na = float(torch.linalg.norm(orig).item())
        nb = float(torch.linalg.norm(recon).item())
        if na < 1e-30 or nb < 1e-30:
            return 0.0
        return dot / (na * nb)

    def quantization_error(
        self, vectors: torch.Tensor, indices: torch.Tensor
    ) -> float:
        """MSE between original vectors and PQ reconstruction."""
        orig = vectors.detach().float().to(self.device)
        recon = self.decode(indices)
        return float(torch.mean((orig - recon) ** 2).item())

    # -- Serialization --

    def state_dict(self) -> dict:
        """Export PQ state for serialization."""
        return {
            "n_subspaces": self.n_subspaces,
            "sub_clusters": self.sub_clusters,
            "head_dim": self.head_dim,
            "finalized": self._finalized,
            "sub_codebooks": [cb.state_dict() for cb in self._codebooks],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore PQ state from serialization."""
        self._finalized = state["finalized"]
        for s, cb_state in enumerate(state["sub_codebooks"]):
            self._codebooks[s].load_state_dict(cb_state)

    def pack_centroids(self) -> torch.Tensor:
        """Pack all sub-codebook centroids into a single tensor.

        Returns:
            [M, sub_clusters, sub_dim] float32 tensor.
            Useful for batched multi-head scoring.
        """
        if not self._finalized:
            raise RuntimeError("Not finalized")
        max_k = max(len(cb.centroids) for cb in self._codebooks)
        packed = torch.zeros(
            self.n_subspaces, max_k, self.sub_dim, device=self.device
        )
        for s in range(self.n_subspaces):
            k_s = len(self._codebooks[s].centroids)
            packed[s, :k_s, :] = self._codebooks[s].centroids
        return packed


def batched_pq_scores(
    q: torch.Tensor,
    centroids_packed: torch.Tensor,
    pq_indices: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Multi-head batched PQ attention scoring.

    Computes approximate attention scores for all heads simultaneously
    using precomputed PQ distance tables. This replaces the O(seq_len * head_dim)
    full Q@K.T with O(M * sub_clusters + M * seq_len) per query.

    For M=8, head_dim=64: 8x compute reduction on the scoring step.

    Args:
        q: [n_heads, q_len, head_dim] query vectors.
        centroids_packed: [n_heads, M, sub_clusters, sub_dim] codebook centroids.
        pq_indices: [n_heads, seq_len, M] uint8 PQ indices for K cache.
        scale: attention scale factor (typically 1/sqrt(head_dim)).

    Returns:
        [n_heads, q_len, seq_len] approximate attention scores.
    """
    n_heads, q_len, head_dim = q.shape
    _, seq_len, M = pq_indices.shape
    sub_dim = head_dim // M

    scores = torch.zeros(n_heads, q_len, seq_len, device=q.device)

    for s in range(M):
        # Query subspace: [n_heads, q_len, sub_dim]
        q_sub = q[:, :, s * sub_dim : (s + 1) * sub_dim]
        # Centroids for this subspace: [n_heads, sub_clusters, sub_dim]
        cents_s = centroids_packed[:, s, :, :]

        # Distance tables: q_sub @ cents_s.T per head via bmm
        # [n_heads, q_len, sub_clusters]
        tables_s = torch.bmm(q_sub, cents_s.transpose(1, 2))

        # Gather: score[h, i, j] += tables_s[h, i, pq_indices[h, j, s]]
        idx_s = pq_indices[:, :, s].long()          # [n_heads, seq_len]
        idx_exp = idx_s.unsqueeze(1).expand(
            -1, q_len, -1
        )                                            # [n_heads, q_len, seq_len]
        scores += torch.gather(tables_s, 2, idx_exp)

    return scores * scale

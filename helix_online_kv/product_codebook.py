"""ProductCodebook: product quantization for high-fidelity compressed-domain attention.

Splits [head_dim] vectors into M contiguous subspaces of sub_dim=head_dim/M
dimensions each. Each subspace gets its own VectorCodebook(n_clusters, sub_dim).
Total representational capacity = sub_clusters^M (vs 256 for full VQ).

At M=8 with head_dim=64, each 8D subspace has 256 centroids. 8D k-means is
dramatically easier than 64D k-means, so reconstruction fidelity is much higher.

Lifecycle (same pattern as VectorCodebook):
    1. feed_calibration(vectors)          -- accumulate [N, head_dim]
    2. finalize_calibration()             -- fit per-subspace k-means
    3. assign(vectors) -> [N, M] uint8    -- nearest centroid per subspace
    4. decode(indices) -> [N, head_dim]   -- reconstruct by concat

Compressed-domain attention primitives:
    precompute_distance_tables(q) -> [M, sub_clusters]
    gather_pq_scores(tables, indices) -> [seq_len]
"""

from __future__ import annotations

import numpy as np

from .vector_codebook import VectorCodebook


class ProductCodebook:
    """Product quantization: M independent sub-codebooks over head_dim."""

    def __init__(
        self,
        n_subspaces: int = 8,
        sub_clusters: int = 256,
        head_dim: int = 64,
        max_iters: int = 30,
        rtol: float = 1e-4,
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

        # One VectorCodebook per subspace
        self._codebooks: list[VectorCodebook] = [
            VectorCodebook(
                n_clusters=sub_clusters,
                head_dim=self.sub_dim,
                max_iters=max_iters,
                rtol=rtol,
            )
            for _ in range(n_subspaces)
        ]
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized

    def feed_calibration(self, vectors: np.ndarray) -> None:
        """Accumulate [N, head_dim] vectors for calibration.

        Splits each vector into M subspaces and feeds each sub-codebook.
        """
        if self._finalized:
            raise RuntimeError("ProductCodebook already finalized")
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.head_dim:
            raise ValueError(
                f"Expected shape [N, {self.head_dim}], got {vectors.shape}"
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
            raise RuntimeError("ProductCodebook already finalized")

        # Check that at least one codebook has data
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
            "calibration_mse_per_sub": [s["calibration_mse"] for s in sub_stats],
            "calibration_mse_total": float(
                np.mean([s["calibration_mse"] for s in sub_stats])
            ),
            "n_iters_per_sub": [s["n_iters"] for s in sub_stats],
        }

    def assign(self, vectors: np.ndarray) -> np.ndarray:
        """Assign vectors to nearest PQ centroid per subspace.

        Args:
            vectors: [N, head_dim] float32 array.

        Returns:
            [N, M] uint8 index array (one index per subspace).
        """
        if not self._finalized:
            raise RuntimeError(
                "ProductCodebook not finalized -- call finalize_calibration() first"
            )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.head_dim:
            raise ValueError(
                f"Expected shape [N, {self.head_dim}], got {vectors.shape}"
            )

        N = vectors.shape[0]
        indices = np.zeros((N, self.n_subspaces), dtype=np.uint8)

        for s in range(self.n_subspaces):
            start = s * self.sub_dim
            end = start + self.sub_dim
            indices[:, s] = self._codebooks[s].assign(vectors[:, start:end])

        return indices

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Decode [N, M] uint8 indices back to [N, head_dim] float32 vectors.

        Args:
            indices: [N, M] uint8 array.

        Returns:
            [N, head_dim] float32 reconstructed vectors.
        """
        if not self._finalized:
            raise RuntimeError("ProductCodebook not finalized")

        indices = np.asarray(indices)
        N = indices.shape[0]
        result = np.zeros((N, self.head_dim), dtype=np.float32)

        for s in range(self.n_subspaces):
            start = s * self.sub_dim
            end = start + self.sub_dim
            result[:, start:end] = self._codebooks[s].decode(indices[:, s])

        return result

    # -- Compressed-domain attention primitives --

    def precompute_distance_tables(self, q: np.ndarray) -> np.ndarray:
        """Precompute dot-product tables: q_sub @ sub_codebook.T per subspace.

        Args:
            q: [head_dim] float32 query vector.

        Returns:
            [M, max_sub_clusters] float32 distance tables.
            For each subspace s: tables[s, j] = q[s*sub_dim:(s+1)*sub_dim] . centroid_j
        """
        if not self._finalized:
            raise RuntimeError("ProductCodebook not finalized")

        q = np.asarray(q, dtype=np.float32).ravel()
        if q.shape[0] != self.head_dim:
            raise ValueError(
                f"Expected query of dim {self.head_dim}, got {q.shape[0]}"
            )

        # Find max actual cluster count across subspaces
        max_k = max(
            len(self._codebooks[s].centroids) for s in range(self.n_subspaces)
        )
        tables = np.zeros((self.n_subspaces, max_k), dtype=np.float32)

        for s in range(self.n_subspaces):
            start = s * self.sub_dim
            end = start + self.sub_dim
            q_sub = q[start:end]
            centroids_s = self._codebooks[s].centroids  # [k_s, sub_dim]
            k_s = len(centroids_s)
            tables[s, :k_s] = q_sub @ centroids_s.T

        return tables

    def gather_pq_scores(
        self, tables: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """Gather and sum PQ scores from precomputed tables.

        For each token t: score_t = sum_s tables[s, indices[t, s]]

        Args:
            tables: [M, sub_clusters] from precompute_distance_tables.
            indices: [seq_len, M] uint8 PQ indices.

        Returns:
            [seq_len] float32 approximate attention scores (pre-scale).
        """
        seq_len = indices.shape[0]
        scores = np.zeros(seq_len, dtype=np.float32)

        for s in range(self.n_subspaces):
            scores += tables[s][indices[:, s]]  # vectorized gather per subspace

        return scores

    # -- Quality metrics --

    def cosine_similarity(
        self, vectors: np.ndarray, indices: np.ndarray
    ) -> float:
        """Cosine similarity between original vectors and PQ reconstruction."""
        orig = np.asarray(vectors, dtype=np.float32).ravel()
        recon = self.decode(indices).ravel()
        dot = float(np.dot(orig, recon))
        na = float(np.linalg.norm(orig))
        nb = float(np.linalg.norm(recon))
        if na < 1e-30 or nb < 1e-30:
            return 0.0
        return dot / (na * nb)

    def quantization_error(
        self, vectors: np.ndarray, indices: np.ndarray
    ) -> float:
        """MSE between original vectors and PQ reconstruction."""
        orig = np.asarray(vectors, dtype=np.float32)
        recon = self.decode(indices)
        return float(np.mean((orig - recon) ** 2))

"""VectorCodebook: vector quantization for compressed-domain attention.

Unlike OnlineCodebook (scalar VQ, [k] centroids), this operates on
[head_dim]-dimensional vectors. Each token's K/V vector maps to ONE uint8
index, enabling precomputed attention: q @ codebook.T once, then gather.

Lifecycle mirrors OnlineCodebook:
    1. feed_calibration(vectors)     -- accumulate [N, head_dim] vectors
    2. finalize_calibration()        -- fit vector k-means, freeze codebook
    3. assign(vectors) -> [N] uint8  -- nearest centroid (L2)
    4. decode(indices) -> [N, head_dim]

Compressed-domain attention primitives:
    precompute_query_scores(q) -> [k]         -- q @ codebook.T
    gather_scores(precomputed, indices) -> [N] -- precomputed[indices]
"""

from __future__ import annotations

import numpy as np


class VectorCodebook:
    """Vector codebook over [head_dim]-dimensional space."""

    def __init__(
        self,
        n_clusters: int = 256,
        head_dim: int = 64,
        max_iters: int = 30,
        rtol: float = 1e-4,
    ):
        self.n_clusters = n_clusters
        self.head_dim = head_dim
        self.max_iters = max_iters
        self.rtol = rtol

        self._calibration_buffer: list[np.ndarray] = []
        self._centroids: np.ndarray | None = None  # [k, head_dim]
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def centroids(self) -> np.ndarray | None:
        return self._centroids

    def feed_calibration(self, vectors: np.ndarray) -> None:
        """Accumulate vectors during calibration phase.

        Args:
            vectors: [N, head_dim] float32 array.
        """
        if self._finalized:
            raise RuntimeError("Codebook already finalized")
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.head_dim:
            raise ValueError(
                f"Expected shape [N, {self.head_dim}], got {vectors.shape}"
            )
        self._calibration_buffer.append(vectors)

    def finalize_calibration(self) -> dict:
        """Fit vector k-means on accumulated calibration data.

        Returns:
            Dict with calibration stats.
        """
        if self._finalized:
            raise RuntimeError("Codebook already finalized")
        if not self._calibration_buffer:
            raise RuntimeError("No calibration data fed")

        data = np.concatenate(self._calibration_buffer, axis=0)
        self._calibration_buffer.clear()

        centroids, assignments, n_iters = self._fit_kmeans(data)
        self._centroids = centroids
        self._finalized = True

        # Calibration MSE
        recon = centroids[assignments]
        mse = float(np.mean((data - recon) ** 2))

        return {
            "n_samples": len(data),
            "n_clusters": len(centroids),
            "head_dim": self.head_dim,
            "n_iters": n_iters,
            "calibration_mse": mse,
        }

    def assign(self, vectors: np.ndarray) -> np.ndarray:
        """Nearest-centroid assignment (L2). Returns uint8 indices.

        Args:
            vectors: [N, head_dim] float32 array.

        Returns:
            [N] uint8 index array.
        """
        if not self._finalized:
            raise RuntimeError(
                "Codebook not finalized -- call finalize_calibration() first"
            )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.head_dim:
            raise ValueError(
                f"Expected shape [N, {self.head_dim}], got {vectors.shape}"
            )
        # L2 distance: ||v - c||^2 = ||v||^2 - 2 v.c + ||c||^2
        # More efficient than full cdist for large N
        v_sq = np.sum(vectors ** 2, axis=1, keepdims=True)   # [N, 1]
        c_sq = np.sum(self._centroids ** 2, axis=1)          # [k]
        dot = vectors @ self._centroids.T                     # [N, k]
        dists = v_sq - 2 * dot + c_sq                        # [N, k]
        return np.argmin(dists, axis=1).astype(np.uint8)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Decode uint8 indices back to [N, head_dim] vectors.

        Args:
            indices: [N] uint8 array.

        Returns:
            [N, head_dim] float32 reconstructed vectors.
        """
        if not self._finalized:
            raise RuntimeError("Codebook not finalized")
        return self._centroids[indices]

    # -- Compressed-domain attention primitives --

    def precompute_query_scores(self, q: np.ndarray) -> np.ndarray:
        """Precompute dot products between query and all centroids.

        Args:
            q: [head_dim] float32 query vector.

        Returns:
            [k] float32 array of q . centroid_i for each centroid.
        """
        if not self._finalized:
            raise RuntimeError("Codebook not finalized")
        q = np.asarray(q, dtype=np.float32).ravel()
        return q @ self._centroids.T  # [k]

    def gather_scores(
        self, precomputed: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """Gather precomputed scores using token indices.

        Args:
            precomputed: [k] float32 from precompute_query_scores.
            indices: [seq_len] uint8 token indices.

        Returns:
            [seq_len] float32 attention scores.
        """
        return precomputed[indices]

    # -- Quality metrics --

    def cosine_similarity(self, vectors: np.ndarray, indices: np.ndarray) -> float:
        """Cosine similarity between original vectors and reconstruction."""
        orig = np.asarray(vectors, dtype=np.float32).ravel()
        recon = self.decode(indices).ravel()
        dot = float(np.dot(orig, recon))
        na = float(np.linalg.norm(orig))
        nb = float(np.linalg.norm(recon))
        if na < 1e-30 or nb < 1e-30:
            return 0.0
        return dot / (na * nb)

    def quantization_error(self, vectors: np.ndarray, indices: np.ndarray) -> float:
        """MSE between original vectors and reconstruction."""
        orig = np.asarray(vectors, dtype=np.float32)
        recon = self.decode(indices)
        return float(np.mean((orig - recon) ** 2))

    # -- Internal --

    def _fit_kmeans(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Vector k-means with percentile-based initialization.

        Uses per-dimension percentile init (same spirit as OnlineCodebook),
        then standard Lloyd's algorithm with L2 distance.

        Returns:
            (centroids [k, head_dim], assignments [N], n_iterations)
        """
        n_samples = len(data)

        # Cap clusters at number of unique vectors
        unique = np.unique(data, axis=0)
        k = min(self.n_clusters, len(unique))

        # Percentile-based initialization: pick vectors at evenly-spaced
        # positions when sorted by L2 norm (spreads across the distribution)
        norms = np.linalg.norm(data, axis=1)
        sorted_idx = np.argsort(norms)
        pick_positions = np.linspace(0, n_samples - 1, k).astype(int)
        centroids = data[sorted_idx[pick_positions]].copy().astype(np.float32)

        # Convergence threshold based on centroid range
        c_range = float(np.max(norms) - np.min(norms))
        if c_range < 1e-30:
            c_range = 1.0
        abs_tol = self.rtol * c_range

        n_iters = 0
        assignments = np.zeros(n_samples, dtype=np.uint8)

        for i in range(self.max_iters):
            n_iters = i + 1

            # Assign: efficient L2 via expansion
            v_sq = np.sum(data ** 2, axis=1, keepdims=True)    # [N, 1]
            c_sq = np.sum(centroids ** 2, axis=1)              # [k]
            dot = data @ centroids.T                            # [N, k]
            dists = v_sq - 2 * dot + c_sq                      # [N, k]
            assignments = np.argmin(dists, axis=1).astype(np.uint8)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = assignments == j
                if np.any(mask):
                    new_centroids[j] = np.mean(data[mask], axis=0)
                else:
                    new_centroids[j] = centroids[j]

            max_delta = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))
            centroids = new_centroids
            if max_delta < abs_tol:
                break

        return centroids, assignments, n_iters

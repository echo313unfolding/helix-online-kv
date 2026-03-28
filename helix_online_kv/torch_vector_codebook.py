"""TorchVectorCodebook: GPU-native vector VQ codebook.

Same algorithm as VectorCodebook (L2-based k-means, nearest-centroid
assignment) but operates entirely in torch. No numpy round-trips.

Drop-in replacement for VectorCodebook when torch tensors are available.
"""

from __future__ import annotations

import torch


class TorchVectorCodebook:
    """Vector codebook over multi-dimensional space, entirely in torch.

    Lifecycle:
        1. feed_calibration(vectors) -- accumulate [N, dim] calibration data
        2. finalize_calibration() -- fit k-means, freeze codebook
        3. assign(vectors) -> [N] uint8 -- nearest centroid (L2)
        4. decode(indices) -> [N, dim] -- reconstruct from centroids
    """

    _ASSIGN_CHUNK_SIZE = 65536  # Cap [N, k] distance matrix at ~64MB

    def __init__(
        self,
        n_clusters: int = 256,
        dim: int = 8,
        max_iters: int = 30,
        rtol: float = 1e-4,
        device: str | torch.device = "cpu",
    ):
        self.n_clusters = n_clusters
        self.dim = dim
        self.max_iters = max_iters
        self.rtol = rtol
        self.device = torch.device(device)

        self._calibration_buffer: list[torch.Tensor] = []
        self._centroids: torch.Tensor | None = None  # [k, dim]
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def centroids(self) -> torch.Tensor | None:
        return self._centroids

    def feed_calibration(self, vectors: torch.Tensor) -> None:
        """Accumulate vectors during calibration phase.

        Args:
            vectors: [N, dim] float32 tensor.
        """
        if self._finalized:
            raise RuntimeError("Codebook already finalized")
        vectors = vectors.detach().float()
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected [N, {self.dim}], got {vectors.shape}")
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

        data = torch.cat(self._calibration_buffer, dim=0).to(self.device)
        self._calibration_buffer.clear()

        centroids, assignments, n_iters = self._fit_kmeans(data)
        self._centroids = centroids
        self._finalized = True

        recon = centroids[assignments.long()]
        mse = float(torch.mean((data - recon) ** 2).item())

        return {
            "n_samples": len(data),
            "n_clusters": len(centroids),
            "dim": self.dim,
            "n_iters": n_iters,
            "calibration_mse": mse,
        }

    def assign(self, vectors: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid assignment (L2). Returns uint8 indices.

        Args:
            vectors: [N, dim] float32 tensor.

        Returns:
            [N] uint8 index tensor.
        """
        if not self._finalized:
            raise RuntimeError(
                "Codebook not finalized -- call finalize_calibration() first"
            )
        vectors = vectors.detach().float().to(self.device)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected [N, {self.dim}], got {vectors.shape}")
        return self._assign_chunked(vectors)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode uint8 indices back to [N, dim] vectors.

        Args:
            indices: [N] uint8 or long tensor.

        Returns:
            [N, dim] float32 reconstructed vectors.
        """
        if not self._finalized:
            raise RuntimeError("Codebook not finalized")
        return self._centroids[indices.long()]

    def quantization_error(
        self, vectors: torch.Tensor, indices: torch.Tensor
    ) -> float:
        """MSE between original vectors and codebook reconstruction."""
        recon = self.decode(indices)
        orig = vectors.detach().float().to(self.device)
        return float(torch.mean((orig - recon) ** 2).item())

    def cosine_similarity(
        self, vectors: torch.Tensor, indices: torch.Tensor
    ) -> float:
        """Cosine similarity between original and reconstructed vectors."""
        orig = vectors.detach().float().reshape(-1).to(self.device)
        recon = self.decode(indices).reshape(-1)
        dot = float(torch.dot(orig, recon).item())
        na = float(torch.linalg.norm(orig).item())
        nb = float(torch.linalg.norm(recon).item())
        if na < 1e-30 or nb < 1e-30:
            return 0.0
        return dot / (na * nb)

    def state_dict(self) -> dict:
        """Export codebook state for serialization."""
        return {
            "centroids": self._centroids,
            "finalized": self._finalized,
            "n_clusters": self.n_clusters,
            "dim": self.dim,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore codebook state from serialization."""
        self._centroids = state["centroids"]
        if self._centroids is not None:
            self._centroids = self._centroids.to(self.device)
        self._finalized = state["finalized"]
        self.n_clusters = state["n_clusters"]

    # -- Internals --

    def _assign_chunked(
        self,
        vectors: torch.Tensor,
        centroids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Nearest-centroid with chunking to cap memory."""
        if centroids is None:
            centroids = self._centroids
        N = len(vectors)
        if N <= self._ASSIGN_CHUNK_SIZE:
            return self._l2_assign(vectors, centroids)

        indices = torch.empty(N, dtype=torch.uint8, device=self.device)
        for start in range(0, N, self._ASSIGN_CHUNK_SIZE):
            end = min(start + self._ASSIGN_CHUNK_SIZE, N)
            indices[start:end] = self._l2_assign(vectors[start:end], centroids)
        return indices

    @staticmethod
    def _l2_assign(
        vectors: torch.Tensor, centroids: torch.Tensor
    ) -> torch.Tensor:
        """L2 nearest-centroid. Returns uint8 indices.

        Uses expansion trick: ||v - c||^2 = ||v||^2 - 2 v.c + ||c||^2
        """
        v_sq = torch.sum(vectors ** 2, dim=1, keepdim=True)  # [N, 1]
        c_sq = torch.sum(centroids ** 2, dim=1)              # [k]
        dot = vectors @ centroids.T                           # [N, k]
        dists = v_sq - 2 * dot + c_sq                        # [N, k]
        return torch.argmin(dists, dim=1).to(torch.uint8)

    def _fit_kmeans(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Vector k-means with norm-based percentile initialization.

        Returns:
            (centroids [k, dim], assignments [N] uint8, n_iterations)
        """
        N = len(data)
        n_unique = len(torch.unique(data, dim=0))
        k = min(self.n_clusters, n_unique)

        # Percentile-based init: pick vectors at evenly-spaced L2 norm positions
        norms = torch.linalg.norm(data, dim=1)
        sorted_idx = torch.argsort(norms)
        pick_pos = torch.linspace(0, N - 1, k, device=data.device).long()
        centroids = data[sorted_idx[pick_pos]].clone().float()

        c_range = float((norms.max() - norms.min()).item())
        if c_range < 1e-30:
            c_range = 1.0
        abs_tol = self.rtol * c_range

        n_iters = 0
        assignments = torch.zeros(N, dtype=torch.uint8, device=data.device)

        for i in range(self.max_iters):
            n_iters = i + 1

            # Assign (chunked to cap memory)
            assignments = self._assign_chunked(data, centroids=centroids)
            assignments_long = assignments.long()

            # Vectorized centroid update via scatter_add
            counts = torch.zeros(k, device=data.device, dtype=data.dtype)
            sums = torch.zeros(k, self.dim, device=data.device, dtype=data.dtype)
            counts.scatter_add_(
                0, assignments_long,
                torch.ones(N, device=data.device, dtype=data.dtype),
            )
            sums.scatter_add_(
                0, assignments_long.unsqueeze(1).expand(-1, self.dim), data,
            )

            alive = counts > 0
            new_centroids = centroids.clone()
            new_centroids[alive] = sums[alive] / counts[alive].unsqueeze(1)

            max_delta = float(
                torch.max(
                    torch.linalg.norm(new_centroids - centroids, dim=1)
                ).item()
            )
            centroids = new_centroids
            if max_delta < abs_tol:
                break

        return centroids, assignments, n_iters

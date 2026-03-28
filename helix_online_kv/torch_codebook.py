"""TorchCodebook: GPU-native scalar VQ codebook.

Same algorithm as OnlineCodebook (percentile-init k-means, nearest-centroid
assignment, EMA drift correction) but operates entirely in torch — no numpy
round-trips, no .cpu() calls. Calibration, assignment, and decode all stay
on the device where the KV cache lives.

Drop-in replacement for OnlineCodebook when torch tensors are available.
"""

from __future__ import annotations

import torch


class TorchCodebook:
    """Scalar codebook that calibrates from initial tokens then assigns via
    nearest-centroid lookup, entirely in torch.

    Lifecycle:
        1. feed_calibration(values) -- accumulate calibration data (on device)
        2. finalize_calibration() -- fit k-means, freeze codebook
        3. assign(values) -- nearest-centroid (uint8 indices), microseconds
        4. maybe_update_centroids(values, error) -- EMA drift correction
    """

    def __init__(
        self,
        n_clusters: int = 256,
        max_iters: int = 10,
        rtol: float = 0.001,
        drift_threshold: float = 0.01,
        drift_ema_alpha: float = 0.05,
        device: str | torch.device = "cpu",
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.rtol = rtol
        self.drift_threshold = drift_threshold
        self.drift_ema_alpha = drift_ema_alpha
        self.device = torch.device(device)

        self._calibration_buffer: list[torch.Tensor] = []
        self._centroids: torch.Tensor | None = None
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def centroids(self) -> torch.Tensor | None:
        return self._centroids

    def feed_calibration(self, values: torch.Tensor) -> None:
        """Accumulate values during calibration phase. Stays on device."""
        if self._finalized:
            raise RuntimeError("Codebook already finalized")
        self._calibration_buffer.append(values.detach().float().reshape(-1))

    def finalize_calibration(self) -> dict:
        """Fit k-means on accumulated calibration data using torch ops.

        Returns:
            Dict with calibration stats (n_samples, n_clusters, n_iters, final_mse).
        """
        if self._finalized:
            raise RuntimeError("Codebook already finalized")
        if not self._calibration_buffer:
            raise RuntimeError("No calibration data fed")

        data = torch.cat(self._calibration_buffer).to(self.device)
        self._calibration_buffer.clear()

        centroids, assignments, n_iters = self._fit_kmeans(data)
        self._centroids = centroids
        self._finalized = True

        recon = centroids[assignments.long()]
        mse = float(torch.mean((data - recon) ** 2).item())

        return {
            "n_samples": len(data),
            "n_clusters": len(centroids),
            "n_iters": n_iters,
            "calibration_mse": mse,
        }

    def assign(self, values: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid assignment. Returns uint8 indices on same device.

        Chunks large inputs to avoid OOM on the [N, K] distance matrix.
        """
        if not self._finalized:
            raise RuntimeError("Codebook not finalized -- call finalize_calibration() first")
        flat = values.detach().float().reshape(-1).to(self.device)
        return self._assign_chunked(flat)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode uint8 indices back to float32 values."""
        if not self._finalized:
            raise RuntimeError("Codebook not finalized")
        return self._centroids[indices.long()]

    def quantization_error(self, values: torch.Tensor, indices: torch.Tensor) -> float:
        """Compute MSE between original values and codebook reconstruction."""
        recon = self.decode(indices)
        orig = values.detach().float().reshape(-1).to(self.device)
        return float(torch.mean((orig - recon) ** 2).item())

    def cosine_similarity(self, values: torch.Tensor, indices: torch.Tensor) -> float:
        """Cosine similarity between original and reconstructed values."""
        orig = values.detach().float().reshape(-1).to(self.device)
        recon = self.decode(indices)
        dot = float(torch.dot(orig, recon).item())
        norm_a = float(torch.linalg.norm(orig).item())
        norm_b = float(torch.linalg.norm(recon).item())
        if norm_a < 1e-30 or norm_b < 1e-30:
            return 0.0
        return dot / (norm_a * norm_b)

    def maybe_update_centroids(self, values: torch.Tensor, error: float) -> bool:
        """EMA drift correction if error exceeds threshold."""
        if not self._finalized:
            return False
        if error <= self.drift_threshold:
            return False

        flat = values.detach().float().reshape(-1).to(self.device)
        indices = self._assign_chunked(flat)
        alpha = self.drift_ema_alpha

        new_centroids = self._centroids.clone()
        indices_long = indices.long()
        for i in range(len(self._centroids)):
            mask = indices_long == i
            if mask.any():
                cluster_mean = flat[mask].mean()
                new_centroids[i] = (1 - alpha) * self._centroids[i] + alpha * cluster_mean

        self._centroids = new_centroids
        return True

    def state_dict(self) -> dict:
        """Export codebook state for serialization."""
        return {
            "centroids": self._centroids,
            "finalized": self._finalized,
            "n_clusters": self.n_clusters,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore codebook state from serialization."""
        self._centroids = state["centroids"]
        if self._centroids is not None:
            self._centroids = self._centroids.to(self.device)
        self._finalized = state["finalized"]
        self.n_clusters = state["n_clusters"]

    # -- internals --

    _ASSIGN_CHUNK_SIZE = 65536  # 64K values per chunk → 64K * 256 * 4 = 64MB peak

    def _assign_chunked(self, flat: torch.Tensor,
                        centroids: torch.Tensor | None = None) -> torch.Tensor:
        """Nearest-centroid with chunking to cap memory at ~64MB.

        Args:
            flat: 1D float32 tensor of values.
            centroids: Override centroids (used during k-means fitting before
                       self._centroids is set). Defaults to self._centroids.
        """
        if centroids is None:
            centroids = self._centroids
        n = len(flat)
        if n <= self._ASSIGN_CHUNK_SIZE:
            dists = torch.abs(flat.unsqueeze(1) - centroids.unsqueeze(0))
            return torch.argmin(dists, dim=1).to(torch.uint8)

        indices = torch.empty(n, dtype=torch.uint8, device=self.device)
        for start in range(0, n, self._ASSIGN_CHUNK_SIZE):
            end = min(start + self._ASSIGN_CHUNK_SIZE, n)
            chunk = flat[start:end]
            dists = torch.abs(chunk.unsqueeze(1) - centroids.unsqueeze(0))
            indices[start:end] = torch.argmin(dists, dim=1).to(torch.uint8)
        return indices

    def _fit_kmeans(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Percentile-init k-means entirely in torch.

        Returns:
            (centroids [K], assignments [N] uint8, n_iterations)
        """
        n_unique = len(torch.unique(data))
        n_clusters = min(self.n_clusters, n_unique)

        # Percentile initialization via torch.quantile
        percentiles = torch.linspace(0, 1, n_clusters, device=data.device)
        centroids = torch.quantile(data.float(), percentiles).float()

        cb_range = float((centroids[-1] - centroids[0]).item())
        if cb_range < 1e-30:
            cb_range = 1.0
        abs_tol = self.rtol * cb_range

        n_iters = 0

        for i in range(self.max_iters):
            n_iters = i + 1

            # Assign (chunked, pass centroids explicitly during fitting)
            assignments = self._assign_chunked(data, centroids=centroids)
            assignments_long = assignments.long()

            # Vectorized centroid update via scatter
            counts = torch.zeros(n_clusters, device=data.device, dtype=data.dtype)
            sums = torch.zeros(n_clusters, device=data.device, dtype=data.dtype)
            counts.scatter_add_(0, assignments_long, torch.ones_like(data))
            sums.scatter_add_(0, assignments_long, data)

            alive = counts > 0
            new_centroids = centroids.clone()
            new_centroids[alive] = sums[alive] / counts[alive]

            max_delta = float(torch.max(torch.abs(new_centroids - centroids)).item())
            centroids = new_centroids
            if max_delta < abs_tol:
                break

        return centroids, assignments, n_iters

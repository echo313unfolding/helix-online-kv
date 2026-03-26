"""OnlineCodebook: calibrate on first N tokens, then stream-assign."""

from __future__ import annotations

import numpy as np


class OnlineCodebook:
    """Scalar codebook that calibrates from initial tokens then assigns via
    nearest-centroid lookup.

    Lifecycle:
        1. feed_calibration(values) — accumulate calibration data
        2. finalize_calibration() — fit k-means, freeze codebook
        3. assign(values) — nearest-centroid (uint8 indices), microseconds
        4. maybe_update_centroids(values, error) — EMA drift correction

    Uses the same percentile-init k-means as helix_substrate.cdna_encoder._simple_kmeans.
    """

    def __init__(
        self,
        n_clusters: int = 256,
        max_iters: int = 10,
        rtol: float = 0.001,
        drift_threshold: float = 0.01,
        drift_ema_alpha: float = 0.05,
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.rtol = rtol
        self.drift_threshold = drift_threshold
        self.drift_ema_alpha = drift_ema_alpha

        self._calibration_buffer: list[np.ndarray] = []
        self._centroids: np.ndarray | None = None
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def centroids(self) -> np.ndarray | None:
        return self._centroids

    def feed_calibration(self, values: np.ndarray) -> None:
        """Accumulate values during calibration phase.

        Args:
            values: Flat float32 array of KV cache values.
        """
        if self._finalized:
            raise RuntimeError("Codebook already finalized")
        self._calibration_buffer.append(values.ravel().astype(np.float32))

    def finalize_calibration(self) -> dict:
        """Fit k-means on accumulated calibration data.

        Returns:
            Dict with calibration stats (n_samples, n_clusters, n_iters, final_mse).
        """
        if self._finalized:
            raise RuntimeError("Codebook already finalized")
        if not self._calibration_buffer:
            raise RuntimeError("No calibration data fed")

        data = np.concatenate(self._calibration_buffer)
        self._calibration_buffer.clear()

        centroids, assignments, n_iters = self._fit_kmeans(data)
        self._centroids = centroids
        self._finalized = True

        # Compute calibration MSE
        recon = centroids[assignments]
        mse = float(np.mean((data - recon) ** 2))

        return {
            "n_samples": len(data),
            "n_clusters": len(centroids),
            "n_iters": n_iters,
            "calibration_mse": mse,
        }

    def assign(self, values: np.ndarray) -> np.ndarray:
        """Nearest-centroid assignment. Returns uint8 indices.

        Args:
            values: Flat float32 array.

        Returns:
            uint8 index array, same length as values.
        """
        if not self._finalized:
            raise RuntimeError("Codebook not finalized — call finalize_calibration() first")
        flat = values.ravel().astype(np.float32)
        dists = np.abs(flat[:, np.newaxis] - self._centroids)
        return np.argmin(dists, axis=1).astype(np.uint8)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Decode uint8 indices back to float32 values.

        Args:
            indices: uint8 index array.

        Returns:
            float32 reconstructed values.
        """
        if not self._finalized:
            raise RuntimeError("Codebook not finalized")
        return self._centroids[indices]

    def quantization_error(self, values: np.ndarray, indices: np.ndarray) -> float:
        """Compute MSE between original values and codebook reconstruction.

        Args:
            values: Original float32 values (flat).
            indices: uint8 indices from assign().

        Returns:
            Mean squared error (float).
        """
        recon = self.decode(indices)
        return float(np.mean((values.ravel().astype(np.float32) - recon) ** 2))

    def cosine_similarity(self, values: np.ndarray, indices: np.ndarray) -> float:
        """Cosine similarity between original and reconstructed values.

        Args:
            values: Original flat float32 values.
            indices: uint8 indices from assign().

        Returns:
            Cosine similarity (float in [-1, 1]).
        """
        orig = values.ravel().astype(np.float32)
        recon = self.decode(indices)
        dot = float(np.dot(orig, recon))
        norm_a = float(np.linalg.norm(orig))
        norm_b = float(np.linalg.norm(recon))
        if norm_a < 1e-30 or norm_b < 1e-30:
            return 0.0
        return dot / (norm_a * norm_b)

    def maybe_update_centroids(self, values: np.ndarray, error: float) -> bool:
        """EMA drift correction if error exceeds threshold.

        Updates centroids via exponential moving average when the quantization
        error on new data exceeds drift_threshold.

        Args:
            values: Recent flat float32 values.
            error: Current quantization MSE on these values.

        Returns:
            True if centroids were updated, False otherwise.
        """
        if not self._finalized:
            return False
        if error <= self.drift_threshold:
            return False

        flat = values.ravel().astype(np.float32)
        indices = self.assign(flat)
        alpha = self.drift_ema_alpha

        new_centroids = self._centroids.copy()
        for i in range(len(self._centroids)):
            mask = indices == i
            if np.any(mask):
                cluster_mean = np.mean(flat[mask])
                new_centroids[i] = (1 - alpha) * self._centroids[i] + alpha * cluster_mean

        self._centroids = new_centroids
        return True

    def _fit_kmeans(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Percentile-init k-means (matches helix_substrate._simple_kmeans).

        Returns:
            (centroids, assignments, n_iterations)
        """
        n_clusters = min(self.n_clusters, len(np.unique(data)))

        # Percentile initialization
        percentiles = np.linspace(0, 100, n_clusters)
        centroids = np.percentile(data, percentiles).astype(np.float32)

        # Convergence threshold
        cb_range = float(centroids[-1] - centroids[0])
        if cb_range < 1e-30:
            cb_range = 1.0
        abs_tol = self.rtol * cb_range

        n_iters = 0
        assignments = np.zeros(len(data), dtype=np.uint8)

        for i in range(self.max_iters):
            n_iters = i + 1
            # Assign to nearest centroid
            dists = np.abs(data[:, np.newaxis] - centroids)
            assignments = np.argmin(dists, axis=1).astype(np.uint8)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(n_clusters):
                mask = assignments == j
                if np.any(mask):
                    new_centroids[j] = np.mean(data[mask])
                else:
                    new_centroids[j] = centroids[j]

            max_delta = float(np.max(np.abs(new_centroids - centroids)))
            centroids = new_centroids
            if max_delta < abs_tol:
                break

        return centroids, assignments, n_iters

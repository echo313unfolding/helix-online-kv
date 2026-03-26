"""Configuration for online KV cache compression."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OnlineKVConfig:
    """Configuration for CompressedKVCache.

    Attributes:
        calibration_tokens: Number of tokens to accumulate before fitting codebook.
        hot_window: Number of recent tokens kept in exact FP16 (Tier 0).
        n_clusters: Codebook size (max 256 for uint8 indices).
        max_kmeans_iters: K-means iterations during calibration.
        rtol: Relative convergence tolerance for k-means.
        drift_threshold: MSE threshold for triggering EMA centroid update.
        drift_ema_alpha: EMA smoothing factor for centroid drift correction.
        exact_layers: Layer indices kept exact (no compression). High-kurtosis
            layers like layer 0 V should go here.
    """
    calibration_tokens: int = 128
    hot_window: int = 256
    n_clusters: int = 256
    max_kmeans_iters: int = 10
    rtol: float = 0.001
    drift_threshold: float = 0.01
    drift_ema_alpha: float = 0.05
    exact_layers: list[int] = field(default_factory=lambda: [0])

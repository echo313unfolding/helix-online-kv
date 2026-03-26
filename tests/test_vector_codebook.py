"""Tests for VectorCodebook: vector quantization and compressed-domain primitives."""

import numpy as np
import pytest

from helix_online_kv.vector_codebook import VectorCodebook


def _make_vectors(n=256, head_dim=64, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, head_dim)).astype(np.float32)


class TestVectorCodebookLifecycle:
    def test_lifecycle(self):
        cb = VectorCodebook(n_clusters=16, head_dim=64)
        assert not cb.finalized
        cb.feed_calibration(_make_vectors(200))
        stats = cb.finalize_calibration()
        assert cb.finalized
        assert stats["n_clusters"] == 16
        assert stats["head_dim"] == 64
        assert cb.centroids.shape == (16, 64)

    def test_assign_returns_uint8(self):
        cb = VectorCodebook(n_clusters=32, head_dim=64)
        cb.feed_calibration(_make_vectors(300))
        cb.finalize_calibration()
        idx = cb.assign(_make_vectors(100, seed=99))
        assert idx.dtype == np.uint8
        assert idx.shape == (100,)

    def test_decode_roundtrip(self):
        """Cosine improves with more clusters; k=N gives near-perfect."""
        # k=N should give near-exact reconstruction (each vector its own centroid)
        data = _make_vectors(64)
        cb = VectorCodebook(n_clusters=64, head_dim=64)
        cb.feed_calibration(data)
        cb.finalize_calibration()
        idx = cb.assign(data)
        cos = cb.cosine_similarity(data, idx)
        assert cos > 0.99, f"k=N cosine should be near-perfect: {cos}"

        # With clustered data (like real KV), k << N still works well
        rng = np.random.default_rng(42)
        centers = rng.standard_normal((16, 64)).astype(np.float32)
        clustered = np.vstack([
            c + rng.standard_normal((20, 64)).astype(np.float32) * 0.3
            for c in centers
        ])  # 320 vectors, 16 true clusters
        cb2 = VectorCodebook(n_clusters=32, head_dim=64)
        cb2.feed_calibration(clustered)
        cb2.finalize_calibration()
        idx2 = cb2.assign(clustered)
        cos2 = cb2.cosine_similarity(clustered, idx2)
        assert cos2 > 0.95, f"Clustered data cosine too low: {cos2}"

    def test_caps_clusters(self):
        """k=256 with only 50 unique vectors -> k capped at 50."""
        cb = VectorCodebook(n_clusters=256, head_dim=64)
        data = _make_vectors(50)
        cb.feed_calibration(data)
        stats = cb.finalize_calibration()
        assert stats["n_clusters"] <= 50

    def test_feed_after_finalize_raises(self):
        cb = VectorCodebook(n_clusters=16, head_dim=64)
        cb.feed_calibration(_make_vectors(100))
        cb.finalize_calibration()
        with pytest.raises(RuntimeError, match="already finalized"):
            cb.feed_calibration(_make_vectors(50))

    def test_assign_before_finalize_raises(self):
        cb = VectorCodebook(n_clusters=16, head_dim=64)
        with pytest.raises(RuntimeError, match="not finalized"):
            cb.assign(_make_vectors(50))

    def test_finalize_empty_raises(self):
        cb = VectorCodebook(n_clusters=16, head_dim=64)
        with pytest.raises(RuntimeError, match="No calibration data"):
            cb.finalize_calibration()

    def test_wrong_dim_raises(self):
        cb = VectorCodebook(n_clusters=16, head_dim=64)
        with pytest.raises(ValueError, match="Expected shape"):
            cb.feed_calibration(np.zeros((10, 32), dtype=np.float32))


class TestCompressedDomainPrimitives:
    def test_precompute_shape(self):
        cb = VectorCodebook(n_clusters=32, head_dim=64)
        cb.feed_calibration(_make_vectors(200))
        cb.finalize_calibration()
        q = np.random.default_rng(7).standard_normal(64).astype(np.float32)
        pre = cb.precompute_query_scores(q)
        assert pre.shape == (32,)

    def test_gather_matches_standard(self):
        """gather_scores(precompute(q), idx) == q @ decode(idx).T exactly."""
        cb = VectorCodebook(n_clusters=64, head_dim=64)
        data = _make_vectors(200)
        cb.feed_calibration(data)
        cb.finalize_calibration()
        idx = cb.assign(data)

        q = np.random.default_rng(7).standard_normal(64).astype(np.float32)

        # Compressed-domain path
        pre = cb.precompute_query_scores(q)
        scores_compressed = cb.gather_scores(pre, idx)

        # Dense path via decode
        K_recon = cb.decode(idx)  # [200, 64]
        scores_dense = q @ K_recon.T  # [200]

        np.testing.assert_allclose(scores_compressed, scores_dense, rtol=1e-5)

    def test_multiple_feed_batches(self):
        """Feeding calibration data in multiple batches works."""
        cb = VectorCodebook(n_clusters=16, head_dim=64)
        cb.feed_calibration(_make_vectors(100, seed=1))
        cb.feed_calibration(_make_vectors(100, seed=2))
        stats = cb.finalize_calibration()
        assert stats["n_samples"] == 200


class TestVectorQuantizationQuality:
    def test_mse_decreases_with_more_clusters(self):
        """MSE on training data should decrease monotonically with k."""
        # Use clustered data for realistic behavior and test on train data
        rng = np.random.default_rng(42)
        centers = rng.standard_normal((32, 64)).astype(np.float32)
        data = np.vstack([
            c + rng.standard_normal((30, 64)).astype(np.float32) * 0.3
            for c in centers
        ])  # 960 vectors with 32 true clusters

        errors = []
        for k in [8, 32, 128]:
            cb = VectorCodebook(n_clusters=k, head_dim=64)
            cb.feed_calibration(data)
            cb.finalize_calibration()
            idx = cb.assign(data)
            errors.append(cb.quantization_error(data, idx))
        assert errors[0] > errors[1] > errors[2], f"MSE should decrease: {errors}"

    def test_quantization_error_positive(self):
        cb = VectorCodebook(n_clusters=32, head_dim=64)
        data = _make_vectors(200)
        cb.feed_calibration(data)
        cb.finalize_calibration()
        idx = cb.assign(data)
        err = cb.quantization_error(data, idx)
        assert err > 0, "Quantization error should be positive with k < N"

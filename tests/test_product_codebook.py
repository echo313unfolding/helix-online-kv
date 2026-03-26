"""Tests for ProductCodebook (product quantization over M subspaces)."""

import numpy as np
import pytest

from helix_online_kv.product_codebook import ProductCodebook
from helix_online_kv.vector_codebook import VectorCodebook


HEAD_DIM = 64


def _make_vectors(n=512, head_dim=HEAD_DIM, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, head_dim)).astype(np.float32)


def _make_clustered_vectors(n=512, head_dim=HEAD_DIM, n_clusters=32, seed=42):
    """Vectors clustered around n_clusters centers (easier for VQ)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, head_dim)).astype(np.float32) * 3
    labels = rng.integers(0, n_clusters, size=n)
    noise = rng.standard_normal((n, head_dim)).astype(np.float32) * 0.3
    return (centers[labels] + noise).astype(np.float32)


class TestProductCodebookLifecycle:
    def test_lifecycle(self):
        """feed -> finalize -> assign -> decode roundtrip."""
        vecs = _make_vectors(256)
        pq = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq.feed_calibration(vecs)
        stats = pq.finalize_calibration()

        assert pq.finalized
        assert stats["n_subspaces"] == 8
        assert stats["sub_dim"] == 8
        assert len(stats["sub_clusters"]) == 8

        idx = pq.assign(vecs)
        recon = pq.decode(idx)
        assert recon.shape == vecs.shape

    def test_assign_shape(self):
        """assign returns [N, M] uint8."""
        vecs = _make_vectors(100)
        pq = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq.feed_calibration(vecs)
        pq.finalize_calibration()

        idx = pq.assign(vecs)
        assert idx.shape == (100, 8)
        assert idx.dtype == np.uint8

    def test_decode_roundtrip(self):
        """cosine > 0.99 on clustered data."""
        vecs = _make_clustered_vectors(512, n_clusters=32)
        pq = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq.feed_calibration(vecs)
        pq.finalize_calibration()

        idx = pq.assign(vecs)
        cos = pq.cosine_similarity(vecs, idx)
        assert cos > 0.99, f"PQ cosine on clustered data: {cos}"

    def test_wrong_dim_raises(self):
        """head_dim mismatch raises ValueError."""
        pq = ProductCodebook(n_subspaces=8, head_dim=64)
        bad = np.zeros((10, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected shape"):
            pq.feed_calibration(bad)

    def test_indivisible_dim_raises(self):
        """head_dim % n_subspaces != 0 raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            ProductCodebook(n_subspaces=7, head_dim=64)

    def test_caps_clusters(self):
        """Fewer unique vectors than sub_clusters doesn't crash."""
        # Only 10 unique vectors — sub_clusters=256 should cap
        rng = np.random.default_rng(77)
        vecs = rng.standard_normal((10, HEAD_DIM)).astype(np.float32)
        pq = ProductCodebook(n_subspaces=8, sub_clusters=256, head_dim=HEAD_DIM)
        pq.feed_calibration(vecs)
        stats = pq.finalize_calibration()

        # Each sub-codebook should have capped its clusters
        for k in stats["sub_clusters"]:
            assert k <= 10

        idx = pq.assign(vecs)
        recon = pq.decode(idx)
        assert recon.shape == vecs.shape


class TestPQCompressedDomainPrimitives:
    def test_distance_tables_shape(self):
        """precompute_distance_tables returns [M, sub_clusters] float32."""
        vecs = _make_vectors(256)
        pq = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq.feed_calibration(vecs)
        pq.finalize_calibration()

        q = np.random.default_rng(99).standard_normal(HEAD_DIM).astype(np.float32)
        tables = pq.precompute_distance_tables(q)
        assert tables.shape[0] == 8
        assert tables.dtype == np.float32

    def test_gather_pq_matches_decode_dot(self):
        """PQ scores ~= q @ decode(idx).T (rtol=1e-5)."""
        vecs = _make_vectors(200)
        pq = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq.feed_calibration(vecs)
        pq.finalize_calibration()

        idx = pq.assign(vecs)
        q = np.random.default_rng(123).standard_normal(HEAD_DIM).astype(np.float32)

        # PQ path: tables + gather
        tables = pq.precompute_distance_tables(q)
        pq_scores = pq.gather_pq_scores(tables, idx)

        # Reference: decode + dense dot
        recon = pq.decode(idx)
        ref_scores = recon @ q

        np.testing.assert_allclose(pq_scores, ref_scores, rtol=1e-4, atol=1e-5)

    def test_multiple_subspace_configs(self):
        """M=4, 8, 16 all work."""
        vecs = _make_vectors(256)
        q = np.random.default_rng(42).standard_normal(HEAD_DIM).astype(np.float32)

        for m in [4, 8, 16]:
            pq = ProductCodebook(n_subspaces=m, sub_clusters=64, head_dim=HEAD_DIM)
            pq.feed_calibration(vecs)
            pq.finalize_calibration()

            idx = pq.assign(vecs)
            assert idx.shape == (256, m)

            tables = pq.precompute_distance_tables(q)
            scores = pq.gather_pq_scores(tables, idx)
            assert scores.shape == (256,)


class TestPQQuantizationQuality:
    def test_mse_decreases_with_more_subspaces(self):
        """M=4 > M=8 > M=16 in MSE (more subspaces = better fidelity)."""
        vecs = _make_vectors(512, seed=77)
        mses = {}

        for m in [4, 8, 16]:
            pq = ProductCodebook(n_subspaces=m, sub_clusters=64, head_dim=HEAD_DIM)
            pq.feed_calibration(vecs)
            pq.finalize_calibration()
            idx = pq.assign(vecs)
            mses[m] = pq.quantization_error(vecs, idx)

        assert mses[4] > mses[8], f"M=4 MSE ({mses[4]}) should > M=8 ({mses[8]})"
        assert mses[8] > mses[16], f"M=8 MSE ({mses[8]}) should > M=16 ({mses[16]})"

    def test_pq_beats_vector_vq(self):
        """Same total centroid storage, PQ has lower MSE than full-space VQ."""
        vecs = _make_vectors(512, seed=88)

        # Full VQ: 256 centroids in 64D → 256 * 64 = 16384 floats
        vq = VectorCodebook(n_clusters=256, head_dim=HEAD_DIM)
        vq.feed_calibration(vecs)
        vq.finalize_calibration()
        vq_idx = vq.assign(vecs)
        vq_mse = vq.quantization_error(vecs, vq_idx)

        # PQ: M=8, 256 per sub → 8 * 256 * 8 = 16384 floats (same storage)
        pq = ProductCodebook(n_subspaces=8, sub_clusters=256, head_dim=HEAD_DIM)
        pq.feed_calibration(vecs)
        pq.finalize_calibration()
        pq_idx = pq.assign(vecs)
        pq_mse = pq.quantization_error(vecs, pq_idx)

        assert pq_mse < vq_mse, (
            f"PQ MSE ({pq_mse}) should be lower than full VQ MSE ({vq_mse})"
        )

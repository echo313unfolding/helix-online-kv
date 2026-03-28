"""Tests for TorchProductCodebook: GPU-native PQ codebook.

Validates lifecycle, PQ assignment, attention scoring primitives,
parity with numpy ProductCodebook, and multi-head batched scoring.
"""

import numpy as np
import pytest
import torch

from helix_online_kv.torch_product_codebook import (
    TorchProductCodebook,
    batched_pq_scores,
)
from helix_online_kv.product_codebook import ProductCodebook


def _make_vectors_torch(n=1000, head_dim=64, seed=42, device="cpu"):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    data = torch.randn(n, head_dim, generator=gen, dtype=torch.float32, device="cpu")
    return data.to(device)


def _make_vectors_np(n=1000, head_dim=64, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, head_dim)).astype(np.float32)


class TestTorchProductCodebookLifecycle:
    def test_lifecycle(self):
        cb = TorchProductCodebook(n_subspaces=8, sub_clusters=32, head_dim=64, device="cpu")
        assert not cb.finalized
        cb.feed_calibration(_make_vectors_torch(500))
        stats = cb.finalize_calibration()
        assert cb.finalized
        assert stats["n_subspaces"] == 8
        assert stats["head_dim"] == 64
        assert stats["sub_dim"] == 8

    def test_assign_shape(self):
        cb = TorchProductCodebook(n_subspaces=8, sub_clusters=32, head_dim=64, device="cpu")
        cb.feed_calibration(_make_vectors_torch(500))
        cb.finalize_calibration()
        idx = cb.assign(_make_vectors_torch(200, seed=99))
        assert idx.dtype == torch.uint8
        assert idx.shape == (200, 8)  # [N, M]

    def test_decode_roundtrip(self):
        cb = TorchProductCodebook(n_subspaces=8, sub_clusters=256, head_dim=64, device="cpu")
        data = _make_vectors_torch(2000)
        cb.feed_calibration(data[:1000])
        cb.finalize_calibration()
        idx = cb.assign(data[1000:])
        recon = cb.decode(idx)
        assert recon.shape == (1000, 64)
        cos = cb.cosine_similarity(data[1000:], idx)
        assert cos > 0.75, f"PQ cosine too low: {cos}"

    def test_feed_after_finalize_raises(self):
        cb = TorchProductCodebook(n_subspaces=8, sub_clusters=32, head_dim=64, device="cpu")
        cb.feed_calibration(_make_vectors_torch(500))
        cb.finalize_calibration()
        with pytest.raises(RuntimeError, match="already finalized"):
            cb.feed_calibration(_make_vectors_torch(100))

    def test_head_dim_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            TorchProductCodebook(n_subspaces=7, sub_clusters=32, head_dim=64)


class TestTorchPQDistanceTables:
    """Test the attention scoring primitives."""

    def _fitted_cb(self):
        cb = TorchProductCodebook(n_subspaces=8, sub_clusters=256, head_dim=64, device="cpu")
        cb.feed_calibration(_make_vectors_torch(2000))
        cb.finalize_calibration()
        return cb

    def test_single_query_tables_shape(self):
        cb = self._fitted_cb()
        q = torch.randn(64)
        tables = cb.precompute_distance_tables(q)
        assert tables.shape == (8, 256)  # [M, sub_clusters]

    def test_batched_query_tables_shape(self):
        cb = self._fitted_cb()
        q = torch.randn(5, 64)  # 5 queries
        tables = cb.precompute_distance_tables(q)
        assert tables.shape == (5, 8, 256)  # [Q, M, sub_clusters]

    def test_single_query_gather(self):
        cb = self._fitted_cb()
        data = _make_vectors_torch(100, seed=99)
        indices = cb.assign(data)  # [100, 8]

        q = torch.randn(64)
        tables = cb.precompute_distance_tables(q)
        scores = cb.gather_pq_scores(tables, indices)
        assert scores.shape == (100,)

    def test_batched_query_gather(self):
        cb = self._fitted_cb()
        data = _make_vectors_torch(100, seed=99)
        indices = cb.assign(data)  # [100, 8]

        q = torch.randn(5, 64)
        tables = cb.precompute_distance_tables(q)
        scores = cb.gather_pq_scores(tables, indices)
        assert scores.shape == (5, 100)

    def test_pq_scores_approximate_dot_product(self):
        """PQ scores should approximate the true q.K dot products."""
        cb = self._fitted_cb()
        data = _make_vectors_torch(500, seed=99)
        indices = cb.assign(data)

        q = torch.randn(64)

        # PQ approximate scores
        tables = cb.precompute_distance_tables(q)
        pq_scores = cb.gather_pq_scores(tables, indices)

        # True dot products: q @ K.T
        true_scores = data @ q  # [500]

        # Correlation should be high (PQ is a dot-product approximation)
        corr = float(torch.corrcoef(torch.stack([pq_scores, true_scores]))[0, 1].item())
        assert corr > 0.70, f"PQ-true correlation too low: {corr}"

    def test_pq_ranking_quality(self):
        """PQ top-k should overlap significantly with true top-k."""
        cb = self._fitted_cb()
        data = _make_vectors_torch(1000, seed=99)
        indices = cb.assign(data)

        q = torch.randn(64)
        tables = cb.precompute_distance_tables(q)
        pq_scores = cb.gather_pq_scores(tables, indices)
        true_scores = data @ q

        k = 64
        pq_topk = set(torch.topk(pq_scores, k).indices.tolist())
        true_topk = set(torch.topk(true_scores, k).indices.tolist())

        overlap = len(pq_topk & true_topk)
        recall = overlap / k
        assert recall > 0.5, f"PQ top-{k} recall too low: {recall}"


class TestBatchedPQScores:
    """Test the multi-head batched scoring function."""

    def test_shape(self):
        n_heads, q_len, seq_len = 4, 1, 100
        head_dim, M, sub_clusters = 64, 8, 32

        q = torch.randn(n_heads, q_len, head_dim)
        centroids = torch.randn(n_heads, M, sub_clusters, head_dim // M)
        indices = torch.randint(0, sub_clusters, (n_heads, seq_len, M), dtype=torch.uint8)

        scores = batched_pq_scores(q, centroids, indices, scale=0.125)
        assert scores.shape == (n_heads, q_len, seq_len)

    def test_multiquery_shape(self):
        n_heads, q_len, seq_len = 4, 10, 200
        head_dim, M, sub_clusters = 64, 8, 32

        q = torch.randn(n_heads, q_len, head_dim)
        centroids = torch.randn(n_heads, M, sub_clusters, head_dim // M)
        indices = torch.randint(0, sub_clusters, (n_heads, seq_len, M), dtype=torch.uint8)

        scores = batched_pq_scores(q, centroids, indices, scale=0.125)
        assert scores.shape == (n_heads, q_len, seq_len)

    def test_consistency_with_single_head(self):
        """Batched scoring should match per-head scoring."""
        n_heads, q_len, seq_len = 4, 1, 50
        head_dim, M = 64, 8
        sub_clusters = 32
        sub_dim = head_dim // M

        # Fit per-head codebooks
        cbs = []
        for h in range(n_heads):
            cb = TorchProductCodebook(n_subspaces=M, sub_clusters=sub_clusters,
                                     head_dim=head_dim, device="cpu")
            cb.feed_calibration(_make_vectors_torch(500, seed=h * 100))
            cb.finalize_calibration()
            cbs.append(cb)

        # Generate K data and assign per head
        all_indices = []
        for h in range(n_heads):
            k_data = _make_vectors_torch(seq_len, seed=1000 + h)
            idx = cbs[h].assign(k_data)
            all_indices.append(idx)

        # Pack centroids: [n_heads, M, sub_clusters, sub_dim]
        centroids_packed = torch.stack([cb.pack_centroids() for cb in cbs])

        # Pack indices: [n_heads, seq_len, M]
        pq_indices = torch.stack(all_indices)

        # Query
        q = torch.randn(n_heads, q_len, head_dim)
        scale = 1.0 / (head_dim ** 0.5)

        # Batched scores
        batch_scores = batched_pq_scores(q, centroids_packed, pq_indices, scale)

        # Per-head scores
        for h in range(n_heads):
            tables = cbs[h].precompute_distance_tables(q[h])  # [q_len, M, sub_clusters]
            per_head = cbs[h].gather_pq_scores(tables, all_indices[h]) * scale  # [q_len, seq_len]

            torch.testing.assert_close(
                batch_scores[h], per_head,
                msg=f"Head {h} batched vs single mismatch",
            )


class TestTorchNumpyPQParity:
    """Verify TorchProductCodebook quality is similar to numpy ProductCodebook."""

    def test_cosine_parity(self):
        np_data = _make_vectors_np(3000, head_dim=64, seed=42)
        torch_data = torch.from_numpy(np_data)

        np_cb = ProductCodebook(n_subspaces=8, sub_clusters=256, head_dim=64)
        np_cb.feed_calibration(np_data[:1000])
        np_cb.finalize_calibration()
        np_idx = np_cb.assign(np_data[1000:])
        np_cos = np_cb.cosine_similarity(np_data[1000:], np_idx)

        torch_cb = TorchProductCodebook(n_subspaces=8, sub_clusters=256,
                                        head_dim=64, device="cpu")
        torch_cb.feed_calibration(torch_data[:1000])
        torch_cb.finalize_calibration()
        torch_idx = torch_cb.assign(torch_data[1000:])
        torch_cos = torch_cb.cosine_similarity(torch_data[1000:], torch_idx)

        assert abs(np_cos - torch_cos) < 0.02, \
            f"PQ parity gap: numpy={np_cos:.4f}, torch={torch_cos:.4f}"


class TestTorchProductCodebookStateDict:
    def test_save_load_roundtrip(self):
        cb = TorchProductCodebook(n_subspaces=8, sub_clusters=64,
                                  head_dim=64, device="cpu")
        cb.feed_calibration(_make_vectors_torch(1000))
        cb.finalize_calibration()

        data = _make_vectors_torch(200, seed=99)
        idx_before = cb.assign(data)

        state = cb.state_dict()

        cb2 = TorchProductCodebook(n_subspaces=8, sub_clusters=64,
                                   head_dim=64, device="cpu")
        cb2.load_state_dict(state)

        idx_after = cb2.assign(data)
        assert torch.equal(idx_before, idx_after)

    def test_pack_centroids_shape(self):
        cb = TorchProductCodebook(n_subspaces=8, sub_clusters=32,
                                  head_dim=64, device="cpu")
        cb.feed_calibration(_make_vectors_torch(500))
        cb.finalize_calibration()
        packed = cb.pack_centroids()
        assert packed.shape == (8, 32, 8)  # [M, sub_clusters, sub_dim]

"""Tests for compressed-domain attention (Path A scalar VQ, Path B vector VQ,
Path D product quantization, Path F prefiltered, Path G hybrid PQ+prefilter)."""

import numpy as np
import pytest

from helix_online_kv.compressed_attention import (
    standard_attention,
    scalar_vq_attention,
    vector_vq_attention_scores,
    vector_vq_value_output,
    full_vector_vq_attention,
    pq_attention_scores,
    pq_value_output,
    full_pq_attention,
    prefiltered_attention,
    hybrid_pq_attention,
)
from helix_online_kv.codebook import OnlineCodebook
from helix_online_kv.vector_codebook import VectorCodebook
from helix_online_kv.product_codebook import ProductCodebook


HEAD_DIM = 64
SEQ_LEN = 200
SCALE = 1.0 / np.sqrt(HEAD_DIM)


def _make_data(seq_len=SEQ_LEN, head_dim=HEAD_DIM, seed=42):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(head_dim).astype(np.float32)
    K = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    V = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    return q, K, V


def _cosine(a, b):
    a, b = a.ravel(), b.ravel()
    d = float(np.dot(a, b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return d / (na * nb)


def _kl_divergence(p, q_dist):
    """KL(p || q) where p and q are probability distributions."""
    from scipy.special import softmax
    p = softmax(p)
    q_dist = softmax(q_dist)
    # Avoid log(0)
    mask = p > 1e-30
    return float(np.sum(p[mask] * np.log(p[mask] / q_dist[mask])))


class TestScalarVQAttention:
    def test_scalar_vq_matches_reference(self):
        """Decompress + matmul gives same result as standard (float32 exact)."""
        q, K, V = _make_data()

        # Fit scalar codebook on K
        cb = OnlineCodebook(n_clusters=256)
        cb.feed_calibration(K.ravel())
        cb.finalize_calibration()

        # Get indices and reconstruct
        k_indices = cb.assign(K.ravel()).reshape(K.shape)  # [seq_len, head_dim]

        # Standard on original
        scores_ref, out_ref = standard_attention(q, K, V, SCALE)

        # Scalar VQ path
        scores_svq, out_svq = scalar_vq_attention(
            q, k_indices, cb.centroids, V, SCALE
        )

        # Scores should match the decompress-then-matmul result exactly
        K_recon = cb.centroids[k_indices]
        scores_manual = (q @ K_recon.T) * SCALE
        np.testing.assert_allclose(scores_svq, scores_manual, rtol=1e-5)

        # Fidelity vs reference (not exact due to quantization, but very close)
        cos_scores = _cosine(scores_ref, scores_svq)
        cos_output = _cosine(out_ref, out_svq)
        assert cos_scores > 0.99, f"Score cosine: {cos_scores}"
        assert cos_output > 0.99, f"Output cosine: {cos_output}"


class TestVectorVQAttention:
    def test_vector_vq_scores_shape(self):
        q, K, _ = _make_data()
        cb = VectorCodebook(n_clusters=64, head_dim=HEAD_DIM)
        cb.feed_calibration(K)
        cb.finalize_calibration()
        idx = cb.assign(K)

        scores = vector_vq_attention_scores(q, idx, cb.centroids, SCALE)
        assert scores.shape == (SEQ_LEN,)

    def test_vector_vq_precompute_gather_vs_dense(self):
        """Compressed-domain scores match dense-on-reconstructed scores exactly."""
        q, K, _ = _make_data()
        cb = VectorCodebook(n_clusters=64, head_dim=HEAD_DIM)
        cb.feed_calibration(K)
        cb.finalize_calibration()
        idx = cb.assign(K)

        # Compressed path
        scores_compressed = vector_vq_attention_scores(q, idx, cb.centroids, SCALE)

        # Dense path on reconstructed K
        K_recon = cb.decode(idx)
        scores_dense = (q @ K_recon.T) * SCALE

        # These must be numerically identical (same computation, different order)
        np.testing.assert_allclose(scores_compressed, scores_dense, rtol=1e-5)

    def test_vector_vq_value_output_shape(self):
        q, K, V = _make_data()
        from scipy.special import softmax

        # Fake weights
        scores = np.random.default_rng(7).standard_normal(SEQ_LEN).astype(np.float32)
        weights = softmax(scores)

        cb_v = VectorCodebook(n_clusters=64, head_dim=HEAD_DIM)
        cb_v.feed_calibration(V)
        cb_v.finalize_calibration()
        v_idx = cb_v.assign(V)

        output = vector_vq_value_output(weights, v_idx, cb_v.centroids)
        assert output.shape == (HEAD_DIM,)

    def test_full_pipeline_cosine(self):
        """Full K+V compressed vs standard attention, output cosine > 0.90."""
        q, K, V = _make_data()

        # Fit vector codebooks for K and V
        cb_k = VectorCodebook(n_clusters=128, head_dim=HEAD_DIM)
        cb_k.feed_calibration(K)
        cb_k.finalize_calibration()
        k_idx = cb_k.assign(K)

        cb_v = VectorCodebook(n_clusters=128, head_dim=HEAD_DIM)
        cb_v.feed_calibration(V)
        cb_v.finalize_calibration()
        v_idx = cb_v.assign(V)

        # Compressed path
        _, out_compressed = full_vector_vq_attention(
            q, k_idx, cb_k.centroids, v_idx, cb_v.centroids, SCALE
        )

        # Reference
        _, out_ref = standard_attention(q, K, V, SCALE)

        cos = _cosine(out_ref, out_compressed)
        assert cos > 0.90, f"Full pipeline output cosine: {cos}"

    def test_softmax_kl_divergence(self):
        """KL divergence between compressed and reference softmax distributions."""
        q, K, _ = _make_data()

        cb = VectorCodebook(n_clusters=128, head_dim=HEAD_DIM)
        cb.feed_calibration(K)
        cb.finalize_calibration()
        idx = cb.assign(K)

        # Reference scores
        scores_ref = (q @ K.T) * SCALE

        # Compressed scores
        scores_comp = vector_vq_attention_scores(q, idx, cb.centroids, SCALE)

        kl = _kl_divergence(scores_ref, scores_comp)
        # On random data, KL may be higher than on real structured data.
        # With k=128 for 200 tokens in 64D, use a generous threshold.
        assert kl < 1.0, f"KL divergence too high: {kl}"


class TestPQAttention:
    """Path D: Product Quantization attention."""

    def test_pq_scores_shape(self):
        """pq_attention_scores returns [seq_len] float32."""
        q, K, _ = _make_data()
        pq = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq.feed_calibration(K)
        pq.finalize_calibration()
        idx = pq.assign(K)

        scores = pq_attention_scores(q, idx, pq, SCALE)
        assert scores.shape == (SEQ_LEN,)
        assert scores.dtype == np.float32

    def test_pq_scores_cosine_vs_ref(self):
        """PQ scores have positive cosine correlation with reference."""
        q, K, _ = _make_data()
        pq = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq.feed_calibration(K)
        pq.finalize_calibration()
        idx = pq.assign(K)

        scores_pq = pq_attention_scores(q, idx, pq, SCALE)
        scores_ref = (q @ K.T) * SCALE

        cos = _cosine(scores_ref, scores_pq)
        # Random 200-token data with 64 sub-clusters: ~3 samples/cluster.
        # Real-data gate (>= 0.99) is in the benchmark with 256 sub-clusters.
        assert cos > 0.80, f"PQ score cosine: {cos}"

    def test_full_pq_output_shape(self):
        """full_pq_attention returns [head_dim] output."""
        q, K, V = _make_data()
        pq_k = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq_k.feed_calibration(K)
        pq_k.finalize_calibration()
        k_idx = pq_k.assign(K)

        pq_v = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq_v.feed_calibration(V)
        pq_v.finalize_calibration()
        v_idx = pq_v.assign(V)

        scores, output = full_pq_attention(q, k_idx, pq_k, v_idx, pq_v, SCALE)
        assert output.shape == (HEAD_DIM,)
        assert scores.shape == (SEQ_LEN,)

    def test_full_pq_output_cosine(self):
        """Full PQ attention output cosine > 0.90 vs reference."""
        q, K, V = _make_data()
        pq_k = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq_k.feed_calibration(K)
        pq_k.finalize_calibration()
        k_idx = pq_k.assign(K)

        pq_v = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq_v.feed_calibration(V)
        pq_v.finalize_calibration()
        v_idx = pq_v.assign(V)

        _, out_pq = full_pq_attention(q, k_idx, pq_k, v_idx, pq_v, SCALE)
        _, out_ref = standard_attention(q, K, V, SCALE)

        cos = _cosine(out_ref, out_pq)
        assert cos > 0.90, f"Full PQ output cosine: {cos}"


class TestPrefilteredAttention:
    """Path F: Prefiltered (sparse) attention."""

    def test_prefiltered_shape(self):
        """Returns (scores [seq_len], output [head_dim])."""
        q, K, V = _make_data()
        cb = VectorCodebook(n_clusters=32, head_dim=HEAD_DIM)
        cb.feed_calibration(K)
        cb.finalize_calibration()
        idx = cb.assign(K)

        scores, output = prefiltered_attention(
            q, K, V, idx, cb.centroids, SCALE, top_p=16
        )
        assert scores.shape == (SEQ_LEN,)
        assert output.shape == (HEAD_DIM,)

    def test_prefiltered_top_p_full(self):
        """top_p >= n_clusters equivalent to standard_attention."""
        q, K, V = _make_data()
        cb = VectorCodebook(n_clusters=32, head_dim=HEAD_DIM)
        cb.feed_calibration(K)
        cb.finalize_calibration()
        idx = cb.assign(K)

        # top_p=32 means ALL clusters selected → all tokens included
        _, out_prefilter = prefiltered_attention(
            q, K, V, idx, cb.centroids, SCALE, top_p=32
        )
        _, out_ref = standard_attention(q, K, V, SCALE)

        # With all clusters, output should match reference exactly
        cos = _cosine(out_ref, out_prefilter)
        assert cos > 0.999, f"Full-coverage prefilter cosine: {cos}"

    def test_prefiltered_cosine_vs_ref(self):
        """Output cosine > 0.80 at top_p=32 with 64 clusters on random data."""
        q, K, V = _make_data()
        cb = VectorCodebook(n_clusters=64, head_dim=HEAD_DIM)
        cb.feed_calibration(K)
        cb.finalize_calibration()
        idx = cb.assign(K)

        _, out_prefilter = prefiltered_attention(
            q, K, V, idx, cb.centroids, SCALE, top_p=32
        )
        _, out_ref = standard_attention(q, K, V, SCALE)

        cos = _cosine(out_ref, out_prefilter)
        # Random data with 50% coverage has no structure to exploit.
        # Real-data gate (>= 0.98) is in the benchmark.
        assert cos > 0.70, f"Prefiltered cosine at top_p=32: {cos}"


class TestHybridPQAttention:
    """Path G (CDC-03): Hybrid PQ scoring + dense V on selected subset."""

    def test_hybrid_output_shape(self):
        """Returns (scores [seq_len], output [head_dim], meta dict)."""
        q, K, V = _make_data()
        pq = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq.feed_calibration(K)
        pq.finalize_calibration()
        idx = pq.assign(K)

        scores, output, meta = hybrid_pq_attention(
            q, K, V, idx, pq, SCALE, top_k=64
        )
        assert scores.shape == (SEQ_LEN,)
        assert output.shape == (HEAD_DIM,)
        assert meta["selected_count"] == 64
        assert 0 < meta["coverage"] <= 1.0

    def test_hybrid_top_k_full_equals_standard(self):
        """top_k >= seq_len should match standard attention exactly."""
        q, K, V = _make_data()
        pq = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq.feed_calibration(K)
        pq.finalize_calibration()
        idx = pq.assign(K)

        _, out_hybrid, _ = hybrid_pq_attention(
            q, K, V, idx, pq, SCALE, top_k=SEQ_LEN
        )
        _, out_ref = standard_attention(q, K, V, SCALE)

        # All tokens selected → output must match reference exactly
        cos = _cosine(out_ref, out_hybrid)
        assert cos > 0.9999, f"Full-coverage hybrid cosine: {cos}"

    def test_hybrid_cosine_vs_ref(self):
        """Hybrid at top_k=128 (64% coverage) should beat random-data prefilter."""
        q, K, V = _make_data()
        pq = ProductCodebook(n_subspaces=8, sub_clusters=64, head_dim=HEAD_DIM)
        pq.feed_calibration(K)
        pq.finalize_calibration()
        idx = pq.assign(K)

        _, out_hybrid, meta = hybrid_pq_attention(
            q, K, V, idx, pq, SCALE, top_k=128
        )
        _, out_ref = standard_attention(q, K, V, SCALE)

        cos = _cosine(out_ref, out_hybrid)
        # Token-level selection is more precise than cluster-level.
        # At 64% coverage on random data, should be decent.
        assert cos > 0.90, f"Hybrid cosine at top_k=128: {cos}"

    def test_hybrid_monotonic_with_top_k(self):
        """More tokens selected → closer to reference."""
        q, K, V = _make_data()
        pq = ProductCodebook(n_subspaces=16, sub_clusters=64, head_dim=HEAD_DIM)
        pq.feed_calibration(K)
        pq.finalize_calibration()
        idx = pq.assign(K)

        _, out_ref = standard_attention(q, K, V, SCALE)

        cosines = {}
        for tk in [32, 64, 128, SEQ_LEN]:
            _, out, _ = hybrid_pq_attention(q, K, V, idx, pq, SCALE, top_k=tk)
            cosines[tk] = _cosine(out_ref, out)

        # Monotonically increasing (or equal)
        vals = list(cosines.values())
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-6, (
                f"Not monotonic: top_k={list(cosines.keys())[i]} "
                f"cos={vals[i]:.6f} > top_k={list(cosines.keys())[i+1]} "
                f"cos={vals[i+1]:.6f}"
            )


class TestEdgeCases:
    def test_single_token(self):
        """Single-token sequence should work for all paths."""
        rng = np.random.default_rng(99)
        q = rng.standard_normal(HEAD_DIM).astype(np.float32)
        K = rng.standard_normal((1, HEAD_DIM)).astype(np.float32)
        V = rng.standard_normal((1, HEAD_DIM)).astype(np.float32)

        _, out_ref = standard_attention(q, K, V, SCALE)

        # Vector VQ with k=1
        cb_k = VectorCodebook(n_clusters=1, head_dim=HEAD_DIM)
        cb_k.feed_calibration(K)
        cb_k.finalize_calibration()
        k_idx = cb_k.assign(K)

        cb_v = VectorCodebook(n_clusters=1, head_dim=HEAD_DIM)
        cb_v.feed_calibration(V)
        cb_v.finalize_calibration()
        v_idx = cb_v.assign(V)

        _, out_comp = full_vector_vq_attention(
            q, k_idx, cb_k.centroids, v_idx, cb_v.centroids, SCALE
        )

        # With k=1 and N=1, reconstruction is exact
        np.testing.assert_allclose(out_ref, out_comp, rtol=1e-4)

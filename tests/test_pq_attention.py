"""Tests for PQ-accelerated attention scoring.

Validates:
1. PQAttentionState lifecycle (accumulate → fit → update → score)
2. PQ decode attention output matches full-attention baseline (approximate)
3. batched_pq_scores correctness
4. _pq_decode_attention correctness vs _hybrid_attention_torch fallback
"""

import sys
from pathlib import Path

import pytest
import torch

from helix_online_kv.torch_product_codebook import TorchProductCodebook, batched_pq_scores
from helix_online_kv.pq_attention import PQAttentionState

# Import the attention functions from echo_runtime
echo_runtime_root = Path(__file__).parent.parent.parent / "echo_runtime"
sys.path.insert(0, str(echo_runtime_root.parent))
from echo_runtime.model_wrapper import _hybrid_attention_torch, _pq_decode_attention


def _make_kv(bsz=1, n_heads=4, seq_len=256, head_dim=64, seed=42):
    """Make random K/V states."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(bsz, n_heads, seq_len, head_dim, generator=gen)
    v = torch.randn(bsz, n_heads, seq_len, head_dim, generator=gen)
    return k, v


def _make_query(bsz=1, n_heads=4, q_len=1, head_dim=64, seed=99):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(bsz, n_heads, q_len, head_dim, generator=gen)


def _fit_pq_codebooks(key_states, n_subspaces=8, sub_clusters=32):
    """Fit per-head PQ codebooks and return (centroids_packed, pq_indices)."""
    bsz, n_heads, seq_len, head_dim = key_states.shape
    codebooks = []
    indices_list = []

    for h in range(n_heads):
        cb = TorchProductCodebook(
            n_subspaces=n_subspaces, sub_clusters=sub_clusters,
            head_dim=head_dim, device="cpu",
        )
        k_data = key_states[0, h]  # [seq_len, head_dim]
        cb.feed_calibration(k_data)
        cb.finalize_calibration()
        codebooks.append(cb)
        indices_list.append(cb.assign(k_data))

    centroids_packed = torch.stack([cb.pack_centroids() for cb in codebooks])
    pq_indices = torch.stack(indices_list)  # [n_heads, seq_len, M]
    return centroids_packed, pq_indices


class TestBatchedPQScoresUnit:
    """Unit tests for the batched_pq_scores function."""

    def test_basic_shape(self):
        n_heads, q_len, seq_len = 4, 1, 100
        head_dim, M, sub_clusters = 64, 8, 32
        sub_dim = head_dim // M

        q = torch.randn(n_heads, q_len, head_dim)
        centroids = torch.randn(n_heads, M, sub_clusters, sub_dim)
        indices = torch.randint(0, sub_clusters, (n_heads, seq_len, M), dtype=torch.uint8)

        scores = batched_pq_scores(q, centroids, indices, scale=0.125)
        assert scores.shape == (n_heads, q_len, seq_len)

    def test_scale_applied(self):
        n_heads, q_len, seq_len = 2, 1, 50
        head_dim, M, sub_clusters = 64, 8, 32
        sub_dim = head_dim // M

        q = torch.randn(n_heads, q_len, head_dim)
        centroids = torch.randn(n_heads, M, sub_clusters, sub_dim)
        indices = torch.randint(0, sub_clusters, (n_heads, seq_len, M), dtype=torch.uint8)

        s1 = batched_pq_scores(q, centroids, indices, scale=1.0)
        s2 = batched_pq_scores(q, centroids, indices, scale=0.5)
        torch.testing.assert_close(s1 * 0.5, s2)


class TestPQDecodeAttention:
    """Test _pq_decode_attention correctness."""

    def test_output_shape(self):
        k, v = _make_kv(seq_len=512)
        q = _make_query()
        centroids, pq_idx = _fit_pq_codebooks(k, sub_clusters=32)

        out = _pq_decode_attention(
            q, k, v, centroids, pq_idx,
            top_k=64, sink_tokens=4, scale=0.125,
        )
        assert out.shape == (1, 4, 1, 64)

    def test_output_matches_baseline_approximate(self):
        """PQ decode output should be close to full-attention baseline.

        Not exact because PQ selects different top-k than true scoring,
        but should be correlated.
        """
        k, v = _make_kv(seq_len=512)
        q = _make_query()
        centroids, pq_idx = _fit_pq_codebooks(k, sub_clusters=64)

        # PQ decode
        pq_out = _pq_decode_attention(
            q, k, v, centroids, pq_idx,
            top_k=128, sink_tokens=4, scale=0.125,
        )

        # Full attention baseline (no PQ, uses full Q@K.T for ranking)
        baseline_out = _hybrid_attention_torch(
            q, k, v, top_k=128, sink_tokens=4, scale=0.125,
        )

        # Both should have same shape
        assert pq_out.shape == baseline_out.shape

        # Cosine similarity should be high (not exact due to different top-k selection)
        pq_flat = pq_out.reshape(-1)
        base_flat = baseline_out.reshape(-1)
        cos = torch.nn.functional.cosine_similarity(
            pq_flat.unsqueeze(0), base_flat.unsqueeze(0)
        ).item()
        assert cos > 0.8, f"PQ vs baseline cosine too low: {cos}"

    def test_with_causal_mask(self):
        """PQ decode should handle causal mask correctly."""
        seq_len = 256
        k, v = _make_kv(seq_len=seq_len)
        q = _make_query()
        centroids, pq_idx = _fit_pq_codebooks(k, sub_clusters=32)

        # Create causal mask (future positions masked out)
        causal_mask = torch.zeros(1, 1, 1, seq_len)
        # Mask out last 50% of positions (simulating attending to first half only)
        causal_mask[:, :, :, seq_len // 2:] = float("-inf")

        out = _pq_decode_attention(
            q, k, v, centroids, pq_idx,
            attention_mask=causal_mask,
            top_k=64, sink_tokens=4, scale=0.125,
        )
        assert out.shape == (1, 4, 1, 64)
        # Output should be finite
        assert torch.isfinite(out).all()


class TestHybridAttentionPQPath:
    """Test _hybrid_attention_torch with PQ args."""

    def test_pq_path_activated_for_decode(self):
        """When PQ args provided and q_len=1, PQ path should be used."""
        k, v = _make_kv(seq_len=512)
        q = _make_query(q_len=1)
        centroids, pq_idx = _fit_pq_codebooks(k, sub_clusters=32)

        out = _hybrid_attention_torch(
            q, k, v,
            top_k=64, sink_tokens=4, scale=0.125,
            pq_centroids_packed=centroids,
            pq_k_indices=pq_idx,
        )
        assert out.shape == (1, 4, 1, 64)

    def test_fallback_for_prefill(self):
        """With q_len>1, should fall back to full Q@K.T even with PQ args."""
        k, v = _make_kv(seq_len=512)
        q = _make_query(q_len=10)  # prefill
        centroids, pq_idx = _fit_pq_codebooks(k, sub_clusters=32)

        out_with_pq = _hybrid_attention_torch(
            q, k, v,
            top_k=64, sink_tokens=4, scale=0.125,
            pq_centroids_packed=centroids,
            pq_k_indices=pq_idx,
        )
        out_without_pq = _hybrid_attention_torch(
            q, k, v,
            top_k=64, sink_tokens=4, scale=0.125,
        )
        # Should be identical (both use full Q@K.T fallback for q_len>1)
        torch.testing.assert_close(out_with_pq, out_without_pq)

    def test_fallback_without_pq(self):
        """Without PQ args, should work as before."""
        k, v = _make_kv(seq_len=512)
        q = _make_query()

        out = _hybrid_attention_torch(
            q, k, v,
            top_k=64, sink_tokens=4, scale=0.125,
        )
        assert out.shape == (1, 4, 1, 64)

    def test_full_attention_short_seq(self):
        """Short sequences should get full attention regardless of PQ."""
        k, v = _make_kv(seq_len=20)
        q = _make_query()

        out = _hybrid_attention_torch(
            q, k, v,
            top_k=64, sink_tokens=4, scale=0.125,
        )
        assert out.shape == (1, 4, 1, 64)


class TestPQAttentionState:
    """Test the PQAttentionState lifecycle manager."""

    def test_accumulate_and_fit(self):
        state = PQAttentionState(
            n_layers=4, n_heads=4, head_dim=64,
            n_subspaces=8, sub_clusters=32,
            exact_layers={0}, device="cpu",
        )

        # Simulate calibration: 50 tokens
        for t in range(50):
            for layer in range(4):
                k, _ = _make_kv(n_heads=4, seq_len=1, seed=t * 4 + layer)
                state.accumulate(k, layer)

        # Layer 0 is exact — should not be fitted
        assert 0 not in state._cal_buffers

        # Layer 1 should have calibration data
        assert 1 in state._cal_buffers
        assert len(state._cal_buffers[1]) == 50

        # Fit layer 1
        k_full, _ = _make_kv(n_heads=4, seq_len=50, seed=999)
        stats = state.fit_layer(1, key_states=k_full)
        assert 1 in state.fitted_layers
        assert stats["n_heads"] == 4

        # Get state should return centroids and indices
        centroids, indices = state.get_state(1)
        assert centroids is not None
        assert centroids.shape[0] == 4  # n_heads
        assert indices is not None
        assert indices.shape == (4, 50, 8)  # [n_heads, seq_len, M]

    def test_incremental_update(self):
        state = PQAttentionState(
            n_layers=2, n_heads=4, head_dim=64,
            n_subspaces=8, sub_clusters=32, device="cpu",
        )

        # Fit with initial 100 tokens
        k_init, _ = _make_kv(n_heads=4, seq_len=100, seed=42)
        state.accumulate(k_init, 0)
        state.fit_layer(0, key_states=k_init)

        _, indices_100 = state.get_state(0)
        assert indices_100.shape[1] == 100

        # Add 10 more tokens
        k_more, _ = _make_kv(n_heads=4, seq_len=110, seed=43)
        state.update_indices(k_more, 0)

        _, indices_110 = state.get_state(0)
        assert indices_110.shape[1] == 110

        # First 100 indices should be unchanged
        torch.testing.assert_close(indices_110[:, :100, :], indices_100)

    def test_exact_layer_ignored(self):
        state = PQAttentionState(
            n_layers=2, n_heads=4, head_dim=64,
            n_subspaces=8, sub_clusters=32,
            exact_layers={0}, device="cpu",
        )

        k, _ = _make_kv(n_heads=4, seq_len=50, seed=42)
        state.accumulate(k, 0)

        # Layer 0 should not have calibration data
        assert 0 not in state._cal_buffers

        # Fitting should return skipped
        stats = state.fit_layer(0)
        assert stats.get("skipped")
        assert 0 not in state.fitted_layers

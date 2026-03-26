"""Tests for fused scalar VQ attention kernel (Triton + numpy reference)."""

import numpy as np
import pytest

from helix_online_kv.triton_attention import fused_scalar_vq_qkt_numpy

HEAD_DIM = 64
SCALE = 1.0 / np.sqrt(HEAD_DIM)


def _make_scalar_vq_data(seq_len=256, head_dim=HEAD_DIM, seed=42):
    """Generate scalar VQ test data: q, k_indices, codebook."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(head_dim).astype(np.float32)
    k_indices = rng.integers(0, 256, size=(seq_len, head_dim)).astype(np.uint8)
    codebook = rng.standard_normal(256).astype(np.float32)
    return q, k_indices, codebook


class TestFusedScalarVQQKT:
    def test_matches_numpy_reference(self):
        """Numpy fused path matches explicit decompress-then-matmul."""
        q, k_indices, codebook = _make_scalar_vq_data(seq_len=200)

        # Fused numpy path
        scores_fused = fused_scalar_vq_qkt_numpy(q, k_indices, codebook, SCALE)

        # Explicit decompress path
        K_recon = codebook[k_indices]  # [200, 64]
        scores_ref = (q @ K_recon.T) * SCALE

        np.testing.assert_allclose(scores_fused, scores_ref, rtol=1e-5)

    def test_output_shape(self):
        """Output is [seq_len] float32."""
        q, k_indices, codebook = _make_scalar_vq_data(seq_len=128)
        scores = fused_scalar_vq_qkt_numpy(q, k_indices, codebook, SCALE)
        assert scores.shape == (128,)
        assert scores.dtype == np.float32

    @pytest.mark.parametrize("seq_len", [1, 16, 64, 256, 1024])
    def test_various_seq_lengths(self, seq_len):
        """Kernel works for various sequence lengths."""
        q, k_indices, codebook = _make_scalar_vq_data(seq_len=seq_len, seed=seq_len)
        scores = fused_scalar_vq_qkt_numpy(q, k_indices, codebook, SCALE)
        assert scores.shape == (seq_len,)

        # Verify correctness
        K_recon = codebook[k_indices]
        scores_ref = (q @ K_recon.T) * SCALE
        np.testing.assert_allclose(scores, scores_ref, rtol=1e-5)


# GPU tests — skip if no CUDA available
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False

try:
    from helix_online_kv.triton_attention import fused_scalar_vq_qkt, _HAS_TRITON
except ImportError:
    _HAS_TRITON = False


@pytest.mark.skipif(not _HAS_CUDA or not _HAS_TRITON, reason="No CUDA/Triton")
class TestFusedKernelGPU:
    def test_gpu_correctness(self):
        """GPU kernel matches CPU reference."""
        q_np, k_indices_np, codebook_np = _make_scalar_vq_data(seq_len=512)

        # CPU reference
        scores_ref = fused_scalar_vq_qkt_numpy(q_np, k_indices_np, codebook_np, SCALE)

        # GPU kernel
        q_gpu = torch.from_numpy(q_np).cuda()
        k_idx_gpu = torch.from_numpy(k_indices_np.astype(np.int32)).cuda()
        cb_gpu = torch.from_numpy(codebook_np).cuda()

        scores_gpu = fused_scalar_vq_qkt(q_gpu, k_idx_gpu, cb_gpu, SCALE)
        scores_gpu_np = scores_gpu.cpu().numpy()

        np.testing.assert_allclose(scores_gpu_np, scores_ref, rtol=1e-4, atol=1e-5)

    def test_bandwidth_improvement(self):
        """Fused kernel faster than dense q@K.T on GPU at 1024 tokens."""
        import time

        seq_len = 1024
        q_np, k_indices_np, codebook_np = _make_scalar_vq_data(seq_len=seq_len, seed=99)

        q_gpu = torch.from_numpy(q_np).cuda()
        k_idx_gpu = torch.from_numpy(k_indices_np.astype(np.int32)).cuda()
        cb_gpu = torch.from_numpy(codebook_np).cuda()

        # Dense reference: materialize K then matmul
        K_dense = torch.from_numpy(codebook_np[k_indices_np]).cuda()

        # Warmup
        for _ in range(10):
            _ = fused_scalar_vq_qkt(q_gpu, k_idx_gpu, cb_gpu, SCALE)
            _ = (q_gpu @ K_dense.T) * SCALE
        torch.cuda.synchronize()

        # Time fused
        n_runs = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = fused_scalar_vq_qkt(q_gpu, k_idx_gpu, cb_gpu, SCALE)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / n_runs

        # Time dense
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = (q_gpu @ K_dense.T) * SCALE
        torch.cuda.synchronize()
        t_dense = (time.perf_counter() - t0) / n_runs

        speedup = t_dense / max(t_fused, 1e-9)
        print(f"\n  Fused: {t_fused*1e6:.1f} us, Dense: {t_dense*1e6:.1f} us, "
              f"Speedup: {speedup:.2f}x")
        # Report but don't gate — bandwidth-bound on T2000 may not show speedup
        # Gate is in bench_compressed_attention.py Part E

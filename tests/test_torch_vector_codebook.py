"""Tests for TorchVectorCodebook: GPU-native vector VQ codebook.

Validates lifecycle, reconstruction quality, parity with numpy VectorCodebook,
and device placement.
"""

import numpy as np
import pytest
import torch

from helix_online_kv.torch_vector_codebook import TorchVectorCodebook
from helix_online_kv.vector_codebook import VectorCodebook


def _make_vectors_torch(n=1000, dim=8, seed=42, device="cpu"):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    data = torch.randn(n, dim, generator=gen, dtype=torch.float32, device="cpu")
    return data.to(device)


def _make_vectors_np(n=1000, dim=8, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


class TestTorchVectorCodebookLifecycle:
    def test_lifecycle(self):
        cb = TorchVectorCodebook(n_clusters=16, dim=8, device="cpu")
        assert not cb.finalized
        cb.feed_calibration(_make_vectors_torch(500))
        stats = cb.finalize_calibration()
        assert cb.finalized
        assert stats["n_clusters"] == 16
        assert stats["dim"] == 8
        assert stats["n_samples"] == 500

    def test_assign_returns_uint8(self):
        cb = TorchVectorCodebook(n_clusters=32, dim=8, device="cpu")
        cb.feed_calibration(_make_vectors_torch(1000))
        cb.finalize_calibration()
        idx = cb.assign(_make_vectors_torch(200, seed=99))
        assert idx.dtype == torch.uint8
        assert idx.shape == (200,)

    def test_decode_roundtrip(self):
        cb = TorchVectorCodebook(n_clusters=256, dim=8, device="cpu")
        data = _make_vectors_torch(2000)
        cb.feed_calibration(data)
        cb.finalize_calibration()
        idx = cb.assign(data)
        recon = cb.decode(idx)
        assert recon.shape == data.shape
        cos = cb.cosine_similarity(data, idx)
        assert cos > 0.80, f"Cosine too low: {cos}"

    def test_feed_after_finalize_raises(self):
        cb = TorchVectorCodebook(n_clusters=16, dim=8, device="cpu")
        cb.feed_calibration(_make_vectors_torch(500))
        cb.finalize_calibration()
        with pytest.raises(RuntimeError, match="already finalized"):
            cb.feed_calibration(_make_vectors_torch(100))

    def test_assign_before_finalize_raises(self):
        cb = TorchVectorCodebook(n_clusters=16, dim=8, device="cpu")
        with pytest.raises(RuntimeError, match="not finalized"):
            cb.assign(_make_vectors_torch(100))

    def test_finalize_empty_raises(self):
        cb = TorchVectorCodebook(n_clusters=16, dim=8, device="cpu")
        with pytest.raises(RuntimeError, match="No calibration data"):
            cb.finalize_calibration()

    def test_wrong_dim_raises(self):
        cb = TorchVectorCodebook(n_clusters=16, dim=8, device="cpu")
        with pytest.raises(ValueError, match="Expected"):
            cb.feed_calibration(torch.randn(10, 16))


class TestTorchVectorCodebookQuality:
    def test_mse_decreases_with_more_clusters(self):
        data = _make_vectors_torch(3000, dim=8)
        errors = []
        for k in [16, 64, 256]:
            cb = TorchVectorCodebook(n_clusters=k, dim=8, device="cpu")
            cb.feed_calibration(data[:1000])
            cb.finalize_calibration()
            idx = cb.assign(data[1000:])
            errors.append(cb.quantization_error(data[1000:], idx))
        assert errors[0] > errors[1] > errors[2], f"MSE should decrease: {errors}"

    def test_generalization(self):
        data = _make_vectors_torch(3000, dim=8)
        cb = TorchVectorCodebook(n_clusters=256, dim=8, device="cpu")
        cb.feed_calibration(data[:1000])
        cb.finalize_calibration()
        idx = cb.assign(data[1000:])
        cos = cb.cosine_similarity(data[1000:], idx)
        assert cos > 0.75, f"Generalization cosine: {cos}"


class TestTorchNumpyVectorParity:
    def test_cosine_parity(self):
        np_data = _make_vectors_np(3000, dim=8, seed=42)
        torch_data = torch.from_numpy(np_data)

        np_cb = VectorCodebook(n_clusters=64, head_dim=8)
        np_cb.feed_calibration(np_data[:1000])
        np_cb.finalize_calibration()
        np_idx = np_cb.assign(np_data[1000:])
        np_cos = np_cb.cosine_similarity(np_data[1000:], np_idx)

        torch_cb = TorchVectorCodebook(n_clusters=64, dim=8, device="cpu")
        torch_cb.feed_calibration(torch_data[:1000])
        torch_cb.finalize_calibration()
        torch_idx = torch_cb.assign(torch_data[1000:])
        torch_cos = torch_cb.cosine_similarity(torch_data[1000:], torch_idx)

        # Both should be >0.95; allow small parity gap
        assert abs(np_cos - torch_cos) < 0.02, \
            f"Parity gap: numpy={np_cos:.4f}, torch={torch_cos:.4f}"


class TestTorchVectorCodebookDevice:
    def test_centroids_on_device(self):
        cb = TorchVectorCodebook(n_clusters=32, dim=8, device="cpu")
        cb.feed_calibration(_make_vectors_torch(500))
        cb.finalize_calibration()
        assert cb.centroids.device == torch.device("cpu")

    def test_indices_on_device(self):
        cb = TorchVectorCodebook(n_clusters=32, dim=8, device="cpu")
        cb.feed_calibration(_make_vectors_torch(500))
        cb.finalize_calibration()
        idx = cb.assign(_make_vectors_torch(100))
        assert idx.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_lifecycle_cuda(self):
        device = "cuda"
        cb = TorchVectorCodebook(n_clusters=64, dim=8, device=device)
        data = _make_vectors_torch(2000, device=device)
        cb.feed_calibration(data[:1000])
        cb.finalize_calibration()
        assert cb.centroids.device.type == "cuda"

        idx = cb.assign(data[1000:])
        assert idx.device.type == "cuda"
        assert idx.dtype == torch.uint8


class TestTorchVectorCodebookStateDict:
    def test_save_load_roundtrip(self):
        cb = TorchVectorCodebook(n_clusters=64, dim=8, device="cpu")
        cb.feed_calibration(_make_vectors_torch(1000))
        cb.finalize_calibration()

        data = _make_vectors_torch(200, seed=99)
        idx_before = cb.assign(data)

        state = cb.state_dict()

        cb2 = TorchVectorCodebook(n_clusters=64, dim=8, device="cpu")
        cb2.load_state_dict(state)

        idx_after = cb2.assign(data)
        assert torch.equal(idx_before, idx_after)

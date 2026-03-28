"""Tests for TorchCodebook: GPU-native scalar VQ codebook.

Validates parity with OnlineCodebook (numpy) and verifies tensors
stay on device throughout the lifecycle.
"""

import numpy as np
import pytest
import torch

from helix_online_kv.torch_codebook import TorchCodebook
from helix_online_kv.codebook import OnlineCodebook


def _make_gaussian_torch(n=10000, seed=42, device="cpu"):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    data = torch.randn(n, generator=gen, dtype=torch.float32, device="cpu")
    return data.to(device)


def _make_gaussian_np(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


class TestTorchCodebookLifecycle:
    def test_lifecycle(self):
        cb = TorchCodebook(n_clusters=16, device="cpu")
        assert not cb.finalized
        cb.feed_calibration(_make_gaussian_torch(1000))
        stats = cb.finalize_calibration()
        assert cb.finalized
        assert stats["n_clusters"] == 16
        assert stats["n_samples"] == 1000

    def test_assign_returns_uint8_tensor(self):
        cb = TorchCodebook(n_clusters=32, device="cpu")
        cb.feed_calibration(_make_gaussian_torch(2000))
        cb.finalize_calibration()
        idx = cb.assign(_make_gaussian_torch(500, seed=99))
        assert idx.dtype == torch.uint8
        assert idx.shape == (500,)

    def test_decode_roundtrip(self):
        cb = TorchCodebook(n_clusters=256, device="cpu")
        data = _make_gaussian_torch(5000)
        cb.feed_calibration(data)
        cb.finalize_calibration()
        idx = cb.assign(data)
        recon = cb.decode(idx)
        cos = torch.nn.functional.cosine_similarity(data.unsqueeze(0), recon.unsqueeze(0)).item()
        assert cos > 0.99, f"Cosine too low: {cos}"

    def test_feed_after_finalize_raises(self):
        cb = TorchCodebook(n_clusters=16, device="cpu")
        cb.feed_calibration(_make_gaussian_torch(1000))
        cb.finalize_calibration()
        with pytest.raises(RuntimeError, match="already finalized"):
            cb.feed_calibration(_make_gaussian_torch(100))

    def test_assign_before_finalize_raises(self):
        cb = TorchCodebook(n_clusters=16, device="cpu")
        with pytest.raises(RuntimeError, match="not finalized"):
            cb.assign(_make_gaussian_torch(100))

    def test_finalize_empty_raises(self):
        cb = TorchCodebook(n_clusters=16, device="cpu")
        with pytest.raises(RuntimeError, match="No calibration data"):
            cb.finalize_calibration()


class TestTorchCodebookGeneralization:
    def test_gaussian_generalization(self):
        gen = torch.Generator(device="cpu").manual_seed(42)
        all_data = torch.randn(10000, generator=gen, dtype=torch.float32)

        cb = TorchCodebook(n_clusters=256, device="cpu")
        cb.feed_calibration(all_data[:2000])
        cb.finalize_calibration()

        test = all_data[2000:]
        idx = cb.assign(test)
        cos = cb.cosine_similarity(test, idx)
        assert cos > 0.998, f"Generalization cosine too low: {cos}"

    def test_bimodal_generalization(self):
        gen = torch.Generator(device="cpu").manual_seed(1)
        a = torch.randn(1000, generator=gen) - 3.0
        b = torch.randn(1000, generator=gen) + 3.0
        train = torch.cat([a, b])

        gen2 = torch.Generator(device="cpu").manual_seed(2)
        a2 = torch.randn(2500, generator=gen2) - 3.0
        b2 = torch.randn(2500, generator=gen2) + 3.0
        test = torch.cat([a2, b2])

        cb = TorchCodebook(n_clusters=256, device="cpu")
        cb.feed_calibration(train)
        cb.finalize_calibration()
        idx = cb.assign(test)
        cos = cb.cosine_similarity(test, idx)
        assert cos > 0.998, f"Bimodal generalization cosine: {cos}"


class TestTorchCodebookDrift:
    def test_drift_triggers_update(self):
        cb = TorchCodebook(n_clusters=64, drift_threshold=0.0001, device="cpu")
        cb.feed_calibration(_make_gaussian_torch(2000, seed=1))
        cb.finalize_calibration()

        shifted = _make_gaussian_torch(1000, seed=2) + 5.0
        idx = cb.assign(shifted)
        error = cb.quantization_error(shifted, idx)
        updated = cb.maybe_update_centroids(shifted, error)
        assert updated, "Expected drift correction to trigger"

    def test_no_drift_on_same_distribution(self):
        cb = TorchCodebook(n_clusters=256, drift_threshold=0.01, device="cpu")
        data = _make_gaussian_torch(5000)
        cb.feed_calibration(data[:2000])
        cb.finalize_calibration()

        test = data[2000:]
        idx = cb.assign(test)
        error = cb.quantization_error(test, idx)
        updated = cb.maybe_update_centroids(test, error)
        assert not updated, f"Unexpected drift correction (error={error:.6f})"


class TestTorchCodebookMSE:
    def test_mse_decreases_with_more_clusters(self):
        data = _make_gaussian_torch(5000)
        errors = []
        for k in [16, 64, 256]:
            cb = TorchCodebook(n_clusters=k, device="cpu")
            cb.feed_calibration(data[:2000])
            cb.finalize_calibration()
            idx = cb.assign(data[2000:])
            errors.append(cb.quantization_error(data[2000:], idx))
        assert errors[0] > errors[1] > errors[2], f"MSE should decrease: {errors}"


class TestTorchNumpyParity:
    """Verify TorchCodebook produces equivalent results to OnlineCodebook."""

    def test_cosine_parity(self):
        """Both codebooks on same data should achieve similar cosine."""
        np_data = _make_gaussian_np(5000, seed=42)
        torch_data = torch.from_numpy(np_data)

        # Numpy path
        np_cb = OnlineCodebook(n_clusters=256)
        np_cb.feed_calibration(np_data[:2000])
        np_cb.finalize_calibration()
        np_idx = np_cb.assign(np_data[2000:])
        np_cos = np_cb.cosine_similarity(np_data[2000:], np_idx)

        # Torch path
        torch_cb = TorchCodebook(n_clusters=256, device="cpu")
        torch_cb.feed_calibration(torch_data[:2000])
        torch_cb.finalize_calibration()
        torch_idx = torch_cb.assign(torch_data[2000:])
        torch_cos = torch_cb.cosine_similarity(torch_data[2000:], torch_idx)

        # Both should be >0.998; allow small difference from float precision
        assert abs(np_cos - torch_cos) < 0.005, \
            f"Parity gap too large: numpy={np_cos:.6f}, torch={torch_cos:.6f}"

    def test_mse_parity(self):
        """Both codebooks should have similar MSE."""
        np_data = _make_gaussian_np(5000, seed=42)
        torch_data = torch.from_numpy(np_data)

        np_cb = OnlineCodebook(n_clusters=256)
        np_cb.feed_calibration(np_data[:2000])
        np_cb.finalize_calibration()
        np_idx = np_cb.assign(np_data[2000:])
        np_mse = np_cb.quantization_error(np_data[2000:], np_idx)

        torch_cb = TorchCodebook(n_clusters=256, device="cpu")
        torch_cb.feed_calibration(torch_data[:2000])
        torch_cb.finalize_calibration()
        torch_idx = torch_cb.assign(torch_data[2000:])
        torch_mse = torch_cb.quantization_error(torch_data[2000:], torch_idx)

        # MSE should be within 20% (different k-means init precision)
        ratio = max(np_mse, torch_mse) / max(min(np_mse, torch_mse), 1e-10)
        assert ratio < 1.2, f"MSE parity: numpy={np_mse:.6f}, torch={torch_mse:.6f}, ratio={ratio:.2f}"


class TestTorchCodebookDevice:
    """Test that tensors stay on the specified device."""

    def test_centroids_on_device(self):
        cb = TorchCodebook(n_clusters=32, device="cpu")
        cb.feed_calibration(_make_gaussian_torch(1000))
        cb.finalize_calibration()
        assert cb.centroids.device == torch.device("cpu")

    def test_indices_on_device(self):
        cb = TorchCodebook(n_clusters=32, device="cpu")
        cb.feed_calibration(_make_gaussian_torch(1000))
        cb.finalize_calibration()
        idx = cb.assign(_make_gaussian_torch(100))
        assert idx.device == torch.device("cpu")

    def test_decode_on_device(self):
        cb = TorchCodebook(n_clusters=32, device="cpu")
        cb.feed_calibration(_make_gaussian_torch(1000))
        cb.finalize_calibration()
        idx = cb.assign(_make_gaussian_torch(100))
        recon = cb.decode(idx)
        assert recon.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_lifecycle_cuda(self):
        """Full lifecycle on CUDA -- nothing should touch CPU."""
        device = "cuda"
        cb = TorchCodebook(n_clusters=64, device=device)
        data = _make_gaussian_torch(5000, device=device)
        cb.feed_calibration(data[:2000])
        stats = cb.finalize_calibration()
        assert cb.centroids.device.type == "cuda"

        idx = cb.assign(data[2000:])
        assert idx.device.type == "cuda"
        assert idx.dtype == torch.uint8

        recon = cb.decode(idx)
        assert recon.device.type == "cuda"

        cos = cb.cosine_similarity(data[2000:], idx)
        assert cos > 0.998, f"CUDA cosine too low: {cos}"


class TestTorchCodebookChunking:
    """Test that chunked assignment matches single-pass."""

    def test_chunked_matches_full(self):
        cb = TorchCodebook(n_clusters=64, device="cpu")
        cb.feed_calibration(_make_gaussian_torch(2000))
        cb.finalize_calibration()

        data = _make_gaussian_torch(200000)  # larger than chunk size

        # Force single-pass by temporarily raising chunk size
        old_chunk = TorchCodebook._ASSIGN_CHUNK_SIZE
        TorchCodebook._ASSIGN_CHUNK_SIZE = 300000
        idx_full = cb.assign(data)
        TorchCodebook._ASSIGN_CHUNK_SIZE = old_chunk

        # Chunked
        idx_chunked = cb.assign(data)
        assert torch.equal(idx_full, idx_chunked)


class TestTorchCodebookStateDict:
    def test_save_load_roundtrip(self):
        cb = TorchCodebook(n_clusters=64, device="cpu")
        cb.feed_calibration(_make_gaussian_torch(2000))
        cb.finalize_calibration()

        data = _make_gaussian_torch(500, seed=99)
        idx_before = cb.assign(data)

        state = cb.state_dict()

        cb2 = TorchCodebook(n_clusters=64, device="cpu")
        cb2.load_state_dict(state)

        idx_after = cb2.assign(data)
        assert torch.equal(idx_before, idx_after)

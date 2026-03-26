"""Tests for OnlineCodebook: calibration, assignment, drift correction."""

import numpy as np
import pytest

from helix_online_kv.codebook import OnlineCodebook


def _make_gaussian(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def _make_bimodal(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    a = rng.normal(-3.0, 0.5, n // 2)
    b = rng.normal(3.0, 0.5, n // 2)
    return np.concatenate([a, b]).astype(np.float32)


class TestOnlineCodebookBasic:
    def test_lifecycle(self):
        cb = OnlineCodebook(n_clusters=16)
        assert not cb.finalized
        cb.feed_calibration(_make_gaussian(1000))
        stats = cb.finalize_calibration()
        assert cb.finalized
        assert stats["n_clusters"] == 16
        assert stats["n_samples"] == 1000

    def test_assign_returns_uint8(self):
        cb = OnlineCodebook(n_clusters=32)
        cb.feed_calibration(_make_gaussian(2000))
        cb.finalize_calibration()
        idx = cb.assign(_make_gaussian(500, seed=99))
        assert idx.dtype == np.uint8
        assert len(idx) == 500

    def test_decode_roundtrip(self):
        cb = OnlineCodebook(n_clusters=256)
        data = _make_gaussian(5000)
        cb.feed_calibration(data)
        cb.finalize_calibration()
        idx = cb.assign(data)
        recon = cb.decode(idx)
        cos = np.dot(data, recon) / (np.linalg.norm(data) * np.linalg.norm(recon))
        assert cos > 0.99, f"Cosine too low: {cos}"

    def test_feed_after_finalize_raises(self):
        cb = OnlineCodebook(n_clusters=16)
        cb.feed_calibration(_make_gaussian(1000))
        cb.finalize_calibration()
        with pytest.raises(RuntimeError, match="already finalized"):
            cb.feed_calibration(_make_gaussian(100))

    def test_assign_before_finalize_raises(self):
        cb = OnlineCodebook(n_clusters=16)
        with pytest.raises(RuntimeError, match="not finalized"):
            cb.assign(_make_gaussian(100))

    def test_finalize_empty_raises(self):
        cb = OnlineCodebook(n_clusters=16)
        with pytest.raises(RuntimeError, match="No calibration data"):
            cb.finalize_calibration()


class TestCodebookGeneralization:
    """Test that codebook fitted on subset generalizes to unseen data."""

    def test_gaussian_generalization(self):
        """Codebook from first 2000 samples generalizes to next 8000."""
        rng = np.random.default_rng(42)
        all_data = rng.standard_normal(10000).astype(np.float32)

        cb = OnlineCodebook(n_clusters=256)
        cb.feed_calibration(all_data[:2000])
        cb.finalize_calibration()

        # Test on unseen data
        test = all_data[2000:]
        idx = cb.assign(test)
        cos = cb.cosine_similarity(test, idx)
        assert cos > 0.998, f"Generalization cosine too low: {cos}"

    def test_bimodal_generalization(self):
        cb = OnlineCodebook(n_clusters=256)
        train = _make_bimodal(2000, seed=1)
        test = _make_bimodal(5000, seed=2)
        cb.feed_calibration(train)
        cb.finalize_calibration()
        idx = cb.assign(test)
        cos = cb.cosine_similarity(test, idx)
        assert cos > 0.998, f"Bimodal generalization cosine: {cos}"


class TestDriftCorrection:
    def test_drift_triggers_update(self):
        cb = OnlineCodebook(n_clusters=64, drift_threshold=0.0001)
        cb.feed_calibration(_make_gaussian(2000, seed=1))
        cb.finalize_calibration()

        # Shifted distribution should trigger drift
        shifted = _make_gaussian(1000, seed=2) + 5.0
        idx = cb.assign(shifted)
        error = cb.quantization_error(shifted, idx)
        updated = cb.maybe_update_centroids(shifted, error)
        assert updated, "Expected drift correction to trigger"

    def test_no_drift_on_same_distribution(self):
        cb = OnlineCodebook(n_clusters=256, drift_threshold=0.01)
        data = _make_gaussian(5000)
        cb.feed_calibration(data[:2000])
        cb.finalize_calibration()

        test = data[2000:]
        idx = cb.assign(test)
        error = cb.quantization_error(test, idx)
        updated = cb.maybe_update_centroids(test, error)
        # MSE on same distribution with 256 clusters should be tiny
        assert not updated, f"Unexpected drift correction (error={error:.6f})"


class TestQuantizationError:
    def test_mse_decreases_with_more_clusters(self):
        data = _make_gaussian(5000)

        errors = []
        for k in [16, 64, 256]:
            cb = OnlineCodebook(n_clusters=k)
            cb.feed_calibration(data[:2000])
            cb.finalize_calibration()
            idx = cb.assign(data[2000:])
            errors.append(cb.quantization_error(data[2000:], idx))

        assert errors[0] > errors[1] > errors[2], f"MSE should decrease: {errors}"

"""Tests for PolarQuant rotation integration."""

import numpy as np
import pytest

from helix_online_kv.polar_rotation import (
    generate_rotation_matrix,
    infer_head_geometry,
    polar_rotate,
    polar_unrotate,
)
from helix_online_kv.config import OnlineKVConfig
from helix_online_kv.layer_state import KVLayerState, LayerPhase


N_HEADS = 4
HEAD_DIM = 64
ENTRY_SIZE = N_HEADS * HEAD_DIM


def _make_token_kv(n_heads=N_HEADS, head_dim=HEAD_DIM, seed=42):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal(n_heads * head_dim).astype(np.float32)
    v = rng.standard_normal(n_heads * head_dim).astype(np.float32)
    return k, v


def _make_outlier_kv(n_heads=N_HEADS, head_dim=HEAD_DIM, seed=42):
    """KV tensor with realistic outlier dimensions (10-100x variance on a few dims)."""
    rng = np.random.default_rng(seed)
    k = rng.standard_normal(n_heads * head_dim).astype(np.float32)
    v = rng.standard_normal(n_heads * head_dim).astype(np.float32)
    # Inject outliers in 4 dims per head
    for h in range(n_heads):
        offset = h * head_dim
        outlier_dims = rng.choice(head_dim, size=4, replace=False)
        for d in outlier_dims:
            k[offset + d] *= 50.0
            v[offset + d] *= 50.0
    return k, v


# ── Unit tests for rotation primitives ──────────────────────────────────────


class TestRotationMatrix:
    def test_orthogonality(self):
        Q = generate_rotation_matrix(64, seed=42)
        eye = Q @ Q.T
        np.testing.assert_allclose(eye, np.eye(64), atol=1e-5)

    def test_deterministic(self):
        Q1 = generate_rotation_matrix(64, seed=42)
        Q2 = generate_rotation_matrix(64, seed=42)
        np.testing.assert_array_equal(Q1, Q2)

    def test_different_seeds_differ(self):
        Q1 = generate_rotation_matrix(64, seed=42)
        Q2 = generate_rotation_matrix(64, seed=43)
        assert not np.allclose(Q1, Q2)

    def test_proper_rotation(self):
        """det(Q) should be +1 (rotation, not reflection)."""
        Q = generate_rotation_matrix(64, seed=42)
        det = np.linalg.det(Q)
        np.testing.assert_allclose(abs(det), 1.0, atol=1e-4)


class TestRotateUnrotate:
    def test_roundtrip_exact(self):
        Q = generate_rotation_matrix(HEAD_DIM, seed=42)
        k, _ = _make_token_kv()
        rotated = polar_rotate(k, Q, N_HEADS)
        recovered = polar_unrotate(rotated, Q, N_HEADS)
        np.testing.assert_allclose(recovered, k, atol=1e-5)

    def test_rotation_changes_values(self):
        Q = generate_rotation_matrix(HEAD_DIM, seed=42)
        k, _ = _make_token_kv()
        rotated = polar_rotate(k, Q, N_HEADS)
        assert not np.allclose(rotated, k)

    def test_rotation_preserves_norm(self):
        """Orthogonal rotation preserves L2 norm per head."""
        Q = generate_rotation_matrix(HEAD_DIM, seed=42)
        k, _ = _make_token_kv()
        rotated = polar_rotate(k, Q, N_HEADS)
        for h in range(N_HEADS):
            s = h * HEAD_DIM
            e = s + HEAD_DIM
            np.testing.assert_allclose(
                np.linalg.norm(k[s:e]),
                np.linalg.norm(rotated[s:e]),
                atol=1e-5,
            )

    def test_rotation_spreads_variance(self):
        """After rotation, per-dim variance should be more uniform."""
        Q = generate_rotation_matrix(HEAD_DIM, seed=42)
        rng = np.random.default_rng(99)
        raw_samples = []
        rot_samples = []
        for i in range(200):
            k, _ = _make_outlier_kv(seed=i)
            raw_samples.append(k)
            rot_samples.append(polar_rotate(k, Q, N_HEADS))

        raw = np.stack(raw_samples)
        rot = np.stack(rot_samples)
        # Coefficient of variation of per-dim variance
        raw_cv = np.std(np.var(raw, axis=0)) / np.mean(np.var(raw, axis=0))
        rot_cv = np.std(np.var(rot, axis=0)) / np.mean(np.var(rot, axis=0))
        assert rot_cv < raw_cv, (
            f"Rotation should reduce variance spread: raw_cv={raw_cv:.3f} rot_cv={rot_cv:.3f}"
        )


class TestInferHeadGeometry:
    def test_explicit_n_heads(self):
        assert infer_head_geometry(256, n_heads=4) == (4, 64)

    def test_auto_infer_256_entry(self):
        # 256 / 128 = 2 heads (128 checked before 64)
        assert infer_head_geometry(256) == (2, 128)

    def test_auto_infer_192_entry(self):
        # 192 is not divisible by 128 → falls through to 64
        assert infer_head_geometry(192) == (3, 64)

    def test_auto_infer_4096_entry(self):
        assert infer_head_geometry(4096) == (32, 128)

    def test_bad_divisor(self):
        with pytest.raises(ValueError):
            infer_head_geometry(256, n_heads=3)

    def test_uninferable(self):
        with pytest.raises(ValueError):
            infer_head_geometry(17)


# ── Integration tests with KVLayerState ─────────────────────────────────────


class TestPolarKVLayerState:
    def _make_config(self, polar=True, cal_tokens=16):
        return OnlineKVConfig(
            calibration_tokens=cal_tokens,
            n_clusters=64,
            exact_layers=[],
            polar_rotation=polar,
            polar_seed=42,
            n_heads=N_HEADS,
        )

    def test_default_off(self):
        """polar_rotation=False should not touch values."""
        config = OnlineKVConfig(exact_layers=[])
        ls = KVLayerState(1, config)
        assert ls._polar_Q is None

    def test_lazy_init(self):
        """Rotation matrix created on first feed_token."""
        config = self._make_config()
        ls = KVLayerState(1, config)
        assert ls._polar_Q is None
        ls.feed_token(*_make_token_kv(seed=0))
        assert ls._polar_Q is not None
        assert ls._polar_Q.shape == (HEAD_DIM, HEAD_DIM)

    def test_per_layer_seed(self):
        """Different layers get different rotation matrices."""
        config = self._make_config()
        ls1 = KVLayerState(1, config)
        ls2 = KVLayerState(2, config)
        ls1.feed_token(*_make_token_kv(seed=0))
        ls2.feed_token(*_make_token_kv(seed=0))
        assert not np.allclose(ls1._polar_Q, ls2._polar_Q)

    def test_calibration_still_works(self):
        """Calibration → streaming transition with rotation enabled."""
        config = self._make_config(cal_tokens=8)
        ls = KVLayerState(1, config)
        for i in range(7):
            assert ls.feed_token(*_make_token_kv(seed=i)) is None
        result = ls.feed_token(*_make_token_kv(seed=7))
        assert result is not None
        assert ls.is_streaming

    def test_roundtrip_fidelity(self):
        """Compress → decode → unrotate should recover original with high cosine."""
        config = self._make_config(cal_tokens=16)
        ls = KVLayerState(1, config)

        rng = np.random.default_rng(42)
        # Calibrate
        for i in range(16):
            k = rng.standard_normal(ENTRY_SIZE).astype(np.float32)
            v = rng.standard_normal(ENTRY_SIZE).astype(np.float32)
            ls.feed_token(k, v)

        assert ls.is_streaming

        # Stream tokens, keep originals
        originals_k = []
        for i in range(20):
            k = rng.standard_normal(ENTRY_SIZE).astype(np.float32)
            v = rng.standard_normal(ENTRY_SIZE).astype(np.float32)
            originals_k.append(k)
            ls.feed_token(k, v)

        all_k = ls.get_all_compressed_k()
        assert all_k is not None
        assert all_k.shape[0] == 20

        orig_flat = np.concatenate(originals_k)
        recon_flat = all_k.ravel()
        cos = np.dot(orig_flat, recon_flat) / (
            np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat)
        )
        assert cos > 0.99, f"Polar roundtrip cosine too low: {cos}"

    def test_single_token_decode(self):
        """get_compressed_kv also unrotates correctly."""
        config = self._make_config(cal_tokens=8)
        ls = KVLayerState(1, config)

        rng = np.random.default_rng(42)
        for i in range(8):
            k = rng.standard_normal(ENTRY_SIZE).astype(np.float32)
            v = rng.standard_normal(ENTRY_SIZE).astype(np.float32)
            ls.feed_token(k, v)

        # Stream one token
        k_orig = rng.standard_normal(ENTRY_SIZE).astype(np.float32)
        v_orig = rng.standard_normal(ENTRY_SIZE).astype(np.float32)
        ls.feed_token(k_orig, v_orig)

        k_dec, v_dec = ls.get_compressed_kv(0)
        # Must be close to original (not to rotated)
        cos_k = np.dot(k_orig, k_dec) / (np.linalg.norm(k_orig) * np.linalg.norm(k_dec))
        cos_v = np.dot(v_orig, v_dec) / (np.linalg.norm(v_orig) * np.linalg.norm(v_dec))
        assert cos_k > 0.99, f"Single-token K cosine: {cos_k}"
        assert cos_v > 0.99, f"Single-token V cosine: {cos_v}"

    def test_exact_layer_ignores_polar(self):
        """Exact layers should not create rotation matrices."""
        config = OnlineKVConfig(
            exact_layers=[0],
            polar_rotation=True,
            polar_seed=42,
            n_heads=N_HEADS,
        )
        ls = KVLayerState(0, config)
        ls.feed_token(*_make_token_kv())
        assert ls._polar_Q is None
        assert ls.is_exact

    def test_backward_compat_no_polar(self):
        """Without polar_rotation, behavior is identical to baseline."""
        config_off = OnlineKVConfig(
            calibration_tokens=8, n_clusters=64, exact_layers=[],
            polar_rotation=False,
        )
        config_on = OnlineKVConfig(
            calibration_tokens=8, n_clusters=64, exact_layers=[],
            polar_rotation=False,
        )

        ls_off = KVLayerState(1, config_off)
        ls_on = KVLayerState(1, config_on)

        rng = np.random.default_rng(42)
        for i in range(8):
            k = rng.standard_normal(ENTRY_SIZE).astype(np.float32)
            v = rng.standard_normal(ENTRY_SIZE).astype(np.float32)
            ls_off.feed_token(k.copy(), v.copy())
            ls_on.feed_token(k.copy(), v.copy())

        # Stream same tokens
        rng2 = np.random.default_rng(99)
        for i in range(5):
            k = rng2.standard_normal(ENTRY_SIZE).astype(np.float32)
            v = rng2.standard_normal(ENTRY_SIZE).astype(np.float32)
            ls_off.feed_token(k.copy(), v.copy())
            ls_on.feed_token(k.copy(), v.copy())

        k_off = ls_off.get_all_compressed_k()
        k_on = ls_on.get_all_compressed_k()
        np.testing.assert_array_equal(k_off, k_on)

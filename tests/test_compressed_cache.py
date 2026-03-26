"""Tests for KVLayerState (numpy-only, no torch dependency)."""

import numpy as np
import pytest

from helix_online_kv.config import OnlineKVConfig
from helix_online_kv.layer_state import KVLayerState, LayerPhase


def _make_token_kv(n_heads=4, head_dim=64, seed=42):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal(n_heads * head_dim).astype(np.float32)
    v = rng.standard_normal(n_heads * head_dim).astype(np.float32)
    return k, v


class TestKVLayerState:
    def test_exact_layer_stays_exact(self):
        config = OnlineKVConfig(exact_layers=[0])
        ls = KVLayerState(0, config)
        assert ls.is_exact
        assert ls.phase == LayerPhase.EXACT
        # Feed should be no-op
        result = ls.feed_token(*_make_token_kv())
        assert result is None

    def test_calibration_to_streaming_transition(self):
        config = OnlineKVConfig(calibration_tokens=8, exact_layers=[])
        ls = KVLayerState(1, config)
        assert ls.is_calibrating

        # Feed 7 tokens — still calibrating
        for i in range(7):
            result = ls.feed_token(*_make_token_kv(seed=i))
            assert result is None
            assert ls.is_calibrating

        # 8th token triggers finalization
        result = ls.feed_token(*_make_token_kv(seed=7))
        assert result is not None
        assert ls.is_streaming
        assert "k_calibration" in result
        assert "v_calibration" in result

    def test_streaming_produces_indices(self):
        config = OnlineKVConfig(calibration_tokens=4, exact_layers=[])
        ls = KVLayerState(1, config)

        # Calibrate
        for i in range(4):
            ls.feed_token(*_make_token_kv(seed=i))

        assert ls.is_streaming

        # Stream 10 more tokens
        for i in range(10):
            ls.feed_token(*_make_token_kv(seed=100 + i))

        assert ls.compressed_token_count == 10

    def test_compressed_decode_cosine(self):
        config = OnlineKVConfig(calibration_tokens=16, n_clusters=64, exact_layers=[])
        ls = KVLayerState(1, config)

        rng = np.random.default_rng(42)
        # Calibrate with consistent distribution
        for i in range(16):
            k = rng.standard_normal(256).astype(np.float32)
            v = rng.standard_normal(256).astype(np.float32)
            ls.feed_token(k, v)

        # Stream and verify decode quality
        originals_k, originals_v = [], []
        for i in range(20):
            k = rng.standard_normal(256).astype(np.float32)
            v = rng.standard_normal(256).astype(np.float32)
            originals_k.append(k)
            originals_v.append(v)
            ls.feed_token(k, v)

        all_k = ls.get_all_compressed_k()
        all_v = ls.get_all_compressed_v()
        assert all_k is not None
        assert all_v is not None
        assert all_k.shape[0] == 20

        # Check cosine on concatenated
        orig_k_flat = np.concatenate(originals_k)
        recon_k_flat = all_k.ravel()
        cos = np.dot(orig_k_flat, recon_k_flat) / (
            np.linalg.norm(orig_k_flat) * np.linalg.norm(recon_k_flat)
        )
        assert cos > 0.99, f"Compressed K cosine too low: {cos}"

    def test_memory_report(self):
        config = OnlineKVConfig(calibration_tokens=4, exact_layers=[])
        ls = KVLayerState(1, config)

        for i in range(4):
            ls.feed_token(*_make_token_kv(seed=i))

        for i in range(10):
            ls.feed_token(*_make_token_kv(seed=100 + i))

        mem = ls.memory_bytes()
        assert mem["codebook_bytes"] > 0
        assert mem["index_bytes"] > 0
        assert mem["total_bytes"] == mem["codebook_bytes"] + mem["index_bytes"]

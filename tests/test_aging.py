"""Tests for AgingPolicy."""

from helix_online_kv.aging_policy import AgingPolicy


class TestAgingPolicy:
    def test_default_hot_window(self):
        ap = AgingPolicy(hot_window=256)
        assert not ap.should_compress(0)
        assert not ap.should_compress(255)
        assert ap.should_compress(256)
        assert ap.should_compress(1000)

    def test_custom_min_age(self):
        ap = AgingPolicy(hot_window=256, min_age_tokens=128)
        assert not ap.should_compress(127)
        assert ap.should_compress(128)

    def test_tier_boundary_small_seq(self):
        ap = AgingPolicy(hot_window=256)
        # Fewer tokens than hot window — everything is Tier 0
        assert ap.tier_boundary(100) == 0

    def test_tier_boundary_exact(self):
        ap = AgingPolicy(hot_window=256)
        assert ap.tier_boundary(256) == 0
        assert ap.tier_boundary(257) == 1
        assert ap.tier_boundary(512) == 256

    def test_tier_boundary_large(self):
        ap = AgingPolicy(hot_window=128)
        assert ap.tier_boundary(1024) == 896

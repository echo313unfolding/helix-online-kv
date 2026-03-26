"""Aging policy: Tier 0 (exact FP16) -> Tier 1 (compressed uint8)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgingPolicy:
    """Controls when tokens age from Tier 0 (exact) to Tier 1 (compressed).

    Tier 0: Recent tokens kept in full precision (FP16/FP32).
    Tier 1: Older tokens stored as uint8 codebook indices.

    The hot_window defines how many recent tokens stay in Tier 0.
    Tokens older than hot_window are demoted to Tier 1 (compressed).

    Attributes:
        hot_window: Number of most-recent tokens kept exact.
        min_age_tokens: Minimum token age before eligible for compression.
            Defaults to hot_window.
    """
    hot_window: int = 256
    min_age_tokens: int | None = None

    @property
    def effective_min_age(self) -> int:
        return self.min_age_tokens if self.min_age_tokens is not None else self.hot_window

    def should_compress(self, token_age: int) -> bool:
        """Whether a token at the given age should be compressed.

        Args:
            token_age: How many tokens ago this was generated
                (0 = current, 1 = previous, etc.)

        Returns:
            True if the token should be in Tier 1 (compressed).
        """
        return token_age >= self.effective_min_age

    def tier_boundary(self, total_tokens: int) -> int:
        """Index of the first Tier 0 token in the sequence.

        Tokens [0, boundary) are Tier 1 (compressed).
        Tokens [boundary, total_tokens) are Tier 0 (exact).

        Args:
            total_tokens: Total number of tokens in the KV cache.

        Returns:
            Index boundary.
        """
        if total_tokens <= self.effective_min_age:
            return 0  # Everything is hot (Tier 0)
        return total_tokens - self.effective_min_age

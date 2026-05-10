"""PolarQuant rotation: random orthogonal rotation before scalar VQ.

Spreads outlier-dimension energy uniformly across all dimensions, improving
centroid utilization and reducing quantization MSE.  Rotation matrix is
deterministic per seed — no storage overhead.

Reference: TurboQuant (ICLR 2026), PolarQuant stage.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _is_torch_tensor(x) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def generate_rotation_matrix(dim: int, seed: int = 42) -> np.ndarray:
    """Random orthogonal matrix via QR decomposition.  Deterministic per seed.

    Returns:
        Q: [dim, dim] orthogonal float32 matrix (det = +1).
    """
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim).astype(np.float32)
    Q, R = np.linalg.qr(H)
    # Fix sign so Q is a proper rotation (det=+1), not a reflection
    Q = Q * np.sign(np.diag(R))
    return Q


def polar_rotate(values, Q: np.ndarray, n_heads: int):
    """Rotate KV values per-head before quantization.

    Args:
        values: Flat [n_heads * head_dim] array (numpy or torch tensor).
        Q: [head_dim, head_dim] orthogonal matrix (numpy).
        n_heads: Number of attention heads.

    Returns:
        Rotated flat array, same type and device as input.
    """
    head_dim = Q.shape[0]
    if _is_torch_tensor(values):
        Q_t = torch.from_numpy(Q).to(device=values.device, dtype=values.dtype)
        reshaped = values.reshape(n_heads, head_dim)
        return (reshaped @ Q_t.T).ravel()
    reshaped = values.reshape(n_heads, head_dim)
    rotated = reshaped @ Q.T
    return rotated.ravel()


def polar_unrotate(values, Q: np.ndarray, n_heads: int):
    """Inverse rotation on decompressed values.  Q is orthogonal: Q^{-1} = Q^T.

    Args:
        values: Flat [n_heads * head_dim] array (numpy or torch tensor, rotated domain).
        Q: [head_dim, head_dim] orthogonal matrix (numpy).
        n_heads: Number of attention heads.

    Returns:
        Un-rotated flat array, same type and device as input.
    """
    head_dim = Q.shape[0]
    if _is_torch_tensor(values):
        Q_t = torch.from_numpy(Q).to(device=values.device, dtype=values.dtype)
        reshaped = values.reshape(n_heads, head_dim)
        return (reshaped @ Q_t).ravel()
    reshaped = values.reshape(n_heads, head_dim)
    unrotated = reshaped @ Q
    return unrotated.ravel()


def infer_head_geometry(entry_size: int, n_heads: int = 0) -> tuple[int, int]:
    """Infer (n_heads, head_dim) from flat entry_size.

    If n_heads is provided and > 0, computes head_dim = entry_size / n_heads.
    Otherwise tries common head_dim values (128, 64, 32) and picks the first
    that divides entry_size evenly.

    Returns:
        (n_heads, head_dim)

    Raises:
        ValueError: If geometry cannot be inferred.
    """
    if n_heads > 0:
        if entry_size % n_heads != 0:
            raise ValueError(
                f"entry_size={entry_size} not divisible by n_heads={n_heads}"
            )
        return n_heads, entry_size // n_heads

    for hd in (128, 64, 32):
        if entry_size % hd == 0:
            return entry_size // hd, hd

    raise ValueError(
        f"Cannot infer head geometry from entry_size={entry_size}. "
        f"Set n_heads explicitly in OnlineKVConfig."
    )

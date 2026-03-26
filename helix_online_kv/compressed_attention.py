"""Compressed-domain attention: compute attention without full K/V reconstruction.

Paths:

Path A (Scalar VQ): Existing OnlineCodebook. Decompress K from scalar codebook,
    then standard Q@K.T. Win = memory (4x smaller index buffer), no compute reduction.

Path B (Vector VQ): New VectorCodebook. Precompute q @ codebook.T (k scores),
    then gather via indices. Win = compute (O(k) vs O(seq_len*head_dim)).

Path D (Product Quantization): ProductCodebook splits head_dim into M subspaces.
    Precompute distance tables per subspace, gather+sum. Higher fidelity than B/C.

Path F (Prefiltered Attention): Coarse VQ clusters select top-p clusters, then
    full-precision attention on selected tokens only. Sparse attention.

Path G (Hybrid PQ + Prefilter): CDC-03. PQ scores rank all tokens cheaply,
    select top-k by approximate score, exact q@K[selected].T + dense V on subset.
    Combines PQ compute reduction with dense V precision.

All functions operate on single-head, single-query vectors for clarity.
Batching/multi-head is the caller's responsibility.
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax as _scipy_softmax


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    return _scipy_softmax(x)


# -- Reference --

def standard_attention(
    q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reference single-query attention.

    Args:
        q: [head_dim] query vector.
        K: [seq_len, head_dim] key matrix.
        V: [seq_len, head_dim] value matrix.
        scale: typically 1/sqrt(head_dim).

    Returns:
        (scores [seq_len], output [head_dim])
    """
    scores = (q @ K.T) * scale                     # [seq_len]
    weights = _softmax(scores)                      # [seq_len]
    output = weights @ V                            # [head_dim]
    return scores, output


# -- Path A: Scalar VQ --

def scalar_vq_attention(
    q: np.ndarray,
    k_indices: np.ndarray,
    k_codebook: np.ndarray,
    V: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Path A: decompress K from scalar codebook, then standard attention.

    Args:
        q: [head_dim] query vector.
        k_indices: [seq_len, head_dim] uint8 indices into scalar codebook.
        k_codebook: [256] float32 scalar centroids.
        V: [seq_len, head_dim] value matrix (uncompressed for now).
        scale: typically 1/sqrt(head_dim).

    Returns:
        (scores [seq_len], output [head_dim])
    """
    K_recon = k_codebook[k_indices]                 # [seq_len, head_dim]
    return standard_attention(q, K_recon, V, scale)


# -- Path B: Vector VQ --

def vector_vq_attention_scores(
    q: np.ndarray,
    k_indices: np.ndarray,
    k_vector_codebook: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Path B K-side: precompute q @ codebook.T, then gather.

    This is the compute reduction path. Instead of q @ K.T (seq_len * head_dim ops),
    we do q @ codebook.T (k * head_dim ops) + index gather (seq_len ops).

    Args:
        q: [head_dim] query vector.
        k_indices: [seq_len] uint8 indices into vector codebook.
        k_vector_codebook: [k, head_dim] vector centroids.
        scale: typically 1/sqrt(head_dim).

    Returns:
        scores [seq_len] (pre-softmax attention scores).
    """
    precomputed = q @ k_vector_codebook.T           # [k]
    scores = precomputed[k_indices] * scale         # [seq_len]
    return scores


def vector_vq_value_output(
    attn_weights: np.ndarray,
    v_indices: np.ndarray,
    v_vector_codebook: np.ndarray,
) -> np.ndarray:
    """Path B V-side: bin attention weights by V-cluster, weighted codebook sum.

    Instead of weights @ V (seq_len * head_dim), we:
    1. Accumulate weights into k bins via scatter-add: bin[idx] += weight
    2. Output = bin_weights @ codebook (k * head_dim ops)

    Args:
        attn_weights: [seq_len] softmax attention weights.
        v_indices: [seq_len] uint8 indices into V vector codebook.
        v_vector_codebook: [k, head_dim] V centroids.

    Returns:
        output [head_dim].
    """
    k = len(v_vector_codebook)
    bin_weights = np.zeros(k, dtype=np.float64)     # accumulate in float64
    np.add.at(bin_weights, v_indices, attn_weights)
    output = bin_weights @ v_vector_codebook         # [head_dim]
    return output.astype(np.float32)


def full_vector_vq_attention(
    q: np.ndarray,
    k_indices: np.ndarray,
    k_codebook: np.ndarray,
    v_indices: np.ndarray,
    v_codebook: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Path B full: K-side scores -> softmax -> V-side output.

    Args:
        q: [head_dim] query vector.
        k_indices: [seq_len] uint8 K indices.
        k_codebook: [k_k, head_dim] K centroids.
        v_indices: [seq_len] uint8 V indices.
        v_codebook: [k_v, head_dim] V centroids.
        scale: typically 1/sqrt(head_dim).

    Returns:
        (scores [seq_len], output [head_dim])
    """
    scores = vector_vq_attention_scores(q, k_indices, k_codebook, scale)
    weights = _softmax(scores)
    output = vector_vq_value_output(weights, v_indices, v_codebook)
    return scores, output


# -- Path D: Product Quantization --

def pq_attention_scores(
    q: np.ndarray,
    k_pq_indices: np.ndarray,
    k_product_codebook,
    scale: float,
) -> np.ndarray:
    """Path D K-side: precompute distance tables, gather+sum per subspace.

    Args:
        q: [head_dim] query vector.
        k_pq_indices: [seq_len, M] uint8 PQ indices.
        k_product_codebook: ProductCodebook instance.
        scale: typically 1/sqrt(head_dim).

    Returns:
        scores [seq_len] (pre-softmax attention scores).
    """
    tables = k_product_codebook.precompute_distance_tables(q)
    scores = k_product_codebook.gather_pq_scores(tables, k_pq_indices) * np.float32(scale)
    return scores


def pq_value_output(
    attn_weights: np.ndarray,
    v_pq_indices: np.ndarray,
    v_product_codebook,
) -> np.ndarray:
    """Path D V-side: per-subspace scatter-add + weighted codebook sum, concat.

    For each subspace s:
        1. Bin attention weights by v_pq_indices[:, s]
        2. Weighted sum of sub_codebook_s centroids -> [sub_dim]
    Concatenate all subspaces -> [head_dim].

    Args:
        attn_weights: [seq_len] softmax attention weights.
        v_pq_indices: [seq_len, M] uint8 PQ indices.
        v_product_codebook: ProductCodebook instance.

    Returns:
        output [head_dim].
    """
    M = v_product_codebook.n_subspaces
    sub_dim = v_product_codebook.sub_dim
    head_dim = v_product_codebook.head_dim

    output = np.zeros(head_dim, dtype=np.float32)

    for s in range(M):
        cb_s = v_product_codebook._codebooks[s]
        k_s = len(cb_s.centroids)
        centroids_s = cb_s.centroids  # [k_s, sub_dim]

        # Accumulate weights into bins (float64 for precision)
        bin_weights = np.zeros(k_s, dtype=np.float64)
        np.add.at(bin_weights, v_pq_indices[:, s], attn_weights)

        # Weighted sum of centroids
        start = s * sub_dim
        end = start + sub_dim
        output[start:end] = (bin_weights @ centroids_s).astype(np.float32)

    return output


def full_pq_attention(
    q: np.ndarray,
    k_pq_indices: np.ndarray,
    k_pq_cb,
    v_pq_indices: np.ndarray,
    v_pq_cb,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Path D full: PQ scores -> softmax -> PQ V-side output.

    Args:
        q: [head_dim] query vector.
        k_pq_indices: [seq_len, M] uint8 K PQ indices.
        k_pq_cb: ProductCodebook for K.
        v_pq_indices: [seq_len, M] uint8 V PQ indices.
        v_pq_cb: ProductCodebook for V.
        scale: typically 1/sqrt(head_dim).

    Returns:
        (scores [seq_len], output [head_dim])
    """
    scores = pq_attention_scores(q, k_pq_indices, k_pq_cb, scale)
    weights = _softmax(scores)
    output = pq_value_output(weights, v_pq_indices, v_pq_cb)
    return scores, output


# -- Path F: Prefiltered Attention --

def prefiltered_attention(
    q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    k_indices: np.ndarray,
    k_codebook: np.ndarray,
    scale: float,
    top_p: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Coarse VQ filter on centroids, full-precision on selected tokens.

    1. q @ codebook.T -> [n_clusters] coarse scores
    2. Select top_p clusters by score
    3. mask = tokens in selected clusters
    4. Full-precision: q @ K[mask].T for selected, -1e9 for unselected
    5. softmax -> weights @ V -> output

    Args:
        q: [head_dim] query vector.
        K: [seq_len, head_dim] key matrix (dense).
        V: [seq_len, head_dim] value matrix (dense).
        k_indices: [seq_len] uint8 cluster indices for K.
        k_codebook: [n_clusters, head_dim] vector centroids.
        scale: typically 1/sqrt(head_dim).
        top_p: number of clusters to select.

    Returns:
        (scores [seq_len], output [head_dim])
        Unselected positions have score = -1e9.
    """
    n_clusters = len(k_codebook)
    seq_len = K.shape[0]

    # Coarse: q @ codebook.T
    coarse_scores = q @ k_codebook.T  # [n_clusters]

    # Select top_p clusters (cap at n_clusters)
    p = min(top_p, n_clusters)
    top_cluster_ids = np.argsort(coarse_scores)[-p:]  # top-p by score

    # Build mask: which tokens belong to selected clusters
    mask = np.isin(k_indices, top_cluster_ids)

    # Full-precision attention on selected tokens, -1e9 on rest
    scores = np.full(seq_len, -1e9, dtype=np.float32)
    if np.any(mask):
        scores[mask] = (q @ K[mask].T) * scale

    weights = _softmax(scores)
    output = (weights @ V).astype(np.float32)

    return scores, output


# -- Path G: Hybrid PQ + Prefilter (CDC-03) --

def hybrid_pq_attention(
    q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    k_pq_indices: np.ndarray,
    k_pq_cb,
    scale: float,
    top_k: int = 256,
    sink_tokens: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """CDC-03 Hybrid: PQ scores rank tokens, dense K+V on selected subset.

    The key insight: compress the scoring (PQ), keep values exact (dense V).
    PQ gives cheap approximate scores for ALL tokens, then we select the
    top-k tokens by approximate score and do full-precision attention on
    just that subset.

    Attention sink guard: the first `sink_tokens` positions are ALWAYS
    included, regardless of PQ score. Position 0 acts as an attention sink
    in transformer models — it receives disproportionate weight but may
    score low on content-based dot products. Missing it collapses output.

    Pipeline:
        1. PQ distance tables → approximate scores for all tokens (cheap)
        2. Force-include first sink_tokens positions
        3. Select remaining top_k tokens by approximate score
        4. Full-precision q @ K[selected].T for exact scores on selected
        5. Softmax over selected tokens (unselected → -inf)
        6. weights @ V[selected] for output

    Args:
        q: [head_dim] query vector.
        K: [seq_len, head_dim] key matrix (dense or decompressed from scalar VQ).
        V: [seq_len, head_dim] value matrix (dense).
        k_pq_indices: [seq_len, M] uint8 PQ indices for K.
        k_pq_cb: ProductCodebook for K scoring.
        scale: typically 1/sqrt(head_dim).
        top_k: number of tokens to select by approximate score.
        sink_tokens: number of initial positions always included (attention sink).

    Returns:
        (scores [seq_len], output [head_dim], meta dict)
        Unselected positions have score = -inf.
        Meta contains: selected_count, coverage, pq_score_range, sink_count.
    """
    seq_len = K.shape[0]

    # Step 1: PQ approximate scores (cheap — table lookup + sum)
    tables = k_pq_cb.precompute_distance_tables(q)
    approx_scores = k_pq_cb.gather_pq_scores(tables, k_pq_indices) * np.float32(scale)

    # Step 2: Build selected set — sink tokens + top-k by PQ score
    k = min(top_k, seq_len)
    actual_sink = min(sink_tokens, seq_len)

    if k >= seq_len:
        selected_idx = np.arange(seq_len)
    else:
        # Always include sink tokens
        sink_set = set(range(actual_sink))

        # Top-k from non-sink tokens
        remaining_budget = k - actual_sink
        if remaining_budget > 0:
            # Mask out sink tokens by setting their approx scores to -inf
            # so they don't consume budget
            masked_scores = approx_scores.copy()
            masked_scores[:actual_sink] = -np.inf
            top_idx = np.argpartition(masked_scores, -remaining_budget)[-remaining_budget:]
            selected_set = sink_set | set(top_idx.tolist())
        else:
            selected_set = sink_set

        selected_idx = np.array(sorted(selected_set), dtype=np.intp)

    # Step 3: Full-precision exact scores on selected tokens only
    K_selected = K[selected_idx]
    exact_scores_selected = (q @ K_selected.T) * scale  # [n_selected]

    # Step 4: Softmax over selected tokens only (truly sparse)
    selected_weights = _softmax(exact_scores_selected)  # [n_selected]

    # Step 5: Sparse V matmul — only touch V[selected]
    V_selected = V[selected_idx]  # [n_selected, head_dim]
    output = (selected_weights @ V_selected).astype(np.float32)  # [head_dim]

    # Build full score vector for diagnostics (unselected = -inf)
    scores = np.full(seq_len, -np.inf, dtype=np.float32)
    scores[selected_idx] = exact_scores_selected

    meta = {
        "selected_count": len(selected_idx),
        "coverage": float(len(selected_idx)) / seq_len,
        "pq_score_range": float(np.max(approx_scores) - np.min(approx_scores)),
        "sink_count": actual_sink,
    }

    return scores, output, meta

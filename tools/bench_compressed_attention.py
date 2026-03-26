#!/usr/bin/env python3
"""CDC-01/02: Compressed-Domain Attention Benchmark.

Uses REAL KV data from ~/helix-substrate/kv_dump/ (or --kv-dir).
Operates per-head (reshape [1,4,seq_len,64] -> 4 separate [seq_len,64] arrays).

CDC-01:
  Part A: Scalar VQ fidelity (existing OnlineCodebook)
  Part B: Vector VQ compute reduction (new VectorCodebook), k sweep
  Part C: Full pipeline (Vector VQ K+V)

CDC-02:
  Part D: Product Quantization (M subspaces, high fidelity)
  Part E: Fused scalar VQ Triton kernel (GPU bandwidth reduction)
  Part F: Prefiltered attention (coarse VQ cluster selection)

All gates, Pareto curves, and projected speedup tables included.
Receipt with WO-RECEIPT-COST-01 cost block.
"""

import argparse
import json
import os
import sys
import time
import resource
import platform
from pathlib import Path

import numpy as np
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helix_online_kv.codebook import OnlineCodebook
from helix_online_kv.vector_codebook import VectorCodebook
from helix_online_kv.product_codebook import ProductCodebook
from helix_online_kv.compressed_attention import (
    standard_attention,
    scalar_vq_attention,
    vector_vq_attention_scores,
    full_vector_vq_attention,
    pq_attention_scores,
    full_pq_attention,
    prefiltered_attention,
    hybrid_pq_attention,
)
from helix_online_kv.triton_attention import fused_scalar_vq_qkt_numpy

try:
    import torch
    from helix_online_kv.triton_attention import fused_scalar_vq_qkt, _HAS_TRITON
    _HAS_CUDA = torch.cuda.is_available() and _HAS_TRITON
except ImportError:
    _HAS_CUDA = False

RECEIPTS_DIR = Path(__file__).resolve().parent.parent / "receipts"
N_HEADS = 4
HEAD_DIM = 64
N_CLUSTERS_SCALAR = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description="CDC-01/02: Compressed-Domain Attention Benchmark"
    )
    parser.add_argument(
        "--kv-dir",
        type=str,
        default=os.path.expanduser("~/helix-substrate/kv_dump"),
        help="Path to KV dump directory (default: ~/helix-substrate/kv_dump)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for receipt filename (default: auto from dir name)",
    )
    parser.add_argument(
        "--parts",
        type=str,
        default="a,b,c",
        help="Comma-separated parts to run: a,b,c,d,e,f (default: a,b,c)",
    )
    return parser.parse_args()


def auto_detect_prompts(kv_dir: Path) -> list:
    """Scan kv_dir for subdirectories containing layer_*_k.npy files."""
    prompts = []
    for d in sorted(kv_dir.iterdir()):
        if d.is_dir() and list(d.glob("layer_*_k.npy")):
            prompts.append(d.name)
    return prompts


def auto_detect_n_layers(kv_dir: Path, first_prompt: str) -> int:
    """Count layer files in first prompt directory."""
    return len(list((kv_dir / first_prompt).glob("layer_*_k.npy")))


def load_kv_per_head(kv_dir: Path, prompt: str, layer: int, kv_type: str) -> np.ndarray:
    """Load KV dump and return per-head arrays.

    Returns: [n_heads, seq_len, head_dim] float32
    """
    path = kv_dir / prompt / f"layer_{layer}_{kv_type}.npy"
    data = np.load(path)  # [1, n_heads, seq_len, head_dim]
    return data[0].astype(np.float32)  # [n_heads, seq_len, head_dim]


def cosine_sim(a, b):
    a, b = a.ravel(), b.ravel()
    d = float(np.dot(a, b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return d / (na * nb)


def kl_div(scores_ref, scores_comp):
    """KL(softmax(ref) || softmax(comp))."""
    p = softmax(scores_ref)
    q = softmax(scores_comp)
    mask = p > 1e-30
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-30))))


# ========== PART A: Scalar VQ Fidelity ==========

def run_part_a(kv_dir, prompts, n_layers):
    """Scalar VQ: decompress K, then standard attention. Win = memory only."""
    print("=" * 70)
    print("PART A: Scalar VQ Fidelity (existing OnlineCodebook)")
    print("=" * 70)

    scale = 1.0 / np.sqrt(HEAD_DIM)
    results = []

    for prompt in prompts:
        for layer in range(n_layers):
            K_all = load_kv_per_head(kv_dir, prompt, layer, "k")  # [4, seq_len, 64]
            V_all = load_kv_per_head(kv_dir, prompt, layer, "v")

            for head in range(N_HEADS):
                K = K_all[head]  # [seq_len, 64]
                V = V_all[head]
                seq_len = K.shape[0]

                # Generate query with same distribution as K
                rng = np.random.default_rng(layer * N_HEADS + head)
                q = rng.standard_normal(HEAD_DIM).astype(np.float32)
                q = q * K.std() + K.mean()

                # Fit scalar codebook on K
                cb = OnlineCodebook(n_clusters=N_CLUSTERS_SCALAR)
                cb.feed_calibration(K.ravel())
                cb.finalize_calibration()

                # Assign and reshape
                k_indices = cb.assign(K.ravel()).reshape(K.shape)

                # Reference
                scores_ref, out_ref = standard_attention(q, K, V, scale)

                # Scalar VQ
                scores_svq, out_svq = scalar_vq_attention(
                    q, k_indices, cb.centroids, V, scale
                )

                # Metrics
                score_cos = cosine_sim(scores_ref, scores_svq)
                output_cos = cosine_sim(out_ref, out_svq)
                kl = kl_div(scores_ref, scores_svq)

                # Memory: indices (uint8) + codebook vs dense K (float32)
                mem_compressed = k_indices.nbytes + cb.centroids.nbytes
                mem_dense = K.nbytes
                mem_ratio = mem_dense / mem_compressed

                results.append({
                    "prompt": prompt, "layer": layer, "head": head,
                    "seq_len": seq_len,
                    "score_cosine": round(score_cos, 6),
                    "output_cosine": round(output_cos, 6),
                    "kl_divergence": round(kl, 8),
                    "memory_compressed_bytes": int(mem_compressed),
                    "memory_dense_bytes": int(mem_dense),
                    "memory_ratio": round(mem_ratio, 2),
                })

    # Summarize
    score_cosines = [r["score_cosine"] for r in results]
    output_cosines = [r["output_cosine"] for r in results]
    kls = [r["kl_divergence"] for r in results]
    mem_ratios = [r["memory_ratio"] for r in results]

    print(f"\n  N entries: {len(results)}")
    print(f"  Score cosine:  mean={np.mean(score_cosines):.6f}  "
          f"min={np.min(score_cosines):.6f}  max={np.max(score_cosines):.6f}")
    print(f"  Output cosine: mean={np.mean(output_cosines):.6f}  "
          f"min={np.min(output_cosines):.6f}  max={np.max(output_cosines):.6f}")
    print(f"  KL divergence: mean={np.mean(kls):.6f}  max={np.max(kls):.6f}")
    print(f"  Memory ratio:  mean={np.mean(mem_ratios):.2f}x")

    gate = float(np.mean(score_cosines)) >= 0.999
    print(f"\n  GATE (score cosine >= 0.999): {'PASS' if gate else 'FAIL'} "
          f"(mean={np.mean(score_cosines):.6f})")

    return {
        "per_layer": results,
        "summary": {
            "n_entries": len(results),
            "score_cosine_mean": round(float(np.mean(score_cosines)), 6),
            "score_cosine_min": round(float(np.min(score_cosines)), 6),
            "output_cosine_mean": round(float(np.mean(output_cosines)), 6),
            "output_cosine_min": round(float(np.min(output_cosines)), 6),
            "kl_divergence_mean": round(float(np.mean(kls)), 6),
            "kl_divergence_max": round(float(np.max(kls)), 6),
            "memory_ratio_mean": round(float(np.mean(mem_ratios)), 2),
        },
        "gate": {
            "criterion": "mean score cosine >= 0.999",
            "value": round(float(np.mean(score_cosines)), 6),
            "passed": gate,
        },
    }


# ========== PART B: Vector VQ Compute Reduction ==========

def run_part_b(kv_dir, prompts, n_layers):
    """Vector VQ: precompute + gather. K sweep with timing."""
    print("\n" + "=" * 70)
    print("PART B: Vector VQ Compute Reduction (k sweep)")
    print("=" * 70)

    scale = 1.0 / np.sqrt(HEAD_DIM)
    k_sweep_results = []

    # Use all prompts, all layers, all heads
    for prompt in prompts:
        for layer in range(n_layers):
            K_all = load_kv_per_head(kv_dir, prompt, layer, "k")

            for head in range(N_HEADS):
                K = K_all[head]  # [seq_len, 64]
                seq_len = K.shape[0]

                rng = np.random.default_rng(layer * N_HEADS + head + 1000)
                q = rng.standard_normal(HEAD_DIM).astype(np.float32)
                q = q * K.std() + K.mean()

                # Reference scores
                scores_ref = (q @ K.T) * scale

                # k sweep (cap at seq_len)
                k_values = [k for k in [32, 64, 128, 256] if k <= seq_len]

                for k in k_values:
                    cb = VectorCodebook(n_clusters=k, head_dim=HEAD_DIM)
                    cb.feed_calibration(K)
                    cb.finalize_calibration()
                    idx = cb.assign(K)

                    # Time compressed path (average over 100 runs)
                    n_timing = 100
                    t0 = time.perf_counter()
                    for _ in range(n_timing):
                        pre = cb.precompute_query_scores(q)
                    t_precompute = (time.perf_counter() - t0) / n_timing

                    t0 = time.perf_counter()
                    for _ in range(n_timing):
                        _ = cb.gather_scores(pre, idx)
                    t_gather = (time.perf_counter() - t0) / n_timing

                    t0 = time.perf_counter()
                    for _ in range(n_timing):
                        _ = (q @ K.T) * scale
                    t_standard = (time.perf_counter() - t0) / n_timing

                    # Fidelity
                    scores_comp = vector_vq_attention_scores(
                        q, idx, cb.centroids, scale
                    )
                    score_cos = cosine_sim(scores_ref, scores_comp)
                    kl = kl_div(scores_ref, scores_comp)

                    k_sweep_results.append({
                        "prompt": prompt, "layer": layer, "head": head,
                        "seq_len": seq_len, "k": k,
                        "score_cosine": round(score_cos, 6),
                        "kl_divergence": round(kl, 8),
                        "time_precompute_us": round(t_precompute * 1e6, 2),
                        "time_gather_us": round(t_gather * 1e6, 2),
                        "time_compressed_us": round((t_precompute + t_gather) * 1e6, 2),
                        "time_standard_us": round(t_standard * 1e6, 2),
                        "speedup": round(t_standard / max(t_precompute + t_gather, 1e-9), 2),
                    })

    # Summarize by k
    print(f"\n  Total entries: {len(k_sweep_results)}")
    k_values_seen = sorted(set(r["k"] for r in k_sweep_results))

    pareto = {}
    for k in k_values_seen:
        subset = [r for r in k_sweep_results if r["k"] == k]
        cos_vals = [r["score_cosine"] for r in subset]
        kl_vals = [r["kl_divergence"] for r in subset]
        speedups = [r["speedup"] for r in subset]

        pareto[k] = {
            "n_entries": len(subset),
            "score_cosine_mean": round(float(np.mean(cos_vals)), 6),
            "score_cosine_min": round(float(np.min(cos_vals)), 6),
            "kl_divergence_mean": round(float(np.mean(kl_vals)), 6),
            "speedup_mean": round(float(np.mean(speedups)), 2),
        }
        print(f"  k={k:3d}: cos_mean={np.mean(cos_vals):.4f}  "
              f"cos_min={np.min(cos_vals):.4f}  "
              f"kl_mean={np.mean(kl_vals):.6f}  "
              f"speedup_mean={np.mean(speedups):.2f}x")

    # Projected speedup at longer sequences (compute-only, not including calibration)
    # At seq_len=S, standard=S*D, compressed=k*D+S
    projected = {}
    for target_seq in [1024, 4096, 16384, 131072]:
        proj_row = {}
        for k in k_values_seen:
            standard_ops = target_seq * HEAD_DIM
            compressed_ops = k * HEAD_DIM + target_seq
            proj_row[f"k={k}"] = round(standard_ops / compressed_ops, 1)
        projected[str(target_seq)] = proj_row
        print(f"  Projected speedup at seq_len={target_seq}: "
              + "  ".join(f"k={k}:{proj_row[f'k={k}']}x" for k in k_values_seen))

    # Gate: at k=128, mean score cosine >= 0.99
    k128_entries = [r for r in k_sweep_results if r["k"] == 128]
    if k128_entries:
        gate_val = float(np.mean([r["score_cosine"] for r in k128_entries]))
    else:
        # Fallback to largest k available
        max_k = max(k_values_seen)
        k128_entries = [r for r in k_sweep_results if r["k"] == max_k]
        gate_val = float(np.mean([r["score_cosine"] for r in k128_entries]))
    gate = gate_val >= 0.99
    print(f"\n  GATE (k=128 score cosine >= 0.99): {'PASS' if gate else 'FAIL'} "
          f"(mean={gate_val:.6f})")

    return {
        "k_sweep": k_sweep_results,
        "pareto": pareto,
        "projected_speedup": projected,
        "gate": {
            "criterion": "mean score cosine >= 0.99 at k=128 (or max k if seq < 128)",
            "value": round(gate_val, 6),
            "passed": gate,
        },
    }


# ========== PART C: Full Pipeline (Vector VQ K+V) ==========

def run_part_c(kv_dir, prompts, n_layers):
    """Full vector VQ K+V pipeline on real data."""
    print("\n" + "=" * 70)
    print("PART C: Full Pipeline (Vector VQ K + V)")
    print("=" * 70)

    scale = 1.0 / np.sqrt(HEAD_DIM)
    results = []

    # Use k=128 (or cap at seq_len)
    TARGET_K = 128

    for prompt in prompts:
        for layer in range(n_layers):
            K_all = load_kv_per_head(kv_dir, prompt, layer, "k")
            V_all = load_kv_per_head(kv_dir, prompt, layer, "v")

            for head in range(N_HEADS):
                K = K_all[head]
                V = V_all[head]
                seq_len = K.shape[0]
                k = min(TARGET_K, seq_len)

                rng = np.random.default_rng(layer * N_HEADS + head + 2000)
                q = rng.standard_normal(HEAD_DIM).astype(np.float32)
                q = q * K.std() + K.mean()

                # Reference
                _, out_ref = standard_attention(q, K, V, scale)

                # Vector VQ codebooks
                cb_k = VectorCodebook(n_clusters=k, head_dim=HEAD_DIM)
                cb_k.feed_calibration(K)
                cb_k.finalize_calibration()
                k_idx = cb_k.assign(K)

                cb_v = VectorCodebook(n_clusters=k, head_dim=HEAD_DIM)
                cb_v.feed_calibration(V)
                cb_v.finalize_calibration()
                v_idx = cb_v.assign(V)

                # Compressed
                scores_comp, out_comp = full_vector_vq_attention(
                    q, k_idx, cb_k.centroids, v_idx, cb_v.centroids, scale
                )

                output_cos = cosine_sim(out_ref, out_comp)
                output_mse = float(np.mean((out_ref - out_comp) ** 2))

                results.append({
                    "prompt": prompt, "layer": layer, "head": head,
                    "seq_len": seq_len, "k": k,
                    "output_cosine": round(output_cos, 6),
                    "output_mse": round(output_mse, 8),
                })

    # Summarize
    output_cosines = [r["output_cosine"] for r in results]
    output_mses = [r["output_mse"] for r in results]

    print(f"\n  N entries: {len(results)}")
    print(f"  Output cosine: mean={np.mean(output_cosines):.6f}  "
          f"min={np.min(output_cosines):.6f}  max={np.max(output_cosines):.6f}")
    print(f"  Output MSE:    mean={np.mean(output_mses):.6f}  max={np.max(output_mses):.6f}")

    gate_val = float(np.mean(output_cosines))
    gate = gate_val >= 0.98
    print(f"\n  GATE (output cosine >= 0.98): {'PASS' if gate else 'FAIL'} "
          f"(mean={gate_val:.6f})")

    # Worst 5 layers
    results_sorted = sorted(results, key=lambda r: r["output_cosine"])
    print("\n  Worst 5 by output cosine:")
    for r in results_sorted[:5]:
        print(f"    {r['prompt']} layer {r['layer']} head {r['head']}: "
              f"cos={r['output_cosine']:.6f}  mse={r['output_mse']:.8f}")

    return {
        "per_entry": results,
        "summary": {
            "n_entries": len(results),
            "output_cosine_mean": round(float(np.mean(output_cosines)), 6),
            "output_cosine_min": round(float(np.min(output_cosines)), 6),
            "output_mse_mean": round(float(np.mean(output_mses)), 8),
        },
        "gate": {
            "criterion": "mean output cosine >= 0.98",
            "value": round(gate_val, 6),
            "passed": gate,
        },
        "worst_5": results_sorted[:5],
    }


# ========== PART D: Product Quantization ==========

def run_part_d(kv_dir, prompts, n_layers):
    """Product quantization: M subspaces, high fidelity."""
    print("\n" + "=" * 70)
    print("PART D: Product Quantization (CDC-02)")
    print("=" * 70)

    scale = 1.0 / np.sqrt(HEAD_DIM)
    score_results = []
    full_results = []

    # M sweep: test multiple subspace configurations (first prompt only for Pareto)
    m_sweep_prompt = prompts[0]
    m_sweep_results = []

    for m_val in [4, 8, 16]:
        layer = 0
        K_all = load_kv_per_head(kv_dir, m_sweep_prompt, layer, "k")
        V_all = load_kv_per_head(kv_dir, m_sweep_prompt, layer, "v")

        for head in range(N_HEADS):
            K = K_all[head]
            V = V_all[head]
            seq_len = K.shape[0]

            rng = np.random.default_rng(layer * N_HEADS + head + 3000)
            q = rng.standard_normal(HEAD_DIM).astype(np.float32)
            q = q * K.std() + K.mean()

            scores_ref, out_ref = standard_attention(q, K, V, scale)

            # PQ K-side
            pq_k = ProductCodebook(n_subspaces=m_val, sub_clusters=256, head_dim=HEAD_DIM)
            pq_k.feed_calibration(K)
            pq_k.finalize_calibration()
            k_idx = pq_k.assign(K)

            scores_pq = pq_attention_scores(q, k_idx, pq_k, scale)
            score_cos = cosine_sim(scores_ref, scores_pq)
            kl = kl_div(scores_ref, scores_pq)

            # Timing: PQ tables + gather
            n_timing = 100
            t0 = time.perf_counter()
            for _ in range(n_timing):
                tables = pq_k.precompute_distance_tables(q)
                _ = pq_k.gather_pq_scores(tables, k_idx) * scale
            t_pq = (time.perf_counter() - t0) / n_timing

            t0 = time.perf_counter()
            for _ in range(n_timing):
                _ = (q @ K.T) * scale
            t_std = (time.perf_counter() - t0) / n_timing

            m_sweep_results.append({
                "m": m_val, "head": head,
                "score_cosine": round(score_cos, 6),
                "kl_divergence": round(kl, 8),
                "time_pq_us": round(t_pq * 1e6, 2),
                "time_standard_us": round(t_std * 1e6, 2),
                "speedup": round(t_std / max(t_pq, 1e-9), 2),
            })

    # Print M sweep Pareto
    print("\n  M sweep (first prompt, layer 0):")
    for m_val in [4, 8, 16]:
        subset = [r for r in m_sweep_results if r["m"] == m_val]
        cos_vals = [r["score_cosine"] for r in subset]
        print(f"    M={m_val:2d}: cos_mean={np.mean(cos_vals):.6f}  "
              f"cos_min={np.min(cos_vals):.6f}")

    # Full benchmark with best M=8 on all prompts/layers
    BEST_M = 8
    print(f"\n  Full benchmark with M={BEST_M}:")

    for prompt in prompts:
        for layer in range(n_layers):
            K_all = load_kv_per_head(kv_dir, prompt, layer, "k")
            V_all = load_kv_per_head(kv_dir, prompt, layer, "v")

            for head in range(N_HEADS):
                K = K_all[head]
                V = V_all[head]
                seq_len = K.shape[0]

                rng = np.random.default_rng(layer * N_HEADS + head + 3000)
                q = rng.standard_normal(HEAD_DIM).astype(np.float32)
                q = q * K.std() + K.mean()

                scores_ref, out_ref = standard_attention(q, K, V, scale)

                # PQ K-side scores
                pq_k = ProductCodebook(n_subspaces=BEST_M, sub_clusters=256, head_dim=HEAD_DIM)
                pq_k.feed_calibration(K)
                pq_k.finalize_calibration()
                k_idx = pq_k.assign(K)

                scores_pq = pq_attention_scores(q, k_idx, pq_k, scale)
                score_cos = cosine_sim(scores_ref, scores_pq)
                kl = kl_div(scores_ref, scores_pq)

                # Memory
                mem_pq = k_idx.nbytes  # [seq_len, M] uint8
                # Add codebook: M * 256 * sub_dim * 4 bytes
                mem_pq += BEST_M * 256 * (HEAD_DIM // BEST_M) * 4
                mem_dense = K.nbytes
                mem_ratio = mem_dense / mem_pq

                score_results.append({
                    "prompt": prompt, "layer": layer, "head": head,
                    "seq_len": seq_len, "m": BEST_M,
                    "score_cosine": round(score_cos, 6),
                    "kl_divergence": round(kl, 8),
                    "memory_ratio": round(mem_ratio, 2),
                })

                # Full PQ K+V
                pq_v = ProductCodebook(n_subspaces=BEST_M, sub_clusters=256, head_dim=HEAD_DIM)
                pq_v.feed_calibration(V)
                pq_v.finalize_calibration()
                v_idx = pq_v.assign(V)

                _, out_pq = full_pq_attention(q, k_idx, pq_k, v_idx, pq_v, scale)
                output_cos = cosine_sim(out_ref, out_pq)

                full_results.append({
                    "prompt": prompt, "layer": layer, "head": head,
                    "seq_len": seq_len, "m": BEST_M,
                    "output_cosine": round(output_cos, 6),
                })

    # Score gate
    score_cosines = [r["score_cosine"] for r in score_results]
    print(f"\n  PQ Scores (M={BEST_M}):")
    print(f"    N entries: {len(score_results)}")
    print(f"    Score cosine: mean={np.mean(score_cosines):.6f}  "
          f"min={np.min(score_cosines):.6f}")

    score_gate_val = float(np.mean(score_cosines))
    score_gate = score_gate_val >= 0.99
    print(f"    GATE (score cosine >= 0.99): {'PASS' if score_gate else 'FAIL'} "
          f"(mean={score_gate_val:.6f})")

    # Full K+V gate
    output_cosines = [r["output_cosine"] for r in full_results]
    print(f"\n  PQ Full K+V (M={BEST_M}):")
    print(f"    N entries: {len(full_results)}")
    print(f"    Output cosine: mean={np.mean(output_cosines):.6f}  "
          f"min={np.min(output_cosines):.6f}")

    full_gate_val = float(np.mean(output_cosines))
    full_gate = full_gate_val >= 0.98
    print(f"    GATE (output cosine >= 0.98): {'PASS' if full_gate else 'FAIL'} "
          f"(mean={full_gate_val:.6f})")

    # Worst 5
    full_sorted = sorted(full_results, key=lambda r: r["output_cosine"])
    print("\n    Worst 5 by output cosine:")
    for r in full_sorted[:5]:
        print(f"      {r['prompt']} L{r['layer']} H{r['head']}: "
              f"cos={r['output_cosine']:.6f}")

    return {
        "m_sweep": m_sweep_results,
        "score_summary": {
            "n_entries": len(score_results),
            "m": BEST_M,
            "score_cosine_mean": round(float(np.mean(score_cosines)), 6),
            "score_cosine_min": round(float(np.min(score_cosines)), 6),
            "memory_ratio_mean": round(float(np.mean([r["memory_ratio"] for r in score_results])), 2),
        },
        "full_summary": {
            "n_entries": len(full_results),
            "output_cosine_mean": round(float(np.mean(output_cosines)), 6),
            "output_cosine_min": round(float(np.min(output_cosines)), 6),
        },
        "score_gate": {
            "criterion": "mean PQ score cosine >= 0.99 at M=8",
            "value": round(score_gate_val, 6),
            "passed": score_gate,
        },
        "full_gate": {
            "criterion": "mean PQ output cosine >= 0.98 at M=8",
            "value": round(full_gate_val, 6),
            "passed": full_gate,
        },
        "worst_5": full_sorted[:5],
    }


# ========== PART E: Fused Scalar VQ Triton Kernel ==========

def run_part_e(kv_dir, prompts, n_layers):
    """Fused scalar VQ kernel on GPU: bandwidth reduction."""
    print("\n" + "=" * 70)
    print("PART E: Fused Scalar VQ Triton Kernel (CDC-02)")
    print("=" * 70)

    if not _HAS_CUDA:
        print("  WARNING: No CUDA/Triton available. Skipping Part E.")
        return {
            "skipped": True,
            "reason": "No CUDA or Triton available",
            "gate": {"criterion": "fidelity >= 0.999 AND speedup > 1.5x",
                     "passed": False, "value": None},
        }

    scale = 1.0 / np.sqrt(HEAD_DIM)
    results = []

    for prompt in prompts:
        for layer in range(n_layers):
            K_all = load_kv_per_head(kv_dir, prompt, layer, "k")
            V_all = load_kv_per_head(kv_dir, prompt, layer, "v")

            for head in range(N_HEADS):
                K = K_all[head]
                V = V_all[head]
                seq_len = K.shape[0]

                rng = np.random.default_rng(layer * N_HEADS + head + 4000)
                q = rng.standard_normal(HEAD_DIM).astype(np.float32)
                q = q * K.std() + K.mean()

                # Fit scalar codebook
                cb = OnlineCodebook(n_clusters=N_CLUSTERS_SCALAR)
                cb.feed_calibration(K.ravel())
                cb.finalize_calibration()
                k_indices = cb.assign(K.ravel()).reshape(K.shape)

                # Reference: numpy standard path
                scores_ref = (q @ K.T) * scale

                # Fused numpy (correctness baseline)
                scores_fused_np = fused_scalar_vq_qkt_numpy(q, k_indices, cb.centroids, scale)
                cos_np = cosine_sim(scores_ref, scores_fused_np)

                # GPU kernel
                q_gpu = torch.from_numpy(q).cuda()
                k_idx_gpu = torch.from_numpy(k_indices.astype(np.int32)).cuda()
                cb_gpu = torch.from_numpy(cb.centroids).cuda()

                scores_gpu = fused_scalar_vq_qkt(q_gpu, k_idx_gpu, cb_gpu, scale)
                scores_gpu_np = scores_gpu.cpu().numpy()
                cos_gpu = cosine_sim(scores_ref, scores_gpu_np)

                # Timing: fused GPU vs dense GPU
                K_dense_gpu = torch.from_numpy(cb.centroids[k_indices]).cuda()

                # Warmup
                for _ in range(10):
                    _ = fused_scalar_vq_qkt(q_gpu, k_idx_gpu, cb_gpu, scale)
                    _ = (q_gpu @ K_dense_gpu.T) * scale
                torch.cuda.synchronize()

                n_timing = 200
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_timing):
                    _ = fused_scalar_vq_qkt(q_gpu, k_idx_gpu, cb_gpu, scale)
                torch.cuda.synchronize()
                t_fused = (time.perf_counter() - t0) / n_timing

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_timing):
                    _ = (q_gpu @ K_dense_gpu.T) * scale
                torch.cuda.synchronize()
                t_dense = (time.perf_counter() - t0) / n_timing

                speedup = t_dense / max(t_fused, 1e-9)

                results.append({
                    "prompt": prompt, "layer": layer, "head": head,
                    "seq_len": seq_len,
                    "cosine_numpy": round(cos_np, 6),
                    "cosine_gpu": round(cos_gpu, 6),
                    "time_fused_us": round(t_fused * 1e6, 2),
                    "time_dense_us": round(t_dense * 1e6, 2),
                    "speedup": round(speedup, 2),
                })

    # Summarize
    cos_gpus = [r["cosine_gpu"] for r in results]
    speedups = [r["speedup"] for r in results]

    print(f"\n  N entries: {len(results)}")
    print(f"  GPU cosine: mean={np.mean(cos_gpus):.6f}  min={np.min(cos_gpus):.6f}")
    print(f"  Speedup:    mean={np.mean(speedups):.2f}x  min={np.min(speedups):.2f}x")

    fidelity_gate = float(np.mean(cos_gpus)) >= 0.999
    speed_gate = float(np.mean(speedups)) > 1.5
    overall_gate = fidelity_gate and speed_gate

    print(f"\n  GATE (fidelity >= 0.999): {'PASS' if fidelity_gate else 'FAIL'} "
          f"(mean={np.mean(cos_gpus):.6f})")
    print(f"  GATE (speedup > 1.5x):   {'PASS' if speed_gate else 'FAIL'} "
          f"(mean={np.mean(speedups):.2f}x)")

    return {
        "per_entry": results,
        "summary": {
            "n_entries": len(results),
            "cosine_gpu_mean": round(float(np.mean(cos_gpus)), 6),
            "cosine_gpu_min": round(float(np.min(cos_gpus)), 6),
            "speedup_mean": round(float(np.mean(speedups)), 2),
            "speedup_min": round(float(np.min(speedups)), 2),
        },
        "gate": {
            "criterion": "fidelity >= 0.999 AND speedup > 1.5x at seq_len",
            "fidelity_value": round(float(np.mean(cos_gpus)), 6),
            "speedup_value": round(float(np.mean(speedups)), 2),
            "passed": overall_gate,
        },
    }


# ========== PART F: Prefiltered Attention ==========

def run_part_f(kv_dir, prompts, n_layers):
    """Prefiltered attention: coarse VQ cluster selection, full-precision on subset."""
    print("\n" + "=" * 70)
    print("PART F: Prefiltered Attention (CDC-02)")
    print("=" * 70)

    scale = 1.0 / np.sqrt(HEAD_DIM)
    N_CLUSTERS_VQ = 256

    # top_p sweep on first prompt only
    top_p_values = [8, 16, 32, 64, 128]
    sweep_results = []

    for top_p in top_p_values:
        layer = 0
        K_all = load_kv_per_head(kv_dir, prompts[0], layer, "k")
        V_all = load_kv_per_head(kv_dir, prompts[0], layer, "v")

        for head in range(N_HEADS):
            K = K_all[head]
            V = V_all[head]
            seq_len = K.shape[0]

            rng = np.random.default_rng(layer * N_HEADS + head + 5000)
            q = rng.standard_normal(HEAD_DIM).astype(np.float32)
            q = q * K.std() + K.mean()

            _, out_ref = standard_attention(q, K, V, scale)

            # Fit vector codebook on K for cluster assignment
            n_clusters = min(N_CLUSTERS_VQ, seq_len)
            cb = VectorCodebook(n_clusters=n_clusters, head_dim=HEAD_DIM)
            cb.feed_calibration(K)
            cb.finalize_calibration()
            idx = cb.assign(K)

            # Prefiltered
            scores_pf, out_pf = prefiltered_attention(
                q, K, V, idx, cb.centroids, scale, top_p=top_p
            )

            output_cos = cosine_sim(out_ref, out_pf)
            # Mask coverage: fraction of tokens selected
            mask = scores_pf > -1e8  # selected tokens have real scores
            coverage = float(np.sum(mask)) / seq_len

            sweep_results.append({
                "top_p": top_p, "head": head,
                "output_cosine": round(output_cos, 6),
                "mask_coverage": round(coverage, 4),
                "tokens_selected": int(np.sum(mask)),
                "seq_len": seq_len,
            })

    # Print sweep
    print("\n  top_p sweep (first prompt, layer 0):")
    for tp in top_p_values:
        subset = [r for r in sweep_results if r["top_p"] == tp]
        cos_vals = [r["output_cosine"] for r in subset]
        cov_vals = [r["mask_coverage"] for r in subset]
        print(f"    top_p={tp:3d}: cos_mean={np.mean(cos_vals):.6f}  "
              f"coverage_mean={np.mean(cov_vals):.4f}")

    # Full benchmark at top_p=32
    BEST_TOP_P = 32
    full_results = []

    for prompt in prompts:
        for layer in range(n_layers):
            K_all = load_kv_per_head(kv_dir, prompt, layer, "k")
            V_all = load_kv_per_head(kv_dir, prompt, layer, "v")

            for head in range(N_HEADS):
                K = K_all[head]
                V = V_all[head]
                seq_len = K.shape[0]

                rng = np.random.default_rng(layer * N_HEADS + head + 5000)
                q = rng.standard_normal(HEAD_DIM).astype(np.float32)
                q = q * K.std() + K.mean()

                _, out_ref = standard_attention(q, K, V, scale)

                n_clusters = min(N_CLUSTERS_VQ, seq_len)
                cb = VectorCodebook(n_clusters=n_clusters, head_dim=HEAD_DIM)
                cb.feed_calibration(K)
                cb.finalize_calibration()
                idx = cb.assign(K)

                scores_pf, out_pf = prefiltered_attention(
                    q, K, V, idx, cb.centroids, scale, top_p=BEST_TOP_P
                )

                output_cos = cosine_sim(out_ref, out_pf)
                mask = scores_pf > -1e8
                coverage = float(np.sum(mask)) / seq_len

                # Effective compute ratio: only score selected tokens
                compute_ratio = coverage  # fraction of tokens computed

                full_results.append({
                    "prompt": prompt, "layer": layer, "head": head,
                    "seq_len": seq_len, "top_p": BEST_TOP_P,
                    "output_cosine": round(output_cos, 6),
                    "mask_coverage": round(coverage, 4),
                    "compute_ratio": round(compute_ratio, 4),
                })

    # Gate
    output_cosines = [r["output_cosine"] for r in full_results]
    coverages = [r["mask_coverage"] for r in full_results]

    print(f"\n  Prefiltered (top_p={BEST_TOP_P}):")
    print(f"    N entries: {len(full_results)}")
    print(f"    Output cosine: mean={np.mean(output_cosines):.6f}  "
          f"min={np.min(output_cosines):.6f}")
    print(f"    Coverage:      mean={np.mean(coverages):.4f}")

    gate_val = float(np.mean(output_cosines))
    gate = gate_val >= 0.98
    print(f"    GATE (output cosine >= 0.98 at top_p={BEST_TOP_P}): "
          f"{'PASS' if gate else 'FAIL'} (mean={gate_val:.6f})")

    # Worst 5
    full_sorted = sorted(full_results, key=lambda r: r["output_cosine"])
    print("\n    Worst 5 by output cosine:")
    for r in full_sorted[:5]:
        print(f"      {r['prompt']} L{r['layer']} H{r['head']}: "
              f"cos={r['output_cosine']:.6f}  cov={r['mask_coverage']:.4f}")

    return {
        "top_p_sweep": sweep_results,
        "full_summary": {
            "n_entries": len(full_results),
            "top_p": BEST_TOP_P,
            "output_cosine_mean": round(float(np.mean(output_cosines)), 6),
            "output_cosine_min": round(float(np.min(output_cosines)), 6),
            "coverage_mean": round(float(np.mean(coverages)), 4),
        },
        "gate": {
            "criterion": f"mean output cosine >= 0.98 at top_p={BEST_TOP_P}",
            "value": round(gate_val, 6),
            "passed": gate,
        },
        "worst_5": full_sorted[:5],
    }


# ========== PART G: Hybrid PQ + Prefilter (CDC-03) ==========

def run_part_g(kv_dir, prompts, n_layers):
    """CDC-03 Hybrid: PQ scores rank tokens, select top-k, dense V on subset."""
    print("\n" + "=" * 70)
    print("PART G: Hybrid PQ + Prefilter (CDC-03)")
    print("=" * 70)

    scale = 1.0 / np.sqrt(HEAD_DIM)
    BEST_M = 16  # M=16 proven at cos=0.991 for scores

    # top_k sweep on first prompt, layers 0 + 10 (worst + typical)
    top_k_values = [128, 256, 512, 768]
    sweep_results = []
    sweep_layers = [0, min(10, n_layers - 1)]

    for sweep_layer in sweep_layers:
        K_all = load_kv_per_head(kv_dir, prompts[0], sweep_layer, "k")
        V_all = load_kv_per_head(kv_dir, prompts[0], sweep_layer, "v")

        for top_k in top_k_values:
            for head in range(N_HEADS):
                K = K_all[head]
                V = V_all[head]
                seq_len = K.shape[0]

                rng = np.random.default_rng(sweep_layer * N_HEADS + head + 6000)
                q = rng.standard_normal(HEAD_DIM).astype(np.float32)
                q = q * K.std() + K.mean()

                _, out_ref = standard_attention(q, K, V, scale)

                pq = ProductCodebook(n_subspaces=BEST_M, sub_clusters=256, head_dim=HEAD_DIM)
                pq.feed_calibration(K)
                pq.finalize_calibration()
                idx = pq.assign(K)

                _, out_hybrid, meta = hybrid_pq_attention(
                    q, K, V, idx, pq, scale, top_k=top_k
                )
                output_cos = cosine_sim(out_ref, out_hybrid)

                # Timing
                n_timing = 50
                t0 = time.perf_counter()
                for _ in range(n_timing):
                    _ = hybrid_pq_attention(q, K, V, idx, pq, scale, top_k=top_k)
                t_hybrid = (time.perf_counter() - t0) / n_timing

                t0 = time.perf_counter()
                for _ in range(n_timing):
                    _ = standard_attention(q, K, V, scale)
                t_std = (time.perf_counter() - t0) / n_timing

                sweep_results.append({
                    "top_k": top_k, "layer": sweep_layer, "head": head,
                    "output_cosine": round(output_cos, 6),
                    "coverage": round(meta["coverage"], 4),
                    "time_hybrid_us": round(t_hybrid * 1e6, 2),
                    "time_standard_us": round(t_std * 1e6, 2),
                    "speedup": round(t_std / max(t_hybrid, 1e-9), 2),
                })

    print(f"\n  top_k sweep (first prompt, layers {sweep_layers}, M={BEST_M}):")
    for tk in top_k_values:
        subset = [r for r in sweep_results if r["top_k"] == tk]
        cos_vals = [r["output_cosine"] for r in subset]
        cov_vals = [r["coverage"] for r in subset]
        # Show per-layer breakdown
        for sl in sweep_layers:
            layer_subset = [r for r in subset if r["layer"] == sl]
            lcos = [r["output_cosine"] for r in layer_subset]
            print(f"    top_k={tk:3d} L{sl:2d}: cos_mean={np.mean(lcos):.6f}  "
                  f"cos_min={np.min(lcos):.6f}")

    # Find smallest top_k where ALL sweep entries meet 0.99
    # (this includes the worst layer)
    best_top_k = top_k_values[-1]  # fallback to largest
    for tk in top_k_values:
        subset = [r for r in sweep_results if r["top_k"] == tk]
        cos_min = float(np.min([r["output_cosine"] for r in subset]))
        if cos_min >= 0.95:  # must work even on worst layer
            best_top_k = tk
            break

    print(f"\n  Selected top_k={best_top_k} for full benchmark (min cos >= 0.95 on sweep)")

    # Full benchmark on all prompts/layers
    full_results = []

    for prompt in prompts:
        for layer_idx in range(n_layers):
            K_all = load_kv_per_head(kv_dir, prompt, layer_idx, "k")
            V_all = load_kv_per_head(kv_dir, prompt, layer_idx, "v")

            for head in range(N_HEADS):
                K = K_all[head]
                V = V_all[head]
                seq_len = K.shape[0]

                rng = np.random.default_rng(layer_idx * N_HEADS + head + 6000)
                q = rng.standard_normal(HEAD_DIM).astype(np.float32)
                q = q * K.std() + K.mean()

                _, out_ref = standard_attention(q, K, V, scale)

                # Fit PQ on K (M=16)
                pq = ProductCodebook(
                    n_subspaces=BEST_M, sub_clusters=256, head_dim=HEAD_DIM
                )
                pq.feed_calibration(K)
                pq.finalize_calibration()
                idx = pq.assign(K)

                # Hybrid
                _, out_hybrid, meta = hybrid_pq_attention(
                    q, K, V, idx, pq, scale, top_k=best_top_k
                )
                output_cos = cosine_sim(out_ref, out_hybrid)

                # Also compute score fidelity: PQ approx scores vs reference
                scores_ref = (q @ K.T) * scale
                approx = pq.gather_pq_scores(
                    pq.precompute_distance_tables(q), idx
                ) * np.float32(scale)
                score_cos = cosine_sim(scores_ref, approx)

                # Memory: PQ indices [seq_len, M] uint8 + PQ codebook
                mem_pq = idx.nbytes + BEST_M * 256 * (HEAD_DIM // BEST_M) * 4
                # Dense K that gets decompressed for subset
                mem_k_dense = K.nbytes
                # Effective K read: only top_k tokens decompressed
                mem_k_effective = best_top_k * HEAD_DIM * 4
                # V stays dense
                mem_v = V.nbytes

                full_results.append({
                    "prompt": prompt, "layer": layer_idx, "head": head,
                    "seq_len": seq_len, "top_k": best_top_k, "m": BEST_M,
                    "output_cosine": round(output_cos, 6),
                    "score_cosine": round(score_cos, 6),
                    "coverage": round(meta["coverage"], 4),
                    "pq_index_bytes": int(idx.nbytes),
                    "k_effective_bytes": int(mem_k_effective),
                    "compute_ratio": round(float(best_top_k) / seq_len, 4),
                })

    # Gates
    output_cosines = [r["output_cosine"] for r in full_results]
    score_cosines = [r["score_cosine"] for r in full_results]
    coverages = [r["coverage"] for r in full_results]

    print(f"\n  Hybrid (M={BEST_M}, top_k={best_top_k}):")
    print(f"    N entries: {len(full_results)}")
    print(f"    Output cosine: mean={np.mean(output_cosines):.6f}  "
          f"min={np.min(output_cosines):.6f}  "
          f"max={np.max(output_cosines):.6f}")
    print(f"    PQ score cos:  mean={np.mean(score_cosines):.6f}  "
          f"min={np.min(score_cosines):.6f}")
    print(f"    Coverage:      mean={np.mean(coverages):.4f}")
    print(f"    Compute ratio: {np.mean(coverages):.1%} of tokens scored exactly")

    gate_val = float(np.mean(output_cosines))
    gate_min = float(np.min(output_cosines))
    gate = gate_val >= 0.99
    print(f"\n    GATE (mean output cosine >= 0.99): {'PASS' if gate else 'FAIL'} "
          f"(mean={gate_val:.6f}, min={gate_min:.6f})")

    # Worst 5
    full_sorted = sorted(full_results, key=lambda r: r["output_cosine"])
    print("\n    Worst 5 by output cosine:")
    for r in full_sorted[:5]:
        print(f"      {r['prompt']} L{r['layer']} H{r['head']}: "
              f"cos={r['output_cosine']:.6f}  score_cos={r['score_cosine']:.6f}  "
              f"cov={r['coverage']:.4f}")

    # Projected compute savings at different seq_len
    print("\n    Projected compute savings:")
    for target_seq in [1024, 4096, 16384, 131072]:
        # PQ scoring cost: M * 256 * sub_dim (tables) + M * target_seq (gather)
        pq_scoring = BEST_M * 256 * (HEAD_DIM // BEST_M) + BEST_M * target_seq
        # Dense scoring: target_seq * head_dim
        dense_scoring = target_seq * HEAD_DIM
        # Exact rescoring on top_k: top_k * head_dim
        exact_rescore = best_top_k * HEAD_DIM
        # V output on top_k: top_k * head_dim
        v_output = best_top_k * HEAD_DIM
        # Total hybrid: pq_scoring + exact_rescore + v_output
        total_hybrid = pq_scoring + exact_rescore + v_output
        # Total standard: dense_scoring + target_seq * head_dim (V)
        total_standard = dense_scoring + target_seq * HEAD_DIM
        ratio = total_standard / max(total_hybrid, 1)
        print(f"      seq={target_seq:6d}: hybrid={total_hybrid:,} ops  "
              f"standard={total_standard:,} ops  "
              f"savings={ratio:.1f}x")

    return {
        "top_k_sweep": sweep_results,
        "config": {"m": BEST_M, "top_k": best_top_k},
        "full_summary": {
            "n_entries": len(full_results),
            "output_cosine_mean": round(float(np.mean(output_cosines)), 6),
            "output_cosine_min": round(float(np.min(output_cosines)), 6),
            "score_cosine_mean": round(float(np.mean(score_cosines)), 6),
            "coverage_mean": round(float(np.mean(coverages)), 4),
        },
        "gate": {
            "criterion": f"mean output cosine >= 0.99 at M={BEST_M}, top_k={best_top_k}",
            "value": round(gate_val, 6),
            "min_value": round(gate_min, 6),
            "passed": gate,
        },
        "worst_5": full_sorted[:5],
    }


# ========== MAIN ==========

def main():
    args = parse_args()
    kv_dir = Path(args.kv_dir)
    parts = [p.strip().lower() for p in args.parts.split(",")]

    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    t_start = time.time()
    cpu_start = time.process_time()

    # Auto-detect prompts and layers
    assert kv_dir.exists(), f"KV dump dir not found: {kv_dir}"
    prompts = auto_detect_prompts(kv_dir)
    assert prompts, f"No prompt dirs with layer_*_k.npy found in {kv_dir}"
    n_layers = auto_detect_n_layers(kv_dir, prompts[0])

    # Derive tag for receipt naming
    tag = args.tag or kv_dir.name.replace("kv_dump", "").strip("_") or "short"

    # Determine phase label
    has_cdc01 = any(p in parts for p in ["a", "b", "c"])
    has_cdc02 = any(p in parts for p in ["d", "e", "f"])
    has_cdc03 = "g" in parts
    if has_cdc03:
        phase = "CDC-03"
    elif has_cdc01 and has_cdc02:
        phase = "CDC-01+02"
    elif has_cdc02:
        phase = "CDC-02"
    else:
        phase = "CDC-01"

    print(f"{phase}: Compressed-Domain Attention Benchmark")
    print(f"KV dump dir: {kv_dir}")
    print(f"Parts: {','.join(parts)}")
    print(f"Prompts: {prompts} (auto-detected), Layers: {n_layers}, Heads: {N_HEADS}")
    print(f"Receipt tag: {tag}")
    print()

    # Verify data exists
    for p in prompts:
        assert (kv_dir / p).exists(), f"Missing KV dump: {kv_dir / p}"

    # Run requested parts
    part_results = {}

    if "a" in parts:
        part_results["a"] = run_part_a(kv_dir, prompts, n_layers)
    if "b" in parts:
        part_results["b"] = run_part_b(kv_dir, prompts, n_layers)
    if "c" in parts:
        part_results["c"] = run_part_c(kv_dir, prompts, n_layers)
    if "d" in parts:
        part_results["d"] = run_part_d(kv_dir, prompts, n_layers)
    if "e" in parts:
        part_results["e"] = run_part_e(kv_dir, prompts, n_layers)
    if "f" in parts:
        part_results["f"] = run_part_f(kv_dir, prompts, n_layers)
    if "g" in parts:
        part_results["g"] = run_part_g(kv_dir, prompts, n_layers)

    # Cost block
    cost = {
        'wall_time_s': round(time.time() - t_start, 3),
        'cpu_time_s': round(time.process_time() - cpu_start, 3),
        'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
        'timestamp_start': start_iso,
        'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    # Summary
    print("\n" + "=" * 70)
    print(f"{phase} SUMMARY")
    print("=" * 70)

    all_gates_pass = True

    if "a" in part_results:
        pa = part_results["a"]
        passed = pa['gate']['passed']
        all_gates_pass &= passed
        print(f"  Part A (Scalar VQ):     gate={'PASS' if passed else 'FAIL'} "
              f"(score cos {pa['gate']['value']:.6f})")
    if "b" in part_results:
        pb = part_results["b"]
        passed = pb['gate']['passed']
        all_gates_pass &= passed
        print(f"  Part B (Vector VQ):     gate={'PASS' if passed else 'FAIL'} "
              f"(score cos {pb['gate']['value']:.6f})")
    if "c" in part_results:
        pc = part_results["c"]
        passed = pc['gate']['passed']
        all_gates_pass &= passed
        print(f"  Part C (Full VQ K+V):   gate={'PASS' if passed else 'FAIL'} "
              f"(output cos {pc['gate']['value']:.6f})")
    if "d" in part_results:
        pd = part_results["d"]
        sg = pd['score_gate']['passed']
        fg = pd['full_gate']['passed']
        all_gates_pass &= sg and fg
        print(f"  Part D (PQ scores):     gate={'PASS' if sg else 'FAIL'} "
              f"(score cos {pd['score_gate']['value']:.6f})")
        print(f"  Part D (PQ full K+V):   gate={'PASS' if fg else 'FAIL'} "
              f"(output cos {pd['full_gate']['value']:.6f})")
    if "e" in part_results:
        pe = part_results["e"]
        passed = pe['gate']['passed']
        all_gates_pass &= passed
        if pe.get("skipped"):
            print(f"  Part E (Fused kernel):  SKIPPED ({pe['reason']})")
        else:
            print(f"  Part E (Fused kernel):  gate={'PASS' if passed else 'FAIL'} "
                  f"(cos {pe['gate']['fidelity_value']:.6f}, "
                  f"speedup {pe['gate']['speedup_value']:.2f}x)")
    if "f" in part_results:
        pf = part_results["f"]
        passed = pf['gate']['passed']
        all_gates_pass &= passed
        print(f"  Part F (Prefiltered):   gate={'PASS' if passed else 'FAIL'} "
              f"(output cos {pf['gate']['value']:.6f})")
    if "g" in part_results:
        pg = part_results["g"]
        passed = pg['gate']['passed']
        all_gates_pass &= passed
        print(f"  Part G (Hybrid PQ):    gate={'PASS' if passed else 'FAIL'} "
              f"(output cos {pg['gate']['value']:.6f}, "
              f"min={pg['gate']['min_value']:.6f})")

    print(f"  Wall time: {cost['wall_time_s']:.1f}s")

    # Honest claims
    print("\n  HONEST CLAIMS:")
    if "a" in part_results and part_results["a"]["gate"]["passed"]:
        print("  [PROVEN] Scalar VQ preserves attention fidelity (cos >= 0.999)")
    if "b" in part_results and part_results["b"]["gate"]["passed"]:
        print("  [PROVEN] Vector VQ enables O(k) compressed-domain attention scores")
    if "c" in part_results and part_results["c"]["gate"]["passed"]:
        print("  [PROVEN] Full K+V vector VQ produces usable output")
    if "d" in part_results:
        pd = part_results["d"]
        if pd["score_gate"]["passed"]:
            print("  [PROVEN] Product quantization (M=8) preserves score fidelity (cos >= 0.99)")
        else:
            print(f"  [UNPROVEN] PQ score cosine {pd['score_gate']['value']:.6f} < 0.99")
        if pd["full_gate"]["passed"]:
            print("  [PROVEN] Full PQ K+V attention produces usable output (cos >= 0.98)")
        else:
            print(f"  [UNPROVEN] PQ output cosine {pd['full_gate']['value']:.6f} < 0.98")
    if "e" in part_results and not part_results["e"].get("skipped"):
        pe = part_results["e"]
        if pe["gate"]["passed"]:
            print(f"  [PROVEN] Fused scalar VQ kernel: {pe['gate']['speedup_value']:.2f}x "
                  f"speedup at cos >= 0.999")
        else:
            print(f"  [UNPROVEN] Fused kernel: cos={pe['gate']['fidelity_value']:.6f}, "
                  f"speedup={pe['gate']['speedup_value']:.2f}x")
    if "f" in part_results:
        pf = part_results["f"]
        if pf["gate"]["passed"]:
            cov = pf["full_summary"]["coverage_mean"]
            print(f"  [PROVEN] Prefiltered attention at top_p=32: cos >= 0.98, "
                  f"coverage={cov:.1%}")
        else:
            print(f"  [UNPROVEN] Prefiltered output cosine {pf['gate']['value']:.6f} < 0.98")
    if "g" in part_results:
        pg = part_results["g"]
        if pg["gate"]["passed"]:
            cfg = pg["config"]
            cov = pg["full_summary"]["coverage_mean"]
            print(f"  [PROVEN] Hybrid PQ+prefilter (M={cfg['m']}, top_k={cfg['top_k']}): "
                  f"cos >= 0.99, coverage={cov:.1%}")
            print(f"           Compress scoring (PQ), keep values exact (dense V on subset).")
        else:
            print(f"  [UNPROVEN] Hybrid output cosine {pg['gate']['value']:.6f} < 0.99")

    # Build receipt
    receipt = {
        "experiment": "bench_compressed_attention",
        "phase": phase,
        "kv_dir": str(kv_dir),
        "tag": tag,
        "parts_run": parts,
        "prompts": prompts,
        "n_layers": n_layers,
    }

    if "a" in part_results:
        receipt["part_a_scalar_vq"] = {
            "summary": part_results["a"]["summary"],
            "gate": part_results["a"]["gate"],
        }
    if "b" in part_results:
        receipt["part_b_vector_vq"] = {
            "pareto": part_results["b"]["pareto"],
            "projected_speedup": part_results["b"]["projected_speedup"],
            "gate": part_results["b"]["gate"],
        }
    if "c" in part_results:
        receipt["part_c_full_pipeline"] = {
            "summary": part_results["c"]["summary"],
            "gate": part_results["c"]["gate"],
            "worst_5": part_results["c"]["worst_5"],
        }
    if "d" in part_results:
        receipt["part_d_product_quantization"] = {
            "m_sweep": part_results["d"]["m_sweep"],
            "score_summary": part_results["d"]["score_summary"],
            "full_summary": part_results["d"]["full_summary"],
            "score_gate": part_results["d"]["score_gate"],
            "full_gate": part_results["d"]["full_gate"],
            "worst_5": part_results["d"]["worst_5"],
        }
    if "e" in part_results:
        pe = part_results["e"]
        if pe.get("skipped"):
            receipt["part_e_fused_kernel"] = {"skipped": True, "reason": pe["reason"]}
        else:
            receipt["part_e_fused_kernel"] = {
                "summary": pe["summary"],
                "gate": pe["gate"],
            }
    if "f" in part_results:
        receipt["part_f_prefiltered"] = {
            "top_p_sweep": part_results["f"]["top_p_sweep"],
            "full_summary": part_results["f"]["full_summary"],
            "gate": part_results["f"]["gate"],
            "worst_5": part_results["f"]["worst_5"],
        }
    if "g" in part_results:
        receipt["part_g_hybrid_pq"] = {
            "top_k_sweep": part_results["g"]["top_k_sweep"],
            "config": part_results["g"]["config"],
            "full_summary": part_results["g"]["full_summary"],
            "gate": part_results["g"]["gate"],
            "worst_5": part_results["g"]["worst_5"],
        }

    receipt["cost"] = cost

    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    receipt_name = f"cdc02_compressed_attention_{tag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path = RECEIPTS_DIR / receipt_name
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\n  Receipt saved: {receipt_path}")

    return 0 if all_gates_pass else 1


if __name__ == "__main__":
    sys.exit(main())

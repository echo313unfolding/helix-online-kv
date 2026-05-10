#!/usr/bin/env python3
"""WO-KVCACHE-POLARQUANT-01: PolarQuant rotation benchmark.

Compares raw scalar VQ vs PolarQuant-rotated VQ on real KV cache tensors.
Measures MSE improvement, cosine similarity, and attention score agreement.

Gate: PolarQuant must improve MSE by >10% at same k to justify complexity.

Usage:
    python3 tools/bench_polarquant.py --model ~/models/tinyllama-1.1b-chat-v1.0
    python3 tools/bench_polarquant.py --kv-dump  # Use pre-dumped KV tensors
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helix_online_kv.config import OnlineKVConfig
from helix_online_kv.layer_state import KVLayerState

KV_DUMP_DIR = Path(os.path.expanduser("~/helix-substrate/kv_dump"))
RECEIPTS_DIR = Path(__file__).resolve().parent.parent / "receipts" / "kv_cache"

N_HEADS = 4
HEAD_DIM = 64
ENTRY_SIZE = N_HEADS * HEAD_DIM
CALIBRATION_TOKENS = 128


def load_kv(prompt: str, layer: int, kv_type: str) -> np.ndarray:
    """Load real KV dump: [seq_len, entry_size]."""
    path = KV_DUMP_DIR / prompt / f"layer_{layer}_{kv_type}.npy"
    data = np.load(path)
    batch, n_heads, seq_len, head_dim = data.shape
    return data[0].transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.ravel(), b.ravel()) / (
        np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12
    ))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.ravel() - b.ravel()) ** 2))


def run_layer_comparison(k_data: np.ndarray, v_data: np.ndarray,
                         layer_idx: int, n_clusters: int = 256) -> dict:
    """Compare raw vs PolarQuant VQ on one layer's KV data."""
    seq_len = k_data.shape[0]
    cal = min(CALIBRATION_TOKENS, seq_len // 2)
    test_start = cal
    test_end = seq_len
    n_test = test_end - test_start

    if n_test < 10:
        return {"layer": layer_idx, "skipped": True, "reason": "too few tokens"}

    # --- Raw VQ (baseline) ---
    config_raw = OnlineKVConfig(
        calibration_tokens=cal, n_clusters=n_clusters, exact_layers=[],
    )
    ls_raw = KVLayerState(layer_idx, config_raw)

    for t in range(cal):
        ls_raw.feed_token(k_data[t], v_data[t])
    for t in range(test_start, test_end):
        ls_raw.feed_token(k_data[t], v_data[t])

    raw_k_decoded = ls_raw.get_all_compressed_k()
    raw_v_decoded = ls_raw.get_all_compressed_v()

    # --- PolarQuant VQ ---
    config_polar = OnlineKVConfig(
        calibration_tokens=cal, n_clusters=n_clusters, exact_layers=[],
        polar_rotation=True, polar_seed=42, n_heads=N_HEADS,
    )
    ls_polar = KVLayerState(layer_idx, config_polar)

    for t in range(cal):
        ls_polar.feed_token(k_data[t], v_data[t])
    for t in range(test_start, test_end):
        ls_polar.feed_token(k_data[t], v_data[t])

    polar_k_decoded = ls_polar.get_all_compressed_k()
    polar_v_decoded = ls_polar.get_all_compressed_v()

    # --- Metrics vs FP32 originals ---
    k_orig = k_data[test_start:test_end]
    v_orig = v_data[test_start:test_end]

    raw_k_mse = mse(k_orig, raw_k_decoded)
    raw_v_mse = mse(v_orig, raw_v_decoded)
    polar_k_mse = mse(k_orig, polar_k_decoded)
    polar_v_mse = mse(v_orig, polar_v_decoded)

    raw_k_cos = cosine_sim(k_orig, raw_k_decoded)
    raw_v_cos = cosine_sim(v_orig, raw_v_decoded)
    polar_k_cos = cosine_sim(k_orig, polar_k_decoded)
    polar_v_cos = cosine_sim(v_orig, polar_v_decoded)

    k_mse_improvement = (raw_k_mse - polar_k_mse) / raw_k_mse * 100 if raw_k_mse > 0 else 0
    v_mse_improvement = (raw_v_mse - polar_v_mse) / raw_v_mse * 100 if raw_v_mse > 0 else 0

    return {
        "layer": layer_idx,
        "n_clusters": n_clusters,
        "n_test_tokens": n_test,
        "raw": {
            "k_mse": raw_k_mse, "v_mse": raw_v_mse,
            "k_cos": raw_k_cos, "v_cos": raw_v_cos,
        },
        "polar": {
            "k_mse": polar_k_mse, "v_mse": polar_v_mse,
            "k_cos": polar_k_cos, "v_cos": polar_v_cos,
        },
        "improvement": {
            "k_mse_pct": round(k_mse_improvement, 2),
            "v_mse_pct": round(v_mse_improvement, 2),
        },
    }


def attention_fidelity(k_data: np.ndarray, v_data: np.ndarray,
                       layer_idx: int, n_clusters: int = 256) -> dict:
    """Compare attention output: FP32 vs raw-VQ vs polar-VQ.

    Catches failure modes where good MSE doesn't translate to good attention
    (the QJL problem: softmax amplifies quantization error).
    """
    seq_len = k_data.shape[0]
    cal = min(CALIBRATION_TOKENS, seq_len // 2)
    test_start = cal
    n_test = seq_len - test_start

    if n_test < 32:
        return {"layer": layer_idx, "skipped": True, "reason": "too few tokens for attention test"}

    # Build decoded arrays for both methods
    k_orig = k_data[test_start:]
    v_orig = v_data[test_start:]

    # Raw VQ
    config_raw = OnlineKVConfig(
        calibration_tokens=cal, n_clusters=n_clusters, exact_layers=[],
    )
    ls_raw = KVLayerState(layer_idx, config_raw)
    for t in range(cal):
        ls_raw.feed_token(k_data[t], v_data[t])
    for t in range(test_start, seq_len):
        ls_raw.feed_token(k_data[t], v_data[t])
    raw_k = ls_raw.get_all_compressed_k()
    raw_v = ls_raw.get_all_compressed_v()

    # Polar VQ
    config_polar = OnlineKVConfig(
        calibration_tokens=cal, n_clusters=n_clusters, exact_layers=[],
        polar_rotation=True, polar_seed=42, n_heads=N_HEADS,
    )
    ls_polar = KVLayerState(layer_idx, config_polar)
    for t in range(cal):
        ls_polar.feed_token(k_data[t], v_data[t])
    for t in range(test_start, seq_len):
        ls_polar.feed_token(k_data[t], v_data[t])
    polar_k = ls_polar.get_all_compressed_k()
    polar_v = ls_polar.get_all_compressed_v()

    # Simulate attention: pick last token as query, all others as context
    # Reshape to per-head: [n_test, n_heads, head_dim]
    def reshape_heads(x):
        return x.reshape(x.shape[0], N_HEADS, HEAD_DIM)

    k_orig_h = reshape_heads(k_orig)
    v_orig_h = reshape_heads(v_orig)
    raw_k_h = reshape_heads(raw_k)
    raw_v_h = reshape_heads(raw_v)
    polar_k_h = reshape_heads(polar_k)
    polar_v_h = reshape_heads(polar_v)

    # Use last token's K as query (simplified attention proxy)
    q = k_orig_h[-1]  # [n_heads, head_dim]
    context_k_orig = k_orig_h[:-1]  # [n_ctx, n_heads, head_dim]
    context_v_orig = v_orig_h[:-1]

    top_k_agreement = {"raw": [], "polar": []}

    for head in range(N_HEADS):
        q_h = q[head]  # [head_dim]
        # Attention scores: q @ K^T / sqrt(d)
        scale = 1.0 / np.sqrt(HEAD_DIM)
        scores_fp32 = (context_k_orig[:, head, :] @ q_h) * scale
        scores_raw = (raw_k_h[:-1, head, :] @ q_h) * scale
        scores_polar = (polar_k_h[:-1, head, :] @ q_h) * scale

        # Softmax
        def softmax(x):
            x = x - x.max()
            e = np.exp(x)
            return e / e.sum()

        attn_fp32 = softmax(scores_fp32)
        attn_raw = softmax(scores_raw)
        attn_polar = softmax(scores_polar)

        # Top-k agreement (k=16)
        top_n = min(16, len(attn_fp32))
        fp32_topk = set(np.argsort(attn_fp32)[-top_n:])
        raw_topk = set(np.argsort(attn_raw)[-top_n:])
        polar_topk = set(np.argsort(attn_polar)[-top_n:])

        raw_agree = len(fp32_topk & raw_topk) / top_n
        polar_agree = len(fp32_topk & polar_topk) / top_n
        top_k_agreement["raw"].append(raw_agree)
        top_k_agreement["polar"].append(polar_agree)

    return {
        "layer": layer_idx,
        "n_context_tokens": n_test - 1,
        "top16_agreement": {
            "raw_mean": round(float(np.mean(top_k_agreement["raw"])), 4),
            "polar_mean": round(float(np.mean(top_k_agreement["polar"])), 4),
            "raw_per_head": [round(x, 4) for x in top_k_agreement["raw"]],
            "polar_per_head": [round(x, 4) for x in top_k_agreement["polar"]],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="PolarQuant rotation benchmark")
    parser.add_argument("--kv-dump", action="store_true", help="Use pre-dumped KV tensors")
    parser.add_argument("--model", type=str, default=None, help="Model path (requires transformers)")
    parser.add_argument("--prompt", type=str, default="science", help="KV dump prompt name")
    parser.add_argument("--layers", type=str, default="1,5,10,15,20",
                        help="Comma-separated layer indices to test")
    parser.add_argument("--clusters", type=str, default="256,64,32",
                        help="Comma-separated cluster counts to test")
    args = parser.parse_args()

    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    t_start = time.time()
    cpu_start = time.process_time()

    layers = [int(x) for x in args.layers.split(",")]
    cluster_sizes = [int(x) for x in args.clusters.split(",")]

    if not KV_DUMP_DIR.exists():
        print(f"ERROR: KV dump directory not found: {KV_DUMP_DIR}")
        print("Run tools/dump_kv_cache_long.py first, or pass --model")
        sys.exit(1)

    prompt_dir = KV_DUMP_DIR / args.prompt
    if not prompt_dir.exists():
        available = [d.name for d in KV_DUMP_DIR.iterdir() if d.is_dir()]
        print(f"ERROR: Prompt '{args.prompt}' not found. Available: {available}")
        sys.exit(1)

    print(f"=== WO-KVCACHE-POLARQUANT-01: PolarQuant Benchmark ===")
    print(f"Prompt: {args.prompt}")
    print(f"Layers: {layers}")
    print(f"Cluster sizes: {cluster_sizes}")
    print()

    all_results = []
    all_attention = []

    for layer in layers:
        k_path = KV_DUMP_DIR / args.prompt / f"layer_{layer}_k.npy"
        v_path = KV_DUMP_DIR / args.prompt / f"layer_{layer}_v.npy"
        if not k_path.exists() or not v_path.exists():
            print(f"  Layer {layer}: SKIP (no dump)")
            continue

        k_data = load_kv(args.prompt, layer, "k")
        v_data = load_kv(args.prompt, layer, "v")
        print(f"Layer {layer}: {k_data.shape[0]} tokens, entry_size={k_data.shape[1]}")

        for n_clusters in cluster_sizes:
            result = run_layer_comparison(k_data, v_data, layer, n_clusters)
            all_results.append(result)

            if result.get("skipped"):
                print(f"  k={n_clusters}: SKIPPED ({result['reason']})")
                continue

            imp = result["improvement"]
            raw = result["raw"]
            polar = result["polar"]
            gate = "PASS" if imp["k_mse_pct"] > 10 else "FAIL"

            print(f"  k={n_clusters}: K MSE {raw['k_mse']:.6f} → {polar['k_mse']:.6f} "
                  f"({imp['k_mse_pct']:+.1f}%) [{gate}]")
            print(f"          V MSE {raw['v_mse']:.6f} → {polar['v_mse']:.6f} "
                  f"({imp['v_mse_pct']:+.1f}%)")
            print(f"          K cos {raw['k_cos']:.6f} → {polar['k_cos']:.6f}")

        # Attention fidelity at k=256
        attn = attention_fidelity(k_data, v_data, layer, 256)
        all_attention.append(attn)
        if not attn.get("skipped"):
            ta = attn["top16_agreement"]
            print(f"  Attention top-16 agreement: raw={ta['raw_mean']:.4f} polar={ta['polar_mean']:.4f}")
        print()

    # Summary
    print("=== SUMMARY ===")
    k256_results = [r for r in all_results if r.get("n_clusters") == 256 and not r.get("skipped")]
    if k256_results:
        avg_k_imp = np.mean([r["improvement"]["k_mse_pct"] for r in k256_results])
        avg_v_imp = np.mean([r["improvement"]["v_mse_pct"] for r in k256_results])
        gate_pass = sum(1 for r in k256_results if r["improvement"]["k_mse_pct"] > 10)
        print(f"k=256: avg K MSE improvement = {avg_k_imp:+.1f}%, V = {avg_v_imp:+.1f}%")
        print(f"Gate (>10% improvement): {gate_pass}/{len(k256_results)} layers pass")

    # Write receipt
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "work_order": "WO-KVCACHE-POLARQUANT-01",
        "description": "PolarQuant rotation benchmark: raw vs rotated scalar VQ on real KV tensors",
        "prompt": args.prompt,
        "layers_tested": layers,
        "cluster_sizes": cluster_sizes,
        "results": all_results,
        "attention_fidelity": all_attention,
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }

    receipt_path = RECEIPTS_DIR / "polarquant_probe.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {receipt_path}")


if __name__ == "__main__":
    main()

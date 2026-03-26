#!/usr/bin/env python3
"""CDC-03 closing experiment: prove hybrid PQ at top_k=128 on layers 1-21.

Layer 0 is routed to exact attention (exact_layers=[0]).
This proves the hybrid works for the other 21 layers at 12.5% coverage.

Receipt with WO-RECEIPT-COST-01 cost block.
"""

import json
import os
import sys
import time
import resource
import platform
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helix_online_kv.product_codebook import ProductCodebook
from helix_online_kv.compressed_attention import (
    standard_attention,
    hybrid_pq_attention,
)

RECEIPTS_DIR = Path(__file__).resolve().parent.parent / "receipts"
N_HEADS = 4
HEAD_DIM = 64
M = 16
TOP_K = 128
SINK_TOKENS = 4


def cosine_sim(a, b):
    a, b = a.ravel(), b.ravel()
    d = float(np.dot(a, b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return d / (na * nb)


def main():
    kv_dir = Path(os.path.expanduser("~/helix-substrate/kv_dump_long"))
    assert kv_dir.exists(), f"KV dump not found: {kv_dir}"

    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    t_start = time.time()
    cpu_start = time.process_time()

    # Auto-detect
    prompts = sorted(
        d.name for d in kv_dir.iterdir()
        if d.is_dir() and list(d.glob("layer_*_k.npy"))
    )
    n_layers = len(list((kv_dir / prompts[0]).glob("layer_*_k.npy")))
    scale = 1.0 / np.sqrt(HEAD_DIM)

    print(f"CDC-03 Closing Experiment: Hybrid PQ on layers 1-{n_layers-1}")
    print(f"Config: M={M}, top_k={TOP_K}, sink_tokens={SINK_TOKENS}")
    print(f"Prompts: {prompts}, Layers: 1-{n_layers-1} (skip 0), Heads: {N_HEADS}")
    print()

    results = []

    for prompt in prompts:
        for layer_idx in range(1, n_layers):  # SKIP LAYER 0
            k_path = kv_dir / prompt / f"layer_{layer_idx}_k.npy"
            v_path = kv_dir / prompt / f"layer_{layer_idx}_v.npy"
            K_all = np.load(k_path)[0].astype(np.float32)
            V_all = np.load(v_path)[0].astype(np.float32)

            for head in range(N_HEADS):
                K = K_all[head]
                V = V_all[head]
                seq_len = K.shape[0]

                rng = np.random.default_rng(layer_idx * N_HEADS + head + 6000)
                q = rng.standard_normal(HEAD_DIM).astype(np.float32)
                q = q * K.std() + K.mean()

                # Reference
                _, out_ref = standard_attention(q, K, V, scale)

                # Hybrid PQ
                pq = ProductCodebook(
                    n_subspaces=M, sub_clusters=256, head_dim=HEAD_DIM
                )
                pq.feed_calibration(K)
                pq.finalize_calibration()
                idx = pq.assign(K)

                _, out_hybrid, meta = hybrid_pq_attention(
                    q, K, V, idx, pq, scale,
                    top_k=TOP_K, sink_tokens=SINK_TOKENS
                )

                output_cos = cosine_sim(out_ref, out_hybrid)

                results.append({
                    "prompt": prompt,
                    "layer": layer_idx,
                    "head": head,
                    "seq_len": seq_len,
                    "output_cosine": round(output_cos, 8),
                    "coverage": round(meta["coverage"], 4),
                    "selected": meta["selected_count"],
                })

            # Progress
            if layer_idx % 5 == 0:
                recent = results[-N_HEADS:]
                cos_vals = [r["output_cosine"] for r in recent]
                print(f"  {prompt} L{layer_idx}: cos={np.mean(cos_vals):.8f}")

    # Summary
    output_cosines = [r["output_cosine"] for r in results]
    coverages = [r["coverage"] for r in results]

    cos_mean = float(np.mean(output_cosines))
    cos_min = float(np.min(output_cosines))
    cos_max = float(np.max(output_cosines))
    cos_std = float(np.std(output_cosines))

    print(f"\n{'='*70}")
    print(f"CDC-03 CLOSING EXPERIMENT RESULTS")
    print(f"{'='*70}")
    print(f"  Layers tested: 1-{n_layers-1} ({n_layers-1} layers)")
    print(f"  Total entries: {len(results)}")
    print(f"  Config: M={M}, top_k={TOP_K}, sink={SINK_TOKENS}")
    print(f"  Coverage: {np.mean(coverages):.4f} ({TOP_K}/{results[0]['seq_len']} tokens)")
    print(f"\n  Output cosine:")
    print(f"    mean = {cos_mean:.8f}")
    print(f"    min  = {cos_min:.8f}")
    print(f"    max  = {cos_max:.8f}")
    print(f"    std  = {cos_std:.8f}")

    # Gates
    gate_mean = cos_mean >= 0.99
    gate_min = cos_min >= 0.95
    overall = gate_mean and gate_min

    print(f"\n  GATE (mean >= 0.99):  {'PASS' if gate_mean else 'FAIL'} ({cos_mean:.8f})")
    print(f"  GATE (min >= 0.95):   {'PASS' if gate_min else 'FAIL'} ({cos_min:.8f})")
    print(f"  OVERALL:              {'PASS' if overall else 'FAIL'}")

    # Worst 10
    results_sorted = sorted(results, key=lambda r: r["output_cosine"])
    print(f"\n  Worst 10:")
    for r in results_sorted[:10]:
        print(f"    {r['prompt']} L{r['layer']:2d} H{r['head']}: "
              f"cos={r['output_cosine']:.8f}  sel={r['selected']}")

    # Distribution
    bins = [0, 0.90, 0.95, 0.99, 0.999, 0.9999, 1.0001]
    hist, _ = np.histogram(output_cosines, bins=bins)
    print(f"\n  Distribution:")
    for i in range(len(hist)):
        pct = hist[i] / len(results) * 100
        print(f"    [{bins[i]:.4f}, {bins[i+1]:.4f}): {hist[i]:4d} ({pct:.1f}%)")

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

    print(f"\n  Wall time: {cost['wall_time_s']:.1f}s")

    # Receipt
    receipt = {
        "experiment": "cdc03_hybrid_layers1_21",
        "config": {
            "m": M,
            "top_k": TOP_K,
            "sink_tokens": SINK_TOKENS,
            "layers_tested": list(range(1, n_layers)),
            "layer_0_excluded": True,
            "reason": "Layer 0 routed to exact attention (attention sink pattern)",
        },
        "kv_dir": str(kv_dir),
        "prompts": prompts,
        "n_layers_tested": n_layers - 1,
        "n_entries": len(results),
        "summary": {
            "output_cosine_mean": round(cos_mean, 8),
            "output_cosine_min": round(cos_min, 8),
            "output_cosine_max": round(cos_max, 8),
            "output_cosine_std": round(cos_std, 8),
            "coverage_mean": round(float(np.mean(coverages)), 4),
        },
        "gates": {
            "mean_cosine_099": {"value": round(cos_mean, 8), "passed": gate_mean},
            "min_cosine_095": {"value": round(cos_min, 8), "passed": gate_min},
            "overall": overall,
        },
        "worst_10": results_sorted[:10],
        "distribution": {
            f"[{bins[i]:.4f},{bins[i+1]:.4f})": int(hist[i])
            for i in range(len(hist))
        },
        "cost": cost,
        "honest_claim": (
            f"Hybrid PQ attention (M={M}, top_k={TOP_K}) on layers 1-{n_layers-1} "
            f"achieves mean output cosine {cos_mean:.6f} (min {cos_min:.6f}) "
            f"at {np.mean(coverages):.1%} coverage on real WikiText-2 KV data. "
            f"Layer 0 excluded (exact_layers=[0]). "
            f"V matmul is truly sparse: only selected tokens touched."
        ),
    }

    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%dT%H%M%S')
    receipt_path = RECEIPTS_DIR / f"cdc03_hybrid_layers1_21_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\n  Receipt saved: {receipt_path}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Phase 1 Experiment: Codebook Calibration Generalization.

Proves that a codebook fitted on the first N tokens of a KV cache
generalizes to all subsequent tokens.

Uses REAL KV dumps from ~/helix-substrate/kv_dump/ (science/history/cooking,
3 prompts, 22 layers each).

Sweeps:
  - Calibration window N = {32, 64, 128, 256}
  - Per-layer cosine similarity on held-out tokens
  - Cross-prompt test: fit on science, test on history/cooking
  - Per-layer kurtosis correlation with generalization quality

Gate: mean cosine >= 0.998 at N=128 across all 22 layers.
"""

import json
import os
import sys
import time
import resource
import platform
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helix_online_kv.codebook import OnlineCodebook

KV_DUMP_DIR = Path(os.path.expanduser("~/helix-substrate/kv_dump"))
RECEIPTS_DIR = Path(__file__).resolve().parent.parent / "receipts"
PROMPTS = ["science", "history", "cooking"]
N_LAYERS = 22
KV_TYPES = ["k", "v"]
CALIBRATION_WINDOWS = [32, 64, 128, 256]
N_CLUSTERS = 256

# Load kv_stats for kurtosis values
STATS_PATH = KV_DUMP_DIR / "kv_stats.json"


def load_kv(prompt: str, layer: int, kv_type: str) -> np.ndarray:
    """Load a KV dump as flat float32 per token.

    Returns: [seq_len, n_heads * head_dim] float32
    """
    path = KV_DUMP_DIR / prompt / f"layer_{layer}_{kv_type}.npy"
    data = np.load(path)  # [1, n_heads, seq_len, head_dim]
    batch, n_heads, seq_len, head_dim = data.shape
    # Reshape to [seq_len, n_heads * head_dim]
    return data[0].transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel()
    b_flat = b.ravel()
    dot = float(np.dot(a_flat, b_flat))
    na = float(np.linalg.norm(a_flat))
    nb = float(np.linalg.norm(b_flat))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return dot / (na * nb)


def run_within_prompt_sweep():
    """Sweep calibration windows: fit on first N tokens, test on rest."""
    print("=" * 70)
    print("WITHIN-PROMPT GENERALIZATION SWEEP")
    print("=" * 70)

    results = {}

    for prompt in PROMPTS:
        results[prompt] = {}
        for N in CALIBRATION_WINDOWS:
            results[prompt][N] = []
            for layer in range(N_LAYERS):
                for kv in KV_TYPES:
                    data = load_kv(prompt, layer, kv)
                    seq_len = data.shape[0]

                    if N >= seq_len:
                        # Not enough tokens for this window
                        results[prompt][N].append({
                            "layer": layer, "kv": kv,
                            "cosine": None, "mse": None,
                            "reason": f"seq_len={seq_len} < N={N}",
                        })
                        continue

                    # Fit codebook on first N tokens (flatten to scalar)
                    cal_data = data[:N].ravel()
                    test_data = data[N:]

                    cb = OnlineCodebook(n_clusters=N_CLUSTERS)
                    cb.feed_calibration(cal_data)
                    cb.finalize_calibration()

                    # Test on held-out tokens
                    test_flat = test_data.ravel()
                    idx = cb.assign(test_flat)
                    cos = cb.cosine_similarity(test_flat, idx)
                    mse = cb.quantization_error(test_flat, idx)

                    results[prompt][N].append({
                        "layer": layer, "kv": kv,
                        "cosine": round(cos, 6),
                        "mse": round(mse, 8),
                        "cal_tokens": N,
                        "test_tokens": seq_len - N,
                    })

            # Print summary for this window
            valid = [r for r in results[prompt][N] if r["cosine"] is not None]
            if valid:
                cosines = [r["cosine"] for r in valid]
                mean_cos = np.mean(cosines)
                min_cos = np.min(cosines)
                min_entry = min(valid, key=lambda r: r["cosine"])
                print(f"  {prompt} N={N:3d}: mean_cos={mean_cos:.6f}  "
                      f"min_cos={min_cos:.6f} (layer {min_entry['layer']} {min_entry['kv']})  "
                      f"n={len(valid)}")

    return results


def run_cross_prompt_test():
    """Fit codebook on science, test on history/cooking."""
    print("\n" + "=" * 70)
    print("CROSS-PROMPT GENERALIZATION (fit on science, test on history/cooking)")
    print("=" * 70)

    N = 128  # Calibration window
    results = {}

    for test_prompt in ["history", "cooking"]:
        results[test_prompt] = []
        for layer in range(N_LAYERS):
            for kv in KV_TYPES:
                # Fit on science
                train = load_kv("science", layer, kv)
                if N > train.shape[0]:
                    continue
                cal_data = train[:N].ravel()

                cb = OnlineCodebook(n_clusters=N_CLUSTERS)
                cb.feed_calibration(cal_data)
                cb.finalize_calibration()

                # Test on different prompt
                test = load_kv(test_prompt, layer, kv)
                test_flat = test.ravel()
                idx = cb.assign(test_flat)
                cos = cb.cosine_similarity(test_flat, idx)
                mse = cb.quantization_error(test_flat, idx)

                results[test_prompt].append({
                    "layer": layer, "kv": kv,
                    "cosine": round(cos, 6),
                    "mse": round(mse, 8),
                    "train_prompt": "science",
                    "test_prompt": test_prompt,
                })

        valid = results[test_prompt]
        if valid:
            cosines = [r["cosine"] for r in valid]
            mean_cos = np.mean(cosines)
            min_cos = np.min(cosines)
            min_entry = min(valid, key=lambda r: r["cosine"])
            print(f"  science -> {test_prompt}: mean_cos={mean_cos:.6f}  "
                  f"min_cos={min_cos:.6f} (layer {min_entry['layer']} {min_entry['kv']})")

    return results


def run_kurtosis_correlation():
    """Correlate per-layer kurtosis with generalization quality."""
    print("\n" + "=" * 70)
    print("KURTOSIS vs GENERALIZATION QUALITY")
    print("=" * 70)

    with open(STATS_PATH) as f:
        kv_stats = json.load(f)

    N = 128
    prompt = "science"

    kurtosis_vals = []
    cosine_vals = []
    rows = []

    for layer_stat in kv_stats[prompt]["layers"]:
        layer = layer_stat["layer"]
        for kv in KV_TYPES:
            kurt = layer_stat[f"{kv}_kurtosis"]
            data = load_kv(prompt, layer, kv)

            if N >= data.shape[0]:
                continue

            cb = OnlineCodebook(n_clusters=N_CLUSTERS)
            cb.feed_calibration(data[:N].ravel())
            cb.finalize_calibration()

            test_flat = data[N:].ravel()
            idx = cb.assign(test_flat)
            cos = cb.cosine_similarity(test_flat, idx)

            kurtosis_vals.append(kurt)
            cosine_vals.append(cos)
            rows.append({
                "layer": layer, "kv": kv,
                "kurtosis": round(kurt, 2),
                "cosine": round(cos, 6),
            })

    # Spearman rank correlation
    from scipy.stats import spearmanr
    try:
        rho, pval = spearmanr(kurtosis_vals, cosine_vals)
        print(f"  Spearman rho = {rho:.4f}, p = {pval:.2e}")
    except ImportError:
        # Fallback: manual rank correlation
        rho, pval = None, None
        print("  (scipy not available — skipping Spearman)")

    # Print worst layers
    rows.sort(key=lambda r: r["cosine"])
    print("\n  Worst 5 by cosine:")
    for r in rows[:5]:
        flag = " *** HIGH KURTOSIS" if r["kurtosis"] > 20 else ""
        print(f"    layer {r['layer']:2d} {r['kv']}: cos={r['cosine']:.6f}  "
              f"kurt={r['kurtosis']:7.2f}{flag}")

    print("\n  Best 5 by cosine:")
    for r in rows[-5:]:
        print(f"    layer {r['layer']:2d} {r['kv']}: cos={r['cosine']:.6f}  "
              f"kurt={r['kurtosis']:7.2f}")

    return rows, rho, pval


def main():
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    t_start = time.time()
    cpu_start = time.process_time()

    print(f"KV dump dir: {KV_DUMP_DIR}")
    print(f"Prompts: {PROMPTS}")
    print(f"Calibration windows: {CALIBRATION_WINDOWS}")
    print(f"Clusters: {N_CLUSTERS}")
    print()

    # Verify data exists
    for p in PROMPTS:
        assert (KV_DUMP_DIR / p).exists(), f"Missing KV dump: {KV_DUMP_DIR / p}"

    # Run experiments
    within_results = run_within_prompt_sweep()
    cross_results = run_cross_prompt_test()
    kurtosis_rows, rho, pval = run_kurtosis_correlation()

    # Check gate: mean cosine >= 0.998 at N=128
    print("\n" + "=" * 70)
    print("GATE CHECK: mean cosine >= 0.998 at N=128")
    print("=" * 70)

    gate_passed = True
    for prompt in PROMPTS:
        valid = [r for r in within_results[prompt][128] if r["cosine"] is not None]
        if valid:
            mean_cos = np.mean([r["cosine"] for r in valid])
            passed = mean_cos >= 0.998
            status = "PASS" if passed else "FAIL"
            print(f"  {prompt}: mean_cos={mean_cos:.6f} [{status}]")
            if not passed:
                gate_passed = False

    print(f"\n  Overall gate: {'PASS' if gate_passed else 'FAIL'}")

    # Build receipt
    cost = {
        'wall_time_s': round(time.time() - t_start, 3),
        'cpu_time_s': round(time.process_time() - cpu_start, 3),
        'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
        'timestamp_start': start_iso,
        'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    receipt = {
        "experiment": "calibrate_codebook_generalization",
        "phase": 1,
        "gate": {
            "criterion": "mean_cosine >= 0.998 at N=128 across all 22 layers",
            "passed": gate_passed,
        },
        "within_prompt": {},
        "cross_prompt": {},
        "kurtosis_correlation": {
            "spearman_rho": rho,
            "spearman_pval": pval,
        },
        "cost": cost,
    }

    # Summarize within-prompt results
    for prompt in PROMPTS:
        receipt["within_prompt"][prompt] = {}
        for N in CALIBRATION_WINDOWS:
            valid = [r for r in within_results[prompt][N] if r["cosine"] is not None]
            if valid:
                cosines = [r["cosine"] for r in valid]
                receipt["within_prompt"][prompt][str(N)] = {
                    "mean_cosine": round(float(np.mean(cosines)), 6),
                    "min_cosine": round(float(np.min(cosines)), 6),
                    "max_cosine": round(float(np.max(cosines)), 6),
                    "n_entries": len(valid),
                }

    # Summarize cross-prompt results
    for test_prompt, entries in cross_results.items():
        if entries:
            cosines = [r["cosine"] for r in entries]
            receipt["cross_prompt"][test_prompt] = {
                "mean_cosine": round(float(np.mean(cosines)), 6),
                "min_cosine": round(float(np.min(cosines)), 6),
                "n_entries": len(entries),
            }

    # Worst layers by kurtosis
    receipt["worst_layers"] = kurtosis_rows[:5]

    # Save receipt
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = RECEIPTS_DIR / f"codebook_generalization_{time.strftime('%Y%m%dT%H%M%S')}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt saved: {receipt_path}")

    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())

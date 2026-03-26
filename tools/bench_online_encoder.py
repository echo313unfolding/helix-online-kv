#!/usr/bin/env python3
"""Phase 2 Benchmark: Online Encoder Latency.

Measures per-token overhead of OnlineCodebook assignment on real KV data.
Target: < 5ms per-token overhead at seq_len=1024.

Uses real KV dumps, simulates token-by-token streaming through KVLayerState.
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
from helix_online_kv.config import OnlineKVConfig
from helix_online_kv.layer_state import KVLayerState

KV_DUMP_DIR = Path(os.path.expanduser("~/helix-substrate/kv_dump"))
RECEIPTS_DIR = Path(__file__).resolve().parent.parent / "receipts"
N_LAYERS = 22
N_HEADS = 4
HEAD_DIM = 64
ENTRY_SIZE = N_HEADS * HEAD_DIM  # 256 floats per token per layer


def load_kv(prompt: str, layer: int, kv_type: str) -> np.ndarray:
    path = KV_DUMP_DIR / prompt / f"layer_{layer}_{kv_type}.npy"
    data = np.load(path)
    batch, n_heads, seq_len, head_dim = data.shape
    return data[0].transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim).astype(np.float32)


def bench_single_layer_streaming(prompt: str = "science", layer: int = 5):
    """Benchmark single-layer streaming latency."""
    k_data = load_kv(prompt, layer, "k")
    v_data = load_kv(prompt, layer, "v")
    seq_len = k_data.shape[0]

    config = OnlineKVConfig(
        calibration_tokens=128,
        n_clusters=256,
        exact_layers=[],
    )
    ls = KVLayerState(layer, config)

    # Phase 1: Calibration
    cal_times = []
    for t in range(min(128, seq_len)):
        t0 = time.perf_counter()
        ls.feed_token(k_data[t], v_data[t])
        cal_times.append(time.perf_counter() - t0)

    cal_finalize_time = cal_times[-1]  # Last one includes finalization

    # Phase 2: Streaming assignment
    stream_times = []
    for t in range(128, seq_len):
        t0 = time.perf_counter()
        ls.feed_token(k_data[t], v_data[t])
        stream_times.append(time.perf_counter() - t0)

    return {
        "layer": layer,
        "seq_len": seq_len,
        "calibration_tokens": 128,
        "streaming_tokens": len(stream_times),
        "cal_mean_us": round(np.mean(cal_times[:-1]) * 1e6, 1) if len(cal_times) > 1 else 0,
        "cal_finalize_us": round(cal_finalize_time * 1e6, 1),
        "stream_mean_us": round(np.mean(stream_times) * 1e6, 1) if stream_times else 0,
        "stream_p50_us": round(np.percentile(stream_times, 50) * 1e6, 1) if stream_times else 0,
        "stream_p99_us": round(np.percentile(stream_times, 99) * 1e6, 1) if stream_times else 0,
        "stream_max_us": round(np.max(stream_times) * 1e6, 1) if stream_times else 0,
    }


def bench_full_model_streaming(prompt: str = "science"):
    """Benchmark all 22 layers processing one token (simulates full forward pass)."""
    # Load all layers
    all_k = {}
    all_v = {}
    for layer in range(N_LAYERS):
        all_k[layer] = load_kv(prompt, layer, "k")
        all_v[layer] = load_kv(prompt, layer, "v")

    seq_len = all_k[0].shape[0]

    config = OnlineKVConfig(
        calibration_tokens=128,
        n_clusters=256,
        exact_layers=[0],  # Layer 0 exact (kurtosis=36.1 V)
    )
    states = [KVLayerState(i, config) for i in range(N_LAYERS)]

    # Calibration phase
    for t in range(min(128, seq_len)):
        for layer in range(N_LAYERS):
            states[layer].feed_token(all_k[layer][t], all_v[layer][t])

    # Streaming: measure per-token latency across ALL layers
    per_token_times = []
    for t in range(128, seq_len):
        t0 = time.perf_counter()
        for layer in range(N_LAYERS):
            states[layer].feed_token(all_k[layer][t], all_v[layer][t])
        per_token_times.append(time.perf_counter() - t0)

    return {
        "n_layers": N_LAYERS,
        "exact_layers": [0],
        "compressed_layers": 21,
        "seq_len": seq_len,
        "streaming_tokens": len(per_token_times),
        "per_token_mean_us": round(np.mean(per_token_times) * 1e6, 1) if per_token_times else 0,
        "per_token_p50_us": round(np.percentile(per_token_times, 50) * 1e6, 1) if per_token_times else 0,
        "per_token_p99_us": round(np.percentile(per_token_times, 99) * 1e6, 1) if per_token_times else 0,
        "per_token_max_us": round(np.max(per_token_times) * 1e6, 1) if per_token_times else 0,
        "per_token_mean_ms": round(np.mean(per_token_times) * 1e3, 3) if per_token_times else 0,
    }


def bench_assignment_microbenchmark():
    """Pure assignment latency: codebook.assign() on pre-calibrated codebook."""
    from helix_online_kv.codebook import OnlineCodebook

    # Calibrate on real data
    k_data = load_kv("science", 5, "k")
    cb = OnlineCodebook(n_clusters=256)
    cb.feed_calibration(k_data[:128].ravel())
    cb.finalize_calibration()

    # Microbenchmark: assign single token (256 floats)
    token = k_data[128].copy()
    times = []
    for _ in range(10000):
        t0 = time.perf_counter()
        cb.assign(token)
        times.append(time.perf_counter() - t0)

    return {
        "op": "codebook.assign",
        "input_size": len(token),
        "n_clusters": 256,
        "n_iterations": 10000,
        "mean_us": round(np.mean(times) * 1e6, 2),
        "p50_us": round(np.percentile(times, 50) * 1e6, 2),
        "p99_us": round(np.percentile(times, 99) * 1e6, 2),
        "min_us": round(np.min(times) * 1e6, 2),
    }


def main():
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    t_start = time.time()
    cpu_start = time.process_time()

    print("=" * 70)
    print("PHASE 2: ONLINE ENCODER LATENCY BENCHMARK")
    print("=" * 70)

    # 1. Assignment microbenchmark
    print("\n--- Assignment Microbenchmark (single token, 256 floats) ---")
    micro = bench_assignment_microbenchmark()
    print(f"  mean={micro['mean_us']:.1f} us  p50={micro['p50_us']:.1f} us  "
          f"p99={micro['p99_us']:.1f} us  (n={micro['n_iterations']})")

    # 2. Single layer streaming
    print("\n--- Single Layer Streaming (layer 5, science) ---")
    single = bench_single_layer_streaming("science", 5)
    print(f"  cal_mean={single['cal_mean_us']:.1f} us  "
          f"cal_finalize={single['cal_finalize_us']:.1f} us")
    print(f"  stream_mean={single['stream_mean_us']:.1f} us  "
          f"p50={single['stream_p50_us']:.1f} us  "
          f"p99={single['stream_p99_us']:.1f} us")

    # 3. Full model streaming (all 22 layers per token)
    print("\n--- Full Model Streaming (22 layers, science) ---")
    full = bench_full_model_streaming("science")
    print(f"  per_token_mean={full['per_token_mean_ms']:.3f} ms  "
          f"({full['per_token_mean_us']:.1f} us)")
    print(f"  p50={full['per_token_p50_us']:.1f} us  "
          f"p99={full['per_token_p99_us']:.1f} us  "
          f"max={full['per_token_max_us']:.1f} us")

    # Gate check
    print("\n--- GATE CHECK ---")
    gate_passed = bool(full["per_token_mean_ms"] < 5.0)
    status = "PASS" if gate_passed else "FAIL"
    print(f"  per-token overhead < 5ms: {full['per_token_mean_ms']:.3f} ms [{status}]")

    # Cost
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
        "experiment": "bench_online_encoder",
        "phase": 2,
        "gate": {
            "criterion": "per-token overhead < 5ms at all 22 layers",
            "passed": gate_passed,
        },
        "assignment_microbenchmark": micro,
        "single_layer_streaming": single,
        "full_model_streaming": full,
        "cost": cost,
    }

    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = RECEIPTS_DIR / f"online_encoder_{time.strftime('%Y%m%dT%H%M%S')}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt saved: {receipt_path}")

    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())

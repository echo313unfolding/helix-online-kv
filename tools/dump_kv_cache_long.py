#!/usr/bin/env python3
"""CDC-01.1 Phase A: Long-Sequence KV Cache Capture.

Dumps 1024-token KV caches from TinyLlama using REAL WikiText-2 text.
Three non-overlapping passages from different corpus positions.

Outputs:
  ~/helix-substrate/kv_dump_long/{wiki_a,wiki_b,wiki_c}/layer_{i}_{k,v}.npy
  ~/helix-substrate/kv_dump_long/kv_stats.json
  ~/helix-substrate/kv_dump_long/receipt.json  (WO-RECEIPT-COST-01)
"""

import json
import os
import sys
import time
import resource
import platform
from pathlib import Path

import numpy as np
import torch

# Path setup: helix-online-kv (parent) + helix-substrate
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(os.path.expanduser("~/helix-substrate"))))

from helix_substrate.model_manager import ModelManager
from helix_substrate.query_classifier import ModelTarget

# ─── Cost tracking ───────────────────────────────────────────────────────
t_start = time.time()
cpu_start = time.process_time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

# ─── Config ──────────────────────────────────────────────────────────────
TARGET_SEQ_LEN = 1024
PASSAGE_OFFSETS = [0, 2048, 4096]  # Token offsets (well-separated)
PASSAGE_NAMES = ["wiki_a", "wiki_b", "wiki_c"]


def kurtosis(arr):
    """Excess kurtosis (Fisher's definition)."""
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


def extract_passages(tokenizer):
    """Extract 3 x 1024-token passages from WikiText-2 test split."""
    from datasets import load_dataset

    print("Loading WikiText-2 test split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text into one corpus
    corpus = "\n".join(row["text"] for row in ds)
    print(f"  Corpus length: {len(corpus):,} chars")

    # Tokenize entire corpus
    all_tokens = tokenizer.encode(corpus)
    print(f"  Total tokens: {len(all_tokens):,}")

    # Verify we have enough tokens
    max_needed = max(PASSAGE_OFFSETS) + TARGET_SEQ_LEN
    assert len(all_tokens) >= max_needed, (
        f"Need {max_needed} tokens but corpus has {len(all_tokens)}"
    )

    passages = {}
    for name, offset in zip(PASSAGE_NAMES, PASSAGE_OFFSETS):
        token_ids = all_tokens[offset : offset + TARGET_SEQ_LEN]
        passages[name] = token_ids
        # Show first 80 chars of decoded text for verification
        preview = tokenizer.decode(token_ids[:30])[:80]
        print(f"  {name}: offset={offset}, tokens={len(token_ids)}, "
              f"preview=\"{preview}...\"")

    return passages


def main():
    out_dir = Path(os.path.expanduser("~/helix-substrate/kv_dump_long"))
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CDC-01.1 PHASE A: Long-Sequence KV Cache Capture")
    print(f"  Target seq_len: {TARGET_SEQ_LEN}")
    print(f"  Source: WikiText-2 test split (REAL text)")
    print("=" * 60)

    # Load TinyLlama
    print("\nLoading TinyLlama via ModelManager...")
    mgr = ModelManager()
    model, tokenizer = mgr.ensure_model(ModelTarget.TINYLLAMA)
    status = mgr.status()
    print(f"  Model: {status['active_model']}, VRAM: {status['vram_mb']} MB")

    # Extract passages
    passages = extract_passages(tokenizer)

    all_stats = {}

    for passage_name, token_ids in passages.items():
        print(f"\n── Passage: {passage_name} ──")
        seq_len = len(token_ids)
        print(f"  Tokens: {seq_len}")

        # Forward pass with KV cache
        input_ids = torch.tensor([token_ids], device=model.device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )

        past_kv = outputs.past_key_values
        n_layers = len(past_kv)
        print(f"  Layers: {n_layers}")
        print(f"  K shape: {past_kv[0][0].shape}")
        print(f"  V shape: {past_kv[0][1].shape}")

        # Save per-layer
        passage_dir = out_dir / passage_name
        passage_dir.mkdir(exist_ok=True)

        layer_stats = []
        for i, (k, v) in enumerate(past_kv):
            k_np = k.cpu().float().numpy()  # [1, n_kv_heads, seq_len, head_dim]
            v_np = v.cpu().float().numpy()

            np.save(passage_dir / f"layer_{i}_k.npy", k_np)
            np.save(passage_dir / f"layer_{i}_v.npy", v_np)

            # Distribution stats
            k_flat = k_np.ravel()
            v_flat = v_np.ravel()

            stats = {
                "layer": i,
                "k_shape": list(k_np.shape),
                "v_shape": list(v_np.shape),
                "k_mean": float(np.mean(k_flat)),
                "k_std": float(np.std(k_flat)),
                "k_min": float(np.min(k_flat)),
                "k_max": float(np.max(k_flat)),
                "k_kurtosis": kurtosis(k_flat),
                "v_mean": float(np.mean(v_flat)),
                "v_std": float(np.std(v_flat)),
                "v_min": float(np.min(v_flat)),
                "v_max": float(np.max(v_flat)),
                "v_kurtosis": kurtosis(v_flat),
                "k_bytes": k_np.nbytes,
                "v_bytes": v_np.nbytes,
            }
            layer_stats.append(stats)

        all_stats[passage_name] = {
            "seq_len": seq_len,
            "n_layers": n_layers,
            "layers": layer_stats,
        }

        # Summary
        total_bytes = sum(s["k_bytes"] + s["v_bytes"] for s in layer_stats)
        print(f"  Total KV size: {total_bytes / 1024 / 1024:.2f} MB")
        print(f"  K kurtosis range: [{min(s['k_kurtosis'] for s in layer_stats):.1f}, "
              f"{max(s['k_kurtosis'] for s in layer_stats):.1f}]")
        print(f"  V kurtosis range: [{min(s['v_kurtosis'] for s in layer_stats):.1f}, "
              f"{max(s['v_kurtosis'] for s in layer_stats):.1f}]")

    # ── Distribution fingerprint ──
    print("\n── Distribution Fingerprint ──")
    print(f"{'Tensor':<12} {'Mean':>8} {'Std':>8} {'Kurtosis':>10} {'Range':>12}")
    print("-" * 55)

    for layer_idx in [0, 11, 21]:
        for kv_type in ["k", "v"]:
            s = all_stats["wiki_a"]["layers"][layer_idx]
            prefix = f"{kv_type.upper()}[{layer_idx}]"
            m = s[f"{kv_type}_mean"]
            sd = s[f"{kv_type}_std"]
            kurt = s[f"{kv_type}_kurtosis"]
            rng = s[f"{kv_type}_max"] - s[f"{kv_type}_min"]
            print(f"{prefix:<12} {m:>8.4f} {sd:>8.4f} {kurt:>10.1f} {rng:>12.4f}")

    # ── Save stats & receipt ──
    stats_path = out_dir / "kv_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    receipt = {
        "experiment": "CDC-01.1-PHASE-A-LONG-KV-DUMP",
        "claim": "Capture 1024-token KV caches from TinyLlama on real WikiText-2 text",
        "source": "WikiText-2 test split (real text, not synthetic)",
        "passages": {
            name: {
                "token_offset": offset,
                "seq_len": TARGET_SEQ_LEN,
            }
            for name, offset in zip(PASSAGE_NAMES, PASSAGE_OFFSETS)
        },
        "model": "TinyLlama-1.1B (HelixLinear compressed)",
        "vram_mb": status["vram_mb"],
        "output_dir": str(out_dir),
        "cost": cost,
    }

    receipt_path = out_dir / "receipt.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"\n  Stats: {stats_path}")
    print(f"  Receipt: {receipt_path}")
    print(f"  Cost: {cost['wall_time_s']}s wall, {cost['peak_memory_mb']} MB peak")
    print()

    # Verify shapes
    sample = np.load(out_dir / "wiki_a" / "layer_0_k.npy")
    print(f"  Shape verification: {sample.shape}")
    assert sample.shape[2] == TARGET_SEQ_LEN, (
        f"Expected seq_len={TARGET_SEQ_LEN}, got {sample.shape[2]}"
    )
    print(f"  VERIFIED: seq_len={TARGET_SEQ_LEN}")

    # Unload to free VRAM
    mgr._unload()
    print("  Model unloaded. VRAM freed.")


if __name__ == "__main__":
    main()

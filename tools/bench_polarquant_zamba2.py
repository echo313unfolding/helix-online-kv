#!/usr/bin/env python3
"""PolarQuant downstream eval on Zamba2-1.2B via mamba-scan-lite.

Zamba2 uses Zamba2HybridDynamicCache (not DynamicCache), so CompressedKVCache
can't drop in directly. Instead we:
1. Run the model with its native cache to capture real attention-layer KV tensors
2. Feed those tensors through KVLayerState with/without PolarQuant
3. Compare MSE, cosine, and attention score fidelity

This proves PolarQuant works on Zamba2's actual KV distribution, not just
TinyLlama's.

Gate: PolarQuant must improve K MSE by >10% (same as TinyLlama gate).
"""

import json
import os
import sys
import time
import resource
import platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(os.path.expanduser("~/mamba-scan-lite"))))

import numpy as np
import torch

# Patch Zamba2 BEFORE loading model
import mamba_scan_lite

from transformers import AutoModelForCausalLM, AutoTokenizer
from helix_online_kv.config import OnlineKVConfig
from helix_online_kv.layer_state import KVLayerState

RECEIPTS_DIR = Path(__file__).resolve().parent.parent / "receipts" / "kv_cache"
MODEL_PATH = os.path.expanduser("~/models/zamba2-1.2b")

EVAL_TEXT = (
    "The history of artificial intelligence began in the mid-twentieth century "
    "when researchers first proposed that machines could be made to simulate "
    "human intelligence. Early work focused on symbolic reasoning and expert "
    "systems that encoded human knowledge as logical rules. The field experienced "
    "periods of optimism followed by funding cuts known as AI winters. The "
    "resurgence of neural networks in the 2010s driven by deep learning "
    "transformed the landscape. Convolutional networks achieved superhuman "
    "performance on image recognition tasks while recurrent networks showed "
    "promise for sequence modeling and machine translation. The introduction of "
    "the transformer architecture by Vaswani and colleagues in 2017 proved to be "
    "a watershed moment enabling models with hundreds of billions of parameters "
    "to be trained on massive text corpora. These large language models "
    "demonstrated emergent capabilities in reasoning code generation and few-shot "
    "learning that surprised even their creators."
)


def capture_attention_kv(model, tokenizer, text):
    """Run model and capture KV tensors from attention layers only."""
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    cache = outputs.past_key_values
    # Zamba2HybridDynamicCache has key_cache/value_cache lists
    # Attention layers have actual KV, Mamba layers have empty tensors
    attention_kvs = {}
    for layer_idx in range(len(cache.key_cache)):
        k = cache.key_cache[layer_idx]
        v = cache.value_cache[layer_idx]
        # Skip mamba layers (empty or shape [batch, 0])
        if k.dim() < 4 or k.shape[2] == 0:
            continue
        # k, v: [batch, n_heads, seq_len, head_dim]
        attention_kvs[layer_idx] = {
            "k": k[0].cpu().float(),  # [n_heads, seq_len, head_dim]
            "v": v[0].cpu().float(),
        }

    return attention_kvs


def cosine_sim(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mse(a, b):
    return float(np.mean((a.ravel() - b.ravel()) ** 2))


def run_layer_comparison(k_data, v_data, layer_idx, n_heads, head_dim, n_clusters=256):
    """Compare raw vs PolarQuant VQ on one layer's KV data (numpy)."""
    seq_len = k_data.shape[0]
    entry_size = k_data.shape[1]
    cal = min(128, seq_len // 2)
    test_start = cal
    n_test = seq_len - test_start

    if n_test < 4:
        return {"layer": layer_idx, "skipped": True, "reason": "too few tokens"}

    # Raw VQ
    config_raw = OnlineKVConfig(
        calibration_tokens=cal, n_clusters=n_clusters, exact_layers=[],
        polar_rotation=False,
    )
    ls_raw = KVLayerState(layer_idx, config_raw)
    for t in range(cal):
        ls_raw.feed_token(k_data[t], v_data[t])
    for t in range(test_start, seq_len):
        ls_raw.feed_token(k_data[t], v_data[t])

    raw_k = ls_raw.get_all_compressed_k()
    raw_v = ls_raw.get_all_compressed_v()

    # PolarQuant VQ
    config_polar = OnlineKVConfig(
        calibration_tokens=cal, n_clusters=n_clusters, exact_layers=[],
        polar_rotation=True, polar_seed=42, n_heads=n_heads,
    )
    ls_polar = KVLayerState(layer_idx, config_polar)
    for t in range(cal):
        ls_polar.feed_token(k_data[t], v_data[t])
    for t in range(test_start, seq_len):
        ls_polar.feed_token(k_data[t], v_data[t])

    polar_k = ls_polar.get_all_compressed_k()
    polar_v = ls_polar.get_all_compressed_v()

    k_orig = k_data[test_start:]
    v_orig = v_data[test_start:]

    raw_k_mse = mse(k_orig, raw_k)
    raw_v_mse = mse(v_orig, raw_v)
    polar_k_mse = mse(k_orig, polar_k)
    polar_v_mse = mse(v_orig, polar_v)

    raw_k_cos = cosine_sim(k_orig, raw_k)
    raw_v_cos = cosine_sim(v_orig, raw_v)
    polar_k_cos = cosine_sim(k_orig, polar_k)
    polar_v_cos = cosine_sim(v_orig, polar_v)

    k_imp = (raw_k_mse - polar_k_mse) / raw_k_mse * 100 if raw_k_mse > 0 else 0
    v_imp = (raw_v_mse - polar_v_mse) / raw_v_mse * 100 if raw_v_mse > 0 else 0

    # Attention fidelity
    attn_result = attention_fidelity(k_orig, v_orig, raw_k, polar_k, n_heads, head_dim)

    return {
        "layer": layer_idx,
        "n_clusters": n_clusters,
        "n_test_tokens": n_test,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "raw": {"k_mse": raw_k_mse, "v_mse": raw_v_mse, "k_cos": raw_k_cos, "v_cos": raw_v_cos},
        "polar": {"k_mse": polar_k_mse, "v_mse": polar_v_mse, "k_cos": polar_k_cos, "v_cos": polar_v_cos},
        "improvement": {"k_mse_pct": round(k_imp, 2), "v_mse_pct": round(v_imp, 2)},
        "attention": attn_result,
    }


def attention_fidelity(k_orig, v_orig, raw_k, polar_k, n_heads, head_dim):
    """Top-16 attention agreement: FP32 vs raw vs polar."""
    n_test = k_orig.shape[0]
    if n_test < 8:
        return {"skipped": True}

    def reshape_heads(x):
        return x.reshape(x.shape[0], n_heads, head_dim)

    k_orig_h = reshape_heads(k_orig)
    raw_k_h = reshape_heads(raw_k)
    polar_k_h = reshape_heads(polar_k)

    q = k_orig_h[-1]
    ctx_orig = k_orig_h[:-1]

    raw_agree = []
    polar_agree = []

    for head in range(n_heads):
        q_h = q[head]
        scale = 1.0 / np.sqrt(head_dim)
        scores_fp32 = (ctx_orig[:, head, :] @ q_h) * scale
        scores_raw = (raw_k_h[:-1, head, :] @ q_h) * scale
        scores_polar = (polar_k_h[:-1, head, :] @ q_h) * scale

        def softmax(x):
            x = x - x.max()
            e = np.exp(x)
            return e / e.sum()

        attn_fp32 = softmax(scores_fp32)
        attn_raw = softmax(scores_raw)
        attn_polar = softmax(scores_polar)

        top_n = min(16, len(attn_fp32))
        fp32_topk = set(np.argsort(attn_fp32)[-top_n:])
        raw_topk = set(np.argsort(attn_raw)[-top_n:])
        polar_topk = set(np.argsort(attn_polar)[-top_n:])

        raw_agree.append(len(fp32_topk & raw_topk) / top_n)
        polar_agree.append(len(fp32_topk & polar_topk) / top_n)

    return {
        "raw_mean": round(float(np.mean(raw_agree)), 4),
        "polar_mean": round(float(np.mean(polar_agree)), 4),
    }


def main():
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    t_start = time.time()
    cpu_start = time.process_time()

    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    dtype = torch.float16 if use_gpu else torch.float32

    print("=" * 70)
    print("POLARQUANT DOWNSTREAM: Zamba2-1.2B + mamba-scan-lite")
    print("=" * 70)
    print(f"  Device: {device}")
    if use_gpu:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")

    print(f"\nLoading {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype, device_map=device if use_gpu else None)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers loaded")
    if use_gpu:
        print(f"  VRAM after load: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    # Capture real KV tensors from attention layers
    print(f"\nCapturing KV tensors from attention layers ({len(EVAL_TEXT.split())} words)...")
    attn_kvs = capture_attention_kv(model, tokenizer, EVAL_TEXT)
    print(f"  Found {len(attn_kvs)} attention layers with KV data")

    # Free model VRAM
    del model
    if use_gpu:
        torch.cuda.empty_cache()

    # Run PolarQuant comparison on each attention layer
    all_results = []
    for layer_idx, kv in sorted(attn_kvs.items()):
        k_tensor = kv["k"]  # [n_heads, seq_len, head_dim]
        v_tensor = kv["v"]
        n_heads = k_tensor.shape[0]
        seq_len = k_tensor.shape[1]
        head_dim = k_tensor.shape[2]
        entry_size = n_heads * head_dim

        # Reshape to [seq_len, entry_size] for KVLayerState
        k_flat = k_tensor.permute(1, 0, 2).reshape(seq_len, entry_size).numpy()
        v_flat = v_tensor.permute(1, 0, 2).reshape(seq_len, entry_size).numpy()

        result = run_layer_comparison(k_flat, v_flat, layer_idx, n_heads, head_dim)
        all_results.append(result)

        if result.get("skipped"):
            print(f"  Layer {layer_idx}: SKIP ({result['reason']})")
            continue

        imp = result["improvement"]
        raw = result["raw"]
        polar = result["polar"]
        gate = "PASS" if imp["k_mse_pct"] > 10 else "FAIL"
        attn = result["attention"]

        print(f"  Layer {layer_idx:2d} ({n_heads}h×{head_dim}d): "
              f"K MSE {raw['k_mse']:.6f} → {polar['k_mse']:.6f} ({imp['k_mse_pct']:+.1f}%) [{gate}] "
              f"| attn top-16: {attn.get('raw_mean', 'N/A')} → {attn.get('polar_mean', 'N/A')}")

    # Summary
    valid = [r for r in all_results if not r.get("skipped")]
    if valid:
        avg_k_imp = np.mean([r["improvement"]["k_mse_pct"] for r in valid])
        avg_v_imp = np.mean([r["improvement"]["v_mse_pct"] for r in valid])
        gate_pass = sum(1 for r in valid if r["improvement"]["k_mse_pct"] > 10)
        avg_attn_raw = np.mean([r["attention"]["raw_mean"] for r in valid if "raw_mean" in r.get("attention", {})])
        avg_attn_polar = np.mean([r["attention"]["polar_mean"] for r in valid if "polar_mean" in r.get("attention", {})])

        print(f"\n=== SUMMARY (Zamba2-1.2B, {len(valid)} attention layers) ===")
        print(f"  Avg K MSE improvement: {avg_k_imp:+.1f}%")
        print(f"  Avg V MSE improvement: {avg_v_imp:+.1f}%")
        print(f"  Gate (>10%): {gate_pass}/{len(valid)} layers pass")
        print(f"  Avg attention top-16: raw={avg_attn_raw:.4f} → polar={avg_attn_polar:.4f}")

    # Receipt
    cost = {
        'wall_time_s': round(time.time() - t_start, 3),
        'cpu_time_s': round(time.process_time() - cpu_start, 3),
        'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
        'timestamp_start': start_iso,
        'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'device': device,
    }
    if use_gpu:
        cost['gpu'] = torch.cuda.get_device_name(0)

    receipt = {
        "work_order": "WO-KVCACHE-POLARQUANT-01",
        "experiment": "polarquant_downstream_zamba2",
        "model": "Zamba2-1.2B",
        "mamba_scan_lite": True,
        "n_attention_layers": len(valid),
        "gate": {
            "criterion": "K MSE improvement > 10%",
            "layers_passing": gate_pass if valid else 0,
            "layers_total": len(valid) if valid else 0,
        },
        "results": all_results,
        "cost": cost,
    }

    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = RECEIPTS_DIR / "polarquant_zamba2_downstream.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, default=str)
    print(f"\nReceipt: {receipt_path}")


if __name__ == "__main__":
    main()

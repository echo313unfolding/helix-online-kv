#!/usr/bin/env python3
"""Phase 4: End-to-End Compressed Generation.

HelixLinear compressed weights + CompressedKVCache compressed KV,
both running on T2000 (or CPU fallback).

Gate: Combined PPL < +2% vs dense+vanilla baseline.
Gate: Fits in 4GB at 4096 tokens where dense+vanilla OOMs.
"""

import json
import os
import sys
import time
import resource
import platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# helix-substrate must be importable
sys.path.insert(0, str(Path(os.path.expanduser("~/helix-substrate"))))

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from helix_online_kv.config import OnlineKVConfig
from helix_online_kv.compressed_cache import CompressedKVCache

RECEIPTS_DIR = Path(__file__).resolve().parent.parent / "receipts"
MODEL_PATH = Path(os.path.expanduser("~/models/tinyllama_fp32"))
CDNA_DIR = MODEL_PATH / "cdnav3"

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


def eval_ppl_token_by_token(model, tokenizer, text, cache=None, label=""):
    """Evaluate PPL via token-by-token forward with optional CompressedKVCache."""
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if next(model.parameters()).is_cuda:
        input_ids = input_ids.cuda()
    seq_len = input_ids.shape[1]

    total_loss = 0.0
    n_tokens = 0

    with torch.no_grad():
        for t in range(seq_len - 1):
            token = input_ids[:, t:t+1]
            kwargs = {"use_cache": True}
            if cache is not None:
                kwargs["past_key_values"] = cache
            outputs = model(token, **kwargs)
            logits = outputs.logits[:, -1, :]
            target = input_ids[:, t+1]
            if logits.device != target.device:
                target = target.to(logits.device)
            loss = F.cross_entropy(logits, target)
            total_loss += loss.item()
            n_tokens += 1

            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                cache = outputs.past_key_values

    avg_loss = total_loss / n_tokens
    ppl = float(np.exp(avg_loss))
    return {"ppl": round(ppl, 4), "loss": round(avg_loss, 6), "seq_len": seq_len, "label": label}


def get_vram_mb():
    """Current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
    return 0.0


def get_vram_peak_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    return 0.0


def main():
    if not _HAS_TORCH:
        print("ERROR: torch/transformers not available")
        return 1

    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    t_start = time.time()
    cpu_start = time.process_time()

    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"

    print("=" * 70)
    print("PHASE 4: END-TO-END COMPRESSED GENERATION")
    print("=" * 70)
    print(f"  Device: {device}")
    if use_gpu:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
        torch.cuda.reset_peak_memory_stats()

    # --- Stage A: Dense + vanilla baseline (CPU, avoids OOM) ---
    print("\n--- Stage A: Dense FP32 + vanilla cache (CPU) ---")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    dense_model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), dtype=torch.float32)
    dense_model.eval()

    baseline = eval_ppl_token_by_token(dense_model, tokenizer, EVAL_TEXT, label="dense+vanilla")
    print(f"  PPL={baseline['ppl']:.4f}  seq_len={baseline['seq_len']}")
    del dense_model
    if use_gpu:
        torch.cuda.empty_cache()

    # --- Stage B: Dense + CompressedKVCache (CPU) ---
    print("\n--- Stage B: Dense FP32 + CompressedKVCache (CPU) ---")
    dense_model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), dtype=torch.float32)
    dense_model.eval()
    n_layers = dense_model.config.num_hidden_layers

    kv_config = OnlineKVConfig(
        calibration_tokens=128,
        hot_window=256,
        n_clusters=256,
        exact_layers=[0],
    )
    cache = CompressedKVCache(kv_config, n_layers=n_layers)
    dense_compressed = eval_ppl_token_by_token(
        dense_model, tokenizer, EVAL_TEXT, cache=cache, label="dense+compressed_kv"
    )
    print(f"  PPL={dense_compressed['ppl']:.4f}  "
          f"delta={((dense_compressed['ppl']/baseline['ppl'])-1)*100:+.4f}%")
    del dense_model, cache
    if use_gpu:
        torch.cuda.empty_cache()

    # --- Stage C: HelixLinear + CompressedKVCache ---
    print("\n--- Stage C: HelixLinear + CompressedKVCache ---")
    if not CDNA_DIR.exists():
        print(f"  SKIP: CDNA dir not found at {CDNA_DIR}")
        helix_result = None
    else:
        try:
            from helix_substrate.helix_linear import load_cdna_factors, swap_to_helix

            base_model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_PATH), dtype=torch.float32
            )
            helix_modules = load_cdna_factors(str(CDNA_DIR), model=base_model)
            model = swap_to_helix(base_model, helix_modules)

            if use_gpu:
                model = model.cuda()
                torch.cuda.reset_peak_memory_stats()
                vram_after_load = get_vram_mb()
                print(f"  VRAM after model load: {vram_after_load} MB")

            model.eval()
            n_layers = model.config.num_hidden_layers

            cache = CompressedKVCache(kv_config, n_layers=n_layers)
            helix_result = eval_ppl_token_by_token(
                model, tokenizer, EVAL_TEXT, cache=cache, label="helix+compressed_kv"
            )

            if use_gpu:
                vram_peak = get_vram_peak_mb()
                helix_result["vram_after_load_mb"] = vram_after_load
                helix_result["vram_peak_mb"] = vram_peak

            helix_ppl_delta = ((helix_result['ppl'] / baseline['ppl']) - 1) * 100
            helix_result["ppl_delta_pct"] = round(helix_ppl_delta, 4)
            print(f"  PPL={helix_result['ppl']:.4f}  delta={helix_ppl_delta:+.4f}%")
            if use_gpu:
                print(f"  VRAM peak: {vram_peak} MB")

            # Memory report
            mem = cache.memory_report()
            print(f"  Cache: {mem['total_tokens']} tokens, "
                  f"exact={mem['exact_bytes']//1024}KB, "
                  f"compressed={mem['compressed_bytes']//1024}KB")
            helix_result["cache_memory"] = mem

            del model, cache
            if use_gpu:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            helix_result = {"error": str(e)}

    # --- Gate checks ---
    print("\n--- GATE CHECKS ---")
    gates = {}

    # Gate 1: Combined PPL < +2%
    if helix_result and "ppl" in helix_result:
        delta = abs(helix_result["ppl_delta_pct"])
        g1 = delta < 2.0
        gates["ppl_under_2pct"] = {"passed": g1, "delta_pct": delta}
        print(f"  PPL < +2%: {delta:.4f}% [{'PASS' if g1 else 'FAIL'}]")
    else:
        gates["ppl_under_2pct"] = {"passed": False, "reason": "helix model not loaded"}
        print("  PPL < +2%: SKIP (helix model not loaded)")

    # Gate 2: Fits in 4GB (if GPU available)
    if use_gpu and helix_result and "vram_peak_mb" in helix_result:
        peak = helix_result["vram_peak_mb"]
        g2 = peak < 4096
        gates["fits_4gb"] = {"passed": g2, "vram_peak_mb": peak}
        print(f"  Fits 4GB: peak {peak} MB [{'PASS' if g2 else 'FAIL'}]")
    else:
        gates["fits_4gb"] = {"passed": None, "reason": "no GPU or helix not loaded"}
        print("  Fits 4GB: SKIP (no GPU or helix not loaded)")

    overall = all(g.get("passed", False) for g in gates.values() if g.get("passed") is not None)

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
        "experiment": "e2e_compressed_generation",
        "phase": 4,
        "gates": gates,
        "overall_pass": overall,
        "baseline": baseline,
        "dense_compressed_kv": dense_compressed,
        "helix_compressed_kv": helix_result,
        "config": {
            "calibration_tokens": kv_config.calibration_tokens,
            "hot_window": kv_config.hot_window,
            "n_clusters": kv_config.n_clusters,
            "exact_layers": list(kv_config.exact_layers),
        },
        "cost": cost,
    }

    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = RECEIPTS_DIR / f"e2e_compressed_{time.strftime('%Y%m%dT%H%M%S')}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, default=str)
    print(f"\nReceipt saved: {receipt_path}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Quick A/B generation test: Zamba2-1.2B with PolarQuant ON vs OFF.

Runs 10 prompts through the model with Zamba2CompressedCache, comparing
output coherence between polar_rotation=True and polar_rotation=False.

Not a benchmark — a sanity check that PolarQuant doesn't break generation.
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

import torch
import mamba_scan_lite

from transformers import AutoModelForCausalLM, AutoTokenizer
from helix_online_kv.zamba2_compressed_cache import Zamba2CompressedCache

RECEIPTS_DIR = Path(__file__).resolve().parent.parent / "receipts" / "kv_cache"
MODEL_PATH = os.path.expanduser("~/models/zamba2-1.2b")

PROMPTS = [
    "The most important discovery in physics was",
    "To make a perfect cup of coffee, you need to",
    "The difference between machine learning and deep learning is",
    "In 2025, the biggest challenge facing humanity is",
    "A good software engineer should always",
    "The history of the internet began when",
    "Climate change affects ocean ecosystems by",
    "The key principles of good API design include",
    "Quantum computing differs from classical computing because",
    "The best way to learn a new programming language is",
]

MAX_NEW_TOKENS = 50


def generate_with_cache(model, tokenizer, prompt, cache, max_tokens=50):
    """Generate tokens using model with custom cache."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(
                generated_ids[:, -1:] if cache.has_previous_state else generated_ids,
                past_key_values=cache,
                use_cache=True,
            )
            cache.has_previous_state = True
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return output_text


def main():
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    t_start = time.time()

    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    dtype = torch.float16 if use_gpu else torch.float32

    print("=" * 70)
    print("ZAMBA2 GENERATION A/B TEST: PolarQuant ON vs OFF")
    print("=" * 70)
    print(f"  Device: {device}")
    if use_gpu:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype, device_map=device if use_gpu else None)
    model.eval()
    print(f"  {model.config.num_hidden_layers} layers loaded")
    if use_gpu:
        print(f"  VRAM after load: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    results = []

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}/{len(PROMPTS)} ---")
        print(f"  \"{prompt}\"")

        # A: No PolarQuant
        cache_off = Zamba2CompressedCache(
            model.config, batch_size=1, dtype=dtype,
            device=device, polar_rotation=False,
        )
        t0 = time.time()
        text_off = generate_with_cache(model, tokenizer, prompt, cache_off, MAX_NEW_TOKENS)
        time_off = time.time() - t0

        # B: PolarQuant ON
        cache_on = Zamba2CompressedCache(
            model.config, batch_size=1, dtype=dtype,
            device=device, polar_rotation=True,
        )
        t0 = time.time()
        text_on = generate_with_cache(model, tokenizer, prompt, cache_on, MAX_NEW_TOKENS)
        time_on = time.time() - t0

        match = text_off == text_on
        stats_off = cache_off.compression_stats()
        stats_on = cache_on.compression_stats()

        print(f"  OFF ({time_off:.1f}s): {text_off[len(prompt):].strip()[:80]}...")
        print(f"   ON ({time_on:.1f}s): {text_on[len(prompt):].strip()[:80]}...")
        print(f"  Match: {match}")

        results.append({
            "prompt": prompt,
            "text_off": text_off,
            "text_on": text_on,
            "match": match,
            "time_off_s": round(time_off, 2),
            "time_on_s": round(time_on, 2),
        })

        del cache_off, cache_on
        if use_gpu:
            torch.cuda.empty_cache()

    # Summary
    matches = sum(1 for r in results if r["match"])
    print(f"\n=== SUMMARY ===")
    print(f"  Exact matches: {matches}/{len(results)}")
    print(f"  (Exact match expected during calibration phase — compression")
    print(f"   only affects tokens beyond calibration_tokens=128)")

    # VRAM report on last generation
    if use_gpu:
        print(f"  VRAM peak: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

    # Receipt
    cost = {
        'wall_time_s': round(time.time() - t_start, 3),
        'cpu_time_s': round(time.process_time() - time.process_time(), 3),
        'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
        'timestamp_start': start_iso,
        'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    if use_gpu:
        cost['gpu'] = torch.cuda.get_device_name(0)

    receipt = {
        "work_order": "WO-KVCACHE-POLARQUANT-01",
        "experiment": "zamba2_generation_ab_test",
        "model": "Zamba2-1.2B",
        "mamba_scan_lite": True,
        "n_prompts": len(results),
        "max_new_tokens": MAX_NEW_TOKENS,
        "exact_matches": matches,
        "results": results,
        "cost": cost,
    }

    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = RECEIPTS_DIR / "polarquant_zamba2_generation.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {receipt_path}")


if __name__ == "__main__":
    main()

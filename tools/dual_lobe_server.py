#!/usr/bin/env python3
"""Echo Dual-Lobe Server: Zamba2-1.2B-HXQ full stack on T2000.

Zamba2 IS a dual-lobe architecture:
  Lobe A (Mamba): 32 layers, sequential processing, SSM state
  Lobe B (Attention): 6 shared layers, exact retrieval, KV cache

This server wires:
  - HXQ compression (HelixLinear, 1.35 GB VRAM)
  - mamba-scan-lite (no CUDA kernel compilation)
  - PolarQuant KV cache compression (93% MSE improvement on attention layers)
  - Zamba2CompressedCache (compresses attention KV, leaves Mamba state untouched)
  - Soulfile identity injection
  - OpenAI-compatible /v1/chat/completions endpoint

Usage:
    python3 tools/dual_lobe_server.py
    curl -X POST http://localhost:8001/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "What is gravity?"}]}'
"""

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.path.expanduser("~/helix-substrate"))
sys.path.insert(0, os.path.expanduser("~/mamba-scan-lite"))

# Register HXQ quantizer BEFORE any model loading
import helix_substrate.hf_quantizer

# Patch Zamba2 Mamba layers
import mamba_scan_lite

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from helix_online_kv.zamba2_compressed_cache import Zamba2CompressedCache

MODEL_PATH = os.path.expanduser("~/models/zamba2-1.2b-helix")
SOULFILE_PATHS = [
    os.path.expanduser("~/.echo/vault/soul.json"),
    os.path.expanduser("~/helix-cdc/config/soul.yaml"),
]
PORT = 8001


def load_soulfile():
    """Load soulfile identity for prompt injection."""
    for path in SOULFILE_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            if path.endswith('.json'):
                try:
                    data = json.loads(content)
                    name = data.get("identity", data.get("name", "Echo"))
                    vows = data.get("vows", [])
                    # Softer injection — don't let vows override factual answers
                    vow_str = ", ".join(v.replace("_", " ") for v in vows) if vows else ""
                    parts = [f"You are {name}, a helpful AI assistant."]
                    if vow_str:
                        parts.append(f"Your guiding principles: {vow_str}.")
                    parts.append("Answer questions accurately and helpfully.")
                    return " ".join(parts)
                except json.JSONDecodeError:
                    return content[:500]
            else:
                return content[:500]
    return ""


def build_prompt(messages, soul_prefix=""):
    """Build prompt from OpenAI-style messages.

    Uses a clean single-turn format to minimize loop hallucination.
    The model sees exactly one User/Assistant exchange.
    """
    parts = []
    if soul_prefix:
        parts.append(soul_prefix)

    # Collect system messages into preamble
    for msg in messages:
        if msg.get("role") == "system":
            parts.append(msg.get("content", ""))

    # Build conversation — only include the last user message to avoid
    # teaching the model the "multi-turn" pattern
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

    # If there's conversation history, summarize it as context
    if len(user_msgs) > 1 or assistant_msgs:
        history = []
        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                history.append(msg)
        # Include full history but keep format tight
        for msg in history:
            role = "Q" if msg["role"] == "user" else "A"
            parts.append(f"{role}: {msg['content']}")
        parts.append("A:")
    else:
        # Single turn — cleanest format
        if user_msgs:
            parts.append(f"Q: {user_msgs[-1]['content']}")
        parts.append("A:")

    return "\n".join(parts)


# Patterns that signal the model is looping into a fake conversation
# or hallucinating new instructions/tasks
_LOOP_PATTERNS = re.compile(
    r'\n(?:Q:|A:|User:|Assistant:|Human:|Bot:|Question:|Answer:|Instruction:)\s',
    re.IGNORECASE,
)
# Also catch "Instruction:" even without leading newline (often mid-sentence)
_INSTRUCTION_PATTERN = re.compile(r'\s+Instruction:\s', re.IGNORECASE)


def truncate_at_loop(text: str) -> str:
    """Cut response at the first sign of a fake turn marker or hallucinated instruction."""
    # Check newline + turn marker
    match = _LOOP_PATTERNS.search(text)
    if match:
        text = text[:match.start()]
    # Check mid-sentence "Instruction:" hallucination
    match = _INSTRUCTION_PATTERN.search(text)
    if match:
        text = text[:match.start()]
    return text.strip()


class EchoLobeServer:
    """Single Zamba2 model with full Echo stack."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("=" * 60)
        print("ECHO DUAL-LOBE SERVER")
        print("=" * 60)

        # Load soulfile
        self.soul_prefix = load_soulfile()
        if self.soul_prefix:
            print(f"  Soulfile: loaded ({len(self.soul_prefix)} chars)")
        else:
            print("  Soulfile: not found")

        # Load model
        print(f"\n  Loading Zamba2-1.2B-HXQ → {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map=self.device if self.device == "cuda" else None,
        )
        self.model.eval()

        n_layers = self.model.config.num_hidden_layers
        attn_layers = [i for i, t in enumerate(self.model.config.layers_block_type) if t == "hybrid"]

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**2
            print(f"  VRAM: {vram:.1f} MB")
        print(f"  Architecture: {n_layers} layers ({len(attn_layers)} attention, {n_layers - len(attn_layers)} Mamba)")
        print(f"  Attention layers: {attn_layers}")
        print(f"  PolarQuant: enabled (93% K MSE improvement)")
        print(f"\n  Ready on :{PORT}")

    def generate(self, messages, max_tokens=200, temperature=0.7):
        """Generate response with compressed cache."""
        prompt = build_prompt(messages, self.soul_prefix)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Create PolarQuant compressed cache
        cache = Zamba2CompressedCache(
            self.model.config,
            batch_size=1,
            dtype=next(self.model.parameters()).dtype,
            device=self.device,
            polar_rotation=True,
        )

        # Encode multiple stop sequences for StoppingCriteria
        stop_seqs = [
            self.tokenizer.encode(s, add_special_tokens=False)
            for s in ["\nQ:", "\nUser:", "\nHuman:", "\nQuestion:", "\nInstruction:"]
        ]

        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnTurnMarker(StoppingCriteria):
            def __init__(self, stop_seqs, prompt_len):
                self.stop_seqs = stop_seqs
                self.prompt_len = prompt_len
            def __call__(self, input_ids, scores, **kwargs):
                gen = input_ids[0][self.prompt_len:]
                for seq in self.stop_seqs:
                    if len(gen) >= len(seq):
                        if gen[-len(seq):].tolist() == seq:
                            return True
                return False

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                past_key_values=cache,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([
                    StopOnTurnMarker(stop_seqs, input_ids.shape[1])
                ]),
            )
        gen_time = time.time() - t0

        # Decode only the new tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Primary defense: truncate at any loop pattern in decoded text
        response = truncate_at_loop(response)

        n_tokens = len(new_tokens)
        tok_per_sec = n_tokens / gen_time if gen_time > 0 else 0

        # Compression stats
        comp_stats = cache.compression_stats()
        mem = cache.memory_report()

        return {
            "text": response.strip(),
            "tokens": n_tokens,
            "tok_per_sec": round(tok_per_sec, 1),
            "gen_time_s": round(gen_time, 2),
            "cache": {
                "exact_kv_kb": mem["exact_kv_bytes"] // 1024,
                "compressed_kb": mem["compressed_index_bytes"] // 1024,
                "mamba_state_kb": mem["mamba_state_bytes"] // 1024,
            },
        }


def main():
    server = EchoLobeServer()

    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        print("\nfastapi/uvicorn not installed. Running in REPL mode.\n")
        print("Type a message (or 'quit' to exit):\n")
        while True:
            try:
                query = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in ("quit", "exit", "q"):
                break
            result = server.generate([{"role": "user", "content": query}])
            print(f"Echo [{result['tok_per_sec']} tok/s]: {result['text']}\n")
        return

    app = FastAPI(title="Echo Dual-Lobe Server")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 200)
        temperature = request.get("temperature", 0.7)

        result = server.generate(messages, max_tokens=max_tokens, temperature=temperature)

        return JSONResponse({
            "id": f"echo-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "zamba2-1.2b-hxq-echo",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"],
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "completion_tokens": result["tokens"],
                "total_tokens": result["tokens"],
            },
            "echo_metadata": {
                "lobe_a": "mamba_32_layers",
                "lobe_b": "attention_6_layers_polarquant",
                "tok_per_sec": result["tok_per_sec"],
                "cache": result["cache"],
            },
        })

    @app.get("/v1/models")
    async def list_models():
        vram = 0
        if torch.cuda.is_available():
            vram = round(torch.cuda.memory_allocated() / 1024**2, 1)
        return {
            "data": [{
                "id": "zamba2-1.2b-hxq-echo",
                "object": "model",
                "architecture": "dual-lobe (32 mamba + 6 shared attention)",
                "compression": "HXQ 4x + PolarQuant KV cache",
                "vram_mb": vram,
            }]
        }

    @app.get("/health")
    async def health():
        vram = 0
        if torch.cuda.is_available():
            vram = round(torch.cuda.memory_allocated() / 1024**2, 1)
        return {"status": "ok", "vram_mb": vram, "model": "zamba2-1.2b-hxq"}

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()

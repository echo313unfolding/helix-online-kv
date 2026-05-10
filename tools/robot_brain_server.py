#!/usr/bin/env python3
"""WO-ROBOT-BRAIN-01: Multi-Model Compressed AI System on Single 24GB GPU.

Three HXQ-compressed models + EchoMemory + FGIP + Soulfile, one process:
  - Zamba2-7B-Instruct (language lobe) — hybrid SSM+Transformer, 213 HelixLinear
  - Qwen2.5-Coder-3B (code lobe) — decoder-only Transformer, 252 HelixLinear
  - CLIP-ViT-L/14 (vision lobe) — dual-encoder, 218 HelixLinear
  - EchoMemory — SQLite hybrid retrieval (TF-IDF + embedding cosine)
  - FGIP graph — 1,896 nodes, 3,411 edges, FTS5 search
  - Soulfile — identity injection on every generation

Total: 683 HelixLinear modules, ~9.2 GB VRAM, one shared materialization buffer.

Usage:
    python3 robot_brain_server.py
    # Language
    curl -X POST http://localhost:8001/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "What is gravity?"}]}'
    # Code (auto-routed or explicit)
    curl -X POST http://localhost:8001/v1/chat/completions \
      -d '{"model": "code", "messages": [{"role": "user", "content": "Write a sort"}]}'
    # Vision
    curl -X POST http://localhost:8001/v1/vision/classify \
      -F "image=@photo.jpg" -F 'labels=["cat","dog","car"]'
    # Memory search
    curl -X POST http://localhost:8001/v1/memory/search \
      -d '{"query": "compression"}'
    # FGIP graph search
    curl -X POST http://localhost:8001/v1/fgip/search \
      -d '{"query": "copper supply"}'
"""

import json
import os
import re
import sqlite3
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.path.expanduser("~/helix-substrate"))
sys.path.insert(0, os.path.expanduser("~/echo_bridge"))

# Register HXQ quantizer BEFORE any model loading
import helix_substrate.hf_quantizer

# Patch Zamba2 Mamba layers
import mamba_scan_lite

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    CLIPModel, CLIPProcessor,
    StoppingCriteria, StoppingCriteriaList,
)
from helix_substrate.helix_linear import HelixLinear

# ── Paths ──
ZAMBA_PATH = os.path.expanduser("~/models/zamba2-7b-instruct-vq2d-helix")
QWEN_PATH = os.path.expanduser("~/models/qwen2.5-coder-3b-hxq")
CLIP_PATH = os.path.expanduser("~/models/clip-vit-large-patch14-helix")
SOULFILE_PATH = os.path.expanduser("~/soul.json")
ECHO_MEMORY_DB = os.path.expanduser("~/echo_bridge/echo_memory.sqlite")
FGIP_DB = os.path.expanduser("~/fgip.db")
PORT = 8001

# ── Code detection keywords ──
_CODE_KEYWORDS = re.compile(
    r'\b(?:def |class |import |function |const |let |var |print\(|console\.|'
    r'```|write (?:a |me )?(?:code|function|script|program|class)|'
    r'debug|refactor|implement|compile|syntax|algorithm|regex|'
    r'python|javascript|typescript|rust|golang|java|cpp|sql)\b',
    re.IGNORECASE,
)

# ── Memory-trigger keywords ──
_MEMORY_STORE_KEYWORDS = re.compile(
    r'\b(?:remember|memorize|store|save|note that|keep in mind)\b',
    re.IGNORECASE,
)
_MEMORY_RECALL_KEYWORDS = re.compile(
    r'\b(?:what do you (?:know|remember)|recall|what did we|'
    r'have we talked|my (?:name|favorite|preference))\b',
    re.IGNORECASE,
)

# ── FGIP-trigger keywords ──
_FGIP_KEYWORDS = re.compile(
    r'\b(?:supply chain|copper|silver|data center|power grid|'
    r'investment|portfolio|bottleneck|commodity|infrastructure|'
    r'Fed|treasury|market|mining|energy)\b',
    re.IGNORECASE,
)

# ── Anti-loop patterns ──
_LOOP_PATTERNS = re.compile(
    r'\n(?:Q:|A:|User:|Assistant:|Human:|Bot:|Question:|Answer:|Instruction:)\s',
    re.IGNORECASE,
)
_INSTRUCTION_PATTERN = re.compile(r'\s+Instruction:\s', re.IGNORECASE)


def truncate_at_loop(text: str) -> str:
    match = _LOOP_PATTERNS.search(text)
    if match:
        text = text[:match.start()]
    match = _INSTRUCTION_PATTERN.search(text)
    if match:
        text = text[:match.start()]
    return text.strip()


# ── Soulfile ──
def load_soulfile():
    if os.path.exists(SOULFILE_PATH):
        with open(SOULFILE_PATH) as f:
            data = json.load(f)
        name = data.get("identity", "Echo")
        vows = data.get("vows", [])
        vow_str = ", ".join(v.replace("_", " ") for v in vows) if vows else ""
        parts = [f"You are {name}, a helpful AI assistant."]
        if vow_str:
            parts.append(f"Your guiding principles: {vow_str}.")
        parts.append("Answer questions accurately and helpfully.")
        return " ".join(parts)
    return "You are Echo, a helpful AI assistant."


# ── EchoMemory (lightweight wrapper — no sentence-transformers needed) ──
class MemoryStore:
    """Lightweight EchoMemory wrapper using TF-IDF search only (no embeddings)."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.available = False

    def connect(self):
        if not os.path.exists(self.db_path):
            print(f"  EchoMemory: DB not found at {self.db_path}")
            return
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.available = True
        # Count items
        try:
            seeds = self.conn.execute("SELECT COUNT(*) FROM mem_items").fetchone()[0]
            interactions = self.conn.execute("SELECT COUNT(*) FROM mem_interactions").fetchone()[0]
            print(f"  EchoMemory: {seeds} seeds, {interactions} interactions")
        except Exception:
            print("  EchoMemory: connected (schema unknown)")

    def search(self, query, top_k=5):
        """Search seed memories by text overlap."""
        if not self.available:
            return []
        try:
            words = re.findall(r'\w+', query.lower())
            if not words:
                return []
            # Simple LIKE-based search across seed items
            conditions = " OR ".join(["text LIKE ?" for _ in words])
            params = [f"%{w}%" for w in words]
            rows = self.conn.execute(
                f"SELECT text, source_file, tags_json FROM mem_items WHERE {conditions} LIMIT ?",
                params + [top_k],
            ).fetchall()
            return [{"text": r["text"][:300], "source": r["source_file"]} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    def search_interactions(self, query, top_k=5):
        """Search past interactions."""
        if not self.available:
            return []
        try:
            words = re.findall(r'\w+', query.lower())
            if not words:
                return []
            conditions = " OR ".join(["query LIKE ? OR response LIKE ?" for _ in words])
            params = []
            for w in words:
                params.extend([f"%{w}%", f"%{w}%"])
            rows = self.conn.execute(
                f"SELECT query, response, model_id, timestamp FROM mem_interactions "
                f"WHERE {conditions} ORDER BY timestamp DESC LIMIT ?",
                params + [top_k],
            ).fetchall()
            return [{"query": r["query"][:200], "response": r["response"][:200]} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    def store_interaction(self, query, response, model_id="robot_brain"):
        """Store a new interaction."""
        if not self.available:
            return
        try:
            import hashlib
            q_hash = hashlib.sha256(query.encode()).hexdigest()
            r_hash = hashlib.sha256(response.encode()).hexdigest()
            self.conn.execute(
                "INSERT OR IGNORE INTO mem_interactions "
                "(trace_id, query, response, task, model_id, query_sha256, response_sha256, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (str(uuid.uuid4()), query, response, "generate", model_id, q_hash, r_hash),
            )
            self.conn.commit()
        except Exception:
            pass


# ── FGIP Graph ──
class FGIPGraph:
    """FGIP knowledge graph with FTS5 search."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.available = False

    def connect(self):
        if not os.path.exists(self.db_path):
            print(f"  FGIP: DB not found at {self.db_path}")
            return
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.available = True
        nodes = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edges = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        print(f"  FGIP: {nodes} nodes, {edges} edges")

    def search(self, query, top_k=5):
        """Search nodes and edges via FTS5."""
        if not self.available:
            return {"nodes": [], "edges": []}
        # Clean query for FTS5
        terms = re.findall(r'\w+', query)
        if not terms:
            return {"nodes": [], "edges": []}
        fts_query = " OR ".join(terms)

        nodes = []
        try:
            rows = self.conn.execute(
                "SELECT node_id, name, description FROM nodes_fts "
                "WHERE nodes_fts MATCH ? LIMIT ?",
                (fts_query, top_k),
            ).fetchall()
            nodes = [{"id": r["node_id"], "name": r["name"],
                      "description": (r["description"] or "")[:300]} for r in rows]
        except Exception:
            pass

        edges = []
        try:
            rows = self.conn.execute(
                "SELECT edge_id, edge_type, from_node_id, to_node_id, notes "
                "FROM edges WHERE notes LIKE ? OR edge_type LIKE ? LIMIT ?",
                (f"%{terms[0]}%", f"%{terms[0]}%", top_k),
            ).fetchall()
            edges = [{"type": r["edge_type"], "from": r["from_node_id"],
                      "to": r["to_node_id"], "notes": (r["notes"] or "")[:200]} for r in rows]
        except Exception:
            pass

        return {"nodes": nodes, "edges": edges}


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


def route_input(messages, explicit_model=None):
    """Route to: language, code, or vision."""
    if explicit_model:
        m = explicit_model.lower()
        if any(k in m for k in ("qwen", "code", "coder")):
            return "code"
        if any(k in m for k in ("clip", "vision", "image")):
            return "vision"
        if any(k in m for k in ("zamba", "language", "chat")):
            return "language"

    user_msgs = [m for m in messages if m.get("role") == "user"]
    if user_msgs:
        last_msg = user_msgs[-1].get("content", "")
        if _CODE_KEYWORDS.search(last_msg):
            return "code"

    return "language"


class RobotBrainServer:
    """Three-model HXQ server + EchoMemory + FGIP + Soulfile."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("=" * 60)
        print("ECHO ROBOT BRAIN  (WO-ROBOT-BRAIN-01)")
        print("=" * 60)

        # ── Soulfile ──
        self.soul_prefix = load_soulfile()
        print(f"  Soulfile: {len(self.soul_prefix)} chars")

        # ── EchoMemory ──
        self.memory = MemoryStore(ECHO_MEMORY_DB)
        self.memory.connect()

        # ── FGIP ──
        self.fgip = FGIPGraph(FGIP_DB)
        self.fgip.connect()

        # ── Load Zamba2-7B (language lobe) ──
        print(f"\n  [1/3] Loading Zamba2-7B-Instruct-HXQ → {self.device}...")
        t0 = time.time()
        self.zamba_tokenizer = AutoTokenizer.from_pretrained(ZAMBA_PATH)
        self.zamba_model = AutoModelForCausalLM.from_pretrained(
            ZAMBA_PATH, device_map=self.device, dtype=torch.bfloat16,
        )
        self.zamba_model.eval()
        zamba_vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        print(f"        {time.time()-t0:.1f}s | {zamba_vram:.0f} MB | "
              f"{sum(1 for m in self.zamba_model.modules() if isinstance(m, HelixLinear))} HelixLinear")

        # ── Load Qwen-Coder-3B (code lobe) ──
        print(f"  [2/3] Loading Qwen2.5-Coder-3B-HXQ → {self.device}...")
        t0 = time.time()
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH, device_map=self.device, dtype=torch.bfloat16,
        )
        self.qwen_model.eval()
        qwen_vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        print(f"        {time.time()-t0:.1f}s | {qwen_vram:.0f} MB | "
              f"{sum(1 for m in self.qwen_model.modules() if isinstance(m, HelixLinear))} HelixLinear")

        # ── Load CLIP (vision lobe) ──
        print(f"  [3/3] Loading CLIP-ViT-L/14-HXQ → {self.device}...")
        t0 = time.time()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)
        self.clip_model = CLIPModel.from_pretrained(CLIP_PATH, dtype=torch.float32)
        try:
            self.clip_model = self.clip_model.to(self.device)
            self.clip_device = self.device
        except (NotImplementedError, RuntimeError):
            self.clip_device = "cpu"
            print("        NOTE: CLIP on CPU (meta tensor fallback)")
        self.clip_model.eval()
        clip_vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        print(f"        {time.time()-t0:.1f}s | {clip_vram:.0f} MB | "
              f"{sum(1 for m in self.clip_model.modules() if isinstance(m, HelixLinear))} HelixLinear")

        # Summary
        total_helix = sum(1 for m in self.zamba_model.modules() if isinstance(m, HelixLinear))
        total_helix += sum(1 for m in self.qwen_model.modules() if isinstance(m, HelixLinear))
        total_helix += sum(1 for m in self.clip_model.modules() if isinstance(m, HelixLinear))
        total_vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

        print(f"\n  Total: {total_helix} HelixLinear | {total_vram:.0f} MB VRAM")
        print(f"  Shared buffer: id={id(HelixLinear._shared_buffer)}")
        print(f"  Components: Soulfile + EchoMemory + FGIP + 3 lobes")
        print(f"\n  Ready on :{PORT}")
        print("=" * 60)

    def _build_chat_messages(self, messages, system_prefix=""):
        """Build ChatML message list, injecting system prefix and context."""
        chat_msgs = []
        if system_prefix:
            chat_msgs.append({"role": "system", "content": system_prefix})
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system" and chat_msgs and chat_msgs[0]["role"] == "system":
                chat_msgs[0]["content"] += "\n" + content
            else:
                chat_msgs.append({"role": role, "content": content})
        return chat_msgs

    def _enrich_with_context(self, messages, system_prefix):
        """Enrich system prompt with EchoMemory and FGIP context if relevant."""
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return system_prefix

        last_query = user_msgs[-1].get("content", "")
        context_parts = [system_prefix]

        # Memory recall: check if user is asking about past interactions
        if _MEMORY_RECALL_KEYWORDS.search(last_query):
            memories = self.memory.search_interactions(last_query, top_k=3)
            if memories and not any("error" in m for m in memories):
                context_parts.append("\n[EchoMemory — past interactions:]")
                for mem in memories:
                    context_parts.append(f"- Q: {mem['query'][:100]} → A: {mem['response'][:100]}")

        # Memory search: check seed memories for relevant context
        mem_results = self.memory.search(last_query, top_k=2)
        if mem_results and not any("error" in m for m in mem_results):
            context_parts.append("\n[EchoMemory — relevant knowledge:]")
            for mem in mem_results:
                context_parts.append(f"- {mem['text'][:200]}")

        # FGIP: check if query touches investment/supply chain topics
        if _FGIP_KEYWORDS.search(last_query):
            fgip_results = self.fgip.search(last_query, top_k=3)
            if fgip_results["nodes"]:
                context_parts.append("\n[FGIP graph — relevant intelligence:]")
                for node in fgip_results["nodes"]:
                    desc = f" — {node['description']}" if node["description"] else ""
                    context_parts.append(f"- {node['name']}{desc}")
                for edge in fgip_results["edges"][:3]:
                    context_parts.append(
                        f"- {edge['from']} → {edge['type']} → {edge['to']}: {edge['notes'][:100]}"
                    )

        return "\n".join(context_parts)

    def generate_language(self, messages, max_tokens=200, temperature=0.7):
        """Generate with Zamba2-7B (language lobe) using ChatML template."""
        # Enrich system prompt with memory/FGIP context
        enriched_prefix = self._enrich_with_context(messages, self.soul_prefix)
        chat_msgs = self._build_chat_messages(messages, enriched_prefix)
        prompt = self.zamba_tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.zamba_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.zamba_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                use_cache=True,
            )
        gen_time = time.time() - t0

        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = self.zamba_tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = truncate_at_loop(response)
        n_tokens = len(new_tokens)

        # Store interaction in EchoMemory
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs and response.strip():
            self.memory.store_interaction(
                user_msgs[-1].get("content", ""), response.strip(), "zamba2-7b-hxq",
            )

        return {
            "text": response.strip(),
            "tokens": n_tokens,
            "tok_per_sec": round(n_tokens / gen_time, 1) if gen_time > 0 else 0,
            "gen_time_s": round(gen_time, 2),
            "model": "zamba2-7b-hxq",
            "lobe": "language",
        }

    def generate_code(self, messages, max_tokens=300, temperature=0.3):
        """Generate with Qwen-Coder-3B (code lobe) using ChatML template."""
        chat_msgs = self._build_chat_messages(
            messages, "You are a coding assistant. Write clean, correct code.",
        )
        prompt = self.qwen_tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.qwen_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.qwen_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                use_cache=True,
            )
        gen_time = time.time() - t0

        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = self.qwen_tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = truncate_at_loop(response)
        n_tokens = len(new_tokens)

        # Store interaction
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs and response.strip():
            self.memory.store_interaction(
                user_msgs[-1].get("content", ""), response.strip(), "qwen-coder-3b-hxq",
            )

        return {
            "text": response.strip(),
            "tokens": n_tokens,
            "tok_per_sec": round(n_tokens / gen_time, 1) if gen_time > 0 else 0,
            "gen_time_s": round(gen_time, 2),
            "model": "qwen2.5-coder-3b-hxq",
            "lobe": "code",
        }

    def classify_image(self, image, labels):
        """Zero-shot image classification with CLIP (vision lobe)."""
        text_prompts = [f"a photo of a {label}" for label in labels]
        inputs = self.clip_processor(
            text=text_prompts, images=image,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.clip_device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        infer_time = time.time() - t0

        probs = outputs.logits_per_image.softmax(dim=-1)[0].cpu().tolist()
        results = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)

        return {
            "classifications": [{"label": l, "score": round(s, 4)} for l, s in results],
            "infer_time_s": round(infer_time, 3),
            "model": "clip-vit-l14-hxq",
            "lobe": "vision",
        }

    def embed_image(self, image):
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.clip_device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def embed_text(self, text):
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.clip_device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()


def main():
    server = RobotBrainServer()

    try:
        from fastapi import FastAPI, File, Form, UploadFile
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        print("\nfastapi/uvicorn not installed. Running in REPL mode.\n")
        while True:
            try:
                query = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in ("quit", "exit", "q"):
                break
            lobe = route_input([{"role": "user", "content": query}])
            if lobe == "code":
                result = server.generate_code([{"role": "user", "content": query}])
            else:
                result = server.generate_language([{"role": "user", "content": query}])
            print(f"[{result['lobe']}] [{result['tok_per_sec']} tok/s]: {result['text']}\n")
        return

    app = FastAPI(title="Echo Robot Brain", version="0.1.0")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 200)
        temperature = request.get("temperature", 0.7)
        explicit_model = request.get("model")

        lobe = route_input(messages, explicit_model)

        if lobe == "code":
            result = server.generate_code(messages, max_tokens, temperature)
        else:
            result = server.generate_language(messages, max_tokens, temperature)

        return JSONResponse({
            "id": f"echo-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result["model"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }],
            "usage": {
                "completion_tokens": result["tokens"],
                "total_tokens": result["tokens"],
            },
            "echo_metadata": {
                "lobe": result["lobe"],
                "tok_per_sec": result["tok_per_sec"],
                "routed_by": "explicit" if explicit_model else "auto",
            },
        })

    @app.post("/v1/vision/classify")
    async def vision_classify(
        image: UploadFile = File(...),
        labels: str = Form(...),
    ):
        from PIL import Image
        import io
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        label_list = json.loads(labels)
        result = server.classify_image(img, label_list)
        return JSONResponse(result)

    @app.post("/v1/vision/embed")
    async def vision_embed(image: UploadFile = File(...)):
        from PIL import Image
        import io
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        embedding = server.embed_image(img)
        return JSONResponse({"embedding": embedding, "dim": len(embedding), "model": "clip-vit-l14-hxq"})

    @app.post("/v1/embeddings")
    async def text_embed(request: dict):
        text = request.get("input", "")
        embedding = server.embed_text(text)
        return JSONResponse({
            "object": "list",
            "data": [{"object": "embedding", "embedding": embedding, "index": 0}],
            "model": "clip-vit-l14-hxq",
        })

    @app.post("/v1/memory/search")
    async def memory_search(request: dict):
        query = request.get("query", "")
        top_k = request.get("top_k", 5)
        seeds = server.memory.search(query, top_k)
        interactions = server.memory.search_interactions(query, top_k)
        return JSONResponse({"seeds": seeds, "interactions": interactions})

    @app.post("/v1/memory/store")
    async def memory_store(request: dict):
        query = request.get("query", "")
        response = request.get("response", "")
        server.memory.store_interaction(query, response)
        return JSONResponse({"status": "stored"})

    @app.post("/v1/fgip/search")
    async def fgip_search(request: dict):
        query = request.get("query", "")
        top_k = request.get("top_k", 5)
        results = server.fgip.search(query, top_k)
        return JSONResponse(results)

    @app.get("/v1/models")
    async def list_models():
        vram = round(torch.cuda.memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else 0
        return {
            "data": [
                {"id": "zamba2-7b-hxq", "object": "model", "lobe": "language",
                 "architecture": "hybrid (Mamba2 + shared Transformer)", "compression": "HXQ VQ2D (6 bits/weight)"},
                {"id": "qwen2.5-coder-3b-hxq", "object": "model", "lobe": "code",
                 "architecture": "decoder-only Transformer", "compression": "HXQ (8 bits/weight)"},
                {"id": "clip-vit-l14-hxq", "object": "model", "lobe": "vision",
                 "architecture": "dual-encoder (ViT-L/14 + Text Transformer)", "compression": "HXQ (8 bits/weight)"},
            ],
            "components": {
                "echo_memory": server.memory.available,
                "fgip": server.fgip.available,
                "soulfile": bool(server.soul_prefix),
            },
            "total_vram_mb": vram,
            "shared_buffer_id": id(HelixLinear._shared_buffer),
        }

    @app.get("/health")
    async def health():
        vram = round(torch.cuda.memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else 0
        return {
            "status": "ok",
            "vram_mb": vram,
            "lobes": {"language": "zamba2-7b-hxq", "code": "qwen2.5-coder-3b-hxq", "vision": "clip-vit-l14-hxq"},
            "components": {
                "echo_memory": server.memory.available,
                "fgip": server.fgip.available,
                "soulfile": bool(server.soul_prefix),
            },
        }

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()

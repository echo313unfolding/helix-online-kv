# helix-online-kv

Online KV cache compression using learned VQ codebooks. Fit codebooks on the first 128 tokens, then VQ-assign all subsequent KV entries in real time. 1.9x more tokens fit in the same VRAM. Compressed-domain attention (CDC-03) skips 87.5% of tokens per layer.

Same codec family as [helix-substrate](https://github.com/voidstr3m33/helix-substrate) weight compression — codebook + uint8 indices — applied to activations instead of weights.

```bash
pip install helix-online-kv[torch]
```

```python
from helix_online_kv import CompressedKVCache, OnlineKVConfig

config = OnlineKVConfig(calibration_tokens=128, hot_window=256, exact_layers=[0])
cache = CompressedKVCache(config=config, n_layers=22)

# Drop into HuggingFace generate() -- subclasses DynamicCache
output = model.generate(input_ids, past_key_values=cache, max_new_tokens=256)
```

## Benchmarks

### KV Cache Compression (TinyLlama 1.1B, Quadro T2000)

| Configuration | PPL | PPL Delta | VRAM Peak | Calibration |
|--------------|-----|-----------|-----------|-------------|
| Dense FP16 + standard KV | 9.607 | baseline | — | — |
| Dense FP16 + compressed KV | 9.607 | **+0.00%** | — | 128 tokens (online) |
| **HelixLinear + compressed KV** | **9.680** | **+0.77%** | **1329 MB** | **128 tokens (online)** |

The compressed KV cache adds **zero measurable PPL degradation** on its own. The +0.77% comes entirely from weight compression. The codebook at k=256 with per-layer fitting is precise enough that reconstruction error falls below noise floor.

### Attention Compute Reduction (CDC-03)

| Method | Output Cosine | Tokens Touched | Compute vs Dense | Calibration |
|--------|--------------|----------------|-----------------|-------------|
| Standard attention | 1.000 | 100% | 1.0x | — |
| KIVI/KVQuant (INT2) | — | 100% | ~1.0x | Offline |
| H2O / Scissorhands | — | ~20% (evicted) | ~0.2x | Activation stats |
| **CDC-03 (ours)** | **0.9997** | **12.5%** | **~0.12x** | **128 tokens (online)** |

\* CDC-03 compute ratio is theoretical op count at seq_len=1024. Fused kernel not yet built. Other methods' fidelity numbers not directly comparable (different models/settings).

**Key difference:** KIVI/KVQuant reduce storage. H2O/Scissorhands reduce compute by dropping tokens permanently. CDC-03 reduces both — all tokens remain accessible via PQ approximate scoring, but only the top 128 get exact attention.

### Memory Scaling

| Seq Length | Dense FP16 | Compressed (Tier 0=256) | Savings |
|-----------|-----------|------------------------|---------|
| 256 | 5.5 MB | 4.2 MB | 1.3x |
| 1K | 22.0 MB | 12.9 MB | 1.7x |
| 4K | 88.0 MB | 47.4 MB | **1.9x** |
| 16K | 352.0 MB | 185.4 MB | **1.9x** |
| 128K | 2.75 GB | 1.45 GB | **1.9x** |

Per-model (22 layers). Tier 0 hot window = 256 tokens. Savings converge to 1.9x as sequence length dominates hot window.

## Key findings

**KV compression is free at k=256.** Six hyperparameter configurations tested (calibration tokens ∈ {64, 128}, hot window ∈ {64, 128, 256}). Every single one produces identical PPL to baseline. The VQ codebook captures the KV distribution so precisely that the error is below measurement noise. Receipt: `receipts/ppl_sweep_20260325T204346.json`.

**128 tokens of calibration generalize perfectly.** Codebooks fit on a science prompt achieve cos >= 0.999 when applied to cooking and history prompts. Cross-domain generalization works because KV value distributions are dominated by layer-specific structure, not content. Receipt: `receipts/codebook_generalization_20260325T202936.json`.

**Real attention patterns are easier than random.** Random synthetic KV data gives worse CDC-03 fidelity because attention is uniformly distributed. Real transformers concentrate attention on a few tokens per head — exactly the structure CDC-03 exploits. This is why 12.5% coverage (top-128 of 1024) achieves 0.9997 cosine: the tokens CDC-03 skips have near-zero attention weight anyway.

**Layer 0 is special and must stay exact.** Layer 0 V has kurtosis 36.1 (vs < 5 for all other layers). It carries the attention sink pattern. Compressing layer 0 drops CDC-03 fidelity from 0.9997 to 0.986. Excluding it costs nothing — one layer of 22 at full precision.

**Seven attention paths were tested. Only one survived.** Scalar VQ decompress (Path A), vector VQ (B), full PQ (D), and prefiltered (F) all failed fidelity gates at long sequences. Hybrid PQ with sparse exact V (Path G / CDC-03) is the only architecture that maintains cos > 0.99 at 1024 tokens. The key insight: approximate the *scores*, but compute exact attention on the winners.

## How it works

1. **Calibration phase** (first 128 tokens): Accumulate K/V scalars, fit per-layer 256-entry codebooks via k-means.
2. **Streaming phase** (subsequent tokens): Each new K/V scalar assigned to nearest centroid (~139 μs/layer). Only uint8 index stored.
3. **Aging**: Recent tokens stay exact in Tier 0 (FP16). Older tokens age into Tier 1 (uint8 VQ indices). Hot window is configurable.

```
Token arrives
     │
     ▼
CompressedKVCache.update()
     │
     ├── Calibrating? ──▶ accumulate, fit codebook at N tokens
     │
     ├── Streaming? ──▶ assign to nearest centroid (uint8 index)
     │
     ├── Aging: hot_window recent tokens stay FP16 (Tier 0)
     │          older tokens demoted to uint8 indices (Tier 1)
     │
     ▼
Attention (per layer)
     │
     ├── Layer 0 (exact)? ──▶ standard Q @ K.T @ V
     │
     ├── Calibration not done? ──▶ standard Q @ K.T @ V
     │
     └── CDC-03 hybrid:
            1. PQ distance table: M=16 subspaces × 256 centroids
            2. Approximate score for ALL tokens via table lookup
            3. Select top_k=128 + 4 sink tokens
            4. Exact Q @ K[selected].T → softmax → @ V[selected]
            5. Unselected tokens get weight ≈ 0 (masked to -inf)
```

## CDC-03: Compressed-domain attention

```
Standard attention:  O(n × d)     -- touch every token
CDC-03 attention:    O(M × 256)   -- PQ distance table (fixed, ~16K ops)
                   + O(n × M)     -- approximate score per token
                   + O(k × d)     -- exact attention on top-k only
```

**Proven metrics (real WikiText-2 KV data, seq=1024, TinyLlama):**

| Metric | Value | Gate |
|--------|-------|------|
| Output cosine (mean) | **0.99973** | >= 0.99 |
| Output cosine (min) | 0.97689 | >= 0.95 |
| Entries at cos >= 0.9999 | **211 / 252** (83.7%) | — |
| Entries below cos 0.95 | 1 / 252 (0.4%) | — |
| Coverage | 12.5% (128 + 4 sink of 1024) | — |

3 WikiText-2 prompts × 21 layers × 4 heads = 252 entries. All on real KV data from TinyLlama inference, not synthetic.

**Projected compute savings (per-head, per-layer K-side dot products):**

| Seq Length | Dense Ops | Hybrid Ops (top_k=128) | Savings |
|-----------|-----------|----------------------|---------|
| 1K | 65K | 8.2K | **8x** |
| 4K | 262K | 8.2K | **29x** |
| 16K | 1M | 8.2K | **128x** |
| 128K | 8.4M | 8.2K | **900x** |

Does not include PQ table precompute (~16K ops fixed per query). Savings are theoretical op counts from the proven fidelity regime, not measured wall-clock speedup.

### Attention path comparison

Seven implementations tested. CDC-03 (Path G) is the only one that passes all gates at long sequences.

| Path | Method | K-side | V-side | Fidelity | Status |
|------|--------|--------|--------|----------|--------|
| A | Scalar VQ decompress | Decompress | Decompress | Baseline | Reference |
| B | Vector VQ scores + gather | Compressed | Decompress | cos 0.96 | Failed at 1K |
| D | Full PQ (K + V compressed) | PQ tables | PQ reconstruct | cos 0.98 | Failed at 1K |
| F | Prefiltered (coarse + dense) | VQ clusters | Exact on subset | cos 0.99 | Borderline |
| **G (CDC-03)** | **Hybrid PQ + sparse exact** | **PQ tables** | **Sparse exact** | **cos 0.9997** | **Production** |

## Positioning against KIVI / KVQuant

| | KIVI / KVQuant | helix-online-kv |
|---|---|---|
| **Method** | 2-bit uniform scalar quantization | 8-bit VQ with learned codebooks |
| **Compression** | 8x+ | 1.9x |
| **Calibration** | Offline (requires calibration data) | Online (fit from first 128 tokens) |
| **Quality** | Calibration-dependent, model-specific | **0% PPL delta** across all configs tested |
| **Compute reduction** | Storage only | Storage + CDC-03 attention sparsity |
| **Codec sharing** | Standalone | Same codec as weight compression |

They compress harder. We compress smarter — codebooks capture distribution shape rather than uniform binning, and CDC-03 adds compute reduction on top of storage reduction. One codec for weights and KV cache.

## Proof chain

Every claim traces to a receipt with a cost block (WO-RECEIPT-COST-01). All receipts in `receipts/`.

| Phase | What | Key Metric | Receipt |
|-------|------|------------|---------|
| 1. Codebook generalization | 128-token calibration generalizes across prompts | cos >= 0.999 all 22 layers | `codebook_generalization_*.json` |
| 2. Encoder latency | Per-token VQ overhead (21 compressed layers) | 2.81 ms/token (p99: 2.91 ms) | `online_encoder_*.json` |
| 3. Memory savings | Tiered compression ratio at 16K tokens | 1.9x | `tiered_memory_*.json` |
| 3b. PPL sensitivity | Sweep cal/hot/clusters (6 configs) | **0% PPL delta** | `ppl_sweep_*.json` |
| 4. End-to-end | HelixLinear + CompressedKVCache on GPU | +0.77% PPL, 1329 MB | `e2e_compressed_*.json` |
| CDC-03 | Hybrid PQ attention, layers 1-21 | cos=0.9997, 12.5% coverage | `cdc03_hybrid_layers1_21_*.json` |

## What's honest

**1.9x is the memory number, not 4x.** 8-bit VQ indices vs 16-bit FP16 gives a 2x theoretical ceiling. With Tier 0 hot window and codebook overhead, the measured number at 16K tokens is 1.9x.

**Encoder latency is 2.81ms for 21 layers — fast enough for generation but not free.** Per-layer assignment is 139 μs. Calibration finalize is a one-time 450ms cost per layer.

**CDC-03 projected savings are op counts, not wall-clock.** The fidelity is proven (cos=0.9997 on real data). The speedup requires a fused kernel that scores via PQ tables and gathers only selected tokens — not yet built.

**Scaling projections are extrapolations.** The 29x at 4K and 900x at 128K assume the same 12.5% coverage ratio holds at longer contexts. Real long-context behavior has not been measured yet.

**The Triton fused kernel (Path A) is proof-of-concept.** CDC-03 (Path G) uses PyTorch ops. A fused CDC-03 kernel would need PQ table lookup + top-k selection + sparse gather in one pass.

## API reference

### OnlineKVConfig

```python
@dataclass
class OnlineKVConfig:
    calibration_tokens: int = 128    # tokens to accumulate before fitting codebook
    hot_window: int = 256            # recent tokens kept in exact FP16 (Tier 0)
    n_clusters: int = 256            # codebook size (max 256 for uint8)
    max_kmeans_iters: int = 10       # k-means iterations during calibration
    rtol: float = 0.001              # relative convergence tolerance
    drift_threshold: float = 0.01    # MSE threshold for centroid EMA update
    drift_ema_alpha: float = 0.05    # EMA smoothing factor
    exact_layers: list[int] = [0]    # layers kept exact
```

### CompressedKVCache

Subclass of `transformers.DynamicCache`. Drop-in replacement for HuggingFace `past_key_values`.

```python
cache = CompressedKVCache(config, n_layers=22)

# Properties
cache.total_tokens         # cumulative token count
cache.calibration_complete # all non-exact layers have codebooks

# Memory reporting
cache.memory_report()
# → {"exact_bytes": 8_380_000, "compressed_bytes": 666_600,
#    "total_bytes": 9_046_600, "n_layers": 22,
#    "total_tokens": 1024, "calibration_complete": True}
```

### ProductCodebook

For CDC-03 compressed-domain attention scoring.

```python
from helix_online_kv import ProductCodebook

pq = ProductCodebook(dim=64, m=16, k=256)  # 16 subspaces, 256 centroids each
pq.fit(key_vectors)                         # fit on calibration KV data
scores = pq.asymmetric_score(query)         # approximate q@K.T via PQ tables
```

### Three codebook variants

| Codebook | Dimensionality | Use case | Fidelity |
|----------|---------------|----------|----------|
| `OnlineCodebook` | Scalar (1D) | Per-element VQ, cheapest | cos > 0.999 |
| `VectorCodebook` | head_dim (64D) | Per-token VQ, better K fidelity | cos > 0.99 |
| **`ProductCodebook`** | **M subspaces of sub_dim** | **PQ scoring for CDC-03** | **cos 0.9997** |

All three follow the same lifecycle: `feed_calibration()` → `finalize_calibration()` → `assign()` / `decode()`.

## Quick start

### Run tests

```bash
make test
```

### Prove codebook generalization

```bash
make prove   # calibrate on one prompt, test on others
```

### Benchmark CDC-03

```bash
make dump-long                # extract 1024-token KV caches from WikiText-2
make bench-attention-cdc03    # hybrid PQ + prefilter on real data
```

### End-to-end generation

```bash
make e2e    # HelixLinear + CompressedKVCache, measures PPL + VRAM
```

## Project structure

```
helix-online-kv/
├── helix_online_kv/
│   ├── config.py               # OnlineKVConfig dataclass
│   ├── codebook.py             # Scalar VQ (1D k-means)
│   ├── vector_codebook.py      # Vector VQ (head_dim-D k-means)
│   ├── product_codebook.py     # Product quantization (M subspaces)
│   ├── aging_policy.py         # Tier 0/1 token aging
│   ├── layer_state.py          # Per-layer compression state machine
│   ├── compressed_cache.py     # Drop-in DynamicCache replacement
│   ├── compressed_attention.py # Paths A–G (CDC-03 = Path G)
│   └── triton_attention.py     # Fused Triton kernel (Path A, PoC)
├── tools/                      # Benchmarking + proof scripts
├── receipts/                   # 15 proof receipts with cost blocks
└── tests/
```

## Companion projects

### [helix-substrate](https://github.com/voidstr3m33/helix-substrate)

Calibration-free weight compression. 4x smaller weights with <1% quality loss. Same VQ codec, applied to model parameters instead of activations. HelixLinear is the drop-in nn.Linear replacement.

### [echo_runtime](https://github.com/voidstr3m33/echo-runtime)

Unified inference runtime wiring HelixLinear + CompressedKVCache + CDC-03 into one forward pass. 155 compressed weight modules + 22-layer KV cache + 21 hybrid attention layers. 1326 MB on Quadro T2000.

## Prior art

- **KIVI (Liu et al., 2024)** — Per-channel INT2/4 KV cache quantization. We differ: VQ codebooks (non-uniform), online calibration, compressed-domain attention.
- **KVQuant (Hooper et al., 2024)** — Per-channel quantization with outlier handling. We differ: VQ instead of linear quantization, no offline calibration pass.
- **Gear (Kang et al., 2024)** — Grouped-head KV cache quantization. We differ: per-layer codebook learning, product quantization for attention scoring.
- **H2O / Scissorhands** — KV cache eviction (drop tokens). We differ: compress, don't drop. All tokens remain accessible via approximate scoring.

## License

Echo Labs LLC. See LICENSE for details.

## Citation

If you use helix-online-kv in research, please cite:

```
@software{helix_online_kv,
  author = {Josh (voidstr3m33)},
  title = {helix-online-kv: Online KV cache compression with compressed-domain attention},
  year = {2026},
  url = {https://github.com/voidstr3m33/helix-online-kv}
}
```

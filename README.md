# helix-online-kv

Online KV cache compression using learned VQ codebooks. Fit codebooks on the first 128 tokens, then VQ-assign all subsequent KV entries in real time. 1.9x more tokens fit in the same VRAM. Compressed-domain attention (CDC-03) skips 87.5% of tokens per layer.

Same codec family as [helix-substrate](https://github.com/voidstr3m33/helix-substrate) weight compression — codebook + uint8 indices — applied to activations instead of weights.

```bash
pip install helix-online-kv[torch]
```

```python
from helix_online_kv import CompressedKVCache, OnlineKVConfig

config = OnlineKVConfig(
    calibration_tokens=128,  # fit codebooks on first 128 tokens
    hot_window=256,          # last 256 tokens stay FP16 (Tier 0)
    n_clusters=256,          # uint8 VQ codebook size
    exact_layers=[0],        # layer 0 stays exact (attention sink)
)

cache = CompressedKVCache(config=config, n_layers=22)

# Drop into HuggingFace generate() -- subclasses DynamicCache
output = model.generate(input_ids, past_key_values=cache, max_new_tokens=256)
```

## How it works

Standard KV caches store every key and value vector in FP16 — 2 * n_layers * seq_len * n_heads * head_dim * 2 bytes. At 4K context on a 7B model, that's ~1 GB of VRAM just for the cache.

helix-online-kv replaces this with a calibrate-then-stream pipeline:

1. **Calibration phase** (first 128 tokens): Accumulate K/V vectors, fit per-layer codebooks via k-means.
2. **Streaming phase** (subsequent tokens): Each new K/V vector assigned to nearest centroid (~139 μs/layer). Only uint8 index stored for Tier 1 tokens.
3. **Aging**: Recent tokens stay exact in Tier 0 (FP16). Older tokens age into Tier 1 (compressed uint8). Hot window is configurable.

Layer 0 is always kept exact (attention sink pattern — layer 0 carries disproportionate KV importance, kurtosis 36.1).

```
Token stream
  │
  ▼
┌──────────────────┐
│  Calibration     │  Accumulate first 128 tokens
│  (k-means fit)   │  per non-exact layer
└────────┬─────────┘
         │ codebook ready
         ▼
┌──────────────────┐     ┌─────────────────┐
│  Tier 0 (hot)    │────▶│  Tier 1 (cold)  │
│  FP16 exact      │ age │  uint8 VQ index  │
│  last 256 tokens │ out │  + codebook      │
└──────────────────┘     └─────────────────┘
         │                        │
         ▼                        ▼
┌────────────────────────────────────────┐
│  Attention forward pass                │
│  Layer 0: exact (attention sink)       │
│  Layers 1-21: CDC-03 hybrid PQ        │
└────────────────────────────────────────┘
```

## Proof chain

Every claim traces to a receipt with a cost block (WO-RECEIPT-COST-01). All receipts in `receipts/`.

| Phase | What | Key Metric | Receipt |
|-------|------|------------|---------|
| 1. Codebook generalization | 128-token calibration generalizes across prompts | cos >= 0.999 across 22 layers | `codebook_generalization_20260325T202936.json` |
| 2. Encoder latency | Per-token VQ overhead (21 compressed layers) | 2.81 ms/token (p99: 2.91 ms) | `online_encoder_20260325T203212.json` |
| 3. Memory savings | Tiered compression ratio at 16K tokens | 1.9x | `tiered_memory_20260325T203218.json` |
| 3b. PPL sensitivity | Sweep cal∈{64,128}, hot∈{64,128,256}, k=256 | 0% PPL delta across all 6 configs | `ppl_sweep_20260325T204346.json` |
| 4. End-to-end | HelixLinear + CompressedKVCache on GPU | +0.77% PPL, 1329 MB on T2000 | `e2e_compressed_20260325T205041.json` |
| CDC-03 | Hybrid PQ attention, layers 1-21 | cos=0.9997, 12.5% coverage | `cdc03_hybrid_layers1_21_20260326T100511.json` |

## CDC-03: Compressed-domain attention

The real savings come from never decompressing. CDC-03 scores all tokens cheaply using product quantization distance tables, selects the top 128 by approximate score, then runs exact attention on only the selected subset.

```
Standard attention:  O(n * d)     -- touch every token, full precision
CDC-03 attention:    O(M * 256)   -- precompute PQ distance table
                   + O(n * M)     -- approximate score per token
                   + O(k * d)     -- exact attention on top-k only
```

**Layer routing:** Layer 0 uses exact attention (attention sink pattern). Layers 1-21 use hybrid PQ.

**Proven metrics (real WikiText-2 KV data, seq=1024, TinyLlama):**

| Metric | Standard | CDC-03 | Gate |
|--------|----------|--------|------|
| Output cosine (mean) | 1.0 | 0.9997 | >= 0.99 |
| Output cosine (min) | 1.0 | 0.977 | >= 0.95 |
| Tokens touched | 1024 | 128 + 4 sink | 87% reduction |

Tested on 3 WikiText-2 prompts × 21 layers × 4 heads = 252 entries. Distribution: 211/252 at cos >= 0.9999, only 1 below 0.95. All on real KV data from TinyLlama inference, not synthetic.

**Projected compute savings (per-head, per-layer K-side dot products):**

| Seq Length | Dense Ops | Hybrid Ops (top_k=128) | Savings |
|-----------|-----------|----------------------|---------|
| 1K | 65K | 8.2K | 8x |
| 4K | 262K | 8.2K | 29x |
| 16K | 1M | 8.2K | 128x |
| 128K | 8.4M | 8.2K | 900x |

Does not include PQ table precompute (~16K ops fixed per query). Savings are theoretical op counts from the proven fidelity regime, not measured wall-clock speedup.

Receipt: `receipts/cdc03_hybrid_layers1_21_20260326T100511.json`

### Attention path comparison

Seven attention implementations were tested during development. CDC-03 (Path G) is the production path.

| Path | Method | K-side | V-side | Fidelity |
|------|--------|--------|--------|----------|
| A | Scalar VQ decompress | Decompress | Decompress | Baseline |
| B | Vector VQ scores + gather | Compressed | Decompress | cos > 0.99 |
| D | Full PQ (K + V compressed) | PQ tables | PQ reconstruct | cos > 0.98 |
| F | Prefiltered (coarse rank + dense) | VQ clusters | Exact on subset | cos > 0.99 |
| **G (CDC-03)** | **Hybrid PQ + prefilter** | **PQ tables** | **Sparse exact** | **cos 0.9997** |

## Benchmarks

### End-to-end with HelixLinear (TinyLlama 1.1B, Quadro T2000)

| Configuration | PPL | PPL Delta | VRAM Peak |
|--------------|-----|-----------|-----------|
| Dense + standard KV | 9.607 | baseline | — |
| Dense + compressed KV | 9.607 | +0.00% | — |
| **HelixLinear + compressed KV** | **9.680** | **+0.77%** | **1329 MB** |

The compressed KV cache adds zero measurable PPL degradation on its own. The +0.77% comes entirely from weight compression.

Receipt: `receipts/e2e_compressed_20260325T205041.json`

### Codebook quality

| Phase | Metric | Result | Gate |
|-------|--------|--------|------|
| Calibration | Generalizes to unseen prompts | cos >= 0.999, cross-prompt validated | PASS |
| Encoder | Per-token assignment (21 layers) | 2.81 ms (p99: 2.91 ms) | < 5 ms |
| Memory | Tier 0 + Tier 1 savings at 16K | 1.9x | > 1.5x |
| PPL | Degradation across 6 configs | 0% delta | < 2% |

## Positioning against KIVI/KVQuant

| | KIVI/KVQuant | helix-online-kv |
|---|---|---|
| Method | 2-bit uniform scalar quantization | 8-bit VQ with learned codebooks |
| Compression | 8x+ | 1.9x |
| Calibration | Calibration-dependent (offline) | Calibration-free (codebooks fit online from first 128 tokens) |
| Compute reduction | Storage only | Storage + CDC-03 attention sparsity |
| Codec sharing | Standalone | Same codec as weight compression (helix-substrate) |

They compress harder. We compress smarter — codebooks capture distribution shape rather than uniform binning, and CDC-03 adds compute reduction on top of storage reduction. One codec for weights and KV cache.

## What's honest

**1.9x is the memory number, not 4x.** 8-bit VQ indices vs 16-bit FP16 gives a 2x theoretical ceiling. With Tier 0 hot window and codebook overhead, the measured number at 16K tokens is 1.9x.

**KV compression alone adds zero measurable PPL loss.** The codebook at k=256 with per-layer fitting is precise enough that the reconstruction error is below noise floor. All measured PPL degradation in the E2E benchmark comes from weight compression, not KV compression.

**Encoder latency is 2.81ms for 21 layers — fast enough for generation but not free.** Per-layer assignment is 139 μs. Calibration finalize is a one-time 450ms cost per layer.

**CDC-03 fidelity is proven on real data, not synthetic.** The cos=0.9997 result is from WikiText-2 KV caches extracted during actual TinyLlama inference, tested on 252 (prompt, layer, head) combinations. Random synthetic data gives worse cosines because attention is uniformly distributed — real transformers have concentrated attention patterns that CDC-03 exploits.

**CDC-03 projected savings are op counts, not wall-clock.** The fidelity is proven. The speedup requires a fused kernel that scores via PQ tables and gathers only selected tokens — not yet built.

**Scaling projections are extrapolations.** The 29x at 4K and 900x at 128K assume the same 12.5% coverage ratio holds at longer contexts. Real long-context behavior has not been measured yet.

**Layer 0 requires exact attention.** The attention sink pattern (kurtosis 36.1 on layer 0 V) defeats VQ and sparse selection. Layer 0 is excluded from all compression.

**The Triton fused kernel (Path A) is proof-of-concept.** It demonstrates 4x bandwidth reduction for scalar VQ attention, but CDC-03 (Path G) uses PyTorch ops. A fused CDC-03 kernel would need PQ table lookup + top-k selection + sparse gather in one pass.

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

Subclass of `transformers.DynamicCache`. Drop-in replacement.

```python
cache = CompressedKVCache(config, n_layers=22)

# Properties
cache.total_tokens         # cumulative token count
cache.calibration_complete # all non-exact layers have codebooks

# Memory reporting
cache.memory_report()
# → {"exact_bytes": ..., "compressed_bytes": ..., "total_bytes": ...,
#    "n_layers": 22, "total_tokens": 1024, "calibration_complete": True}
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
| `OnlineCodebook` | Scalar (1D) | Per-element VQ, cheapest | Good for values |
| `VectorCodebook` | head_dim (64D) | Per-token VQ, better K fidelity | cos > 0.99 |
| `ProductCodebook` | M subspaces of sub_dim | PQ scoring for attention | cos 0.9997 |

All three follow the same lifecycle: `feed_calibration()` → `finalize_calibration()` → `assign()` / `decode()`.

## Tests

```bash
make test   # full test suite
pytest tests/ -v
```

### Make targets

```bash
make prove                   # Codebook generalization proof
make dump-long               # Extract 1024-token KV caches from WikiText-2
make bench-attention-cdc03   # Hybrid PQ + prefilter on real data
make e2e                     # HelixLinear + CompressedKVCache, measures PPL
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
│   ├── compressed_attention.py # Paths A, B, D, F, G (CDC-03)
│   └── triton_attention.py     # Fused Triton kernel (Path A)
├── tools/
│   ├── prove_hybrid_layers1_21.py   # CDC-03 final proof
│   ├── bench_compressed_attention.py # All-paths benchmark
│   └── e2e_compressed_generation.py  # End-to-end PPL measurement
├── receipts/                    # 15 proof receipts with cost blocks
└── tests/                       # Unit tests
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

.PHONY: test prove bench bench-latency bench-memory bench-ppl e2e lint clean

# Run all tests
test:
	python3 -m pytest tests/ -v

# Phase 1: Prove codebook calibration generalizes
prove:
	python3 tools/calibrate_codebook_generalization.py

# Phase 2: Benchmark online encoder latency
bench-latency:
	python3 tools/bench_online_encoder.py

# Phase 3: Benchmark tiered memory savings
bench-memory:
	python3 tools/bench_tiered_memory.py

# Phase 3: PPL sweep
bench-ppl:
	python3 tools/bench_ppl_sweep.py

# Phase 4: End-to-end (HelixLinear + CompressedKVCache)
e2e:
	python3 tools/e2e_compressed_generation.py

# Phase CDC-01: Compressed-domain attention benchmark
bench-attention:
	python3 tools/bench_compressed_attention.py

# CDC-01.1: Generate long (1024-token) KV dumps from WikiText-2
dump-long:
	python3 tools/dump_kv_cache_long.py

# CDC-01.1: Benchmark on long sequences
bench-attention-long: dump-long
	python3 tools/bench_compressed_attention.py --kv-dir ~/helix-substrate/kv_dump_long --tag long

# CDC-02: All new approaches (PQ + fused kernel + prefilter)
bench-attention-cdc02:
	python3 tools/bench_compressed_attention.py \
	  --kv-dir ~/helix-substrate/kv_dump_long --tag cdc02 --parts d,e,f

# CDC-02: PQ only (fastest test)
bench-attention-pq:
	python3 tools/bench_compressed_attention.py \
	  --kv-dir ~/helix-substrate/kv_dump_long --tag pq --parts d

# CDC-03: Hybrid PQ + prefilter (production path)
bench-attention-cdc03:
	python3 tools/bench_compressed_attention.py \
	  --kv-dir ~/helix-substrate/kv_dump_long --tag cdc03 --parts g

# All benchmarks
bench: bench-latency bench-memory bench-ppl

# Lint
lint:
	python3 -m py_compile helix_online_kv/codebook.py
	python3 -m py_compile helix_online_kv/config.py
	python3 -m py_compile helix_online_kv/layer_state.py
	python3 -m py_compile helix_online_kv/aging_policy.py
	python3 -m py_compile helix_online_kv/compressed_cache.py
	python3 -m py_compile helix_online_kv/vector_codebook.py
	python3 -m py_compile helix_online_kv/compressed_attention.py
	python3 -m py_compile helix_online_kv/product_codebook.py
	python3 -m py_compile helix_online_kv/triton_attention.py

clean:
	rm -rf __pycache__ helix_online_kv/__pycache__ tests/__pycache__
	rm -rf *.egg-info build dist

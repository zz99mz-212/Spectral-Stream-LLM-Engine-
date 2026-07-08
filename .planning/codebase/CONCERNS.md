# Codebase Concerns

**Analysis Date:** 2026-07-07

> Focus: Technical debt, bugs, security, performance, fragility, scaling limits, dependencies at risk, missing features, and TEST COVERAGE GAPS for the Spectral-Stream LLM Engine.
> Context: This repo has a documented history of inflated/non-honest metrics (`c66016e` "replace fabricated metrics with true end-to-end compression measurements", `b213e4f` "honest-metrics-windows-compat"). This audit specifically interrogates whether reported numbers are REAL end-to-end measurements or proxies/estimates, and whether the flagship compression methods actually work.

---

## Tech Debt

### TD-01 — 2964 registered "methods" but production full-model run uses only 2 (HIGH)
- **Issue:** The engine registers **2964 compression methods** (`final_compression_log.txt` line: `Registered compression method: ...` then `Discovered and registered 2964 methods`). However, the honest full-model run (`run_full_model_honest_results.json`) used only `fp16_passthrough` (1429/2011 tensors = 71%) and `int8_blockwise+zlib` (582/2011 = 29%). 2962 of 2964 methods are never exercised by real compression.
- **Files:** `spectralstream/compression/engine/_orchestrator.py`, `spectralstream/compression/methods/` (thousands of stub/unused modules), `final_compression_log.txt:3-4`, `run_full_model_honest_results.json` (`method` distribution).
- **Impact:** Massive hidden surface area (3479 `.py` files under `spectralstream/`), slow import/startup (`final_compression_log.txt` shows method registration taking ~2s and `Engine ready: 2964 methods`), and maintenance burden. Most "methods" are aspirational/non-functional and mislead about capability.
- **Fix approach:** Prune or quarantine unused methods into a clearly-labeled `experimental/` namespace; keep only methods with passing tests + demonstrated real-weight results in the active registry. Add a test asserting the method registry size stays bounded.

### TD-02 — Broken walk-based method auto-discovery (MEDIUM)
- **Issue:** `final_compression_log.txt:1` logs `WARNING Cannot import METHOD_CLASSES` then `Walk-discovered 0 methods`, then falls back to `Registered 1617 methods from METHOD_CLASSES`. The `_discover_by_walk()` fallback returns an empty dict — discovery by directory walking is dead/broken.
- **Files:** `spectralstream/compression/engine/method_discovery.py:143-187` (`_discover_by_walk`), called at `method_discovery.py:66`.
- **Impact:** The documented "auto-discovery" mechanism does not work; the registry depends entirely on the hard-coded `METHOD_CLASSES` import succeeding. If that import fails, the engine silently registers 0 methods and proceeds to fail later or pass through.
- **Fix approach:** Either fix `_discover_by_walk` to match the actual directory layout, or remove it and the misleading "Walk-discovered 0 methods" log.

### TD-03 — `fp16_passthrough` expansion + SSF container overhead dominate "compression" (HIGH)
- **Issue:** In `run_full_model_honest_results.json`, `fp16_passthrough` tensors show `orig_bf16_bytes / compressed_bytes ≈ 0.077` (i.e., the stored bytes are ~13x LARGER than the BF16 source). 71% of all tensors fall in this near-lossless/passthrough category. The SSF container uses 4096-byte page alignment plus 256-byte header + 128-byte footer per tensor (`tests/test_coverage_gaps.py:82-85`, `SSF_HEADER_SIZE=256`, `SSF_FOOTER_SIZE=128`, `SSF_PAGE_SIZE=4096`), so thousands of small tensors incur huge alignment/padding overhead.
- **Files:** `spectralstream/format/core.py` (constants), `spectralstream/format/ssf_format_pipeline.py`, `run_full_model_honest_results.json` (`method` distribution, per-tensor `compressed_bytes`).
- **Impact:** The headline "2.306x compression vs BF16" is REAL but misleading: it is carried almost entirely by the 29% of weight matrices using `int8_blockwise+zlib` (~6.5x), while 71% of tensors are stored losslessly (and expanded by format padding). The "compressed size 4.443 GB" includes substantial non-algorithmic padding, not pure savings.
- **Fix approach:** Report per-category ratios; pack small tensors into shared pages instead of per-tensor 4096-byte alignment; stop counting pass-through tensors as "compression"; surface container overhead separately from algorithmic ratio.

### TD-04 — `_archive/` contains 12 abandoned scripts (LOW)
- **Issue:** `_archive/v1/scripts/` holds 12 legacy scripts (`legacy_advanced_upgrades.py` 3099 lines, `legacy_quasar_extraction.py`, `profile_full_scale.py`, etc.) that are dead code maintained alongside the active tree.
- **Files:** `_archive/v1/scripts/*`.
- **Impact:** Confuses contributors about what is current; some still reference removed APIs. No test coverage.
- **Fix approach:** Remove or move to a clearly-versioned archive with a README explaining why each is abandoned.

### TD-05 — Functional error is 100x+ larger than the headline reconstruction error (HIGH — honesty)
- **Issue:** `run_full_model_honest.log:40` reports reconstruction `Rel MSE mean = 2.491e-05` (excellent) AND `rel_mse median = 0.000e+00` (71% of tensors are exactly 0 because they are pass-through/expanded). But the same log's "Functional error (`||Wx - What x|| / ||Wx||`)" section (lines 22-55) shows `rel_err` up to **0.014 (1.4%)** on sampled `self_attn.*` and `mlp.*` matrices — i.e., the actual inference-relevant error is ~560x the mean reconstruction MSE. The reconstruction `rel_mse` is dominated by the 1427 zero-error pass-through tensors and understates real quality impact.
- **Files:** `run_full_model_honest.log:40-55`, `run_full_model_honest_results.json` (`rel_mse_median: 0.0`, `functional` array).
- **Impact:** Headline quality metrics (rel_mse, cosine_sim 0.999988) are optimistic; the metric that matters for model output (functional error) is materially worse (up to 1.4% on attention projections). Risk of overclaiming quality.
- **Fix approach:** Lead with functional/downstream error, not reconstruction rel_mse of a heavily pass-through model; report functional error distribution prominently; stop reporting `rel_mse_median` as `0.0` without caveats.

---

## Known Bugs

### BUG-01 — Flagship "5-stage cascade" produces 72–92% reconstruction error on real weights (HIGH)
- **Issue:** The most recently added headline feature — the 5-stage cascade (`cascade_5stage.FiveStageCascade`, commit `de4e3bf` "implement 5-stage cascade pipeline") — produces catastrophic results on real Gemma-4 weights:
  - `wave4_results.json`: cascade `rel_mse = 0.721–0.923` (i.e., 72–92% error) across all 8 tensor categories, while the `block_int8` baseline gets `rel_mse = 0.0000–0.0001`.
  - `stage_diagnosis.json`: cascade final `rel_mse = 0.858`, `snr_db = 0.66`, `cosine_sim = 0.376`; diagnosis confirms `TT-SVD rank truncation` captures only ~14% of variance, and subsequent stages (Sparse top-1%, Ergodic, SIREN) add `SNR = 0.02 / 0.00 / 0.00 dB` (essentially nothing).
- **Files:** `wave4_pipeline.py` (uses `spectralstream.compression.cascade_5stage.FiveStageCascade`), `spectralstream/compression/cascade_5stage.py`, `wave4_results.json`, `stage_diagnosis.json`, `spectralstream/compression/methods/novel/cascade_1200.py`.
- **Impact:** The flagship method is non-functional as a compressor. Any claim that the cascade achieves its stated ratio while preserving quality is false for real weights.
- **Fix approach:** Gate cascade selection on error budgets (reject rel_mse > threshold); do not expose cascade as a default/front-page method until it beats block_int8 on real weights; or redesign the cascade stages (the diagnosis already lists `why_tt_fails`).

### BUG-02 — Cascade reports identical 22.31x ratio as block_int8 despite 85% error (HIGH — metric bug)
- **Issue:** In `wave4_results.json`, EVERY tensor category reports `ratio_vs_bf16 = 22.31` for BOTH the broken cascade (85% error) and the working `block_int8` (0% error). INT8 blockwise quantization of a dense matrix cannot yield 22.31x vs BF16 (realistic ≈ 2–4x). The ratio appears to be a shared/derived constant, not an independent per-method measurement, and is not gated on reconstruction error — so a method that destroys 85% of the signal is still reported as "22.31x compression."
- **Files:** `wave4_results.json` (`results.<cat>.5stage_cascade.ratio_vs_bf16` and `.block_int8.ratio_vs_bf16` both = 22.31), `spectralstream/compression/honest_metrics.py` (`dual_ratio`, `end_to_end_error`).
- **Impact:** Reported compression ratios are not trustworthy end-to-end measurements; they can be achieved by discarding data. Directly contradicts the "honest metrics" effort.
- **Fix approach:** Compute `ratio` from true serialized container bytes (including headers) measured independently per method; reject/flag any method whose `rel_mse` exceeds a threshold; never report a ratio without its paired error.

### BUG-03 — FP32-comparison inflates ratio 2x (MEDIUM — metric bug)
- **Issue:** `run_full_model_honest.log:38-39` reports `Ratio vs on-disk (BF16): 2.306x` and `Ratio vs FP32: 4.613x`. The input is already BF16; comparing against FP32 doubles the headline number. `benchmark_industry_comparison.json` and `final_benchmark_results.json` similarly compute `ratio_vs_fp32` alongside `ratio_vs_bf16`.
- **Files:** `run_full_model_honest.log:38-39`, `run_full_model_honest_results.json` (`ratio_vs_fp32: 4.613`, `ratio_vs_disk: 2.306`), `final_benchmark_results.json`, `benchmark_industry_comparison.json`.
- **Impact:** The most favorable number (4.6x) is the one most likely to be quoted; the honest comparison is 2.3x vs the actual on-disk format.
- **Fix approach:** Make `ratio_vs_fp32` secondary/non-default; lead with `ratio_vs_disk`.

### BUG-04 — `BaseTokenizer.encode` raises `NotImplementedError`; `token_count` calls it (MEDIUM)
- **Issue:** `spectralstream/utils/tokenizer_engine.py:133` `BaseTokenizer.encode` raises `NotImplementedError`, and `token_count` (line 142) calls `self.encode`, so calling `token_count` on a `BaseTokenizer` instance crashes. Concrete subclasses exist (`BPETokenizer`, `SentencePieceTokenizer`, `TiktokenTokenizer`, `CachedTokenizer`, `ParallelTokenizer`, `SpectralTokenizer`), but several depend on external libs absent from core deps.
- **Files:** `spectralstream/utils/tokenizer_engine.py:110-150`, `178-2132` (subclasses).
- **Impact:** If a base/unsupported tokenizer is used in the inference path (`spectralstream/orchestrator.py`, `spectralstream/utils/__init__.py`), encoding/counting raises at runtime. Risk of silent breakage in serving.
- **Fix approach:** Make `BaseTokenizer` truly abstract (ABC) so misuse fails at import; ensure production `AutoTokenizer` always resolves to a working concrete impl with a clear error if the backing lib is missing.

### BUG-05 — `EvictionPolicy.select_eviction` is a stub base (LOW)
- **Issue:** `spectralstream/kv_cache/eviction.py:14` base `EvictionPolicy.select_eviction` raises `NotImplementedError`. Only `SpectralEviction` and a few subclasses implement it; any other subclass or direct base use crashes.
- **Files:** `spectralstream/kv_cache/eviction.py:14`.
- **Impact:** Low — subclasses exist, but the base contract is unenforced.
- **Fix approach:** Convert to `abc.abstractmethod`.

### BUG-06 — Other `NotImplementedError` / TODO stubs (LOW-MEDIUM)
- **Issue:** Found across the tree:
  - `spectralstream/compression/world_model/compression_intelligence_v2.py:1688` `raise NotImplementedError("Decompress via decompress_to_safetensors")` — `decompress` path unimplemented in a "world model" module.
  - `spectralstream/compression/engine/dynamic_tuning/spectral_tuning.py:223,226` — two unimplemented tuning hooks.
  - `spectralstream/compression/methods/novel/topological/topological_skeleton.py:136` `# TODO: use randomized for large` SVD (full SVD on large matrices → memory blowup risk).
  - `spectralstream/utils/predictor_ensemble.py:87`, `spectralstream/kv_cache/eviction.py:18` — stubs.
  - `spectralstream/inference/pipeline.py:349` `# TODO: migrate _kv_cache_dict (flat dict) to KVCacheManager`.
- **Files:** as listed.
- **Impact:** Incomplete subsystems; `compression_intelligence_v2.decompress` is a hard crash for a documented capability.
- **Fix approach:** Implement or remove; add tests that exercise decompress paths.

---

## Security Considerations

### SEC-01 — `signal.alarm`/`exec` used for timeouts (Linux-only, fragile) (MEDIUM)
- **Issue:** `benchmark_physics_real_weights.py:24-26` sets a SIGALRM handler that does `exec("raise TimeoutError()")` inside a lambda. This is non-portable (SIGALRM does not exist on Windows) and uses `exec` of a string. The repo's own branch is `fix/honest-metrics-windows-compat`, confirming Windows breakage.
- **Files:** `benchmark_physics_real_weights.py:20-30`.
- **Impact:** Crashes/hangs on non-Linux; `exec` of string is a code-injection-ish anti-pattern (here benign but a poor precedent).
- **Fix approach:** Use `concurrent.futures` or `multiprocessing` timeouts; remove `exec`.

### SEC-02 — Hardcoded absolute model path to author's machine (LOW-MEDIUM)
- **Issue:** `benchmark_physics_real_weights.py:38` hardcodes `/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors`. `final_compression_log.txt` and `run_full_model_honest.log` reference `models/gemma-4-E2B/model.safetensors` (also absent from the repo; `models/` is gitignored).
- **Files:** `benchmark_physics_real_weights.py:38`, `final_compression_log.txt:147`, `run_full_model_honest.log:58`.
- **Impact:** Scripts are not reproducible off the original author's machine; results cannot be independently verified by others. Combined with fabricated-looking comparisons (below), this undermines the "honest metrics" claim's verifiability.
- **Fix approach:** Read model path from env/CLI arg; document required artifacts; publish the model hash used.

### SEC-03 — Benchmark "industry comparison" compares against non-existent methods (HIGH — honesty/fabrication)
- **Issue:** `benchmark_industry_comparison.json` reports `GGUF`, `GPTQ`, `AWQ`, and `SpectralStream` ratios/errors/times per matrix. But:
  - `GPTQ` and `AWQ` are **not in dependencies** (`pyproject.toml` has no gptq/awq; `grep -rn "import gptq\|import awq"` returns nothing). The only AWQ code is a local `_awqlike.py` DCT-based approximation.
  - GPTQ/AWQ values are **identical across different matrices** (e.g., `embedding` and `output` both show `GPTQ = 3.9844357976653697`, `AWQ = 3.9689922480620154`); GGUF ratio is a constant `3.5555...` (= 4/1.125, a formula, not a measurement). Identical per-matrix floats prove these are looked-up/estimated, not measured.
  - `certificate.py:440-463` builds an `industry_comparison` from **hardcoded constants** (e.g., `("GGML Q8_0", 2.5, ...)`, `("SqueezeLLM", 8.0, ...)`) — textbook ratios, not measurements.
- **Files:** `benchmark_industry_comparison.json`, `spectralstream/compression/certificate.py:440-463`, `spectralstream/compression/methods/novel/quantization_massive/_awqlike.py`.
- **Impact:** Direct continuation of the repo's historical "fabricated metrics" problem. Presents estimates/hardcoded constants as measured competitive benchmarks.
- **Fix approach:** Only compare against methods that are actually executed in-repo; label any external/estimated numbers explicitly as "literature estimates, not measured here"; remove hardcoded ratio tables.

### SEC-04 — No input validation / size limits on deserialization paths (LOW)
- **Issue:** Format readers (`spectralstream/format/ssf_format_pipeline.py`, `gguf_parser_engine.py`) parse attacker-influenced offsets/lengths. Tests like `test_coverage_gaps.py:264-267` only check a few truncated-name cases; there is no evidence of fuzz testing for malicious container files.
- **Files:** `spectralstream/format/ssf_format_pipeline.py:2906` (`encode`), `spectralstream/format/gguf_parser_engine.py`.
- **Impact:** Potential OOB reads / resource exhaustion from crafted model files.
- **Fix approach:** Validate offsets against file size; add fuzz/negative tests; cap allocation.

---

## Performance Bottlenecks

### PERF-01 — Per-tensor 4096-byte alignment wastes gigabytes (HIGH)
- **Issue:** SSF container aligns every tensor to 4096-byte pages with 256+128 bytes header/footer (`tests/test_coverage_gaps.py:82-85`). With 2011 tensors and 71% being small pass-through tensors, alignment/padding dominates stored size (see TD-03). The "compressed 4.443 GB" is inflated by this overhead.
- **Files:** `spectralstream/format/core.py`, `spectralstream/format/ssf_format_pipeline.py`.
- **Impact:** Real on-disk savings are smaller than reported; inflates the model file.
- **Fix approach:** Shared paged allocation; pack small tensors; report overhead separately.

### PERF-02 — Pathological slowness spike in full-model run (MEDIUM)
- **Issue:** `run_full_model_honest.log:7-8` shows checkpoints 600→700 at 298s but 700→800 at **1383s** — ~1085s for 100 tensors vs ~40s/100 before. A single tensor (or method branch) caused a >25x slowdown. `Wall time: 3213.2s` (~54 min) for a 10GB model.
- **Files:** `run_full_model_honest.log:7-8,57`, `scripts/run_5stage_on_weights.py` (pipeline that produced it).
- **Impact:** Throughput is unpredictable; some tensors trigger algorithms with poor scaling (cf. `topological_skeleton.py:136` full SVD TODO).
- **Fix approach:** Per-tensor timeout; per-method time budgets; profile the slow tensor class.

### PERF-03 — `block_int8` per-matrix time is 1–3 seconds (MEDIUM)
- **Issue:** `benchmark_industry_comparison.json` shows `SpectralStream (block_int8)` `time_ms = 1145–3023` per matrix (e.g., `ffn_gate` 3023ms). Basic INT8 blockwise quantization+entropy coding should be milliseconds. Either the method does excessive work or the timing includes model load/overhead.
- **Files:** `benchmark_industry_comparison.json` (`SpectralStream.time_ms`), `spectralstream/compression/unified_quantizer.py` (`block_int8` path).
- **Impact:** Throughput claims for the only working method are weak; 1-3s/matrix × thousands of tensors is impractical.
- **Fix approach:** Profile `block_int8`; separate load time from compute; vectorize.

### PERF-04 — Pure-NumPy LLM inference has no GPU/accel path (HIGH — architecture)
- **Issue:** Core deps (`pyproject.toml`) are `numpy/scipy/psutil/safetensors/zstandard`; `torch` is only an OPTIONAL `ml` extra. The engine is described as an "LLM inference engine" but there is no BLAS/GPU acceleration for the dominant linear-algebra workload.
- **Files:** `pyproject.toml` (dependencies), `spectralstream/tensor/tensor_ops_engine.py`, `spectralstream/serving/*`.
- **Impact:** Inference/compression throughput is orders of magnitude below any real serving engine; not production-viable for LLM serving.
- **Fix approach:** Provide an optional torch/numpy-BLAS backend; document that CPU-NumPy is research-only.

---

## Fragile Areas

### FRAG-01 — `fp16_passthrough` can overflow/NaN (MEDIUM)
- **Issue:** Storing tensors as `fp16_passthrough` converts BF16→FP16. BF16 has exponent range ~3.4e38; FP16 max ≈ 65504. Any tensor with large-norm values will overflow to `inf`/`nan` during the cast. `run_full_model_honest_results.json` shows 1427/1429 pass-through tensors have `rel_mse == 0`, which only holds if all values happened to be in FP16 range — a coincidental, fragile property, not a guarantee.
- **Files:** `spectralstream/compression/engine/_orchestrator.py` (method selection), `run_full_model_honest_results.json`.
- **Impact:** A single out-of-range tensor silently corrupts the model (NaN propagation) on future models with larger activations/weights.
- **Fix approach:** Detect out-of-FP16-range values and fall back to BF16-passthrough (no cast) or int8; add a test with large-magnitude inputs.

### FRAG-02 — `unified_quantizer.py` is a 4749-line monolith (MEDIUM)
- **Issue:** `spectralstream/compression/unified_quantizer.py` (4749 lines) is one of the largest modules and holds the only working quantization path. No obvious internal decomposition.
- **Files:** `spectralstream/compression/unified_quantizer.py`.
- **Impact:** Hard to modify safely; high regression risk; difficult to test in isolation.
- **Fix approach:** Split into per-method modules (mirroring the `methods/` layout); add unit tests per quantizer.

### FRAG-03 — Largest files are untested monoliths (MEDIUM)
- **Issue:** The biggest modules have ~0 test references: `spectralstream/serving/batching_engine.py` (4540), `spectralstream/serving/production_stack.py` (4022), `spectralstream/utils/meta_controller.py` (4010), `spectralstream/agents/engine.py` (3545), `spectralstream/tensor/hpc_engine.py` (3077), `spectralstream/utils/multimodal_prompt.py` (3433), `spectralstream/utils/integration.py` (2800). See Test Coverage Gaps.
- **Files:** as listed.
- **Impact:** Core serving/agent/tensor subsystems can break without any test failing.
- **Fix approach:** Add smoke + integration tests; enforce coverage minimums on these modules.

### FRAG-04 — `stage_diagnosis.json` is honest but shows cascade is fundamentally unsound (MEDIUM)
- **Issue:** The diagnosis (`stage_diagnosis.json`) itself is a good, honest artifact: it proves TT-SVD folding creates a mode-0 dim of only 32 for a 2048×1536 matrix, capturing ~14% variance, and that Ergodic/SIREN stages add ~0 dB. This is a design flaw, not a tuning issue.
- **Files:** `stage_diagnosis.json` (`analysis.why_tt_fails`, `ablation_summary`), `spectralstream/compression/methods/novel/cascade_1200.py`.
- **Impact:** The cascade cannot be "tuned" into working without redesign; yet it is presented as a headline feature.
- **Fix approach:** Either redesign TT folding (match tensor modes to TT cores) or remove cascade from advertised capabilities until fixed.

---

## Scaling Limits

### SCALE-01 — Method registry does not scale (MEDIUM)
- **Issue:** 2964 methods registered at startup; registration is O(methods) and import of the full tree is heavy (`final_compression_log.txt` ~2s just to register). Adding more methods linearly degrades startup.
- **Files:** `spectralstream/compression/engine/method_discovery.py`, `spectralstream/compression/methods/__init__.py`.
- **Impact:** Cold-start latency; memory for unused method objects.
- **Fix approach:** Lazy registration; only register methods referenced by the active config.

### SCALE-02 — Full-model compression is single-pass, ~54 min for 10GB (MEDIUM)
- **Issue:** `run_full_model_honest.log` shows 3213s wall time for one 10GB model, single pass, with a 25x slow spike. No streaming/distributed path is evident for larger models.
- **Files:** `scripts/run_5stage_on_weights.py`, `run_full_model_honest.log`.
- **Impact:** Cannot realistically compress 70B+ models in reasonable time on CPU.
- **Fix approach:** Worker pool (`CompressionConfig.num_workers` exists but efficacy unverified), streaming chunks, GPU backend.

### SCALE-03 — In-memory tensor materialization for 256x256+ slices (LOW)
- **Issue:** `benchmark_physics_real_weights.py:47-49` materializes `np.array(view[:256,:256])` and several methods do full SVD on copies (`topological_skeleton.py:136`). Larger slices will OOM on the working set.
- **Files:** `benchmark_physics_real_weights.py:47-49`, `spectralstream/compression/methods/novel/topological/topological_skeleton.py:136`.
- **Impact:** Memory blowup on large matrices; only tiny slices are benchmarked (representativeness gap).
- **Fix approach:** Randomized SVD; memory-mapped ops; cap slice size.

---

## Dependencies at Risk

### DEP-01 — No GPU/accel dependency; torch is optional (HIGH)
- **Issue:** `pyproject.toml` lists only `numpy/scipy/psutil/safetensors/zstandard` as required; `torch` is an optional `ml` extra. A "LLM inference engine" with no mandatory accelerated linear algebra is architecturally at risk vs any real baseline (llama.cpp, vLLM, HF transformers).
- **Files:** `pyproject.toml`.
- **Impact:** Performance (PERF-04) makes the engine non-competitive; risk of abandonment as a "real" engine.
- **Fix approach:** Commit to a backend strategy; make a fast path mandatory or clearly scope the project as research/prototype.

### DEP-02 — `requirements.txt` diverges from `pyproject.toml` (MEDIUM)
- **Issue:** `requirements.txt` pins only `tree-sitter*` packages (for code-analysis features), while `pyproject.toml` declares the real runtime deps. Two dependency sources can drift; `requirements.txt` alone cannot install the package.
- **Files:** `requirements.txt`, `pyproject.toml`.
- **Impact:** Confusing environment setup; missing `requirements.txt` entry for `numpy` etc. leads to broken installs.
- **Fix approach:** Single source of truth; remove or align `requirements.txt`.

### DEP-03 — Exotic method modules depend on unpinned/optional libs (LOW)
- **Issue:** Some methods import `torch`, `scikit-learn`, `ml_dtypes` (optional extras). If an optional method is selected at runtime without its extra installed, it raises `ImportError` mid-compression.
- **Files:** `spectralstream/compression/methods/novel/*`, `spectralstream/profiler/scanner.py:80` (`raise ImportError("GGUF parser engine not available")`).
- **Impact:** Runtime failures depend on which optional extras were installed.
- **Fix approach:** Declare method→extra mapping; fail fast with actionable message at method selection.

---

## Missing Critical Features

### MISS-01 — No real end-to-end text-generation / perplexity validation (HIGH)
- **Issue:** The headline quality story relies on tensor reconstruction + functional error on a handful of sampled matrices. There is no measured downstream metric (perplexity, task accuracy) of a model compressed by SpectralStream vs the original. `benchmark_suite.py:3236` (`bench_perplexity`) exists but the repo's result JSONs contain none.
- **Files:** `spectralstream/benchmark/benchmark_suite.py:3236-3357`, `tests/` (no perplexity result artifact).
- **Impact:** Cannot substantiate that "compression preserves model quality" — the central claim — for actual generation.
- **Fix approach:** Add a real eval (e.g., Wikitext perplexity) comparing original vs compressed model; publish results.

### MISS-02 — No working production tokenizer guaranteed (MEDIUM)
- **Issue:** Concrete tokenizers (`BPETokenizer`, `SentencePieceTokenizer`, `TiktokenTokenizer`) depend on external libs not in core deps; `BaseTokenizer.encode` is a stub. It is unclear which tokenizer actually works out-of-the-box for inference on Gemma-4.
- **Files:** `spectralstream/utils/tokenizer_engine.py`, `spectralstream/orchestrator.py`.
- **Impact:** End-to-end inference path may not be runnable without extra installs.
- **Fix approach:** Ship one self-contained, tested default tokenizer.

### MISS-03 — `compression_intelligence_v2.decompress` unimplemented (MEDIUM)
- **Issue:** `spectralstream/compression/world_model/compression_intelligence_v2.py:1688` raises `NotImplementedError` for decompress. A "world model" compression capability cannot round-trip.
- **Files:** `spectralstream/compression/world_model/compression_intelligence_v2.py:1688`.
- **Impact:** Advertised capability is non-functional.
- **Fix approach:** Implement or remove from the public API.

### MISS-04 — No CI / automated quality gate in repo (MEDIUM)
- **Issue:** No CI config (GitHub Actions, etc.) is present; tests are run manually via `tests/run_all_tests.py`. The `.coverage` file exists (manual run) but there is no enforced gate preventing metric/feature regressions.
- **Files:** repo root (no `.github/workflows`), `tests/run_all_tests.py`, `.coverage`.
- **Impact:** Regressions (like fabricated metrics) can reappear unnoticed; no PR gate.
- **Fix approach:** Add CI running `pytest` + a metrics-honesty lint (assert ratios are paired with errors, no hardcoded competitor tables).

---

## Test Coverage Gaps

> The repo has **83 test files** and **2758 test functions** — but the source tree has **3479 `.py` files** under `spectralstream/` plus 15 under `scripts/`. Coverage is heavily concentrated on the `format/` and `compression/engine` dataclasses; entire large subsystems have effectively **no tests**.

### COV-01 — `tensor/`, `agents/`, `utils/`, `benchmark/` have ~0 test references (HIGH)
- **Issue:** Grepping test imports shows ~0 test files reference `spectralstream.tensor`, `spectralstream.agents`, `spectralstream.utils`, `spectralstream.benchmark`. Meanwhile these include the largest, most critical modules:
  - `spectralstream/tensor/hpc_engine.py` (3077 lines) — **0 tests**
  - `spectralstream/agents/engine.py` (3545), `cascade.py`, `swarm.py`, `cascade_controller.py` — **0 tests**
  - `spectralstream/utils/meta_controller.py` (4010), `multimodal_prompt.py` (3433), `integration.py` (2800), `sampler_engine.py` (2611), `tokenizer_engine.py` (2438) — **0 tests**
  - `spectralstream/benchmark/benchmark_suite.py` (4051) — **0 tests** (the very tool that produces quality metrics!)
- **Files:** as listed; `tests/` directory inventory.
- **Impact:** The subsystems that would actually serve/infer/measure are unverified. A regression in `benchmark_suite.py` or `tokenizer_engine.py` would not be caught.
- **Fix approach:** Add smoke + unit tests for each; prioritize `tokenizer_engine.py` (inference-critical) and `benchmark_suite.py` (metrics-critical).

### COV-02 — Serving stack barely tested (MEDIUM)
- **Issue:** `spectralstream/serving/batching_engine.py` (4540), `production_stack.py` (4022), `server.py` (2384) have only `tests/test_serving_api.py` (2 test files referenced). No load/integration tests.
- **Files:** `spectralstream/serving/*`, `tests/test_serving_api.py`.
- **Impact:** Production serving behavior unverified.
- **Fix approach:** Add request/lifecycle tests; error-path tests.

### COV-03 — The "honest metrics" path itself is untested (HIGH)
- **Issue:** `spectralstream/compression/honest_metrics.py` (`dual_ratio`, `end_to_end_error`, `ErrorMetrics`) — the module explicitly built to stop metric fabrication — has no dedicated test asserting that (a) ratios are computed from true serialized bytes, and (b) a method with high `rel_mse` is flagged/rejected. Yet `wave4_results.json` shows it emitting a 22.31x ratio for an 85%-error method (BUG-02).
- **Files:** `spectralstream/compression/honest_metrics.py`, `tests/` (no `test_honest_metrics.py`).
- **Impact:** The anti-fabrication safeguard has no guardrail; the bug recurred.
- **Fix approach:** Add `tests/test_honest_metrics.py` asserting ratio↔error coupling and rejecting error>threshold.

### COV-04 — `fp16`/`bf16` round-trip and overflow untested (MEDIUM)
- **Issue:** `fp16_passthrough` (FRAG-01) has no test with out-of-FP16-range values; the only dtype test is `test_coverage_gaps.py:92-115` (enum mapping, not round-trip correctness).
- **Files:** `spectralstream/compression/engine/_orchestrator.py`, `tests/test_coverage_gaps.py:92-115`.
- **Impact:** Overflow/NaN risk (FRAG-01) undetected.
- **Fix approach:** Add round-trip + overflow-fallback tests.

### COV-05 — Cascade method has no correctness test on real weights (HIGH)
- **Issue:** `spectralstream/compression/cascade_5stage.py` (`FiveStageCascade`) has no test asserting acceptable `rel_mse` on a real matrix. The failure only surfaced in `wave4_results.json`/`stage_diagnosis.json` artifacts, not in the test suite.
- **Files:** `spectralstream/compression/cascade_5stage.py`, `tests/` (no cascade correctness test).
- **Impact:** A fundamentally broken method ships as a headline feature with green tests.
- **Fix approach:** Add a test that compresses a real weight slice and asserts `rel_mse < 0.05`; mark cascade as `xfail`/disabled until it passes.

### COV-06 — Dead "archive" tests inflate the count (LOW)
- **Issue:** `tests/test_archive_*.py` (7 files: `test_archive_architecture_compressor`, `test_archive_combined_pipeline`, `test_archive_edge_cases`, `test_archive_extreme_compression`, `test_archive_extreme_compressor`, `test_archive_gguf_conversion`, `test_archive_intelligence_engine`) test code in `_archive/` that is abandoned. These pad the 2758-function count without covering active code.
- **Files:** `tests/test_archive_*.py`, `_archive/`.
- **Impact:** Inflates perceived coverage; misleads reviewers.
- **Fix approach:** Delete archive tests or move them with the archive.

### COV-07 — `compression_intelligence_v2` decompress path untested (LOW)
- **Issue:** No test exercises `decompress` on `compression_intelligence_v2` (which raises `NotImplementedError`, BUG-06).
- **Files:** `spectralstream/compression/world_model/compression_intelligence_v2.py:1688`, `tests/`.
- **Impact:** A crashing public method is untested.
- **Fix approach:** Add test that asserts decompress works or is explicitly unsupported.

---

## Priority Summary

| ID | Area | Severity | One-line |
|----|------|----------|----------|
| BUG-01 | Cascade broken (72–92% err) | HIGH | Flagship 5-stage cascade fails on real weights |
| BUG-02 | Cascade 22.31x ratio w/ 85% err | HIGH | Reported ratios not error-gated → not honest |
| SEC-03 | Fabricated industry comparison | HIGH | GPTQ/AWQ/GGUF are hardcoded/estimated, not measured |
| TD-01 | 2964 methods, 2 used | HIGH | Massive dead surface; misleading capability |
| TD-03 | fp16 passthrough + SSF overhead | HIGH | Headline 2.3x is mostly format padding, not compression |
| TD-05 | Functional err 100x reconstruction | HIGH | Quality overclaimed via pass-through-dominated rel_mse |
| PERF-04 | No GPU/accel backend | HIGH | CPU-NumPy LLM engine not production-viable |
| MISS-01 | No perplexity/eval | HIGH | Central "quality preserved" claim unsubstantiated |
| COV-01 | tensor/agents/utils untested | HIGH | Largest critical modules uncovered |
| COV-03 | honest_metrics untested | HIGH | Anti-fabrication safeguard has no guardrail |
| COV-05 | cascade untested on real weights | HIGH | Broken method ships green |
| BUG-03 | FP32 ratio 2x inflation | MEDIUM | Use ratio_vs_disk as default |
| BUG-04 | BaseTokenizer stub | MEDIUM | Inference-encoding can crash |
| SEC-01 | signal.alarm/exec timeout | MEDIUM | Non-portable, Windows breakage |
| SEC-02 | Hardcoded model path | MEDIUM | Results not reproducible elsewhere |
| PERF-01/02/03 | SSF alignment / spikes / slow int8 | MEDIUM | Throughput & size overhead |
| FRAG-01 | fp16 overflow/NaN | MEDIUM | Silent model corruption risk |
| DEP-01/02 | torch optional, reqs diverge | MEDIUM | Unclear/risky dependency story |
| MISS-02/03/04 | tokenizer/decompress/CI gaps | MEDIUM | Missing runnable/verifiable essentials |
| TD-02/04, BUG-05/06, FRAG-02/03/04, SCALE-*, DEP-03, COV-02/04/06/07 | LOW | Cleanup/stub/monolith/fuzz gaps |

---

*Concerns audit: 2026-07-07*

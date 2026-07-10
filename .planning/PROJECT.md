# Spectral-Stream LLM Engine

## What This Is

A pure-Python (NumPy/SciPy, no C++), CPU-targeted LLM weight & KV-cache compression engine that doubles as a local OpenAI-compatible inference server. Its thesis: instead of compressing weights as flat tensors, treat them as continuous manifolds / spectral fields / physical systems and exploit multiple redundancy sources (low-rank, spectral decay, cross-layer correlation, entropy, physics symmetries) through a cascade of independent methods.

The engine delivers **~4–6× compression vs FP32 (≈2–3× vs BF16)** on real LLM weights via its verified INT8/INT4 quantization path. A broad registry of 2,964 methods across 43 categories exists as an explicit, labeled **research catalog** — most are aspirational or unvalidated on real weights. The project's energy is split between (1) maintaining an honest, reliable compression core, and (2) actively exploring activation-based pruning, LLM neuroanatomy, per-layer importance analysis and dynamic reduction, and deeper architecture / math optimizations.

**🔬 ACTIVE R&D — NOT PRODUCTION READY.** License: AGPL-3.0.

## Core Value

**Honest compression that actually works on real weights.** The metric that matters is byte-exact ratio + paired error — never one without the other — and every advertised number must be verifiably measured, not estimated or aspirational. The research catalog lives alongside this core, but the default advertised capability tells the truth about what works today.

## Requirements

### Validated

Capabilities that work on real weights or are structurally sound:

- ✓ **INT8 blockwise quantization (`block_int8`)** — ~4.6× vs FP32 (2.3× vs BF16), SNR ~42 dB, on real Gemma-4 E2B (2011 tensors, 10 GB → 4.4 GB on disk) — the entire verified compression capability
- ✓ **FP16 passthrough** — Lossless byte-for-byte identity (with SSF container padding overhead)
- ✓ **Honest-metrics infrastructure** — `serialized_nbytes()`, `end_to_end_error()`, `dual_ratio()`, `ErrorMetrics(rel_mse, cosine_sim, max_abs, snr_db)` — byte-exact ratio + error measurement
- ✓ **Registry-driven orchestrator** — `CompressionIntelligenceEngine` (profile→allocate→select→compress→validate) with lazy method loading and tier-based selection
- ✓ **SSF v2/v3 binary format** — Full read/write/index/header I/O with zstd container codec
- ✓ **CLI** — 10+ subcommands (`compress`, `profile`, `validate`, `benchmark`, `list-methods`, `info`, `convert`, `infer`, `generate-certificate`, `dial-in`)
- ✓ **GGUF/safetensors converters** — Pure-Python weight loading and format conversion
- ✓ **KVCacheManager** — 30+ eviction/compression policies unified under a single adapter
- ✓ **5-stage cascade pipeline (2 live stages)** — EinSort + TT-SVD/block-quant implemented; stages 3–5 (sparse, ergodic, SIREN) defined but not wired
- ✓ **Method registry (broad structural)** — 2,964 registered methods across decomposition, spectral, structural, functional, physics, quantization, entropy, lossless, hybrid, and novel categories — registration infrastructure works; per-method validation on real weights varies from zero to complete
- ✓ **Streaming/chunked/memory-mapped compressors** — Memory-bounded execution for large tensors
- ✓ **Pure-Python GGUF weight dequantizer** — Imports GGUF models without llama.cpp dependency
- ✓ **CLI dashboards** — Rich terminal dashboards, progress bars, tables
- ✓ **Error-gated ratio emission (`apply_gate`)** — Validated in Phase 01 (METRICS-01): high-error methods (`rel_mse > 0.05`) emit no ratio (gated: `ratio_vs_bf16:None`), good methods keep byte-exact `ratio_vs_bf16`/`ratio_vs_fp32` via `serialized_nbytes`
- ✓ **BF16 as headline ratio** — Validated in Phase 01 (METRICS-02): CLI summary leads with `ratio_vs_disk` (BF16), demotes FP32 to secondary, filters gated tensors from means, emits a GATED marker
- ✓ **Honest competitor comparison** — Validated in Phase 01 (METRICS-03): 9 competitor constants extracted to `literature_estimates.py` labeled "literature estimates, not measured here"; `certificate.py` is a pure consumer; orphaned `benchmark_industry_comparison.json` deleted
- ✓ **honest_metrics unit tests** — Validated in Phase 01 (METRICS-04): `tests/test_honest_metrics.py` asserts ratio↔error coupling, strict-`>` boundary, all `serialized_nbytes` payload shapes, BF16 headline, gated-None

### Active

v1 scope: fix honesty gaps AND consolidate the core. All items are hypotheses until shipped.

- [ ] **METRICS-01**: Error-gate all reported ratios — a method with `rel_mse > threshold` must not report a ratio as if it succeeded (fixes BUG-02)
- [ ] **METRICS-02**: Make `ratio_vs_disk` (BF16) the default headline number; demote `ratio_vs_fp32` to secondary (fixes BUG-03)
- [ ] **METRICS-03**: Replace fabricated industry comparison in `benchmark_industry_comparison.json` and `certificate.py` with either real measurements or explicit "literature estimates, not measured here" labels (fixes SEC-03)
- [ ] **METRICS-04**: Add dedicated unit tests for `honest_metrics.py` — assert ratio↔error coupling, reject methods above error threshold, verify `serialized_nbytes` handles all payload shapes (fixes COV-03)
- [ ] **CASCADE-01**: Fix the 5-stage cascade to produce `rel_mse < 0.05` on real weight slices, or remove it as a default/headline method until it does (fixes BUG-01)
- [ ] **CASCADE-02**: Wire stages 3–5 or update documentation to honestly state that only stages 1–2 are live
- [ ] **REGISTRY-01**: Split the 2,964-method registry into a validated active set (methods with tests + real-weight results) versus a labeled `experimental/` namespace (fixes TD-01)
- [ ] **REGISTRY-02**: Fix or remove the broken walk-based auto-discovery (`_discover_by_walk`) (fixes TD-02)
- [ ] **FORMAT-01**: Separate SSF container overhead (per-tensor 4096-byte page alignment + header/footer) from algorithmic compression ratio; pack small tensors into shared pages (fixes TD-03 / PERF-01)
- [ ] **EVAL-01**: Implement and publish at least one real downstream eval — Wikitext perplexity on original vs compressed model (fixes MISS-01)
- [ ] **DOCS-01**: Rewrite README, docs/QUANTIZATION.md, and compression-roadmap.md to honestly frame what works (4–6× vs FP32), label aspirational numbers as targets, and describe the method registry as a research catalog
- [ ] **DOCS-02**: Create explicit "Research Catalog" documentation explaining the maturity levels of experimental methods
- [ ] **RND-01**: Implement weight-activation importance scoring (APoZ / Wanda-class / SparseGPT) and validate on real weight slices — build the neural-activity-analysis layer
- [ ] **RND-02**: Build LLM layer neuroanatomy profiling pipeline: per-layer singular-value distribution, activation sparsity patterns, cross-layer correlation analysis
- [ ] **RND-03**: Prototype dynamic per-layer reduction — given importance metrics per layer, choose compression parameters or pruning ratio adaptively
- [ ] **RND-04**: Explore deep architecture optimizations and math-primitive improvements (e.g., alternative factorization schedules, hybrid spectral-decomposition variants)

### Out of Scope

- **Production-grade GPU-accelerated LLM serving** — The engine is research-grade CPU-only. No vLLM/TRT-LLM/TensorRT path is planned.
- **Training-from-scratch compression methods** (BitNet, ternary-from-scratch, etc.) — Only post-training compression. The 800×+ ratios require training methods; that is a different research agenda.
- **The 2000–5000× numbers as achieved results** — These are theoretical upper bounds from exotic methods (quantum, holographic, chaotic), not end-to-end measurements. They live in the research catalog; README must not present them as current capability.
- **Real-time model serving at scale** — No load balancing, horizontal scaling, or production SLAs.
- **Full CI/CD pipeline** — v1 targets a metrics-honesty gate and optional puzzles eval, not a full CI/CD lifecycle.

## Context

**The honesty story:** The repo has a documented history of inflated metrics (commit `c66016e` "replace fabricated metrics with true end-to-end compression measurements"). The `honest_metrics.py` module was built specifically to stop this. It works — but CONCERNS.md (COV-03) finds it has no tests, and BUG-02 shows a method with 85% error still reports a 22.31× ratio. The honest-metrics infrastructure is in place but unhardened.

**The cascade gap:** The flagship 5-stage cascade is documented as EinSort → TT-SVD → Sparse Residual → Ergodic → SIREN, but only stages 1–2 are actually wired. On real Gemma-4 weights, the cascade yields 72–92% reconstruction error (TT-SVD captures ~14% variance). The diagnosis (`stage_diagnosis.json`) is honest but damning.

**The registry size:** 2,964 registered methods, but the honest full-model run (`run_full_model_honest_results.json`) used only `fp16_passthrough` (71% of tensors) and `int8_blockwise+zlib` (29%). The registry is overwhelmingly aspirational — a research catalog, not an active toolkit.

**Research directions (the R&D half):** The active exploration agenda includes:
- *Activation-based pruning* — APoZ (average % of zeros), Wanda (weight × activation), SparseGPT on real weights
- *LLM neuroanatomy* — Per-layer importance profiling, singular-value distribution analysis, activation sparsity patterns
- *Dynamic reduction* — Given per-layer importance metrics, adaptively choose compression parameters or prune ratios per layer
- *Deep architecture optimization* — Factorizations, hybrid spectral-decomposition variants, math-primitive improvements
- *The existing 2,964-method research catalog* — Maturity-labeled and honestly documented, not claimed as working

## Constraints

- **Tech stack**: Pure Python (NumPy ≥1.24, SciPy ≥1.10). No C++ extensions, no GPU acceleration, no torch dependency in the core path. CPU-targeted only.
- **License**: AGPL-3.0
- **Maturity**: Research-grade. Performance is orders of magnitude below real LLM serving engines. Single-threaded (GIL), no GPU, inference ~2–3× slower than llama.cpp at best.
- **Metrics honesty**: ALL ratio/error numbers must flow through `honest_metrics.py` (byte-exact). No estimates, no per-stage products, no len(dict).
- **Format overhead**: SSF container uses 4096-byte page alignment + 256-byte header + 128-byte footer per tensor. ALWAYS report algorithmic ratio separate from overhead.
- **Windows compatibility**: The `fix/honest-metrics-windows-compat` branch exists. `signal.alarm`/`exec`-based timeouts (Linux-only) must be replaced.
- **Dependency divergence**: `pyproject.toml` and `requirements.txt` disagree. Single-source deps needed.
- **No eval baseline**: Zero perplexity or downstream task evaluations exist. This is the single biggest quality-trust gap.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Dual core + R&D | Keep the broad method registry as an explicit research catalog; make the honest ~4–6× path the only advertised/default capability. Honesty and exploration coexist but are clearly labeled. | — Pending |
| v1 = Both honesty + consolidation | Fix metric gaps (BUG-02, SEC-03, COV-03) AND consolidate core (TD-01 registry, TD-03 SSF overhead, MISS-01 eval). "Either/or" leaves known dishonesty or unsustainable bloat. | — Pending |
| Activation-based pruning as R&D track | APoZ / Wanda / SparseGPT are partially present as stubs. The real work is building the analysis layer (neuroanatomy, importance scoring, dynamic gating) and validating on real weights. | — Pending |
| Research catalog maturity labels | The 2,964 methods are not removed but moved to a labeled namespace with documented validation status. This preserves the exploration value while preventing misrepresentation. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-08 after Phase 1 (Metrics Trust Loop) completion*
*Last updated: 2026-07-08 after project initialization*

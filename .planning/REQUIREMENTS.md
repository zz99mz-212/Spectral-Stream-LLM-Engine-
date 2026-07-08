# Requirements: Spectral-Stream LLM Engine

**Defined:** 2026-07-08
**Core Value:** Honest compression that actually works on real weights — every ratio paired with its error, every claim verifiably measured.

## v1 Requirements

Requirements for initial consolidation release. Each maps to roadmap phases.

### Metrics Trust Loop

Foundational honesty infrastructure. Fixes the broken relationship between ratio and error that made BUG-02 and the fabricated comparison possible.

- [ ] **METRICS-01**: Error-gate every reported compression ratio — a method with `rel_mse > threshold` must not emit a ratio as if it succeeded (fixes BUG-02)
- [ ] **METRICS-02**: Make `ratio_vs_disk` (BF16 baseline) the default headline; demote `ratio_vs_fp32` to secondary annotation (fixes BUG-03)
- [ ] **METRICS-03**: Replace fabricated industry comparisons in `benchmark_industry_comparison.json` and `certificate.py:440-463` — remove hardcoded GPTQ/AWQ/GGUF constants; label any external reference as "literature estimates, not measured here" (fixes SEC-03)
- [ ] **METRICS-04**: Add `tests/test_honest_metrics.py` — assert ratio↔error coupling, reject methods above error threshold, verify `serialized_nbytes` handles all payload shapes (fixes COV-03)

### Cascade Correction

Fix or honestly scope the project's headline feature.

- [ ] **CASCADE-01**: The 5-stage cascade must produce `rel_mse < 0.05` on real weight slices, OR be removed as a default/headline method and moved to experimental (fixes BUG-01)
- [ ] **CASCADE-02**: Wire stages 3–5 (sparse, ergodic, SIREN) into the compress path, or update documentation to state honestly that only stages 1–2 are live (fixes the known anti-pattern)

### Eval Subsystem

The single biggest trust gap — without perplexity or downstream-task evaluation, "quality preserved" is an unsubstantiated claim.

- [ ] **EVAL-01**: Implement and publish at least one real downstream eval — WikiText-2 perplexity (seq len 2048) comparing original FP16 model vs compressed model; produce a verifiable JSON artifact (fixes MISS-01)
- [ ] **EVAL-02**: Fix `BaseTokenizer.encode` to not raise `NotImplementedError`; ship one working default tokenizer; add test (fixes BUG-04, MISS-02)
- [ ] **EVAL-03**: Replace hardcoded absolute model path (`/home/mike/.../gemma-4-E2B`) with env/CLI arg; document required model for reproducing results (fixes SEC-02)

### Registry & Depth Reduction

The 2,964-method registry is the project's largest maintenance liability and most misleading capability indicator.

- [ ] **REGISTRY-01**: Split the registry into a validated active set (methods with passing tests + real-weight results) versus a labeled `experimental/` namespace (fixes TD-01)
- [ ] **REGISTRY-02**: Fix or remove the broken walk-based auto-discovery (`_discover_by_walk`); it silently returns 0 methods (fixes TD-02)

### Format Transparency

SSF container overhead inflates reported compression ratios and masks true algorithmic savings.

- [ ] **FORMAT-01**: Separate SSF container overhead (per-tensor 4096-byte page alignment + 256B header + 128B footer) from algorithmic compression ratio in reporting (fixes TD-03 / PERF-01)
- [ ] **FORMAT-02**: Pack small tensors into shared pages instead of per-tensor alignment (reduces padding overhead)

### Documentation & Honest Framing

- [ ] **DOCS-01**: Rewrite README, docs/QUANTIZATION.md, compression-roadmap.md to honestly frame what works (~4–6× vs FP32); label all aspirational numbers as targets/theoretical (fixes TD-05 documentation gap)
- [ ] **DOCS-02**: Create explicit "Research Catalog" documentation explaining the maturity levels of experimental methods, so the broad registry is honest about its nature

### R&D Exploration

Build the analysis layer that enables activation-based pruning, neuroanatomy study, and dynamic reduction. Depends on calibration infrastructure from later phases.

- [ ] **RND-01**: Implement weight-activation importance scoring on real weight slices — APoZ / Wanda-class / SparseGPT
- [ ] **RND-02**: Build LLM layer neuroanatomy profiling pipeline — per-layer singular-value distributions, activation sparsity patterns, cross-layer correlation analysis
- [ ] **RND-03**: Prototype dynamic per-layer reduction — given importance metrics, choose compression parameters or pruning ratio adaptively
- [ ] **RND-04**: Explore deep architecture optimizations and math-primitive improvements (alternative factorization schedules, hybrid spectral-decomposition variants)

## v2 Requirements

Deferred to future releases. Tracked but not in current roadmap.

- **INT4-01**: Groupwise INT4 with column-wise OBS error compensation — uses calibration + eval subsystems (depends on Phases 3–4 from research)
- **INF-01**: Promote `infer` CLI to first-class `inference/` subsystem with stable format contract
- **GGUF-01**: Add GGUF writer for llama.cpp interop (reader exists)
- **CI-01**: Add CI with metrics-honesty lint + perplexity gate (MISS-04)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Production-grade GPU-accelerated LLM serving | Research-grade CPU-only; no vLLM/TRT-LLM/TensorRT path planned (from PROJECT.md) |
| Training-from-scratch compression (BitNet, ternary models) | Only post-training compression; the 800×+ ratios require training methods (from PROJECT.md) |
| 2000–5000× as achieved results | Theoretical upper bounds from exotic methods; live in research catalog only (from PROJECT.md) |
| Marlin/EXL2/speculative-decoding kernels | GPU-kernel features violate pure-Python/CPU-only constraint (from research) |
| Real-time model serving at scale | No load balancing, horizontal scaling, or production SLAs |
| YAML config support | `spectralstream/config.py` claims YAML support but only JSON is implemented; no PyYAML dep planned |
| tree-sitter code analysis features | 10 language grammars listed in `requirements.txt` but no `import tree_sitter` exists anywhere |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| METRICS-01 | Phase 1 (Metrics Trust Loop) | Pending |
| METRICS-02 | Phase 1 (Metrics Trust Loop) | Pending |
| METRICS-03 | Phase 1 (Metrics Trust Loop) | Pending |
| METRICS-04 | Phase 1 (Metrics Trust Loop) | Pending |
| CASCADE-01 | Phase 3 (Cascade Correction) | Pending |
| CASCADE-02 | Phase 3 (Cascade Correction) | Pending |
| EVAL-01 | Phase 2 (Eval Subsystem) | Pending |
| EVAL-02 | Phase 2 (Eval Subsystem) | Pending |
| EVAL-03 | Phase 2 (Eval Subsystem) | Pending |
| REGISTRY-01 | Phase 6 (Registry & Depth Reduction) | Pending |
| REGISTRY-02 | Phase 6 (Registry & Depth Reduction) | Pending |
| FORMAT-01 | Phase 7 (Format Transparency) | Pending |
| FORMAT-02 | Phase 7 (Format Transparency) | Pending |
| DOCS-01 | Phase 8 (Documentation & Honest Framing) | Pending |
| DOCS-02 | Phase 8 (Documentation & Honest Framing) | Pending |
| RND-01 | Phase 4 (Calibration & Neuroanatomy Foundation) | Pending |
| RND-02 | Phase 4 (Calibration & Neuroanatomy Foundation) | Pending |
| RND-03 | Phase 5 (Dynamic Reduction & Architecture Optimization) | Pending |
| RND-04 | Phase 5 (Dynamic Reduction & Architecture Optimization) | Pending |
| INT4-01 (v2) | — | Deferred |
| INF-01 (v2) | — | Deferred |
| GGUF-01 (v2) | — | Deferred |
| CI-01 (v2) | — | Deferred |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0 ✓ (100% coverage)

---
*Requirements defined: 2026-07-08*
*Last updated: 2026-07-08 after initial definition*

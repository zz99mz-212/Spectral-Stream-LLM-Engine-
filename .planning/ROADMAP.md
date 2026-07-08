# Roadmap: Spectral-Stream LLM Engine v1

## Overview

Spectral-Stream v1 is an honesty-first consolidation of a pure-Python, CPU-targeted LLM compression engine. The journey runs the research critical path: harden the metrics trust loop so every number is error-gated and byte-exact (the cheapest, highest-leverage fix), stand up an independent eval subsystem so "quality preserved" is provable, honestly scope or fix the flagship cascade, build a torch-free calibration substrate that unblocks the entire R&D track, then reduce registry/format bloat and document the truth. Every phase is an independent, verifiable MVP outcome; INT4 and inference promotion are deliberately deferred to v2.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3 ...): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Metrics Trust Loop** - Every reported ratio is error-gated and measured end-to-end; fabricated comparisons removed (completed 2026-07-08)
  - [x] 01-01-PLAN.md — Walking skeleton: central `apply_gate` chokepoint + `serialized_nbytes` shapes test (METRICS-01, METRICS-04)
  - [x] 01-02-PLAN.md — Gate second CLI block + BF16-led summary with GATED marker (METRICS-01, METRICS-02)
  - [x] 01-03-PLAN.md — De-hardcode competitor constants into `literature_estimates.py`, delete orphaned JSON (METRICS-03)
- [x] **Phase 2: Eval Subsystem** - Reproducible WikiText-2 perplexity proves quality is preserved on real weights (completed 2026-07-08)
- [ ] **Phase 3: Cascade Correction** - Flagship 5-stage cascade fixed to honest error bounds or honestly scoped as experimental
- [ ] **Phase 4: Calibration & Neuroanatomy Foundation** - Torch-free activation capture enables real-weight importance scoring and neuroanatomy profiling
- [ ] **Phase 5: Dynamic Reduction & Architecture Optimization** - Per-layer importance drives adaptive compression; math-primitive exploration
- [ ] **Phase 6: Registry & Depth Reduction** - Split 2,964-method registry into validated active set + labeled experimental namespace; kill broken auto-discovery
- [ ] **Phase 7: Format Transparency** - Separate SSF container overhead from algorithmic ratio; pack small tensors
- [ ] **Phase 8: Documentation & Honest Framing** - README/docs tell the truth; research catalog maturity documented

## Phase Details

### Phase 1: Metrics Trust Loop

**Goal**: Every reported compression ratio is error-gated and measured end-to-end through `serialized_nbytes()`; no method with high error can emit a ratio, and fabricated competitor comparisons are removed.
**Mode**: mvp
**Depends on**: Nothing (first phase)
**Requirements**: METRICS-01, METRICS-02, METRICS-03, METRICS-04
**Success Criteria** (what must be TRUE):

  1. A method whose `rel_mse` exceeds the configured threshold emits no compression-ratio report (gated) — verifiable by running a deliberately-bad method and confirming no ratio appears in output.
  2. All ratio reporting flows through `serialized_nbytes()` (byte-exact); `ratio_vs_disk` (BF16) is the default headline and `ratio_vs_fp32` appears only as a secondary annotation.
  3. `benchmark_industry_comparison.json` and `certificate.py` (lines ~440-463) contain no hardcoded competitor constants; any external reference is labeled "literature estimates, not measured here."
  4. `tests/test_honest_metrics.py` exists and passes — asserting ratio↔error coupling, rejection of over-threshold methods, and `serialized_nbytes` handling all payload shapes.

**Plans**: 3/3 plans complete

### Phase 2: Eval Subsystem

**Goal**: A reproducible, independent eval subsystem proves quality is preserved by measuring WikiText-2 perplexity on original vs compressed weights, closing the project's single biggest trust gap.
**Mode**: mvp
**Depends on**: Phase 1 (eval reports must use honest metrics)
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):

  1. An `eval/` subsystem runs WikiText-2 perplexity (seq len 2048) on original FP16 vs compressed model and emits a verifiable JSON artifact with both PPL values and a recovery ratio.
  2. A default tokenizer ships and `BaseTokenizer.encode` no longer raises `NotImplementedError`; a test asserts encode/decode round-trips on sample text.
  3. Model loading uses an env var / CLI arg instead of the hardcoded `/home/mike/.../gemma-4-E2B` path; README documents the required model for reproducing results.
  4. A recovery gate (`compressed/base ≥ configured threshold`) is enforced and reported in the eval artifact.

**Plans**: 4/4 plans complete
**Wave 1**

- [x] 02-01-PLAN.md — WikiText-2 perplexity grader with honest JSON artifact (EVAL-01)
- [x] 02-02-PLAN.md — BaseTokenizer byte-identity fallback and default tokenizer round-trip (EVAL-02)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02-03-PLAN.md — Parameterize model paths and document required model in README (EVAL-03)
- [x] 02-04-PLAN.md — Wire model-native tokenizer into eval (D-02 gap closure)

### Phase 3: Cascade Correction

**Goal**: The flagship 5-stage cascade is either fixed to honest error bounds or honestly scoped — it never reports as a working headline method when it is not.
**Mode**: mvp
**Depends on**: Phase 1 (metrics gate enforces the `rel_mse` threshold)
**Requirements**: CASCADE-01, CASCADE-02
**Success Criteria** (what must be TRUE):

  1. If the cascade runs as a default method, it produces `rel_mse < 0.05` on real weight slices (verifiable artifact); otherwise it is removed as a default/headline method and relocated to the experimental namespace.
  2. The CLI no longer advertises the cascade as a working headline method unless it passes the error gate; running it on real weights yields honest scope messaging, not a false ratio.
  3. Stages 3–5 (sparse, ergodic, SIREN) are either wired into the compress path and validated, OR documentation explicitly states only stages 1–2 are live — with no contradictory claim remaining anywhere in code or docs.
  4. A documented decision record states the cascade's current status (live vs experimental) consistent with what the code actually does.

**Plans**: TBD

### Phase 4: Calibration & Neuroanatomy Foundation

**Goal**: A torch-free calibration subsystem captures per-layer activations, enabling real-weight importance scoring and LLM neuroanatomy profiling — the critical substrate for the entire R&D track.
**Mode**: mvp
**Depends on**: Phase 2 (eval validates quality); Phase 1 (metrics integrity)
**Requirements**: RND-01, RND-02
**Success Criteria** (what must be TRUE):

  1. A `calibration/` subsystem captures per-layer activation statistics (e.g., X, Hessian H⁻¹) from a dataset via a tokenizer + forward hook, torch-free, and exposes an activation-stats protocol consumed by `methods/`.
  2. Weight-activation importance scoring (APoZ / Wanda-class / SparseGPT) runs on real weight slices and outputs a per-tensor importance metric validated against a sanity check on the eval model.
  3. A neuroanatomy profiling pipeline reports per-layer singular-value distributions, activation sparsity patterns, and cross-layer correlation, emitted as a verifiable artifact on a real model.
  4. Every RND output is gated behind "runs on real weights" — no method claims validity without a real-weight result artifact.

**Plans**: TBD

### Phase 5: Dynamic Reduction & Architecture Optimization

**Goal**: Given per-layer importance metrics, the engine adaptively chooses compression/pruning parameters, and explores deeper architecture math primitives — prune-first, never quant-then-prune.
**Mode**: mvp
**Depends on**: Phase 4 (needs calibration + importance scoring)
**Requirements**: RND-03, RND-04
**Success Criteria** (what must be TRUE):

  1. A dynamic per-layer reduction routine selects compression parameters or pruning ratio per layer from importance metrics and emits a per-layer plan artifact (verifiable JSON).
  2. The dynamic plan, when applied, is validated by the Phase 2 eval subsystem (recovery gate) and shows no worse-than-baseline degradation on the eval model.
  3. At least one architecture-math optimization (alternative factorization schedule or hybrid spectral-decomposition variant) is prototyped and benchmarked against the current baseline with a recorded result.
  4. Pruning is applied on dense weights before quantization (prune-first invariant) and enforced in the pipeline, with a test asserting the order.

**Plans**: TBD

### Phase 6: Registry & Depth Reduction

**Goal**: The 2,964-method registry is split into a validated active set and a labeled experimental namespace, and the broken walk-based auto-discovery is fixed or removed.
**Mode**: mvp
**Depends on**: Phase 1 (honest metrics to classify the validated set); parallelizable with eval/calibration
**Requirements**: REGISTRY-01, REGISTRY-02
**Success Criteria** (what must be TRUE):

  1. The registry is split into a validated active set (methods with passing tests + real-weight results) and a labeled `experimental/` namespace; the active set is documented.
  2. A bounded registry-size test asserts the active set count matches an expected small number (methods that actually run on real weights), preventing silent catalog inflation.
  3. `_discover_by_walk` is fixed or removed; if removed, no code path calls it and a test asserts discovery returns the expected methods (not 0).
  4. Loading the engine lists only validated methods by default, with experimental accessible via an explicit flag.

**Plans**: TBD

### Phase 7: Format Transparency

**Goal**: SSF container overhead is reported separately from algorithmic compression, and small tensors are packed to reduce padding.
**Mode**: mvp
**Depends on**: Phase 1 (metrics consistency); parallelizable
**Requirements**: FORMAT-01, FORMAT-02
**Success Criteria** (what must be TRUE):

  1. Compression reports show algorithmic ratio separate from SSF container overhead (per-tensor 4096B alignment + 256B header + 128B footer), so padding is never counted as compression.
  2. Small tensors are packed into shared pages instead of per-tensor alignment, reducing stored size — verifiable by comparing on-disk size before/after on a real model.
  3. A report artifact documents both the raw algorithmic ratio and the true on-disk ratio with overhead broken out line-by-line.

**Plans**: TBD

### Phase 8: Documentation & Honest Framing

**Goal**: README and docs honestly frame what works, label aspirational numbers as targets, and explain the research catalog maturity.
**Mode**: mvp
**Depends on**: Phase 6 (registry split), Phase 7 (format overhead), Phase 3 (cascade status)
**Requirements**: DOCS-01, DOCS-02
**Success Criteria** (what must be TRUE):

  1. README, docs/QUANTIZATION.md, and compression-roadmap.md state honest ~4–6× vs FP32 capability, with all aspirational numbers explicitly labeled as targets/theoretical.
  2. The method registry is described as a research catalog with maturity levels, and no claim presents 2,964 methods as a working toolkit.
  3. A "Research Catalog" document exists explaining maturity levels of experimental methods and how to access the experimental namespace.
  4. A fresh clone can follow docs to reproduce the headline number without encountering fabricated comparisons or hardcoded paths.

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Metrics Trust Loop | 3/3 | Complete    | 2026-07-08 |
| 2. Eval Subsystem | 4/4 | Complete    | 2026-07-08 |
| 3. Cascade Correction | 0/0 | Not started | - |
| 4. Calibration & Neuroanatomy Foundation | 0/0 | Not started | - |
| 5. Dynamic Reduction & Architecture Optimization | 0/0 | Not started | - |
| 6. Registry & Depth Reduction | 0/0 | Not started | - |
| 7. Format Transparency | 0/0 | Not started | - |
| 8. Documentation & Honest Framing | 0/0 | Not started | - |

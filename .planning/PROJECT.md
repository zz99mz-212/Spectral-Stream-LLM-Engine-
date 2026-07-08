# Spectral-Stream LLM Engine

## What This Is

Spectral-Stream is a **pure-Python (NumPy/SciPy, no real C++), CPU-targeted LLM weight-compression and KV-cache compression engine** that also runs as a local, OpenAI-compatible inference server. It compresses model weights by treating them as continuous manifolds / spectral fields / physical systems and removing redundant structure through a registry of ~3,000 candidate methods, then serves the compressed model on consumer hardware. Status: **ACTIVE R&D — NOT PRODUCTION READY** (AGPL-3.0).

## Core Value

**Compress real LLM weights honestly and run them on CPU** — every reported ratio and error must be a measured, byte-exact end-to-end number, never an estimate or a hardcoded constant.

## Requirements

### Validated

_(Inferred from the existing codebase — `spectralstream/` + the committed map in `.planning/codebase/`.)_

- ✓ **Method-registry compression engine** — `CompressionIntelligenceEngine` (`compression/engine/_orchestrator.py:206`) profiles → allocates → selects → compresses → validates over a 2,964-method registry across 43 categories (`compression/methods/`).
- ✓ **Working quantization path** — `block_int8` / `block_int4` / Hadamard-int8 + entropy coding delivers the only verified result: **~4–6× vs FP32 (≈2–3× vs BF16) on real Gemma-4 E2B weights** with 2.3–34% error by bit-width (`run_full_model_honest_results.json`, `final_benchmark_results.json`).
- ✓ **5-stage cascade framework** — `FiveStageCascade` (`compression/cascade_5stage.py:652`) defines EinSort → TT-SVD → Sparse → Ergodic → SIREN (currently only stages 1–2 are wired).
- ✓ **Honest-metrics layer** — `honest_metrics.py` provides byte-exact `serialized_nbytes()`, `end_to_end_error()`, `dual_ratio()` and `ErrorMetrics(rel_mse, cosine_sim, max_abs, snr_db)` as the canonical measurement path.
- ✓ **SSF serialization + converters** — SSF v2/v3, SST v3, SSCX containers (`format/`) plus safetensors/GGUF readers/writers; `zstandard` compression.
- ✓ **Inference pipeline + strategy stack** — `InferencePipeline` (`inference/pipeline.py:62`) and a 6-level unified engine (FORWARDLESS/HDC → RESONANT/Vlasov → SPECTRAL_BLOCK → SPECTRAL_VERIFY → STANDARD → FALLBACK) with exotic R&D engines (Vlasov mean-field attention, HDC, HRR, COCONUT, TimeCrystal resonance).
- ✓ **KV-cache compression** — `KVCacheManager` unifying 30+ eviction policies (spectral, h2o, sliding, resonance, etc.) and compression methods (fwht, dct, svd, e8_lattice).
- ✓ **Serving layer** — FastAPI OpenAI-compatible server (`serving/unified_server.py`), batching engine, production stack.
- ✓ **Test suite** — 83 test files / ~2,758 pytest functions (concentrated on `format/` + engine dataclasses).

### Active

_(The work this project is being steered toward — the user's stated direction: honest 10–60× weight compression, 200–600× aspirational targets, leveraging the breadth of R&D methods.)_

- [ ] **ACHV-01**: Achieve a real, honestly-measured **10–60× weight compression** (the practical target) end-to-end on a real model, validated by `honest_metrics`.
- [ ] **ACHV-02**: Advance toward the **200–600× aspirational ratio targets** via multiplicative cascades (e.g., decomposition → spectral → quantization → entropy), gated by real reconstruction + functional error.
- [ ] **ACHV-03**: Fix the 5-stage cascade so it actually compresses (currently 72–92% reconstruction error on real weights) — either wire/correct stages 3–5 or down-scope the advertised pipeline to match code.
- [ ] **ACHV-04**: Make honesty enforceable — add tests for `honest_metrics.py` and a CI gate that rejects ratios reported without paired error, eliminating fabricated comparisons (GPTQ/AWQ/GGUF hardcoded constants).
- [ ] **ACHV-05**: Establish **downstream quality validation** (perplexity / task accuracy on the compressed vs original model) to substantiate the "quality preserved" claim.
- [ ] **ACHV-06**: Prune/quarantine the 2,964-method registry to working + tested methods; bound registry size and remove dead/branded SVD wrappers.
- [ ] **ACHV-07**: Improve throughput/scaling realism (per-tensor time budgets, worker pool, realistic single-pass runtime for larger models).

### Out of Scope

- **GPU / accelerated backend** — engine is explicitly CPU/NumPy research-grade; `torch` is an optional extra only. Production serving is a different class of project.
- **Training from scratch / distillation to reach 800×+** — `REAL_WORLD_BENCHMARK.md` shows post-training compression is Shannon-capped near ~6:1; 800×+ requires BitNet/low-rank training, a separate effort.
- **Cloud hosting / multi-tenant serving** — local single-process / local FastAPI only; no containers/CI/deploy manifests today.
- **Non-Python / C++ core rewrite** — `CMakeLists.txt` is a stub; the project is pure Python by design.
- **Fabricated or estimated benchmark comparisons** — explicitly excluded per the honesty mandate; only measured-in-repo methods may be compared.

## Context

- **Brownfield, R&D-phase project.** The repo carries a documented history of inflated/non-honest metrics (commits `c66016e` "replace fabricated metrics with true end-to-end compression measurements", `b213e4f` "honest-metrics-windows-compat"). `compression-roadmap.md` itself documents past accounting bugs (e.g., `len(dict)` mis-used as byte length producing fake 588:1/2000:1 ratios).
- **The ratio numbers are a spectrum, not a single fact.** Per the research docs: working reality is ~4–6× vs FP32 on real BF16 weights (flat singular-value spectrum, no exploitable low-rank structure → Shannon ceiling ≈ 6:1, `REAL_WORLD_BENCHMARK.md`). The "10–60×" framing comes from `docs/QUANTIZATION.md` ("6–60× better than GGUF Q4_K_M"); "200–400× realistic / 2000–5000× aspirational" from README/roadmap; "200–600×" blends these repo figures. KV-cache figures (3277:1 holographic phase, 332:1 quantum weight) are mostly theoretical or qualified ("at 75% retrieval similarity, not <0.02% loss"). Single-method/theoretical explorations reach far higher (Spectral Envelope 9915:1, Strange-Attractor 1,000,000:1) but are math upper-bounds, not measured on real weights.
- **Codebase map** at `.planning/codebase/` (7 docs) is the authoritative current-state reference for planning.

## Constraints

- **Tech stack**: Pure Python 3.10+, NumPy ≥1.24 / SciPy ≥1.10 / psutil / safetensors / zstandard. No real C/C++ extension. Optional extras: `web` (fastapi/uvicorn/jinja2/pydantic), `gguf`, `ml` (torch/sklearn/ml-dtypes), `finetune` (datasets), `dev` (pytest/rich).
- **Platform**: CPU-only inference target. No GPU requirement.
- **License**: AGPL-3.0.
- **Honesty mandate**: All ratios/errors MUST flow through `honest_metrics`; no `len(dict)` estimates, no hardcoded competitor tables (per `CONCERNS.md` SEC-03, HIGH).
- **Real-weight ceiling**: Post-training BF16 compression is Shannon-limited near ~6:1; larger claims require architectural/training changes, not post-hoc compression.
- **Reproducibility**: Benchmarks currently hardcode the author's model path (`models/gemma-4-E2B/model.safetensors`, gitignored) and use `signal.alarm`/`exec` timeouts that break on Windows (branch is `fix/honest-metrics-windows-compat`).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Registry-driven, tiered, lazy-evaluated engine | Maximal method breadth; only ~10 methods pre-loaded, rest lazy | ✓ Good for exploration; ⚠️ 2,964 registered / ~2 used in practice — needs pruning |
| Honesty-first metrics (`honest_metrics`) | Past fabricated-ratio history; ratios must be byte-exact | ✓ Right direction; ⚠️ not yet enforced by tests/CI |
| Pure-Python / CPU scope | Research-grade, dependency-light, runs on consumer hardware | ⚠️ Revisit — non-competitive vs real serving engines for latency |
| 5-stage cascade as flagship narrative | EinSort→TT-SVD→Sparse→Ergodic→SIREN is the conceptual through-line | ⚠️ Revisit — only stages 1–2 wired; 72–92% error on real weights |

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
*Last updated: 2026-07-08 after project initialization (brownfield, via /gsd-new-project)*

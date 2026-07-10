# Walking Skeleton — Spectral-Stream LLM Engine (Metrics Trust Loop)

**Phase:** 1
**Generated:** 2026-07-08

## Capability Proven End-to-End

> A compression method whose reconstruction error (`rel_mse`) exceeds the gate threshold emits **no compression-ratio number** and is instead marked `gated: True` with a `gate_reason` string, while a good method emits an honest, byte-exact ratio whose headline baseline is **BF16 (vs disk)** — proven by `pytest tests/test_honest_metrics.py` and observable through the real CLI `--honest-metrics` summary path.

This is a pure-Python CLI + library (an existing compression engine), not a web/DB app, so the "full stack" proven here is: byte-exact measurement primitive → central error-gate chokepoint → CLI emission → passing test suite.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Gate location | `honest_metrics.py` (central `apply_gate()` chokepoint) | All ratio emission must route through one pure function so no emitter can silently skip the error check (root cause of BUG-02). CLI, certificate, and future reporters all call it. |
| Headline baseline | `ratio_vs_bf16` is the default headline; `ratio_vs_fp32` is secondary | BF16/disk is the dtype the model actually ships as; FP32-led reporting (current CLI) was dishonest vs the on-disk reality. |
| Default threshold | `ERROR_GATE_THRESHOLD = 0.05` (`rel_mse`), strict `>` | Consistent with the Phase-3 cascade acceptance bound; one `UPPER_SNAKE_CASE` constant, not per-call-site literals. |
| Gated-but-not-dropped | Suppress ratio + emit `gated: True` + `gate_reason` | A silently missing ratio can be misread as "1.0× = no compression"; an explicit marker renders "GATED" instead. |
| External references | `literature_estimates.py` module, labeled "literature estimates, not measured here" | Moves fabrication risk out of the reporting path; `certificate.py` becomes a pure consumer. |
| Development style | Test-first (TDD) for the gate + `serialized_nbytes` shapes | The gate is the highest-leverage honesty fix; a failing test anchors it before implementation. |

## Stack Touched in Phase 1

- [x] Library core (`honest_metrics.py` — `apply_gate`, `ERROR_GATE_THRESHOLD`)
- [x] CLI emission path (`cli.py` — both honest-metrics blocks gated; summary BF16-led)
- [x] Test coverage (`tests/test_honest_metrics.py` — gate coupling, boundary, payload shapes, literature label)
- [x] "Deployment" = the green pytest suite + a `python -m spectralstream.compression.cli ... --honest-metrics` invocation that exercises the loop end-to-end

## Out of Scope (Deferred to Later Slices)

> Anything that is *not* in the skeleton. Be explicit — this list prevents future phases from re-litigating Phase 1's minimalism.

- Perplexity / downstream eval (Phase 2, EVAL-01..03)
- 5-stage cascade correctness / honest scoping (Phase 3, CASCADE-01..02)
- INT4, registry depth reduction, SSF format transparency (later phases)
- CI metrics-honesty lint + perplexity gate (v2, CI-01)

## Subsequent Slice Plan

Each later phase adds one vertical slice on top of this skeleton without altering its architectural decisions (gate lives in `honest_metrics.py`; BF16 is the headline; threshold is `0.05`):

- Phase 2: Reproducible WikiText-2 perplexity proves quality is preserved, using the gate from this phase.
- Phase 3: Flagship cascade either passes `< 0.05` or is honestly scoped, enforced by the same `apply_gate`.
- Phase 4: Torch-free calibration enables real-weight importance scoring.
- Phase 5: Dynamic per-layer reduction driven by importance metrics.
- Phase 6: Registry split into validated active set + experimental namespace.
- Phase 7: SSF format overhead reported separately from algorithmic ratio.
- Phase 8: Documentation honestly frames what works.

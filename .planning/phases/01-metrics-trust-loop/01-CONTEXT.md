# Phase 1: Metrics Trust Loop - Context

**Gathered:** 2026-07-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Every reported compression ratio is error-gated and measured end-to-end through
`serialized_nbytes()`. No method whose reconstruction error exceeds a configured
threshold may emit a compression-ratio number, and all fabricated competitor
comparisons are removed from the codebase or explicitly relabeled as external
literature estimates.

This phase covers: (1) an error gate coupling ratio↔error, (2) routing all
ratio reporting through the byte-exact `serialized_nbytes()`/`dual_ratio()` path
with BF16 as the headline, (3) de-hardcoding competitor constants, and (4) a new
`tests/test_honest_metrics.py`.

Out of scope: perplexity/eval (Phase 2), cascade correctness (Phase 3), INT4.
</domain>

<decisions>
## Implementation Decisions

### Error Gate Behavior
- When `rel_mse` exceeds the configured threshold, the ratio number is **suppressed
  and replaced with an explicit marker** (e.g. `gated: true`, `gate_reason="rel_mse
  0.11 > 0.05"`), not silently dropped. The method still runs; only its ratio claim
  is withheld.
- The gate is a central chokepoint in `honest_metrics.py` (a `gated_ratio(...)` /
  `apply_gate(...)` helper) so every emission path flows through it.
- Threshold is **0.05 rel_mse** by default, consistent with the Phase 3 cascade
  acceptance bound. It is configurable (constant + optional config/env override),
  not hardcoded at every call site.

### Byte-Exact Ratio Reporting
- All surfaced ratios derive from `serialized_nbytes()` (via `dual_ratio`), never
  `len(dict)` or products of per-stage estimates.
- `ratio_vs_bf16` (BF16 / "vs disk") is the **default headline**; `ratio_vs_fp32`
  appears only as a secondary annotation.
- The CLI already computes `dual_ratio` per tensor — extend/confirm that the
  headline summary leads with BF16 and labels FP32 as secondary.

### Competitor Comparison Constants
- Move the 9 hardcoded competitor constants out of `certificate.py` into a new
  module (e.g. `spectralstream/compression/literature_estimates.py`) whose values
  are explicitly labeled **"literature estimates, not measured here."**
- `certificate.py` imports these; it contains no bare competitor numbers itself.
- Delete `benchmark_industry_comparison.json` (fabricated, unmeasured, unreferenced
  by any code or test).
- Preserve existing `test_certificate*` behavior: `industry_comparison` still
  exposes `comparisons`, `beats_standard_quant`, `beats_int4`, rank, etc.

### Tests (tests/test_honest_metrics.py)
- **Priority anchor: ratio↔error coupling gate** — a deliberately-bad (high
  rel_mse) method emits no ratio (gated), a good one emits a ratio.
- Also cover: `serialized_nbytes` across all payload shapes (bytes, ndarray,
  nested dict/list, tuples, scalars, None), rejection of over-threshold methods,
  and that the literature-estimate label is present / no fabricated constants leak.

### Claude's Discretion
- Exact helper names, config key/env-var name for the threshold, and module path
  for literature estimates are at Claude's discretion, following existing naming
  conventions (`honest_metrics.py`, `UPPER_SNAKE_CASE` constants, `_`-prefixed
  private helpers).
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `spectralstream/compression/honest_metrics.py` — already byte-exact:
  `serialized_nbytes`, `end_to_end_error` (returns `ErrorMetrics` with `rel_mse`,
  `cosine_sim`, `max_abs`, `snr_db`), `dual_ratio`, `ratio_vs_fp32/bf16`. The gate
  belongs here.
- `spectralstream/compression/cli.py` (~L839-865) already builds a per-tensor
  `honest_metrics_dict` from `dual_ratio` + `end_to_end_error`, and prints an
  "Honest ratio (vs FP32/BF16)" summary (~L1414-1425).
- `spectralstream/compression/engine/_helpers.py` — `_compute_metrics`,
  `_compute_ratio`, `_enrich_meta` populate `compression_ratio`/`relative_error`
  into metadata.

### Established Patterns
- `from __future__ import annotations` header mandatory on every module.
- Methods return `(payload_bytes, metadata_dict)`; ratios flow into `meta
  ["compression_ratio"]`.
- Tests live in top-level `tests/`, `test_*.py`, box-drawing section banners.

### Integration Points
- Certificate constants: `certificate.py` `_compute_industry_comparison` (~L429-463)
  and `comparisons` list (~L432-443).
- Fabricated artifact: `benchmark_industry_comparison.json` (repo root).
- Gate consumers: CLI honest-metrics summary; `CompressedTensor.compression_ratio`
  surfaced in `cli.py` compress loop and certificate builders.
</code_context>

<specifics>
## Specific Ideas

- Gate marker should be inspectable in the per-tensor honest_metrics dict so the
  CLI can render "GATED" instead of a fabricated ratio.
- Keep the change additive: existing `dual_ratio`/`serialized_nbytes` signatures
  must not break current callers or `test_certificate*`.
</specifics>

<deferred>
## Deferred Ideas

- CI metrics-honesty lint + perplexity gate — already deferred to v2 (STATE.md).
- Perplexity-based quality proof — Phase 2.
</deferred>

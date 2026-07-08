---
phase: 01
plan: 01
subsystem: honest_metrics
tags: [metrics, trust-loop, tdd, gate, error-coupling]
dependency_graph:
  requires: []
  provides: [apply_gate, ERROR_GATE_THRESHOLD]
  affects: [cli.py World-Model honest-metrics block]
tech_stack:
  added: [apply_gate pure function]
  patterns: [central chokepoint for ratio emission]
key_files:
  created: [tests/test_honest_metrics.py]
  modified: [spectralstream/compression/honest_metrics.py, spectralstream/compression/cli.py]
decisions:
  - "Single ERROR_GATE_THRESHOLD = 0.05 constant controls gate; no per-call-site thresholds"
  - "apply_gate is THE ONLY ratio-emission decision point; both CLI blocks and future reporters route through it"
  - "Strict > boundary on rel_mse gates (at exactly threshold, NOT gated); consistent with Phase-3 cascade acceptance"
  - "High-error methods suppress ratio (None) but retain error metrics; no fabricated 0.0x/1.0x"
metrics:
  duration: ~25 minutes
  completed_date: "2026-07-08"
status: complete
---

# Phase 1 Plan 01: Honest Metrics Trust Loop — Walking Skeleton Summary

## One-Liner

Established the central `apply_gate()` chokepoint that couples every compression ratio to its reconstruction error, gating high-error methods (suppressing their ratio) while low-error methods retain byte-exact ratios via `serialized_nbytes` — verified by a RED→GREEN TDD cycle.

## Objective Achievement

Plan 01 delivered the highest-leverage honesty fix for BUG-02: a single decision point where every compression ratio is gated by its reconstruction error. The `apply_gate()` function provides:

1. **Single source of truth**: One `ERROR_GATE_THRESHOLD = 0.05` constant governs all gating
2. **Central chokepoint**: All ratio emission flows through `apply_gate()`; no scattered per-call-site thresholds
3. **Strict boundary**: At exactly `rel_mse == 0.05`, NOT gated (uses strict `>`, consistent with Phase-3 cascade acceptance criteria)
4. **Error retention**: Gate suppresses the ratio claim but always retains `rel_mse`, `cosine_sim`, `max_abs`, `snr_db` — never a fabricated zero
5. **Byte-exact measurement**: Ratios derived from `serialized_nbytes()` (honest_metrics.py:25), never by inference or estimate

## Tasks Completed

### Task 1 (RED): Failing honest-metrics test file
- **File**: `tests/test_honest_metrics.py` (created)
- **Action**: Wrote tests asserting ratio↔error coupling before `apply_gate` existed
- **Result**: Fails with `ImportError: cannot import name 'apply_gate'` (RED state proven)
- **Commit**: `test(01-01): add failing honest_metrics apply_gate + serialized_nbytes shapes test` (80bc392)

### Task 2 (GREEN): Implement apply_gate + gate World-Model CLI block
- **File**: `spectralstream/compression/honest_metrics.py` (modified)
  - Added `ERROR_GATE_THRESHOLD = 0.05` module constant (line 24)
  - Added `apply_gate(payload, original_elements, rel_mse, threshold=ERROR_GATE_THRESHOLD) -> Dict[str, Any]` function (after `dual_ratio`, line ~164)
- **File**: `spectralstream/compression/cli.py` (modified)
  - Updated import to include `apply_gate` (line 30)
  - World-Model honest-metrics block (lines 838-867) now calls `apply_gate(data, original.size, err.rel_mse)` instead of directly assigning `dual_ratio` results
  - Gate result merged into `honest_metrics_dict`; gated tensors get `gated:True` + `ratio_vs_bf16:None`
- **Result**: All 14 tests pass (GREEN state achieved)
- **Commit**: `feat(01-01): implement apply_gate chokepoint + gate World-Model CLI block` (7e021a5)

## Test Coverage

`tests/test_honest_metrics.py` covers:

| Test | Assertions |
|------|-----------|
| `test_good_method_emits_ratio` | Low-error method: `gated is False`, both ratios not None |
| `test_bad_method_gated` | High-error method: `gated is True`, both ratios None, `"rel_mse" in gate_reason` |
| `test_boundary_exactly_threshold_not_gated` | `rel_mse == 0.05`: `gated is False` (strict >) |
| `test_boundary_just_above_gated` | `rel_mse == 0.05 + 1e-6`: `gated is True` |
| `test_serialized_nbytes_shapes` (parametrized, 11 cases) | `serialized_nbytes` correctly counts bytes for None, bytes, bytearray, ndarray, np.generic, dict, list, tuple, bool, int, float, str |

## Deviations from Plan

None — plan executed exactly as specified.

## Auth Gates

None encountered.

## Known Stubs

None.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: integrity | `spectralstream/compression/honest_metrics.py::apply_gate` | Central ratio-emission decision point; all CLI blocks and future reporters MUST route through it (not per-call-site) |

## Code Quality Notes

- Both files maintain `from __future__ import annotations` header and 4-space indentation
- Box-drawing ASCII banners follow project convention (test_certificate_comprehensive.py pattern)
- `apply_gate` is pure (no side effects); returns a dict with explicit keys
- CLI block preserves error metrics and only replaces ratio computation with the gate result

## Verification Status

- [x] `ERROR_GATE_THRESHOLD = 0.05` constant exists in honest_metrics.py
- [x] `apply_gate` function exists with signature `(payload, original_elements, rel_mse, threshold=ERROR_GATE_THRESHOLD)`
- [x] High-error methods gated (ratio suppressed to None); good methods retain byte-exact ratios
- [x] World-Model CLI block (lines ~838-867) routes through `apply_gate`
- [x] `tests/test_honest_metrics.py` created, begins with `from __future__ import annotations`, PASSES (14 tests)
- [x] RED→GREEN TDD cycle completed: failing test (80bc392) before implementation (7e021a5)

## Next Steps

This gate is the foundation for Phase 1 Plans 02-03:
- Plan 02: Gate the second CLI block + update BF16-led summary with GATED marker
- Plan 03: De-hardcode competitor constants into `literature_estimates.py`

All later phases (eval, cascade, registry, format) route through this gate to ensure honesty.

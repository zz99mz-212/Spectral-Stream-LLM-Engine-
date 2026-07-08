---
phase: 01-metrics-trust-loop
plan: 02
subsystem: compression-cli
tags: [metrics, honest-ratios, gate, bf16, fp32, gated]
requires:
  - 01-01
provides:
  - gated-second-block
  - bf16-led-summary
  - gated-marker
  - none-filter
affects:
  - cli.py
  - tests/test_honest_metrics.py
tech_stack:
  added: []
  patterns:
    - apply_gate chokepoint (both CLI blocks)
    - BF16-led summary ordering
    - None-filter for gated tensors
    - GATED marker emission
key_files:
  created: []
  modified:
    - spectralstream/compression/cli.py
    - tests/test_honest_metrics.py
decisions:
  - "Both CLI honest-metrics blocks route through apply_gate (no ratio leak)"
  - "Summary leads with BF16 (vs disk), demotes FP32 to secondary"
  - "Gated tensors yield None in honest_metrics (not 0), so means filter them out"
  - "GATED marker emitted when any tensor exceeds threshold"
metrics:
  duration: 0.3 hours
  completed: 2026-07-08
status: complete
---

# Phase 01 Plan 02: Finish Honest-Metrics Vertical Slice Summary

Gate the second CLI honest-metrics block (so no emission path leaks a ratio past the gate),
and make the CLI summary honest — lead with BF16 (vs disk), demote FP32 to secondary,
filter gated `None`s out of the mean, and render a "GATED" marker.

This completes the metric-trust-loop vertical slice: both emission paths are now gated by
`apply_gate()`, and the surfaced numbers are honest by construction.

## Objective

METRICS-01 requires BOTH blocks gated; METRICS-02 requires BF16 as the headline baseline.
This was the user-visible end of the trust loop proven by Plan 01's `apply_gate` chokepoint.

## Tasks Completed

### Task 1: Gate the second CLI block + reorder summary to BF16-led

**Files modified:**
- `spectralstream/compression/cli.py`

**Changes:**
1. Removed the `dual_ratio()` call in the second CLI block (lines 1278-1280) and replaced
   with `apply_gate()` call (lines 1293-1294 after MethodDiscovery + decompression + error calc).
   - **Result:** Both CLI blocks (World-Model + second) now route through the central
     `apply_gate()` chokepoint. No emission path can leak a ratio past the gate.
   - Verification: `grep -c "dual_ratio(" cli.py` == 0; `grep -c "apply_gate(" cli.py` == 2

2. Updated the honest-metrics summary block (lines 1394-1443):
   - Changed `.get("ratio_vs_bf16", 0)` and `.get("ratio_vs_fp32", 0)` defaults to `None`
     so a gated tensor yields `None`, not `0` (means will filter it out).
   - Added filtering before computing means: `hm_bf16_valid = [r for r in hm_bf16 if r is not None]`
     and `hm_fp32_valid = [r for r in hm_fp32 if r is not None]`.
   - Reordered so BF16 leads: "Honest ratio (vs BF16 / disk)" printed BEFORE "(secondary, vs FP32)".
   - Added GATED marker: if any compressed tensor has `c.get("honest_metrics", {}).get("gated")`,
     emit a log line with the count.
   - Verification: `grep -n "vs BF16 / disk" cli.py` and `grep -n "GATED" cli.py` present.

**Acceptance criteria met:**
- [x] `grep -c "dual_ratio(" spectralstream/compression/cli.py` == 0 (both sites replaced by apply_gate)
- [x] `grep -c "apply_gate(" spectralstream/compression/cli.py` == 2 (World-Model + second block)
- [x] BF16-led summary line present, printed before FP32 secondary
- [x] GATED marker log line emitted when any tensor is gated
- [x] `pytest tests/test_honest_metrics.py` still passes (existing Plan 01 tests unaffected)

### Task 2: Add headline-ordering + None-filter test anchor

**Files modified:**
- `tests/test_honest_metrics.py`

**Changes:**
Appended three new test functions after the existing Plan 01 tests:

1. `test_bf16_is_headline_key`: For a good method (`rel_mse <= threshold`), `apply_gate()`
   result has both `ratio_vs_bf16` and `ratio_vs_fp32` non-None; asserts `gated is False`
   (BF16 is the valid surfaced baseline because gating did not trip).

2. `test_gated_ratio_is_none_not_zero`: For a bad method (`rel_mse > threshold`), `apply_gate()`
   returns `ratio_vs_bf16 is None` and `ratio_vs_fp32 is None` (never 0.0x). Asserts exactly
   that — the summary None-filter depends on `None`, not `0`.

3. `test_gate_reason_format`: For `rel_mse=0.11, threshold=0.05`, asserts
   `gate_reason == "rel_mse 0.1100 > 0.05"` (human-readable format).

**Acceptance criteria met:**
- [x] Test file contains `test_bf16_is_headline_key`, `test_gated_ratio_is_none_not_zero`,
      `test_gate_reason_format`
- [x] All three new tests pass (`pytest tests/test_honest_metrics.py -k "headline or none or gate_reason"` exits 0)
- [x] Full file passes (`pytest tests/test_honest_metrics.py -v` -> 18 passed)

## Deviations from Plan

None — plan executed exactly as written.

## Verification

**Static checks (grep):**
```bash
grep -c "dual_ratio(" spectralstream/compression/cli.py        # → 0
grep -c "apply_gate(" spectralstream/compression/cli.py        # → 2
grep -n "vs BF16 / disk" spectralstream/compression/cli.py     # → 1433
grep -n "GATED" spectralstream/compression/cli.py              # → 1422, 1431
```

**Dynamic checks (pytest):**
```bash
python -m pytest tests/test_honest_metrics.py -v               # → 18 passed
python -m pytest tests/test_honest_metrics.py -k "headline or none or gate_reason"  # → 4 passed
```

## Integration Notes

- Both CLI honest-metrics emission paths now route through `apply_gate()` (World-Model + second).
- The summary block correctly filters gated tensors (None, not 0) before computing means.
- BF16 is surfaced as the headline baseline; FP32 is demoted to a secondary annotation.
- GATED marker provides user-visible feedback when the trust loop suppresses a ratio.
- No existing tests were modified; the new tests follow the same box-drawing banner style.

## Success Criteria — FULLY MET

- [x] Every CLI ratio emission path is gated; summary leads with BF16 and demotes FP32.
- [x] Gated tensors are excluded from means and marked "GATED".
- [x] Tests anchor BF16 headline + gated-None behavior.

## Known Stubs

None.

## Threat Flags

None — both blocks route through `apply_gate()` which is the central chokepoint established
in Plan 01 (T-01-01 mitigated). This plan mitigates T-01-03 (second block leak) and T-01-04
(BF16-led ordering honesty).

## Commits

- `6dd0d81` feat(01-02): gate second CLI block + reorder summary to BF16-led
- `168a6fd` test(01-02): add headline-ordering + None-filter test anchors

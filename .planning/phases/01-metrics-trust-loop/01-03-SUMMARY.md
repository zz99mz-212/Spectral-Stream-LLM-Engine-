---
phase: 01-metrics-trust-loop
plan: 03
subsystem: compression
tags: [metrics, honest-metrics, literature-estimates, certificate, disclaimer]

# Dependency graph
requires:
  - phase: 01-01
    provides: apply_gate chokepoint, ERROR_GATE_THRESHOLD
  - phase: 01-02
    provides: BF16-led summary, GATED marker
provides:
  - literature_estimates module (9 competitor tuples + disclaimer)
  - certificate.py as pure consumer of literature_estimates
  - disclaimer rendering in to_text/to_markdown
  - deleted orphaned benchmark_industry_comparison.json
affects: [phase-2-eval, phase-6-registry]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "External references must be labeled via LITERATURE_DISCLAIMER (anti-fabrication)"
    - "Certificate.py consumes pre-computed constants instead of hardcoding literals"

key-files:
  created:
    - spectralstream/compression/literature_estimates.py
  modified:
    - spectralstream/compression/certificate.py
    - tests/test_honest_metrics.py

key-decisions:
  - "Extracted 9 competitor tuples to literature_estimates.py with LITERATURE_DISCLAIMER label"
  - "Current-run row ('SpectralStream (current)') stays computed at runtime, not in literature_estimates"
  - "Industry comparison contract structure unchanged: keys, math, format preserved"

patterns-established:
  - "Centralized external estimates: all competitor numbers flow from literature_estimates module"

requirements-completed: [METRICS-03]

coverage:
  - id: D1
    description: "Literature estimates module created with 9 competitor tuples + disclaimer"
    requirement: "METRICS-03"
    verification:
      - kind: unit
        ref: "tests/test_honest_metrics.py#test_literature_disclaimer_present"
        status: pass
    human_judgment: false
  - id: D2
    description: "Certificate.py imports from literature_estimates, no hardcoded competitor literals"
    requirement: "METRICS-03"
    verification:
      - kind: unit
        ref: "tests/test_honest_metrics.py#test_no_competitor_literals_in_certificate_source"
        status: pass
    human_judgment: false
  - id: D3
    description: "Industry comparison dict contract preserved (keys + math unchanged)"
    requirement: "METRICS-03"
    verification:
      - kind: unit
        ref: "tests/test_certificate_comprehensive.py::TestIndustryComparison (3 tests)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Disclaimer rendered in to_text() and to_markdown() output"
    requirement: "METRICS-03"
    verification:
      - kind: unit
        ref: "tests/test_honest_metrics.py#test_disclaimer_rendered_in_certificate"
        status: pass
    human_judgment: false

# Metrics
duration: 45min
completed: 2026-07-08
status: complete
---

# Phase 1 Plan 3: De-hardcode Competitor Constants Summary

**Removed 9 hardcoded competitor constants from certificate.py; created literature_estimates.py with disclaimer label "literature estimates, not measured here"; preserved industry comparison contract; deleted orphaned JSON**

## Performance

- **Duration:** 45 min
- **Started:** 2026-07-08T18:45:37Z
- **Completed:** 2026-07-08T19:30:12Z
- **Tasks:** 2 (TDD RED + GREEN)
- **Files modified:** 3 (literature_estimates.py, certificate.py, test_honest_metrics.py)

## Accomplishments

- Created `spectralstream/compression/literature_estimates.py` with 9 competitor tuples marked as literature estimates
- Refactored `certificate.py` to import 9 tuples from literature_estimates instead of hardcoding them
- Preserved the `industry_comparison` dict contract exactly (keys, math, format, current-run row computation)
- Rendered the anti-fabrication disclaimer in both `to_text()` and `to_markdown()` output
- Deleted orphaned `benchmark_industry_comparison.json` from repo root (zero references confirmed)

## Task Commits

1. **Task 1 (RED): Create literature_estimates module + METRICS-03 tests** - `b6b73df` (test)
2. **Task 2 (GREEN): Refactor certificate.py + delete orphaned JSON** - `3a3f675` (feat)

**Plan metadata:** `RED` commit `b6b73df` precedes `GREEN` commit `3a3f675` per TDD+MVP gate enforcement.

## Files Created/Modified

- `spectralstream/compression/literature_estimates.py` - NEW: 9 competitor tuples + LITERATURE_DISCLAIMER
- `spectralstream/compression/certificate.py` - Import from literature_estimates; no hardcoded literals; render disclaimer
- `tests/test_honest_metrics.py` - 4 new METRICS-03 tests appended (literature disclaimer, competitor literals guard, contract preservation, rendered disclaimer)
- `benchmark_industry_comparison.json` - DELETED

## Decisions Made

- **External estimates blanket-labeled:** All competitor references (GPTQ/AWQ/SqueezeLLM/GGML Q) marked with "literature estimates, not measured here" disclaimer to prevent fabrication/misattribution surface
- **Current-run row NOT in literature_estimates:** ("SpectralStream (current)", ratio, ...) is computed at runtime from the actual `overall_ratio` variable, never hardcoded in the estimates module. This keeps the estimates module as pure external reference data
- **Contract structure preserved:** No change to `better_count`/`total_known`/`rank` math or keys (`comparisons`, `beats_standard_quant`, `beats_int4`, `rank`, `better_than_count`, `total_compared`). Only data source changed (imported vs inline)
- **Orphaned JSON deletion safe:** `benchmark_industry_comparison.json` had zero `.py`/`.json` references per grep; deletion confirmed safe

## Deviations from Plan

**None - plan executed exactly as written.**

## Issues Encountered

None.

**Note on unrelated test failures:** 4 round-trip file I/O tests in `tests/test_certificate_comprehensive.py` (test_txt_roundtrip, test_md_roundtrip, test_html_roundtrip, test_all_formats_roundtrip) fail on Windows due to UTF-8/CP1252 encoding mismatch in the test harness itself (test reads file without specifying `encoding="utf-8"`). This is a pre-existing Windows-specific test bug unrelated to METRICS-03. The 3 industry_comparison contract tests pass, confirming the contract is preserved. All METRICS-03 tests pass (22/22 in test_honest_metrics.py).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 1 Metrics Trust Loop is now **complete** (3/3 plans done)
- All compression ratios are error-gated + byte-exact (Plan 01-01, 01-02)
- Competitor comparisons are now non-fabricated and clearly labeled (Plan 01-03)
- Ready for Phase 2 (Eval Subsystem), Phase 6 (Registry Reduction), and other downstream work

---

*Phase: 01-metrics-trust-loop*
*Completed: 2026-07-08*

---
phase: 02-eval-subsystem
plan: 03
type: execute
wave: 2
status: complete
completed_date: 2026-07-09
duration_minutes: 15
tasks_total: 3
tasks_completed: 3
requirements:
  - EVAL-03
key_files_modified:
  - benchmark_physics_real_weights.py
  - wave4_pipeline.py
  - README.md
key_files_created: []
depends_on:
  - 02-01
---

# Phase 2 Plan 3: Remove Hardcoded Model Paths Summary

**One-liner:** Removed hardcoded author-specific `/home/mike/` paths from two scripts, wired them to the shared `resolve_model_path` helper with `--model` CLI+env var support, and documented the required Gemma-4 E2B model in README.

## Objective

Replace remaining hardcoded model paths with the shared `resolve_model_path` helper, add CLI `--model` support to both `benchmark_physics_real_weights.py` and `wave4_pipeline.py`, and update README to document the required Gemma-4 E2B model and how to supply it. This eliminates the environment-specific path leakage (SEC-02 / EVAL-03) and makes the phase reproducible on a fresh clone.

## Tasks

### Task 1: Parameterize benchmark_physics_real_weights.py

- **Status:** Complete
- **Commit:** `539d87d`
- **Actions:**
  - Added `from eval.model_path import resolve_model_path` import (after `sys.path.insert(0, ".")`)
  - Added argparse `--model` CLI flag with description
  - Replaced hardcoded `"/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"` with `resolve_model_path(args.model)`
  - Kept existing `signal.alarm`/`exec` timeout code unchanged per plan scope boundary
- **Verification:**
  - `grep "/home/mike"` returns nothing
  - `--help` lists `--model` flag
  - `--model /nonexistent` raises `FileNotFoundError` before model loading

### Task 2: Parameterize wave4_pipeline.py

- **Status:** Complete
- **Commit:** `3dc8bf0`
- **Actions:**
  - Added `from eval.model_path import resolve_model_path` import
  - Added argparse `--model` CLI flag in `main()` function
  - Replaced hardcoded `model_path = "models/gemma-4-E2B/model.safetensors"` with `resolve_model_path(args.model)`
  - Maintains relative fallback behavior via `resolve_model_path`'s internal default
- **Verification:**
  - `grep "/home/mike"` returns nothing
  - `--help` lists `--model` flag
  - `--model /nonexistent` raises `FileNotFoundError` before model loading

### Task 3: Document model path in README

- **Status:** Complete
- **Commit:** `fe02722`
- **Actions:**
  - Added new "Reproducing results" section after the status paragraph blockquote
  - Documents resolution order: `SPECTRALSTREAM_MODEL_PATH` env var → `--model PATH` → relative `models/gemma-4-E2B/model.safetensors` fallback
  - Concrete shell examples for benchmark, wave4, and eval scripts
  - Preserved existing "Current status" line intact
- **Verification:**
  - `SPECTRALSTREAM_MODEL_PATH` present (2 occurrences)
  - `--model` present (6 occurrences)
  - `Gemma-4 E2B` present (2 occurrences)
  - `models/gemma-4-E2B/model.safetensors` present (2 occurrences)

## Deviations from Plan

None. The plan was executed exactly as written. One minor import-reordering was required: `from eval.model_path import resolve_model_path` had to be placed after `sys.path.insert(0, ".")` in `benchmark_physics_real_weights.py` to match the script's existing import pattern (the `eval/` package is at the repo root). This is consistent with how other custom imports in that file work and is not a deviation.

## Threat Model Compliance

| Threat ID | Category | Severity | Disposition | Status |
|-----------|----------|----------|-------------|--------|
| T-02-03-01 | Tampering | high | mitigate | Both scripts delegate path validation to `resolve_model_path`, which rejects `..` traversal and verifies existence |
| T-02-03-02 | Information Disclosure | low | accept | README documents only user-supplied local path configuration |

Both threat mitigations are in place. No new threat surfaces were introduced.

## Threat Flags

None. All security-relevant surface (CLI `--model` → filesystem path) was already scoped in the threat model and is handled by the shared validation in `eval/model_path.py`.

## Known Stubs

None. All changes are complete wiring with no placeholder or mock data.

## Key Decisions

1. **Import order:** In `benchmark_physics_real_weights.py`, the `resolve_model_path` import is placed after `sys.path.insert(0, ".")` to match the script's existing pattern (other custom imports follow the same convention). In `wave4_pipeline.py` the same pattern is used consistently.

## Verification Summary

- `grep -rn "/home/mike"` on both target scripts: 0 hits
- `--help` on both scripts: `--model` flag listed
- README contains all 4 required strings: `SPECTRALSTREAM_MODEL_PATH`, `--model`, `Gemma-4 E2B`, `models/gemma-4-E2B/model.safetensors`
- Existing `signal.alarm`/`exec` code in `benchmark_physics_real_weights.py`: untouched
- "Current status" line in README: preserved

## Self-Check: PASSED

All claims verified:
- [x] benchmark_physics_real_weights.py: no `/home/mike` paths, `--help` works, `--model` accepted
- [x] wave4_pipeline.py: no `/home/mike` paths, `--help` works, `--model` accepted
- [x] README.md: all required strings present
- [x] signal.alarm code untouched
- [x] Current status line preserved
- [x] 3 commits created (1 per task)

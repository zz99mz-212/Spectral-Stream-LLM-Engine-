---
phase: 02-eval-subsystem
plan: 01
name: "WikiText-2 Perplexity Grader with Honest JSON Artifact"
type: tdd
wave: 1
subsystem: eval
tags:
  - eval
  - perplexity
  - grader
  - recovery-gate
  - vocab-blocked
dependency:
  requires:
    - Phase 1 (honest_metrics conventions)
  provides:
    - eval.run_eval CLI
    - eval.grader.grade
    - eval.artifact D-09 schema
  affects:
    - spectralstream/inference/pipeline.py (blocked log-softmax)
tech-stack:
  added:
    - eval package (7 modules)
    - scripts/fetch_eval_corpus.py
  patterns:
    - %-style lazy logging
    - eval.constants module for tunable defaults
    - TDD (RED -> GREEN -> REFACTOR)
key-files:
  created:
    - eval/__init__.py
    - eval/artifact.py
    - eval/constants.py
    - eval/corpus.py
    - eval/grader.py
    - eval/model_path.py
    - eval/run_eval.py
    - eval/data/wikitext2_sample.txt
    - eval/data/wikitext2_sample.tokens.json
    - scripts/fetch_eval_corpus.py
    - tests/test_eval_grader.py
  modified:
    - spectralstream/inference/pipeline.py
decisions:
  - "recovery_ratio = base_ppl / compressed_ppl (retained-quality fraction, per D-07 correction)"
  - "Gate threshold 0.95 configurable via --threshold (per D-08)"
  - "eval/ modules placed at project root (imported as eval.*, not spectralstream.eval.*)"
  - "Vocab-blocked log-softmax with BLOCK_SIZE=4096 to cap OOM (T-02-01-02)"
  - "Base model closed before compressed model loaded (sequential loading)"
metrics:
  duration: "~25 min"
  completed: "2026-07-09"
  tasks: 3 (RED, GREEN, REFACTOR)
  files_created: 11
  files_modified: 1
status: complete
---

# Phase 2 Plan 1: WikiText-2 Perplexity Grader with Honest JSON Artifact Summary

## One-Liner

Delivered the end-to-end eval grader (Slice 1): a runnable CLI (`eval/run_eval.py`) that measures WikiText-2 perplexity on base vs compressed models, computes recovery ratio as `base_ppl / compressed_ppl`, enforces the gate at threshold 0.95, and emits a D-09 JSON artifact with both measured PPL values. Patched `InferencePipeline.measure_perplexity` with vocab-blocked log-softmax to cap per-window logits memory from ~4 GB to ~64 MB.

## What Was Built

The eval subsystem is a standalone, importable Python package at the project root (`eval/`):

| Module | Purpose |
|--------|---------|
| `eval/__init__.py` | Barrel exports: `compute_recovery_ratio`, `build_eval_artifact`, `write_artifact`, `grade`, `run_ppl`, `resolve_model_path`, `resolve_corpus` |
| `eval/constants.py` | Tunable defaults: `RECOVERY_GATE_THRESHOLD=0.95`, `DEFAULT_SEQ_LEN=2048`, `DEFAULT_STRIDE=512`, `VOCAB_LOG_SOFTMAX_BLOCK_SIZE=4096` |
| `eval/model_path.py` | `resolve_model_path(cli_model)` with CLI -> env -> fallback resolution order and traversal-guard regex (mirrors `cli.py`) |
| `eval/corpus.py` | `resolve_corpus(corpus_path)` loads JSON token ids or raw text (via byte-level tokenizer); defaults to committed sample |
| `eval/artifact.py` | `compute_recovery_ratio`, `build_eval_artifact` (D-09 schema), `write_artifact` (pretty-printed JSON) |
| `eval/grader.py` | `run_ppl` wraps `InferencePipeline` as context manager; `grade` runs base -> close -> compressed -> close sequence |
| `eval/run_eval.py` | CLI with `--model`, `--compressed`, `--corpus`, `--tokenizer`, `--seq-len`, `--stride`, `--threshold`, `--output` |
| `eval/data/wikitext2_sample.*` | Committed ~2 KB raw text sample and pre-tokenized byte-level token ids for offline reproducibility |
| `scripts/fetch_eval_corpus.py` | Downloads full WikiText-2 test set via `urllib` (stdlib, no `datasets` or `requests` dependency) |

### Pipeline Patch

`spectralstream/inference/pipeline.py`:
- Added module-level `_blocked_log_sum_exp(logits, block_size)` that computes `log(sum(exp(logits)))` in vocab chunks, accumulating in float64. Peak memory: `block_size * seq_len * 8` (~64 MB at 4096x2048) vs `vocab_size * seq_len * 8` (~4 GB for 262k vocab).
- Upstreamed `VOCAB_LOG_SOFTMAX_BLOCK_SIZE = 4096` constant.
- `measure_perplexity` now uses `_blocked_log_sum_exp` instead of `np.exp(log_probs).sum(axis=-1)`.

## TDD Gate Compliance

| Phase | Commit | Gate |
|-------|--------|------|
| RED | `75a0816` | `test(02-01): add failing tests for eval grader subsystem` |
| GREEN | `02d0895` | `feat(02-01): implement eval grader subsystem with RED/GREEN/REFACTOR` |
| REFACTOR | (no delta) | Path validation already consolidated in GREEN; logging style already %-style |

## Verification

- Automated tests: `python -m pytest tests/test_eval_grader.py -x -q` passes with 17/17 tests.
- CLI help: `python -m eval.run_eval --help` lists all 8 flags as specified.
- Static check: `grep -rn "/home/mike" eval/` returns nothing -- no hardcoded author paths in eval modules.
- Pre-existing `/home/mike` paths in `scripts/dial_in_spectral.py`, `scripts/global_migration_validate.py`, and `benchmark_physics_real_weights.py` were NOT modified (out of scope).

## Deviations from Plan

### Plan Text Alignment

- **CLI invocation:** The plan says `python -m spectralstream.eval.run_eval` but all file paths (`eval/__init__.py`, `eval/run_eval.py`) place `eval/` at the project root, not under `spectralstream/eval/`. Implemented as `python -m eval.run_eval` per the file path convention. This aligns with how `tests/` and `scripts/` are accessed at the project root.
- **No REFACTOR delta:** Path validation consolidation and %-style logging were already satisfied by the GREEN implementation. No additional code changes needed.

### Auto-fixed Issues

**1. [Test Bug] Mock context manager return value** -- Tests using `unittest.mock.patch` for `InferencePipeline` need `mock_pipe.__enter__.return_value = mock_pipe` for the context manager protocol to return the correct mock instance. Fixed in all 4 mock sites.

**2. [Test Bug] Windows path separator** -- `test_write_artifact_default_dir` asserted `result.startswith("eval/artifacts/")` but `os.path.join` uses `\` on Windows, producing `eval\artifacts\...`. Fixed by normalizing path with `os.path.normpath` for the assertion.

**3. [Bug] `datetime.utcnow()` deprecated** -- Replaced with `datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")` per Python 3.13 deprecation.

## Threat Model Compliance

| Threat ID | Category | Mitigation | Status |
|-----------|----------|------------|--------|
| T-02-01-01 | Tampering | Path traversal guard in `eval/model_path.py` mirrors `cli.py` regex | Mitigated |
| T-02-01-02 | DoS | `_blocked_log_sum_exp` caps per-window logits memory; sequential model loading | Mitigated |
| T-02-01-03 | Tampering | `recovery_ratio` and `gate_passed` derived solely from measured PPLs | Mitigated |
| T-02-01-04 | Tampering | `layers_loaded` recorded via `len(pipe.tensor_names)`; sequential loading | Mitigated |
| T-02-01-05 | Information Disclosure | `fetch_eval_corpus.py` downloads public WikiText-2 via stdlib only | Accepted |

## Test Coverage

17 tests covering:
- Recovery ratio direction and threshold (4 tests)
- Artifact D-09 schema completeness and type correctness (2 tests)
- Artifact JSON writing (2 tests)
- `run_ppl` lifecycle: enter + measure + close (1 test)
- `grade` identical windowing for both models (1 test)
- `layers_loaded` non-zero tracking (1 test)
- `resolve_model_path` traversal rejection (2 tests)
- `resolve_corpus` JSON loading and defaults (2 tests)
- Package importability (1 test)
- Empty/inf token list edge case (1 test)

## Commits

| Hash | Type | Message |
|------|------|---------|
| `75a0816` | `test` | `test(02-01): add failing tests for eval grader subsystem` |
| `02d0895` | `feat` | `feat(02-01): implement eval grader subsystem with RED/GREEN/REFACTOR` |

## Self-Check: PASSED

- [x] 17/17 tests pass (`python -m pytest tests/test_eval_grader.py -x -q`)
- [x] CLI help lists all flags (`python -m eval.run_eval --help`)
- [x] No `/home/mike` paths in `eval/`
- [x] All created files exist
- [x] Both commits exist in git log
- [x] SUMMARY.md written to correct path

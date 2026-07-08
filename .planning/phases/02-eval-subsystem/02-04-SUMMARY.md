---
phase: 02-eval-subsystem
plan: 04
name: "Wire Model-Native Tokenizer into Eval (D-02 Gap Closure)"
type: tdd
wave: 2
subsystem: eval
tags:
  - eval
  - tokenizer
  - d-02-gap-closure
  - tdd
  - gap-closure
dependency:
  requires:
    - "02-01 (eval grader subsystem)"
    - "02-02 (tokenizer fallback)"
  provides:
    - D-02 model-native tokenizer actually wired into tokenization path
    - D-13 transparency documentation in eval/data/README.md
  affects:
    - eval/corpus.py
    - eval/run_eval.py
    - eval/__init__.py
    - eval/constants.py
    - eval/data/README.md
    - tests/test_eval_grader.py
    - .planning/phases/02-eval-subsystem/02-01-PLAN.md
tech-stack:
  added:
    - eval/data/README.md (D-13 transparency doc)
  patterns:
    - TYPE_CHECKING import for type hints (avoid runtime dependency cycle)
    - Lazy import of build_default_tokenizer inside function body
    - Graceful degradation via try/except with logged warning on loader failure
    - unittest.mock.patch at source module for cross-module mock isolation
key-files:
  created:
    - eval/data/README.md
  modified:
    - eval/corpus.py (new tokenizer parameter + conditional default fallback)
    - eval/run_eval.py (AutoTokenizer wiring + graceful degradation)
    - eval/__init__.py (docstring update)
    - eval/constants.py (DEFAULT_SAMPLE_TOKENS docstring update)
    - tests/test_eval_grader.py (6 new TDD tests)
    - .planning/phases/02-eval-subsystem/02-01-PLAN.md (must_have reworded)
decisions:
  - "AutoTokenizer.from_pretrained and from_gguf are called from eval.run_eval, not eval.corpus (CLI-layer responsibility, not data-layer)"
  - "JSON pre-tokenized paths ignore the injected tokenizer (committed byte-level sample is the offline default per D-12)"
  - "Loader failures fall back to build_default_tokenizer() with a logged warning, never crash (T-02-04-01)"
  - "Must_have in 02-01-PLAN.md reworded to accurately scope the D-02 behavior (model tokenizer wired WHEN SUPPLIED, byte-level fallback otherwise)"
metrics:
  duration: "~5 min"
  completed: "2026-07-08"
  tasks: 3 (Task 1 RED/GREEN, Task 2 docs, Task 3 meta)
  files_created: 1
  files_modified: 6
status: complete
---

# Phase 2 Plan 4: Wire Model-Native Tokenizer into Eval (D-02 Gap Closure) Summary

## One-Liner

Closed the single failing verification gap from Phase 2 (02-VERIFICATION.md Truth #7): `--tokenizer` is now actually wired to `AutoTokenizer.from_pretrained` / `from_gguf` so the model's own tokenizer is used for raw-text encoding when supplied, while the committed byte-level sample remains the offline fallback when `--tokenizer` is omitted (D-02, D-12, D-13).

## What Was Built

### Modified Modules

| File | Change | Purpose |
|------|--------|---------|
| `eval/corpus.py` | Added `tokenizer` parameter to `resolve_corpus(corpus_path, tokenizer=None)` | Injected tokenizer honored for raw text; byte-level fallback preserved; JSON paths unchanged |
| `eval/run_eval.py` | Replaced metadata-only `tokenizer_name` with actual 3-branch tokenizer loading | `--tokenizer.json` calls `AutoTokenizer.from_pretrained`; `.gguf` calls `AutoTokenizer.from_gguf`; omitted uses `build_default_tokenizer()`; graceful fallback on loader failure |
| `eval/__init__.py` | Docstring update | Documents the new `tokenizer` parameter on `resolve_corpus` |
| `eval/constants.py` | `DEFAULT_SAMPLE_TOKENS` docstring | Explicitly states vocab_size=256 byte-level and references README |
| `eval/data/README.md` | **NEW** (~50 lines) | Documents the committed byte-level sample as offline fallback (D-12/D-13), model-native reproduction steps (D-02), and locked decisions |

### New Tests (6)

All in `tests/test_eval_grader.py`:

| Test | Verifies |
|------|----------|
| `test_resolve_corpus_uses_injected_tokenizer_for_raw_text` | Injected custom tokenizer's result is returned, not overwritten by default |
| `test_resolve_corpus_json_path_ignores_tokenizer` | JSON pre-tokenized paths ignore the tokenizer argument |
| `test_resolve_corpus_raw_text_default_uses_byte_level` | `tokenizer=None` preserves the byte-level `build_default_tokenizer()` fallback |
| `test_run_eval_loads_tokenizer_from_pretrained` | `--tokenizer tokenizer.json` triggers `AutoTokenizer.from_pretrained` call |
| `test_run_eval_loads_tokenizer_from_gguf` | `--tokenizer model.gguf` triggers `AutoTokenizer.from_gguf` call |
| `test_run_eval_omitted_tokenizer_uses_default` | No `--tokenizer` means no AutoTokenizer loader calls; `resolve_corpus` still receives a tokenizer |

## TDD Gate Compliance

| Phase | Commit | Gate |
|-------|--------|------|
| RED | `46a12e3` | `test(02-04): add failing tests for model-native tokenizer wiring` |
| GREEN | `7f7f18c` | `feat(02-04): wire model-native tokenizer into resolve_corpus and run_eval` |
| REFACTOR | (no delta) | No additional clean-up needed; code is already clean |

## Verification

- Automated tests: `python -m pytest tests/test_eval_grader.py -x -q` passes with 23/23 (17 pre-existing + 6 new).
- CLI help: `python -m eval.run_eval --help` lists all flags, unchanged.
- Static check: `resolve_corpus` signature is `(corpus_path: str | None = None, tokenizer: BaseTokenizer | None = None) -> list[int]`.
- Static check: `AutoTokenizer.from_pretrained` and `AutoTokenizer.from_gguf` both have call sites in `eval/run_eval.py`.
- Canary test: `test_resolve_corpus_uses_injected_tokenizer_for_raw_text` passes (would fail before the fix because the old code unconditionally overwrote the tokenizer with `build_default_tokenizer()`).
- Static check: `eval/data/README.md` exists and references D-02/D-12/D-13.

## Deviations from Plan

### Auto-fixed Issues

None. Plan executed exactly as written.

### Plan Text Alignment

- **Mock target for run_eval tests:** The plan suggests patching at `eval.run_eval.AutoTokenizer.from_pretrained`, but patching at `spectralstream.utils.tokenizer_engine.AutoTokenizer.from_pretrained` provides cross-phase RED/GREEN compatibility (the source-module attribute exists before the GREEN import is added). Both approaches produce the same mock isolation because the import creates a reference to the same class object.

### Out-of-Scope Discoveries

None.

## Threat Model Compliance

| Threat ID | Category | Mitigation | Status |
|-----------|----------|------------|--------|
| T-02-04-01 | DoS | Each loader call (`from_pretrained` / `from_gguf`) wrapped in try/except with logged warning + `build_default_tokenizer()` fallback | Mitigated |
| T-02-04-02 | Tampering | JSON path ignores injected `tokenizer` argument (documented in docstring) | Accepted |
| T-02-04-03 | Information Disclosure | `tokenizer_name` records only `os.path.basename` (e.g. `auto_from_pretrained:tokenizer.json`) | Accepted |
| T-02-04-04 | Repudiation | 6 new TDD tests mock loaders via `unittest.mock.patch` at source module; assert call args; commits auditable | Mitigated |

### Closed Risks

- **T-02-01-06 (old):** The `--tokenizer` metadata-only gap is now closed. `AutoTokenizer.from_pretrained` / `from_gguf` are actually called and the loaded tokenizer is passed to `resolve_corpus`. T-02-04-04 covers the regression test.

## Test Coverage

23 tests total (17 pre-existing + 6 new):

**Pre-existing (17):**
- Recovery ratio direction and threshold (4)
- Artifact D-09 schema completeness and type correctness (2)
- Artifact JSON writing (2)
- `run_ppl` lifecycle: enter + measure + close (1)
- `grade` identical windowing for both models (1)
- `layers_loaded` non-zero tracking (1)
- `resolve_model_path` traversal rejection (2)
- `resolve_corpus` JSON loading and defaults (2)
- Package importability (1)
- Empty/inf token list edge case (1)

**New (6):**
- `resolve_corpus` injected tokenizer honored (1)
- `resolve_corpus` JSON path ignores tokenizer (1)
- `resolve_corpus` tokenizer=None uses byte-level (1)
- `run_eval` loads from `from_pretrained` (1)
- `run_eval` loads from `from_gguf` (1)
- `run_eval` omitted tokenizer uses default (1)

## Verification

All automated checks pass.

## Commits

| Hash | Type | Message |
|------|------|---------|
| `46a12e3` | `test` | `test(02-04): add failing tests for model-native tokenizer wiring` |
| `7f7f18c` | `feat` | `feat(02-04): wire model-native tokenizer into resolve_corpus and run_eval` |
| `6697d72` | `docs` | `docs(02-04): document committed byte-level sample as offline fallback` |

## Self-Check: PASSED

- [x] 23/23 tests pass (`python -m pytest tests/test_eval_grader.py -x -q`)
- [x] CLI help works (`python -m eval.run_eval --help`)
- [x] `resolve_corpus` signature includes `tokenizer` parameter
- [x] `AutoTokenizer.from_pretrained` and `from_gguf` call sites exist in `eval/run_eval.py`
- [x] New canary test passes and would fail BEFORE the fix
- [x] `eval/data/README.md` exists and references D-02/D-12/D-13
- [x] `must_have` in 02-01-PLAN.md reworded to accurate scope
- [x] All commits exist in git log
- [x] SUMMARY.md written to correct path

---
phase: 2
slug: eval-subsystem
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-08
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (already in `pyproject.toml` dev extra) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` (timeout=120, testpaths=tests) |
| **Quick run command** | `python -m pytest tests/test_eval_subsystem.py tests/test_default_tokenizer.py -q` |
| **Full suite command** | `python -m pytest tests/ -q` |
| **Estimated runtime** | ~60–300s (PPL on a small CPU test model is the dominant cost; tokenizer round-trip + gate unit tests are fast) |

---

## Sampling Rate

- **After every task commit:** Run the quick command above.
- **After every plan wave:** Run the full suite.
- **Before `/gsd-verify-work`:** Full suite must be green.
- **Max feedback latency:** 300 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01 | eval-grader | 1 | EVAL-01 | T-02-01 / — | Loads only local model paths; no network; no eval of arbitrary remote code | integration | `python -m pytest tests/test_eval_subsystem.py -q` | ❌ W0 | ⬜ pending |
| 02-02 | default-tokenizer | 1 | EVAL-02 | T-02-02 / — | N/A (pure local text transform) | unit | `python -m pytest tests/test_default_tokenizer.py -q` | ❌ W0 | ⬜ pending |
| 02-03 | model-path-config | 1 | EVAL-03 | T-02-03 / — | Env/CLI path only; no hardcoded absolute author path | unit | `python -m pytest tests/test_model_path_config.py -q` | ❌ W0 | ⬜ pending |
| 02-04 | recovery-gate | 2 | EVAL-01 | T-02-04 / — | Gate predicate assertable; rejects >5% degradation, accepts lossless | unit | `python -m pytest tests/test_eval_subsystem.py -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_eval_subsystem.py` — stubs asserting: JSON artifact shape, both PPL values measured, `recovery_ratio = base/compressed`, gate `>= 0.95` passes lossless / fails 10× degradation, `layers_loaded` guard.
- [ ] `tests/test_default_tokenizer.py` — `BaseTokenizer.encode/decode` no longer raises; `encode→decode` round-trip on sample text.
- [ ] `tests/test_model_path_config.py` — env var + `--model` resolve; README documents required model.

*Existing infrastructure (pytest) covers the framework; only test-file stubs are new.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Reproduce headline PPL on the real Gemma-4 E2B model | EVAL-01 | Requires the multi-GB model weights the user supplies locally; not in CI | `python -m spectralstream.eval.run_eval --model <path> --corpus eval/data/wikitext2_sample.json` and inspect `eval/artifacts/*.json` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 300s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

---
phase: 01
slug: metrics-trust-loop
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-08
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Derived from
> `01-RESEARCH.md` §Validation Architecture. TDD mode is enabled, so each behavior
> has a failing/automated test as its first task.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x (project standard `dev` extra) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` + `tests/conftest.py` |
| **Quick run command** | `python -m pytest tests/test_honest_metrics.py -v` |
| **Full suite command** | `python -m pytest tests/test_honest_metrics.py tests/test_certificate.py tests/test_certificate_comprehensive.py -v` |
| **Estimated runtime** | ~10 seconds (new suite is <1s/case; cert suite already exists) |

> **pytest-timeout caveat:** env has pytest 9.0.2 with `pytest-timeout>=2.0`. If the
> timeout plugin errors on import, run the new file with `-p no:timeout` (does not
> block the phase — new tests are sub-second).

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_honest_metrics.py -v`
- **After every plan wave:** Run the full suite command above
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | METRICS-01 | T-01-01 | Gate marks `gated:True`, never emits ratio for over-threshold error | unit (TDD RED) | `pytest tests/test_honest_metrics.py -v -k gate` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | METRICS-01 | — | Strict `>` boundary at exactly 0.05 not gated | unit | `pytest tests/test_honest_metrics.py -v -k boundary` | ❌ W0 | ⬜ pending |
| 01-02-01 | 01 | 1 | METRICS-02 | — | All ratios route through `serialized_nbytes`/`dual_ratio`; BF16 headline | unit | `pytest tests/test_honest_metrics.py -v -k ratio` | ❌ W0 | ⬜ pending |
| 01-02-02 | 01 | 1 | METRICS-02 | — | CLI summary leads BF16, FP32 secondary | unit + CLI smoke | `pytest tests/test_honest_metrics.py -v -k headline` | ❌ W0 | ⬜ pending |
| 01-03-01 | 01 | 1 | METRICS-03 | T-01-02 | No competitor literals in `certificate.py`; disclaimer present | unit + static | `pytest tests/test_honest_metrics.py -v -k literature`; `grep -rn "GPTQ\|AWQ\|SqueezeLLM\|GGML Q" spectralstream/compression/certificate.py` | ❌ W0 | ⬜ pending |
| 01-03-02 | 01 | 1 | METRICS-03 | — | `industry_comparison` contract preserved; orphaned JSON deleted | unit (existing) | `pytest tests/test_certificate.py tests/test_certificate_comprehensive.py -v` | ✅ keep | ⬜ pending |
| 01-04-01 | 01 | 1 | METRICS-04 | — | `serialized_nbytes` handles all payload shapes | unit | `pytest tests/test_honest_metrics.py -v -k serialized` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_honest_metrics.py` — created this phase (Wave 0 stubs for METRICS-01..04)
- [x] `tests/conftest.py` — shared fixtures already exist
- [x] pytest — already installed (project `dev` extra)

*Existing infrastructure covers all phase requirements; no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CLI smoke: run a deliberately-bad (noisy) method and confirm "GATED" renders and no numeric ratio line appears for that tensor | METRICS-01 | Requires a live model/CLI invocation with a synthetic noise method impossible to express purely as a unit test | Add a fixture/script invoking `apply_gate()` on a `rel_mse>0.05` `ErrorMetrics` and assert `gated:True` + `ratio_vs_bf16 is None`; the unit test covers the same path without a live model. |

*All phase behaviors have automated verification; the manual row is covered equivalently by a unit test.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved

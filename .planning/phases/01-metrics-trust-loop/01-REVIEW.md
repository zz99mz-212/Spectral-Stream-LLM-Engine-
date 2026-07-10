---
phase: 01-metrics-trust-loop
reviewed: 2026-07-08T00:00:00Z
depth: deep
files_reviewed: 5
files_reviewed_list:
  - spectralstream/compression/honest_metrics.py
  - spectralstream/compression/cli.py
  - spectralstream/compression/certificate.py
  - spectralstream/compression/literature_estimates.py
  - tests/test_honest_metrics.py
findings:
  critical: 1
  warning: 2
  info: 1
  total: 4
status: issues_found
---

# Phase 1: Code Review Report

**Reviewed:** 2026-07-08T00:00:00Z
**Depth:** deep
**Files Reviewed:** 5
**Status:** issues_found

## Summary

The Phase 1 metrics-trust-loop changes are largely sound at the level of the new
`apply_gate` chokepoint and the METRICS-03 literature-estimate refactoring.
`apply_gate` correctly implements the strict `>` boundary at `rel_mse == 0.05`,
returns `None` (not `0.0`) for suppressed ratios, and retains the error metrics —
and the CLI honest-metrics summary correctly filters `None` ratios before averaging,
so the gate boundary math is honored end-to-end. The TDD anchors in
`tests/test_honest_metrics.py` are thorough and all pass.

However, the review surfaced one severe, pre-existing crash that lives in a reviewed
file (cli.py) and one contract inconsistency in the certificate builder that undercuts
the stated "THE ONLY ratio-emission decision point" mandate. The most important
deliverable-specific issue: the anti-fabrication disclaimer that is the entire point of
METRICS-03 is rendered in `to_text`/`to_markdown` but is **absent from the HTML
certificate** (the "professional" artifact users share).

Note: the gate boundary logic, byte-exactness of `dual_ratio`/`serialized_nbytes`, and
the None-filtering in the CLI summary were all verified correct — no defects found there.

## Critical Issues

### CR-01: `pattern_counts`/`pattern_ratios` referenced before assignment in World-Model branch — `UnboundLocalError` on the default `compress` path

**File:** `spectralstream/compression/cli.py:876-877` (definition at `:1041-1042`)
**Issue:** In `cmd_compress`, the World-Model Auto branch (`if args.auto:` — which is
the **default**, since `--auto` defaults to `True`) increments `pattern_counts[wm_pattern]`
and `pattern_ratios` at lines 876-877 inside `for name, result in results_dict.items():`.
Those two names are first *assigned* at lines 1041-1042, which sit **after** the branch's
`return` at line 955. Because Python binds a name as function-local the moment it is
assigned anywhere in the function, `pattern_counts` is treated as local for the whole of
`cmd_compress`; on the first real tensor the reference at line 876 raises
`UnboundLocalError` (a `NameError` subclass) and aborts the entire compression run.

Verified by static analysis: the WM branch starts at line 801, returns at line 955, and
the first `pattern_counts` binding is at line 1041; the `pattern_counts[wm_pattern]`
reference at line 876 is both before that binding and before the branch return.

This means `python -m spectralstream.compression.cli compress model.safetensors out.ssf`
(the default invocation) crashes at the first tensor for any model with ≥1 tensor.

**Severity note:** This is **pre-existing** — the diff base (`11524a8^`) already had the
same reference/definition ordering (base lines 871 / 1036). Phase 1 did not introduce it,
but it is a real BLOCKER in a reviewed file and must be fixed before the default path works.
It is reported here for transparency rather than as a Phase 1 regression.

**Fix:** Move the declaration of `pattern_counts` / `pattern_ratios` /
`tensor_type_methods` (and any other names the WM branch touches) to the top of
`cmd_compress`, before the `if args.auto:` branch, e.g.:

```python
def cmd_compress(args: argparse.Namespace) -> None:
    pattern_counts: Dict[str, int] = {}
    pattern_ratios: Dict[str, List[float]] = {}
    tensor_type_methods: Dict[str, Dict[str, int]] = {}
    # ... existing body; remove the re-declarations near line 1041
```

## Warnings

### WR-01: Certificate builder bypasses `apply_gate` — per-tensor ratios emitted ungated, contradicting the chokepoint contract

**File:** `spectralstream/compression/certificate.py:19-300` (builder), vs `honest_metrics.py:168-219`
**Issue:** `apply_gate`'s docstring states it is "THE ONLY ratio-emission decision point.
Both CLI blocks and future reporters must call it (not per-call-site thresholds)." Yet
`CertificateBuilder.from_compressed_tensors` / `from_compression_report` populate each
`TensorCertificate.compression_ratio` from `ct.compression_ratio` (the engine's claimed
ratio) and never consult `ct.params["honest_metrics"]` (where `apply_gate` would have set
`ratio_vs_bf16: None` and `gated: True` for high-error tensors). A tensor whose
`rel_mse > 0.05` therefore still appears in the certificate with its full compression ratio
and no suppression — the exact behavior the gate was created to prevent. The honest-metrics
result is computed by the CLI and attached to `ct.params["honest_metrics"]`, but the
certificate builder ignores it.

This does not "fabricate" a number (ratio + error + grade are all shown), but it means the
central gate is not actually the single chokepoint the module documents, and the certificate
quietly diverges from the CLI summary's gating.

**Fix:** Have the builder read `ct.params.get("honest_metrics", {})`; when `gated` is `True`,
suppress/surface the ratio as gated (e.g. emit a `gated` flag and omit the numeric ratio from
the headline, or mark the row), consistent with `apply_gate`'s contract.

### WR-02: Anti-fabrication disclaimer missing from the HTML certificate

**File:** `spectralstream/compression/certificate.py:507-719` (`to_html`) — disclaimer only added in `to_markdown` (`:777-784`) and `to_text` (`:836-837`)
**Issue:** METRICS-03's central guarantee is that competitor ratios are flagged "literature
estimates, not measured here." The refactor correctly renders `LITERATURE_DISCLAIMER` in
`to_text()` and `to_markdown()`, and `test_disclaimer_rendered_in_certificate` asserts it
appears in both. But `to_html()` — the "professional" certificate users are most likely to
share or present — contains no disclaimer at all. The disclaimer key is present in the
`industry_comparison` dict but is never surfaced in the HTML template, so the
anti-fabrication control is defeated exactly where it matters most.

**Fix:** In `to_html()`, after the Industry Comparison table (near the `</div>` closing that
section), render:

```python
disclaimer = self.industry_comparison.get("disclaimer", "")
if disclaimer:
    html += f'<p style="margin-top:10px;color:#888;">⚠️ {disclaimer}</p>\n'
```

## Info

### IN-01: Unused `dual_ratio` import in cli.py after gating refactor

**File:** `spectralstream/compression/cli.py:30-34` (import), no call sites remain
**Issue:** The Phase 1 diff replaced both CLI `dual_ratio(...)` call sites with
`apply_gate(...)`. The `dual_ratio` import is now dead (grep for `dual_ratio(` in cli.py
returns zero call sites). This is harmless but is a dangling import that should be dropped
to keep the "single chokepoint" intent clear and avoid confusion.

**Fix:**
```python
from spectralstream.compression.honest_metrics import (
    end_to_end_error,
    apply_gate,
)
```

---

_Reviewed: 2026-07-08T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_

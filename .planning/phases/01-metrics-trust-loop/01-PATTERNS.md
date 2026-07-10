# Phase 01: metrics-trust-loop - Pattern Map

**Mapped:** 2026-07-08
**Files analyzed:** 6 (5 new/modified + 1 delete)
**Analogs found:** 6 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `spectralstream/compression/honest_metrics.py` | utility/lib (MODIFY) | transform (pure fn) | `honest_metrics.py` itself (existing `dual_ratio`/`end_to_end_error`) | exact |
| `spectralstream/compression/literature_estimates.py` | data module (NEW) | static catalog | `honest_metrics.py` conventions (UPPER_SNAKE_CASE, `from __future__ import annotations`) | role-match |
| `spectralstream/compression/certificate.py` | builder (MODIFY) | transform | `certificate.py` itself (`_compute_industry_comparison`, `to_text`/`to_markdown`) | exact |
| `spectralstream/compression/cli.py` | CLI (MODIFY) | request-response | `cli.py` itself (both honest-metrics blocks, summary print) | exact |
| `tests/test_honest_metrics.py` | test (NEW) | batch | `tests/test_certificate_comprehensive.py` (section banners, fixtures, contracts) | role-match |
| `benchmark_industry_comparison.json` | data artifact (DELETE) | n/a | none (orphaned; only referenced by planning docs, not `.py`/`.json`) | n/a |

## Pattern Assignments

### `spectralstream/compression/honest_metrics.py` (utility/lib, transform — MODIFY)

**Analog:** itself. Add `ERROR_GATE_THRESHOLD` constant + `apply_gate()` / `gated_ratio()` helper. Do NOT change existing `serialized_nbytes`, `dual_ratio`, `end_to_end_error`, `ErrorMetrics`, `ratio_vs_fp32/bf16` signatures — they are called by `cli.py` and `engine/_helpers.py`.

**Module header + imports pattern** (lines 1-22):
```python
"""Honest measurement helpers for compression ratio and error reporting.
[... mandatory module docstring describing byte-exact mandate ...]
"""

from __future__ import annotations

import struct
from typing import Any, Dict, NamedTuple, Tuple

import numpy as np

BF16_BYTES_PER_ELEMENT = 2
FP32_BYTES_PER_ELEMENT = 4
```
> New constant must follow `UPPER_SNAKE_CASE` and sit next to `BF16_BYTES_PER_ELEMENT` / `FP32_BYTES_PER_ELEMENT`.

**Existing byte-exact primitives to reuse (do NOT reimplement)** (lines 25-163):
```python
def serialized_nbytes(payload: Any) -> int: ...           # recursive: bytes/ndarray/dict/list/tuple/scalars/None
def dual_ratio(original_elements: int, payload: Any) -> Dict[str, float]:
    return {"ratio_vs_fp32": ..., "ratio_vs_bf16": ...}  # line 133
class ErrorMetrics(NamedTuple):                           # line 76
    rel_mse: float; cosine_sim: float; max_abs: float; snr_db: float
def end_to_end_error(original: np.ndarray, reconstructed: np.ndarray) -> ErrorMetrics: ...  # line 83
```

**New gate helper (proposed, grounded in `dual_ratio` + `ErrorMetrics`)** — place after `dual_ratio` (line 147):
```python
ERROR_GATE_THRESHOLD = 0.05  # rel_mse; configurable via SpectralStreamConfig / env (SS_* precedence)

def apply_gate(
    payload: Any,
    original_elements: int,
    rel_mse: float,
    threshold: float = ERROR_GATE_THRESHOLD,
) -> Dict[str, Any]:
    """Central chokepoint: decide whether a ratio may be emitted for a tensor.

    Uses STRICT `>` so rel_mse == threshold is NOT gated (Pitfall 1). When the
    error budget is exceeded the ratio is suppressed and replaced with an
    explicit marker (`gated: True`, `gate_reason`) so callers render "GATED"
    rather than a missing/zero number (Anti-Pattern: silent drop).
    """
    ratios = dual_ratio(original_elements, payload)
    gated = bool(rel_mse > threshold)
    return {
        "ratio_vs_bf16": (ratios["ratio_vs_bf16"] if not gated else None),
        "ratio_vs_fp32": (ratios["ratio_vs_fp32"] if not gated else None),
        "rel_mse": float(rel_mse),
        "gated": gated,
        "gate_reason": (f"rel_mse {rel_mse:.4f} > {threshold}" if gated else ""),
    }
```
> `apply_gate` is the single source of truth for ALL ratio emission (METRICS-01). Both CLI blocks and any future certificate builder must call it.

---

### `spectralstream/compression/literature_estimates.py` (data module, static catalog — NEW)

**Analog:** `honest_metrics.py` module conventions (header, `from __future__ import annotations`, `UPPER_SNAKE_CASE`, typed module-level constants). No existing "constant catalog" file exists, so mirror the byte-exact lib's style.

**Proposed file shape:**
```python
"""External competitor compression ratios.

⚠️ LITERATURE ESTIMATES, NOT MEASURED HERE. ⚠️
Every number below is a published/claimed figure from external methods. SpectralStream
does NOT measure these; they are shown only for context and MUST be labeled as
external literature estimates wherever surfaced (see LITERATURE_DISCLAIMER).
"""

from __future__ import annotations

from typing import List, Tuple

LITERATURE_DISCLAIMER = "literature estimates, not measured here"

# (name, ratio, description, type) — external methods; NOT measured by this engine.
LITERATURE_ESTIMATES: List[Tuple[str, float, str, str]] = [
    ("FP16 (baseline)", 2.0, "2x storage savings", "lossless"),
    ("INT8 quantization", 4.0, "Standard 8-bit quantization", "lossy"),
    ("INT4 quantization", 8.0, "Standard 4-bit quantization", "lossy"),
    ("NF4 (QLoRA)", 4.0, "Normal float 4, QLoRA standard", "lossy"),
    ("GPTQ 4-bit", 8.0, "Post-training quantization", "lossy"),
    ("AWQ 4-bit", 8.0, "Activation-aware quantization", "lossy"),
    ("GGML Q4_0", 4.5, "llama.cpp Q4_0 quantization", "lossy"),
    ("GGML Q8_0", 2.5, "llama.cpp Q8_0 quantization", "lossy"),
    ("SqueezeLLM", 8.0, "Non-uniform quantization", "lossy"),
]
```
> The 9 tuples are verbatim from `certificate.py` `_compute_industry_comparison` (lines 432-442). Do NOT include the `"SpectralStream (current)"` row — that is computed from `ratio` at runtime in `_compute_industry_comparison` and must stay there (Pitfall 3).

---

### `spectralstream/compression/certificate.py` (builder, transform — MODIFY)

**Analog:** itself. `_compute_industry_comparison` (lines 429-463) keeps its output dict contract; only the 9 competitor tuples move to `literature_estimates` import. `LITERATURE_DISCLAIMER` must surface in the `industry_comparison` dict AND in `to_text`/`to_markdown`/`to_html`.

**Current competitor block** (lines 429-463) — the lines to replace:
```python
def _compute_industry_comparison(self):
    """Compare compression power against known methods."""
    ratio = self.overall_ratio
    comparisons = [
        ("FP16 (baseline)", 2.0, ...),          # lines 432-442 — DELETE these 9 literals
        ...
        ("SqueezeLLM", 8.0, "Non-uniform quantization", "lossy"),
        ("SpectralStream (current)", round(ratio, 1), "This run", "hybrid"),  # KEEP — computed from ratio
    ]
    better_count = sum(1 for _, r, _, _ in comparisons if r < ratio and r != ratio)
    total_known = sum(1 for _, r, _, _ in comparisons if r != ratio)
    rank = sum(1 for _, r, _, _ in comparisons if r >= ratio)
    self.industry_comparison = {
        "comparisons": [
            {"name": n, "ratio": r, "description": d, "type": t,
             "beats": ratio > r if r != ratio else None}
            for n, r, d, t in comparisons
        ],
        "beats_standard_quant": ratio > 4.0,
        "beats_int4": ratio > 8.0,
        "rank": f"{rank}/{total_known}",
        "better_than_count": better_count,
        "total_compared": total_known,
    }
```

**Required change:** import at top of file (alongside other `from spectralstream...` imports) and replace the 9 hardcoded tuples:
```python
from spectralstream.compression.literature_estimates import (
    LITERATURE_ESTIMATES,
    LITERATURE_DISCLAIMER,
)
```
Inside `_compute_industry_comparison`, build `comparisons` from `LITERATURE_ESTIMATES` plus the current-run row, and add a `disclaimer` key:
```python
ratio = self.overall_ratio
comparisons = list(LITERATURE_ESTIMATES) + [
    ("SpectralStream (current)", round(ratio, 1), "This run", "hybrid"),
]
# ... better_count / total_known / rank math unchanged ...
self.industry_comparison = {
    "comparisons": [...],            # identical shape
    "beats_standard_quant": ratio > 4.0,
    "beats_int4": ratio > 8.0,
    "rank": f"{rank}/{total_known}",
    "better_than_count": better_count,
    "total_compared": total_known,
    "disclaimer": LITERATURE_DISCLAIMER,   # NEW — surfaced to machine-readable consumers + rendered text
}
```
> **Contract MUST be preserved** (test_certificate_comprehensive.py asserts): keys `comparisons`, `beats_standard_quant`, `beats_int4`, `rank`, `better_than_count`, `total_compared`; `len(comparisons) >= 9`; current method included; `beats_standard_quant`/`beats_int4` booleans unchanged. Adding `disclaimer` is additive and will not break these tests.

**Rendering surface** — disclaimer must appear in `to_text()` (line 784), `to_markdown()` (line 723), `to_html()` (line 509) near the industry-comparison section. Append a line such as:
```python
f"  ⚠️ {self.industry_comparison.get('disclaimer', '')}"
```
so the "literature estimates, not measured here" label reaches users (Pitfall 5).

---

### `spectralstream/compression/cli.py` (CLI, request-response — MODIFY)

**Analog:** itself. Two honest-metrics blocks + one summary print. Both blocks route through `apply_gate()`.

**CLI import line** (line 30) — extend:
```python
from spectralstream.compression.honest_metrics import dual_ratio, end_to_end_error
# → add apply_gate (and optionally ERROR_GATE_THRESHOLD if surfacing the constant)
from spectralstream.compression.honest_metrics import dual_ratio, end_to_end_error, apply_gate
```

**Block 1 — World Model path (lines 839-865).** Current populates `honest_metrics_dict` directly with `dual_ratio` + `end_to_end_error` results. Replace the ratio population with `apply_gate` so that when `err.rel_mse > ERROR_GATE_THRESHOLD` the ratio is suppressed+flagged:
```python
# BEFORE (line 846-848, 861-865)
ratios = dual_ratio(original.size, data)
honest_metrics_dict["ratio_vs_fp32"] = ratios["ratio_vs_fp32"]
honest_metrics_dict["ratio_vs_bf16"] = ratios["ratio_vs_bf16"]
...
err = end_to_end_error(original, recon)
honest_metrics_dict["rel_mse"] = err.rel_mse
honest_metrics_dict["cosine_sim"] = err.cosine_sim
honest_metrics_dict["max_abs"] = err.max_abs
honest_metrics_dict["snr_db"] = err.snr_db

# AFTER — gate wraps ratio emission; error metrics still stored unconditionally
honest_metrics_dict.update(apply_gate(data, original.size, err.rel_mse))
honest_metrics_dict["cosine_sim"] = err.cosine_sim
honest_metrics_dict["max_abs"] = err.max_abs
honest_metrics_dict["snr_db"] = err.snr_db
```
> NOTE: `apply_gate` returns `rel_mse` too, so the explicit assignment is optional. Keep both error scalars so the gate result + raw error coexist.

**Block 2 — second compress path (lines 1270-1295).** Identical transformation; `original.size` → `tensor.size` (line 1273 uses `tensor.size`). Both blocks MUST be gated (Pitfall 2 — grep `dual_ratio(` returns 2 sites; both must route through `apply_gate`).

**Summary print (lines 1414-1417) — lead with BF16 (METRICS-02).** Current:
```python
if hm_fp32:
    logger.info("  Honest ratio (vs FP32): avg %.1fx", float(np.mean(hm_fp32)))
if hm_bf16:
    logger.info("  Honest ratio (vs BF16): avg %.1fx", float(np.mean(hm_bf16)))
```
Required reorder (also handle gated entries where `ratio_vs_bf16` is `None` — filter them out of the mean, or the mean breaks):
```python
hm_bf16_valid = [r for r in hm_bf16 if r is not None]
hm_fp32_valid = [r for r in hm_fp32 if r is not None]
if hm_bf16_valid:
    logger.info("  Honest ratio (vs BF16 / disk): avg %.1fx", float(np.mean(hm_bf16_valid)))
if hm_fp32_valid:
    logger.info("  (secondary, vs FP32): avg %.1fx", float(np.mean(hm_fp32_valid)))
```
> The gated `None` values from `apply_gate` land in `hm_bf16`/`hm_fp32` collections (lines 1390-1397 read `c.get("honest_metrics", {}).get("ratio_vs_bf16", 0)`). Add `.get("ratio_vs_bf16")` then filter `None` before `np.mean` so a gated tensor renders "GATED" not a fabricated 0.0×. CLI should also emit a "GATED" marker line when `gated` is True in any tensor's `honest_metrics`.

**`hm_bf16`/`hm_fp32` collection (lines 1388-1397)** — these pull `ratio_vs_bf16`/`ratio_vs_fp32` from `honest_metrics`. With the gate, gated tensors yield `None`; filtering at the summary stage (above) handles it. No change needed at collection except tolerating `None`.

---

### `tests/test_honest_metrics.py` (test, batch — NEW)

**Analog:** `tests/test_certificate_comprehensive.py` (header, fixtures, box-drawing section banners, contract assertions at lines 1263-1362). Mirror its structure: `from __future__ import annotations` first line, `sys.path.insert(0, os.path.join(...))`, imports from `spectralstream.compression.honest_metrics`, `np.random` data, section banners.

**Imports / header pattern** (from test_certificate_comprehensive.py lines 1-27):
```python
"""Tests for honest metrics: ratio↔error gate, byte-exact serialized_nbytes, literature labels."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.honest_metrics import (
    serialized_nbytes,
    dual_ratio,
    end_to_end_error,
    apply_gate,
    ERROR_GATE_THRESHOLD,
)
from spectralstream.compression.literature_estimates import (
    LITERATURE_ESTIMATES,
    LITERATURE_DISCLAIMER,
)
```

**Section banner pattern** (line 30):
```python
# ═══════════════════════════════════════════════════════════════════
# Ratio ↔ Error Coupling Gate
# ═══════════════════════════════════════════════════════════════════
```

**Priority anchor test (METRICS-01, gate coupling)** — good method emits ratio, bad (high rel_mse) method is gated:
```python
def test_good_method_emits_ratio():
    orig = np.random.RandomState(0).randn(8, 8).astype(np.float32)
    recon = orig + 1e-4 * np.random.RandomState(1).randn(8, 8).astype(np.float32)
    err = end_to_end_error(orig, recon)
    assert err.rel_mse <= ERROR_GATE_THRESHOLD
    res = apply_gate(orig.tobytes(), orig.size, err.rel_mse)
    assert res["gated"] is False
    assert res["ratio_vs_bf16"] is not None
    assert res["ratio_vs_fp32"] is not None

def test_bad_method_gated():
    orig = np.random.RandomState(0).randn(8, 8).astype(np.float32)
    recon = np.random.RandomState(9).randn(8, 8).astype(np.float32)  # uncorrelated noise
    err = end_to_end_error(orig, recon)
    assert err.rel_mse > ERROR_GATE_THRESHOLD
    res = apply_gate(orig.tobytes(), orig.size, err.rel_mse)
    assert res["gated"] is True
    assert res["ratio_vs_bf16"] is None
    assert res["ratio_vs_fp32"] is None
    assert "rel_mse" in res["gate_reason"]
```

**Boundary test (Pitfall 1) — strict `>`**:
```python
def test_boundary_exactly_threshold_not_gated():
    res = apply_gate(b"\x00" * 64, 32, ERROR_GATE_THRESHOLD)
    assert res["gated"] is False

def test_boundary_just_above_gated():
    res = apply_gate(b"\x00" * 64, 32, ERROR_GATE_THRESHOLD + 1e-6)
    assert res["gated"] is True
```

**serialized_nbytes payload shapes (METRICS-04)** — one case per branch in `serialized_nbytes` (lines 38-67):
```python
@pytest.mark.parametrize("payload,expected", [
    (None, 0),
    (b"abc", 3),
    (bytearray(b"xy"), 2),
    (np.zeros((4, 4), dtype=np.float32), 64),   # nbytes
    (np.float32(1.0), 4),                        # np.generic
    ({"a": b"xy", "b": 3}, len(b"a") + len(b"b") + 2 + 8),  # nested dict (key utf-8 + recursive)
    ([b"ab", (4, 4)], 2 + 8 + 8),               # list + tuple
    (True, 1),                                  # bool
    (7, 8),                                     # int
    (2.5, 8),                                   # float
    ("hi", 2),                                  # str utf-8
])
def test_serialized_nbytes_shapes(payload, expected):
    assert serialized_nbytes(payload) == expected
```

**Literature label / no fabricated constants (METRICS-03)**:
```python
def test_literature_disclaimer_present():
    assert "literature estimates" in LITERATURE_DISCLAIMER.lower()
    assert len(LITERATURE_ESTIMATES) == 9

def test_no_gptq_literal_in_certificate_source():
    # static guard: certificate.py must not hardcode competitor ratios
    cert_src = (Path(__file__).parent.parent /
                "spectralstream/compression/certificate.py").read_text()
    assert "GPTQ" not in cert_src
    assert "AWQ" not in cert_src
    assert "SqueezeLLM" not in cert_src
```
> For the static guard, use `pathlib.Path`. Confirm `certificate.py` after refactor imports these only via `LITERATURE_ESTIMATES`.

---

### `benchmark_industry_comparison.json` (DELETE)

**Analog:** none. Confirmed orphaned — `Grep` for `benchmark_industry_comparison` returns matches ONLY in `.planning/**` docs, zero in `spectralstream/**` or any `.json`. Safe to `git rm` / delete. No test or module references it.

---

## Shared Patterns

### Byte-exact ratio mandate
**Source:** `spectralstream/compression/honest_metrics.py` (lines 1-12, `serialized_nbytes` L25)
**Apply to:** All ratio emission in `cli.py` blocks and `apply_gate`. Never use `len(dict)` or per-stage products. Ratios MUST flow through `serialized_nbytes` via `dual_ratio` / `apply_gate`.
```python
def serialized_nbytes(payload: Any) -> int: ...   # the ONLY correct byte counter
def dual_ratio(original_elements: int, payload: Any) -> Dict[str, float]: ...
```

### Error metrics source
**Source:** `honest_metrics.end_to_end_error` → `ErrorMetrics(rel_mse, cosine_sim, max_abs, snr_db)` (L76-130)
**Apply to:** CLI blocks compute `err = end_to_end_error(original, recon)` then pass `err.rel_mse` into `apply_gate`. Keep storing all four scalars in `honest_metrics_dict` even when gated (gate only suppresses the ratio, not error reporting).

### `from __future__ import annotations` + module docstring
**Source:** every module (honest_metrics.py L1-14, certificate.py, all tests)
**Apply to:** `literature_estimates.py` and `test_honest_metrics.py` — mandatory first line + descriptive docstring.

### UPPER_SNAKE_CASE constants + `_`-prefixed private helpers
**Source:** `honest_metrics.py` (`BF16_BYTES_PER_ELEMENT`, `FP32_BYTES_PER_ELEMENT`), CLAUDE.md Conventions
**Apply to:** `ERROR_GATE_THRESHOLD`, `LITERATURE_ESTIMATES`, `LITERATURE_DISCLAIMER`. Any CLI helper (e.g. `_honest_metrics_for(name, data, meta, original)` to DRY the two blocks) uses a leading underscore.

### Test section banners + contract preservation
**Source:** `tests/test_certificate_comprehensive.py` (L30 box-drawing banner; L1263-1362 `industry_comparison` contract)
**Apply to:** `test_honest_metrics.py` mirrors banner style; existing `test_certificate*.py` MUST stay green — the `_compute_industry_comparison` refactor is additive (new `disclaimer` key, tuples imported).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|-------|
| `benchmark_industry_comparison.json` | artifact | n/a | Orphaned; intentional deletion, no analog needed. |

## Metadata

**Analog search scope:** `spectralstream/compression/` (honest_metrics.py, certificate.py, cli.py, engine/_helpers.py), `tests/` (test_certificate.py, test_certificate_comprehensive.py), repo root (benchmark_industry_comparison.json)
**Files scanned:** 8 (4 source + 2 test + 1 json + 1 new-module analog)
**Pattern extraction date:** 2026-07-08
**Pitfalls flagged for planner:** (1) strict `>` boundary; (2) gate BOTH cli blocks; (3) preserve industry_comparison dict contract (`>=9` comparisons, current method included, booleans); (4) pytest-timeout may warn on pytest 9 — run new test with `-p no:timeout` if needed; (5) disclaimer must surface in to_text/to_markdown + no competitor literals left in certificate.py source.

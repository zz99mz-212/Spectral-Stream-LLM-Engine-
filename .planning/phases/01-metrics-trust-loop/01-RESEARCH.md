# Phase 1: Metrics Trust Loop - Research

**Researched:** 2026-07-08
**Domain:** Compression-ratio honesty infrastructure (error-gate, byte-exact ratio routing, de-hardcoding competitor constants, new test suite)
**Confidence:** HIGH

## Summary

This is a **research-grade codebase with an existing, working core** — not greenfield. The byte-exact measurement primitives already exist and are correct: `spectralstream/compression/honest_metrics.py` provides `serialized_nbytes()` (recursive, handles bytes/ndarray/dict/list/tuple/scalars/None), `dual_ratio(original_elements, payload)` (returns both `ratio_vs_fp32` and `ratio_vs_bf16`), and `end_to_end_error()` returning `ErrorMetrics(rel_mse, cosine_sim, max_abs, snr_db)`. The Phase-1 work is therefore **additive wrapping**, not new math: add a central gate helper in `honest_metrics.py`, route CLI emission through it, demote FP32 to secondary, and move the 9 hardcoded competitor constants out of `certificate.py` into a labeled `literature_estimates.py` module.

The fabricated artifact `benchmark_industry_comparison.json` is confirmed **orphaned** — it is referenced by zero `.py` and zero `.json` files, so deletion is safe. The only location of hardcoded competitor constants in live code is `certificate.py:_compute_industry_comparison` (lines ~429-463); no other shipped `.py` carries bare GPTQ/AWQ/SqueezeLLM/GGML ratio literals (the matches in `tests/test_calibration_quantizer.py` and `_archive/` are legitimate tests or dead code, not reporting paths). Existing `tests/test_certificate*.py` assert the `industry_comparison` shape (`comparisons`, `beats_standard_quant`, `beats_int4`, `rank`, `better_than_count`, `total_compared`, `>=9` entries, current method included) and MUST keep passing — the refactor preserves this dict's contract.

The CLI contains **two** parallel honest-metrics blocks (World Model path ~L839-865 and a second at ~L1270-1295), both populating `honest_metrics_dict`, and a single summary print (~L1414-1417) that currently leads with **FP32** — the opposite of METRICS-02's requirement. The gate must wrap both blocks; the headline summary must lead with BF16.

**Primary recommendation:** Implement a single central `apply_gate()`/`gated_ratio()` helper in `honest_metrics.py` (threshold default `0.05` rel_mse, strict `>`), route both CLI honest-metrics blocks through it so the ratio is suppressed-and-flagged (`gated: True`, `gate_reason`) when error exceeds threshold, lead the summary with `ratio_vs_bf16`, extract the 9 competitor tuples into `spectralstream/compression/literature_estimates.py` tagged "literature estimates, not measured here", delete the orphaned JSON, and add `tests/test_honest_metrics.py`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Error Gate Behavior**: When `rel_mse` exceeds the configured threshold, the ratio is **suppressed and replaced with an explicit marker** (`gated: True`, `gate_reason="rel_mse 0.11 > 0.05"`), not silently dropped. The method still runs; only its ratio claim is withheld. The gate is a central chokepoint in `honest_metrics.py` (`gated_ratio(...)` / `apply_gate(...)`). Threshold is **0.05 rel_mse** by default, consistent with Phase-3 cascade acceptance, and configurable (constant + optional config/env override), not hardcoded at every call site.
- **Byte-Exact Ratio Reporting**: All surfaced ratios derive from `serialized_nbytes()` (via `dual_ratio`); `ratio_vs_bf16` is the **default headline**, `ratio_vs_fp32` is a secondary annotation only. CLI already computes `dual_ratio` per tensor — extend/confirm headline leads with BF16 and labels FP32 as secondary.
- **Competitor Comparison Constants**: Move the 9 hardcoded constants out of `certificate.py` into `spectralstream/compression/literature_estimates.py`, explicitly labeled **"literature estimates, not measured here."** `certificate.py` imports these; contains no bare competitor numbers. Delete `benchmark_industry_comparison.json`. Preserve existing `test_certificate*` behavior (`industry_comparison` still exposes `comparisons`, `beats_standard_quant`, `beats_int4`, rank, etc.).
- **Tests (`tests/test_honest_metrics.py`)**: Priority anchor = ratio↔error coupling gate (deliberately-bad high-rel_mse method emits no ratio; good one emits a ratio). Also cover `serialized_nbytes` across all payload shapes, rejection of over-threshold methods, and literature-estimate label presence / no fabricated constants leak.

### Claude's Discretion
- Exact helper names, config key/env-var name for the threshold, and module path for literature estimates are at Claude's discretion, following existing naming conventions (`honest_metrics.py`, `UPPER_SNAKE_CASE` constants, `_`-prefixed private helpers).

### Deferred Ideas (OUT OF SCOPE)
- CI metrics-honesty lint + perplexity gate (v2, CI-01).
- Perplexity-based quality proof (Phase 2, EVAL-01).
- Int4, registry depth, format transparency (later phases).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| METRICS-01 | Error-gate every reported compression ratio — `rel_mse > threshold` must not emit a ratio | Gate helper in `honest_metrics.py`; wrap both CLI honest-metrics blocks; CLI summary must render "GATED" not a fabricated number |
| METRICS-02 | Make `ratio_vs_bf16` the default headline; demote `ratio_vs_fp32` to secondary | `dual_ratio()` already returns both; CLI summary (~L1414-1417) currently prints FP32 FIRST — reorder to BF16-first |
| METRICS-03 | Remove hardcoded GPTQ/AWQ/GGUF constants; label externals "literature estimates, not measured here" | Only `certificate.py:_compute_industry_comparison` (~L429-463) holds them; JSON is orphaned; extract to `literature_estimates.py` |
| METRICS-04 | Add `tests/test_honest_metrics.py` asserting ratio↔error coupling, threshold rejection, all `serialized_nbytes` payload shapes | New file; no existing file to break; mirrors existing test conventions |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Error-gate ratio emission | `honest_metrics.py` (core lib) | CLI summary renderer | Gate is a pure function over (payload, ErrorMetrics, threshold); the chokepoint must live in the byte-exact lib so ALL emitters route through it, not scatter threshold checks across CLI/certificate. |
| Byte-exact ratio computation | `honest_metrics.py` (`serialized_nbytes`/`dual_ratio`) | CLI, certificate builders | `dual_ratio` is the single source of truth for ratios; CLI/certificate must call it, never `len(dict)` or `len(bytes)`. |
| Competitor-constant storage | `literature_estimates.py` (new module) | `certificate.py` (importer) | Moves fabrication risk out of the reporting path; `certificate.py` becomes a pure consumer. |
| Ratio rendering / labeling | CLI summary + certificate text/HTML/MD | `honest_metrics.py` (data shape) | Presentation layer decides headline order (BF16 first); data layer only supplies both values. |
| Test coverage | `tests/test_honest_metrics.py` (new) | existing `test_certificate*.py` | New anchor tests for the gate; existing cert tests guard the `industry_comparison` contract. |

## Standard Stack

No new external packages are introduced. The phase is pure-Python and reuses existing runtime dependencies.

### Core (already present)
| Library | Version (verified) | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `numpy` | 2.4.2 (env) / `>=1.24` (pyproject) | Tensor math, `end_to_end_error` | Foundational; already used pervasively. `[VERIFIED: runtime import]` |
| `scipy` | 1.17.1 (env) / `>=1.10` (pyproject) | (not directly needed this phase) | Existing constraint. |
| `pytest` | 9.0.2 (env) / `>=7.0` (pyproject `dev`) | Test runner for `tests/test_honest_metrics.py` | Project test runner. `[VERIFIED: pytest --version]` |

> Note: env has pytest 9.0.2. The project pins `pytest-timeout>=2.0` for the 120s per-test timeout. pytest-timeout 2.x may warn on pytest 9 — if it breaks, the new test file should be short enough to not need the timeout and can be marked `-p no:timeout` or exempted. See Pitfalls.

### Supporting
| Library | Purpose | When to Use |
|---------|---------|-------------|
| `pytest-timeout` | Per-test timeout enforcement | Existing global config; not a Phase-1 concern unless it conflicts. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Central `apply_gate()` in `honest_metrics.py` | Per-call-site `if rel_mse > 0.05` checks in CLI | Per-site checks drift, get hardcoded, and miss emitters (already the root cause of BUG-02). Central chokepoint is mandatory per CONTEXT. |

**Installation:** None — pure internal refactor + new test. No `pip install` required.

## Package Legitimacy Audit

**N/A — no external packages are installed or introduced in this phase.** The work reuses `numpy`, `scipy`, `pytest` already declared in `pyproject.toml` and present in the environment. No new registry packages, no `npm`/`pip` additions. The Package Legitimacy Gate does not apply.

## Architecture Patterns

### System Architecture Diagram

```
                         compression produces (data, meta)
                                        │
                                        ▼
              ┌─────────────────────────────────────────────┐
   original ─▶│  honest_metrics.apply_gate(                │
   tensor     │     payload=data,                           │
              │     error=end_to_end_error(orig, recon),   │
              │     threshold=ERROR_GATE_THRESHOLD          │
              │  )  → {ratio_vs_bf16, ratio_vs_fp32,      │
              │          rel_mse, gated, gate_reason}      │
              └─────────────────────────────────────────────┘
                                        │
                  ┌─────────────────────┴──────────────────────┐
           rel_mse ≤ 0.05                          rel_mse > 0.05
                  │                                            │
                  ▼                                            ▼
        emit {ratio_vs_bf16 (headline),              emit {gated: True,
               ratio_vs_fp32 (secondary)}           gate_reason: "rel_mse X > 0.05",
                  │                                 ratio_vs_bf16: None}
                  ▼                                            │
        CLI summary (BF16-led) ◀──────────────┐               │
        certificate.industry_comparison ◀──────┤               │
                                               │               │
                              literature_estimates.py ◀────────┘
                              (9 constants, labeled
                               "literature estimates,
                                not measured here")
```

### Recommended Project Structure (additions only)
```
spectralstream/compression/
├── honest_metrics.py          # + apply_gate() / gated_ratio() helper, ERROR_GATE_THRESHOLD const
├── literature_estimates.py    # NEW: LITERATURE_ESTIMATES list, LITERATURE_DISCLAIMER string
├── certificate.py            # _compute_industry_comparison imports from literature_estimates
└── cli.py                    # both honest-metrics blocks route through apply_gate(); summary BF16-led

tests/
└── test_honest_metrics.py    # NEW: gate coupling, threshold rejection, serialized_nbytes shapes
```

### Pattern 1: Central Gate Chokepoint
**What:** A single pure helper in `honest_metrics.py` decides whether a ratio may be emitted; every caller passes its `ErrorMetrics` + serialized payload to it.
**When to use:** Every path that surfaces a compression ratio to a user or report.
**Example (proposed signature, grounded in existing `dual_ratio`/`ErrorMetrics`):**
```python
# Source: spectralstream/compression/honest_metrics.py (existing dual_ratio, end_to_end_error)
ERROR_GATE_THRESHOLD = 0.05  # rel_mse; configurable via SpectralStreamConfig / env

def apply_gate(
    payload: Any,
    original_elements: int,
    rel_mse: float,
    threshold: float = ERROR_GATE_THRESHOLD,
) -> Dict[str, Any]:
    ratios = dual_ratio(original_elements, payload)
    gated = bool(rel_mse > threshold)
    return {
        "ratio_vs_bf16": (ratios["ratio_vs_bf16"] if not gated else None),
        "ratio_vs_fp32": (ratios["ratio_vs_fp32"] if not gated else None),
        "rel_mse": float(rel_mse),
        "gated": gated,
        "gate_reason": (
            f"rel_mse {rel_mse:.4f} > {threshold}" if gated else ""
        ),
    }
```

### Pattern 2: Literature-Estimate Module
**What:** Competitor constants live in a dedicated module whose very first line / docstring and every consumer string declares "literature estimates, not measured here."
**When to use:** Any reference to external methods' ratios/numbers not measured by this engine.
**Example (proposed):**
```python
# spectralstream/compression/literature_estimates.py
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

### Anti-Patterns to Avoid
- **`len(dict)` for ratio** — counts keys, not bytes; `serialized_nbytes` exists precisely to prevent this. Never reintroduce.
- **Hardcoding `0.05` at each call site** — threshold must be one `UPPER_SNAKE_CASE` constant (+ optional config/env), so it stays consistent with Phase-3 cascade acceptance.
- **Silently dropping the ratio on gate** — must emit `gated: True` + `gate_reason` so the CLI renders "GATED" rather than a missing/zero number that could be misread as "1.0× = no compression."
- **Leading with `ratio_vs_fp32`** — the summary print at cli.py ~L1414-1417 currently does this; reorder to BF16-first per METRICS-02.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Byte-exact payload byte counting | A custom `len()`/`.nbytes` per type | `honest_metrics.serialized_nbytes()` | Already handles all real payload shapes recursively; hand-rolling misses nested dict/list/None cases. |
| Ratio vs baselines | Ad-hoc `orig_bytes/len(data)` | `honest_metrics.dual_ratio()` | Single source of truth for both bf16+fp32 baselines. |
| Error metrics | Custom MSE snippet | `honest_metrics.end_to_end_error()` → `ErrorMetrics` | Already returns the 4-tuple the gate consumes. |
| Test runner | Custom harness | `pytest` (existing `tests/conftest.py`, `tests/run_all_tests.py`) | Project standard; new `test_honest_metrics.py` drops into `tests/`. |

**Key insight:** The measurement primitives are correct and present. The failure mode was never the math — it was *where the gate lived*. Centralizing it in `honest_metrics.py` closes BUG-02.

## Runtime State Inventory

> Trigger: This is a code/config-only refactor (no rename, rebrand, or migration of runtime state). N/A.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore stores phase constants | None. |
| Live service config | None | None. |
| OS-registered state | None | None. |
| Secrets/env vars | `ERROR_GATE_THRESHOLD` may later be overridable via env at Claude's discretion; no secret involved | Optional; if added, document the var name. |
| Build artifacts | None — no compiled artifacts; pure Python | None. |

**Nothing found requiring runtime-state migration.** Deleting `benchmark_industry_comparison.json` is a repo-file deletion, not runtime state.

## Common Pitfalls

### Pitfall 1: Threshold boundary ambiguity (`0.05` exactly)
**What goes wrong:** Gate defined with `>=` vs `>` inconsistent between helper and tests; a method at exactly `rel_mse=0.05` is sometimes gated, sometimes not.
**Why it happens:** CONTEXT says "exceeds the configured threshold" → strictly greater, but code authors default to `>=`.
**How to avoid:** Use strict `rel_mse > threshold` everywhere; add an explicit test at exactly `0.05` asserting NOT gated.
**Warning signs:** Tests assert both `0.0501`→gated and `0.0500`→not-gated.

### Pitfall 2: Only wiring ONE of the two CLI honest-metrics blocks
**What goes wrong:** The World-Model path (~L839) gets the gate but the second block (~L1270) does not (or vice-versa), so a ratio still leaks through the unwired path.
**Why it happens:** Two near-duplicate blocks; easy to patch one and miss the other.
**How to avoid:** Both blocks call the same `apply_gate()`; consider a tiny shared helper `cli._honest_metrics_for(name, data, meta, original)` to DRY them.
**Warning signs:** Grep for `dual_ratio(` in cli.py returns 2 sites — both must be gated.

### Pitfall 3: Breaking `test_certificate*` contract
**What goes wrong:** Refactor drops a key from `industry_comparison` (`beats_int4`, `rank`) or changes comparison count, failing `test_certificate_comprehensive.py` (asserts `>=9` comparisons, `beats_standard_quant`/`beats_int4` booleans, current method included).
**Why it happens:** The list moves to `literature_estimates.py` and the builder is rewritten without preserving the output shape.
**How to avoid:** Keep `_compute_industry_comparison` producing the identical dict contract; only the *source* of the 9 tuples changes (now imported). The current-run tuple `"SpectralStream (current)"` is still computed from `ratio`, not from the literature module.
**Warning signs:** `pytest tests/test_certificate.py tests/test_certificate_comprehensive.py` must stay green.

### Pitfall 4: pytest-timeout vs pytest 9 incompatibility
**What goes wrong:** Env has pytest 9.0.2; `pytest-timeout>=2.0` may emit errors/warnings on pytest 9, and the 120s global timeout could interfere.
**Why it happens:** Version drift between pinned-lower-bound `dev` extra and installed env.
**How to avoid:** New test is fast (<1s per case). If timeout plugin errors, run the new file with `-p no:timeout` in CI/validation, or confirm `pytest-timeout` works on pytest 9 first.
**Warning signs:** `pytest tests/test_honest_metrics.py` errors before any test runs with a `pytest-timeout` plugin complaint.

### Pitfall 5: Literature disclaimer leakage / missing label
**What goes wrong:** After moving constants, a competitor number still appears in `certificate.py` itself, or the "literature estimates, not measured here" label is absent from the rendered certificate.
**Why it happens:** Copy-paste leaves a literal in `certificate.py`; label only in the module docstring, not surfaced to users.
**How to avoid:** `certificate.py` imports `LITERATURE_ESTIMATES` and `LITERATURE_DISCLAIMER`; the disclaimer string is injected into the `industry_comparison` dict (e.g., a `"disclaimer"` key) and rendered in the markdown/text/HTML sections.
**Warning signs:** Test asserting `LITERATURE_DISCLAIMER` substring present in `to_markdown()`/`to_text()` output, and no `8.0`/`GPTQ` literals remain in `certificate.py` source.

## Code Examples

Grounded in the read source (verified HIGH):

### Existing byte-exact primitives (do not reimplement)
```python
# Source: spectralstream/compression/honest_metrics.py:25,133,83 (verified by read)
def serialized_nbytes(payload) -> int: ...          # recursive: bytes/ndarray/dict/list/tuple/scalars/None
def dual_ratio(original_elements: int, payload) -> Dict[str, float]:
    return {"ratio_vs_fp32": ..., "ratio_vs_bf16": ...}
class ErrorMetrics(NamedTuple):
    rel_mse, cosine_sim, max_abs, snr_db
def end_to_end_error(original, reconstructed) -> ErrorMetrics: ...
```

### CLI honest-metrics block (World Model path, ~cli.py:846-865, verified by read)
```python
ratios = dual_ratio(original.size, data)          # leading with bf16 is a CLI print-order fix, not a dual_ratio change
honest_metrics_dict["ratio_vs_fp32"] = ratios["ratio_vs_fp32"]
honest_metrics_dict["ratio_vs_bf16"] = ratios["ratio_vs_bf16"]
err = end_to_end_error(original, recon)
honest_metrics_dict["rel_mse"] = err.rel_mse
# → wrap these in apply_gate() so ratio is suppressed when err.rel_mse > 0.05
```

### CLI summary print-order fix (cli.py ~L1414-1417, verified by read)
```python
# CURRENT (wrong order): FP32 printed before BF16
if hm_fp32: logger.info("  Honest ratio (vs FP32): avg %.1fx", ...)
if hm_bf16: logger.info("  Honest ratio (vs BF16): avg %.1fx", ...)
# REQUIRED (METRICS-02): lead with BF16, label FP32 secondary
if hm_bf16: logger.info("  Honest ratio (vs BF16 / disk): avg %.1fx", ...)
if hm_fp32: logger.info("  (secondary, vs FP32): avg %.1fx", ...)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `len(dict)` / per-stage ratio products emitted as "compression ratio" | `serialized_nbytes()` + `dual_ratio()` byte-exact | Honest-metrics module already added (pre-Phase-1) | Ratios are now real; gate is the missing coupling to error. |
| Competitor constants hardcoded in `certificate.py` + orphaned `benchmark_industry_comparison.json` | Move to `literature_estimates.py` with disclaimer; delete JSON | This phase (METRICS-03) | Removes fabrication surface; preserves `industry_comparison` contract. |
| Ratio printed with FP32 first | BF16-first headline + FP32 secondary | This phase (METRICS-02) | Honest vs the dtype the model actually ships (BF16/disk). |

**Deprecated/outdated:**
- `benchmark_industry_comparison.json` — orphaned fabricated artifact; delete.
- Per-call-site error thresholds — replaced by single `ERROR_GATE_THRESHOLD` constant.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Only `certificate.py:_compute_industry_comparison` holds live hardcoded competitor constants; no other shipped `.py` reports them | Standard Stack / METRICS-03 | Low — grep of `*.py` for GPTQ/AWQ/SqueezeLLM/GGML found only tests (`test_calibration_quantizer.py`, legitimate) and `_archive/`. A missed literal would still be caught by METRICS-03's "no fabricated constants leak" test. |
| A2 | The 9-competitor tuple list is exactly the set at certificate.py L432-442 | METRICS-03 | Low — extract verbatim; if a 10th exists it is visible in the read source (lines 432-443). |
| A3 | pytest-timeout 2.x works on installed pytest 9.0.2 | Pitfalls P4 | Low/Med — if it errors, run new test with `-p no:timeout`; does not block the phase. |
| A4 | `benchmark_industry_comparison.json` deletion is safe (zero references) | METRICS-03 | Low — confirmed by grep: no `.py`/`.json` references it. |

## Open Questions

1. **Config/env override mechanism for the threshold** — CONTEXT leaves the exact key/env-var name to Claude's discretion. Recommend `SpectralStreamConfig` field + `SS_ERROR_GATE_THRESHOLD` env, mirroring existing `SS_*` precedence (env > JSON > default). **Recommendation:** Add field to `SpectralStreamConfig` only if it fits existing dataclass layout; otherwise a module-level constant + optional env read is sufficient for Phase 1.
2. **Where the disclaimer surfaces** — should "literature estimates, not measured here" appear in the `industry_comparison` dict (machine-readable) and/or only in rendered markdown/text/HTML? **Recommendation:** both — a `disclaimer` key in the dict plus rendered text, so `test_honest_metrics.py` can assert it.
3. **Gate marker shape for certificate builders** — `CompressedTensor.params["honest_metrics"]` already carries the dict; the gate result can live there directly. **Recommendation:** store `apply_gate()` output in `honest_metrics_dict` so both CLI and certificate consume one structure.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | All | ✓ | 3.x (env) | — |
| numpy | honest_metrics | ✓ | 2.4.2 | — |
| scipy | (transitive) | ✓ | 1.17.1 | — |
| pytest | tests/test_honest_metrics.py | ✓ | 9.0.2 | Use `pytest` directly |
| pytest-timeout | global 120s timeout | ✓ (may warn on pytest 9) | 2.x | `-p no:timeout` if it errors |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None blocking.

## Validation Architecture

> Required by mandate. Proves the phase goal (every ratio error-gated, byte-exact, no fabricated competitor constants) is achieved.

### Measurable Outcomes (success criteria → proof)
1. **(METRICS-01/SC-1)** A deliberately-bad method (e.g., returns noise/zeros, `rel_mse > 0.05`) emits NO numeric ratio — its `honest_metrics_dict` has `gated: True` and `ratio_vs_bf16: None`. Verifiable by running such a method and confirming no ratio appears, and by unit test.
2. **(METRICS-02/SC-2)** All ratios flow through `serialized_nbytes()` via `dual_ratio`; `ratio_vs_bf16` is the headline (printed first, labeled "(vs BF16 / disk)"), `ratio_vs_fp32` is secondary. Verifiable by inspecting CLI summary order + a test asserting `apply_gate` returns bf16-first and routes through `serialized_nbytes`.
3. **(METRICS-03/SC-3)** `benchmark_industry_comparison.json` deleted; `certificate.py` contains no bare competitor constants; any external reference carries "literature estimates, not measured here". Verifiable by grep + test.
4. **(METRICS-04/SC-4)** `tests/test_honest_metrics.py` exists and passes: ratio↔error coupling, over-threshold rejection, `serialized_nbytes` across all payload shapes.

### Requirement → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| METRICS-01 | high-rel_mse method → `gated: True`, no ratio | unit | `pytest tests/test_honest_metrics.py -v -k gate` | ❌ new |
| METRICS-01 | good method (rel_mse ≤ 0.05) → ratio emitted | unit | `pytest tests/test_honest_metrics.py -v -k gate` | ❌ new |
| METRICS-01 | threshold boundary exactly 0.05 → NOT gated | unit | `pytest tests/test_honest_metrics.py -v -k boundary` | ❌ new |
| METRICS-02 | `dual_ratio`/`serialized_nbytes` are the ratio source | unit | `pytest tests/test_honest_metrics.py -v -k ratio` | ❌ new |
| METRICS-02 | BF16 leads, FP32 secondary in output | unit + CLI smoke | `pytest tests/test_honest_metrics.py -v -k headline` + CLI invocation | ❌ new |
| METRICS-03 | no competitor literals in `certificate.py`; disclaimer present | unit + static | `pytest tests/test_honest_metrics.py -v -k literature`; `grep -rn "GPTQ\|AWQ\|SqueezeLLM" spectralstream/compression/certificate.py` returns nothing | ❌ new |
| METRICS-03 | `industry_comparison` contract preserved | unit (existing) | `pytest tests/test_certificate.py tests/test_certificate_comprehensive.py -v` | ✅ keep |
| METRICS-04 | `serialized_nbytes` all payload shapes | unit | `pytest tests/test_honest_metrics.py -v -k serialized` | ❌ new |

### Concrete Verification Commands
```bash
# 1. New anchor test suite
python -m pytest tests/test_honest_metrics.py -v

# 2. Preserve existing certificate contract (must stay green)
python -m pytest tests/test_certificate.py tests/test_certificate_comprehensive.py -v

# 3. Static check: no fabricated competitor constants remain in certificate.py
grep -rn "GPTQ\|AWQ\|SqueezeLLM\|GGML Q" spectralstream/compression/certificate.py
# expected: no matches (constants now imported from literature_estimates.py)

# 4. Orphaned JSON removed
ls benchmark_industry_comparison.json
# expected: "No such file"

# 5. CLI smoke: run a deliberately-bad method and confirm no ratio emitted
python -m spectralstream.compression.cli cmd_compress --help   # confirm CLI entry; then run on a tiny model with a noisy method and assert "GATED" in output and no numeric ratio line for that tensor
# (concrete invocation depends on cli arg shape — planner should add a fixture/script using a synthetic noise method)
```

### Edge Cases / Failure Modes the Validation MUST catch
- **`serialized_nbytes` payload shapes:** `None` → 0; `bytes`/`bytearray`/`memoryview` → len; `np.ndarray` → `.nbytes`; `np.generic` → cast; nested `dict` (keys counted as utf-8 bytes + recursive values); `list`/`tuple` (sum recursive); `bool` → 1; `int`/`float` → 8; `str` → utf-8 len; unknown scalar → `repr()` fallback. Each needs a test case.
- **Threshold boundary at exactly 0.05:** strict `>` — `0.0500` NOT gated, `0.0501` gated.
- **Literature-estimate label leakage:** disclaimer substring present in `to_markdown()`/`to_text()`/`to_dict()` output; no bare competitor ratio literal remains in `certificate.py` source.
- **Both CLI honest-metrics blocks gated:** grep for `dual_ratio(` in `cli.py` must show both sites wrapped by `apply_gate`.
- **Current-run tuple still computed from real `ratio`,** not from the literature module (so `beats_standard_quant`/`beats_int4` logic is unchanged and existing cert tests pass).
- **Method still runs when gated:** gate suppresses only the ratio claim, not execution — confirm `data` is still produced/decompressed.

## Security Domain

> `security_enforcement` not explicitly disabled; include this section. This phase touches reporting honesty (SEC-03) but introduces no auth/session/crypto surface.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | N/A — no auth in compression reporting. |
| V3 Session Management | no | N/A. |
| V4 Access Control | no | N/A. |
| V5 Input Validation | partial | Gate validates `rel_mse`/`payload` types before computing ratios; no untrusted input path added. |
| V6 Cryptography | no | N/A — no crypto; math is float compression only. |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Fabricated/ misleading metrics (SEC-03) | Spoofing/Integrity | Central `apply_gate` + `serialized_nbytes` byte-exact + labeled literature estimates; tests assert no leaked constants. |
| Silent ratio suppression misread as "1.0×" | Tampering/Integrity | Explicit `gated: True` + `gate_reason` marker; CLI renders "GATED". |

## Sources

### Primary (HIGH confidence)
- `spectralstream/compression/honest_metrics.py` (read in full) — `serialized_nbytes`, `dual_ratio`, `end_to_end_error`, `ErrorMetrics`. `[VERIFIED: source read]`
- `spectralstream/compression/cli.py` (read L820-940, L1270-1468) — two honest-metrics blocks, summary print order. `[VERIFIED: source read]`
- `spectralstream/compression/certificate.py` (read L1-90, L420-510, grep of industry/marketing) — `_compute_industry_comparison` ~L429-463. `[VERIFIED: source read]`
- `spectralstream/compression/engine/_helpers.py` (read in full) — `_enrich_meta`, `_compute_ratio`, `_grade_error`. `[VERIFIED: source read]`
- `benchmark_industry_comparison.json` — confirmed orphaned via grep (zero `.py`/`.json` references). `[VERIFIED: filesystem + grep]`
- `tests/test_certificate.py`, `tests/test_certificate_comprehensive.py` — asserted `industry_comparison` contract. `[VERIFIED: source read]`

### Secondary (MEDIUM confidence)
- `.planning/REQUIREMENTS.md` (METRICS-01..04), `.planning/phases/01-metrics-trust-loop/01-CONTEXT.md` (locked decisions). `[CITED: planning docs]`
- `.claude/CLAUDE.md` — tech-stack constraints (pure Python, numpy/scipy, honest_metrics mandate). `[CITED: project CLAUDE.md]`

### Tertiary (LOW confidence)
- None. All findings derived from direct source reads.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — reuses existing pinned deps; verified present in env.
- Architecture: HIGH — gate placement, dual CLI blocks, certificate contract all read directly from source.
- Pitfalls: HIGH — each pitfall traced to a specific read line range (cli.py L1414-1417 order, two dual_ratio sites, cert test assertions).

**Research date:** 2026-07-08
**Valid until:** 2026-08-07 (stable internal codebase; re-verify only if `honest_metrics.py`/`certificate.py` change before planning).

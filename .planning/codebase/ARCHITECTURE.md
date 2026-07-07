<!-- refreshed: 2026-07-07 -->
# Architecture

**Analysis Date:** 2026-07-07

SpectralStream is a pure-Python (NumPy-vectorized, no C++ extensions) LLM inference + weight-compression engine. The central subsystem is the **compression intelligence engine**, a tiered, method-registry-driven orchestrator over 80+ compression methods plus a 5-stage cascade pipeline (`spectralstream/compression/cascade_5stage.py`). This document captures the pipeline, the engines, and the orchestration logic.

## System Overview

```text
                         ┌──────────────────────────────────────────────────────────────┐
                         │                     ENTRY POINTS                              │
                         │  cli.py (cmd_compress/profile/validate/gen/convert/info/...) │
                         │  wave4_pipeline.py · scripts/*.py · tests/*.py               │
                         └───────────────────────────┬──────────────────────────────────┘
                                                     │  CompressionIntelligenceEngine
                                                     ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                         CompressionIntelligenceEngine  (engine/_orchestrator.py:206)            │
│                                                                                                 │
│   PROFILE ──▶ ALLOCATE ──▶ SELECT ──▶ COMPRESS ──▶ VALIDATE                                    │
│   (_profiler) (_allocator) (oracle/  (LazyMethodDict +  (_helpers.                                              │
│                (holographic  compress_tensor_    compress_tensor_                                   │
│                 /quantum      with_validation)    with_validation)                                  │
│                 cascade)                                                                          │
└───────────────┬───────────────────────────────┬─────────────────────────────────────────────────┘
                │ method registry                │ SSF binary read/write
                ▼                                ▼
┌───────────────────────────────┐   ┌─────────────────────────────────────────────────────────┐
│ registry/ (CompressionMethod) │   │ format/  — SSF v2/v3 reader/writer/index/header/core       │
│ enum (200+ members)           │   │ reader.py · writer.py · index.py · header.py · core.py     │
│ registry.py · metadata.py     │   │ ssf_format_pipeline.py · streaming_converter.py · gguf_*   │
│ registration._register_all()  │   └─────────────────────────────────────────────────────────┘
│ METHOD_CLASSES (methods/__init__) │
└───────────────┬───────────────┘
                │ each method class has .compress()/.decompress()
                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ METHODS  (spectralstream/compression/methods/) — 9 categories + novel + tensor-network + massive │
│   decomposition/ spectral/ structural/ entropy/ functional/ physics/ quantization/ lossless/      │
│   hybrid/ + novel.* (breakthrough, fractal_chaos, quantum, revolutionary, ...)                    │
└───────────────┬───────────────────────────────────────────────────────────────────────────────────┘
                │ 5-stage cascade is itself a method
                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  FiveStageCascade  (compression/cascade_5stage.py:652)                                           │
│                                                                                                 │
│   W  (4096×4096)                                                                                │
│     │  Stage 1  _einsort_stage1        (cascade_5stage.py:164)  → row/col perms                  │
│     ▼                                                                                           │
│   permuted                                                                                      │
│     │  Stage 2  _tt_svd_decompose     (cascade_5stage.py:183)  OR                                │
│     │            _svd_truncated       (cascade_5stage.py:248)  OR                                │
│     │            _block_quant (INT2/4)(cascade_5stage.py:551)  [flat-spectrum / ratio>100 path] │
│     ▼                                                                                           │
│   cores (or quant_stages)  +  primary_residual                                                 │
│                                                                                                 │
│   NOTE: stages 3/4/5 (sparse, ergodic, SIREN) are DEFINED as helpers                            │
│   (_sparse_residual_stage3:314, _ergodic_trajectory_stage4:370,                                │
│    _siren_fit_2d:450) but are NOT invoked by the compress path (see Anti-Patterns).             │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  core/math_primitives/ — 18 submodules (DCT, FWHT, FFT, wavelets, NTT, Lloyd-Max,               │
│  HRR, bfloat16, metrics, decomposition) — canonical math layer used by ALL methods.             │
│  Compression metrics: honest_metrics.py (serialized_nbytes:25, end_to_end_error:83,             │
│  dual_ratio:133) — honest byte-exact ratio + error measurement.                                 │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  DOWNSTREAM:  format/ → inference/pipeline.py (InferencePipeline:62) → kv_cache/ → serving/     │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `CompressionIntelligenceEngine` | Top orchestrator: profile→allocate→select→compress→validate, lazy method loading, sub-engine wiring | `spectralstream/compression/engine/_orchestrator.py:206` |
| `LazyMethodDict` | Memory-efficient dict; only 10 built-ins pre-loaded, others instantiated on access | `spectralstream/compression/engine/_orchestrator.py:67` |
| `CompressionConfig` / `CompressedTensor` / `TensorProfile` / `CompressionReport` | Core dataclasses for the engine | `spectralstream/compression/engine/_dataclasses.py` |
| `CompressionProfiler` | Per-tensor statistic profiling (sensitivity, rank, spectral decay, energy) | `spectralstream/compression/engine/_profiler.py` |
| `ErrorBudgetAllocator` | Distributes error budget per tensor by sensitivity | `spectralstream/compression/engine/_allocator.py` |
| `MethodDiscovery` | Discovers/instantiates all methods, builds category/tier indices | `spectralstream/compression/engine/method_discovery.py` |
| `get_tier` / `MethodTier` | Category→tier mapping (Tier1 real-compression → Tier5 quantization) | `spectralstream/compression/engine/method_tiers.py` |
| `_compress_tensor_with_validation` | Runs candidate methods within budget, picks best, validates | `spectralstream/compression/engine/_helpers.py` |
| `FiveStageCascade` | 5-stage cascade pipeline (EinSort + TT-SVD/quant, plus defined-but-unwired sparse/ergodic/SIREN helpers) | `spectralstream/compression/cascade_5stage.py:652` |
| `Cascade5StageMethod` | Engine-compatible wrapper around `FiveStageCascade` | `spectralstream/compression/cascade_5stage.py:898` |
| `ErgodicHyperfunction` | Ergodic irrational-winding compression (prime√ frequencies); registered as `ergodic_hyperfunction` | `spectralstream/compression/methods/functional/ergodic_hyperfunction.py:89` |
| `CompressionMethod` (enum) | Enumerates 200+ methods across 9+ categories | `spectralstream/compression/registry/enum.py:17` |
| `MethodRegistry` | In-memory registry of `MethodMetadata` keyed by enum | `spectralstream/compression/registry/registry.py:9` |
| `_register_all` | Auto-registers every method from `METHOD_CLASSES` with metadata | `spectralstream/compression/registry/registration.py:54` |
| `METHOD_CLASSES` | Source-of-truth dict of all method classes (eager + lazy `_load_extra`) | `spectralstream/compression/methods/__init__.py:936` |
| Engine built-in methods (`_BlockINT8`, `_SVDCompress`, `_TensorTrain`, etc.) | Always-available compressors (bytes interface) | `spectralstream/compression/engine/_methods.py` |
| `honest_metrics` | Byte-exact ratio + multi-metric error measurement (anti-fabrication) | `spectralstream/compression/honest_metrics.py` |
| `CompressionCertificate` | JSON/HTML/MD/TXT report generation | `spectralstream/compression/certificate.py` |
| CLI (`cmd_compress`, etc.) | Unified CLI: compress/profile/list-methods/validate/benchmark/generate/convert/info | `spectralstream/compression/cli.py` (`cmd_compress:601`, `build_parser:2701`) |
| `SSFReader` / `SSFWriter` / `SSFIndex` / `SSFHeader` | SSF v2/v3 binary format I/O | `spectralstream/format/reader.py`, `writer.py`, `index.py`, `header.py` |
| `core/math_primitives` | Canonical math layer (DCT/FWHT/FFT/wavelets/NTT/Lloyd-Max/HRR/bf16/metrics) | `spectralstream/core/math_primitives/__init__.py` |
| `InferencePipeline` | Loads SSF/safetensors, runs Gemma-4 forward pass + generation + KV cache | `spectralstream/inference/pipeline.py:62` |
| `SpectralStreamConfig` | Layered config (dataclass sections + `SS_*` env vars + model overrides) | `spectralstream/config.py:197` |
| `Wave4Pipeline` (`wave4_pipeline.py`) | Standalone harness: runs 5-stage cascade + BlockINT8 on Gemma-4-E2B tensor slices | `wave4_pipeline.py:164` |
| `DirectCascadeEngine` | Alternate "direct" cascade orchestration (129k lines) | `spectralstream/compression/engine/direct_cascade.py` |
| `QuantumCascadeEngine` / `QuantumSuperpositionEngine` | Parallel method-testing via "quantum superposition" metaphor | `spectralstream/compression/engine/quantum_cascade.py` |
| `HolographicOracle` | Associative-memory zero-shot method selection | `spectralstream/compression/engine/holographic_oracle.py` |
| `WorldModelCompressor` / `MethodOracle` | Perf-history model oracle for method selection | `spectralstream/compression/engine/world_model_compressor.py`, `world_model/method_oracle.py` |

## Pattern Overview

**Overall:** Registry-driven, tiered, lazy-evaluated compression intelligence engine with a cascade sub-pipeline.

**Key Characteristics:**
- **Method Registry + Enum pattern:** Every method is a class with `name`/`category`/`compress()`/`decompress()`. Names map to a `CompressionMethod` enum (`registry/enum.py`) and a `MethodMetadata` record (`registry/registry.py`). Registration is automatic via `_register_all()` (`registry/registration.py:54`).
- **Tiered priority:** Tier is derived from method *category* (`method_tiers.py`), not a manual map — Tier1 (decomposition/spectral/tensor-network/functional, score 10) → Tier5 (quantization, score 0.3). The selector tries Tier-1 first and cascades down when the error budget isn't met.
- **Lazy loading:** `METHOD_CLASSES` is a `_MethodClassesDict` that triggers `_load_extra()` (heavy breakthrough/massive/novel sections) only on first access (`methods/__init__.py:608`). `ALL_METHODS` / `LazyMethodDict` defer instantiation similarly. This keeps the import footprint and RAM low (only ~10 built-ins pre-loaded).
- **Honest-metrics mandate:** All ratios/errors MUST flow through `honest_metrics.py` (`serialized_nbytes:25`, `end_to_end_error:83`, `dual_ratio:133`) — never `len(dict)` or products of per-stage estimates (see Error Handling).
- **BF16-aware I/O:** All compress/decompress methods accept `uint16` BF16 tensors, convert to float32 for arithmetic, and flag via `_input_was_bf16` (`engine/_methods.py:38`, `_dtype_utils.py`).
- **Memory-bounded:** Chunked/memory-mapped/streaming compressors kick in above a memory budget (`compress_within_budget:376`, `chunked_compressor.py`, `memory_mapped_engine.py`, `streaming_pipeline.py`).

## Layers

**Layer 1 — Entry / CLI (`spectralstream/compression/cli.py`, `wave4_pipeline.py`, `scripts/`)**
- Purpose: User-facing commands and run harnesses.
- Location: `spectralstream/compression/cli.py`
- Contains: `build_parser:2701`, `cmd_compress:601`, `cmd_validate:1816`, `cmd_benchmark:2154`, etc.
- Depends on: `engine.CompressionIntelligenceEngine`, `format.reader`, `honest_metrics`.
- Used by: End users, `tests/`, `scripts/`.

**Layer 2 — Orchestration (`spectralstream/compression/engine/_orchestrator.py`)**
- Purpose: Profile → allocate → select → compress → validate loop.
- Location: `spectralstream/compression/engine/`
- Contains: `CompressionIntelligenceEngine`, `LazyMethodDict`, profiler/allocator/selector.
- Depends on: registry, methods, honest_metrics, format.
- Used by: CLI, scripts, inference.

**Layer 3 — Method Registry & Selection (`registry/`, `methods/`, `engine/method_tiers.py`, `method_discovery.py`)**
- Purpose: Discover, catalog, and tier all methods; pick the best per tensor.
- Depends on: `methods/`, `core/math_primitives`.
- Used by: Orchestrator.

**Layer 4 — Compression Methods (`spectralstream/compression/methods/`)**
- Purpose: Individual `compress()`/`decompress()` implementations across categories.
- Location: `spectralstream/compression/methods/{decomposition,spectral,structural,entropy,functional,physics,quantization,lossless,hybrid,novel,...}/`
- Depends on: `core/math_primitives`.
- Used by: Orchestrator, cascade.

**Layer 5 — 5-Stage Cascade (`spectralstream/compression/cascade_5stage.py`)**
- Purpose: A specific method that chains stage helpers on the residual of the previous.
- Depends on: `core/math_primitives`, `_dtype_utils`, `honest_metrics`.
- Used by: Orchestrator (`cascade_5stage` method), `wave4_pipeline.py`.

**Layer 6 — Format & Math (`spectralstream/format/`, `spectralstream/core/math_primitives/`)**
- Purpose: SSF binary I/O and the canonical math primitives.
- Used by: Every layer above.

**Layer 7 — Downstream (`spectralstream/inference/`, `kv_cache/`, `serving/`)**
- Purpose: Decompress SSF at inference time and run the model forward pass.
- Depends on: format, compression.

## Data Flow

### Primary Compression Path (engine)

1. **Entry** — `CompressionIntelligenceEngine(config)` is constructed; `LazyMethodDict` loads 10 built-ins, `MethodDiscovery.discover()` indexes all methods by category/tier (`_orchestrator.py:217-246`).
2. **Request** — `compress_model()` / `compress_tensor()` / `compress_fast()` (`_orchestrator.py:1273`, `474`).
3. **Profile** — `profile_tensor()` builds a `TensorProfile` (`engine/_profiler.py`); tensor classified by name via `_classify_by_name` (`engine/_helpers.py`).
4. **Allocate** — `ErrorBudgetAllocator` computes per-tensor error budget from `max_error/target_ratio` (`_orchestrator.py:482`, `engine/_allocator.py`).
5. **Select** — `HolographicOracle.select_method()` or `MethodOracle.select_with_bypass()` returns a ranked method list (`_orchestrator.py:486-545`).
6. **Compress** — `compress_tensor_with_validation()` runs each candidate, measures with `honest_metrics`, picks the best within budget (`_orchestrator.py:462`, `engine/_helpers.py`).
7. **Serialize** — payload bytes written via `format.writer.SSFWriter` (`cmd_compress:601` → `_write_ssf:1640`).
8. **Report** — `certificate.py` produces JSON/HTML/MD/TXT per-tensor metrics; `dual_ratio()` reports vs FP32 and BF16.

### 5-Stage Cascade Path (per tensor)

1. **Normalize** — `FiveStageCascade.compress()` casts to float64; reshapes >2D to 2D; dispatches to `_compress_1d` (`:702`) or `_compress_2d` (`:733`).
2. **Stage 1 — EinSort** — `_einsort_stage1()` (`:164`) sorts rows/cols by second-moment norms, producing `row_perm`/`col_perm`. This is the permutation-space alignment.
3. **Path selection** — `_detect_flat_spectrum()` (`:514`) checks whether >30% rank is needed for 90% energy (true for LLM weights). If flat OR `target_ratio > 100` → quantization path; else SVD/TT path.
4. **Stage 2a — TT-SVD** (structured path) — `_tt_svd_decompose()` (`:183`) folds the matrix into a d-dimensional tensor (`_auto_fold_dims:130`), runs sequential SVD, applies energy-based rank selection, stores TT cores (float16). OR `_svd_truncated()` (`:248`) for small matrices.
5. **Stage 2b — Block Quant** (flat-spectrum path) — multi-stage residual `_block_quant()` (`:551`, INT2/INT4 with bit-packing) looped `n_stages` times (`:757`).
6. **Output** — payload = `{s1_row_perm, s1_col_perm, s2_type, s2_cores | s2_quant_stages}`; metadata stores `used_stages=[1,2]`, `primary_rel_error`, `_storage_dtype`.
7. **Decompress** — `_decompress_2d()` (`:838`) reconstructs TT/quant, then `_inverse_permute()` (`:175`) undoes EinSort.

**Stages 3 (Sparse), 4 (Ergodic), 5 (SIREN):** Helper functions exist — `_sparse_residual_stage3:314`, `_ergodic_trajectory_stage4:370`, `_siren_fit_2d:450` (plus ergonic reconstructors) — but they are **NOT called** in the compress path. See Anti-Patterns.

**State Management:** Per-call statelessness in the cascade (payload+metadata fully describe a tensor). The engine keeps performance history in `MethodOracle`/`HolographicOracle` for zero-shot selection. Config state lives in `SpectralStreamConfig` (`config.py:197`).

### Metrics Flow

- `honest_metrics.serialized_nbytes(payload)` (`:25`) → true byte cost (handles bytes/ndarray/dict/list/scalars recursively).
- `end_to_end_error(orig, recon)` (`:83`) → `ErrorMetrics(rel_mse, cosine_sim, max_abs, snr_db)`.
- `dual_ratio(elements, payload)` (`:133`) → `ratio_vs_fp32`, `ratio_vs_bf16`.
- `wave4_pipeline.run_5stage()` (`:108`) runs cascade, pickles payload, computes `end_to_end_error` + `dual_ratio`, writes `wave4_results.json`.

## Key Abstractions

**`CompressionMethod` enum** (`registry/enum.py:17`): 200+ members; auto-creates members for unknown names via `_missing_` (`:472`). Source of canonical identities for methods.

**`MethodMetadata`** (`registry/metadata.py:9`): `compression_ratio_range`, `expected_error_range`, `category`, `is_lossless`, `requires_calibration`, `supports_streaming`. Stored per enum in `MethodRegistry`.

**`METHOD_CLASSES`** (`methods/__init__.py:936`): `dict[name → class]`. The single source of truth for all method implementations. Eager section populated at import; ~10 heavy sections loaded lazily via `_load_extra()` (`:422`).

**`LazyMethodDict`** (`_orchestrator.py:67`): Defers method instantiation; only `_BUILTIN_NAMES` (10) pre-loaded.

**`FiveStageCascade`** (`cascade_5stage.py:652`): A compressor exposing `compress(tensor, target_ratio)→(payload, meta)` and `decompress(payload, meta)→tensor`. Its `compress` actually runs a 2-stage subset; rest are helper functions.

**`CompressionConfig`** (`engine/_dataclasses.py`): `target_ratio`, `max_error`, and other knobs consumed by the orchestrator.

**`CompressedTensor`** (`engine/_dataclasses.py`): Result object carrying payload bytes, metadata, ratios, and errors.

## Entry Points

**CLI** (`spectralstream/compression/cli.py`): invoked as `python -m spectralstream.compression.cli <cmd>`. Subcommands: `compress`, `profile`, `list-methods`, `validate`, `verify`, `benchmark`, `dial-in`, `infer`, `generate-certificate`, `convert`, `info`. Parser built at `build_parser:2701`.

**Wave 4 harness** (`wave4_pipeline.py:164`): `main()` discovers Gemma-4-E2B tensors by category, runs `run_5stage` (`:108`) and `run_block_int8` (`:139`) on 512×512 slices, writes `wave4_results.json`.

**Scripts** (`scripts/`): `e2e_validation.py`, `e2e_test.py`, `compress_gemma4.py`, `run_benchmark.py`, `diagnose_5stage.py`, `run_5stage_on_model.py`, `tune_cascade_quality.py`, `baseline_honest_test.py`, `dial_in_spectral.py`, `final_benchmark.py`, `benchmark_compression.py`, `global_migration_validate.py`.

**Tests** (`tests/`): `test_pipeline.py`, `test_cli.py`, `test_complete_system.py`, `test_certificate*.py`, `test_method_*.py`, `test_unified_*.py`, `test_archive_*.py`, `test_kv_cache_*.py`, `test_inference_*.py`, etc. (pytest, `testpaths=["tests"]` in `pyproject.toml`).

**Inference** (`spectralstream/inference/pipeline.py:62`): `InferencePipeline("model.ssf").generate("Hello")`.

**Note:** There is **no** `__main__.py` in the package (verified — `find . -name __main__.py` returns none). All execution goes through `python -m spectralstream.compression.cli` or the top-level scripts.

## Architectural Constraints

- **Threading:** Single-threaded Python (NumPy/GIL). `InferencePipeline` documents "thread-safe generation (re-entrant via per-call state)" (`inference/pipeline.py:76`), and `kv_cache` has a "Thread-safe" note, but the compression engine itself has no explicit threading — parallelism comes from `QuantumCascadeEngine` (process/pool-based "superposition" of method trials, `quantum_cascade.py`) and `ParallelCompressor` (`engine/parallel_compressor.py`).
- **Global state:** `SpectralStreamConfig` is instantiated per-call (no module-level singleton). `MethodRegistry._methods` is class-level (registry/registry.py:12) — a shared mutable global dictionary populated at import by `_register_all()`. `_METHOD_TIER_MAP_CACHE` (method_tiers.py:140) and `_EXTRA_LOADED` flag (methods/__init__.py:419) are module-level globals controlling lazy registration.
- **Circular imports:** Deliberately broken via lazy `__getattr__` / deferred imports. `method_tiers.py` uses module-level `__getattr__` (`:155`) to avoid `method_tiers → method_discovery → method_tiers`. `engine/__init__.py` re-exports everything, so importers should import submodules directly or accept the heavy import. `METHOD_CLASSES` uses `_load_extra()` lazy trigger rather than importing heavy modules at module top.
- **dtypes:** Storage dtype is auto-detected and encoded into metadata (`_dtype_utils.encode_dtype_code`); BF16 carried as `uint16`. All arithmetic cast to float64/float32.
- **Determinism:** RNG seeds fixed (`np.random.RandomState(42)` in cascade `_siren_fit_2d:467` and `_detect_flat_spectrum:528`) for reproducibility.

## Anti-Patterns

### 5-Stage cascade only wires 2 stages
**What happens:** `README.md` and commit messages describe a 5-stage pipeline (EinSort → TT-SVD → Sparse Residual → Ergodic → SIREN), but `FiveStageCascade._compress_2d` (`cascade_5stage.py:733`) sets `used_stages=[1,2]` and calls only `_einsort_stage1` (Stage 1) + `_tt_svd_decompose`/`_svd_truncated`/`_block_quant` (Stage 2). The stage-3/4/5 helper functions (`_sparse_residual_stage3:314`, `_ergodic_trajectory_stage4:370`, `_siren_fit_2d:450`) are defined and even imported (`ErgodicHyperfunction` imported at `cascade_5stage.py:17` from `methods/functional/ergodic_hyperfunction.py`) but never invoked in `compress()`.
**Why it's wrong:** The actual compression does not match the documented/advertised pipeline; downstream consumers (e.g. `wave4_pipeline.py` reporting) measure only the 2-stage behavior. Metrics and documentation diverge from code.
**Do this instead:** Either invoke the residual stages in `_compress_2d`/`_decompress_2d` (chaining the `residual` output of each stage into the next, accumulating `used_stages`) or update `README.md`/docs to state the cascade currently implements EinSort + TT/Quant only. The helper functions already have matching reconstructors and are close to being wired.

### Fabricated compression ratios (historical, now guarded)
**What happens:** Per `honest_metrics.py:1-12`, earlier cascade engines reported ratios from `len(dict)` (counting keys) or by multiplying per-stage *estimated* ratios instead of measuring real serialized bytes.
**Why it's wrong:** Produced ratios off by orders of magnitude vs the true on-disk size.
**Do this instead:** ALL ratio/error numbers must be derived from `honest_metrics.serialized_nbytes()` / `end_to_end_error()` / `dual_ratio()` (byte-exact). Any new stage must return a measurable payload, not an estimate.

### Module-level re-export bloat
**What happens:** `engine/__init__.py` re-exports ~200 names (lines 1-576) including archive-migration classes (`intelligence_real`, `unified_quant_system`, `compression_intelligence`, etc.). Importing `from spectralstream.compression.engine import ...` triggers a large import graph.
**Why it's wrong:** Slow imports; obscures which subsystem is canonical vs archived.
**Do this instead:** Import the live orchestrator (`_orchestrator.CompressionIntelligenceEngine`) directly; treat `*_real`/`*_v2`/`archive*` re-exports as legacy.

### Redundant method representations
**What happens:** The same algorithm exists in multiple places — engine-built-ins (`_methods.py`), `methods/__init__.py` `METHOD_CLASSES`, the `CompressionMethod` enum, and `_register_all` `_NAME_TO_ENUM` map. Keeping them in sync is manual.
**Why it's wrong:** High risk of drift; e.g. a method added to `METHOD_CLASSES` but missing from the enum/name-map silently fails registration.
**Do this instead:** Treat `METHOD_CLASSES` as the single source; derive enum/metadata automatically (as `_register_all` does), and avoid hand-duplicating names.

## Error Handling

**Strategy:** Defensive per-method; failures degrade rather than abort. `compress_tensor_with_validation` tries each candidate method and selects the best within budget. Lazy loaders swallow `ImportError`/`Exception` (e.g. `methods/__init__.py:604`, `LazyMethodDict._resolve` catches `Exception`).

**Patterns:**
- Methods return `(payload_bytes, metadata_dict)`; metadata carries `_input_was_bf16` and method name.
- Numeric fallbacks: `_randomized_svd` (`_methods.py:69`) falls back to `np.eye`/`np.ones` on `LinAlgError`.
- `_detect_flat_spectrum` (`:514`) catches all exceptions and returns `False` (safe default).
- CLI validates paths against traversal (`cli.py:_validate_input_path:81`, regex `:78`).
- Honest error reporting via `ErrorMetrics` (rel_mse, cosine_sim, max_abs, snr_db) — never a single fabricated scalar.

## Cross-Cutting Concerns

**Logging:** Standard `logging` module; `spectralstream/logging_config.py` configures format/handlers; `core/math_primitives/metrics_dashboard.py` formats summaries.

**Validation:** `certificate.py` produces validation certificates; `e2e_validation.py` runs end-to-end with exit code 0 (pass) / 1 (threshold breach). Quality grading in `engine/loss_metrics.py` (`_grade_quality`, SNR/MSE/cosine thresholds).

**Configuration:** `SpectralStreamConfig` (`config.py:197`) — 9 dataclass sections, loaded from file → env (`SS_*` prefix) → defaults; per-model overrides (`_MODEL_OVERRIDES:144`) and per-hardware tuning (`for_hardware:314`).

**Authentication:** None (local CLI tool). No auth layer.

**Security:** Path-traversal guard in CLI (`:78-89`); otherwise no secrets handling. `.gitignore` excludes `.venv/`, `models/`, `.intelligence/`.

**Performance:** Memory budget gating (`_orchestrator.compress_within_budget:376`) routes large tensors to chunked/memory-mapped/streaming compressors. Large-file support via mmap (`format/reader.py`, `engine/memory_mapped_engine.py`).

**Metrics honesty:** Mandated via `honest_metrics.py`; this is the central cross-cutting correctness guarantee of the whole engine.

---

*Architecture analysis: 2026-07-07*

# Codebase Structure

**Analysis Date:** 2026-07-07

## Directory Layout

```
.
├── pyproject.toml                  # setuptools build, pytest config, deps
├── requirements.txt                # tree-sitter deps (not core runtime)
├── README.md                       # project overview + 5-stage pipeline docs
├── AGENTS.md                       # agent/contributor guidance
├── LICENSE                         # AGPL-3.0
├── MATH_RESEARCH_REPORT.md
├── REAL_WORLD_BENCHMARK.md
├── compression-roadmap.md
├── wave4_pipeline.py               # WAVE 4: 5-stage cascade harness on Gemma-4-E2B
├── wave4_results.json              # output: per-category 5stage vs block_int8 metrics
├── run_full_model_honest.log / .json
├── final_benchmark_results.json
├── stage_diagnosis.json
├── benchmark_industry_comparison.json
├── final_compression_log.txt       # 240KB run log
├── potential_path.txt
├── CMakeLists.txt                  # (legacy; not used by Python build)
├── configs/                        # runtime configs for LM Studio / vLLM / Ollama / OpenAI / Codex / OpenCode
├── docs/                           # research docs (KV cache, quantization, inference, novel methods)
│   └── whitepaper/                 # LaTeX + PDF whitepaper source
├── _archive/                       # ARCHIVED code (committed, not active)
│   └── v1/
│       └── scripts/                # legacy dial-in / profile / decomposition scripts
├── .codegraph/                     # codegraph tool cache (generated)
├── .opencode/                      # opencode config (generated)
├── .planning/                      # GSD planning artifacts (this doc lives here)
│   └── codebase/
├── spectralstream.egg-info/        # pip/setuptools metadata (generated)
├── spectralstream/                 # ── MAIN PACKAGE ──
│   ├── __init__.py
│   ├── version.py
│   ├── config.py                   # SpectralStreamConfig (dataclass + SS_ env)
│   ├── logging_config.py
│   ├── orchestrator.py             # top-level orchestrator (32KB)
│   ├── audit.py
│   ├── unified_core.py / unified_attention.py / unified_kv_cache.py / unified_quantizer.py  # compat shims
│   ├── gemma4_config.py / gguf_parser_engine.py / llama_bridge.py / model_compressor.py / supreme_quant_engine.py
│   ├── core/
│   │   ├── tensor_ops.py
│   │   ├── validation.py
│   │   └── math_primitives/        # 18 submodules (canonical math layer)
│   │       ├── bfloat16.py fft.py transforms.py spectral.py numerical.py coherence.py
│   │       ├── quantization.py kernels.py rotators.py hd_vectors.py wavelets.py ntt.py
│   │       ├── compressed_sensing.py decomposition.py metrics.py metrics_dashboard.py
│   │       ├── prng.py dtype_detection.py chirplet.py legacy_ops.py quality.py
│   ├── compression/
│   │   ├── __init__.py             # exports engine + benchmark
│   │   ├── cascade_5stage.py       # FiveStageCascade (32KB) — 5-stage pipeline
│   │   ├── honest_metrics.py       # byte-exact ratio/error (anti-fabrication)
│   │   ├── certificate.py          # JSON/HTML/MD/TXT reports (52KB)
│   │   ├── cli.py                  # unified CLI (124KB, cmd_* functions)
│   │   ├── cli_dashboard.py
│   │   ├── cutting_edge.py
│   │   ├── _dtype_utils.py / _imports.py / method_registry.py
│   │   ├── adaptive_rank.py / optimizer.py / physics_compression.py / novel_compression_library.py
│   │   ├── novel_operators.py / noise_aware_compressor.py / unified_compression_pipeline.py
│   │   ├── model_compressor.py
│   │   ├── registry/               # CompressionMethod enum, MethodRegistry, auto-registration
│   │   │   ├── enum.py metadata.py registry.py registration.py __init__.py
│   │   ├── methods/                # ALL compression method implementations
│   │   │   ├── __init__.py         # METHOD_CLASSES (eager + lazy _load_extra)
│   │   │   ├── decomposition/ spectral/ structural/ entropy/ functional/ physics/
│   │   │   ├── quantization/ lossless/ hybrid/ information/ cross_layer/ novel/
│   │   │   └── novel/              # breakthrough/, fractal_chaos/, quantum/, revolutionary/,
│   │   │                           #   structural/, topological/, entropy_info/, physics/,
│   │   │                           #   quantization_massive/ (deeeep nesting)
│   │   ├── engine/                 # the intelligence engine + sub-engines
│   │   │   ├── __init__.py _orchestrator.py _methods.py _helpers.py _profiler.py
│   │   │   ├── _allocator.py _dataclasses.py _sensitivity.py _selector.py _tier_common.py
│   │   │   ├── _lru_cache.py _utils.py _io.py _tensor_type_strategy.py
│   │   │   ├── direct_cascade.py   # 129KB alternate cascade engine
│   │   │   ├── quantum_cascade.py holographic_oracle.py self_evolving_intelligence.py
│   │   │   ├── world_model_compressor.py grouping_optimizer.py resonant_grouping.py
│   │   │   ├── parallel_compressor.py streaming_compressor.py streaming_pipeline.py
│   │   │   ├── memory_mapped_engine.py chunked_compressor.py progressive_release.py
│   │   │   ├── hpc_kernel_fusion.py cascade_learner.py dynamic_method_tester.py
│   │   │   ├── quantization_engine.py loss_metrics.py model_intelligence.py
│   │   │   ├── model_calibrator.py method_discovery.py method_tiers.py method_validation.py
│   │   │   ├── moe_compression.py multi_shard_io.py float32_support.py
│   │   │   ├── compression_intelligence.py intelligence.py intelligence_real.py
│   │   │   ├── dynamic_tensor_intelligence.py unified_quant_system.py compression_profiler.py
│   │   │   ├── stacking_engine.py tiered_error.py cascade_configs.py
│   │   │   ├── dynamic_selector2.py dynamic_tuning/  # time_crystal, quantum_field_cascade,
│   │   │   │                    #   plasma_confinement, multiplicative_stacking, nas/
│   │   │   │                    #   f1_cascade_optimizer, cascade_engine, quantization_tuning
│   │   │   ├── quantum_plasma_fusion/ streaming/ world_model/
│   │   ├── advanced/               # hyper_compression_v2, sparsity_engine, tt_pq_engine, + standalone .py
│   │   ├── benchmark/              # benchmark_runner, report_generator, loss_calculator, dial_in_optimizer
│   │   ├── cutting_edge/           # 25 cutting-edge method modules (_*.py)
│   │   ├── novel_compression_library/  # standalone novel library methods
│   │   ├── profiler/ streaming/ world_model/
│   ├── format/                     # SSF v2/v3 binary format + GGUF conversion
│   │   ├── reader.py writer.py index.py header.py core.py compression.py
│   │   ├── converter.py ssf_format.py sscx_format.py sst_format.py
│   │   ├── ssf_format_pipeline.py  # 125KB master SSF pipeline
│   │   ├── streaming_converter.py  # 68KB
│   │   ├── gguf_converter.py gguf_parser_engine.py model_converter.py conversion_report.py
│   ├── inference/                  # CPU inference engine
│   │   ├── pipeline.py engine.py loader.py layer.py ffn.py attention.py generation.py
│   │   ├── vlasov*.py coconut.py hdc_engine.py hrr_memory.py coherence.py resonance.py
│   │   ├── mean_field.py mmap_engine.py unified.py unified_loader.py etc.
│   ├── kv_cache/                   # unified KV cache (core, manager, eviction, compressor, v2, ultimate, spectral)
│   ├── model/                      # gemma4_config, gguf_model, model_config, model_targets
│   ├── memory/  embeddings/  finetuning/  tensor/  agents/  attention/  serving/  utils/
├── scripts/                        # runnable harnesses (chmod +x)
│   ├── e2e_validation.py e2e_test.py compress_gemma4.py run_benchmark.py
│   ├── run_5stage_on_model.py run_5stage_on_weights.py test_5stage_cascade.py test_cascade_on_weights.py
│   ├── diagnose_5stage.py tune_cascade_quality.py baseline_honest_test.py
│   ├── final_benchmark.py dial_in_spectral.py benchmark_compression.py global_migration_validate.py
├── tests/                          # pytest suite (testpaths=["tests"])
│   ├── conftest.py run_all_tests.py
│   ├── data/ output/
│   ├── test_pipeline.py test_cli.py test_complete_system.py
│   ├── test_certificate*.py test_method_*.py test_unified_*.py test_archive_*.py
│   ├── test_kv_cache_*.py test_inference_*.py test_loss_metrics_engine.py
│   ├── benchmark*.py  (also many *.json result files committed)
```

## Directory Purposes

**`spectralstream/`** — Main package. Everything importable lives here.

**`spectralstream/compression/`** — Compression intelligence engine: orchestrator, 80+ methods, registry, cascade, CLI, certificate, metrics.

**`spectralstream/compression/engine/`** — The orchestrator and its sub-engines (selection, profiling, allocation, cascade variants, world-model, holographic oracle, streaming, quantum cascade). `direct_cascade.py` (129KB) is a parallel/legacy cascade implementation.

**`spectralstream/compression/methods/`** — Every individual compression method class. `METHOD_CLASSES` (`methods/__init__.py:936`) is the source of truth. Heavy subsections (novel/breakthrough/massive) load lazily.

**`spectralstream/compression/registry/`** — `CompressionMethod` enum (200+ members), `MethodRegistry`, `MethodMetadata`, and `_register_all()` auto-registration.

**`spectralstream/core/math_primitives/`** — Canonical math used by all methods: DCT, FWHT, FFT, wavelets, NTT, Lloyd-Max, HRR, BF16, metrics, decomposition.

**`spectralstream/format/`** — SSF v2/v3 binary format read/write, index/header/core, GGUF conversion, streaming converter.

**`spectralstream/inference/`** — Production CPU inference pipeline, Gemma-4 forward pass, token generation, KV cache integration, Vlasov/HDC/COCONUT engines.

**`spectralstream/kv_cache/`** — Unified KV cache with 30+ eviction/compression policies.

**`spectralstream/model/`, `configs/`, `docs/`** — Model configs, external tool configs, research documentation.

**`scripts/`** — Standalone runnable Python harnesses (e2e validation, benchmarks, cascade tuning/diagnosis).

**`tests/`** — pytest suite + committed `.json` benchmark/result artifacts.

**`_archive/v1/`** — Archived code (committed, not active). README says the web dashboard was archived here; migrate into active package to use.

**`wave4_pipeline.py`** (repo root) — Wave 4 harness comparing the 5-stage cascade vs BlockINT8 on real Gemma-4-E2B tensor slices; writes `wave4_results.json`.

## Key File Locations

**Entry Points:**
- CLI: `spectralstream/compression/cli.py` (`build_parser:2701`, `cmd_compress:601`)
- Wave4 harness: `wave4_pipeline.py` (`main:164`)
- Scripts: `scripts/e2e_validation.py`, `scripts/compress_gemma4.py`, `scripts/run_benchmark.py`, `scripts/diagnose_5stage.py`
- Inference: `spectralstream/inference/pipeline.py` (`InferencePipeline:62`)

**Configuration:**
- `pyproject.toml` (build/test/deps), `requirements.txt`, `spectralstream/config.py` (`SpectralStreamConfig:197`), `configs/*.json`
- `.gitignore` (excludes `.venv/`, `models/`, `.intelligence/`, `__pycache__/`)

**Core Logic:**
- Orchestrator: `spectralstream/compression/engine/_orchestrator.py` (`CompressionIntelligenceEngine:206`)
- 5-stage cascade: `spectralstream/compression/cascade_5stage.py` (`FiveStageCascade:652`)
- Method registry: `spectralstream/compression/registry/`
- Method source: `spectralstream/compression/methods/__init__.py` (`METHOD_CLASSES:936`)
- Honest metrics: `spectralstream/compression/honest_metrics.py`
- Math primitives: `spectralstream/core/math_primitives/`
- Format I/O: `spectralstream/format/`

**Tests:**
- `tests/` (pytest, `testpaths=["tests"]` in `pyproject.toml`)

## Naming Conventions

**Files:**
- Methods: snake_case module per method or grouped `_class_wrappers.py` (e.g. `decomposition/_class_wrappers.py`).
- Engine internals: underscore-prefixed modules (`_orchestrator.py`, `_methods.py`, `_helpers.py`, `_profiler.py`).
- Cutting-edge method modules: underscore-prefixed (`cutting_edge/_algebraicgeometrycompression.py`).
- CLI command handlers: `cmd_<verb>` (`cli.py:cmd_compress`, `cmd_validate`).
- Registry enum members: UPPER_SNAKE_CASE (`BLOCK_INT8`, `TT_SVD`).

**Classes:**
- Method classes: PascalCase, expose `name` (snake) + `category` class attributes and `compress()`/`decompress()` methods (e.g. `ErgodicHyperfunction`, `_BlockINT8`).
- Engine classes: PascalCase (`CompressionIntelligenceEngine`, `MethodRegistry`).

**Functions:**
- Stage helpers in cascade: `_<stage>_stage<n>` (`_einsort_stage1`, `_tt_svd_decompose` [stage2], `_sparse_residual_stage3`, `_ergodic_trajectory_stage4`, `_siren_fit_2d` [stage5]).
- Metrics: `snake_case` (`serialized_nbytes`, `end_to_end_error`, `dual_ratio`).

**Directories:**
- Category directories mirror method categories: `decomposition/`, `spectral/`, `structural/`, `entropy/`, `functional/`, `physics/`, `quantization/`, `lossless/`, `hybrid/`, `novel/`.
- Deeply nested `novel/` subcategories (e.g. `novel/breakthrough/breakthrough_massive/`).

## Where to Add New Code

**New compression method (recommended path):**
1. Implement a class with `name`, `category`, `compress(tensor, **kw)→(payload, meta)`, `decompress(payload, meta)→tensor` in the appropriate `spectralstream/compression/methods/<category>/` module.
2. Add it to the `METHOD_CLASSES` dict in `spectralstream/compression/methods/__init__.py` (`:649`). If it's a heavy archive/massive section, add a loader branch in `_load_extra()` (`:422`).
3. If it needs a `CompressionMethod` enum identity, add a member in `spectralstream/compression/registry/enum.py` and map the name in `_NAME_TO_ENUM` (`registry/registration.py:134`) — otherwise `_find_enum_for_name` auto-creates one.
4. The orchestrator will automatically discover, tier, and select it.

**New engine sub-component:**
- `spectralstream/compression/engine/` (add module; re-export from `engine/__init__.py` only if it's the canonical live API — avoid adding archive duplicates).

**New math primitive:**
- `spectralstream/core/math_primitives/` (add module; re-export from `__init__.py`).

**New format feature:**
- `spectralstream/format/` (reader/writer/index/header/core).

**New CLI command:**
- Add `cmd_<verb>` in `spectralstream/compression/cli.py` and register it in `build_parser` (`:2701`).

**New run harness:**
- `scripts/` (make it `chmod +x`).

**New test:**
- `tests/test_*.py` (pytest auto-discovers; markers: `gemma4`, `validation`, `slow` in `pyproject.toml`).

## Special Directories

**Generated at runtime (NOT committed unless noted):**
- `__pycache__/` (excluded by `.gitignore`) — Python bytecode caches, present in most dirs.
- `.codegraph/` — codegraph tool cache.
- `.opencode/` (`opencode.json`) — opencode editor config.
- `.pytest_cache/`, `.ruff_cache/`, `.coverage` — test/coverage caches.
- `.intelligence/`, `intelligence/index/audit_log.json` — runtime intelligence state (excluded by `.gitignore`).
- `models/` — model weights (excluded by `.gitignore`; `wave4_pipeline.py` expects `models/gemma-4-E2B/model.safetensors`).

**Committed build metadata:**
- `spectralstream.egg-info/` — setuptools install metadata (committed; regenerate on `pip install`).

**Committed generated artifacts (note: these are data, not source):**
- `tests/*.json`, `tests/data/`, `tests/output/` — committed benchmark/result JSON.
- `wave4_results.json`, `run_full_model_honest_results.json`, `stage_diagnosis.json`, `final_benchmark_results.json`, `benchmark_industry_comparison.json`, `final_compression_log.txt` — committed run outputs at repo root.

**Archived (committed, inactive):**
- `_archive/v1/` — old scripts/serving; superseded by active code. Per README, the web dashboard was archived here and must be migrated into `spectralstream/serving/` to use.

**Source of truth vs derived:**
- `METHOD_CLASSES` (`methods/__init__.py`) is the canonical method source; the `CompressionMethod` enum and `MethodMetadata` registry are derived/registered from it. Prefer editing `METHOD_CLASSES` over hand-editing the enum maps.

---

*Structure analysis: 2026-07-07*

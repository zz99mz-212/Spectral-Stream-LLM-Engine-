# Testing Patterns

**Analysis Date:** 2026-07-07

## Test Framework

**Runner:**
- pytest `>=7.0` (declared in `pyproject.toml` `[project.optional-dependencies]` `dev`).
- `pytest-timeout>=2.0` provides the `timeout` fixture/marker.
- Config block `[tool.pytest.ini_options]` in `pyproject.toml` (lines 28-35):
  ```toml
  [tool.pytest.ini_options]
  timeout = 120
  testpaths = ["tests"]
  markers = [
      "gemma4: tests requiring Gemma 4 model weights",
      "validation: compression method validation tests",
      "slow: slow tests that take significant time",
  ]
  ```

**Assertion Library:**
- pytest's built-in `assert` plus `numpy.testing.assert_allclose` for array equality (e.g., `tests/test_compression_engine.py` line 109: `np.testing.assert_allclose(x, idct(dct(x)), atol=1e-4)`).
- `pytest.approx` for floating-point tolerance: `assert m.mse == pytest.approx(4.0)` (`tests/test_loss_metrics_engine.py` lines 70-71).

**Run Commands:**
```bash
python -m pytest tests/                      # run all collected tests
python -m pytest tests/ -v --tb=short        # verbose + short traceback
python -m pytest tests/ -x --timeout=120     # stop on first failure, 120s cap
python -m pytest tests/test_config.py        # single file
python -m pytest tests/ -k "test_roundtrip"  # filter by name (pytest -k)
python tests/run_all_tests.py                # custom runner: summary JSON + optional coverage
python tests/run_all_tests.py --coverage     # runs with --cov=spectralstream
```

## Test File Organization

**Location:** Tests are stored exclusively in the **top-level `tests/` directory** — NONE are co-located with source. The package (`spectralstream/`) contains no `test_*.py` files.

**Naming:** `test_*.py` (e.g., `test_config.py`, `test_registry.py`, `test_integration.py`). A few files are benchmark scripts rather than pytest suites (`tests/benchmark.py`, `tests/benchmark_all_models.py`, `tests/benchmark_final.py`, `tests/benchmark_real.py`, `tests/benchmark_full_context.py`, `tests/run_all_tests.py`) — these use `if __name__ == "__main__"` guards and are not collected as tests.

**Structure:**
- Two styles coexist:
  1. **Module-level functions** (no classes): `tests/test_version.py`, `tests/test_config.py`. Each `def test_*` is a standalone function.
  2. **Class-based suites** (`class Test*:`): the dominant style. Each class groups related tests (`tests/test_compression_engine.py` has 19 `Test*` classes: `TestDCT`, `TestFWHT`, `TestLloydMax`, `TestWavelet`, `TestHRR`, `TestNumerical`, `TestHadamardRotator`, `TestSymAntiSym`, `TestCompressedSensing`, `TestCompressionMethods`, `TestCompressionIntelligenceEngine`, `TestCompressionProfiler`, `TestMethodSelector`, `TestErrorBudgetAllocator`, `TestEdgeCases`, `TestQualityMetrics`, `TestSerialization`, `TestConcurrentCompression`, `TestErrorPropagation`, `TestMemoryAndPerformance`, `TestIntegration`, `TestParametrized`).
- Test class names are `PascalCase` prefixed with `Test`: `TestMethodRegistry`, `TestMethodTier`, `TestSpectralMetrics`, `TestStatisticalMetrics`, `TestFullPipeline`, `TestUnifiedQuantizer`.
- Test functions are `snake_case` prefixed with `test_`: `test_default_config_creation`, `test_registry_block_int8_roundtrip`, `test_known_mse`.

## Test Structure

**Suite Organization (class-based with fixtures):**
```python
import numpy as np
import pytest
from spectralstream.compression.engine import (
    CompressionIntelligenceEngine, METHOD_REGISTRY, ...
)

@pytest.fixture
def rng():
    return np.random.RandomState(42)

@pytest.fixture
def small_tensor(rng):
    return rng.randn(16, 16).astype(np.float32)

@pytest.fixture
def engine():
    return CompressionIntelligenceEngine()

class TestCompressionMethods:
    def test_roundtrip_shape(self, small_tensor, engine):
        inst = METHOD_REGISTRY["block_int8"]
        data, meta = inst.compress(small_tensor)
        recon = inst.decompress(data, meta).reshape(small_tensor.shape)
        assert recon.shape == small_tensor.shape
```
(Pattern from `tests/test_integration.py` lines 43-78, `tests/test_loss_metrics_engine.py` lines 40-47, `tests/test_registry.py` lines 35-77.)

**Patterns:**
- Setup is done inside the test body or via fixtures; there is no explicit `tearDown` in most files — fixtures use `yield` + cleanup (`tempfile.TemporaryDirectory`, `os.unlink(tmp_path)`).
- An autouse fixture `auto_gc` forces `gc.collect()` after every test to prevent cross-test memory leaks (`tests/conftest.py` lines 25-29).
- Assertion style: direct `assert` on shapes/dtypes/values, `np.testing.assert_allclose(..., atol=...)`, and `pytest.approx(...)`.

## Mocking

**Framework:** `unittest.mock` is available; Grep across `tests/` found references to `MagicMock`, `patch(`, `monkeypatch` (76 occurrences across 17 files). However, mocking is **not** the dominant pattern — most compression tests exercise real numpy/scipy code with synthetic tensors.

**Patterns observed:**
- `monkeypatch` is imported and used in several files (`tests/test_bfloat16.py`, `tests/test_comprehensive.py`, `tests/test_compression_engine.py`, `tests/test_coverage_gaps.py`, `tests/test_dial_in_engine.py`, `tests/test_cutting_edge.py`, `tests/test_format_*.py`, `tests/test_inference_loader.py`, `tests/test_integration.py`, `tests/test_noise_aware_compressor.py`, `tests/test_unified_model_compression_engine.py`, `tests/test_unified_quantizer.py`, `tests/test_validation_gemma4.py`).
- `pytest`'s built-in `tmp_path` / `tempfile.TemporaryDirectory` is used for filesystem isolation instead of file mocking.

**What to Mock:**
- External model loaders / GGUF parsing that require real weight files (these tests are in `collect_ignore` or skip-marked — see below).
- Environment variables are set/unset manually with save-and-restore (e.g., `tests/test_config.py` `test_from_env` saves `original` and restores in `finally`).

**What NOT to Mock:**
- The compression math itself — roundtrip tests use real `compress()`/`decompress()` and assert shape/dtype/approx-equality. This is the core validation strategy.

## Fixtures and Factories

**Shared fixtures (`tests/conftest.py`):**
- `auto_gc` (autouse) — garbage collection after each test.
- `small_tensor` — `np.random.RandomState(42).randn(16,16).astype(np.float32)`, freed after test.
- `medium_tensor` — same 16x16 float32 tensor (note: currently identical to `small_tensor` in the file).
- `tiny_engine` — `CompressionIntelligenceEngine()` restricted to built-in methods only, to avoid OOM from 3000+ lazy-loaded methods; calls `engine.close()` and `gc.collect()` on teardown.

**Per-file fixtures (defined locally):**
- `rng` → `np.random.RandomState(42)` (fixed seed for reproducibility) — `tests/test_integration.py` line 44, `tests/test_loss_metrics_engine.py` line 46.
- `small_tensor` / `multi_tensor_model` (dict of synthetic named tensors mimicking a model) — `tests/test_integration.py` lines 49-72.
- `engine` → `CompressionIntelligenceEngine()` — `tests/test_integration.py` line 76, `tests/test_loss_metrics_engine.py` line 41.
- `temp_dir` → `tempfile.TemporaryDirectory()` context manager — `tests/test_integration.py` line 81.

**Factories:** No dedicated factory classes; tests construct objects inline (e.g., `SpectralStreamConfig()`, `np.random.randn(...)`). The `multi_tensor_model` fixture acts as a model-tensor factory.

## Coverage

**Requirements:** No coverage threshold is enforced (no `fail_under` in config, no `[tool.coverage]` section). Coverage is opt-in via the custom runner.

**Artifacts:**
- `.coverage` file exists at repo root (`D:\compression engine\Spectral-Stream-LLM-Engine\.coverage`, 94,208 bytes, dated 2026-07-06) — a `.coverage` SQLite data file produced by `coverage.py`.
- `tests/run_all_tests.py` `run_coverage()` invokes pytest with `--cov=<root>/spectralstream --cov-report=json --cov-report=term-missing` and writes `tests/coverage.json`, then reads `total_percent` and per-file `percent_covered`.
- `tests/test_results.json` and `tests/coverage.json` hold last run summaries.

**View coverage:**
```bash
python tests/run_all_tests.py --coverage
# or directly:
python -m pytest tests/ --cov=spectralstream --cov-report=term-missing --cov-report=json
```

## Test Types

**Unit Tests:** The majority — test individual methods/classes in isolation (e.g., `tests/test_loss_metrics_engine.py` tests each metric: `TestSpectralMetrics`, `TestStatisticalMetrics`, `TestStructuralMetrics`, `TestCompressionMetrics`). Registry roundtrip tests (`tests/test_registry.py`) are unit tests over each compression method.

**Integration Tests:** `tests/test_integration.py` (`TestFullPipeline`, `TestIntegration`) drive the whole pipeline: profile → compress → decompress → certificate, across a synthetic multi-tensor model. `tests/test_compression_engine.py` `TestIntegration` and `TestConcurrentCompression` also exercise engine-level integration.

**E2E Tests:** Not formal E2E. Files `test_gemma4.py`, `test_method_validation_on_gemma4.py`, `test_validation_gemma4.py`, `test_inference_loader.py`, `test_inference_pipeline.py`, `test_serving_api.py` require real Gemma 4 / GGUF weights and are either in `collect_ignore` or skip-marked.

**Performance/Benchmark:** `tests/benchmark*.py` are standalone scripts (not pytest-collected). `tests/test_performance.py` exists as a collected test; `tests/test_memory_streaming.py` tests streaming behavior.

## Common Patterns

**Async Testing:** No async/await tests detected — the engine is synchronous. No `pytest-asyncio` in dependencies.

**Error Testing:**
- Tuple-of-exceptions form via `pytest.raises`:
  ```python
  with pytest.raises((RuntimeError, ValueError, TypeError)):
      engine.compress_fast("not_a_tensor", name="bad")
  ```
  (`tests/test_integration.py` lines 779-782). `with pytest.raises(KeyError): reader.get_tensor("nonexistent")` (line 853).
- Config error paths assert on returned warning strings rather than raised exceptions:
  ```python
  cfg.hdc.dim = 0
  warnings = cfg.validate()
  assert "HDC dim must be positive" in warnings
  ```
  (`tests/test_config.py` lines 122-126).
- Legacy style `try/except ... assert False` still present in `tests/test_config.py` (`test_from_file_nonexistent`, `test_from_file_invalid_json`).

**Parametrized Testing:** `@pytest.mark.parametrize` is used (76 occurrences across 17 files). Examples:
```python
@pytest.mark.parametrize("n", [4, 8, 16])
def test_various_sizes(self, n):
    x = np.random.randn(n)
    np.testing.assert_allclose(x, idct(dct(x)), atol=1e-4)
```
(`tests/test_compression_engine.py` lines 106-109). Method-matrix parametrization:
```python
METHOD_PARAMS = [...]  # (name, params) pairs
@pytest.mark.parametrize("mname,params", METHOD_PARAMS)
def test_roundtrip_shape(self, mname, params, small_tensor):
    inst = METHOD_REGISTRY[mname]
    cd, meta = inst.compress(small_tensor, **params)
    recon = inst.decompress(cd, meta).reshape(small_tensor.shape)
    assert recon.shape == small_tensor.shape
```
(`tests/test_compression_engine.py` lines 286-291).

**Skipping / Collection Control (`tests/conftest.py`):**
- `collect_ignore` (lines 44-68) lists files that fail import (archive-only modules, files needing real model weights) so pytest doesn't error at collection: `test_rans_hadamard.py`, `test_sscx_format.py`, `test_supreme_quant_engine.py`, `test_unified_system.py`, archive `test_archive_*.py`, Gemma4 `test_gemma4.py`, `test_method_validation_on_gemma4.py`, `test_validation_gemma4.py`, `test_inference_pipeline.py`, `test_inference_loader.py`, `test_serving_api.py`, `test_holographic_fractal_chaos_real_weights.py`.
- `pytest_collection_modifyitems` (lines 71-174) dynamically applies `pytest.mark.skip` to: whole archive files, whole test classes referencing un-migrated modules (e.g., `TestHrrMemory`, `TestInferenceEngine`, `TestVlasovMeanFieldAttention`), Gemma4 tests needing real weights (except `test_metadata`/`test_config`), and individual API-mismatch tests (e.g., `TestUnifiedCoreUtilities::test_gibbs_softmax`, `TestQAOABitAllocator`).
- Custom markers `timeout(N)`, `gemma4`, `validation`, `slow` are registered in `pytest_configure` (lines 32-39) to silence `PytestUnknownMarkWarning`.

**Reproducibility:** Tests rely on `np.random.RandomState(42)` fixed seeds (via `rng` fixture or inline) for deterministic tensors.

## Test Catalog (tests/ directory)

| Test File | What It Tests | Key Test Classes / Functions |
|-----------|---------------|------------------------------|
| `tests/test_version.py` | `spectralstream.version` | `test_version_string`, `test_version_info`, `test_version_matches_info` |
| `tests/test_config.py` | `SpectralStreamConfig` + nested dataclasses | module-level funcs: `test_default_config_creation`, `test_validate_*`, `test_for_model_*`, `test_from_file_*`, `test_from_env`, `test_to_file_save_roundtrip` |
| `tests/test_registry.py` | `METHOD_REGISTRY`, `MethodTier` | `TestMethodRegistry`, `TestMethodTier` (registry roundtrips, tier mappings) |
| `tests/test_integration.py` | Full compress→certify pipeline | `TestFullPipeline`, fixtures `multi_tensor_model`, `engine`, `temp_dir` |
| `tests/test_loss_metrics_engine.py` | `LossMetricsIntelligenceEngine` metrics | `TestSpectralMetrics`, `TestStatisticalMetrics`, `TestStructuralMetrics`, `TestCompressionMetrics` |
| `tests/test_compression_engine.py` | Engine + math primitives + methods | 19 `Test*` classes incl. `TestDCT`, `TestFWHT`, `TestLloydMax`, `TestCompressionMethods`, `TestErrorBudgetAllocator`, `TestEdgeCases`, `TestQualityMetrics`, `TestConcurrentCompression` |
| `tests/test_comprehensive.py` | Broad engine coverage | multiple `Test*` classes |
| `tests/test_coverage_gaps.py` | Targets untested code paths | multiple `Test*` classes |
| `tests/test_cutting_edge.py` | Novel/cutting-edge methods | multiple `Test*` classes |
| `tests/test_unified_quantizer.py` | Unified quantizer + SSF blocks | `TestHierarchicalDCT`, `TestTensorTrain`, `TestVariableBitQuantizer`, `TestEntropyCoder`, `TestUnifiedQuantizer`, `TestSSFBlock`, `TestEdgeCases` |
| `tests/test_unified_model_compression_engine.py` | Model compression engine | `Test*` classes |
| `tests/test_unified_system.py` | End-to-end system (IN `collect_ignore` — not collected) | system-level |
| `tests/test_unified_cascade_engine.py` | Cascade engine | `Test*` classes |
| `tests/test_unified_compression_world_model.py` | World-model compressor | `Test*` classes |
| `tests/test_unified_engine_comprehensive.py` | Comprehensive engine | `Test*` classes |
| `tests/test_unified_method_oracle.py` | Method selection oracle | `Test*` classes |
| `tests/test_unified_streaming_pipeline.py` | Streaming pipeline | `Test*` classes |
| `tests/test_kv_cache_core.py` | KV-cache core | `Test*` classes |
| `tests/test_kv_cache_compressor.py` | KV-cache compressor | `Test*` classes |
| `tests/test_kv_cache_eviction.py` | KV-cache eviction | `Test*` classes |
| `tests/test_kv_cache_integration.py` | KV-cache integration | `Test*` classes |
| `tests/test_kv_cache_manager.py` | KV-cache manager | `Test*` classes |
| `tests/test_attention_unified.py` | Unified attention | `Test*` classes |
| `tests/test_bfloat16.py` | BF16 / `bfloat16` primitives | `Test*` classes |
| `tests/test_calibration_quantizer.py` | Calibration quantizer | `Test*` classes |
| `tests/test_certificate.py` | Compression certificate | `Test*` classes |
| `tests/test_certificate_comprehensive.py` | Comprehensive certificate (79KB) | `Test*` classes |
| `tests/test_cli.py` | CLI entrypoint | `Test*` classes |
| `tests/test_complete_system.py` | Complete system (some classes skipped) | `TestHierarchicalMPSCompressor`, `TestCompressionPipeline2000`, `TestUnifiedCoreUtilities` (partial skips) |
| `tests/test_dial_in_engine.py` | Dial-in engine | `Test*` classes |
| `tests/test_engine_orchestrator.py` | Engine orchestrator | `Test*` classes |
| `tests/test_finetune.py` | Finetune (un-skipped) | `Test*` classes |
| `tests/test_finetuning_engine.py` | Finetuning engine (un-skipped) | `Test*` classes |
| `tests/test_format_compression.py` | Format compression | `Test*` classes |
| `tests/test_format_converter.py` | Format converter | `Test*` classes |
| `tests/test_format_core.py` | Format core | `Test*` classes |
| `tests/test_format_gguf_parser.py` | GGUF parser | `Test*` classes |
| `tests/test_format_header.py` | Format header | `Test*` classes |
| `tests/test_format_index.py` | Format index | `Test*` classes |
| `tests/test_format_ssf.py` | SSF format (6 parametrize) | `Test*` classes |
| `tests/test_gemma4.py` | Gemma 4 (IN `collect_ignore`) | metadata/config only collected |
| `tests/test_holographic_fractal_chaos_real_weights.py` | Real-weights script (IN `collect_ignore`) | standalone script |
| `tests/test_inference_loader.py` | Inference loader (IN `collect_ignore`) | `Test*` classes |
| `tests/test_inference_pipeline.py` | Inference pipeline (IN `collect_ignore`) | `Test*` classes |
| `tests/test_method_categories.py` | Method categories | `Test*` classes |
| `tests/test_method_category_entropy.py` | Entropy category | `Test*` classes |
| `tests/test_method_category_functional.py` | Functional category | `Test*` classes |
| `tests/test_method_category_quantization.py` | Quantization category | `Test*` classes |
| `tests/test_method_category_spectral.py` | Spectral category | `Test*` classes |
| `tests/test_method_category_structural.py` | Structural category | `Test*` classes |
| `tests/test_method_discovery.py` | Method discovery | `Test*` classes |
| `tests/test_method_registration.py` | Method registration | `Test*` classes |
| `tests/test_method_validation_on_gemma4.py` | Validation on Gemma4 (IN `collect_ignore`) | validation |
| `tests/test_noise_aware_compressor.py` | Noise-aware compressor | `Test*` classes |
| `tests/test_performance.py` | Performance | `Test*` classes |
| `tests/test_physics_compression.py` | Physics-based compression | `Test*` classes |
| `tests/test_pipeline.py` | Pipeline (skipped — archive) | `Test*` classes |
| `tests/test_rans_hadamard.py` | RANS + Hadamard (IN `collect_ignore`) | `Test*` classes |
| `tests/test_registry.py` | Method registry | `TestMethodRegistry`, `TestMethodTier` |
| `tests/test_revolutionary_compression.py` | Revolutionary compression | `Test*` classes |
| `tests/test_security_compliance.py` | Security/compliance | `Test*` classes |
| `tests/test_selector_priority.py` | Selector priority | `Test*` classes |
| `tests/test_serving_api.py` | Serving API (IN `collect_ignore`) | `Test*` classes |
| `tests/test_sscx_format.py` | SSCX format (IN `collect_ignore`) | `Test*` classes |
| `tests/test_stacking_priority.py` | Stacking priority | `Test*` classes |
| `tests/test_supreme_quant_engine.py` | Supreme quant engine (IN `collect_ignore`) | `Test*` classes |
| `tests/test_tier_priority.py` | Tier priority | `Test*` classes |
| `tests/test_audit.py` | Audit (skipped — archive) | `Test*` classes |
| `tests/test_validation_gemma4.py` | Validation on Gemma4 (IN `collect_ignore`) | validation |
| `tests/test_archive_architecture_compressor.py` | Archive (IN `collect_ignore`) | `Test*` classes |
| `tests/test_archive_combined_pipeline.py` | Archive (IN `collect_ignore`) | `Test*` classes |
| `tests/test_archive_edge_cases.py` | Archive (IN `collect_ignore`) | `Test*` classes |
| `tests/test_archive_extreme_compression.py` | Archive (IN `collect_ignore`) | `Test*` classes |
| `tests/test_archive_extreme_compressor.py` | Archive (IN `collect_ignore`) | `Test*` classes |
| `tests/test_archive_gguf_conversion.py` | Archive (IN `collect_ignore`) | `Test*` classes |
| `tests/test_archive_intelligence_engine.py` | Archive (skipped at collection) | `Test*` classes |
| `tests/benchmark.py`, `benchmark_all_models.py`, `benchmark_final.py`, `benchmark_real.py`, `benchmark_full_context.py` | Standalone benchmark scripts (not collected) | `if __name__ == "__main__"` |
| `tests/run_all_tests.py` | Custom test runner (subprocess + coverage JSON) | `run_pytest`, `run_coverage`, `parse_pytest_output`, `main` |
| `tests/conftest.py` | Shared fixtures + collection/skip control | `auto_gc`, `small_tensor`, `medium_tensor`, `tiny_engine`, `pytest_configure`, `pytest_collection_modifyitems` |

**Relationship to `spectralstream` package:** Tests import the installed package root `spectralstream.*` (after adding repo root to `sys.path`). They target: config (`config.py`), version (`version.py`), compression engine (`compression/engine/*`), math primitives (`core/math_primitives/*`), registry (`compression/engine/_methods.py`, `method_registry.py`), certificate (`compression/certificate.py`), format I/O (`format/*`), KV-cache (`kv_cache/*`), attention (`attention/*`), quantizers (`unified_quantizer.py`), world-model (`compression/world_model/*`), finetuning (`finetuning/*`), serving (`serving/*`), and inference (`inference/*`). Many compression method files under `spectralstream/compression/methods/` and `spectralstream/compression/advanced/` are exercised indirectly through the engine/registry rather than via dedicated test files.

---

*Testing analysis: 2026-07-07*

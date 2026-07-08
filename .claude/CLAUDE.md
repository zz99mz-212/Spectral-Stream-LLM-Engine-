<!-- GSD:project-start source:PROJECT.md -->

## Project

**Spectral-Stream LLM Engine**

A pure-Python (NumPy/SciPy, no C++), CPU-targeted LLM weight & KV-cache compression engine that doubles as a local OpenAI-compatible inference server. Its thesis: instead of compressing weights as flat tensors, treat them as continuous manifolds / spectral fields / physical systems and exploit multiple redundancy sources (low-rank, spectral decay, cross-layer correlation, entropy, physics symmetries) through a cascade of independent methods.

The engine delivers **~4–6× compression vs FP32 (≈2–3× vs BF16)** on real LLM weights via its verified INT8/INT4 quantization path. A broad registry of 2,964 methods across 43 categories exists as an explicit, labeled **research catalog** — most are aspirational or unvalidated on real weights. The project's energy is split between (1) maintaining an honest, reliable compression core, and (2) actively exploring activation-based pruning, LLM neuroanatomy, per-layer importance analysis and dynamic reduction, and deeper architecture / math optimizations.

**🔬 ACTIVE R&D — NOT PRODUCTION READY.** License: AGPL-3.0.

**Core Value:** **Honest compression that actually works on real weights.** The metric that matters is byte-exact ratio + paired error — never one without the other — and every advertised number must be verifiably measured, not estimated or aspirational. The research catalog lives alongside this core, but the default advertised capability tells the truth about what works today.

### Constraints

- **Tech stack**: Pure Python (NumPy ≥1.24, SciPy ≥1.10). No C++ extensions, no GPU acceleration, no torch dependency in the core path. CPU-targeted only.
- **License**: AGPL-3.0
- **Maturity**: Research-grade. Performance is orders of magnitude below real LLM serving engines. Single-threaded (GIL), no GPU, inference ~2–3× slower than llama.cpp at best.
- **Metrics honesty**: ALL ratio/error numbers must flow through `honest_metrics.py` (byte-exact). No estimates, no per-stage products, no len(dict).
- **Format overhead**: SSF container uses 4096-byte page alignment + 256-byte header + 128-byte footer per tensor. ALWAYS report algorithmic ratio separate from overhead.
- **Windows compatibility**: The `fix/honest-metrics-windows-compat` branch exists. `signal.alarm`/`exec`-based timeouts (Linux-only) must be replaced.
- **Dependency divergence**: `pyproject.toml` and `requirements.txt` disagree. Single-source deps needed.
- **No eval baseline**: Zero perplexity or downstream task evaluations exist. This is the single biggest quality-trust gap.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Python 3.10+ (`requires-python = ">=3.10"` in `pyproject.toml`; README badge "Python 3.10+"). Source uses modern syntax: type hints with subscripted generics `list[dict]` (`spectralstream/serving/lmstudio.py:12`), `from __future__ import annotations` (`spectralstream/utils/multimodal_prompt.py:27`), f-strings, dataclasses.
- The entire engine is **pure Python** — no compiled extension is actually built or imported (see C/C++ section).
- C/C++ — `CMakeLists.txt` exists at repo root but its own header comment states all C++ SIMD targets were **removed**. No `cpp_ext/` directory, no `.so`/`.pyd`, no pybind11. Functionally unused.
- JSON — configuration, model metadata, certificates, and benchmark results are stored as `.json` files (`spectralstream/config.py`, `spectralstream/format/compression.py`, `configs/*.json`).
- HTML / CSS / JavaScript — web dashboard and chat UI templates under `spectralstream/serving/templates/` and `spectralstream/serving/static/`.
- Jinja-templated config — SQL/Modelfile-style integration configs (`configs/ollama.Modelfile`, `configs/*.json`).

## Runtime

- CPython 3.10, 3.11, 3.12 (lower bound only is pinned; no upper bound). CPU-targeted ("CPU Inference Engine" per `README.md`).
- No GPU requirement declared. `torch` (optional `ml` extra) runs on CPU by default; `vllm.json` sets `gpu_memory_utilization: 0.0`.
- pip + setuptools (build backend `setuptools.build_meta:__legacy__`).
- Lockfile: **missing** — `pyproject.toml` uses open range specifiers (`>=`), no `poetry.lock` / `pip.lock` / `Pipfile.lock` present.
- A separate `requirements.txt` exists at repo root with its *own* independent dependency set (see Discrepancies).

## Frameworks

- NumPy `>=1.24` — the foundational numerical framework. All tensor math, FFT (`numpy.fft` / pocketfft via `scipy.fft`), `einsum`, BLAS-backed matmul, vectorised SIMD-equivalent kernels. Used pervasively (300+ files).
- SciPy `>=1.10` — `scipy.linalg.svd`, `scipy.fft`, `scipy.optimize.minimize`, `scipy.integrate.solve_ivp`, `scipy.stats`, `scipy.signal.fftconvolve`, `scipy.ndimage.uniform_filter`, `scipy.spatial.distance.cdist`.
- Standard library only otherwise — `json`, `os`, `math`, `dataclasses`, `typing`, `argparse`, `http.client`, `subprocess`, `struct`, `base64`, `io`, `tempfile`, `queue`, `socket`, `ast`, `re`.
- pytest `>=7.0` (dev extra) — test runner. Configured in `pyproject.toml` `[tool.pytest.ini_options]`: `timeout = 120`, `testpaths = ["tests"]`, markers `gemma4`, `validation`, `slow`.
- pytest-timeout `>=2.0` (dev extra) — enforces the 120s per-test timeout.
- `tests/run_all_tests.py` and `tests/conftest.py` provide custom harness/collection around pytest.
- setuptools `>=68.0` — build backend.
- rich `>=13.0` (dev extra) — terminal dashboards, progress bars, tables in CLI (`spectralstream/compression/cli.py`, `spectralstream/compression/cli_dashboard.py`, `spectralstream/compression/benchmark/report_generator.py`). Imported with fallback guards (`_has_rich`).
- CMake `>=3.20` — present only as a stub; no build targets active.

## Key Dependencies

| Package | Constraint | Version area | Why it matters |
|---------|-----------|--------------|----------------|
| `numpy` | `>=1.24` | Core | Foundation of all tensor ops, FFTs (pocketfft), einsum/BLAS matmul. The "SIMD" kernels are pure-NumPy vectorised ops (`spectralstream/utils/simd_backend.py`). |
| `scipy` | `>=1.10` | Core | SVD (TT-SVD, structured/tensor decomposition), ODE/PDE integration (`solve_ivp`), optimization (`minimize`), KDE, signal filtering (`fftconvolve`, `uniform_filter`), spatial distance. Drives the 5-stage cascade and physics methods. |
| `psutil` | `>=5.9` | Core | Hardware introspection: RAM estimate (`spectralstream/config.py:104`), CPU core counts, process monitoring in benchmarks (`spectralstream/compression/benchmark/benchmark_runner.py`). Imported defensively with `/proc/meminfo` fallback. |
| `safetensors` | `>=0.4` | Core | Reads model weight tensors from `.safetensors` files (`safe_open`) — primary on-disk model format (`scripts/compress_gemma4.py`, `spectralstream/compression/world_model/dial_in_engine.py`). |
| `zstandard` | `>=0.22` | Core | Native ZSTD compression used by the `.ssf` Serialized Spectral Format (`spectralstream/format/compression.py`, imported as `import zstandard as zstd`). |
| Extra | Package | Constraint | Purpose |
|-------|---------|-----------|---------|
| `dev` | `pytest` | `>=7.0` | Test runner |
| `dev` | `pytest-timeout` | `>=2.0` | Per-test timeout enforcement |
| `dev` | `rich` | `>=13.0` | Terminal dashboards / progress |
| `web` | `fastapi` | `>=0.100` | HTTP API server (`spectralstream/serving/api/`, `spectralstream/serving/unified_server.py`) |
| `web` | `uvicorn` | `>=0.22` | ASGI server that launches the FastAPI app (`spectralstream/serving/unified_server.py`) |
| `web` | `jinja2` | `>=3.1` | Template engine via `fastapi.templating.Jinja2Templates` (HTML dashboards) |
| `web` | `python-multipart` | `>=0.0.6` | `multipart/form-data` file upload parsing for the API |
| `web` | `pydantic` | `>=2.0` | Request/response models (`BaseModel`, `Field`) in `spectralstream/serving/api/_*.py` |
| `gguf` | `gguf` | `>=0.6` | Reads `.gguf` model files via `GGUFReader` (`spectralstream/utils/tokenizer_engine.py`, benchmarks) |
| `ml` | `scikit-learn` | `>=1.3` | ML utilities (declared; limited in-package use) |
| `ml` | `ml-dtypes` | `>=0.3` | `bfloat16`/`float8` dtype support (declared) |
| `ml` | `torch` | `>=2.0` | Optional tensor backend; imported lazily only in `spectralstream/core/validation.py` (`import torch` inside a function) |
| `finetune` | `datasets` | `>=2.0` | HF datasets for fine-tuning (`spectralstream/finetuning/*`) |

- `scikit-learn`, `ml-dtypes`, `datasets` — declared in extras but no `import sklearn` / `import ml_dtypes` / `import datasets` found in the package (functionality lives behind extras that are not exercised by the committed code).
- `tree-sitter*` (`requirements.txt`) — `tree-sitter>=0.23.0` plus 9 language grammars are listed in `requirements.txt`, but **no `import tree_sitter` exists** in any `.py` file. Only `AGENTS.md` prose references "tree-sitter AST for 10 languages" (the `/index` intelligence feature). Effectively dead/aspirational in the committed code.
- `requests` — optionally imported in `spectralstream/utils/multimodal_prompt.py:68` (`HAS_REQUESTS` guard). Not declared anywhere.
- `matplotlib` — optionally imported (with `Agg` backend) in `spectralstream/utils/multimodal_prompt.py:1283` for plots. Not declared anywhere.
- `openai` — used in the sample integration script `configs/openai_sdk.py:1` (`from openai import OpenAI`). Not declared.

## Configuration

- Single central dataclass config: `spectralstream/config.py` → `SpectralStreamConfig` (`@dataclass`) composed of sub-configs: `HDCConfig`, `SpectralConfig`, `ConfidenceGateConfig`, `BlockEmissionConfig`, `OnlineLearningConfig`, `ServerConfig`, `MonitoringConfig`, `PersistenceConfig`, `HardwareConfig`.
- Load precedence: **environment variables > JSON file > dataclass defaults**.
- Per-model overrides in `_MODEL_OVERRIDES` (`config.py:144`) — gemma-4-2b/9b/27b, llama-3.x, qwen2.5-*, deepseek-*.
- Per-hardware auto-tuning `for_hardware()` (`config.py:314`) based on RAM/CPU.
- Validation via `.validate()` (`config.py:338`).
- `pyproject.toml` is the single build manifest. Build system `setuptools>=68.0`, backend `setuptools.build_meta:__legacy__`.
- `[tool.setuptools.packages.find]` → `include = ["spectralstream*"]`.
- `[tool.pytest.ini_options]` → `timeout=120`, `testpaths=["tests"]`, markers `gemma4`/`validation`/`slow`.
- Installed editable package metadata present at `spectralstream.egg-info/` (`PKG-INFO`, `SOURCES.txt`, `requires.txt`, `top_level.txt` = `spectralstream`).

## Platform Requirements

- Python 3.10+.
- pip-installable; core install needs only numpy/scipy/psutil/safetensors/zstandard (all pure-Python or wheeled — no compiler required because the C++ extension was removed).
- `psutil` has a `/proc/meminfo` Linux fallback path (`config.py:110`); on Windows the psutil import is used directly.
- Optional features pulled via extras: `pip install spectralstream[web,gguf,dev]`.
- CPU-only inference target (no GPU required). `vllm.json` sets `gpu_memory_utilization: 0.0`.
- Serves as an OpenAI-compatible / LM Studio / vLLM / Ollama backend (see `INTEGRATIONS.md`) over local HTTP (`127.0.0.1` default port `1234`, `config.py:72`).
- State persisted to `~/.spectralstream/state/` (`config.py:90`).
- Models (`.safetensors` / `.gguf`) loaded from local filesystem (`models/` is git-ignored per `.gitignore`).

## Discrepancies & Notes

- **Version mismatch:** `pyproject.toml` declares `version = "2.0.0"` while `spectralstream/version.py` reports `__version__ = "1.0.0"`.
- **Two divergent manifests:** `pyproject.toml` (canonical, used by build) and `requirements.txt` (a separate, independent set dominated by unused `tree-sitter` packages). They do not stay in sync.
- **No compiled extension:** Despite the project narrative ("All SIMD via NumPy"), `CMakeLists.txt` explicitly states C++ SIMD kernels were removed and no `cpp_ext/` exists. There is no native code path.
- **YAML claimed but unsupported:** `spectralstream/config.py` docstring says "YAML/JSON config file" but only `json.load` is implemented; no `PyYAML` dependency.
- **Undeclared optional deps:** `requests`, `matplotlib`, `openai` are used/referenced but absent from `pyproject.toml` dependencies/extras, so `pip install spectralstream` without extras will fail when those code paths execute.

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Main package modules use `snake_case` single-word or compound names: `config.py`, `orchestrator.py`, `version.py`, `logging_config.py`, `audit.py`, `model_compressor.py`.
- Test files use the `test_*.py` prefix, always in the top-level `tests/` directory (never co-located with source): `tests/test_config.py`, `tests/test_registry.py`, `tests/test_integration.py`.
- Sub-implementation modules are heavily prefixed with a single underscore `_` to mark them "private"/internal: `spectralstream/compression/engine/_orchestrator.py`, `spectralstream/compression/engine/_selector.py`, `spectralstream/compression/engine/_methods.py`.
- Compression method classes are kept in files named after the algorithm, often wrapped in `_`-prefixed helpers: `spectralstream/compression/methods/spectral/dct_spectral.py`, `spectralstream/compression/methods/quantization/nf4_quant.py`.
- `snake_case` for functions and methods: `setup_logging`, `get_logger`, `profile_tensor`, `compress_fast`, `from_file`, `to_dict`, `for_model`.
- `_camelCase` private helper modules under compression methods use leading-underscore module names but expose `CamelCase` classes.
- Dunder methods are used per Python convention: `__version__`, `__version_info__`, `__all__`.
- `snake_case` for locals and attributes: `spectral_rank`, `kv_compression`, `num_lsh_tables`, `lsh_bits_per_key`.
- Module-level constants use `UPPER_SNAKE_CASE` where present, e.g. `METHOD_REGISTRY`, `METHOD_TIER_MAP`, `CATEGORY_TIER_MAP`, `METHOD_PARAMS` (in `tests/test_compression_engine.py`).
- `PascalCase` for classes and enums: `CompressionIntelligenceEngine`, `CompressionConfig`, `MethodTier` (enum), `MethodDiscovery`, `DynamicIntelligenceSelector`, `ErrorBudgetAllocator`, `CompressionLogger`, `PerformanceTimer`, `AuditLogger`, `SpectralStreamConfig`.
- Configuration is modeled with `@dataclass` classes: `HDCConfig`, `SpectralConfig`, `ConfidenceGateConfig`, `ServerConfig`, `HardwareConfig`, `PersistenceConfig` (in `spectralstream/config.py`).
- Compression method instances follow a `<Verb><Type>` naming convention exposed in the registry: `block_int8`, `hadamard_int4`, `svd_compress`, `dct_spectral`, `tensor_train`, `fwht_compress` (see `spectralstream/compression/engine/_methods.py`).
- Result/value objects use `PascalCase`: `CompressedTensor`, `TensorProfile`, `CompressionReport`, `CompressionCertificate`.

## Code Style

- No enforced formatter detected. `pyproject.toml` contains only `[build-system]`, `[project]`, `[tool.setuptools.packages.find]`, and `[tool.pytest.ini_options]`. There is no `[tool.black]`, `[tool.ruff]`, `[tool.isort]`, `[tool.pylint]`, or `[tool.mypy]` section.
- The project root has no `setup.cfg`, `.flake8`, `ruff.toml`, `.pre-commit-config.yaml`, or `.editorconfig` (Glob for `**/*.cfg` and `**/*.ini` returned no files; only `pyproject.toml` and `CMakeLists.txt` exist).
- Code consistently uses 4-space indentation and a `from __future__ import annotations` header at the top of nearly every module and test file (observed in `spectralstream/logging_config.py`, `spectralstream/config.py`, `spectralstream/version.py`, and all sampled test files).
- Visual section dividers use box-drawing ASCII banners: `════════════════` used in `spectralstream/logging_config.py` and throughout test files (`tests/test_integration.py`, `tests/test_loss_metrics_engine.py`, `tests/test_compression_engine.py`).
- No linter configured (no ruff/flake8/pylint/mypy). `dev` optional dependency is `pytest>=7.0`, `pytest-timeout>=2.0`, `rich>=13.0` (`pyproject.toml` `[project.optional-dependencies]`).
- Not enforced by config; source files observed keep well under 100 columns.

## Import Organization

- No custom path aliases or `src` layout. Package root is `spectralstream/` and tests import via the installed/root package `spectralstream.*`.

## Error Handling

- Errors are raised with built-in exceptions using descriptive messages: `ValueError` (config/file errors), `KeyError` (missing tensor/key), `RuntimeError` / `TypeError` (bad inputs), `NotImplementedError`, `IndexError`, `FileNotFoundError`, `AssertionError`. Grep across `spectralstream/` found 281 `raise` statements in 116 files.
- Config validation does NOT raise; it returns a `list[str]` of warning strings. `SpectralStreamConfig.validate()` returns `[]` when valid and appends human-readable messages otherwise (`spectralstream/config.py`; tests assert `"HDC dim must be positive" in warnings`).
- Tests assert errors via `pytest.raises`, often with a tuple of acceptable exceptions:
- Some older test files use the `try/except ... pass` style instead of `pytest.raises`:
- File I/O is wrapped in `try/finally` with `os.unlink(tmp_path)` cleanup in tests (`tests/test_config.py` lines 321-329).

## Logging

- `setup_logging(level="INFO", json_output=False, log_file=None)` — configures root logger `logging.getLogger("spectralstream")`. Idempotent via module-level `_CONFIGURED` flag.
- `get_logger(name)` — returns a child logger `spectralstream.<name>`, auto-calling `setup_logging()` if not configured.
- Two formatters: `_JSONFormatter` (one JSON object per line, with `timestamp`, `level`, `logger`, `message`, `module`, `function`, `line`, plus optional `exc_info` and `data`), and `_HumanFormatter` (colored console output, ANSI colors for DEBUG/INFO/WARNING/ERROR/CRITICAL, color only when `sys.stderr.isatty()`).
- `CompressionLogger` — records compression/decompression events with metrics (`ratio`, `error`, `time_s`) into a bounded `deque` (maxlen 10000), thread-safe via `threading.Lock`.
- `PerformanceTimer` — context manager (`with PerformanceTimer("quantize") as timer:`) timing code blocks via `time.perf_counter()`; `timed_operation(label, logger)` shorthand yields elapsed seconds.
- `AuditLogger` — audit trail of compression decisions/overrides into a bounded `deque` (maxlen 100_000), `export_trail(path)` writes JSON.
- Child loggers are obtained via `get_logger("compression")`, `get_logger("timer")`, `get_logger("audit")` rather than `logging.getLogger(__name__)`.
- Logs go to `sys.stderr` (console) and optionally a JSON-lines file via `log_file`.
- Messages use `%`-style lazy formatting: `self.logger.info("Compressed %s with %s: ratio=%.2fx, error=%.6f", tensor_name, method, ratio, error, time_s)`.

## Comments

- Module docstrings are present at the top of most files describing purpose and exported components (e.g., `spectralstream/logging_config.py` lines 1-12, `spectralstream/config.py` lines 1-17, `spectralstream/version.py`).
- Inline comments are used sparingly to explain non-obvious math/physics steps (e.g., `# Large enough to trigger power iteration` in `tests/test_loss_metrics_engine.py` line 107).
- Section headers in large files use ASCII banners for logical grouping.
- NumPy-style docstrings with `Parameters` / `Returns` sections are used on public functions (e.g., `setup_logging` in `spectralstream/logging_config.py` lines 96-106, `pytest_collection_modifyitems` in `tests/conftest.py` lines 78-82).
- Test files and small helper functions often omit docstrings; class-level docstrings summarize the test group (e.g., `class TestFullPipeline: """Tests the complete compression pipeline...""` in `tests/test_integration.py` line 92).

## Function Design

- Functions are generally small-to-medium. Logging/utility functions (`get_logger`, `setup_logging`) are compact. Some test helper/engine methods are larger, but no single function exceeds a few hundred lines in sampled files.
- Compression methods expose a consistent `compress(tensor, **params) -> (bytes, dict)` / `decompress(data, meta) -> np.ndarray` interface. Test asserts: `isinstance(result, tuple)` with `len == 2`, `result[0]` is `bytes`, `result[1]` is `dict` (`tests/test_registry.py` lines 129-136).
- Type hints are used throughout (`tensor: np.ndarray`, `name: str = "compression"`, `level: str = "INFO"`).
- `from __future__ import annotations` allows full annotations without runtime cost.
- Config classes use `@dataclass` fields with defaults; e.g. `HDCConfig(dim: int = 10000, ngram_order: int = 4, ...)`.
- Methods return tuples for multi-value results (`(data, meta)`, `(data, meta, ratio, error)`).
- Validation returns `list[str]` (empty = valid).
- Properties like `ratio`, `error`, `elapsed` are returned as floats; rounding to fixed precision is done at log time (`round(ratio, 4)`).

## Module Design

- `__all__` is declared in 103 of the package's Python files (Grep count), listing public names. Examples:
- The package `__init__.py` re-exports the top-level orchestrator, logging tools, audit tools, and the supreme quant engine. Lower-level engine/methods are imported directly from submodules by tests (`from spectralstream.compression.engine import ...`).
- `__init__.py` files act as barrels in subpackages (e.g., `spectralstream/compression/engine/__init__.py` exports `CompressionIntelligenceEngine`, `METHOD_REGISTRY`, etc.). Private sub-implementations live in `_`-prefixed sibling modules imported by the barrel.
- Present at the top of virtually every `.py` file (source and test). Treat as mandatory for new files.

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

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

- **Method Registry + Enum pattern:** Every method is a class with `name`/`category`/`compress()`/`decompress()`. Names map to a `CompressionMethod` enum (`registry/enum.py`) and a `MethodMetadata` record (`registry/registry.py`). Registration is automatic via `_register_all()` (`registry/registration.py:54`).
- **Tiered priority:** Tier is derived from method *category* (`method_tiers.py`), not a manual map — Tier1 (decomposition/spectral/tensor-network/functional, score 10) → Tier5 (quantization, score 0.3). The selector tries Tier-1 first and cascades down when the error budget isn't met.
- **Lazy loading:** `METHOD_CLASSES` is a `_MethodClassesDict` that triggers `_load_extra()` (heavy breakthrough/massive/novel sections) only on first access (`methods/__init__.py:608`). `ALL_METHODS` / `LazyMethodDict` defer instantiation similarly. This keeps the import footprint and RAM low (only ~10 built-ins pre-loaded).
- **Honest-metrics mandate:** All ratios/errors MUST flow through `honest_metrics.py` (`serialized_nbytes:25`, `end_to_end_error:83`, `dual_ratio:133`) — never `len(dict)` or products of per-stage estimates (see Error Handling).
- **BF16-aware I/O:** All compress/decompress methods accept `uint16` BF16 tensors, convert to float32 for arithmetic, and flag via `_input_was_bf16` (`engine/_methods.py:38`, `_dtype_utils.py`).
- **Memory-bounded:** Chunked/memory-mapped/streaming compressors kick in above a memory budget (`compress_within_budget:376`, `chunked_compressor.py`, `memory_mapped_engine.py`, `streaming_pipeline.py`).

## Layers

- Purpose: User-facing commands and run harnesses.
- Location: `spectralstream/compression/cli.py`
- Contains: `build_parser:2701`, `cmd_compress:601`, `cmd_validate:1816`, `cmd_benchmark:2154`, etc.
- Depends on: `engine.CompressionIntelligenceEngine`, `format.reader`, `honest_metrics`.
- Used by: End users, `tests/`, `scripts/`.
- Purpose: Profile → allocate → select → compress → validate loop.
- Location: `spectralstream/compression/engine/`
- Contains: `CompressionIntelligenceEngine`, `LazyMethodDict`, profiler/allocator/selector.
- Depends on: registry, methods, honest_metrics, format.
- Used by: CLI, scripts, inference.
- Purpose: Discover, catalog, and tier all methods; pick the best per tensor.
- Depends on: `methods/`, `core/math_primitives`.
- Used by: Orchestrator.
- Purpose: Individual `compress()`/`decompress()` implementations across categories.
- Location: `spectralstream/compression/methods/{decomposition,spectral,structural,entropy,functional,physics,quantization,lossless,hybrid,novel,...}/`
- Depends on: `core/math_primitives`.
- Used by: Orchestrator, cascade.
- Purpose: A specific method that chains stage helpers on the residual of the previous.
- Depends on: `core/math_primitives`, `_dtype_utils`, `honest_metrics`.
- Used by: Orchestrator (`cascade_5stage` method), `wave4_pipeline.py`.
- Purpose: SSF binary I/O and the canonical math primitives.
- Used by: Every layer above.
- Purpose: Decompress SSF at inference time and run the model forward pass.
- Depends on: format, compression.

## Data Flow

### Primary Compression Path (engine)

### 5-Stage Cascade Path (per tensor)

### Metrics Flow

- `honest_metrics.serialized_nbytes(payload)` (`:25`) → true byte cost (handles bytes/ndarray/dict/list/scalars recursively).
- `end_to_end_error(orig, recon)` (`:83`) → `ErrorMetrics(rel_mse, cosine_sim, max_abs, snr_db)`.
- `dual_ratio(elements, payload)` (`:133`) → `ratio_vs_fp32`, `ratio_vs_bf16`.
- `wave4_pipeline.run_5stage()` (`:108`) runs cascade, pickles payload, computes `end_to_end_error` + `dual_ratio`, writes `wave4_results.json`.

## Key Abstractions

## Entry Points

## Architectural Constraints

- **Threading:** Single-threaded Python (NumPy/GIL). `InferencePipeline` documents "thread-safe generation (re-entrant via per-call state)" (`inference/pipeline.py:76`), and `kv_cache` has a "Thread-safe" note, but the compression engine itself has no explicit threading — parallelism comes from `QuantumCascadeEngine` (process/pool-based "superposition" of method trials, `quantum_cascade.py`) and `ParallelCompressor` (`engine/parallel_compressor.py`).
- **Global state:** `SpectralStreamConfig` is instantiated per-call (no module-level singleton). `MethodRegistry._methods` is class-level (registry/registry.py:12) — a shared mutable global dictionary populated at import by `_register_all()`. `_METHOD_TIER_MAP_CACHE` (method_tiers.py:140) and `_EXTRA_LOADED` flag (methods/__init__.py:419) are module-level globals controlling lazy registration.
- **Circular imports:** Deliberately broken via lazy `__getattr__` / deferred imports. `method_tiers.py` uses module-level `__getattr__` (`:155`) to avoid `method_tiers → method_discovery → method_tiers`. `engine/__init__.py` re-exports everything, so importers should import submodules directly or accept the heavy import. `METHOD_CLASSES` uses `_load_extra()` lazy trigger rather than importing heavy modules at module top.
- **dtypes:** Storage dtype is auto-detected and encoded into metadata (`_dtype_utils.encode_dtype_code`); BF16 carried as `uint16`. All arithmetic cast to float64/float32.
- **Determinism:** RNG seeds fixed (`np.random.RandomState(42)` in cascade `_siren_fit_2d:467` and `_detect_flat_spectrum:528`) for reproducibility.

## Anti-Patterns

### 5-Stage cascade only wires 2 stages

### Fabricated compression ratios (historical, now guarded)

### Module-level re-export bloat

### Redundant method representations

## Error Handling

- Methods return `(payload_bytes, metadata_dict)`; metadata carries `_input_was_bf16` and method name.
- Numeric fallbacks: `_randomized_svd` (`_methods.py:69`) falls back to `np.eye`/`np.ones` on `LinAlgError`.
- `_detect_flat_spectrum` (`:514`) catches all exceptions and returns `False` (safe default).
- CLI validates paths against traversal (`cli.py:_validate_input_path:81`, regex `:78`).
- Honest error reporting via `ErrorMetrics` (rel_mse, cosine_sim, max_abs, snr_db) — never a single fabricated scalar.

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->

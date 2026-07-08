# Technology Stack

**Analysis Date:** 2026-07-07

## Languages

**Primary:**
- Python 3.10+ (`requires-python = ">=3.10"` in `pyproject.toml`; README badge "Python 3.10+"). Source uses modern syntax: type hints with subscripted generics `list[dict]` (`spectralstream/serving/lmstudio.py:12`), `from __future__ import annotations` (`spectralstream/utils/multimodal_prompt.py:27`), f-strings, dataclasses.
- The entire engine is **pure Python** — no compiled extension is actually built or imported (see C/C++ section).

**Secondary:**
- C/C++ — `CMakeLists.txt` exists at repo root but its own header comment states all C++ SIMD targets were **removed**. No `cpp_ext/` directory, no `.so`/`.pyd`, no pybind11. Functionally unused.
- JSON — configuration, model metadata, certificates, and benchmark results are stored as `.json` files (`spectralstream/config.py`, `spectralstream/format/compression.py`, `configs/*.json`).
- HTML / CSS / JavaScript — web dashboard and chat UI templates under `spectralstream/serving/templates/` and `spectralstream/serving/static/`.
- Jinja-templated config — SQL/Modelfile-style integration configs (`configs/ollama.Modelfile`, `configs/*.json`).

## Runtime

**Environment:**
- CPython 3.10, 3.11, 3.12 (lower bound only is pinned; no upper bound). CPU-targeted ("CPU Inference Engine" per `README.md`).
- No GPU requirement declared. `torch` (optional `ml` extra) runs on CPU by default; `vllm.json` sets `gpu_memory_utilization: 0.0`.

**Package Manager:**
- pip + setuptools (build backend `setuptools.build_meta:__legacy__`).
- Lockfile: **missing** — `pyproject.toml` uses open range specifiers (`>=`), no `poetry.lock` / `pip.lock` / `Pipfile.lock` present.
- A separate `requirements.txt` exists at repo root with its *own* independent dependency set (see Discrepancies).

## Frameworks

**Core:**
- NumPy `>=1.24` — the foundational numerical framework. All tensor math, FFT (`numpy.fft` / pocketfft via `scipy.fft`), `einsum`, BLAS-backed matmul, vectorised SIMD-equivalent kernels. Used pervasively (300+ files).
- SciPy `>=1.10` — `scipy.linalg.svd`, `scipy.fft`, `scipy.optimize.minimize`, `scipy.integrate.solve_ivp`, `scipy.stats`, `scipy.signal.fftconvolve`, `scipy.ndimage.uniform_filter`, `scipy.spatial.distance.cdist`.
- Standard library only otherwise — `json`, `os`, `math`, `dataclasses`, `typing`, `argparse`, `http.client`, `subprocess`, `struct`, `base64`, `io`, `tempfile`, `queue`, `socket`, `ast`, `re`.

**Testing:**
- pytest `>=7.0` (dev extra) — test runner. Configured in `pyproject.toml` `[tool.pytest.ini_options]`: `timeout = 120`, `testpaths = ["tests"]`, markers `gemma4`, `validation`, `slow`.
- pytest-timeout `>=2.0` (dev extra) — enforces the 120s per-test timeout.
- `tests/run_all_tests.py` and `tests/conftest.py` provide custom harness/collection around pytest.

**Build/Dev:**
- setuptools `>=68.0` — build backend.
- rich `>=13.0` (dev extra) — terminal dashboards, progress bars, tables in CLI (`spectralstream/compression/cli.py`, `spectralstream/compression/cli_dashboard.py`, `spectralstream/compression/benchmark/report_generator.py`). Imported with fallback guards (`_has_rich`).
- CMake `>=3.20` — present only as a stub; no build targets active.

## Key Dependencies

**Critical (declared runtime / `dependencies` in `pyproject.toml`):**

| Package | Constraint | Version area | Why it matters |
|---------|-----------|--------------|----------------|
| `numpy` | `>=1.24` | Core | Foundation of all tensor ops, FFTs (pocketfft), einsum/BLAS matmul. The "SIMD" kernels are pure-NumPy vectorised ops (`spectralstream/utils/simd_backend.py`). |
| `scipy` | `>=1.10` | Core | SVD (TT-SVD, structured/tensor decomposition), ODE/PDE integration (`solve_ivp`), optimization (`minimize`), KDE, signal filtering (`fftconvolve`, `uniform_filter`), spatial distance. Drives the 5-stage cascade and physics methods. |
| `psutil` | `>=5.9` | Core | Hardware introspection: RAM estimate (`spectralstream/config.py:104`), CPU core counts, process monitoring in benchmarks (`spectralstream/compression/benchmark/benchmark_runner.py`). Imported defensively with `/proc/meminfo` fallback. |
| `safetensors` | `>=0.4` | Core | Reads model weight tensors from `.safetensors` files (`safe_open`) — primary on-disk model format (`scripts/compress_gemma4.py`, `spectralstream/compression/world_model/dial_in_engine.py`). |
| `zstandard` | `>=0.22` | Core | Native ZSTD compression used by the `.ssf` Serialized Spectral Format (`spectralstream/format/compression.py`, imported as `import zstandard as zstd`). |

**Infrastructure / Optional extras (`[project.optional-dependencies]`):**

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

**Dependencies declared but NOT imported anywhere in `spectralstream/` or `tests/`:**
- `scikit-learn`, `ml-dtypes`, `datasets` — declared in extras but no `import sklearn` / `import ml_dtypes` / `import datasets` found in the package (functionality lives behind extras that are not exercised by the committed code).
- `tree-sitter*` (`requirements.txt`) — `tree-sitter>=0.23.0` plus 9 language grammars are listed in `requirements.txt`, but **no `import tree_sitter` exists** in any `.py` file. Only `AGENTS.md` prose references "tree-sitter AST for 10 languages" (the `/index` intelligence feature). Effectively dead/aspirational in the committed code.

**Optional imports present in code but NOT declared in `pyproject.toml`:**
- `requests` — optionally imported in `spectralstream/utils/multimodal_prompt.py:68` (`HAS_REQUESTS` guard). Not declared anywhere.
- `matplotlib` — optionally imported (with `Agg` backend) in `spectralstream/utils/multimodal_prompt.py:1283` for plots. Not declared anywhere.
- `openai` — used in the sample integration script `configs/openai_sdk.py:1` (`from openai import OpenAI`). Not declared.

## Configuration

**Environment:**
- Single central dataclass config: `spectralstream/config.py` → `SpectralStreamConfig` (`@dataclass`) composed of sub-configs: `HDCConfig`, `SpectralConfig`, `ConfidenceGateConfig`, `BlockEmissionConfig`, `OnlineLearningConfig`, `ServerConfig`, `MonitoringConfig`, `PersistenceConfig`, `HardwareConfig`.
- Load precedence: **environment variables > JSON file > dataclass defaults**.
  - `SpectralStreamConfig.load(path=None)` (`config.py:236`) and `.from_env()` (`:222`) and `.from_file()` (`:210`).
  - Env vars use `SS_` prefix mapped via `_ENV_MAP` (`config.py:120`) — e.g. `SS_HDC_DIM`, `SS_KV_COMPRESSION`, `SS_PORT`, `SS_STATE_DIR`.
  - JSON file via `from_file` using `json.load` (module docstring mentions YAML but **only JSON is implemented**; no YAML parser dependency present).
- Per-model overrides in `_MODEL_OVERRIDES` (`config.py:144`) — gemma-4-2b/9b/27b, llama-3.x, qwen2.5-*, deepseek-*.
- Per-hardware auto-tuning `for_hardware()` (`config.py:314`) based on RAM/CPU.
- Validation via `.validate()` (`config.py:338`).

**Build:**
- `pyproject.toml` is the single build manifest. Build system `setuptools>=68.0`, backend `setuptools.build_meta:__legacy__`.
- `[tool.setuptools.packages.find]` → `include = ["spectralstream*"]`.
- `[tool.pytest.ini_options]` → `timeout=120`, `testpaths=["tests"]`, markers `gemma4`/`validation`/`slow`.
- Installed editable package metadata present at `spectralstream.egg-info/` (`PKG-INFO`, `SOURCES.txt`, `requires.txt`, `top_level.txt` = `spectralstream`).

## Platform Requirements

**Development:**
- Python 3.10+.
- pip-installable; core install needs only numpy/scipy/psutil/safetensors/zstandard (all pure-Python or wheeled — no compiler required because the C++ extension was removed).
- `psutil` has a `/proc/meminfo` Linux fallback path (`config.py:110`); on Windows the psutil import is used directly.
- Optional features pulled via extras: `pip install spectralstream[web,gguf,dev]`.

**Production:**
- CPU-only inference target (no GPU required). `vllm.json` sets `gpu_memory_utilization: 0.0`.
- Serves as an OpenAI-compatible / LM Studio / vLLM / Ollama backend (see `INTEGRATIONS.md`) over local HTTP (`127.0.0.1` default port `1234`, `config.py:72`).
- State persisted to `~/.spectralstream/state/` (`config.py:90`).
- Models (`.safetensors` / `.gguf`) loaded from local filesystem (`models/` is git-ignored per `.gitignore`).

---

## Discrepancies & Notes

- **Version mismatch:** `pyproject.toml` declares `version = "2.0.0"` while `spectralstream/version.py` reports `__version__ = "1.0.0"`.
- **Two divergent manifests:** `pyproject.toml` (canonical, used by build) and `requirements.txt` (a separate, independent set dominated by unused `tree-sitter` packages). They do not stay in sync.
- **No compiled extension:** Despite the project narrative ("All SIMD via NumPy"), `CMakeLists.txt` explicitly states C++ SIMD kernels were removed and no `cpp_ext/` exists. There is no native code path.
- **YAML claimed but unsupported:** `spectralstream/config.py` docstring says "YAML/JSON config file" but only `json.load` is implemented; no `PyYAML` dependency.
- **Undeclared optional deps:** `requests`, `matplotlib`, `openai` are used/referenced but absent from `pyproject.toml` dependencies/extras, so `pip install spectralstream` without extras will fail when those code paths execute.

*Stack analysis: 2026-07-07*

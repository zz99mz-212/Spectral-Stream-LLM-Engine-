# Coding Conventions

**Analysis Date:** 2026-07-07

## Naming Patterns

**Files:**
- Main package modules use `snake_case` single-word or compound names: `config.py`, `orchestrator.py`, `version.py`, `logging_config.py`, `audit.py`, `model_compressor.py`.
- Test files use the `test_*.py` prefix, always in the top-level `tests/` directory (never co-located with source): `tests/test_config.py`, `tests/test_registry.py`, `tests/test_integration.py`.
- Sub-implementation modules are heavily prefixed with a single underscore `_` to mark them "private"/internal: `spectralstream/compression/engine/_orchestrator.py`, `spectralstream/compression/engine/_selector.py`, `spectralstream/compression/engine/_methods.py`.
- Compression method classes are kept in files named after the algorithm, often wrapped in `_`-prefixed helpers: `spectralstream/compression/methods/spectral/dct_spectral.py`, `spectralstream/compression/methods/quantization/nf4_quant.py`.

**Functions:**
- `snake_case` for functions and methods: `setup_logging`, `get_logger`, `profile_tensor`, `compress_fast`, `from_file`, `to_dict`, `for_model`.
- `_camelCase` private helper modules under compression methods use leading-underscore module names but expose `CamelCase` classes.
- Dunder methods are used per Python convention: `__version__`, `__version_info__`, `__all__`.

**Variables:**
- `snake_case` for locals and attributes: `spectral_rank`, `kv_compression`, `num_lsh_tables`, `lsh_bits_per_key`.
- Module-level constants use `UPPER_SNAKE_CASE` where present, e.g. `METHOD_REGISTRY`, `METHOD_TIER_MAP`, `CATEGORY_TIER_MAP`, `METHOD_PARAMS` (in `tests/test_compression_engine.py`).

**Types / Classes:**
- `PascalCase` for classes and enums: `CompressionIntelligenceEngine`, `CompressionConfig`, `MethodTier` (enum), `MethodDiscovery`, `DynamicIntelligenceSelector`, `ErrorBudgetAllocator`, `CompressionLogger`, `PerformanceTimer`, `AuditLogger`, `SpectralStreamConfig`.
- Configuration is modeled with `@dataclass` classes: `HDCConfig`, `SpectralConfig`, `ConfidenceGateConfig`, `ServerConfig`, `HardwareConfig`, `PersistenceConfig` (in `spectralstream/config.py`).
- Compression method instances follow a `<Verb><Type>` naming convention exposed in the registry: `block_int8`, `hadamard_int4`, `svd_compress`, `dct_spectral`, `tensor_train`, `fwht_compress` (see `spectralstream/compression/engine/_methods.py`).
- Result/value objects use `PascalCase`: `CompressedTensor`, `TensorProfile`, `CompressionReport`, `CompressionCertificate`.

## Code Style

**Formatting:**
- No enforced formatter detected. `pyproject.toml` contains only `[build-system]`, `[project]`, `[tool.setuptools.packages.find]`, and `[tool.pytest.ini_options]`. There is no `[tool.black]`, `[tool.ruff]`, `[tool.isort]`, `[tool.pylint]`, or `[tool.mypy]` section.
- The project root has no `setup.cfg`, `.flake8`, `ruff.toml`, `.pre-commit-config.yaml`, or `.editorconfig` (Glob for `**/*.cfg` and `**/*.ini` returned no files; only `pyproject.toml` and `CMakeLists.txt` exist).
- Code consistently uses 4-space indentation and a `from __future__ import annotations` header at the top of nearly every module and test file (observed in `spectralstream/logging_config.py`, `spectralstream/config.py`, `spectralstream/version.py`, and all sampled test files).
- Visual section dividers use box-drawing ASCII banners: `════════════════` used in `spectralstream/logging_config.py` and throughout test files (`tests/test_integration.py`, `tests/test_loss_metrics_engine.py`, `tests/test_compression_engine.py`).

**Linting:**
- No linter configured (no ruff/flake8/pylint/mypy). `dev` optional dependency is `pytest>=7.0`, `pytest-timeout>=2.0`, `rich>=13.0` (`pyproject.toml` `[project.optional-dependencies]`).

**Line length:**
- Not enforced by config; source files observed keep well under 100 columns.

## Import Organization

**Order (observed convention — not enforced by a tool):**
1. Standard library: `import json`, `import os`, `import sys`, `import math`, `import tempfile`, `import copy`, `import time`, `import threading`.
2. Third-party: `import numpy as np` (always aliased `np`), `import pytest`, `from scipy import ...`.
3. Project imports: `from spectralstream.config import SpectralStreamConfig`, `from spectralstream.compression.engine import ...`.
4. Many test files insert the project root onto `sys.path` before importing the package:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

(See `tests/test_registry.py`, `tests/test_integration.py`, `tests/test_loss_metrics_engine.py`.) Note inconsistency: `tests/test_version.py` and `tests/test_config.py` use `sys.path.insert(0, ".")` instead.

**`from __future__ import annotations`** appears as the first import in essentially every module, enabling postponed evaluation of annotations for full type hints.

**Typing imports** follow PEP 604 / modern style: `from typing import Any, Optional, Dict, List` (e.g., `tests/test_integration.py` line 10), and many modules use `Optional[str]` from typing (`spectralstream/logging_config.py` line 27).

**Path Aliases:**
- No custom path aliases or `src` layout. Package root is `spectralstream/` and tests import via the installed/root package `spectralstream.*`.

## Error Handling

**Patterns:**
- Errors are raised with built-in exceptions using descriptive messages: `ValueError` (config/file errors), `KeyError` (missing tensor/key), `RuntimeError` / `TypeError` (bad inputs), `NotImplementedError`, `IndexError`, `FileNotFoundError`, `AssertionError`. Grep across `spectralstream/` found 281 `raise` statements in 116 files.
- Config validation does NOT raise; it returns a `list[str]` of warning strings. `SpectralStreamConfig.validate()` returns `[]` when valid and appends human-readable messages otherwise (`spectralstream/config.py`; tests assert `"HDC dim must be positive" in warnings`).
- Tests assert errors via `pytest.raises`, often with a tuple of acceptable exceptions:
  ```python
  with pytest.raises((RuntimeError, ValueError, TypeError)):
      engine.compress_fast("not_a_tensor", name="bad")
  ```
  (`tests/test_integration.py` lines 781-782). `with pytest.raises(KeyError):` for missing keys (line 853).
- Some older test files use the `try/except ... pass` style instead of `pytest.raises`:
  ```python
  try:
      SpectralStreamConfig.from_file("/tmp/nonexistent_config_file.json")
      assert False, "Expected ValueError"
  except ValueError:
      pass
  ```
  (`tests/test_config.py` lines 333-337). Prefer `pytest.raises` for new code.
- File I/O is wrapped in `try/finally` with `os.unlink(tmp_path)` cleanup in tests (`tests/test_config.py` lines 321-329).

## Logging

**Framework:** Python standard `logging` module (structured, not `print`).

**Infrastructure:** `spectralstream/logging_config.py` provides:
- `setup_logging(level="INFO", json_output=False, log_file=None)` — configures root logger `logging.getLogger("spectralstream")`. Idempotent via module-level `_CONFIGURED` flag.
- `get_logger(name)` — returns a child logger `spectralstream.<name>`, auto-calling `setup_logging()` if not configured.
- Two formatters: `_JSONFormatter` (one JSON object per line, with `timestamp`, `level`, `logger`, `message`, `module`, `function`, `line`, plus optional `exc_info` and `data`), and `_HumanFormatter` (colored console output, ANSI colors for DEBUG/INFO/WARNING/ERROR/CRITICAL, color only when `sys.stderr.isatty()`).
- `CompressionLogger` — records compression/decompression events with metrics (`ratio`, `error`, `time_s`) into a bounded `deque` (maxlen 10000), thread-safe via `threading.Lock`.
- `PerformanceTimer` — context manager (`with PerformanceTimer("quantize") as timer:`) timing code blocks via `time.perf_counter()`; `timed_operation(label, logger)` shorthand yields elapsed seconds.
- `AuditLogger` — audit trail of compression decisions/overrides into a bounded `deque` (maxlen 100_000), `export_trail(path)` writes JSON.

**Patterns:**
- Child loggers are obtained via `get_logger("compression")`, `get_logger("timer")`, `get_logger("audit")` rather than `logging.getLogger(__name__)`.
- Logs go to `sys.stderr` (console) and optionally a JSON-lines file via `log_file`.
- Messages use `%`-style lazy formatting: `self.logger.info("Compressed %s with %s: ratio=%.2fx, error=%.6f", tensor_name, method, ratio, error, time_s)`.

## Comments

**When to Comment:**
- Module docstrings are present at the top of most files describing purpose and exported components (e.g., `spectralstream/logging_config.py` lines 1-12, `spectralstream/config.py` lines 1-17, `spectralstream/version.py`).
- Inline comments are used sparingly to explain non-obvious math/physics steps (e.g., `# Large enough to trigger power iteration` in `tests/test_loss_metrics_engine.py` line 107).
- Section headers in large files use ASCII banners for logical grouping.

**Docstrings:**
- NumPy-style docstrings with `Parameters` / `Returns` sections are used on public functions (e.g., `setup_logging` in `spectralstream/logging_config.py` lines 96-106, `pytest_collection_modifyitems` in `tests/conftest.py` lines 78-82).
- Test files and small helper functions often omit docstrings; class-level docstrings summarize the test group (e.g., `class TestFullPipeline: """Tests the complete compression pipeline...""` in `tests/test_integration.py` line 92).

## Function Design

**Size:**
- Functions are generally small-to-medium. Logging/utility functions (`get_logger`, `setup_logging`) are compact. Some test helper/engine methods are larger, but no single function exceeds a few hundred lines in sampled files.
- Compression methods expose a consistent `compress(tensor, **params) -> (bytes, dict)` / `decompress(data, meta) -> np.ndarray` interface. Test asserts: `isinstance(result, tuple)` with `len == 2`, `result[0]` is `bytes`, `result[1]` is `dict` (`tests/test_registry.py` lines 129-136).

**Parameters:**
- Type hints are used throughout (`tensor: np.ndarray`, `name: str = "compression"`, `level: str = "INFO"`).
- `from __future__ import annotations` allows full annotations without runtime cost.
- Config classes use `@dataclass` fields with defaults; e.g. `HDCConfig(dim: int = 10000, ngram_order: int = 4, ...)`.

**Return Values:**
- Methods return tuples for multi-value results (`(data, meta)`, `(data, meta, ratio, error)`).
- Validation returns `list[str]` (empty = valid).
- Properties like `ratio`, `error`, `elapsed` are returned as floats; rounding to fixed precision is done at log time (`round(ratio, 4)`).

## Module Design

**Exports:**
- `__all__` is declared in 103 of the package's Python files (Grep count), listing public names. Examples:
  - `spectralstream/version.py`: `__all__ = ["__version__", "__version_info__"]`
  - `spectralstream/__init__.py`: `__all__ = ["SpectralOrchestrator", "setup_logging", "get_logger", "CompressionLogger", "PerformanceTimer", "timed_operation", "AuditLogger", "CompressionAudit", "InferenceAudit", "AuditTrail", "SupremeQuantEngine", "CompressionPipeline", "CompressedWeight", "CompressionBudget"]`
- The package `__init__.py` re-exports the top-level orchestrator, logging tools, audit tools, and the supreme quant engine. Lower-level engine/methods are imported directly from submodules by tests (`from spectralstream.compression.engine import ...`).

**Barrel Files:**
- `__init__.py` files act as barrels in subpackages (e.g., `spectralstream/compression/engine/__init__.py` exports `CompressionIntelligenceEngine`, `METHOD_REGISTRY`, etc.). Private sub-implementations live in `_`-prefixed sibling modules imported by the barrel.

**`from __future__ import annotations`:**
- Present at the top of virtually every `.py` file (source and test). Treat as mandatory for new files.

---

*Convention analysis: 2026-07-07*

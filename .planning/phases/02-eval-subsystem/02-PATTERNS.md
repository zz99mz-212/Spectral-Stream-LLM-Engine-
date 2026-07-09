# Phase 2: Eval Subsystem - Pattern Map

**Mapped:** 2026-07-08
**Files analyzed:** 16 (new/modified across `eval/`, `scripts/`, `tests/`, `spectralstream/`, root scripts, README)
**Analogs found:** 16 / 16 (every new file has a concrete in-repo analog; `eval/` is greenfield but each module maps to an explicit source pattern)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `eval/__init__.py` | module | — | `spectralstream/compression/engine/__init__.py` (barrel) | role-match |
| `eval/constants.py` | config | transform | `spectralstream/compression/honest_metrics.py:21-24` (module-level constants) | exact |
| `eval/model_path.py` | utility | request-response | `spectralstream/compression/cli.py:85-93` (`_validate_input_path`) + `benchmark_physics_real_weights.py:37` (hardcoded path target) | exact |
| `eval/corpus.py` | utility | file-I/O | `spectralstream/inference/pipeline.py:737-770` (token consumption) + `scripts/` conventions | role-match |
| `eval/artifact.py` | utility | transform | `spectralstream/compression/honest_metrics.py:168-219` (`apply_gate`) + `spectralstream/compression/certificate.py` | exact |
| `eval/grader.py` | service | batch | `spectralstream/inference/pipeline.py:737-770` (`measure_perplexity`) + `:774-791` (`close`) | exact |
| `eval/run_eval.py` | controller/CLI | request-response | `spectralstream/compression/cli.py:2558-2602` (`cmd_generate_certificate`) + `:2719-2802` (`build_parser`) | role-match |
| `eval/data/wikitext2_sample.txt` | data | file-I/O | `spectralstream/inference/` tokenizer test corpus (committed data convention) | partial |
| `eval/data/wikitext2_sample.tokens.json` | data | file-I/O | `eval/data/*.txt` counterpart; JSON-wrapped token ids | partial |
| `scripts/fetch_eval_corpus.py` | script | file-I/O | `scripts/compress_gemma4.py` (script conventions) | role-match |
| `tests/test_eval_subsystem.py` | test | CRUD | `tests/test_config.py` + `tests/test_registry.py` | exact |
| `spectralstream/utils/tokenizer_engine.py` (modify) | model | transform | `build_default_tokenizer` at `:2146` + `BPETokenizer._build_byte_fallback` at `:439-445` | exact (in-place) |
| `benchmark_physics_real_weights.py` (modify) | script | request-response | `benchmark_physics_real_weights.py:37-39` (current hardcoded path) | exact (in-place) |
| `wave4_pipeline.py` (modify) | script | request-response | `wave4_pipeline.py:164-165` (current hardcoded path) | exact (in-place) |
| `spectralstream/compression/cli.py` (modify, optional) | controller | request-response | `cli.py:2775-2802` (`sub.add_parser`) + `:3379-3398` (dispatch) | role-match |
| `README.md` (modify) | doc | — | `README.md:19` (Gemma-4 E2B model reference) | exact |

## Pattern Assignments

### `eval/grader.py` (service, batch) — **CORE of EVAL-01**

**Analog:** `spectralstream/inference/pipeline.py:737-791` (`measure_perplexity` + `close`)

**Imports pattern** (pipeline.py:9-40 — always present):
```python
from __future__ import annotations
import gc
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from spectralstream.inference.pipeline import InferencePipeline, InferenceConfig
```

**Core PPL pattern — reuse verbatim, do NOT re-implement (pipeline.py:737-770):**
```python
def measure_perplexity(
    self,
    test_tokens: List[int],
    stride: int = 512,
    max_seq_len: Optional[int] = None,
) -> float:
    """Measure perplexity on a sequence of tokens using sliding window."""
    max_seq = max_seq_len or self.model_config.MAX_POSITION_EMBEDDINGS
    max_seq = min(max_seq, 8192)
    nll = 0.0
    n_tokens = 0
    total_len = len(test_tokens)
    for start in range(0, total_len, stride):
        end = min(start + max_seq, total_len)
        if end - start < 10:
            break
        chunk = test_tokens[start:end]
        self.kv_cache.clear()
        self._kv_cache_dict.clear()
        tokens = np.array(chunk[:-1], dtype=np.int32)
        logits = self.forward(tokens)
        log_probs = logits - logits.max(axis=-1, keepdims=True)
        log_probs = log_probs - np.log(
            np.exp(log_probs).sum(axis=-1, keepdims=True) + 1e-30
        )
        target_tokens = np.array(chunk[1:], dtype=np.int32)
        token_log_probs = log_probs[np.arange(len(target_tokens)), target_tokens]
        nll += -float(np.sum(token_log_probs))
        n_tokens += len(target_tokens)
    if n_tokens == 0:
        return float("inf")
    return float(np.exp(nll / n_tokens))
```

**Sequential-load + close pattern (pipeline.py:774-791) — REQUIRED to avoid OOM (Pitfall 1):**
```python
def close(self) -> None:
    if self.use_unified:
        if hasattr(self, "_unified"):
            self._unified.close()
        return
    if self._ssf_reader is not None:
        self._ssf_reader.close()
        self._ssf_reader = None
    if self._loader is not None:
        close_fn = getattr(self._loader, "close", None)
        if callable(close_fn):
            close_fn()
        self._loader = None
    self.kv_cache.clear()
    self.layers.clear()
    self._layer_cache.clear()
    self._kv_cache_dict.clear()
    gc.collect()
```

**Grader skeleton to copy (research Pattern 1 + D-04/D-05):**
```python
def run_ppl(model_path: str, test_tokens: list[int], seq_len: int, stride: int) -> float:
    cfg = InferenceConfig()  # keep forward minimal/deterministic
    with InferencePipeline(model_path, config=cfg, use_unified=False) as pipe:
        return pipe.measure_perplexity(test_tokens, stride=stride, max_seq_len=seq_len)
```
Both runs MUST pass identical `test_tokens`, `stride`, `max_seq_len` (Pitfall 4), and use the
same `use_unified` value. Load base → `close()` → load compressed (never both in RAM).

**Pitfall 3 guard — silent layer drop (pipeline.py:279-304):**
`_get_layer_weights` returns `None` and `_get_layer` silently skips the layer if any of the 9
expected weight tensors is missing. Record `len(pipe.tensor_names)` (pipeline.py:308-314) and
compare to expected `model_config.NUM_HIDDEN_LAYERS * 9 + embeddings + lm_head`
(`config.py:14` → `NUM_HIDDEN_LAYERS = 35`, `VOCAB_SIZE = 262144`). Emit `layers_loaded` in artifact.

---

### `eval/artifact.py` (utility, transform) — **EVAL-01 artifact / honesty gate**

**Analog:** `spectralstream/compression/honest_metrics.py:21-24, 168-219` (`ERROR_GATE_THRESHOLD` constant + `apply_gate` chokepoint)

**Constant convention (honest_metrics.py:21-24):**
```python
ERROR_GATE_THRESHOLD = 0.05  # rel_mse; strict > gate; consistent with Phase-3 cascade acceptance
```

**Gate chokepoint convention (honest_metrics.py:168-219) — single decision point, strict `>`, retains raw metrics even when gated:**
```python
def apply_gate(payload, original_elements, rel_mse, threshold=ERROR_GATE_THRESHOLD) -> Dict[str, Any]:
    ratios = dual_ratio(original_elements, payload)
    gated = bool(rel_mse > threshold)  # STRICT >, never >=
    gate_reason = f"rel_mse {rel_mse:.4f} > {threshold}" if gated else ""
    return {
        "ratio_vs_bf16": ratios["ratio_vs_bf16"] if not gated else None,
        "ratio_vs_fp32": ratios["ratio_vs_fp32"] if not gated else None,
        "rel_mse": float(rel_mse),
        "gated": gated,
        "gate_reason": gate_reason,
    }
```

**Recovery gate — CRITICAL direction fix (D-07, Research Pitfall 2, Open Question Q1):**
```python
# CORRECT (retained-quality fraction, 1.0 = lossless):
recovery_ratio = ppl_base / ppl_compressed           # NOT ppl_compressed / ppl_base
gate_passed = recovery_ratio >= RECOVERY_GATE_THRESHOLD  # default 0.95 (D-08)
```
The locked spec's `ppl_compressed / ppl_base` with `>= 0.95` is **inverted** — it fails on
*better* compressed models and passes arbitrarily *worse* ones. Implement `base/compressed`
(or keep compressed/base but gate `<= 1.05`). Record the exact formula string in the artifact.

**Artifact schema (D-09) — all REAL measured values, no estimates:**
```python
artifact = {
    "model": model_name,
    "method": method_name,
    "tokenizer": tokenizer_name,
    "base_ppl": float(base_ppl),
    "compressed_ppl": float(compressed_ppl),
    "recovery_ratio": float(recovery_ratio),          # == ppl_base / ppl_compressed
    "recovery_gate_threshold": float(RECOVERY_GATE_THRESHOLD),
    "gate_passed": bool(gate_passed),
    "seq_len": int(seq_len),
    "stride": int(stride),
    "n_tokens": int(n_tokens),
    "layers_loaded": int(layers_loaded),
    "timestamp": iso_timestamp,
    "git_ref": git_sha,
}
```
Mirror `honest_metrics` honesty: never compute from per-stage estimates; both PPLs from
`measure_perplexity`. Write under `eval/artifacts/` (git-ignore emitted; commit sample under `eval/data/`).

---

### `eval/model_path.py` (utility, request-response) — **EVAL-03**

**Analog:** `spectralstream/compression/cli.py:85-93` (`_validate_input_path`) + `benchmark_physics_real_weights.py:37-39` (path target)

**Path-validation pattern (cli.py:82-93) — reuse the traversal regex:**
```python
_PATH_TRAVERSAL_PATTERN = re.compile(r"\.\./|\.\.\\|/\.\.|\\\.\.")

def _validate_input_path(path: str) -> Path:
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")
    if _PATH_TRAVERSAL_PATTERN.search(path):
        raise ValueError(f"Path traversal detected: {path!r}")
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved
```

**`resolve_model_path` target (D-10/D-11, Research Pattern 3) — replace `benchmark_physics_real_weights.py:37-39`:**
```python
# benchmark_physics_real_weights.py:37-39 (REMOVE this hardcoded absolute path):
#   mmap = MemoryMappedTensorEngine(
#       "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"
#   )
def resolve_model_path(cli_model: str | None = None) -> str:
    path = (
        cli_model
        or os.environ.get("SPECTRALSTREAM_MODEL_PATH")
        or "models/gemma-4-E2B/model.safetensors"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Set SPECTRALSTREAM_MODEL_PATH "
            f"or pass --model to a local .safetensors/.ssf path."
        )
    return path
```
Resolution order: `--model` → `SPECTRALSTREAM_MODEL_PATH` → relative `models/gemma-4-E2B/model.safetensors`.
Never a hardcoded absolute author path (SEC-02).

---

### `spectralstream/utils/tokenizer_engine.py` (modify) — **EVAL-02 / BUG-04**

**Analog:** `build_default_tokenizer` at `:2146-2161` + `BPETokenizer._build_byte_fallback` at `:439-445`

**Current raise sites (tokenizer_engine.py:133-137) — REPLACE with byte-identity fallback (D-01):**
```python
def encode(self, text: str) -> list[int]:
    raise NotImplementedError          # ← remove

def decode(self, token_ids: list[int]) -> str:
    raise NotImplementedError          # ← remove
```
**New (round-trips any text, never raises — satisfies D-03):**
```python
def encode(self, text: str) -> list[int]:
    # Byte-identity fallback: every byte is its own token id (0..255).
    return list(text.encode("utf-8"))

def decode(self, token_ids: list[int]) -> str:
    return bytes(int(t) & 0xFF for t in token_ids).decode("utf-8", errors="replace")
```

**Shipped default — unchanged but confirmed (tokenizer_engine.py:2146-2161):**
```python
def build_default_tokenizer(vocab_size: int = 32000) -> BPETokenizer:
    tok = BPETokenizer()
    tok._tokens = [bytes([i]) for i in range(min(vocab_size, 256))]
    tok._vocab_size = len(tok._tokens)
    tok._scores = [0.0] * tok._vocab_size
    tok._token_type = [1] * tok._vocab_size
    tok._vocab = {tok._tokens[i]: i for i in range(tok._vocab_size)}
    tok._id_to_token = dict(enumerate(tok._tokens))
    tok.bos_id = 1
    tok.eos_id = 2
    tok.unk_id = 0
    return tok
```
`AutoTokenizer` (`:1457-1466`) returns `BPETokenizer()` when no model tokenizer is available;
`auto_tokenizer_for_model` (`:2196-2234`) falls back to `build_default_tokenizer()` on failure.
The eval's *actual* perplexity tokenizer is the model's OWN via `AutoTokenizer.from_gguf` / `from_pretrained`
(D-02) — do NOT pull in tiktoken/transformers/sentencepiece.

---

### `eval/run_eval.py` (controller/CLI, request-response) — **EVAL-01 entry**

**Analog:** `spectralstream/compression/cli.py:2558-2602` (`cmd_generate_certificate` body) + `:2719-2802` (`build_parser`) + `:3379-3398` (dispatch)

**`cmd_` function body pattern (cli.py:2558-2765) — file-existence guard + logger + sys.exit(1):**
```python
def cmd_generate_certificate(args: argparse.Namespace) -> None:
    if not os.path.exists(args.ssf_file):
        logger.error("SSF file not found: %s", args.ssf_file)
        sys.exit(1)
    logger.info("Generating certificate from: %s", args.ssf_file)
    try:
        reader = SSFReader(args.ssf_file, mmap_mode=True)
    except Exception as e:
        logger.error("Failed to open SSF file: %s", e)
        sys.exit(1)
```
Logger convention: `logger = logging.getLogger(__name__)` (cli.py:61). Use `%`-style lazy formatting.

**Argparse subparser pattern (cli.py:2775-2802):**
```python
sub = parser.add_subparsers(dest="command", required=True)
cp = sub.add_parser("eval", help="Run WikiText-2 perplexity quality eval")
cp.add_argument("--model", help="Path to base model (.safetensors)")
cp.add_argument("--compressed", required=True, help="Path to compressed (.ssf)")
cp.add_argument("--corpus", help="Path to token corpus (default: committed sample)")
cp.add_argument("--tokenizer", help="Path to model tokenizer.json (optional)")
cp.add_argument("--seq-len", type=int, default=2048)
cp.add_argument("--stride", type=int, default=512)
cp.add_argument("--threshold", type=float, default=0.95)
cp.add_argument("--output", help="Artifact output path")
```

---

### `tests/test_eval_subsystem.py` (test, CRUD) — **EVAL-01/02/03**

**Analog:** `tests/test_config.py` + `tests/test_registry.py` (assert-style, `from __future__ import annotations`)

**Required assertions (per Research Validation Map):**
- `test_base_tokenizer_roundtrip`: `BaseTokenizer().encode`/`decode` round-trips sample text, does NOT raise (D-03).
- `test_default_tokenizer`: `build_default_tokenizer()` round-trips AND `vocab_size == 256`.
- `test_resolve_model_path`: honors env > `--model` > relative; raises `FileNotFoundError` on missing.
- `test_artifact_schema`: artifact contains all D-09 fields; both PPLs real (not inf/nan when weights present).
- `test_recovery_gate`: worse compressed → `gate_passed=False`; equal/better → `True` (depends Q1 resolution).
- `test_layers_loaded`: `layers_loaded` recorded / asserted (Pitfall 3).
- Static: `grep -rn "/home/mike" scripts/ benchmark_*.py` returns nothing (EVAL-03).

Test files live at top-level `tests/`, prefix `test_*.py` (never co-located). Use `pytest.raises`.

---

### `benchmark_physics_real_weights.py` & `wave4_pipeline.py` (modify) — **EVAL-03**

**Target sites:**
- `benchmark_physics_real_weights.py:37-39` — hardcoded
  `"/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"`.
- `wave4_pipeline.py:164-165` — `model_path = "models/gemma-4-E2B/model.safetensors"` (relative;
  augment with `os.environ.get("SPECTRALSTREAM_MODEL_PATH")` / `--model` arg per D-10).

Replace with `resolve_model_path(...)` (or inline env→fallback). Note: `signal.alarm`/`exec` timeouts
in `benchmark_physics_real_weights.py` are Windows-incompat (handled by `fix/honest-metrics-windows-compat` branch) —
out of phase scope; only the path parameterization (D-10) is required here.

---

## Shared Patterns

### Header / module conventions (apply to ALL new files)
**Source:** every `spectralstream/*.py` (observed at `config.py`, `version.py`, `logging_config.py`)
```python
from __future__ import annotations
```
Mandatory on every module. `snake_case` functions, `from __future__ import annotations`,
4-space indent, `git_ref`/`timestamp` from artifact should use `subprocess`/`datetime` (stdlib).

### Honesty / no-fabrication reporting
**Source:** `spectralstream/compression/honest_metrics.py:1-12, 168-219`
**Apply to:** `eval/grader.py`, `eval/artifact.py`
Both PPL values MUST be REAL measured outputs of `measure_perplexity`. No estimates, no
per-stage products. Gate is a single chokepoint with strict `>` / `>=` and retains raw metrics.

### Logging
**Source:** `spectralstream/compression/cli.py:61`
**Apply to:** `eval/run_eval.py`, `eval/grader.py`
```python
logger = logging.getLogger(__name__)
```
Use `%`-style lazy formatting (`logger.info("Compressed %s: ppl=%.2f", name, ppl)`).

### Path validation (ASVS V5)
**Source:** `spectralstream/compression/cli.py:82-93`
**Apply to:** `eval/model_path.py`, `eval/run_eval.py` (`--model`/`--corpus`/`--tokenizer`)
Reuse `_PATH_TRAVERSAL_PATTERN`; reject `..` segments; verify existence.

## No Analog Found

All files have a concrete in-repo analog. The `eval/` directory is greenfield but every module
maps to an explicit source pattern listed above. No file requires external-package invention
(D-02/D-04 forbid torch/lm_eval/tiktoken/transformers/sentencepiece).

## Metadata

**Analog search scope:** `spectralstream/inference/pipeline.py`, `spectralstream/utils/tokenizer_engine.py`, `spectralstream/compression/honest_metrics.py`, `spectralstream/compression/cli.py`, `spectralstream/inference/config.py`, `benchmark_physics_real_weights.py`, `wave4_pipeline.py`, `tests/test_config.py`, `README.md`.
**Files scanned:** 16 source + 2 research/context docs.
**Pattern extraction date:** 2026-07-08
**Open blocker flagged:** Recovery-gate direction (Research Open Question Q1 / Pitfall 2) — planner must resolve before Plan 02-01. Recommend `recovery_ratio = ppl_base / ppl_compressed`, gate `>= 0.95`.
**Critical risk flagged:** Logits OOM at seq_len 2048 (Pitfall 1) — `VOCAB_SIZE=262144` (config.py:15) → `[2048,262144]` float32 ≈ 2.15 GB/window. Mitigate with sequential load (`close()` between runs) + vocab-blocked log-softmax.

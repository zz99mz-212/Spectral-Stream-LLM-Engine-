# Phase 2: Eval Subsystem - Research

**Researched:** 2026-07-08
**Domain:** Pure-Python (NumPy/SciPy, no torch) downstream-quality evaluation — WikiText-2 perplexity grader wrapping `InferencePipeline`, tokenizer hardering, and reproducible model-path wiring.
**Confidence:** HIGH for API surface, behaviors, and the memory/gate findings (all verified against source). MEDIUM for corpus-sourcing specifics (depends on user-supplied tokenizer artifacts). LOW for tokenizer-faithfulness claims when the byte-level fallback is used.

## Summary

Phase 2 wraps **existing** code; it introduces **no new external dependencies** (locked by D-02/D-04). The three loads are: (1) `InferencePipeline.measure_perplexity()` already produces real logit-based sliding-window perplexity — the eval is a thin "independent grader" that runs it on original FP16 vs a compressed `.ssf` and emits a JSON artifact; (2) `BaseTokenizer.encode/decode` raise `NotImplementedError` (tokenizer_engine.py:134) — fixed by giving the ABC a byte-identity fallback (round-trips any text, never raises) while `build_default_tokenizer()` / `AutoTokenizer` keep the byte-level BPE as the shipped default; (3) hardcoded `/home/mike/.../gemma-4-E2B` paths (benchmark_physics_real_weights.py:38, wave4_pipeline.py:165) are replaced by `SPECTRALSTREAM_MODEL_PATH` env + `--model` CLI, resolved against a relative `models/gemma-4-E2B/model.safetensors` fallback.

Two findings dominate the planning risk. **(A) Memory:** Gemma-4 E2B has `VOCAB_SIZE = 262144` (config.py:15). With `seq_len = 2048` (roadmap-locked) and the existing `measure_perplexity` materializing a `[2048, 262144]` float32 logits matrix per window, each forward pass needs **~2.15 GB** just for logits, plus the full model in fp32 (~10–20 GB). This will OOM typical dev machines unless (i) the two models are loaded **sequentially** (measure base, `close()`, then load compressed), and (ii) the vocab log-softmax is computed in **column blocks** to cap peak logits memory. **(B) Inverted gate:** the locked spec D-07/D-08 defines `recovery_ratio = ppl_compressed / ppl_base` AND "gate passes when `recovery_ratio >= 0.95`". Because lower PPL is better, this gate **fails precisely when compressed is *better* than base and passes for arbitrarily *worse* models** — it is inverted relative to "quality preserved." This must be resolved before the gate is wired (see Open Questions Q1). The artifact should record `recovery_ratio` in a documented direction plus a separately-defined gate predicate.

**Primary recommendation:** Build `eval/` as a thin grader over `InferencePipeline.measure_perplexity` with sequential model loading, vocab-blocked log-softmax, an identical windowing/corpus for both runs, a byte-identity fallback in `BaseTokenizer`, a shared `resolve_model_path()` helper, and a committed offline byte-level WikiText-2 sample. Fix the gate direction before enforcing it.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `BaseTokenizer.encode`/`decode` MUST NOT raise `NotImplementedError`. Give the ABC a byte-level fallback. `build_default_tokenizer()` (byte-level BPE, 256 tokens, tokenizer_engine.py:2146) becomes the shipped default returned by `AutoTokenizer` when no model tokenizer is available.
- **D-02:** For the eval's *actual* perplexity tokenization, reuse the model's OWN tokenizer via existing `AutoTokenizer.from_gguf` / `auto_tokenizer_for_model`. Do NOT pull in `tiktoken`, `transformers`, or `sentencepiece`.
- **D-03:** A test asserts `encode` → `decode` round-trips on sample text (round-trip correctness, not linguistic fidelity).
- **D-04:** EVAL-01 uses the NATIVE in-core NumPy forward pass via `InferencePipeline.measure_perplexity(test_tokens, stride, max_seq_len)` (pipeline.py:737). NO `lm_eval`/torch subprocess.
- **D-05:** The eval adds a thin independent grader: corpus load → tokenize → original (FP16/safetensors) → compressed (.ssf) → record PPL → recovery ratio → gate → JSON artifact.
- **D-06:** `seq_len = 2048` (roadmap-locked). Default `stride = 512` (method's existing default), configurable.
- **D-07:** `recovery_ratio = ppl_compressed / ppl_base`. [Research flags this as inverted vs the gate intent — see Open Questions Q1.]
- **D-08:** Default threshold = 0.95. Configurable via constant + env/CLI override.
- **D-09:** Artifact JSON fields: `model`, `method`, `tokenizer`, `base_ppl`, `compressed_ppl`, `recovery_ratio`, `recovery_gate_threshold`, `gate_passed`, `seq_len`, `stride`, `n_tokens`, `timestamp`, `git_ref`. Both PPL values are REAL MEASURED numbers. Written under `eval/artifacts/` (or `eval/results/`).
- **D-10:** Remove hardcoded paths (benchmark_physics_real_weights.py:38, wave4_pipeline.py:165); replace with `SPECTRALSTREAM_MODEL_PATH` env + `--model` CLI. Thread through eval entry + parameterize the scripts.
- **D-11:** README documents required model (Gemma-4 E2B) + how user supplies it. Resolution order: env var → `--model` → relative `models/gemma-4-E2B/model.safetensors` fallback — never a hardcoded absolute author path.
- **D-12:** Commit a small deterministic sampled slice of WikiText-2 test tokens (e.g. `eval/data/wikitext2_sample.*`) so the default run is offline/reproducible. Provide `scripts/fetch_eval_corpus.py` for the full set + `--corpus PATH` override.
- **D-13:** Corpus tokenization uses the model's own tokenizer (D-02). Committed sample is pre-tokenized for the default tokenizer; commit raw text too for transparency.

### Claude's Discretion
- Artifact filename/path, env-var name (`SPECTRALSTREAM_MODEL_PATH` vs shorter), default stride, exact sample size, and whether to fully support the 2048-token window under memory limits.
- WikiText-2 vs WikiText-103 — pick whichever the committed sample represents.
- New module/file names under `eval/` and the CLI subcommand name — following `snake_case` CLI conventions.

### Deferred Ideas (OUT OF SCOPE)
- CI perplexity gate — deferred to v2 (CI-01 / MISS-04).
- lm-eval-harness oracle — explicitly NOT chosen (would add torch).
- INT4 quality proof — Phase 4.
- Full WikiText-2 vs small sample — full corpus opt-in; committed sample keeps runs offline.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-01 | WikiText-2 perplexity (seq len 2048) original FP16 vs compressed; verifiable JSON artifact with both PPL + recovery ratio | `InferencePipeline.measure_perplexity` (pipeline.py:737) is the real source; grader wraps it; artifact schema D-09 |
| EVAL-02 | `BaseTokenizer.encode` must not raise; ship one working default tokenizer; add test | Byte-identity fallback in `BaseTokenizer` (tokenizer_engine.py:134) + `build_default_tokenizer()` (line 2146) byte-level BPE |
| EVAL-03 | Replace hardcoded absolute model path with env/CLI arg; document required model | `resolve_model_path()` helper + parameterize benchmark_physics_real_weights.py:38, wave4_pipeline.py:165; README doc |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Perplexity measurement (forward pass + logits) | Model/Inference tier (in-process) | — | `InferencePipeline.measure_perplexity` runs the Gemma-4 forward pass natively; no external service |
| Corpus tokenization | Tokenization tier (in-process, pure-Python) | Model tokenizer loader (GGUF/tokenizer.json) | `AutoTokenizer`/`build_default_tokenizer` are in-process; model's own tokenizer optional |
| Model loading & path resolution | Config/Env tier | CLI entry | `SPECTRALSTREAM_MODEL_PATH` → `--model` → relative fallback |
| Recovery gate computation | Eval grader tier (script/CLI) | — | Pure arithmetic on two measured PPLs; must be deterministic |
| Artifact emission | Eval grader tier (script/CLI) | Filesystem (`eval/artifacts/`) | JSON write; mirrors `honest_metrics` "real measured values" convention |
| Deterministic RNG / corpus | Eval grader tier | Committed data (`eval/data/`) | Sample committed for offline reproducibility |

## Standard Stack

No new external libraries. The phase reuses the locked in-core stack (NumPy/SciPy, `safetensors`, `zstandard` already present). This is mandated by D-02/D-04 — pulling in `torch`, `lm_eval`, `tiktoken`, `transformers`, or `sentencepiece` would cross the pure-Python/no-torch boundary.

### Core (reused, not added)
| Asset | Location | Purpose | Why Standard |
|-------|----------|---------|--------------|
| `InferencePipeline` | `spectralstream/inference/pipeline.py:62` | Loads SSF/safetensors, runs Gemma-4 forward, `measure_perplexity` | The locked, real perplexity source (D-04) |
| `measure_perplexity` | `spectralstream/inference/pipeline.py:737` | Sliding-window logit PPL | Real measured perplexity; verbatim reused |
| `BaseTokenizer` + `build_default_tokenizer` | `spectralstream/utils/tokenizer_engine.py:110, 2146` | ABC fix + byte-level default | Existing tokenizer core (D-01/D-02) |
| `AutoTokenizer` / `auto_tokenizer_for_model` | `spectralstream/utils/tokenizer_engine.py:1444, 2196` | Load model's own tokenizer | GGUF / tokenizer.json detection already implemented |
| `honest_metrics` conventions | `spectralstream/compression/honest_metrics.py` | Byte-exact / measured-value reporting pattern | The honesty template the artifact mirrors (D-09) |
| `InferenceConfig` | `spectralstream/inference/pipeline.py:44` | Pipeline config dataclass | Extend for eval flags if needed |

### Supporting
| Asset | Purpose | When to Use |
|-------|---------|-------------|
| `scripts/fetch_eval_corpus.py` (new) | Download full WikiText-2 test set (opt-in) | User wants > committed sample |
| `eval/run_eval.py` (new) | Grader entry with argparse | Default eval invocation |

**Installation:** None — no new dependencies. All imports come from the existing locked stack.

## Package Legitimacy Audit

**N/A — this phase installs zero external packages.** D-02/D-04 explicitly forbid adding `torch`, `lm_eval`, `tiktoken`, `transformers`, `sentencepiece`. All functionality reuses in-repo modules (`InferencePipeline`, `AutoTokenizer`, `BaseTokenizer`, `honest_metrics`). No `npm view` / `pip index` check is required because no new registry package is introduced. The only new *user-supplied* artifact is the Gemma-4 tokenizer (a `tokenizer.json` the user points at via `--tokenizer`), which is data, not a dependency.

## Architecture Patterns

### System Architecture Diagram

```
                          ┌─────────────────────────────────────────┐
                          │  eval/run_eval.py  (grader / CLI)        │
                          └───────────────┬─────────────────────────┘
                                          │
            resolve_model_path()           │        resolve_corpus()        resolve_tokenizer()
   env SPECTRALSTREAM_MODEL_PATH ──┐      │      ┌── committed sample ──┐   ┌── model tokenizer.json ──┐
   --model PATH ────────────────────┼──────┘      └── --corpus PATH ─────┼───└── build_default (byte) ──┘
   relative models/.../model.safetensors    │                             │
                                          ▼                             ▼
                          ┌──────────────────────────┐      ┌────────────────────────────┐
                          │ tokenize(corpus_text)    │      │ AutoTokenizer / default    │
                          │ → List[int] test_tokens │      │ (deterministic id list)    │
                          └────────────┬─────────────┘      └─────────────┬──────────────┘
                                       │                                  │
                                       └──────────────┬───────────────────┘
                                                      ▼
                       ┌──────────────────────────────────────────────────────────┐
                       │  GRADED RUN #1 (base)            GRADED RUN #2 (compressed)│
                       │  InferencePipeline(model_fp16)   InferencePipeline(.ssf)   │
                       │        │ close() first                  │                    │
                       │        ▼                              ▼                    │
                       │  measure_perplexity(test_tokens,     measure_perplexity(   │
                       │     stride, max_seq_len=2048)        same args)            │
                       │        │  [seq load: NO overlap]     │                     │
                       │        ▼                              ▼                     │
                       │  base_ppl (measured float)     compressed_ppl (measured)   │
                       └───────────────────────────┬──────────────────────────────┘
                                                    ▼
                       ┌──────────────────────────────────────────────────────────┐
                       │  recovery gate: compute ratio, compare to threshold,      │
                       │  record gate_passed  →  emit JSON artifact               │
                       │  (eval/artifacts/<model>_<method>_eval.json)            │
                       └──────────────────────────────────────────────────────────┘
```

Key invariants for honesty:
- **Both runs use IDENTICAL `test_tokens`, `stride`, `max_seq_len`.** Only the weights differ, so `recovery_ratio` is apples-to-apples even if absolute PPL isn't perfectly standard.
- **Sequential model loading** (measure base → `close()` → load compressed) to avoid holding two ~20 GB models in RAM.
- **Vocab-blocked log-softmax** inside the PPL loop (see Pitfall 1) to cap peak memory.

### Recommended Project Structure
```
eval/
├── __init__.py
├── run_eval.py          # grader entry: argparse (--model, --compressed, --corpus, --tokenizer, --seq-len, --stride, --threshold, --output)
├── grader.py            # Evaluator: loads pipeline, runs PPL, computes recovery, gate
├── corpus.py            # resolve_corpus(): committed sample + --corpus override
├── model_path.py        # resolve_model_path(): env → --model → relative fallback  (shared helper)
├── constants.py         # RECOVERY_GATE_THRESHOLD, DEFAULT_SEQ_LEN, DEFAULT_STRIDE, ARTIFACT_DIR
├── artifact.py          # build_eval_artifact() + write_json()
└── artifacts/           # emitted JSON (git-ignored or committed? default: git-ignored)
eval/data/
├── wikitext2_sample.txt          # raw WikiText-2 text (transparency, D-13)
└── wikitext2_sample.tokens.json # byte-level token ids for default tokenizer (offline default)
scripts/
└── fetch_eval_corpus.py # opt-in full WikiText-2 download
tests/
└── test_eval_subsystem.py  # EVAL-01/02/03 assertions
```

### Pattern 1: Wrap `measure_perplexity` (thin grader, D-05)
**What:** A grader that calls the existing method twice and compares — it does NOT re-implement the forward pass.
**When to use:** Every eval run. This is the entire EVAL-01 implementation surface.
**Example:**
```python
# Source: spectralstream/inference/pipeline.py:737 (verified)
from spectralstream.inference.pipeline import InferencePipeline, InferenceConfig

def run_ppl(model_path: str, test_tokens: list[int], seq_len: int, stride: int) -> float:
    cfg = InferenceConfig()  # keep forward minimal/deterministic
    with InferencePipeline(model_path, config=cfg, use_unified=False) as pipe:
        return pipe.measure_perplexity(test_tokens, stride=stride, max_seq_len=seq_len)
```
**Note:** `use_unified=False` routes `forward()` directly to `_legacy_forward` (pipeline.py:399-400), avoiding the Unified engine's wrapper overhead/strategy surface. Both runs MUST use the same `use_unified` value.

### Pattern 2: Byte-identity fallback in `BaseTokenizer` (D-01, EVAL-02)
**What:** The ABC's `encode`/`decode` map text ↔ bytes directly so they never raise and always round-trip.
**When to use:** Default behavior of the abstract base; subclasses (BPE/SentencePiece/Tiktoken) still override with full tokenizers.
**Example:**
```python
# Source: spectralstream/utils/tokenizer_engine.py:133-137 (verified raise sites)
class BaseTokenizer:
    def encode(self, text: str) -> list[int]:
        # Byte-identity fallback: every byte is its own token id (0..255).
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(int(t) & 0xFF for t in token_ids).decode("utf-8", errors="replace")
```
This guarantees `decode(encode(t)) == t` for ANY text (byte-exact), satisfying D-03, and never raises. `build_default_tokenizer()` (BPETokenizer with `_build_byte_fallback`, line 439-445) remains the *shipped default* returned by `AutoTokenizer` when no model tokenizer is available.

### Pattern 3: `resolve_model_path` (D-10, EVAL-03)
**What:** Single resolution order env → CLI → relative fallback, reused by eval and the parameterized scripts.
**When to use:** Anywhere a model path was previously hardcoded.
**Example:**
```python
import os

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

### Anti-Patterns to Avoid
- **Holding both models in RAM:** loads ~20 GB fp32 each — OOM. Load sequentially, `close()` between.
- **Faithfulness claim with byte-level fallback:** do NOT claim "faithful to the reference token distribution" when the default byte tokenizer is used (no Gemma-4 SentencePiece). The recovery ratio is still valid (both runs share the same tokenizer); only the *absolute* PPL is unfaithful.
- **Divergent windowing between runs:** different `stride`/`max_seq_len`/corpus between base and compressed silently invalidates `recovery_ratio`. Pass identical args.
- **Re-implementing the forward pass:** never add a torch/NumPy-from-scratch PPL; wrap `measure_perplexity`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Perplexity / forward pass | New NumPy transformer forward | `InferencePipeline.measure_perplexity` | Real, already-validated Gemma-4 forward; re-implementing risks silent correctness bugs (D-04) |
| Tokenizer | New tokenizer for Gemma-4 | `AutoTokenizer` / `auto_tokenizer_for_model` + `build_default_tokenizer` | GGUF/tokenizer.json detection already exists; re-inventing breaks faithfulness |
| Byte-exact ratio / error reporting | Ad-hoc JSON numbers | Mirror `honest_metrics` (`serialized_nbytes`, measured values) | Project honesty mandate; the artifact must contain REAL measured PPL, never estimates |
| Model loading | Custom safetensors/SSF loaders | `InferencePipeline` (wraps `ModelLoader`/`SafeTensorsLoader`/`SSFReader`) | Already handles `.ssf`, `.safetensors`, SSF metadata override |

**Key insight:** Every "new" capability here already exists in-core. The phase is integration, not invention. Hand-rolling any of the above would both duplicate code and risk correctness/honesty regressions.

## Common Pitfalls

### Pitfall 1: Logits matrix OOM at seq_len 2048 (CRITICAL)
**What goes wrong:** Gemma-4 E2B has `VOCAB_SIZE = 262144` (config.py:15). `measure_perplexity` builds `logits = hidden @ lm_head.T` → shape `[seq_len, 262144]` float32. At `seq_len = 2048` (roadmap-locked), that is `2048 × 262144 × 4 bytes ≈ 2.15 GB` per window, materialized again in the `np.exp(log_probs).sum(...)` line. With overlapping stride-512 windows this repeats per window, and the full model sits in fp32 (~10–20 GB).
**Why it happens:** The existing method was likely tuned for smaller vocab (e.g. 32k) and stride 512; the 262k Gemma-4 vocab makes the matrix 8× larger than expected.
**How to avoid:**
- Load models **sequentially** (measure base, `close()`, then compressed) — never both at once.
- Replace the inline `np.exp(log_probs).sum(axis=-1)` with a **vocab-blocked log-softmax**: iterate over vocab columns in chunks of e.g. 4096, accumulate `log(sum(exp(block)))`, so peak logits memory is `seq_len × block` not `seq_len × 262144`. Provide this as a thin wrapper or a patched `measure_perplexity` call.
- Consider measuring on a **subset** of windows when memory is constrained (document the window count in the artifact).
**Warning signs:** RSS climbing past available RAM during the first window; `MemoryError` / swap thrash; process killed.

### Pitfall 2: Inverted recovery gate (CRITICAL — must resolve before wiring)
**What goes wrong:** Locked spec D-07 defines `recovery_ratio = ppl_compressed / ppl_base` and D-08 says "gate passes when `recovery_ratio >= 0.95`". Because lower PPL is better, this predicate **fails when compressed is *better* than base and passes for arbitrarily *worse* models** (e.g. `ratio = 2.0` → `2.0 >= 0.95` → PASS). It is inverted relative to "quality preserved."
**Why it happens:** `compressed/base` near 1.0 means equality; but the `>= 0.95` test only rejects `ratio < 0.95`, which occurs when compressed is *more than 5% better* — the opposite of the intent.
**How to avoid:** Pick ONE consistent definition. Recommended: define `recovery_ratio = ppl_base / ppl_compressed` (≥ 0.95 means base PPL is at most ~5% above the compressed model, i.e. compressed is within 5% of base quality) and gate on `recovery_ratio >= threshold`. OR keep `ratio = ppl_compressed/ppl_base` but gate on `ratio <= 1.05` (upper bound, not lower). Either way, the gate must treat *worse* compressed models as failures. The artifact must document the exact formula used. See Open Questions Q1.
**Warning signs:** A compressed model with dramatically *lower* PPL failing the gate; a 2×-worse compressed model passing.

### Pitfall 3: Silent layer-dropping yields a plausible-but-wrong PPL
**What goes wrong:** `InferencePipeline._get_layer_weights` returns `None` (and `_get_layer` silently skips the layer) if any of the 9 expected weight tensors for a layer is missing. If a `.ssf`/`safetensors` is missing layers, `measure_perplexity` still returns a finite, believable PPL computed over a *partial* model — silently wrong.
**Why it happens:** Graceful skip in the pipeline (pipeline.py:365-366) prioritizes not-crashing over correctness.
**How to avoid:** Before measuring, assert/record `len(pipe.tensor_names)` and compare against expected layer count (`model_config.NUM_HIDDEN_LAYERS * 9 + embeddings + lm_head`); emit a `layers_loaded` field and warn/abort if short. Record this in the artifact.
**Warning signs:** PPL suspiciously close to a trivial baseline; `tensor_names` count far below expected.

### Pitfall 4: Tokenizer mismatch between base and compressed runs
**What goes wrong:** If base run uses the model's SentencePiece tokenizer but the compressed run accidentally uses the byte-level default (or vice-versa), the two PPLs are computed on different token sequences and `recovery_ratio` is meaningless.
**Why it happens:** The grader might resolve the tokenizer per-run instead of once and reuse.
**How to avoid:** Resolve the tokenizer **exactly once** and pass the same `test_tokens` to both runs (tokenize before measuring, never inside the per-run loop). This is already enforced by Pattern 1's call shape.
**Warning signs:** `tokenizer` field differs between the two runs in the artifact; `n_tokens` differs.

### Pitfall 5: `BaseTokenizer` fix breaks subclass contracts
**What goes wrong:** Giving the ABC a real `encode`/`decode` could shadow intended subclass behavior if callers relied on the base raising.
**Why it happens:** Subclasses (BPETokenizer etc.) already override these; a base default only affects direct `BaseTokenizer()` instances, which previously raised.
**How to avoid:** The byte-identity fallback is safe: subclasses override first (MRO), and direct `BaseTokenizer()` instances now work for round-trip tests. Add `test_eval_subsystem.py` asserting `BaseTokenizer().encode/decode` round-trips.

## Code Examples

### Run both PPLs sequentially (honest grader core)
```python
# Source: spectralstream/inference/pipeline.py:737, :79 (verified signatures)
from spectralstream.inference.pipeline import InferencePipeline, InferenceConfig
from eval.model_path import resolve_model_path

def measure(model_path: str, test_tokens: list[int], seq_len: int, stride: int) -> float:
    cfg = InferenceConfig()
    with InferencePipeline(model_path, config=cfg, use_unified=False) as pipe:
        return pipe.measure_perplexity(test_tokens, stride=stride, max_seq_len=seq_len)

base_ppl = measure(resolve_model_path(args.model), toks, 2048, 512)      # close() after `with`
comp_ppl = measure(args.compressed, toks, 2048, 512)                   # same toks/stride/seq_len
```

### Default tokenizer round-trip (EVAL-02, D-03)
```python
# Source: spectralstream/utils/tokenizer_engine.py:2146 (verified)
from spectralstream.utils.tokenizer_engine import build_default_tokenizer
tok = build_default_tokenizer()  # BPETokenizer, 256 byte tokens
text = "The quick brown fox 🦊 jumps!"
assert tok.decode(tok.encode(text)) == text
```

### Model's own tokenizer (D-02) — only when user supplies it
```python
# Source: spectralstream/utils/tokenizer_engine.py:1470, :2196 (verified)
from spectralstream.utils.tokenizer_engine import AutoTokenizer, auto_tokenizer_for_model

def load_eval_tokenizer(model_dir_or_gguf: str | None):
    if model_dir_or_gguf and os.path.exists(model_dir_or_gguf):
        try:
            return AutoTokenizer.from_pretrained(model_dir_or_gguf)  # dir w/ tokenizer.json or .gguf
        except Exception:
            pass
    return build_default_tokenizer()  # byte-level fallback (unfaithful but consistent)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No eval baseline existed (MISS-01) | `InferencePipeline.measure_perplexity` provides real logit PPL | already present (pipeline.py:737) | Eval feasible in-core; phase wraps it |
| `BaseTokenizer.encode` raised `NotImplementedError` | Byte-identity fallback + `build_default_tokenizer` default | this phase (D-01) | Tokenizer usable for tests |
| Hardcoded `/home/mike/.../gemma-4-E2B` | `SPECTRALSTREAM_MODEL_PATH` + `--model` + relative fallback | this phase (D-10) | Fresh clone can reproduce |
| Fabricated competitor tables (Phase 1 fixed) | Artifact contains ONLY real measured PPL | Phase 1 done | Honesty carried into eval |

**Deprecated/outdated:**
- `signal.alarm`/`exec` timeouts in `benchmark_physics_real_weights.py` (Linux-only) — Windows-compat is handled by the `fix/honest-metrics-windows-compat` branch, **out of phase scope**; note only that the script still needs the path parameterized (D-10) regardless.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Gemma-4 E2B ships solely as `.safetensors` (README references `model.safetensors`); tokenizer must be supplied separately as `tokenizer.json` | Standard Stack / Pitfall 4 | If the safetensors bundles a tokenizer, faithfulness path changes — but default byte fallback still works for the gate |
| A2 | WikiText-2 test set is obtainable offline as committed text; full set available via HuggingFace `wikitext` | Corpus sourcing | If no offline sample committed, default run needs network — breaks reproducibility (D-12) |
| A3 | `use_unified=False` produces identical logits to `use_unified=True` (unified wraps `_legacy_forward`) | Pattern 1 | If unified adds logit transforms, forward paths diverge; verify both runs use same flag |
| A4 | Both grader runs share one resolved `test_tokens` list (tokenize once) | Pitfall 4 | If tokenized per-run, mismatch risk |

**If this table is empty:** (not applicable — see A1–A4, all LOW-risk and verifiable during planning/implementation).

## Open Questions

1. **(BLOCKER for gate) Recovery-gate direction is inverted in the locked spec.** D-07 (`recovery_ratio = ppl_compressed/ppl_base`) + D-08 (`>= 0.95`) together reject better models and accept worse ones. *What we know:* the artifact must contain both measured PPLs and a gate. *What's unclear:* the exact gate predicate the user intends. *Recommendation:* adopt `recovery_ratio = ppl_base / ppl_compressed` with gate `>= 0.95` (or keep `compressed/base` but gate `<= 1.05`). Record the formula explicitly in the artifact. **Planner must resolve before Plan 02-01.**

2. **Committed sample size vs memory.** Full 2048-window WikiText-2 over many windows may OOM even with vocab-blocking. *Recommendation:* commit a sample of N tokens (e.g. 2k–8k) sufficient for a stable PPL estimate; document `n_tokens`. Pick WikiText-2 (not 103) for the sample.

3. **Artifact directory git policy.** `eval/artifacts/` — commit or git-ignore? *Recommendation:* git-ignore emitted artifacts but commit `eval/data/` sample. Keep one reference artifact in docs for transparency.

4. **Lazy vs eager weight load for memory.** Confirm `SafeTensorsLoader`/`ModelLoader` do not eagerly materialize all fp32 weights before first forward; if they do, the sequential-load mitigation is insufficient and we need memory-mapped (already present: `MemoryMappedTensorEngine`) or per-layer streaming.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | All | ✓ (branch CPython) | 3.10+ | — |
| NumPy ≥1.24 | All tensor math | ✓ | (lockfile missing; verify `import numpy`) | — |
| SciPy ≥1.10 | (indirect) | ✓ | — | — |
| `safetensors` | Load `.safetensors` model | ✓ (core dep) | ≥0.4 | — |
| `zstandard` | `.ssf` container | ✓ (core dep) | ≥0.22 | — |
| Gemma-4 E2B weights | Actual eval run | ✗ (user-supplied) | — | Committed sample + README documents required model; grader must no-op gracefully without weights |
| Model tokenizer.json | Faithful tokenization | ✗ (user-supplied) | — | `build_default_tokenizer()` byte fallback |
| `datasets` (ml extra) | `fetch_eval_corpus.py` full download | ✗ (optional) | — | Download raw WikiText-2 text from public URL; committed sample is the default |

**Missing dependencies with no fallback:** Gemma-4 E2B weights (intrinsic — must be user-supplied; the eval cannot run end-to-end without them, but the grader + tests must run offline against the committed sample/tiny stub).

**Missing dependencies with fallback:** Model tokenizer (byte fallback), full corpus (committed sample).

## Validation Architecture

Per `.planning/config.json`, `workflow.nyquist_validation` is enabled (true) → this section is required. The eval subsystem is verified along four independent dimensions.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=7.0 (dev extra, configured in pyproject.toml) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (timeout=120, testpaths=["tests"]) |
| Quick run command | `python -m pytest tests/test_eval_subsystem.py -x -q` |
| Full suite command | `python -m pytest tests/ -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-02 | `BaseTokenizer().encode`/`decode` round-trips sample text and does not raise | unit | `python -m pytest tests/test_eval_subsystem.py::test_base_tokenizer_roundtrip -x` | ❌ Wave 0 |
| EVAL-02 | `build_default_tokenizer()` round-trips and `vocab_size` == 256 | unit | `...::test_default_tokenizer` | ❌ Wave 0 |
| EVAL-03 | `resolve_model_path` honors env > --model > relative, raises on missing | unit | `...::test_resolve_model_path` | ❌ Wave 0 |
| EVAL-03 | Hardcoded `/home/mike/...` removed from benchmark_physics_real_weights.py & wave4_pipeline.py | static/grep | `grep -rn "/home/mike" scripts/ benchmark_*.py` returns nothing | ❌ Wave 0 |
| EVAL-01 | Grader produces JSON artifact with `base_ppl`, `compressed_ppl`, `recovery_ratio`, `gate_passed`, all required D-09 fields, both PPLs real (not inf/nan when weights present) | integration (needs weights) | `python -m pytest tests/test_eval_subsystem.py::test_artifact_schema -x` | ❌ Wave 0 |
| EVAL-01 | `recovery_ratio` same direction for both runs (identical corpus/stride/seq_len) | integration | `...::test_identical_windowing` | ❌ Wave 0 |
| EVAL-01 | Gate predicate behaves correctly (worse compressed → fail; equal/better → pass) — depends on Q1 resolution | unit | `...::test_recovery_gate` | ❌ Wave 0 |
| EVAL-01 | No silent layer-dropping: `layers_loaded` recorded / asserted | integration | `...::test_layers_loaded` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_eval_subsystem.py -x -q` (focused)
- **Per wave merge:** `python -m pytest tests/ -q` (full suite, 120s timeout per test)
- **Phase gate:** Full suite green + one reference artifact emitted and committed under `docs/` for transparency.

### Wave 0 Gaps
- [ ] `tests/test_eval_subsystem.py` — covers EVAL-01/02/03
- [ ] `eval/grader.py`, `eval/run_eval.py`, `eval/corpus.py`, `eval/model_path.py`, `eval/constants.py`, `eval/artifact.py` — new modules
- [ ] `eval/data/wikitext2_sample.txt` + `wikitext2_sample.tokens.json` — committed offline sample
- [ ] `scripts/fetch_eval_corpus.py` — opt-in full download
- [ ] README update documenting Gemma-4 E2B requirement + `SPECTRALSTREAM_MODEL_PATH` / `--model`
- [ ] Patch `measure_perplexity` call (or wrapper) for vocab-blocked log-softmax if memory testing shows OOM (Pitfall 1)

## Security Domain

`security_enforcement` is enabled (config.json). This phase is a local CPU-only eval grader; ASVS relevance is limited but the path-resolution work touches a known trust gap (SEC-02).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth in eval |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | Local single-user |
| V5 Input Validation | yes | `resolve_model_path` must reject path traversal / non-existent paths; `--corpus`/`--model` args validated against traversal (mirror `cli.py:_validate_input_path` regex at cli.py:78) |
| V6 Cryptography | no | No crypto |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via `--model`/`--corpus` | Tampering | Validate resolved path exists & is within allowed roots; reject `..` segments (reuse CLI regex pattern) |
| Hardcoded author path leaking environment | Info disclosure (SEC-02) | Removed by D-10; replaced with user-supplied resolution |

## Sources

### Primary (HIGH confidence — verified against source)
- `spectralstream/inference/pipeline.py` — `InferencePipeline.__init__` (:79), `forward` (:393), `measure_perplexity` (:737), `InferenceConfig` (:44)
- `spectralstream/inference/config.py` — `Gemma4Config.VOCAB_SIZE = 262144` (:15), `MAX_POSITION_EMBEDDINGS = 131072` (:16)
- `spectralstream/utils/tokenizer_engine.py` — `BaseTokenizer` raise sites (:134-137), `build_default_tokenizer` (:2146), `_build_byte_fallback` (:439), `AutoTokenizer.from_pretrained` (:1470), `auto_tokenizer_for_model` (:2196)
- `spectralstream/compression/honest_metrics.py` — `serialized_nbytes`, `apply_gate`, `ErrorMetrics` (byte-exact convention)
- `benchmark_physics_real_weights.py:38`, `wave4_pipeline.py:165` — hardcoded path targets (D-10)
- `README.md:19` — Gemma-4 E2B = 10.2 GB / 2011 tensors; INT8 4.6× vs FP32

### Secondary (MEDIUM confidence — project research/context)
- `.planning/research/SUMMARY.md` — recovery gate ≥ 0.95 convention, eval as #1 trust gap
- `.planning/phases/02-eval-subsystem/02-CONTEXT.md` — all D-01..D-13 locked decisions
- `.planning/REQUIREMENTS.md` — EVAL-01/02/03 definitions, MISS-01/02, BUG-04, SEC-02 mappings

### Tertiary (LOW confidence — to validate during implementation)
- A1–A4 in Assumptions Log (tokenizer bundling, corpus availability, unified-forward equivalence, single-tokenize)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all components are existing in-repo modules verified by reading source; no new deps.
- Architecture: HIGH — wrapper pattern over `measure_perplexity` is direct; grader flow traced end-to-end.
- Pitfalls: HIGH — memory math (2048×262144×4) and gate-inversion are arithmetic/logical deductions from verified source.

**Research date:** 2026-07-08
**Valid until:** 2026-08-08 (30 days — stable internal API; re-verify if `measure_perplexity` signature or Gemma4Config changes).

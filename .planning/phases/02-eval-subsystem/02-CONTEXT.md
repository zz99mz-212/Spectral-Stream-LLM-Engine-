# Phase 2: Eval Subsystem - Context

**Gathered:** 2026-07-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a reproducible, independent `eval/` subsystem that proves quality is preserved by
measuring WikiText-2 perplexity (seq len 2048) on original FP16 weights vs the compressed
model, and emits a verifiable JSON artifact with both PPL values plus a recovery ratio
gated by `compressed/base ≥ threshold`. This closes the project's single biggest trust gap
(no perplexity/downstream eval exists at all — `MISS-01`).

This phase also: (a) ships a working default tokenizer so `BaseTokenizer.encode` no longer
raises `NotImplementedError` (EVAL-02 / BUG-04), and (b) replaces the hardcoded
`/home/mike/.../gemma-4-E2B` model path with env/CLI configuration and documents the
required model for reproduction (EVAL-03 / SEC-02).

Depends on Phase 1: eval reports must reuse the honest-metrics conventions (byte-exact,
real measured values, no estimates). Out of scope: calibration (Phase 3/4), INT4 quality
proof (Phase 4), cascade correctness (Phase 3), registry/format work (Phase 6/7).
</domain>

<decisions>
## Implementation Decisions

### Default Tokenizer to Ship (EVAL-02 / BUG-04)
- **D-01:** `BaseTokenizer.encode` / `decode` MUST NOT raise `NotImplementedError`. Give the
  abstract base a byte-level fallback implementation (byte → token id via the byte vocab /
  `chr` mapping) so the base class never raises by default. `build_default_tokenizer()`
  (byte-level BPE, 256 tokens — already at `tokenizer_engine.py:2146`) becomes the shipped
  default returned by `AutoTokenizer` when no model tokenizer is available.
- **D-02:** For the eval subsystem's *actual* perplexity tokenization, reuse the model's OWN
  tokenizer loaded from the Gemma-4 model via the existing `AutoTokenizer.from_gguf` /
  `auto_tokenizer_for_model` (SentencePiece / BPE / GGUF). Do NOT pull in `tiktoken`,
  `transformers`, or `sentencepiece` pip packages — that would cross the pure-Python /
  no-torch boundary. This keeps the eval faithful to the reference token distribution.
- **D-03:** A test asserts `encode` → `decode` round-trips on sample text for the default
  tokenizer (and ideally the model tokenizer). The bar is round-trip correctness, not
  linguistic fidelity.

### Eval Execution Model — true perplexity source (Decisions Needed #2)
- **D-04:** EVAL-01 uses the NATIVE in-core NumPy forward pass via
  `InferencePipeline.measure_perplexity(test_tokens, stride, max_seq_len)`
  (`pipeline.py:737`) — real logit-based sliding-window perplexity. NO `lm_eval` / torch
  subprocess oracle: it would violate the pure-Python/no-torch constraint and add a heavy
  dev dependency. The eval layer *wraps* `InferencePipeline`; it does not re-implement the
  forward pass.
- **D-05:** `InferencePipeline` already provides the forward pass + `measure_perplexity`.
  The eval subsystem adds the thin "independent grader": corpus load → tokenize → run on
  original (FP16/safetensors) → run on compressed (.ssf) → record PPL → compute recovery
  ratio → enforce gate → emit JSON artifact.

### Recovery Gate & Artifact (EVAL-01 / success criteria #1, #4)
- **D-06:** seq_len = 2048 (per ROADMAP). Use `measure_perplexity`'s sliding window; default
  stride = 512 (the method's existing default), configurable.
- **D-07:** `recovery_ratio = ppl_compressed / ppl_base` — matches the requirement's literal
  `compressed/base`. Gate passes when `recovery_ratio >= threshold`.
- **D-08:** Default threshold = 0.95 (research recovery ≥ 0.95). Configurable via a constant
  (e.g. `RECOVERY_GATE_THRESHOLD`) plus optional env/CLI override. The verified INT8 baseline
  should clear it comfortably and documents the in-repo quality bar.
- **D-09:** Artifact is a JSON file with fields: `model`, `method`, `tokenizer`, `base_ppl`,
  `compressed_ppl`, `recovery_ratio`, `recovery_gate_threshold`, `gate_passed`, `seq_len`,
  `stride`, `n_tokens`, `timestamp`, `git_ref`. Both PPL values are REAL MEASURED numbers —
  no estimates, no proxies. Written under `eval/artifacts/` (or `eval/results/`).

### Model Loading & Path (EVAL-03 / SEC-02)
- **D-10:** Remove hardcoded absolute paths (`benchmark_physics_real_weights.py:38`,
  `wave4_pipeline.py:165`) and replace with `SPECTRALSTREAM_MODEL_PATH` env var + `--model`
  CLI arg. `InferencePipeline` already takes `model_path`, so thread it through the eval
  entry point and parameterize the scripts.
- **D-11:** README documents the required model (Gemma-4 E2B) and how a user supplies it
  (local `.safetensors` path they provide). Resolution order: env var → `--model` → relative
  `models/gemma-4-E2B/model.safetensors` fallback — never a hardcoded absolute author path.

### WikiText-2 Corpus Sourcing (determinism / reproducibility)
- **D-12:** The eval needs a tokenized WikiText-2 (or WikiText-103) test corpus. Commit a
  small, deterministic sampled slice of WikiText-2 test tokens to the repo (e.g.
  `eval/data/wikitext2_sample.*`) so the default run is offline and reproducible. Provide a
  `scripts/fetch_eval_corpus.py` that downloads the full WikiText-2 test set, plus a
  `--corpus PATH` CLI/env override.
- **D-13:** Corpus tokenization uses the model's own tokenizer (D-02). The committed sample is
  pre-tokenized for the default model to avoid requiring the model at corpus-load time; commit
  the raw text too for transparency.

### Claude's Discretion
- Exact artifact filename/path, the env-var name (`SPECTRALSTREAM_MODEL_PATH` vs a shorter
  form), default stride, exact sample size, and whether to fully support the 2048-token window
  under memory limits — all at Claude's discretion per project conventions.
- WikiText-2 vs WikiText-103 — pick whichever the committed sample represents; document it.
- New module/file names under `eval/` (e.g. `eval/evaluator.py`, `eval/run_eval.py`) and the
  CLI subcommand name — Claude's discretion, following existing `snake_case` CLI conventions.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap & Requirements
- `.planning/ROADMAP.md` § "Phase 2: Eval Subsystem" — goal, 4 success criteria, depends on Phase 1
- `.planning/REQUIREMENTS.md` "Eval Subsystem" — EVAL-01, EVAL-02, EVAL-03 definitions
- `.planning/PROJECT.md` "Constraints" — pure Python (no torch), metrics honesty, "No eval baseline" gap
- `.planning/PROJECT.md` "Validated" — INT8 ~4.6× vs FP32 on Gemma-4 E2B is the baseline quality reference

### Research (Decisions Needed)
- `.planning/research/SUMMARY.md` "Decisions Needed" #1 (tokenizer strategy) and #2 (eval execution model) — both land in this phase
- `.planning/research/SUMMARY.md` "Key Findings" #1 — eval subsystem is the #1 trust gap; recovery gate ≥ 0.95

### Code (reusable assets / integration points)
- `spectralstream/inference/pipeline.py:79` — `InferencePipeline.__init__(model_path, ...)`; already parameterizable
- `spectralstream/inference/pipeline.py:737` — `measure_perplexity(test_tokens, stride=512, max_seq_len)`; real logit PPL
- `spectralstream/inference/pipeline.py:393` — `forward(tokens) -> logits`; reused by the eval layer
- `spectralstream/utils/tokenizer_engine.py:110` — `BaseTokenizer` (raises `NotImplementedError` at :134 — EVAL-02 target)
- `spectralstream/utils/tokenizer_engine.py:2146` — `build_default_tokenizer()` (byte-level BPE default)
- `spectralstream/utils/tokenizer_engine.py:1444` — `AutoTokenizer.from_gguf` / `auto_tokenizer_for_model` (model tokenizer)
- `benchmark_physics_real_weights.py:38` — hardcoded `/home/mike/.../gemma-4-E2B` path (EVAL-03 target)
- `wave4_pipeline.py:165` — hardcoded `models/gemma-4-E2B/model.safetensors` (parameterize per EVAL-03)

### Honesty / format conventions
- `spectralstream/compression/honest_metrics.py` — byte-exact reporting conventions to mirror in the artifact

[No external SPEC.md — requirements fully captured above from ROADMAP.md / REQUIREMENTS.md]
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `spectralstream/inference/pipeline.py` — `InferencePipeline` already loads SSF/safetensors,
  runs a Gemma-4 forward pass, and exposes `measure_perplexity()` producing TRUE logit-based
  perplexity via sliding window. The eval subsystem wraps this; no new math needed.
- `spectralstream/utils/tokenizer_engine.py` — `BaseTokenizer` ABC, `BPETokenizer`,
  `SentencePieceTokenizer`, `TiktokenTokenizer`, `AutoTokenizer` (GGUF/SentencePiece/BPE
  detection), and `build_default_tokenizer()` (byte-level fallback). The core tokenizer exists;
  only the abstract base's `NotImplementedError` and a "shipped default" need fixing.
- `spectralstream/compression/honest_metrics.py` — error-gated, byte-exact reporting pattern
  to follow for the eval artifact (real measured values, gated, no estimates).

### Established Patterns
- `from __future__ import annotations` mandatory on every module; `snake_case` functions;
  `tests/test_*.py` in top-level `tests/`.
- CLI additions follow `spectralstream/compression/cli.py` conventions (subcommands, argparse).
- `InferenceConfig` dataclass drives pipeline behavior — extend it for eval flags if needed.

### Integration Points
- Eval entry point instantiates `InferencePipeline` twice (original vs compressed) — reuse
  `model_path` param, add env/CLI `SPECTRALSTREAM_MODEL_PATH` / `--model`.
- Tokenization: `AutoTokenizer.from_gguf(model)` for the default model's tokenizer.
- Hardcoded paths: `benchmark_physics_real_weights.py`, `wave4_pipeline.py` — parameterize.
- README: document the required model + how to reproduce the headline PPL.
</code_context>

<specifics>
## Specific Ideas

- `recovery_ratio` definition is `ppl_compressed / ppl_base` (literal "compressed/base"); gate
  passes when `recovery_ratio >= 0.95`. This is the single most important honesty check.
- seq_len = 2048 is fixed by the roadmap success criterion; do not change it without revisiting ROADMAP.
- The artifact must contain BOTH `base_ppl` and `compressed_ppl` as measured values — the
  recovery ratio is derived, never supplied.
</specifics>

<deferred>
## Deferred Ideas

- **CI perplexity gate** — deferred to v2 (CI-01 / MISS-04). Eval subsystem lands first; wiring
  it into CI is a later phase.
- **lm-eval-harness oracle** — explicitly NOT chosen (D-04): would add torch, violating the
  pure-Python constraint. If ever needed, it is its own phase, not this one.
- **INT4 quality proof** — belongs to Phase 4 (calibration) + validation via this eval subsystem.
- **Full WikiText-2 vs small sample** — full corpus is opt-in via `fetch_eval_corpus.py`; the
  default committed sample keeps runs offline/reproducible.

None — discussion stayed within phase scope.
</deferred>

---

*Phase: 2-Eval Subsystem*
*Context gathered: 2026-07-08*

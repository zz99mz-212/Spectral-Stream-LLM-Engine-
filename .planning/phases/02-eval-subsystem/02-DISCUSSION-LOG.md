# Phase 2: Eval Subsystem - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-08
**Phase:** 2-eval-subsystem
**Areas discussed:** Default Tokenizer, Eval Execution Model, Recovery Gate & Artifact, Model Loading & Path, WikiText-2 Corpus Sourcing

---

## Default Tokenizer (EVAL-02 / BUG-04)

| Option | Description | Selected |
|--------|-------------|----------|
| Byte-level fallback as `BaseTokenizer` default + `build_default_tokenizer` as shipped default | `BaseTokenizer.encode/decode` get a byte-level fallback so they never raise; `build_default_tokenizer()` (byte-level BPE) is the shipped `AutoTokenizer` default. Model's own GGUF/SentencePiece tokenizer reused for actual eval. | ✓ |
| Pull in `tiktoken` / `transformers` / vendored sentencepiece | Real subword tokenizer, but crosses the pure-Python/no-torch boundary and adds deps. | |
| Ship only the abstract `BaseTokenizer` untouched | Leaves EVAL-02's "encode no longer raises" requirement unmet. | |

**User's choice:** Auto-selected recommended option (--auto mode).
**Notes:** EVAL-02 requires `BaseTokenizer.encode` to stop raising `NotImplementedError`; the model's own tokenizer (from Gemma-4 GGUF) is used for faithful eval PPL, not a generic subword tokenizer.

## Eval Execution Model — true perplexity source

| Option | Description | Selected |
|--------|-------------|----------|
| Native in-core NumPy via `InferencePipeline.measure_perplexity` | Reuse existing Gemma-4 forward pass + real logit PPL; no torch; honors pure-Python constraint. | ✓ |
| External `lm_eval` subprocess oracle | Cleaner harness API but requires torch; violates the no-torch constraint. | |

**User's choice:** Auto-selected recommended option (--auto mode).
**Notes:** `measure_perplexity` (pipeline.py:737) already exists; the eval layer is a thin wrapper, not a new forward pass.

## Recovery Gate & Artifact (EVAL-01 / success #1,#4)

| Option | Description | Selected |
|--------|-------------|----------|
| `recovery_ratio = ppl_compressed / ppl_base`, gate passes when `>= 0.95` (configurable) | Matches the requirement's literal `compressed/base`; 0.95 from research recovery gate. | ✓ |
| Inverse ratio `ppl_base / ppl_compressed` | Also viable but does not match the requirement wording. | |
| No recovery gate | Fails success criterion #4. | |

**User's choice:** Auto-selected recommended option (--auto mode).
**Notes:** seq_len fixed at 2048 by roadmap; both PPL values must be real measured, not estimated.

## Model Loading & Path (EVAL-03 / SEC-02)

| Option | Description | Selected |
|--------|-------------|----------|
| `SPECTRALSTREAM_MODEL_PATH` env + `--model` CLI arg, README documents required model | Replaces hardcoded `/home/mike/.../gemma-4-E2B`; threads `model_path` already accepted by `InferencePipeline`. | ✓ |
| Keep a hardcoded default path | Violates SEC-02 / reproducibility for fresh clones. | |

**User's choice:** Auto-selected recommended option (--auto mode).
**Notes:** Hardcoded paths in `benchmark_physics_real_weights.py:38` and `wave4_pipeline.py:165` must be parameterized.

## WikiText-2 Corpus Sourcing

| Option | Description | Selected |
|--------|-------------|----------|
| Commit a small deterministic sampled slice + `fetch_eval_corpus.py` for full set + `--corpus` override | Offline reproducible default; full corpus opt-in. | ✓ |
| Require network download of full WikiText-2 at run time | Breaks offline/reproducible runs. | |
| User must hand-provide corpus every time | Poor default DX. | |

**User's choice:** Auto-selected recommended option (--auto mode).
**Notes:** Committed sample pre-tokenized for the default model; raw text committed for transparency.

## Claude's Discretion

- Artifact filename/path, env-var name, default stride, sample size, full 2048-window memory handling.
- WikiText-2 vs WikiText-103 choice (document whichever the sample represents).
- New `eval/` module/file names and CLI subcommand name.

## Deferred Ideas

- CI perplexity gate → v2 (CI-01 / MISS-04).
- lm-eval-harness oracle → rejected (adds torch); own phase if ever needed.
- INT4 quality proof → Phase 4 (calibration) + validation via this eval subsystem.
- Full WikiText-2 corpus → opt-in via `fetch_eval_corpus.py`.

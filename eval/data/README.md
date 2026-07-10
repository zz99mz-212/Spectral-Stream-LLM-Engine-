# Eval Corpus Data

This directory contains the committed WikiText-2 sample files used by the
eval subsystem for offline/reproducible runs.

## Files

- `wikitext2_sample.txt` -- Small raw-text slice of WikiText-2 test (a
  self-authored didactic sample, not the full WikiText-2 distribution).
  Do NOT use this for community-comparable PPL numbers; it is a small
  smoke-test corpus for verifying the eval pipeline works.

- `wikitext2_sample.tokens.json` -- Byte-level tokenization of the above
  raw-text sample produced by `build_default_tokenizer()` (the
  `BaseTokenizer` byte-level fallback, `vocab_size=256`). This is the
  **offline fallback** so default runs (`python -m eval.run_eval --model
  ... --compressed ...`) do not require a model tokenizer file.

## Default Behavior (D-12 / D-13)

When `--tokenizer` is omitted and no `--corpus` is given, `resolve_corpus`
first looks for `wikitext2_sample.tokens.json` and loads the pre-tokenized
byte-level IDs. If that file is missing it falls back to the raw text and
uses `build_default_tokenizer()` (byte-level).

This means default runs are **offline and reproducible** without any model
tokenizer file -- but the absolute PPL values will **not** match what a
community-standard implementation (e.g. HuggingFace `transformers` for
Gemma-4) would produce, because the token IDs are byte-level, not the
model's native SentencePiece / BPE tokens.

## Model-Faithful PPL (D-02)

For absolute PPL values that are comparable to community-standard
implementations:

1. Supply a real Gemma-4 `tokenizer.json` or `.gguf` file via
   `--tokenizer path/to/tokenizer.json`.
2. Supply raw text (NOT the committed JSON sample) via
   `--corpus path/to/wikitext2-test.txt`.
3. `eval/run_eval.py` will use `AutoTokenizer.from_pretrained` (for
   `.json`) or `AutoTokenizer.from_gguf` (for `.gguf`) to load the
   model-native tokenizer and encode the raw text.

The model-native pre-tokenization cannot be committed because the model
file (which contains the tokenizer data) is large and git-ignored per
project conventions.

## Key Locked Decisions

| ID | Summary |
|----|---------|
| D-02 | Model-native tokenizer used when `--tokenizer` is supplied; byte-level fallback otherwise |
| D-12 | Offline default: committed sample enables reproducible runs without a model |
| D-13 | Committed sample is byte-level (not model-native); raw text also committed for transparency |

## References

- `eval/corpus.py` -- `resolve_corpus(corpus_path, tokenizer=None)` honors
  the injected tokenizer for raw text; JSON paths ignore the tokenizer arg.
- `eval/run_eval.py` -- CLI dispatches to `AutoTokenizer.from_pretrained` /
  `from_gguf` when `--tokenizer` is supplied.
- `scripts/fetch_eval_corpus.py` -- Downloads the full WikiText-2 test set
  via stdlib `urllib` (no external dependencies).

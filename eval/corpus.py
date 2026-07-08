"""
Corpus resolution for the eval subsystem.

Supports two input types:
- ``.json`` files: loaded as a ``list[int]`` of pre-tokenized token ids.
- Raw text files (any other extension): encoded with the byte-level default
  tokenizer via ``build_default_tokenizer()``.

The default corpus is the committed ``wikitext2_sample.tokens.json`` sample
(see D-12 / D-13).
"""

from __future__ import annotations

import json
import os

from eval.constants import DEFAULT_SAMPLE_TOKENS, DEFAULT_SAMPLE_TXT


def resolve_corpus(corpus_path: str | None = None) -> list[int]:
    """Resolve a token-list corpus from *corpus_path*.

    Parameters
    ----------
    corpus_path : str or None
        Path to a ``.json`` (pre-tokenized) or raw-text corpus file.
        If ``None``, the default sample is used.

    Returns
    -------
    list[int]
        Token ids (for all subsequent eval runs).
    """
    path: str = corpus_path if corpus_path else ""

    if not path:
        # Prefer the pre-tokenized sample; fall back to raw text
        if os.path.exists(DEFAULT_SAMPLE_TOKENS):
            path = DEFAULT_SAMPLE_TOKENS
        elif os.path.exists(DEFAULT_SAMPLE_TXT):
            path = DEFAULT_SAMPLE_TXT
        else:
            raise FileNotFoundError(
                f"Default sample not found at {DEFAULT_SAMPLE_TOKENS!r} "
                f"or {DEFAULT_SAMPLE_TXT!r}"
            )
    else:
        # Validate the user-supplied path exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corpus file not found: {path}")

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            ids: list[int] = json.load(f)
        if not isinstance(ids, list) or (ids and not isinstance(ids[0], int)):
            raise ValueError(f"JSON corpus must be a list of ints, got {type(ids)}")
        return ids

    # Raw text → byte-level tokenization
    from spectralstream.utils.tokenizer_engine import build_default_tokenizer

    tokenizer = build_default_tokenizer()
    with open(path, "r", encoding="utf-8") as f:
        text: str = f.read()
    encoded = tokenizer.encode(text)
    return encoded

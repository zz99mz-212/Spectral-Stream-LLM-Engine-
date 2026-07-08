"""
Corpus resolution for the eval subsystem.

Supports two input types:
- ``.json`` files: loaded as a ``list[int]`` of pre-tokenized token ids.
  The ``tokenizer`` argument is ignored for JSON paths — pre-tokenized data
  is returned as-is (see D-12 offline fallback).
- Raw text files (any other extension): encoded with the supplied tokenizer
  (model-native via ``--tokenizer``) or the byte-level default tokenizer
  (``build_default_tokenizer()``) when ``tokenizer`` is ``None``.

The default corpus is the committed ``wikitext2_sample.tokens.json`` sample
(see D-12 / D-13).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from eval.constants import DEFAULT_SAMPLE_TOKENS, DEFAULT_SAMPLE_TXT

if TYPE_CHECKING:
    from spectralstream.utils.tokenizer_engine import BaseTokenizer


def resolve_corpus(
    corpus_path: str | None = None,
    tokenizer: BaseTokenizer | None = None,
) -> list[int]:
    """Resolve a token-list corpus from *corpus_path*.

    Parameters
    ----------
    corpus_path : str or None
        Path to a ``.json`` (pre-tokenized) or raw-text corpus file.
        If ``None``, the default sample is used.
    tokenizer : BaseTokenizer or None
        Tokenizer with an ``encode(text) -> list[int]`` method. Used for
        raw-text files only. When ``None``, the byte-level default tokenizer
        (``build_default_tokenizer``) is used as the offline fallback.
        Ignored for ``.json`` paths (pre-tokenized data is returned as-is).

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

    # Raw text → tokenization (model-native if supplied, byte-level fallback)
    if tokenizer is None:
        from spectralstream.utils.tokenizer_engine import build_default_tokenizer

        tokenizer = build_default_tokenizer()
    with open(path, "r", encoding="utf-8") as f:
        text: str = f.read()
    return tokenizer.encode(text)

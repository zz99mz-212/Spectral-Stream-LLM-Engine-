"""
Tests for the BaseTokenizer byte-identity fallback and default tokenizer
round-trip (EVAL-02).

Ensures:
- ``BaseTokenizer().encode(text)`` returns a ``list[int]`` of UTF-8 byte values
  and never raises ``NotImplementedError``.
- ``BaseTokenizer().decode(token_ids)`` returns the original text and never
  raises ``NotImplementedError``.
- ``build_default_tokenizer()`` has ``vocab_size == 256`` and round-trips text.
"""

from __future__ import annotations

import pytest

from spectralstream.utils.tokenizer_engine import BaseTokenizer, build_default_tokenizer

# ── Sample texts ─────────────────────────────────────────────────────────

ASCII_SAMPLE = "hello"
ASCII_EXPECTED_ENCODE = [104, 101, 108, 108, 111]

MULTIBYTE_SAMPLE = "Café"  # cafe + combining acute accent
EMOJI_SAMPLE = "\U0001f600"  # grinning face


# ── BaseTokenizer byte-identity fallback ────────────────────────────────


class TestBaseTokenizerFallback:
    """BaseTokenizer.encode/decode must not raise NotImplementedError."""

    def test_base_tokenizer_encode_returns_byte_ids(self):
        tok = BaseTokenizer()
        result = tok.encode(ASCII_SAMPLE)
        assert isinstance(result, list), "encode must return a list"
        assert all(isinstance(t, int) for t in result), "all elements must be ints"
        assert result == ASCII_EXPECTED_ENCODE, (
            f"encode({ASCII_SAMPLE!r}) expected {ASCII_EXPECTED_ENCODE}, got {result}"
        )

    def test_base_tokenizer_decode_roundtrips_ascii(self):
        tok = BaseTokenizer()
        result = tok.decode(ASCII_EXPECTED_ENCODE)
        assert result == ASCII_SAMPLE, (
            f"decode({ASCII_EXPECTED_ENCODE}) expected {ASCII_SAMPLE!r}, got {result!r}"
        )

    def test_base_tokenizer_roundtrip_ascii(self):
        tok = BaseTokenizer()
        token_ids = tok.encode(ASCII_SAMPLE)
        decoded = tok.decode(token_ids)
        assert decoded == ASCII_SAMPLE, (
            f"round-trip failed: encode({ASCII_SAMPLE!r}) -> {token_ids} -> {decoded!r}"
        )

    def test_base_tokenizer_roundtrip_multibyte(self):
        tok = BaseTokenizer()
        token_ids = tok.encode(MULTIBYTE_SAMPLE)
        decoded = tok.decode(token_ids)
        assert decoded == MULTIBYTE_SAMPLE, (
            f"round-trip failed for multibyte: "
            f"encode({MULTIBYTE_SAMPLE!r}) -> {token_ids} -> {decoded!r}"
        )

    def test_base_tokenizer_roundtrip_emoji(self):
        tok = BaseTokenizer()
        token_ids = tok.encode(EMOJI_SAMPLE)
        decoded = tok.decode(token_ids)
        assert decoded == EMOJI_SAMPLE, (
            f"round-trip failed for emoji: "
            f"encode({EMOJI_SAMPLE!r}) -> {token_ids} -> {decoded!r}"
        )

    def test_base_tokenizer_does_not_raise(self):
        """Calling encode/decode on BaseTokenizer does not raise NotImplementedError."""
        tok = BaseTokenizer()
        # encode
        ids = tok.encode("test")
        assert isinstance(ids, list)
        # decode
        text = tok.decode([116, 101, 115, 116])
        assert text == "test"


# ── build_default_tokenizer ──────────────────────────────────────────────


class TestBuildDefaultTokenizer:
    """build_default_tokenizer() must have vocab_size == 256 and round-trip."""

    def test_default_tokenizer_vocab_size(self):
        tok = build_default_tokenizer()
        assert tok.vocab_size == 256, (
            f"expected vocab_size=256, got {tok.vocab_size}"
        )

    def test_default_tokenizer_roundtrip_ascii(self):
        tok = build_default_tokenizer()
        token_ids = tok.encode(ASCII_SAMPLE)
        decoded = tok.decode(token_ids)
        assert decoded == ASCII_SAMPLE, (
            f"default tokenizer round-trip failed for ascii: "
            f"encode({ASCII_SAMPLE!r}) -> {token_ids} -> {decoded!r}"
        )

    def test_default_tokenizer_roundtrip_multibyte(self):
        tok = build_default_tokenizer()
        token_ids = tok.encode(MULTIBYTE_SAMPLE)
        decoded = tok.decode(token_ids)
        assert decoded == MULTIBYTE_SAMPLE, (
            f"default tokenizer round-trip failed for multibyte: "
            f"encode({MULTIBYTE_SAMPLE!r}) -> {token_ids} -> {decoded!r}"
        )

    def test_default_tokenizer_roundtrip_emoji(self):
        tok = build_default_tokenizer()
        token_ids = tok.encode(EMOJI_SAMPLE)
        decoded = tok.decode(token_ids)
        assert decoded == EMOJI_SAMPLE, (
            f"default tokenizer round-trip failed for emoji: "
            f"encode({EMOJI_SAMPLE!r}) -> {token_ids} -> {decoded!r}"
        )

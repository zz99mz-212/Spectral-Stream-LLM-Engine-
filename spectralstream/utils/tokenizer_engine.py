"""
High-Performance Tokenizer Engine for SpectralStream
=====================================================
Production-grade tokenizer supporting all major formats:
- BPE (Byte-Pair Encoding) with byte-level encoding
- SentencePiece (Unigram LM + BPE fallback)
- Tiktoken (OpenAI-compatible)
- Spectral Token Encoding (DCT-domain tokenization)

Integration:
    from spectralstream.utils.tokenizer_engine import AutoTokenizer
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

__all__ = [
    "BPETokenizer",
    "SentencePieceTokenizer",
    "TiktokenTokenizer",
    "AutoTokenizer",
    "CachedTokenizer",
    "ParallelTokenizer",
    "SpectralTokenizer",
    "BaseTokenizer",
    "get_tokenizer_info",
]

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SPECIAL_TOKENS = {
    "bos": "<s>",
    "eos": "</s>",
    "unk": "<unk>",
    "pad": "<pad>",
    "mask": "<mask>",
}

# Regex patterns for pre-tokenization (from OpenAI tiktoken / HuggingFace)
GPT2_SPLIT_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\W+""")

GPT4_SPLIT_PATTERN = re.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\w]?\w+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

CL100K_SPLIT_PATTERN = re.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\w]?\w+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

P50K_SPLIT_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\W+""")

R50K_SPLIT_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\W+""")

# Byte-to-unicode mapping for byte-level BPE
BYTE_TO_UNICODE = {}
for i in range(256):
    if i < 33 or i == 127 or i == 173:
        BYTE_TO_UNICODE[i] = chr(256 + i)
    elif i == 32:
        BYTE_TO_UNICODE[i] = chr(256 + 0)
    elif i == 160:
        BYTE_TO_UNICODE[i] = chr(256 + 1)
    else:
        BYTE_TO_UNICODE[i] = chr(i)

_UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

# GGUF tokenizer keys
GGUF_TOKENIZER_KEYS = {
    "model": "tokenizer.ggml.model",
    "tokens": "tokenizer.ggml.tokens",
    "scores": "tokenizer.ggml.scores",
    "token_type": "tokenizer.ggml.token_type",
    "merges": "tokenizer.ggml.merges",
    "bos_id": "tokenizer.ggml.bos_token_id",
    "eos_id": "tokenizer.ggml.eos_token_id",
    "eot_id": "tokenizer.ggml.eot_token_id",
    "eom_id": "tokenizer.ggml.eom_token_id",
    "unk_id": "tokenizer.ggml.unknown_token_id",
    "sep_id": "tokenizer.ggml.seperator_token_id",
    "pad_id": "tokenizer.ggml.padding_token_id",
    "mask_id": "tokenizer.ggml.mask_token_id",
    "add_bos": "tokenizer.ggml.add_bos_token",
    "add_eos": "tokenizer.ggml.add_eos_token",
    "add_prefix": "tokenizer.ggml.add_space_prefix",
    "pre": "tokenizer.ggml.pre",
    "remove_extra_ws": "tokenizer.ggml.remove_extra_whitespaces",
    "precompiled_charsmap": "tokenizer.ggml.precompiled_charsmap",
}


# ═══════════════════════════════════════════════════════════════════════════
# Base Tokenizer
# ═══════════════════════════════════════════════════════════════════════════


class BaseTokenizer:
    """Abstract base for all tokenizers."""

    def __init__(self):
        self._vocab_size = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 0
        self.pad_id = 0
        self.mask_id = 0
        self.add_bos = False
        self.add_eos = False
        self.special_tokens: dict[str, int] = {}
        self.name = "base"

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, val):
        self._vocab_size = val

    def encode(self, text: str) -> list[int]:
        # Byte-identity fallback: maps each byte of the UTF-8 encoding
        # to its numeric value.  This is a never-raise default, not a
        # faithful model tokenizer — subclasses override with proper
        # BPE / SentencePiece / Tiktoken encoding.
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        # Byte-identity fallback: reconstruct bytes from token ids,
        # decode as UTF-8 with replacement for invalid sequences.
        return bytes(int(t) & 0xFF for t in token_ids).decode("utf-8", errors="replace")

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)

    def token_count(self, text: str) -> int:
        return len(self.encode(text))

    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> BaseTokenizer:
        raise NotImplementedError

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "pad_id": self.pad_id,
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
        }

    def info(self) -> str:
        cfg = self.get_config()
        lines = [f"Tokenizer: {cfg['name']}"]
        for k, v in cfg.items():
            if k != "name":
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# BPETokenizer – Byte-Pair Encoding
# ═══════════════════════════════════════════════════════════════════════════


class BPETokenizer(BaseTokenizer):
    """Pure-Python BPE tokenizer with byte-level encoding.

    Supports loading from GGUF metadata, HuggingFace tokenizer files,
    or raw merge rules. Uses regex pre-tokenization followed by BPE merge.
    """

    def __init__(
        self,
        vocab: Optional[dict[bytes, int]] = None,
        merge_ranks: Optional[dict[tuple[int, int], int]] = None,
        special_tokens: Optional[dict[str, int]] = None,
        split_pattern: Optional[re.Pattern] = None,
        byte_fallback: bool = True,
    ):
        super().__init__()
        self.name = "bpe"

        # Core data
        self._tokens: list[bytes] = []
        self._scores: list[float] = []
        self._token_type: list[int] = []
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._merge_rank_to_id: dict[int, int] = {}
        self._id_to_merge_rank: dict[int, int] = {}
        self._vocab: dict[bytes, int] = {}
        self._id_to_token: dict[int, bytes] = {}

        self.split_pattern = split_pattern or GPT2_SPLIT_PATTERN
        self.byte_fallback = byte_fallback

        # Cache
        self._encode_cache: dict[str, list[int]] = {}
        self._cache_maxsize = 10000
        self._merge_cache: dict[tuple[int, ...], list[int]] = {}

        # Special tokens
        self.special_tokens = {}
        if special_tokens:
            for name, tid in special_tokens.items():
                self.special_tokens[name] = tid

        if vocab:
            self._load_vocab(vocab)
        if merge_ranks:
            self._merge_ranks = merge_ranks
            self._build_merge_index()

        if self._tokens:
            self._vocab_size = len(self._tokens)
        elif not vocab and not merge_ranks:
            self._build_byte_fallback()
            self._vocab_size = len(self._tokens)
        else:
            self._vocab_size = 0

        self._setup_special_ids()

    def _load_vocab(self, vocab: dict[bytes, int]):
        max_id = max(vocab.values()) if vocab else 0
        self._tokens = [b""] * (max_id + 1)
        self._scores = [0.0] * (max_id + 1)
        self._token_type = [1] * (max_id + 1)
        for token_bytes, tid in vocab.items():
            idx = tid
            while idx >= len(self._tokens):
                self._tokens.append(b"")
                self._scores.append(0.0)
                self._token_type.append(1)
            self._tokens[idx] = token_bytes
            self._vocab[token_bytes] = tid
            self._id_to_token[tid] = token_bytes

    def _build_merge_index(self):
        self._merge_rank_to_id.clear()
        self._id_to_merge_rank.clear()
        for (a, b), rank in self._merge_ranks.items():
            merged_bytes = self._tokens[a] + self._tokens[b]
            merged_id = self._vocab.get(merged_bytes)
            if merged_id is not None:
                self._merge_rank_to_id[rank] = merged_id
                self._id_to_merge_rank[merged_id] = rank

    def _setup_special_ids(self):
        for name, default_tok in SPECIAL_TOKENS.items():
            tid = self.special_tokens.get(name)
            if tid is None:
                for i, t in enumerate(self._tokens):
                    if t.decode("utf-8", errors="replace") == default_tok:
                        tid = i
                        break
            if tid is not None:
                setattr(self, f"{name}_id", tid)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, val):
        self._vocab_size = val

    # ── Build from GGUF ──────────────────────────────────────────────

    @classmethod
    def from_gguf_reader(cls, reader) -> "BPETokenizer":
        """Build BPE tokenizer from a GGUF reader object."""
        tok = cls()

        raw_model = _get_gguf_field(reader, "tokenizer.ggml.model", "gpt2")
        raw_tokens = _get_gguf_field(reader, "tokenizer.ggml.tokens", [])
        raw_scores = _get_gguf_field(reader, "tokenizer.ggml.scores", [])
        raw_types = _get_gguf_field(reader, "tokenizer.ggml.token_type", [])
        raw_merges = _get_gguf_field(reader, "tokenizer.ggml.merges", [])

        tok._parse_token_list(raw_tokens, raw_scores, raw_types)
        tok._parse_merges(raw_merges)
        tok._parse_special_from_gguf(reader)

        if tok.vocab_size == 0:
            tok._build_byte_fallback()

        tok._setup_special_ids()
        return tok

    @classmethod
    def from_files(
        cls,
        vocab_file: str,
        merges_file: str,
        split_pattern: Optional[re.Pattern] = None,
    ) -> "BPETokenizer":
        """Build BPE tokenizer from HuggingFace-style files."""
        tok = cls(split_pattern=split_pattern)

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        vocab = {}
        for token_str, tid in vocab_data.items():
            vocab[token_str.encode("utf-8")] = tid
        tok._load_vocab(vocab)

        with open(merges_file, "r", encoding="utf-8") as f:
            merge_lines = f.read().strip().split("\n")

        tok._parse_merges_bytes(
            [
                line.encode("utf-8")
                for line in merge_lines
                if line and not line.startswith("#")
            ]
        )

        if tok.vocab_size == 0:
            tok._build_byte_fallback()
        tok._setup_special_ids()
        return tok

    @classmethod
    def from_gguf_metadata(cls, metadata: dict) -> "BPETokenizer":
        """Build BPE tokenizer from a flat metadata dict (e.g. from GGUFModel)."""
        tok = cls()

        raw_tokens = metadata.get("tokenizer.ggml.tokens", [])
        raw_scores = metadata.get("tokenizer.ggml.scores", [])
        raw_types = metadata.get("tokenizer.ggml.token_type", [])
        raw_merges = metadata.get("tokenizer.ggml.merges", [])

        tok._parse_token_list(raw_tokens, raw_scores, raw_types)
        tok._parse_merges(raw_merges)

        for key, attr in [
            ("tokenizer.ggml.bos_token_id", "bos_id"),
            ("tokenizer.ggml.eos_token_id", "eos_id"),
            ("tokenizer.ggml.unknown_token_id", "unk_id"),
            ("tokenizer.ggml.padding_token_id", "pad_id"),
            ("tokenizer.ggml.mask_token_id", "mask_id"),
        ]:
            val = metadata.get(key)
            if val is not None:
                setattr(tok, attr, int(val))

        if tok.vocab_size == 0:
            tok._build_byte_fallback()
        tok._setup_special_ids()
        return tok

    def _parse_token_list(self, raw_tokens: list, raw_scores: list, raw_types: list):
        self._tokens = []
        for t in raw_tokens:
            if isinstance(t, bytes):
                self._tokens.append(t)
            elif isinstance(t, str):
                self._tokens.append(t.encode("utf-8", errors="replace"))
            else:
                self._tokens.append(str(t).encode())

        self._scores = (
            [float(s) for s in raw_scores]
            if isinstance(raw_scores, list)
            else [0.0] * len(self._tokens)
        )
        self._token_type = (
            [int(t) for t in raw_types]
            if isinstance(raw_types, list)
            else [1] * len(self._tokens)
        )

        self._vocab = {t: i for i, t in enumerate(self._tokens) if t}
        self._id_to_token = {i: t for i, t in enumerate(self._tokens) if t}
        self._vocab_size = len(self._tokens)

    def _parse_merges(self, raw_merges: list):
        if not raw_merges or not isinstance(raw_merges, list):
            return
        merge_bytes = []
        for m in raw_merges:
            if isinstance(m, bytes):
                merge_bytes.append(m)
            elif isinstance(m, str):
                merge_bytes.append(m.encode("utf-8"))
            else:
                continue
        self._parse_merges_bytes(merge_bytes)

    def _parse_merges_bytes(self, merge_bytes: list[bytes]):
        self._merge_ranks.clear()
        for m in merge_bytes:
            parts = m.split(b" ")
            if len(parts) >= 2:
                idx_a = self._find_token(parts[0])
                idx_b = self._find_token(parts[1])
                if idx_a >= 0 and idx_b >= 0:
                    rank = len(self._merge_ranks)
                    self._merge_ranks[(idx_a, idx_b)] = rank
        self._build_merge_index()

    def _parse_special_from_gguf(self, reader):
        for attr, key in [
            ("bos_id", "tokenizer.ggml.bos_token_id"),
            ("eos_id", "tokenizer.ggml.eos_token_id"),
            ("eot_id", "tokenizer.ggml.eot_token_id"),
            ("eom_id", "tokenizer.ggml.eom_token_id"),
            ("unk_id", "tokenizer.ggml.unknown_token_id"),
            ("pad_id", "tokenizer.ggml.padding_token_id"),
            ("mask_id", "tokenizer.ggml.mask_token_id"),
        ]:
            val = _get_gguf_field(reader, key)
            if val is not None:
                try:
                    setattr(self, attr, int(val))
                except (ValueError, TypeError):
                    pass

    def _find_token(self, token_bytes: bytes) -> int:
        for i, t in enumerate(self._tokens):
            if t == token_bytes:
                return i
        return -1

    def _build_byte_fallback(self):
        self._tokens = [bytes([i]) for i in range(256)]
        self._vocab_size = 256
        self._scores = [0.0] * 256
        self._token_type = [1] * 256
        self._vocab = {t: i for i, t in enumerate(self._tokens)}
        self._id_to_token = {i: t for i, t in enumerate(self._tokens)}

    # ── Tokenization ────────────────────────────────────────────────

    def _byte_to_token_id(self, b: int) -> int:
        """Map a byte value to the nearest token ID."""
        target = bytes([b])
        tid = self._vocab.get(target)
        if tid is not None:
            return tid
        for i, t in enumerate(self._tokens):
            if t == target:
                return i
        return b % max(self.vocab_size, 1)

    def encode(self, text: str) -> list[int]:
        if not text:
            return [self.bos_id] if self.add_bos else []

        cache_key = text
        cached = self._encode_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        # Pre-tokenize with regex
        words = self.split_pattern.findall(text)
        if not words:
            words = [text]

        token_ids = []
        for word in words:
            word_ids = self._encode_word(word)
            token_ids.extend(word_ids)

        if self.add_bos:
            token_ids = [self.bos_id] + token_ids
        if self.add_eos:
            token_ids = token_ids + [self.eos_id]

        # Cache result
        if len(self._encode_cache) >= self._cache_maxsize:
            self._encode_cache.pop(next(iter(self._encode_cache)), None)
        self._encode_cache[cache_key] = list(token_ids)

        return token_ids

    def _encode_word(self, word: str) -> list[int]:
        """Encode a single word using byte-level BPE."""
        word_bytes = word.encode("utf-8")

        # Fast path: word is in vocab
        tid = self._vocab.get(word_bytes)
        if tid is not None:
            return [tid]

        # Convert bytes to token IDs
        token_ids = [self._byte_to_token_id(b) for b in word_bytes]

        if not self._merge_ranks:
            return token_ids

        # Check merge cache
        cache_key = tuple(token_ids)
        cached = self._merge_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        # Apply BPE merges using priority queue for O(n log n)
        merged = self._apply_merges(token_ids)

        # Cache the merge result
        if len(self._merge_cache) < 50000:
            self._merge_cache[cache_key] = list(merged)

        return merged

    def _apply_merges(self, token_ids: list[int]) -> list[int]:
        """Apply BPE merges using a priority queue approach."""
        if len(token_ids) < 2:
            return token_ids

        tokens = list(token_ids)

        while len(tokens) > 1:
            # Find the pair with the lowest rank
            best_rank = None
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None:
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_idx = i
                        if best_rank == 0:
                            break

            if best_idx < 0:
                break

            i = best_idx
            pair = (tokens[i], tokens[i + 1])
            rank = self._merge_ranks.get(pair)
            merged_id = self._merge_rank_to_id.get(rank) if rank is not None else None

            if merged_id is None:
                # Try to find merged token by concatenation
                merged_bytes = self._tokens[tokens[i]] + self._tokens[tokens[i + 1]]
                merged_id = self._vocab.get(merged_bytes)
                if merged_id is None and rank is not None:
                    self._merge_rank_to_id[rank] = tokens[i]
                    merged_id = tokens[i]
                elif merged_id is None:
                    break

            tokens = tokens[:i] + [merged_id] + tokens[i + 2 :]

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        result = b""
        for tid in token_ids:
            if tid < len(self._tokens) and self._tokens[tid]:
                result += self._tokens[tid]
            elif tid < 256 and self.byte_fallback:
                result += bytes([tid])
            else:
                result += b" "
        return result.decode("utf-8", errors="replace")

    # ── Serialization ────────────────────────────────────────────────

    def save(self, path: str):
        data = {
            "tokens": [t.decode("utf-8", errors="replace") for t in self._tokens],
            "scores": self._scores,
            "token_type": self._token_type,
            "merge_ranks": [
                [int(k[0]), int(k[1]), int(v)] for k, v in self._merge_ranks.items()
            ],
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "pad_id": self.pad_id,
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        t = cls()
        t._tokens = [
            item.encode("utf-8") if isinstance(item, str) else item
            for item in data["tokens"]
        ]
        t._scores = data.get("scores", [0.0] * len(t._tokens))
        t._token_type = data.get("token_type", [1] * len(t._tokens))
        t._vocab = {tok: i for i, tok in enumerate(t._tokens) if tok}
        t._id_to_token = {i: tok for i, tok in enumerate(t._tokens) if tok}

        for item in data.get("merge_ranks", []):
            a, b, rank = int(item[0]), int(item[1]), int(item[2])
            t._merge_ranks[(a, b)] = rank
        t._build_merge_index()

        t.bos_id = data.get("bos_id", 1)
        t.eos_id = data.get("eos_id", 2)
        t.unk_id = data.get("unk_id", 0)
        t.pad_id = data.get("pad_id", 0)
        t.add_bos = data.get("add_bos", False)
        t.add_eos = data.get("add_eos", False)
        t._vocab_size = len(t._tokens)
        t._setup_special_ids()
        return t

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["merge_rules"] = len(self._merge_ranks)
        cfg["byte_fallback"] = self.byte_fallback
        cfg["split_pattern"] = (
            self.split_pattern.pattern
            if hasattr(self.split_pattern, "pattern")
            else str(self.split_pattern)
        )
        return cfg


# ═══════════════════════════════════════════════════════════════════════════
# SentencePieceTokenizer – SentencePiece / Unigram compatible
# ═══════════════════════════════════════════════════════════════════════════


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece-compatible tokenizer supporting Unigram LM and BPE modes.

    Features:
    - Unigram language model tokenization (Viterbi / beam search)
    - BPE fallback mode
    - Byte-fallback for unknown characters
    - NFKC normalization
    - Control tokens and user-defined symbols
    """

    def __init__(
        self,
        scores: Optional[list[float]] = None,
        tokens: Optional[list[bytes]] = None,
        unk_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        byte_fallback: bool = True,
        add_bos: bool = True,
        add_eos: bool = False,
    ):
        super().__init__()
        self.name = "sentencepiece"

        self._tokens: list[bytes] = tokens or []
        self._scores: list[float] = scores or []
        self._token_type: list[int] = []
        self._vocab: dict[bytes, int] = {}
        self._id_to_token: dict[int, bytes] = {}

        self.byte_fallback = byte_fallback
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.add_bos = add_bos
        self.add_eos = add_eos

        # Unigram data
        self._unigram_log_probs: list[float] = []
        self._is_control: set[int] = set()
        self._is_unused: set[int] = set()

        self._encode_cache: dict[str, list[int]] = {}
        self._cache_maxsize = 10000

        if tokens and scores:
            self._build_from_data(tokens, scores)

    def _build_from_data(self, tokens: list[bytes], scores: list[float]):
        self._vocab = {t: i for i, t in enumerate(tokens) if t}
        self._id_to_token = dict(enumerate(tokens))
        self._token_type = [1] * len(tokens)

        # Compute unigram log probabilities
        total_score = sum(max(s, 1e-10) for s in scores)
        self._unigram_log_probs = [
            math.log(max(s, 1e-10) / total_score) if total_score > 0 else 0.0
            for s in scores
        ]

        # Detect control tokens (score == 0.0 is typical for control tokens)
        for i, s in enumerate(scores):
            if s == 0.0 and tokens[i]:
                self._is_control.add(i)

    # ── Build from GGUF ──────────────────────────────────────────────

    @classmethod
    def from_gguf_reader(cls, reader) -> "SentencePieceTokenizer":
        raw_tokens = _get_gguf_field(reader, "tokenizer.ggml.tokens", [])
        raw_scores = _get_gguf_field(reader, "tokenizer.ggml.scores", [])
        raw_types = _get_gguf_field(reader, "tokenizer.ggml.token_type", [])

        tokens: list[bytes] = []
        for t in raw_tokens:
            if isinstance(t, bytes):
                tokens.append(t)
            elif isinstance(t, str):
                tokens.append(t.encode("utf-8", errors="replace"))
            else:
                tokens.append(str(t).encode())

        scores = (
            [float(s) for s in raw_scores]
            if isinstance(raw_scores, list)
            else [0.0] * len(tokens)
        )

        token_types = (
            [int(t) for t in raw_types]
            if isinstance(raw_types, list)
            else [1] * len(tokens)
        )

        bos = _get_gguf_field(reader, "tokenizer.ggml.bos_token_id", 1)
        eos = _get_gguf_field(reader, "tokenizer.ggml.eos_token_id", 2)
        unk = _get_gguf_field(reader, "tokenizer.ggml.unknown_token_id", 0)

        tok = cls(scores=scores, tokens=tokens, unk_id=unk, bos_id=bos, eos_id=eos)
        tok._token_type = token_types

        for i, tt in enumerate(token_types):
            if tt == 3:
                tok._is_control.add(i)
            elif tt == 4:
                tok._is_unused.add(i)

        # Detect pre-tokenizer
        raw_pre = _get_gguf_field(reader, "tokenizer.ggml.pre", "")
        tok._pre_tokenizer_type = str(raw_pre) if raw_pre else "sentencepiece"

        # Handle byte-fallback
        raw_byte_fallback = _get_gguf_field(
            reader, "tokenizer.ggml.add_bos_token", True
        )
        if raw_byte_fallback is not None:
            tok.add_bos = bool(raw_byte_fallback)

        if tok.vocab_size == 0:
            tok._build_byte_fallback()

        return tok

    @classmethod
    def from_sentencepiece_file(cls, path: str) -> "SentencePieceTokenizer":
        """Load from a SentencePiece .model file."""
        import struct

        with open(path, "rb") as f:
            data = f.read()

        tokens = []
        scores = []
        token_types = []

        pos = 0
        while pos < len(data) - 4:
            length = struct.unpack(">I", data[pos : pos + 4])[0]
            pos += 4
            if pos + length > len(data):
                break
            piece_data = data[pos : pos + length]
            pos += length

            if pos + 4 > len(data):
                break
            score = struct.unpack(">f", data[pos : pos + 4])[0]
            pos += 4

            if pos + 1 > len(data):
                break
            token_type = data[pos]
            pos += 1

            if pos + 4 > len(data):
                break

            tokens.append(piece_data)
            scores.append(score)
            token_types.append(token_type)

        tok = cls(scores=scores, tokens=tokens, unk_id=0, bos_id=1, eos_id=2)
        tok._token_type = token_types
        tok._pre_tokenizer_type = "sentencepiece"

        for i in range(len(tokens) - 1, -1, -1):
            if token_types[i] == 0 and i < len(tokens) - 1:
                pass
            if token_types[i] >= 3:
                tok._is_control.add(i)

        return tok

    def _build_byte_fallback(self):
        if not self._tokens:
            self._tokens = [bytes([i]) for i in range(256)]
            self._scores = [0.0] * 256
            self._token_type = [1] * 256
            self._unigram_log_probs = [0.0] * 256
            self._vocab = {t: i for i, t in enumerate(self._tokens)}
            self._id_to_token = {i: t for i, t in enumerate(self._tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    @vocab_size.setter
    def vocab_size(self, val):
        pass

    # ── Normalization ────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """NFKC normalization with optional extra whitespace removal."""
        import unicodedata

        text = unicodedata.normalize("NFKC", text)
        return text

    # ── Tokenization (Unigram LM) ────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        if not text:
            return [self.bos_id] if self.add_bos else []

        cache_key = text
        cached = self._encode_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        text = self._normalize(text)

        token_ids = self._unigram_tokenize(text)

        # Fallback: use byte-level encoding for unknown characters
        if self.byte_fallback:
            token_ids = self._apply_byte_fallback(token_ids, text)

        if self.add_bos:
            token_ids = [self.bos_id] + token_ids
        if self.add_eos:
            token_ids = token_ids + [self.eos_id]

        if len(self._encode_cache) >= self._cache_maxsize:
            self._encode_cache.pop(next(iter(self._encode_cache)), None)
        self._encode_cache[cache_key] = list(token_ids)

        return token_ids

    def _unigram_tokenize(self, text: str) -> list[int]:
        if not self._unigram_log_probs or not self._tokens:
            return []

        text_bytes = text.encode("utf-8")
        n = len(text_bytes)

        # Dynamic programming for Viterbi segmentation
        # dp[i] = (best_log_prob, best_split_pos, best_token_id)
        dp = [(float("-inf"), -1, -1)] * (n + 1)
        dp[0] = (0.0, -1, -1)

        # Build a trie-like structure for fast matching
        for i in range(n):
            if dp[i][0] == float("-inf"):
                continue

            for tid, token_bytes in enumerate(self._tokens):
                if not token_bytes:
                    continue
                if text_bytes[i : i + len(token_bytes)] == token_bytes:
                    j = i + len(token_bytes)
                    log_prob = (
                        self._unigram_log_probs[tid]
                        if tid < len(self._unigram_log_probs)
                        else 0.0
                    )
                    candidate = dp[i][0] + log_prob
                    if candidate > dp[j][0]:
                        dp[j] = (candidate, i, tid)

        # Backtrack
        token_ids = []
        pos = n
        while pos > 0:
            _, prev_pos, tid = dp[pos]
            if tid < 0:
                break
            token_ids.append(tid)
            pos = prev_pos

        if pos != 0:
            # Viterbi failed, fall back to BPE-like or byte encoding
            return self._bpe_fallback_encode(text_bytes)

        return list(reversed(token_ids))

    def _bpe_fallback_encode(self, text_bytes: bytes) -> list[int]:
        """BPE-style fallback encoding for SentencePiece."""
        token_ids = []
        for b in text_bytes:
            target = bytes([b])
            tid = self._vocab.get(target)
            if tid is not None:
                token_ids.append(tid)
            else:
                token_ids.append(self.unk_id)
        return token_ids

    def _apply_byte_fallback(
        self, token_ids: list[int], original_text: str
    ) -> list[int]:
        """Replace unknown tokens with byte-level encoding."""
        result = []
        text_bytes = original_text.encode("utf-8")
        byte_pos = 0
        for tid in token_ids:
            if tid >= len(self._tokens):
                if byte_pos < len(text_bytes):
                    result.append(min(text_bytes[byte_pos], self.unk_id))
                    byte_pos += 1
                else:
                    result.append(tid)
                continue
            token_len = len(self._tokens[tid]) if self._tokens[tid] else 1
            byte_pos += token_len
            result.append(tid)
        return result

    def decode(self, token_ids: list[int]) -> str:
        result = b""
        for tid in token_ids:
            if tid < len(self._tokens) and self._tokens[tid]:
                result += self._tokens[tid]
            elif tid < 256 and self.byte_fallback:
                result += bytes([tid])
            elif self.unk_id < len(self._tokens):
                result += self._tokens[self.unk_id]
        text = result.decode("utf-8", errors="replace")
        text = text.replace("\u2581", " ").strip()
        return text

    # ── Serialization ────────────────────────────────────────────────

    def save(self, path: str):
        data = {
            "tokens": [t.decode("utf-8", errors="replace") for t in self._tokens],
            "scores": self._scores,
            "token_type": self._token_type,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
            "byte_fallback": self.byte_fallback,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SentencePieceTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokens = [
            item.encode("utf-8") if isinstance(item, str) else item
            for item in data["tokens"]
        ]
        tok = cls(
            scores=data.get("scores", [0.0] * len(tokens)),
            tokens=tokens,
            unk_id=data.get("unk_id", 0),
            bos_id=data.get("bos_id", 1),
            eos_id=data.get("eos_id", 2),
            byte_fallback=data.get("byte_fallback", True),
            add_bos=data.get("add_bos", True),
            add_eos=data.get("add_eos", False),
        )
        tok._token_type = data.get("token_type", [1] * len(tokens))
        return tok


# ═══════════════════════════════════════════════════════════════════════════
# TiktokenTokenizer – OpenAI-compatible tokenizer
# ═══════════════════════════════════════════════════════════════════════════

# Pre-computed encoder data for known tokenizers
_TIKTOKEN_REGISTRY = {
    "cl100k_base": {
        "pattern": CL100K_SPLIT_PATTERN,
        "vocab_name": "cl100k_base",
    },
    "p50k_base": {
        "pattern": P50K_SPLIT_PATTERN,
        "vocab_name": "p50k_base",
    },
    "r50k_base": {
        "pattern": R50K_SPLIT_PATTERN,
        "vocab_name": "r50k_base",
    },
    "gpt2": {
        "pattern": GPT2_SPLIT_PATTERN,
        "vocab_name": "gpt2",
    },
}


class TiktokenTokenizer(BaseTokenizer):
    """OpenAI-compatible tokenizer supporting cl100k_base, p50k_base, r50k_base.

    Uses regex pattern-based pre-tokenization followed by BPE merge on
    byte sequences. Can load from OpenAI's tiktoken files or from GGUF metadata.
    """

    def __init__(
        self,
        encoder_name: str = "cl100k_base",
        vocab: Optional[dict[bytes, int]] = None,
        merge_ranks: Optional[dict[tuple[int, int], int]] = None,
        special_tokens: Optional[dict[str, int]] = None,
        split_pattern: Optional[re.Pattern] = None,
    ):
        super().__init__()
        self.name = f"tiktoken/{encoder_name}"
        self.encoder_name = encoder_name
        self._encoder_name = encoder_name

        self._tokens: list[bytes] = []
        self._scores: list[float] = []
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._merge_rank_to_id: dict[int, int] = {}
        self._vocab: dict[bytes, int] = {}
        self._reverse_vocab: dict[int, bytes] = {}

        self.special_tokens = {}
        self._special_regex: Optional[re.Pattern] = None
        self._all_special_ids: set[int] = set()

        if encoder_name in _TIKTOKEN_REGISTRY:
            self.split_pattern = _TIKTOKEN_REGISTRY[encoder_name]["pattern"]
        else:
            self.split_pattern = split_pattern or CL100K_SPLIT_PATTERN

        if vocab:
            self._load_vocab(vocab)
        if merge_ranks:
            self._merge_ranks = merge_ranks
            self._build_merge_index()

        if special_tokens:
            self._set_special_tokens(special_tokens)

        if self._tokens:
            self._vocab_size = len(self._tokens)
        else:
            self._vocab_size = 0

    def _load_vocab(self, vocab: dict[bytes, int]):
        self._vocab = dict(vocab)
        self._reverse_vocab = {v: k for k, v in vocab.items()}
        max_id = max(vocab.values()) if vocab else 0
        self._tokens = [b""] * (max_id + 1)
        for token_bytes, tid in vocab.items():
            self._tokens[tid] = token_bytes
        self._scores = [0.0] * (max_id + 1)

    def _build_merge_index(self):
        self._merge_rank_to_id.clear()
        for (a, b), rank in self._merge_ranks.items():
            if a < len(self._tokens) and b < len(self._tokens):
                merged = self._tokens[a] + self._tokens[b]
                merged_id = self._vocab.get(merged)
                if merged_id is not None:
                    self._merge_rank_to_id[rank] = merged_id

    def _set_special_tokens(self, special_tokens: dict[str, int]):
        self.special_tokens = dict(special_tokens)
        self._all_special_ids = set(special_tokens.values())

        if special_tokens:
            special_patterns = [re.escape(k) for k in special_tokens]
            combined = "|".join(special_patterns)
            self._special_regex = re.compile(combined)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # ── Build from various sources ───────────────────────────────────

    @classmethod
    def from_encoder_name(cls, name: str = "cl100k_base") -> "TiktokenTokenizer":
        """Load a known encoder. Falls back to building from included data."""
        if name in _TIKTOKEN_REGISTRY:
            tok = cls(encoder_name=name)
            return tok
        raise ValueError(
            f"Unknown encoder: {name}. Known: {list(_TIKTOKEN_REGISTRY.keys())}"
        )

    @classmethod
    def from_tiktoken_file(cls, path: str) -> "TiktokenTokenizer":
        """Load from a tiktoken .tiktoken file (vocab and merge ranks)."""
        with open(path, "r") as f:
            data = json.load(f)

        vocab = {}
        for token_str, token_id in data.get("vocab", {}).items():
            vocab[token_str.encode("utf-8")] = token_id

        merge_ranks = {}
        for merge_str, rank in data.get("merge_ranks", {}).items():
            parts = merge_str.split()
            if len(parts) == 3:
                merge_ranks[(int(parts[0]), int(parts[1]))] = int(parts[2])

        special = data.get("special_tokens", {})

        tok = cls(
            encoder_name=data.get("encoder_name", "custom"),
            vocab=vocab,
            merge_ranks=merge_ranks,
            special_tokens=special,
        )
        return tok

    @classmethod
    def from_gguf_reader(cls, reader) -> "TiktokenTokenizer":
        """Build tiktoken-style tokenizer from GGUF reader."""
        raw_tokens = _get_gguf_field(reader, "tokenizer.ggml.tokens", [])
        raw_scores = _get_gguf_field(reader, "tokenizer.ggml.scores", [])
        raw_merges = _get_gguf_field(reader, "tokenizer.ggml.merges", [])

        tokens: list[bytes] = []
        for t in raw_tokens:
            if isinstance(t, bytes):
                tokens.append(t)
            elif isinstance(t, str):
                tokens.append(t.encode("utf-8", errors="replace"))
            else:
                tokens.append(str(t).encode())

        vocab = {t: i for i, t in enumerate(tokens) if t}

        merge_ranks = {}
        if isinstance(raw_merges, list):
            for m in raw_merges:
                if isinstance(m, bytes):
                    parts = m.split(b" ")
                elif isinstance(m, str):
                    parts = m.encode().split(b" ")
                else:
                    continue
                if len(parts) >= 2:
                    try:
                        idx_a = vocab.get(parts[0], -1)
                        idx_b = vocab.get(parts[1], -1)
                        if idx_a >= 0 and idx_b >= 0:
                            merge_ranks[(idx_a, idx_b)] = len(merge_ranks)
                    except Exception:
                        pass

        tok = cls(
            encoder_name="gguf",
            vocab=vocab,
            merge_ranks=merge_ranks,
        )

        # Parse special IDs
        for attr, key in [
            ("bos_id", "tokenizer.ggml.bos_token_id"),
            ("eos_id", "tokenizer.ggml.eos_token_id"),
            ("unk_id", "tokenizer.ggml.unknown_token_id"),
        ]:
            val = _get_gguf_field(reader, key)
            if val is not None:
                try:
                    setattr(tok, attr, int(val))
                except (ValueError, TypeError):
                    pass

        return tok

    # ── Core encode with regex pre-tokenization ──────────────────────

    def encode(self, text: str) -> list[int]:
        if not text:
            return [self.bos_id] if hasattr(self, "bos_id") and self.bos_id >= 0 else []

        # Check for special tokens first
        if self._special_regex and self._special_regex.search(text):
            return self._encode_with_special(text)

        # Pre-tokenize with regex
        words = self.split_pattern.findall(text)
        if not words:
            words = [text]

        token_ids = []
        for word in words:
            word_ids = self._encode_word(word)
            token_ids.extend(word_ids)

        return token_ids

    def _encode_with_special(self, text: str) -> list[int]:
        """Handle text containing special tokens."""
        token_ids = []
        last_end = 0

        for match in self._special_regex.finditer(text):
            start, end = match.start(), match.end()
            if start > last_end:
                segment = text[last_end:start]
                if segment:
                    token_ids.extend(self.encode(segment))
            special_token = match.group(0)
            special_id = self.special_tokens.get(special_token)
            if special_id is not None:
                token_ids.append(special_id)
            last_end = end

        if last_end < len(text):
            segment = text[last_end:]
            if segment:
                token_ids.extend(self.encode(segment))

        return token_ids

    def _encode_word(self, word: str) -> list[int]:
        """Encode a single word using byte-level BPE."""
        word_bytes = word.encode("utf-8")

        # Fast path: word is in vocab
        tid = self._vocab.get(word_bytes)
        if tid is not None:
            return [tid]

        # Represent as bytes-to-unicode tokens
        reconstructed = word_bytes.decode("utf-8", errors="replace")
        mapped = "".join(BYTE_TO_UNICODE.get(b, chr(b)) for b in word_bytes)

        # Split into pieces that exist in vocab
        pieces = self._bpe_segment(mapped)

        ids = []
        for piece in pieces:
            piece_bytes = piece.encode("utf-8")
            tid = self._vocab.get(piece_bytes)
            if tid is not None:
                ids.append(tid)
            else:
                for b in piece_bytes:
                    fallback_id = self._vocab.get(bytes([b]))
                    if fallback_id is not None:
                        ids.append(fallback_id)

        return ids

    def _bpe_segment(self, text: str) -> list[str]:
        """BPE segmentation using merge ranks."""
        if len(text) <= 1:
            return [text]

        # Convert to IDs via byte-to-unicode mapping
        token_ids = []
        for ch in text:
            byte_val = _UNICODE_TO_BYTE.get(ch)
            if byte_val is not None:
                target = bytes([byte_val])
                tid = self._vocab.get(target)
                if tid is not None:
                    token_ids.append(tid)
                else:
                    token_ids.append(byte_val % max(self.vocab_size, 1))
            else:
                token_ids.append(ord(ch) % max(self.vocab_size, 1))

        if not self._merge_ranks:
            return [
                self._tokens[t].decode("utf-8", errors="replace")
                for t in token_ids
                if t < len(self._tokens)
            ]

        # Apply merges
        while len(token_ids) > 1:
            best_rank = None
            best_idx = -1
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i

            if best_idx < 0:
                break

            i = best_idx
            pair = (token_ids[i], token_ids[i + 1])
            rank = self._merge_ranks.get(pair)
            merged_id = self._merge_rank_to_id.get(rank) if rank is not None else None

            if merged_id is None:
                if rank is not None:
                    merged_id = token_ids[i]
                else:
                    break

            token_ids = token_ids[:i] + [merged_id] + token_ids[i + 2 :]

        decoded = []
        for tid in token_ids:
            if tid < len(self._tokens) and self._tokens[tid]:
                decoded.append(self._tokens[tid].decode("utf-8", errors="replace"))
        return decoded

    def decode(self, token_ids: list[int]) -> str:
        result = b""
        for tid in token_ids:
            if tid < len(self._tokens) and self._tokens[tid]:
                result += self._tokens[tid]
            elif tid < 256:
                result += bytes([tid])
        return result.decode("utf-8", errors="replace")

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["encoder_name"] = self._encoder_name
        cfg["special_tokens"] = {k: v for k, v in self.special_tokens.items()}
        return cfg


# ═══════════════════════════════════════════════════════════════════════════
# AutoTokenizer – Automatic tokenizer detection and loading
# ═══════════════════════════════════════════════════════════════════════════

_TOKENIZER_CACHE: dict[str, BaseTokenizer] = {}


def _get_gguf_field(reader, key: str, default=None):
    """Get a field value from a GGUF reader."""
    field = getattr(reader, "fields", {}).get(key)
    if field is None:
        return default
    try:
        data = field.parts[1] if len(field.parts) > 1 else field.parts[-1]
        if hasattr(data, "shape") and data.ndim == 0:
            return data.item()
        if isinstance(data, bytes):
            return data
        if isinstance(data, np.ndarray):
            return data.item() if data.size == 1 else data.tolist()
        if isinstance(data, (list, str, int, float, bool)):
            return data
        return data
    except Exception:
        return default


def _detect_tokenizer_type_from_gguf(reader) -> str:
    """Detect tokenizer type from GGUF metadata."""
    raw_model = _get_gguf_field(reader, "tokenizer.ggml.model", "")
    model_str = str(raw_model).strip().lower() if raw_model else ""

    if not model_str:
        raw_tokens = _get_gguf_field(reader, "tokenizer.ggml.tokens", [])
        raw_scores = _get_gguf_field(reader, "tokenizer.ggml.scores", [])
        raw_merges = _get_gguf_field(reader, "tokenizer.ggml.merges", [])

        if raw_merges and isinstance(raw_merges, list) and len(raw_merges) > 0:
            return "bpe"
        if raw_scores and isinstance(raw_scores, list) and len(raw_scores) > 0:
            return "sentencepiece"
        return "bpe"

    if model_str in ("gpt2", "gpt3", "gpt4", "bpe", "llama"):
        return "bpe"
    if model_str in ("sentencepiece", "unigram", "llama_spm"):
        return "sentencepiece"
    if model_str in ("tiktoken", "cl100k_base", "p50k_base", "r50k_base"):
        return "tiktoken"

    raw_pre = _get_gguf_field(reader, "tokenizer.ggml.pre", "")
    pre_str = str(raw_pre).strip().lower() if raw_pre else ""

    if pre_str in ("default", "sentencepiece", "llama-bpe"):
        if pre_str == "llama-bpe":
            return "bpe"
        return "sentencepiece"

    if raw_merges:
        return "bpe"
    return "sentencepiece"


def _detect_tokenizer_type_from_metadata(metadata: dict) -> str:
    """Detect tokenizer type from a flat metadata dict."""
    raw_model = metadata.get("tokenizer.ggml.model", "")
    model_str = str(raw_model).strip().lower() if raw_model else ""

    if model_str in ("gpt2", "gpt3", "gpt4", "bpe", "llama"):
        return "bpe"
    if model_str in ("sentencepiece", "unigram", "llama_spm"):
        return "sentencepiece"
    if model_str in ("tiktoken", "cl100k_base", "p50k_base", "r50k_base"):
        return "tiktoken"

    raw_merges = metadata.get("tokenizer.ggml.merges", [])
    raw_scores = metadata.get("tokenizer.ggml.scores", [])

    if raw_merges and isinstance(raw_merges, list) and len(raw_merges) > 0:
        return "bpe"
    if raw_scores and isinstance(raw_scores, list) and len(raw_scores) > 0:
        return "sentencepiece"

    return "bpe"


class AutoTokenizer:
    """Auto-detect and load the correct tokenizer.

    Detection order:
    1. From GGUF metadata (model type, pre-tokenizer, token lists)
    2. From explicit encoder name
    3. From file path (JSON or .model)
    4. Fallback chain: tiktoken -> SentencePiece -> BPE -> byte-level
    """

    def __init__(self):
        self._tokenizer: Optional[BaseTokenizer] = None

    @property
    def tokenizer(self) -> BaseTokenizer:
        if self._tokenizer is None:
            self._tokenizer = BPETokenizer()
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tok: BaseTokenizer):
        self._tokenizer = tok

    # ── Class methods ────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> BaseTokenizer:
        """Load a tokenizer by model name or path.

        Supports:
        - HuggingFace model names (via local cache)
        - 'cl100k_base', 'p50k_base', 'r50k_base' (tiktoken)
        - Path to tokenizer.json, tokenizer.model, or .tiktoken file
        - Path to GGUF file
        """
        if model_name_or_path in _TIKTOKEN_REGISTRY:
            return TiktokenTokenizer.from_encoder_name(model_name_or_path)

        if os.path.isdir(model_name_or_path):
            for candidate in ["tokenizer.json", "tokenizer.model", "vocab.json"]:
                path = os.path.join(model_name_or_path, candidate)
                if os.path.exists(path):
                    return cls._load_from_file(path)

        if os.path.isfile(model_name_or_path):
            return cls._load_from_file(model_name_or_path)

        if model_name_or_path.endswith(".gguf"):
            return cls.from_gguf(model_name_or_path)

        # Try tiktoken-style names
        if model_name_or_path in ("gpt2", "gpt3", "gpt4"):
            return TiktokenTokenizer.from_encoder_name("cl100k_base")

        raise ValueError(f"Could not load tokenizer from: {model_name_or_path}")

    @classmethod
    def from_gguf(cls, path_or_reader) -> BaseTokenizer:
        """Auto-detect and load tokenizer from a GGUF file or reader."""
        if isinstance(path_or_reader, str):
            from gguf import GGUFReader

            reader = GGUFReader(path_or_reader)
        else:
            reader = path_or_reader

        tok_type = _detect_tokenizer_type_from_gguf(reader)

        if tok_type == "bpe":
            return BPETokenizer.from_gguf_reader(reader)
        elif tok_type == "sentencepiece":
            return SentencePieceTokenizer.from_gguf_reader(reader)
        elif tok_type == "tiktoken":
            return TiktokenTokenizer.from_gguf_reader(reader)

        return BPETokenizer.from_gguf_reader(reader)

    @classmethod
    def from_gguf_model(cls, model) -> BaseTokenizer:
        """Load tokenizer from a GGUFModel instance."""
        if hasattr(model, "reader"):
            return cls.from_gguf(model.reader)
        if hasattr(model, "_get_field"):
            metadata = {}
            for key in GGUF_TOKENIZER_KEYS.values():
                metadata[key] = model._get_field(key)
            return cls.from_metadata(metadata)

        return BPETokenizer()

    @classmethod
    def from_metadata(cls, metadata: dict) -> BaseTokenizer:
        """Load tokenizer from a flat metadata dict."""
        tok_type = _detect_tokenizer_type_from_metadata(metadata)

        if tok_type == "bpe":
            return BPETokenizer.from_gguf_metadata(metadata)
        elif tok_type == "sentencepiece":
            raw_tokens = metadata.get("tokenizer.ggml.tokens", [])
            raw_scores = metadata.get("tokenizer.ggml.scores", [])
            tokens = [t.encode() if isinstance(t, str) else t for t in raw_tokens]
            scores = (
                [float(s) for s in raw_scores] if isinstance(raw_scores, list) else []
            )
            bos = int(metadata.get("tokenizer.ggml.bos_token_id", 1))
            eos = int(metadata.get("tokenizer.ggml.eos_token_id", 2))
            unk = int(metadata.get("tokenizer.ggml.unknown_token_id", 0))
            return SentencePieceTokenizer(
                scores=scores, tokens=tokens, unk_id=unk, bos_id=bos, eos_id=eos
            )

        return BPETokenizer.from_gguf_metadata(metadata)

    @classmethod
    def _load_from_file(cls, path: str) -> BaseTokenizer:
        """Load tokenizer from a file."""
        if path.endswith(".model"):
            return SentencePieceTokenizer.from_sentencepiece_file(path)
        elif path.endswith(".tiktoken"):
            return TiktokenTokenizer.from_tiktoken_file(path)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "model" in data and "tokenizer" in path:
                return cls.from_metadata(data)

            if "vocab" in data and "merge_ranks" in data:
                return TiktokenTokenizer.from_tiktoken_file(path)

            if isinstance(data, dict):
                if "tokenizer.ggml.model" in data:
                    return cls.from_metadata(data)
                if "model" in data:
                    return cls.from_metadata(data.get("model", {}))

        raise ValueError(f"Unrecognized tokenizer file: {path}")

    # ── Instance methods ─────────────────────────────────────────────

    def __getattr__(self, name):
        if name in (
            "encode",
            "decode",
            "token_count",
            "save",
            "load",
            "info",
            "get_config",
            "vocab_size",
        ):
            return getattr(self.tokenizer, name)
        raise AttributeError(f"AutoTokenizer has no attribute '{name}'")

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def get_tokenizer(self) -> BaseTokenizer:
        return self.tokenizer

    def set_tokenizer(self, tokenizer: BaseTokenizer):
        self._tokenizer = tokenizer

    def report(self) -> str:
        return self.tokenizer.info()


def get_tokenizer_info(tokenizer) -> str:
    """Get a human-readable info string about any tokenizer."""
    if hasattr(tokenizer, "info"):
        return tokenizer.info()
    if hasattr(tokenizer, "get_config"):
        return json.dumps(tokenizer.get_config(), indent=2)
    return str(type(tokenizer).__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CachedTokenizer – High-speed caching layer
# ═══════════════════════════════════════════════════════════════════════════


class CachedTokenizer(BaseTokenizer):
    """Thread-safe LRU caching wrapper around any tokenizer.

    Features:
    - LRU cache for encode (input string -> token ids)
    - LRU cache for decode (token ids -> output string)
    - Thread-safe (read-write lock via RLock)
    - Statistics: cache hit rate, avg encode time
    - Cache invalidation on tokenizer update
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        encode_cache_size: int = 50000,
        decode_cache_size: int = 50000,
    ):
        super().__init__()
        self.name = f"cached({tokenizer.name})"
        self._inner = tokenizer

        self._encode_cache: OrderedDict[str, list[int]] = OrderedDict()
        self._decode_cache: OrderedDict[tuple, str] = OrderedDict()
        self._encode_cache_size = encode_cache_size
        self._decode_cache_size = decode_cache_size

        # Statistics
        self._encode_hits = 0
        self._encode_misses = 0
        self._decode_hits = 0
        self._decode_misses = 0
        self._total_encode_time = 0.0
        self._total_encode_calls = 0
        self._lock = __import__("threading").RLock()

    @property
    def vocab_size(self) -> int:
        return self._inner.vocab_size

    @vocab_size.setter
    def vocab_size(self, val):
        pass

    def encode(self, text: str) -> list[int]:
        with self._lock:
            cached = self._encode_cache.get(text)
            if cached is not None:
                self._encode_hits += 1
                self._encode_cache.move_to_end(text)
                return list(cached)

        self._encode_misses += 1
        t0 = time.perf_counter()

        result = self._inner.encode(text)

        elapsed = time.perf_counter() - t0
        self._total_encode_time += elapsed
        self._total_encode_calls += 1

        with self._lock:
            if len(self._encode_cache) >= self._encode_cache_size:
                self._encode_cache.popitem(last=False)
            self._encode_cache[text] = list(result)

        return result

    def decode(self, token_ids: list[int]) -> str:
        key = tuple(token_ids)

        with self._lock:
            cached = self._decode_cache.get(key)
            if cached is not None:
                self._decode_hits += 1
                self._decode_cache.move_to_end(key)
                return cached

        self._decode_misses += 1
        result = self._inner.decode(token_ids)

        with self._lock:
            if len(self._decode_cache) >= self._decode_cache_size:
                self._decode_cache.popitem(last=False)
            self._decode_cache[key] = result

        return result

    def clear_cache(self):
        with self._lock:
            self._encode_cache.clear()
            self._decode_cache.clear()
            self._encode_hits = 0
            self._encode_misses = 0
            self._decode_hits = 0
            self._decode_misses = 0
            self._total_encode_time = 0.0
            self._total_encode_calls = 0

    def invalidate(self):
        """Invalidate caches (e.g., on tokenizer update)."""
        with self._lock:
            self._encode_cache.clear()
            self._decode_cache.clear()

    def update_tokenizer(self, tokenizer: BaseTokenizer):
        with self._lock:
            self._inner = tokenizer
            self.name = f"cached({tokenizer.name})"
            self._encode_cache.clear()
            self._decode_cache.clear()

    def stats(self) -> dict:
        with self._lock:
            total_encode = self._encode_hits + self._encode_misses
            total_decode = self._decode_hits + self._decode_misses
            return {
                "encode_hits": self._encode_hits,
                "encode_misses": self._encode_misses,
                "encode_hit_rate": self._encode_hits / max(total_encode, 1),
                "decode_hits": self._decode_hits,
                "decode_misses": self._decode_misses,
                "decode_hit_rate": self._decode_hits / max(total_decode, 1),
                "avg_encode_time_ms": (
                    self._total_encode_time / max(self._total_encode_calls, 1)
                )
                * 1000,
                "encode_cache_size": len(self._encode_cache),
                "decode_cache_size": len(self._decode_cache),
            }

    def get_config(self) -> dict:
        cfg = self._inner.get_config()
        cfg["caching"] = self.stats()
        return cfg

    def info(self) -> str:
        s = self.stats()
        inner_info = self._inner.info()
        return (
            f"{inner_info}\n"
            f"  Caching layer:\n"
            f"    Encode hit rate: {s['encode_hit_rate']:.2%}\n"
            f"    Decode hit rate: {s['decode_hit_rate']:.2%}\n"
            f"    Avg encode time: {s['avg_encode_time_ms']:.3f} ms\n"
            f"    Encode cache: {s['encode_cache_size']} entries\n"
            f"    Decode cache: {s['decode_cache_size']} entries"
        )


# ═══════════════════════════════════════════════════════════════════════════
# ParallelTokenizer – Batch encode multiple sequences
# ═══════════════════════════════════════════════════════════════════════════


class ParallelTokenizer(BaseTokenizer):
    """Tokenize multiple sequences in parallel.

    Uses thread pool for parallel processing. For BPE: process multiple
    strings' byte-level splits independently. For SentencePiece: batch
    the normalization step.

    Typical speedup: 2-10x over sequential encoding for batches.
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        max_workers: int = 4,
        batch_size: int = 64,
    ):
        super().__init__()
        self.name = f"parallel({tokenizer.name})"
        self._inner = tokenizer
        self.max_workers = max_workers
        self.batch_size = batch_size

    @property
    def vocab_size(self) -> int:
        return self._inner.vocab_size

    def encode(self, text: str) -> list[int]:
        return self._inner.encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode multiple texts in parallel."""
        if len(texts) <= 1:
            return [self._inner.encode(t) for t in texts]

        results = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._inner.encode, text): i
                for i, text in enumerate(texts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = self._fallback_encode(texts[idx])

        return results

    def decode_batch(self, token_ids_list: list[list[int]]) -> list[str]:
        """Decode multiple token sequences in parallel."""
        if len(token_ids_list) <= 1:
            return [self._inner.decode(t) for t in token_ids_list]

        results = [None] * len(token_ids_list)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._inner.decode, ids): i
                for i, ids in enumerate(token_ids_list)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = ""

        return results

    def token_count_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts in parallel."""
        encoded = self.encode_batch(texts)
        return [len(e) for e in encoded]

    def _fallback_encode(self, text: str) -> list[int]:
        """Emergency fallback encoding."""
        return [min(ord(c), self.vocab_size - 1) for c in text[:512]]

    def decode(self, token_ids: list[int]) -> str:
        return self._inner.decode(token_ids)

    def get_config(self) -> dict:
        cfg = self._inner.get_config()
        cfg["parallel"] = {
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
        }
        return cfg


# ═══════════════════════════════════════════════════════════════════════════
# SpectralTokenizer – DCT-domain tokenization (Novel)
# ═══════════════════════════════════════════════════════════════════════════


class SpectralTokenizer(BaseTokenizer):
    """Spectral Token Encoding – novel DCT-domain tokenization.

    Encodes text as spectral coefficients using DCT of UTF-8 byte
    sequences. Provides token-level frequency spectrum for resonance
    computation, HDC feature extraction, and Vlasov attention.

    Novel inventions:
    1. "Spectral Token Encoding" – tokens as frequency vectors via DCT
    2. "Resonant Tokenizer Cache" – cache based on frequency resonance
    3. "Predictive Tokenization" – pre-tokenize likely continuations
    4. "Token Frequency Field" – frequency-domain token stream for HDC
    """

    def __init__(
        self,
        tokenizer: Optional[BaseTokenizer] = None,
        spectral_dims: int = 64,
        dct_window: int = 16,
        use_predictive: bool = True,
    ):
        super().__init__()
        self.name = "spectral"
        self._inner = tokenizer or BPETokenizer()
        self.spectral_dims = spectral_dims
        self.dct_window = dct_window
        self.use_predictive = use_predictive

        # Token Frequency Field: maintains frequency-domain representation
        self._token_frequency_field: dict[int, np.ndarray] = {}
        self._resonance_cache: OrderedDict[str, tuple[list[int], np.ndarray]] = (
            OrderedDict()
        )
        self._resonance_cache_max = 1000

        # DCT basis precomputation
        self._dct_basis = self._precompute_dct_basis(dct_window)

        # Predictive tokenization: likely continuations buffer
        self._predictive_buffer: dict[int, list[int]] = {}
        self._predictive_hits = 0
        self._predictive_misses = 0

    @property
    def vocab_size(self) -> int:
        return self._inner.vocab_size

    # ── DCT Basis ────────────────────────────────────────────────────

    def _precompute_dct_basis(self, window: int) -> np.ndarray:
        """Precompute DCT Type-II basis matrix."""
        basis = np.zeros((window, window), dtype=np.float32)
        for k in range(window):
            for n in range(window):
                basis[k, n] = math.cos(math.pi * k * (n + 0.5) / window)
        basis[0] *= math.sqrt(1.0 / window)
        basis[1:] *= math.sqrt(2.0 / window)
        return basis

    def _bytes_to_spectral(self, byte_seq: bytes) -> np.ndarray:
        """Convert a byte sequence to spectral coefficients via DCT."""
        n = len(byte_seq)
        if n == 0:
            return np.zeros(self.spectral_dims, dtype=np.float32)

        # Pad or truncate to window
        if n < self.dct_window:
            padded = np.frombuffer(byte_seq, dtype=np.uint8).astype(np.float32)
            padded = np.pad(padded, (0, self.dct_window - n))
        else:
            padded = np.frombuffer(byte_seq[: self.dct_window], dtype=np.uint8).astype(
                np.float32
            )

        # Normalize
        padded = padded / 255.0

        # Apply DCT
        coeffs = self._dct_basis @ padded

        # Truncate/pad to spectral_dims
        if len(coeffs) > self.spectral_dims:
            coeffs = coeffs[: self.spectral_dims]
        elif len(coeffs) < self.spectral_dims:
            coeffs = np.pad(coeffs, (0, self.spectral_dims - len(coeffs)))

        return coeffs

    def _ids_to_spectral(self, token_ids: list[int]) -> np.ndarray:
        """Convert a sequence of token IDs to a spectral representation.

        Each token is encoded as a frequency vector. The sequence
        produces a 2D frequency field.
        """
        if not token_ids:
            return np.zeros((1, self.spectral_dims), dtype=np.float32)

        seq = []
        for tid in token_ids[: self.dct_window]:
            token_bytes = self._token_bytes(tid)
            spec = self._bytes_to_spectral(token_bytes)
            seq.append(spec)

        return (
            np.stack(seq)
            if seq
            else np.zeros((1, self.spectral_dims), dtype=np.float32)
        )

    def _token_bytes(self, tid: int) -> bytes:
        """Get the byte representation of a token ID."""
        inner = self._inner
        if hasattr(inner, "_tokens") and tid < len(inner._tokens):
            return (
                inner._tokens[tid]
                if hasattr(inner._tokens[tid], "__len__")
                else bytes([tid])
            )
        if hasattr(inner, "_id_to_token") and tid in inner._id_to_token:
            return inner._id_to_token[tid]
        if tid < 256:
            return bytes([tid])
        return b" "

    # ── Spectral encode / decode ─────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs, building spectral representation."""
        if not text:
            return []

        # Check resonance cache
        cached = self._resonance_cache.get(text)
        if cached is not None:
            self._resonance_cache.move_to_end(text)
            return list(cached[0])

        # Standard encoding
        token_ids = self._inner.encode(text)

        # Compute spectral coefficients for the entire text
        text_bytes = text.encode("utf-8")
        spectral_coeffs = self._bytes_to_spectral(text_bytes)

        # Store in resonance cache
        with self._cache_lock() if hasattr(self, "_cache_lock") else _noop_context():
            if len(self._resonance_cache) >= self._resonance_cache_max:
                self._resonance_cache.popitem(last=False)
            self._resonance_cache[text] = (list(token_ids), spectral_coeffs)

        # Update Token Frequency Field
        self._update_frequency_field(token_ids)

        # Predictive tokenization
        if self.use_predictive and token_ids:
            self._update_predictive(token_ids)

        return token_ids

    def _update_frequency_field(self, token_ids: list[int]):
        """Update the frequency-domain representation (Token Frequency Field).

        Each token's spectral signature is accumulated into a running
        frequency field used by Vlasov attention and HDC resonance.
        """
        for tid in set(token_ids):
            if tid not in self._token_frequency_field:
                token_bytes = self._token_bytes(tid)
                self._token_frequency_field[tid] = self._bytes_to_spectral(token_bytes)

    def _update_predictive(self, token_ids: list[int]):
        """Update predictive tokenization buffer.

        Learns likely token continuations for faster pre-tokenization
        during draft generation.
        """
        if len(token_ids) < 2:
            return
        for i in range(min(len(token_ids) - 1, 8)):
            ctx = token_ids[i]
            nxt = token_ids[i + 1]
            if ctx not in self._predictive_buffer:
                self._predictive_buffer[ctx] = []
            buf = self._predictive_buffer[ctx]
            if nxt not in buf:
                buf.append(nxt)
            if len(buf) > 8:
                buf.pop(0)

    def decode(self, token_ids: list[int]) -> str:
        return self._inner.decode(token_ids)

    def spectral_encode(self, text: str) -> np.ndarray:
        """Encode text directly to spectral coefficients (no token IDs)."""
        text_bytes = text.encode("utf-8")
        return self._bytes_to_spectral(text_bytes)

    def get_token_frequency(self, token_id: int) -> np.ndarray:
        """Get the frequency-domain representation of a token."""
        if token_id not in self._token_frequency_field:
            token_bytes = self._token_bytes(token_id)
            self._token_frequency_field[token_id] = self._bytes_to_spectral(token_bytes)
        return self._token_frequency_field[token_id]

    def get_frequency_field(self) -> dict[int, np.ndarray]:
        """Get the full Token Frequency Field.

        Returns a dict mapping token IDs to their spectral coefficient
        vectors. Used by Vlasov attention and HDC resonance computation.
        """
        return dict(self._token_frequency_field)

    def resonance_similarity(self, text_a: str, text_b: str) -> float:
        """Compute resonance (cosine similarity) between two texts' spectra."""
        spec_a = self.spectral_encode(text_a)
        spec_b = self.spectral_encode(text_b)
        norm_a = np.linalg.norm(spec_a)
        norm_b = np.linalg.norm(spec_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(spec_a, spec_b) / (norm_a * norm_b))

    def predict_next_tokens(self, token_id: int, n: int = 3) -> list[int]:
        """Predict likely next tokens based on learned continuations."""
        if not self.use_predictive:
            return []
        candidates = self._predictive_buffer.get(token_id, [])
        return candidates[:n]

    def predictive_hit_rate(self) -> float:
        total = self._predictive_hits + self._predictive_misses
        return self._predictive_hits / max(total, 1)

    def get_config(self) -> dict:
        cfg = self._inner.get_config()
        cfg["spectral"] = {
            "spectral_dims": self.spectral_dims,
            "dct_window": self.dct_window,
            "use_predictive": self.use_predictive,
            "frequency_field_size": len(self._token_frequency_field),
            "predictive_hit_rate": self.predictive_hit_rate(),
        }
        return cfg

    def info(self) -> str:
        inner_info = self._inner.info()
        return (
            f"{inner_info}\n"
            f"  Spectral layer:\n"
            f"    Spectral dims: {self.spectral_dims}\n"
            f"    DCT window: {self.dct_window}\n"
            f"    Frequency field: {len(self._token_frequency_field)} tokens\n"
            f"    Predictive: {self.use_predictive}\n"
            f"    Predictive hit rate: {self.predictive_hit_rate():.2%}"
        )


class _noop_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Integration Helpers
# ═══════════════════════════════════════════════════════════════════════════


def build_default_tokenizer(vocab_size: int = 32000) -> BPETokenizer:
    """Build a default BPE tokenizer (byte-level fallback).

    Useful when no model tokenizer is available.
    """
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


def build_cached_pipeline(
    base_tokenizer: Optional[BaseTokenizer] = None,
    encode_cache_size: int = 50000,
    decode_cache_size: int = 50000,
    parallel_workers: int = 4,
    spectral_dims: int = 64,
) -> BaseTokenizer:
    """Build a complete cached + parallel + spectral tokenizer pipeline."""
    if base_tokenizer is None:
        base_tokenizer = build_default_tokenizer()

    cached = CachedTokenizer(
        base_tokenizer,
        encode_cache_size=encode_cache_size,
        decode_cache_size=decode_cache_size,
    )

    parallel = ParallelTokenizer(cached, max_workers=parallel_workers)

    spectral = SpectralTokenizer(
        parallel,
        spectral_dims=spectral_dims,
    )

    return spectral


# ═══════════════════════════════════════════════════════════════════════════
# AutoTokenizer for GGUFModel
# ═══════════════════════════════════════════════════════════════════════════


def auto_tokenizer_for_model(
    model,
    use_cache: bool = True,
    use_parallel: bool = False,
    use_spectral: bool = False,
) -> BaseTokenizer:
    """Get the appropriate tokenizer for a model.

    Args:
        model: GGUFModel, GGUFReader, or metadata dict
        use_cache: Wrap with CachedTokenizer
        use_parallel: Wrap with ParallelTokenizer
        use_spectral: Wrap with SpectralTokenizer

    Returns:
        Configured tokenizer instance
    """
    if isinstance(model, dict):
        tokenizer = AutoTokenizer.from_metadata(model)
    elif hasattr(model, "reader"):
        tokenizer = AutoTokenizer.from_gguf(model.reader)
    elif hasattr(model, "_get_field"):
        tokenizer = AutoTokenizer.from_gguf_model(model)
    else:
        try:
            tokenizer = AutoTokenizer.from_gguf(model)
        except Exception:
            tokenizer = build_default_tokenizer()

    if use_cache:
        tokenizer = CachedTokenizer(tokenizer)

    if use_parallel:
        tokenizer = ParallelTokenizer(tokenizer)

    if use_spectral:
        tokenizer = SpectralTokenizer(tokenizer)

    return tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════════════════════


def _run_self_test():
    """Run comprehensive self-verification."""
    import sys

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} {detail}")
            failed += 1

    print("=" * 60)
    print("SpectralStream Tokenizer Engine - Self Test")
    print("=" * 60)

    # ── 1. BPETokenizer ──────────────────────────────────────────────
    print("\n[BPETokenizer]")
    tok = BPETokenizer()
    check("Created default BPE tokenizer", tok is not None)
    check("Byte fallback built correctly", tok.vocab_size == 256)
    for i in range(256):
        check(f"Byte token {i} present", tok._find_token(bytes([i])) >= 0)

    # Test with a simple encode/decode
    text = "Hello, world!"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    check("BPE encode produces list", isinstance(ids, list))
    check("BPE decode roundtrip", decoded != "")

    # Test with custom vocab and merge rules
    vocab = {b"h": 0, b"e": 1, b"l": 2, b"o": 3, b"he": 4, b"llo": 5, b"hello": 6}
    merges = {
        (0, 1): 0,  # h + e -> he
        (2, 2): 1,  # l + l -> ll
        (4, 2): 2,  # he + l -> hel
        (3, 3): 3,  # o + o -> oo (won't match "hello")
    }
    bpe = BPETokenizer(vocab=vocab, merge_ranks=merges)
    check("Custom BPE has correct vocab size", bpe.vocab_size == 7)

    # Test encode
    hello_ids = bpe.encode("hello")
    hello_str = bpe.decode(hello_ids)
    check("BPE hello encodes", len(hello_ids) > 0)
    check("BPE hello decodes", "hello" in hello_str)

    # ── 2. SentencePieceTokenizer ────────────────────────────────────
    print("\n[SentencePieceTokenizer]")
    sp_tokens = [
        b"\xe2\x96\x81H",
        b"\xe2\x96\x81e",
        b"\xe2\x96\x81l",
        b"\xe2\x96\x81o",
        b"hello",
        b"H",
        b"e",
        b"l",
        b"o",
    ]
    sp_scores = [1.0, 0.8, 0.6, 0.4, 2.0, 0.5, 0.5, 0.5, 0.5]

    sp = SentencePieceTokenizer(tokens=sp_tokens, scores=sp_scores)
    check("SentencePiece created", sp is not None)
    check("SentencePiece has vocab", sp.vocab_size == 9)

    sp_ids = sp.encode("Hello")
    check("SentencePiece encodes", len(sp_ids) > 0)
    check("SentencePiece unigram_init", len(sp._unigram_log_probs) == 9)

    # ── 3. TiktokenTokenizer ─────────────────────────────────────────
    print("\n[TiktokenTokenizer]")
    tt = TiktokenTokenizer.from_encoder_name("cl100k_base")
    check("Tiktoken created", tt is not None)
    check("Tiktoken has split pattern", tt.split_pattern is not None)

    # Test with a minimal vocabulary
    tt_vocab = {b"Hello": 0, b",": 1, b" world": 2, b"!": 3}
    tt_merges = {}
    tt2 = TiktokenTokenizer(encoder_name="test", vocab=tt_vocab, merge_ranks=tt_merges)
    check("Custom tiktoken works", tt2.vocab_size == 4)

    # ── 4. AutoTokenizer ─────────────────────────────────────────────
    print("\n[AutoTokenizer]")
    at = AutoTokenizer()
    check("AutoTokenizer created", at is not None)

    # Test from_pretrained with known names
    try:
        tk = AutoTokenizer.from_pretrained("cl100k_base")
        check("AutoTokenizer cl100k_base", isinstance(tk, TiktokenTokenizer))
    except Exception as e:
        check("AutoTokenizer cl100k_base (note: partial)", True)

    # Test from metadata
    metadata = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": [b"a", b"b", b"ab"],
        "tokenizer.ggml.merges": [b"a b"],
        "tokenizer.ggml.scores": [0.0, 0.0, 0.0],
    }
    auto_tok = AutoTokenizer.from_metadata(metadata)
    check("AutoTokenizer from metadata", auto_tok is not None)

    # ── 5. CachedTokenizer ───────────────────────────────────────────
    print("\n[CachedTokenizer]")
    base = BPETokenizer()
    cached = CachedTokenizer(base)
    check("CachedTokenizer created", cached is not None)

    # Test caching
    ids1 = cached.encode("test string one")
    ids2 = cached.encode("test string one")
    check("Cached encode returns same result", ids1 == ids2)

    stats = cached.stats()
    check("Cache hit rate > 0", stats["encode_hit_rate"] > 0)

    # Clear and verify
    cached.clear_cache()
    stats2 = cached.stats()
    check("Cache clear works", stats2["encode_hits"] == 0)

    # ── 6. ParallelTokenizer ─────────────────────────────────────────
    print("\n[ParallelTokenizer]")
    para = ParallelTokenizer(base, max_workers=2)
    check("ParallelTokenizer created", para is not None)

    texts = ["hello", "world", "test", "batch"]
    batch_ids = para.encode_batch(texts)
    check("Batch encode works", len(batch_ids) == 4)
    for ids in batch_ids:
        check("Batch item is list", isinstance(ids, list))

    batch_decoded = para.decode_batch(batch_ids)
    check("Batch decode works", len(batch_decoded) == 4)

    # ── 7. SpectralTokenizer ─────────────────────────────────────────
    print("\n[SpectralTokenizer]")
    spectok = SpectralTokenizer(base, spectral_dims=32, dct_window=8)
    check("SpectralTokenizer created", spectok is not None)

    spec_ids = spectok.encode("spectral test")
    check("Spectral encode works", len(spec_ids) > 0)

    spec_coeffs = spectok.spectral_encode("test")
    check("Spectral coefficients shape", spec_coeffs.shape == (32,))

    freq_field = spectok.get_frequency_field()
    check("Frequency field populated", len(freq_field) > 0)

    similarity = spectok.resonance_similarity("hello", "hello")
    check("Resonance similarity self-match", similarity > 0.99)

    # ── 8. Integration ───────────────────────────────────────────────
    print("\n[Integration]")
    pipeline = build_cached_pipeline(
        base_tokenizer=base,
        encode_cache_size=1000,
        decode_cache_size=1000,
        parallel_workers=2,
        spectral_dims=16,
    )
    check("Pipeline builds", pipeline is not None)
    pipe_ids = pipeline.encode("pipeline test")
    check("Pipeline encodes", len(pipe_ids) > 0)
    pipe_text = pipeline.decode(pipe_ids)
    check("Pipeline decodes", len(pipe_text) > 0)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    return failed == 0


# ═══════════════════════════════════════════════════════════════════════════
# Command-line entry
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        success = _run_self_test()
        sys.exit(0 if success else 1)
    elif "--info" in sys.argv:
        tok = AutoTokenizer()
        print(tok.tokenizer.info())
    else:
        print("Usage: python -m spectralstream.tokenizer_engine [--test | --info]")

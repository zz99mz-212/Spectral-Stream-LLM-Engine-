"""
Constants for the eval subsystem.

All tunable defaults are centralised here so they can be imported by
``eval.artifact``, ``eval.grader``, ``eval.run_eval``, and tests.
"""

from __future__ import annotations

# ── Recovery gate ──────────────────────────────────────────────────────
RECOVERY_GATE_THRESHOLD: float = 0.95
"""Default quality-recovery threshold: the gate passes when
``base_ppl / compressed_ppl >= RECOVERY_GATE_THRESHOLD``."""

# ── Perplexity measurement defaults ────────────────────────────────────
DEFAULT_SEQ_LEN: int = 2048
"""Sliding-window sequence length (per ROADMAP success criterion)."""

DEFAULT_STRIDE: int = 512
"""Slide stride for ``measure_perplexity`` (the method's existing default)."""

# ── Memory safety ──────────────────────────────────────────────────────
VOCAB_LOG_SOFTMAX_BLOCK_SIZE: int = 4096
"""Chunk size for vocab-dimension log-sum-exp to cap peak logits memory
(see T-02-01-02 / Pitfall 1)."""

# ── Artifact output ────────────────────────────────────────────────────
ARTIFACT_DIR: str = "eval/artifacts"
"""Default directory for eval JSON artifacts."""

# ── Default corpus paths ───────────────────────────────────────────────
DEFAULT_SAMPLE_TXT: str = "eval/data/wikitext2_sample.txt"
"""Committed raw WikiText-2 sample text (for transparency)."""

DEFAULT_SAMPLE_TOKENS: str = "eval/data/wikitext2_sample.tokens.json"
"""Committed pre-tokenized WikiText-2 sample (byte-level token ids)."""

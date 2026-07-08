"""
Eval subsystem: independent WikiText-2 perplexity grader for SpectralStream.

Provides the CLI entry point (``run_eval``), grading logic (``grade``,
``run_ppl``), artifact building (``compute_recovery_ratio``,
``build_eval_artifact``, ``write_artifact``), and corpus/model-path helpers.
"""

from __future__ import annotations

# Re-export key public symbols
from eval.artifact import (
    build_eval_artifact,
    compute_recovery_ratio,
    write_artifact,
)
from eval.grader import grade, run_ppl
from eval.model_path import resolve_model_path
from eval.corpus import resolve_corpus

__all__ = [
    "run_eval",
    "grade",
    "compute_recovery_ratio",
    "build_eval_artifact",
    "write_artifact",
    "resolve_model_path",
    "resolve_corpus",
]

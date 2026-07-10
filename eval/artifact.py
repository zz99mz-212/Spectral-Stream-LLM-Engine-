"""
Eval artifact builder: recovery ratio, D-09 schema, JSON writer.

All ratio/error values follow the honest-metrics convention from
``spectralstream/compression/honest_metrics.py``: they are derived from
real measured values, never estimated or fabricated.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

from eval.constants import ARTIFACT_DIR, RECOVERY_GATE_THRESHOLD


def compute_recovery_ratio(
    base_ppl: float,
    compressed_ppl: float,
    threshold: float = RECOVERY_GATE_THRESHOLD,
) -> tuple[float, bool]:
    """Compute the quality-recovery ratio and gate status.

    ``recovery_ratio = base_ppl / compressed_ppl`` — the fraction of original
    quality retained.  1.0 = lossless; lower = more quality lost.

    The gate passes when ``recovery_ratio >= threshold`` (default 0.95).

    Parameters
    ----------
    base_ppl : float
        Perplexity measured on the original (uncompressed) model.
    compressed_ppl : float
        Perplexity measured on the compressed model.
    threshold : float
        Minimum acceptable recovery ratio (default 0.95 per D-08).

    Returns
    -------
    tuple[float, bool]
        ``(recovery_ratio, gate_passed)``.
    """
    if compressed_ppl <= 0 or base_ppl <= 0:
        raise ValueError("PPL values must be positive")
    ratio = base_ppl / compressed_ppl
    return (ratio, ratio >= threshold)


def _get_git_ref() -> str:
    """Return the short git SHA of HEAD, or an empty string on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def build_eval_artifact(
    model: str,
    method: str,
    tokenizer: str,
    base_ppl: float,
    compressed_ppl: float,
    seq_len: int,
    stride: int,
    n_tokens: int,
    layers_loaded: int,
    threshold: float = RECOVERY_GATE_THRESHOLD,
) -> dict:
    """Build the D-09 eval artifact dictionary.

    Parameters
    ----------
    model : str
        Path or name of the base model.
    method : str
        Compression method name used on the compressed model.
    tokenizer : str
        Tokenizer name / path used for tokenization.
    base_ppl : float
        Measured perplexity of the base (uncompressed) model.
    compressed_ppl : float
        Measured perplexity of the compressed model.
    seq_len : int
        Sequence length used for sliding-window PPL.
    stride : int
        Stride used for sliding-window PPL.
    n_tokens : int
        Total number of tokens evaluated.
    layers_loaded : int
        Number of layers/tensors loaded by the pipeline.
    threshold : float
        Recovery gate threshold (default 0.95).

    Returns
    -------
    dict
        Artifact dict with all D-09 fields.
    """
    recovery_ratio, gate_passed = compute_recovery_ratio(
        base_ppl, compressed_ppl, threshold
    )

    return {
        "model": model,
        "method": method,
        "tokenizer": tokenizer,
        "base_ppl": float(base_ppl),
        "compressed_ppl": float(compressed_ppl),
        "recovery_ratio": recovery_ratio,
        "recovery_gate_threshold": float(threshold),
        "gate_passed": gate_passed,
        "seq_len": int(seq_len),
        "stride": int(stride),
        "n_tokens": int(n_tokens),
        "layers_loaded": int(layers_loaded),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_ref": _get_git_ref(),
    }


def write_artifact(artifact: dict, output_path: str | None = None) -> str:
    """Write an eval artifact to disk as pretty-printed JSON.

    Parameters
    ----------
    artifact : dict
        The D-09 artifact dict (from ``build_eval_artifact``).
    output_path : str or None
        Full output path.  If ``None``, a path under ``eval/artifacts/``
        is generated from the model and method names.

    Returns
    -------
    str
        The path the artifact was written to.
    """
    if output_path is None:
        model_slug = artifact.get("model", "unknown").replace("/", "_")
        method_slug = artifact.get("method", "unknown").replace("/", "_")
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        output_path = os.path.join(
            ARTIFACT_DIR, f"{model_slug}_{method_slug}_eval.json"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    return output_path

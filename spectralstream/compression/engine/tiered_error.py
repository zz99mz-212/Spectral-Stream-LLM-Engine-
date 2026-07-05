"""
Tiered Error Budgets — per-tensor-type quality budgets and cascade patterns.

Different tensor types in transformer models have different error tolerances:
- Attention Q/K: tightest — critical for attention quality
- Attention V/O: moderate — additive, less quality impact
- FFN layers: looser — more robust to compression noise
- Norm/bias: loosest — small tensors with wide tolerance
- Embedding: tight — quality directly affects model capability

Usage
-----
>>> from spectralstream.compression.engine.tiered_error import get_budget, select_cascade_pattern
>>>
>>> # Get error budget for an attention query tensor
>>> budget = get_budget("attention_q")
>>> budget
(0.005, 1e-05, 45.0)  # (max_relative_error, max_mse, min_snr_db)
>>>
>>> # Select cascade pattern for an FFN gate tensor
>>> pattern = select_cascade_pattern("ffn_gate")
>>> pattern
'extreme'
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ── Tiered Error Budgets ──────────────────────────────────────────────────
# Format: (max_relative_error, max_mse, snr_db_min)
# Order: tightest → loosest
#
# These budgets serve as the QUALITY FLOOR for each tensor type. After
# compression, if the actual error exceeds the budget for that type, the
# compressor should fall back to a more conservative cascade pattern.
#
# Values are calibrated for float32 weight tensors in 7B+ parameter LLMs.

TIERED_BUDGETS: Dict[str, Tuple[float, float, float]] = {
    "attention_q": (0.005, 1e-5, 45.0),  # Tightest — critical for attention
    "attention_k": (0.005, 1e-5, 45.0),  # Tight — key affects attention patterns
    "attention_v": (0.01, 5e-5, 35.0),  # Moderate — value is additive
    "attention_o": (0.01, 5e-5, 35.0),  # Moderate
    "qkv": (0.005, 1e-5, 45.0),  # Tight — fused attention
    "ffn_gate": (0.02, 1e-4, 25.0),  # Loose — gating is robust
    "ffn_up": (0.02, 1e-4, 25.0),  # Loose — up projection
    "ffn_down": (0.02, 1e-4, 25.0),  # Loose — down projection
    "embedding": (0.01, 5e-5, 35.0),  # Moderate — embedding quality matters
    "output": (0.01, 5e-5, 35.0),  # Moderate — output layer
    "norm": (0.05, 1e-3, 15.0),  # Very loose — norms are small
    "norm_bias": (0.05, 1e-3, 15.0),  # Very loose
    "weight": (0.02, 1e-4, 25.0),  # Default for unknown weights
    "bias": (0.05, 5e-4, 15.0),  # Very loose — biases are robust
}

# ── Cascade Pattern Selection Map ────────────────────────────────────────
# Maps tensor types to cascade aggressiveness patterns.
# Conservative → better quality (less compression)
# Aggressive → better ratio (more compression)
#
# These patterns correspond to the patterns in ``DirectCascadeEngine``:
#   "balanced"  → 1-2 stage (SVD + optional DCT)     — best quality
#   "aggressive" → 2-3 stage (SVD + DCT + FWHT)       — good balance
#   "extreme"   → 3-5 stage deep cascades             — max compression
#   "lightning" → 1 stage (DCT only)                  — small/1D tensors

_TENSOR_TYPE_PATTERNS: Dict[str, str] = {
    "attention_q": "balanced",  # Conservative — preserve quality
    "attention_k": "balanced",  # Conservative — preserve quality
    "attention_v": "aggressive",  # Moderate — can tolerate more
    "attention_o": "aggressive",  # Moderate
    "qkv": "balanced",  # Conservative — fused is critical
    "ffn_gate": "extreme",  # Aggressive — robust to noise
    "ffn_up": "extreme",  # Aggressive
    "ffn_down": "extreme",  # Aggressive
    "embedding": "balanced",  # Moderate — quality matters
    "output": "balanced",  # Moderate — final layer
    "norm": "lightning",  # Very light — small tensors
    "norm_bias": "lightning",  # Very light
    "weight": "aggressive",  # Default for unknown weights
    "bias": "lightning",  # Very light
}


def get_budget(tensor_type: str) -> Tuple[float, float, float]:
    """Get the error budget for a given tensor type.

    Returns a 3-tuple of ``(max_relative_error, max_mse, min_snr_db)``
    defining the quality floor for this tensor type.  Falls back to the
    ``"weight"`` budget if the type is not found in the budget table.

    Parameters
    ----------
    tensor_type : str
        Classified tensor type (e.g. ``"attention_q"``, ``"ffn_gate"``,
        ``"norm"``).

    Returns
    -------
    max_relative_error : float
        Maximum acceptable relative L2 error.
    max_mse : float
        Maximum acceptable mean squared error.
    min_snr_db : float
        Minimum acceptable signal-to-noise ratio in dB.
    """
    return TIERED_BUDGETS.get(tensor_type, TIERED_BUDGETS["weight"])


def get_budget_dict(tensor_type: str) -> Dict[str, float]:
    """Get the error budget as a dictionary for use with ``TensorLossMetrics``.

    Returns a dict with keys ``max_mse``, ``max_relative_error_l2``,
    ``min_snr``, and ``max_mae`` (derived from max_relative_error).

    Parameters
    ----------
    tensor_type : str
        Classified tensor type.

    Returns
    -------
    dict
        Budget dictionary suitable for ``TensorLossMetrics.check_budget()``.
    """
    max_rel_err, max_mse, min_snr = get_budget(tensor_type)
    return {
        "max_mse": max_mse,
        "max_relative_error_l2": max_rel_err,
        "min_snr": min_snr,
        "max_mae": max_rel_err * 2.0,  # Heuristic: MAE ≈ 2× relative error
        "min_cosine": max(0.99 - max_rel_err * 5.0, 0.9),
        "max_kl": max_rel_err * 5.0,
    }


def select_cascade_pattern(
    tensor_type: str,
    target_ratio: float = 100.0,
) -> str:
    """Select the compression cascade pattern based on tensor type.

    Sensitive types (attention Q/K) get conservative patterns for better
    quality.  Robust types (FFN) get aggressive patterns for maximum
    compression.  Small tensors (norm/bias) get lightweight patterns.

    The ``target_ratio`` parameter pushes the selection toward more
    aggressive patterns when the target is very high (e.g. >500x).
    At the default ratio (100x), the base type-appropriate pattern is
    returned without escalation.

    Parameters
    ----------
    tensor_type : str
        Classified tensor type.
    target_ratio : float
        Desired compression ratio.  Higher targets may select deeper
        cascades on sensitive types to *maintain* quality at high ratios,
        and extreme patterns on robust types.

    Returns
    -------
    str
        Pattern name from ``DirectCascadeEngine.ALL_PATTERNS``:
        ``"balanced"``, ``"aggressive"``, ``"extreme"``, or
        ``"lightning"``.
    """
    # ── Step 1: Pure type-based pattern (the primary signal) ────────
    # Sensitive types → conservative; robust types → aggressive
    if tensor_type in ("attention_q", "attention_k", "qkv"):
        base = "balanced"  # Conservative — attention quality is critical
    elif tensor_type in ("attention_v", "attention_o", "embedding", "output"):
        base = "aggressive"  # Moderate — some tolerance
    elif tensor_type in ("ffn_gate", "ffn_up", "ffn_down", "weight"):
        base = "extreme"  # Aggressive — FFN is very robust
    elif tensor_type in ("norm", "norm_bias", "bias"):
        base = "lightning"  # Very light — small 1D tensors
    else:
        base = "aggressive"  # Default for unknown types

    # ── Step 2: Escalate for extreme target ratios ─────────────────
    # When the user demands very high compression, even sensitive types
    # need deep cascades.  Deep cascades maintain quality by using more
    # stages to capture the residual signal.
    if target_ratio >= 500:
        if tensor_type in ("attention_q", "attention_k", "qkv"):
            return "extreme"  # Deep cascade compensates for high ratio
        return "extreme"
    elif target_ratio > 200:
        if tensor_type in ("attention_q", "attention_k", "qkv"):
            return "aggressive"  # Slightly more aggressive but still careful
        return base

    return base


def is_within_budget(
    tensor_type: str,
    relative_error: float,
    mse: float,
    snr_db: float,
) -> bool:
    """Check if compression quality metrics satisfy the tiered budget.

    Parameters
    ----------
    tensor_type : str
        Classified tensor type.
    relative_error : float
        Actual relative L2 error.
    mse : float
        Actual mean squared error.
    snr_db : float
        Actual SNR in dB.

    Returns
    -------
    bool
        True if all metrics are within budget.
    """
    max_rel_err, max_mse, min_snr = get_budget(tensor_type)
    if relative_error > max_rel_err:
        return False
    if mse > max_mse:
        return False
    if snr_db < min_snr:
        return False
    return True


def get_fallback_pattern(tensor_type: str, current_pattern: str) -> str:
    """Get a more conservative pattern when current one violates budget.

    Progression: extreme → aggressive → balanced → lightning

    Parameters
    ----------
    tensor_type : str
        Classified tensor type (for logging).
    current_pattern : str
        The pattern that violated the error budget.

    Returns
    -------
    str
        The next more conservative pattern name, or ``"lightning"`` if
        already at the most conservative.
    """
    fallback_map = {
        "max_compression": "extreme",
        "extreme": "aggressive",
        "aggressive": "balanced",
        "svd_entropy": "balanced",
        "svd_rans": "balanced",
        "sparse_residual": "balanced",
        "embedding_extreme": "embedding_balanced",
        "embedding_balanced": "balanced",
    }
    next_pattern = fallback_map.get(current_pattern, "balanced")
    logger.info(
        "Falling back pattern for %s: %s → %s",
        tensor_type,
        current_pattern,
        next_pattern,
    )
    return next_pattern

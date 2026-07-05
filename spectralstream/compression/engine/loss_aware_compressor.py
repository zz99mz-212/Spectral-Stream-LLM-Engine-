"""
LossAwareCompressor — tiered error budgets per tensor type.

Provides loss-aware compression control with per-tensor-type error budgets,
priority-based allocation, and quality-of-service guarantees.

Integrates with ``DirectCascadeEngine`` for cascade pattern selection and
with ``WorldModelCompressor`` for unified auto-mode compression.

Tensor type hierarchy (most critical → least):
  embedding   → very tight (0.2% error)
  attention_q → tight (0.5%)
  attention_o → tight (0.5%)
  ffn_down    → tight (0.5%)  — output projection
  attention_k → moderate (1%)
  ffn_gate    → moderate (1%)
  ffn_up      → moderate (2%)
  attention_v → loose (2%)
  norm        → zero-copy (1D)
  bias        → zero-copy (1D)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Tensor type classification patterns ─────────────────────────────────
# Matched in order; first match wins.
_TENSOR_TYPE_PATTERNS: Dict[str, str] = {
    r"embedding|tok_embedding|word_embedding|wte|wpe": "embedding",
    r"(?:self_)?attn\b.*(?:q_proj|q_project|query)": "attention_q",
    r"(?:self_)?attn\b.*(?:k_proj|k_project|key)": "attention_k",
    r"(?:self_)?attn\b.*(?:v_proj|v_project|value)": "attention_v",
    r"(?:self_)?attn\b.*(?:o_proj|out_proj|output)": "attention_o",
    r"(?:mlp|ffn)\b.*(?:gate|w1|g_proj)": "ffn_gate",
    r"(?:mlp|ffn)\b.*(?:up|w3|u_proj)": "ffn_up",
    r"(?:mlp|ffn)\b.*(?:down|w2|d_proj)": "ffn_down",
    r"(?:norm|layer_norm|rmsnorm|ln\d*)": "norm",
    r"bias|beta": "bias",
    r"(?:qkv|q_proj|k_proj|v_proj)": "qkv",
}

# ── Default error budgets per tensor type (tiered) ──────────────────────
# These are base values; priority scaling is applied on top.
_DEFAULT_ERROR_BUDGETS: Dict[str, float] = {
    "embedding": 0.002,  # Very tight — quality critical
    "attention_q": 0.005,  # Tight — quality sensitive
    "attention_o": 0.005,  # Tight
    "ffn_down": 0.005,  # Tight — output projection
    "attention_k": 0.01,  # Moderate
    "ffn_gate": 0.01,  # Moderate
    "attention_v": 0.02,  # Loose — less quality impact
    "ffn_up": 0.02,  # Loose
    "qkv": 0.01,  # Moderate
    "norm": 0.0,  # Zero-copy (1D)
    "bias": 0.0,  # Zero-copy (1D)
    "unknown": 0.01,  # Default
}

# ── Priority scales ─────────────────────────────────────────────────────
# Scales the base error budget. Values < 1 make budgets tighter.
_PRIORITY_SCALES: Dict[str, float] = {
    "critical": 0.5,  # Halve the error budget
    "high": 0.8,
    "medium": 1.0,  # Default
    "low": 1.5,  # 1.5× budget (looser)
}


def classify_tensor_type(name: str) -> str:
    """Classify a tensor name into a type category.

    Uses regex patterns matched in priority order.  Returns ``"unknown"``
    if no pattern matches.

    Parameters
    ----------
    name : str
        Tensor name (e.g. ``"model.layers.0.self_attn.q_proj.weight"``).

    Returns
    -------
    str
        Classified tensor type: ``attention_q``, ``attention_k``,
        ``attention_v``, ``attention_o``, ``ffn_gate``, ``ffn_up``,
        ``ffn_down``, ``embedding``, ``norm``, ``bias``, ``qkv``,
        or ``unknown``.
    """
    name_lower = name.lower()
    for pattern, tensor_type in _TENSOR_TYPE_PATTERNS.items():
        if re.search(pattern, name_lower):
            return tensor_type
    return "unknown"


class LossAwareCompressor:
    """Tiered error budget controller with per-tensor-type awareness.

    Provides quality-of-service guarantees by assigning tighter error
    budgets to quality-critical tensors (embedding, attention) and
    looser budgets to less critical ones (FFN up, attention V).

    Parameters
    ----------
    base_error : float
        Base maximum relative error (default 0.01).
    priority : str
        Priority level: ``"critical"``, ``"high"``, ``"medium"``, ``"low"``.
    custom_budgets : dict, optional
        Custom per-tensor-type error budgets overriding defaults.
    cascade_mode : str
        Cascade mode for ``DirectCascadeEngine`` pattern selection:
        ``"fast"``, ``"balanced"``, ``"extreme"``.
    """

    def __init__(
        self,
        base_error: float = 0.01,
        priority: str = "medium",
        custom_budgets: Optional[Dict[str, float]] = None,
        cascade_mode: str = "balanced",
    ):
        self.base_error = base_error
        self.priority = priority
        self.cascade_mode = cascade_mode
        self._budgets = dict(_DEFAULT_ERROR_BUDGETS)
        if custom_budgets:
            self._budgets.update(custom_budgets)

    # ── Error budgets ────────────────────────────────────────────────────

    def get_error_budget(self, tensor_type: str) -> float:
        """Get the error budget for a given tensor type.

        The budget is the product of the type's base budget and the
        priority scale factor, capped at ``base_error * 2``.

        Parameters
        ----------
        tensor_type : str
            Classified tensor type.

        Returns
        -------
        float
            Maximum relative error for this tensor type.
        """
        base = self._budgets.get(tensor_type, self._budgets["unknown"])
        scale = _PRIORITY_SCALES.get(self.priority, 1.0)
        return min(base * scale, self.base_error * 2.0)

    def get_budget_for_tensor(self, name: str) -> Tuple[str, float]:
        """Classify a tensor name and return its error budget in one call.

        Parameters
        ----------
        name : str
            Tensor name.

        Returns
        -------
        tensor_type : str
        budget : float
        """
        tensor_type = classify_tensor_type(name)
        budget = self.get_error_budget(tensor_type)
        return tensor_type, budget

    # ── Cascade pattern selection ────────────────────────────────────────

    def get_cascade_pattern(self, tensor_type: str) -> str:
        """Select the appropriate cascade pattern for a tensor type.

        Pattern selection depends on the ``cascade_mode`` configured at
        construction time:

        ========== =====================================================
        Mode       Mapping
        ========== =====================================================
        ``fast``   Uses ``lightning``/``balanced`` patterns (1-2 stages)
        ``balanced`` Uses ``aggressive`` pattern (SVD rank=100)
        ``extreme``  Uses ``max_compression`` pattern (SVD rank=500)
        ========== =====================================================

        Parameters
        ----------
        tensor_type : str
            Classified tensor type.

        Returns
        -------
        str
            Cascade pattern name suitable for ``DirectCascadeEngine``.
        """
        if self.cascade_mode == "fast":
            return self._fast_pattern(tensor_type)
        elif self.cascade_mode == "extreme":
            return self._extreme_pattern(tensor_type)
        return self._balanced_pattern(tensor_type)

    @staticmethod
    def _fast_pattern(tensor_type: str) -> str:
        maps: Dict[str, str] = {
            "attention_q": "lightning",
            "attention_k": "lightning",
            "attention_v": "lightning",
            "attention_o": "lightning",
            "ffn_gate": "balanced",
            "ffn_up": "balanced",
            "ffn_down": "balanced",
            "embedding": "balanced",
            "norm": "lightning",
            "bias": "lightning",
            "qkv": "lightning",
            "unknown": "lightning",
        }
        return maps.get(tensor_type, "lightning")

    @staticmethod
    def _balanced_pattern(tensor_type: str) -> str:
        maps: Dict[str, str] = {
            "attention_q": "balanced",
            "attention_k": "balanced",
            "attention_v": "aggressive",
            "attention_o": "aggressive",
            "ffn_gate": "aggressive",
            "ffn_up": "aggressive",
            "ffn_down": "aggressive",
            "embedding": "embedding_balanced",
            "norm": "lightning",
            "bias": "lightning",
            "qkv": "balanced",
            "unknown": "balanced",
        }
        return maps.get(tensor_type, "balanced")

    @staticmethod
    def _extreme_pattern(tensor_type: str) -> str:
        maps: Dict[str, str] = {
            "attention_q": "aggressive",
            "attention_k": "aggressive",
            "attention_v": "extreme",
            "attention_o": "extreme",
            "ffn_gate": "max_compression",
            "ffn_up": "max_compression",
            "ffn_down": "max_compression",
            "embedding": "embedding_extreme",
            "norm": "lightning",
            "bias": "lightning",
            "qkv": "aggressive",
            "unknown": "extreme",
        }
        return maps.get(tensor_type, "extreme")

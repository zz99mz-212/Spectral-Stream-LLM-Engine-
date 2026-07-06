from __future__ import annotations

from enum import IntEnum
from typing import Dict, Optional


class MethodTier(IntEnum):
    TIER1_REAL_COMPRESSION = 1
    TIER2_STRUCTURAL = 2
    TIER3_ENTROPY = 3
    TIER4_HYBRID = 4
    TIER5_QUANTIZATION = 5


CATEGORY_TIER_MAP: Dict[str, MethodTier] = {
    "decomposition": MethodTier.TIER1_REAL_COMPRESSION,
    "spectral": MethodTier.TIER1_REAL_COMPRESSION,
    "tensor_network": MethodTier.TIER1_REAL_COMPRESSION,
    "functional": MethodTier.TIER1_REAL_COMPRESSION,
    "functional_weight_space": MethodTier.TIER1_REAL_COMPRESSION,
    "novel": MethodTier.TIER1_REAL_COMPRESSION,
    "novel_algorithmic": MethodTier.TIER1_REAL_COMPRESSION,
    "revolutionary_gauge": MethodTier.TIER1_REAL_COMPRESSION,
    "revolutionary_topological": MethodTier.TIER1_REAL_COMPRESSION,
    "revolutionary": MethodTier.TIER1_REAL_COMPRESSION,
    "breakthrough_decomposition": MethodTier.TIER1_REAL_COMPRESSION,
    "breakthrough_signal": MethodTier.TIER1_REAL_COMPRESSION,
    "breakthrough_math": MethodTier.TIER1_REAL_COMPRESSION,
    "novel_signal": MethodTier.TIER1_REAL_COMPRESSION,
    "novel_info": MethodTier.TIER1_REAL_COMPRESSION,
    "novel_cross": MethodTier.TIER1_REAL_COMPRESSION,
    "novel_chaotic": MethodTier.TIER1_REAL_COMPRESSION,
    "tensor_quantum": MethodTier.TIER1_REAL_COMPRESSION,
    "quantum_compression": MethodTier.TIER1_REAL_COMPRESSION,
    "quantum_engine": MethodTier.TIER1_REAL_COMPRESSION,
    "fractal_holographic": MethodTier.TIER1_REAL_COMPRESSION,
    "information_theory_2": MethodTier.TIER1_REAL_COMPRESSION,
    "breakthrough_info": MethodTier.TIER1_REAL_COMPRESSION,
    "structural": MethodTier.TIER2_STRUCTURAL,
    "physics": MethodTier.TIER2_STRUCTURAL,
    "novel_structural": MethodTier.TIER2_STRUCTURAL,
    "novel_physics": MethodTier.TIER2_STRUCTURAL,
    "novel_chaos": MethodTier.TIER2_STRUCTURAL,
    "novel_topological": MethodTier.TIER2_STRUCTURAL,
    "novel_biological": MethodTier.TIER2_STRUCTURAL,
    "breakthrough_physics": MethodTier.TIER2_STRUCTURAL,
    "unified_physics_quantum2": MethodTier.TIER2_STRUCTURAL,
    "topological_biological": MethodTier.TIER2_STRUCTURAL,
    "geometric_topological_manifold": MethodTier.TIER2_STRUCTURAL,
    "topological_biological": MethodTier.TIER2_STRUCTURAL,
    "entropy": MethodTier.TIER3_ENTROPY,
    "lossless": MethodTier.TIER3_ENTROPY,
    "novel_entropy": MethodTier.TIER3_ENTROPY,
    "novel_fractal": MethodTier.TIER3_ENTROPY,
    "hybrid": MethodTier.TIER4_HYBRID,
    "cascade": MethodTier.TIER4_HYBRID,
    "breakthrough_hybrid": MethodTier.TIER4_HYBRID,
    "quantization": MethodTier.TIER5_QUANTIZATION,
    "transform_quant": MethodTier.TIER5_QUANTIZATION,
    "sparsity_quant": MethodTier.TIER5_QUANTIZATION,
    "delta_quant": MethodTier.TIER5_QUANTIZATION,
    "mixed": MethodTier.TIER5_QUANTIZATION,
}

MANUAL_TIER_OVERRIDES: Dict[str, MethodTier] = {
    "block_int8": MethodTier.TIER5_QUANTIZATION,
    "block_int4": MethodTier.TIER5_QUANTIZATION,
    "hadamard_int8": MethodTier.TIER5_QUANTIZATION,
    "hadamard_int4": MethodTier.TIER5_QUANTIZATION,
    "sparsity_int4": MethodTier.TIER5_QUANTIZATION,
    "delta_int4": MethodTier.TIER5_QUANTIZATION,
    "svd_compress": MethodTier.TIER1_REAL_COMPRESSION,
    "dct_spectral": MethodTier.TIER1_REAL_COMPRESSION,
    "tensor_train": MethodTier.TIER1_REAL_COMPRESSION,
    "fwht_compress": MethodTier.TIER1_REAL_COMPRESSION,
    # Toeplitz/Hankel: structural assumptions rarely hold on real weights
    # Lowered from Tier 1 (decomposition) to Tier 5 (last resort)
    "toeplitz": MethodTier.TIER5_QUANTIZATION,
    "hankel": MethodTier.TIER5_QUANTIZATION,
}

DEFAULT_TIER = MethodTier.TIER1_REAL_COMPRESSION


def get_method_tier(method_name: str, category: Optional[str] = None) -> MethodTier:
    if method_name in MANUAL_TIER_OVERRIDES:
        return MANUAL_TIER_OVERRIDES[method_name]
    if category and category in CATEGORY_TIER_MAP:
        return CATEGORY_TIER_MAP[category]
    return DEFAULT_TIER


def get_tier(method_name: str, category: str = "") -> MethodTier:
    if method_name in CATEGORY_TIER_MAP:
        return CATEGORY_TIER_MAP[method_name]
    effective_category = category if category else method_name
    return get_method_tier(method_name, effective_category)


def tier_score(tier: MethodTier) -> float:
    scores = {
        MethodTier.TIER1_REAL_COMPRESSION: 10.0,
        MethodTier.TIER2_STRUCTURAL: 5.0,
        MethodTier.TIER3_ENTROPY: 2.0,
        MethodTier.TIER4_HYBRID: 1.5,
        MethodTier.TIER5_QUANTIZATION: 0.3,
    }
    return scores.get(tier, 1.0)

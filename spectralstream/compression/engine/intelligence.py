"""
Compression Intelligence Engine — profiles + method evaluation + auto-select
=============================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MethodScore:
    name: str
    expected_ratio: float
    expected_error: float
    score: float
    suitable_categories: List[str] = field(default_factory=list)


CATEGORY_AFFINITY: Dict[str, List[str]] = {
    "low_rank": [
        "svd_lowrank",
        "tt_decomposition",
        "tr_decomposition",
        "cp_decomposition",
        "tucker_decomposition",
        "kronecker_approx",
    ],
    "sparse": ["structured_nm", "block_sparsity", "sparsegpt", "group_lasso"],
    "spectral": [
        "dct_spectral",
        "wavelet_threshold",
        "hadamard_transform",
        "butterfly_sparse",
        "random_projection",
    ],
    "quantization": [
        "lloyd_max",
        "product_quantization",
        "residual_vq",
        "additive_codebook",
        "e8_lattice",
        "mixed_precision",
        "hessian_aware",
        "nf4_quant",
    ],
    "entropy": ["huffman_coding", "rans_coding", "arithmetic_coding"],
    "cross_layer": ["delta_encoding", "basis_sharing"],
    "noise_aware": ["noise_floor", "bf16_exploit"],
    "physics": [
        "hamiltonian_dynamical",
        "topological_quant",
        "state_space_waveform",
        "vlasov_field",
        "quantum_state",
        "plasma_oscillation",
        "manifold_embedding",
        "optimal_transport",
        "resonance_modes",
    ],
}


class MethodEvaluator:
    def __init__(self):
        self._method_cache: Dict[str, Any] = {}

    def evaluate(
        self,
        tensor: np.ndarray,
        profile: "TensorProfile",
        target_ratio: float,
        all_methods: Optional[Dict[str, Any]] = None,
    ) -> List[MethodScore]:
        if all_methods is None:
            all_methods = {}
        scores = []
        for name in all_methods:
            try:
                est_ratio = 4.0  # default estimate
                if hasattr(all_methods[name], "estimate_ratio"):
                    est_ratio = all_methods[name].estimate_ratio(
                        tensor, target_ratio=target_ratio
                    )
                est_error = self._estimate_error(profile, name)
                ratio_score = max(
                    0, 1.0 - abs(est_ratio - target_ratio) / max(target_ratio, 0.01)
                )
                category_bonus = self._category_bonus(name, profile)
                score = (
                    0.4 * ratio_score + 0.3 * (1.0 - est_error) + 0.3 * category_bonus
                )
                scores.append(
                    MethodScore(
                        name=name,
                        expected_ratio=est_ratio,
                        expected_error=est_error,
                        score=score,
                    )
                )
            except Exception as e:
                logger.debug(f"Method {name} failed evaluation: {e}")
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores

    def _estimate_error(self, profile: "TensorProfile", method_name: str) -> float:
        base_error = 0.05
        if profile.sparsity > 0.5 and method_name in CATEGORY_AFFINITY.get(
            "sparse", []
        ):
            base_error *= 0.3
        if profile.spectral_entropy < 0.5 and method_name in CATEGORY_AFFINITY.get(
            "spectral", []
        ):
            base_error *= 0.4
        if method_name in CATEGORY_AFFINITY.get("low_rank", []):
            base_error *= 0.3
        return min(base_error, 0.95)

    def _category_bonus(self, method_name: str, profile: "TensorProfile") -> float:
        category = self._infer_category(profile)
        preferred = CATEGORY_AFFINITY.get(category, [])
        if method_name in preferred:
            return 1.0
        all_methods = []
        for methods in CATEGORY_AFFINITY.values():
            all_methods.extend(methods)
        if method_name in all_methods:
            return 0.5
        return 0.3

    @staticmethod
    def _infer_category(profile: "TensorProfile") -> str:
        if profile.sparsity > 0.5:
            return "sparse"
        if profile.energy_concentration > 0.7:
            return "spectral"
        if profile.effective_rank < 0.3 * max(int(np.sqrt(profile.n_elements)), 1):
            return "low_rank"
        return "quantization"


class CompressionIntelligence:
    def __init__(self):
        self.evaluator = MethodEvaluator()

    def auto_compress(
        self,
        tensor: np.ndarray,
        target_ratio: float = 0.25,
        all_methods: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if all_methods is None:
            all_methods = {}
        flat = tensor.ravel().astype(np.float64)
        profile = type(
            "TensorProfile",
            (),
            {
                "n_elements": flat.size,
                "sparsity": float(np.mean(np.abs(flat) < 1e-10)),
                "spectral_entropy": 0.5,
                "energy_concentration": 0.5,
                "effective_rank": 1.0,
                "shape": tensor.shape,
            },
        )()
        scores = self.evaluator.evaluate(tensor, profile, target_ratio, all_methods)
        if not scores:
            return {"error": "no_methods"}, {"method": "none"}
        best = scores[0]
        data, meta = {}, {"method": best.name, "score": best.score}
        return data, meta

    def auto_compress_compose(
        self,
        tensor: np.ndarray,
        target_ratio: float = 0.25,
        all_methods: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.auto_compress(tensor, target_ratio, all_methods, **kwargs)

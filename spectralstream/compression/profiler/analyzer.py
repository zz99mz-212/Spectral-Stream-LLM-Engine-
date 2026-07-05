from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct,
    spectral_entropy as _spectral_entropy,
)


LAYER_SENSITIVITY: Dict[str, float] = {
    "embed": 1.0,
    "tok_embeddings": 1.0,
    "wte": 1.0,
    "attn_q": 1.0,
    "q_proj": 1.0,
    "wq": 1.0,
    "query": 1.0,
    "attn_k": 0.92,
    "k_proj": 0.92,
    "wk": 0.92,
    "key": 0.92,
    "attn_v": 0.88,
    "v_proj": 0.88,
    "wv": 0.88,
    "value": 0.88,
    "attn_o": 1.0,
    "o_proj": 1.0,
    "wo": 1.0,
    "attn_norm": 0.7,
    "ln_1": 0.7,
    "layernorm": 0.7,
    "ffn_gate": 0.55,
    "gate_proj": 0.55,
    "w1": 0.55,
    "ffn_up": 0.60,
    "up_proj": 0.60,
    "w3": 0.60,
    "ffn_down": 0.65,
    "down_proj": 0.65,
    "w2": 0.65,
    "ffn_norm": 0.50,
    "ln_2": 0.50,
    "norm": 0.50,
    "final_norm": 0.50,
    "rms_norm": 0.50,
    "output": 1.0,
    "lm_head": 1.0,
    "head": 1.0,
}


def _get_sensitivity(name: str) -> float:
    name_lower = name.lower()
    for key, val in LAYER_SENSITIVITY.items():
        if key in name_lower:
            return val
    if "bias" in name_lower:
        return 0.95
    if "weight" in name_lower:
        return 0.7
    return 0.5


@dataclass
class TensorProfile:
    name: str = ""
    shape: Tuple[int, ...] = (0,)
    dtype: str = ""
    n_elements: int = 0
    nbytes: int = 0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    dynamic_range: float = 0.0
    outlier_ratio: float = 0.0
    kurtosis: float = 0.0
    skewness: float = 0.0
    effective_rank: float = 0.0
    spectral_decay_rate: float = 0.0
    energy_concentration: float = 0.0
    spectral_entropy: float = 0.0
    block_structure_score: float = 0.0
    toeplitz_score: float = 0.0
    circulant_score: float = 0.0
    optimal_bits: int = 8
    sensitivity: float = 0.5
    sensitivity_category: str = "unknown"
    tensor_type: str = "generic"
    recommended_method: str = "block_int8"
    recommended_bits: int = 8
    compression_difficulty: float = 0.5


@dataclass
class SensitivityHeatmap:
    tensor_names: List[str] = field(default_factory=list)
    sensitivities: List[float] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    tensor_types: List[str] = field(default_factory=list)
    sizes_bytes: List[int] = field(default_factory=list)


class SensitivityAnalyzer:
    def __init__(self, max_sample_elements: int = 100_000) -> None:
        self.max_sample = max_sample_elements

    def profile(self, tensor: np.ndarray, name: str = "") -> TensorProfile:
        tensor = np.asarray(tensor)
        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        p = TensorProfile(
            name=name,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            n_elements=n,
            nbytes=tensor.nbytes,
        )
        if n == 0:
            return p
        self._statistical_profile(p, flat)
        self._spectral_profile(p, flat, tensor)
        self._structural_profile(p, tensor)
        p.spectral_entropy = _spectral_entropy(flat[: min(n, 4096)]) if n >= 4 else 0.0
        p.sensitivity = _get_sensitivity(name) if name else 0.5
        p.sensitivity_category = (
            "HIGH"
            if p.sensitivity >= 0.8
            else ("MEDIUM" if p.sensitivity >= 0.5 else "LOW")
        )
        self._classify_tensor_type(p, p.name.lower())
        self._recommend_method(p)
        p.compression_difficulty = self._estimate_difficulty(p)
        return p

    def _statistical_profile(self, p: TensorProfile, flat: np.ndarray) -> None:
        n = flat.size
        sample_n = min(n, self.max_sample)
        fs = (
            flat
            if sample_n >= n
            else flat[np.random.choice(n, sample_n, replace=False)]
        )
        p.mean = float(np.mean(fs))
        p.std = float(np.std(fs))
        p.min_val = float(np.min(flat))
        p.max_val = float(np.max(flat))
        p.dynamic_range = p.max_val - p.min_val
        if p.std > 1e-10:
            centered = (fs - p.mean) / p.std
            p.kurtosis = float(np.mean(centered**4) - 3.0)
            p.skewness = float(np.mean(centered**3))
            p.outlier_ratio = float(np.mean(np.abs(centered) > 3.0))
        p.optimal_bits = 4 if p.std < 1e-10 else (6 if p.std < 0.001 else 8)

    def _spectral_profile(
        self, p: TensorProfile, flat: np.ndarray, tensor: np.ndarray
    ) -> None:
        if tensor.ndim >= 2 and all(s > 1 for s in tensor.shape[:2]):
            mat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)
            try:
                max_dim = min(mat.shape[0], mat.shape[1], 64)
                if max_dim >= 2:
                    sv = np.linalg.svd(
                        mat[:max_dim, : min(mat.shape[1], max_dim)], compute_uv=False
                    )
                    sv = sv / (sv[0] + 1e-10)
                    sn2 = sv / (np.sum(sv) + 1e-10)
                    nnz = sn2[sn2 > 1e-10]
                    p.effective_rank = (
                        float(np.exp(-np.sum(nnz * np.log(nnz))))
                        if len(nnz) > 0
                        else 1.0
                    )
                    if len(sv) > 2:
                        nf = min(20, len(sv))
                        positive = sv[:nf] > 1e-10
                        if np.any(positive):
                            lsv = np.log(sv[:nf][positive] + 1e-30)
                            A = np.vstack([np.arange(len(lsv)), np.ones(len(lsv))]).T
                            p.spectral_decay_rate = max(
                                0.0, -float(np.linalg.lstsq(A, lsv, rcond=None)[0][0])
                            )
            except np.linalg.LinAlgError:
                pass
        if len(flat) >= 4:
            try:
                s = min(len(flat), 2048)
                pw = dct(flat[:s]) ** 2
                total_power = float(np.sum(pw))
                if total_power > 1e-30:
                    sorted_pw = np.sort(pw)[::-1]
                    top_n = max(1, int(s * 0.1))
                    p.energy_concentration = (
                        float(np.sum(sorted_pw[:top_n])) / total_power
                    )
            except Exception:
                p.energy_concentration = 0.0

    def _structural_profile(self, p: TensorProfile, tensor: np.ndarray) -> None:
        if tensor.ndim >= 2 and min(tensor.shape) >= 4:
            sub = tensor[:64, :64].astype(np.float64)
            nd = min(sub.shape) - 1
            diag_stds = []
            for k in range(-nd, nd + 1):
                diag = np.diag(sub, k)
                if len(diag) > 1:
                    diag_std = float(np.std(diag))
                    diag_mean = float(np.mean(np.abs(diag)))
                    if diag_mean > 1e-10:
                        diag_stds.append(diag_std / diag_mean)
            if diag_stds:
                p.toeplitz_score = max(0.0, 1.0 - float(np.mean(diag_stds)))
            if min(sub.shape) >= 4:
                h = min(sub.shape)
                half = h // 2
                c = float(
                    np.corrcoef(
                        sub[:half, :half].ravel(),
                        sub[half : 2 * half, half : 2 * half].ravel(),
                    )[0, 1]
                )
                p.block_structure_score = min(c * 0.5 + 0.5, 1.0)
            p.circulant_score = 1.0 if tensor.shape[0] == tensor.shape[1] else 0.0

    def _classify_tensor_type(self, p: TensorProfile, nl: str) -> None:
        if p.shape[0] > 10000 and p.n_elements > 1_000_000:
            p.tensor_type = "embedding"
        elif any(k in nl for k in ("attn", "q_proj", "k_proj", "v_proj", "o_proj")):
            p.tensor_type = "attention"
        elif any(k in nl for k in ("ffn", "gate", "up_proj", "down_proj", "mlp")):
            p.tensor_type = "ffn"
        elif p.nbytes < 1024:
            p.tensor_type = "norm"
        else:
            p.tensor_type = "weight"

    def _recommend_method(self, p: TensorProfile) -> None:
        if p.nbytes < 1024 or p.tensor_type == "norm":
            p.recommended_method, p.recommended_bits = "passthrough", 16
        elif p.tensor_type == "embedding":
            p.recommended_method, p.recommended_bits = (
                ("hadamard_int8", 8) if p.outlier_ratio > 0.01 else ("block_int8", 8)
            )
        elif p.outlier_ratio > 0.3:
            p.recommended_method, p.recommended_bits = "sparsity_int4", 4
        elif p.effective_rank < 32 and p.energy_concentration > 0.8:
            p.recommended_method, p.recommended_bits = "hadamard_int8", 8
        elif p.energy_concentration > 0.7:
            p.recommended_method, p.recommended_bits = "block_int8", 8
        else:
            p.recommended_method, p.recommended_bits = "block_int8", 8

    @staticmethod
    def _estimate_difficulty(p: TensorProfile) -> float:
        factors = [
            1.0 - min(p.energy_concentration, 1.0),
            p.outlier_ratio,
            1.0 - min(p.toeplitz_score, 1.0),
            1.0 - p.sensitivity,
        ]
        return max(0.0, min(float(np.mean(factors)), 1.0))

    def sensitivity_analysis(
        self, tensor: np.ndarray, name: str = ""
    ) -> Dict[str, Any]:
        p = self.profile(tensor, name)
        return {
            "sensitivity": p.sensitivity,
            "category": p.sensitivity_category,
            "tensor_type": p.tensor_type,
            "difficulty": p.compression_difficulty,
        }

    def generate_heatmap(
        self, profiles: Dict[str, TensorProfile]
    ) -> SensitivityHeatmap:
        return SensitivityHeatmap(
            tensor_names=list(profiles.keys()),
            sensitivities=[p.sensitivity for p in profiles.values()],
            categories=[p.sensitivity_category for p in profiles.values()],
            tensor_types=[p.tensor_type for p in profiles.values()],
            sizes_bytes=[p.nbytes for p in profiles.values()],
        )

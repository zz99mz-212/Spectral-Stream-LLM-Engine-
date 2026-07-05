"""
Compression Intelligence Engine — Dynamic strategy orchestration
=================================================================
Tensor analysis → strategy scoring → bit-budget allocation →
Lagrangian RD optimization → adaptive feedback → orchestration.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    HadamardRotator,
    LloydMaxQuantizer,
    dct,
    fwht,
    idct,
    next_power_of_two,
    softmax,
    spectral_entropy,
)
from spectralstream.compression.engine._dataclasses import TensorProfile

logger = logging.getLogger(__name__)


class TensorCategory(IntEnum):
    LOW_RANK = 0
    SPARSE = 1
    SPECTRAL_COMPACT = 2
    UNIFORM = 3
    STRUCTURED = 4


@dataclass
class StrategyScore:
    name: str
    estimated_ratio: float
    estimated_error: float
    score: float
    params: Dict = field(default_factory=dict)


class TensorAnalyzer:
    def __init__(self, energy_threshold: float = 0.95):
        self.energy_threshold = energy_threshold

    def analyze(self, tensor: np.ndarray) -> TensorProfile:
        tensor = np.asarray(tensor, dtype=np.float64)
        flat = tensor.ravel()
        n_elements = tensor.size
        mean_mag = float(np.mean(np.abs(flat)))
        std_mag = float(np.std(flat))
        sparsity_ratio = float(np.mean(np.abs(flat) < 1e-10))
        ent = spectral_entropy(flat)
        rank_est, cond_num = self._estimate_rank(tensor)
        energy_conc = self._energy_concentration(tensor)
        best_basis = self._select_best_basis(tensor)
        coherence = self._compute_coherence(tensor)
        category = self._classify(
            rank_est, n_elements, sparsity_ratio, energy_conc, ent, std_mag
        )
        return TensorProfile(
            name="",
            shape=tensor.shape,
            n_elements=n_elements,
            nbytes=tensor.nbytes,
            mean=mean_mag,
            std=std_mag,
            min_val=float(np.min(flat)),
            max_val=float(np.max(flat)),
            recommended_bits=4,
            energy_concentration=energy_conc,
            spectral_entropy=ent,
        )

    def _estimate_rank(self, tensor: np.ndarray) -> Tuple[int, float]:
        if tensor.ndim < 2:
            return 1, 1.0
        m, n = tensor.shape[0], int(np.prod(tensor.shape[1:]))
        mat = tensor.reshape(m, n)
        try:
            sv = np.linalg.svd(mat, compute_uv=False)
            sv_max = float(sv[0]) if len(sv) > 0 else 1.0
            sv_min = float(sv[-1]) if len(sv) > 0 else 1e-10
            cond = sv_max / max(sv_min, 1e-10)
            cumulative = np.cumsum(sv**2) / (np.sum(sv**2) + 1e-30)
            rank = int(np.searchsorted(cumulative, self.energy_threshold)) + 1
            return min(rank, len(sv)), cond
        except np.linalg.LinAlgError:
            return 1, 1.0

    def _energy_concentration(self, tensor: np.ndarray) -> float:
        flat = tensor.ravel()
        coeffs = dct(flat)
        power = coeffs**2
        total = np.sum(power)
        if total < 1e-30:
            return 0.0
        n = len(power)
        n_keep = max(1, int(n * 0.1))
        top_power = np.sum(np.sort(power)[::-1][:n_keep])
        return float(top_power / total)

    def _select_best_basis(self, tensor: np.ndarray) -> str:
        flat = tensor.ravel()
        dct_coeffs = dct(flat)
        dct_energy = float(np.sum(dct_coeffs[: len(flat) // 10] ** 2))
        dct_total = float(np.sum(dct_coeffs**2) + 1e-30)
        pad_len = next_power_of_two(len(flat))
        padded = np.pad(flat, (0, pad_len - len(flat)))
        had_coeffs = fwht(padded)
        had_energy = float(np.sum(had_coeffs[: pad_len // 10] ** 2))
        had_total = float(np.sum(had_coeffs**2) + 1e-30)
        dct_ratio = dct_energy / dct_total
        had_ratio = had_energy / had_total
        return "dct" if dct_ratio > had_ratio else "hadamard"

    def _compute_coherence(self, tensor: np.ndarray) -> float:
        if tensor.ndim < 2 or tensor.shape[0] < 2:
            return 0.0
        m = min(tensor.shape[0], 128)
        mat = tensor[:m].reshape(m, -1)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        mat_normed = mat / norms
        gram = mat_normed @ mat_normed.T
        np.fill_diagonal(gram, 0.0)
        return float(np.max(np.abs(gram)))

    def _classify(
        self,
        rank_est: int,
        n_elements: int,
        sparsity_ratio: float,
        energy_conc: float,
        entropy: float,
        std_mag: float,
    ) -> TensorCategory:
        rank_ratio = rank_est / max(int(np.sqrt(n_elements)), 1)
        if sparsity_ratio > 0.7:
            return TensorCategory.SPARSE
        elif rank_ratio < 0.1:
            return TensorCategory.LOW_RANK
        elif energy_conc > 0.8:
            return TensorCategory.SPECTRAL_COMPACT
        elif std_mag < 0.01:
            return TensorCategory.UNIFORM
        else:
            return TensorCategory.STRUCTURED


class CompressionStrategySelector:
    def __init__(self):
        self._strategies: List[Callable] = [
            self._score_scalar_quant_4bit,
            self._score_scalar_quant_6bit,
            self._score_scalar_quant_8bit,
            self._score_dct_lowpass,
            self._score_dct_zigzag,
            self._score_hadamard_quant,
            self._score_low_rank_svd,
            self._score_tt_decomposition,
            self._score_block_sparsity,
            self._score_nm_sparsity,
            self._score_structured_pruning,
            self._score_wavelet_threshold,
            self._score_sketch_count,
            self._score_nystrom_approx,
            self._score_random_fourier,
            self._score_butterfly_transform,
            self._score_tensor_ring,
            self._score_cp_decomposition,
            self._score_huffman_delta,
            self._score_rle_sparse,
            self._score_csc_sparse,
            self._score_mixed_precision,
        ]

    def _score_scalar_quant_4bit(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        ratio = 32.0 / 4.0
        error = profile.std * 0.1
        return StrategyScore("scalar_quant_4bit", ratio, error, ratio / (error + 1e-6))

    def _score_scalar_quant_6bit(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        ratio = 32.0 / 6.0
        error = profile.std * 0.05
        return StrategyScore("scalar_quant_6bit", ratio, error, ratio / (error + 1e-6))

    def _score_scalar_quant_8bit(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        ratio = 32.0 / 8.0
        error = profile.std * 0.02
        return StrategyScore("scalar_quant_8bit", ratio, error, ratio / (error + 1e-6))

    def _score_dct_lowpass(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        keep = max(1, int(tensor.size * 0.2))
        ratio = tensor.size / max(keep, 1)
        error = profile.std * (1.0 - profile.energy_concentration) * 0.3
        return StrategyScore(
            "dct_lowpass", ratio, error, ratio / (error + 1e-6), {"keep_fraction": 0.2}
        )

    def _score_dct_zigzag(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        ratio = 32.0 / max(4.0, 32.0 * (1.0 - profile.energy_concentration))
        error = profile.std * 0.15
        return StrategyScore("dct_zigzag", ratio, error, ratio / (error + 1e-6))

    def _score_hadamard_quant(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        ratio = 32.0 / 6.0
        error = profile.std * 0.08
        bonus = 1.2 if getattr(profile, "best_basis", "") == "hadamard" else 0.8
        score = ratio / (error + 1e-6) * bonus
        return StrategyScore("hadamard_quant", ratio, error, score)

    def _score_low_rank_svd(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        rank = getattr(profile, "rank_estimate", 1)
        m, n = tensor.shape[0], int(np.prod(tensor.shape[1:]))
        ratio = (m * n) / max(rank * (m + n), 1)
        error = profile.std * (1.0 - profile.energy_concentration) * 0.5
        return StrategyScore(
            "low_rank_svd", ratio, error, ratio / (error + 1e-6), {"rank": rank}
        )

    def _score_tt_decomposition(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        rank = min(getattr(profile, "rank_estimate", 1), 32)
        n_modes = min(tensor.ndim, 4)
        tt_size = sum(
            int(np.prod(tensor.shape[: i + 1])) * rank * rank
            for i in range(n_modes - 1)
        )
        ratio = tensor.size / max(tt_size, 1)
        error = profile.std * 0.2
        return StrategyScore(
            "tt_decomposition", ratio, error, ratio / (error + 1e-6), {"rank": rank}
        )

    def _score_block_sparsity(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        target_sparsity = 0.8
        ratio = 1.0 / max(1.0 - target_sparsity, 0.01)
        error = profile.std * 0.15
        return StrategyScore(
            "block_sparsity",
            ratio,
            error,
            ratio / (error + 1e-6),
            {"sparsity": target_sparsity},
        )

    def _score_nm_sparsity(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        ratio = 4.0 / 2.0
        error = profile.std * 0.12
        return StrategyScore(
            "nm_sparsity", ratio, error, ratio / (error + 1e-6), {"n": 2, "m": 4}
        )

    def _score_structured_pruning(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        ratio = 1.0 / max(1.0 - getattr(profile, "sparsity_ratio", 0), 0.01)
        error = profile.std * 0.1
        return StrategyScore("structured_pruning", ratio, error, ratio / (error + 1e-6))

    def _score_wavelet_threshold(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        keep = max(1, int(tensor.size * 0.15))
        ratio = tensor.size / max(keep, 1)
        error = profile.std * 0.25
        return StrategyScore("wavelet_threshold", ratio, error, ratio / (error + 1e-6))

    def _score_sketch_count(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        sketches = max(1, int(np.sqrt(tensor.size) * 0.3))
        ratio = tensor.size / max(sketches, 1)
        error = profile.std * 0.3
        return StrategyScore("sketch_count", ratio, error, ratio / (error + 1e-6))

    def _score_nystrom_approx(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        n_landmarks = max(1, int(np.sqrt(tensor.size) * 0.1))
        ratio = tensor.size / max(n_landmarks * tensor.shape[-1] * 2, 1)
        error = profile.std * 0.2
        return StrategyScore("nystrom_approx", ratio, error, ratio / (error + 1e-6))

    def _score_random_fourier(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        n_features = max(1, int(tensor.size * 0.05))
        ratio = tensor.size / max(n_features, 1)
        error = profile.std * 0.35
        return StrategyScore("random_fourier", ratio, error, ratio / (error + 1e-6))

    def _score_butterfly_transform(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        ratio = 8.0
        error = profile.std * 0.15
        return StrategyScore(
            "butterfly_transform", ratio, error, ratio / (error + 1e-6)
        )

    def _score_tensor_ring(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        rank = min(getattr(profile, "rank_estimate", 1), 16)
        ring_size = rank * rank * tensor.ndim * tensor.shape[0]
        ratio = tensor.size / max(ring_size, 1)
        error = profile.std * 0.18
        return StrategyScore(
            "tensor_ring", ratio, error, ratio / (error + 1e-6), {"rank": rank}
        )

    def _score_cp_decomposition(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        rank = min(getattr(profile, "rank_estimate", 1), 32)
        cp_size = rank * sum(tensor.shape)
        ratio = tensor.size / max(cp_size, 1)
        error = profile.std * 0.22
        return StrategyScore(
            "cp_decomposition", ratio, error, ratio / (error + 1e-6), {"rank": rank}
        )

    def _score_huffman_delta(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        ratio = 32.0 / max(8.0, 32.0 * 0.5)
        error = 0.0
        return StrategyScore("huffman_delta", ratio, error, ratio)

    def _score_rle_sparse(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        sparsity = getattr(profile, "sparsity_ratio", 0)
        if sparsity < 0.3:
            return StrategyScore("rle_sparse", 1.0, 0.0, 0.0)
        ratio = 1.0 / max(1.0 - sparsity, 0.01) * 0.8
        return StrategyScore("rle_sparse", ratio, 0.0, ratio)

    def _score_csc_sparse(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        sparsity = getattr(profile, "sparsity_ratio", 0)
        if sparsity < 0.3:
            return StrategyScore("csc_sparse", 1.0, 0.0, 0.0)
        ratio = 1.0 / max(1.0 - sparsity, 0.01) * 0.85
        return StrategyScore("csc_sparse", ratio, 0.0, ratio)

    def _score_mixed_precision(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> StrategyScore:
        ratio = 32.0 / 6.0
        error = profile.std * 0.06
        return StrategyScore("mixed_precision", ratio, error, ratio / (error + 1e-6))

    def evaluate(
        self, tensor: np.ndarray, profile: TensorProfile
    ) -> List[StrategyScore]:
        scores = []
        for strategy_fn in self._strategies:
            try:
                score = strategy_fn(tensor, profile)
                scores.append(score)
            except Exception as e:
                logger.warning("Strategy %s failed: %s", strategy_fn.__name__, e)
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores


@dataclass
class BitBudget:
    total_bits: int
    allocations: Dict[str, int]
    per_tensor_bits: Dict[str, int]
    estimated_total_error: float


class BitBudgetOptimizer:
    def __init__(self, total_bits: int = 1_000_000):
        self.total_bits = total_bits

    def _distortion_model(self, bits: float, tensor_size: int, std: float) -> float:
        if bits <= 0:
            return float(std**2 * tensor_size)
        bits_per_elem = bits / max(tensor_size, 1)
        return float(std**2 * tensor_size * np.exp(-2.0 * bits_per_elem))

    def optimize(
        self, tensor_sizes: Dict[str, int], tensor_stds: Dict[str, float]
    ) -> BitBudget:
        names = list(tensor_sizes.keys())
        if not names:
            return BitBudget(self.total_bits, {}, {}, 0.0)
        lambda_lo = 0.0
        lambda_hi = 1e10
        for _ in range(50):
            lambda_mid = (lambda_lo + lambda_hi) / 2.0
            total_alloc = 0
            for name in names:
                size = tensor_sizes[name]
                std = tensor_stds.get(name, 0.1)
                b_opt = (
                    size
                    / 2.0
                    * np.log(
                        max(2.0 * std**2 * size / max(lambda_mid * size, 1e-30), 1.0)
                    )
                )
                b_opt = max(0, min(b_opt, size * 32))
                total_alloc += int(b_opt)
            if total_alloc > self.total_bits:
                lambda_lo = lambda_mid
            else:
                lambda_hi = lambda_mid
        allocations = {}
        per_tensor_bits = {}
        total_error = 0.0
        for name in names:
            size = tensor_sizes[name]
            std = tensor_stds.get(name, 0.1)
            b_opt = (
                size
                / 2.0
                * np.log(max(2.0 * std**2 * size / max(lambda_hi * size, 1e-30), 1.0))
            )
            b_opt = max(0, min(b_opt, size * 32))
            b_int = int(b_opt)
            allocations[name] = b_int
            per_tensor_bits[name] = b_int
            total_error += self._distortion_model(b_int, size, std)
        return BitBudget(self.total_bits, allocations, per_tensor_bits, total_error)


@dataclass
class RateDistortionPoint:
    rate: float
    distortion: float
    lambda_val: float
    params: Dict = field(default_factory=dict)


class LagrangianRateDistortion:
    def __init__(self, n_lambda_points: int = 50):
        self.n_lambda_points = n_lambda_points

    def compute_rd_curve(
        self, tensor: np.ndarray, candidate_methods: Optional[List[str]] = None
    ) -> List[RateDistortionPoint]:
        tensor = np.asarray(tensor, dtype=np.float64)
        std = float(np.std(tensor))
        n = tensor.size
        points = []
        for i in range(self.n_lambda_points):
            lambda_val = 10 ** (-6 + 8 * i / max(self.n_lambda_points - 1, 1))
            rate = (
                n
                / 2.0
                * np.log(max(2.0 * std**2 * n / max(lambda_val * n, 1e-30), 1.0))
            )
            rate = max(0, min(rate, n * 32))
            distortion = std**2 * n * np.exp(-2.0 * rate / max(n, 1))
            points.append(
                RateDistortionPoint(
                    rate=rate / n, distortion=distortion / n, lambda_val=lambda_val
                )
            )
        points.sort(key=lambda p: p.rate)
        return points

    def find_optimal_lambda(self, tensor: np.ndarray, target_rate: float) -> float:
        rd_curve = self.compute_rd_curve(tensor)
        if not rd_curve:
            return 1.0
        best = min(rd_curve, key=lambda p: abs(p.rate - target_rate))
        return best.lambda_val


@dataclass
class AdaptationState:
    current_method: str
    current_error: float
    error_history: List[float]
    method_history: List[str]
    adaptation_count: int


class AdaptiveMethodSelector:
    def __init__(self, error_threshold: float = 0.05, switch_patience: int = 3):
        self.error_threshold = error_threshold
        self.switch_patience = switch_patience
        self._states: Dict[str, AdaptationState] = {}
        self._selector = CompressionStrategySelector()

    def register_tensor(self, name: str, initial_method: str = "scalar_quant_8bit"):
        self._states[name] = AdaptationState(
            current_method=initial_method,
            current_error=0.0,
            error_history=[],
            method_history=[initial_method],
            adaptation_count=0,
        )

    def record_error(self, name: str, error: float) -> bool:
        if name not in self._states:
            self.register_tensor(name)
        state = self._states[name]
        state.current_error = error
        state.error_history.append(error)
        if len(state.error_history) > 100:
            state.error_history = state.error_history[-100:]
        if error > self.error_threshold:
            state.adaptation_count += 1
            if state.adaptation_count >= self.switch_patience:
                return self._switch_method(name)
        else:
            state.adaptation_count = 0
        return False

    def _switch_method(self, name: str) -> bool:
        state = self._states[name]
        methods = [
            "scalar_quant_4bit",
            "scalar_quant_6bit",
            "scalar_quant_8bit",
            "dct_lowpass",
            "hadamard_quant",
            "low_rank_svd",
            "block_sparsity",
            "mixed_precision",
        ]
        current_idx = 0
        for i, m in enumerate(methods):
            if m == state.current_method:
                current_idx = i
                break
        new_method = methods[(current_idx + 1) % len(methods)]
        state.current_method = new_method
        state.method_history.append(new_method)
        state.adaptation_count = 0
        return True

    def get_current_method(self, name: str) -> str:
        if name not in self._states:
            return "scalar_quant_8bit"
        return self._states[name].current_method

    def get_stats(self, name: str) -> Dict:
        if name not in self._states:
            return {}
        state = self._states[name]
        return {
            "current_method": state.current_method,
            "current_error": state.current_error,
            "mean_error": float(np.mean(state.error_history))
            if state.error_history
            else 0.0,
            "n_switches": len(set(state.method_history)) - 1,
            "total_adaptations": len(state.method_history),
        }


@dataclass
class CompressionPlan:
    tensor_plans: Dict[str, StrategyScore]
    bit_budget: Optional[BitBudget]
    rd_curve: Optional[List[RateDistortionPoint]]
    total_estimated_ratio: float
    total_estimated_error: float


class CompressionOrchestrator:
    def __init__(
        self, total_bit_budget: Optional[int] = None, error_threshold: float = 0.05
    ):
        self.analyzer = TensorAnalyzer()
        self.strategy_selector = CompressionStrategySelector()
        self.budget_optimizer = BitBudgetOptimizer(total_bit_budget or 10_000_000)
        self.rd_optimizer = LagrangianRateDistortion()
        self.adaptive_selector = AdaptiveMethodSelector(error_threshold=error_threshold)
        self._profiles: Dict[str, TensorProfile] = {}
        self._plans: Dict[str, StrategyScore] = {}

    def analyze_tensor(self, name: str, tensor: np.ndarray) -> TensorProfile:
        profile = self.analyzer.analyze(tensor)
        self._profiles[name] = profile
        return profile

    def select_strategy(self, name: str, tensor: np.ndarray) -> StrategyScore:
        if name not in self._profiles:
            self.analyze_tensor(name, tensor)
        profile = self._profiles[name]
        scores = self.strategy_selector.evaluate(tensor, profile)
        if scores:
            best = scores[0]
            self._plans[name] = best
            self.adaptive_selector.register_tensor(name, best.name)
            return best
        return StrategyScore("none", 1.0, 0.0, 0.0)

    def optimize_bit_budget(self, tensors: Dict[str, np.ndarray]) -> BitBudget:
        sizes = {name: t.size for name, t in tensors.items()}
        stds = {name: float(np.std(t)) for name, t in tensors.items()}
        return self.budget_optimizer.optimize(sizes, stds)

    def compute_rd_curve(self, tensor: np.ndarray) -> List[RateDistortionPoint]:
        return self.rd_optimizer.compute_rd_curve(tensor)

    def create_compression_plan(
        self, tensors: Dict[str, np.ndarray], total_bits: Optional[int] = None
    ) -> CompressionPlan:
        tensor_plans = {
            name: self.select_strategy(name, t) for name, t in tensors.items()
        }
        bit_budget = None
        if total_bits is not None or self.budget_optimizer.total_bits > 0:
            budget_total = total_bits or self.budget_optimizer.total_bits
            optimizer = BitBudgetOptimizer(budget_total)
            sizes = {n: t.size for n, t in tensors.items()}
            stds = {n: float(np.std(t)) for n, t in tensors.items()}
            bit_budget = optimizer.optimize(sizes, stds)
        total_ratio = (
            float(np.mean([p.estimated_ratio for p in tensor_plans.values()]))
            if tensor_plans
            else 1.0
        )
        total_error = (
            float(np.mean([p.estimated_error for p in tensor_plans.values()]))
            if tensor_plans
            else 0.0
        )
        return CompressionPlan(
            tensor_plans=tensor_plans,
            bit_budget=bit_budget,
            rd_curve=None,
            total_estimated_ratio=total_ratio,
            total_estimated_error=total_error,
        )

    def record_feedback(self, name: str, error: float) -> bool:
        return self.adaptive_selector.record_error(name, error)

    def get_tensor_plan(self, name: str) -> Optional[StrategyScore]:
        return self._plans.get(name)

    def get_global_stats(self) -> Dict:
        return {
            "n_tensors": len(self._plans),
            "mean_ratio": float(
                np.mean([p.estimated_ratio for p in self._plans.values()])
            )
            if self._plans
            else 0.0,
            "mean_error": float(
                np.mean([p.estimated_error for p in self._plans.values()])
            )
            if self._plans
            else 0.0,
        }

"""
Neural Architecture Search for Compression Stacking Patterns.

Discovers Pareto-optimal sequences of compression methods per tensor type
using evolutionary search, synergy tracking, and meta-learning.

Design philosophy (F1 aerodynamicist analogy):
  Each compression method is an aerodynamic device (diffuser, wing, DRS).
  Stacking them in the right order unlocks multiplicative gains, but the
  wrong order creates turbulence (error cascade).  We run thousands of
  virtual tunnel combinations to find the perfect setup per tensor type.
"""

from __future__ import annotations

import copy
import gc
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.engine._helpers import (
    _compute_metrics,
    _compute_ratio,
    _estimate_entropy_rate,
    _estimate_noise_floor,
)
from spectralstream.compression.engine.method_discovery import MethodDiscovery
from spectralstream.compression.engine._tier_common import (
    get_method_tier,
    MethodTier,
    tier_score,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_POPULATION_SIZE = 50
_GENERATIONS = 20
_TOURNAMENT_SIZE = 5
_CROSSOVER_RATE = 0.7
_MUTATION_RATE = 0.3
_PARETO_EPSILON = 1e-9
_MAX_STAGES = 4
_MIN_STAGES = 1
_CACHE_TTL = 3600  # seconds

# Tier 1-2 methods suitable as "primary" (first-stage) compressors
_PRIMARY_METHODS = [
    "svd_compress",
    "dct_spectral",
    "tensor_train",
    "fwht_compress",
    "butterfly",
    "monarch",
    "cp_decomposition",
    "kronecker",
    "nystrom",
    "random_feature",
    "h_matrix",
    "cur_decomposition",
    "einsort_tt",
    "lotr",
    "svd_truncated",
    "tensor_network",
    "hierarchical_mps",
    "decomp_tensor_train",
    "tensor_ring",
]

# Tier 5 quantization methods suitable as "secondary" (precision-reduction) stage
_SECONDARY_METHODS = [
    "block_int8",
    "block_int4",
    "hadamard_int8",
    "hadamard_int4",
    "sparsity_int4",
    "delta_int4",
]

# Tier 3 entropy/lossless methods for final stage
_FINAL_METHODS = [
    "arithmetic_coding",
    "ans",
    "huffman",
    "range_coding",
    "zstd",
    "rans",
    "lz4",
]

# ── Data Structures ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TensorSignature:
    """Lightweight fingerprint of a tensor for similarity matching."""

    log2_elements: float = 0.0
    aspect_ratio: float = 1.0
    rank_estimate: float = 0.0
    entropy_rate: float = 0.0
    noise_floor: float = 0.0
    sparsity: float = 0.0
    spectral_decay: float = 0.0
    dynamic_range: float = 0.0
    outlier_ratio: float = 0.0
    energy_concentration: float = 0.0
    tensor_type: str = "weight"

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.log2_elements,
                min(self.aspect_ratio, 100.0),
                self.rank_estimate,
                self.entropy_rate,
                self.noise_floor,
                self.sparsity,
                self.spectral_decay,
                min(self.dynamic_range, 100.0),
                self.outlier_ratio,
                self.energy_concentration,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_profile(cls, profile: Any) -> TensorSignature:
        shape = getattr(profile, "shape", (1,))
        n_elements = getattr(profile, "n_elements", max(1, np.prod(shape)))
        aspect = 1.0
        if len(shape) >= 2 and shape[1] > 0:
            aspect = shape[0] / shape[1]
        return cls(
            log2_elements=math.log2(max(n_elements, 1)),
            aspect_ratio=float(aspect),
            rank_estimate=float(getattr(profile, "effective_rank", 0.0)),
            entropy_rate=float(getattr(profile, "entropy_rate", 0.0)),
            noise_floor=float(getattr(profile, "noise_floor", 0.0)),
            sparsity=float(
                getattr(profile, "unstructured_sparsity_score", 0.0)
                + getattr(profile, "nm_sparsity_score", 0.0)
            )
            / 2.0,
            spectral_decay=float(getattr(profile, "spectral_decay_rate", 0.0)),
            dynamic_range=float(getattr(profile, "dynamic_range", 0.0)),
            outlier_ratio=float(getattr(profile, "outlier_ratio", 0.0)),
            energy_concentration=float(getattr(profile, "energy_concentration", 0.0)),
            tensor_type=str(getattr(profile, "tensor_type", "weight")),
        )


@dataclass
class StackingPattern:
    """Ordered sequence of compression stages."""

    stages: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.stages, list):
            self.stages = [(s[0], dict(s[1])) for s in self.stages]

    def copy(self) -> StackingPattern:
        return StackingPattern(stages=[(s[0], dict(s[1])) for s in self.stages])

    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, idx: int) -> Tuple[str, Dict[str, Any]]:
        return self.stages[idx]

    def __iter__(self):
        return iter(self.stages)

    def to_dict(self) -> Dict[str, Any]:
        return {"stages": [(name, dict(params)) for name, params in self.stages]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> StackingPattern:
        return StackingPattern(stages=[(s[0], dict(s[1])) for s in d.get("stages", [])])


@dataclass
class PatternScore:
    """Normalized scores for a stacking pattern."""

    ratio: float = 0.0
    error: float = 1.0
    speed: float = 0.0
    synergy_score: float = 0.0

    def dominates(self, other: PatternScore) -> bool:
        """Pareto dominance: better in at least one, no worse in any."""
        return (
            self.ratio > other.ratio + _PARETO_EPSILON
            or self.error < other.error - _PARETO_EPSILON
            or self.speed > other.speed + _PARETO_EPSILON
            or self.synergy_score > other.synergy_score + _PARETO_EPSILON
        ) and not (
            self.ratio < other.ratio - _PARETO_EPSILON
            or self.error > other.error + _PARETO_EPSILON
            or self.speed < other.speed - _PARETO_EPSILON
            or self.synergy_score < other.synergy_score - _PARETO_EPSILON
        )

    def fitness(self, target_ratio: float) -> float:
        """Primary fitness: ratio / (1 + error * 100).

        Maximizes ratio while heavily penalizing error.
        Also incorporates speed and synergy as secondary objectives.
        """
        primary = self.ratio / max(1.0 + self.error * 100.0, 1e-8)
        speed_bonus = math.tanh(self.speed / 100.0) * 0.1
        synergy_bonus = math.tanh(self.synergy_score * 2.0) * 0.1
        return primary + speed_bonus + synergy_bonus


# ── Meta-Learning Cache ────────────────────────────────────────────────────


class MetaLearningCache:
    """Caches best stacking patterns for tensor signatures.

    Stores (signature_vector → (pattern, score, timestamp)) entries.
    Retrieves warm-start patterns via cosine similarity.
    """

    def __init__(self, max_entries: int = 512, similarity_threshold: float = 0.85):
        self._entries: List[
            Tuple[np.ndarray, StackingPattern, PatternScore, float]
        ] = []
        self._max_entries = max_entries
        self._similarity_threshold = similarity_threshold

    def lookup(
        self, sig: TensorSignature
    ) -> Optional[Tuple[StackingPattern, PatternScore]]:
        """Return cached pattern+score if a similar signature exists."""
        vec = sig.to_vector()
        best_sim = 0.0
        best_idx = -1
        for i, (cached_vec, pattern, score, _ts) in enumerate(self._entries):
            sim = self._cosine_sim(vec, cached_vec)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_sim >= self._similarity_threshold and best_idx >= 0:
            _, pattern, score, _ = self._entries[best_idx]
            logger.debug("Meta-cache hit: similarity=%.3f", best_sim)
            return pattern.copy(), PatternScore(**vars(score))
        return None

    def store(
        self, sig: TensorSignature, pattern: StackingPattern, score: PatternScore
    ) -> None:
        """Store a pattern+score under the given signature."""
        if len(self._entries) >= self._max_entries:
            self._entries.pop(0)
        vec = sig.to_vector()
        self._entries.append(
            (vec, pattern.copy(), PatternScore(**vars(score)), time.monotonic())
        )

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an < 1e-12 or bn < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    def clear(self) -> None:
        self._entries.clear()


# ── Synergy Matrix ─────────────────────────────────────────────────────────


class SynergyMatrix:
    """N×N matrix tracking pairwise method synergy.

    Positive score → methods work well together (cascading gain).
    Negative score → methods interfere.
    """

    def __init__(self, n_methods: int = 128):
        self._n = n_methods
        self._matrix: Dict[Tuple[int, int], List[float]] = {}
        self._method_names: Dict[str, int] = {}
        self._next_id = 0

    def _get_id(self, name: str) -> int:
        if name not in self._method_names:
            self._method_names[name] = self._next_id
            self._next_id += 1
        return self._method_names[name]

    def update(self, pair: Tuple[str, str], score_delta: float) -> None:
        """Record synergy observation between two methods."""
        i, j = self._get_id(pair[0]), self._get_id(pair[1])
        key = (i, j) if i <= j else (j, i)
        if key not in self._matrix:
            self._matrix[key] = []
        self._matrix[key].append(score_delta)
        if len(self._matrix[key]) > 32:
            self._matrix[key] = self._matrix[key][-32:]

    def get_synergy(self, a: str, b: str) -> float:
        """Return mean synergy score between two methods."""
        i, j = self._get_id(a), self._get_id(b)
        key = (i, j) if i <= j else (j, i)
        scores = self._matrix.get(key)
        if not scores:
            return 0.0
        return float(np.mean(scores))

    def synergy_for_pattern(self, stages: List[Tuple[str, Dict[str, Any]]]) -> float:
        """Compute aggregate synergy score for a full pattern."""
        if len(stages) < 2:
            return 0.0
        names = [s[0] for s in stages]
        scores = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                scores.append(self.get_synergy(names[i], names[j]))
        return float(np.mean(scores)) if scores else 0.0


# ── Default Templates ──────────────────────────────────────────────────────


_TENSOR_TYPE_TEMPLATES: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
    "attention": [
        ("svd_compress", {"rank_ratio": 0.25}),
        ("hadamard_int8", {"block_size": 128}),
    ],
    "attention_q": [
        ("svd_compress", {"rank_ratio": 0.2}),
        ("block_int8", {"block_size": 128}),
    ],
    "attention_k": [
        ("svd_compress", {"rank_ratio": 0.15}),
        ("block_int8", {"block_size": 128}),
    ],
    "attention_v": [
        ("svd_compress", {"rank_ratio": 0.3}),
        ("delta_int4", {"block_size": 64}),
    ],
    "attention_o": [
        ("svd_compress", {"rank_ratio": 0.25}),
        ("hadamard_int8", {"block_size": 128}),
    ],
    "ffn": [
        ("dct_spectral", {"coeff_ratio": 0.3}),
        ("sparsity_int4", {"block_size": 64}),
    ],
    "ffn_gate": [
        ("dct_spectral", {"coeff_ratio": 0.35}),
        ("block_int4", {"block_size": 64}),
    ],
    "ffn_up": [
        ("dct_spectral", {"coeff_ratio": 0.4}),
        ("block_int4", {"block_size": 64}),
    ],
    "ffn_down": [
        ("fwht_compress", {"coeff_ratio": 0.3}),
        ("hadamard_int4", {"block_size": 64}),
    ],
    "embedding": [
        ("tensor_train", {"rank": 32}),
        ("block_int8", {"block_size": 128}),
    ],
    "output": [
        ("svd_compress", {"rank_ratio": 0.15}),
        ("block_int8", {"block_size": 128}),
    ],
    "norm": [
        ("block_int8", {"block_size": 256}),
    ],
    "norm_bias": [
        ("block_int8", {"block_size": 256}),
    ],
    "qkv_fused": [
        ("svd_compress", {"rank_ratio": 0.2}),
        ("hadamard_int8", {"block_size": 128}),
        ("zstd", {}),
    ],
    "weight": [
        ("dct_spectral", {"coeff_ratio": 0.4}),
        ("block_int8", {"block_size": 128}),
    ],
}

_METHOD_POOL: List[str] = _PRIMARY_METHODS + _SECONDARY_METHODS + _FINAL_METHODS


# ── NAS Optimizer ──────────────────────────────────────────────────────────


class NASCompressionOptimizer:
    """Neural Architecture Search for optimal compression stacking patterns.

    Uses evolutionary search (genetic algorithm) with:
      - Tournament selection
      - Simulated binary crossover (pattern-level)
      - Three mutation operators: replace/insert/delete stage
      - Pareto dominance ranking
      - Synergy matrix for pairwise method compatibility
      - Meta-learning cache for warm-start from similar tensors
    """

    def __init__(
        self,
        engine: Any,
        population_size: int = _POPULATION_SIZE,
        generations: int = _GENERATIONS,
        max_stages: int = _MAX_STAGES,
        cache_max: int = 512,
        random_state: int = 42,
    ):
        self._engine = engine
        self._population_size = population_size
        self._generations = generations
        self._max_stages = max_stages
        self._rng = random.Random(random_state)
        self._np_rng = np.random.RandomState(random_state)

        self.synergy = SynergyMatrix()
        self.cache = MetaLearningCache(max_entries=cache_max)

        self._n_total_evaluations = 0
        self._n_cache_hits = 0

        # Build available method lookup from the engine
        self._available_methods: Dict[str, Any] = {}
        try:
            self._available_methods = engine.get_available_methods()
        except Exception:
            try:
                discovered = MethodDiscovery.discover()
                self._available_methods = {
                    name: info.get("instance") or info.get("class")
                    for name, info in discovered.items()
                }
            except Exception:
                pass

        # Cache method tiers for fast lookup
        self._method_tiers: Dict[str, MethodTier] = {}
        for mname in self._available_methods:
            try:
                self._method_tiers[mname] = get_method_tier(mname)
            except Exception:
                self._method_tiers[mname] = MethodTier.TIER5_QUANTIZATION

    # ── Public API ─────────────────────────────────────────────────────────

    def recommend(
        self,
        profile: Any,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        max_search_time: float = 5.0,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Recommend an optimal stacking pattern for the given tensor profile.

        Returns a dict with keys:
          stages          — list of (method_name, params)
          expected_ratio  — predicted compression ratio
          expected_error  — predicted relative error
          synergy_score   — pairwise method synergy
          search_time     — wall-clock time spent searching
          cache_hit       — whether the result came from cache
        """
        t0 = time.monotonic()
        sig = TensorSignature.from_profile(profile)

        # 1. Check meta-learning cache
        cached = self.cache.lookup(sig) if use_cache else None
        if cached is not None:
            self._n_cache_hits += 1
            cached_pattern, cached_score = cached
            return self._build_recommendation(
                cached_pattern,
                cached_score,
                time.monotonic() - t0,
                cache_hit=True,
            )

        # 2. Determine tensor type for template seeding
        tensor_type = self._resolve_tensor_type(profile)

        # 3. Run evolutionary search
        best_pattern, best_score = self._evolve(
            tensor_type=tensor_type,
            target_ratio=target_ratio,
            max_error=max_error,
            max_time=max_search_time - (time.monotonic() - t0),
            profile=profile,
        )

        # 4. Cache the result
        self.cache.store(sig, best_pattern, best_score)

        elapsed = time.monotonic() - t0
        return self._build_recommendation(
            best_pattern,
            best_score,
            elapsed,
            cache_hit=False,
        )

    def search(
        self,
        tensor: np.ndarray,
        profile: Optional[Any] = None,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        max_search_time: float = 10.0,
    ) -> Dict[str, Any]:
        """Full search: profile → evolve → validate on the actual tensor.

        This is more expensive than recommend() because it runs actual
        compression evaluations during evolution.
        """
        if profile is None:
            from spectralstream.compression.engine._profiler import CompressionProfiler

            profiler = CompressionProfiler()
            profile = profiler.profile_tensor(tensor, name="search")

        t0 = time.monotonic()
        sig = TensorSignature.from_profile(profile)

        cached = self.cache.lookup(sig)
        if cached is not None:
            self._n_cache_hits += 1
            cached_pattern, base_score = cached
            score = self._evaluate_pattern(tensor, cached_pattern)
            elapsed = time.monotonic() - t0
            return self._build_recommendation(
                cached_pattern, score, elapsed, cache_hit=True
            )

        tensor_type = self._resolve_tensor_type(profile)
        best_pattern, best_score = self._evolve(
            tensor_type=tensor_type,
            target_ratio=target_ratio,
            max_error=max_error,
            max_time=max_search_time - (time.monotonic() - t0),
            profile=profile,
            tensor=tensor,
        )

        self.cache.store(sig, best_pattern, best_score)
        elapsed = time.monotonic() - t0
        return self._build_recommendation(
            best_pattern, best_score, elapsed, cache_hit=False
        )

    def suggest_synergy(self, method_a: str, method_b: str) -> float:
        """Query pairwise synergy score between two methods."""
        return self.synergy.get_synergy(method_a, method_b)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "n_evaluations": self._n_total_evaluations,
            "n_cache_hits": self._n_cache_hits,
            "cache_size": len(self.cache._entries),
            "synergy_pairs_tracked": len(self.synergy._matrix),
            "available_methods": len(self._available_methods),
        }

    # ── Evolutionary Search Core ───────────────────────────────────────────

    def _evolve(
        self,
        tensor_type: str,
        target_ratio: float,
        max_error: float,
        max_time: float,
        profile: Any,
        tensor: Optional[np.ndarray] = None,
    ) -> Tuple[StackingPattern, PatternScore]:
        """Run genetic algorithm to discover optimal stacking pattern.

        If *tensor* is provided, evaluates patterns on real data (expensive).
        Otherwise uses a surrogate fitness model.
        """
        t_deadline = time.monotonic() + max_time

        # Initialize population from templates and random patterns
        population: List[Tuple[StackingPattern, PatternScore]] = []
        template = _TENSOR_TYPE_TEMPLATES.get(
            tensor_type, _TENSOR_TYPE_TEMPLATES["weight"]
        )

        # Seed with template
        base_pattern = StackingPattern(stages=template)
        score = self._score_pattern(base_pattern, profile, tensor, target_ratio)
        if score is not None:
            population.append((base_pattern, score))

        # Seed with template variations
        for _ in range(max(2, self._population_size // 8)):
            variant = self._mutate_pattern(base_pattern.copy())
            score = self._score_pattern(variant, profile, tensor, target_ratio)
            if score is not None:
                population.append((variant, score))

        # Seed random patterns
        while len(population) < self._population_size:
            pattern = self._random_pattern(tensor_type)
            score = self._score_pattern(pattern, profile, tensor, target_ratio)
            if score is not None:
                population.append((pattern, score))
            if time.monotonic() > t_deadline:
                break

        if not population:
            fallback = StackingPattern(stages=template)
            return fallback, PatternScore(
                ratio=1.0, error=1.0, speed=0.0, synergy_score=0.0
            )

        best_pattern, best_score = population[0]

        for gen in range(self._generations):
            if time.monotonic() > t_deadline:
                break

            next_pop: List[Tuple[StackingPattern, PatternScore]] = []

            # Elitism: keep top 20%
            population.sort(key=lambda x: x[1].fitness(target_ratio), reverse=True)
            n_elite = max(2, self._population_size // 5)
            for i in range(min(n_elite, len(population))):
                next_pop.append(population[i])

            # Fill rest via tournament selection + crossover + mutation
            while len(next_pop) < self._population_size:
                if time.monotonic() > t_deadline:
                    break

                parent_a = self._tournament_select(population, target_ratio)
                parent_b = self._tournament_select(population, target_ratio)

                if self._rng.random() < _CROSSOVER_RATE:
                    child = self._crossover(parent_a[0], parent_b[0])
                else:
                    child = parent_a[0].copy()

                if self._rng.random() < _MUTATION_RATE:
                    child = self._mutate_pattern(child)

                score = self._score_pattern(child, profile, tensor, target_ratio)
                if score is not None:
                    next_pop.append((child, score))

                    # Track best
                    if score.fitness(target_ratio) > best_score.fitness(target_ratio):
                        best_pattern, best_score = child, score

            population = next_pop

            # Adaptive mutation: increase if no improvement
            if gen > 0 and len(population) > 1:
                prev_best = best_score.fitness(target_ratio)
                curr_best = max(p[1].fitness(target_ratio) for p in population)
                if curr_best <= prev_best + 1e-6:
                    pass  # stagnation — mutation rate already applied

        return best_pattern, best_score

    # ── Pattern Evaluation ─────────────────────────────────────────────────

    def _score_pattern(
        self,
        pattern: StackingPattern,
        profile: Any,
        tensor: Optional[np.ndarray],
        target_ratio: float,
    ) -> Optional[PatternScore]:
        """Score a pattern. Uses real compression if tensor provided, else surrogate."""
        if tensor is not None:
            return self._evaluate_pattern(tensor, pattern)
        return self._surrogate_score(pattern, profile, target_ratio)

    def _evaluate_pattern(
        self, tensor: np.ndarray, pattern: StackingPattern
    ) -> Optional[PatternScore]:
        """Run actual compression through all stages and compute metrics."""
        t0 = time.monotonic()
        tensor = np.asarray(tensor, dtype=np.float32)
        current = tensor
        total_ratio = 1.0
        stages_metadata: List[dict] = []
        method_names: List[str] = []

        for method_name, params in pattern.stages:
            inst = self._available_methods.get(method_name)
            if inst is None:
                return None
            try:
                data, meta = inst.compress(current, **params)
                ratio = _compute_ratio(current.nbytes, data)
                total_ratio *= ratio
                recon = inst.decompress(data, meta)
                current = recon.astype(np.float32, copy=False)
                meta["method"] = method_name
                meta["stage_ratio"] = ratio
                stages_metadata.append(meta)
                method_names.append(method_name)
                del data
            except Exception:
                return None

        elapsed = time.monotonic() - t0
        metrics = _compute_metrics(tensor, current)
        error = metrics.get("relative_error", 1.0)

        # Compute synergy
        synergy_score = self.synergy.synergy_for_pattern(
            [(m, {}) for m in method_names]
        )

        # Update synergy matrix with this observation
        if error < 0.05 and total_ratio > 2.0:
            effective_synergy = total_ratio / max(len(method_names), 1)
            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    self.synergy.update(
                        (method_names[i], method_names[j]),
                        effective_synergy,
                    )

        speed = (tensor.nbytes / 1e6) / max(elapsed, 1e-6)
        self._n_total_evaluations += 1

        gc.collect()
        return PatternScore(
            ratio=total_ratio,
            error=error,
            speed=speed,
            synergy_score=synergy_score,
        )

    def _surrogate_score(
        self,
        pattern: StackingPattern,
        profile: Any,
        target_ratio: float,
    ) -> Optional[PatternScore]:
        """Fast surrogate scoring without running real compression.

        Uses tensor profile statistics to estimate achievable ratio and error
        based on method tiers and synergy.
        """
        if not pattern.stages:
            return None

        # Base estimate from profile
        entropy = getattr(profile, "entropy_rate", 4.0)
        rank = getattr(
            profile,
            "effective_rank",
            max(
                1,
                min(profile.shape) // 4
                if hasattr(profile, "shape") and len(profile.shape) >= 2
                else 64,
            ),
        )
        noise = getattr(profile, "noise_floor", 0.01)
        sparsity = (
            getattr(profile, "unstructured_sparsity_score", 0.0)
            + getattr(profile, "nm_sparsity_score", 0.0)
        ) / 2.0

        n_elements = getattr(profile, "n_elements", 1)
        nbytes = getattr(profile, "nbytes", n_elements * 4)

        total_ratio = 1.0
        total_error_accum = 0.0
        method_names: List[str] = []

        for method_name, params in pattern.stages:
            tier = self._method_tiers.get(method_name, MethodTier.TIER5_QUANTIZATION)
            tier_val = tier_score(tier)

            # Estimate per-stage ratio based on tier and profile
            if tier in (MethodTier.TIER1_REAL_COMPRESSION,):
                rank_ratio = params.get("rank_ratio", 0.3)
                coeff_ratio = params.get("coeff_ratio", 0.4)
                r = 2.0 / max(rank_ratio * coeff_ratio, 0.01)
                err = noise * (1.0 + (1.0 - rank_ratio))
            elif tier == MethodTier.TIER5_QUANTIZATION:
                bits = 8 if "int8" in method_name else 4 if "int4" in method_name else 8
                r = 32.0 / bits
                err = 1.0 / (2.0 ** (bits - 1)) * (1.0 + entropy * 0.5)
            elif tier in (MethodTier.TIER3_ENTROPY,):
                r = 1.2 + entropy * 0.3
                err = 0.0
            else:
                r = 2.0
                err = noise * 0.5

            # Sparsity bonus
            if sparsity > 0.1 and tier == MethodTier.TIER5_QUANTIZATION:
                r *= 1.0 + sparsity

            total_ratio *= max(r, 1.01)
            total_error_accum += err
            method_names.append(method_name)

        # Synergy bonus
        synergy_score = self.synergy.synergy_for_pattern(
            [(m, {}) for m in method_names]
        )
        if synergy_score > 0:
            total_ratio *= 1.0 + 0.15 * math.tanh(synergy_score)
        elif synergy_score < 0:
            total_ratio *= 1.0 + 0.1 * synergy_score  # penalty

        # Cap error
        total_error = min(total_error_accum / max(len(pattern.stages), 1), 1.0)

        # Speed estimate (inverse of total complexity)
        speed_estimate = (
            nbytes / 1e6 / (0.001 * len(pattern.stages) * max(math.log2(n_elements), 1))
        )

        self._n_total_evaluations += 1
        return PatternScore(
            ratio=total_ratio,
            error=total_error,
            speed=speed_estimate,
            synergy_score=synergy_score,
        )

    # ── Genetic Operators ──────────────────────────────────────────────────

    def _tournament_select(
        self,
        population: List[Tuple[StackingPattern, PatternScore]],
        target_ratio: float,
    ) -> Tuple[StackingPattern, PatternScore]:
        """Tournament selection: pick best from k random individuals."""
        k = min(_TOURNAMENT_SIZE, len(population))
        idxs = self._rng.sample(range(len(population)), k)
        best_idx = max(idxs, key=lambda i: population[i][1].fitness(target_ratio))
        return population[best_idx]

    def _crossover(self, a: StackingPattern, b: StackingPattern) -> StackingPattern:
        """Simulated binary crossover: swap subsequences between patterns."""
        if len(a) < 2 and len(b) < 2:
            return a.copy()

        # Choose crossover points
        max_len = max(len(a), len(b))
        if max_len < 2:
            return a.copy()

        cp = self._rng.randint(1, max_len)

        child_stages: List[Tuple[str, Dict[str, Any]]] = []
        for i in range(cp):
            if i < len(a):
                child_stages.append(a[i])
        for i in range(cp, max_len):
            if i < len(b):
                child_stages.append(b[i])

        # Truncate to max stages
        child_stages = child_stages[: self._max_stages]
        if not child_stages:
            child_stages = a.stages[:1]
        return StackingPattern(stages=child_stages)

    def _mutate_pattern(self, pattern: StackingPattern) -> StackingPattern:
        """Apply one of several mutation operators."""
        if not pattern.stages:
            return pattern.copy()

        operator = self._rng.choice(["replace", "insert", "delete", "reorder"])
        mutated = pattern.copy()

        if operator == "replace" and mutated.stages:
            idx = self._rng.randint(0, len(mutated.stages) - 1)
            new_method = self._rng.choice(_METHOD_POOL)
            new_params = self._random_params(new_method)
            mutated.stages[idx] = (new_method, new_params)
        elif operator == "insert":
            if len(mutated.stages) < self._max_stages:
                idx = self._rng.randint(0, len(mutated.stages))
                new_method = self._rng.choice(_METHOD_POOL)
                new_params = self._random_params(new_method)
                mutated.stages.insert(idx, (new_method, new_params))
        elif operator == "delete" and len(mutated.stages) > _MIN_STAGES:
            idx = self._rng.randint(0, len(mutated.stages) - 1)
            mutated.stages.pop(idx)
        elif operator == "reorder" and len(mutated.stages) >= 3:
            idx_a, idx_b = self._rng.sample(range(len(mutated.stages)), 2)
            mutated.stages[idx_a], mutated.stages[idx_b] = (
                mutated.stages[idx_b],
                mutated.stages[idx_a],
            )

        return mutated

    def _random_pattern(self, tensor_type: str) -> StackingPattern:
        """Generate a random stacking pattern."""
        n_stages = self._rng.randint(_MIN_STAGES, min(self._max_stages, 3))

        # Smart selection: avoid putting quantization before decomposition
        stages: List[Tuple[str, Dict[str, Any]]] = []
        categories_used: set = set()

        for stage_idx in range(n_stages):
            if stage_idx == 0:
                pool = _PRIMARY_METHODS
            elif stage_idx == n_stages - 1:
                pool = _FINAL_METHODS
            else:
                pool = _SECONDARY_METHODS

            # Ensure variety: pick from a different category if possible
            for _ in range(20):
                mname = self._rng.choice(pool)
                if mname not in [s[0] for s in stages]:
                    break
            else:
                mname = self._rng.choice(pool)

            params = self._random_params(mname)
            stages.append((mname, params))

        return StackingPattern(stages=stages)

    @staticmethod
    def _random_params(method_name: str) -> Dict[str, Any]:
        """Generate random parameters for a given method."""
        if "svd" in method_name or "decomp" in method_name or "tensor" in method_name:
            return {"rank_ratio": round(random.uniform(0.05, 0.5), 3)}
        if "dct" in method_name or "fwht" in method_name or "spectral" in method_name:
            return {"coeff_ratio": round(random.uniform(0.1, 0.8), 3)}
        if "int8" in method_name:
            return {"block_size": random.choice([64, 128, 256])}
        if "int4" in method_name:
            return {"block_size": random.choice([16, 32, 64])}
        if method_name in ("zstd", "rans", "lz4"):
            return {}
        return {}

    # ── Helpers ────────────────────────────────────────────────────────────

    def _resolve_tensor_type(self, profile: Any) -> str:
        """Determine tensor type from profile."""
        tensor_type = getattr(profile, "tensor_type", "")
        if tensor_type and tensor_type != "generic":
            return tensor_type

        name = getattr(profile, "name", "")
        nl = name.lower()
        if not nl:
            return "weight"

        if any(k in nl for k in ("embed", "tok_embeddings", "wte")):
            return "embedding"
        if any(k in nl for k in ("attn_q", "q_proj", "wq", "query")):
            return "attention_q"
        if any(k in nl for k in ("attn_k", "k_proj", "wk", "key")):
            return "attention_k"
        if any(k in nl for k in ("attn_v", "v_proj", "wv", "value")):
            return "attention_v"
        if any(k in nl for k in ("attn_o", "o_proj", "wo", "out")):
            return "attention_o"
        if any(k in nl for k in ("qkv",)):
            return "qkv_fused"
        if any(k in nl for k in ("ffn_gate", "gate_proj", "w1", "fc_gate")):
            return "ffn_gate"
        if any(k in nl for k in ("ffn_up", "up_proj", "w3")):
            return "ffn_up"
        if any(k in nl for k in ("ffn_down", "down_proj", "w2")):
            return "ffn_down"
        if any(k in nl for k in ("ffn", "mlp", "expert")):
            return "ffn"
        if any(k in nl for k in ("norm", "ln_", "rms")):
            return "norm"
        if any(k in nl for k in ("output", "lm_head", "head")):
            return "output"
        return "weight"

    def _build_recommendation(
        self,
        pattern: StackingPattern,
        score: PatternScore,
        elapsed: float,
        cache_hit: bool = False,
    ) -> Dict[str, Any]:
        return {
            "stages": [(name, dict(params)) for name, params in pattern.stages],
            "expected_ratio": round(score.ratio, 2),
            "expected_error": round(score.error, 6),
            "synergy_score": round(score.synergy_score, 4),
            "search_time": round(elapsed, 4),
            "cache_hit": cache_hit,
        }

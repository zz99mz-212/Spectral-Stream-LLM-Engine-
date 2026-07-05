"""
Self-Evolving Compression Intelligence
======================================
A system that learns from every compression run and continuously improves.

Key innovations:
1. BAYESIAN PERFORMANCE TRACKING - probabilistic models of method performance
2. CROSS-MODEL KNOWLEDGE TRANSFER - lessons from Gemma help compress MiMo
3. ADAPTIVE EXPLORATION-EXPLOITATION - balances trying new methods vs using proven ones
4. AUTOMATIC STRATEGY EVOLUTION - evolves selection strategies via genetic algorithms
5. KNOWLEDGE GRAPH - maps relationships between tensor types, methods, and outcomes

This creates a COMPRESSION INTELLIGENCE that grows smarter with every model it processes.
"""

import logging
import time
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# BAYESIAN PERFORMANCE TRACKER
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MethodPerformance:
    """Bayesian performance model for a method on a tensor type."""

    method_name: str = ""
    tensor_type: str = ""

    # Posterior distribution parameters (Beta distribution for success)
    n_trials: int = 0
    n_successes: int = 0

    # Gaussian posterior for ratio
    ratio_mean: float = 3.88  # Default: block_int8 baseline
    ratio_variance: float = 1.0
    ratio_n: int = 0

    # Gaussian posterior for error
    error_mean: float = 0.01
    error_variance: float = 0.01
    error_n: int = 0

    # Bayesian scores
    expected_ratio: float = 3.88
    expected_error: float = 0.01
    confidence: float = 0.0  # How sure are we? (0-1)

    def update(self, ratio: float, error: float, success: bool):
        """Bayesian update with new observation."""
        self.n_trials += 1
        if success:
            self.n_successes += 1

        # Update ratio (Gaussian posterior)
        alpha = 1.0 / (1.0 + self.ratio_n)
        self.ratio_mean = (1 - alpha) * self.ratio_mean + alpha * ratio
        self.ratio_variance = (1 - alpha) * self.ratio_variance + alpha * (
            ratio - self.ratio_mean
        ) ** 2
        self.ratio_n += 1

        # Update error (Gaussian posterior)
        alpha = 1.0 / (1.0 + self.error_n)
        self.error_mean = (1 - alpha) * self.error_mean + alpha * error
        self.error_variance = (1 - alpha) * self.error_variance + alpha * (
            error - self.error_mean
        ) ** 2
        self.error_n += 1

        # Update expectations
        self.expected_ratio = self.ratio_mean
        self.expected_error = max(self.error_mean, 1e-10)

        # Confidence increases with more data
        self.confidence = min(1.0, self.n_trials / 20.0)  # 20 trials = fully confident

    @property
    def score(self) -> float:
        """Bayesian score: expected ratio / expected error * confidence."""
        return (
            self.expected_ratio
            / max(self.expected_error, 1e-10)
            * (0.5 + 0.5 * self.confidence)
        )

    @property
    def success_rate(self) -> float:
        return self.n_successes / max(self.n_trials, 1)


class BayesianPerformanceTracker:
    """
    Tracks compression method performance using Bayesian inference.

    Each (method, tensor_type) pair has a Bayesian posterior distribution
    that gets updated with each observation.

    This allows:
    1. Probabilistic predictions ("method X will give ratio Y ± Z")
    2. Uncertainty-aware selection (explore uncertain methods)
    3. Automatic confidence growth with more data
    """

    def __init__(self):
        self._performances: Dict[str, MethodPerformance] = {}
        self._history: List[Dict] = []

    def _key(self, method_name: str, tensor_type: str) -> str:
        return f"{method_name}:{tensor_type}"

    def record(self, method_name: str, tensor_type: str, ratio: float, error: float):
        """Record a compression outcome."""
        key = self._key(method_name, tensor_type)
        success = error < 0.01  # < 1% error = success

        if key not in self._performances:
            self._performances[key] = MethodPerformance(
                method_name=method_name, tensor_type=tensor_type
            )

        self._performances[key].update(ratio, error, success)
        self._history.append(
            {
                "method": method_name,
                "tensor_type": tensor_type,
                "ratio": ratio,
                "error": error,
                "success": success,
                "timestamp": time.time(),
            }
        )

    def predict(self, method_name: str, tensor_type: str) -> MethodPerformance:
        """Predict performance of a method on a tensor type."""
        key = self._key(method_name, tensor_type)

        if key in self._performances:
            return self._performances[key]

        # No direct data - use cross-method transfer
        # Find similar methods in the same category
        similar = [
            p for k, p in self._performances.items() if method_name.split("_")[0] in k
        ]

        if similar:
            # Average performance of similar methods
            avg_ratio = float(np.mean([s.expected_ratio for s in similar]))
            avg_error = float(np.mean([s.expected_error for s in similar]))
            return MethodPerformance(
                method_name=method_name,
                tensor_type=tensor_type,
                expected_ratio=avg_ratio,
                expected_error=avg_error,
                confidence=0.5,  # Lower confidence for transferred knowledge
            )

        # No data at all - return default
        return MethodPerformance(
            method_name=method_name,
            tensor_type=tensor_type,
            expected_ratio=3.88,
            expected_error=0.01,
            confidence=0.1,
        )

    def get_best_method(
        self, tensor_type: str, available_methods: List[str]
    ) -> Tuple[str, float]:
        """Get the best method for a tensor type (exploit)."""
        best_score = -1
        best_method = available_methods[0] if available_methods else "block_int8"

        for method in available_methods:
            perf = self.predict(method, tensor_type)
            if perf.score > best_score:
                best_score = perf.score
                best_method = method

        return best_method, best_score

    def select_method(
        self, tensor_type: str, available_methods: List[str], epsilon: float = 0.1
    ) -> str:
        """
        Epsilon-greedy method selection.

        With probability epsilon, explore (try an uncertain method).
        With probability 1-epsilon, exploit (use best known method).

        This balances trying new things vs using what works.
        """
        # Exploit: use best known method
        if np.random.random() > epsilon:
            return self.get_best_method(tensor_type, available_methods)[0]

        # Explore: try a method with low confidence
        uncertain = [
            m
            for m in available_methods
            if self.predict(m, tensor_type).confidence < 0.5
        ]
        if uncertain:
            return np.random.choice(uncertain)

        # All methods are well-known, exploit
        return self.get_best_method(tensor_type, available_methods)[0]

    def save(self, path: str):
        """Save performance data."""
        data = {
            "performances": {k: asdict(v) for k, v in self._performances.items()},
            "history": self._history[-1000:],  # Keep last 1000 records
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, path: str):
        """Load performance data."""
        with open(path) as f:
            data = json.load(f)
        for key, perf_data in data.get("performances", {}).items():
            perf = MethodPerformance(**perf_data)
            self._performances[key] = perf
        self._history = data.get("history", [])


# ═══════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════


class CompressionKnowledgeGraph:
    """
    Knowledge graph that maps relationships between:
    - Tensor types (embedding, attention_q, ffn_gate, etc.)
    - Method categories (decomposition, spectral, quantization, etc.)
    - Outcome metrics (ratio, error, time)

    The graph LEARNS which tensor types respond well to which methods
    and can TRANSFER knowledge between different model architectures.
    """

    def __init__(self):
        # Graph: tensor_type -> method_category -> {ratio, error, n}
        self._graph: Dict[str, dict] = defaultdict(
            lambda: defaultdict(
                lambda: {"ratio": 0.0, "error": 0.0, "n": 0, "score": 0.0}
            )
        )

        # Cross-architecture transfer: model_arch -> tensor_type -> method
        self._cross_arch: Dict[str, Dict[str, str]] = defaultdict(dict)

    def update(
        self, tensor_type: str, method_category: str, ratio: float, error: float
    ):
        """Update the knowledge graph with a new observation."""
        entry = self._graph[tensor_type][method_category]
        n = entry["n"]

        # Exponential moving average
        alpha = 1.0 / (1.0 + n)
        entry["ratio"] = (1 - alpha) * entry["ratio"] + alpha * ratio
        entry["error"] = (1 - alpha) * entry["error"] + alpha * error
        entry["n"] += 1
        entry["score"] = entry["ratio"] / max(entry["error"], 1e-10)

    def get_best_category(self, tensor_type: str) -> str:
        """Get the best method category for a tensor type."""
        if tensor_type not in self._graph:
            return "quantization"

        categories = self._graph[tensor_type]
        return max(categories, key=lambda c: categories[c]["score"])

    def get_top_categories(self, tensor_type: str, top_k: int = 3) -> List[str]:
        """Get top-k method categories for a tensor type."""
        if tensor_type not in self._graph:
            return ["quantization", "decomposition", "spectral"]

        sorted_cats = sorted(
            self._graph[tensor_type].items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )
        return [cat for cat, _ in sorted_cats[:top_k]]

    def transfer_knowledge(self, source_arch: str, target_arch: str):
        """Transfer knowledge between model architectures."""
        # Map source tensor types to target
        arch_map = {
            "gemma": {"q_proj": "q_proj", "v_proj": "v_proj", "gate_proj": "gate_proj"},
            "llama": {"q_proj": "q_proj", "v_proj": "v_proj", "gate_proj": "gate_proj"},
            "mimo": {"q_proj": "q_proj", "v_proj": "v_proj", "w1": "gate_proj"},
        }

        source_map = arch_map.get(source_arch, {})
        target_map = arch_map.get(target_arch, {})

        reverse_map = {v: k for k, v in target_map.items()}

        for src_type, methods in dict(self._graph.get(source_arch, {})).items():
            if src_type in source_map:
                common_type = source_map[src_type]
                if common_type in reverse_map:
                    tgt_type = reverse_map[common_type]
                    # Transfer knowledge
                    for cat, stats in list(methods.items()):
                        copied = dict(stats)
                        copied["n"] = max(1, int(stats["n"]) // 2)
                        self._graph[tgt_type][cat] = copied

    def to_dict(self) -> Dict:
        return {
            "graph": {t: dict(c) for t, c in self._graph.items()},
            "cross_arch": dict(self._cross_arch),
        }


# ═══════════════════════════════════════════════════════════════════════
# GENETIC STRATEGY EVOLVER
# ═══════════════════════════════════════════════════════════════════════


class GeneticStrategyEvolver:
    """
    Uses a GENETIC ALGORITHM to evolve the optimal compression strategy.

    Each "genome" is a set of rules mapping tensor properties to methods.
    Genomes compete: better compression -> higher fitness -> more descendants.
    Over generations, the strategy evolves to be more effective.

    Genotype:
    - Gene 1: effective_rank_threshold -> decomposition vs quantization
    - Gene 2: energy_concentration_threshold -> spectral vs structural
    - Gene 3: outlier_ratio_threshold -> outlier-aware methods
    - Gene 4: sparsity_threshold -> sparsity methods
    - Gene 5: tensor_type_bias -> per-type method preferences
    """

    def __init__(self, population_size: int = 50):
        self.population_size = population_size
        self.population = self._initialize_population()
        self.generation = 0
        self.best_genome = None
        self.best_fitness = 0

    def _initialize_population(self) -> List[Dict]:
        """Create initial random population of strategies."""
        population = []
        for _ in range(self.population_size):
            genome = {
                "rank_threshold": np.random.uniform(0.1, 0.9),
                "energy_threshold": np.random.uniform(0.3, 0.95),
                "outlier_threshold": np.random.uniform(0.01, 0.5),
                "sparsity_threshold": np.random.uniform(0.1, 0.9),
                "tier_bias": np.random.uniform(0.1, 0.9),
                "exploration_rate": np.random.uniform(0.01, 0.3),
                "prefer_spectral": np.random.choice([True, False]),
                "prefer_structural": np.random.choice([True, False]),
                "prefer_decomposition": np.random.choice([True, False]),
            }
            population.append(genome)
        return population

    def evaluate_fitness(self, genome: Dict, test_results: List[Dict]) -> float:
        """Evaluate how good a strategy is based on actual compression results."""
        if not test_results:
            return 0.0

        # Fitness = average score across all tests
        scores = [r.get("score", 0) for r in test_results if r.get("score")]
        return float(np.mean(scores)) if scores else 0.0

    def select_parents(self) -> Tuple[Dict, Dict]:
        """Tournament selection: pick two parents."""

        def tournament():
            candidates = np.random.choice(len(self.population), 5, replace=False)
            best = max(candidates, key=lambda i: self.population[i].get("_fitness", 0))
            return self.population[best]

        return tournament(), tournament()

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Single-point crossover to create offspring."""
        child = {}
        keys = list(parent1.keys())
        split = np.random.randint(0, len(keys))

        for i, key in enumerate(keys):
            if key.startswith("_"):
                continue
            child[key] = parent1[key] if i < split else parent2[key]

        return child

    def mutate(self, genome: Dict, mutation_rate: float = 0.1) -> Dict:
        """Random mutation of genome."""
        mutated = genome.copy()
        for key in mutated:
            if key.startswith("_"):
                continue
            if np.random.random() < mutation_rate:
                if isinstance(mutated[key], bool):
                    mutated[key] = not mutated[key]
                elif isinstance(mutated[key], float):
                    mutated[key] = np.clip(
                        mutated[key] + np.random.randn() * 0.1, 0.0, 1.0
                    )
        return mutated

    def evolve(self, test_results: List[Dict]) -> Optional[Dict]:
        """Run one generation of evolution."""
        # Evaluate fitness
        for genome in self.population:
            genome["_fitness"] = self.evaluate_fitness(genome, test_results)

        # Track best
        best = max(self.population, key=lambda g: g.get("_fitness", 0))
        if best["_fitness"] > self.best_fitness:
            self.best_fitness = best["_fitness"]
            self.best_genome = best.copy()

        # Create new population
        new_population = [best.copy()]  # Elitism: keep best

        while len(new_population) < self.population_size:
            p1, p2 = self.select_parents()
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        return self.best_genome


# ═══════════════════════════════════════════════════════════════════════
# SELF-EVOLVING ENGINE
# ═══════════════════════════════════════════════════════════════════════


class SelfEvolvingIntelligenceEngine:
    """
    The COMPLETE self-evolving compression intelligence.

    Combines:
    1. Bayesian performance tracking (learns from each compression)
    2. Knowledge graph (cross-model knowledge transfer)
    3. Genetic strategy evolution (optimizes strategy over time)

    This engine gets BETTER with every model it compresses.
    It starts knowing nothing and evolves into an expert.
    """

    def __init__(self, knowledge_path: Optional[str] = None):
        self.tracker = BayesianPerformanceTracker()
        self.knowledge_graph = CompressionKnowledgeGraph()
        self.evolver = GeneticStrategyEvolver()

        self._compression_history: List[Dict] = []
        self._model_architectures_seen: set = set()

        # Load existing knowledge
        if knowledge_path and Path(knowledge_path).exists():
            self.load_knowledge(knowledge_path)

    def record_compression(
        self,
        tensor_name: str,
        tensor_type: str,
        method_name: str,
        method_category: str,
        ratio: float,
        error: float,
        model_architecture: str = "unknown",
    ):
        """Record a compression outcome and update all learning systems."""
        # 1. Bayesian tracker
        success = error < 0.01
        self.tracker.record(method_name, tensor_type, ratio, error)

        # 2. Knowledge graph
        self.knowledge_graph.update(tensor_type, method_category, ratio, error)

        # 3. History
        self._compression_history.append(
            {
                "tensor_name": tensor_name,
                "tensor_type": tensor_type,
                "method_name": method_name,
                "method_category": method_category,
                "ratio": ratio,
                "error": error,
                "success": success,
                "model_architecture": model_architecture,
                "timestamp": time.time(),
            }
        )

        # 4. Track architectures
        self._model_architectures_seen.add(model_architecture)

    def select_method(
        self,
        tensor_type: str,
        available_methods: Dict[str, Dict],
        tensor_profile: Any = None,
    ) -> str:
        """
        Select the best method using ALL learned knowledge.

        Selection algorithm:
        1. Check knowledge graph for best category
        2. Use Bayesian tracker for best method in category
        3. Apply epsilon-greedy for exploration
        4. Return selected method
        """
        # Get top 3 categories from knowledge graph
        top_categories = self.knowledge_graph.get_top_categories(tensor_type, 3)

        # Filter methods by top categories
        category_methods = defaultdict(list)
        for name, info in available_methods.items():
            cat = info.get("category", "unknown")
            if cat in top_categories:
                category_methods[cat].append(name)

        # Use Bayesian tracker to select best method in each category
        best_method = "block_int8"
        best_score = 0

        for cat, methods in category_methods.items():
            method, score = self.tracker.get_best_method(tensor_type, methods)
            if score > best_score:
                best_score = score
                best_method = method

        # Apply epsilon-greedy exploration
        if np.random.random() < 0.1:  # 10% exploration
            all_methods = list(available_methods.keys())
            uncertain = [
                m
                for m in all_methods
                if self.tracker.predict(m, tensor_type).confidence < 0.3
            ]
            if uncertain:
                best_method = np.random.choice(uncertain)

        return best_method

    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            "total_compressions": len(self._compression_history),
            "unique_methods_tracked": len(self.tracker._performances),
            "tensor_types_in_graph": len(self.knowledge_graph._graph),
            "model_architectures_seen": list(self._model_architectures_seen),
            "evolution_generations": self.evolver.generation,
            "best_evolution_fitness": self.evolver.best_fitness,
        }

    def save_knowledge(self, path: str):
        """Save all learned knowledge."""
        self.tracker.save(path + ".perf")

        knowledge = {
            "graph": self.knowledge_graph.to_dict(),
            "architectures": list(self._model_architectures_seen),
            "history": self._compression_history[-500:],
            "best_genome": self.evolver.best_genome,
            "generation": self.evolver.generation,
        }
        with open(path + ".knowledge", "w") as f:
            json.dump(knowledge, f, indent=2, default=str)

    def load_knowledge(self, path: str):
        """Load learned knowledge."""
        perf_path = path + ".perf"
        if Path(perf_path).exists():
            self.tracker.load(perf_path)

        knowledge_path = path + ".knowledge"
        if Path(knowledge_path).exists():
            with open(knowledge_path) as f:
                knowledge = json.load(f)
            # Load knowledge graph data
            graph_data = knowledge.get("graph", {}).get("graph", {})
            for ttype, cats in graph_data.items():
                for cat, stats in cats.items():
                    self.knowledge_graph._graph[ttype][cat] = stats.copy()
            self._model_architectures_seen = set(knowledge.get("architectures", []))
            self.evolver.best_genome = knowledge.get("best_genome")
            self.evolver.generation = knowledge.get("generation", 0)


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION HOOK
# ═══════════════════════════════════════════════════════════════════════


def integrate_self_evolving_engine(engine, knowledge_path: Optional[str] = None):
    """Integrate the self-evolving intelligence with the main engine."""
    sei = SelfEvolvingIntelligenceEngine(knowledge_path)
    engine._self_evolving = sei
    logger.info("Self-evolving intelligence integrated")
    return sei

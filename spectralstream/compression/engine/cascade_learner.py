from __future__ import annotations

import gc
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_STAGE_NAMES = (
    "svd_compress",
    "dct_2d",
    "fwht_compress",
    "tensor_train",
    "block_int8",
    "block_int4",
    "hadamard_int8",
    "hadamard_int4",
    "delta_int4",
    "sparsity_int4",
    "uniform_quantize",
    "arithmetic_encode",
    "range_encode",
    "zstd_compress",
)


class CascadePattern:
    """A learned cascade pattern: sequence of (method, params) with expected outcomes."""

    def __init__(
        self,
        tensor_type: str,
        stages: List[Tuple[str, dict]],
        expected_ratio: float,
        expected_cosine: float,
        n_observations: int = 1,
    ) -> None:
        self.tensor_type = tensor_type
        self.stages = stages
        self.expected_ratio = expected_ratio
        self.expected_cosine = expected_cosine
        self.n_observations = n_observations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_type": self.tensor_type,
            "stages": [(m, p) for m, p in self.stages],
            "expected_ratio": self.expected_ratio,
            "expected_cosine": self.expected_cosine,
            "n_observations": self.n_observations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CascadePattern:
        return cls(
            tensor_type=data["tensor_type"],
            stages=[(m, dict(p)) for m, p in data["stages"]],
            expected_ratio=data["expected_ratio"],
            expected_cosine=data["expected_cosine"],
            n_observations=data.get("n_observations", 1),
        )


_TENSOR_TYPES = (
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_o",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "embedding",
    "norm",
    "weight",
    "unknown",
)


class CascadeLearner:
    """Learns optimal cascade patterns per tensor type from compression experience.

    Uses a lightweight approach:
    - Tensor profile fingerprint -> hash -> lookup known patterns
    - Pattern database stored as JSON (compressed tensor characteristics)
    - Bayesian update: posterior = prior + observation
    - Epsilon-greedy exploration: 10% chance to try novel patterns
    """

    def __init__(self, knowledge_path: Optional[str] = None) -> None:
        self._patterns: Dict[str, List[CascadePattern]] = defaultdict(list)
        self._tensor_profiles: Dict[str, Dict[str, float]] = {}
        self._exploration_rate: float = 0.1
        self._knowledge_path = knowledge_path or os.path.expanduser(
            "~/.spectralstream/cascade_knowledge.json"
        )
        self._rng = np.random.default_rng()
        self._load_knowledge()

    # ── public API ───────────────────────────────────────────────────

    def get_tensor_fingerprint(self, profile: Any) -> str:
        """Create a hash from tensor profile characteristics.

        Uses: shape, effective_rank, spectral_energy_concentration,
        sparsity, sensitivity, std/mean ratio.
        Returns a category string like 'attention_q_2048x1536_rank60'
        """
        tensor_type = getattr(profile, "tensor_type", None) or getattr(
            profile, "name", "unknown"
        )

        shape = getattr(profile, "shape", None)
        if shape is None and isinstance(profile, dict):
            shape = profile.get("shape", (1,))
        if shape is None:
            shape = (1,)
        if isinstance(shape, (list, tuple)):
            shape_str = "x".join(str(d) for d in shape[:4])
        else:
            shape_str = str(shape)

        effective_rank = getattr(profile, "effective_rank", None)
        if effective_rank is None and isinstance(profile, dict):
            effective_rank = profile.get("effective_rank", 0.5)
        if effective_rank is None:
            effective_rank = 0.5

        spec_energy = getattr(profile, "energy_concentration_dct", None)
        if spec_energy is None and isinstance(profile, dict):
            spec_energy = profile.get("energy_concentration_dct", 0.5)
        if spec_energy is None:
            spec_energy = 0.5

        return f"{tensor_type}_{shape_str}_rank{int(effective_rank * 100)}"

    def suggest_cascade(
        self,
        tensor_type: str,
        target_ratio: float,
        max_error: float,
        profile: Any,
    ) -> Optional[List[Tuple[str, dict]]]:
        """Suggest optimal cascade pattern from learned knowledge.

        Returns list of (method_name, params) or None if no knowledge exists.
        With epsilon probability, returns a random valid pattern (exploration).
        """
        fp = (
            self.get_tensor_fingerprint(profile) if profile is not None else tensor_type
        )

        patterns = self._patterns.get(tensor_type, [])

        if self._rng.random() < self._exploration_rate:
            return self._generate_exploration_pattern(tensor_type, target_ratio)

        if not patterns:
            return None

        best = max(patterns, key=lambda p: p.expected_cosine * p.n_observations)
        return best.stages

    def record_result(
        self,
        tensor_type: str,
        profile: Any,
        stages: List[Tuple[str, dict]],
        ratio: float,
        cosine_similarity: float,
        error: float,
    ) -> None:
        """Record a compression result to improve future predictions.

        Bayesian update: new_estimate = (prior * n + observation) / (n + 1)
        """
        if not stages:
            return

        existing = [
            p for p in self._patterns.get(tensor_type, []) if p.stages == stages
        ]

        if existing:
            pat = existing[0]
            n = pat.n_observations
            pat.expected_ratio = (pat.expected_ratio * n + ratio) / (n + 1)
            pat.expected_cosine = (pat.expected_cosine * n + cosine_similarity) / (
                n + 1
            )
            pat.n_observations = n + 1
        else:
            pat = CascadePattern(
                tensor_type=tensor_type,
                stages=stages,
                expected_ratio=ratio,
                expected_cosine=cosine_similarity,
                n_observations=1,
            )
            self._patterns[tensor_type].append(pat)

        fp = (
            self.get_tensor_fingerprint(profile) if profile is not None else tensor_type
        )
        self._tensor_profiles[fp] = {
            "tensor_type": tensor_type,
            "ratio": ratio,
            "cosine_similarity": cosine_similarity,
            "error": error,
        }

        self._save_knowledge()

        del stages
        gc.collect()

    def get_best_pattern(self, tensor_type: str) -> Optional[CascadePattern]:
        """Return the best-known pattern for a tensor type."""
        patterns = self._patterns.get(tensor_type, [])
        if not patterns:
            return None
        return max(patterns, key=lambda p: p.expected_cosine * p.n_observations)

    def get_statistics(self) -> Dict[str, Any]:
        """Return learning statistics: patterns learned, confidence, etc."""
        total_patterns = sum(len(v) for v in self._patterns.values())
        total_observations = sum(
            p.n_observations for v in self._patterns.values() for p in v
        )
        best_by_type: Dict[str, Dict[str, Any]] = {}
        for ttype in self._patterns:
            best = self.get_best_pattern(ttype)
            if best is not None:
                best_by_type[ttype] = {
                    "stages": [m for m, _ in best.stages],
                    "expected_ratio": round(best.expected_ratio, 1),
                    "expected_cosine": round(best.expected_cosine, 4),
                    "confidence": round(min(best.n_observations / 10.0, 1.0), 3),
                    "n_observations": best.n_observations,
                }

        return {
            "knowledge_path": self._knowledge_path,
            "n_tensor_types": len(self._patterns),
            "n_patterns_total": total_patterns,
            "n_observations_total": total_observations,
            "exploration_rate": self._exploration_rate,
            "best_patterns": best_by_type,
        }

    # ── persistence ─────────────────────────────────────────────────

    def _load_knowledge(self) -> None:
        """Load learned patterns from JSON file."""
        if not os.path.exists(self._knowledge_path):
            return
        try:
            with open(self._knowledge_path, "r") as f:
                data = json.load(f)
            raw_patterns: Dict[str, List[dict]] = data.get("patterns", {})
            for ttype, plist in raw_patterns.items():
                self._patterns[ttype] = [CascadePattern.from_dict(pd) for pd in plist]
            self._tensor_profiles = data.get("profiles", {})
        except (json.JSONDecodeError, KeyError, TypeError):
            self._patterns.clear()
            self._tensor_profiles.clear()

    def _save_knowledge(self) -> None:
        """Save learned patterns to JSON file."""
        os.makedirs(os.path.dirname(self._knowledge_path), exist_ok=True)
        data = {
            "patterns": {
                ttype: [p.to_dict() for p in plist]
                for ttype, plist in self._patterns.items()
            },
            "profiles": self._tensor_profiles,
        }
        with open(self._knowledge_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── internal helpers ────────────────────────────────────────────

    def _generate_exploration_pattern(
        self, tensor_type: str, target_ratio: float
    ) -> List[Tuple[str, dict]]:
        """Generate a random valid cascade pattern for exploration."""
        n_stages = max(2, min(int(np.log2(max(target_ratio, 2.0))), 6))
        available = list(_STAGE_NAMES)
        chosen = self._rng.choice(available, size=n_stages, replace=False).tolist()
        return [(name, {}) for name in chosen]

import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...method_tiers import get_method_tier
from ._common import (
    ERROR_GRADIENT_MAP,
    _decomp_error_gradient,
    _entropy_error_gradient,
    _quant_error_gradient,
    _spectral_error_gradient,
    _structural_error_gradient,
)
from ._stackingcandidate import StackingCandidate
from ._stackingplan import StackingPlan
from ._stackingstage import StackingStage
from .__minimaltensorprofile import _MinimalTensorProfile


logger = logging.getLogger(__name__)


class MultiplicativeStackingEngine:
    """
    Stack MULTIPLE compression methods on the SAME tensor.

    Each method compresses what the previous method couldn't (the residual).
    This gives MULTIPLICATIVE compression ratios instead of additive.

    Stacking patterns (most aggressive to most conservative):

    Pattern 1: "Max Compression" (5000:1+ target)
      Decomposition (SVD rank 16) x Spectral (DCT 10%) x Quant (INT4) x Entropy
      Expected: 100 x 5 x 8 x 2 = 8000x, error ~1-2%

    Pattern 2: "High Quality" (1200:1 target)
      Decomposition (SVD rank 32) x Spectral (DCT 30%) x Quant (INT8) x Entropy
      Expected: 50 x 3 x 4 x 2 = 1200x, error ~0.3-0.6%

    Pattern 3: "Lossless-like" (200:1 target)
      Decomposition (SVD rank 64) x Spectral (DCT 50%) x Entropy
      Expected: 25 x 2 x 2 = 100x, error ~0.05-0.1%
    """

    CASCADE_METHOD_NAMES = [
        "cascade_stage1_structural",
        "cascade_stage2_delta",
        "cascade_stage3_hypernetwork",
        "cascade_stage4_entropy",
        "cascade_full_1200",
    ]

    STACKING_PATTERNS = {
        # ── Phase 1: Tier 1 only (real compression — no quantization) ─────
        "tier1_decomp_spectral": {
            "stages": [
                {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
                {"method_type": "spectral", "params": {"keep_frac": 0.3}},
            ],
            "expected_ratio": 150,
            "expected_error": 0.002,
        },
        "tier1_spectral_only": {
            "stages": [
                {"method_type": "spectral", "params": {"keep_frac": 0.15}},
            ],
            "expected_ratio": 6,
            "expected_error": 0.001,
        },
        "tier1_decomp_only": {
            "stages": [
                {"method_type": "decomposition", "params": {"rank_frac": 0.01}},
            ],
            "expected_ratio": 100,
            "expected_error": 0.005,
        },
        # ── Phase 2: Tier 1 + Tier 2 (structural/physics) ────────────────
        "tier1_tier2": {
            "stages": [
                {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
                {"method_type": "spectral", "params": {"keep_frac": 0.3}},
                {"method_type": "structural", "params": {"block_size": 64}},
            ],
            "expected_ratio": 300,
            "expected_error": 0.005,
        },
        # ── Phase 3: Tier 1 + Tier 2 + Tier 3 (entropy — lossless) ────────
        "tier1_tier2_tier3": {
            "stages": [
                {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
                {"method_type": "spectral", "params": {"keep_frac": 0.3}},
                {"method_type": "entropy", "params": {"method": "rans"}},
            ],
            "expected_ratio": 300,
            "expected_error": 0.002,
        },
        # ── Phase 4: Tier 1 + Tier 2 + Tier 3 + Tier 4 (hybrid) ──────────
        "tier1_through_tier4": {
            "stages": [
                {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
                {"method_type": "spectral", "params": {"keep_frac": 0.3}},
                {"method_type": "structural", "params": {"block_size": 64}},
                {"method_type": "entropy", "params": {"method": "rans"}},
            ],
            "expected_ratio": 600,
            "expected_error": 0.005,
        },
        # ── Compression-only: NO QUANTIZATION (Tiers 1-3 only) ─────────────
        "compression_only": {
            "stages": [
                {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
                {"method_type": "spectral", "params": {"keep_frac": 0.3}},
                {"method_type": "structural", "params": {"block_size": 64}},
            ],
            "expected_ratio": 200,
            "expected_error": 0.005,
            "tiers": [1, 1, 2],  # NO QUANTIZATION — pure compression
        },
        "compression_plus_entropy": {
            "stages": [
                {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
                {"method_type": "spectral", "params": {"keep_frac": 0.3}},
                {"method_type": "entropy", "params": {"method": "rans"}},
            ],
            "expected_ratio": 300,
            "expected_error": 0.005,
            "tiers": [1, 1, 3],  # NO QUANTIZATION — compression + lossless entropy
        },
        "full_cascade": {
            "stages": [
                {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
                {"method_type": "spectral", "params": {"keep_frac": 0.3}},
                {"method_type": "structural", "params": {"block_size": 64}},
                {"method_type": "entropy", "params": {"method": "rans"}},
            ],
            "expected_ratio": 400,
            "expected_error": 0.01,
            "tiers": [1, 1, 2, 3],  # NO QUANTIZATION UNLESS NEEDED
        },
        # ── Phase 5: Including quantization — ABSOLUTE LAST RESORT ────────
        "max_compression": {
            "stages": [
                {"method_type": "decomposition", "params": {"rank_frac": 0.01}},
                {"method_type": "spectral", "params": {"keep_frac": 0.1}},
                {"method_type": "quantization", "params": {"bits": 4}},
                {"method_type": "entropy", "params": {"method": "rans"}},
            ],
            "expected_ratio": 8000,
            "expected_error": 0.02,
        },
        "high_quality": {
            "stages": [
                {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
                {"method_type": "spectral", "params": {"keep_frac": 0.3}},
                {"method_type": "quantization", "params": {"bits": 8}},
                {"method_type": "entropy", "params": {"method": "rans"}},
            ],
            "expected_ratio": 1200,
            "expected_error": 0.006,
        },
        "lossless_like": {
            "stages": [
                {"method_type": "decomposition", "params": {"rank_frac": 0.04}},
                {"method_type": "spectral", "params": {"keep_frac": 0.5}},
                {"method_type": "entropy", "params": {"method": "rans"}},
            ],
            "expected_ratio": 200,
            "expected_error": 0.001,
        },
        # ── Cascade 1200:1 patterns ────────────────────────────────────
        "cascade_svd_delta": {
            "stages": [
                {"method_type": "cascade_stage1_structural", "params": {}},
                {"method_type": "cascade_stage2_delta", "params": {}},
            ],
            "expected_ratio": 20,
            "expected_error": 0.01,
        },
        "cascade_svd_delta_hyper": {
            "stages": [
                {"method_type": "cascade_stage1_structural", "params": {}},
                {"method_type": "cascade_stage2_delta", "params": {}},
                {"method_type": "cascade_stage3_hypernetwork", "params": {}},
            ],
            "expected_ratio": 300,
            "expected_error": 0.02,
        },
        "cascade_full_1200": {
            "stages": [
                {"method_type": "cascade_stage1_structural", "params": {}},
                {"method_type": "cascade_stage2_delta", "params": {}},
                {"method_type": "cascade_stage3_hypernetwork", "params": {}},
                {"method_type": "cascade_stage4_entropy", "params": {}},
            ],
            "expected_ratio": 1200,
            "expected_error": 0.05,
        },
    }

    @staticmethod
    def build_cascade_config(
        target_ratio: float,
    ) -> List[Dict[str, Any]]:
        """Build optimal stacking cascade config for any target ratio.

        Dynamically constructs a tier-ordered cascade pattern based on
        *target_ratio*.  More aggressive ratios include more stages and
        may add quantization as the absolute last resort.

        Tier ordering (strict — never violated):
          1. Decomposition (Tier 1) — highest compression potential
          2. Spectral (Tier 1) — residual energy concentration
          3. Structural (Tier 2) — matrix structure exploitation
          4. Entropy (Tier 3) — lossless squeeze (zero added error)
          5. Quantization (Tier 5) — ABSOLUTE LAST RESORT, only for 3000:1+

        Parameters
        ----------
        target_ratio : float
            Desired compression ratio.

        Returns
        -------
        list of dict
            Stage configurations for plan_stacking.  Each dict has
            ``method_type`` and ``params`` keys.
        """
        stages: List[Dict[str, Any]] = []

        # ── Stage 1: Decomposition (Tier 1) — always included ──
        if target_ratio <= 200:
            rank_frac = 0.04
        elif target_ratio <= 1200:
            rank_frac = 0.02
        else:
            rank_frac = 0.01
        stages.append(
            {
                "method_type": "decomposition",
                "params": {"rank_frac": rank_frac},
            }
        )

        # ── Stage 2: Spectral (Tier 1) — always included ──
        if target_ratio <= 200:
            keep_frac = 0.5
        elif target_ratio <= 1200:
            keep_frac = 0.3
        else:
            keep_frac = 0.1
        stages.append(
            {
                "method_type": "spectral",
                "params": {"keep_frac": keep_frac},
            }
        )

        # ── Stage 3: Structural (Tier 2) — only for extreme targets ──
        if target_ratio > 2000:
            stages.append(
                {
                    "method_type": "structural",
                    "params": {"block_size": 64},
                }
            )

        # ── Stage 4: Entropy (Tier 3) — lossless coding ──
        if target_ratio > 400:
            stages.append(
                {
                    "method_type": "entropy",
                    "params": {"method": "rans"},
                }
            )

        # ── Stage 5: Quantization (Tier 5) — LAST RESORT ──
        if target_ratio > 3000:
            stages.append(
                {
                    "method_type": "quantization",
                    "params": {"bits": 4},
                }
            )
        elif target_ratio > 800:
            stages.append(
                {
                    "method_type": "quantization",
                    "params": {"bits": 8},
                }
            )

        return stages

    def __init__(self, engine):
        self.engine = engine
        self._method_cache = {}

    # ── Compression-First Pattern Selection ───────────────────────────────

    COMPRESSION_ONLY_PATTERN_NAMES = [
        "compression_only",
        "compression_plus_entropy",
        "full_cascade",
        "tier1_decomp_spectral",
        "tier1_tier2",
        "tier1_tier2_tier3",
        "tier1_through_tier4",
    ]

    def get_compression_only_patterns(self) -> Dict[str, Any]:
        """
        Return patterns that explicitly avoid quantization (Tier 5).
        These patterns use only decomposition, spectral, structural,
        and entropy methods — pure compression without bit pruning.

        Returns
        -------
        dict
            Subset of STACKING_PATTERNS with no quantization stages.
        """
        return {
            name: pat
            for name, pat in self.STACKING_PATTERNS.items()
            if not any(
                s.get("method_type", "").lower() == "quantization"
                for s in pat.get("stages", [])
            )
        }

    def has_quantization(self, pattern_name: str) -> bool:
        """Check if a stacking pattern includes quantization."""
        pat = self.STACKING_PATTERNS.get(pattern_name, {})
        return any(
            s.get("method_type", "").lower() == "quantization"
            for s in pat.get("stages", [])
        )

    def get_highest_tier_in_pattern(self, pattern_name: str) -> int:
        """Return the highest tier number used in a stacking pattern."""
        pat = self.STACKING_PATTERNS.get(pattern_name, {})
        tiers = pat.get("tiers", [])
        if not tiers:
            # Infer from stage method_types
            tiers = []
            for s in pat.get("stages", []):
                mt = s.get("method_type", "").lower()
                if mt in (
                    "decomposition",
                    "spectral",
                    "functional",
                    "tensor_network",
                    "novel",
                ):
                    tiers.append(1)
                elif mt in ("structural", "physics"):
                    tiers.append(2)
                elif mt in ("entropy", "lossless"):
                    tiers.append(3)
                elif mt in ("hybrid", "cascade"):
                    tiers.append(4)
                elif mt in ("quantization",):
                    tiers.append(5)
                else:
                    tiers.append(5)
        return max(tiers) if tiers else 5

    # ── Enhancement 1: Per-Stage Method Selection ───────────────────────

    def select_method_for_stage(
        self,
        residual: np.ndarray,
        stage_type: str,
        previous_methods: Optional[List[str]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[Tuple[str, Any]]]:
        """
        Select the best method for this stage using model intelligence.

        Uses the TargetRatioEngine to find the Pareto-optimal method from
        candidates that match *stage_type*.

        For large tensors (>1M elements), bypasses TargetRatioEngine
        and returns the preferred method directly for speed.
        """
        candidates: List[Tuple[str, Any]] = []

        stage_type_lower = stage_type.lower()

        for name, inst in self.engine._methods.items():
            cat = getattr(inst, "category", "").lower()

            if stage_type_lower == "decomposition" and any(
                t in cat
                for t in (
                    "decomposition",
                    "breakthrough_decomposition",
                    "svd",
                    "tensor_train",
                    "cp_decomp",
                    "low_rank",
                )
            ):
                candidates.append((name, inst))

            elif stage_type_lower == "spectral" and any(
                t in cat
                for t in (
                    "spectral",
                    "breakthrough_signal",
                    "dct",
                    "wavelet",
                    "fft",
                    "fourier",
                    "transform",
                )
            ):
                candidates.append((name, inst))

            elif stage_type_lower == "quantization" and any(
                t in cat
                for t in (
                    "quantization",
                    "int",
                    "quant",
                    "sparsity_quant",
                    "delta_quant",
                    "transform_quant",
                )
            ):
                candidates.append((name, inst))

            elif stage_type_lower == "entropy" and any(
                t in cat for t in ("entropy", "rans", "huffman", "zstd", "lossless")
            ):
                candidates.append((name, inst))

            elif stage_type_lower == "structural" and any(
                t in cat
                for t in ("structural", "functional", "tensor_network", "physics")
            ):
                candidates.append((name, inst))

            elif stage_type_lower.startswith("cascade_stage") and any(
                t in cat for t in ("cascade",)
            ):
                candidates.append((name, inst))

        if not candidates:
            logger.warning("No methods found for stage type '%s'", stage_type)
            return None, []

        if previous_methods:
            prev_set = set(previous_methods)
            candidates = [(n, inst) for n, inst in candidates if n not in prev_set]

        if not candidates:
            logger.warning(
                "All candidates for '%s' already used in previous stages", stage_type
            )
            return None, []

        # For large tensors (>1M elements), bypass TargetRatioEngine and
        # use the preferred method for each stage type (fast path)
        if residual.size > 1_000_000:
            preferred = {
                "decomposition": "svd_compress",
                "spectral": "dct_spectral",
                "quantization": "block_int8",
                "entropy": "rans",
                "structural": "einsort",
            }
            pref_name = preferred.get(stage_type_lower)
            if pref_name:
                for n, inst in candidates:
                    if n == pref_name:
                        try:
                            compressed, meta = inst.compress(residual)
                            ratio = residual.nbytes / max(len(compressed), 1)
                            recon = inst.decompress(compressed, meta)
                            err = float(
                                np.linalg.norm(residual.ravel() - recon.ravel())
                                / (np.linalg.norm(residual.ravel()) + 1e-30)
                            )
                            cat = getattr(inst, "category", "")
                            return {
                                "method": n,
                                "params": {},
                                "tier": int(get_method_tier(n, cat)),
                                "predicted_ratio": ratio,
                                "predicted_error": err,
                                "actual_ratio": ratio,
                                "actual_error": err,
                                "score": ratio / max(err, 1e-10),
                            }, candidates
                        except Exception:
                            pass

        # For smaller tensors, use TargetRatioEngine for optimal selection
        from ..target_ratio_engine import TargetRatioEngine

        candidate_methods: Dict[str, Any] = {n: inst for n, inst in candidates}
        tre = TargetRatioEngine(methods=candidate_methods)

        try:
            best = tre.find_best_method(
                residual,
                target_ratio=10.0,
                max_error=0.01,
                validate=True,
            )
            return best, candidates
        except Exception as exc:
            logger.warning(
                "TargetRatioEngine selection failed for '%s': %s", stage_type, exc
            )
            if candidates:
                first_name, first_inst = candidates[0]
                try:
                    compressed, meta = first_inst.compress(residual)
                    ratio = residual.nbytes / max(len(compressed), 1)
                    recon = first_inst.decompress(compressed, meta)
                    err = float(
                        np.linalg.norm(residual.ravel() - recon.ravel())
                        / (np.linalg.norm(residual.ravel()) + 1e-30)
                    )
                    first_cat = getattr(first_inst, "category", "")
                    return {
                        "method": first_name,
                        "params": {},
                        "tier": int(get_method_tier(first_name, first_cat)),
                        "predicted_ratio": ratio,
                        "predicted_error": err,
                        "actual_ratio": ratio,
                        "actual_error": err,
                        "score": ratio / max(err, 1e-10),
                    }, candidates
                except Exception:
                    pass
            return None, candidates

    # ── Enhancement 2: Quality Validation Loop ──────────────────────────

    def execute_with_quality_loop(
        self,
        tensor: np.ndarray,
        tensor_name: str = "",
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        force_stage_types: Optional[List[str]] = None,
    ) -> Optional[StackingCandidate]:
        """
        Execute stacking with a quality validation loop.

        Tries multiple patterns and parameter variations, keeping the
        result with the best score (ratio / error). If quality meets
        *max_error* early, returns immediately.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to compress.
        tensor_name : str
            Name for logging / metadata.
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable relative reconstruction error.
        force_stage_types : list of str, optional
            If set, only use stacking patterns whose stage method_types
            match this list (order-insensitive). Used for tier-ordered
            cascade: Tier 1, Tier 1+2, Tier 1+2+3, etc.

        Returns
        -------
        StackingCandidate or None
            Best candidate found.
        """
        best_candidate: Optional[StackingCandidate] = None

        if force_stage_types is not None:
            allowed_set = set(force_stage_types)
            all_patterns_list = [
                name
                for name, pat in self.STACKING_PATTERNS.items()
                if set(s["method_type"] for s in pat["stages"]).issubset(allowed_set)
            ]
            if not all_patterns_list:
                all_patterns_list = list(self.STACKING_PATTERNS.keys())
        else:
            all_patterns_list = list(self.STACKING_PATTERNS.keys())

        # ── Order patterns: compression-first first, quant last ──
        patterns_to_try = sorted(
            all_patterns_list,
            key=lambda name: (
                # Primary sort: patterns with quantization go LAST
                1 if self.has_quantization(name) else 0,
                # Secondary sort: by highest tier (lower tiers first)
                self.get_highest_tier_in_pattern(name),
                # Tertiary sort: by expected ratio (lower ratio first = more conservative)
                self.STACKING_PATTERNS.get(name, {}).get("expected_ratio", 99999),
            ),
        )

        for pattern_name in patterns_to_try:
            for variation in range(3):
                plan = self.plan_stacking(
                    tensor,
                    tensor_name,
                    target_ratio,
                    max_error,
                    pattern_name=pattern_name,
                    variation=variation,
                )

                if plan.total_ratio < 1.0:
                    continue

                data, meta = self.execute_stacking(plan, tensor)
                reconstructed = self.unstack(data, meta, tensor.shape)

                actual_error = float(
                    np.linalg.norm(tensor.ravel() - reconstructed.ravel())
                    / (np.linalg.norm(tensor.ravel()) + 1e-30)
                )
                actual_ratio = tensor.nbytes / max(len(data), 1)
                score = actual_ratio / max(actual_error, 1e-10)

                candidate = StackingCandidate(
                    plan=plan,
                    data=data,
                    metadata=meta,
                    ratio=actual_ratio,
                    error=actual_error,
                    score=score,
                    pattern_name=pattern_name,
                )

                if best_candidate is None or score > best_candidate.score:
                    best_candidate = candidate

                if actual_error <= max_error:
                    return best_candidate

        # If nothing passed quality, try the candidate with highest score
        if best_candidate is not None:
            logger.info(
                "Quality loop ended — best score=%.2f at ratio=%.1fx error=%.4f%%",
                best_candidate.score,
                best_candidate.ratio,
                best_candidate.error * 100,
            )

        return best_candidate

    # ── Enhancement 3: Multiple Candidate Stacks with Pareto Selection ──

    def generate_stacking_candidates(
        self,
        tensor: np.ndarray,
        tensor_name: str = "",
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        n_candidates: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple stacking candidates with different approaches.

        Each candidate represents a different strategy (aggressive vs.
        conservative, different method combinations) to produce diverse
        trade-offs on the Pareto frontier.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to compress.
        tensor_name : str
            Name for logging / metadata.
        target_ratio : float
            Desired compression ratio (used as reference).
        max_error : float
            Maximum acceptable relative error.
        n_candidates : int
            Number of candidate strategies to generate.

        Returns
        -------
        list of dict
            Each dict has key "stages" -> list of stage configs.
        """
        n = n_candidates

        # Build a diverse set of candidate stage configurations
        # Compression-first candidates (NO quantization) — tried FIRST
        base_candidates: List[Dict[str, Any]] = [
            # Candidate 1: Pure compression — decomposition + spectral + structural (no quant)
            {
                "stages": [
                    {"method_type": "decomposition", "rank_frac": 0.02},
                    {"method_type": "spectral", "keep_frac": 0.3},
                    {"method_type": "structural", "block_size": 64},
                ],
                "_label": "compression_only",
            },
            # Candidate 2: Compression + entropy (still no quant)
            {
                "stages": [
                    {"method_type": "decomposition", "rank_frac": 0.02},
                    {"method_type": "spectral", "keep_frac": 0.3},
                    {"method_type": "entropy", "method": "rans"},
                ],
                "_label": "compression_plus_entropy",
            },
            # Candidate 3: Full compression cascade (no quant)
            {
                "stages": [
                    {"method_type": "decomposition", "rank_frac": 0.02},
                    {"method_type": "spectral", "keep_frac": 0.3},
                    {"method_type": "structural", "block_size": 64},
                    {"method_type": "entropy", "method": "rans"},
                ],
                "_label": "full_cascade",
            },
            # Candidate 4: Aggressive decomposition + light spectral
            {
                "stages": [
                    {"method_type": "decomposition", "rank_frac": 0.01},
                    {"method_type": "spectral", "keep_frac": 0.2},
                    {"method_type": "entropy", "method": "rans"},
                ],
                "_label": "agg_compression_entropy",
            },
            # Candidate 5: Decomposition + structural only
            {
                "stages": [
                    {"method_type": "decomposition", "rank_frac": 0.03},
                    {"method_type": "structural", "block_size": 128},
                ],
                "_label": "decomp_structural",
            },
            # Candidate 6: Spectral heavy + entropy
            {
                "stages": [
                    {"method_type": "spectral", "keep_frac": 0.15},
                    {"method_type": "entropy", "method": "rans"},
                ],
                "_label": "spectral_entropy",
            },
            # Candidate 7: Decomposition only
            {
                "stages": [
                    {"method_type": "decomposition", "rank_frac": 0.01},
                ],
                "_label": "decomp_only",
            },
            # ── Quantization-including candidates (tried LAST) ──
            # Candidate 8: Aggressive decomposition + light spectral + quant
            {
                "stages": [
                    {"method_type": "decomposition", "rank_frac": 0.01},
                    {"method_type": "spectral", "keep_frac": 0.2},
                    {"method_type": "quantization", "bits": 4},
                ],
                "_label": "decomp_spectral_quant",
            },
            # Candidate 9: Moderate decomposition + spectral + quant
            {
                "stages": [
                    {"method_type": "decomposition", "rank_frac": 0.03},
                    {"method_type": "spectral", "keep_frac": 0.4},
                    {"method_type": "quantization", "bits": 8},
                ],
                "_label": "moderate_quant",
            },
            # Candidate 10: Full stack with everything
            {
                "stages": [
                    {"method_type": "decomposition", "rank_frac": 0.02},
                    {"method_type": "spectral", "keep_frac": 0.3},
                    {"method_type": "structural", "block_size": 64},
                    {"method_type": "quantization", "bits": 8},
                    {"method_type": "entropy", "method": "rans"},
                ],
                "_label": "full_with_quant",
            },
            # Candidate 11: Triple quant (fallback)
            {
                "stages": [
                    {"method_type": "quantization", "bits": 8},
                    {"method_type": "quantization", "bits": 4},
                    {"method_type": "entropy", "method": "rans"},
                ],
                "_label": "triple_quant",
            },
        ]

        # Generate variations by adjusting each candidate's parameters
        candidates: List[Dict[str, Any]] = []
        for base in base_candidates:
            candidates.append(base)

            # Two variations: more aggressive and more conservative
            for scale, suffix in [(0.5, "_agg"), (2.0, "_cons")]:
                var_stages: List[Dict[str, Any]] = []
                for stage in base["stages"]:
                    s = dict(stage)
                    for key in ("rank_frac", "keep_frac"):
                        if key in s:
                            s[key] = s[key] * scale
                    for key in ("bits",):
                        if key in s:
                            s[key] = max(2, min(16, int(s[key] * scale)))
                    var_stages.append(s)
                candidates.append({"stages": var_stages})

        return candidates[: max(n * 3, 24)]

    def evaluate_stacking_candidates(
        self,
        tensor: np.ndarray,
        tensor_name: str = "",
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        n_candidates: int = 5,
    ) -> List[StackingCandidate]:
        """
        Generate, execute, and evaluate multiple stacking candidates.

        Returns a Pareto-sorted list of candidates so the caller can
        pick the one with the best ratio/error trade-off.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to compress.
        tensor_name : str
            Name for logging / metadata.
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable reconstruction error.
        n_candidates : int
            Number of base candidates to generate.

        Returns
        -------
        list of StackingCandidate
            Sorted by score descending (best first).
        """
        stage_configs = self.generate_stacking_candidates(
            tensor, tensor_name, target_ratio, max_error, n_candidates
        )

        evaluated: List[StackingCandidate] = []

        for config in stage_configs:
            # Build a plan from this config
            plan = self._plan_from_config(tensor, config["stages"], tensor_name)

            if plan is None or plan.total_ratio < 1.0:
                continue

            try:
                data, meta = self.execute_stacking(plan, tensor)
                reconstructed = self.unstack(data, meta, tensor.shape)

                actual_error = float(
                    np.linalg.norm(tensor.ravel() - reconstructed.ravel())
                    / (np.linalg.norm(tensor.ravel()) + 1e-30)
                )
                actual_ratio = tensor.nbytes / max(len(data), 1)
                score = actual_ratio / max(actual_error, 1e-10)

                evaluated.append(
                    StackingCandidate(
                        plan=plan,
                        data=data,
                        metadata=meta,
                        ratio=actual_ratio,
                        error=actual_error,
                        score=score,
                    )
                )
            except Exception as exc:
                logger.debug("Candidate evaluation failed: %s", exc)
                continue

        # Sort by score descending
        evaluated.sort(key=lambda c: -c.score)

        logger.info(
            "Evaluated %d/%d candidates, best: ratio=%.1fx error=%.4f%% score=%.2f",
            len(evaluated),
            len(stage_configs),
            evaluated[0].ratio if evaluated else 0,
            evaluated[0].error * 100 if evaluated else 0,
            evaluated[0].score if evaluated else 0,
        )

        return evaluated

    def select_pareto_optimal_candidate(
        self,
        candidates: List[StackingCandidate],
        target_ratio: float,
        max_error: float,
    ) -> Optional[StackingCandidate]:
        """
        Select the Pareto-optimal candidate from a list.

        A candidate A dominates B if A.ratio >= B.ratio AND
        A.error <= B.error (with at least one strict).  Among the
        non-dominated candidates, the one closest to *target_ratio*
        with error <= *max_error* is returned.

        Parameters
        ----------
        candidates : list of StackingCandidate
            Evaluated candidates to filter.
        target_ratio : float
            Target compression ratio — used as tie-breaker.
        max_error : float
            Maximum acceptable error — used as hard constraint.

        Returns
        -------
        StackingCandidate or None
        """
        if not candidates:
            return None

        # Vectorized Pareto dominance filter via broadcasting
        ratios = np.array([c.ratio for c in candidates])
        errors = np.array([c.error for c in candidates])
        ratio_ge = ratios[:, None] >= ratios[None, :]
        error_le = errors[:, None] <= errors[None, :]
        any_strict = (ratios[:, None] > ratios[None, :]) | (
            errors[:, None] < errors[None, :]
        )
        dominated_mask = np.any(ratio_ge & error_le & any_strict, axis=1)
        non_dominated = [c for i, c in enumerate(candidates) if not dominated_mask[i]]

        if not non_dominated:
            return candidates[0]

        # Among non-dominated, score by closeness to target + error constraint
        nd_ratios = np.array([c.ratio for c in non_dominated])
        nd_errors = np.array([c.error for c in non_dominated])
        nd_scores = np.array([c.score for c in non_dominated])
        ratio_penalty = np.abs(nd_ratios - target_ratio) / max(target_ratio, 1.0)
        error_ok = (nd_errors <= max_error).astype(np.float64)
        pareto_scores = nd_scores * error_ok - ratio_penalty * 10.0
        best_idx = int(np.argmax(pareto_scores))
        return non_dominated[best_idx]

    # ── Enhancement 4: Progressive Refinement ────────────────────────────

    def progressive_stack(
        self,
        tensor: np.ndarray,
        tensor_name: str = "",
        max_error: float = 0.01,
        start_ratio: float = 10.0,
        max_ratio: float = 100000.0,
        step_factor: float = 2.0,
    ) -> Optional[StackingCandidate]:
        """
        Progressive stacking: start conservative, increase until quality
        degrades.  Finds the MAXIMUM compression ratio automatically.

        The algorithm doubles *target_ratio* each iteration and runs a
        full stacking plan + execute + validate cycle.  As soon as the
        reconstructed error exceeds *max_error*, the previous best is
        returned.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to compress.
        tensor_name : str
            Name for logging / metadata.
        max_error : float
            Maximum acceptable relative reconstruction error.
        start_ratio : float
            Initial compression ratio target (default 10:1).
        max_ratio : float
            Upper bound to stop searching (default 100000:1).
        step_factor : float
            Multiplicative step per iteration (default 2.0).

        Returns
        -------
        StackingCandidate or None
            The best candidate found before quality degraded.
        """
        best_candidate: Optional[StackingCandidate] = None
        target_ratio = start_ratio

        while target_ratio <= max_ratio:
            try:
                plan = self.plan_stacking(tensor, tensor_name, target_ratio, max_error)
            except Exception as exc:
                logger.warning(
                    "plan_stacking failed at ratio %.1f: %s", target_ratio, exc
                )
                break

            if plan.total_ratio < 1.0:
                target_ratio *= step_factor
                continue

            try:
                data, meta = self.execute_stacking(plan, tensor)
                reconstructed = self.unstack(data, meta, tensor.shape)
            except Exception as exc:
                logger.warning(
                    "execute/unstack failed at ratio %.1f: %s", target_ratio, exc
                )
                break

            actual_error = float(
                np.linalg.norm(tensor.ravel() - reconstructed.ravel())
                / (np.linalg.norm(tensor.ravel()) + 1e-30)
            )
            actual_ratio = tensor.nbytes / max(len(data), 1)
            score = actual_ratio / max(actual_error, 1e-10)

            candidate = StackingCandidate(
                plan=plan,
                data=data,
                metadata=meta,
                ratio=actual_ratio,
                error=actual_error,
                score=score,
            )

            if actual_error <= max_error:
                best_candidate = candidate
                logger.info(
                    "Progressive: ratio=%.1fx error=%.4f%% score=%.2f — acceptable",
                    actual_ratio,
                    actual_error * 100,
                    score,
                )
                target_ratio *= step_factor
            else:
                logger.info(
                    "Progressive: ratio=%.1fx error=%.4f%% — quality degraded, stopping",
                    actual_ratio,
                    actual_error * 100,
                )
                break

        if best_candidate is not None:
            logger.info(
                "Progressive stacking complete — best: ratio=%.1fx error=%.4f%%",
                best_candidate.ratio,
                best_candidate.error * 100,
            )
        else:
            logger.warning("Progressive stacking failed to find any acceptable result")

        return best_candidate

    # ── Enhancement 5: Dynamic Pattern Selection via Digital Twin ───────

    def design_optimal_pattern(
        self, tensor: np.ndarray, tensor_name: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Design the optimal stacking pattern for this specific tensor.

        Uses the tensor's profile (digital twin) to determine which method
        types to stack and in what priority order.
        """
        dt = self._profile_tensor(tensor, tensor_name)
        stages: List[Dict[str, Any]] = []

        er = getattr(dt, "effective_rank", 1.0)
        er = er if isinstance(er, (int, float)) else 1.0
        ec = getattr(
            dt,
            "energy_concentration_dct",
            getattr(dt, "energy_concentration", 0.0),
        )
        ec = ec if isinstance(ec, (int, float)) else 0.0
        ts = getattr(dt, "toeplitz_score", 0.0)
        ts = ts if isinstance(ts, (int, float)) else 0.0
        bs = getattr(dt, "block_structure_score", 0.0)
        bs = bs if isinstance(bs, (int, float)) else 0.0

        if er < 0.5:
            stages.append({"method_type": "decomposition", "priority": 1.0})

        if ec > 0.6:
            stages.append({"method_type": "spectral", "priority": 0.8})

        if ts > 0.6 or bs > 0.6:
            stages.append({"method_type": "structural", "priority": 0.6})

        stages.append({"method_type": "quantization", "priority": 0.3})
        stages.append({"method_type": "entropy", "priority": 0.1})

        stages.sort(key=lambda s: -s["priority"])
        return stages

    def _profile_tensor(self, tensor: np.ndarray, tensor_name: str) -> Any:
        profiler = getattr(self.engine, "profiler", None)
        if profiler is not None and hasattr(profiler, "profile_tensor"):
            try:
                return profiler.profile_tensor(tensor, tensor_name)
            except Exception:
                pass
        return _MinimalTensorProfile(tensor)

    # ── Enhancement 6: Adaptive Stage Count ─────────────────────────────

    @staticmethod
    def determine_stage_count(dt: Any, target_ratio: float) -> int:
        """Determine how many stages to use based on tensor and target."""
        cs = getattr(dt, "compressibility_score", 0.0)
        cs = cs if isinstance(cs, (int, float)) else 0.0
        er = getattr(dt, "effective_rank", 1.0)
        er = er if isinstance(er, (int, float)) else 1.0

        if cs > 0.8 and target_ratio < 500:
            return 2
        if cs > 0.6 or target_ratio < 2000:
            return 3
        if target_ratio < 10000:
            return 4
        if er < 0.2 and target_ratio >= 10000:
            return 5
        return 4

    # ── Enhancement 7: Lagrangian-Optimal Sub-Ratio Allocation ──────────

    @staticmethod
    def lagrangian_allocate_sub_ratios(
        target_ratio: float,
        stage_method_types: List[str],
        stage_min_ratios: Optional[List[float]] = None,
        stage_max_ratios: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Lagrangian-optimal allocation of sub-ratios across stages.

        Minimises: total_error = Σ error_i(ratio_i)
        Subject to: Π ratio_i = target_ratio
                    ratio_min_i ≤ ratio_i ≤ ratio_max_i

        Entropy stages (lossless, zero error) are pinned at a fixed minimal
        ratio and excluded from the optimisation.
        """
        n = len(stage_method_types)
        if n == 0:
            return []

        if stage_min_ratios is None:
            stage_min_ratios = [1.2] * n
        if stage_max_ratios is None:
            stage_max_ratios = [500.0] * n

        fixed_indices = [
            i
            for i, mt in enumerate(stage_method_types)
            if ERROR_GRADIENT_MAP.get(mt, _quant_error_gradient)
            is _entropy_error_gradient
        ]
        free_indices = [i for i in range(n) if i not in fixed_indices]

        if not free_indices:
            g = max(1.5, target_ratio ** (1.0 / n))
            return [g] * n

        ratios = [0.0] * n
        fixed_product = 1.0
        for i in fixed_indices:
            r = max(stage_min_ratios[i], 1.5)
            ratios[i] = r
            fixed_product *= r

        free_target = target_ratio / fixed_product
        n_free = len(free_indices)

        if n_free == 0 or free_target <= 1.0:
            return ratios

        # Gradient-projection approach:
        #   x_i = log(r_i), constrain sum(x_i) = log(free_target)
        #   At each step shift ratios inversely proportional to error gradient,
        #   then renormalise to satisfy the product constraint.
        #
        #   For stage i with error model e_i(r), we have:
        #     ∂E/∂x_i = r_i * de_i/dr_i   (chain rule via r_i = exp(x_i))
        #   We project the gradient to have zero mean (keeping sum(x_i) fixed)
        #   and step in the opposite direction, i.e. shrink stages where error
        #   is most sensitive to ratio changes.

        log_ratios = np.full(n_free, np.log(free_target) / n_free)
        lr = 0.5
        free_types = [stage_method_types[i] for i in free_indices]
        grad_fns = np.array(
            [ERROR_GRADIENT_MAP.get(mt, _quant_error_gradient) for mt in free_types],
            dtype=object,
        )
        min_vals = np.array([stage_min_ratios[i] for i in free_indices])
        max_vals = np.array([stage_max_ratios[i] for i in free_indices])

        for _ in range(100):
            ratios_free = np.exp(log_ratios)
            grads = ratios_free * np.array(
                [fn(r) for fn, r in zip(grad_fns, ratios_free)]
            )

            grads -= np.mean(grads)
            log_ratios -= lr * grads

            ratios_free = np.exp(log_ratios)
            ratios_free = np.clip(ratios_free, min_vals, max_vals)
            log_ratios = np.log(ratios_free)

            current_log = np.sum(log_ratios)
            log_ratios += (np.log(free_target) - current_log) / n_free

            if np.std(grads) < 1e-6:
                break

        for j, idx in enumerate(free_indices):
            ratios[idx] = float(np.exp(log_ratios[j]))

        return ratios

    # ── Enhancement 8: Error Feedback Between Stages ────────────────────

    @staticmethod
    def compute_residual(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """Compute residual for next stage, with error shaping (whitening)."""
        residual = original - reconstructed
        r_mean = np.mean(residual)
        r_std = np.std(residual)
        if r_std > 1e-30:
            residual = (residual - r_mean) / r_std
        return residual

    # ── Enhancement 9: Quality Gate Between Stages ──────────────────────

    @staticmethod
    def check_quality_gate(
        original: np.ndarray, reconstructed: np.ndarray, target_error: float
    ) -> bool:
        """
        Check if current quality is acceptable.
        Returns True if relative error <= target_error, allowing early exit.
        """
        denom = np.linalg.norm(original)
        if denom < 1e-30:
            return True
        error = float(np.linalg.norm(original - reconstructed) / denom)
        return error <= target_error

    # ── Existing Methods (enhanced) ──────────────────────────────────────

    def _get_method(self, method_type: str) -> Tuple[Optional[str], Optional[Any]]:
        # Priority-ordered: prefer engine built-in methods first, then find best match
        # For each type, known working method names are tried first.
        preferred = {
            "decomposition": "svd_compress",
            "spectral": "dct_spectral",
            "quantization": "block_int8",
            "entropy": "rans",
            "structural": "einsort",
        }
        pref = preferred.get(method_type)
        if pref and pref in self.engine._methods:
            return pref, self.engine._methods[pref]

        # Fallback: check category match (word-boundary safe)
        cat_map = {
            "decomposition": "decomposition",
            "spectral": "spectral",
            "quantization": "quant",
            "entropy": "entropy",
            "structural": "structural",
        }
        target_cat = cat_map.get(method_type, method_type)
        for name, inst in self.engine._methods.items():
            cat = getattr(inst, "category", "").lower()
            if target_cat in cat.split("_"):
                return name, inst
        # Last resort: try matching target in name with underscore boundaries
        for name, inst in self.engine._methods.items():
            name_lower = name.lower()
            if f"_{target_cat}_" in f"_{name_lower}_":
                return name, inst
        return None, None

    def _plan_from_config(
        self,
        tensor: np.ndarray,
        stage_configs: List[Dict[str, Any]],
        tensor_name: str = "",
    ) -> Optional[StackingPlan]:
        """
        Build a StackingPlan from a list of stage config dicts.

        Each *stage_config* must have at least a ``"method_type"`` key.
        Optional keys include ``"rank_frac"``, ``"keep_frac"``, ``"bits"``,
        ``"block_size"``, etc., which are passed as tuning hints.

        Returns None if no viable methods could be resolved.
        """
        plan = StackingPlan(tensor_name=tensor_name)
        residual = tensor.copy()
        n_stages = len(stage_configs)

        for i, config in enumerate(stage_configs):
            method_type = config["method_type"]

            # Try ModelIntelligence-based selection first
            best, candidates = self.select_method_for_stage(
                residual,
                method_type,
                previous_methods=[s.method_name for s in plan.stages],
            )

            if best is not None and best.get("method") is not None:
                method_name = best["method"]
                method_inst = self.engine._methods.get(method_name)
                if method_inst is not None:
                    tuned_params = self._tune_stage_params(
                        method_inst,
                        method_name,
                        residual,
                        10.0,
                        0.01 / max(n_stages, 1),
                    )
                    # Merge in config-level hints
                    for k, v in config.items():
                        if k != "method_type":
                            tuned_params.setdefault(k, v)

                    try:
                        compressed, meta = method_inst.compress(
                            residual, **tuned_params
                        )
                    except Exception:
                        compressed, meta = method_inst.compress(residual)
                        tuned_params = {}

                    try:
                        decompressed = method_inst.decompress(compressed, meta)
                        if decompressed.shape != residual.shape:
                            decompressed = decompressed.reshape(residual.shape)

                        stage_ratio = residual.nbytes / max(len(compressed), 1)
                        stage_error = float(
                            np.linalg.norm(residual.ravel() - decompressed.ravel())
                            / (np.linalg.norm(residual.ravel()) + 1e-30)
                        )

                        residual = residual - decompressed

                        tier = get_method_tier(
                            method_name, getattr(method_inst, "category", "")
                        )

                        plan.stages.append(
                            StackingStage(
                                method_name=method_name,
                                category=getattr(method_inst, "category", ""),
                                tier=tier,
                                params=tuned_params,
                                sub_ratio=stage_ratio,
                                sub_error=stage_error,
                                compressed_data=compressed,
                                metadata=meta,
                            )
                        )

                        plan.total_ratio = tensor.nbytes / max(
                            sum(
                                len(s.compressed_data)
                                for s in plan.stages
                                if s.compressed_data
                            )
                            + len(compressed),
                            1,
                        )

                        logger.debug(
                            "Plan stage %d (%s): ratio=%.2fx",
                            i + 1,
                            method_name,
                            stage_ratio,
                        )

                    except Exception as e:
                        logger.warning(
                            "Plan stage %d (%s) failed: %s", i + 1, method_name, e
                        )
                        continue

            if not plan.stages or len(plan.stages) <= i:
                # Fallback to pattern-based lookup
                method_name, method_inst = self._get_method(method_type)
                if method_name is None or method_inst is None:
                    logger.warning("No method found for type %s, skipping", method_type)
                    continue

                tuned_params = self._tune_stage_params(
                    method_inst,
                    method_name,
                    residual,
                    10.0,
                    0.01 / max(n_stages, 1),
                )
                for k, v in config.items():
                    if k != "method_type":
                        tuned_params.setdefault(k, v)

                try:
                    compressed, meta = method_inst.compress(residual, **tuned_params)
                except Exception:
                    compressed, meta = method_inst.compress(residual)
                    tuned_params = {}

                try:
                    decompressed = method_inst.decompress(compressed, meta)
                    if decompressed.shape != residual.shape:
                        decompressed = decompressed.reshape(residual.shape)

                    stage_ratio = residual.nbytes / max(len(compressed), 1)

                    residual = residual - decompressed

                    tier = get_method_tier(
                        method_name, getattr(method_inst, "category", "")
                    )

                    plan.stages.append(
                        StackingStage(
                            method_name=method_name,
                            category=getattr(method_inst, "category", ""),
                            tier=tier,
                            params=tuned_params,
                            sub_ratio=stage_ratio,
                            sub_error=0.0,
                            compressed_data=compressed,
                            metadata=meta,
                        )
                    )

                    plan.total_ratio = tensor.nbytes / max(
                        sum(
                            len(s.compressed_data)
                            for s in plan.stages
                            if s.compressed_data
                        ),
                        1,
                    )

                except Exception as e:
                    logger.warning(
                        "Plan stage %d (%s) failed: %s", i + 1, method_name, e
                    )
                    continue

        if not plan.stages:
            return None

        # Compute actual total error from residual
        plan.total_error = float(
            np.linalg.norm(residual.ravel()) / (np.linalg.norm(tensor.ravel()) + 1e-30)
        )

        return plan

    def plan_stacking(
        self,
        tensor: np.ndarray,
        tensor_name: str = "",
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        pattern_name: Optional[str] = None,
        variation: int = 0,
        use_dynamic_pattern: bool = True,
        use_lagrangian_ratios: bool = True,
        use_quality_gate: bool = True,
        use_error_shaping: bool = True,
    ) -> StackingPlan:
        """
        Build a multi-stage stacking plan for *tensor*.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to compress.
        tensor_name : str
            Name for logging / metadata.
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable reconstruction error.
        pattern_name : str, optional
            Override pattern selection.  One of "max_compression",
            "high_quality", "lossless_like".  If None, selected
            automatically via digital twin (or via *target_ratio* when
            *use_dynamic_pattern* is False).
        variation : int
            Variation index (0, 1, 2).  Higher values apply more
            aggressive parameter scaling to explore the trade-off space.
        use_dynamic_pattern : bool
            Use digital-twin-driven pattern selection (default True).
        use_lagrangian_ratios : bool
            Use Lagrangian-optimal sub-ratio allocation (default True).
        use_quality_gate : bool
            Enable early-exit quality gates between stages (default True).
        use_error_shaping : bool
            Enable residual whitening between stages (default True).

        Returns
        -------
        StackingPlan
        """
        # ── Dynamic pattern + stage count ──
        dt = self._profile_tensor(tensor, tensor_name)

        if use_dynamic_pattern:
            stages_config = self.design_optimal_pattern(tensor, tensor_name)
            n_stages = self.determine_stage_count(dt, target_ratio)
            stages_config = stages_config[:n_stages]
        elif pattern_name is not None:
            pattern = self.STACKING_PATTERNS.get(pattern_name)
            if pattern is None:
                pattern = self.STACKING_PATTERNS["high_quality"]
            stages_config = pattern["stages"]
            n_stages = len(stages_config)
        else:
            # ── Compression-first priority: try tier-ordered patterns ──
            # Start with pure compression (Tier 1-2), escalate only as needed
            compression_only_pats = self.get_compression_only_patterns()
            if target_ratio <= 200:
                # Low compression needed — pure compression without quant
                pat_name = (
                    "compression_only"
                    if "compression_only" in compression_only_pats
                    else "tier1_tier2"
                )
                pat = self.STACKING_PATTERNS.get(
                    pat_name, self.STACKING_PATTERNS["lossless_like"]
                )
            elif target_ratio <= 400:
                # Moderate compression — add entropy (still no quant)
                pat_name = (
                    "full_cascade"
                    if "full_cascade" in compression_only_pats
                    else "tier1_through_tier4"
                )
                pat = self.STACKING_PATTERNS.get(
                    pat_name, self.STACKING_PATTERNS["lossless_like"]
                )
            elif target_ratio <= 1200:
                # Higher compression — use high quality pattern with quant
                pat = self.STACKING_PATTERNS["high_quality"]
            else:
                # Extreme compression — use max compression pattern
                pat = self.STACKING_PATTERNS["max_compression"]
            stages_config = pat["stages"]
            n_stages = len(stages_config)

        if not stages_config:
            stages_config = self.STACKING_PATTERNS["high_quality"]["stages"]
            n_stages = len(stages_config)

        # ── Lagrangian sub-ratio allocation ──
        stage_method_types = [s["method_type"] for s in stages_config]

        if use_lagrangian_ratios:
            sub_ratios = self.lagrangian_allocate_sub_ratios(
                target_ratio,
                stage_method_types,
            )
        else:
            sub_ratios = [max(1.5, target_ratio ** (1.0 / n_stages))] * n_stages

        # ── Execute stages ──
        plan = StackingPlan(tensor_name=tensor_name)
        reconstructed = np.zeros_like(tensor)
        residual = tensor.copy()

        for i, stage_config in enumerate(stages_config):
            method_type = stage_config["method_type"]

            best, candidates = self.select_method_for_stage(
                residual,
                method_type,
                previous_methods=[s.method_name for s in plan.stages],
            )

            method_name: Optional[str] = None
            method_inst: Optional[Any] = None
            using_intelligence = False

            if best is not None and best.get("method") is not None:
                name_candidate = best["method"]
                inst_candidate = self.engine._methods.get(name_candidate)
                if inst_candidate is not None:
                    method_name = name_candidate
                    method_inst = inst_candidate
                    using_intelligence = True

            if method_name is None or method_inst is None:
                method_name, method_inst = self._get_method(method_type)

            if method_name is None or method_inst is None:
                logger.warning("No method found for type %s, skipping", method_type)
                continue

            sub_ratio = sub_ratios[i] if i < len(sub_ratios) else 2.0
            if variation > 0:
                sub_ratio *= 1.0 + 0.25 * variation
            stage_target_error = max_error / max(n_stages - i, 1)

            tuned_params = self._tune_stage_params(
                method_inst,
                method_name,
                residual,
                sub_ratio,
                stage_target_error,
            )

            config_params = stage_config.get("params", {})
            for k, v in config_params.items():
                tuned_params.setdefault(k, v)

            try:
                compressed, meta = method_inst.compress(residual, **tuned_params)
            except Exception:
                compressed, meta = method_inst.compress(residual)
                tuned_params = {}

            try:
                decompressed = method_inst.decompress(compressed, meta)

                if decompressed.shape != residual.shape:
                    decompressed = decompressed.reshape(residual.shape)

                stage_ratio = residual.nbytes / max(len(compressed), 1)

                reconstructed += decompressed

                tier = get_method_tier(
                    method_name, getattr(method_inst, "category", "")
                )

                plan.stages.append(
                    StackingStage(
                        method_name=method_name,
                        category=getattr(method_inst, "category", ""),
                        tier=tier,
                        params=tuned_params,
                        sub_ratio=stage_ratio,
                        sub_error=0.0,
                        compressed_data=compressed,
                        metadata=meta,
                    )
                )

                plan.total_ratio = tensor.nbytes / max(
                    sum(
                        len(s.compressed_data) for s in plan.stages if s.compressed_data
                    ),
                    1,
                )

                logger.debug(
                    "Stage %d (%s): ratio=%.2fx using_intelligence=%s",
                    i + 1,
                    method_name,
                    stage_ratio,
                    using_intelligence,
                )

                # ── Quality gate: exit early if error budget is met ──
                if use_quality_gate and self.check_quality_gate(
                    tensor, reconstructed, max_error
                ):
                    logger.debug(
                        "Quality gate PASSED after stage %d (%s) — "
                        "skipping remaining %d stages",
                        i + 1,
                        method_name,
                        n_stages - i - 1,
                    )
                    break

                # ── Error-shaped residual for next stage ──
                if use_error_shaping:
                    residual = self.compute_residual(tensor, reconstructed)
                else:
                    residual = tensor - reconstructed

            except Exception as e:
                logger.warning("Stage %d (%s) failed: %s", i + 1, method_name, e)
                continue

        # ── Compute actual total error (not additive) ──
        plan.total_error = float(
            np.linalg.norm(tensor.ravel() - reconstructed.ravel())
            / (np.linalg.norm(tensor.ravel()) + 1e-30)
        )

        return plan

    def _tune_stage_params(
        self,
        method_inst: Any,
        method_name: str,
        tensor: np.ndarray,
        target_ratio: float,
        target_error: float,
    ) -> Dict:
        default_params: Dict[str, Any] = {}

        name_lower = method_name.lower()
        cat_lower = getattr(method_inst, "category", "").lower()

        # Decomposition methods (SVD, low-rank, tensor train, CP)
        if any(
            k in name_lower for k in ("svd", "low_rank", "tensor_train", "cp_decomp")
        ):
            m = tensor.shape[0]
            n = max(1, np.prod(tensor.shape[1:]) if len(tensor.shape) > 1 else 1)
            k = int(m * n / (target_ratio * (m + n + 1)))
            k = max(1, min(k, min(m, n)))
            default_params["rank"] = k

        # Spectral methods (DCT, FFT, Fourier, wavelet)
        elif any(k in name_lower for k in ("dct", "fft", "fourier", "wavelet")):
            default_params["keep_ratio"] = 1.0 / max(target_ratio, 1.5)

        # Quantization methods
        elif any(
            k in name_lower or k in cat_lower
            for k in ("quant", "int", "int8", "int4", "sparsity", "delta")
        ):
            block_size = max(16, min(512, int(128 * target_ratio / 4)))
            block_size = max(block_size // 2 * 2, 2)
            default_params["block_size"] = block_size

            if "hadamard" in name_lower or "transform" in cat_lower:
                bits = max(1, min(8, int(32 / max(target_ratio, 1.0))))
                default_params["bits"] = bits
            elif "sparsity" in name_lower:
                default_params["group_size"] = 32
            elif "delta" in name_lower:
                default_params["block_size"] = block_size

        # Entropy methods (RANS, Zstd, Huffman) — no tuning needed
        elif any(k in name_lower for k in ("rans", "zstd", "huffman")):
            pass

        return default_params

    def execute_stacking(
        self, plan: StackingPlan, tensor: np.ndarray
    ) -> Tuple[bytes, dict]:
        buf = bytearray()
        import json

        buf += struct.pack("<I", len(plan.stages))
        buf += struct.pack("<d", plan.total_ratio)
        buf += struct.pack("<d", plan.total_error)

        for stage in plan.stages:
            if stage.compressed_data is None:
                continue

            name_bytes = stage.method_name.encode("utf-8")
            buf += struct.pack("<I", len(name_bytes))
            buf += name_bytes

            meta_json = json.dumps(
                {
                    k: v
                    for k, v in (stage.metadata or {}).items()
                    if isinstance(v, (str, int, float, bool, list, tuple))
                },
                default=str,
            ).encode("utf-8")
            buf += struct.pack("<I", len(meta_json))
            buf += meta_json

            buf += struct.pack(
                "<II", len(stage.compressed_data), int(stage.sub_ratio * 1000)
            )
            buf += stage.compressed_data

        metadata = {
            "method": "multiplicative_stacking",
            "n_stages": len(plan.stages),
            "stages": [s.method_name for s in plan.stages],
            "total_ratio": plan.total_ratio,
            "total_error": plan.total_error,
            "tensor_name": plan.tensor_name,
        }

        return bytes(buf), metadata

    def unstack(self, data: bytes, metadata: dict, original_shape: tuple) -> np.ndarray:
        import json

        pos = 0

        n_stages = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        pos += 8  # skip total_ratio
        pos += 8  # skip total_error

        result = np.zeros(original_shape, dtype=np.float32)

        for _ in range(n_stages):
            if pos >= len(data):
                break

            name_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            method_name = data[pos : pos + name_len].decode("utf-8")
            pos += name_len

            meta_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            stage_meta = json.loads(data[pos : pos + meta_len].decode("utf-8"))
            pos += meta_len

            if "original_shape" not in stage_meta:
                stage_meta["original_shape"] = list(original_shape)

            data_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            pos += 4  # skip sub_ratio

            stage_data = data[pos : pos + data_len]
            pos += data_len

            inst = self.engine._methods.get(method_name)
            if inst is not None:
                try:
                    decompressed = inst.decompress(stage_data, stage_meta)
                    if decompressed.shape != original_shape:
                        decompressed = decompressed.reshape(original_shape)
                    result += decompressed
                except Exception as e:
                    logger.warning("Failed to unstack stage %s: %s", method_name, e)

        return result.astype(np.float32)

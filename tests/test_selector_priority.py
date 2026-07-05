"""
Tests: DynamicIntelligenceSelector — Compression-First Priority
================================================================
Verifies the selector always prioritizes compression methods (decomposition,
spectral, structural) over quantization (Tier 5). Ensures the scoring formula
gives Tier 1 methods a significant advantage even when quantization methods
have a perfect profile match.

Key invariants enforced:
1. Tier 1 (decomp/spectral) gets tier_score 10.0; Tier 5 (quant) gets 0.3
2. Quantization match is capped at 1.0 (never exceeds Tier 1 baseline)
3. Explicit quantization penalty (0.5x) applied in _score_method()
4. select() sorts by tier first, then by score within each tier
5. rank_methods_for_tensor() sorts by tier first, then score
6. For low-rank tensors, decomposition methods appear before quantization
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.engine import (
    DynamicIntelligenceSelector,
    CompressionProfiler,
    TensorProfile,
    MethodTier,
    tier_score,
    get_tier,
)
from spectralstream.compression.engine.dynamic_selector2 import (
    TensorIntelligence,
    TensorIntelligenceAnalyzer,
    MethodPerformancePredictor,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_profile(
    shape: Tuple[int, ...],
    name: str = "test_tensor",
    effective_rank: float = 10.0,
    energy_concentration: float = 0.5,
    entropy: float = 5.0,
    std: float = 1.0,
    sparsity: float = 0.0,
    toeplitz: float = 0.0,
    circulant: float = 0.0,
    sensitivity: float = 0.5,
    outlier_ratio: float = 0.05,
    dynamic_range: float = 5.0,
    n_elements: int = 0,
    nbytes: int = 0,
) -> TensorProfile:
    """Build a synthetic TensorProfile with controllable properties."""
    if n_elements == 0:
        n_elements = int(np.prod(shape)) if shape else 1024
    if nbytes == 0:
        nbytes = n_elements * 4

    p = TensorProfile(
        name=name,
        shape=shape,
        dtype="float32",
        n_elements=n_elements,
        nbytes=nbytes,
        mean=0.0,
        std=std,
        min_val=-3.0 * std,
        max_val=3.0 * std,
        dynamic_range=dynamic_range,
        kurtosis=0.0,
        skewness=0.0,
        outlier_ratio=outlier_ratio,
        entropy_rate=entropy,
        energy_concentration=energy_concentration,
        spectral_decay_rate=1.0,
        effective_rank=effective_rank,
        spectral_entropy=0.3,
        toeplitz_score=toeplitz,
        circulant_score=circulant,
        block_diagonal_score=0.0,
        nm_sparsity_score=sparsity,
        sensitivity=sensitivity,
        tensor_type="weight",
    )
    return p


def _extract_candidate_tiers(
    candidates: List[Tuple[str, Any, Dict[str, Any]]],
) -> List[int]:
    """Extract tier numbers from a candidate list by looking up method category."""
    tiers: List[int] = []
    for name, inst, params in candidates:
        cat = "unknown"
        if inst is not None:
            cat = getattr(inst, "category", getattr(type(inst), "category", "unknown"))
        else:
            # Infer from name
            if name == "passthrough":
                cat = "lossless"
            elif any(k in name for k in ("svd", "low_rank", "tensor_train")):
                cat = "decomposition"
            elif any(k in name for k in ("dct", "fwht", "spectral")):
                cat = "spectral"
            elif any(
                k in name for k in ("block_int", "hadamard_int", "delta_int", "nf4")
            ):
                cat = "quantization"
            elif any(k in name for k in ("sparsity_int4",)):
                cat = "sparsity_quant"
        tier = get_tier(name, cat)
        tiers.append(int(tier))
    return tiers


# ── Tests ──────────────────────────────────────────────────────────────────


class TestTierScores:
    """Verify tier_score() values match compression-first priority."""

    def test_tier1_score_well_above_tier5(self):
        """Tier 1 score must be >= 10x Tier 5 score."""
        s1 = tier_score(MethodTier.TIER1_REAL_COMPRESSION)
        s5 = tier_score(MethodTier.TIER5_QUANTIZATION)
        assert s1 >= 10.0 * s5, f"Tier 1 score ({s1}) should be >= 10x Tier 5 ({s5})"
        print(f"  Tier 1 score: {s1}, Tier 5 score: {s5}, ratio: {s1 / s5:.1f}x")

    def test_tier_scores_decrease_monotonically(self):
        """Higher tier numbers must have strictly lower scores."""
        scores = [
            tier_score(MethodTier.TIER1_REAL_COMPRESSION),
            tier_score(MethodTier.TIER2_STRUCTURAL),
            tier_score(MethodTier.TIER3_ENTROPY),
            tier_score(MethodTier.TIER4_HYBRID),
            tier_score(MethodTier.TIER5_QUANTIZATION),
        ]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], (
                f"Tier {i + 1} score ({scores[i]}) must be > Tier {i + 2} ({scores[i + 1]})"
            )
        print(f"  Tier scores: {scores}")


class TestSelectorPriority:
    """Verify DynamicIntelligenceSelector prioritizes compression over quantization."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.selector = DynamicIntelligenceSelector()
        self.profiler = CompressionProfiler()
        self.selector.register_discoverable_batch()
        # Yield to test
        yield

    def test_selector_returns_tier1_first_for_low_rank(self):
        """Low-rank tensor (rank=3) — decomposition methods should dominate."""
        # Build a rank-3 matrix: A(32x3) @ B(3x32) → rank is ~3
        np.random.seed(42)
        A = np.random.randn(32, 3)
        B = np.random.randn(3, 32)
        tensor = (A @ B).astype(np.float32)

        profile = self.profiler.profile_tensor(tensor, "low_rank_test")
        candidates = self.selector.select(profile, error_budget=0.01, target_ratio=1000)

        tiers = _extract_candidate_tiers(candidates)
        print(f"\n  Low-rank tensor candidates (tiers): {tiers}")
        for i, (name, inst, params) in enumerate(candidates[:8]):
            tier = tiers[i] if i < len(tiers) else -1
            print(f"    #{i + 1}: {name:40s} → Tier {tier}")

        # Top candidate should be Tier 1 (decomposition)
        assert len(candidates) > 0, "Should return at least one candidate"
        assert tiers[0] == 1, (
            f"Top method is Tier {tiers[0]}, expected Tier 1 (decomposition)"
        )

        # No Tier 5 (quantization) in top 3 for a low-rank tensor
        top3_tiers = tiers[:3]
        assert 5 not in top3_tiers, (
            f"Quantization (Tier 5) found in top 3: {top3_tiers}"
        )

    def test_selector_cascading_for_extreme_ratio(self):
        """For extreme ratios (>100:1), cascade suggestions should be available."""
        np.random.seed(42)
        tensor = np.random.randn(16, 16).astype(np.float32) * 0.1
        profile = self.profiler.profile_tensor(tensor, "extreme_ratio_test")

        candidates = self.selector.select(profile, error_budget=0.01, target_ratio=5000)

        assert len(candidates) > 0, "Should return at least one candidate"
        tiers = _extract_candidate_tiers(candidates)
        print(f"\n  Extreme ratio candidates (tiers): {tiers}")
        for i, (name, inst, params) in enumerate(candidates[:8]):
            tier = tiers[i] if i < len(tiers) else -1
            print(f"    #{i + 1}: {name:40s} → Tier {tier}")

        # Tier 1 should still dominate top positions
        top_tiers = tiers[:5]
        assert 1 in top_tiers, f"Tier 1 method not found in top 5: {top_tiers}"


class TestScoreMethod:
    """Verify _score_method() gives Tier 1 methods a significant advantage."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.selector = DynamicIntelligenceSelector()
        yield

    def _check_tier_dominance(
        self,
        ti: TensorIntelligence,
        tier1_name: str,
        tier1_cat: str,
        tier5_name: str,
        tier5_cat: str,
    ):
        """Helper: verify Tier 1 method scores higher than Tier 5."""
        score1 = self.selector._score_method(
            tier1_name,
            MethodTier.TIER1_REAL_COMPRESSION,
            tier1_cat,
            ti,
            error_budget=0.01,
            target_ratio=100,
        )
        score5 = self.selector._score_method(
            tier5_name,
            MethodTier.TIER5_QUANTIZATION,
            tier5_cat,
            ti,
            error_budget=0.01,
            target_ratio=100,
        )
        assert score1 > score5, (
            f"Tier 1 ({tier1_name}: {score1:.2f}) should beat "
            f"Tier 5 ({tier5_name}: {score5:.2f})"
        )
        return score1, score5

    def test_tier1_decomp_beats_tier5_quant(self):
        """Decomposition (Tier 1) beats quantization (Tier 5) even on generic tensor."""
        ti = TensorIntelligence(
            name="test",
            shape=(64, 64),
            n_elements=4096,
            nbytes=16384,
            mean=0.0,
            std=1.0,
            min_val=-3.0,
            max_val=3.0,
            dynamic_range=6.0,
            kurtosis=0.0,
            skewness=0.0,
            outlier_ratio=0.05,
            entropy=5.0,
            energy_concentration=0.5,
            spectral_decay_rate=0.5,
            effective_rank=8,
            spectral_entropy=0.5,
            sparsity_ratio=0.0,
            sensitivity=0.5,
            tensor_type="weight",
        )
        s1, s5 = self._check_tier_dominance(
            ti, "svd_compress", "decomposition", "block_int8", "quantization"
        )
        print(f"  SVD (Tier 1): {s1:.2f}, BlockINT8 (Tier 5): {s5:.2f}")

    def test_tier1_spectral_beats_tier5_quant(self):
        """Spectral (Tier 1) beats quantization (Tier 5) on energy-concentrated tensor."""
        ti = TensorIntelligence(
            name="test",
            shape=(64, 64),
            n_elements=4096,
            nbytes=16384,
            mean=0.0,
            std=0.5,
            min_val=-2.0,
            max_val=2.0,
            dynamic_range=4.0,
            kurtosis=2.0,
            skewness=0.0,
            outlier_ratio=0.02,
            entropy=4.0,
            energy_concentration=0.85,
            spectral_decay_rate=2.0,
            effective_rank=12,
            spectral_entropy=0.3,
            sparsity_ratio=0.0,
            sensitivity=0.5,
            tensor_type="weight",
        )
        s1, s5 = self._check_tier_dominance(
            ti, "dct_spectral", "spectral", "hadamard_int8", "quantization"
        )
        print(f"  DCT (Tier 1): {s1:.2f}, HadamardINT8 (Tier 5): {s5:.2f}")

    def test_tier1_dominance_ratio_worst_case(self):
        """
        Worst-case scenario: Tier 1 with terrible match vs Tier 5 with perfect match.

        Even when:
        - The tensor has NO structure for decomposition (norm_rank > 0.3 → match=0.3)
        - Quantization has all bonuses active (hadamard + sparsity + delta + int8)

        Tier 1 should still score higher due to tier_score * penalty.
        """
        ti = TensorIntelligence(
            name="worst_case",
            shape=(64, 64),
            n_elements=4096,
            nbytes=16384,
            mean=0.0,
            std=2.0,
            min_val=-2.0,
            max_val=2.0,
            dynamic_range=1.5,  # Low range → delta bonus
            kurtosis=0.0,
            skewness=0.0,
            outlier_ratio=0.1,
            entropy=6.5,
            energy_concentration=0.75,  # High → hadamard bonus
            spectral_decay_rate=0.2,
            effective_rank=30,  # High rank → bad for decomp
            spectral_entropy=0.5,
            sparsity_ratio=0.5,  # High sparsity → sparsity bonus
            sensitivity=0.3,
            tensor_type="weight",
        )
        # Tier 1: decomposition method that doesn't match (high effective rank)
        # Tier 5: quant method that triggers ALL bonuses
        s1, s5 = self._check_tier_dominance(
            ti, "svd_compress", "decomposition", "hadamard_int8", "quantization"
        )
        ratio = s1 / max(s5, 1e-10)
        print(f"  SVD (Tier 1, bad match): {s1:.2f}")
        print(f"  HadamardINT8 (Tier 5, perfect match): {s5:.2f}")
        print(f"  Dominance ratio: {ratio:.2f}x")
        assert ratio >= 1.5, (
            f"Tier 1 must dominate Tier 5 even in worst case (ratio: {ratio:.2f}x)"
        )


class TestRankMethodsForTensor:
    """Verify rank_methods_for_tensor() sorts by tier first, then score."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.selector = DynamicIntelligenceSelector()
        self.profiler = CompressionProfiler()
        self.selector.register_discoverable_batch()
        yield

    def test_ranked_list_sorted_by_tier_then_score(self):
        """ranked list must have Tier 1 before Tier 5, then highest score first."""
        np.random.seed(42)
        tensor = np.random.randn(32, 32).astype(np.float32) * 0.1
        profile = self.profiler.profile_tensor(tensor, "ranked_test")

        ranked = self.selector.rank_methods_for_tensor(
            profile, error_budget=0.01, target_ratio=100
        )

        assert len(ranked) > 0, "Should return at least one ranked method"
        print("\n  Ranked methods (tier-first sorted):")
        for i, r in enumerate(ranked[:15]):
            print(
                f"    #{i + 1}: {r['method']:40s} Tier {r['tier']} Score {r['score']:.2f}"
            )

        # Verify tier-first ordering
        prev_tier = -1
        for r in ranked:
            tier = r["tier"]
            assert tier >= prev_tier, (
                f"Not sorted by tier first: saw tier {tier} after tier {prev_tier}"
            )
            prev_tier = tier

        # Verify score-descending within same tier
        for i in range(len(ranked) - 1):
            if ranked[i]["tier"] == ranked[i + 1]["tier"]:
                assert ranked[i]["score"] >= ranked[i + 1]["score"], (
                    f"Within tier {ranked[i]['tier']}, scores not descending: "
                    f"{ranked[i]['score']:.2f} then {ranked[i + 1]['score']:.2f}"
                )


class TestMethodPerformancePredictor:
    """Verify predictor gives compression methods higher ratios than quantization."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.predictor = MethodPerformancePredictor()
        yield

    def test_decomp_predicts_high_ratio_for_low_rank(self):
        """Decomposition should predict 200x ratio for very low rank."""
        ti = TensorIntelligence(
            shape=(64, 64),
            n_elements=4096,
            effective_rank=4,
        )
        ratio = self.predictor.predict_ratio(ti, "svd_compress", "decomposition")
        assert ratio >= 100, f"Expected >=100x for low rank, got {ratio}x"
        print(f"  Low-rank decomposition ratio: {ratio:.0f}x")

    def test_quant_predicts_modest_ratio(self):
        """Quantization should predict at most 8x for int4."""
        ti = TensorIntelligence(shape=(64, 64), n_elements=4096)
        ratio = self.predictor.predict_ratio(ti, "block_int4", "quantization")
        assert ratio <= 12, f"Expected <=12x for quantization, got {ratio}x"
        print(f"  Quantization ratio: {ratio:.0f}x")


class TestSelectSorting:
    """Verify select() sorts by tier then score within tier."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.selector = DynamicIntelligenceSelector()
        self.profiler = CompressionProfiler()
        self.selector.register_discoverable_batch()
        yield

    def test_select_candidates_sorted_by_tier_then_score(self):
        """
        select() must return candidates sorted by tier first (Tier 1 highest),
        then by combined score descending within the same tier.
        """
        np.random.seed(42)
        tensor = np.random.randn(64, 64).astype(np.float32)
        profile = self.profiler.profile_tensor(tensor, "sort_test")

        candidates = self.selector.select(profile, error_budget=0.01, target_ratio=100)

        assert len(candidates) > 0, "Should return at least one candidate"
        print("\n  Select candidates (tier-first sorted):")
        tiers = _extract_candidate_tiers(candidates)
        for i, (name, inst, params) in enumerate(candidates[:10]):
            tier = tiers[i] if i < len(tiers) else -1
            print(f"    #{i + 1}: {name:40s} → Tier {tier}")

        # Verify tier-first: tiers should be non-decreasing
        for i in range(len(tiers) - 1):
            assert tiers[i] <= tiers[i + 1], (
                f"Candidate list not sorted by tier: "
                f"Tier {tiers[i]} at position {i} before Tier {tiers[i + 1]} at {i + 1}"
            )


class TestQuantizationMatchCapping:
    """Verify quantization match is capped and penalty is applied."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.selector = DynamicIntelligenceSelector()
        yield

    def test_quant_match_capped_at_one(self):
        """_tensor_method_match should cap quantization at 1.0."""
        # Create a tensor with all properties that trigger quant bonuses
        ti = TensorIntelligence(
            shape=(64, 64),
            energy_concentration=0.8,  # Triggers hadamard bonus
            sparsity_ratio=0.5,  # Triggers sparsity bonus
            dynamic_range=1.5,  # Triggers delta bonus
        )

        # Test hadamard_int8 — all bonuses except delta
        match = self.selector._tensor_method_match("hadamard_int8", "quantization", ti)
        assert match <= 1.0, f"Quantization match ({match:.2f}) should be capped at 1.0"
        print(f"  HadamardINT8 quantization match (capped): {match:.2f}")

        # Test delta_int4
        match2 = self.selector._tensor_method_match("delta_int4", "quantization", ti)
        assert match2 <= 1.0, (
            f"Quantization match ({match2:.2f}) should be capped at 1.0"
        )
        print(f"  DeltaINT4 quantization match (capped): {match2:.2f}")

    def test_quant_penalty_applied(self):
        """Explicit quantization penalty (0.5x) must reduce score."""
        ti = TensorIntelligence(
            shape=(128, 128),
            n_elements=16384,
            energy_concentration=0.5,
            effective_rank=32,
            sensitivity=0.5,
        )

        score_quant = self.selector._score_method(
            "block_int8",
            MethodTier.TIER5_QUANTIZATION,
            "quantization",
            ti,
            error_budget=0.01,
            target_ratio=100,
        )
        score_decomp = self.selector._score_method(
            "svd_compress",
            MethodTier.TIER1_REAL_COMPRESSION,
            "decomposition",
            ti,
            error_budget=0.01,
            target_ratio=100,
        )

        print(f"  Quant score (with penalty): {score_quant:.4f}")
        print(f"  Decomp score: {score_decomp:.4f}")

        # The quant penalty + tier_score disparity should make decomp dominant
        assert score_decomp > score_quant * 2, (
            f"After quant penalty, decomp ({score_decomp:.2f}) should be "
            f"> 2x quant ({score_quant:.2f})"
        )


class TestTensorMethodMatch:
    """Verify _tensor_method_match() gives appropriate scores."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.selector = DynamicIntelligenceSelector()
        yield

    def test_decomp_match_for_low_rank(self):
        """Decomposition methods get high match for low-rank tensors."""
        ti = TensorIntelligence(
            shape=(64, 64),
            effective_rank=2,  # norm_rank = 2/64 ≈ 0.03 < 0.05
        )
        match = self.selector._tensor_method_match("svd_compress", "decomposition", ti)
        assert match == 3.0, f"Expected 3.0 for low rank, got {match}"
        print(f"  Low-rank decomposition match: {match:.1f}")

    def test_spectral_match_for_high_energy(self):
        """Spectral methods get high match for energy-concentrated tensors."""
        ti = TensorIntelligence(energy_concentration=0.92)
        match = self.selector._tensor_method_match("dct_spectral", "spectral", ti)
        assert match == 3.0, f"Expected 3.0 for high energy, got {match}"
        print(f"  High-energy spectral match: {match:.1f}")

    def test_quant_match_baseline_no_bonus(self):
        """Quantization methods start at 0.5 baseline (no bonus triggers)."""
        ti = TensorIntelligence()
        # Use a quant method that doesn't trigger any bonus conditions
        match = self.selector._tensor_method_match("nf4", "quantization", ti)
        assert match == 0.5, f"Expected baseline 0.5, got {match}"
        print(f"  Quantization baseline match (nf4): {match:.1f}")

    def test_quant_match_int8_gets_modest_bonus(self):
        """block_int8 gets 1.2x bonus but is still capped."""
        ti = TensorIntelligence()
        match = self.selector._tensor_method_match("block_int8", "quantization", ti)
        assert match == 0.6, f"Expected 0.6 (0.5 * 1.2), got {match}"
        print(f"  Quantization match (block_int8): {match:.1f}")

    def test_quant_match_capped_at_one(self):
        """Even with all bonuses, quantization match never exceeds 1.0."""
        ti = TensorIntelligence(
            energy_concentration=0.8,
            sparsity_ratio=0.5,
            dynamic_range=1.0,
        )
        # A method that would trigger all bonuses: hadamard + sparsity + delta + int8
        match = self.selector._tensor_method_match(
            "hadamard_sparsity_delta_int8", "quantization", ti
        )
        assert match <= 1.0, f"Quantization match ({match:.2f}) should be capped at 1.0"
        print(f"  Full-bonus quantization match (capped): {match:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

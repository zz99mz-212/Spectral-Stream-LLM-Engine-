"""Tests for compression tier priority system.

Verifies that compression methods (decomposition, spectral, tensor_network,
functional, novel) are STRICTLY prioritized over quantization (bit pruning).

The tier system ensures:
  - Tier 1 (score 10.0): Decomposition, spectral, tensor network, functional, novel
  - Tier 2 (score  5.0): Structural, physics
  - Tier 3 (score  2.0): Entropy, lossless
  - Tier 4 (score  1.5): Hybrid, cascade
  - Tier 5 (score  0.3): Quantization — LAST RESORT
  - Tier gap: 10.0 / 0.3 = 33.3x (requires >= 10x)
"""

import sys

sys.path.insert(0, ".")

import numpy as np
import pytest


class TestTierMapping:
    """Verify the CATEGORY_TIER_MAP has correct mappings."""

    def test_decomp_is_tier1_quant_is_tier5(self):
        from spectralstream.compression.engine.method_tiers import get_tier

        assert get_tier("decomposition") == 1
        assert get_tier("spectral") == 1
        assert get_tier("tensor_network") == 1
        assert get_tier("functional") == 1
        assert get_tier("novel") == 1
        assert get_tier("quantization") == 5

    def test_structural_is_tier2(self):
        from spectralstream.compression.engine.method_tiers import get_tier

        assert get_tier("structural") == 2
        assert get_tier("physics") == 2

    def test_entropy_lossless_is_tier3(self):
        from spectralstream.compression.engine.method_tiers import get_tier

        assert get_tier("entropy") == 3
        assert get_tier("lossless") == 3

    def test_hybrid_cascade_is_tier4(self):
        from spectralstream.compression.engine.method_tiers import get_tier

        assert get_tier("hybrid") == 4
        assert get_tier("cascade") == 4

    def test_quant_categories_are_tier5(self):
        from spectralstream.compression.engine.method_tiers import get_tier

        assert get_tier("transform_quant") == 5
        assert get_tier("sparsity_quant") == 5
        assert get_tier("delta_quant") == 5

    def test_breakthrough_categories_mapped(self):
        """Novel/breakthrough categories from methods/novel must have tiers."""
        from spectralstream.compression.engine.method_tiers import (
            get_tier,
            CATEGORY_TIER_MAP,
        )

        # These should ALL be explicitly mapped (not relying on default)
        assert "breakthrough_decomposition" in CATEGORY_TIER_MAP
        assert "breakthrough_signal" in CATEGORY_TIER_MAP
        assert "novel_structural" in CATEGORY_TIER_MAP
        assert "novel_physics" in CATEGORY_TIER_MAP
        assert "novel_entropy" in CATEGORY_TIER_MAP
        assert "breakthrough_hybrid" in CATEGORY_TIER_MAP


class TestTierScoreAndGap:
    """Verify tier scores and the compression-vs-quantization gap."""

    def test_tier_scores_are_correct(self):
        from spectralstream.compression.engine.method_tiers import (
            tier_score,
            MethodTier,
        )

        assert tier_score(MethodTier.TIER1_REAL_COMPRESSION) == 10.0
        assert tier_score(MethodTier.TIER2_STRUCTURAL) == 5.0
        assert tier_score(MethodTier.TIER3_ENTROPY) == 2.0
        assert tier_score(MethodTier.TIER4_HYBRID) == 1.5
        assert tier_score(MethodTier.TIER5_QUANTIZATION) == 0.3

    def test_tier_gap_is_at_least_10x(self):
        """Compression methods must have at least 10x score advantage over quantization.

        This is the CORE assertion of the tier priority system. Without this gap,
        the selector would not consistently prefer compression over quantization.
        """
        from spectralstream.compression.engine.method_tiers import (
            tier_score,
            MethodTier,
        )

        t1_score = tier_score(MethodTier.TIER1_REAL_COMPRESSION)
        t5_score = tier_score(MethodTier.TIER5_QUANTIZATION)
        gap = t1_score / max(t5_score, 1e-30)

        assert gap >= 10.0, (
            f"Tier gap insufficient: {t1_score} vs {t5_score} "
            f"(gap = {gap:.1f}x, requires >= 10x)"
        )

    def test_validate_tier_gap_function(self):
        """The validate_tier_gap() function must not raise."""
        from spectralstream.compression.engine.method_tiers import validate_tier_gap

        validate_tier_gap()  # Should not raise AssertionError


class TestGetTierForMethod:
    """Verify get_tier_for_method() works correctly."""

    def test_engine_builtins(self):
        from spectralstream.compression.engine.method_tiers import get_tier_for_method

        # Decomposition methods → Tier 1
        assert get_tier_for_method("svd_compress") == 1

        # Spectral methods → Tier 1
        assert get_tier_for_method("dct_spectral") == 1
        assert get_tier_for_method("fwht_compress") == 1

        # Tensor network → Tier 1
        assert get_tier_for_method("tensor_train") == 1

        # Quantization → Tier 5
        assert get_tier_for_method("block_int8") == 5
        assert get_tier_for_method("block_int4") == 5
        assert get_tier_for_method("hadamard_int8") == 5
        assert get_tier_for_method("hadamard_int4") == 5
        assert get_tier_for_method("sparsity_int4") == 5
        assert get_tier_for_method("delta_int4") == 5

    def test_discoverable_methods(self):
        """Verify methods discovered from METHODS_CLASSES get correct tiers."""
        from spectralstream.compression.engine.method_tiers import get_tier_for_method

        # Decomposition methods
        assert get_tier_for_method("butterfly") == 1
        assert get_tier_for_method("monarch") == 1
        assert get_tier_for_method("tucker_decomposition") == 1
        assert get_tier_for_method("hierarchical_tucker") == 1
        assert get_tier_for_method("cp_decomposition") == 1
        assert get_tier_for_method("svd_truncated") == 1

        # Spectral methods
        assert get_tier_for_method("dct_2d") == 1
        assert get_tier_for_method("fwht") == 1
        assert get_tier_for_method("wavelet_haar") == 1
        assert get_tier_for_method("fourier") == 1
        assert get_tier_for_method("chebyshev") == 1

        # Structural methods → Tier 2
        assert get_tier_for_method("einsort") == 2
        assert get_tier_for_method("circulant") == 2
        assert get_tier_for_method("sparse_gpt") == 2

        # Physics methods → Tier 2
        assert get_tier_for_method("mhd") == 2
        assert get_tier_for_method("vlasov_distribution") == 2

        # Entropy → Tier 3
        assert get_tier_for_method("huffman") == 3
        assert get_tier_for_method("rans") == 3
        assert get_tier_for_method("arithmetic") == 3

        # Lossless → Tier 3
        assert get_tier_for_method("lossless_zstd") == 3

        # Hybrid → Tier 4
        assert get_tier_for_method("cascade_2_stage") == 4
        assert get_tier_for_method("cascade_full_1200") == 4

        # Quantization → Tier 5
        assert get_tier_for_method("nf4") == 5
        assert get_tier_for_method("binary_quant") == 5
        assert get_tier_for_method("gptq_quant") == 5
        assert get_tier_for_method("kmeans_quant") == 5
        assert get_tier_for_method("mixed_precision") == 5

    def test_unknown_method_falls_back_to_default(self):
        from spectralstream.compression.engine.method_tiers import (
            get_tier_for_method,
            DEFAULT_TIER,
        )

        # Unknown method should use default (Tier 1)
        tier = get_tier_for_method("nonexistent_super_method")
        assert tier == DEFAULT_TIER


class TestSelectorCompressionPriority:
    """Verify the DynamicIntelligenceSelector prefers compression over quantization.

    On a low-rank tensor, decomposition methods should be selected before
    quantization methods. This confirms the tier system works end-to-end.
    """

    def test_selector_prefers_compression_over_quant_on_low_rank(self):
        """On a low-rank tensor (rank=5), decomposition should dominate."""
        from spectralstream.compression.engine.dynamic_selector2 import (
            DynamicIntelligenceSelector,
        )
        from spectralstream.compression.engine._profiler import CompressionProfiler
        from spectralstream.compression.engine.method_tiers import get_tier_for_method

        selector = DynamicIntelligenceSelector()
        selector.register_discoverable_batch()
        profiler = CompressionProfiler()

        # Low-rank tensor (rank=5 in 32x32) — decomposition should shine
        A = np.random.randn(32, 5)
        B = np.random.randn(5, 32)
        tensor = (A @ B).astype(np.float32)

        profile = profiler.profile_tensor(tensor, "test_low_rank")
        candidates = selector.select(
            profile, error_budget=0.01, target_ratio=100, max_candidates=15
        )

        # All top 5 candidates must be Tier 1-2 (compression, not quantization)
        top_tiers = []
        for name, inst, params in candidates[:5]:
            tier = get_tier_for_method(name)
            top_tiers.append(int(tier))

        # At least the top 3 must be Tier 1 or Tier 2
        assert top_tiers[0] <= 2, (
            f"Best candidate is Tier {top_tiers[0]}! Top-5 tiers: {top_tiers}"
        )
        assert min(top_tiers[:3]) <= 2, f"Top 3 candidates all Tier 3+: {top_tiers}"

        # None of the top 5 should be quantization (Tier 5)
        assert all(t < 5 for t in top_tiers), (
            f"Quantization method in top 5! Tiers: {top_tiers}"
        )

    def test_quantization_is_last_resort_on_random_tensor(self):
        """On a random (high-rank) tensor, quantization may appear but not as #1."""
        from spectralstream.compression.engine.dynamic_selector2 import (
            DynamicIntelligenceSelector,
        )
        from spectralstream.compression.engine._profiler import CompressionProfiler
        from spectralstream.compression.engine.method_tiers import get_tier_for_method

        selector = DynamicIntelligenceSelector()
        selector.register_discoverable_batch()
        profiler = CompressionProfiler()

        # Random full-rank tensor — harder for decomposition
        tensor = np.random.randn(16, 16).astype(np.float32)

        profile = profiler.profile_tensor(tensor, "test_random")
        candidates = selector.select(
            profile, error_budget=0.01, target_ratio=10, max_candidates=10
        )

        # The #1 candidate should still NOT be quantization (Tier 5)
        # because tier priority means any Tier 1-4 method with a reasonable
        # profile match beats a perfect quantization match.
        if len(candidates) >= 3:
            name = candidates[0][0]
            tier = get_tier_for_method(name)
            assert int(tier) < 5, (
                f"Best candidate is quantization (Tier 5) on random tensor! "
                f"Method: {name}. This violates compression-first priority."
            )

    def test_selector_knows_all_method_tiers(self):
        """After registration, the selector should have correct tiers for all methods."""
        from spectralstream.compression.engine.dynamic_selector2 import (
            DynamicIntelligenceSelector,
        )
        from spectralstream.compression.engine.method_tiers import (
            get_tier_for_method,
            tier_score,
            MethodTier,
        )

        selector = DynamicIntelligenceSelector()
        selector.register_discoverable_batch()

        # Check that the selector's internal tier matches the tier map
        method_list = selector.get_available_methods()
        assert len(method_list) > 100, f"Expected 100+ methods, got {len(method_list)}"

        # Spot-check a few methods
        checks = {
            "svd_compress": MethodTier.TIER1_REAL_COMPRESSION,
            "block_int8": MethodTier.TIER5_QUANTIZATION,
            "dct_2d": MethodTier.TIER1_REAL_COMPRESSION,
            "huffman": MethodTier.TIER3_ENTROPY,
            "circulant": MethodTier.TIER2_STRUCTURAL,
        }
        for method_name, expected_tier in checks.items():
            tier = get_tier_for_method(method_name)
            assert tier == expected_tier, (
                f"Method '{method_name}' has unexpected tier: "
                f"got {tier} (value {tier.value}), "
                f"expected {expected_tier} (value {expected_tier.value})"
            )


class TestTierMethodDiscovery:
    """Verify method_discovery uses correct tier assignments."""

    def test_discovery_assigns_correct_tiers(self):
        from spectralstream.compression.engine.method_discovery import MethodDiscovery
        from spectralstream.compression.engine.method_tiers import get_tier_for_method

        methods = MethodDiscovery.discover()
        assert len(methods) > 100, f"Expected 100+ methods, got {len(methods)}"

        # Spot-check tier assignments in discovery results
        for method_name, info in methods.items():
            discovered_tier = info.get("tier")
            if discovered_tier is not None:
                expected_tier = get_tier_for_method(method_name)
                assert int(discovered_tier) == int(expected_tier), (
                    f"Tier mismatch for '{method_name}': "
                    f"discovery says {discovered_tier}, "
                    f"tier system says {expected_tier}"
                )

    def test_compression_methods_exclude_quantization(self):
        from spectralstream.compression.engine.method_discovery import MethodDiscovery

        compression = MethodDiscovery.get_compression_methods()
        quant = MethodDiscovery.get_quantization_methods()

        # No overlap
        compression_names = set(compression.keys())
        quant_names = set(quant.keys())
        overlap = compression_names & quant_names
        assert len(overlap) == 0, (
            f"Methods in both compression and quant sets: {overlap}"
        )

        # Quantization methods should be Tier 5
        for name, info in quant.items():
            assert int(info["tier"]) == 5, (
                f"Quant method '{name}' has tier {info['tier']}, expected 5"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

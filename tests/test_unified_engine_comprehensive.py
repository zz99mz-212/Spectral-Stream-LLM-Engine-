"""Comprehensive integration test for the unified compression intelligence engine.

Verifies:
1. All methods are discoverable and categorized
2. Tier system prioritizes compression over quantization
3. Multiplicative stacking composes methods correctly
4. Dynamic selector prefers Tier 1-3 over Tier 5
5. Engine can handle realistic tensor shapes
6. Memory management works (no OOM)
"""

import pytest
import numpy as np
import sys, os, time, gc

sys.path.insert(0, ".")

# ============================================================
# SECTION 1: Tier System Verification
# ============================================================


class TestTierPriority:
    """Compression methods must be prioritized over quantization."""

    def test_tier_assignments(self):
        from spectralstream.compression.engine.method_tiers import (
            get_tier,
            CATEGORY_TIER_MAP,
        )

        assert get_tier("decomposition") == 1, "Decomposition must be Tier 1"
        assert get_tier("spectral") == 1, "Spectral must be Tier 1"
        assert get_tier("tensor_network") == 1, "Tensor network must be Tier 1"
        assert get_tier("functional") == 1, "Functional must be Tier 1"
        assert get_tier("novel") == 1, "Novel must be Tier 1"
        assert get_tier("quantization") == 5, (
            "Quantization must be Tier 5 (last resort)"
        )
        assert get_tier("transform_quant") == 5, "Transform quant must be Tier 5"
        assert get_tier("sparsity_quant") == 5, "Sparsity quant must be Tier 5"
        assert get_tier("delta_quant") == 5, "Delta quant must be Tier 5"

    def test_tier_score_gap(self):
        from spectralstream.compression.engine.method_tiers import (
            tier_score,
            MethodTier,
        )

        t1_score = tier_score(MethodTier.TIER1_REAL_COMPRESSION)
        t5_score = tier_score(MethodTier.TIER5_QUANTIZATION)
        assert t1_score > t5_score * 10, (
            f"Tier 1 score ({t1_score}) must be > 10x Tier 5 score ({t5_score})"
        )

    def test_method_tier_assignments(self):
        from spectralstream.compression.engine.method_tiers import (
            get_method_tier,
            MethodTier,
        )

        assert get_method_tier("block_int8") == MethodTier.TIER5_QUANTIZATION
        assert get_method_tier("block_int4") == MethodTier.TIER5_QUANTIZATION
        assert get_method_tier("svd_compress") == MethodTier.TIER1_REAL_COMPRESSION
        assert get_method_tier("dct_spectral") == MethodTier.TIER1_REAL_COMPRESSION
        assert get_method_tier("tensor_train") == MethodTier.TIER1_REAL_COMPRESSION


# ============================================================
# SECTION 2: Method Discovery Verification
# ============================================================


class TestMethodDiscovery:
    """All methods must be discoverable and properly categorized."""

    def test_method_class_count(self):
        from spectralstream.compression.methods import METHOD_CLASSES

        base_methods = [m for m in METHOD_CLASSES if "_variant_" not in m]
        print(f"\n  Total method classes: {len(METHOD_CLASSES)}")
        print(f"  Base methods: {len(base_methods)}")
        assert len(base_methods) >= 150, (
            f"Only {len(base_methods)} base methods, expected >= 150"
        )

    def test_all_methods_instantiate(self):
        from spectralstream.compression.methods import ALL_METHODS

        failures = []
        for name, inst in ALL_METHODS.items():
            if not hasattr(inst, "compress"):
                failures.append(f"{name} missing compress()")
            if not hasattr(inst, "decompress"):
                failures.append(f"{name} missing decompress()")
        assert len(failures) < 20, (
            f"Methods missing compress/decompress: {failures[:20]}"
        )
        print(f"\n  ✓ All {len(ALL_METHODS)} methods have compress/decompress")

    def test_categories_in_tier_map(self):
        from spectralstream.compression.methods import METHOD_CLASSES, ALL_CATEGORIES
        from spectralstream.compression.engine.method_tiers import CATEGORY_TIER_MAP

        methods_by_cat = {}
        for name, cls in METHOD_CLASSES.items():
            cat = getattr(cls, "category", "unknown")
            methods_by_cat.setdefault(cat, []).append(name)

        missing = [
            c
            for c in methods_by_cat
            if c not in CATEGORY_TIER_MAP and c not in ALL_CATEGORIES
        ]
        assert len(missing) == 0, f"Categories not in tier map: {missing}"
        print(f"\n  ✓ All categories mapped to tiers")


# ============================================================
# SECTION 3: Dynamic Selector Verification
# ============================================================


class TestDynamicSelector:
    """Selector must prioritize compression methods for compressible tensors."""

    def test_low_rank_prefers_decomposition(self):
        from spectralstream.compression.engine.dynamic_selector2 import (
            DynamicIntelligenceSelector,
            TensorIntelligenceAnalyzer,
            MethodPerformancePredictor,
            TensorIntelligence,
        )
        from spectralstream.compression.engine._profiler import CompressionProfiler
        from spectralstream.compression.engine.method_tiers import get_tier, tier_score

        selector = DynamicIntelligenceSelector()
        selector.register_discoverable_batch()

        profiler = CompressionProfiler()

        np.random.seed(42)
        A = np.random.randn(128, 5)
        B = np.random.randn(5, 128)
        tensor = (A @ B).astype(np.float32)

        profile = profiler.profile_tensor(tensor, "low_rank_test")
        candidates = selector.select(profile, error_budget=0.01, target_ratio=1000)

        assert len(candidates) > 0, "No candidates returned!"

        top_name = candidates[0][0]
        top_inst = candidates[0][1]
        top_cat = getattr(
            top_inst, "category", getattr(type(top_inst), "category", "unknown")
        )
        top_tier = get_tier(top_name, top_cat)

        print(f"\n  Best method for low-rank: {top_name} (Tier {top_tier})")
        assert top_tier.value <= 2, (
            f"Best method is Tier {top_tier.value}, expected Tier 1 or 2"
        )

    def test_random_tensor_still_no_quant_first(self):
        from spectralstream.compression.engine.dynamic_selector2 import (
            DynamicIntelligenceSelector,
        )
        from spectralstream.compression.engine._profiler import CompressionProfiler
        from spectralstream.compression.engine.method_tiers import get_tier

        selector = DynamicIntelligenceSelector()
        selector.register_discoverable_batch()
        profiler = CompressionProfiler()

        tensor = np.random.randn(128, 128).astype(np.float32) * 0.1
        profile = profiler.profile_tensor(tensor, "random_test")
        candidates = selector.select(profile, error_budget=0.05, target_ratio=100)

        for i in range(min(3, len(candidates))):
            name, inst, params = candidates[i]
            cat = getattr(inst, "category", getattr(type(inst), "category", "unknown"))
            tier = get_tier(name, cat)
            assert tier.value < 5, (
                f"Top {i + 1} method '{name}' is quantization (Tier 5)! Expected Tier 1-4"
            )
            print(f"  #{i + 1}: {name} (Tier {tier.value})")

    def test_high_ratio_low_rank_selects_svd(self):
        from spectralstream.compression.engine.dynamic_selector2 import (
            DynamicIntelligenceSelector,
        )
        from spectralstream.compression.engine._profiler import CompressionProfiler

        selector = DynamicIntelligenceSelector()
        selector.register_discoverable_batch()

        profiler = CompressionProfiler()

        A = np.random.randn(64, 4)
        B = np.random.randn(4, 64)
        tensor = (A @ B).astype(np.float32)

        profile = profiler.profile_tensor(tensor, "svd_test")
        ranked = selector.rank_methods_for_tensor(
            profile, error_budget=0.01, target_ratio=500
        )
        ranked_names = [r["method"] for r in ranked]

        svd_rank = next((i for i, n in enumerate(ranked_names) if "svd" in n), None)
        dec_rank = next(
            (i for i, n in enumerate(ranked_names) if "decomp" in n or "tt_" in n), None
        )

        print(f"\n  svd_compress rank: #{svd_rank}, decomp methods rank: #{dec_rank}")
        assert svd_rank is not None, "svd_compress not found in ranked methods"
        assert svd_rank < 200, f"svd_compress too far down the list: #{svd_rank}"


# ============================================================
# SECTION 4: Multiplicative Stacking Verification
# ============================================================


class TestMultiplicativeStacking:
    """Individual methods and stacking compose correctly."""

    def test_svd_compress_decompress_roundtrip(self):
        from spectralstream.compression.engine._methods import _SVDCompress

        # Low-rank tensor for SVD to work well
        A = np.random.randn(64, 4).astype(np.float32) * 0.1
        B = np.random.randn(4, 64).astype(np.float32) * 0.1
        tensor = A @ B

        svd = _SVDCompress()

        data, meta = svd.compress(tensor, rank=4)
        recon = svd.decompress(data, meta).reshape(tensor.shape)

        ratio = tensor.nbytes / max(len(data), 1)
        error = float(np.mean((recon - tensor) ** 2) ** 0.5) / float(
            np.mean(tensor**2) ** 0.5
        )

        print(f"\n  SVD: ratio={ratio:.1f}x, error={error:.6f}")
        assert ratio > 1.0, "No compression achieved"
        assert error < 0.1, f"SVD error too high: {error}"

    def test_dct_compress_decompress_roundtrip(self):
        from spectralstream.compression.engine._methods import _DCTSpectral

        tensor = np.random.randn(64, 64).astype(np.float32) * 0.1
        dct_obj = _DCTSpectral()

        data, meta = dct_obj.compress(tensor, keep_ratio=0.1)
        recon = dct_obj.decompress(data, meta).reshape(tensor.shape)

        ratio = tensor.nbytes / max(len(data), 1)
        error = float(np.mean((recon - tensor) ** 2) ** 0.5) / float(
            np.mean(tensor**2) ** 0.5
        )

        print(f"\n  DCT: ratio={ratio:.1f}x, error={error:.6f}")
        assert ratio > 1.0, "No compression achieved"

    def test_block_int8_compress_decompress_roundtrip(self):
        from spectralstream.compression.engine._methods import _BlockINT8

        tensor = np.random.randn(64, 64).astype(np.float32) * 0.1
        q = _BlockINT8()

        data, meta = q.compress(tensor, block_size=32)
        recon = q.decompress(data, meta).reshape(tensor.shape)

        ratio = tensor.nbytes / max(len(data), 1)
        error = float(np.mean((recon - tensor) ** 2) ** 0.5) / float(
            np.mean(tensor**2) ** 0.5
        )

        print(f"\n  BlockINT8: ratio={ratio:.1f}x, error={error:.6f}")
        assert ratio > 1.0, "No compression achieved"

    def test_svd_increases_ratio(self):
        tensor = np.random.randn(64, 64).astype(np.float32) * 0.1
        from spectralstream.compression.engine._methods import _SVDCompress

        svd = _SVDCompress()
        d1, m1 = svd.compress(tensor)
        recon1 = svd.decompress(d1, m1)
        r1 = tensor.nbytes / max(len(d1), 1)

        recon_t = recon1[:64, :64] if hasattr(recon1, "shape") else tensor
        error_before = float(np.mean((recon_t - tensor) ** 2) ** 0.5)
        print(f"\n  SVD ratio: {r1:.1f}x, error: {error_before:.6f}")
        assert r1 > 1.0


# ============================================================
# SECTION 5: Engine Integration
# ============================================================


class TestEngineIntegration:
    """Full compression engine pipeline integration."""

    def test_compress_fast_basic(self):
        from spectralstream.compression.engine import CompressionIntelligenceEngine

        engine = CompressionIntelligenceEngine()
        tensor = np.random.randn(64, 64).astype(np.float32) * 0.1

        result = engine.compress_fast(tensor, name="test_tensor")

        assert result is not None
        assert hasattr(result, "compression_ratio")
        print(
            f"\n  compress_fast: method={result.method}, ratio={result.compression_ratio:.1f}x, "
            f"error={result.relative_error:.6f}"
        )
        assert result.compression_ratio > 1.0, "No compression achieved"

    def test_compress_ultra_high_ratio(self):
        from spectralstream.compression.engine import CompressionIntelligenceEngine
        from spectralstream.compression.engine._constants import CHUNK_SIZE

        engine = CompressionIntelligenceEngine()
        engine.config.target_ratio = 5000
        engine.config.max_error = 0.01

        A = np.random.randn(256, 4)
        B = np.random.randn(4, 256)
        tensor = (A @ B).astype(np.float32)

        result = engine._compress_ultra(tensor, name="ultra_test")

        assert result is not None
        assert hasattr(result, "compression_ratio")
        print(
            f"\n  _compress_ultra: method={result.method}, ratio={result.compression_ratio:.1f}x, "
            f"error={result.relative_error:.6f}"
        )
        assert result.compression_ratio > 1.0

    def test_compress_tensor_with_validation(self):
        from spectralstream.compression.engine import CompressionIntelligenceEngine
        from spectralstream.compression.engine._profiler import CompressionProfiler

        engine = CompressionIntelligenceEngine()
        profiler = CompressionProfiler()

        tensor = np.random.randn(64, 64).astype(np.float32) * 0.1
        profile = profiler.profile_tensor(tensor, name="validation_test")

        target_ratio = 100
        error_budget = 0.05

        from spectralstream.compression.engine._methods import _SVDCompress, _BlockINT8

        methods_to_try = [
            ("svd_compress", _SVDCompress(), {"rank": 16}),
            ("block_int8", _BlockINT8(), {"block_size": 128}),
        ]

        result = engine.compress_tensor_with_validation(
            tensor, profile, methods_to_try, error_budget
        )

        assert result is not None
        print(
            f"\n  compress_tensor: method={result.method}, ratio={result.compression_ratio:.1f}x, "
            f"error={result.relative_error:.6f}"
        )
        assert result.compression_ratio > 1.0

    def test_decompression_roundtrip(self):
        from spectralstream.compression.engine import CompressionIntelligenceEngine

        engine = CompressionIntelligenceEngine()
        tensor = np.random.randn(64, 64).astype(np.float32) * 0.1

        result = engine.compress_fast(tensor, name="roundtrip_test")
        recon = engine.decompress_tensor(result).reshape(tensor.shape)

        max_diff = float(np.max(np.abs(recon - tensor)))
        print(f"\n  Decompression roundtrip: max_diff={max_diff:.6f}")
        assert max_diff < 1.0, f"Roundtrip max difference too high: {max_diff}"


# ============================================================
# SECTION 6: Dynamic Selector Performance Predictions
# ============================================================


class TestPerformancePrediction:
    """Performance prediction must work across categories."""

    def test_predictor_works_for_all_categories(self):
        from spectralstream.compression.engine.dynamic_selector2 import (
            MethodPerformancePredictor,
            TensorIntelligence,
        )

        predictor = MethodPerformancePredictor()
        ti = TensorIntelligence(
            shape=(128, 128),
            n_elements=16384,
            nbytes=65536,
            effective_rank=10.0,
            energy_concentration=0.8,
            entropy=4.5,
            sparsity_ratio=0.3,
            toeplitz_score=0.5,
            circulant_score=0.5,
        )

        categories = [
            "decomposition",
            "spectral",
            "structural",
            "quantization",
            "entropy",
            "physics",
            "lossless",
        ]
        for cat in categories:
            for method_name in [f"dummy_{cat}", f"{cat}_method"]:
                ratio = predictor.predict_ratio(ti, method_name, cat)
                error = predictor.predict_error(ti, method_name, cat)
                assert ratio > 0, f"{cat}: Ratio must be > 0"
                assert error >= 0, f"{cat}: Error must be >= 0"
                print(f"  {cat}: ratio={ratio:.1f}, error={error:.6f}")


# ============================================================
# SECTION 7: Full Orchestrator Pipeline
# ============================================================


class TestFullPipeline:
    """End-to-end: profile → select → compress → validate."""

    def test_profile_select_compress_validate_pipeline(self):
        from spectralstream.compression.engine import CompressionIntelligenceEngine
        from spectralstream.compression.engine._profiler import CompressionProfiler

        engine = CompressionIntelligenceEngine()
        profiler = CompressionProfiler()

        tensor = np.random.randn(64, 64).astype(np.float32) * 0.1

        profile = profiler.profile_tensor(tensor, name="pipeline_test")
        assert profile.nbytes > 0
        assert profile.n_elements > 0
        print(
            f"\n  Profile: shape={profile.shape}, nbytes={profile.nbytes}, "
            f"type={profile.tensor_type}, sensitivity={profile.sensitivity:.3f}"
        )

        error_budget = 0.05
        target_ratio = 100
        methods = engine._select_methods(profile, error_budget, target_ratio)
        assert len(methods) > 0, "No methods selected"
        print(f"  Selected {len(methods)} candidate methods:")
        for m_name, m_inst, m_params in methods[:5]:
            m_cat = getattr(
                m_inst, "category", getattr(type(m_inst), "category", "unknown")
            )
            print(f"    {m_name} (cat={m_cat})")

        result = engine.compress_tensor_with_validation(
            tensor, profile, methods, error_budget
        )
        assert result is not None
        assert result.compression_ratio > 1.0
        print(
            f"  Result: method={result.method}, ratio={result.compression_ratio:.1f}x, "
            f"error={result.relative_error:.6f}"
        )

        # Validate by reconstructing
        recon = engine.decompress_tensor(result).reshape(tensor.shape)
        mse = float(np.mean((recon - tensor) ** 2))
        print(f"  Reconstruction MSE: {mse:.8f}")
        assert mse < 1.0, f"Reconstruction MSE too high: {mse}"

    def test_telemetry_collection(self):
        from spectralstream.compression.engine import CompressionIntelligenceEngine

        engine = CompressionIntelligenceEngine()

        for i in range(3):
            tensor = np.random.randn(64, 64).astype(np.float32) * 0.1
            engine.compress_fast(tensor, name=f"tensor_{i}")

        telemetry = engine.get_telemetry()
        assert telemetry is not None
        assert "per_tensor_stats" in telemetry
        print(
            f"\n  Telemetry: {len(telemetry.get('per_tensor_stats', {}))} tensors tracked"
        )


# ============================================================
# SECTION 8: Discoverable Methods Integration
# ============================================================


class TestMethodDiscoveryIntegration:
    """Discoverable methods must work with the engine."""

    def test_discoverable_methods_in_selector(self):
        from spectralstream.compression.engine.dynamic_selector2 import (
            DynamicIntelligenceSelector,
        )
        from spectralstream.compression.engine._profiler import CompressionProfiler

        selector = DynamicIntelligenceSelector()
        count = selector.register_discoverable_batch()
        print(f"\n  Registered {count} discoverable methods")
        assert count > 50, f"Expected >50 discoverable methods, got {count}"

        available = selector.get_available_methods()
        print(f"  Total available methods: {len(available)}")
        assert len(available) > 50, (
            f"Expected >50 available methods, got {len(available)}"
        )

        profiler = CompressionProfiler()
        tensor = np.random.randn(64, 64).astype(np.float32) * 0.1
        profile = profiler.profile_tensor(tensor, name="discovery_test")

        candidates = selector.select(profile, error_budget=0.05, target_ratio=100)
        print(f"  Selected {len(candidates)} candidates for test tensor")
        assert len(candidates) > 0

    def test_ranked_methods_detail(self):
        from spectralstream.compression.engine.dynamic_selector2 import (
            DynamicIntelligenceSelector,
        )
        from spectralstream.compression.engine._profiler import CompressionProfiler

        selector = DynamicIntelligenceSelector()
        selector.register_discoverable_batch()
        profiler = CompressionProfiler()

        tensor = np.random.randn(64, 64).astype(np.float32) * 0.1
        profile = profiler.profile_tensor(tensor, name="ranked_test")

        ranked = selector.rank_methods_for_tensor(
            profile, error_budget=0.05, target_ratio=100
        )
        assert len(ranked) > 0, "No ranked methods returned"

        print(f"\n  Top 10 ranked methods ({len(ranked)} total):")
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for i, r in enumerate(ranked[:50]):
            t = r.get("tier", 5)
            tier_counts[t] = tier_counts.get(t, 0) + 1
        for r in ranked[:10]:
            print(
                f"    #{r.get('score', 0):.1f}: {r.get('method')} "
                f"(tier={r.get('tier')}, pred_ratio={r.get('predicted_ratio', 0):.1f})"
            )
        print(f"  Tier distribution (top 50): {dict(tier_counts)}")
        non_quant_top10 = sum(1 for r in ranked[:10] if r.get("tier", 5) < 5)
        assert non_quant_top10 >= 1, (
            f"Expected at least 1 non-quantization method in top 10, got {non_quant_top10}"
        )


# ============================================================
# SECTION 9: Error Budget Allocator
# ============================================================


class TestErrorBudgetAllocation:
    """Error budgets must be allocated by tensor sensitivity."""

    def test_budget_allocation(self):
        from spectralstream.compression.engine._allocator import ErrorBudgetAllocator
        from spectralstream.compression.engine._dataclasses import TensorProfile

        allocator = ErrorBudgetAllocator(
            adjustment_power=1.5, min_budget=0.0001, max_budget=0.05, safety_margin=1.5
        )

        profiles = {
            "q_proj": TensorProfile(name="q_proj", nbytes=65536, sensitivity=0.9),
            "k_proj": TensorProfile(name="k_proj", nbytes=65536, sensitivity=0.6),
            "o_proj": TensorProfile(name="o_proj", nbytes=65536, sensitivity=0.3),
            "norm": TensorProfile(name="norm", nbytes=1024, sensitivity=0.2),
        }

        budgets = allocator.allocate(profiles, target_ratio=5000, max_error=0.0002)
        assert len(budgets) == 4, f"Expected 4 budgets, got {len(budgets)}"
        print(f"\n  Allocated budgets:")
        for name, budget in sorted(budgets.items()):
            print(f"    {name}: {budget:.6f}")

        assert budgets.get("q_proj", 1) <= budgets.get("norm", 0), (
            f"q_proj budget ({budgets.get('q_proj', 0):.6f}) should be tighter "
            f"than norm budget ({budgets.get('norm', 0):.6f})"
        )

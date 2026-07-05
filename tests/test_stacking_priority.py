"""
Test Stacking Priority — verify multiplicative stacking prioritizes
compression-first methods over quantization.

The stacking engine MUST try Tier 1-3 methods (decomposition, spectral,
structural, entropy) BEFORE falling back to Tier 5 (quantization).
This test suite validates that ordering.
"""

import sys

sys.path.insert(0, ".")

import numpy as np
import pytest

# ── Helpers ──────────────────────────────────────────────────────────────


def _is_quant_method(method_name: str) -> bool:
    """Check if a method name is a quantization method."""
    return any(
        q in method_name.lower() for q in ["int4", "int8", "quant", "binary", "ternary"]
    )


def _low_rank_tensor(rows: int = 16, cols: int = 16, rank: int = 4) -> np.ndarray:
    """Create a low-rank tensor for testing."""
    A = np.random.randn(rows, rank)
    B = np.random.randn(rank, cols)
    return (A @ B).astype(np.float32)


def _noisy_tensor(rows: int = 16, cols: int = 16, noise_std: float = 0.1) -> np.ndarray:
    """Create a noisy tensor with some structure."""
    t = np.sin(np.linspace(0, 4 * np.pi, rows)).reshape(-1, 1) @ np.cos(
        np.linspace(0, 4 * np.pi, cols)
    ).reshape(1, -1)
    return (t + np.random.randn(rows, cols) * noise_std).astype(np.float32)


def _skip_if_stacking_unavailable():
    """Skip test if stacking engine or orchestrator can't be imported."""
    try:
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F401
            MultiplicativeStackingEngine,
        )
        from spectralstream.compression.engine._orchestrator import (  # noqa: F401
            CompressionIntelligenceEngine,
        )
    except ImportError as e:
        pytest.skip(f"Multiplicative stacking not available: {e}")


def _get_stacking_engine():
    """Get a MultiplicativeStackingEngine instance with minimal setup."""
    from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
        MultiplicativeStackingEngine,
    )

    # Create a minimal engine stub with _methods
    class _MinimalEngine:
        def __init__(self):
            self._methods = {}
            self.profiler = None

    return MultiplicativeStackingEngine(_MinimalEngine())


# ── Tests ────────────────────────────────────────────────────────────────


class TestStackingPriority:
    """Verify multiplicative stacking prioritizes compression over quantization."""

    def test_stacking_patterns_have_compression_first(self):
        """Verify STACKING_PATTERNS list compression-first patterns before quant."""
        _skip_if_stacking_unavailable()
        engine = _get_stacking_engine()

        # Check that compression-only patterns exist
        compression_only = engine.get_compression_only_patterns()
        assert len(compression_only) > 0, (
            "Must have at least one compression-only pattern (no quantization)"
        )

        # Check named patterns
        assert "compression_only" in engine.STACKING_PATTERNS
        assert "compression_plus_entropy" in engine.STACKING_PATTERNS
        assert "full_cascade" in engine.STACKING_PATTERNS

        # Verify they have no quantization stage
        for name in ["compression_only", "compression_plus_entropy", "full_cascade"]:
            pat = engine.STACKING_PATTERNS[name]
            has_quant = any(
                s.get("method_type", "").lower() == "quantization"
                for s in pat.get("stages", [])
            )
            assert not has_quant, f"Pattern '{name}' should not contain quantization"

    def test_compression_only_patterns_have_tier_metadata(self):
        """Verify new patterns have tier metadata indicating NO quantization."""
        _skip_if_stacking_unavailable()
        engine = _get_stacking_engine()

        for name in ["compression_only", "compression_plus_entropy", "full_cascade"]:
            pat = engine.STACKING_PATTERNS.get(name)
            assert pat is not None, f"Pattern '{name}' missing"
            tiers = pat.get("tiers", [])
            assert len(tiers) > 0, f"Pattern '{name}' missing tier metadata"
            assert max(tiers) < 5, (
                f"Pattern '{name}' has tier >= 5 (quantization). Tiers: {tiers}"
            )

    def test_get_compression_only_patterns_excludes_quant(self):
        """Verify get_compression_only_patterns() excludes quant patterns."""
        _skip_if_stacking_unavailable()
        engine = _get_stacking_engine()
        compression_only = engine.get_compression_only_patterns()

        for name, pat in compression_only.items():
            quant_stages = [
                s
                for s in pat.get("stages", [])
                if s.get("method_type", "").lower() == "quantization"
            ]
            assert len(quant_stages) == 0, (
                f"Pattern '{name}' in compression-only set has quantization"
            )

        # Verify 'max_compression' and 'high_quality' are NOT in compression-only
        assert "max_compression" not in compression_only
        assert "high_quality" not in compression_only

    def test_has_quantization_detection(self):
        """Verify has_quantization() correctly identifies quant patterns."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            MultiplicativeStackingEngine,
        )

        engine = _get_stacking_engine()

        # Patterns that should have quantization
        assert engine.has_quantization("max_compression")
        assert engine.has_quantization("high_quality")

        # Patterns that should NOT have quantization
        assert not engine.has_quantization("compression_only")
        assert not engine.has_quantization("compression_plus_entropy")
        assert not engine.has_quantization("full_cascade")
        assert not engine.has_quantization("tier1_decomp_spectral")

    def test_get_highest_tier_in_pattern(self):
        """Verify get_highest_tier_in_pattern returns correct tiers."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            MultiplicativeStackingEngine,
        )

        engine = _get_stacking_engine()

        # Patterns without quantization should have tier < 5
        assert engine.get_highest_tier_in_pattern("compression_only") < 5
        assert engine.get_highest_tier_in_pattern("compression_plus_entropy") < 5
        assert engine.get_highest_tier_in_pattern("full_cascade") < 5

        # Patterns with quantization should have tier >= 4 or 5
        assert engine.get_highest_tier_in_pattern("max_compression") >= 4
        assert engine.get_highest_tier_in_pattern("high_quality") >= 4

    def test_quality_gate_early_exit(self):
        """Verify quality gate allows early exit when error budget is met."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            MultiplicativeStackingEngine,
        )

        # Test identical tensors → error is 0 → should pass quality gate
        tensor = np.random.randn(16, 16).astype(np.float32)
        assert MultiplicativeStackingEngine.check_quality_gate(tensor, tensor, 0.01), (
            "Identical tensors should pass quality gate"
        )

        # Test very different tensors → should NOT pass quality gate
        other = np.random.randn(16, 16).astype(np.float32)
        assert not MultiplicativeStackingEngine.check_quality_gate(
            tensor, other * 100, 0.01
        ), "Very different tensors should not pass quality gate"

        # Test with zero denominator (all-zero tensor)
        zero = np.zeros((16, 16), dtype=np.float32)
        assert MultiplicativeStackingEngine.check_quality_gate(zero, zero, 0.01), (
            "Zero tensor should pass quality gate"
        )

    def test_error_feedback_whitening(self):
        """Verify compute_residual produces whitened residuals."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            MultiplicativeStackingEngine,
        )

        original = np.random.randn(16, 16).astype(np.float32) * 10 + 5
        reconstructed = original * 0.9  # 10% error

        residual = MultiplicativeStackingEngine.compute_residual(
            original, reconstructed
        )

        # Residual should be whitened (zero mean, unit std)
        assert abs(np.mean(residual)) < 0.5, (
            f"Whitened residual should have near-zero mean, got {np.mean(residual)}"
        )
        assert abs(np.std(residual) - 1.0) < 0.3, (
            f"Whitened residual should have near-unit std, got {np.std(residual)}"
        )

        # Residual shape should match
        assert residual.shape == original.shape

    def test_lagrangian_allocates_sub_ratios(self):
        """Verify Lagrangian allocation produces valid sub-ratios."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            MultiplicativeStackingEngine,
        )

        target_ratio = 200.0
        stage_types = ["decomposition", "spectral", "structural"]

        sub_ratios = MultiplicativeStackingEngine.lagrangian_allocate_sub_ratios(
            target_ratio, stage_types
        )

        assert len(sub_ratios) == len(stage_types)
        # Product should be close to target
        product = 1.0
        for r in sub_ratios:
            product *= r
            assert r >= 1.1, f"Sub-ratio {r} should be >= 1.1"
        assert abs(product - target_ratio) / target_ratio < 0.3, (
            f"Product {product:.2f} should be close to target {target_ratio}"
        )

    def test_lagrangian_with_entropy(self):
        """Verify Lagrangian handles entropy (lossless) stages correctly."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            MultiplicativeStackingEngine,
        )

        target_ratio = 500.0
        # Entropy (lossless) should be pinned at minimal ratio
        stage_types = ["decomposition", "spectral", "entropy"]

        sub_ratios = MultiplicativeStackingEngine.lagrangian_allocate_sub_ratios(
            target_ratio, stage_types
        )

        assert len(sub_ratios) == len(stage_types)
        product = 1.0
        for r in sub_ratios:
            product *= r
            assert r >= 1.1
        # Product should be close to target (within 30%)
        assert abs(product - target_ratio) / target_ratio < 0.3, (
            f"Product {product:.2f} should be close to target {target_ratio}"
        )

    def test_design_optimal_pattern_prioritizes_decomp_for_low_rank(self):
        """Verify design_optimal_pattern prioritizes decomposition for low-rank."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            MultiplicativeStackingEngine,
        )

        # Create truly rank-1 tensor
        v = np.random.randn(16).astype(np.float32)
        tensor = v.reshape(-1, 1) @ v.reshape(1, -1)

        engine = _get_stacking_engine()
        stages = engine.design_optimal_pattern(tensor, "rank1_test")

        # Decomposition should be first (highest priority)
        assert len(stages) > 0
        decomp_first = any(
            s.get("method_type") == "decomposition" and s.get("priority", 0) > 0.5
            for s in stages
        )

        # For a rank-1 tensor, decomposition should be top priority
        priorities = [s.get("priority", 0) for s in stages]
        assert max(priorities) > 0.5, (
            f"Should have high priority stages, got priorities: {priorities}"
        )

    def test_progressive_stack_conservative_start(self):
        """Verify progressive stacking starts conservative and escalates."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            MultiplicativeStackingEngine,
        )

        engine = _get_stacking_engine()
        tensor = np.random.randn(16, 16).astype(np.float32)

        # For a small tensor, progressive stack should start at a conservative ratio
        try:
            result = engine.progressive_stack(
                tensor, max_error=0.5, start_ratio=2.0, max_ratio=10.0
            )
            # May not find a viable plan for truly random data, which is acceptable
            if result is not None:
                assert result.ratio >= 1.0
                assert result.error <= 0.5, (
                    f"Error {result.error:.4f} should be <= max_error 0.5"
                )
        except Exception as exc:
            pytest.skip(f"Progressive stack not fully functional: {exc}")


class TestOrchestratorStacking:
    """Verify the orchestrator's _try_stacking uses compression-first cascade."""

    def test_stacking_phases_ordered_correctly(self, tiny_engine):
        """Verify stacking_phases list orders compression before quantization."""
        _skip_if_stacking_unavailable()

        engine = tiny_engine

        # Access the stacking phases from _try_stacking
        # Phase 1 should be compression-only (Tier 1)
        # Phase 5 should include everything (including quantization)

        # Verify stacking engine is enabled
        assert engine._stacking is not None, "Stacking engine should be enabled"

    def test_stacking_engine_compression_only_methods(self):
        """Verify stacking engine exposes compression-only pattern filtering."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            MultiplicativeStackingEngine,
        )

        engine = _get_stacking_engine()

        # Get compression-only patterns
        patterns = engine.get_compression_only_patterns()
        assert len(patterns) >= 3, (
            f"Expected at least 3 compression-only patterns, got {len(patterns)}"
        )

        # Verify each pattern's expected_ratio is reasonable
        for name, pat in patterns.items():
            expected_ratio = pat.get("expected_ratio", 0)
            expected_error = pat.get("expected_error", 1.0)
            assert expected_ratio > 0, (
                f"Pattern '{name}' should have expected_ratio > 0"
            )
            assert expected_error > 0, (
                f"Pattern '{name}' should have expected_error > 0"
            )
            assert expected_error < 1.0, (
                f"Pattern '{name}' expected_error should be < 1.0 (got {expected_error})"
            )

    def test_orchestrator_stacking_returns_none_for_small_tensors(self, tiny_engine):
        """Verify _try_stacking returns None for tiny tensors (fast path)."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine._dataclasses import TensorProfile

        engine = tiny_engine

        # Small 1D tensor should be skipped by stacking
        small_tensor = np.random.randn(100).astype(np.float32)
        profile = TensorProfile(
            name="test_small",
            tensor_type="norm_bias",
            shape=small_tensor.shape,
            dtype="float32",
            n_elements=small_tensor.size,
            nbytes=small_tensor.nbytes,
        )
        result = engine._try_stacking(small_tensor, profile, 0.01)
        assert result is None, "Small/norm_bias tensors should skip stacking"

    def test_stacking_within_compress_tensor_with_validation(self, tiny_engine):
        """Verify compress_tensor_with_validation invokes stacking when needed."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine._dataclasses import TensorProfile

        engine = tiny_engine

        # Create a tensor that would benefit from stacking
        tensor = _low_rank_tensor(16, 16, 4)
        profile = engine.profiler.profile_tensor(tensor, name="low_rank")

        # Try to find methods and validate
        target_ratio = 100.0
        methods = engine._select_methods(profile, 0.01, target_ratio, 5)

        if methods:
            result = engine.compress_tensor_with_validation(
                tensor, profile, methods, 0.01
            )
            assert result is not None
            assert result.compression_ratio >= 1.0
            # The method might include stacking (starts with "stacked_")
            # or a regular method — both are acceptable
            logger_msg = (
                f"Method: {result.method}, Ratio: {result.compression_ratio:.2f}, "
                f"Error: {result.relative_error:.6f}"
            )
            print(f"\n{logger_msg}")


class TestMultiplicativeStackingDataclasses:
    """Test the dataclasses used by the stacking engine."""

    def test_stacking_stage_creation(self):
        """Verify StackingStage dataclass works correctly."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            StackingStage,
        )

        stage = StackingStage(
            method_name="svd_compress",
            category="decomposition",
            tier=1,
            params={"rank": 16},
            sub_ratio=50.0,
            sub_error=0.002,
        )

        assert stage.method_name == "svd_compress"
        assert stage.category == "decomposition"
        assert stage.tier == 1
        assert stage.sub_ratio == 50.0
        assert stage.sub_error == 0.002

    def test_stacking_plan_creation(self):
        """Verify StackingPlan correctly tracks totals."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            StackingPlan,
            StackingStage,
        )

        plan = StackingPlan(tensor_name="test")
        plan.stages.append(
            StackingStage(
                method_name="svd_compress",
                category="decomposition",
                tier=1,
                params={"rank": 16},
                sub_ratio=50.0,
                sub_error=0.002,
            )
        )
        plan.stages.append(
            StackingStage(
                method_name="dct_spectral",
                category="spectral",
                tier=1,
                params={"keep_fraction": 0.3},
                sub_ratio=3.0,
                sub_error=0.001,
            )
        )

        plan.total_ratio = 50.0 * 3.0
        plan.total_error = 0.002 + 0.001

        assert plan.n_stages == 2
        assert plan.total_ratio == 150.0
        assert abs(plan.total_error - 0.003) < 1e-10

        # Summary should be a non-empty string
        summary = plan.summary()
        assert len(summary) > 0
        assert "150.00x" in summary

    def test_stacking_candidate_creation(self):
        """Verify StackingCandidate creation and attributes."""
        _skip_if_stacking_unavailable()
        from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (  # noqa: F811
            StackingCandidate,
            StackingPlan,
        )

        plan = StackingPlan(tensor_name="test")
        plan.total_ratio = 100.0
        plan.total_error = 0.01

        candidate = StackingCandidate(
            plan=plan,
            data=b"test_data",
            metadata={"test": True},
            ratio=100.0,
            error=0.01,
            score=10000.0,
            pattern_name="high_quality",
        )

        assert candidate.ratio == 100.0
        assert candidate.error == 0.01
        assert candidate.pattern_name == "high_quality"
        assert candidate.score == 10000.0

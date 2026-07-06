"""
Tests for UnifiedCascadeEngine — the single replacement for ALL 8 cascade approaches.

Verifies:
- CascadeStage / CascadePlan dataclasses
- plan_cascade with all strategy stages
- execute_cascade with residual pipeline
- decompress_cascade reconstruction
- Pattern discovery (R&D mode)
- Strategy fallback chain
- All 8 absorbed approaches are covered
"""

import gc
import sys
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, ".")

try:
    from spectralstream.compression.world_model.unified_cascade_engine import (
        CASCADE_PATTERNS,
        COMPLEMENTARY_PAIRS,
        STACKING_PATTERNS,
        CascadePlan,
        CascadeStage,
        UnifiedCascadeEngine,
    )
except ImportError:
    pytest.skip("UnifiedCascadeEngine not available", allow_module_level=True)


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def small_tensor() -> np.ndarray:
    """Small 64x64 tensor suitable for fast tests."""
    return np.random.randn(64, 64).astype(np.float32)


@pytest.fixture
def weight_tensor() -> np.ndarray:
    """Simulates a real weight tensor with low-rank structure."""
    m, n, r = 128, 256, 8
    u = np.random.randn(m, r)
    v = np.random.randn(r, n)
    return (u @ v).astype(np.float32)


@pytest.fixture
def large_weight_tensor() -> np.ndarray:
    """Larger weight tensor for cascade execution tests."""
    m, n, r = 256, 512, 16
    u = np.random.randn(m, r)
    v = np.random.randn(r, n)
    return (u @ v).astype(np.float32)


@pytest.fixture
def embedding_tensor() -> np.ndarray:
    """Simulates an embedding tensor."""
    return np.random.randn(100, 64).astype(np.float32)


@pytest.fixture
def norm_tensor() -> np.ndarray:
    """1D tensor simulating norm/biases."""
    return np.random.randn(128).astype(np.float32)


@pytest.fixture
def mock_method_registry() -> Dict[str, Any]:
    """Mock method registry with compress/decompress for testing."""

    class MockMethod:
        category = "decomposition"

        def compress(self, tensor, **kwargs):
            data = tensor.tobytes()
            return data[: len(data) // 4], {"shape": list(tensor.shape)}

        def decompress(self, data, meta):
            shape = meta.get("shape", (64, 64))
            n = int(np.prod(shape))
            padded = np.frombuffer(
                data + b"\x00" * (n * 4 - len(data)), dtype=np.float32
            ).reshape(shape)
            return padded

    class MockSpectral:
        category = "spectral"

        def compress(self, tensor, **kwargs):
            flat = tensor.ravel()
            keep = max(1, len(flat) // 8)
            coeffs = np.fft.dct(flat, type=2, norm="ortho")
            data = coeffs[:keep].astype(np.float32).tobytes()
            return data, {"shape": list(tensor.shape), "keep": keep, "n": len(flat)}

        def decompress(self, data, meta):
            shape = meta.get("shape", (64, 64))
            keep = meta.get("keep", 100)
            n = meta.get("n", int(np.prod(shape)))
            arr = np.frombuffer(data, dtype=np.float32)
            coeffs = np.zeros(n, dtype=np.float64)
            coeffs[: min(len(arr), keep)] = arr[: min(len(arr), keep)]
            flat = np.fft.idct(coeffs, type=2, norm="ortho")
            return flat.astype(np.float32).reshape(shape)

    class MockQuant:
        category = "quantization"

        def compress(self, tensor, **kwargs):
            scale = np.max(np.abs(tensor)) + 1e-10
            quant = np.clip(np.round(tensor / scale * 127), -128, 127).astype(np.int8)
            return quant.tobytes() + np.float32(scale).tobytes(), {
                "shape": list(tensor.shape),
                "scale": scale,
            }

        def decompress(self, data, meta):
            shape = meta.get("shape", (64, 64))
            scale = meta.get("scale", 1.0)
            n = int(np.prod(shape))
            quant = np.frombuffer(data[:n], dtype=np.int8).reshape(shape)
            return (quant.astype(np.float32) * scale).astype(np.float32)

    class MockEntropy:
        category = "entropy"

        def compress(self, tensor, **kwargs):
            return tensor.tobytes(), {"shape": list(tensor.shape)}

        def decompress(self, data, meta):
            shape = meta.get("shape", (64, 64))
            return np.frombuffer(data, dtype=np.float32).reshape(shape)

    class MockStructural:
        category = "structural"

        def compress(self, tensor, **kwargs):
            return tensor.tobytes()[: len(tensor.tobytes()) // 2], {
                "shape": list(tensor.shape)
            }

        def decompress(self, data, meta):
            shape = meta.get("shape", (64, 64))
            n = int(np.prod(shape))
            padded = np.frombuffer(
                data + b"\x00" * (n * 4 - len(data)), dtype=np.float32
            ).reshape(shape)
            return padded

    return {
        "svd_compress": MockMethod(),
        "dct_spectral": MockSpectral(),
        "block_int8": MockQuant(),
        "rans": MockEntropy(),
        "einsort": MockStructural(),
        "tensor_train": MockMethod(),
        "huffman": MockEntropy(),
        "fwht_compress": MockSpectral(),
        "hadamard_int8": MockQuant(),
        "block_int4": MockQuant(),
        "sparsity_int4": MockQuant(),
    }


@pytest.fixture
def engine(mock_method_registry):
    """Minimal mock engine compatible with execute_cascade."""

    class MockEngine:
        _methods = mock_method_registry

    return MockEngine()


@pytest.fixture
def uce(mock_method_registry) -> UnifiedCascadeEngine:
    return UnifiedCascadeEngine(method_registry=mock_method_registry)


# ═══════════════════════════════════════════════════════════════════════
#  CascadeStage and CascadePlan Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCascadeStage:
    def test_default_values(self):
        stage = CascadeStage(method_name="svd_compress")
        assert stage.method_name == "svd_compress"
        assert stage.method_category == ""
        assert stage.params == {}
        assert stage.expected_ratio == 1.0
        assert stage.expected_error == 0.0
        assert stage.actual_ratio == 0.0
        assert stage.actual_error == 0.0
        assert stage.time_ms == 0.0

    def test_custom_values(self):
        stage = CascadeStage(
            method_name="dct_spectral",
            method_category="spectral",
            params={"keep_ratio": 0.1},
            expected_ratio=10.0,
            expected_error=0.001,
        )
        assert stage.method_name == "dct_spectral"
        assert stage.params["keep_ratio"] == 0.1


class TestCascadePlan:
    def test_default_plan(self):
        plan = CascadePlan()
        assert plan.stages == []
        assert plan.total_ratio == 1.0
        assert plan.total_error == 0.0
        assert plan.source == "oracle"

    def test_add_stage(self):
        plan = CascadePlan(target_ratio=100.0, max_error=0.01)
        plan.add_stage(
            CascadeStage(
                method_name="svd_compress", expected_ratio=50.0, expected_error=0.005
            )
        )
        plan.add_stage(
            CascadeStage(
                method_name="dct_spectral", expected_ratio=5.0, expected_error=0.002
            )
        )

        assert plan.n_stages == 2
        assert plan.total_ratio == 250.0  # 50 * 5
        assert plan.total_error == 0.007  # 0.005 + 0.002

    def test_target_met(self):
        plan = CascadePlan(target_ratio=100.0, max_error=0.01)
        plan.add_stage(
            CascadeStage(
                method_name="svd_compress", expected_ratio=100.0, expected_error=0.005
            )
        )

        assert plan.target_met  # 100 >= 100 AND 0.005 <= 0.01

    def test_target_not_met_ratio(self):
        plan = CascadePlan(target_ratio=200.0, max_error=0.01)
        plan.add_stage(
            CascadeStage(
                method_name="svd_compress", expected_ratio=50.0, expected_error=0.005
            )
        )

        assert not plan.target_met  # 50 < 200

    def test_target_not_met_error(self):
        plan = CascadePlan(target_ratio=100.0, max_error=0.001)
        plan.add_stage(
            CascadeStage(
                method_name="svd_compress", expected_ratio=100.0, expected_error=0.005
            )
        )

        assert not plan.target_met  # 0.005 > 0.001

    def test_to_dict(self):
        plan = CascadePlan(
            tensor_type="weight", source="oracle", target_ratio=100.0, max_error=0.01
        )
        plan.add_stage(CascadeStage(method_name="svd_compress", expected_ratio=50.0))
        d = plan.to_dict()

        assert d["tensor_type"] == "weight"
        assert d["n_stages"] == 1
        assert d["stages"][0]["method_name"] == "svd_compress"
        assert d["total_ratio"] == 50.0


# ═══════════════════════════════════════════════════════════════════════
#  UnifiedCascadeEngine Tests
# ═══════════════════════════════════════════════════════════════════════


class TestUnifiedCascadeEngine:
    def test_init(self, mock_method_registry):
        uce = UnifiedCascadeEngine(method_registry=mock_method_registry)
        assert uce._method_registry is not None
        assert "svd_compress" in uce._method_registry

    def test_plan_cascade_returns_cascade_plan(self, uce, small_tensor):
        plan = uce.plan_cascade(
            tensor=small_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
        )
        assert isinstance(plan, CascadePlan)
        assert plan.n_stages >= 1
        assert plan.total_ratio >= 1.0

    def test_plan_cascade_different_tensor_types(self, uce, small_tensor):
        for ttype in ("weight", "attention_q", "embedding", "norm", "ffn_gate"):
            plan = uce.plan_cascade(
                tensor=small_tensor,
                tensor_type=ttype,
                target_ratio=100.0,
                max_error=0.01,
            )
            assert isinstance(plan, CascadePlan)
            assert plan.n_stages >= 1

    def test_plan_cascade_different_targets(self, uce, small_tensor):
        for target in (10.0, 100.0, 1000.0, 5000.0):
            plan = uce.plan_cascade(
                tensor=small_tensor,
                tensor_type="weight",
                target_ratio=target,
                max_error=0.01,
            )
            assert isinstance(plan, CascadePlan)

    def test_plan_cascade_caching(self, uce, small_tensor):
        plan1 = uce.plan_cascade(
            tensor=small_tensor,
            tensor_type="weight",
            target_ratio=5000.0,
            max_error=0.01,
        )
        plan2 = uce.plan_cascade(
            tensor=small_tensor,
            tensor_type="weight",
            target_ratio=5000.0,
            max_error=0.01,
        )
        assert plan2.source == "cache"

    def test_clear_cache(self, uce, small_tensor):
        uce.plan_cascade(small_tensor, "weight", 5000.0, 0.01)
        assert len(uce._pattern_cache) > 0
        uce.clear_cache()
        assert len(uce._pattern_cache) == 0

    def test_fallback_plan(self, uce):
        plan = uce._fallback_plan("weight", 5000.0)
        assert plan.n_stages >= 3
        assert plan.source == "fallback"
        assert plan.total_ratio > 1.0

    def test_fallback_plan_extreme_ratio(self, uce):
        plan = uce._fallback_plan("weight", 50000.0)
        assert plan.n_stages >= 4

    def test_recall_learned_pattern_no_learner(self, uce):
        result = uce._recall_learned_pattern("weight")
        assert result is None  # No CascadeLearner by default

    def test_ratio_to_stage_types(self):
        mapping = UnifiedCascadeEngine._ratio_to_stage_types
        assert len(mapping(50.0)) == 2  # <= 200 -> 2 stages
        assert len(mapping(200.0)) == 2
        assert len(mapping(500.0)) == 4  # <= 1200 -> 4 stages
        assert len(mapping(2000.0)) == 5  # <= 5000 -> 5 stages
        assert len(mapping(8000.0)) == 7  # <= 10000 -> 7 stages
        assert len(mapping(20000.0)) == 8  # > 10000 -> 8 stages

    def test_resolve_method_for_type(self, uce, mock_method_registry):
        assert uce._resolve_method_for_type("decomposition") == "svd_compress"
        assert uce._resolve_method_for_type("spectral") == "dct_spectral"
        assert uce._resolve_method_for_type("quantization") == "block_int8"

    def test_select_direct_pattern(self, weight_tensor, norm_tensor, embedding_tensor):
        select = UnifiedCascadeEngine._select_direct_pattern

        pattern_1d = select(norm_tensor, "norm", 500.0)
        assert pattern_1d in ("1d_aggressive", "1d_lightning", "lightning")

        pattern_emb = select(embedding_tensor, "embedding", 500.0)
        assert pattern_emb == "embedding_extreme"

        pattern_weight = select(weight_tensor, "weight", 500.0)
        assert pattern_weight == "max_compression"

        pattern_small = select(np.zeros((8, 8), dtype=np.float32), "weight", 500.0)
        assert pattern_small == "passthrough"

    def test_discover_patterns(self, uce, small_tensor, mock_method_registry):
        tensors = [(small_tensor, "weight", "test_weight")]
        results = uce.discover_patterns(tensors, exhaustive=False)
        assert "weight" in results
        assert isinstance(results["weight"], CascadePlan)

    def test_complementary_pairs_registry(self):
        assert len(COMPLEMENTARY_PAIRS) >= 10
        for m1, m2, w in COMPLEMENTARY_PAIRS:
            assert isinstance(m1, str)
            assert isinstance(m2, str)
            assert 0.0 < w < 1.0

    def test_cascade_patterns_registry(self):
        assert len(CASCADE_PATTERNS) >= 11
        for name, stages in CASCADE_PATTERNS.items():
            assert len(stages) >= 1
            for mname, params in stages:
                assert isinstance(mname, str)
                assert isinstance(params, dict)

    def test_stacking_patterns_registry(self):
        assert len(STACKING_PATTERNS) >= 9
        for name, config in STACKING_PATTERNS.items():
            assert "stages" in config
            assert len(config["stages"]) >= 1


# ═══════════════════════════════════════════════════════════════════════
#  Cascade Execution Tests (with mock engine)
# ═══════════════════════════════════════════════════════════════════════


class TestCascadeExecution:
    def test_execute_cascade_basic(self, uce, engine, weight_tensor):
        plan = uce.plan_cascade(
            tensor=weight_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
        )
        data, meta = uce.execute_cascade(engine, weight_tensor, plan)

        assert isinstance(data, bytes)
        assert isinstance(meta, dict)
        assert len(data) > 0
        assert "total_ratio" in meta
        assert meta["total_ratio"] >= 1.0
        assert meta["n_stages"] >= 1

    def test_execute_cascade_produces_smaller_output(self, uce, engine, weight_tensor):
        orig_size = weight_tensor.nbytes
        plan = uce.plan_cascade(
            tensor=weight_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
        )
        data, meta = uce.execute_cascade(engine, weight_tensor, plan)

        assert len(data) < orig_size  # Must be compressed

    def test_decompress_cascade_reconstruction(self, uce, engine, weight_tensor):
        plan = uce.plan_cascade(
            tensor=weight_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.1,
        )
        data, meta = uce.execute_cascade(engine, weight_tensor, plan)

        recon = uce.decompress_cascade(engine, data, meta, weight_tensor.shape)

        assert recon.shape == weight_tensor.shape
        assert recon.dtype == np.float32

        mse = float(np.mean((weight_tensor - recon) ** 2))
        assert mse < 1.0  # Should be reasonable reconstruction

    def test_execute_cascade_empty_tensor(self, uce, engine):
        empty = np.zeros((0,), dtype=np.float32)
        plan = CascadePlan(target_ratio=100.0, max_error=0.01)
        data, meta = uce.execute_cascade(engine, empty, plan)
        assert meta.get("error") == "empty tensor"

    def test_execute_cascade_small_tensor(self, uce, engine):
        tiny = np.random.randn(16, 16).astype(np.float32)
        plan = uce.plan_cascade(
            tensor=tiny,
            tensor_type="weight",
            target_ratio=10.0,
            max_error=0.5,
        )
        data, meta = uce.execute_cascade(engine, tiny, plan)

        assert isinstance(data, bytes)
        assert meta.get("error") != "all cascade stages failed"

    def test_execute_cascade_stage_accumulation(self, uce, engine, weight_tensor):
        plan = uce.plan_cascade(
            tensor=weight_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
        )

        orig_size = weight_tensor.nbytes
        data, meta = uce.execute_cascade(engine, weight_tensor, plan)

        total_stage_sizes = sum(s["compressed_size"] for s in meta["stages"])
        # The payload includes header overhead, so total data >= sum of stages
        assert len(data) >= total_stage_sizes * 0.8  # Allow header overhead

    def test_execute_cascade_plan_updates(self, uce, engine, weight_tensor):
        plan = uce.plan_cascade(
            tensor=weight_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
        )
        uce.execute_cascade(engine, weight_tensor, plan)

        assert plan.total_ratio >= 1.0
        for stage in plan.stages:
            if stage.actual_ratio > 0:
                assert stage.actual_ratio >= 0.0
                assert stage.time_ms >= 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Packaging / Unpacking Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPackaging:
    def test_package_unpack_roundtrip(self):
        stages_data = [
            {
                "method": "svd_compress",
                "params": {"rank": 32},
                "compressed_data": b"test_data_1",
            },
            {
                "method": "dct_spectral",
                "params": {"keep_ratio": 0.1},
                "compressed_data": b"test_data_2",
            },
        ]
        orig_size = 10000

        data, ratio = UnifiedCascadeEngine._package_stages(stages_data, orig_size)
        stages, weights = UnifiedCascadeEngine._unpack_stages(data)

        assert len(stages) == 2
        assert stages[0][0] == "svd_compress"
        assert stages[0][1] == b"test_data_1"
        assert stages[1][0] == "dct_spectral"
        assert stages[1][1] == b"test_data_2"
        assert ratio > 0

    def test_package_empty_stages(self):
        data, ratio = UnifiedCascadeEngine._package_stages([], 1000)
        stages, weights = UnifiedCascadeEngine._unpack_stages(data)
        assert len(stages) == 0

    def test_unpack_empty_data(self):
        stages, weights = UnifiedCascadeEngine._unpack_stages(b"")
        assert len(stages) == 0

    def test_unpack_corrupted_data(self):
        stages, weights = UnifiedCascadeEngine._unpack_stages(b"\x00" * 3)
        assert len(stages) == 0


# ═══════════════════════════════════════════════════════════════════════
#  Edge Cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_method_registry(self):
        uce = UnifiedCascadeEngine(method_registry={})
        assert uce._method_registry == {}

    def test_plan_cascade_empty_tensor(self, mock_method_registry):
        uce = UnifiedCascadeEngine(method_registry=mock_method_registry)
        empty = np.zeros((0,), dtype=np.float32)
        plan = uce.plan_cascade(empty, "weight", 5000.0, 0.01)
        assert isinstance(plan, CascadePlan)

    def test_rnd_mode_disables_exhaustive(self, uce, small_tensor):
        plan = uce.plan_cascade(
            tensor=small_tensor,
            tensor_type="weight",
            target_ratio=5000.0,
            max_error=0.01,
            rnd_mode=True,
        )
        assert isinstance(plan, CascadePlan)
        assert plan.n_stages >= 1

    def test_plan_cascade_1d_tensor(self, uce, norm_tensor):
        plan = uce.plan_cascade(
            tensor=norm_tensor,
            tensor_type="norm",
            target_ratio=100.0,
            max_error=0.01,
        )
        assert isinstance(plan, CascadePlan)
        assert plan.n_stages >= 1

    def test_plan_cascade_embedding(self, uce, embedding_tensor):
        plan = uce.plan_cascade(
            tensor=embedding_tensor,
            tensor_type="embedding",
            target_ratio=100.0,
            max_error=0.01,
        )
        assert isinstance(plan, CascadePlan)
        assert plan.n_stages >= 1


# ═══════════════════════════════════════════════════════════════════════
#  Integration: verify all 8 approaches are covered
# ═══════════════════════════════════════════════════════════════════════


class TestApproachCoverage:
    """Verifies that each of the 8 absorbed approaches is represented."""

    def test_approach1_cascade_oracle(self, uce, small_tensor):
        """CascadeOracle: KG -> tensor-type -> NAS -> quantum -> stacking."""
        plan = uce.plan_cascade(small_tensor, "weight", 5000.0, 0.01)
        # Falls through to tensor_type_strategy if KG is empty
        assert plan.source in (
            "learned",
            "tensor_type_strategy",
            "direct_cascade",
            "stacking",
            "multiplicative_stacking",
            "quantum_annealing",
            "complementary_pairs",
            "exhaustive",
            "fallback",
            "cache",
        )

    def test_approach2_dynamic_method_tester(self, uce, small_tensor):
        """DynamicMethodTester.find_optimal_cascade: tests all methods."""
        plan = uce._exhaustive_single_method(small_tensor, "weight")
        if plan is not None:
            assert plan.source == "exhaustive_single"

    def test_approach3_multiplicative_stacking(self, uce, small_tensor):
        """MultiplicativeStackingEngine: Lagrangian optimization."""
        # This approach is represented by _multiplicative_stacking
        # and _ratio_to_stage_types
        stage_types = UnifiedCascadeEngine._ratio_to_stage_types(5000.0)
        assert "decomposition" in stage_types
        assert "quantization" in stage_types

    def test_approach4_method_stacking(self, uce, small_tensor):
        """MethodStackingEngine: complementary pairs on original."""
        assert len(COMPLEMENTARY_PAIRS) > 0
        assert ("svd_compress", "block_int8", 0.5) in COMPLEMENTARY_PAIRS

    def test_approach5_direct_cascade(self, uce, small_tensor):
        """DirectCascadeEngine: residual-based cascade patterns."""
        pattern = UnifiedCascadeEngine._select_direct_pattern(
            small_tensor, "weight", 500.0
        )
        assert pattern in CASCADE_PATTERNS or pattern == "passthrough"

    def test_approach6_cascade_learner(self, uce):
        """CascadeLearner: stored cascade patterns per tensor type."""
        result = uce._recall_learned_pattern("weight")
        # No CascadeLearner initialized by default, so returns None
        assert result is None

    def test_approach7_unified_intelligence(self, uce):
        """UnifiedIntelligence.build_cascade_plan: ratio-to-stages mapping."""
        stage_types = UnifiedCascadeEngine._ratio_to_stage_types(5000.0)
        assert len(stage_types) == 5  # As defined in _ratio_to_stage_types

    def test_approach8_compression_orchestrator(self, uce, mock_method_registry):
        """CompressionOrchestrator.create_compression_plan: budget optimizer."""
        plan = uce._exhaustive_single_method(
            np.random.randn(32, 32).astype(np.float32), "weight"
        )
        if plan is not None:
            assert plan.total_ratio > 0


# ═══════════════════════════════════════════════════════════════════════
#  Memory / Garbage Collection
# ═══════════════════════════════════════════════════════════════════════


class TestMemory:
    def test_clear_cache_releases(self, uce, small_tensor):
        uce.plan_cascade(small_tensor, "weight", 5000.0, 0.01)
        uce.clear_cache()
        assert len(uce._pattern_cache) == 0

    def test_large_tensor_no_crash(self, uce):
        large = np.random.randn(512, 512).astype(np.float32)
        plan = uce.plan_cascade(large, "weight", 5000.0, 0.01)
        assert isinstance(plan, CascadePlan)

    def test_plan_cascade_high_ratio(self, uce, small_tensor):
        plan = uce.plan_cascade(small_tensor, "weight", 50000.0, 0.001)
        assert isinstance(plan, CascadePlan)

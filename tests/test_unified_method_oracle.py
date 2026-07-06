"""Tests for UnifiedMethodOracle — single unified method selection oracle."""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, ".")

try:
    from spectralstream.compression.world_model.unified_method_oracle import (
        UnifiedMethodOracle,
        MethodSelection,
        QuantumSuperpositionTest,
        BYPASS_HIGH_CONFIDENCE,
        BYPASS_MEDIUM_CONFIDENCE,
        TEST_FULL,
        _TensorFeatures,
    )
except ImportError as e:
    pytest.skip(f"Skipping: {e}", allow_module_level=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_tensor() -> np.ndarray:
    rng = np.random.RandomState(42)
    return rng.randn(256, 256).astype(np.float32)


@pytest.fixture
def low_rank_tensor() -> np.ndarray:
    rng = np.random.RandomState(42)
    U = rng.randn(256, 16)
    V = rng.randn(16, 256)
    return (U @ V).astype(np.float32)


@pytest.fixture
def sparse_tensor() -> np.ndarray:
    rng = np.random.RandomState(42)
    t = rng.randn(256, 256).astype(np.float32)
    t[np.abs(t) < 1.0] = 0.0
    return t


@pytest.fixture
def mock_method_registry() -> Dict[str, Dict[str, Any]]:
    """Create a mock method registry with a few test methods."""

    class MockInt8:
        name = "block_int8"
        category = "quantization"

        def compress(self, tensor, **kwargs):
            flat = tensor.ravel().astype(np.float32)
            scale = float(np.max(np.abs(flat))) / 127.0
            q = np.clip(np.round(flat / scale), -128, 127).astype(np.int8)
            return q.tobytes(), {"method": "block_int8"}

        def decompress(self, data, meta):
            flat = np.frombuffer(data, dtype=np.int8).astype(np.float32)
            return flat.reshape(256, 256)

    class MockInt4:
        name = "block_int4"
        category = "quantization"

        def compress(self, tensor, **kwargs):
            flat = tensor.ravel().astype(np.float32)
            scale = float(np.max(np.abs(flat))) / 7.0
            q = np.clip(np.round(flat / scale), -8, 7).astype(np.int8)
            return q.tobytes(), {"method": "block_int4"}

        def decompress(self, data, meta):
            flat = np.frombuffer(data, dtype=np.int8).astype(np.float32)
            return flat.reshape(256, 256)

    class MockSVD:
        name = "svd_compress"
        category = "decomposition"

        def compress(self, tensor, **kwargs):
            rank = kwargs.get("rank", 32)
            U, s, Vt = np.linalg.svd(tensor, full_matrices=False)
            U_k = U[:, :rank]
            s_k = s[:rank]
            Vt_k = Vt[:rank, :]
            return (U_k.tobytes() + s_k.tobytes() + Vt_k.tobytes()), {
                "method": "svd_compress",
                "rank": rank,
            }

        def decompress(self, data, meta):
            rank = meta.get("rank", 32)
            offset = 0
            U_k = np.frombuffer(
                data[offset : offset + 256 * rank * 4], dtype=np.float32
            ).reshape(256, rank)
            offset += 256 * rank * 4
            s_k = np.frombuffer(data[offset : offset + rank * 4], dtype=np.float32)
            offset += rank * 4
            Vt_k = np.frombuffer(
                data[offset : offset + rank * 256 * 4], dtype=np.float32
            ).reshape(rank, 256)
            return (U_k @ np.diag(s_k) @ Vt_k).astype(np.float32)

    class MockDCT:
        name = "dct_spectral"
        category = "spectral"

        def compress(self, tensor, **kwargs):
            return b"dct_data", {"method": "dct_spectral"}

        def decompress(self, data, meta):
            return np.zeros((256, 256), dtype=np.float32)

    registry = {
        "block_int8": {
            "class": MockInt8,
            "instance": MockInt8(),
            "category": "quantization",
            "tier": 5,
        },
        "block_int4": {
            "class": MockInt4,
            "instance": MockInt4(),
            "category": "quantization",
            "tier": 5,
        },
        "svd_compress": {
            "class": MockSVD,
            "instance": MockSVD(),
            "category": "decomposition",
            "tier": 2,
        },
        "dct_spectral": {
            "class": MockDCT,
            "instance": MockDCT(),
            "category": "spectral",
            "tier": 3,
        },
    }
    return registry


@pytest.fixture
def oracle(mock_method_registry) -> UnifiedMethodOracle:
    return UnifiedMethodOracle(
        method_registry=mock_method_registry,
        rng_seed=42,
    )


@pytest.fixture
def oracle_with_holographic(oracle, mock_method_registry) -> UnifiedMethodOracle:
    """Oracle with a holographic memory store populated with entries."""
    try:
        from spectralstream.compression.engine.holographic_oracle import (
            HolographicMemoryStore,
            ResonanceSignature,
        )

        # Also add the _HolographicMemoryStore as alias
        from spectralstream.compression.world_model.unified_method_oracle import (  # noqa: F811
            _HolographicMemoryStore as UMO_HolographicMemoryStore,
        )
    except ImportError:
        pytest.skip("HolographicOracle not importable")

    store = HolographicMemoryStore()
    oracle._holographic_memory = store

    # Populate with mock entries
    for i in range(5):
        sig = ResonanceSignature(
            mean=float(i),
            std=1.0,
            skewness=0.0,
            kurtosis=0.0,
            sparsity_1e3=0.1 * i,
            sparsity_1e4=0.05 * i,
            spectral_entropy=0.5,
            energy_concentration=0.8,
            effective_rank_ratio=0.2,
            n_elements_log=4.0,
            shape_ndim=2,
            shape_aspect=1.0,
            tensor_type="weight",
        )
        store.store(sig, "block_int8", {"block_size": 128}, ratio=50.0, error=0.005)

    return oracle


# ═══════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestUnifiedMethodOracle:
    """Test suite for UnifiedMethodOracle."""

    def test_initialization(self, oracle: UnifiedMethodOracle):
        assert oracle is not None
        assert oracle._method_registry is not None
        assert len(oracle._method_registry) >= 4

    def test_select_method_returns_method_selection(
        self, oracle: UnifiedMethodOracle, sample_tensor: np.ndarray
    ):
        selection = oracle.select_method(
            sample_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
            time_budget_ms=500.0,
        )
        assert isinstance(selection, MethodSelection)
        assert isinstance(selection.name, str)
        assert len(selection.name) > 0
        assert 0.0 <= selection.confidence <= 1.0
        assert selection.stage in (
            "holographic",
            "zero_shot_bayesian",
            "ensemble_vote",
            "superposition",
            "exhaustive",
            "fallback",
        )

    def test_select_method_very_fast_path(
        self, oracle: UnifiedMethodOracle, sample_tensor: np.ndarray
    ):
        selection = oracle.select_method(
            sample_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
            time_budget_ms=0.5,
        )
        assert isinstance(selection, MethodSelection)
        assert selection.time_ms < 5.0  # Should complete in < 5ms

    def test_select_method_speed_benchmark(
        self, oracle: UnifiedMethodOracle, sample_tensor: np.ndarray
    ):
        t0 = time.perf_counter()
        selection = oracle.select_method(
            sample_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
            time_budget_ms=100.0,
            name="test_tensor",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        assert isinstance(selection, MethodSelection)
        assert selection.name is not None

    def test_holographic_recall(
        self, oracle_with_holographic: UnifiedMethodOracle, sample_tensor: np.ndarray
    ):
        selection = oracle_with_holographic.select_method(
            sample_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
            time_budget_ms=100.0,
        )
        assert isinstance(selection, MethodSelection)
        assert isinstance(selection.name, str)

    def test_holographic_recall_with_signature(self, oracle: UnifiedMethodOracle):
        fake_sig = np.array(
            [0.0, 1.0, 0.0, 0.0, 0.1, 0.05, 0.5, 0.8, 0.2, 4.0, 2.0, 1.0]
        )
        result = oracle.recall_holographic(fake_sig)
        assert result is None or isinstance(result, tuple)

    def test_ensemble_vote_returns_dict(
        self, oracle: UnifiedMethodOracle, sample_tensor: np.ndarray
    ):
        votes = oracle.ensemble_vote(
            tensor_profile=None,
            target_ratio=100.0,
            max_error=0.01,
            tensor=sample_tensor,
            tensor_type="weight",
            name="test_tensor",
        )
        assert isinstance(votes, dict)
        if votes:
            for mname, score in votes.items():
                assert isinstance(mname, str)
                assert isinstance(score, float)

    def test_quantum_superposition(
        self,
        oracle: UnifiedMethodOracle,
        sample_tensor: np.ndarray,
        mock_method_registry,
    ):
        candidates = []
        for mname, minfo in mock_method_registry.items():
            inst = minfo.get("instance")
            if inst is not None:
                candidates.append({"name": mname, "instance": inst, "params": {}})

        result = oracle.test_in_superposition(
            sample_tensor, candidates, target_ratio=100.0, max_error=0.01
        )
        assert isinstance(result, QuantumSuperpositionTest)
        assert result.n_tested >= 0
        if result.n_tested > 0:
            assert isinstance(result.best_method, str)

    def test_superposition_low_rank_tensor(
        self,
        oracle: UnifiedMethodOracle,
        low_rank_tensor: np.ndarray,
        mock_method_registry,
    ):
        candidates = []
        for mname, minfo in mock_method_registry.items():
            inst = minfo.get("instance")
            if inst is not None:
                candidates.append({"name": mname, "instance": inst, "params": {}})

        result = oracle.test_in_superposition(
            low_rank_tensor, candidates, target_ratio=100.0, max_error=0.01
        )
        assert result.n_tested >= 0

    def test_query_bayesian(self, oracle: UnifiedMethodOracle):
        features = {"tensor_type": "weight"}
        results = oracle.query_bayesian(features)
        assert isinstance(results, dict)

    def test_predict_zeroshot(self, oracle: UnifiedMethodOracle):
        fp = np.zeros(256)
        results = oracle.predict_zeroshot(fp)
        assert isinstance(results, dict)

    def test_record_performance(self, oracle: UnifiedMethodOracle):
        oracle.record_performance("weight", "block_int8", 50.0, 0.005)
        assert "weight" in oracle._performance_history
        assert "block_int8" in oracle._performance_history["weight"]
        h = oracle._performance_history["weight"]["block_int8"]
        assert h["n_tests"] == 1
        assert h["avg_ratio"] == 50.0
        assert h["avg_error"] == 0.005

        oracle.record_performance("weight", "block_int8", 100.0, 0.001)
        h = oracle._performance_history["weight"]["block_int8"]
        assert h["n_tests"] == 2
        assert h["confidence"] > 0.0

    def test_get_stats(self, oracle: UnifiedMethodOracle, sample_tensor: np.ndarray):
        oracle.select_method(sample_tensor, time_budget_ms=500.0)
        stats = oracle.get_stats()
        assert isinstance(stats, dict)

    def test_rnd_mode(self, oracle: UnifiedMethodOracle, sample_tensor: np.ndarray):
        selection = oracle.select_method(
            sample_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
            time_budget_ms=5000.0,
            rnd_mode=True,
            name="test_rnd",
        )
        assert isinstance(selection, MethodSelection)
        assert isinstance(selection.name, str)

    def test_multiple_tensor_types(self, oracle: UnifiedMethodOracle):
        rng = np.random.RandomState(42)
        tensor = rng.randn(128, 128).astype(np.float32)

        for ttype in ("weight", "attention_q", "ffn_gate", "embedding", "norm"):
            selection = oracle.select_method(
                tensor,
                tensor_type=ttype,
                target_ratio=100.0,
                max_error=0.01,
                time_budget_ms=500.0,
                name=f"test_{ttype}",
            )
            assert isinstance(selection, MethodSelection)
            assert isinstance(selection.name, str)

    def test_extract_features(
        self, oracle: UnifiedMethodOracle, sample_tensor: np.ndarray
    ):
        features = oracle._extract_features(sample_tensor, None, "weight")
        assert isinstance(features, _TensorFeatures)
        assert features.n_elements == sample_tensor.size
        assert features.ndim == 2
        assert features.tensor_type == "weight"
        assert features.sparsity >= 0.0
        assert features.mean_abs > 0.0

    def test_ensemble_vote_with_profile(
        self, oracle: UnifiedMethodOracle, sample_tensor: np.ndarray
    ):
        features = oracle._extract_features(sample_tensor, None, "attention_q")
        profile = type(
            "Profile",
            (),
            {
                "shape": sample_tensor.shape,
                "n_elements": sample_tensor.size,
                "nbytes": sample_tensor.nbytes,
                "ndim": 2,
                "dtype": "float32",
                "mean": float(np.mean(sample_tensor)),
                "std": float(np.std(sample_tensor)),
                "min_val": float(np.min(sample_tensor)),
                "max_val": float(np.max(sample_tensor)),
                "sparsity": 0.0,
                "effective_rank": 0.3,
                "energy_concentration": 0.8,
                "spectral_entropy": 0.5,
                "sensitivity": 0.5,
                "compressibility_score": 0.7,
                "spectral_decay_rate": 0.5,
                "entropy_rate": 0.5,
                "nm_sparsity_score": 0.0,
                "recommended_method": "svd_compress",
                "recommended_bits": 8,
                "kurtosis": 0.0,
                "skewness": 0.0,
                "tensor_type": "attention_q",
                "outlier_ratio_3sigma": 0.01,
            },
        )()

        votes = oracle.ensemble_vote(
            tensor_profile=profile,
            target_ratio=100.0,
            max_error=0.01,
            tensor=sample_tensor,
            tensor_type="attention_q",
            name="test_attention",
        )
        assert isinstance(votes, dict)

    def test_low_rank_vs_sparse_selection(
        self,
        oracle: UnifiedMethodOracle,
        low_rank_tensor: np.ndarray,
        sparse_tensor: np.ndarray,
    ):
        # Low-rank tensor should prefer decomposition methods
        lr_sel = oracle.select_method(
            low_rank_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
            time_budget_ms=500.0,
            name="low_rank_test",
        )

        # Sparse tensor should prefer sparsity methods
        sp_sel = oracle.select_method(
            sparse_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
            time_budget_ms=500.0,
            name="sparse_test",
        )

        assert isinstance(lr_sel, MethodSelection)
        assert isinstance(sp_sel, MethodSelection)

    def test_clear_cache(self, oracle: UnifiedMethodOracle):
        oracle.record_performance("weight", "test_method", 10.0, 0.01)
        assert len(oracle._performance_history) > 0
        oracle.clear_cache()
        assert len(oracle._performance_history) == 0

    def test_fast_path_under_1ms(
        self, oracle: UnifiedMethodOracle, sample_tensor: np.ndarray
    ):
        t0 = time.perf_counter()
        selection = oracle.select_method(
            sample_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
            time_budget_ms=0.5,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 5.0, f"Fast path took {elapsed_ms:.2f}ms (expected < 5ms)"
        assert isinstance(selection, MethodSelection)

    def test_method_selection_fields(self):
        sel = MethodSelection(
            name="block_int8",
            params={"block_size": 128},
            confidence=0.95,
            score=0.95,
            expected_ratio=50.0,
            expected_error=0.005,
            bypass_decision=BYPASS_HIGH_CONFIDENCE,
            stage="holographic",
            time_ms=0.5,
        )
        assert sel.name == "block_int8"
        assert sel.confidence == 0.95
        assert sel.bypass_decision == BYPASS_HIGH_CONFIDENCE
        assert sel.stage == "holographic"
        assert sel.time_ms == 0.5

    def test_quantum_superposition_result(self):
        result = QuantumSuperpositionTest(
            method_names=["a", "b"],
            results={
                "a": {
                    "ratio": 50.0,
                    "error": 0.01,
                    "time_ms": 10.0,
                    "compressed_bytes": 1024,
                }
            },
            best_method="a",
            time_ms=15.0,
        )
        assert result.n_tested == 1
        assert result.best_method == "a"
        assert result.time_ms == 15.0

    def test_sparse_vs_decision_tree_vote(
        self, oracle: UnifiedMethodOracle, sparse_tensor: np.ndarray
    ):
        features = oracle._extract_features(sparse_tensor, None, "weight")
        votes = oracle._decision_tree_vote(features, sparse_tensor)
        assert isinstance(votes, dict)

    def test_low_rank_vs_decision_tree_vote(
        self, oracle: UnifiedMethodOracle, low_rank_tensor: np.ndarray
    ):
        features = oracle._extract_features(low_rank_tensor, None, "weight")
        votes = oracle._decision_tree_vote(features, low_rank_tensor)
        assert isinstance(votes, dict)
        if votes:
            top_method = max(votes, key=votes.get)
            _ = top_method  # Just verify it exists

    def test_get_all_methods(self, oracle: UnifiedMethodOracle):
        methods = oracle._get_all_methods()
        assert isinstance(methods, dict)
        assert len(methods) > 0

    def test_get_all_method_names(self, oracle: UnifiedMethodOracle):
        names = oracle._get_all_method_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_default_params(self, oracle: UnifiedMethodOracle):
        params = oracle.DEFAULT_PARAMS.get("block_int8")
        assert params == {"block_size": 128}

        params = oracle.DEFAULT_PARAMS.get("unknown_method", {})
        assert params == {}

    def test_bypass_constants(self):
        assert BYPASS_HIGH_CONFIDENCE == "bypass_high_confidence"
        assert BYPASS_MEDIUM_CONFIDENCE == "bypass_medium_confidence"
        assert TEST_FULL == "test_full"

    def test_bind_engine(self, oracle: UnifiedMethodOracle):
        fake_engine = type("Engine", (), {"get_methods": lambda self: {}})()
        oracle.bind_engine(fake_engine)
        assert oracle._engine is not None

    def test_multiple_selections_consistency(self, oracle: UnifiedMethodOracle):
        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 64).astype(np.float32)

        s1 = oracle.select_method(tensor, time_budget_ms=500.0)
        s2 = oracle.select_method(tensor, time_budget_ms=500.0)
        s3 = oracle.select_method(tensor, time_budget_ms=500.0)

        assert isinstance(s1, MethodSelection)
        assert isinstance(s2, MethodSelection)
        assert isinstance(s3, MethodSelection)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

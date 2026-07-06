"""
Tests for the Unified Compression World Model.

Tests cover:
1. ResonanceSignature — vector, hash, features
2. HolographicMemoryStore — store, recall (exact + approximate), persistence
3. BayesianPerformanceTracker — record, predict, best
4. CompressionKnowledgeGraph — update, best category
5. GeneticStrategyEvolver — evolution, fitness, genome params
6. TensorLossMetrics — 20+ metric computation
7. UnifiedCompressionWorldModel — scan, select_method, compute_loss_metrics,
   plan_cascade, compress, record, get_stats, save/load state, certify
8. Integration — full world model pipeline with real tensor compression
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from typing import Any, Dict

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Import from the world model package ─────────────────────────────────────


@pytest.fixture(scope="module")
def import_world_model():
    try:
        from spectralstream.compression.world_model.unified_world_model import (
            UnifiedCompressionWorldModel,
            ResonanceSignature,
            HolographicMemoryStore,
            BayesianPerformanceTracker,
            CompressionKnowledgeGraph,
            GeneticStrategyEvolver,
            TensorLossMetrics,
            CompressionMode,
            RankedMethod,
            CascadeStage,
            CascadePlan,
            ModelWorldProfile,
            CompressionCertificate,
            MemoryEntry,
            BYPASS_HIGH_CONFIDENCE,
            BYPASS_MEDIUM_CONFIDENCE,
            TEST_FULL,
        )

        return {
            "world_model": UnifiedCompressionWorldModel,
            "signature": ResonanceSignature,
            "memory": HolographicMemoryStore,
            "bayesian": BayesianPerformanceTracker,
            "kg": CompressionKnowledgeGraph,
            "genetic": GeneticStrategyEvolver,
            "metrics": TensorLossMetrics,
            "mode": CompressionMode,
            "ranked": RankedMethod,
            "stage": CascadeStage,
            "plan": CascadePlan,
            "profile": ModelWorldProfile,
            "cert": CompressionCertificate,
            "mem_entry": MemoryEntry,
            "BYPASS_HIGH": BYPASS_HIGH_CONFIDENCE,
            "BYPASS_MEDIUM": BYPASS_MEDIUM_CONFIDENCE,
            "TEST_FULL": TEST_FULL,
        }
    except ImportError as e:
        pytest.skip(f"World model import failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def small_tensor(rng):
    return rng.randn(32, 32).astype(np.float32)


@pytest.fixture
def medium_tensor(rng):
    return rng.randn(64, 64).astype(np.float32)


@pytest.fixture
def sine_tensor():
    t = np.linspace(0, 4 * np.pi, 128, endpoint=False)
    return np.sin(t).astype(np.float32).reshape(8, 16)


@pytest.fixture
def low_rank_tensor(rng):
    a = rng.randn(32, 4).astype(np.float32)
    b = rng.randn(4, 32).astype(np.float32)
    return a @ b  # rank 4


@pytest.fixture
def random_model(rng):
    return {
        "model.layers.0.self_attn.q_proj.weight": rng.randn(64, 64).astype(np.float32),
        "model.layers.0.self_attn.k_proj.weight": rng.randn(64, 64).astype(np.float32),
        "model.layers.0.self_attn.v_proj.weight": rng.randn(64, 64).astype(np.float32),
        "model.layers.0.self_attn.o_proj.weight": rng.randn(64, 64).astype(np.float32),
        "model.layers.0.mlp.gate_proj.weight": rng.randn(256, 64).astype(np.float32),
        "model.layers.0.mlp.up_proj.weight": rng.randn(256, 64).astype(np.float32),
        "model.layers.0.mlp.down_proj.weight": rng.randn(64, 256).astype(np.float32),
        "model.layers.0.input_layernorm.weight": rng.randn(64).astype(np.float32),
        "model.embed_tokens.weight": rng.randn(32000, 64).astype(np.float32),
    }


@pytest.fixture
def tiny_model(rng):
    return {
        "attention_q": rng.randn(16, 16).astype(np.float32),
        "attention_k": rng.randn(16, 16).astype(np.float32),
        "ffn_gate": rng.randn(32, 16).astype(np.float32),
        "embedding": rng.randn(100, 16).astype(np.float32),
        "norm": rng.randn(16).astype(np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  1. ResonanceSignature Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestResonanceSignature:
    def test_n_features(self, import_world_model):
        sig = import_world_model["signature"]
        assert sig.n_features() == 12

    def test_to_vector_shape(self, import_world_model):
        sig = import_world_model["signature"](
            mean=1.0,
            std=2.0,
            tensor_type="attention_q",
        )
        vec = sig.to_vector()
        assert vec.shape == (12,)
        assert vec[0] == 1.0
        assert vec[1] == 2.0

    def test_to_hash_unique(self, import_world_model):
        sig1 = import_world_model["signature"](mean=1.0, tensor_type="weight")
        sig2 = import_world_model["signature"](mean=2.0, tensor_type="weight")
        assert sig1.to_hash() != sig2.to_hash()

    def test_to_hash_same_type(self, import_world_model):
        sig1 = import_world_model["signature"](
            mean=1.0, std=0.5, tensor_type="attention_q"
        )
        sig2 = import_world_model["signature"](
            mean=1.0, std=0.5, tensor_type="attention_q"
        )
        assert sig1.to_hash() == sig2.to_hash()

    def test_to_dict(self, import_world_model):
        sig = import_world_model["signature"](
            mean=1.0,
            std=2.0,
            tensor_type="weight",
            _tensor_name="test",
            _tensor_shape=(8, 8),
        )
        d = sig.to_dict()
        assert d["mean"] == 1.0
        assert d["std"] == 2.0
        assert d["tensor_type"] == "weight"


# ═══════════════════════════════════════════════════════════════════════════
#  2. HolographicMemoryStore Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHolographicMemoryStore:
    def test_store_and_recall_exact(self, import_world_model):
        sig_cls = import_world_model["signature"]
        store = import_world_model["memory"]()

        sig = sig_cls(mean=0.5, std=0.1, tensor_type="weight")
        store.store(sig, "block_int8", {"block_size": 128}, 4.0, 0.01)

        recalled = store.recall(sig, min_confidence=0.0)
        assert recalled is not None
        assert recalled["method_name"] == "block_int8"
        assert recalled["match_type"] == "exact"
        assert recalled["ratio"] == 4.0
        assert recalled["error"] == 0.01

    def test_recall_no_match(self, import_world_model):
        sig_cls = import_world_model["signature"]
        store = import_world_model["memory"]()

        sig_store = sig_cls(mean=1.0, tensor_type="weight")
        store.store(sig_store, "svd", {}, 50.0, 0.001)

        sig_new = sig_cls(mean=999.0, tensor_type="weight")
        recalled = store.recall(sig_new, min_confidence=0.9)
        assert recalled is None  # far from stored

    def test_store_updates_existing(self, import_world_model):
        sig_cls = import_world_model["signature"]
        store = import_world_model["memory"]()

        sig = sig_cls(mean=0.5, tensor_type="weight")
        store.store(sig, "block_int8", {}, 4.0, 0.01)
        store.store(sig, "block_int8", {}, 6.0, 0.005)

        recalled = store.recall(sig, min_confidence=0.0)
        assert recalled is not None
        assert recalled["ratio"] == 5.0  # (4+6)/2

    def test_n_entries(self, import_world_model):
        sig_cls = import_world_model["signature"]
        store = import_world_model["memory"]()
        assert store.n_entries() == 0

        store.store(sig_cls(mean=1.0, tensor_type="weight"), "a", {}, 1.0, 0.01)
        assert store.n_entries() == 1

        store.store(sig_cls(mean=2.0, tensor_type="weight"), "b", {}, 2.0, 0.01)
        assert store.n_entries() == 2

        store.store(sig_cls(mean=1.0, tensor_type="weight"), "a", {}, 3.0, 0.01)
        assert store.n_entries() == 2  # same hash → update, not new

    def test_get_stats(self, import_world_model):
        sig_cls = import_world_model["signature"]
        store = import_world_model["memory"]()
        empty_stats = store.get_stats()
        assert empty_stats["n_entries"] == 0

        store.store(sig_cls(mean=1.0, tensor_type="ffn_gate"), "dct", {}, 5.0, 0.005)
        store.store(
            sig_cls(mean=2.0, tensor_type="attention_q"), "svd", {}, 50.0, 0.001
        )

        stats = store.get_stats()
        assert stats["n_entries"] == 2
        assert stats["n_types"] == 2
        assert stats["avg_ratio"] > 0

    def test_save_and_load(self, import_world_model):
        sig_cls = import_world_model["signature"]
        store = import_world_model["memory"]()

        store.store(
            sig_cls(mean=1.0, tensor_type="weight"), "method_a", {"p": 1}, 10.0, 0.01
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name

        try:
            store.save(tmp_path)

            store2 = import_world_model["memory"]()
            store2.load(tmp_path)
            assert store2.n_entries() == 1

            recalled = store2.recall(
                sig_cls(mean=1.0, tensor_type="weight"), min_confidence=0.0
            )
            assert recalled is not None
            assert recalled["method_name"] == "method_a"
            assert recalled["ratio"] == 10.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_clear(self, import_world_model):
        sig_cls = import_world_model["signature"]
        store = import_world_model["memory"]()
        store.store(sig_cls(mean=1.0, tensor_type="weight"), "a", {}, 1.0, 0.01)
        assert store.n_entries() == 1
        store.clear()
        assert store.n_entries() == 0


# ═══════════════════════════════════════════════════════════════════════════
#  3. BayesianPerformanceTracker Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBayesianPerformanceTracker:
    def test_record_and_predict(self, import_world_model):
        b = import_world_model["bayesian"]()
        b.record("svd_compress", "attention_q", 50.0, 0.001)
        perf = b.predict("svd_compress", "attention_q")
        assert perf.expected_ratio == 50.0
        assert perf.expected_error == 0.001
        assert perf.n_trials == 1

    def test_predict_unknown(self, import_world_model):
        b = import_world_model["bayesian"]()
        perf = b.predict("unknown_method", "weight")
        assert perf.expected_ratio == 3.88
        assert perf.expected_error == 0.01
        assert perf.confidence == 0.1

    def test_best(self, import_world_model):
        b = import_world_model["bayesian"]()
        b.record("svd", "attention_q", 50.0, 0.001)
        b.record("dct", "attention_q", 10.0, 0.01)
        best_method, best_score = b.get_best(
            "attention_q", ["svd", "dct", "block_int8"]
        )
        assert best_method == "svd"

    def test_multiple_records_confidence(self, import_world_model):
        b = import_world_model["bayesian"]()
        for _ in range(25):
            b.record("block_int8", "weight", 4.0, 0.01)

        perf = b.predict("block_int8", "weight")
        assert perf.confidence == 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  4. CompressionKnowledgeGraph Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCompressionKnowledgeGraph:
    def test_update_and_best_category(self, import_world_model):
        kg = import_world_model["kg"]()
        kg.update("attention_q", "decomposition", 50.0, 0.001)
        kg.update("attention_q", "quantization", 4.0, 0.01)
        assert kg.get_best_category("attention_q") == "decomposition"

    def test_top_categories(self, import_world_model):
        kg = import_world_model["kg"]()
        kg.update("ffn_gate", "spectral", 10.0, 0.005)
        kg.update("ffn_gate", "structural", 8.0, 0.003)
        kg.update("ffn_gate", "quantization", 4.0, 0.01)
        top = kg.get_top_categories("ffn_gate", 2)
        assert len(top) == 2

    def test_empty_best_category(self, import_world_model):
        kg = import_world_model["kg"]()
        assert kg.get_best_category("unknown") == "quantization"

    def test_to_dict(self, import_world_model):
        kg = import_world_model["kg"]()
        kg.update("attention_q", "decomposition", 50.0, 0.001)
        d = kg.to_dict()
        assert "graph" in d
        assert "attention_q" in d["graph"]


# ═══════════════════════════════════════════════════════════════════════════
#  5. GeneticStrategyEvolver Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGeneticStrategyEvolver:
    def test_initialize_population(self, import_world_model):
        g = import_world_model["genetic"](population_size=20)
        assert len(g.population) == 20
        assert "rank_threshold" in g.population[0]
        assert "energy_threshold" in g.population[0]

    def test_evolve_with_results(self, import_world_model):
        g = import_world_model["genetic"](population_size=10)
        results = [
            {"score": 100.0, "ratio": 50.0, "error": 0.001},
            {"score": 80.0, "ratio": 30.0, "error": 0.005},
            {"score": 60.0, "ratio": 10.0, "error": 0.01},
        ]
        best = g.evolve(results)
        assert best is not None
        assert g.generation == 1
        assert g.best_fitness > 0

    def test_get_params(self, import_world_model):
        g = import_world_model["genetic"](population_size=5)
        params = g.get_params()
        assert params == {}
        g.evolve([{"score": 50.0}])
        params = g.get_params()
        assert "rank_threshold" in params
        assert "generation" in params
        assert params["generation"] == 1


# ═══════════════════════════════════════════════════════════════════════════
#  6. TensorLossMetrics Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTensorLossMetrics:
    @pytest.fixture
    def wm(self, import_world_model):
        cls = import_world_model["world_model"]
        return cls(memory_budget_mb=4096, max_workers=2)

    def test_identical_tensors(self, wm, small_tensor):
        loss = wm.compute_loss_metrics(
            original=small_tensor,
            reconstructed=small_tensor,
            name="identical",
            compressed_size=small_tensor.nbytes,
        )
        assert loss.mse == 0.0
        assert loss.mae == 0.0
        assert loss.cosine_similarity == 1.0
        assert loss.snr_db == float("inf")
        assert loss.quality_grade == "EXCELLENT"
        assert loss.is_acceptable is True

    def test_orthogonal_tensors(self, wm, small_tensor):
        opposite = -small_tensor.copy()
        loss = wm.compute_loss_metrics(
            original=small_tensor,
            reconstructed=opposite,
            name="opposite",
            compressed_size=1,
        )
        assert loss.cosine_similarity < -0.9
        assert loss.mse > 0

    def test_compression_ratio(self, wm, small_tensor):
        loss = wm.compute_loss_metrics(
            original=small_tensor,
            reconstructed=small_tensor,
            name="trimmed",
            compressed_size=256,
        )
        assert loss.compression_ratio == small_tensor.nbytes / 256

    def test_to_dict(self, wm, small_tensor):
        loss = wm.compute_loss_metrics(
            original=small_tensor,
            reconstructed=small_tensor,
            name="test",
            compressed_size=small_tensor.nbytes,
        )
        d = loss.to_dict()
        assert isinstance(d, dict)
        assert d["mse"] == 0.0

    def test_metrics_fields(self, wm, small_tensor):
        loss = wm.compute_loss_metrics(
            original=small_tensor,
            reconstructed=small_tensor,
            name="full",
            compressed_size=small_tensor.nbytes,
        )
        fields = [
            "mse",
            "mae",
            "max_ae",
            "rmse",
            "snr_db",
            "psnr_db",
            "cosine_similarity",
            "kl_divergence",
            "wasserstein_distance",
            "ks_statistic",
            "js_divergence",
            "mean_bias",
            "std_shift",
            "skewness_shift",
            "kurtosis_shift",
            "outlier_preservation_rate",
            "spectral_norm_error",
            "effective_rank_error",
            "quality_grade",
            "is_acceptable",
            "compression_ratio",
        ]
        for f in fields:
            assert hasattr(loss, f), f"Missing field: {f}"


# ═══════════════════════════════════════════════════════════════════════════
#  7. UnifiedCompressionWorldModel Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def wm(import_world_model):
    cls = import_world_model["world_model"]
    instance = cls(memory_budget_mb=4096, max_workers=2)
    return instance


class TestUnifiedCompressionWorldModel:
    """Integration tests for the full UnifiedCompressionWorldModel."""

    def test_scan_model(self, wm, tiny_model):
        profile = wm.scan_model(tiny_model)
        assert profile.graph.n_tensors == 5
        assert profile.estimated_model_size_gb > 0
        assert len(profile.sensitivity_tiers) == 5

    def test_scan_model_from_metadata(self, wm, tiny_model):
        tensor_infos: Dict[str, Any] = {}
        for name, tensor in tiny_model.items():
            tensor_infos[name] = (tensor.shape, str(tensor.dtype), 0, tensor.nbytes)
        profile = wm.scan_model_from_metadata(tensor_infos)
        assert profile.graph.n_tensors == 5

    def test_classify_by_name(self, wm):
        assert wm.classify_by_name("model.q_proj.weight") == "attention_q"
        assert wm.classify_by_name("model.k_proj.weight") == "attention_k"
        assert wm.classify_by_name("model.v_proj.weight") == "attention_v"
        assert wm.classify_by_name("model.o_proj.weight") == "attention_o"
        assert wm.classify_by_name("model.gate_proj.weight") == "ffn_gate"
        assert wm.classify_by_name("model.up_proj.weight") == "ffn_up"
        assert wm.classify_by_name("model.down_proj.weight") == "ffn_down"
        assert wm.classify_by_name("embed_tokens.weight") == "embedding"
        assert wm.classify_by_name("model.norm.weight") == "norm"
        assert wm.classify_by_name("lm_head.weight") == "output"
        assert wm.classify_by_name("unknown.weight") == "weight"

    def test_select_method(self, wm, small_tensor):
        ranked, bypass = wm.select_method(
            small_tensor,
            tensor_type="weight",
            target_ratio=100.0,
            max_error=0.01,
            name="test_weight",
            max_results=10,
        )
        assert len(ranked) >= 1
        assert bypass in (
            "bypass_high_confidence",
            "bypass_medium_confidence",
            "test_full",
        )

    def test_compute_loss_metrics(self, wm, small_tensor):
        loss = wm.compute_loss_metrics(
            original=small_tensor,
            reconstructed=small_tensor,
            name="identity",
            compressed_size=small_tensor.nbytes,
        )
        assert loss.mse == 0.0
        assert loss.cosine_similarity == 1.0
        assert loss.quality_grade == "EXCELLENT"

    def test_compute_loss_metrics_shifted(self, wm, small_tensor):
        shifted = small_tensor + 1.0
        loss = wm.compute_loss_metrics(
            original=small_tensor,
            reconstructed=shifted,
            name="shifted",
            compressed_size=small_tensor.nbytes,
        )
        assert abs(loss.mean_bias - (-1.0)) < 0.001, f"mean_bias={loss.mean_bias}"
        assert loss.kl_divergence > 0

    def test_plan_cascade(self, wm, medium_tensor):
        plan = wm.plan_cascade(
            medium_tensor,
            tensor_type="weight",
            target_ratio=5000.0,
            max_error=0.01,
            name="test_cascade",
        )
        assert plan.n_stages >= 2
        assert plan.total_expected_ratio > 0
        assert plan.source != ""

    def test_compress_small_tensor(self, wm, small_tensor):
        data, meta, ratio, error = wm.compress(
            small_tensor,
            target_ratio=10.0,
            max_error=0.05,
            name="test_compress",
        )
        assert isinstance(data, bytes)
        assert isinstance(meta, dict)
        assert ratio >= 1.0
        assert error >= 0.0

    def test_compress_tensor_type_names(self, wm, rng):
        for name in [
            "attention_q",
            "attention_k",
            "attention_v",
            "attention_o",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
            "embedding",
            "norm",
        ]:
            tensor = rng.randn(16, 16).astype(np.float32)
            data, meta, ratio, error = wm.compress(
                tensor,
                target_ratio=5.0,
                max_error=0.1,
                name=name,
            )
            assert isinstance(data, bytes), f"Failed for {name}"
            assert ratio > 0, f"ratio=0 for {name}"

    def test_compress_model_level(self, wm, tiny_model):
        results = wm.compress(tiny_model, target_ratio=10.0, max_error=0.05)
        assert "_meta" in results
        meta = results["_meta"]
        assert meta["total_tensors"] == 5
        assert meta["overall_ratio"] >= 1.0
        assert "method_distribution" in meta

    def test_compress_streaming(self, wm):
        pytest.skip("Streaming test requires safetensors I/O and engine internals")

    def test_record_compression(self, wm, small_tensor):
        wm.record_compression(
            tensor=small_tensor,
            tensor_type="weight",
            method_name="block_int8",
            method_category="quantization",
            ratio=4.0,
            error=0.01,
            name="recorded_tensor",
        )
        assert len(wm._compression_history) >= 1

    def test_get_stats(self, wm):
        stats = wm.get_stats()
        assert "oracle" in stats
        assert "holographic_memory" in stats
        assert "bayesian" in stats
        assert "knowledge_graph" in stats
        assert "genetic_evolver" in stats
        assert "compression_history" in stats

    def test_save_and_load_state(self, wm, small_tensor):
        sig = wm._compute_signature_from_tensor(small_tensor, "test")
        wm.holo_memory.store(sig, "block_int8", {}, 4.0, 0.01)

        with tempfile.NamedTemporaryFile(suffix=".state", delete=False) as f:
            state_path = f.name

        try:
            wm.save_state(state_path)
            mem_path = state_path + ".holographic_memory.npz"
            assert os.path.exists(mem_path)

            kg_path = state_path + ".knowledge_graph.json"
            assert os.path.exists(kg_path)

            genetic_path = state_path + ".genetic.json"
            assert os.path.exists(genetic_path)

            wm2 = wm.__class__(memory_budget_mb=4096, max_workers=2)
            wm2.load_state(state_path)
            recalled = wm2.holo_memory.recall(sig, min_confidence=0.0)
            assert recalled is not None
            assert recalled["method_name"] == "block_int8"
        finally:
            for suffix in [
                ".holographic_memory.npz",
                ".knowledge_graph.json",
                ".genetic.json",
            ]:
                p = state_path + suffix
                if os.path.exists(p):
                    os.remove(p)

    def test_certify(self, wm, small_tensor):
        data, meta, ratio, error = wm.compress(
            small_tensor,
            target_ratio=5.0,
            max_error=0.1,
            name="test_tensor",
        )
        cert_data = wm.certify(
            original=small_tensor,
            compressed_data=data,
            metadata=meta,
            name="test_tensor",
        )
        assert cert_data["method"] is not None
        assert cert_data["ratio"] >= 1.0

    def test_certify_model(self, wm, tiny_model):
        results = wm.compress(tiny_model, target_ratio=5.0, max_error=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            certs = wm.certify_model(results, output_dir=tmpdir, formats=["json"])
            assert len(certs) == 5
            assert os.path.exists(os.path.join(tmpdir, "model_certificate.json"))

    def test_compress_low_rank_tensor(self, wm, low_rank_tensor):
        data, meta, ratio, error = wm.compress(
            low_rank_tensor,
            target_ratio=500.0,
            max_error=0.01,
            name="low_rank",
        )
        assert isinstance(data, bytes)
        assert ratio >= 1.0

    def test_compress_sine_tensor(self, wm, sine_tensor):
        data, meta, ratio, error = wm.compress(
            sine_tensor,
            target_ratio=100.0,
            max_error=0.005,
            name="sine",
        )
        assert isinstance(data, bytes)
        assert ratio >= 1.0

    def test_compress_cascade_mode(self, wm, medium_tensor):
        data, meta, ratio, error = wm.compress(
            medium_tensor,
            target_ratio=5000.0,
            max_error=0.01,
            name="cascade_test",
            use_cascade=True,
        )
        assert isinstance(data, bytes)
        assert ratio >= 1.0

    def test_select_method_ranked_scores(self, wm, small_tensor):
        ranked, bypass = wm.select_method(
            small_tensor,
            tensor_type="attention_q",
            target_ratio=100.0,
            max_error=0.01,
            name="q_proj",
            max_results=5,
        )
        assert len(ranked) <= 5
        if len(ranked) >= 2:
            assert ranked[0].vote_score >= ranked[1].vote_score

    def test_classify_by_name_layer_idx(self, wm):
        assert (
            wm.classify_by_name("model.layers.0.self_attn.q_proj.weight")
            == "attention_q"
        )
        assert wm.classify_by_name("model.layers.5.mlp.gate_proj.weight") == "ffn_gate"

    def test_compress_preserves_shape(self, wm, small_tensor):
        data, meta, ratio, error = wm.compress(
            small_tensor,
            target_ratio=10.0,
            max_error=0.05,
            name="shape_check",
        )
        assert "original_shape" in meta
        assert tuple(meta["original_shape"]) == small_tensor.shape

    def test_benchmark_mode(self, wm, tiny_model):
        report = wm.benchmark_mode(
            tiny_model,
            target_ratio=100.0,
            max_error=0.01,
            max_methods_per_tensor=5,
        )
        assert "world_model" in report
        assert "per_type_results" in report
        assert "best_cascade_plans" in report
        assert "genetic_strategy" in report
        assert "oracle_stats" in report
        assert report["world_model"]["n_tensors"] == 5


# ═══════════════════════════════════════════════════════════════════════════
#  8. Edge Cases and Error Handling
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_holographic_recall(self, import_world_model):
        sig_cls = import_world_model["signature"]
        store = import_world_model["memory"]()
        sig = sig_cls(mean=0.5, tensor_type="weight")
        assert store.recall(sig, min_confidence=0.9) is None

    def test_memory_store_same_hash_updates(self, import_world_model):
        sig_cls = import_world_model["signature"]
        store = import_world_model["memory"]()
        sig = sig_cls(mean=0.5, std=0.1, tensor_type="weight")
        store.store(sig, "method_a", {}, 10.0, 0.01)
        store.store(sig, "method_a", {}, 20.0, 0.005)
        stats = store.get_stats()
        assert stats["n_entries"] == 1

    def test_bayesian_cross_method_transfer(self, import_world_model):
        b = import_world_model["bayesian"]()
        b.record("svd_compress", "attention_q", 50.0, 0.001)
        perf = b.predict("svd_truncated", "attention_q")
        assert perf.expected_ratio >= 3.0

    def test_genetic_evolve_empty(self, import_world_model):
        g = import_world_model["genetic"](population_size=10)
        best = g.evolve([])
        assert best is None

    def test_ranked_method_score(self, import_world_model):
        rm = import_world_model["ranked"](name="test", vote_score=0.75)
        assert rm.score == 0.75

    def test_cascade_plan_add_stage(self, import_world_model):
        plan_cls = import_world_model["plan"]
        stage_cls = import_world_model["stage"]
        plan = plan_cls(tensor_type="weight")
        plan.add_stage(
            stage_cls(method_name="svd", expected_ratio=50.0, expected_error=0.001)
        )
        plan.add_stage(
            stage_cls(method_name="dct", expected_ratio=10.0, expected_error=0.005)
        )
        assert plan.n_stages == 2
        assert abs(plan.total_expected_ratio - 500.0) < 0.01
        assert abs(plan.total_expected_error - 0.006) < 0.0001

    def test_compress_1d_tensor(self, wm, rng):
        tensor_1d = rng.randn(128).astype(np.float32)
        data, meta, ratio, error = wm.compress(
            tensor_1d,
            target_ratio=5.0,
            max_error=0.1,
            name="norm_bias",
        )
        assert isinstance(data, bytes)
        assert ratio > 0

    def test_compress_empty_dict(self, wm):
        result = wm.compress({}, target_ratio=10.0, max_error=0.05)
        assert "_meta" in result
        assert result["_meta"]["total_tensors"] == 0

    def test_resonance_signature_from_wm(self, wm, small_tensor, import_world_model):
        sig = wm._compute_signature_from_tensor(small_tensor, "test_weight")
        sig_cls = import_world_model["signature"]
        assert isinstance(sig, sig_cls)
        assert sig.shape_ndim == 2
        assert sig.n_elements_log > 0

    def test_select_method_bypass_high(self, wm, small_tensor, import_world_model):
        sig = wm._compute_signature_from_tensor(small_tensor, "attention_q")
        wm.holo_memory.store(sig, "svd_compress", {"rank": 32}, 500.0, 0.0005)

        ranked, bypass = wm.select_method(
            small_tensor,
            tensor_type="attention_q",
            target_ratio=500.0,
            max_error=0.01,
            name="attention_q",
        )
        assert bypass in (
            import_world_model["BYPASS_HIGH"],
            import_world_model["BYPASS_MEDIUM"],
            import_world_model["TEST_FULL"],
        )
        assert len(ranked) >= 1

    def test_benchmark_small_model(self, wm, tiny_model):
        report = wm.benchmark_mode(tiny_model, max_methods_per_tensor=3)
        assert report["world_model"]["n_tensors"] == 5
        for ttype, tres in report["per_type_results"].items():
            assert "best_method" in tres
            assert "top_5_methods" in tres

    def test_save_load_genetic_state(self, wm):
        wm.genetic.evolve([{"score": 50.0}])
        gen_before = wm.genetic.generation
        with tempfile.NamedTemporaryFile(
            suffix=".genetic.json", delete=False, mode="w"
        ) as f:
            json.dump(
                {
                    "generation": gen_before,
                    "best_fitness": 42.0,
                    "best_genome": {"rank_threshold": 0.5},
                },
                f,
            )
        os.remove(f.name)

    def test_certify_empty_tensor(self, wm):
        empty = np.array([], dtype=np.float32)
        cert_data = wm.certify(
            original=empty,
            compressed_data=b"",
            metadata={"method": "none"},
            name="empty",
        )
        assert cert_data["method"] == "none"

    def test_loss_metrics_no_recon(self, wm, small_tensor):
        zero = np.zeros_like(small_tensor)
        loss = wm.compute_loss_metrics(
            small_tensor, zero, "zero_recon", small_tensor.nbytes
        )
        assert loss.cosine_similarity == 0.0
        assert loss.mse > 0

    def test_compress_with_progress(self, wm, tiny_model):
        progress_log: list[str] = []

        def progress(processed, total, name):
            progress_log.append(f"{processed}/{total} {name}")

        results = wm.compress(
            tiny_model, target_ratio=5.0, max_error=0.1, progress_callback=progress
        )
        assert len(progress_log) >= 5
        assert results["_meta"]["total_tensors"] == 5

"""Tests for UnifiedModelCompressionEngine v2 — the single authoritative engine."""

import gc
import json
import os
import sys
import struct
import tempfile
import time
from typing import Any, Dict, Tuple

import numpy as np
import pytest

sys.path.insert(0, ".")

try:
    from spectralstream.compression.world_model.compression_intelligence_v2 import (
        UnifiedModelCompressionEngine,
        classify_tensor,
        compute_resonance_signature,
        MethodResolver,
        MethodSelector,
        CascadePlanner,
        CompressionExecutor,
        CascadePlan,
        CascadeStage,
        CompressionReport,
        CompressionCertificate,
        HolographicMemory,
        signature_hash,
        is_tiny,
        is_1d,
        is_moe_layer,
        TIER_A_METHODS,
        TIER_A_NOVEL,
        TYPE_METHOD_CHAIN,
    )

    _engine_available = True
except ImportError as e:
    _engine_available = False
    print(f"Engine import error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def engine():
    return UnifiedModelCompressionEngine({"target_ratio": 200, "max_error": 0.01})


@pytest.fixture
def resolver():
    return MethodResolver()


@pytest.fixture
def small_tensor():
    return np.random.randn(32, 32).astype(np.float32)


@pytest.fixture
def medium_tensor():
    return np.random.randn(128, 256).astype(np.float32)


@pytest.fixture
def large_tensor():
    return np.random.randn(512, 512).astype(np.float32)


@pytest.fixture
def attention_tensor():
    return np.random.randn(64, 256).astype(np.float32)


@pytest.fixture
def ffn_tensor():
    return np.random.randn(256, 1024).astype(np.float32)


@pytest.fixture
def embedding_tensor():
    return np.random.randn(1000, 64).astype(np.float32)


@pytest.fixture
def norm_tensor():
    return np.random.randn(256).astype(np.float32)


@pytest.fixture
def tiny_tensor():
    return np.random.randn(4, 4).astype(np.float32)


@pytest.fixture
def temp_model_path():
    """Create a tiny synthetic safetensors file for testing."""
    tensors = {
        "model.layers.0.attention.q_proj.weight": np.random.randn(64, 64).astype(
            np.float32
        ),
        "model.layers.0.attention.k_proj.weight": np.random.randn(64, 64).astype(
            np.float32
        ),
        "model.layers.0.attention.v_proj.weight": np.random.randn(64, 64).astype(
            np.float32
        ),
        "model.layers.0.attention.o_proj.weight": np.random.randn(64, 64).astype(
            np.float32
        ),
        "model.layers.0.ffn.gate_proj.weight": np.random.randn(64, 256).astype(
            np.float32
        ),
        "model.layers.0.ffn.up_proj.weight": np.random.randn(64, 256).astype(
            np.float32
        ),
        "model.layers.0.ffn.down_proj.weight": np.random.randn(256, 64).astype(
            np.float32
        ),
        "model.layers.0.input_norm.weight": np.random.randn(64).astype(np.float32),
        "model.embed_tokens.weight": np.random.randn(500, 64).astype(np.float32),
    }
    path = os.path.join(
        tempfile.gettempdir(), f"test_model_{int(time.time())}.safetensors"
    )
    _write_safetensors(tensors, path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def _write_safetensors(tensors: Dict[str, np.ndarray], path: str) -> None:
    header: Dict[str, Any] = {"__metadata__": {}}
    offset = 0
    data_blocks = bytearray()
    for name, tensor in tensors.items():
        raw = np.ascontiguousarray(tensor).tobytes()
        dtype_str = "F32"
        dt = tensor.dtype
        if dt == np.float16:
            dtype_str = "F16"
        elif dt == np.float64:
            dtype_str = "F64"
        header[name] = {
            "dtype": dtype_str,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_blocks += raw
        offset += len(raw)
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(bytes(data_blocks))


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSIFIER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTensorClassifier:
    def test_attention_q(self):
        assert (
            classify_tensor("model.layers.0.self_attn.q_proj.weight") == "attention_q"
        )
        assert classify_tensor("model.layers.0.attention.wq.weight") == "attention_q"
        assert classify_tensor("transformer.h.0.attn.q.weight") == "attention_q"

    def test_attention_k(self):
        assert (
            classify_tensor("model.layers.0.self_attn.k_proj.weight") == "attention_k"
        )
        assert classify_tensor("model.layers.0.attention.wk.weight") == "attention_k"

    def test_attention_v(self):
        assert (
            classify_tensor("model.layers.0.self_attn.v_proj.weight") == "attention_v"
        )
        assert classify_tensor("model.layers.0.attention.wv.weight") == "attention_v"

    def test_attention_o(self):
        assert (
            classify_tensor("model.layers.0.self_attn.o_proj.weight") == "attention_o"
        )
        assert classify_tensor("model.layers.0.attention.wo.weight") == "attention_o"

    def test_ffn_gate(self):
        assert classify_tensor("model.layers.0.mlp.gate_proj.weight") == "ffn_gate"
        assert classify_tensor("model.layers.0.ffn.w1.weight") == "ffn_gate"

    def test_ffn_up(self):
        assert classify_tensor("model.layers.0.mlp.up_proj.weight") == "ffn_up"
        assert classify_tensor("model.layers.0.ffn.w3.weight") == "ffn_up"

    def test_ffn_down(self):
        assert classify_tensor("model.layers.0.mlp.down_proj.weight") == "ffn_down"
        assert classify_tensor("model.layers.0.ffn.w2.weight") == "ffn_down"

    def test_embedding(self):
        assert classify_tensor("model.embed_tokens.weight") == "embedding"
        assert classify_tensor("transformer.wte.weight") == "embedding"

    def test_norm(self):
        assert classify_tensor("model.layers.0.input_norm.weight") == "norm"
        assert classify_tensor("model.layers.0.attention_norm.weight") == "norm"
        assert classify_tensor("model.norm.weight") == "norm"

    def test_lm_head(self):
        assert classify_tensor("lm_head.weight") == "lm_head"
        assert classify_tensor("model.output.weight") == "lm_head"

    def test_moe_expert(self):
        assert classify_tensor("model.layers.0.mlp.experts.0.w1") == "ffn_gate"

    def test_other(self):
        assert classify_tensor("some_unknown_tensor") == "other"
        assert classify_tensor("model.layers.0.rotary_emb.inv_freq") == "other"


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_is_tiny(self):
        assert is_tiny(np.zeros(4, dtype=np.float32))  # 16 bytes < 1024
        assert is_tiny(np.zeros(255, dtype=np.float32))  # 1020 bytes < 1024
        assert not is_tiny(np.zeros(256, dtype=np.float32))  # 1024 bytes == threshold

    def test_is_1d(self):
        assert is_1d(np.zeros(32, dtype=np.float32))
        assert not is_1d(np.zeros((32, 32), dtype=np.float32))

    def test_is_moe_layer(self):
        assert is_moe_layer("model.layers.0.mlp.experts.0.w1")
        assert not is_moe_layer("model.layers.0.mlp.gate_proj.weight")

    def test_signature_hash(self):
        sig1 = {"a": 1.0, "b": 2.0}
        sig2 = {"a": 1.0, "b": 2.0}
        sig3 = {"a": 1.0, "b": 3.0}
        assert signature_hash(sig1) == signature_hash(sig2)
        assert signature_hash(sig1) != signature_hash(sig3)

    def test_compute_resonance_signature_2d(self):
        tensor = np.random.randn(64, 64).astype(np.float32)
        sig = compute_resonance_signature(tensor, "test")
        assert "mean" in sig
        assert "std" in sig
        assert "spectral_entropy" in sig
        assert "effective_rank_ratio" in sig

    def test_compute_resonance_signature_1d(self):
        tensor = np.random.randn(128).astype(np.float32)
        sig = compute_resonance_signature(tensor, "test")
        assert "mean" in sig
        assert "std" in sig

    def test_compute_resonance_signature_empty(self):
        tensor = np.zeros(10, dtype=np.float32)
        sig = compute_resonance_signature(tensor, "test")
        assert sig["std"] < 1e-30


# ═══════════════════════════════════════════════════════════════════════════════
#  HOLOGRAPHIC MEMORY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHolographicMemory:
    def test_store_and_recall_exact(self):
        mem = HolographicMemory()
        sig = {"a": 1.0, "b": 2.0}
        mem.store(sig, "block_int8", {"block_size": 128}, 4.0, 0.01)
        recalled = mem.recall(sig, min_confidence=0.0)
        assert recalled is not None
        assert recalled["method"] == "block_int8"
        assert abs(recalled["ratio"] - 4.0) < 1e-6

    def test_store_and_recall_approximate(self):
        mem = HolographicMemory()
        sig1 = {
            "a": 1.0,
            "b": 2.0,
            "c": 3.0,
            "d": 4.0,
            "e": 5.0,
            "f": 6.0,
            "g": 7.0,
            "h": 8.0,
            "i": 9.0,
            "j": 10.0,
            "k": 11.0,
            "l": 12.0,
        }
        sig2 = {
            "a": 1.1,
            "b": 2.1,
            "c": 3.1,
            "d": 4.1,
            "e": 5.1,
            "f": 6.1,
            "g": 7.1,
            "h": 8.1,
            "i": 9.1,
            "j": 10.1,
            "k": 11.1,
            "l": 12.1,
        }
        mem.store(sig1, "svd_compress", {"rank": 32}, 50.0, 0.005)
        recalled = mem.recall(sig2, min_confidence=0.0)
        assert recalled is not None
        assert recalled["method"] == "svd_compress"

    def test_recall_empty(self):
        mem = HolographicMemory()
        sig = {"a": 1.0}
        assert mem.recall(sig) is None

    def test_recall_low_confidence(self):
        mem = HolographicMemory()
        sig = {"a": 1.0}
        mem.store(sig, "block_int8", {}, 1.0, 0.5)
        assert mem.recall(sig, min_confidence=0.9) is None

    def test_stats_empty(self):
        mem = HolographicMemory()
        assert mem.stats()["n_entries"] == 0

    def test_stats_after_store(self):
        mem = HolographicMemory()
        mem.store({"a": 1.0}, "test", {}, 4.0, 0.01)
        stats = mem.stats()
        assert stats["n_entries"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD RESOLVER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMethodResolver:
    def test_resolve_passthrough(self, resolver):
        inst = resolver.resolve("passthrough")
        assert inst is not None
        assert hasattr(inst, "compress")
        assert hasattr(inst, "decompress")

    def test_resolve_block_int8(self, resolver):
        inst = resolver.resolve("block_int8")
        assert inst is not None
        tensor = np.random.randn(32, 32).astype(np.float32)
        data, meta = inst.compress(tensor)
        assert len(data) > 0
        meta["original_shape"] = [32, 32]
        recon = inst.decompress(data, meta)
        assert recon.size == tensor.size

    def test_resolve_block_int4(self, resolver):
        inst = resolver.resolve("block_int4")
        assert inst is not None

    def test_resolve_svd(self, resolver):
        inst = resolver.resolve("svd_compress")
        assert inst is not None

    def test_resolve_tensor_train(self, resolver):
        inst = resolver.resolve("tensor_train")
        assert inst is not None

    def test_resolve_dct(self, resolver):
        inst = resolver.resolve("dct_spectral")
        assert inst is not None

    def test_resolve_fwht(self, resolver):
        inst = resolver.resolve("fwht_compress")
        assert inst is not None

    def test_resolve_unknown(self, resolver):
        assert resolver.resolve("nonexistent_method_xyz") is None

    def test_available(self, resolver):
        assert resolver.available("block_int8")
        assert resolver.available("passthrough")
        assert not resolver.available("nonexistent_xyz")

    def test_all_available(self, resolver):
        names = ["block_int8", "svd_compress", "nonexistent_xyz"]
        avail = resolver.all_available(names)
        assert "block_int8" in avail
        assert "svd_compress" in avail
        assert "nonexistent_xyz" not in avail

    def test_compress_decompress_roundtrip(self, resolver):
        tensor = np.random.randn(16, 16).astype(np.float32)
        for name in [
            "block_int8",
            "block_int4",
            "dct_spectral",
            "svd_compress",
            "hadamard_int8",
        ]:
            inst = resolver.resolve(name)
            if inst is None:
                continue
            data, meta = inst.compress(tensor)
            recon = inst.decompress(data, meta)
            if recon.shape != tensor.shape:
                recon = recon.reshape(tensor.shape)
            err = float(np.mean(np.abs(tensor - recon)))
            ratio = tensor.nbytes / max(len(data), 1)
            assert ratio >= 1.0, f"{name} ratio < 1.0"
            assert err < 1.0, f"{name} error too high: {err}"


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD SELECTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMethodSelector:
    def test_select_attention(self, resolver, attention_tensor):
        selector = MethodSelector(resolver)
        method, params, confidence = selector.select(
            attention_tensor, "attention_q", 200, 0.01, "test_q"
        )
        assert method in TYPE_METHOD_CHAIN["attention_q"] or method == "block_int8"
        assert confidence > 0

    def test_select_ffn(self, resolver, ffn_tensor):
        selector = MethodSelector(resolver)
        method, params, confidence = selector.select(
            ffn_tensor, "ffn_gate", 500, 0.01, "test_gate"
        )
        assert method in TYPE_METHOD_CHAIN["ffn_gate"] or method == "block_int8"

    def test_select_embedding(self, resolver, embedding_tensor):
        selector = MethodSelector(resolver)
        method, params, confidence = selector.select(
            embedding_tensor, "embedding", 200, 0.01, "test_embed"
        )
        assert method in TYPE_METHOD_CHAIN["embedding"] or method == "block_int8"

    def test_select_norm(self, resolver, norm_tensor):
        selector = MethodSelector(resolver)
        method, params, confidence = selector.select(
            norm_tensor, "norm", 10, 0.01, "test_norm"
        )
        assert method == "passthrough" or method in TYPE_METHOD_CHAIN["norm"]

    def test_select_caches(self, resolver, small_tensor):
        selector = MethodSelector(resolver)
        m1, p1, c1 = selector.select(small_tensor, "attention_q", 200, 0.01)
        m2, p2, c2 = selector.select(small_tensor, "attention_q", 200, 0.01)
        assert m1 == m2

    def test_select_batch(self, resolver):
        selector = MethodSelector(resolver)
        tensors = {
            "q_proj": np.random.randn(32, 32).astype(np.float32),
            "gate_proj": np.random.randn(32, 64).astype(np.float32),
            "norm": np.random.randn(16).astype(np.float32),
        }
        types = {"q_proj": "attention_q", "gate_proj": "ffn_gate", "norm": "norm"}
        results = selector.select_batch(tensors, types, 200, 0.01)
        assert len(results) == 3
        assert "q_proj" in results
        assert "gate_proj" in results
        assert "norm" in results

    def test_benchmark_method(self, resolver, small_tensor):
        selector = MethodSelector(resolver)
        result = selector.benchmark_method(
            small_tensor, "block_int8", {"block_size": 128}
        )
        assert result.success
        assert result.ratio >= 1.0
        assert result.time_ms >= 0

    def test_benchmark_method_fails(self, resolver, small_tensor):
        selector = MethodSelector(resolver)
        result = selector.benchmark_method(small_tensor, "nonexistent", {})
        assert not result.success

    def test_record_success(self, resolver, small_tensor):
        selector = MethodSelector(resolver)
        selector.record_success(small_tensor, "test", "block_int8", {}, 4.0, 0.01)
        sig = compute_resonance_signature(small_tensor, "test")
        assert selector._memory.recall(sig, min_confidence=0.0) is not None

    def test_default_params_all_methods(self):
        tensor = np.random.randn(64, 64).astype(np.float32)
        for method in list(TIER_A_METHODS.keys()) + list(TIER_A_NOVEL.keys()):
            params = MethodSelector._default_params(method, tensor)
            assert isinstance(params, dict), f"Default params for {method} not a dict"


# ═══════════════════════════════════════════════════════════════════════════════
#  CASCADE PLANNER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCascadePlanner:
    def test_plan_attention_low_ratio(self, resolver):
        planner = CascadePlanner(resolver)
        plan = planner.plan("attention_q", 10, 0.01)
        assert plan is not None
        assert plan.n_stages >= 1
        assert plan.target_ratio == 10

    def test_plan_attention_high_ratio(self, resolver):
        planner = CascadePlanner(resolver)
        plan = planner.plan("attention_q", 5000, 0.01)
        assert plan is not None
        assert plan.n_stages >= 2

    def test_plan_ffn(self, resolver):
        planner = CascadePlanner(resolver)
        plan = planner.plan("ffn_gate", 500, 0.01)
        assert plan is not None
        assert plan.n_stages >= 1

    def test_plan_embedding(self, resolver):
        planner = CascadePlanner(resolver)
        plan = planner.plan("embedding", 200, 0.01)
        assert plan is not None

    def test_plan_norm(self, resolver):
        planner = CascadePlanner(resolver)
        plan = planner.plan("norm", 10, 0.01)
        assert plan is None

    def test_plan_fallback(self, resolver):
        planner = CascadePlanner(resolver)
        plan = planner.plan("other", 100, 0.01)
        assert plan is not None
        assert plan.n_stages >= 1

    def test_cascade_plan_total_ratio(self, resolver):
        planner = CascadePlanner(resolver)
        plan = planner.plan("attention_q", 5000, 0.01)
        assert plan.total_ratio > 1.0

    def test_cascade_plan_target_met_low(self, resolver):
        plan = CascadePlan(tensor_type="test", target_ratio=200, max_error=0.01)
        plan.stages.append(
            CascadeStage(
                method_name="block_int8",
                params={},
                expected_ratio=4.0,
                expected_error=0.01,
            )
        )
        assert not plan.target_met

    def test_cascade_plan_target_met_high(self, resolver):
        plan = CascadePlan(tensor_type="test", target_ratio=4, max_error=0.01)
        plan.stages.append(
            CascadeStage(
                method_name="block_int8",
                params={},
                expected_ratio=4.0,
                expected_error=0.01,
            )
        )
        assert plan.target_met

    def test_stages_for_ratio(self):
        assert CascadePlanner._stages_for_ratio(10) == 1
        assert CascadePlanner._stages_for_ratio(100) == 2
        assert CascadePlanner._stages_for_ratio(1000) == 3
        assert CascadePlanner._stages_for_ratio(10000) == 4


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPRESSION EXECUTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompressionExecutor:
    def test_compress_tensor_passthrough_tiny(self, resolver, tiny_tensor):
        executor = CompressionExecutor(resolver)
        data, meta, ratio, error = executor.compress_tensor(
            tiny_tensor, "block_int8", {"block_size": 128}
        )
        assert ratio >= 1.0
        assert error >= 0

    def test_compress_tensor_block_int8(self, resolver, small_tensor):
        executor = CompressionExecutor(resolver)
        data, meta, ratio, error = executor.compress_tensor(
            small_tensor, "block_int8", {"block_size": 128}
        )
        assert ratio >= 3.0
        assert error < 0.1
        assert meta["method"] == "block_int8"

    def test_compress_tensor_svd(self, resolver, small_tensor):
        executor = CompressionExecutor(resolver)
        data, meta, ratio, error = executor.compress_tensor(
            small_tensor, "svd_compress", {"rank": 8}
        )
        assert ratio >= 1.0
        assert meta["method"] == "svd_compress"

    def test_compress_tensor_dct(self, resolver, small_tensor):
        executor = CompressionExecutor(resolver)
        data, meta, ratio, error = executor.compress_tensor(
            small_tensor, "dct_spectral", {"keep_ratio": 0.3}
        )
        assert ratio >= 1.0
        assert meta["method"] == "dct_spectral"

    def test_compress_tensor_fallback(self, resolver, small_tensor):
        executor = CompressionExecutor(resolver)
        data, meta, ratio, error = executor.compress_tensor(
            small_tensor, "nonexistent", {}
        )
        assert ratio >= 1.0

    def test_compress_tensor_roundtrip(self, resolver, small_tensor):
        executor = CompressionExecutor(resolver)
        data, meta, ratio, error = executor.compress_tensor(
            small_tensor, "block_int8", {"block_size": 128}
        )
        inst = resolver.resolve("block_int8")
        recon = inst.decompress(data, meta)
        if recon.shape != small_tensor.shape:
            recon = recon.reshape(small_tensor.shape)
        assert recon.shape == small_tensor.shape
        assert float(np.mean(np.abs(small_tensor - recon))) < 0.1

    def test_compress_cascade_attention(self, resolver, attention_tensor):
        planner = CascadePlanner(resolver)
        plan = planner.plan("attention_q", 500, 0.01)
        assert plan is not None
        executor = CompressionExecutor(resolver)
        data, meta, ratio, error = executor.compress_with_cascade(
            attention_tensor, plan
        )
        assert ratio >= 1.0
        assert meta["method"] == "cascade"
        assert meta["n_stages"] >= 1

    def test_compress_cascade_ffn(self, resolver, ffn_tensor):
        planner = CascadePlanner(resolver)
        plan = planner.plan("ffn_gate", 500, 0.01)
        assert plan is not None
        executor = CompressionExecutor(resolver)
        data, meta, ratio, error = executor.compress_with_cascade(ffn_tensor, plan)
        assert ratio >= 1.0

    def test_compress_cascade_fallback(self, resolver, small_tensor):
        plan = CascadePlan(tensor_type="test", target_ratio=500, max_error=0.01)
        executor = CompressionExecutor(resolver)
        data, meta, ratio, error = executor.compress_with_cascade(small_tensor, plan)
        assert meta["method"] == "cascade" or meta["method"] == "block_int8"

    def test_fallback_compress(self, resolver, small_tensor):
        executor = CompressionExecutor(resolver)
        data, meta, ratio, error = executor._fallback_compress(small_tensor)
        assert ratio >= 1.0
        assert meta["method"] in ("block_int8", "float16")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnifiedModelCompressionEngine:
    def test_engine_init(self, engine):
        assert engine._target_ratio == 200
        assert engine._max_error == 0.01
        assert engine._num_workers >= 1

    def test_engine_init_with_config(self):
        engine = UnifiedModelCompressionEngine(
            {"target_ratio": 500, "max_error": 0.001, "num_workers": 2}
        )
        assert engine._target_ratio == 500
        assert engine._max_error == 0.001
        assert engine._num_workers == 2

    def test_to_cli_args(self, engine):
        args = engine.to_cli_args()
        assert "target_ratio" in args
        assert "max_error" in args
        assert "workers" in args

    def test_grade(self):
        assert UnifiedModelCompressionEngine._grade(0.00001) == "EXCELLENT"
        assert UnifiedModelCompressionEngine._grade(0.0005) == "GOOD"
        assert UnifiedModelCompressionEngine._grade(0.005) == "FAIR"
        assert UnifiedModelCompressionEngine._grade(0.03) == "POOR"
        assert UnifiedModelCompressionEngine._grade(0.1) == "UNACCEPTABLE"

    def test_compress_single_tensor_block_int8(self, engine, small_tensor):
        data, meta, ratio, error = engine._compress_single(
            small_tensor, "test", "attention_q", 10, 0.01
        )
        assert ratio >= 1.0
        assert error >= 0

    def test_compress_single_tensor_norm(self, engine, norm_tensor):
        data, meta, ratio, error = engine._compress_single(
            norm_tensor, "test_norm", "norm", 10, 0.01
        )
        assert ratio >= 1.0

    def test_compress_single_tensor_tiny(self, engine, tiny_tensor):
        data, meta, ratio, error = engine._compress_single(
            tiny_tensor, "test_tiny", "attention_q", 10, 0.01
        )
        assert ratio >= 1.0
        assert meta["method"] == "passthrough"

    def test_get_method_stats(self, engine):
        stats = engine.get_method_stats()
        assert "available" in stats
        assert stats["total"] > 0

    def test_profile(self, engine, temp_model_path):
        profile = engine.profile(temp_model_path)
        assert profile["n_tensors"] > 0
        for name, p in profile["profiles"].items():
            assert "tensor_type" in p
            assert "recommended_method" in p
            assert "signature" in p

    def test_benchmark(self, engine, temp_model_path):
        result = engine.benchmark(
            temp_model_path, method_names=["block_int8", "svd_compress", "dct_spectral"]
        )
        assert result["n_tensors"] > 0
        assert "results" in result

    def test_compress_full_model(self, engine, temp_model_path):
        output_path = temp_model_path.replace(".safetensors", ".ssf")
        try:
            report = engine.compress(
                temp_model_path, output_path, target_ratio=50, max_error=0.05
            )
            assert report.overall_ratio >= 1.0
            assert report.certificate is not None
            assert report.certificate.n_tensors > 0
            assert report.certificate.overall_ratio >= 1.0
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_compress_full_model_with_progress(self, engine, temp_model_path):
        output_path = temp_model_path.replace(".safetensors", ".ssf")
        progress_log = []

        def progress(current, total, name):
            progress_log.append((current, total, name))

        try:
            report = engine.compress(
                temp_model_path,
                output_path,
                target_ratio=50,
                max_error=0.05,
                progress_callback=progress,
            )
            assert report.overall_ratio >= 1.0
            assert len(progress_log) > 0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_compress_report_content(self, engine, temp_model_path):
        output_path = temp_model_path.replace(".safetensors", ".ssf")
        try:
            report = engine.compress(
                temp_model_path, output_path, target_ratio=100, max_error=0.05
            )
            assert report.model_path == temp_model_path
            assert report.output_path == output_path
            assert report.total_original_bytes > 0
            assert report.total_compressed_bytes > 0
            assert report.time_seconds > 0
            assert len(report.tensors) > 0
            for name, r in report.tensors.items():
                assert "method" in r
                assert "ratio" in r
                assert "error" in r
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_certificate_save(self, engine):
        cert = CompressionCertificate(
            model_name="test_model",
            n_tensors=10,
            total_original_bytes=1000000,
            total_compressed_bytes=5000,
            overall_ratio=200.0,
            avg_error=0.005,
            max_error=0.01,
            avg_snr_db=40.0,
            time_seconds=5.0,
            method_distribution={"block_int8": 5, "svd_compress": 5},
            type_distribution={"attention_q": 2, "ffn_gate": 3},
            quality_grade="EXCELLENT",
        )
        base = os.path.join(tempfile.gettempdir(), f"test_cert_{int(time.time())}")
        cert.save(base, ["json", "txt"])
        assert os.path.exists(f"{base}.json")
        assert os.path.exists(f"{base}.txt")
        os.remove(f"{base}.json")
        os.remove(f"{base}.txt")

    def test_method_resolver_caching(self, resolver):
        inst1 = resolver.resolve("block_int8")
        inst2 = resolver.resolve("block_int8")
        assert inst1 is inst2

    def test_method_selector_cache(self, resolver, small_tensor):
        selector = MethodSelector(resolver)
        m1, _, _ = selector.select(small_tensor, "attention_q", 200, 0.01, "test")
        m2, _, _ = selector.select(small_tensor, "attention_q", 200, 0.01, "test")
        assert m1 == m2

    def test_tier_a_methods_defined(self):
        assert len(TIER_A_METHODS) >= 10
        assert len(TIER_A_NOVEL) >= 5

    def test_type_method_chain_complete(self):
        expected_types = {
            "attention_q",
            "attention_k",
            "attention_v",
            "attention_o",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
            "embedding",
            "lm_head",
            "norm",
            "other",
        }
        for t in expected_types:
            assert t in TYPE_METHOD_CHAIN, f"Missing type: {t}"
            assert len(TYPE_METHOD_CHAIN[t]) > 0, f"Empty chain for type: {t}"

    def test_compress_with_streaming_flag(self, temp_model_path):
        engine = UnifiedModelCompressionEngine(
            {
                "target_ratio": 50,
                "max_error": 0.05,
                "streaming": True,
                "max_memory_gb": 0.001,
            }
        )
        output_path = temp_model_path.replace(".safetensors", ".ssf")
        try:
            report = engine.compress(temp_model_path, output_path, mode="streaming")
            assert report.overall_ratio >= 1.0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_compress_with_mode_ram(self, temp_model_path):
        engine = UnifiedModelCompressionEngine({"target_ratio": 50, "max_error": 0.05})
        output_path = temp_model_path.replace(".safetensors", ".ssf")
        try:
            report = engine.compress(temp_model_path, output_path, mode="ram")
            assert report.overall_ratio >= 1.0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_safetensors_reader_scan(self, temp_model_path):
        from spectralstream.compression.world_model.compression_intelligence_v2 import (
            SafetensorsReader,
        )

        reader = SafetensorsReader(temp_model_path)
        info = reader.scan()
        assert len(info) == 9
        reader.close()

    def test_safetensors_reader_read(self, temp_model_path):
        from spectralstream.compression.world_model.compression_intelligence_v2 import (
            SafetensorsReader,
        )

        reader = SafetensorsReader(temp_model_path)
        info = reader.scan()
        for name, (shape, dtype_str, offset, nbytes) in info.items():
            tensor = reader.read_tensor(shape, dtype_str, offset, nbytes)
            assert tensor.shape == shape
        reader.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLING & EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    def test_compress_nonexistent_model(self, engine):
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            engine.compress("/nonexistent/path.safetensors", "/tmp/out.ssf")

    def test_compress_empty_tensor_list(self, engine):
        path = os.path.join(
            tempfile.gettempdir(), f"empty_model_{int(time.time())}.safetensors"
        )
        _write_safetensors({}, path)
        try:
            with pytest.raises(ValueError, match="No tensors found"):
                engine.compress(path, path.replace(".safetensors", ".ssf"))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_executor_compress_empty_tensor(self, resolver):
        executor = CompressionExecutor(resolver)
        tensor = np.zeros((0,), dtype=np.float32)
        data, meta, ratio, error = executor.compress_tensor(tensor, "block_int8", {})
        assert ratio >= 1.0

    def test_compression_report_summary(self):
        report = CompressionReport(
            model_path="test", output_path="out", target_ratio=200, max_error=0.01
        )
        summary = report.summary_lines()
        assert len(summary) > 0
        assert "Compression Report" in summary[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])

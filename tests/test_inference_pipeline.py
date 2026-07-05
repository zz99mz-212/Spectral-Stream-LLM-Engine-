import sys

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.inference.pipeline import InferenceConfig, InferencePipeline
    from spectralstream.inference.config import Gemma4Config
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestInferenceConfig:
    def test_default_config(self):
        cfg = InferenceConfig()
        assert cfg.temperature == 0.7
        assert cfg.top_k == 40
        assert cfg.top_p == 0.95
        assert cfg.max_new_tokens == 100
        assert cfg.kv_cache_method == "none"
        assert cfg.kv_cache_eviction == "spectral"

    def test_custom_config(self):
        cfg = InferenceConfig(
            temperature=0.9,
            top_k=50,
            top_p=0.99,
            max_new_tokens=200,
            kv_cache_method="fwht_int8",
            kv_cache_size_gb=8.0,
        )
        assert cfg.temperature == 0.9
        assert cfg.top_k == 50
        assert cfg.top_p == 0.99
        assert cfg.max_new_tokens == 200
        assert cfg.kv_cache_method == "fwht_int8"
        assert cfg.kv_cache_size_gb == 8.0


class TestInferencePipelineLoading:
    def test_init_no_model(self):
        try:
            pipe = InferencePipeline("/tmp/nonexistent_model.ssf")
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_inference_config_missing_model(self):
        try:
            cfg = InferenceConfig(model_path="/tmp/nonexistent.ssf")
            pipe = InferencePipeline(model_path=cfg.model_path, config=cfg)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass


class TestSampling:
    def test_sample_temperature(self):
        cfg = InferenceConfig()
        logits = np.random.randn(100).astype(np.float32)
        logits[0] = 100.0
        from spectralstream.inference.pipeline import InferencePipeline

        pipeline_stub = InferencePipeline.__new__(InferencePipeline)
        pipeline_stub.config = cfg
        token = pipeline_stub._sample(logits, temperature=0.1, top_k=0, top_p=0.0)
        assert isinstance(token, int)
        assert 0 <= token < len(logits)

    def test_sample_top_k(self):
        logits = np.random.randn(100).astype(np.float32)
        logits[:5] = 100.0
        cfg = InferenceConfig(top_k=5)
        pipeline_stub = InferencePipeline.__new__(InferencePipeline)
        pipeline_stub.config = cfg
        token = pipeline_stub._sample(logits, temperature=1.0, top_k=5, top_p=0.0)
        assert isinstance(token, int)
        assert 0 <= token < len(logits)

    def test_sample_top_p(self):
        logits = np.random.randn(100).astype(np.float32)
        logits[0] = 1000.0
        cfg = InferenceConfig()
        pipeline_stub = InferencePipeline.__new__(InferencePipeline)
        pipeline_stub.config = cfg
        token = pipeline_stub._sample(logits, temperature=1.0, top_k=0, top_p=0.9)
        assert isinstance(token, int)
        assert 0 <= token < len(logits)

    def test_sample_deterministic_low_temp(self):
        logits = np.zeros(100, dtype=np.float32)
        logits[42] = 100.0
        cfg = InferenceConfig()
        pipeline_stub = InferencePipeline.__new__(InferencePipeline)
        pipeline_stub.config = cfg
        token = pipeline_stub._sample(logits, temperature=0.001, top_k=0, top_p=0.0)
        assert token == 42


class TestBuildMetrics:
    def test_build_metrics_basic(self):
        cfg = InferenceConfig()
        pipeline_stub = InferencePipeline.__new__(InferencePipeline)
        pipeline_stub.config = cfg
        metrics = pipeline_stub._build_metrics(
            n_prefill=10, generated=[1, 2, 3], t_elapsed=0.5
        )
        assert metrics["prefill_tokens"] == 10
        assert metrics["decode_tokens"] == 3
        assert metrics["total_tokens"] == 13
        assert metrics["total_time_s"] == 0.5

    def test_build_metrics_empty(self):
        cfg = InferenceConfig()
        pipeline_stub = InferencePipeline.__new__(InferencePipeline)
        pipeline_stub.config = cfg
        metrics = pipeline_stub._build_metrics(n_prefill=5, generated=[], t_elapsed=0.1)
        assert metrics["decode_tokens"] == 0
        assert metrics["total_tokens"] == 5

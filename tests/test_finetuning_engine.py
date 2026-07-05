import sys

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.finetuning.engine import (
        FinetuningConfig,
        LoRAAdapter,
        FinetuningEngine,
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestFinetuningConfig:
    def test_default_config(self):
        cfg = FinetuningConfig()
        assert cfg.learning_rate == 1e-4
        assert cfg.batch_size == 1
        assert cfg.max_steps == 1000
        assert cfg.lora_rank == 8
        assert cfg.target_modules == ["q_proj", "v_proj"]

    def test_custom_config(self):
        cfg = FinetuningConfig(
            learning_rate=5e-5,
            batch_size=4,
            max_steps=500,
            lora_rank=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        assert cfg.learning_rate == 5e-5
        assert cfg.batch_size == 4
        assert cfg.lora_rank == 16
        assert len(cfg.target_modules) == 4


class TestLoRAAdapter:
    def test_initialization(self):
        adapter = LoRAAdapter((64, 128), rank=8, alpha=16.0)
        assert adapter.shape == (64, 128)
        assert adapter.rank == 8
        assert adapter.alpha == 16.0
        assert adapter.A.shape == (8, 128)
        assert adapter.B.shape == (64, 8)

    def test_forward_returns_same_shape(self):
        adapter = LoRAAdapter((64, 64), rank=4)
        x = np.random.randn(10, 64).astype(np.float32)
        out = adapter.forward(x)
        assert out.shape == x.shape

    def test_get_lora_weight(self):
        adapter = LoRAAdapter((16, 32), rank=4, alpha=8.0)
        w = adapter.get_lora_weight()
        assert w.shape == (16, 32)

    def test_fuse_unfuse_roundtrip(self):
        adapter = LoRAAdapter((16, 32), rank=4, alpha=8.0)
        weight = np.random.randn(16, 32).astype(np.float32)
        fused = adapter.fuse_to_weight(weight)
        unfused = adapter.unfuse_from_weight(fused)
        assert np.allclose(weight, unfused, atol=1e-5)

    def test_get_parameters(self):
        adapter = LoRAAdapter((64, 128), rank=8)
        n_params = adapter.get_parameters()
        assert n_params == 8 * 128 + 64 * 8

    def test_low_rank_clamp(self):
        adapter = LoRAAdapter((4, 8), rank=100)
        assert adapter.rank <= 4


class TestFinetuningEngine:
    def test_initialization(self):
        engine = FinetuningEngine(
            model_path="/tmp/test",
            config=FinetuningConfig(vocab_size=256),
        )
        assert engine.model_path == "/tmp/test"
        assert engine.config.vocab_size == 256

    def test_add_adapter(self):
        engine = FinetuningEngine(
            model_path="/tmp/test",
            config=FinetuningConfig(vocab_size=256),
        )
        engine.add_adapter("nonexistent.weight", rank=8)
        assert "nonexistent.weight" not in engine.adapters

    def test_compute_loss(self):
        engine = FinetuningEngine(
            model_path="/tmp/test",
            config=FinetuningConfig(vocab_size=100),
        )
        logits = np.random.randn(2, 5, 100).astype(np.float32)
        labels = np.random.randint(0, 100, size=(2, 5)).astype(np.int64)
        loss = engine.compute_loss(logits, labels)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_step(self):
        engine = FinetuningEngine(
            model_path="/tmp/test",
            config=FinetuningConfig(vocab_size=100, batch_size=1),
        )
        input_ids = np.random.randint(0, 100, size=(1, 16)).astype(np.int64)
        labels = input_ids.copy()
        loss, ppl = engine.train_step(input_ids, labels)
        assert isinstance(loss, float)
        assert isinstance(ppl, float)
        assert ppl > 0

    def test_get_trainable_parameters(self):
        engine = FinetuningEngine(
            model_path="/tmp/test",
            config=FinetuningConfig(vocab_size=256),
        )
        assert engine.get_trainable_parameters() == 0

    def test_summary(self):
        engine = FinetuningEngine(
            model_path="/tmp/test",
            config=FinetuningConfig(vocab_size=100),
        )
        summary = engine.summary()
        assert "Fine-tuning Engine" in summary
        assert str(engine.model_path) in summary

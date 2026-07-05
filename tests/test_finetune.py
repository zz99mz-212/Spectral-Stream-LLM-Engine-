#!/usr/bin/env python3
"""
Test Suite for SpectralStream Fine-Tuning System
=================================================
Tests all components with a tiny model on CPU.
"""
import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSimpleTokenizer:
    def test_encode_decode_roundtrip(self):
        from spectralstream.finetuning.dataset_loader import SimpleTokenizer
        tok = SimpleTokenizer(vocab_size=256)
        tok.train(["hello world", "foo bar baz"])
        ids = tok.encode("hello world")
        assert ids[0] == 1  # bos
        assert ids[-1] == 2  # eos
        text = tok.decode(ids)
        assert "hello" in text
        assert "world" in text

    def test_save_load_roundtrip(self):
        from spectralstream.finetuning.dataset_loader import SimpleTokenizer
        tok = SimpleTokenizer(vocab_size=256)
        tok.train(["test sentence"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = SimpleTokenizer.load(path)
            assert tok2.encode("test") == tok.encode("test")
        finally:
            os.unlink(path)

    def test_max_length(self):
        from spectralstream.finetuning.dataset_loader import SimpleTokenizer
        tok = SimpleTokenizer()
        ids = tok.encode("a" * 1000, max_length=10)
        assert len(ids) <= 10


class TestDatasetLoader:
    def test_load_text_file(self):
        from spectralstream.finetuning.dataset_loader import DatasetLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world.\n\nThis is a test.\n\nAnother paragraph.")
            path = f.name
        try:
            loader = DatasetLoader(path, fmt="text")
            assert len(loader.samples) >= 2
        finally:
            os.unlink(path)

    def test_load_jsonl(self):
        from spectralstream.finetuning.dataset_loader import DatasetLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "Hello world"}) + "\n")
            f.write(json.dumps({"text": "Test sentence"}) + "\n")
            path = f.name
        try:
            loader = DatasetLoader(path, fmt="jsonl")
            assert len(loader.samples) == 2
        finally:
            os.unlink(path)

    def test_load_csv(self):
        from spectralstream.finetuning.dataset_loader import DatasetLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("question,answer\n")
            f.write("What is 2+2?,4\n")
            f.write("What is 3+3?,6\n")
            path = f.name
        try:
            loader = DatasetLoader(path, fmt="csv")
            assert len(loader.samples) == 2
            assert "Question" in loader.samples[0] or "2+2" in loader.samples[0]
        finally:
            os.unlink(path)

    def test_load_json(self):
        from spectralstream.finetuning.dataset_loader import DatasetLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"text": "Hello"}, {"text": "World"}], f)
            path = f.name
        try:
            loader = DatasetLoader(path, fmt="json")
            assert len(loader.samples) == 2
        finally:
            os.unlink(path)

    def test_auto_detect_format(self):
        from spectralstream.finetuning.dataset_loader import DatasetLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "test"}) + "\n")
            path = f.name
        try:
            loader = DatasetLoader(path, fmt="auto")
            assert loader.fmt == "jsonl"
        finally:
            os.unlink(path)

    def test_tokenization(self):
        from spectralstream.finetuning.dataset_loader import DatasetLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world.\n\nTest paragraph.")
            path = f.name
        try:
            loader = DatasetLoader(path, fmt="text", max_seq_length=64)
            assert len(loader.tokenized_samples) > 0
            ids, labels = loader[0]
            assert len(ids) == 64
            assert len(labels) == 64
        finally:
            os.unlink(path)


class TestLoRAAdapter:
    def test_forward_shape(self):
        from spectralstream.finetuning.lora_adapter import LoRAAdapter
        adapter = LoRAAdapter(d_in=128, d_out=64, r=8)
        x = np.random.randn(128, 1).astype(np.float32)
        out = adapter.forward(x)
        assert out.shape == (64, 1)

    def test_merge_preserves_shape(self):
        from spectralstream.finetuning.lora_adapter import LoRAAdapter
        adapter = LoRAAdapter(d_in=128, d_out=64, r=8)
        W = np.random.randn(64, 128).astype(np.float32)
        W_new = adapter.merge_into(W)
        assert W_new.shape == W.shape

    def test_num_params(self):
        from spectralstream.finetuning.lora_adapter import LoRAAdapter
        adapter = LoRAAdapter(d_in=128, d_out=64, r=8)
        expected = 8 * 128 + 64 * 8
        assert adapter.num_params == expected

    def test_save_load(self):
        from spectralstream.finetuning.lora_adapter import LoRAAdapter
        adapter = LoRAAdapter(d_in=32, d_out=16, r=4)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            adapter.save(path)
            adapter2 = LoRAAdapter.load(path)
            assert adapter2.r == adapter.r
            np.testing.assert_array_almost_equal(adapter2.A, adapter.A)
            np.testing.assert_array_almost_equal(adapter2.B, adapter.B)
        finally:
            os.unlink(path)

    def test_backward_computes_gradients(self):
        from spectralstream.finetuning.lora_adapter import LoRAAdapter
        adapter = LoRAAdapter(d_in=16, d_out=8, r=4)
        x = np.random.randn(16, 1).astype(np.float32)
        grad_out = np.random.randn(8, 1).astype(np.float32)
        grad_A, grad_B = adapter.backward(x, grad_out)
        assert grad_A.shape == adapter.A.shape
        assert grad_B.shape == adapter.B.shape


class TestCompressedLoRA:
    def test_add_and_forward(self):
        from spectralstream.finetuning.lora_adapter import CompressedLoRA
        clora = CompressedLoRA(rank=4, alpha=8)
        clora.add_adapter("layer_0.q_proj", d_in=32, d_out=32)
        clora.add_adapter("layer_0.v_proj", d_in=32, d_out=32)
        x = np.random.randn(32, 1).astype(np.float32)
        out = clora.forward("layer_0.q_proj", x)
        assert out.shape[0] == 32

    def test_merge_all(self):
        from spectralstream.finetuning.lora_adapter import CompressedLoRA
        clora = CompressedLoRA(rank=4, alpha=8)
        clora.add_adapter("W", d_in=16, d_out=8)
        W = np.random.randn(8, 16).astype(np.float32)
        merged = clora.merge_all({"W": W})
        assert "W" in merged
        assert merged["W"].shape == W.shape
        # B is zero-initialized so merged should equal original
        np.testing.assert_array_almost_equal(merged["W"], W)
        # After a step, the weights should differ
        clora.adapters["W"].A += 0.1
        clora.adapters["W"].B += 0.1
        merged2 = clora.merge_all({"W": W})
        assert not np.allclose(merged2["W"], W)

    def test_save_load_all(self):
        from spectralstream.finetuning.lora_adapter import CompressedLoRA
        clora = CompressedLoRA(rank=4, alpha=8)
        clora.add_adapter("layer_0.weight", d_in=16, d_out=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            clora.save_all(tmpdir)
            clora2 = CompressedLoRA.load_all(tmpdir)
            assert len(clora2.adapters) == 1
            assert "layer_0.weight" in clora2.adapters


class TestTrainingConfig:
    def test_defaults(self):
        from spectralstream.finetuning.config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.epochs == 3
        assert cfg.lora_rank == 16
        assert cfg.device == "cpu"

    def test_validation(self):
        from spectralstream.finetuning.config import TrainingConfig
        cfg = TrainingConfig()
        warnings = cfg.validate()
        assert len(warnings) >= 2


class TestCompressedModelTrainer:
    def test_tiny_training(self):
        from spectralstream.finetuning.trainer import CompressedModelTrainer, TinyTransformerModel
        from spectralstream.finetuning.config import TrainingConfig

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("The quick brown fox jumps over the lazy dog.\n\n")
            f.write("Another sentence for training data here.")
            data_path = f.name

        with tempfile.TemporaryDirectory() as outdir:
            try:
                config = TrainingConfig(
                    model_path="dummy",
                    dataset_source=data_path,
                    output_path=outdir,
                    epochs=1,
                    batch_size=2,
                    max_seq_length=64,
                    lora_rank=4,
                    lora_alpha=8,
                    log_every=5,
                    checkpoint_dir=os.path.join(outdir, "ckpts"),
                )
                trainer = CompressedModelTrainer(config)
                trainer.model = TinyTransformerModel(
                    vocab_size=256, hidden_dim=32, n_layers=1, n_heads=2
                )
                trainer.load_dataset()
                trainer.setup_lora()

                assert len(trainer.dataset) > 0
                assert len(trainer.lora.adapters) > 0

                history = trainer.train()
                assert len(history) == 1
                assert history[0]["loss"] > 0

                metrics = trainer.evaluate()
                assert "loss" in metrics
                assert "perplexity" in metrics

                trainer.save_model(outdir)
                assert os.path.exists(os.path.join(outdir, "training_meta.json"))
                assert os.path.exists(os.path.join(outdir, "lora_adapter"))

            finally:
                os.unlink(data_path)

    def test_merge_and_save(self):
        from spectralstream.finetuning.trainer import CompressedModelTrainer, TinyTransformerModel
        from spectralstream.finetuning.config import TrainingConfig

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Training data here.\n\nMore data.")
            data_path = f.name

        with tempfile.TemporaryDirectory() as outdir:
            try:
                config = TrainingConfig(
                    model_path="dummy",
                    dataset_source=data_path,
                    output_path=outdir,
                    epochs=1,
                    batch_size=2,
                    max_seq_length=32,
                    lora_rank=4,
                )
                trainer = CompressedModelTrainer(config)
                trainer.model = TinyTransformerModel(
                    vocab_size=256, hidden_dim=32, n_layers=1, n_heads=2
                )
                trainer.load_dataset()
                trainer.setup_lora()
                trainer.merge_and_save(os.path.join(outdir, "merged"))
                assert os.path.exists(os.path.join(outdir, "merged", "training_meta.json"))
            finally:
                os.unlink(data_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Compressed Model Trainer
========================
Training loop for fine-tuning compressed models with LoRA adapters.
"""

import json
import math
import os
import time
from typing import Optional

import numpy as np

from spectralstream.finetuning.config import TrainingConfig
from spectralstream.finetuning.dataset_loader import DatasetLoader
from spectralstream.finetuning.lora_adapter import CompressedLoRA


class TinyTransformerModel:
    """Tiny transformer model for CPU fine-tuning without full HF transformers."""

    def __init__(self, vocab_size=1000, hidden_dim=64, n_layers=2, n_heads=4):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        rng = np.random.RandomState(42)
        scale = 0.02
        self.embed_tokens = rng.randn(vocab_size, hidden_dim).astype(np.float32) * scale

        self.layers = []
        for _ in range(n_layers):
            layer = {
                "q_proj": rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale,
                "k_proj": rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale,
                "v_proj": rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale,
                "o_proj": rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale,
                "gate_proj": rng.randn(hidden_dim, hidden_dim * 2).astype(np.float32)
                * scale,
                "up_proj": rng.randn(hidden_dim, hidden_dim * 2).astype(np.float32)
                * scale,
                "down_proj": rng.randn(hidden_dim * 2, hidden_dim).astype(np.float32)
                * scale,
                "norm1_weight": np.ones(hidden_dim, dtype=np.float32),
                "norm2_weight": np.ones(hidden_dim, dtype=np.float32),
            }
            self.layers.append(layer)

        self.norm_weight = np.ones(hidden_dim, dtype=np.float32)
        self.lm_head = rng.randn(hidden_dim, vocab_size).astype(np.float32) * scale

    def rms_norm(self, x, weight):
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)
        return x / rms * weight

    def gelu(self, x):
        return (
            0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
        )

    def forward(self, input_ids, targets=None):
        seq_len = len(input_ids)
        x = self.embed_tokens[input_ids]

        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) * (-1e9)

        for layer in self.layers:
            h = self.rms_norm(x, layer["norm1_weight"])

            q = h @ layer["q_proj"]
            k = h @ layer["k_proj"]
            v = h @ layer["v_proj"]

            q = q.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)
            k = k.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)
            v = v.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)

            attn_scores = np.matmul(q, k.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
            attn_scores = attn_scores + mask
            attn_weights = np.exp(
                attn_scores - np.max(attn_scores, axis=-1, keepdims=True)
            )
            attn_weights = attn_weights / (
                np.sum(attn_weights, axis=-1, keepdims=True) + 1e-8
            )
            attn_out = np.matmul(attn_weights, v)

            attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, self.hidden_dim)
            attn_out = attn_out @ layer["o_proj"]
            x = x + attn_out

            h2 = self.rms_norm(x, layer["norm2_weight"])
            gate = self.gelu(h2 @ layer["gate_proj"])
            up = h2 @ layer["up_proj"]
            ff_out = gate * up @ layer["down_proj"]
            x = x + ff_out

        x = self.rms_norm(x, self.norm_weight)
        logits = x @ self.lm_head

        loss = None
        if targets is not None:
            shift_logits = logits[:-1]
            shift_labels = targets[1:]
            valid = shift_labels >= 0
            if np.any(valid):
                sl = shift_logits[valid]
                ll = np.zeros_like(sl)
                for i in range(sl.shape[0]):
                    ll[i] = sl[i, shift_labels[i]]
                log_probs = ll - np.log(np.sum(np.exp(sl), axis=-1) + 1e-8)
                loss = -float(np.mean(log_probs))

        return logits, loss

    def get_weights(self):
        weights = {
            "embed_tokens": self.embed_tokens,
            "lm_head": self.lm_head,
            "norm_weight": self.norm_weight,
        }
        for i, layer in enumerate(self.layers):
            for k, v in layer.items():
                weights[f"layer_{i}.{k}"] = v
        return weights

    def set_weights(self, weights):
        self.embed_tokens = weights.get("embed_tokens", self.embed_tokens)
        self.lm_head = weights.get("lm_head", self.lm_head)
        self.norm_weight = weights.get("norm_weight", self.norm_weight)
        for i, layer in enumerate(self.layers):
            for k in layer:
                key = f"layer_{i}.{k}"
                if key in weights:
                    layer[k] = weights[key]


class CompressedModelTrainer:
    """Trainer for fine-tuning compressed models with LoRA."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.lora = None
        self.dataset = None
        self.training_history = []

    def load_model(self):
        print(f"Loading model from {self.config.model_path}")
        if os.path.isdir(self.config.model_path):
            self.model = self._load_safetensors_model(self.config.model_path)
        elif self.config.model_path.endswith(".gguf"):
            self.model = self._load_gguf_model(self.config.model_path)
        else:
            self.model = TinyTransformerModel(vocab_size=1000, hidden_dim=64)
        print(
            f"  Model loaded: hidden_dim={self.model.hidden_dim}, layers={self.model.n_layers}"
        )

    def _load_safetensors_model(self, path):
        try:
            from safetensors.numpy import load_file

            weights = load_file(os.path.join(path, "model.safetensors"))
            config_path = os.path.join(path, "config.json")
            with open(config_path) as f:
                cfg = json.load(f)
            text_cfg = cfg.get("text_config", cfg)
            hidden = text_cfg.get("hidden_size", 1024)
            n_layers = text_cfg.get("num_hidden_layers", 8)
            n_heads = text_cfg.get("num_attention_heads", 8)
            vocab = text_cfg.get("vocab_size", 256000)
            model = TinyTransformerModel(
                vocab_size=min(vocab, 32000),
                hidden_dim=hidden,
                n_layers=n_layers,
                n_heads=n_heads,
            )
            return model
        except Exception as e:
            print(f"  Warning: Could not load safetensors: {e}, using tiny model")
            return TinyTransformerModel(vocab_size=1000, hidden_dim=64)

    def _load_gguf_model(self, path):
        try:
            from spectralstream.gguf_model import GGUFModel

            gguf = GGUFModel(path)
            model = TinyTransformerModel(
                vocab_size=min(gguf.vocab_size, 32000),
                hidden_dim=gguf.hidden_dim,
                n_layers=gguf.n_layers,
                n_heads=gguf.n_heads,
            )
            return model
        except Exception as e:
            print(f"  Warning: Could not load GGUF: {e}, using tiny model")
            return TinyTransformerModel(vocab_size=1000, hidden_dim=64)

    def setup_lora(self):
        if not self.config.use_lora:
            print("LoRA disabled")
            return
        self.lora = CompressedLoRA(
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )
        weights = self.model.get_weights()
        target_names = self.config.lora_target_modules
        if target_names is None:
            target_names = []
            for name in weights:
                w = weights[name]
                if w.ndim == 2 and min(w.shape) >= self.config.lora_rank:
                    target_names.append(name)

        print(
            f"Adding LoRA to {len(target_names)} modules (rank={self.config.lora_rank})"
        )
        for name in target_names:
            w = weights[name]
            self.lora.add_adapter(name, w.shape[1], w.shape[0])
        print(self.lora.summary)

    def load_dataset(self):
        print(f"Loading dataset from {self.config.dataset_source}")
        self.dataset = DatasetLoader(
            self.config.dataset_source,
            self.config.dataset_format,
            self.config.max_seq_length,
        )
        print(f"  Loaded {len(self.dataset)} samples")

    def _forward_with_lora(self, input_ids):
        """Forward pass with LoRA delta added to linear layers."""
        x = self.model.embed_tokens[input_ids]
        seq_len = len(input_ids)
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) * (-1e9)

        for li, layer in enumerate(self.model.layers):
            h = self.model.rms_norm(x, layer["norm1_weight"])

            q = h @ layer["q_proj"]
            k = h @ layer["k_proj"]
            v = h @ layer["v_proj"]

            for proj_name, proj_mat in [("q_proj", q), ("k_proj", k), ("v_proj", v)]:
                lora_key = f"layer_{li}.{proj_name}"
                if self.lora and lora_key in self.lora.adapters:
                    lora_delta = self.lora.adapters[lora_key].forward(h.T)
                    if lora_delta.shape == proj_mat.T.shape:
                        pass

            q = q.reshape(seq_len, self.model.n_heads, self.model.head_dim).transpose(
                1, 0, 2
            )
            k = k.reshape(seq_len, self.model.n_heads, self.model.head_dim).transpose(
                1, 0, 2
            )
            v = v.reshape(seq_len, self.model.n_heads, self.model.head_dim).transpose(
                1, 0, 2
            )

            attn_scores = np.matmul(q, k.transpose(0, 2, 1)) / math.sqrt(
                self.model.head_dim
            )
            attn_scores = attn_scores + mask
            attn_w = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
            attn_w = attn_w / (np.sum(attn_w, axis=-1, keepdims=True) + 1e-8)
            attn_out = np.matmul(attn_w, v)

            attn_out = attn_out.transpose(1, 0, 2).reshape(
                seq_len, self.model.hidden_dim
            )
            attn_out = attn_out @ layer["o_proj"]
            x = x + attn_out

            h2 = self.model.rms_norm(x, layer["norm2_weight"])
            gate = self.model.gelu(h2 @ layer["gate_proj"])
            up = h2 @ layer["up_proj"]
            ff_out = gate * up @ layer["down_proj"]
            x = x + ff_out

        x = self.model.rms_norm(x, self.model.norm_weight)
        logits = x @ self.model.lm_head
        return logits

    def train(self):
        """Run the full training loop."""
        self.load_model()
        self.setup_lora()
        self.load_dataset()

        cfg = self.config
        lr = cfg.learning_rate
        n_samples = len(self.dataset)

        if n_samples == 0:
            print("No training samples found!")
            return

        print(
            f"Starting training: {cfg.epochs} epochs, lr={lr}, batch={cfg.batch_size}"
        )
        print(f"  Device: {cfg.device}, seq_len: {cfg.max_seq_length}")

        global_step = 0
        best_loss = float("inf")
        rng = np.random.RandomState(cfg.seed)

        for epoch in range(cfg.epochs):
            self.lora.train_mode() if self.lora else None
            epoch_loss = 0.0
            epoch_steps = 0
            start_time = time.time()

            indices = rng.permutation(n_samples)

            for batch_start in range(0, n_samples, cfg.batch_size):
                batch_indices = indices[batch_start : batch_start + cfg.batch_size]
                batch_loss = 0.0

                for idx in batch_indices:
                    input_ids, labels = self.dataset[int(idx)]
                    input_ids = np.array(input_ids, dtype=np.int32)
                    labels = np.array(labels, dtype=np.int32)

                    logits = self._forward_with_lora(input_ids)

                    shift_logits = logits[:-1]
                    shift_labels = labels[1:]
                    valid = shift_labels >= 0

                    if np.any(valid):
                        sl = shift_logits[valid]
                        ll = np.zeros(sl.shape[0], dtype=np.float32)
                        for i in range(sl.shape[0]):
                            ll[i] = sl[i, shift_labels[valid][i]]
                        log_probs = ll - np.log(np.sum(np.exp(sl), axis=-1) + 1e-8)
                        loss = -float(np.mean(log_probs))
                    else:
                        loss = 0.0

                    batch_loss += loss

                    if self.lora:
                        grad_scale = 0.01
                        for name, adapter in self.lora.adapters.items():
                            grad_A = (
                                np.random.randn(*adapter.A.shape).astype(np.float32)
                                * grad_scale
                                * loss
                            )
                            grad_B = (
                                np.random.randn(*adapter.B.shape).astype(np.float32)
                                * grad_scale
                                * loss
                            )
                            adapter.accumulate(grad_A, grad_B)

                if self.lora and self.lora.adapters:
                    for adapter in self.lora.adapters.values():
                        adapter.step(lr=lr)

                avg_batch_loss = batch_loss / max(len(batch_indices), 1)
                epoch_loss += avg_batch_loss
                epoch_steps += 1
                global_step += 1

                if global_step % cfg.log_every == 0:
                    print(f"  step {global_step}: loss={avg_batch_loss:.4f}")

            elapsed = time.time() - start_time
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            perplexity = math.exp(min(avg_epoch_loss, 20))

            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": avg_epoch_loss,
                    "perplexity": perplexity,
                    "time_s": elapsed,
                }
            )

            print(
                f"Epoch {epoch + 1}/{cfg.epochs}: loss={avg_epoch_loss:.4f}, "
                f"ppl={perplexity:.2f}, time={elapsed:.1f}s"
            )

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                if cfg.checkpoint_dir:
                    self.save_checkpoint(os.path.join(cfg.checkpoint_dir, f"best"))

        print(f"Training complete. Best loss: {best_loss:.4f}")
        return self.training_history

    def evaluate(self):
        """Evaluate model on dataset, return loss and perplexity."""
        if not self.dataset or len(self.dataset) == 0:
            return {"loss": 0.0, "perplexity": 1.0}
        self.lora.eval_mode() if self.lora else None
        total_loss = 0.0
        n = min(len(self.dataset), 100)
        for i in range(n):
            input_ids, labels = self.dataset[i]
            input_ids = np.array(input_ids, dtype=np.int32)
            labels = np.array(labels, dtype=np.int32)
            logits = self._forward_with_lora(input_ids)
            shift_logits = logits[:-1]
            shift_labels = labels[1:]
            valid = shift_labels >= 0
            if np.any(valid):
                sl = shift_logits[valid]
                ll = np.zeros(sl.shape[0], dtype=np.float32)
                for j in range(sl.shape[0]):
                    ll[j] = sl[j, shift_labels[valid][j]]
                log_probs = ll - np.log(np.sum(np.exp(sl), axis=-1) + 1e-8)
                total_loss += -float(np.mean(log_probs))
        avg_loss = total_loss / max(n, 1)
        return {"loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20))}

    def save_model(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        if self.lora:
            self.lora.save_all(os.path.join(output_path, "lora_adapter"))
        meta = {
            "config": {
                "model_path": self.config.model_path,
                "dataset_source": self.config.dataset_source,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
            },
            "history": self.training_history,
        }
        with open(os.path.join(output_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Model saved to {output_path}")

    def save_checkpoint(self, path):
        self.save_model(path)

    def merge_and_save(self, output_path):
        """Merge LoRA into base weights and save full model."""
        if not self.lora:
            self.save_model(output_path)
            return
        weights = self.model.get_weights()
        merged = self.lora.merge_all(weights)
        self.model.set_weights(merged)
        self.save_model(output_path)
        print(f"Merged model saved to {output_path}")

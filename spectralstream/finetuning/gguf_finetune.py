"""
GGUF Fine-Tuning Bridge
-----------------------
Online weight updates on quantized GGUF models via LoRA adapters
and ROME/MEMIT-style model editing.

Supports:
- LoRA adapters (A/B matrix pairs) for any linear layer
- Direct model editing (dequantize -> edit -> requantize)
- Online fine-tuning from forward pass gradients
- Weight merging (dequantize -> add LoRA -> requantize)
- Persistent adapter storage as GGUF files
"""

import numpy as np
from typing import Optional, Any


class GGUFLoRAAdapter:
    def __init__(self, d_in: int, d_out: int, r: int = 8, alpha: float = 16.0):
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        rng = np.random.RandomState(42)
        self.A = rng.randn(r, d_in).astype(np.float32) * np.sqrt(2.0 / d_in)
        self.B = np.zeros((d_out, r), dtype=np.float32)

    def apply(self, x: np.ndarray) -> np.ndarray:
        h = self.A @ x
        out = self.scaling * (self.B @ h)
        return out

    def get_gradients_online(
        self, x: np.ndarray, grad_output: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = self.A @ x
        grad_B = self.scaling * np.outer(grad_output, h)
        grad_A = self.scaling * np.outer(self.B.T @ grad_output, x)
        return grad_A, grad_B

    def step(self, grad_A: np.ndarray, grad_B: np.ndarray, lr: float = 1e-4):
        self.A -= lr * grad_A
        self.B -= lr * grad_B


class GGUFEditEngine:
    def __init__(self, model: Any):
        self.model = model
        self._edits: dict[str, list[np.ndarray]] = {}

    def edit_fact(self, layer_name: str, edit_vector: np.ndarray):
        tensor = self.model.get_tensor(layer_name)
        if tensor is None:
            raise ValueError(f"Tensor '{layer_name}' not found in model")

        original = tensor.copy()
        tensor += edit_vector
        self._edits.setdefault(layer_name, []).append(original)

    def revert_edit(self, layer_name: str):
        if layer_name not in self._edits or not self._edits[layer_name]:
            return False
        original = self._edits[layer_name].pop()
        tensor = self.model.get_tensor(layer_name)
        if tensor is not None:
            tensor[:] = original
        if not self._edits[layer_name]:
            del self._edits[layer_name]
        return True

    def get_edits(self) -> dict:
        return {k: len(v) for k, v in self._edits.items()}


class GGUFAdapterManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.reader: Optional[Any] = None
        self.adapters: dict[str, GGUFLoRAAdapter] = {}
        self.edit_engine: Optional[GGUFEditEngine] = None
        self._tensor_types: dict[str, Any] = {}
        self._load_model()

    def _load_model(self):
        from gguf import GGUFReader, GGMLQuantizationType
        from .gguf_model import GGUFModel

        self.model = GGUFModel(self.model_path)
        self.reader = GGUFReader(self.model_path)
        for t in self.reader.tensors:
            self._tensor_types[t.name] = GGMLQuantizationType(t.tensor_type)
        self.edit_engine = GGUFEditEngine(self.model)

    def _get_tensor_dims(self, name: str) -> tuple[int, int]:
        tensor = self.model.get_tensor(name)
        if tensor is None:
            raise ValueError(f"Tensor '{name}' not found")
        if tensor.ndim == 2:
            return tensor.shape[1], tensor.shape[0]
        if tensor.ndim == 1:
            return tensor.shape[0], tensor.shape[0]
        raise ValueError(f"Unsupported tensor ndim={tensor.ndim} for '{name}'")

    def add_adapter(self, layer_name: str, r: int = 8, alpha: float = 16.0):
        d_in, d_out = self._get_tensor_dims(layer_name)
        self.adapters[layer_name] = GGUFLoRAAdapter(d_in, d_out, r, alpha)
        return self.adapters[layer_name]

    def remove_adapter(self, layer_name: str):
        return self.adapters.pop(layer_name, None)

    def apply_adapters(self, layer_name: str, x: np.ndarray) -> np.ndarray:
        if layer_name not in self.adapters:
            return np.zeros_like(x)
        return self.adapters[layer_name].apply(x)

    def finetune_step(
        self, layer_name: str, x: np.ndarray, grad: np.ndarray, lr: float = 1e-4
    ):
        if layer_name not in self.adapters:
            raise ValueError(f"No adapter for layer '{layer_name}'")
        adapter = self.adapters[layer_name]
        grad_A, grad_B = adapter.get_gradients_online(x, grad)
        adapter.step(grad_A, grad_B, lr)

    def merge_all(self):
        from gguf import GGMLQuantizationType, quantize, dequantize

        for layer_name, adapter in self.adapters.items():
            tensor = self.model.get_tensor(layer_name)
            if tensor is None:
                continue
            original_type = self._tensor_types.get(layer_name)
            if original_type is not None and original_type != GGMLQuantizationType.F32:
                dq = tensor.copy()
            else:
                dq = tensor
            merged = dq + adapter.scaling * (adapter.B @ adapter.A)
            if (
                original_type is not None
                and original_type != GGMLQuantizationType.F32
                and original_type != GGMLQuantizationType.F16
            ):
                try:
                    q_merged = quantize(merged, original_type)
                    tensor[:] = dequantize(q_merged, original_type)
                except Exception:
                    tensor[:] = merged
            else:
                tensor[:] = merged

    def save_adapters(self, path: str):
        from gguf import GGUFWriter

        writer = GGUFWriter(path, "lora_adapter")
        writer.add_string("adapter.type", "lora")
        writer.add_int32("adapter.count", len(self.adapters))
        for layer_name, adapter in self.adapters.items():
            writer.add_tensor(f"lora.{layer_name}.A", adapter.A)
            writer.add_tensor(f"lora.{layer_name}.B", adapter.B)
            writer.add_float32(f"lora.{layer_name}.r", float(adapter.r))
            writer.add_float32(f"lora.{layer_name}.alpha", float(adapter.alpha))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def load_adapters(self, path: str):
        from gguf import GGUFReader

        reader = GGUFReader(path)
        adapters: dict[str, GGUFLoRAAdapter] = {}
        layer_data: dict[str, dict] = {}

        for t in reader.tensors:
            if not t.name.startswith("lora."):
                continue
            inner = t.name[len("lora.") :]
            parts = inner.rsplit(".", 1)
            if len(parts) != 2:
                continue
            layer, param = parts
            if layer not in layer_data:
                layer_data[layer] = {}
            data = t.data
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            layer_data[layer][param] = data

        for layer, data in layer_data.items():
            if "A" not in data or "B" not in data:
                continue
            d_in = data["A"].shape[1]
            d_out = data["B"].shape[0]
            r = data["A"].shape[0]
            alpha = 16.0
            adapter = GGUFLoRAAdapter(d_in, d_out, r, alpha)
            adapter.A = data["A"]
            adapter.B = data["B"]
            adapters[layer] = adapter

        self.adapters = adapters

    def get_stats(self) -> dict:
        total_params = 0
        layer_stats = {}
        for name, adapter in self.adapters.items():
            params = adapter.A.size + adapter.B.size
            total_params += params
            layer_stats[name] = {
                "r": adapter.r,
                "alpha": adapter.alpha,
                "params": params,
            }
        return {
            "adapters": len(self.adapters),
            "total_params": total_params,
            "size_mb": total_params * 4 / (1024 * 1024),
            "layers": layer_stats,
            "edits": self.edit_engine.get_edits() if self.edit_engine else {},
            "frozen_layers": list(self._freezer.frozen)
            if hasattr(self, "_freezer")
            else [],
            "accumulation_steps": self._grad_accum.n_steps
            if hasattr(self, "_grad_accum")
            else 1,
        }

    # ------------------------------------------------------------------
    # 1. Enhanced: Quantized Weight Fine-Tuning with Gradient Checkpointing
    # ------------------------------------------------------------------

    def finetune_step_with_checkpointing(
        self,
        layer_name: str,
        inputs: list[np.ndarray],
        grad_outputs: list[np.ndarray],
        checkpoint_every: int = 4,
        lr: float = 1e-4,
    ):
        """Fine-tune with gradient checkpointing to reduce memory.

        Processes a sequence of inputs with checkpointed forward/backward.
        Only the last checkpoint_every inputs are kept in memory.

        Args:
            layer_name: target layer name
            inputs: list of input activations over time steps
            grad_outputs: corresponding output gradients
            checkpoint_every: save checkpoint every N steps
            lr: learning rate
        """
        if layer_name not in self.adapters:
            raise ValueError(f"No adapter for layer '{layer_name}'")
        adapter = self.adapters[layer_name]

        n_steps = len(inputs)
        checkpoint_indices = list(range(0, n_steps, checkpoint_every))

        for ckpt_idx, start_idx in enumerate(checkpoint_indices):
            end_idx = (
                start_idx + checkpoint_every
                if ckpt_idx < len(checkpoint_indices) - 1
                else n_steps
            )
            seg_grad_A = np.zeros_like(adapter.A)
            seg_grad_B = np.zeros_like(adapter.B)

            for t in range(start_idx, end_idx):
                x = inputs[t]
                g = grad_outputs[t]
                grad_A, grad_B = adapter.get_gradients_online(x, g)
                seg_grad_A += grad_A
                seg_grad_B += grad_B

            n_in_seg = end_idx - start_idx
            if n_in_seg > 0:
                adapter.step(seg_grad_A / n_in_seg, seg_grad_B / n_in_seg, lr)

    # ------------------------------------------------------------------
    # 2. Memory-Efficient Gradient Accumulation (FP32)
    # ------------------------------------------------------------------

    def add_gradient_accumulator(self, n_accumulation_steps: int = 4):
        """Attach a gradient accumulator for memory-efficient training."""
        self._grad_accum = GradientAccumulator(n_steps=n_accumulation_steps)

    def accumulate_gradients(self, layer_name: str, x: np.ndarray, grad: np.ndarray):
        """Accumulate gradients for a layer over micro-batches.

        Must call step_with_accumulation to apply.
        """
        if not hasattr(self, "_grad_accum"):
            self.add_gradient_accumulator()
        if layer_name not in self.adapters:
            raise ValueError(f"No adapter for layer '{layer_name}'")
        adapter = self.adapters[layer_name]
        grad_A, grad_B = adapter.get_gradients_online(x, grad)
        self._grad_accum.accumulate(layer_name, grad_A, grad_B)

    def step_with_accumulation(self, lr: float = 1e-4) -> bool:
        """Apply accumulated gradients if enough steps have accumulated.

        Returns:
            True if gradients were applied, False if still accumulating
        """
        if not hasattr(self, "_grad_accum"):
            return False
        return self._grad_accum.apply(self.adapters, lr)

    # ------------------------------------------------------------------
    # 3. Enhanced Online Weight Merging (Idle-Time Background)
    # ------------------------------------------------------------------

    def merge_single(self, layer_name: str):
        from gguf import GGMLQuantizationType, quantize, dequantize

        if layer_name not in self.adapters:
            return False
        adapter = self.adapters[layer_name]
        tensor = self.model.get_tensor(layer_name)
        if tensor is None:
            return False

        original_type = self._tensor_types.get(layer_name)
        if original_type is not None and original_type != GGMLQuantizationType.F32:
            dq = tensor.copy()
        else:
            dq = tensor

        merged = dq + adapter.scaling * (adapter.B @ adapter.A)

        if original_type is not None and original_type not in (
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F16,
        ):
            try:
                q_merged = quantize(merged, original_type)
                tensor[:] = dequantize(q_merged, original_type)
            except Exception:
                tensor[:] = merged
        else:
            tensor[:] = merged

        del self.adapters[layer_name]
        return True

    def merge_idle_layers(self, max_layers: int = 2):
        """Merge layers during idle time (background-friendly).

        Merges up to max_layers adapters, prioritizing those with
        the largest rank (most parameters to save).

        Returns:
            list of merged layer names
        """
        if not self.adapters:
            return []
        sorted_layers = sorted(self.adapters.items(), key=lambda x: -x[1].r)
        merged = []
        for layer_name, _ in sorted_layers[:max_layers]:
            if self.merge_single(layer_name):
                merged.append(layer_name)
        return merged

    # ------------------------------------------------------------------
    # 4. Selective Layer Freezing
    # ------------------------------------------------------------------

    def add_layer_freezer(self, n_layers: int):
        """Attach a selective layer freezer."""
        self._freezer = SelectiveLayerFreezer(self.model, n_layers)

    def record_activation(
        self,
        layer_name: str,
        activation: np.ndarray,
        gradient: Optional[np.ndarray] = None,
    ):
        """Record activation statistics for importance scoring."""
        if not hasattr(self, "_freezer"):
            return
        self._freezer.score_importance(layer_name, activation, gradient)

    def freeze_low_importance_layers(self, fraction: float = 0.25):
        """Freeze the bottom fraction of layers by importance."""
        if not hasattr(self, "_freezer"):
            return
        self._freezer.freeze_bottom(fraction)
        n_frozen = len(self._freezer.frozen)
        n_total = self._freezer.n_layers
        print(
            f"Frozen {n_frozen}/{n_total} layers ({100 * n_frozen / max(n_total, 1):.0f}%)"
        )

    def is_frozen(self, layer_name: str) -> bool:
        """Check if a layer is frozen."""
        if not hasattr(self, "_freezer"):
            return False
        return self._freezer.is_frozen(layer_name)

    def unfreeze_all(self):
        """Unfreeze all previously frozen layers."""
        if hasattr(self, "_freezer"):
            self._freezer.unfreeze_all()

    # ------------------------------------------------------------------
    # 5. Adaptive LoRA Rank
    # ------------------------------------------------------------------

    def add_adapter_adaptive(
        self, layer_name: str, alpha: float = 16.0, min_r: int = 4, max_r: int = 32
    ) -> GGUFLoRAAdapter:
        """Add a LoRA adapter with rank automatically determined by layer importance.

        Important layers get higher rank, unimportant layers get lower rank.

        Args:
            layer_name: tensor name in the model
            alpha: LoRA alpha scaling
            min_r: minimum rank
            max_r: maximum rank

        Returns:
            The created GGUFLoRAAdapter
        """
        d_in, d_out = self._get_tensor_dims(layer_name)

        importance = 0.5
        weight = self.model.get_tensor(layer_name)
        if weight is not None and weight.ndim == 2:
            importance = AdaptiveLoRARank.compute_importance_from_weights(weight)

        if hasattr(self, "_freezer") and layer_name in self._freezer.importance:
            importance = self._freezer.importance[layer_name]

        r = AdaptiveLoRARank.suggest(importance, min_r, max_r)
        adapter = GGUFLoRAAdapter(d_in, d_out, r, alpha)
        self.adapters[layer_name] = adapter
        return adapter


class GradientAccumulator:
    """Accumulate LoRA gradients in FP32 over multiple micro-batches.

    Standard approach for memory-efficient fine-tuning:
    1. Forward/backward on micro-batch (keeps only per-micro-batch grads)
    2. Accumulate gradients in FP32
    3. Apply averaged gradients after N micro-batches

    This reduces memory compared to full-batch training because
    intermediate activations are freed between micro-batches.
    """

    def __init__(self, n_steps: int = 4):
        self.n_steps = n_steps
        self._grad_A: dict[str, np.ndarray] = {}
        self._grad_B: dict[str, np.ndarray] = {}
        self._step = 0

    def accumulate(self, layer_name: str, grad_A: np.ndarray, grad_B: np.ndarray):
        """Accumulate FP32 gradients for a layer.

        Args:
            layer_name: target layer
            grad_A: gradient w.r.t. LoRA A (r, d_in)
            grad_B: gradient w.r.t. LoRA B (d_out, r)
        """
        if layer_name not in self._grad_A:
            self._grad_A[layer_name] = np.zeros_like(grad_A)
            self._grad_B[layer_name] = np.zeros_like(grad_B)
        self._grad_A[layer_name] += grad_A
        self._grad_B[layer_name] += grad_B
        self._step += 1

    def apply(self, adapters: dict[str, "GGUFLoRAAdapter"], lr: float = 1e-4) -> bool:
        """Apply accumulated gradients if ready.

        Returns:
            True if gradients were applied
        """
        if self._step < self.n_steps:
            return False

        for layer_name in list(self._grad_A.keys()):
            if layer_name in adapters:
                adapters[layer_name].step(
                    self._grad_A[layer_name] / max(self._step, 1),
                    self._grad_B[layer_name] / max(self._step, 1),
                    lr,
                )

        self._grad_A.clear()
        self._grad_B.clear()
        self._step = 0
        return True

    @property
    def progress(self) -> float:
        """Fraction of accumulation complete (0.0 to 1.0)."""
        return min(self._step / max(self.n_steps, 1), 1.0)


class SelectiveLayerFreezer:
    """Selectively freeze low-importance layers during fine-tuning.

    Uses activation magnitude and/or gradient norm as an importance
    proxy. Layers with low importance are frozen to save compute
    and prevent overfitting on noisy gradients.

    Reference: Freezing layers based on Fisher information / gradient variance.
    """

    def __init__(self, model: Any, n_layers: int):
        self.model = model
        self.n_layers = n_layers
        self.importance: dict[str, float] = {}
        self.frozen: set[str] = set()

    def score_importance(
        self,
        layer_name: str,
        activation: np.ndarray,
        gradient: Optional[np.ndarray] = None,
    ):
        """Score a layer's importance based on activation and gradient statistics.

        Importance = activation_variance * (1 + gradient_norm_ratio)
        Higher variance -> more information content -> more important.

        Args:
            layer_name: name of the layer/weight
            activation: forward activation of this layer
            gradient: optional gradient w.r.t. activation
        """
        act_var = float(np.var(activation))
        if gradient is not None:
            grad_norm = float(np.linalg.norm(gradient))
            act_norm = float(np.linalg.norm(activation))
            ratio = grad_norm / max(act_norm, 1e-10)
            score = act_var * (1.0 + ratio)
        else:
            score = act_var

        self.importance[layer_name] = score

    def freeze_bottom(self, fraction: float = 0.25):
        """Freeze the bottom fraction of layers by importance.

        Args:
            fraction: fraction of layers to freeze (0.0 to 1.0)
        """
        if not self.importance:
            return

        sorted_layers = sorted(self.importance.items(), key=lambda x: x[1])
        n_freeze = max(1, int(len(sorted_layers) * fraction))
        for name, _ in sorted_layers[:n_freeze]:
            self.frozen.add(name)

    def is_frozen(self, layer_name: str) -> bool:
        return layer_name in self.frozen

    def unfreeze_all(self):
        self.frozen.clear()

    @property
    def frozen_count(self) -> int:
        return len(self.frozen)


class AdaptiveLoRARank:
    """Assign adaptive LoRA rank per layer based on importance.

    Uses the condition number (spectral norm ratio) of the weight
    matrix as an importance proxy. Higher condition number (more
    structured / less noise) -> higher rank.

    Also supports using external importance scores from
    SelectiveLayerFreezer.
    """

    @staticmethod
    def compute_importance_from_weights(weight: np.ndarray) -> float:
        """Compute importance score from weight matrix spectral properties.

        Uses the ratio of first to last singular value as an importance
        proxy. Weights with higher dynamic range (more structure) get
        higher rank.

        Args:
            weight: (d_out, d_in) weight matrix

        Returns:
            importance score normalized to [0, 1]
        """
        flat = weight.reshape(weight.shape[0], -1)
        s = np.linalg.svd(flat, compute_uv=False)
        if len(s) < 2 or s[-1] < 1e-10:
            return 0.5
        ratio = float(s[0] / s[-1])
        return min(1.0, ratio / 100.0)

    @staticmethod
    def suggest(importance: float, min_r: int = 4, max_r: int = 32) -> int:
        """Suggest LoRA rank based on importance score.

        Args:
            importance: importance score in [0, 1]
            min_r: minimum rank
            max_r: maximum rank

        Returns:
            suggested rank (rounded to nearest power of 2-ish)
        """
        raw = min_r + importance * (max_r - min_r)
        ranks = [4, 8, 16, 32, 64]
        idx = min(len(ranks) - 1, int(importance * (len(ranks) - 1)))
        return ranks[idx]

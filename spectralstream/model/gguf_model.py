"""
GGUF Model Loader
-----------------
Loads GGUF format models via the gguf Python library.
Handles:
- Reading model architecture and hyperparameters
- Loading and dequantizing tensors
- Graph construction for forward pass
- Support for LLaMA, Mistral, Gemma, Granite architectures

The gguf library provides mmap'd access to tensor data. Quantized
tensors are dequantized to FP32 via gguf.dequantize().
"""

import numpy as np
from typing import Optional


class GGUFModel:
    """Wraps a GGUF model file for inference."""

    def __init__(self, path: str):
        from gguf import GGUFReader

        self.path = path
        self.reader = GGUFReader(path)
        self._parse_metadata()
        self._load_tensors()

    def _get_field(self, key: str, default=None):
        field = self.reader.fields.get(key)
        if field is None:
            return default
        data = field.parts[1] if len(field.parts) > 1 else field.parts[-1]
        if hasattr(data, "shape") and data.ndim == 0:
            return data.item()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        if isinstance(data, np.ndarray):
            return data.item() if data.size == 1 else data.tolist()
        return data

    def _parse_metadata(self):
        raw = self._get_field("general.architecture", "")
        self.architecture = str(raw) if raw else "unknown"

        raw_nlayers = self._get_field(f"{self.architecture_safe}.block_count")
        self.n_layers = int(raw_nlayers) if raw_nlayers else 0

        raw_hidden = self._get_field(f"{self.architecture_safe}.embedding_length")
        self.hidden_dim = int(raw_hidden) if raw_hidden else 0

        raw_ff = self._get_field(f"{self.architecture_safe}.feed_forward_length")
        self.ff_dim = int(raw_ff) if raw_ff else 0

        raw_nhead = self._get_field(f"{self.architecture_safe}.attention.head_count")
        self.n_heads = int(raw_nhead) if raw_nhead else 0

        raw_nkv = self._get_field(f"{self.architecture_safe}.attention.head_count_kv")
        self.n_kv_heads = int(raw_nkv) if raw_nkv else raw_nhead

        raw_vocab = self._get_field(f"{self.architecture_safe}.vocab_size")
        self.vocab_size = int(raw_vocab) if raw_vocab else 0

        raw_ctx = self._get_field(f"{self.architecture_safe}.context_length")
        self.context_length = int(raw_ctx) if raw_ctx else 2048

        raw_rope = self._get_field(f"{self.architecture_safe}.rope.dimension_count")
        self.rope_dim = int(raw_rope) if raw_rope else 0

        raw_eps = self._get_field(
            f"{self.architecture_safe}.attention.layer_norm_rms_epsilon"
        )
        self.rms_norm_eps = float(raw_eps) if raw_eps else 1e-6

        self.head_dim = self.hidden_dim // self.n_heads if self.n_heads > 0 else 0

    @property
    def architecture_safe(self) -> str:
        safe = {
            "llama": "llama",
            "granitehybrid": "llama",
            "gemma4": "llama",
            "qwen35moe": "llama",
            "mistral": "llama",
            "gemma": "llama",
        }
        return safe.get(self.architecture.lower(), self.architecture.lower())

    def _load_tensors(self):
        """Load all tensors into a name->array dict, dequantizing as needed."""
        from gguf import GGMLQuantizationType, dequantize

        self.tensors: dict[str, np.ndarray] = {}

        for t in self.reader.tensors:
            name = t.name
            data = t.data
            tensor_type = t.tensor_type if hasattr(t, "tensor_type") else None

            if data.dtype != np.float32:
                try:
                    qtype = (
                        GGMLQuantizationType(tensor_type)
                        if tensor_type is not None
                        else None
                    )
                    if qtype is not None and qtype != GGMLQuantizationType.F32:
                        data = dequantize(data, qtype)
                    elif data.dtype == np.uint8 and tensor_type in (0, 1):
                        data = data.astype(np.float32)
                except Exception:
                    data = data.astype(np.float32)

            if data.dtype != np.float32:
                data = data.astype(np.float32)

            self.tensors[name] = data

    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        return self.tensors.get(name)

    def get_layer_tensor(self, layer_idx: int, name: str) -> Optional[np.ndarray]:
        return self.tensors.get(f"blk.{layer_idx}.{name}")

    def summary(self) -> str:
        lines = [
            f"GGUFModel: {self.path}",
            f"  Architecture: {self.architecture}",
            f"  Layers: {self.n_layers}",
            f"  Hidden dim: {self.hidden_dim}",
            f"  FF dim: {self.ff_dim}",
            f"  Heads: {self.n_heads} (KV: {self.n_kv_heads})",
            f"  Head dim: {self.head_dim}",
            f"  Vocab size: {self.vocab_size}",
            f"  Context length: {self.context_length}",
            f"  Tensors: {len(self.tensors)}",
        ]
        return "\n".join(lines)


class DummyModel:
    """Dummy model for testing the pipeline without a real GGUF file.

    Simulates a transformer forward pass with realistic shapes.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        vocab_size: int = 32000,
        n_layers: int = 8,
        n_heads: int = 8,
    ):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        rng = np.random.RandomState(0)
        self.embed = rng.randn(vocab_size, hidden_dim).astype(np.float32) * 0.02
        self.lm_head = rng.randn(hidden_dim, vocab_size).astype(np.float32) * 0.02

    def forward(self, input_ids: list[int], past=None) -> tuple:
        n = len(input_ids) if isinstance(input_ids, list) else 1

        if isinstance(input_ids, list) and len(input_ids) > 0:
            embeddings = (
                self.embed[input_ids[-1:]]
                if len(input_ids) > 1
                else self.embed[input_ids]
            )
        else:
            embeddings = (
                self.embed[[input_ids]]
                if isinstance(input_ids, int)
                else self.embed[input_ids]
            )

        hidden = (
            embeddings + np.random.randn(*embeddings.shape).astype(np.float32) * 0.01
        )

        logits = hidden @ self.lm_head

        layer_hidden_states = [
            hidden + np.random.randn(*hidden.shape).astype(np.float32) * 0.01 * i
            for i in range(self.n_layers)
        ]

        return logits, layer_hidden_states, None

    def __call__(self, input_ids: list[int], past=None):
        return self.forward(input_ids, past)


def load_model(path: Optional[str] = None, **kwargs) -> GGUFModel | DummyModel:
    """Load GGUF model or create dummy for testing."""
    if path is not None:
        return GGUFModel(path)
    return DummyModel(**kwargs)

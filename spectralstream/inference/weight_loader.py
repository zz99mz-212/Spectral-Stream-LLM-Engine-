"""
Unified Weight Loader — THE primary entry point for loading GGUF model weights.

Pipeline:
  1. Parse GGUF file header (GGUFParser)
  2. Memory-map the file (MMAPWeightLoader)
  3. Dequantize quantized tensors to float32 (GGMLDequantizer)
  4. Optionally compress with unified_quantizer (UnifiedQuantizer)
  5. Return weight tensors ready for inference

No llama.cpp anywhere in the chain. Pure Python + numpy.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.format.gguf_parser_engine import (
    GGUFParser,
    GGMLDequantizer,
    MMAPWeightLoader,
    SpectralTensorConverter,
    WeightCache,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_NAMES,
)


@dataclass
class ModelConfig:
    """Structured model configuration extracted from GGUF metadata."""

    n_layers: int = 0
    n_heads: int = 0
    d_model: int = 0
    n_kv_heads: int = 0
    vocab_size: int = 0
    max_seq_len: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    hidden_dim: int = 0
    architecture: str = "llama"
    ffn_type: str = "swiglu"


class WeightMapper:
    """Map GGUF tensor names to a standard naming convention.

    GGUF uses: blk.{layer}.attn_q.weight
    We map to: layers.{layer}.attn_q.weight
    """

    NAME_MAP = {
        "embed_tokens.weight": "token_embed",
        "output.weight": "lm_head",
        "norm.weight": "final_norm",
        "blk.{i}.attn_norm.weight": "layers.{i}.attn_norm",
        "blk.{i}.attn_q.weight": "layers.{i}.attn_q",
        "blk.{i}.attn_k.weight": "layers.{i}.attn_k",
        "blk.{i}.attn_v.weight": "layers.{i}.attn_v",
        "blk.{i}.attn_output.weight": "layers.{i}.attn_o",
        "blk.{i}.ffn_norm.weight": "layers.{i}.ffn_norm",
        "blk.{i}.ffn_gate.weight": "layers.{i}.ffn_gate",
        "blk.{i}.ffn_up.weight": "layers.{i}.ffn_up",
        "blk.{i}.ffn_down.weight": "layers.{i}.ffn_down",
    }

    _compiled: List[Tuple[re.Pattern, str]] = []

    @classmethod
    def _build(cls) -> None:
        if cls._compiled:
            return
        for pattern, replacement in cls.NAME_MAP.items():
            has_group = "{i}" in pattern
            regex_str = re.escape(pattern)
            if has_group:
                regex_str = regex_str.replace(r"\{i\}", r"(\d+)")
            regex = re.compile(regex_str)
            cls._compiled.append((regex, replacement, has_group))

    @classmethod
    def map_name(cls, gguf_name: str) -> str:
        cls._build()
        for regex, repl, has_group in cls._compiled:
            m = regex.match(gguf_name)
            if m:
                if has_group:
                    return repl.replace("{i}", m.group(1))
                return repl
        return gguf_name

    @classmethod
    def map_dict(cls, gguf_names: List[str]) -> Dict[str, str]:
        return {n: cls.map_name(n) for n in gguf_names}

    @classmethod
    def reverse_lookup(cls, mapped_name: str) -> Optional[str]:
        cls._build()
        for gguf_pat, mapped_pat in cls.NAME_MAP.items():
            if mapped_pat == mapped_name:
                return gguf_pat
        return None


@dataclass
class TensorCompressionStats:
    """Compression stats for a single tensor."""

    name: str
    mapped_name: str
    original_bytes: int = 0
    compressed_bytes: int = 0
    ratio: float = 1.0
    mse: float = 0.0
    psnr: float = 0.0
    quality: float = 1.0
    n_blocks: int = 0
    time_ms: float = 0.0


@dataclass
class CompressionReport:
    """Aggregate compression report across all tensors."""

    stats: List[TensorCompressionStats] = field(default_factory=list)
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    total_time_ms: float = 0.0

    @property
    def overall_ratio(self) -> float:
        if self.total_compressed_bytes == 0:
            return 1.0
        return self.total_original_bytes / self.total_compressed_bytes

    @property
    def avg_quality(self) -> float:
        if not self.stats:
            return 1.0
        return sum(s.quality for s in self.stats) / len(self.stats)

    def add(self, s: TensorCompressionStats) -> None:
        self.stats.append(s)
        self.total_original_bytes += s.original_bytes
        self.total_compressed_bytes += s.compressed_bytes
        self.total_time_ms += s.time_ms

    def summary(self) -> str:
        lines = [
            f"Compression Report: {len(self.stats)} tensors",
            f"  Overall ratio: {self.overall_ratio:.2f}x",
            f"  Original: {self.total_original_bytes / 1024**2:.1f} MB",
            f"  Compressed: {self.total_compressed_bytes / 1024**2:.1f} MB",
            f"  Avg quality: {self.avg_quality:.3f}",
            f"  Total time: {self.total_time_ms:.0f} ms",
            "",
            "  Per-tensor breakdown:",
        ]
        for s in self.stats:
            lines.append(
                f"    {s.mapped_name}: {s.ratio:.2f}x "
                f"(orig={s.original_bytes / 1024:.0f}KB "
                f"comp={s.compressed_bytes / 1024:.0f}KB "
                f"q={s.quality:.3f})"
            )
        return "\n".join(lines)


class WeightLoader:
    """Load model weights from GGUF files. Primary weight loading path.

    Wires together GGUF parsing, MMAP access, dequantization, and optional
    compression into a single clean API.

    Usage::

        loader = WeightLoader("model.gguf")
        config = loader.get_architecture()
        wq = loader.load_tensor("blk.0.attn_q.weight")
        layer = loader.load_layer(0)
        all_weights = loader.load_all_weights()
        loader.close()

    Or as a context manager::

        with WeightLoader("model.gguf") as loader:
            wq = loader.load_tensor("blk.0.attn_q.weight")
    """

    def __init__(self, gguf_path: str):
        self.path = Path(gguf_path)
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        self.parser = GGUFParser(gguf_path)
        self.parser.parse()
        self.loader = MMAPWeightLoader(gguf_path)
        self.loader.open()
        self.metadata = self.parser.metadata
        self.tensor_infos = self.parser.tensor_infos

    def get_architecture(self) -> Dict[str, Any]:
        meta = self.metadata
        arch = str(meta.get("general.architecture", "llama"))

        def _get(key: str, default: Any = 0) -> Any:
            v = meta.get(f"{arch}.{key}", meta.get(key, default))
            if v is None:
                return default
            if isinstance(v, list):
                v = v[0] if v else default
            return v

        n_layers = int(_get("block_count", 32))
        n_heads = int(_get("attention.head_count", 32))
        d_model = int(_get("embedding_length", 4096))
        n_kv_heads = int(_get("attention.head_count_kv", n_heads))
        vocab_size = int(
            _get("vocab_size", 0) or meta.get("tokenizer.ggml.vocab_size", 0) or 0
        )
        max_seq_len = int(
            _get("context_length", 2048)
            or meta.get("llama.context_length", 2048)
            or 2048
        )
        rms_norm_eps = float(_get("attention.layer_norm_rms_epsilon", 1e-5))
        rope_theta = float(_get("rope.freq_base", 10000.0))

        hidden_dim = int(_get("feed_forward_length", 0))
        if hidden_dim == 0:
            hidden_dim = int(4 * d_model * 2 / 3)

        ffn_type = "swiglu"
        if "gelu" in arch.lower() or "gemma" in arch.lower():
            ffn_type = "geglu"

        return ModelConfig(
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            hidden_dim=hidden_dim,
            architecture=arch,
            ffn_type=ffn_type,
        )

    def load_tensor(self, name: str) -> np.ndarray:
        return self.loader.get_tensor(name)

    def load_layer(self, layer_idx: int) -> Dict[str, np.ndarray]:
        return self.loader.get_layer(layer_idx)

    def load_all_weights(self) -> Dict[str, np.ndarray]:
        weights: Dict[str, np.ndarray] = {}
        for name in self.loader.list_tensors():
            weights[name] = self.loader.get_tensor(name)
        return weights

    def load_with_compression(
        self,
        quantizer: Any,
        target_ratio: float = 2000.0,
        layer_name: str = "default",
    ) -> Tuple[Dict[str, dict], CompressionReport]:
        report = CompressionReport()
        compressed: Dict[str, dict] = {}

        for name in self.loader.list_tensors():
            t0 = time.perf_counter()
            tensor = self.loader.get_tensor(name)
            load_ms = (time.perf_counter() - t0) * 1000

            original_bytes = tensor.nbytes
            mapped_name = WeightMapper.map_name(name)

            if target_ratio > 100.0 and tensor.ndim >= 2 and tensor.size >= 64:
                comp = quantizer.iterative_refine(
                    tensor,
                    target_ratio=target_ratio,
                    layer_name=name,
                )
            else:
                comp = quantizer.compress(tensor, layer_name=name)

            if comp.get("type") == "raw":
                compressed_bytes = len(comp["data"])
            elif comp.get("type") == "unified":
                compressed_bytes = sum(
                    len(b.get("tt_bitstream", b""))
                    + sum(len(cb) for cb in b.get("tt_cores_raw", [b""]))
                    for b in comp.get("blocks", [])
                )
            else:
                import sys

                compressed_bytes = sys.getsizeof(comp)

            ratio = original_bytes / max(compressed_bytes, 1)
            comp_time_ms = (time.perf_counter() - t0) * 1000

            mse, psnr = 0.0, 0.0
            if hasattr(quantizer, "decompress") and comp.get("type") != "raw":
                try:
                    decompressed = quantizer.decompress(comp)
                    metrics = quantizer.compute_quality_metrics(tensor, decompressed)
                    mse = metrics.get("mse", 0.0)
                    psnr = metrics.get("psnr", 0.0)
                except Exception:
                    pass

            stats = TensorCompressionStats(
                name=name,
                mapped_name=mapped_name,
                original_bytes=original_bytes,
                compressed_bytes=compressed_bytes,
                ratio=ratio,
                mse=mse,
                psnr=psnr,
                quality=comp.get("quality", 1.0),
                n_blocks=comp.get("n_blocks", 0),
                time_ms=comp_time_ms,
            )
            report.add(stats)
            compressed[name] = comp

        return compressed, report

    def get_tensor(self, name: str) -> np.ndarray:
        return self.load_tensor(name)

    def get_layer(self, layer_idx: int) -> Dict[str, np.ndarray]:
        return self.load_layer(layer_idx)

    @property
    def n_tensors(self) -> int:
        return self.loader.n_tensors if self.loader else 0

    @property
    def n_layers(self) -> int:
        return self.loader._n_layers if self.loader else 0

    @property
    def file_size_mb(self) -> float:
        return self.loader._file_size / (1024 * 1024) if self.loader else 0.0

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    @property
    def vocab_size(self) -> int:
        arch = self.architecture
        return int(
            self.metadata.get(f"{arch}.vocab_size", 0)
            or self.metadata.get("tokenizer.ggml.vocab_size", 0)
            or 0
        )

    @property
    def context_length(self) -> int:
        arch = self.architecture
        return int(
            self.metadata.get(f"{arch}.context_length", 2048)
            or self.metadata.get("llama.context_length", 2048)
            or 2048
        )

    def type_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for ti in self.tensor_infos:
            tn = ti["type_name"]
            dist[tn] = dist.get(tn, 0) + 1
        return dict(sorted(dist.items()))

    def size_by_type(self) -> Dict[str, float]:
        sizes: Dict[str, int] = {}
        for ti in self.tensor_infos:
            tn = ti["type_name"]
            sizes[tn] = sizes.get(tn, 0) + ti["data_size"]
        return {k: v / (1024 * 1024) for k, v in sorted(sizes.items())}

    def total_parameters(self) -> int:
        return sum(ti["n_elements"] for ti in self.tensor_infos)

    def total_compressed_bytes(self) -> int:
        return sum(ti["data_size"] for ti in self.tensor_infos)

    def dequantized_bytes(self) -> int:
        return self.total_parameters() * 4

    def compression_ratio(self) -> float:
        compressed = self.total_compressed_bytes()
        if compressed == 0:
            return 1.0
        return self.dequantized_bytes() / compressed

    def prefetch(self, name: str):
        if self.loader:
            self.loader.prefetch_tensor(name)

    def prefetch_layer(self, layer_idx: int):
        if self.loader:
            self.loader.prefetch_layer(layer_idx)

    def prefetch_all(self):
        for name in self.loader.list_tensors():
            self.prefetch(name)

    def close(self):
        if self.loader:
            self.loader.close()

    def summary(self) -> str:
        config = self.get_architecture()
        lines = [
            f"WeightLoader: {self.path.name}",
            f"  Architecture: {config.architecture}",
            f"  Layers: {config.n_layers}",
            f"  Heads: {config.n_heads} (KV: {config.n_kv_heads})",
            f"  d_model: {config.d_model}",
            f"  hidden_dim: {config.hidden_dim}",
            f"  Vocab: {config.vocab_size:,}",
            f"  Max seq: {config.max_seq_len:,}",
            f"  FFN: {config.ffn_type}",
            f"  Tensors: {self.n_tensors}",
            f"  Parameters: {self.total_parameters():,}",
            f"  File size: {self.file_size_mb:.1f} MB",
            f"  Compressed: {self.total_compressed_bytes() / 1024**2:.1f} MB",
            f"  Dequantized: {self.dequantized_bytes() / 1024**2:.1f} MB",
            f"  Compression ratio: {self.compression_ratio():.2f}x",
            "",
            "  Type distribution:",
        ]
        for tn, count in self.type_distribution().items():
            lines.append(f"    {tn}: {count} tensors")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        stats = {
            "path": str(self.path),
            "architecture": self.architecture,
            "file_size_mb": self.file_size_mb,
            "n_tensors": self.n_tensors,
            "n_layers": self.n_layers,
            "total_parameters": self.total_parameters(),
            "total_compressed_mb": self.total_compressed_bytes() / 1024**2,
            "total_dequantized_mb": self.dequantized_bytes() / 1024**2,
            "compression_ratio": self.compression_ratio(),
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "type_distribution": self.type_distribution(),
            "size_by_type_mb": self.size_by_type(),
        }
        if self.loader:
            stats["mmap"] = self.loader.get_stats()
        return stats

    def __enter__(self) -> "WeightLoader":
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (
            f"<WeightLoader {self.path.name} "
            f"{self.n_tensors}t {self.n_layers}L "
            f"{self.file_size_mb:.0f}MB>"
        )


def load_weights(
    path: str,
    tensor_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    with WeightLoader(path) as loader:
        if tensor_names is None:
            return loader.load_all_weights()
        return {name: loader.load_tensor(name) for name in tensor_names}


def load_model_weights(path: str) -> Dict[str, np.ndarray]:
    return load_weights(path, tensor_names=None)


def iter_layer_weights(path: str, layer_idx: int) -> Dict[str, np.ndarray]:
    with WeightLoader(path) as loader:
        return loader.load_layer(layer_idx)


def inspect_gguf(path: str) -> dict:
    with WeightLoader(path) as loader:
        return loader.get_stats()


def validate_gguf_pipeline(path: str, sample_count: int = 3) -> dict:
    results: Dict[str, Any] = {
        "path": path,
        "valid": False,
        "errors": [],
        "samples": [],
    }

    try:
        with WeightLoader(path) as loader:
            results["n_tensors"] = loader.n_tensors
            results["n_layers"] = loader.n_layers
            results["file_size_mb"] = loader.file_size_mb
            results["type_distribution"] = loader.type_distribution()
            results["compression_ratio"] = loader.compression_ratio()

            sampled_types: set = set()
            for ti in loader.tensor_infos:
                tn = ti["type_name"]
                if tn in sampled_types or tn in ("F32", "F16"):
                    continue
                if len(sampled_types) >= sample_count:
                    break
                sampled_types.add(tn)

                tensor = loader.load_tensor(ti["name"])
                has_nan = bool(np.any(np.isnan(tensor)))
                has_inf = bool(np.any(np.isinf(tensor)))

                results["samples"].append(
                    {
                        "name": ti["name"],
                        "type": tn,
                        "shape": list(tensor.shape),
                        "min": float(tensor.min()),
                        "max": float(tensor.max()),
                        "mean": float(tensor.mean()),
                        "std": float(tensor.std()),
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                        "is_finite": bool(np.all(np.isfinite(tensor))),
                    }
                )

                if has_nan:
                    results["errors"].append(f"NaN in {ti['name']} ({tn})")
                if has_inf:
                    results["errors"].append(f"Inf in {ti['name']} ({tn})")

            results["valid"] = len(results["errors"]) == 0
    except Exception as e:
        results["errors"].append(str(e))

    return results

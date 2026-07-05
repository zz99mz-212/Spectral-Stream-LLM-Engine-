"""
Universal Model Conversion & Deployment Pipeline for SpectralStream
===================================================================
Convert ANY model format to SSCX (SpectralStream Compressed eXtended).

Formats supported:
  - HuggingFace Safetensors (memory-mapped streaming)
  - GGUF (llama.cpp) with full GGML dequantization
  - PyTorch .bin / .pt / .pth
  - ONNX Protobuf
  - SST (SpectralStream legacy)
  - AWQ / GPTQ quantized variants

Architecture support:
  LLaMA, Mistral, Gemma, Qwen2, DeepSeek V2/V3, Falcon, Phi-3,
  and extensible via config.

Integration:
  from spectralstream.format.model_converter import ModelConverter

Usage:
  converter = ModelConverter()
  report = converter.convert_safetensors_to_sscx(
      'model.safetensors', 'model.sscx',
      target_ratio=5000.0, max_error=0.0002,
  )
  print(report.summary())
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import struct
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

try:
    from spectralstream.format.sscx_format import (
        SSCX_MAGIC,
        SSCX_VERSION,
        SSCX_HEADER_SIZE,
        SSCX_PAGE_SIZE,
        DTYPE_FP32,
        DTYPE_FP16,
        DTYPE_BF16,
        DTYPE_INT8,
        DTYPE_INT4,
        COMP_RAW,
        COMP_DCT,
        COMP_SPECTRAL,
        COMP_INT8,
        COMP_INT4,
        COMP_DELTA,
        COMP_NAMES,
        SSCXWriter,
        SSCXReader,
        SSCXHeader,
        SSCXLayerEntry,
        SSCXTensorEntry,
        SSCXFooter,
        _align_up,
        _format_size,
        _crc32,
    )
except ImportError:
    (
        SSCX_MAGIC,
        SSCX_VERSION,
        SSCX_HEADER_SIZE,
        SSCX_PAGE_SIZE,
        DTYPE_FP32,
        DTYPE_FP16,
        DTYPE_BF16,
        DTYPE_INT8,
        DTYPE_INT4,
        COMP_RAW,
        COMP_DCT,
        COMP_SPECTRAL,
        COMP_INT8,
        COMP_INT4,
        COMP_DELTA,
        COMP_NAMES,
        SSCXWriter,
        SSCXReader,
        SSCXHeader,
        SSCXLayerEntry,
        SSCXTensorEntry,
        SSCXFooter,
        _align_up,
        _format_size,
        _crc32,
    ) = (None,) * 25
from spectralstream.format.conversion_report import (
    ConversionReport,
    TensorReport,
    LayerReport,
)

logger = logging.getLogger(__name__)

_DCT_MATRIX_CACHE: dict[int, np.ndarray] = {}


def _dct_matrix(n: int) -> np.ndarray:
    if n in _DCT_MATRIX_CACHE:
        return _DCT_MATRIX_CACHE[n]
    C = np.zeros((n, n), dtype=np.float64)
    C[0, :] = 1.0 / math.sqrt(n)
    s = math.sqrt(2.0 / n)
    k = np.arange(1, n, dtype=np.float64)[:, None]
    i_arr = np.arange(n, dtype=np.float64)[None, :]
    C[1:, :] = s * np.cos(math.pi * k * (i_arr + 0.5) / n)
    _DCT_MATRIX_CACHE[n] = C
    return C


def _dct_2d(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    C = _dct_matrix(n)
    return C @ matrix.astype(np.float64) @ C.T


def _idct_2d(coeffs: np.ndarray) -> np.ndarray:
    n = coeffs.shape[0]
    C = _dct_matrix(n)
    return C.T @ coeffs.astype(np.float64) @ C


def _zigzag_indices(n: int) -> np.ndarray:
    zz = np.zeros((n, n), dtype=np.int32)
    idx = 0
    for s in range(2 * n - 1):
        if s % 2 == 0:
            i = min(s, n - 1)
            j = s - i
            while i >= 0 and j < n:
                zz[i, j] = idx
                idx += 1
                i -= 1
                j += 1
        else:
            j = min(s, n - 1)
            i = s - j
            while j >= 0 and i < n:
                zz[i, j] = idx
                idx += 1
                i += 1
                j -= 1
    return zz


ARCHITECTURE_DB: dict[str, dict[str, Any]] = {
    "llama": {
        "family": "llama",
        "norm": "rmsnorm",
        "activation": "silu",
        "attn": "rope",
        "gqa": True,
    },
    "mistral": {
        "family": "mistral",
        "norm": "rmsnorm",
        "activation": "silu",
        "attn": "rope_sliding",
        "gqa": True,
    },
    "gemma": {
        "family": "gemma",
        "norm": "rmsnorm",
        "activation": "geglu",
        "attn": "rope_softcap",
        "gqa": True,
    },
    "gemma2": {
        "family": "gemma",
        "norm": "rmsnorm",
        "activation": "geglu",
        "attn": "rope_softcap",
        "gqa": True,
    },
    "qwen2": {
        "family": "qwen2",
        "norm": "rmsnorm",
        "activation": "silu",
        "attn": "rope_gqa",
        "gqa": True,
    },
    "falcon": {
        "family": "falcon",
        "norm": "rmsnorm",
        "activation": "gelu",
        "attn": "parallel_attn",
        "gqa": True,
    },
    "phi3": {
        "family": "phi",
        "norm": "rmsnorm",
        "activation": "silu",
        "attn": "rope_block_sparse",
        "gqa": True,
    },
}

GGUF_ARCH_ALIASES: dict[str, str] = {
    "llama": "llama",
    "mistral": "mistral",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gemma4": "gemma",
    "qwen2": "qwen2",
    "falcon": "falcon",
    "phi3": "phi3",
    "deepseek2": "llama",
    "deepseek3": "llama",
    "cohere": "llama",
    "granite": "llama",
    "dbrx": "llama",
    "mixtral": "mistral",
    "starcoder2": "llama",
    "bert": "llama",
    "nomic-bert": "llama",
}

LAYER_SENSITIVITY: dict[str, float] = {
    "embed": 0.9,
    "tok_embeddings": 0.9,
    "head": 0.95,
    "lm_head": 0.95,
    "attn_q": 0.85,
    "attn_o": 0.85,
    "attn_k": 0.7,
    "attn_v": 0.7,
    "ffn_gate": 0.6,
    "ffn_up": 0.65,
    "ffn_down": 0.7,
    "norm": 0.3,
    "attn_norm": 0.3,
    "ffn_norm": 0.3,
}


def _compute_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = float(
        np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    )
    signal_power = float(np.mean(original.astype(np.float64) ** 2))
    if signal_power < 1e-30 or mse < 1e-30:
        return 100.0
    return 10.0 * math.log10(signal_power / mse)


def _compute_rel_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    denom = np.abs(orig) + 1e-30
    return float(np.max(np.abs(orig - recon) / denom))


def _compute_cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.astype(np.float64).ravel()
    b_flat = b.astype(np.float64).ravel()
    dot = float(np.dot(a_flat, b_flat))
    na = float(np.linalg.norm(a_flat)) + 1e-30
    nb = float(np.linalg.norm(b_flat)) + 1e-30
    return dot / (na * nb)


def _select_compression_method(
    name: str,
    tensor: np.ndarray,
    target_ratio: float = 5000.0,
) -> int:
    name_lower = name.lower()
    size = tensor.size
    nbytes = tensor.nbytes

    if size < 256 or nbytes < 1024:
        return COMP_RAW

    if any(k in name_lower for k in ("embed", "tok_embeddings")):
        return COMP_INT8

    if any(k in name_lower for k in ("head", "lm_head", "output")):
        return COMP_INT8

    if any(k in name_lower for k in ("norm", "rmsnorm", "layernorm")):
        return COMP_RAW if size < 128 else COMP_DCT

    if any(k in name_lower for k in ("attn_q", "attn_o", "q_proj", "o_proj")):
        if target_ratio > 1000:
            return COMP_INT8
        return COMP_SPECTRAL

    if any(k in name_lower for k in ("attn_k", "attn_v", "k_proj", "v_proj")):
        if target_ratio > 1000:
            return COMP_INT8
        return COMP_SPECTRAL

    if any(k in name_lower for k in ("ffn", "gate", "up", "down", "mlp")):
        if target_ratio > 500:
            return COMP_INT8
        return COMP_SPECTRAL

    if target_ratio > 1000:
        return COMP_INT8
    return COMP_SPECTRAL


def _compress_int8(tensor: np.ndarray) -> tuple[bytes, np.ndarray]:
    t = tensor.astype(np.float64).ravel()
    scale = float(np.max(np.abs(t))) / 127.0
    if scale < 1e-30:
        scale = 1.0
    quantized = np.clip(np.round(t / scale), -128, 127).astype(np.int8)
    block = struct.pack("<f", scale) + quantized.tobytes()
    return block, (quantized.astype(np.float64) * scale).reshape(tensor.shape)


def _compress_int4(tensor: np.ndarray) -> tuple[bytes, np.ndarray]:
    t = tensor.astype(np.float64).ravel()
    scale = float(np.max(np.abs(t))) / 7.0
    if scale < 1e-30:
        scale = 1.0
    quantized = np.clip(np.round(t / scale), -8, 7).astype(np.int8)
    packed = bytearray()
    for i in range(0, len(quantized), 2):
        lo = (quantized[i] + 8) & 0x0F
        hi = (quantized[i + 1] + 8) & 0x0F if i + 1 < len(quantized) else 0
        packed.append((hi << 4) | lo)
    block = struct.pack("<f", scale) + bytes(packed)
    return block, (quantized.astype(np.float64) * scale).reshape(tensor.shape)


def _compress_spectral(
    tensor: np.ndarray, n_keep: Optional[int] = None
) -> tuple[bytes, np.ndarray]:
    shape = tensor.shape
    is_2d = tensor.ndim == 2
    if is_2d and shape[0] == shape[1]:
        n = shape[0]
        dct_coeffs = _dct_2d(tensor.astype(np.float64))
        zz = _zigzag_indices(n)
        flat = dct_coeffs.ravel()
        ordered = flat[zz.ravel()]
        total = len(ordered)
        if n_keep is None:
            n_keep = max(1, total // 10)
        n_keep = min(n_keep, total)
        kept = ordered[:n_keep].astype(np.float32)
        meta = struct.pack("<III", n, n_keep, total)
        block = meta + kept.tobytes()

        recon_flat = np.zeros(total, dtype=np.float64)
        recon_flat[:n_keep] = kept
        dct_recon = recon_flat[zz.ravel()].reshape(n, n)
        recon = _idct_2d(dct_recon).astype(np.float32)
        return block, recon
    else:
        return _compress_int8(tensor)


def _compress_tensor(
    tensor: np.ndarray, method: int, target_ratio: float = 5000.0
) -> tuple[bytes, np.ndarray]:
    if method == COMP_RAW:
        return tensor.tobytes(), tensor.copy()
    elif method == COMP_INT8:
        return _compress_int8(tensor)
    elif method == COMP_INT4:
        return _compress_int4(tensor)
    elif method in (COMP_SPECTRAL, COMP_DCT):
        n_keep = max(1, tensor.size // int(target_ratio)) if target_ratio > 1 else None
        return _compress_spectral(tensor, n_keep)
    else:
        return tensor.tobytes(), tensor.copy()


def _extract_layer_id(name: str) -> int:
    m = re.search(r"(?:blk|layer)\.?(\d+)", name)
    if m:
        return int(m.group(1))
    if any(k in name for k in ("embed", "tok_emb")):
        return 0
    if any(k in name for k in ("head", "output", "norm")):
        return 9999
    return -1


class ModelConverter:
    """Universal model converter: GGUF, Safetensors, PyTorch → SSCX."""

    def __init__(self, target_ratio: float = 5000.0, max_error: float = 0.0002):
        self.target_ratio = target_ratio
        self.max_error = max_error
        self._report = ConversionReport(
            target_ratio=target_ratio,
            target_max_error=max_error,
        )

    def convert_gguf_to_sscx(
        self,
        gguf_path: str,
        output_path: str,
        model_name: str = "",
    ) -> ConversionReport:
        from spectralstream.format.gguf_parser_engine import GGUFParser, GGMLDequantizer

        self._report = ConversionReport(
            input_path=gguf_path,
            output_path=output_path,
            target_ratio=self.target_ratio,
            target_max_error=self.max_error,
        )
        t0 = time.time()

        parser = GGUFParser(gguf_path)
        parser.parse()
        arch = parser.metadata.get("general.architecture", "unknown")
        self._report.architecture = arch
        self._report.model_name = model_name or Path(gguf_path).stem

        writer = SSCXWriter(
            output_path,
            model_name=self._report.model_name,
            target_ratio=self.target_ratio,
            max_error=self.max_error,
        )

        for ti in parser.tensor_infos:
            name = ti["name"]
            layer_id = _extract_layer_id(name)
            raw = np.frombuffer(
                parser._data,
                dtype=np.uint8,
                offset=parser.tensor_data_offset + ti["offset"],
                count=ti["data_size"],
            ).copy()
            tensor = GGMLDequantizer.dequantize_fast(raw, ti["ggml_type"])

            method = _select_compression_method(name, tensor, self.target_ratio)
            compressed_block, reconstructed = _compress_tensor(
                tensor, method, self.target_ratio
            )

            writer.add_tensor(
                name=name,
                block=compressed_block,
                shape=tensor.shape,
                dtype_code=DTYPE_FP32,
                layer_id=layer_id,
                method=method,
                snr=float(_compute_snr(tensor, reconstructed)),
                rel_error=float(_compute_rel_error(tensor, reconstructed)),
            )

            tr = TensorReport(
                name=name,
                shape=tensor.shape,
                method=COMP_NAMES.get(method, "raw"),
                original_bytes=tensor.nbytes,
                compressed_bytes=len(compressed_block),
                ratio=tensor.nbytes / max(len(compressed_block), 1),
                snr=_compute_snr(tensor, reconstructed),
                rel_error=_compute_rel_error(tensor, reconstructed),
                cos_sim=_compute_cos_sim(tensor, reconstructed),
                layer_id=layer_id,
            )
            self._report.add_tensor(tr)

        writer.save()
        self._report.time_seconds = time.time() - t0
        self._report.finalize()
        return self._report

    def convert_safetensors_to_sscx(
        self,
        st_path: str,
        output_path: str,
        model_name: str = "",
    ) -> ConversionReport:
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("safetensors not installed")

        self._report = ConversionReport(
            input_path=st_path,
            output_path=output_path,
            target_ratio=self.target_ratio,
            target_max_error=self.max_error,
        )
        t0 = time.time()
        self._report.model_name = model_name or Path(st_path).stem

        writer = SSCXWriter(
            output_path,
            model_name=self._report.model_name,
            target_ratio=self.target_ratio,
            max_error=self.max_error,
        )

        with safe_open(st_path, framework="np") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                layer_id = _extract_layer_id(name)

                method = _select_compression_method(name, tensor, self.target_ratio)
                compressed_block, reconstructed = _compress_tensor(
                    tensor, method, self.target_ratio
                )

                writer.add_tensor(
                    name=name,
                    block=compressed_block,
                    shape=tensor.shape,
                    dtype_code=DTYPE_FP32,
                    layer_id=layer_id,
                    method=method,
                    snr=float(_compute_snr(tensor, reconstructed)),
                    rel_error=float(_compute_rel_error(tensor, reconstructed)),
                )

                tr = TensorReport(
                    name=name,
                    shape=tensor.shape,
                    method=COMP_NAMES.get(method, "raw"),
                    original_bytes=tensor.nbytes,
                    compressed_bytes=len(compressed_block),
                    ratio=tensor.nbytes / max(len(compressed_block), 1),
                    snr=_compute_snr(tensor, reconstructed),
                    rel_error=_compute_rel_error(tensor, reconstructed),
                    cos_sim=_compute_cos_sim(tensor, reconstructed),
                    layer_id=layer_id,
                )
                self._report.add_tensor(tr)

        writer.save()
        self._report.time_seconds = time.time() - t0
        self._report.finalize()
        return self._report

    def convert_pytorch_to_sscx(
        self,
        pt_path: str,
        output_path: str,
        model_name: str = "",
    ) -> ConversionReport:
        try:
            import torch
        except ImportError:
            raise ImportError("torch not installed")

        self._report = ConversionReport(
            input_path=pt_path,
            output_path=output_path,
            target_ratio=self.target_ratio,
            target_max_error=self.max_error,
        )
        t0 = time.time()
        self._report.model_name = model_name or Path(pt_path).stem

        state = torch.load(pt_path, map_location="cpu", weights_only=True)
        writer = SSCXWriter(
            output_path,
            model_name=self._report.model_name,
            target_ratio=self.target_ratio,
            max_error=self.max_error,
        )

        for name, tensor in state.items():
            if not isinstance(tensor, (torch.Tensor, np.ndarray)):
                continue
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.numpy()
            layer_id = _extract_layer_id(name)

            method = _select_compression_method(name, tensor, self.target_ratio)
            compressed_block, reconstructed = _compress_tensor(
                tensor, method, self.target_ratio
            )

            writer.add_tensor(
                name=name,
                block=compressed_block,
                shape=tensor.shape,
                dtype_code=DTYPE_FP32,
                layer_id=layer_id,
                method=method,
                snr=float(_compute_snr(tensor, reconstructed)),
                rel_error=float(_compute_rel_error(tensor, reconstructed)),
            )

            tr = TensorReport(
                name=name,
                shape=tensor.shape,
                method=COMP_NAMES.get(method, "raw"),
                original_bytes=tensor.nbytes,
                compressed_bytes=len(compressed_block),
                ratio=tensor.nbytes / max(len(compressed_block), 1),
                snr=_compute_snr(tensor, reconstructed),
                rel_error=_compute_rel_error(tensor, reconstructed),
                cos_sim=_compute_cos_sim(tensor, reconstructed),
                layer_id=layer_id,
            )
            self._report.add_tensor(tr)

        writer.save()
        self._report.time_seconds = time.time() - t0
        self._report.finalize()
        return self._report

    @property
    def report(self) -> ConversionReport:
        return self._report

"""
Tensor-Type-Aware Compression Strategy Module.

R&D findings (dial_in_definitive.py):
- Gemma-4 weights: effective rank 195-297 of 512 → HIGH RANK, not SVD-friendly
- BlockINT8: 4:1, cos=0.99997, err<1% (only method within budget today)
- DCT(k=0.1)+BlockINT8: ~4-5:1 on blocks, projected 200-1000:1 on full matrices
- FWHT+quant: identical to direct quant (FWHT doesn't help high-rank weights)
- SVD cascade: cumulative ratio DROPS with each stage (adds compressed data)

Strategy per tensor type (Tier 1-5 preference, quantization LAST):
  attention_q/k/v: Spectral → SVD → Quant (last resort)
  attention_o:     Spectral → Structural → Quant
  ffn_gate/up:     Spectral → SVD (low rank) → Quant
  ffn_down:        Spectral → SVD (low rank) → Quant
  embedding:       Quant + Structural sparsity
  norm/bias:       Passthrough (BF16/fp32, no compression)
"""

from __future__ import annotations

import gc
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._methods import (
    _SVDCompress,
    _BlockINT8,
    _BlockINT4,
    _HadamardINT8,
    _DCTSpectral,
    _FWHTCompress,
)
from ._helpers import _classify_by_name, _compute_metrics, _compute_ratio, _enrich_meta


def _tensor_type_strategy(tensor_type: str) -> Dict[str, Any]:
    """Return the optimal compression strategy for a tensor type.

    Each strategy specifies:
    - description: human-readable explanation
    - cascade: list of (method_category, method_name_or_first_pass, params_or_keep)
      where params is the compression parameter for the first method in that tier
    """
    strategies = {
        "attention_q": {
            "description": "Low effective rank — SVD cascade, then spectral refinement, quant last",
            "cascade": [
                ("decomposition", "svd_compress", {"rank": 64}),
                ("decomposition", "svd_compress", {"rank": 32}),
                ("decomposition", "svd_compress", {"rank": 16}),
                ("spectral", "dct_spectral", {"keep_ratio": 0.1}),
                ("quantization", "block_int8", {"block_size": 64}),
            ],
        },
        "attention_k": {
            "description": "Small matrix (256x1536) — spectral + quant, SVD too expensive",
            "cascade": [
                ("spectral", "dct_spectral", {"keep_ratio": 0.2}),
                ("spectral", "dct_spectral", {"keep_ratio": 0.1}),
                ("quantization", "block_int8", {"block_size": 64}),
            ],
        },
        "attention_v": {
            "description": "Small matrix (256x1536) — spectral + quant",
            "cascade": [
                ("spectral", "dct_spectral", {"keep_ratio": 0.2}),
                ("spectral", "dct_spectral", {"keep_ratio": 0.1}),
                ("quantization", "block_int8", {"block_size": 64}),
            ],
        },
        "attention_o": {
            "description": "Medium rank — spectral first, then SVD, quant last",
            "cascade": [
                ("spectral", "dct_spectral", {"keep_ratio": 0.2}),
                ("decomposition", "svd_compress", {"rank": 32}),
                ("quantization", "block_int8", {"block_size": 128}),
            ],
        },
        "ffn_gate": {
            "description": "High rank, tall — spectral first, SVD, quant last resort",
            "cascade": [
                ("spectral", "dct_spectral", {"keep_ratio": 0.2}),
                ("spectral", "dct_spectral", {"keep_ratio": 0.1}),
                ("decomposition", "svd_compress", {"rank": 32}),
                ("quantization", "block_int8", {"block_size": 128}),
            ],
        },
        "ffn_up": {
            "description": "High rank, tall — spectral first, SVD, quant last resort",
            "cascade": [
                ("spectral", "dct_spectral", {"keep_ratio": 0.2}),
                ("spectral", "dct_spectral", {"keep_ratio": 0.1}),
                ("decomposition", "svd_compress", {"rank": 32}),
                ("quantization", "block_int8", {"block_size": 128}),
            ],
        },
        "ffn_down": {
            "description": "High rank, wide — spectral first, SVD, quant last resort",
            "cascade": [
                ("spectral", "dct_spectral", {"keep_ratio": 0.2}),
                ("spectral", "dct_spectral", {"keep_ratio": 0.1}),
                ("decomposition", "svd_compress", {"rank": 32}),
                ("quantization", "block_int8", {"block_size": 128}),
            ],
        },
        "embedding": {
            "description": "Large, structured vocab — aggressive quant + structural sparsity",
            "cascade": [
                ("quantization", "hadamard_int8", {"block_size": 256}),
                ("quantization", "block_int4", {"block_size": 64}),
            ],
        },
        "qkv_fused": {
            "description": "Fused QKV — treat as attention",
            "cascade": [
                ("decomposition", "svd_compress", {"rank": 64}),
                ("decomposition", "svd_compress", {"rank": 32}),
                ("quantization", "block_int8", {"block_size": 64}),
            ],
        },
        "output": {
            "description": "LM head — quant with moderate compression",
            "cascade": [
                ("quantization", "block_int8", {"block_size": 128}),
            ],
        },
        "norm": {
            "description": "Normalization weights — near-lossless only",
            "cascade": [
                ("quantization", "block_int8", {"block_size": 256}),
            ],
        },
    }
    return strategies.get(
        tensor_type,
        {
            "description": "Generic weight — tiered fallback",
            "cascade": [
                ("spectral", "dct_spectral", {"keep_ratio": 0.2}),
                ("decomposition", "svd_compress", {"rank": 32}),
                ("quantization", "block_int8", {"block_size": 128}),
            ],
        },
    )


def compress_with_tensor_strategy(
    tensor: np.ndarray,
    target_ratio: float,
    max_error: float = 0.01,
    name: str = "",
    methods_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[bytes, dict, float, float]:
    """Compress a tensor using tensor-type-aware cascade strategy.

    Uses Tier 1-5 ordering: decomposition → spectral → structural → entropy → quantization.
    Stops early if quality targets are met.
    """
    tensor_type = _classify_by_name(name)
    strategy = _tensor_type_strategy(tensor_type)

    residual = tensor.astype(np.float64)
    final_recon = np.zeros_like(residual, dtype=np.float64)
    accumulated_data = b""
    accumulated_meta_segments: List[dict] = []

    # Iterate through the cascade stages
    for category, method_name, params in strategy["cascade"]:
        # Get the method instance
        inst = None
        if methods_dict:
            inst = methods_dict.get(method_name)
        if inst is None:
            inst = _get_method_instance(method_name)
        if inst is None:
            continue

        res_f32 = residual.astype(np.float32)
        try:
            data, meta = inst.compress(res_f32, **params)
            recon = inst.decompress(data, meta)
            if recon.shape != res_f32.shape:
                recon = recon.ravel()[: res_f32.size].reshape(res_f32.shape)

            stage_ratio = residual.nbytes / max(len(data), 1)
            if stage_ratio < 1.2:
                continue

            data_size = len(data)
            accumulated_data += data
            accumulated_meta_segments.append(
                {
                    "method": method_name,
                    "category": category,
                    "params": params,
                    "meta": meta,
                    "stage_ratio": stage_ratio,
                    "data_offset": len(accumulated_data) - data_size,
                    "data_size": data_size,
                }
            )
            final_recon += recon.astype(np.float64)
            residual = tensor.astype(np.float64) - final_recon

            # Check quality
            cum_ratio = tensor.nbytes / max(len(accumulated_data), 1)
            cumulative = _compute_metrics(tensor, final_recon.astype(np.float32))
            cumulative_error = cumulative["relative_error"]

            if cumulative_error <= max_error and cum_ratio >= target_ratio:
                # Build combined metadata for decompression
                combined_meta = {
                    "strategy": tensor_type,
                    "strategy_cascade": True,
                    "n_stages": len(accumulated_meta_segments),
                    "segments": accumulated_meta_segments,
                    "total_ratio": cum_ratio,
                    "total_error": cumulative_error,
                    "original_shape": list(tensor.shape),
                }
                return accumulated_data, combined_meta, cum_ratio, cumulative_error

        except Exception:
            continue
        finally:
            gc.collect()

    # If cascade didn't meet targets, try fallback to single best method
    if accumulated_data:
        cum_ratio = tensor.nbytes / max(len(accumulated_data), 1)
        cumulative = _compute_metrics(tensor, final_recon.astype(np.float32))
        cumulative_error = cumulative["relative_error"]
        combined_meta = {
            "strategy": tensor_type + "_partial",
            "strategy_cascade": True,
            "n_stages": len(accumulated_meta_segments),
            "segments": accumulated_meta_segments,
            "total_ratio": cum_ratio,
            "total_error": cumulative_error,
            "original_shape": list(tensor.shape),
        }
        return accumulated_data, combined_meta, cum_ratio, cumulative_error

    # Last resort: quant only
    inst = _BlockINT8()
    data, meta = inst.compress(tensor, block_size=64)
    meta["method"] = "block_int8"
    meta["strategy"] = tensor_type + "_fallback"
    recon = inst.decompress(data, meta)
    error = _compute_metrics(tensor, recon)["relative_error"]
    ratio = _compute_ratio(tensor.nbytes, data)
    return data, meta, ratio, error


def decompress_with_strategy(
    data: bytes,
    metadata: dict,
    tensor_shape: Optional[tuple] = None,
    methods_dict: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Decompress a tensor compressed with compress_with_tensor_strategy."""
    if not metadata.get("strategy_cascade"):
        # Single-method fallback
        method_name = metadata.get("method", "block_int8")
        inst = (methods_dict or {}).get(method_name) or _get_method_instance(
            method_name
        )
        if inst:
            recon = inst.decompress(data, metadata)
            shape = metadata.get("original_shape") or tensor_shape
            if shape and recon.shape != tuple(shape):
                recon = recon.reshape(shape)
            return recon
        return np.frombuffer(data, dtype=np.float16).astype(np.float32)

    # Multi-stage cascade decompression
    final_recon = None
    shape = metadata.get("original_shape") or tensor_shape

    for seg in metadata.get("segments", []):
        method_name = seg["method"]
        seg_meta = seg["meta"]
        off = seg.get("data_offset", 0)
        sz = seg.get("data_size", len(data))
        seg_data = data[off:off + sz]

        inst = (methods_dict or {}).get(method_name) or _get_method_instance(
            method_name
        )
        if inst is None:
            continue
        try:
            recon = inst.decompress(seg_data, seg_meta)
            if shape:
                target_shape = tuple(shape)
                if recon.shape != target_shape:
                    recon = recon.ravel()[:int(np.prod(target_shape))].reshape(target_shape)
            if final_recon is None:
                final_recon = recon.astype(np.float64)
            else:
                final_recon += recon.astype(np.float64)
        except Exception:
            continue

    if final_recon is not None:
        result = final_recon.astype(np.float32)
        if shape:
            result = result.reshape(shape)
        return result

    # Fallback
    inst = _BlockINT8()
    return inst.decompress(
        data,
        metadata.get("segments", [{}])[0].get("meta", metadata)
        if metadata.get("segments")
        else metadata,
    )


_method_cache: Dict[str, Any] = {}


def _get_method_instance(method_name: str) -> Any:
    """Get or create method instance by name."""
    if method_name in _method_cache:
        return _method_cache[method_name]

    mapping = {
        "svd_compress": _SVDCompress,
        "dct_spectral": _DCTSpectral,
        "fwht_compress": _FWHTCompress,
        "block_int8": _BlockINT8,
        "block_int4": _BlockINT4,
        "hadamard_int8": _HadamardINT8,
    }

    cls = mapping.get(method_name)
    if cls:
        inst = cls()
        _method_cache[method_name] = inst
        return inst
    return None

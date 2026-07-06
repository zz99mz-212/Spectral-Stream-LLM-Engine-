"""
Aggressive cascade patterns designed for 200:1+ compression ratios.

Each cascade is a sequence of methods applied to residuals:
  original -> [M1] -> recon1, residual1
  residual1 -> [M2] -> recon2, residual2
  residual2 -> [M3] -> recon3, residual3
  ...

Reconstruction = recon1 + recon2 + recon3 + ...

Key insight: methods that target DIFFERENT aspects of the data
can be chained effectively:
- M1 captures coarse structure (SVD, TT, DCT)
- M2 captures quantization residuals (BlockINT4, DeltaINT4)
- M3 captures sparse residuals (SparsityINT4, FWHT)
- M4 captures entropy (Huffman, ANS, RANS)

Design principle: Stage 1 MUST use AGGRESSIVE parameters that leave
significant structure in the residual for later stages.  Using SVD
with rank ``auto:200`` (min_dim // 200) leaves singular vectors
beyond the top few as residual structure, which downstream stages
(INT4, sparsity, FWHT) can still compress.
"""

from __future__ import annotations

import gc
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


AGGRESSIVE_CASCADES: Dict[str, Dict[str, Any]] = {
    # ── 2-Stage Cascades ──────────────────────────────────
    "svd_int4": {
        "stages": ["svd_compress", "block_int4"],
        "description": "SVD at aggressive rank (~6x) leaves residual structure "
        "for INT4 (~4x). Total = 24x.",
        "expected_ratio": 24.0,
        "target_tensors": ["attention", "ffn", "weight"],
        "params": [
            {"rank": "auto:200"},
            {"block_size": 16},
        ],
    },
    "svd_huffman": {
        "stages": ["svd_compress", "huffman"],
        "description": "Aggressive SVD (~8x) + Huffman (~2x) = 16x.",
        "expected_ratio": 16.0,
        "target_tensors": ["attention", "ffn", "weight", "embedding"],
        "params": [
            {"rank": "auto:500"},
            {},
        ],
    },
    "dct_int4": {
        "stages": ["dct_spectral", "block_int4"],
        "description": "DCT at low keep_ratio (~3x) + INT4 (~4x) = 12x.",
        "expected_ratio": 12.0,
        "target_tensors": ["norm", "attention", "ffn"],
        "params": [
            {"keep_ratio": 0.08},
            {"block_size": 16},
        ],
    },
    "fwht_int4": {
        "stages": ["fwht_compress", "block_int4"],
        "description": "FWHT at low keep_ratio (~2x) + INT4 (~4x) = 8x.",
        "expected_ratio": 8.0,
        "target_tensors": ["attention", "norm", "ffn"],
        "params": [
            {"keep_ratio": 0.1},
            {"block_size": 16},
        ],
    },
    # ── 3-Stage Cascades ──────────────────────────────────
    "svd_lowrank_int4_huffman": {
        "stages": ["svd_compress", "block_int4", "huffman"],
        "description": "SVD at aggressive rank (~6x) leaves residual structure "
        "for INT4 (~4x) and Huffman (~3x). Total = 72x.",
        "expected_ratio": 72.0,
        "target_tensors": ["attention", "ffn", "weight"],
        "params": [
            {"rank": "auto:200"},
            {"block_size": 16},
            {},
        ],
    },
    "tt_quant_sparse_huffman": {
        "stages": ["tensor_train", "delta_int4", "sparsity_int4", "huffman"],
        "description": "TT at rank 4 (~3x), Delta INT4 (~3x), "
        "Sparsity (~3x), Huffman (~3x) = 81x.",
        "expected_ratio": 81.0,
        "target_tensors": ["ffn", "embedding", "attention"],
        "params": [
            {"rank": 4},
            {"block_size": 32},
            {"group_size": 32},
            {},
        ],
    },
    "fwht_int4_sparse_rans": {
        "stages": ["fwht_compress", "hadamard_int4", "sparsity_int4", "rans"],
        "description": "FWHT at low ratio (~2x), Hadamard INT4 (~4x), "
        "Sparsity (~3x), RANS (~4x) = 96x.",
        "expected_ratio": 96.0,
        "target_tensors": ["attention", "norm", "ffn"],
        "params": [
            {"keep_ratio": 0.08},
            {"block_size": 16},
            {"group_size": 32},
            {},
        ],
    },
    "svd_fwht_int4": {
        "stages": ["svd_compress", "fwht_compress", "block_int4"],
        "description": "SVD (~5x) leaves spectral residual for FWHT (~2x) "
        "and INT4 (~4x). Total = 40x.",
        "expected_ratio": 40.0,
        "target_tensors": ["attention", "weight"],
        "params": [
            {"rank": "auto:200"},
            {"keep_ratio": 0.15},
            {"block_size": 16},
        ],
    },
    # ── 4-Stage Cascades ──────────────────────────────────
    "svd_int4_sparse_huffman": {
        "stages": ["svd_compress", "block_int4", "sparsity_int4", "huffman"],
        "description": "Aggressive SVD (~5x) leaves INT4-compressible residual (~4x), "
        "sparse residual (~3x), Huffman (~4x). Total = 240x.",
        "expected_ratio": 240.0,
        "target_tensors": ["attention", "ffn", "weight", "embedding"],
        "params": [
            {"rank": "auto:200"},
            {"block_size": 16},
            {"group_size": 32},
            {},
        ],
    },
    "dct_int4_sparse_huffman": {
        "stages": ["dct_spectral", "block_int4", "sparsity_int4", "huffman"],
        "description": "DCT at low keep_ratio (~3x) + INT4 (~4x) + "
        "Sparsity (~3x) + Huffman (~4x) = 144x.",
        "expected_ratio": 144.0,
        "target_tensors": ["norm", "attention", "ffn"],
        "params": [
            {"keep_ratio": 0.06},
            {"block_size": 16},
            {"group_size": 32},
            {},
        ],
    },
    "svd_delta_huffman_rans": {
        "stages": ["svd_compress", "delta_int4", "huffman", "rans"],
        "description": "SVD (~5x) + Delta INT4 (~4x) + Huffman (~3x) "
        "+ RANS (~3x) = 180x.",
        "expected_ratio": 180.0,
        "target_tensors": ["ffn", "output", "weight"],
        "params": [
            {"rank": "auto:200"},
            {"block_size": 32},
            {},
            {},
        ],
    },
    "svd_sparse_rans": {
        "stages": ["svd_compress", "sparsity_int4", "rans"],
        "description": "Aggressive SVD (~6x) + Sparsity (~3x) + RANS (~3x) = 54x.",
        "expected_ratio": 54.0,
        "target_tensors": ["ffn", "attention"],
        "params": [
            {"rank": "auto:200"},
            {"group_size": 32},
            {},
        ],
    },
    # ── 5-Stage Cascades (for 200:1+) ─────────────────────
    "tt_quant_sparse_fwht_huffman": {
        "stages": [
            "tensor_train",
            "delta_int4",
            "sparsity_int4",
            "fwht_compress",
            "huffman",
        ],
        "description": "TT rank 4 (~3x) + Delta INT4 (~3x) + Sparsity (~3x) + "
        "FWHT (~2x) + Huffman (~4x) = 216x.",
        "expected_ratio": 216.0,
        "target_tensors": ["ffn", "embedding"],
        "params": [
            {"rank": 4},
            {"block_size": 32},
            {"group_size": 32},
            {"keep_ratio": 0.1},
            {},
        ],
    },
    "svd_tt_quant_sparse_huffman": {
        "stages": [
            "svd_compress",
            "tensor_train",
            "block_int4",
            "sparsity_int4",
            "huffman",
        ],
        "description": "SVD (~5x) + TT on residual (~2x) + INT4 (~4x) + "
        "Sparsity (~3x) + Huffman (~4x) = 480x.",
        "expected_ratio": 480.0,
        "target_tensors": ["ffn", "attention", "weight"],
        "params": [
            {"rank": "auto:200"},
            {"rank": 4},
            {"block_size": 16},
            {"group_size": 32},
            {},
        ],
    },
    "svd_dct_int4_sparse_huffman": {
        "stages": [
            "svd_compress",
            "dct_spectral",
            "block_int4",
            "sparsity_int4",
            "huffman",
        ],
        "description": "SVD (~5x) + DCT on residual (~2x) + INT4 (~4x) + "
        "Sparsity (~3x) + Huffman (~4x) = 480x.",
        "expected_ratio": 480.0,
        "target_tensors": ["attention", "ffn"],
        "params": [
            {"rank": "auto:200"},
            {"keep_ratio": 0.12},
            {"block_size": 16},
            {"group_size": 32},
            {},
        ],
    },
    # ── Embedding-specific (huge vocab matrices) ──────────
    "embedding_triple_cascade": {
        "stages": ["svd_compress", "block_int4", "sparsity_int4", "huffman"],
        "description": "Large vocab embeddings. "
        "SVD at rank auto:200 (~32x for 262k x 1536), "
        "INT4 (~4x), Sparsity (~3x), Huffman (~4x) = 1536x.",
        "expected_ratio": 1500.0,
        "target_tensors": ["embedding", "output"],
        "params": [
            {"rank": "auto:200"},
            {"block_size": 16},
            {"group_size": 32},
            {},
        ],
    },
    "embedding_svd_rans": {
        "stages": ["svd_compress", "rans"],
        "description": "SVD at auto:500 + RANS entropy for embedding matrices.",
        "expected_ratio": 150.0,
        "target_tensors": ["embedding"],
        "params": [
            {"rank": "auto:500"},
            {},
        ],
    },
    # ── Norm-specific (small vectors, batch effect) ───────
    "norm_aggressive": {
        "stages": ["block_int4", "huffman"],
        "description": "Norms are small vectors. INT4 (~4x) + Huffman (~4x) = 16x. "
        "Batch many norms together for better ratio.",
        "expected_ratio": 16.0,
        "target_tensors": ["norm"],
        "params": [
            {"block_size": 16},
            {},
        ],
    },
    "norm_fwht_int4": {
        "stages": ["fwht_compress", "block_int4"],
        "description": "FWHT (~2x) + INT4 (~4x) = 8x for norm vectors.",
        "expected_ratio": 8.0,
        "target_tensors": ["norm"],
        "params": [
            {"keep_ratio": 0.15},
            {"block_size": 16},
        ],
    },
}


def select_aggressive_cascade(
    tensor: np.ndarray,
    tensor_type: str = "weight",
    target_ratio: float = 200.0,
    n_elements: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Select the best aggressive cascade pattern for a tensor.

    Selection logic:
    - 1D tensors (norms, biases) -> norm_aggressive or norm_fwht_int4
    - Embeddings -> embedding_triple_cascade
    - target_ratio >= 500 -> 5-stage cascades
    - target_ratio >= 200 -> 4-stage cascades
    - target_ratio >= 100 -> 3-stage cascades
    - target_ratio >= 50 -> 2-stage cascades

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to compress.
    tensor_type : str
        Type hint.
    target_ratio : float
        Desired compression ratio.
    n_elements : int, optional
        Number of elements (overrides tensor.size).

    Returns
    -------
    dict or None
        Cascade config dict with keys: stages, params, expected_ratio, description, target_tensors.
        None if no cascade is suitable.
    """
    ne = n_elements if n_elements is not None else tensor.size
    ndim = tensor.ndim

    # 1D tensors
    if ndim <= 1:
        if target_ratio >= 10:
            return AGGRESSIVE_CASCADES.get("norm_aggressive")
        return AGGRESSIVE_CASCADES.get("norm_fwht_int4")

    # Embeddings (only by type name)
    if (
        tensor_type == "embedding"
        or tensor_type == "output"
        or "embed" in str(tensor_type).lower()
    ):
        if target_ratio >= 100:
            return AGGRESSIVE_CASCADES.get("embedding_triple_cascade")
        return AGGRESSIVE_CASCADES.get("embedding_svd_rans")

    # Ratio-based selection
    if target_ratio >= 500:
        for name in [
            "svd_tt_quant_sparse_huffman",
            "svd_dct_int4_sparse_huffman",
            "tt_quant_sparse_fwht_huffman",
        ]:
            cfg = AGGRESSIVE_CASCADES.get(name)
            if cfg and (
                tensor_type in cfg.get("target_tensors", [])
                or any(tt in tensor_type for tt in cfg.get("target_tensors", []))
            ):
                return cfg
        return AGGRESSIVE_CASCADES.get("svd_tt_quant_sparse_huffman")

    if target_ratio >= 200:
        for name in [
            "svd_int4_sparse_huffman",
            "dct_int4_sparse_huffman",
            "svd_delta_huffman_rans",
        ]:
            cfg = AGGRESSIVE_CASCADES.get(name)
            if cfg and (
                tensor_type in cfg.get("target_tensors", [])
                or any(tt in tensor_type for tt in cfg.get("target_tensors", []))
            ):
                return cfg
        return AGGRESSIVE_CASCADES.get("svd_int4_sparse_huffman")

    if target_ratio >= 100:
        for name in [
            "tt_quant_sparse_huffman",
            "fwht_int4_sparse_rans",
            "svd_lowrank_int4_huffman",
        ]:
            cfg = AGGRESSIVE_CASCADES.get(name)
            if cfg and (
                tensor_type in cfg.get("target_tensors", [])
                or any(tt in tensor_type for tt in cfg.get("target_tensors", []))
            ):
                return cfg
        return AGGRESSIVE_CASCADES.get("svd_lowrank_int4_huffman")

    if target_ratio >= 50:
        for name in ["svd_fwht_int4", "svd_sparse_rans"]:
            cfg = AGGRESSIVE_CASCADES.get(name)
            if cfg and (
                tensor_type in cfg.get("target_tensors", [])
                or any(tt in tensor_type for tt in cfg.get("target_tensors", []))
            ):
                return cfg
        return AGGRESSIVE_CASCADES.get("svd_int4")

    if target_ratio >= 20:
        return AGGRESSIVE_CASCADES.get("svd_huffman")

    return None


def build_cascade_stages(
    engine: Any,
    tensor: np.ndarray,
    cascade_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Execute a cascade config on a tensor, returning stage results.

    Each stage compresses the current residual and accumulates
    into the reconstruction.

    Parameters
    ----------
    engine : CompressionIntelligenceEngine
        Engine with _methods registry.
    tensor : np.ndarray
        Float32 tensor to compress.
    cascade_config : dict
        Cascade config with keys: stages, params.

    Returns
    -------
    list of dict
        Stage results with keys: method, params, compressed_data, metadata,
        stage_ratio, stage_error, recon.
    """
    method_names = cascade_config.get("stages", [])
    params_list = cascade_config.get("params", [{} for _ in method_names])

    original = np.ascontiguousarray(tensor, dtype=np.float32)
    residual = original.copy().astype(np.float64)
    reconstruction = np.zeros_like(original, dtype=np.float64)

    stages: List[Dict[str, Any]] = []

    for i, method_name in enumerate(method_names):
        params = params_list[i] if i < len(params_list) else {}
        inst = engine._methods.get(method_name)
        if inst is None:
            logger.warning("Method '%s' not found, skipping stage %d", method_name, i)
            continue

        try:
            resolved = dict(params)
            if method_name == "svd_compress" and "rank" in params:
                if isinstance(params["rank"], str) and params["rank"].startswith(
                    "auto:"
                ):
                    divisor = int(params["rank"].split(":")[1])
                    min_dim = min(s for s in original.shape if s > 0)
                    resolved["rank"] = max(min_dim // divisor, 2)

            stage_input = np.ascontiguousarray(residual, dtype=np.float32)
            data, meta = inst.compress(stage_input, **resolved)
            recon = inst.decompress(data, meta)
            if recon.shape != original.shape:
                recon = recon.reshape(original.shape)

            stage_ratio = float(original.nbytes / max(len(data), 1))
            stage_error = float(
                np.abs(residual.ravel() - recon.ravel().astype(np.float64)).mean()
            )

            reconstruction += recon.astype(np.float64)
            residual = original.astype(np.float64) - reconstruction

            stages.append(
                {
                    "method": method_name,
                    "params": resolved,
                    "compressed_data": data,
                    "metadata": meta,
                    "stage_ratio": stage_ratio,
                    "stage_error": stage_error,
                    "recon": recon,
                }
            )

        except Exception as exc:
            logger.warning("Stage %d (%s) failed: %s", i, method_name, exc)
            continue

    del residual, reconstruction
    gc.collect()

    return stages


def estimate_cascade_ratio(cascade_config: Dict[str, Any]) -> float:
    """Estimate the expected compression ratio for a cascade config."""
    return cascade_config.get("expected_ratio", 10.0)


def estimate_cascade_error(cascade_config: Dict[str, Any]) -> float:
    """Estimate the expected cumulative error for a cascade config.

    Error is additive across stages (each stage captures residual
    that previous stages didn't). Expected per-stage errors:
    - SVD: 0.5%
    - TT/DCT/FWHT: 0.3%
    - INT4/DeltaINT4: 1.0%
    - SparsityINT4: 0.5%
    - Huffman/RANS: 0.0% (lossless)
    """
    stage_error_map = {
        "svd_compress": 0.005,
        "tensor_train": 0.005,
        "dct_spectral": 0.003,
        "fwht_compress": 0.003,
        "block_int4": 0.01,
        "hadamard_int4": 0.01,
        "delta_int4": 0.01,
        "sparsity_int4": 0.005,
        "huffman": 0.0,
        "rans": 0.0,
    }

    stages = cascade_config.get("stages", [])
    total_error = sum(stage_error_map.get(m, 0.01) for m in stages)
    return min(total_error, 0.10)


def get_aggressive_pattern_names(min_ratio: float = 50.0) -> List[str]:
    """Get names of aggressive cascade patterns that meet a minimum ratio."""
    return [
        name
        for name, cfg in AGGRESSIVE_CASCADES.items()
        if cfg.get("expected_ratio", 0) >= min_ratio
    ]


def get_aggressive_patterns_by_target(
    target_tensor: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Get aggressive cascade patterns matching a target tensor type."""
    results: List[Tuple[str, Dict[str, Any]]] = []
    for name, cfg in AGGRESSIVE_CASCADES.items():
        targets = cfg.get("target_tensors", [])
        if target_tensor in targets or any(t in target_tensor for t in targets):
            results.append((name, cfg))
    return sorted(results, key=lambda x: -x[1].get("expected_ratio", 0))

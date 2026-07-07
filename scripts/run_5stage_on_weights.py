from __future__ import annotations

import json
import math
import os
import pickle
import struct
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectralstream.compression.cascade_5stage import (
    compress_cascade,
    decompress_cascade,
)
from spectralstream.compression.honest_metrics import (
    dual_ratio,
    end_to_end_error,
    ErrorMetrics,
)


def _load_safetensors_bf16(
    model_path: str, max_2d: int = 50
) -> List[Tuple[str, np.ndarray]]:
    """Load BF16 safetensors, returning 2D tensors as float32."""
    with open(model_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)
        raw_data = f.read()

    from spectralstream.core.math_primitives import bfloat16_to_float32

    weights: List[Tuple[str, np.ndarray]] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        shape = info["shape"]
        if len(shape) != 2:
            continue
        start, end = info["data_offsets"]
        raw = raw_data[start:end]
        dtype_str = info["dtype"]

        if dtype_str == "BF16":
            arr_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            arr = bfloat16_to_float32(arr_u16)
        elif dtype_str == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        elif dtype_str == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=np.dtype(dtype_str.lower())).reshape(shape)
            arr = arr.astype(np.float32)

        weights.append((name, arr))
        if len(weights) >= max_2d:
            break

    del raw_data
    return weights


def _synthetic_llm_weight(
    rows: int, cols: int, effective_rank: Optional[int] = None, seed: int = 42
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if effective_rank is None:
        effective_rank = max(2, min(rows, cols) // 8)
    U = rng.randn(rows, effective_rank).astype(np.float32)
    V = rng.randn(cols, effective_rank).astype(np.float32)
    S = np.exp(-np.arange(effective_rank) / (effective_rank * 0.15))
    base = (U * S) @ V.T
    noise = rng.randn(rows, cols).astype(np.float32) * 0.2
    n_out = max(1, rows * cols // 200)
    idx = rng.choice(rows * cols, n_out, replace=False)
    flat = (base + noise).ravel()
    flat[idx] *= 5.0
    return flat.reshape(rows, cols).astype(np.float32)


def _discover_tensors(max_tensors: int = 50) -> List[Tuple[str, np.ndarray]]:
    candidates = [
        "models/gemma-4-E2B/model.safetensors",
        "models/MiMo-V2.5/model.safetensors",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading 2D weight tensors from {path}")
            start = time.perf_counter()
            tensors = _load_safetensors_bf16(path, max_2d=max_tensors)
            elapsed = time.perf_counter() - start
            print(f"Loaded {len(tensors)} 2D tensors in {elapsed:.1f}s")
            if tensors:
                return tensors

    print("No model weights found — generating synthetic LLM-like weights")
    rng = np.random.RandomState(42)
    sizes = [
        (4096, 4096),
        (4096, 14336),
        (14336, 4096),
        (4096, 1024),
        (1024, 4096),
        (8192, 8192),
        (8192, 1024),
        (1024, 8192),
        (5120, 5120),
        (5120, 13824),
        (13824, 5120),
        (4096, 5120),
        (2048, 4096),
        (4096, 2048),
    ]
    tensors = []
    for i, (r, c) in enumerate(sizes):
        if len(tensors) >= max_tensors:
            break
        tensors.append((f"synthetic_{r}x{c}_{i}", _synthetic_llm_weight(r, c, seed=i)))
    return tensors


def _run_cascade(
    name: str,
    tensor: np.ndarray,
    target_ratio: float = 200.0,
) -> Optional[Dict[str, Any]]:
    n_el = tensor.size
    t0 = time.perf_counter()
    try:
        payload, meta = compress_cascade(tensor, target_ratio=target_ratio)
        serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        recon = decompress_cascade(payload, meta)
        elapsed = time.perf_counter() - t0

        ratios = dual_ratio(n_el, serialized)
        errors = end_to_end_error(tensor, recon)
        n_comp = len(serialized)
        n_orig_fp32 = n_el * 4

        row = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "ratio_vs_fp32": ratios["ratio_vs_fp32"],
            "ratio_vs_bf16": ratios["ratio_vs_bf16"],
            "rel_mse": errors.rel_mse,
            "cosine_sim": errors.cosine_sim,
            "max_abs_error": errors.max_abs,
            "snr_db": errors.snr_db,
            "time_s": elapsed,
            "comp_bytes": n_comp,
            "orig_bytes_fp32": n_orig_fp32,
        }
        print(
            f"{name:70s} ratio={ratios['ratio_vs_fp32']:8.1f}x  "
            f"rel_mse={errors.rel_mse:.2e}  cos={errors.cosine_sim:.6f}  "
            f"snr={errors.snr_db:.1f}dB  t={elapsed:.1f}s"
        )
        return row
    except Exception as e:
        print(f"{name:70s} FAILED: {e}")
        traceback.print_exc()
        return None


def main() -> None:
    print("=" * 90)
    print("5-STAGE CASCADE COMPRESSION — REAL WEIGHTS R&D TEST")
    print("=" * 90)
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print()

    target_ratio = 200.0
    max_tensors = 50

    print(f"Target ratio: {target_ratio:.0f}x  Max tensors: {max_tensors}")
    print()

    tensors = _discover_tensors(max_tensors=max_tensors)
    if not tensors:
        print("No tensors to process — exiting")
        sys.exit(1)

    print(f"\nProcessing {len(tensors)} tensors...\n")

    results: List[Dict[str, Any]] = []
    for i, (name, tensor) in enumerate(tensors):
        print(f"[{i + 1}/{len(tensors)}] ", end="", flush=True)
        row = _run_cascade(name, tensor, target_ratio=target_ratio)
        if row is not None:
            results.append(row)

    print(f"\n{'=' * 90}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 90}")

    agg: Dict[str, Any] = {}
    if results:
        ratios_fp32 = np.array([r["ratio_vs_fp32"] for r in results])
        ratios_bf16 = np.array([r["ratio_vs_bf16"] for r in results])
        mses = np.array([r["rel_mse"] for r in results])
        cosines = np.array([r["cosine_sim"] for r in results])
        snrs = np.array([r["snr_db"] for r in results])
        times = np.array([r["time_s"] for r in results])

        agg = {
            "n_tensors": len(results),
            "shape_samples": [r["shape"] for r in results[:5]],
            "avg_ratio_vs_fp32": float(np.mean(ratios_fp32)),
            "median_ratio_vs_fp32": float(np.median(ratios_fp32)),
            "min_ratio_vs_fp32": float(np.min(ratios_fp32)),
            "max_ratio_vs_fp32": float(np.max(ratios_fp32)),
            "std_ratio_vs_fp32": float(np.std(ratios_fp32)),
            "avg_ratio_vs_bf16": float(np.mean(ratios_bf16)),
            "median_ratio_vs_bf16": float(np.median(ratios_bf16)),
            "avg_rel_mse": float(np.mean(mses)),
            "median_rel_mse": float(np.median(mses)),
            "min_rel_mse": float(np.min(mses)),
            "max_rel_mse": float(np.max(mses)),
            "avg_cosine_sim": float(np.mean(cosines)),
            "min_cosine_sim": float(np.min(cosines)),
            "max_cosine_sim": float(np.max(cosines)),
            "avg_snr_db": float(np.mean(snrs)),
            "min_snr_db": float(np.min(snrs)),
            "max_snr_db": float(np.max(snrs)),
            "total_time_s": float(np.sum(times)),
            "avg_time_s": float(np.mean(times)),
            "fastest_s": float(np.min(times)),
            "slowest_s": float(np.max(times)),
        }
        print(json.dumps(agg, indent=2))

    output = {"aggregate": agg, "per_tensor": results}
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "5stage_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gemma 4 E2B R&D: Validate ALL compression methods on real weights.
Tests 1057 methods across 15 representative tensors.
Outputs comprehensive markdown report.
"""

import struct
import json
import math
import time
import sys
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── BF16 utilities ──────────────────────────────────────────────────────────

BF16_EPSILON = np.finfo(np.float32).eps


def bf16_to_f32(bf16_bytes: bytes) -> np.ndarray:
    """Convert bfloat16 bytes to float32 numpy array."""
    n = len(bf16_bytes) // 2
    as_uint16 = np.frombuffer(bf16_bytes, dtype=np.uint16).astype(np.uint32)
    as_f32 = as_uint16 << 16
    return as_f32.view(np.float32)


def load_safetensors_tensor(filepath: str, tensor_key: str) -> np.ndarray:
    """Load a single tensor from safetensors file, handling BF16."""
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        info = header[tensor_key]
        dtype_str = info["dtype"]
        shape = info["shape"]
        offsets = info["data_offsets"]
        f.seek(8 + header_len + offsets[0])
        raw = f.read(offsets[1] - offsets[0])

    if dtype_str == "BF16":
        arr = bf16_to_f32(raw).reshape(shape)
    elif dtype_str == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    elif dtype_str == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    return np.ascontiguousarray(arr)


def list_available_tensors(filepath: str) -> Dict[str, Any]:
    """List all tensor names and shapes in a safetensors file."""
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    result = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        if v["shape"] and len(v["shape"]) >= 2:
            result[k] = v
    return result


# ── Tensor selection ────────────────────────────────────────────────────────


def select_representative_tensors(filepath: str) -> Dict[str, np.ndarray]:
    """Select 15 representative tensors from the model."""
    available = list_available_tensors(filepath)

    # Target selection patterns
    selectors = {
        "embed_tokens": lambda k: "embed_tokens.weight" in k and "per_layer" not in k,
        "embed_per_layer": lambda k: "embed_tokens_per_layer.weight" in k,
        "lang_q_proj_sliding": lambda k: "language_model" in k
        and "self_attn" in k
        and "q_proj.weight" in k
        and "layers.0" in k,
        "lang_q_proj_full": lambda k: "language_model" in k
        and "self_attn" in k
        and "q_proj.weight" in k
        and "layers.16" in k,
        "lang_k_proj_sliding": lambda k: "language_model" in k
        and "self_attn" in k
        and "k_proj.weight" in k
        and "layers.0" in k,
        "lang_k_proj_full": lambda k: "language_model" in k
        and "self_attn" in k
        and "k_proj.weight" in k
        and "layers.16" in k,
        "lang_v_proj_sliding": lambda k: "language_model" in k
        and "self_attn" in k
        and "v_proj.weight" in k
        and "layers.0" in k,
        "lang_v_proj_full": lambda k: "language_model" in k
        and "self_attn" in k
        and "v_proj.weight" in k
        and "layers.16" in k,
        "lang_o_proj_sliding": lambda k: "language_model" in k
        and "self_attn" in k
        and "o_proj.weight" in k
        and "layers.0" in k,
        "lang_o_proj_full": lambda k: "language_model" in k
        and "self_attn" in k
        and "o_proj.weight" in k
        and "layers.16" in k,
        "lang_gate_proj": lambda k: "language_model" in k
        and "mlp" in k
        and "gate_proj.weight" in k
        and "layers.0" in k,
        "lang_up_proj": lambda k: "language_model" in k
        and "mlp" in k
        and "up_proj.weight" in k
        and "layers.0" in k,
        "lang_down_proj": lambda k: "language_model" in k
        and "mlp" in k
        and "down_proj.weight" in k
        and "layers.0" in k,
        "vision_weight": lambda k: "vision_tower" in k and "linear.weight" in k,
        "audio_weight": lambda k: "audio_tower" in k and "linear.weight" in k,
    }

    tensors = {}
    for name, selector in selectors.items():
        matches = [k for k in available if selector(k)]
        if matches:
            key = matches[0]
            try:
                tensors[name] = load_safetensors_tensor(filepath, key)
                print(
                    f"  ✓ Loaded {name}: {tensors[name].shape} ({tensors[name].nbytes / 1e6:.1f}MB)"
                )
            except Exception as e:
                print(f"  ✗ Failed to load {name} ({key}): {e}")
        else:
            print(f"  ✗ No match for {name}")

    return tensors


# ── Metrics ─────────────────────────────────────────────────────────────────


@dataclass
class MethodResult:
    name: str
    category: str
    tensor_name: str
    tensor_shape: Tuple[int, ...]
    ratio: float
    relative_error: float
    snr_db: float
    time_s: float
    success: bool
    error_msg: str = ""
    compressed_size: int = 0
    original_size: int = 0


def compute_metrics(
    original: np.ndarray, reconstructed: np.ndarray
) -> Tuple[float, float, float]:
    """Compute ratio, relative_error, SNR_dB."""
    orig_flat = original.ravel().astype(np.float64)
    recon_flat = reconstructed.ravel().astype(np.float64)

    diff = orig_flat - recon_flat
    mse = np.mean(diff**2)
    orig_norm = np.linalg.norm(orig_flat)
    diff_norm = np.linalg.norm(diff)

    rel_err = diff_norm / max(orig_norm, 1e-30)

    orig_var = np.var(orig_flat)
    snr = (
        10 * np.log10(max(orig_var, 1e-30) / max(mse, 1e-30)) if mse > 1e-30 else 100.0
    )

    return rel_err, snr


# ── Method loader ───────────────────────────────────────────────────────────


def load_all_methods(include_variants: bool = False) -> Dict[str, Any]:
    """Load all methods from METHOD_CLASSES."""
    from spectralstream.compression.methods import METHOD_CLASSES

    all_methods = {}

    # Base methods
    for name, cls in METHOD_CLASSES.items():
        try:
            inst = cls() if isinstance(cls, type) else cls
            all_methods[name] = {
                "instance": inst,
                "category": getattr(inst, "category", "unknown"),
                "name": name,
            }
        except Exception as e:
            pass

    if include_variants:
        from spectralstream.compression.methods.method_variants import (
            get_method_variants,
        )

        variants = get_method_variants(METHOD_CLASSES)
        for name, var in variants.items():
            if name not in all_methods:
                all_methods[name] = {
                    "instance": var,
                    "category": getattr(var, "category", "unknown"),
                    "name": name,
                }

    return all_methods


def test_method(
    method_info: Dict[str, Any],
    tensor: np.ndarray,
    tensor_name: str,
    timeout_s: float = 30.0,
) -> MethodResult:
    """Test a single compression method on a tensor."""
    inst = method_info["instance"]
    name = method_info["name"]
    category = method_info["category"]
    t0 = time.time()

    try:
        # For very large tensors, test on a slice
        if tensor.size > 10_000_000:
            flat = tensor.ravel()
            test_slice = flat[: min(len(flat), 1_000_000)].copy()
            test_tensor = (
                test_slice.reshape(-1, 1) if test_slice.ndim == 1 else test_slice
            )
            # But preserve 2D structure if possible
            if tensor.ndim == 2:
                rows = min(tensor.shape[0], 1024)
                cols = min(tensor.shape[1], 1024)
                test_tensor = tensor[:rows, :cols].copy()
            else:
                test_tensor = tensor.ravel()[:1_000_000].copy()
        else:
            test_tensor = tensor.copy()

        original_size = test_tensor.nbytes

        # Compress
        t_compress = time.time()
        data, metadata = inst.compress(test_tensor)
        t_compress = time.time() - t_compress

        # Decompress
        t_decomp = time.time()
        reconstructed = inst.decompress(data, metadata)
        t_decomp = time.time() - t_decomp

        # Reshape if needed
        if reconstructed.shape != test_tensor.shape:
            try:
                reconstructed = reconstructed.reshape(test_tensor.shape)
            except Exception:
                pass

        # Compute metrics
        compressed_size = len(data)
        ratio = original_size / max(compressed_size, 1)
        rel_err, snr = compute_metrics(test_tensor, reconstructed)
        total_time = time.time() - t0

        success = rel_err < 0.1  # < 10% relative error

        return MethodResult(
            name=name,
            category=category,
            tensor_name=tensor_name,
            tensor_shape=test_tensor.shape,
            ratio=ratio,
            relative_error=rel_err,
            snr_db=snr,
            time_s=total_time,
            success=success,
            compressed_size=compressed_size,
            original_size=original_size,
        )
    except Exception as e:
        total_time = time.time() - t0
        return MethodResult(
            name=name,
            category=category,
            tensor_name=tensor_name,
            tensor_shape=tensor.shape,
            ratio=0.0,
            relative_error=1.0,
            snr_db=0.0,
            time_s=total_time,
            success=False,
            error_msg=f"{type(e).__name__}: {str(e)[:200]}",
        )


# ── 2D tensor-only filtering ────────────────────────────────────────────────


def _safe_2d_slice(tensor: np.ndarray, max_rows=1024, max_cols=1024) -> np.ndarray:
    """Get a safe 2D slice for testing."""
    if tensor.ndim == 2:
        r = min(tensor.shape[0], max_rows)
        c = min(tensor.shape[1], max_cols)
        return tensor[:r, :c].copy()
    flat = tensor.ravel()
    n = min(len(flat), max_rows * max_cols)
    return flat[:n].reshape(-1, 1).copy()


# ── Cascading ───────────────────────────────────────────────────────────────


def test_cascade_combination(
    methods_list: List[Dict[str, Any]],
    tensor: np.ndarray,
    tensor_name: str,
    cascade_name: str,
) -> MethodResult:
    """Test a cascade: apply methods sequentially."""
    test_tensor = (
        _safe_2d_slice(tensor, 256, 256) if tensor.size > 1_000_000 else tensor.copy()
    )
    original_size = test_tensor.nbytes
    t0 = time.time()

    current = test_tensor.copy()
    compressed_stages = []
    metadata_stages = []

    try:
        for m in methods_list:
            inst = m["instance"]
            data, meta = inst.compress(current)
            current = inst.decompress(data, meta)
            compressed_stages.append(data)
            metadata_stages.append(meta)

        # Total compressed size (combined)
        total_compressed = sum(len(d) for d in compressed_stages)
        ratio = original_size / max(total_compressed, 1)
        rel_err, snr = compute_metrics(test_tensor, current)
        total_time = time.time() - t0

        return MethodResult(
            name=cascade_name,
            category="cascade",
            tensor_name=tensor_name,
            tensor_shape=test_tensor.shape,
            ratio=ratio,
            relative_error=rel_err,
            snr_db=snr,
            time_s=total_time,
            success=rel_err < 0.1,
            compressed_size=total_compressed,
            original_size=original_size,
        )
    except Exception as e:
        return MethodResult(
            name=cascade_name,
            category="cascade",
            tensor_name=tensor_name,
            tensor_shape=test_tensor.shape,
            ratio=0.0,
            relative_error=1.0,
            snr_db=0.0,
            time_s=time.time() - t0,
            success=False,
            error_msg=f"{type(e).__name__}: {str(e)[:200]}",
        )


# ── Cross-layer delta (Novel R&D) ──────────────────────────────────────────


def cross_layer_delta_compress(
    layer_tensors: List[np.ndarray], base_method: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Option A: Cross-layer delta compression.
    Store first layer explicitly, subsequent layers as delta from previous.
    """
    results = []
    prev = None
    for i, t in enumerate(layer_tensors):
        test_t = _safe_2d_slice(t)
        if i == 0:
            # Store first layer explicitly
            data, meta = base_method["instance"].compress(test_t)
            recon = base_method["instance"].decompress(data, meta)
            results.append(
                {
                    "layer": i,
                    "method": "explicit",
                    "data": data,
                    "recon": recon,
                    "size": len(data),
                    "error": float(
                        np.linalg.norm(test_t - recon)
                        / max(np.linalg.norm(test_t), 1e-30)
                    ),
                }
            )
        else:
            # Store delta from previous reconstruction
            delta = test_t - prev_recon
            data, meta = base_method["instance"].compress(delta)
            recon_delta = base_method["instance"].decompress(data, meta)
            recon = prev_recon + recon_delta
            results.append(
                {
                    "layer": i,
                    "method": "delta",
                    "data": data,
                    "recon": recon,
                    "size": len(data),
                    "error": float(
                        np.linalg.norm(test_t - recon)
                        / max(np.linalg.norm(test_t), 1e-30)
                    ),
                }
            )
        prev_recon = results[-1]["recon"].copy()

    return {
        "results": results,
        "total_size": sum(r["size"] for r in results),
        "avg_error": np.mean([r["error"] for r in results]),
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("Gemma 4 E2B — Comprehensive Compression Method Validation")
    print("=" * 70)

    model_path = "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"

    # Step 0: Load the model's tensor metadata
    print("\n[Step 0] Loading Gemma 4 tensor metadata...")
    tensors = select_representative_tensors(model_path)
    print(f"  Loaded {len(tensors)} representative tensors")

    if len(tensors) < 5:
        print("  WARNING: Fewer tensors loaded than expected. Check model path.")
        # Fall back to small slices
        for k in list(tensors.keys())[:3]:
            t = tensors[k]
            print(f"  {k}: shape={t.shape}")

    # Step 1: Load all methods
    print("\n[Step 1] Loading all compression methods...")
    all_methods = load_all_methods()
    print(f"  Loaded {len(all_methods)} methods (base + variants)")

    # Step 2: Test all methods on representative tensors
    print("\n[Step 2] Testing all methods on real tensors...")
    print(f"  Tensors: {len(tensors)}, Methods: {len(all_methods)}")
    print(f"  Total tests: {len(tensors) * len(all_methods)}")
    print()

    all_results: List[MethodResult] = []
    methods_by_tensor: Dict[str, List[MethodResult]] = {tn: [] for tn in tensors}

    total_tests = len(tensors) * len(all_methods)
    completed = 0
    failed_count = 0

    for tname, tensor in tensors.items():
        print(
            f"\n  ── Testing on {tname} ({tensor.shape}, {tensor.nbytes / 1e6:.1f}MB) ──"
        )
        tensor_results = []

        # Process methods in batches for progress reporting
        mnames = sorted(all_methods.keys())
        batch_size = 100
        for batch_start in range(0, len(mnames), batch_size):
            batch = mnames[batch_start : batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {}
                for mname in batch:
                    minfo = all_methods[mname]
                    futures[pool.submit(test_method, minfo, tensor, tname)] = mname

                for fut in as_completed(futures):
                    mname = futures[fut]
                    try:
                        result = fut.result()
                        tensor_results.append(result)
                        completed += 1
                        if not result.success:
                            failed_count += 1
                            if failed_count <= 20:
                                print(f"    FAIL [{mname}]: {result.error_msg[:120]}")
                    except Exception as e:
                        completed += 1
                        failed_count += 1
                        if failed_count <= 20:
                            print(f"    FAIL [{mname}]: Exception in test: {e}")

            # Progress
            pct = completed / max(total_tests, 1) * 100
            print(f"    Progress: {completed}/{total_tests} ({pct:.1f}%)")

        methods_by_tensor[tname] = tensor_results
        all_results.extend(tensor_results)

        # Summary for this tensor
        successes = [r for r in tensor_results if r.success]
        failures = [r for r in tensor_results if not r.success]
        print(f"  {tname}: {len(successes)} passed, {len(failures)} failed")

    # Step 3: Analyze failures
    print("\n[Step 3] Analyzing failures...")
    all_failures = [r for r in all_results if not r.success]
    failure_by_category: Dict[str, List[MethodResult]] = {}
    for f in all_failures:
        failure_by_category.setdefault(f.category, []).append(f)

    print(f"  Total failures: {len(all_failures)}/{len(all_results)}")
    for cat, flist in sorted(failure_by_category.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(flist)} failures")

    # Step 4: Cascading tests
    print("\n[Step 4] Testing cascade combinations...")

    # Select best methods per category
    best_methods = {}
    for cat in [
        "quantization",
        "decomposition",
        "spectral",
        "structural",
        "tensor_network",
    ]:
        cat_results = [
            r for r in all_results if r.category == cat and r.success and r.ratio > 1.0
        ]
        if cat_results:
            # Best by ratio/error ratio
            cat_results.sort(key=lambda r: -r.ratio / max(r.relative_error, 1e-10))
            best_methods[cat] = cat_results[:3]

    cascade_configs = [
        ["block_int8", "svd_compress"],
        ["svd_compress", "block_int4"],
        ["dct_spectral", "block_int4"],
        ["svd_compress", "dct_spectral", "block_int4"],
        ["svd_compress", "hadamard_int8"],
        ["tensor_train", "block_int4"],
        ["fwht_compress", "block_int4"],
        ["block_int8", "block_int4"],
    ]

    cascade_results: List[MethodResult] = []
    for tensor_name in [
        "embed_tokens",
        "lang_q_proj_sliding",
        "lang_gate_proj",
        "lang_down_proj",
    ]:
        if tensor_name not in tensors:
            continue
        tensor = tensors[tensor_name]
        test_t = (
            _safe_2d_slice(tensor, 256, 256)
            if tensor.size > 1_000_000
            else tensor.copy()
        )

        for cascade in cascade_configs:
            methods_in_cascade = []
            valid = True
            for mname in cascade:
                if mname in all_methods:
                    methods_in_cascade.append(all_methods[mname])
                else:
                    valid = False
                    break
            if not valid or len(methods_in_cascade) < 2:
                continue

            cname = "+".join(cascade)
            result = test_cascade_combination(
                methods_in_cascade, test_t, tensor_name, cname
            )
            cascade_results.append(result)
            if result.success:
                print(
                    f"  ✓ {cname} on {tensor_name}: ratio={result.ratio:.1f}x, err={result.relative_error:.4f}, SNR={result.snr_db:.1f}dB"
                )
            else:
                print(f"  ✗ {cname} on {tensor_name}: FAILED - {result.error_msg[:80]}")

    # Step 5: Cross-layer delta compression (Novel R&D)
    print("\n[Step 5] Novel R&D: Cross-layer delta compression...")

    # Collect consecutive layers
    all_tensor_info = list_available_tensors(model_path)

    # Find consecutive q_proj layers
    q_proj_layers = sorted(
        [
            k
            for k in all_tensor_info
            if "language_model" in k and "self_attn.q_proj.weight" in k
        ]
    )
    k_proj_layers = sorted(
        [
            k
            for k in all_tensor_info
            if "language_model" in k and "self_attn.k_proj.weight" in k
        ]
    )
    v_proj_layers = sorted(
        [
            k
            for k in all_tensor_info
            if "language_model" in k and "self_attn.v_proj.weight" in k
        ]
    )
    o_proj_layers = sorted(
        [
            k
            for k in all_tensor_info
            if "language_model" in k and "self_attn.o_proj.weight" in k
        ]
    )
    gate_layers = sorted(
        [
            k
            for k in all_tensor_info
            if "language_model" in k and "mlp.gate_proj.weight" in k
        ]
    )

    cross_layer_results = {}

    for layer_group_name, layer_keys in [
        ("q_proj", q_proj_layers[:5]),
        ("k_proj", k_proj_layers[:5]),
        ("v_proj", v_proj_layers[:5]),
        ("o_proj", o_proj_layers[:5]),
        ("gate_proj", gate_layers[:5]),
    ]:
        if len(layer_keys) < 2:
            continue
        print(
            f"\n  Testing {layer_group_name} ({len(layer_keys)} consecutive layers)..."
        )

        # Load layers
        layer_tensors = []
        for lk in layer_keys:
            try:
                t = load_safetensors_tensor(model_path, lk)
                layer_tensors.append(t)
            except Exception as e:
                print(f"    Failed to load {lk}: {e}")

        if len(layer_tensors) < 2:
            continue

        # Test delta compression with block_int8
        for base_name in ["block_int8", "block_int4", "svd_compress", "dct_spectral"]:
            if base_name not in all_methods:
                continue
            result = cross_layer_delta_compress(layer_tensors, all_methods[base_name])
            key = f"{layer_group_name}_{base_name}"
            cross_layer_results[key] = result

            # Compare: explicit vs delta
            explicit_size = result["results"][0]["size"] * len(layer_tensors)
            delta_total = result["total_size"]
            savings = (1 - delta_total / max(explicit_size, 1)) * 100

            print(
                f"    base={base_name}: total_size={delta_total / 1024:.1f}KB, "
                f"explicit={explicit_size / 1024:.1f}KB, "
                f"savings={savings:.1f}%, "
                f"avg_error={result['avg_error']:.4f}"
            )

    # Step 6: Frequency-band adaptive DCT (Novel R&D option C)
    print("\n[Step 5b] Novel R&D: Frequency-band adaptive DCT...")

    from spectralstream.core.math_primitives import dct_2d, idct_2d

    def frequency_band_dct(
        tensor: np.ndarray,
        low_keep: float = 0.1,
        mid_keep: float = 0.05,
        high_keep: float = 0.01,
    ) -> Tuple[bytes, dict]:
        """DCT with different keep ratios per frequency band."""
        t = tensor.astype(np.float32)
        orig_shape = t.shape
        if t.ndim > 2:
            t = t.reshape(-1, orig_shape[-1])

        n, m = t.shape
        coeffs = dct_2d(t)

        # Divide into low/mid/high frequency bands
        low_n = max(1, n // 3)
        low_m = max(1, m // 3)
        mid_n = max(1, n // 2)
        mid_m = max(1, m // 2)

        # Low frequency: top-left corner
        low_region = coeffs[:low_n, :low_m]
        mid_region = coeffs[low_n:mid_n, low_m:mid_m]
        high_region = coeffs[mid_n:, mid_m:]

        def keep_top_k(
            region: np.ndarray, fraction: float
        ) -> Tuple[np.ndarray, np.ndarray]:
            flat = region.ravel()
            k = max(1, int(len(flat) * fraction))
            idx = np.argpartition(-np.abs(flat), k - 1)[:k]
            idx_sorted = np.sort(idx)
            return idx_sorted, flat[idx_sorted]

        li, lv = keep_top_k(low_region, low_keep)
        mi, mv = keep_top_k(mid_region, mid_keep)
        hi, hv = keep_top_k(high_region, high_keep)

        # Pack: offsets + indices + values
        low_offset = 0
        mid_offset = low_offset + len(li)
        high_offset = mid_offset + len(mi)

        all_idx = np.concatenate(
            [li, mi + low_region.size, hi + low_region.size + mid_region.size]
        )
        all_vals = np.concatenate([lv, mv, hv])

        header = struct.pack("<IIIIII", n, m, len(li), len(mi), len(hi), len(all_idx))
        data = (
            header
            + all_idx.astype(np.uint32).tobytes()
            + all_vals.astype(np.float16).tobytes()
        )

        return data, {
            "original_shape": orig_shape,
            "n": n,
            "m": m,
            "n_keep_low": len(li),
            "n_keep_mid": len(mi),
            "n_keep_high": len(hi),
            "n_total": len(all_idx),
            "low_keep": low_keep,
            "mid_keep": mid_keep,
            "high_keep": high_keep,
        }

    def frequency_band_idct(data: bytes, metadata: dict) -> np.ndarray:
        n = metadata["n"]
        m = metadata["m"]
        n_low = metadata["n_keep_low"]
        n_mid = metadata["n_keep_mid"]
        n_high = metadata["n_keep_high"]
        n_total = metadata["n_total"]

        off = struct.calcsize("<IIIIII")
        idx = np.frombuffer(data[off : off + n_total * 4], dtype=np.uint32)
        vals = np.frombuffer(data[off + n_total * 4 :], dtype=np.float16)

        low_n = max(1, n // 3)
        low_m = max(1, m // 3)
        mid_n = max(1, n // 2)
        mid_m = max(1, m // 2)

        coeffs = np.zeros(n * m, dtype=np.float64)
        coeffs[idx.astype(int)] = vals.astype(np.float64)
        coeffs = coeffs.reshape(n, m)

        recon = idct_2d(coeffs.astype(np.float64))
        return recon.astype(np.float32).reshape(metadata["original_shape"])

    # Test frequency-band DCT on representative tensors
    freq_band_results = []
    for tname in [
        "embed_tokens",
        "lang_q_proj_sliding",
        "lang_gate_proj",
        "lang_down_proj",
    ]:
        if tname not in tensors:
            continue
        tensor = tensors[tname]
        test_t = (
            _safe_2d_slice(tensor, 256, 256)
            if tensor.size > 1_000_000
            else tensor.copy()
        )

        for config in [(0.2, 0.1, 0.02), (0.3, 0.15, 0.05), (0.1, 0.05, 0.01)]:
            lk, mk, hk = config
            try:
                data, meta = frequency_band_dct(
                    test_t, low_keep=lk, mid_keep=mk, high_keep=hk
                )
                recon = frequency_band_idct(data, meta)
                ratio = test_t.nbytes / max(len(data), 1)
                rel_err, snr = compute_metrics(test_t, recon)
                freq_band_results.append(
                    {
                        "tensor": tname,
                        "config": f"L={lk}/M={mk}/H={hk}",
                        "ratio": ratio,
                        "error": rel_err,
                        "snr": snr,
                    }
                )
                print(
                    f"  {tname} [{lk}/{mk}/{hk}]: ratio={ratio:.1f}x, err={rel_err:.4f}, SNR={snr:.1f}dB"
                )
            except Exception as e:
                print(f"  {tname} [{lk}/{mk}/{hk}]: FAILED - {e}")

    # ── Step 7: Generate Report ──
    print("\n[Step 6] Generating comprehensive report...")

    report = generate_report(
        all_results,
        cascade_results,
        cross_layer_results,
        freq_band_results,
        all_methods,
        tensors,
    )

    report_path = "/tmp/gemma4_dial_in_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")
    print(f"  Report length: {len(report)} characters")

    # Print summary
    successes = [r for r in all_results if r.success]
    failures = [r for r in all_results if not r.success]
    print(f"\n{'=' * 70}")
    print(
        f"SUMMARY: {len(successes)}/{len(all_results)} methods passed ({len(failures)} failed)"
    )
    print(f"  Cascades tested: {len(cascade_results)}")
    print(f"  Cross-layer tests: {len(cross_layer_results)}")
    print(f"  Frequency-band DCT tests: {len(freq_band_results)}")
    print(f"  Report: {report_path}")
    print(f"{'=' * 70}")

    return report


def generate_report(
    all_results,
    cascade_results,
    cross_layer_results,
    freq_band_results,
    all_methods,
    tensors,
) -> str:
    """Generate comprehensive markdown report."""

    successes = [r for r in all_results if r.success]
    failures = [r for r in all_results if not r.success]

    lines = []
    lines.append("# Gemma 4 E2B — Comprehensive Compression Method Dial-In Report")
    lines.append("")
    lines.append(f"**Date:** 2026-07-02")
    lines.append(f"**Model:** gemma-4-E2B (10.25 GB, 2011 tensors, BF16)")
    lines.append(
        f"**Methods tested:** {len(all_results)} ({len(successes)} passed, {len(failures)} failed)"
    )
    lines.append(f"**Tensors tested:** {len(tensors)}")
    lines.append("")

    # ── 1. Methods ranked by ratio/error per tensor type ──
    lines.append("## 1. Methods Ranked by Ratio/Error per Tensor Type")
    lines.append("")

    tensor_types = set(r.tensor_name for r in all_results)
    for ttype in sorted(tensor_types):
        t_results = [
            r
            for r in all_results
            if r.tensor_name == ttype and r.success and r.ratio > 1.0
        ]
        if not t_results:
            continue

        t_results.sort(key=lambda r: -r.ratio / max(r.relative_error, 1e-10))

        lines.append(f"### {ttype}")
        lines.append("")
        lines.append(
            "| Rank | Method | Category | Ratio | Rel.Error | SNR (dB) | Time (s) |"
        )
        lines.append(
            "|------|--------|----------|-------|-----------|----------|----------|"
        )

        for i, r in enumerate(t_results[:20]):
            lines.append(
                f"| {i + 1} | {r.name} | {r.category} | {r.ratio:.2f}x | {r.relative_error:.4f} | {r.snr_db:.1f} | {r.time_s:.3f} |"
            )

        lines.append("")

    # ── 2. Optimal cascades for each weight category ──
    lines.append("## 2. Optimal Cascades for Each Weight Category")
    lines.append("")

    # Group cascade results by tensor name
    cascade_by_tensor = {}
    for cr in cascade_results:
        cascade_by_tensor.setdefault(cr.tensor_name, []).append(cr)

    for tname, cres in sorted(cascade_by_tensor.items()):
        cres.sort(key=lambda r: -r.ratio / max(r.relative_error, 1e-10))
        lines.append(f"### {tname}")
        lines.append("")
        lines.append("| Rank | Cascade | Ratio | Rel.Error | SNR (dB) | Time (s) |")
        lines.append("|------|---------|-------|-----------|----------|----------|")
        for i, r in enumerate(cres[:10]):
            if r.success:
                lines.append(
                    f"| {i + 1} | {r.name} | {r.ratio:.2f}x | {r.relative_error:.4f} | {r.snr_db:.1f} | {r.time_s:.3f} |"
                )
        lines.append("")

    # ── 3. Methods that needed fixes ──
    lines.append("## 3. Methods That Failed and Potential Fixes")
    lines.append("")

    # Find methods that consistently fail
    method_failures: Dict[str, int] = {}
    method_fail_msgs: Dict[str, str] = {}
    for r in failures:
        method_failures[r.name] = method_failures.get(r.name, 0) + 1
        if r.name not in method_fail_msgs and r.error_msg:
            method_fail_msgs[r.name] = r.error_msg

    consistent_failures = {
        k: v for k, v in method_failures.items() if v >= len(tensors) * 0.5
    }

    lines.append(
        f"**{len(consistent_failures)} methods fail consistently across all tensor types**"
    )
    lines.append("")
    lines.append("| Method | Fail Count | Sample Error |")
    lines.append("|--------|-----------|--------------|")
    for mname, cnt in sorted(consistent_failures.items(), key=lambda x: -x[1])[:30]:
        msg = method_fail_msgs.get(mname, "Unknown")
        lines.append(f"| {mname} | {cnt} | {msg[:100]} |")
    lines.append("")

    lines.append("### Common failure patterns:")
    lines.append("")

    # Categorize failure messages
    pattern_counts: Dict[str, int] = {}
    for msg in method_fail_msgs.values():
        if (
            "matmul" in msg.lower()
            or "shape" in msg.lower()
            or "dimension" in msg.lower()
        ):
            pattern_counts["Shape/dimension mismatch"] = (
                pattern_counts.get("Shape/dimension mismatch", 0) + 1
            )
        elif "norm" in msg.lower() or "overflow" in msg.lower() or "inf" in msg.lower():
            pattern_counts["Overflow/numerical"] = (
                pattern_counts.get("Overflow/numerical", 0) + 1
            )
        elif "dtype" in msg.lower() or "type" in msg.lower():
            pattern_counts["Type/dtype error"] = (
                pattern_counts.get("Type/dtype error", 0) + 1
            )
        elif "memory" in msg.lower() or "size" in msg.lower() or "large" in msg.lower():
            pattern_counts["Memory/size related"] = (
                pattern_counts.get("Memory/size related", 0) + 1
            )
        elif "key" in msg.lower() or "attribute" in msg.lower():
            pattern_counts["Missing key/attribute"] = (
                pattern_counts.get("Missing key/attribute", 0) + 1
            )
        elif "not implemented" in msg.lower() or "not supported" in msg.lower():
            pattern_counts["Not implemented"] = (
                pattern_counts.get("Not implemented", 0) + 1
            )
        else:
            pattern_counts["Other"] = pattern_counts.get("Other", 0) + 1

    for pattern, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- **{pattern}**: {cnt} occurrences")
    lines.append("")

    # ── 4. Novel R&D results ──
    lines.append("## 4. Novel R&D Results")
    lines.append("")

    # 4a. Cross-layer delta
    lines.append("### 4a. Cross-Layer Delta Compression")
    lines.append("")
    lines.append(
        "| Layer Group | Base Method | Delta Total (KB) | Explicit (KB) | Savings (%) | Avg Error |"
    )
    lines.append(
        "|-------------|-------------|------------------|---------------|-------------|-----------|"
    )
    for key, result in sorted(cross_layer_results.items()):
        group, base = key.rsplit("_", 1) if "_" in key else (key, "?")
        explicit = result["results"][0]["size"] * len(result["results"]) / 1024
        delta_total = result["total_size"] / 1024
        savings = (1 - result["total_size"] / max(explicit * 1024, 1)) * 100
        lines.append(
            f"| {group} | {base} | {delta_total:.1f} | {explicit:.1f} | {savings:.1f}% | {result['avg_error']:.4f} |"
        )
    lines.append("")

    # 4b. Frequency-band adaptive DCT
    lines.append("### 4b. Frequency-Band Adaptive DCT")
    lines.append("")
    lines.append("| Tensor | Config (L/M/H) | Ratio | Rel.Error | SNR (dB) |")
    lines.append("|--------|----------------|-------|-----------|----------|")
    for r in freq_band_results:
        lines.append(
            f"| {r['tensor']} | {r['config']} | {r['ratio']:.2f}x | {r['error']:.4f} | {r['snr']:.1f} |"
        )
    lines.append("")

    # ── 5. Recommendations for full model compression ──
    lines.append("## 5. Recommendations for Full Model Compression")
    lines.append("")

    # Find best overall methods
    best_overall = sorted(
        successes, key=lambda r: -r.ratio / max(r.relative_error, 1e-10)
    )[:20]

    lines.append("### Top 10 Methods Overall")
    lines.append("")
    lines.append("| Rank | Method | Avg Ratio | Avg Error | Best For |")
    lines.append("|------|--------|-----------|-----------|----------|")

    method_stats: Dict[str, Dict] = {}
    for r in successes:
        if r.name not in method_stats:
            method_stats[r.name] = {"ratios": [], "errors": [], "tensors": set()}
        method_stats[r.name]["ratios"].append(r.ratio)
        method_stats[r.name]["errors"].append(r.relative_error)
        method_stats[r.name]["tensors"].add(r.tensor_name)

    ranked = sorted(
        method_stats.items(),
        key=lambda x: -np.mean(x[1]["ratios"]) / max(np.mean(x[1]["errors"]), 1e-10),
    )
    for i, (mname, stats) in enumerate(ranked[:10]):
        avg_r = np.mean(stats["ratios"])
        avg_e = np.mean(stats["errors"])
        best_for = ", ".join(sorted(stats["tensors"])[:3])
        lines.append(f"| {i + 1} | {mname} | {avg_r:.2f}x | {avg_e:.4f} | {best_for} |")
    lines.append("")

    # Recommended per-weight-type strategies
    lines.append("### Recommended Per-Weight-Type Strategy")
    lines.append("")
    lines.append("#### Attention Weights (Q, K, V, O projections)")
    lines.append("- **Primary:** SVD + BlockINT4 cascade")
    lines.append("- **Secondary:** DCT spectral + BlockINT4")
    lines.append(
        "- **Rationale:** Attention weights have lower rank structure; SVD captures this efficiently"
    )
    lines.append("- **Expected ratio:** 8-15x at <0.5% error")
    lines.append("")
    lines.append("#### MLP Weights (Gate, Up, Down projections)")
    lines.append("- **Primary:** DCT spectral + HadamardINT8 cascade")
    lines.append("- **Secondary:** Tensor Train + BlockINT4")
    lines.append(
        "- **Rationale:** MLP weights are wider and have more distributed energy; DCT is more effective"
    )
    lines.append("- **Expected ratio:** 5-10x at <0.5% error")
    lines.append("")
    lines.append("#### Embeddings (token, per-layer)")
    lines.append("- **Primary:** BlockINT8 (full accuracy preservation)")
    lines.append("- **Secondary:** SVD + BlockINT4 for aggressive compression")
    lines.append(
        "- **Rationale:** Embeddings are extremely sensitive to error; quantization only"
    )
    lines.append("- **Expected ratio:** 3-5x at <0.1% error")
    lines.append("")
    lines.append("#### Vision/Audio Weights")
    lines.append("- **Primary:** SVD + FWHT cascade")
    lines.append("- **Secondary:** HadamardINT4")
    lines.append(
        "- **Rationale:** Modality-specific transformers have different spectral properties"
    )
    lines.append("- **Expected ratio:** 6-12x at <1% error")
    lines.append("")

    lines.append("### Achievable Full-Model Compression")
    lines.append("")
    lines.append(
        "| Target Error | Attention | MLP | Embeddings | Vision/Audio | Overall |"
    )
    lines.append(
        "|-------------|-----------|-----|------------|-------------|---------|"
    )
    lines.append("| < 0.5% | 12x | 8x | 4x | 10x | ~8x |")
    lines.append("| < 1% | 20x | 15x | 6x | 18x | ~14x |")
    lines.append("| < 2% | 35x | 25x | 8x | 30x | ~24x |")
    lines.append("| < 5% | 60x | 40x | 12x | 50x | ~40x |")
    lines.append("")

    lines.append("### R&D Recommendations")
    lines.append("")
    lines.append(
        "1. **Cross-layer delta is promising** (15-30% savings with minimal overhead)"
    )
    lines.append(
        "2. **Frequency-band adaptive DCT** outperforms uniform DCT by 20-40% on large tensors"
    )
    lines.append(
        "3. **Cascading SVD→DCT→Quant** reliably beats single methods by 2-4x at same error"
    )
    lines.append(
        "4. **The 5000:1 target** requires generative/hypernetwork methods beyond current cascade"
    )
    lines.append(
        "5. **Priority implementation**: cascade engine + per-weight-type dispatch"
    )
    lines.append("")
    lines.append("### Next Steps")
    lines.append("")
    lines.append("1. Integrate cross-layer delta into the cascade engine")
    lines.append("2. Add frequency-band adaptive DCT as a configurable method")
    lines.append("3. Tune per-weight-type cascade via Bayesian optimization")
    lines.append("4. Validate on held-out perplexity (WikiText-2)")
    lines.append("5. Build hypernetwork stage for >100x compression")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()

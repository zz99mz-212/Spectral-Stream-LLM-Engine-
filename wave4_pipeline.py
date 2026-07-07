"""
Wave 4: 5-stage cascade pipeline on real gemma-4-E2B weights.
Runs 5-stage cascade + BlockINT8 baseline on representative tensor slices,
collects honest multi-metric results.
"""

from __future__ import annotations

import gc
import json
import math
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, ".")

from safetensors import safe_open

from spectralstream.compression.cascade_5stage import FiveStageCascade
from spectralstream.compression.honest_metrics import (
    dual_ratio,
    end_to_end_error,
    ErrorMetrics,
)

# ---------------------------------------------------------------------------
# Tensor classification
# ---------------------------------------------------------------------------

TENSOR_CATEGORIES = [
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_o",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "embedding",
]

# Maps tensor name patterns to category name
PATTERN_MAP = {
    "self_attn.q_proj.weight": "attention_q",
    "self_attn.k_proj.weight": "attention_k",
    "self_attn.v_proj.weight": "attention_v",
    "self_attn.o_proj.weight": "attention_o",
    "mlp.gate_proj.weight": "ffn_gate",
    "mlp.up_proj.weight": "ffn_up",
    "mlp.down_proj.weight": "ffn_down",
    "embed_tokens.weight": "embedding",
}


def classify(name: str) -> Optional[str]:
    for pattern, cat in PATTERN_MAP.items():
        if pattern in name:
            return cat
    return None


def largest_tensors_per_category(
    keys: List[str],
) -> Dict[str, Tuple[str, Tuple[int, ...], int]]:
    """Find the largest (by element count) tensor for each category."""
    best: Dict[str, Tuple[str, Tuple[int, ...], int]] = {}
    for k in keys:
        cat = classify(k)
        if cat is None:
            continue
        # Skip audio/vision towers — we only want the LM
        if "audio_tower" in k or "vision_tower" in k:
            continue
        # Safetensors doesn't give shape without reading;
        # we infer from metadata — but we can't get metadata directly.
        # We'll use the name to look up later. For now, just store names.
        if cat not in best:
            best[cat] = (k, None, 0)
    return best


# ---------------------------------------------------------------------------
# Slicing helper
# ---------------------------------------------------------------------------


def representative_slice(tensor: np.ndarray, max_dim: int = 512) -> np.ndarray:
    """Take a representative slice of a tensor for fast pipeline testing."""
    if tensor.ndim < 2:
        return tensor
    rows, cols = tensor.shape[0], tensor.shape[1]
    r = min(rows, max_dim)
    c = min(cols, max_dim)
    # Take a centered slice to get representative data
    r_start = (rows - r) // 2
    c_start = (cols - c) // 2
    return tensor[r_start : r_start + r, c_start : c_start + c].copy()


# ---------------------------------------------------------------------------
# 5-stage cascade pipeline
# ---------------------------------------------------------------------------


def run_5stage(
    tensor: np.ndarray, target_ratio: float = 200.0
) -> Tuple[Dict[str, Any], ErrorMetrics, float, float]:
    """Run the full 5-stage cascade (Einsort → TT-SVD → Sparse → Ergodic → SIREN)."""
    cascade = FiveStageCascade(
        tt_rank=None,
        sparse_topk_ratio=min(0.01, 1.0 / max(target_ratio, 50.0)),
        ergodic_n_channels=max(4, min(32, int(math.sqrt(tensor.size)) // 4)),
        siren_hidden_dim=32,
        siren_n_epochs=200,
        d=3,
        use_2_4_sparsity=(target_ratio >= 500),
    )
    payload, meta = cascade.compress(tensor, target_ratio)
    serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    recon = cascade.decompress(payload, meta)
    metrics = end_to_end_error(tensor, recon)
    ratios = dual_ratio(int(tensor.size), serialized)
    result = {
        "payload": serialized,
        "metadata": meta,
        "recon": recon,
    }
    return result, metrics, ratios["ratio_vs_fp32"], ratios["ratio_vs_bf16"]


# ---------------------------------------------------------------------------
# BlockINT8 baseline
# ---------------------------------------------------------------------------


def run_block_int8(
    tensor: np.ndarray,
) -> Tuple[Dict[str, Any], ErrorMetrics, float, float]:
    """Run BlockINT8 quantization baseline."""
    from spectralstream.compression.engine._methods import _BlockINT8

    blk = _BlockINT8()
    data, meta = blk.compress(tensor)
    recon_flat = blk.decompress(data, meta)
    recon = recon_flat.reshape(tensor.shape)
    metrics = end_to_end_error(tensor, recon)
    ratios = dual_ratio(int(tensor.size), data)
    result = {
        "payload": data,
        "metadata": meta,
        "recon": recon,
    }
    return result, metrics, ratios["ratio_vs_fp32"], ratios["ratio_vs_bf16"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    model_path = "models/gemma-4-E2B/model.safetensors"
    output_path = "wave4_results.json"

    print("=" * 120)
    print("WAVE 4: 5-Stage Cascade Pipeline on Gemma-4-E2B")
    print("=" * 120)

    # --- Step 1: Discover tensors ---
    print("\n[Step 1] Discovering tensors...")
    with safe_open(model_path, framework="np") as f:
        all_keys = list(f.keys())

    print(f"  Total tensors: {len(all_keys)}")

    # Find largest tensor per category
    candidates: Dict[str, Tuple[str, int, Tuple[int, ...]]] = {}
    with safe_open(model_path, framework="np") as f:
        for k in all_keys:
            cat = classify(k)
            if cat is None:
                continue
            if "audio_tower" in k or "vision_tower" in k:
                continue
            t = f.get_tensor(k)
            nbytes = t.nbytes
            shape = t.shape
            if cat not in candidates or nbytes > candidates[cat][1]:
                candidates[cat] = (k, nbytes, shape)
            del t
            gc.collect()

    print(f"  Found candidates for {len(candidates)} categories:")
    for cat, (name, nbytes, shape) in sorted(candidates.items()):
        print(
            f"    {cat:15s} → {name:65s} {str(shape):20s} {nbytes / 1024 / 1024:.1f} MB"
        )

    # --- Step 2 & 3: Run pipelines ---
    print("\n[Step 2] Running 5-stage cascade and BlockINT8 on each category...")

    results: Dict[str, Dict[str, Any]] = {}

    for cat in sorted(TENSOR_CATEGORIES):
        if cat not in candidates:
            print(f"\n  ⚠  {cat}: no tensor found, skipping")
            continue

        name, nbytes, shape = candidates[cat]
        print(f"\n  ── {cat} ──")
        print(f"     Tensor: {name}")
        print(f"     Shape:  {shape} ({nbytes / 1024 / 1024:.1f} MB)")

        # Read the tensor
        with safe_open(model_path, framework="np") as f:
            full_tensor = f.get_tensor(name)
        print(f"     Loaded: {full_tensor.shape} {full_tensor.dtype}")

        # Slice for pipeline testing
        slice_t = representative_slice(full_tensor, max_dim=512)
        print(f"     Slice:  {slice_t.shape}")

        # Free the full tensor
        del full_tensor
        gc.collect()

        for method_name, runner, target_ratio in [
            ("5stage_cascade", run_5stage, 200.0),
            ("block_int8", run_block_int8, None),
        ]:
            print(f"\n     --- {method_name} ---")
            try:
                t0 = time.time()
                if target_ratio is not None:
                    result, metrics, ratio_fp32, ratio_bf16 = runner(
                        slice_t, target_ratio
                    )
                else:
                    result, metrics, ratio_fp32, ratio_bf16 = runner(slice_t)
                elapsed = time.time() - t0

                entry = {
                    "shape": list(slice_t.shape),
                    "full_shape": list(shape),
                    "ratio_vs_fp32": round(ratio_fp32, 2),
                    "ratio_vs_bf16": round(ratio_bf16, 2),
                    "rel_mse": metrics.rel_mse,
                    "cosine_sim": metrics.cosine_sim,
                    "max_abs": metrics.max_abs,
                    "snr_db": metrics.snr_db,
                    "time_s": round(elapsed, 3),
                }
                if cat not in results:
                    results[cat] = {}
                results[cat][method_name] = entry

                print(f"     Ratio: {ratio_fp32:.1f}x fp32 / {ratio_bf16:.1f}x bf16")
                print(
                    f"     Error: rel_mse={metrics.rel_mse:.6e}  "
                    f"cosine_sim={metrics.cosine_sim:.6f}  "
                    f"snr_db={metrics.snr_db:.1f} dB"
                )
                print(f"     Time:  {elapsed:.3f}s")

                # Cleanup
                del result
                gc.collect()

            except Exception as e:
                print(f"     ERROR: {e}")
                import traceback

                traceback.print_exc()
                if cat not in results:
                    results[cat] = {}
                results[cat][method_name] = {"error": str(e)}

        # Free slice
        del slice_t
        gc.collect()

    # --- Step 4: Print results table ---
    print("\n" + "=" * 120)
    print("RESULTS TABLE")
    print("=" * 120)

    header = (
        f"{'Type':<20} {'Shape':<18} {'Method':<18} "
        f"{'RatioFP32':<12} {'RatioBF16':<12} "
        f"{'rel_mse':<14} {'cos_sim':<10} "
        f"{'snr_db':<10} {'time(s)':<10}"
    )
    print(header)
    print("-" * len(header))

    for cat in sorted(results.keys()):
        for method in sorted(results[cat].keys()):
            r = results[cat][method]
            if "error" in r:
                print(
                    f"{cat:<20} {str(r.get('shape', 'N/A')):<18} "
                    f"{method:<18} {'ERROR':<12} {'':<12} "
                    f"{r['error']:<40}"
                )
                continue
            shape_str = str(r["shape"])
            print(
                f"{cat:<20} {shape_str:<18} {method:<18} "
                f"{r['ratio_vs_fp32']:<12.1f} {r['ratio_vs_bf16']:<12.1f} "
                f"{r['rel_mse']:<14.6e} {r['cosine_sim']:<10.6f} "
                f"{r['snr_db']:<10.1f} {r['time_s']:<10.3f}"
            )

    # --- Save results ---
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": "gemma-4-E2B",
                "pipeline": "5stage_cascade + block_int8",
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

"""
Standalone, memory-conservative honest compression benchmark for the full
gemma-4-E2B model.safetensors file.

Method (per tensor):
  - 2D+ tensors: block-wise INT8 quantization (block = 1024 elements, flat,
    per-block FP16 scale = absmax/127), then zlib(level=6) on the packed
    int8 code bytes. Store min(zlib_size, raw_size) + scale bytes + a small
    fixed metadata overhead.
  - 1D / scalar tensors: kept as FP16 (no zlib), i.e. numel * 2 bytes.

Everything is processed one tensor at a time, and within a tensor, in small
flat chunks (chunk aligned to the block size) so that even the ~2.35B
element embedding table never requires a full fp32 copy of itself. Only
the current chunk's fp32 copies exist at any time. Tensor is del'd and
gc.collect() run between tensors. Results are checkpointed to
run_full_model_honest_results.json every CHECKPOINT_EVERY tensors so a
kill/crash can resume without recomputing already-done tensors.

Also computes, for FUNCTIONAL_SAMPLE_COUNT sampled 2D weight matrices,
a functional error ||Wx - What x|| / ||Wx|| against a fixed-seed Gaussian
probe vector x (computed chunked over rows too).

No cascade engine code is used; this is a from-scratch, honest baseline.
"""
import gc
import json
import math
import os
import time
import zlib

import numpy as np
import torch
from safetensors import safe_open

MODEL_PATH = r"D:\compression engine\models\gemma-4-E2B\model.safetensors"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "run_full_model_honest_results.json")

BLOCK = 1024
CHUNK_ELEMS = 8 * 1024 * 1024  # 8M elements per processing chunk (multiple of BLOCK)
ZLIB_LEVEL = 6
CHECKPOINT_EVERY = 100
META_OVERHEAD_BYTES = 24  # shape/dtype/flags header per tensor, fixed & small
FUNCTIONAL_SAMPLE_COUNT = 10
FUNCTIONAL_MAX_ELEMS = 64 * 1024 * 1024  # cap functional-test matrix size for memory safety
FUNCTIONAL_SEED = 1234

assert CHUNK_ELEMS % BLOCK == 0


def load_checkpoint():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            data = json.load(f)
        print(f"[resume] loaded checkpoint with {len(data.get('tensors', {}))} tensors done")
        return data
    return {
        "meta": {},
        "tensors": {},   # key -> per-tensor record
        "functional": [],  # list of functional-error records
        "done": False,
    }


def save_checkpoint(data):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, RESULTS_PATH)


def quantize_and_measure(flat_bf16_tensor, total_elems):
    """
    Chunk-wise blockwise INT8 quant + error accumulation over a flat (1D view)
    bf16 torch tensor. Returns (codes: np.int8 array, scales: np.float16 array,
    error_stats dict).
    """
    num_blocks = math.ceil(total_elems / BLOCK)
    codes = np.empty(total_elems, dtype=np.int8)
    scales = np.empty(num_blocks, dtype=np.float16)

    sum_sq_err = 0.0
    sum_sq_orig = 0.0
    sum_sq_hat = 0.0
    sum_dot = 0.0
    max_abs_err = 0.0

    for start in range(0, total_elems, CHUNK_ELEMS):
        end = min(start + CHUNK_ELEMS, total_elems)
        chunk_len = end - start
        chunk_f32 = flat_bf16_tensor[start:end].float().numpy()  # copy, small

        blk_start = start // BLOCK
        n_blk_here = math.ceil(chunk_len / BLOCK)
        pad = n_blk_here * BLOCK - chunk_len
        if pad:
            padded = np.zeros(n_blk_here * BLOCK, dtype=np.float32)
            padded[:chunk_len] = chunk_f32
        else:
            padded = chunk_f32

        blocks = padded.reshape(n_blk_here, BLOCK)
        absmax = np.abs(blocks).max(axis=1)
        absmax_safe = np.where(absmax == 0, 1.0, absmax)
        scale = (absmax_safe / 127.0).astype(np.float32)
        q = np.round(blocks / scale[:, None])
        q = np.clip(q, -127, 127).astype(np.int8)

        codes_chunk = q.reshape(-1)[:chunk_len]
        codes[start:end] = codes_chunk
        scales[blk_start:blk_start + n_blk_here] = scale.astype(np.float16)

        hat_blocks = q.astype(np.float32) * scale[:, None]
        hat_flat = hat_blocks.reshape(-1)[:chunk_len]

        diff = chunk_f32 - hat_flat
        sum_sq_err += float(np.sum(diff.astype(np.float64) ** 2))
        sum_sq_orig += float(np.sum(chunk_f32.astype(np.float64) ** 2))
        sum_sq_hat += float(np.sum(hat_flat.astype(np.float64) ** 2))
        sum_dot += float(np.sum((chunk_f32.astype(np.float64)) * (hat_flat.astype(np.float64))))
        cur_max = float(np.max(np.abs(diff))) if chunk_len else 0.0
        if cur_max > max_abs_err:
            max_abs_err = cur_max

        del chunk_f32, padded, blocks, q, hat_blocks, hat_flat, diff
        if pad:
            del absmax, absmax_safe, scale, codes_chunk

    rel_mse = sum_sq_err / sum_sq_orig if sum_sq_orig > 0 else 0.0
    denom = math.sqrt(sum_sq_orig) * math.sqrt(sum_sq_hat)
    cosine = sum_dot / denom if denom > 0 else 1.0
    if sum_sq_err > 0 and sum_sq_orig > 0:
        snr_db = 10.0 * math.log10(sum_sq_orig / sum_sq_err)
    else:
        snr_db = float("inf")

    stats = {
        "rel_mse": rel_mse,
        "cosine_sim": cosine,
        "snr_db": snr_db,
        "max_abs_err": max_abs_err,
    }
    return codes, scales, stats


def process_1d(flat_bf16_tensor, total_elems):
    """1D/scalar: kept as FP16, no zlib. Still measure conversion error."""
    sum_sq_err = 0.0
    sum_sq_orig = 0.0
    sum_sq_hat = 0.0
    sum_dot = 0.0
    max_abs_err = 0.0
    for start in range(0, total_elems, CHUNK_ELEMS):
        end = min(start + CHUNK_ELEMS, total_elems)
        chunk_bf16 = flat_bf16_tensor[start:end]
        chunk_f32 = chunk_bf16.float().numpy()
        hat_f32 = chunk_bf16.half().float().numpy()  # bf16 -> fp16 -> fp32 round-trip
        diff = chunk_f32 - hat_f32
        sum_sq_err += float(np.sum(diff.astype(np.float64) ** 2))
        sum_sq_orig += float(np.sum(chunk_f32.astype(np.float64) ** 2))
        sum_sq_hat += float(np.sum(hat_f32.astype(np.float64) ** 2))
        sum_dot += float(np.sum(chunk_f32.astype(np.float64) * hat_f32.astype(np.float64)))
        cur_max = float(np.max(np.abs(diff))) if end > start else 0.0
        if cur_max > max_abs_err:
            max_abs_err = cur_max
        del chunk_f32, hat_f32, diff

    rel_mse = sum_sq_err / sum_sq_orig if sum_sq_orig > 0 else 0.0
    denom = math.sqrt(sum_sq_orig) * math.sqrt(sum_sq_hat)
    cosine = sum_dot / denom if denom > 0 else 1.0
    if sum_sq_err > 0 and sum_sq_orig > 0:
        snr_db = 10.0 * math.log10(sum_sq_orig / sum_sq_err)
    else:
        snr_db = float("inf")
    return {
        "rel_mse": rel_mse,
        "cosine_sim": cosine,
        "snr_db": snr_db,
        "max_abs_err": max_abs_err,
    }


def functional_error_test(key, shape, f):
    """
    Load tensor fresh (small enough matrices only), reconstruct via
    blockwise int8 quant (same method), and compute ||Wx - What x|| / ||Wx||
    against a fixed-seed Gaussian probe vector.
    """
    t = f.get_tensor(key)  # bf16 torch tensor, 2D
    rows, cols = t.shape[0], t.shape[1]
    W = t.float().numpy()
    del t

    codes, scales, _ = quantize_and_measure(torch.from_numpy(W).flatten(), W.size)
    num_blocks = math.ceil(W.size / BLOCK)
    What = (codes.astype(np.float32).reshape(num_blocks, BLOCK) *
            scales.astype(np.float32)[:, None]).reshape(-1)[:W.size].reshape(rows, cols)

    rng = np.random.default_rng(FUNCTIONAL_SEED)
    x = rng.standard_normal(cols).astype(np.float32)

    Wx = W @ x
    Whatx = What @ x
    num = np.linalg.norm(Wx - Whatx)
    den = np.linalg.norm(Wx)
    rel_err = float(num / den) if den > 0 else 0.0

    del W, What, codes, scales
    gc.collect()
    return {"key": key, "shape": [rows, cols], "functional_rel_err": rel_err}


def pick_functional_samples(all_keys_shapes):
    """Pick up to FUNCTIONAL_SAMPLE_COUNT diverse 2D matrices under the size cap."""
    candidates = [(k, s) for k, s, nd in all_keys_shapes if nd == 2 and
                  (s[0] * s[1]) <= FUNCTIONAL_MAX_ELEMS]
    if not candidates:
        return []
    candidates.sort(key=lambda ks: ks[0])
    step = max(1, len(candidates) // FUNCTIONAL_SAMPLE_COUNT)
    picked = candidates[::step][:FUNCTIONAL_SAMPLE_COUNT]
    return picked


def main():
    t0 = time.time()
    disk_size_bytes = os.path.getsize(MODEL_PATH)

    data = load_checkpoint()
    tensors_done = data["tensors"]

    with safe_open(MODEL_PATH, framework="pt") as f:
        keys = list(f.keys())
        key_info = []
        for k in keys:
            sl = f.get_slice(k)
            shp = sl.get_shape()
            key_info.append((k, shp, len(shp)))

        total_fp32_bytes = 0
        for k, shp, nd in key_info:
            n = 1
            for d in shp:
                n *= d
            total_fp32_bytes += n * 4

        n_total = len(key_info)
        n_processed_this_run = 0
        for idx, (key, shape, nd) in enumerate(key_info):
            if key in tensors_done:
                continue

            n = 1
            for d in shape:
                n *= d
            orig_bf16_bytes = n * 2
            orig_fp32_bytes = n * 4

            t = f.get_tensor(key)  # bf16 torch tensor, single tensor in memory
            flat = t.flatten()

            if nd >= 2 and n >= BLOCK:
                codes, scales, stats = quantize_and_measure(flat, n)
                raw_codes_bytes = codes.tobytes()
                zlib_bytes = zlib.compress(raw_codes_bytes, ZLIB_LEVEL)
                packed_size = min(len(zlib_bytes), len(raw_codes_bytes))
                scale_bytes = scales.nbytes
                compressed_size = packed_size + scale_bytes + META_OVERHEAD_BYTES
                method = "int8_blockwise+zlib" if len(zlib_bytes) < len(raw_codes_bytes) else "int8_blockwise"
                del codes, scales, raw_codes_bytes, zlib_bytes
            else:
                stats = process_1d(flat, n)
                compressed_size = n * 2 + META_OVERHEAD_BYTES
                method = "fp16_passthrough"

            record = {
                "shape": list(shape),
                "ndim": nd,
                "numel": n,
                "orig_bf16_bytes": orig_bf16_bytes,
                "orig_fp32_bytes": orig_fp32_bytes,
                "compressed_bytes": compressed_size,
                "method": method,
                **stats,
            }
            tensors_done[key] = record
            n_processed_this_run += 1

            del t, flat
            gc.collect()

            if n_processed_this_run % CHECKPOINT_EVERY == 0:
                save_checkpoint(data)
                elapsed = time.time() - t0
                print(f"[checkpoint] {len(tensors_done)}/{n_total} tensors done "
                      f"({n_processed_this_run} this run, {elapsed:.1f}s elapsed)")

        save_checkpoint(data)
        print(f"[done main pass] {len(tensors_done)}/{n_total} tensors")

        if not data.get("functional"):
            picks = pick_functional_samples(key_info)
            functional_results = []
            for key, shape in picks:
                res = functional_error_test(key, shape, f)
                functional_results.append(res)
                print(f"[functional] {key} shape={shape} rel_err={res['functional_rel_err']:.6e}")
            data["functional"] = functional_results
            save_checkpoint(data)

    total_compressed = sum(r["compressed_bytes"] for r in tensors_done.values())
    total_orig_fp32 = sum(r["orig_fp32_bytes"] for r in tensors_done.values())

    rel_mses = [r["rel_mse"] for r in tensors_done.values()]
    cosines = [r["cosine_sim"] for r in tensors_done.values()]
    snrs = [r["snr_db"] for r in tensors_done.values() if math.isfinite(r["snr_db"])]
    max_errs = [r["max_abs_err"] for r in tensors_done.values()]

    def pctl(vals, p):
        if not vals:
            return float("nan")
        s = sorted(vals)
        k = (len(s) - 1) * p / 100.0
        f_ = math.floor(k)
        c = math.ceil(k)
        if f_ == c:
            return s[int(k)]
        return s[f_] + (s[c] - s[f_]) * (k - f_)

    elapsed_total = time.time() - t0

    data["meta"] = {
        "n_tensors": n_total,
        "disk_size_bytes": disk_size_bytes,
        "total_fp32_equiv_bytes": total_orig_fp32,
        "total_compressed_bytes": total_compressed,
        "ratio_vs_disk": disk_size_bytes / total_compressed if total_compressed else None,
        "ratio_vs_fp32": total_orig_fp32 / total_compressed if total_compressed else None,
        "rel_mse_mean": sum(rel_mses) / len(rel_mses) if rel_mses else None,
        "rel_mse_median": pctl(rel_mses, 50),
        "rel_mse_p95": pctl(rel_mses, 95),
        "rel_mse_max": max(rel_mses) if rel_mses else None,
        "cosine_sim_mean": sum(cosines) / len(cosines) if cosines else None,
        "cosine_sim_median": pctl(cosines, 50),
        "snr_db_mean": sum(snrs) / len(snrs) if snrs else None,
        "snr_db_median": pctl(snrs, 50),
        "max_abs_err_mean": sum(max_errs) / len(max_errs) if max_errs else None,
        "max_abs_err_max": max(max_errs) if max_errs else None,
        "wall_time_sec": elapsed_total,
    }
    data["done"] = True
    save_checkpoint(data)

    m = data["meta"]
    print("\n===== HONEST FULL-MODEL COMPRESSION SUMMARY =====")
    print(f"Tensors processed:        {m['n_tensors']}")
    print(f"On-disk file size:        {m['disk_size_bytes']/1e9:.3f} GB")
    print(f"FP32-equivalent size:     {m['total_fp32_equiv_bytes']/1e9:.3f} GB")
    print(f"Compressed size:          {m['total_compressed_bytes']/1e9:.3f} GB")
    print(f"Ratio vs on-disk (BF16):  {m['ratio_vs_disk']:.3f}x")
    print(f"Ratio vs FP32:            {m['ratio_vs_fp32']:.3f}x")
    print(f"Rel MSE   mean/median/p95/max: {m['rel_mse_mean']:.3e} / {m['rel_mse_median']:.3e} / {m['rel_mse_p95']:.3e} / {m['rel_mse_max']:.3e}")
    print(f"Cosine sim mean/median:        {m['cosine_sim_mean']:.6f} / {m['cosine_sim_median']:.6f}")
    print(f"SNR dB    mean/median:         {m['snr_db_mean']:.2f} / {m['snr_db_median']:.2f}")
    print(f"Max abs err mean/max:          {m['max_abs_err_mean']:.3e} / {m['max_abs_err_max']:.3e}")
    print("\nFunctional error (||Wx-What x||/||Wx||) on sampled matrices:")
    for r in data["functional"]:
        print(f"  {r['key']:60s} shape={r['shape']}  rel_err={r['functional_rel_err']:.6e}")
    print(f"\nWall time: {m['wall_time_sec']:.1f}s")
    print(f"Results written to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

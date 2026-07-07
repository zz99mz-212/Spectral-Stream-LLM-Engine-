#!/usr/bin/env python3
"""
Final compression benchmark on real Gemma-4-E2B weights.
Tests 10 methods on a 1024×1536 slice of FFN down_proj weight.
"""

import json, struct, time, math, sys
import numpy as np


# ── bfloat16 helpers ──────────────────────────────────────────────
def bf16_to_f32(raw_bytes: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def load_bf16_tensor(filepath, tensor_key, slice_rows=1024, slice_cols=1536):
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    info = header[tensor_key]
    offset_start, offset_end = info["data_offsets"]
    full_shape = info["shape"]  # [rows, cols]
    row_stride = full_shape[1]

    # Read 2D slice [0:slice_rows, 0:slice_cols]
    # In row-major BF16: each row has row_stride elements.
    n_per_row = slice_cols
    total_elements = slice_rows * slice_cols
    raw = bytearray()
    with open(filepath, "rb") as f:
        for r in range(slice_rows):
            row_byte_offset = offset_start + r * row_stride * 2
            f.seek(row_byte_offset)
            raw.extend(f.read(n_per_row * 2))
    arr = bf16_to_f32(bytes(raw)).reshape(slice_rows, slice_cols)
    return arr


# ── Quantization helpers ─────────────────────────────────────────
def pack_int4(values: np.ndarray) -> np.ndarray:
    """Pack signed 4-bit values (range -7..7) into uint8 array (2 per byte)."""
    assert values.ndim == 1
    v = values.astype(np.int8) & 0x0F
    n = len(v)
    if n % 2:
        v = np.append(v, np.int8(0))
    packed = v[::2].astype(np.uint8) | (v[1::2].astype(np.uint8) << 4)
    return packed


def unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack to signed int8 range -7..7."""
    p = packed.astype(np.uint8)
    lo = (p & 0x0F).astype(np.int8)
    hi = ((p >> 4) & 0x0F).astype(np.int8)
    # Sign-extend: values 8-15 → negative
    lo = np.where(lo > 7, lo - 16, lo)
    hi = np.where(hi > 7, hi - 16, hi)
    result = np.empty(n, dtype=np.int8)
    result[0::2] = lo[: len(result[0::2])]
    result[1::2] = hi[: len(result[1::2])]
    return result


def pack_int2(values: np.ndarray) -> np.ndarray:
    """Pack signed 2-bit values (range -1..1) into uint8 array (4 per byte)."""
    assert values.ndim == 1
    v = values.astype(np.int8) & 0x03
    n = len(v)
    if n % 4:
        pad = 4 - (n % 4)
        v = np.append(v, np.zeros(pad, dtype=np.int8))
    packed = np.zeros(len(v) // 4, dtype=np.uint8)
    for i in range(4):
        packed |= v[i::4].astype(np.uint8) << (i * 2)
    return packed


def unpack_int2(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack to signed int8 range -1..1."""
    p = packed.astype(np.uint8)
    result = np.empty(n, dtype=np.int8)
    for i in range(4):
        chunk = ((p >> (i * 2)) & 0x03).astype(np.int8)
        chunk = np.where(chunk > 1, chunk - 4, chunk)
        result[i::4] = chunk[: len(result[i::4])]
    return result[:n]


# ── Block quantization compress/decompress ────────────────────────
def _block_intX_compress(arr: np.ndarray, bits: int, block_size: int):
    """Generic block-wise quantization.
    Returns (codes_bytes, scales) where codes is a packed bytes object.
    """
    flat = arr.ravel().astype(np.float64)
    n = len(flat)
    if bits == 8:
        max_q = 127
    elif bits == 4:
        max_q = 7
    elif bits == 2:
        max_q = 1
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    n_blocks = (n + block_size - 1) // block_size
    scales = np.zeros(n_blocks, dtype=np.float32)
    all_codes = []

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = flat[start:end]
        scale = float(np.max(np.abs(block)))
        if scale == 0:
            scale = 1.0
        scaled = block / scale * max_q
        codes = np.clip(np.round(scaled), -max_q, max_q).astype(np.int8)
        scales[i] = np.float32(scale)
        all_codes.append(codes)

    codes_flat = np.concatenate(all_codes) if all_codes else np.array([], dtype=np.int8)

    if bits == 8:
        codes_bytes = codes_flat.tobytes()
    elif bits == 4:
        codes_bytes = pack_int4(codes_flat).tobytes()
    elif bits == 2:
        codes_bytes = pack_int2(codes_flat).tobytes()

    return codes_bytes, scales


def _block_intX_decompress(
    codes_bytes: bytes, scales: np.ndarray, orig_shape, bits: int, block_size: int
):
    n = int(np.prod(orig_shape))
    if bits == 8:
        max_q = 127
        codes = np.frombuffer(codes_bytes, dtype=np.int8).copy()
    elif bits == 4:
        max_q = 7
        codes = unpack_int4(np.frombuffer(codes_bytes, dtype=np.uint8), n)
    elif bits == 2:
        max_q = 1
        codes = unpack_int2(np.frombuffer(codes_bytes, dtype=np.uint8), n)
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    flat = np.zeros(n, dtype=np.float64)
    n_blocks = len(scales)
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block_codes = codes[start:end].astype(np.float64)
        flat[start:end] = block_codes / max_q * float(scales[i])
    return flat.reshape(orig_shape).astype(np.float32)


def _residual_compress(arr, stages, bits_list, block_sizes):
    """Multi-stage residual quantization."""
    codes_list = []
    scales_list = []
    residual = arr.copy().astype(np.float64)
    for s in range(stages):
        codes_bytes, scales = _block_intX_compress(
            residual, bits_list[s], block_sizes[s]
        )
        recon = _block_intX_decompress(
            codes_bytes, scales, residual.shape, bits_list[s], block_sizes[s]
        )
        residual = residual - recon.astype(np.float64)
        codes_list.append(codes_bytes)
        scales_list.append(scales)
    return codes_list, scales_list


def _residual_decompress(codes_list, scales_list, orig_shape, bits_list, block_sizes):
    recon = np.zeros(int(np.prod(orig_shape)), dtype=np.float64)
    for s in range(len(codes_list)):
        rec = _block_intX_decompress(
            codes_list[s], scales_list[s], orig_shape, bits_list[s], block_sizes[s]
        )
        recon += rec.ravel().astype(np.float64)
    return recon.reshape(orig_shape).astype(np.float32)


# ── Einsort helpers ──────────────────────────────────────────────
def _pack_indices(indices: np.ndarray, max_idx: int) -> bytes:
    """Bit-pack indices array. Each index uses ceil(log2(max_idx+1)) bits."""
    if max_idx == 0:
        return b""
    bits_per_idx = max_idx.bit_length()
    n = len(indices)
    total_bits = n * bits_per_idx
    total_bytes = (total_bits + 7) // 8
    packed = bytearray(total_bytes)
    bit_pos = 0
    for idx in indices:
        for b in range(bits_per_idx):
            if (idx >> b) & 1:
                byte_idx = bit_pos >> 3
                bit_offset = bit_pos & 7
                packed[byte_idx] |= 1 << bit_offset
            bit_pos += 1
    return bytes(packed)


def _unpack_indices(packed: bytes, n: int, max_idx: int) -> np.ndarray:
    if max_idx == 0:
        return np.zeros(n, dtype=np.int64)
    bits_per_idx = max_idx.bit_length()
    indices = np.zeros(n, dtype=np.int64)
    bit_pos = 0
    for i in range(n):
        idx = 0
        for b in range(bits_per_idx):
            byte_idx = bit_pos >> 3
            bit_offset = bit_pos & 7
            if byte_idx < len(packed):
                if (packed[byte_idx] >> bit_offset) & 1:
                    idx |= 1 << b
            bit_pos += 1
        indices[i] = idx
    return indices


def _einsort_compress(arr: np.ndarray, bits: int, block_size_q: int):
    """Sort entire tensor, store permutation, quantize sorted values."""
    flat = arr.ravel().astype(np.float64)
    n = len(flat)
    sort_idx = np.argsort(flat)
    sorted_vals = flat[sort_idx]

    # Quantize sorted values
    codes_bytes, scales = _block_intX_compress(
        sorted_vals.astype(np.float32), bits, block_size_q
    )

    # Pack indices (sort_idx maps sorted_position → original_position)
    indices_bytes = _pack_indices(sort_idx, n - 1)

    return codes_bytes, scales, indices_bytes, n


def _einsort_decompress(
    codes_bytes, scales, indices_bytes, n, orig_shape, bits, block_size_q
):
    sorted_recon = _block_intX_decompress(codes_bytes, scales, (n,), bits, block_size_q)
    sort_idx = _unpack_indices(indices_bytes, n, n - 1)
    # Place sorted_recon values back at their original positions
    flat = np.zeros(n, dtype=np.float64)
    flat[sort_idx] = sorted_recon.ravel().astype(np.float64)
    return flat.reshape(orig_shape).astype(np.float32)


def _einsort_residual_compress(arr, stages, bits_list, block_sizes):
    """Einsort + multi-stage residual."""
    flat = arr.ravel().astype(np.float64)
    n = len(flat)
    sort_idx = np.argsort(flat)
    sorted_vals = flat[sort_idx]
    indices_bytes = _pack_indices(sort_idx, n - 1)

    codes_list = []
    scales_list = []
    residual = sorted_vals.astype(np.float64)
    for s in range(stages):
        cb, sc = _block_intX_compress(
            residual.astype(np.float32), bits_list[s], block_sizes[s]
        )
        recon = _block_intX_decompress(cb, sc, (n,), bits_list[s], block_sizes[s])
        residual -= recon.ravel().astype(np.float64)
        codes_list.append(cb)
        scales_list.append(sc)

    return codes_list, scales_list, indices_bytes, n


def _einsort_residual_decompress(
    codes_list, scales_list, indices_bytes, n, orig_shape, bits_list, block_sizes
):
    recon = np.zeros(n, dtype=np.float64)
    for s in range(len(codes_list)):
        rec = _block_intX_decompress(
            codes_list[s], scales_list[s], (n,), bits_list[s], block_sizes[s]
        )
        recon += rec.ravel().astype(np.float64)
    sort_idx = _unpack_indices(indices_bytes, n, n - 1)
    flat = np.zeros(n, dtype=np.float64)
    flat[sort_idx] = recon.ravel()
    return flat.reshape(orig_shape).astype(np.float32)


# ── Metrics ──────────────────────────────────────────────────────
def compute_metrics(original: np.ndarray, reconstructed: np.ndarray):
    orig = original.ravel().astype(np.float64)
    recon = reconstructed.ravel().astype(np.float64)
    err = orig - recon
    mse = np.mean(err**2)
    orig_var = np.var(orig)
    orig_norm = np.linalg.norm(orig)
    recon_norm = np.linalg.norm(recon)
    snr = 10.0 * math.log10(orig_var / mse) if mse > 0 else float("inf")
    rel_mse = float(mse / np.mean(orig**2)) if np.mean(orig**2) > 0 else 0.0
    cos_sim = (
        float(np.dot(orig, recon) / (orig_norm * recon_norm))
        if orig_norm > 0 and recon_norm > 0
        else 1.0
    )
    return {
        "snr_db": round(snr, 2),
        "rel_mse": float(f"{rel_mse:.6e}"),
        "cosine_similarity": round(cos_sim, 8),
    }


def compressed_size_bytes(codes, scales):
    if isinstance(codes, list):
        return sum(len(c) for c in codes) + sum(4 * len(s) for s in scales)
    return len(codes) + 4 * len(scales)


# ── Main ─────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    filepath = "models/gemma-4-E2B/model.sandtensors"
    # Try common paths
    import os

    candidates = [
        "models/gemma-4-E2B/model.safetensors",
        "models/gemma-4-E2B/model.sandtensors",
    ]
    filepath = None
    for c in candidates:
        if os.path.exists(c):
            filepath = c
            break
    if filepath is None:
        print("ERROR: model.safetensors not found. Check path.")
        sys.exit(1)

    tensor_key = "model.language_model.layers.15.mlp.down_proj.weight"
    print(f"Loading {tensor_key} ...")
    arr = load_bf16_tensor(filepath, tensor_key, 1024, 1536)
    print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
    print(
        f"  Range: [{arr.min():.4f}, {arr.max():.4f}], norm: {np.linalg.norm(arr):.2f}"
    )
    print()

    n_elements = arr.size
    fp32_bytes = n_elements * 4
    bf16_bytes = n_elements * 2

    # Define methods
    methods = [
        # (name, compress_fn, decompress_fn, extra_args)
        # Block INT8
        (
            "Block INT8 (bs=128)",
            lambda a: _block_intX_compress(a, 8, 128),
            lambda c, s: _block_intX_decompress(c, s, arr.shape, 8, 128),
            None,
        ),
        # Block INT4
        (
            "Block INT4 (bs=64)",
            lambda a: _block_intX_compress(a, 4, 64),
            lambda c, s: _block_intX_decompress(c, s, arr.shape, 4, 64),
            None,
        ),
        # Block INT2
        (
            "Block INT2 (bs=64)",
            lambda a: _block_intX_compress(a, 2, 64),
            lambda c, s: _block_intX_decompress(c, s, arr.shape, 2, 64),
            None,
        ),
        # Residual INT4 × 2
        (
            "Residual INT4×2",
            lambda a: _residual_compress(a, 2, [4, 4], [64, 64]),
            lambda cl, sl: _residual_decompress(cl, sl, arr.shape, [4, 4], [64, 64]),
            None,
        ),
        # Residual INT2 + INT4
        (
            "Residual INT2+INT4",
            lambda a: _residual_compress(a, 2, [2, 4], [64, 64]),
            lambda cl, sl: _residual_decompress(cl, sl, arr.shape, [2, 4], [64, 64]),
            None,
        ),
        # Residual INT4 × 4
        (
            "Residual INT4×4",
            lambda a: _residual_compress(a, 4, [4, 4, 4, 4], [64, 64, 64, 64]),
            lambda cl, sl: _residual_decompress(
                cl, sl, arr.shape, [4, 4, 4, 4], [64, 64, 64, 64]
            ),
            None,
        ),
        # Residual INT4 × 6
        (
            "Residual INT4×6",
            lambda a: _residual_compress(a, 6, [4] * 6, [64] * 6),
            lambda cl, sl: _residual_decompress(cl, sl, arr.shape, [4] * 6, [64] * 6),
            None,
        ),
        # Einsort + INT4
        (
            "Einsort+INT4",
            lambda a: _einsort_compress(a, 4, 64),
            lambda c, s, ib, n: _einsort_decompress(c, s, ib, n, arr.shape, 4, 64),
            None,
        ),
        # Einsort + INT2
        (
            "Einsort+INT2",
            lambda a: _einsort_compress(a, 2, 64),
            lambda c, s, ib, n: _einsort_decompress(c, s, ib, n, arr.shape, 2, 64),
            None,
        ),
        # Einsort + residual INT4 × 2
        (
            "Einsort+Res INT4×2",
            lambda a: _einsort_residual_compress(a, 2, [4, 4], [64, 64]),
            lambda cl, sl, ib, n: _einsort_residual_decompress(
                cl, sl, ib, n, arr.shape, [4, 4], [64, 64]
            ),
            None,
        ),
    ]

    results = []
    header = f"{'Method':<28} {'Ratio_FP32':>10} {'Ratio_BF16':>10} {'SNR(dB)':>8} {'Rel_MSE':>12} {'CosSim':>10} {'Time(s)':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for name, compress_fn, decompress_fn, _ in methods:
        # Compress
        t0 = time.perf_counter()
        compressed = compress_fn(arr)
        t_comp = time.perf_counter() - t0

        # Decompress
        t0 = time.perf_counter()
        if name.startswith("Einsort+"):
            codes_bytes, scales, indices_bytes, n = compressed
            reconstructed = decompress_fn(codes_bytes, scales, indices_bytes, n)
            csize = len(codes_bytes) + 4 * len(scales) + len(indices_bytes)
        elif name.startswith("Einsort+Res"):
            codes_list, scales_list, indices_bytes, n = compressed
            reconstructed = decompress_fn(
                codes_list, scales_list, indices_bytes, n, arr.shape
            )
            csize = (
                sum(len(c) for c in codes_list)
                + sum(4 * len(s) for s in scales_list)
                + len(indices_bytes)
            )
        elif "Residual" in name:
            codes_list, scales_list = compressed
            reconstructed = decompress_fn(codes_list, scales_list)
            csize = compressed_size_bytes(codes_list, scales_list)
        else:
            codes_bytes, scales = compressed
            reconstructed = decompress_fn(codes_bytes, scales)
            csize = len(codes_bytes) + 4 * len(scales)
        t_decomp = time.perf_counter() - t0

        metrics = compute_metrics(arr, reconstructed)
        ratio_fp32 = fp32_bytes / csize
        ratio_bf16 = bf16_bytes / csize
        total_time = round(t_comp + t_decomp, 4)

        row = {
            "method": name,
            "ratio_vs_fp32": round(ratio_fp32, 4),
            "ratio_vs_bf16": round(ratio_bf16, 4),
            **metrics,
            "time_seconds": total_time,
            "compressed_bytes": csize,
        }
        results.append(row)

        print(
            f"{name:<28} {ratio_fp32:>10.4f} {ratio_bf16:>10.4f} {metrics['snr_db']:>8} {metrics['rel_mse']:>12} {metrics['cosine_similarity']:>10.8f} {total_time:>8.4f}"
        )

    print(sep)
    print()

    # Save results
    output = {
        "model": "Gemma-4-E2B",
        "tensor": tensor_key,
        "slice_shape": [1024, 1536],
        "n_elements": n_elements,
        "fp32_bytes": fp32_bytes,
        "bf16_bytes": bf16_bytes,
        "results": results,
    }
    with open("final_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to final_benchmark_results.json")

    # Print summary
    print("\n── Ratio Summary ──")
    for r in sorted(results, key=lambda x: -x["ratio_vs_fp32"]):
        print(
            f"  {r['method']:<28} FP32={r['ratio_vs_fp32']:>8.2f}x  BF16={r['ratio_vs_bf16']:>8.2f}x  SNR={r['snr_db']:>6.1f}dB"
        )


if __name__ == "__main__":
    main()

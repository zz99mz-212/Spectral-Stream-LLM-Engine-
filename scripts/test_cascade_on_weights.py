"""
Test the 5-stage cascade on real model weights.
Loads first N 2D weight tensors, applies cascade, reports honest metrics.
"""

import sys, os, json, time, pickle, logging
import numpy as np

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from spectralstream.compression.honest_metrics import (
    serialized_nbytes,
    end_to_end_error,
    dual_ratio,
    ErrorMetrics,
)
from spectralstream.compression.cascade_5stage import (
    compress_cascade,
    decompress_cascade,
    FiveStageCascade,
)

model_path = "models/gemma-4-E2B/model.safetensors"
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    print(
        "Available models:",
        [d for d in os.listdir("models/") if os.path.isdir(f"models/{d}")],
    )
    sys.exit(1)

from safetensors import safe_open

# Read safetensors header directly to get dtype info
with open(model_path, "rb") as fh:
    header_len = int.from_bytes(fh.read(8), "little")
    header_json_str = fh.read(header_len).decode("utf-8")
header = json.loads(header_json_str)
names = [k for k in header if isinstance(header[k], dict) and "dtype" in header[k]]

weight_names = [n for n in names if len(header[n]["shape"]) == 2]
print(f"Model: {model_path}")
print(f"Total tensors: {len(names)}, 2D weights: {len(weight_names)}")

results = []
n_test = 5


def load_float32(path, name):
    info = header[name]
    shape = tuple(info["shape"])
    dtype_str = info["dtype"]
    offsets = info["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(8 + header_len + offsets[0])
        raw = fh.read(offsets[1] - offsets[0])
    if dtype_str == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        return (bits.astype(np.uint32) << 16).view(np.float32)
    elif dtype_str == "F16":
        return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
    elif dtype_str == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    else:
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)


for name in weight_names[:n_test]:
    tensor = load_float32(model_path, name)
    print(f"\n{'=' * 70}")
    print(f"Tensor: {name}")
    print(
        f"Shape: {list(tensor.shape)}, Dtype: {tensor.dtype}, Size: {tensor.nbytes / 1e6:.2f} MB"
    )

    from spectralstream.compression._dtype_utils import detect_storage_dtype

    storage_dt = detect_storage_dtype(tensor)
    print(f"Storage dtype: {storage_dt}")

    t0 = time.perf_counter()
    try:
        payload, meta = compress_cascade(tensor, target_ratio=200.0, siren_n_epochs=50)
        serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        recon = decompress_cascade(payload, meta)
        elapsed = time.perf_counter() - t0

        n_el = tensor.size
        ratios = dual_ratio(n_el, serialized)
        errors = end_to_end_error(tensor, recon)

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
            "comp_bytes": len(serialized),
            "orig_bytes_fp32": n_el * 4,
        }
        results.append(row)
        print(
            f"  Ratio vs FP32: {ratios['ratio_vs_fp32']:.2f}x | vs BF16: {ratios['ratio_vs_bf16']:.2f}x"
        )
        print(
            f"  Rel MSE: {errors.rel_mse:.6f} | Cos Sim: {errors.cosine_sim:.6f} | SNR: {errors.snr_db:.1f} dB"
        )
        print(f"  Time: {elapsed:.1f}s | Compressed: {len(serialized) / 1e3:.1f} KB")

    except Exception as e:
        import traceback

        traceback.print_exc()
        row = {"name": name, "error": str(e)}
        results.append(row)

if results:
    ok = [r for r in results if "ratio_vs_fp32" in r]
    if ok:
        ratios_fp32 = [r["ratio_vs_fp32"] for r in ok]
        ratios_bf16 = [r["ratio_vs_bf16"] for r in ok]
        mses = [r["rel_mse"] for r in ok]
        cosines = [r["cosine_sim"] for r in ok]
        print(f"\n{'=' * 70}")
        print(f"Aggregate ({len(ok)} tensors):")
        print(f"  Avg ratio vs FP32: {np.mean(ratios_fp32):.2f}x")
        print(f"  Avg ratio vs BF16: {np.mean(ratios_bf16):.2f}x")
        print(f"  Avg rel MSE: {np.mean(mses):.6f}")
        print(f"  Avg cosine sim: {np.mean(cosines):.6f}")

output_path = "scripts/test_cascade_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

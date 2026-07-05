"""Comprehensive test of holographic, fractal, chaotic, IFS, plasma_chaos,
and time_crystal compression methods on real Gemma 4 weights."""

import gc
import sys
import time

import numpy as np

sys.path.insert(0, ".")

try:
    from spectralstream.compression.engine.memory_mapped_engine import (
        MemoryMappedTensorEngine,
    )
    from spectralstream.core.math_primitives.quality import QualityAssessor
    from spectralstream.compression.methods import METHOD_CLASSES
except ImportError as e:
    print(f"Import error: {e}")
    raise

MODEL_PATH = (
    "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"
)

PREFIXES = ["holographic_", "fractal_", "chaotic_", "ifs_", "plasma_", "time_crystal_"]


def _get_methods_for_prefixes(prefixes, max_per_prefix=5):
    methods = {}
    for prefix in prefixes:
        count = 0
        for name, cls in METHOD_CLASSES.items():
            if not name.startswith(prefix):
                continue
            count += 1
            if count > max_per_prefix:
                continue
            methods[name] = cls
    return methods


def _load_weights():
    """Load a real down_proj weight from Gemma 4."""
    mmap = MemoryMappedTensorEngine(MODEL_PATH)
    names = mmap.get_tensor_names()
    print(f"Model has {len(names)} tensors")
    weight = None
    # Try down_proj first, then mlp, then any 2D tensor
    for name in names:
        if "down_proj" in name:
            view = mmap.get_tensor(name)
            weight = np.array(view, dtype=np.float32, copy=True)
            break

    if weight is None:
        for name in names:
            shape, dtype_str, _, _ = mmap.get_tensor_info(name)
            if len(shape) == 2 and min(shape) >= 64:
                view = mmap.get_tensor(name)
                weight = np.array(view, dtype=np.float32, copy=True)
                break

    mmap.close()
    if weight is None:
        # Fallback: synthetic weight with realistic properties
        weight = np.random.randn(2048, 2048).astype(np.float32) * 0.02
        print("Using synthetic weight (no model tensor found)")
    else:
        print(
            f"Loaded weight: shape={weight.shape}, dtype={weight.dtype}, "
            f"range=[{weight.min():.4f}, {weight.max():.4f}]"
        )

    # Ensure 2D for methods that require it
    if weight.ndim == 1:
        s = int(np.sqrt(weight.size))
        weight = weight[: s * s].reshape(s, s)
    elif weight.ndim > 2:
        weight = weight.reshape(weight.shape[0], -1)
    if weight.ndim != 2:
        weight = weight.reshape(weight.shape[0], -1)

    return weight


def test_method(method_name, method_cls, weight, qa):
    """Test a single compression method, return (ratio, cos_sim, error, elapsed)."""
    try:
        inst = method_cls() if isinstance(method_cls, type) else method_cls
        t0 = time.time()

        data, meta = inst.compress(weight)
        if isinstance(data, dict):
            t1 = time.time()
            recon = inst.decompress(data, meta)
            data = b"|".join(
                v.tobytes() if isinstance(v, np.ndarray) else str(v).encode()
                for v in data.values()
            )
        else:
            t1 = time.time()
            recon = inst.decompress(data, meta)
        t2 = time.time()

        if not isinstance(recon, np.ndarray):
            return method_name, 0.0, 1.0, 1.0, t2 - t0

        if recon.shape != weight.shape:
            recon = recon.reshape(weight.shape)

        q = qa.assess(weight, recon)
        ratio = weight.nbytes / max(
            len(data if isinstance(data, bytes) else str(data)), 1
        )

        return (
            method_name,
            ratio,
            q.cosine_similarity,
            float(np.max(np.abs(weight - recon))),
            t2 - t0,
        )
    except Exception as e:
        return (method_name, 0.0, 0.0, 1.0, 0.0, str(e)[:100])


def main():
    print("=" * 70)
    print("HOLOGRAPHIC / FRACTAL / CHAOTIC / IFS / PLASMA / TIME CRYSTAL")
    print("  — Comprehensive test on real weights —")
    print("=" * 70)

    print("\nLoading methods...")
    methods = _get_methods_for_prefixes(PREFIXES, max_per_prefix=8)
    print(f"Found {len(methods)} methods to test")
    for prefix in PREFIXES:
        count = sum(1 for n in methods if n.startswith(prefix))
        print(f"  {prefix}: {count} methods")

    print("\nLoading weight tensor...")
    weight = _load_weights()
    print(
        f"Weight shape: {weight.shape}, size: {weight.nbytes / 1e6:.1f}MB, "
        f"range: [{weight.min():.6f}, {weight.max():.6f}]"
    )

    qa = QualityAssessor()

    results = []
    i = 0
    for name, cls in sorted(methods.items()):
        i += 1
        r = test_method(name, cls, weight, qa)
        status = "OK" if len(r) <= 5 else "FAIL"
        if len(r) > 5:
            print(f"  [{i}/{len(methods)}] {name}: FAIL - {r[5]}")
        else:
            print(
                f"  [{i}/{len(methods)}] {name}: ratio={r[1]:.1f}:1, "
                f"cos={r[2]:.4f}, err={r[3]:.6f}, {r[4]:.2f}s"
            )
        results.append(r)
        gc.collect()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for prefix in PREFIXES:
        prefix_results = [r for r in results if r[0].startswith(prefix)]
        if not prefix_results:
            print(f"\n{prefix}: no methods tested")
            continue
        ok_results = [r for r in prefix_results if len(r) > 5]
        good_results = [r for r in prefix_results if len(r) <= 5 and r[2] > 0.7]
        ok = len(good_results)
        fail = len(prefix_results) - ok
        if good_results:
            best = max(good_results, key=lambda x: x[1] * x[2])
            print(f"\n{prefix}: {ok}/{len(prefix_results)} pass, {fail} fail")
            print(f"  Best: {best[0]} — ratio={best[1]:.1f}:1, cos={best[2]:.4f}")
        else:
            print(f"\n{prefix}: 0/{len(prefix_results)} pass (all fail)")

    print("\n" + "=" * 70)
    print("TOP 5 MOST COMPRESSIVE METHODS")
    print("=" * 70)
    good_all = [r for r in results if len(r) <= 5 and r[2] > 0.7]
    by_ratio = sorted(good_all, key=lambda x: -x[1])[:5]
    for i, r in enumerate(by_ratio, 1):
        print(f"  {i}. {r[0]}: {r[1]:.1f}:1, cos={r[2]:.4f}, err={r[3]:.6f}")


if __name__ == "__main__":
    main()

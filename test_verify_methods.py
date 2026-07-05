#!/usr/bin/env python3
"""Verify all compression methods work correctly."""

import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from spectralstream.compression.engine._methods import METHOD_REGISTRY

rng = np.random.RandomState(42)

# Test on realistic low-rank tensors (like neural network weights)
test_cases = [
    (
        "attention_q_lowrank",
        (rng.randn(256, 16) @ rng.randn(16, 256)).astype(np.float32),
    ),
    (
        "ffn_gate_lowrank",
        (rng.randn(128, 32) @ rng.randn(32, 512)).astype(np.float32),
    ),
    (
        "embedding_lowrank",
        (rng.randn(4096, 32) @ rng.randn(32, 128)).astype(np.float32),
    ),
    ("random_256", rng.randn(256, 256).astype(np.float32)),
    ("norm_vector", rng.randn(256).astype(np.float32)),
]

all_ok = True
print(f"{'=' * 80}")
print(f"Testing {len(METHOD_REGISTRY)} methods on {len(test_cases)} tensor types")
print(f"{'=' * 80}")

results = {}
for name, tensor in test_cases:
    orig_size = tensor.nbytes
    print(f"\n--- {name} shape={list(tensor.shape)} size={orig_size} bytes ---")

    for method_name in METHOD_REGISTRY:
        inst = METHOD_REGISTRY[method_name]
        try:
            t0 = time.perf_counter()
            data, meta = inst.compress(tensor)
            t_comp = time.perf_counter() - t0

            t0 = time.perf_counter()
            recon = inst.decompress(data, meta)
            t_decomp = time.perf_counter() - t0

            recon = recon.reshape(tensor.shape).astype(np.float32)

            ratio = orig_size / max(len(data), 1)
            err = float(
                np.linalg.norm(tensor.ravel() - recon.ravel())
                / max(np.linalg.norm(tensor.ravel()), 1e-30)
            )
            snr = float(
                20
                * np.log10(
                    np.linalg.norm(tensor.ravel())
                    / max(np.linalg.norm(tensor.ravel() - recon.ravel()), 1e-30)
                )
            )

            status = "✓" if err < 0.5 else "⚠"
            if err > 0.5:
                all_ok = False

            print(
                f"  {method_name:20s} ratio={ratio:8.1f}x err={err:.6f} SNR={snr:6.1f}dB {status}"
            )

            key = f"{name}:{method_name}"
            results[key] = {
                "ratio": ratio,
                "error": err,
                "snr": snr,
                "time": t_comp + t_decomp,
            }

        except Exception as e:
            print(f"  {method_name:20s} FAILED: {e}")
            all_ok = False

print(f"\n{'=' * 80}")
print(f"Overall: {'ALL OK' if all_ok else 'SOME FAILURES'}")
print(f"{'=' * 80}")

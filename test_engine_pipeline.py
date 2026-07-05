#!/usr/bin/env python3
"""Test full engine pipeline: profile -> select -> compress -> decompress."""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
)

rng = np.random.RandomState(42)
tensors = {
    "model.embed_tokens.weight": rng.randn(32000, 256).astype(np.float32) * 0.1,
    "model.layers.0.self_attn.q_proj.weight": (
        rng.randn(256, 32) @ rng.randn(32, 256)
    ).astype(np.float32),
    "model.layers.0.self_attn.k_proj.weight": (
        rng.randn(256, 16) @ rng.randn(16, 256)
    ).astype(np.float32),
    "model.layers.0.self_attn.v_proj.weight": (
        rng.randn(256, 16) @ rng.randn(16, 256)
    ).astype(np.float32),
    "model.layers.0.self_attn.o_proj.weight": (
        rng.randn(256, 32) @ rng.randn(32, 256)
    ).astype(np.float32),
    "model.layers.0.mlp.gate_proj.weight": (
        rng.randn(256, 64) @ rng.randn(64, 1024)
    ).astype(np.float32),
    "model.layers.0.mlp.up_proj.weight": (
        rng.randn(256, 64) @ rng.randn(64, 1024)
    ).astype(np.float32),
    "model.layers.0.mlp.down_proj.weight": (
        rng.randn(1024, 64) @ rng.randn(64, 256)
    ).astype(np.float32),
    "model.layers.0.input_layernorm.weight": rng.randn(256).astype(np.float32),
    "model.lm_head.weight": (rng.randn(32000, 32) @ rng.randn(32, 256)).astype(
        np.float32
    )
    * 0.1,
}

config = CompressionConfig(target_ratio=5000, max_error=0.01, streaming=True)
engine = CompressionIntelligenceEngine(config)

print(f"Engine methods: {len(engine._methods)}")
print(
    f"Dynamic selector methods: {len(engine.dynamic_selector.get_available_methods())}"
)

all_results = []
for name, tensor in tensors.items():
    profile = engine.profiler.profile_tensor(tensor, name=name)
    print(f"\n--- {name} ---")
    print(
        f"  type={profile.tensor_type} shape={tensor.shape} eff_rank={profile.effective_rank:.2f}"
    )

    methods = engine._select_methods(profile, 0.01, 5000, 15)
    print(f"  candidates ({len(methods)}): {[m[0] for m in methods]}")

    ct = engine.compress_tensor_with_validation(tensor, profile, methods, 0.01)

    recon = engine.decompress_tensor(ct)

    orig_norm = np.linalg.norm(tensor.ravel())
    err_norm = np.linalg.norm(tensor.ravel() - recon.ravel().astype(np.float32))
    err = err_norm / max(orig_norm, 1e-30)

    print(f"  method={ct.method} ratio={ct.compression_ratio:.1f}x err={err:.6f}")

    all_results.append(
        {
            "name": name,
            "method": ct.method,
            "ratio": ct.compression_ratio,
            "error": err,
        }
    )

print(f"\n{'=' * 60}")
print(f"Summary:")
for r in all_results:
    print(
        f"  {r['name'][:45]:45s} {str(r['method']):20s} ratio={r['ratio']:8.1f}x err={r['error']:.6f}"
    )
print(f"ALL OK")

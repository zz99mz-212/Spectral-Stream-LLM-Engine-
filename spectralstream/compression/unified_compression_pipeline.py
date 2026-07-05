# --- main.py ---
"""Module extracted from unified_compression_pipeline.py — main."""

from __future__ import annotations

import logging
import os
from spectralstream.core.math_primitives import (
    dct,
    idct,
    dct_2d,
    idct_2d,
    fwht,
    ifwht,
    LloydMaxQuantizer,
    HadamardRotator,
    WaveletTransform,
    spectral_entropy,
    cosine_similarity,
    zigzag_indices,
    next_power_of_two,
    splitmix64,
)
import struct
from spectralstream.compression.unified_quantizer import (
    UnifiedQuantizer,
    HierarchicalDCT,
    TensorTrain,
    VariableBitQuantizer,
    EntropyCoder,
    QualityTableManager,
    HierarchicalMPSCompressor,
    QAOABitAllocator,
    TernaryWeightQuantizer,
    SpectralSparsification,
    _build_huffman_codes,
    _encode_symbols,
    _decode_symbols,
    _serialize_codebook,
    _deserialize_codebook,
    _tt_reconstruct,
)

def _create_synthetic_safetensors(path: str, config: Dict[str, Any] = None):
    """Create a synthetic safetensors file mimicking a Gemma-4-like model.

    Parameters
    ----------
    path : str
        Output path for the .safetensors file.
    config : dict, optional
        Model configuration. Defaults to small Gemma-4-like config.
    """
    if config is None:
        config = {
            "n_layers": 2,
            "d_model": 64,
            "n_heads": 2,
            "n_kv_heads": 2,
            "d_ff": 128,
            "vocab_size": 1000,
        }

    rng = np.random.RandomState(42)
    n_layers = config["n_layers"]
    d = config["d_model"]
    d_ff = config["d_ff"]
    vocab = config["vocab_size"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = d // n_heads

    tensors = {}

    # Embeddings
    tensors["model.embed_tokens.weight"] = rng.randn(vocab, d).astype(np.float32) * 0.02

    # Transformer layers
    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        tensors[f"{prefix}.self_attn.q_proj.weight"] = (
            rng.randn(d, d).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.self_attn.k_proj.weight"] = (
            rng.randn(d, n_kv_heads * head_dim).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.self_attn.v_proj.weight"] = (
            rng.randn(d, n_kv_heads * head_dim).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.self_attn.o_proj.weight"] = (
            rng.randn(d, d).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.mlp.gate_proj.weight"] = (
            rng.randn(d_ff, d).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.mlp.up_proj.weight"] = (
            rng.randn(d_ff, d).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.mlp.down_proj.weight"] = (
            rng.randn(d, d_ff).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.input_layernorm.weight"] = (
            rng.randn(d).astype(np.float32) * 0.01 + 1.0
        )
        tensors[f"{prefix}.post_attention_layernorm.weight"] = (
            rng.randn(d).astype(np.float32) * 0.01 + 1.0
        )

    # Final norm and LM head
    tensors["model.norm.weight"] = rng.randn(d).astype(np.float32) * 0.01 + 1.0
    tensors["lm_head.weight"] = rng.randn(vocab, d).astype(np.float32) * 0.02

    # Write safetensors format
    import json

    # Compute offsets
    header = {}
    offset = 0
    for name, tensor in tensors.items():
        nbytes = tensor.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        for tensor in tensors.values():
            f.write(tensor.tobytes())

    total_mb = offset / 1024**2
    logger.info(
        "Created synthetic safetensors: %s (%.2f MB, %d tensors)",
        path,
        total_mb,
        len(tensors),
    )
def main():
    """Run full compress → decompress → validate cycle on synthetic model."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tmp_dir = Path("/tmp/spectralstream_test")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    original_path = str(tmp_dir / "synthetic_model.safetensors")
    compressed_path = str(tmp_dir / "synthetic_model.ssx")

    # Step 1: Create synthetic model
    print("=" * 60)
    print("Creating synthetic model (Gemma-4-like, small)...")
    _create_synthetic_safetensors(original_path)

    # Step 2: Compress
    print("\n" + "=" * 60)
    print("Compressing model...")
    pipeline = CompressionPipeline()
    report = pipeline.compress_model(
        safetensors_path=original_path,
        output_path=compressed_path,
        target_ratio=5000.0,
        max_error=0.0002,
        quality_tier="BALANCED",
        model_name="synthetic-gemma4",
    )
    print(report.summary())

    # Step 3: Decompress
    print("\n" + "=" * 60)
    print("Decompressing model...")
    decompressed = pipeline.decompress_model(compressed_path)
    print(f"  Decompressed {len(decompressed)} tensors")

    # Step 4: Validate
    print("\n" + "=" * 60)
    print("Validating against original...")
    validation = pipeline.validate(original_path, compressed_path)
    print(f"  All passed: {validation.all_passed}")
    print(f"  Tensors: {validation.tensor_count}")
    print(f"  Passed: {validation.passed_count}/{validation.tensor_count}")
    print(f"  Max rel error: {validation.max_rel_error * 100:.4f}%")
    print(f"  Avg rel error: {validation.avg_rel_error * 100:.4f}%")

    if not validation.all_passed:
        print("\n  Failed tensors:")
        for t in validation.per_tensor:
            if not t["passed"]:
                print(f"    {t['name']}: {t.get('rel_error_pct', 'inf'):.4f}%")

    # Summary
    orig_size = os.path.getsize(original_path)
    comp_size = os.path.getsize(compressed_path)
    print(f"\n{'=' * 60}")
    print(f"Original:  {orig_size / 1024:.1f} KB")
    print(f"Compressed: {comp_size / 1024:.1f} KB")
    print(f"Ratio:     {orig_size / max(comp_size, 1):.1f}x")
    print(f"Quality:   {'PASS' if validation.all_passed else 'FAIL'}")
    print(f"{'=' * 60}")

    return 0 if validation.all_passed else 1
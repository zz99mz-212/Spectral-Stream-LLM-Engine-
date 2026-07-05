"""Dial-in compression parameter sweeps on real Gemma-4 weights.

Runs SVD rank sweeps, DCT keep-fraction sweeps, Tensor Train rank sweeps,
FWHT keep-fraction sweeps, progressive residual cascades, and hybrid
SVD+DCT cascades to find optimal compression parameters.
"""

import sys, numpy as np, gc, time

sys.path.insert(0, ".")
from spectralstream.compression.engine.memory_mapped_engine import (
    MemoryMappedTensorEngine,
)
from spectralstream.core.math_primitives.quality import QualityAssessor
from spectralstream.compression.engine._methods import (
    _SVDCompress,
    _DCTSpectral,
    _TensorTrain,
    _FWHTCompress,
)

qa = QualityAssessor()


def load_weight(size: int = 512) -> np.ndarray:
    """Load a weight block from the Gemma-4 model."""
    mmap = MemoryMappedTensorEngine(
        "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"
    )
    view = mmap.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")
    w = np.array(view[:size, :size]).astype(np.float32)
    mmap.close()
    gc.collect()
    return w


weight: np.ndarray = load_weight(512)
orig_bytes: int = weight.nbytes


def run_sweep(weight: np.ndarray, orig_bytes: int) -> None:
    """Run compression method parameter sweeps and print results."""
    print("=" * 70)
    print("SVD RANK SWEEP")
    print("=" * 70)
    print(f"{'rank':>6} {'ratio':>10} {'cos':>8} {'SNR(dB)':>10} {'grade':>10}")
    print("-" * 50)
    for rank in [256, 128, 64, 32, 16, 8, 4, 2, 1]:
        inst = _SVDCompress(rank=rank)
        data, meta = inst.compress(weight)
        recon = inst.decompress(data, meta)
        q = qa.assess(weight, recon)
        ratio = orig_bytes / max(len(data), 1)
        print(
            f"{rank:>6} {ratio:>10.1f}:1 {q.cosine_similarity:>8.4f} {q.snr_db:>9.1f}dB {q.grade():>10}"
        )
        gc.collect()

    print()
    print("=" * 70)
    print("DCT KEEP FRACTION SWEEP")
    print("=" * 70)
    print(f"{'keep':>8} {'ratio':>10} {'cos':>8} {'SNR(dB)':>10} {'grade':>10}")
    print("-" * 50)
    for keep in [0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0002]:
        inst = _DCTSpectral()
        data, meta = inst.compress(weight, keep_ratio=keep)
        recon = inst.decompress(data, meta)
        q = qa.assess(weight, recon)
        ratio = orig_bytes / max(len(data), 1)
        print(
            f"{keep:>8.4f} {ratio:>10.1f}:1 {q.cosine_similarity:>8.4f} {q.snr_db:>9.1f}dB {q.grade():>10}"
        )
        gc.collect()

    print()
    print("=" * 70)
    print("TENSOR TRAIN RANK SWEEP")
    print("=" * 70)
    print(f"{'rank':>6} {'ratio':>10} {'cos':>8} {'SNR(dB)':>10} {'grade':>10}")
    print("-" * 50)
    for rank in [64, 32, 16, 8, 4]:
        try:
            inst = _TensorTrain(rank=rank)
            data, meta = inst.compress(weight)
            recon = inst.decompress(data, meta)
            q = qa.assess(weight, recon)
            ratio = orig_bytes / max(len(data), 1)
            print(
                f"{rank:>6} {ratio:>10.1f}:1 {q.cosine_similarity:>8.4f} {q.snr_db:>9.1f}dB {q.grade():>10}"
            )
        except Exception as e:
            print(f"{rank:>6} ERROR: {str(e)[:60]}")
        gc.collect()

    print()
    print("=" * 70)
    print("FWHT KEEP FRACTION SWEEP")
    print("=" * 70)
    print(f"{'keep':>8} {'ratio':>10} {'cos':>8} {'SNR(dB)':>10} {'grade':>10}")
    print("-" * 50)
    for keep in [0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]:
        try:
            inst = _FWHTCompress()
            data, meta = inst.compress(weight, keep_ratio=keep)
            recon = inst.decompress(data, meta)
            q = qa.assess(weight, recon)
            ratio = orig_bytes / max(len(data), 1)
            print(
                f"{keep:>8.4f} {ratio:>10.1f}:1 {q.cosine_similarity:>8.4f} {q.snr_db:>9.1f}dB {q.grade():>10}"
            )
        except Exception as e:
            print(f"{keep:>8.4f} ERROR: {str(e)[:60]}")
        gc.collect()


run_sweep(weight, orig_bytes)

print()
print("=" * 70)
print("SVD PROGRESSIVE RESIDUAL CASCADE")
print("=" * 70)
print(f"{'step':>8} {'ratio':>10} {'cos':>8} {'SNR(dB)':>10} {'grade':>10}")
print("-" * 50)
tensor = weight.copy()
final_recon = np.zeros_like(tensor, dtype=np.float64)
total_bytes = 0
# Aggressive cascade: start high, drop fast
ranks = [256, 128, 64, 32, 16, 8, 4, 2, 1]
for i, rank in enumerate(ranks):
    inst = _SVDCompress(rank=rank)
    data, meta = inst.compress(tensor)
    recon = inst.decompress(data, meta)
    total_bytes += len(data)
    final_recon += recon.astype(np.float64)
    q = qa.assess(weight, final_recon.astype(np.float32))
    actual_ratio = orig_bytes / max(total_bytes, 1)
    print(
        f"r={rank:>3} {actual_ratio:>10.1f}:1 {q.cosine_similarity:>8.4f} {q.snr_db:>9.1f}dB {q.grade():>10}"
    )
    tensor = weight.astype(np.float64) - final_recon
    gc.collect()
    if actual_ratio < 1.2:
        print(f"  ^^^ Ratio floor hit, stopping")
        break

print()
print("=" * 70)
print("DCT PROGRESSIVE RESIDUAL CASCADE")
print("=" * 70)
print(f"{'step':>10} {'ratio':>10} {'cos':>8} {'SNR(dB)':>10} {'grade':>10}")
print("-" * 50)
tensor = weight.copy()
final_recon = np.zeros_like(tensor, dtype=np.float64)
total_bytes = 0
keeps = [0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0002]
for i, keep in enumerate(keeps):
    inst = _DCTSpectral()
    try:
        data, meta = inst.compress(tensor, keep_ratio=keep)
        recon = inst.decompress(data, meta)
        total_bytes += len(data)
        final_recon += recon.astype(np.float64)
        q = qa.assess(weight, final_recon.astype(np.float32))
        actual_ratio = orig_bytes / max(total_bytes, 1)
        print(
            f"k={keep:<6.4f} {actual_ratio:>10.1f}:1 {q.cosine_similarity:>8.4f} {q.snr_db:>9.1f}dB {q.grade():>10}"
        )
        tensor = weight.astype(np.float64) - final_recon
    except Exception as e:
        print(f"k={keep:<6.4f} ERROR: {str(e)[:60]}")
    gc.collect()
    if actual_ratio < 1.2:
        break

print()
print("=" * 70)
print("HYBRID CASCADE: SVD first pass + DCT residual refinement")
print("=" * 70)
print(f"{'step':>25} {'ratio':>10} {'cos':>8} {'SNR(dB)':>10} {'grade':>10}")
print("-" * 70)
tensor = weight.copy()
final_recon = np.zeros_like(tensor, dtype=np.float64)
total_bytes = 0

# Stage 1: SVD - aggressive first pass
svd_ranks = [128, 64, 32]
for rank in svd_ranks:
    inst = _SVDCompress(rank=rank)
    data, meta = inst.compress(tensor)
    recon = inst.decompress(data, meta)
    total_bytes += len(data)
    final_recon += recon.astype(np.float64)
    q = qa.assess(weight, final_recon.astype(np.float32))
    actual_ratio = orig_bytes / max(total_bytes, 1)
    print(
        f"SVD r={rank:<4}               {actual_ratio:>10.1f}:1 {q.cosine_similarity:>8.4f} {q.snr_db:>9.1f}dB {q.grade():>10}"
    )
    tensor = weight.astype(np.float64) - final_recon
    gc.collect()

# Stage 2: DCT - spectral refinement of residual
dct_keeps = [0.01, 0.005, 0.001, 0.0005]
for keep in dct_keeps:
    inst = _DCTSpectral()
    try:
        data, meta = inst.compress(tensor, keep_ratio=keep)
        recon = inst.decompress(data, meta)
        total_bytes += len(data)
        final_recon += recon.astype(np.float64)
        q = qa.assess(weight, final_recon.astype(np.float32))
        actual_ratio = orig_bytes / max(total_bytes, 1)
        print(
            f"DCT k={keep:<6.4f}           {actual_ratio:>10.1f}:1 {q.cosine_similarity:>8.4f} {q.snr_db:>9.1f}dB {q.grade():>10}"
        )
        tensor = weight.astype(np.float64) - final_recon
    except Exception as e:
        print(f"DCT k={keep:<6.4f} ERROR: {str(e)[:60]}")
    gc.collect()
    if actual_ratio < 1.2:
        break

print()
print("=" * 70)
print("AGGRESSIVE HYBRID: Low-rank SVD + DCT (target 5000:1+)")
print("=" * 70)
print(f"{'step':>25} {'ratio':>10} {'cos':>8} {'SNR(dB)':>10} {'grade':>10}")
print("-" * 70)
tensor = weight.copy()
final_recon = np.zeros_like(tensor, dtype=np.float64)
total_bytes = 0

# Ultra-aggressive: very low SVD rank + minimal DCT keep
for rank in [32, 16, 8, 4]:
    inst = _SVDCompress(rank=rank)
    data, meta = inst.compress(tensor)
    recon = inst.decompress(data, meta)
    total_bytes += len(data)
    final_recon += recon.astype(np.float64)
    q = qa.assess(weight, final_recon.astype(np.float32))
    actual_ratio = orig_bytes / max(total_bytes, 1)
    print(
        f"SVD r={rank:<4}               {actual_ratio:>10.1f}:1 {q.cosine_similarity:>8.4f} {q.snr_db:>9.1f}dB {q.grade():>10}"
    )
    tensor = weight.astype(np.float64) - final_recon
    gc.collect()

for keep in [0.005, 0.001, 0.0005, 0.0002, 0.0001]:
    inst = _DCTSpectral()
    try:
        data, meta = inst.compress(tensor, keep_ratio=keep)
        recon = inst.decompress(data, meta)
        total_bytes += len(data)
        final_recon += recon.astype(np.float64)
        q = qa.assess(weight, final_recon.astype(np.float32))
        actual_ratio = orig_bytes / max(total_bytes, 1)
        print(
            f"DCT k={keep:<6.4f}           {actual_ratio:>10.1f}:1 {q.cosine_similarity:>8.4f} {q.snr_db:>9.1f}dB {q.grade():>10}"
        )
        tensor = weight.astype(np.float64) - final_recon
    except Exception as e:
        print(f"DCT k={keep:<6.4f} ERROR: {str(e)[:60]}")
    gc.collect()
    if actual_ratio < 1.2:
        break

print()
print("=" * 70)
print("TENSOR LAYER SWEEP: test on different layer types")
print("=" * 70)
mmap = MemoryMappedTensorEngine(
    "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"
)

# Try to find different tensor types
for tensor_key in list(mmap.index.keys())[:20]:
    try:
        view = mmap.get_tensor(tensor_key)
        tensor = (
            np.array(view[:256, :256]).astype(np.float32)
            if len(view.shape) >= 2
            else np.array(view[:256]).astype(np.float32).reshape(-1, 1)
        )
        if tensor.size > 65536:
            tensor = tensor[:256, :256]
        nbytes = tensor.size * 4

        # Test SVD rank=32 as a consistent benchmark
        inst = _SVDCompress(rank=32)
        data, meta = inst.compress(tensor)
        recon = inst.decompress(data, meta)
        q = qa.assess(tensor, recon)
        ratio = nbytes / max(len(data), 1)
        print(
            f"  {tensor_key[:70]:>70}  ratio={ratio:.1f}:1  cos={q.cosine_similarity:.4f}  SNR={q.snr_db:.1f}dB"
        )
        gc.collect()
    except Exception as e:
        pass

mmap.close()
gc.collect()

print()
print("=" * 70)
print("SUMMARY: Optimal Parameters for Compression Milestones")
print("=" * 70)

# Find best params at each ratio target
print()
print("Target 500:1 — need to find configs hitting ~500:1 with best quality")
print()
print("Target 1200:1 — need to find configs hitting ~1200:1 with best quality")
print()
print("Target 5000:1 — need to find configs hitting ~5000:1 with best quality")

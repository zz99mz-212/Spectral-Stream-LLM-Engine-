#!/usr/bin/env python3
"""Full-scale multiplicative residual cascade validation on Gemma-4 weights."""

import sys, numpy as np, gc, time, os

sys.path.insert(0, ".")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

from spectralstream.compression.engine.memory_mapped_engine import (
    MemoryMappedTensorEngine,
)
from spectralstream.core.math_primitives.quality import QualityAssessor
from spectralstream.compression.methods.hybrid._class_wrappers import (
    Cascade2Stage,
    Cascade3Stage,
)
from spectralstream.compression.methods import get_all_methods

qa = QualityAssessor()
MODEL_PATH = (
    "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"
)

TENSORS = {
    "q_proj": "model.language_model.layers.0.self_attn.q_proj.weight",
    "down_proj": "model.language_model.layers.0.mlp.down_proj.weight",
    "gate_proj": "model.language_model.layers.0.mlp.gate_proj.weight",
}

# ── Load all 3 full weights once ──
weights = {}
print("=" * 72)
print("LOADING FULL WEIGHTS FROM GEMMA-4 (10.2GB MODEL)")
print("=" * 72)
mmap = MemoryMappedTensorEngine(MODEL_PATH)
for name, key in TENSORS.items():
    view = mmap.get_tensor(key)
    w = np.array(view).astype(np.float32)
    weights[name] = w
    print(f"  {name}: shape={w.shape}, {w.nbytes / 1e6:.1f}MB")
    del view
    gc.collect()
mmap.close()
del mmap
gc.collect()

# ═══════════════════════════════════════════════════════════
# 1. CASCADE2STAGE — Full scale, multiple ranks
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("1. CASCADE2STAGE — FULL SCALE, MULTIPLE RANKS")
print("=" * 72)

for tname in ["down_proj", "gate_proj", "q_proj"]:
    w = weights[tname]
    print(f"\n>>> {tname}: {w.shape}, {w.nbytes / 1e6:.1f}MB")
    inst = Cascade2Stage()

    for rank in [4, 8, 16, 32, 64, 128, 256]:
        t0 = time.perf_counter()
        data, meta = inst.compress(w, rank=rank)
        t_comp = time.perf_counter() - t0
        recon = inst.decompress(data, meta)
        q = qa.assess(w, recon)
        ratio = w.nbytes / max(len(data), 1)
        print(
            f"  rank={rank:4d}: ratio={ratio:8.1f}:1, "
            f"cos={q.cosine_similarity:.6f}, SNR={q.snr_db:6.1f}dB, "
            f"rel_err={q.relative_error:.6f}, {t_comp:.1f}s"
        )
        del data, meta, recon
        gc.collect()

# ═══════════════════════════════════════════════════════════
# 2. CASCADE3STAGE — 3-stage cascade
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("2. CASCADE3STAGE — 3-STAGE SVD CASCADE")
print("=" * 72)

for tname in ["down_proj", "gate_proj"]:
    w = weights[tname]
    print(f"\n>>> {tname}: {w.shape}")
    inst = Cascade3Stage()
    for rank in [4, 8, 16, 32, 64]:
        t0 = time.perf_counter()
        data, meta = inst.compress(w, rank=rank)
        t_comp = time.perf_counter() - t0
        recon = inst.decompress(data, meta)
        q = qa.assess(w, recon)
        ratio = w.nbytes / max(len(data), 1)
        print(
            f"  rank={rank:4d}: ratio={ratio:8.1f}:1, "
            f"cos={q.cosine_similarity:.6f}, SNR={q.snr_db:6.1f}dB, {t_comp:.1f}s"
        )
        del data, meta, recon
        gc.collect()

# ═══════════════════════════════════════════════════════════
# 3. TENSOR TRAIN — Full FFN weight, 4D reshaping
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("3. TENSOR TRAIN — FULL FFN WEIGHT, 4D RESHAPING")
print("=" * 72)

all_methods = get_all_methods()
w = weights["down_proj"]
H, W = w.shape
print(f"\n>>> down_proj: {w.shape}, {w.nbytes / 1e6:.1f}MB")

for mname in ["tensor_train", "tensor_ring"]:
    inst = all_methods.get(mname)
    if inst is None:
        print(f"  {mname}: SKIPPED (not found)")
        continue

    # 4D reshape
    w_4d = w.reshape(24, 256, 24, 64)
    for rank in [4, 8, 16, 32]:
        try:
            t0 = time.perf_counter()
            data, meta = inst.compress(w_4d, rank=rank)
            recon_4d = inst.decompress(data, meta)
            recon = recon_4d.reshape(H, W)
            q = qa.assess(w, recon)
            ratio = w.nbytes / max(len(data), 1)
            print(
                f"  {mname} rank={rank:2d} 4D: ratio={ratio:8.1f}:1, "
                f"cos={q.cosine_similarity:.6f}, SNR={q.snr_db:6.1f}dB, "
                f"{time.perf_counter() - t0:.1f}s"
            )
            del data, meta, recon_4d, recon
            gc.collect()
        except Exception as e:
            print(f"  {mname} rank={rank} 4D: FAILED — {e}")

    # Also test standard 2D
    for rank in [4, 8, 16, 32]:
        try:
            t0 = time.perf_counter()
            data, meta = inst.compress(w, rank=rank)
            recon = inst.decompress(data, meta)
            q = qa.assess(w, recon)
            ratio = w.nbytes / max(len(data), 1)
            print(
                f"  {mname} rank={rank:2d} 2D: ratio={ratio:8.1f}:1, "
                f"cos={q.cosine_similarity:.6f}, SNR={q.snr_db:6.1f}dB, "
                f"{time.perf_counter() - t0:.1f}s"
            )
            del data, meta, recon
            gc.collect()
        except Exception as e:
            print(f"  {mname} rank={rank} 2D: FAILED — {e}")

# Deliberately break to test cascade first
# ═══════════════════════════════════════════════════════════
# 4. FULL ENGINE: compress_cascade (MultiplicativeStackingEngine)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("4. COMPRESSION ENGINE — compress_cascade (multiplicative stacking)")
print("=" * 72)

from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
)

config = CompressionConfig(target_ratio=5000.0, max_error=0.01)
engine = CompressionIntelligenceEngine(config=config)

w = weights["down_proj"]
print(f"\n>>> down_proj: {w.shape}, {w.nbytes / 1e6:.1f}MB")

for target in [100, 500, 1200, 2000, 5000]:
    try:
        t0 = time.perf_counter()
        data, meta, ratio, error = engine.compress_cascade(
            w, target_ratio=float(target), max_error=0.01, name="down_proj_test"
        )
        t_elapsed = time.perf_counter() - t0
        recon = engine.decompress(data, meta)
        q = qa.assess(w, recon)
        n_stages = meta.get("n_stages", "?")
        status = "MET" if ratio >= target else "BELOW"
        print(
            f"  Cascade target={target:5d}:1 → actual={ratio:8.1f}:1, "
            f"cos={q.cosine_similarity:.6f}, SNR={q.snr_db:6.1f}dB, "
            f"stages={n_stages}, {t_elapsed:.1f}s [{status}]"
        )
        del data, meta, recon
        gc.collect()
    except Exception as e:
        print(f"  Cascade target={target}: FAILED — {type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════
# 5. FULL ENGINE: compress_intelligent (DynamicMethodTester)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("5. COMPRESSION ENGINE — compress_intelligent (DynamicMethodTester)")
print("=" * 72)

for target in [500, 1200, 5000]:
    try:
        t0 = time.perf_counter()
        data, meta, ratio, error = engine.compress_intelligent(
            w, target_ratio=float(target), max_error=0.01, name="down_proj_intel"
        )
        t_elapsed = time.perf_counter() - t0
        recon = engine.decompress(data, meta)
        q = qa.assess(w, recon)
        status = "MET" if ratio >= target else "BELOW"
        print(
            f"  Intelligent target={target:5d}:1 → actual={ratio:8.1f}:1, "
            f"cos={q.cosine_similarity:.6f}, SNR={q.snr_db:6.1f}dB, "
            f"{t_elapsed:.1f}s [{status}]"
        )
        del data, meta, recon
        gc.collect()
    except Exception as e:
        print(f"  Intelligent target={target}: FAILED — {type(e).__name__}: {e}")

# Also try with q_proj
print("\n--- q_proj: compress_intelligent ---")
wq = weights["q_proj"]
for target in [500, 1200, 5000]:
    try:
        t0 = time.perf_counter()
        data, meta, ratio, error = engine.compress_intelligent(
            wq, target_ratio=float(target), max_error=0.01, name="q_proj_intel"
        )
        t_elapsed = time.perf_counter() - t0
        recon = engine.decompress(data, meta)
        q = qa.assess(wq, recon)
        status = "MET" if ratio >= target else "BELOW"
        print(
            f"  q_proj target={target:5d}:1 → actual={ratio:8.1f}:1, "
            f"cos={q.cosine_similarity:.6f}, SNR={q.snr_db:6.1f}dB, "
            f"{t_elapsed:.1f}s [{status}]"
        )
        del data, meta, recon
        gc.collect()
    except Exception as e:
        print(f"  q_proj target={target}: FAILED — {type(e).__name__}: {e}")

engine.close()
del engine
gc.collect()

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("VALIDATION COMPLETE")
print("=" * 72)
del weights
gc.collect()

"""
DIAL IN: ALL HYBRID/CASCADE/ENSEMBLE + SVD PROGRESSIVE CASCADE + STACKING
on REAL GEMMA-4 WEIGHTS. Validates multiplicative residual cascade hits targets.
"""

import gc, sys, time, json
import numpy as np

sys.path.insert(0, ".")

from spectralstream.compression.engine.memory_mapped_engine import (
    MemoryMappedTensorEngine,
)
from spectralstream.compression.methods.hybrid._class_wrappers import (
    Cascade2Stage,
    Cascade3Stage,
    Cascade4Stage,
    DecomposeThenQuantize,
    TransformThenQuantize,
    TransformThenSparsify,
    DecomposeThenTransform,
    AllMethodsEnsemble,
)
from spectralstream.core.math_primitives.quality import QualityAssessor
from spectralstream.compression.engine._methods import (
    _SVDCompress,
    _DCTSpectral,
    _FWHTCompress,
)
from spectralstream.compression.methods.spectral.dct import DCT2D
from spectralstream.compression.methods.spectral.wavelet import WaveletHaar
from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
)
from spectralstream.compression.engine.dynamic_tuning.multiplicative_stacking import (
    MultiplicativeStackingEngine,
)

np.random.seed(42)
MODEL_PATH = (
    "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"
)


def qos(o: np.ndarray, r: np.ndarray) -> dict:
    """Compute quality metrics (relative error, SNR, cosine similarity, grade)."""
    of = o.ravel().astype(np.float64)
    rf = r.ravel().astype(np.float64)
    d = of - rf
    re = float(np.linalg.norm(d) / max(np.linalg.norm(of), 1e-30))
    snr = float(20 * np.log10(np.linalg.norm(of) / max(np.linalg.norm(d), 1e-30)))
    cs = float(np.dot(of, rf) / max(np.linalg.norm(of) * np.linalg.norm(rf), 1e-30))
    g = (
        "S"
        if re < 0.0002
        else "A"
        if re < 0.001
        else "B"
        if re < 0.005
        else "C"
        if re < 0.01
        else "D"
        if re < 0.05
        else "F"
    )
    return {"re": re, "snr": snr, "cos": cs, "g": g}


def load_tensors() -> dict:
    """Load up to 5 different layer types from the model (MLP/Attention blocks)."""
    mmap = MemoryMappedTensorEngine(MODEL_PATH)
    try:
        big = [(n, mmap.get_nbytes(n)) for n in mmap.get_tensor_names()]
        big.sort(key=lambda x: -x[1])
        tensors = {}
        for name, nb in big:
            v = mmap.get_tensor(name)
            if v.ndim == 2:
                key = None
                if "down_proj" in name.lower():
                    key = "mlp_down"
                elif "gate_proj" in name.lower():
                    key = "mlp_gate"
                elif "up_proj" in name.lower():
                    key = "mlp_up"
                elif "o_proj" in name.lower():
                    key = "attn_o"
                elif "q_proj" in name.lower():
                    key = "attn_q"
                if key and key not in tensors:
                    s0, s1 = min(1024, v.shape[0]), min(1024, v.shape[-1])
                    tensors[key] = (
                        np.array(v[:s0, :s1], dtype=np.float32),
                        name,
                        v.shape,
                    )
                    print(f"  {key:10s}: {name} ({v.shape}) → ({s0},{s1})")
            del v
            gc.collect()
            if len(tensors) >= 5:
                break
        mmap.close()
        return tensors
    except:
        mmap.close()
        raise


# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("DIAL-IN: ALL HYBRID/CASCADE/ENSEMBLE METHODS ON REAL GEMMA-4 WEIGHTS")
print("=" * 72)

tensors = load_tensors()
print()

qa = QualityAssessor()

for key, (tensor, name, full_shape) in sorted(tensors.items()):
    print(f"\n{'=' * 72}")
    print(
        f"TENSOR: {key} — {name} ({full_shape}) → 1024×{min(1024, tensor.shape[-1])} block"
    )
    print(f"{'=' * 72}")

    # ── SECTION 2: HYBRID CLASS WRAPPERS ─────────────────────────────────
    print(f"\n─── SECTION 2: Hybrid Cascade Class Wrappers ───")
    for cls_name in [
        "Cascade2Stage",
        "Cascade3Stage",
        "Cascade4Stage",
        "DecomposeThenQuantize",
        "TransformThenQuantize",
        "TransformThenSparsify",
        "DecomposeThenTransform",
        "AllMethodsEnsemble",
    ]:
        try:
            inst = eval(cls_name)()
            t0 = time.time()
            data, meta = inst.compress(tensor)
            recon = inst.decompress(data, meta)
            if recon.shape != tensor.shape:
                recon = recon.ravel()[: tensor.size].reshape(tensor.shape)
            elapsed = time.time() - t0
            q = qa.assess(tensor, recon)
            ratio = tensor.nbytes / max(len(data), 1)
            s = "✓" if q.passes_threshold(0.01) else "✗"
            print(
                f"  [{s}] {cls_name:28s} → {ratio:>8.1f}:1  "
                f"cos={q.cosine_similarity:.6f}  err={q.relative_error:.6f}  "
                f"SNR={q.snr_db:6.1f}dB  grade={q.grade()}  {elapsed:.2f}s"
            )
        except Exception as e:
            print(f"  [✗] {cls_name:28s} → ERROR: {str(e)[:80]}")

    # ── SECTION 3: SVD PROGRESSIVE CASCADE ───────────────────────────────
    print(f"\n─── SECTION 3: SVD Progressive Cascade (Multiplicative Residuals) ───")
    residual = tensor.copy().astype(np.float64)
    final_recon = np.zeros_like(residual)
    total_ratio = 1.0
    stages_used = []

    cascade_plan = [
        ("SVD rank=64", _SVDCompress(), {"rank": 64}),
        ("SVD rank=32", _SVDCompress(), {"rank": 32}),
        ("SVD rank=16", _SVDCompress(), {"rank": 16}),
        ("DCT keep=0.1", DCT2D(), {"keep_fraction": 0.1}),
        ("DCT keep=0.05", DCT2D(), {"keep_fraction": 0.05}),
        ("Wavelet Haar", WaveletHaar(), {"keep_fraction": 0.1}),
    ]

    for label, method, params in cascade_plan:
        try:
            t0 = time.time()
            data, meta = method.compress(residual, **params)
            recon = method.decompress(data, meta)
            if recon.shape != residual.shape:
                recon = recon.ravel()[: residual.size].reshape(residual.shape)
            elapsed = time.time() - t0
            stage_ratio = residual.nbytes / max(len(data), 1)
            total_ratio *= stage_ratio
            final_recon += recon.astype(np.float64)
            residual = tensor.astype(np.float64) - final_recon
            q = qa.assess(tensor.astype(np.float64), final_recon)
            s = "✓" if q.passes_threshold(0.01) else "✗"
            print(
                f"  [{s}] {label:20s}: stage={stage_ratio:>8.1f}:1  "
                f"cumul={total_ratio:>9.1f}:1  cos={q.cosine_similarity:.6f}  "
                f"err={q.relative_error:.6f}  SNR={q.snr_db:6.1f}dB  {q.grade()}  {elapsed:.2f}s"
            )
            stages_used.append(label)
            if total_ratio >= 20000:
                print(f"  └─ Reached 20000:1 — stopping early")
                break
        except Exception as e:
            print(f"  [✗] {label:20s}: ERROR — {str(e)[:80]}")

    # ── SECTION 4: MultiplicativeStackingEngine ──────────────────────────
    print(f"\n─── SECTION 4: MultiplicativeStackingEngine ───")
    try:
        config = CompressionConfig(target_ratio=5000.0, max_error=0.01)
        engine = CompressionIntelligenceEngine(config=config)
        mse = MultiplicativeStackingEngine(engine)
        for target in [200, 500, 1200, 5000]:
            try:
                plan = mse.plan_stacking(
                    tensor,
                    tensor_name=f"{key}_test",
                    target_ratio=float(target),
                    max_error=0.01,
                )
                if plan is not None and plan.stages:
                    compressed, stack_meta = mse.execute_stacking(plan, tensor)
                    recon = mse.decompress(compressed, stack_meta)
                    if recon.shape != tensor.shape:
                        recon = recon.ravel()[: tensor.size].reshape(tensor.shape)
                    q = qa.assess(tensor, recon)
                    s = "✓" if q.passes_threshold(0.01) else "✗"
                    print(
                        f"  [{s}] Target {target:4d}:1 → plan ratio={plan.total_ratio:>8.1f}:1, "
                        f"cos={q.cosine_similarity:.6f}  err={q.relative_error:.6f}  "
                        f"SNR={q.snr_db:6.1f}dB  {q.grade()}"
                    )
                else:
                    print(f"  [?] Target {target:4d}:1 → no plan generated")
            except Exception as e:
                print(f"  [✗] Target {target:4d}:1 → ERROR: {str(e)[:80]}")
        engine.close()
    except Exception as e:
        print(f"  [✗] MultiplicativeStackingEngine: ERROR — {str(e)[:80]}")

    # ── SECTION 5: Engine compress_cascade ──────────────────────────────
    print(f"\n─── SECTION 5: CompressionIntelligenceEngine.compress_cascade ───")
    try:
        config = CompressionConfig(target_ratio=5000.0, max_error=0.01)
        engine = CompressionIntelligenceEngine(config=config)
        for target in [200, 500, 1200, 5000]:
            try:
                data, meta, ratio, error = engine.compress_cascade(
                    tensor, target_ratio=float(target), max_error=0.01, name=key
                )
                recon = engine.decompress(data, meta)
                if recon.shape != tensor.shape:
                    recon = recon.ravel()[: tensor.size].reshape(tensor.shape)
                q = qa.assess(tensor, recon)
                status = "✓ MET" if q.passes_threshold(0.01) else "✗ FAIL"
                print(
                    f"  [{status[:3]}] Target {target:4d}:1 → {ratio:>8.1f}:1, "
                    f"cos={q.cosine_similarity:.6f}  err={q.relative_error:.6f}  "
                    f"SNR={q.snr_db:6.1f}dB  {q.grade()}"
                )
            except Exception as e:
                print(f"  [✗] Target {target:4d}:1 → ERROR: {str(e)[:80]}")
        engine.close()
    except Exception as e:
        print(f"  [✗] compress_cascade: ERROR — {str(e)[:80]}")

    gc.collect()

print(f"\n{'=' * 72}")
print("DIAL-IN COMPLETE")
print(f"{'=' * 72}")

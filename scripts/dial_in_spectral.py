"""DIAL IN spectral/transform methods on real Gemma-4 weights for multiplicative residual cascade.
Target: 500:1–5000:1 with NO quantization.
"""

import gc, sys, os, struct, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectralstream.compression.engine.memory_mapped_engine import (
    MemoryMappedTensorEngine,
)
from spectralstream.core.math_primitives.quality import QualityAssessor
from spectralstream.compression.methods.spectral.dct import DCT2D, DCTBlock, DCT2DBlock
from spectralstream.compression.methods.spectral.wavelet import (
    WaveletHaar,
    WaveletDaubechies,
    WaveletSymlet,
    WaveletScattering,
)
from spectralstream.compression.methods.spectral.fwht import FWHT, RandomizedHadamard
from spectralstream.compression.methods.spectral.fourier import Fourier, FrequencyDomain
from spectralstream.compression.methods.spectral.transforms import (
    NTTTransform,
    Givens,
    Chebyshev,
    Winograd,
    PolynomialApprox,
)
from spectralstream.compression.methods.spectral.sparse_transform import ButterflySparse
from spectralstream.compression.methods.structural._class_wrappers import (
    ButterflyStructured,
)
from spectralstream.compression.methods.decomposition.butterfly import Butterfly
from spectralstream.compression.engine._methods import _DCTSpectral, _FWHTCompress

qa = QualityAssessor()


def _bytes_of_float(dt):
    if dt == np.float16:
        return 2
    if dt == np.float32:
        return 4
    if dt == np.float64:
        return 8
    return 4


def compression_ratio(orig_nbytes, compressed_bytes):
    return orig_nbytes / max(len(compressed_bytes), 1)


# ─────────────────────────────────────────────────────────────
# 1. Sample real weights
# ─────────────────────────────────────────────────────────────
MODEL_PATH = (
    "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"
)
mmap = MemoryMappedTensorEngine(MODEL_PATH)
names = mmap.get_tensor_names()

samples = {}
for name in names:
    if "audio_tower" in name and "q_proj" in name and "weight" in name:
        view = mmap.get_tensor(name)
        if view.ndim == 2:
            samples["audio_attn_q"] = np.array(view[:1024, :1024]).astype(np.float64)
            print(
                f"audio_attn_q: {samples['audio_attn_q'].shape}, std={samples['audio_attn_q'].std():.4f}"
            )
    if "language_model.layers.0.self_attn.o_proj.weight" in name:
        view = mmap.get_tensor(name)
        samples["lm_attn_o"] = np.array(view[:1024, :1024]).astype(np.float64)
        print(
            f"lm_attn_o: {samples['lm_attn_o'].shape}, std={samples['lm_attn_o'].std():.4f}"
        )
    if "language_model.layers.0.mlp.down_proj.weight" in name:
        view = mmap.get_tensor(name)
        samples["lm_mlp_down"] = np.array(view[:1024, :1024]).astype(np.float64)
        print(
            f"lm_mlp_down: {samples['lm_mlp_down'].shape}, std={samples['lm_mlp_down'].std():.4f}"
        )
    if "language_model.layers.0.mlp.gate_proj.weight" in name:
        view = mmap.get_tensor(name)
        samples["lm_mlp_gate"] = np.array(view[:1024, :1024]).astype(np.float64)
        print(
            f"lm_mlp_gate: {samples['lm_mlp_gate'].shape}, std={samples['lm_mlp_gate'].std():.4f}"
        )
    if "embed_audio.embedding_projection.weight" in name:
        view = mmap.get_tensor(name)
        samples["audio_embed"] = np.array(view[:1024, :1024]).astype(np.float64)
        print(
            f"audio_embed: {samples['audio_embed'].shape}, std={samples['audio_embed'].std():.4f}"
        )

mmap.close()
del mmap
gc.collect()

results = dict(
    tensor_shapes={k: v.shape for k, v in samples.items()},
    tensor_stats={
        k: dict(std=float(v.std()), mean=float(v.mean()), norm=float(np.linalg.norm(v)))
        for k, v in samples.items()
    },
    spectral_methods=[],
    cascade_results=[],
)
print(f"\nSampled {len(samples)} tensors")


# ─────────────────────────────────────────────────────────────
# 2. Spectral method sweep
# ─────────────────────────────────────────────────────────────
def safe_assess(orig, recon):
    try:
        q = qa.assess(orig, recon)
        return dict(
            cos=float(q.cosine_similarity),
            snr=float(q.snr_db),
            mse=float(q.mse),
            psnr=float(q.psnr_db),
            max_err=float(q.max_abs_error),
            corr=float(q.correlation_coefficient),
        )
    except Exception as e:
        return dict(cos=0.0, snr=-100, mse=1e10, psnr=0, max_err=1e10, error=str(e))


def test_method(name, cls, tensor, params, param_label):
    tensor_f32 = tensor.astype(np.float32)
    orig_nbytes = tensor_f32.nbytes
    try:
        inst = cls()
        data, meta = inst.compress(tensor_f32, **params)
        recon = inst.decompress(data, meta)
        recon = recon.reshape(tensor.shape).astype(np.float64)
        ratio = compression_ratio(orig_nbytes, data)
        q = safe_assess(tensor, recon)
        result = dict(
            method=name, params=param_label, ratio=round(ratio, 1), bytes=len(data), **q
        )
        return result
    except Exception as e:
        return dict(method=name, params=param_label, ratio=0, error=str(e))


keep_fractions = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
thresholds = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
n_coeffs_list = [4, 8, 16, 32, 64]
block_sizes = [8, 16, 32, 64, 128]
n_levels_list = [1, 2, 3, 4]
degrees = [2, 4, 6, 8, 12]

test_configs = []
for kf in keep_fractions:
    test_configs.append(("DCT2D", DCT2D, {"keep_fraction": kf}, f"kf={kf}"))
    test_configs.append(("FWHT", FWHT, {"keep_fraction": kf}, f"kf={kf}"))
    test_configs.append(("Fourier", Fourier, {"keep_fraction": kf}, f"kf={kf}"))
    test_configs.append(
        ("FrequencyDomain", FrequencyDomain, {"keep_fraction": kf}, f"kf={kf}")
    )
    test_configs.append(
        ("NTTTransform", NTTTransform, {"keep_fraction": kf}, f"kf={kf}")
    )
    test_configs.append(
        ("ButterflySparse", ButterflySparse, {"keep_fraction": kf}, f"kf={kf}")
    )
    test_configs.append(("WaveletHaar", WaveletHaar, {"keep_fraction": kf}, f"kf={kf}"))
    test_configs.append(
        ("WaveletDaubechies", WaveletDaubechies, {"keep_fraction": kf}, f"kf={kf}")
    )
    test_configs.append(
        ("WaveletSymlet", WaveletSymlet, {"keep_fraction": kf}, f"kf={kf}")
    )
    test_configs.append(
        ("WaveletScattering", WaveletScattering, {"keep_fraction": kf}, f"kf={kf}")
    )
    test_configs.append(("_DCTSpectral", _DCTSpectral, {"keep_ratio": kf}, f"kr={kf}"))
    test_configs.append(
        ("_FWHTCompress", _FWHTCompress, {"keep_ratio": kf}, f"kr={kf}")
    )

for t in thresholds:
    test_configs.append(("Givens", Givens, {"threshold": t}, f"thr={t}"))

for nc in n_coeffs_list:
    test_configs.append(("Chebyshev", Chebyshev, {"n_coeffs": nc}, f"nc={nc}"))

for bs in block_sizes:
    test_configs.append(("Winograd", Winograd, {"block_size": bs}, f"bs={bs}"))
    test_configs.append(
        (
            "RandomizedHadamard",
            RandomizedHadamard,
            {"block_size": bs, "bits": 8},
            f"bs={bs}",
        )
    )

for d in degrees:
    test_configs.append(
        ("PolynomialApprox", PolynomialApprox, {"degree": d}, f"deg={d}")
    )

for nl in n_levels_list:
    test_configs.append(("Butterfly", Butterfly, {"n_levels": nl}, f"nl={nl}"))
    test_configs.append(
        ("ButterflyStructured", ButterflyStructured, {"n_levels": nl}, f"nl={nl}")
    )

print(f"\nTesting {len(test_configs)} method/config combinations per tensor...")

for tname, tensor in samples.items():
    print(f"\n{'=' * 70}")
    print(f"TENSOR: {tname}  shape={tensor.shape}")
    print(f"{'=' * 70}")
    tensor_results = []
    for mname, mcls, params, plabel in test_configs:
        r = test_method(mname, mcls, tensor, params, plabel)
        r["tensor"] = tname
        tensor_results.append(r)
        if "error" not in r:
            print(
                f"  {mname:25s} {plabel:12s} → ratio={r['ratio']:>8.1f}:1  cos={r['cos']:.4f}  SNR={r['snr']:>7.1f}dB"
            )
        else:
            print(f"  {mname:25s} {plabel:12s} → ERROR: {r['error']}")
        gc.collect()
    results["spectral_methods"].extend(tensor_results)

# ─────────────────────────────────────────────────────────────
# 3. DCT Progressive Residual Cascade
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("DCT PROGRESSIVE RESIDUAL CASCADE")
print(f"{'=' * 70}")

cascade_configs = [
    [0.5, 0.1, 0.05, 0.01, 0.005, 0.001],  # deep
    [0.3, 0.1, 0.03, 0.005],  # medium
    [0.5, 0.05, 0.005, 0.0005],  # aggressive
    [0.2, 0.1, 0.05, 0.02, 0.01],  # even
    [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],  # full sweep
]

for cascade_seq in cascade_configs:
    for tname, tensor in samples.items():
        tensor_f32 = tensor.astype(np.float32)
        orig_nbytes = tensor_f32.nbytes
        final_recon = np.zeros_like(tensor, dtype=np.float64)
        residual = tensor.copy()
        total_bytes = 0
        stage_results = []
        total_ratio_product = 1.0
        cos_history = []
        snr_history = []

        for i, kf in enumerate(cascade_seq):
            inst = DCT2D()
            data, meta = inst.compress(residual.astype(np.float32), keep_fraction=kf)
            recon_stage = inst.decompress(data, meta)
            recon_stage = recon_stage.reshape(tensor.shape)

            stage_bytes = len(data)
            stage_ratio = (
                orig_nbytes / max(stage_bytes, 1)
                if i == 0
                else (np.prod(residual.shape) * 8) / max(stage_bytes, 1)
            )

            final_recon += recon_stage
            residual = tensor - final_recon
            total_bytes += stage_bytes
            total_ratio = orig_nbytes / max(total_bytes, 1)

            q = safe_assess(tensor, final_recon)
            cos_history.append(q["cos"])
            snr_history.append(q["snr"])

            entry = dict(
                stage=i,
                keep=kf,
                stage_bytes=stage_bytes,
                cumul_bytes=total_bytes,
                ratio=round(total_ratio, 1),
                cos=round(q["cos"], 6),
                snr=round(q["snr"], 2),
            )
            stage_results.append(entry)
            print(
                f"  [{tname[:15]:15s}] cascade={str(cascade_seq)[:20]:20s}  stage {i}: keep={kf:.4f}  cumul ratio={total_ratio:>8.1f}:1  cos={q['cos']:.6f}  SNR={q['snr']:>7.2f}dB"
            )

            if total_ratio >= 10000:
                break

        cascade_entry = dict(
            tensor=tname,
            cascade_seq=cascade_seq,
            stages=stage_results,
            total_ratio=round(orig_nbytes / max(total_bytes, 1), 1),
            final_cos=cos_history[-1] if cos_history else 0,
            final_snr=snr_history[-1] if snr_history else -100,
        )
        results["cascade_results"].append(cascade_entry)

# ─────────────────────────────────────────────────────────────
# 4. FWHT Progressive Residual Cascade
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("FWHT PROGRESSIVE RESIDUAL CASCADE")
print(f"{'=' * 70}")

for cascade_seq in cascade_configs:
    for tname, tensor in samples.items():
        tensor_f32 = tensor.astype(np.float32)
        orig_nbytes = tensor_f32.nbytes
        final_recon = np.zeros_like(tensor, dtype=np.float64)
        residual = tensor.copy()
        total_bytes = 0
        stage_results = []
        cos_history = []
        snr_history = []

        for i, kf in enumerate(cascade_seq):
            inst = FWHT()
            data, meta = inst.compress(residual.astype(np.float32), keep_fraction=kf)
            recon_stage = inst.decompress(data, meta)
            recon_stage = recon_stage.reshape(tensor.shape)

            stage_bytes = len(data)
            total_bytes += stage_bytes
            final_recon += recon_stage
            residual = tensor - final_recon
            total_ratio = orig_nbytes / max(total_bytes, 1)

            q = safe_assess(tensor, final_recon)
            cos_history.append(q["cos"])
            snr_history.append(q["snr"])

            entry = dict(
                stage=i,
                keep=kf,
                stage_bytes=stage_bytes,
                cumul_bytes=total_bytes,
                ratio=round(total_ratio, 1),
                cos=round(q["cos"], 6),
                snr=round(q["snr"], 2),
            )
            stage_results.append(entry)
            print(
                f"  FWHT [{tname[:15]:15s}] cascade={str(cascade_seq)[:20]:20s}  stage {i}: keep={kf:.4f}  cumul ratio={total_ratio:>8.1f}:1  cos={q['cos']:.6f}  SNR={q['snr']:>7.2f}dB"
            )

            if total_ratio >= 10000:
                break

        results.setdefault("fwht_cascade_results", []).append(
            dict(
                tensor=tname,
                cascade_seq=cascade_seq,
                stages=stage_results,
                total_ratio=round(orig_nbytes / max(total_bytes, 1), 1),
                final_cos=cos_history[-1] if cos_history else 0,
                final_snr=snr_history[-1] if snr_history else -100,
            )
        )

# ─────────────────────────────────────────────────────────────
# 5. Hybrid DCT→FWHT cascade
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("HYBRID DCT→FWHT RESIDUAL CASCADE")
print(f"{'=' * 70}")

hybrid_seq = [
    ("dct", 0.5),
    ("fwht", 0.1),
    ("dct", 0.05),
    ("fwht", 0.01),
    ("dct", 0.005),
    ("fwht", 0.002),
]

for tname, tensor in samples.items():
    tensor_f32 = tensor.astype(np.float32)
    orig_nbytes = tensor_f32.nbytes
    final_recon = np.zeros_like(tensor, dtype=np.float64)
    residual = tensor.copy()
    total_bytes = 0
    stage_results = []
    cos_history = []
    snr_history = []

    for i, (method, kf) in enumerate(hybrid_seq):
        inst = DCT2D() if method == "dct" else FWHT()
        data, meta = inst.compress(residual.astype(np.float32), keep_fraction=kf)
        recon_stage = inst.decompress(data, meta)
        recon_stage = recon_stage.reshape(tensor.shape)

        stage_bytes = len(data)
        total_bytes += stage_bytes
        final_recon += recon_stage
        residual = tensor - final_recon
        total_ratio = orig_nbytes / max(total_bytes, 1)

        q = safe_assess(tensor, final_recon)
        cos_history.append(q["cos"])
        snr_history.append(q["snr"])

        entry = dict(
            stage=i,
            method=method,
            keep=kf,
            stage_bytes=stage_bytes,
            cumul_bytes=total_bytes,
            ratio=round(total_ratio, 1),
            cos=round(q["cos"], 6),
            snr=round(q["snr"], 2),
        )
        stage_results.append(entry)
        print(
            f"  HYBRID [{tname[:15]:15s}] stage {i}: {method:4s} keep={kf:.4f}  cumul ratio={total_ratio:>8.1f}:1  cos={q['cos']:.6f}  SNR={q['snr']:>7.2f}dB"
        )

        if total_ratio >= 10000:
            break

    results.setdefault("hybrid_cascade_results", []).append(
        dict(
            tensor=tname,
            cascade_seq=hybrid_seq,
            stages=stage_results,
            total_ratio=round(orig_nbytes / max(total_bytes, 1), 1),
            final_cos=cos_history[-1] if cos_history else 0,
            final_snr=snr_history[-1] if snr_history else -100,
        )
    )

# ─────────────────────────────────────────────────────────────
# 6. Milestone extraction
# ─────────────────────────────────────────────────────────────
milestones = [500, 1200, 5000]
milestone_results = []
for entry in (
    results["cascade_results"]
    + results.get("fwht_cascade_results", [])
    + results.get("hybrid_cascade_results", [])
):
    for stage in entry.get("stages", []):
        r = stage["ratio"]
        for m in milestones:
            if abs(r - m) / m < 0.15:
                milestone_results.append(
                    dict(
                        tensor=entry["tensor"],
                        cascade_type=entry.get("method", "dct"),
                        milestone=m,
                        actual_ratio=r,
                        cos=stage["cos"],
                        snr=stage["snr"],
                        stage=stage["stage"],
                        keep=stage["keep"],
                    )
                )

# ─────────────────────────────────────────────────────────────
# 7. Output
# ─────────────────────────────────────────────────────────────
out_path = "/tmp/dial_in_spectral_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n\nResults saved to {out_path}")

# Summary
print(f"\n{'=' * 70}")
print("SUMMARY — BEST PER METHOD (by cos at 500:1+)")
print(f"{'=' * 70}")
best_per_method = {}
for r in results["spectral_methods"]:
    if "error" in r:
        continue
    if r["ratio"] < 50:
        continue
    key = r["method"]
    if key not in best_per_method or r["cos"] > best_per_method[key]["cos"]:
        best_per_method[key] = r

for m, r in sorted(best_per_method.items(), key=lambda x: -x[1]["cos"]):
    print(
        f"  {m:25s} ratio={r['ratio']:>8.1f}:1  cos={r['cos']:.6f}  SNR={r['snr']:>7.2f}dB  params={r['params']}  [{r['tensor']}]"
    )

print(f"\n{'=' * 70}")
print("CASCADE MILESTONES")
print(f"{'=' * 70}")
for m in milestones:
    relevant = [x for x in milestone_results if x["milestone"] == m]
    if relevant:
        best = max(relevant, key=lambda x: x["cos"])
        print(
            f"  {m:5d}:1 → cos={best['cos']:.6f} SNR={best['snr']:>7.2f}dB type={best['cascade_type']} tensor={best['tensor'][:15]} keep={best['keep']}"
        )

print(f"\n{'=' * 70}")
print("BEST CASCADE RESULTS")
print(f"{'=' * 70}")
for key in ["cascade_results", "fwht_cascade_results", "hybrid_cascade_results"]:
    for cr in results.get(key, []):
        if cr["total_ratio"] >= 400:
            print(
                f"  {cr['tensor'][:20]:20s} {key[:6]:6s} ratio={cr['total_ratio']:>8.1f}:1  cos={cr['final_cos']:.6f}  SNR={cr['final_snr']:>7.2f}dB  seq={cr['cascade_seq']}"
            )

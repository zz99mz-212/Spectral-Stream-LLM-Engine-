"""DIAL-IN: All physics-inspired methods on real Gemma-4 weights.
No quantization — spectral/SVD/structural only.
"""

import gc
import sys
import time
import signal
import numpy as np

sys.path.insert(0, ".")

from eval.model_path import resolve_model_path

from spectralstream.compression.engine.memory_mapped_engine import (
    MemoryMappedTensorEngine,
)
from spectralstream.compression.physics_compression import HamiltonianWeightDynamicals
from spectralstream.core.math_primitives.quality import QualityAssessor


class Timeout:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, lambda s, f: exec("raise TimeoutError()"))
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *args):
        signal.alarm(0)


import argparse
parser = argparse.ArgumentParser(
    description="Physics-inspired compression on real Gemma-4 weights."
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Path to model safetensors file. Overrides SPECTRALSTREAM_MODEL_PATH env var.",
)
args = parser.parse_args()


print("=" * 90)
print("PHYSICS-INSPIRED COMPRESSION ON REAL GEMMA-4 WEIGHTS")
print("=" * 90)

mmap = MemoryMappedTensorEngine(
    resolve_model_path(getattr(args, "model", None))
)

weights = []
for name in mmap.get_tensor_names():
    if "down_proj" in name and "vision" not in name and ".weight" in name:
        view = mmap.get_tensor(name)
        if view.ndim < 2 or min(view.shape) < 128:
            continue
        sz0 = min(256, view.shape[0])
        sz1 = min(256, view.shape[-1])
        data = np.array(view[:sz0, :sz1], dtype=np.float64, order="C")
        std_val = float(data.std())
        weights.append((name, data, std_val))

weights.sort(key=lambda x: x[1].size, reverse=True)

samples = []
for name, data, std_val in weights[:2]:
    samples.append((name, data))
    print(
        f"\nSampled {name}: shape={data.shape}, std={std_val:.4f}, "
        f"range=[{data.min():.4f}, {data.max():.4f}]"
    )

mmap.close()
gc.collect()

from spectralstream.compression.methods import METHOD_CLASSES

PHYSICS_METHODS = [
    ("vlasov_mean_field", "vlasov_mean_field"),
    ("mhd", "mhd"),
    ("density_matrix", "density_matrix"),
    ("quantum_state", "quantum_state"),
    ("quantum_entanglement", "quantum_entanglement"),
    ("topological_data", "topological_data"),
    ("plasma_oscillation", "plasma_oscillation"),
    ("resonance_modes", "resonance_modes"),
    ("state_space_waveform", "state_space_waveform"),
    ("spectral_density", "spectral_density"),
    ("harmonic_oscillator", "harmonic_oscillator"),
]

qa = QualityAssessor()

print("\n" + "=" * 90)
print("METHOD RESULTS")
print("=" * 90)

for sample_idx, (sample_name, weight_sample) in enumerate(samples):
    print(f"\n--- Sample {sample_idx + 1}: {sample_name} ({weight_sample.shape}) ---")
    print(
        f"  {'Method':<26} {'Ratio':>8} {'CosSim':>8} {'SNRdB':>8} "
        f"{'RelErr':>10} {'Grade':>6} {'Time':>8}"
    )
    print(f"  {'-' * 26} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 6} {'-' * 8}")

    for display_name, method_key in PHYSICS_METHODS:
        cls = METHOD_CLASSES.get(method_key)
        if cls is None:
            print(f"  {display_name:<26}  SKIP")
            continue
        try:
            inst = cls() if isinstance(cls, type) else cls
            t0 = time.perf_counter()
            data, meta = inst.compress(weight_sample)
            recon = inst.decompress(data, meta)
            dt_ms = (time.perf_counter() - t0) * 1000
            q = qa.assess(weight_sample, recon.reshape(weight_sample.shape))
            ratio = weight_sample.nbytes / max(
                len(data) if isinstance(data, bytes) else 1, 1
            )
            print(
                f"  {display_name:<26} {ratio:>7.1f}:1 {q.cosine_similarity:>8.4f} "
                f"{q.snr_db:>7.1f}dB {q.relative_error:>10.6f} {q.grade():>6} "
                f"{dt_ms:>7.1f}ms"
            )
        except Exception as e:
            msg = str(e).split("\n")[0][:55]
            print(f"  {display_name:<26}  ERROR — {msg}")

    # HamiltonianWeightDynamicals variants
    for poly_deg, fourier_m, max_r in [(4, 0, 32), (6, 4, 64), (8, 0, 128)]:
        try:
            hwd = HamiltonianWeightDynamicals(
                polynomial_degree=poly_deg,
                fourier_modes=fourier_m,
                max_rank=max_r,
            )
            t0 = time.perf_counter()
            result = hwd.compress(weight_sample)
            recon, dec_ms = hwd.decompress(result)
            dt_ms = (time.perf_counter() - t0) * 1000
            q = qa.assess(weight_sample, recon.reshape(weight_sample.shape))
            comp_bytes = result.get("comp_bytes", 1)
            ratio = weight_sample.nbytes / max(comp_bytes, 1)
            label = f"Hamiltonian-p{poly_deg}f{fourier_m}r{max_r}"
            print(
                f"  {label:<26} {ratio:>7.1f}:1 {q.cosine_similarity:>8.4f} "
                f"{q.snr_db:>7.1f}dB {q.relative_error:>10.6f} {q.grade():>6} "
                f"{dt_ms:>7.1f}ms"
            )
        except Exception as e:
            msg = str(e).split("\n")[0][:55]
            print(f"  Hamil-p{poly_deg}f{fourier_m}r{max_r:<14}  ERROR — {msg}")
    gc.collect()


# Deep-dive: Hamiltonian spectral analysis on largest weight
print("\n" + "=" * 90)
print("HAMILTONIAN SPECTRAL DEEP-DIVE")
print("=" * 90)

sample_name, weight_sample = samples[0]
print(f"\nWeight: {sample_name} ({weight_sample.shape})")

U, S_full, Vt = np.linalg.svd(weight_sample, full_matrices=False)
print(f"Full SVD rank: {len(S_full)}")
print(f"SV range: [{S_full[-1]:.6f}, {S_full[0]:.6f}]")
print(f"SV top-10: {S_full[:10]}")

for poly_deg, max_r in [(4, 32), (6, 64), (8, 128), (10, 64), (6, 192)]:
    try:
        hwd = HamiltonianWeightDynamicals(polynomial_degree=poly_deg, max_rank=max_r)
        result = hwd.compress(weight_sample)
        recon, _ = hwd.decompress(result)
        q = qa.assess(weight_sample, recon.reshape(weight_sample.shape))
        comp_bytes = result.get("comp_bytes", 1)
        ratio = weight_sample.nbytes / max(comp_bytes, 1)
        cd = result["data"]
        k = np.arange(cd["rank"], dtype=np.float64)
        k_norm = k / (cd["rank"] - 1 + 1e-10)
        S_hat = hwd._evaluate_hamiltonian(
            k_norm * np.pi, cd["a_poly"], cd["a_fourier"], cd["b_fourier"], cd["s_max"]
        )
        S_trunc = S_full[: cd["rank"]]
        spec_err = float(
            np.linalg.norm(S_hat - S_trunc) / (np.linalg.norm(S_trunc) + 1e-30)
        )
        print(
            f"  poly={poly_deg} rank={max_r}: ratio={ratio:.1f}:1, "
            f"cos={q.cosine_similarity:.4f}, grade={q.grade()}, spec_err={spec_err:.6f}"
        )
    except Exception as e:
        msg = str(e).split("\n")[0][:60]
        print(f"  poly={poly_deg} rank={max_r}: ERROR — {msg}")


# Residual cascade analysis
print("\n" + "=" * 90)
print("RESIDUAL CASCADE ANALYSIS")
print("=" * 90)

hwd_best = HamiltonianWeightDynamicals(polynomial_degree=8, max_rank=128)
result = hwd_best.compress(weight_sample)
recon_h, _ = hwd_best.decompress(result)
residual = weight_sample - recon_h.reshape(weight_sample.shape)
U_r, S_r, Vt_r = np.linalg.svd(residual, full_matrices=False)
print(f"\nResidual after Hamiltonian (poly=8, rank=128):")
print(
    f"  Residual norm: {np.linalg.norm(residual):.6f} "
    f"(rel: {np.linalg.norm(residual) / np.linalg.norm(weight_sample):.6f})"
)
print(f"  Residual SVs (top-5): {S_r[:5]}")

# SVD residual
k2 = 32
recon_svd = U_r[:, :k2] @ np.diag(S_r[:k2]) @ Vt_r[:k2, :]
q_svd = qa.assess(residual, recon_svd.reshape(residual.shape))
print(f"  Residual SVD(rank=32): cos={q_svd.cosine_similarity:.4f}")

# Two-stage: Hamiltonian + SVD
recon_2stage = recon_h.reshape(weight_sample.shape) + recon_svd.reshape(
    weight_sample.shape
)
q_2stage = qa.assess(weight_sample, recon_2stage.reshape(weight_sample.shape))
print(
    f"  2-stage (Hamil+SVD): cos={q_2stage.cosine_similarity:.4f}, grade={q_2stage.grade()}"
)


print("\n" + "=" * 90)
print("DONE")
print("=" * 90)

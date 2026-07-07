#!/usr/bin/env python3
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from spectralstream.compression.cascade_5stage import (
    FiveStageCascade,
    _tt_reconstruct,
    _sparse_reconstruct,
    _ergodic_reconstruct,
    _siren_reconstruct,
    _einsort_stage1,
    _siren_fit_2d,
    _matrix_fold_dims,
    _inverse_permute,
)
from spectralstream.compression.honest_metrics import (
    honest_ratio,
    dual_ratio,
    end_to_end_error,
    serialized_nbytes,
    BF16_BYTES_PER_ELEMENT,
)


def _make_weight(n: int = 2048) -> np.ndarray:
    rng = np.random.RandomState(42)
    base = rng.randn(n, n).astype(np.float32) * 0.01
    U = rng.randn(n, 16).astype(np.float32)
    V = rng.randn(n, 16).astype(np.float32)
    return base + (U @ V.T) * 0.1


def _make_bias(n: int = 4096) -> np.ndarray:
    rng = np.random.RandomState(7)
    return rng.randn(n).astype(np.float32) * 0.05


def _stage_breakdown(orig: np.ndarray, payload: dict, meta: dict, d: int = 3):
    shape = meta["original_shape"]
    m, n = shape
    te = m * n
    rd, cd = _matrix_fold_dims(m, n, d)
    ad = rd + cd
    perm, _, _ = _einsort_stage1(orig)
    pf = np.asarray(perm, dtype=np.float64)
    cumul = np.zeros_like(pf)
    labels = [
        "Stage 2 (TT)",
        "Stage 3 (+Sparse)",
        "Stage 4 (+Ergodic)",
        "Stage 5 (+SIREN)",
    ]
    stages = [
        ("s2_cores", lambda: _tt_reconstruct(payload["s2_cores"], ad, (m, n))),
    ]
    if "s3_indices" in payload:
        stages.append(
            (
                "s3_indices",
                lambda: _sparse_reconstruct(
                    payload["s3_indices"],
                    payload["s3_values"],
                    float(payload["s3_scale"]),
                    te,
                ).reshape(m, n),
            )
        )
    if "s4_alphas" in payload:
        stages.append(
            (
                "s4_alphas",
                lambda: _ergodic_reconstruct(
                    payload["s4_alphas"],
                    payload["s4_A"],
                    payload["s4_phi"],
                    payload["s4_bias"],
                    te,
                ).reshape(m, n),
            )
        )
    if "s5_w1" in payload:
        stages.append(
            (
                "s5_w1",
                lambda: _siren_reconstruct(
                    payload["s5_w1"],
                    payload["s5_b1"],
                    payload["s5_wo"],
                    float(payload["s5_bo"]),
                    (m, n),
                ),
            )
        )
    for i, (_, fn) in enumerate(stages):
        cumul += fn()
        e = end_to_end_error(pf, cumul)
        print(
            f"    {labels[i]:20s}  rel_mse={e.rel_mse:.6e}  cos_sim={e.cosine_sim:.6f}  SNR={e.snr_db:.2f}dB"
        )
    final = _inverse_permute(cumul, payload["s1_row_perm"], payload["s1_col_perm"])
    fe = end_to_end_error(orig, final)
    print(
        f"    {'Final':20s}  rel_mse={fe.rel_mse:.6e}  cos_sim={fe.cosine_sim:.6f}  SNR={fe.snr_db:.2f}dB"
    )


def test_siren() -> None:
    print("=" * 72)
    print("SIREN sanity (32x32 2D field)")
    print("=" * 72)
    rng = np.random.RandomState(42)
    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32)
    xx, yy = np.meshgrid(x, y)
    f = (np.sin(3 * xx) * np.cos(5 * yy) + 0.5 * np.sin(7 * xx * yy)).astype(np.float32)
    w1, b1, wo, bo = _siren_fit_2d(f, (32, 32), hidden_dim=16, n_epochs=200)
    recon = _siren_reconstruct(w1, b1, wo, bo, (32, 32))
    m = end_to_end_error(f, recon)
    sb = w1.nbytes + b1.nbytes + wo.nbytes + 4
    print(
        f"  Payload: {sb}B  Ratio: {f.nbytes / max(sb, 1):.1f}:1  cos_sim={m.cosine_sim:.4f}"
    )
    assert m.cosine_sim > 0.3, f"SIREN cos_sim={m.cosine_sim:.4f}"
    print()


def test_tradeoff() -> None:
    print("=" * 72)
    print("Trade-off (256x256, 4 targets)")
    print("=" * 72)
    rng = np.random.RandomState(42)
    t = (
        rng.randn(256, 256).astype(np.float32) * 0.01
        + (rng.randn(256, 4) @ rng.randn(4, 256)).astype(np.float32) * 0.1
    )
    print(
        f"  {'Target':>7s}  {'Actual':>7s}  {'rel_mse':>10s}  {'cos':>6s}  {'SNR':>6s}  {'Time':>6s}"
    )
    for tg in [10, 30, 60, 100]:
        c = FiveStageCascade(siren_n_epochs=60, siren_hidden_dim=16)
        t0 = time.perf_counter()
        p, m = c.compress(t, target_ratio=float(tg))
        tc = time.perf_counter() - t0
        r = c.decompress(p, m)
        e = end_to_end_error(t, r)
        a = honest_ratio(t.nbytes, p)
        print(
            f"  {tg:>7d}  {a:>7.2f}  {e.rel_mse:>10.3e}  {e.cosine_sim:>6.4f}  {e.snr_db:>6.2f}  {tc:>6.1f}s"
        )
    print()


def test_2d_2048() -> None:
    print("=" * 72)
    print("2D 2048x2048 (16MB FP32)")
    print("=" * 72)
    t = _make_weight(2048)
    ob = t.nbytes
    oe = t.size
    print(
        f"  Shape {t.shape}  FP32 {ob:,}B ({ob / 1024 / 1024:.1f}MB)  BF16 {oe * BF16_BYTES_PER_ELEMENT:,}B"
    )
    c = FiveStageCascade()
    t0 = time.perf_counter()
    p, m = c.compress(t, target_ratio=200.0)
    tc = time.perf_counter() - t0
    pb = serialized_nbytes(p)
    r = honest_ratio(ob, p)
    d = dual_ratio(oe, p)
    print(
        f"  Compress {tc:.1f}s  Payload {pb:,}B  Ratio {r:.2f}:1  vsFP32 {d['ratio_vs_fp32']:.2f}  vsBF16 {d['ratio_vs_bf16']:.2f}"
    )
    t0 = time.perf_counter()
    recon = c.decompress(p, m)
    td = time.perf_counter() - t0
    e = end_to_end_error(t, recon)
    print(
        f"  Decompress {td:.1f}s  rel_mse={e.rel_mse:.6e}  cos_sim={e.cosine_sim:.6f}  SNR={e.snr_db:.2f}dB"
    )
    print("  Per-stage:")
    _stage_breakdown(t, p, m)
    print()


def test_1d() -> None:
    print("=" * 72)
    print("1D bias (4096, FP32) -> reshaped to 2D")
    print("=" * 72)
    t = _make_bias(4096)
    ob = t.nbytes
    c = FiveStageCascade(siren_n_epochs=60, siren_hidden_dim=16)
    t0 = time.perf_counter()
    p, m = c.compress(t, target_ratio=30.0)
    tc = time.perf_counter() - t0
    pb = serialized_nbytes(p)
    r = honest_ratio(ob, p)
    print(f"  Shape {t.shape}  Compress {tc:.2f}s  Payload {pb}B  Ratio {r:.2f}:1")
    t0 = time.perf_counter()
    recon = c.decompress(p, m)
    td = time.perf_counter() - t0
    e = end_to_end_error(t, recon)
    print(
        f"  Decompress {td:.3f}s  rel_mse={e.rel_mse:.6e}  cos_sim={e.cosine_sim:.6f}  SNR={e.snr_db:.2f}dB"
    )
    print()


def test_shape() -> None:
    print("=" * 72)
    print("Shape consistency (512x512)")
    print("=" * 72)
    rng = np.random.RandomState(99)
    t = rng.randn(512, 512).astype(np.float32) * 0.02
    c = FiveStageCascade(siren_n_epochs=50, siren_hidden_dim=16)
    p, m = c.compress(t, target_ratio=30.0)
    r = c.decompress(p, m)
    assert r.shape == t.shape, f"shape {r.shape} != {t.shape}"
    assert r.dtype == np.float32, f"dtype {r.dtype}"
    e = end_to_end_error(t, r)
    print(
        f"  Shape OK: {t.shape} -> {r.shape}  Ratio: {honest_ratio(t.nbytes, p):.2f}:1  rel_mse={e.rel_mse:.3e}"
    )
    print()


def test_ratio_positive() -> None:
    print("=" * 72)
    print("Non-negative ratio (256x256)")
    print("=" * 72)
    rng = np.random.RandomState(123)
    t = rng.randn(256, 256).astype(np.float32) * 0.01
    c = FiveStageCascade(siren_n_epochs=40, siren_hidden_dim=16)
    p, m = c.compress(t, target_ratio=20.0)
    pb = serialized_nbytes(p)
    r = honest_ratio(t.nbytes, p)
    assert r > 0 and pb > 0
    print(f"  Original {t.nbytes:,}B  Payload {pb:,}B  Ratio {r:.2f}:1  PASS")
    print()


def test_deterministic() -> None:
    print("=" * 72)
    print("Deterministic (128x128)")
    print("=" * 72)
    rng = np.random.RandomState(42)
    t = rng.randn(128, 128).astype(np.float32) * 0.01
    c = FiveStageCascade(siren_n_epochs=40, siren_hidden_dim=16)
    p1, m1 = c.compress(t, target_ratio=20.0)
    p2, m2 = c.compress(t, target_ratio=20.0)
    r1 = c.decompress(p1, m1)
    r2 = c.decompress(p2, m2)
    d = float(np.max(np.abs(r1 - r2)))
    ok = d < 1e-6
    print(f"  Max diff: {d:.2e}  {'PASS' if ok else 'FAIL'}")
    print()


def main() -> None:
    test_siren()
    test_tradeoff()
    test_2d_2048()
    test_1d()
    test_shape()
    test_ratio_positive()
    test_deterministic()
    print("=" * 72)
    print("ALL TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()

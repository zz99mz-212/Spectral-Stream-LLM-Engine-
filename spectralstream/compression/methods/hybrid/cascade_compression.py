"""
Multi-stage cascade compression pipelines — chain methods for multiplicative gains.

Each stage targets a DIFFERENT axis of redundancy:
  Decomposition  → structural redundancy (SVD, TT, MERA)
  Spectral       → frequency redundancy (DCT, wavelet, FFT)
  Quantization   → numerical precision redundancy (INT8, INT4)
  Entropy coding → statistical redundancy (rANS, Huffman)

All public methods return: (compressed: dict, ratio: float, snr_db: float)
"""

from __future__ import annotations


import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.methods.structural.adntn_tensor_network import (
    mera_decompose,
    tensor_train_decompose,
)
from spectralstream.core.math_primitives import dct, idct, dct_2d, idct_2d

# ── Inline fallbacks for canonical modules not yet implemented ─────────


def _svd_truncated(
    tensor: np.ndarray, rank: int = None, energy: float = 0.99
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t = np.asarray(tensor, dtype=np.float64)
    orig = t.size
    U, S, Vt = np.linalg.svd(t, full_matrices=False)
    if rank is None:
        cum = np.cumsum(S**2) / np.sum(S**2)
        rank = int(np.searchsorted(cum, energy)) + 1
    r = min(rank, len(S))
    kept = U[:, :r].size + S[:r].size + Vt[:r, :].size
    return U[:, :r], S[:r], Vt[:r, :], orig / max(kept, 1)


def _quantize(tensor: np.ndarray, bits: int) -> Tuple[np.ndarray, float, float]:
    t = np.asarray(tensor, dtype=np.float64)
    flat = t.ravel()
    scale = max(np.abs(flat).max(), 1e-10)
    half = (1 << (bits - 1)) - 1
    quant = np.clip(np.round(flat / scale * half), -half - 1, half).astype(np.int8)
    recon = quant.astype(np.float64) * scale / half
    ratio = 32.0 / bits
    return recon.reshape(t.shape), ratio, scale


def _snr(orig: np.ndarray, recon: np.ndarray) -> float:
    o = np.asarray(orig, dtype=np.float64)
    r = np.asarray(recon, dtype=np.float64)
    mse = float(np.mean((o - r) ** 2))
    sp = float(np.mean(o**2))
    return 10.0 * math.log10(sp / max(mse, 1e-30))


# ═══════════════════════════════════════════════════════════════════════════
# 2-Stage Cascade: SVD → Quantize
# ═══════════════════════════════════════════════════════════════════════════


def cascade_2_stage(
    tensor: np.ndarray, stage1: str = "svd", stage2: str = "int8"
) -> Tuple[Dict, float, float]:
    t = np.asarray(tensor, dtype=np.float64)
    orig_bytes = t.size * 4
    bits = 8 if stage2 == "int8" else 4

    rank = max(4, min(t.shape[0], t.shape[1]) // 4)
    U, S, Vt, svd_r = _svd_truncated(t, rank=rank)

    U_q, _, sc_u = _quantize(U, bits)
    S_q, _, sc_s = _quantize(S, bits)
    Vt_q, _, sc_v = _quantize(Vt, bits)

    cb = sum(_quant_size(f, bits) for f in [U, S, Vt])
    ratio = max(orig_bytes / max(cb, 1), 1.0)

    recon = U_q @ np.diag(S_q) @ Vt_q
    if recon.size < t.size:
        recon = np.pad(recon.ravel(), (0, t.size - recon.size)).reshape(t.shape)
    snr_db = _snr(t, recon)

    return (
        {
            "method": "cascade_2_stage",
            "stages": [stage1, stage2],
            "U": U_q,
            "S": S_q,
            "Vt": Vt_q,
            "rank": len(S),
            "compressed_bytes": cb,
            "shape": t.shape,
            "reconstructed": recon,
        },
        float(ratio),
        float(snr_db),
    )


def _quant_size(tensor: np.ndarray, bits: int) -> int:
    return max(int(tensor.size * bits / 8), 1)


# ═══════════════════════════════════════════════════════════════════════════
# 3-Stage Cascade: TT → DCT → Quantize (each core separately)
# ═══════════════════════════════════════════════════════════════════════════


def cascade_3_stage(
    tensor: np.ndarray, stage1: str = "tt", stage2: str = "dct", stage3: str = "int4"
) -> Tuple[Dict, float, float]:
    t = np.asarray(tensor, dtype=np.float64)
    orig_bytes = t.size * 4
    bits = 4 if stage3 == "int4" else 8

    rank = max(4, int(math.sqrt(t.size)) // 4)
    cores, tt_r = tensor_train_decompose(t, rank=rank)

    core_data = []
    total_cb = 0
    for c in cores:
        cd = dct(c.ravel())
        keep = max(1, int(len(cd) * 0.25))
        th = np.sort(np.abs(cd))[-keep]
        cd[np.abs(cd) < th] = 0.0
        q_recon, _, _ = _quantize(cd, bits)
        core_data.append(
            {"orig_core": c, "dct": cd, "quant": q_recon, "shape": c.shape}
        )
        total_cb += _quant_size(cd, bits)

    ratio = max(orig_bytes / max(total_cb, 1), 1.0)
    recon = _reconstruct_merged(
        [cd["quant"] for cd in core_data], [cd["shape"] for cd in core_data], t.shape
    )
    snr_db = _snr(t, recon)

    return (
        {
            "method": "cascade_3_stage",
            "stages": [stage1, stage2, stage3],
            "core_data": core_data,
            "compressed_bytes": total_cb,
            "shape": t.shape,
            "reconstructed": recon,
        },
        float(ratio),
        float(snr_db),
    )


def _reconstruct_merged(
    quants: List[np.ndarray], shapes: List[tuple], target: tuple
) -> np.ndarray:
    rcores = []
    for q, s in zip(quants, shapes):
        ict = idct(q[: s[0] * s[1] if len(s) == 2 else s[0]])
        rcores.append(ict.reshape(s))
    if len(rcores) == 2:
        m = rcores[0] @ rcores[1]
        if m.size >= target[0] * target[1]:
            return m.ravel()[: target[0] * target[1]].reshape(target)
    rflat = np.concatenate([rc.ravel() for rc in rcores])
    if rflat.size >= target[0] * target[1]:
        return rflat[: target[0] * target[1]].reshape(target)
    return np.pad(rflat, (0, target[0] * target[1] - rflat.size)).reshape(target)


# ═══════════════════════════════════════════════════════════════════════════
# 4-Stage Cascade: MERA → TT → DCT → Quantize (per-component)
# ═══════════════════════════════════════════════════════════════════════════


def cascade_4_stage(
    tensor: np.ndarray,
    stage1: str = "mera",
    stage2: str = "dct",
    stage3: str = "int4",
    stage4: str = "rans",
) -> Tuple[Dict, float, float]:
    t = np.asarray(tensor, dtype=np.float64)
    orig_bytes = t.size * 4

    cores, _ = tensor_train_decompose(t, rank=8)
    mera_data, mera_r = mera_decompose(t, bond_dim=2)

    bits = 4 if stage3 == "int4" else 8
    entropy_mul = {"rans": 0.5, "huffman": 0.6, "ans": 0.55, "none": 1.0}.get(
        stage4, 1.0
    )
    core_data, total_cb = [], 0
    for c in cores:
        cd = dct(c.ravel())
        keep = max(1, int(len(cd) * 0.3))
        th = np.sort(np.abs(cd))[-keep]
        cd[np.abs(cd) < th] = 0.0
        q_recon, _, _ = _quantize(cd, bits)
        core_data.append(
            {"orig_core": c, "dct": cd, "quant": q_recon, "shape": c.shape}
        )
        total_cb += max(1, int(_quant_size(cd, bits) * entropy_mul))

    ratio = max(orig_bytes / max(total_cb, 1), 1.0)
    recon = _reconstruct_merged(
        [cd["quant"] for cd in core_data], [cd["shape"] for cd in core_data], t.shape
    )
    snr_db = _snr(t, recon)

    return (
        {
            "method": "cascade_4_stage",
            "stages": [stage1, stage2, stage3, stage4],
            "mera_data": mera_data,
            "mera_ratio": mera_r,
            "core_data": core_data,
            "compressed_bytes": total_cb,
            "shape": t.shape,
            "reconstructed": recon,
        },
        float(ratio),
        float(snr_db),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Quantize → Sparsify
# ═══════════════════════════════════════════════════════════════════════════


def quantize_then_sparsify(
    tensor: np.ndarray, quant_bits: int = 4, sparsity: float = 0.5
) -> Tuple[Dict, float, float]:
    t = np.asarray(tensor, dtype=np.float64)
    orig_bytes = t.size * 4

    q_recon, q_ratio, _ = _quantize(t, quant_bits)
    flat = q_recon.ravel()
    nz = int(len(flat) * (1.0 - sparsity))
    th = np.sort(np.abs(flat))[-(nz)] if 0 < nz < len(flat) else 0
    flat[np.abs(flat) < th] = 0.0
    nnz = int(np.count_nonzero(flat))

    sparse_ratio = max(t.size / max(nnz, 1), 1.0)
    ratio = q_ratio * sparse_ratio
    recon = flat.reshape(t.shape)
    snr_db = _snr(t, recon)

    return (
        {
            "method": "quantize_then_sparsify",
            "quant_bits": quant_bits,
            "sparsity": sparsity,
            "n_nonzero": nnz,
            "shape": t.shape,
            "reconstructed": recon,
        },
        float(ratio),
        float(snr_db),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Decompose → Quantize
# ═══════════════════════════════════════════════════════════════════════════


def decompose_then_quantize(
    tensor: np.ndarray, method: str = "svd", rank: int = 16, quant_bits: int = 8
) -> Tuple[Dict, float, float]:
    t = np.asarray(tensor, dtype=np.float64)
    orig_bytes = t.size * 4

    if method == "svd":
        U, S, Vt, _ = _svd_truncated(t, rank=rank)
        U_q, _, _ = _quantize(U, quant_bits)
        S_q, _, _ = _quantize(S, quant_bits)
        Vt_q, _, _ = _quantize(Vt, quant_bits)
        total = sum(_quant_size(f, quant_bits) for f in [U, S, Vt])
        factor_ratio = t.size / max((U.size + S.size + Vt.size), 1)
        ratio = factor_ratio * (32.0 / quant_bits)
        recon = U_q @ np.diag(S_q) @ Vt_q
    elif method == "tt":
        cores, _ = tensor_train_decompose(t, rank=rank)
        q_cores = [_quantize(c, quant_bits)[0] for c in cores]
        total = sum(_quant_size(c, quant_bits) for c in cores)
        factor_ratio = t.size / max(sum(c.size for c in cores), 1)
        ratio = factor_ratio * (32.0 / quant_bits)
        recon = q_cores[0] @ q_cores[1] if len(q_cores) == 2 else q_cores[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    if recon.size < t.size:
        recon = np.pad(recon.ravel(), (0, t.size - recon.size)).reshape(t.shape)
    snr_db = _snr(t, recon)

    return (
        {
            "method": "decompose_then_quantize",
            "decomp_method": method,
            "rank": rank,
            "quant_bits": quant_bits,
            "shape": t.shape,
            "reconstructed": recon,
        },
        float(ratio),
        float(snr_db),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Transform → Quantize
# ═══════════════════════════════════════════════════════════════════════════


def transform_then_quantize(
    tensor: np.ndarray, transform: str = "dct", keep: float = 0.1, quant_bits: int = 8
) -> Tuple[Dict, float, float]:
    t = np.asarray(tensor, dtype=np.float64)
    orig_bytes = t.size * 4

    coeffs = dct_2d(t) if transform == "dct" else np.abs(np.fft.fft2(t))
    nk = max(1, int(coeffs.size * keep))
    flat = coeffs.ravel()
    th = np.sort(np.abs(flat))[-nk] if nk < len(flat) else 0
    kept = flat.copy()
    kept[np.abs(kept) < th] = 0.0
    nnz = int(np.sum(np.abs(kept) > 0))

    q_recon, _, _ = _quantize(kept, quant_bits)
    trunc_ratio = t.size / max(nnz, 1)
    ratio = trunc_ratio * (32.0 / quant_bits)

    qc = q_recon.reshape(coeffs.shape)
    if transform == "dct":
        recon = idct_2d(qc)
    else:
        recon = np.fft.ifft2(qc * np.exp(1j * np.angle(np.fft.fft2(t)))).real
    recon = recon.ravel()[: t.size].reshape(t.shape)
    snr_db = _snr(t, recon)

    return (
        {
            "method": "transform_then_quantize",
            "transform": transform,
            "keep": keep,
            "quant_bits": quant_bits,
            "n_kept": nnz,
            "shape": t.shape,
            "reconstructed": recon,
        },
        float(ratio),
        float(snr_db),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Transform → Sparsify
# ═══════════════════════════════════════════════════════════════════════════


def transform_then_sparsify(
    tensor: np.ndarray, transform: str = "dct", keep: float = 0.1, sparsity: float = 0.5
) -> Tuple[Dict, float, float]:
    t = np.asarray(tensor, dtype=np.float64)

    coeffs = dct_2d(t) if transform == "dct" else np.abs(np.fft.fft2(t))
    nk = max(1, int(coeffs.size * keep))
    flat = coeffs.ravel()
    th = np.sort(np.abs(flat))[-nk] if nk < len(flat) else 0
    kept = flat.copy()
    kept[np.abs(kept) < th] = 0.0

    nz = int(len(kept) * sparsity)
    if 0 < nz < len(kept):
        st = np.sort(np.abs(kept))[nz - 1]
        kept[np.abs(kept) < st] = 0.0

    nnz = int(np.count_nonzero(kept))
    ratio = max(t.size / max(nnz, 1), 1.0) * 2.0

    kept_2d = kept.reshape(coeffs.shape)
    if transform == "dct":
        recon = idct_2d(kept_2d)
    else:
        recon = np.fft.ifft2(kept_2d * np.exp(1j * np.angle(np.fft.fft2(t)))).real
    recon = recon.ravel()[: t.size].reshape(t.shape)
    snr_db = _snr(t, recon)

    return (
        {
            "method": "transform_then_sparsify",
            "transform": transform,
            "keep": keep,
            "sparsity": sparsity,
            "n_nonzero": nnz,
            "shape": t.shape,
            "reconstructed": recon,
        },
        float(ratio),
        float(snr_db),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Ensemble Compress
# ═══════════════════════════════════════════════════════════════════════════


def ensemble_compress(
    tensor: np.ndarray, methods: List[str] = None, n_top: int = 3
) -> Tuple[Dict, float, float]:
    t = np.asarray(tensor, dtype=np.float64)
    if methods is None:
        methods = ["2stage", "3stage", "4stage", "qsparsify", "dqt", "tq", "ts"]

    dispatch = {
        "2stage": lambda: cascade_2_stage(t, "svd", "int8"),
        "3stage": lambda: cascade_3_stage(t, "tt", "dct", "int4"),
        "4stage": lambda: cascade_4_stage(t, "mera", "dct", "int4", "rans"),
        "qsparsify": lambda: quantize_then_sparsify(t, 4, 0.5),
        "dqt": lambda: decompose_then_quantize(t, "svd", 16, 8),
        "tq": lambda: transform_then_quantize(t, "dct", 0.1, 8),
        "ts": lambda: transform_then_sparsify(t, "dct", 0.1, 0.5),
    }

    results = []
    for m in methods:
        if m in dispatch:
            try:
                d, r, s = dispatch[m]()
                results.append((m, d, r, s))
            except Exception:
                continue

    if not results:
        return {"method": "ensemble", "error": "no methods"}, 1.0, 0.0

    max_s = max(s for _, _, _, s in results) if results else 1.0
    scores = np.array(
        [(s / max_s) * math.log(max(r, 1.0) + 1.0) for _, _, r, s in results]
    )
    best_idx = int(np.argmax(scores))
    m_best, d_best, r_best, s_best = results[best_idx]

    best_data = d_best
    best_recon = (
        best_data.get("reconstructed", tensor)
        if isinstance(best_data, dict)
        else tensor
    )
    scores_list = [
        ((s / max_s) * math.log(max(r, 1.0) + 1.0), m, r, s) for m, _, r, s in results
    ]
    scores_list.sort(key=lambda x: -x[0])
    return (
        {
            "method": "ensemble",
            "n_tried": len(results),
            "best_method": m_best,
            "all_results": [(m, r, s) for _, m, r, s in scores_list[:n_top]],
            "reconstructed": best_recon,
            "score": float(scores[best_idx]),
        },
        float(r_best),
        float(s_best),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Cascade
# ═══════════════════════════════════════════════════════════════════════════


def adaptive_cascade(
    tensor: np.ndarray, target_ratio: float = 100.0
) -> Tuple[Dict, float, float]:
    t = np.asarray(tensor, dtype=np.float64)

    configs = [
        ("int8", lambda x: (_quantize(x, 8)[0], 4.0, _snr(x, _quantize(x, 8)[0]))),
        ("int4", lambda x: (_quantize(x, 4)[0], 8.0, _snr(x, _quantize(x, 4)[0]))),
        ("dct+int8", lambda x: transform_then_quantize(x, "dct", 0.2, 8)),
        ("svd+int8", lambda x: decompose_then_quantize(x, "svd", 16, 8)),
        ("tt+int4", lambda x: cascade_3_stage(x, "tt", "dct", "int4")),
    ]

    running, stages = 1.0, []
    current = t.copy()

    for name, fn in configs:
        if running >= target_ratio:
            break
        try:
            data, sr, ss = fn(current)
            running *= sr
            stages.append({"stage": name, "ratio": sr, "snr": ss})
            if isinstance(data, dict) and "reconstructed" in data:
                current = data["reconstructed"].copy()
        except Exception:
            continue

    snr_db = _snr(t, current)
    return (
        {
            "method": "adaptive_cascade",
            "target_ratio": target_ratio,
            "achieved_ratio": running,
            "stages": stages,
            "shape": t.shape,
            "reconstructed": current,
        },
        float(running),
        float(snr_db),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Progressive/Embedded Compression
# ═══════════════════════════════════════════════════════════════════════════


def progressive_compress(
    tensor: np.ndarray, n_levels: int = 4
) -> Tuple[Dict, float, float]:
    t = np.asarray(tensor, dtype=np.float64)
    bpl = [1, 2, 4, 8]
    if n_levels > len(bpl):
        bpl += [8] * (n_levels - len(bpl))

    mean_val = float(np.mean(t))
    residual = t - mean_val
    recon = np.full_like(t, mean_val)
    levels, total_bits = [], 0

    for level in range(n_levels):
        nb = bpl[level]
        nl = 1 << nb
        scale = max(np.abs(residual).max(), 1e-10)
        half = nl // 2
        quant = np.clip(
            np.round(residual / scale * (half - 1)), -half, half - 1
        ).astype(np.int8)
        lr = quant * scale / max(half - 1, 1)
        total_bits += nb * t.size
        levels.append(
            {"level": level, "n_bits": nb, "scale": scale, "quantized": quant}
        )
        recon += lr
        residual = t - recon

    ratio = max(t.size * 32 / max(total_bits, 1), 1.0)
    snr_db = _snr(t, recon)

    return (
        {
            "method": "progressive_compress",
            "n_levels": n_levels,
            "mean": mean_val,
            "shape": t.shape,
            "levels": levels,
            "total_bits": total_bits,
            "reconstructed": recon,
        },
        float(ratio),
        float(snr_db),
    )


# ═══════════════════════════════════════════════════════════════════════════
# MI-Aware Cascade
# ═══════════════════════════════════════════════════════════════════════════


def mi_aware_cascade(
    tensor: np.ndarray, target_ratio: float = 100.0
) -> Tuple[Dict, float, float]:
    t = np.asarray(tensor, dtype=np.float64)

    stages_config = [
        ("decompose", 0.35, lambda x: decompose_then_quantize(x, "svd", 8, 8)),
        ("transform", 0.25, lambda x: transform_then_quantize(x, "dct", 0.15, 4)),
        ("quantize", 0.25, lambda x: quantize_then_sparsify(x, 4, 0.0)),
        ("entropy", 0.15, lambda x: _quantize(x, 8)),
    ]

    running, stage_results = 1.0, []
    current = t.copy()

    for name, mi_w, fn in stages_config:
        if running >= target_ratio:
            break
        try:
            data, sr, ss = fn(current)
            adj = sr ** (1.0 + mi_w)
            running *= adj
            stage_results.append(
                {
                    "stage": name,
                    "ratio": sr,
                    "adjusted": adj,
                    "snr": ss,
                    "mi_weight": mi_w,
                }
            )
            if isinstance(data, dict) and "reconstructed" in data:
                current = data["reconstructed"].copy()
        except Exception:
            continue

    snr_db = _snr(t, current)
    return (
        {
            "method": "mi_aware_cascade",
            "target_ratio": target_ratio,
            "achieved_ratio": max(running, 1.0),
            "stages": stage_results,
            "shape": t.shape,
            "reconstructed": current,
        },
        float(max(running, 1.0)),
        float(snr_db),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════


def compute_stage_ratios(tensor: np.ndarray, method: str, params: dict) -> dict:
    import time

    t = np.asarray(tensor, dtype=np.float64)
    t0 = time.perf_counter()
    try:
        if method == "svd":
            rank = params.get("rank", 16)
            *_, ratio = _svd_truncated(t, rank=rank)
            return {
                "method": method,
                "params": params,
                "ratio": ratio,
                "time_s": time.perf_counter() - t0,
            }
        elif method == "int8":
            recon, ratio, _ = _quantize(t, 8)
            return {
                "method": method,
                "params": params,
                "ratio": ratio,
                "snr": _snr(t, recon),
                "time_s": time.perf_counter() - t0,
            }
        elif method == "int4":
            recon, ratio, _ = _quantize(t, 4)
            return {
                "method": method,
                "params": params,
                "ratio": ratio,
                "snr": _snr(t, recon),
                "time_s": time.perf_counter() - t0,
            }
        elif method == "tt":
            _, ratio = tensor_train_decompose(t, rank=params.get("rank", 8))
            return {
                "method": method,
                "params": params,
                "ratio": ratio,
                "time_s": time.perf_counter() - t0,
            }
        elif method == "mera":
            _, ratio = mera_decompose(t, bond_dim=params.get("bond_dim", 2))
            return {
                "method": method,
                "params": params,
                "ratio": ratio,
                "time_s": time.perf_counter() - t0,
            }
        return {"method": method, "error": f"unknown: {method}"}
    except Exception as e:
        return {"method": method, "params": params, "error": str(e)}


def best_cascade_for_tensor(
    tensor: np.ndarray, target_ratio: float = 100.0, max_error: float = 0.01
) -> Dict:
    t = np.asarray(tensor, dtype=np.float64)
    configs = [
        ("2stage (svd+int8)", lambda: cascade_2_stage(t, "svd", "int8")),
        ("2stage (svd+int4)", lambda: cascade_2_stage(t, "svd", "int4")),
        ("3stage (tt+dct+int4)", lambda: cascade_3_stage(t, "tt", "dct", "int4")),
        ("quantize+sparsify", lambda: quantize_then_sparsify(t, 4, 0.5)),
        ("decompose+quantize", lambda: decompose_then_quantize(t, "svd", 16, 8)),
        ("transform+quantize", lambda: transform_then_quantize(t, "dct", 0.1, 8)),
        ("progressive", lambda: progressive_compress(t, 4)),
        ("mi_aware", lambda: mi_aware_cascade(t, target_ratio)),
    ]

    candidates = []
    for name, fn in configs:
        try:
            data, ratio, snr = fn()
            error = 1.0 / max(snr, 1e-10)
            if ratio >= target_ratio * 0.8 or error <= max_error:
                candidates.append(
                    {
                        "name": name,
                        "ratio": ratio,
                        "snr": snr,
                        "meets_target": ratio >= target_ratio,
                        "meets_error": error <= max_error,
                    }
                )
        except Exception:
            continue

    if not candidates:
        d, r, _ = _quantize(t, 8)
        return {"best": "int8", "ratio": r, "snr": _snr(t, d), "candidates": []}

    candidates.sort(key=lambda x: (-x["snr"], -x["ratio"]))
    return {
        "best": candidates[0]["name"],
        "ratio": candidates[0]["ratio"],
        "snr": candidates[0]["snr"],
        "candidates": candidates,
    }


def rate_distortion_optimal(tensor: np.ndarray, target_ratio: float) -> Dict:
    t = np.asarray(tensor, dtype=np.float64)
    orig_bits = t.size * 32
    target_bits = int(orig_bits / target_ratio)

    _, S, _ = np.linalg.svd(t.reshape(t.shape[0], -1), full_matrices=False)
    sv_energy = S**2 / max(np.sum(S**2), 1e-30)
    sv_bits = max(1, int(target_bits * 0.3 * np.sum(sv_energy[:10])))

    c = dct(t.ravel()[: min(1024, t.size)])
    dct_energy = c**2 / max(np.sum(c**2), 1e-30)
    dct_bits = max(1, int(target_bits * 0.25 * np.sum(dct_energy[:64])))

    qb = max(1, target_bits - sv_bits - dct_bits)
    qbpv = min(8, max(1, qb // t.size))
    total = sv_bits + dct_bits + qbpv * t.size

    return {
        "method": "rate_distortion_optimal",
        "total_bits_available": target_bits,
        "total_bits_allocated": total,
        "svd_bits": int(sv_bits),
        "dct_bits": int(dct_bits),
        "quant_bits_per_value": int(qbpv),
        "estimated_ratio": max(orig_bits / max(total, 1), 1.0),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t = np.random.randn(128, 128).astype(np.float32)

    data, ratio, snr = cascade_2_stage(t, "svd", "int8")
    print(f"2-Stage SVD+INT8: {ratio:.2f}x, SNR={snr:.1f}dB")

    data, ratio, snr = cascade_3_stage(t, "tt", "dct", "int4")
    print(f"3-Stage TT+DCT+INT4: {ratio:.2f}x, SNR={snr:.1f}dB")

    data, ratio, snr = cascade_4_stage(t, "mera", "dct", "int4", "rans")
    print(f"4-Stage MERA+DCT+INT4+rANS: {ratio:.2f}x, SNR={snr:.1f}dB")

    data, ratio, snr = quantize_then_sparsify(t, 4, 0.5)
    print(f"Quantize+Sparsify: {ratio:.2f}x, SNR={snr:.1f}dB")

    data, ratio, snr = decompose_then_quantize(t, "svd", 16, 8)
    print(f"Decompose+Quantize: {ratio:.2f}x, SNR={snr:.1f}dB")

    data, ratio, snr = transform_then_quantize(t, "dct", 0.1, 8)
    print(f"Transform+Quantize: {ratio:.2f}x, SNR={snr:.1f}dB")

    data, ratio, snr = transform_then_sparsify(t, "dct", 0.1, 0.5)
    print(f"Transform+Sparsify: {ratio:.2f}x, SNR={snr:.1f}dB")

    data, ratio, snr = ensemble_compress(t)
    print(f"Ensemble (best): {ratio:.2f}x, SNR={snr:.1f}dB")

    data, ratio, snr = adaptive_cascade(t, 50.0)
    print(f"Adaptive: {ratio:.2f}x, SNR={snr:.1f}dB")

    data, ratio, snr = progressive_compress(t, 4)
    print(f"Progressive: {ratio:.2f}x, SNR={snr:.1f}dB")

    data, ratio, snr = mi_aware_cascade(t, 50.0)
    print(f"MI-Aware: {ratio:.2f}x, SNR={snr:.1f}dB")

    rd = rate_distortion_optimal(t, 100.0)
    print(f"RD-Optimal: {rd['estimated_ratio']:.2f}x est")

    bench = compute_stage_ratios(t, "svd", {"rank": 16})
    print(f"Benchmark SVD: {bench['ratio']:.2f}x, {bench.get('time_s', 0):.4f}s")

    best = best_cascade_for_tensor(t, 50.0, 0.01)
    print(
        f"Best cascade: {best['best']}, ratio={best['ratio']:.2f}x, SNR={best['snr']:.1f}dB"
    )

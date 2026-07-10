"""
Honest measurement helpers for compression ratio and error reporting.

These utilities exist because several cascade engines in this codebase were
found to fabricate compression ratios (e.g. taking ``len(dict)`` — the
number of dict keys — instead of the actual serialized byte size, or
multiplying together per-stage "expected"/estimated ratios instead of
measuring real output size). All ratio and error numbers that get surfaced
to a user or written into a report MUST be derived from these functions
(or an equivalent byte-exact measurement), never from a product of
per-stage estimates or from len() on a non-bytes object.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, NamedTuple, Tuple

import numpy as np

BF16_BYTES_PER_ELEMENT = 2
FP32_BYTES_PER_ELEMENT = 4

ERROR_GATE_THRESHOLD = 0.05  # rel_mse; strict > gate; consistent with Phase-3 cascade acceptance


def serialized_nbytes(payload: Any) -> int:
    """Recursively compute the true number of bytes a payload occupies.

    Handles the shapes actually produced by this codebase's `compress()`
    methods: raw bytes, numpy arrays (codebooks/indices/scales), dicts of
    the above (nested), lists/tuples of the above, and JSON-serializable
    scalars/strings (whose on-disk cost is approximated by struct/utf-8
    encoding, not by Python object size).

    This must NEVER be approximated by ``len(payload)`` when ``payload``
    is a dict — ``len(dict)`` counts keys, not bytes, and silently produces
    fabricated ratios many orders of magnitude off.
    """
    if payload is None:
        return 0
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(payload)
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    if isinstance(payload, np.generic):
        return int(np.asarray(payload).nbytes)
    if isinstance(payload, dict):
        total = 0
        for k, v in payload.items():
            total += len(str(k).encode("utf-8"))
            total += serialized_nbytes(v)
        return total
    if isinstance(payload, (list, tuple)):
        return sum(serialized_nbytes(v) for v in payload)
    if isinstance(payload, bool):
        return 1
    if isinstance(payload, int):
        return 8
    if isinstance(payload, float):
        return 8
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    # Unknown scalar/metadata type: fall back to a conservative struct-like
    # count rather than pretending it's free.
    try:
        return len(repr(payload).encode("utf-8"))
    except Exception:
        return 8


def honest_ratio(original_bytes: int, payload: Any) -> float:
    """Compute original_bytes / serialized_nbytes(payload), byte-exact."""
    comp_bytes = max(serialized_nbytes(payload), 1)
    return float(original_bytes) / float(comp_bytes)


class ErrorMetrics(NamedTuple):
    rel_mse: float
    cosine_sim: float
    max_abs: float
    snr_db: float


def end_to_end_error(original: np.ndarray, reconstructed: np.ndarray) -> ErrorMetrics:
    """Compute multiple independent error metrics between original and
    reconstructed tensors — never reduce this to a single made-up number.

    Parameters
    ----------
    original : np.ndarray
        The ground-truth tensor (any dtype; cast to float64 for measurement).
    reconstructed : np.ndarray
        The fully reconstructed tensor (after summing/decoding ALL cascade
        stages), same shape as ``original``.

    Returns
    -------
    ErrorMetrics
        rel_mse    : mean squared error / variance of original (0 = perfect)
        cosine_sim : cosine similarity between flattened tensors (1 = perfect)
        max_abs    : maximum absolute elementwise error
        snr_db     : signal-to-noise ratio in dB (higher = better; inf if error is 0)
    """
    orig = np.asarray(original, dtype=np.float64).ravel()
    recon = np.asarray(reconstructed, dtype=np.float64).ravel()
    if recon.shape != orig.shape:
        recon = recon.reshape(orig.shape)

    diff = orig - recon
    mse = float(np.mean(diff * diff))
    var = float(np.var(orig))
    rel_mse = mse / var if var > 1e-30 else float(mse)

    orig_norm = float(np.linalg.norm(orig))
    recon_norm = float(np.linalg.norm(recon))
    if orig_norm > 1e-30 and recon_norm > 1e-30:
        cosine_sim = float(np.dot(orig, recon) / (orig_norm * recon_norm))
    else:
        cosine_sim = 1.0 if orig_norm < 1e-30 and recon_norm < 1e-30 else 0.0

    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0

    signal_power = float(np.mean(orig * orig))
    if mse > 1e-30 and signal_power > 1e-30:
        snr_db = 10.0 * float(np.log10(signal_power / mse))
    else:
        snr_db = float("inf") if mse <= 1e-30 else float("-inf")

    return ErrorMetrics(
        rel_mse=rel_mse, cosine_sim=cosine_sim, max_abs=max_abs, snr_db=snr_db
    )


def dual_ratio(original_elements: int, payload: Any) -> Dict[str, float]:
    """Report ratio vs both fp32 and bf16 baselines, so numbers are honest
    against the dtype the model actually ships as.

    ``original_elements`` is the element count of the original tensor
    (NOT the byte count — this function derives fp32/bf16 byte counts
    from the element count so both baselines are computed consistently).
    """
    comp_bytes = max(serialized_nbytes(payload), 1)
    fp32_bytes = original_elements * FP32_BYTES_PER_ELEMENT
    bf16_bytes = original_elements * BF16_BYTES_PER_ELEMENT
    return {
        "ratio_vs_fp32": float(fp32_bytes) / float(comp_bytes),
        "ratio_vs_bf16": float(bf16_bytes) / float(comp_bytes),
    }


def ratio_vs_fp32(original_elements: int, payload: Any) -> float:
    """Convenience wrapper: compression ratio measured against an fp32
    (4 bytes/element) baseline of the original tensor."""
    comp_bytes = max(serialized_nbytes(payload), 1)
    fp32_bytes = original_elements * FP32_BYTES_PER_ELEMENT
    return float(fp32_bytes) / float(comp_bytes)


def ratio_vs_bf16(original_elements: int, payload: Any) -> float:
    """Convenience wrapper: compression ratio measured against a bf16
    (2 bytes/element) baseline of the original tensor."""
    comp_bytes = max(serialized_nbytes(payload), 1)
    bf16_bytes = original_elements * BF16_BYTES_PER_ELEMENT
    return float(bf16_bytes) / float(comp_bytes)


def apply_gate(
    payload: Any,
    original_elements: int,
    rel_mse: float,
    threshold: float = ERROR_GATE_THRESHOLD,
) -> Dict[str, Any]:
    """Central chokepoint: couple every compression ratio to its reconstruction error.

    This is THE ONLY ratio-emission decision point. Both CLI blocks and future
    reporters must call it (not per-call-site thresholds). A high-error method
    is gated (marked, ratio suppressed); a good method keeps its byte-exact
    ratio derived from ``serialized_nbytes``.

    Boundary: at exactly ``rel_mse == threshold`` the gate is NOT triggered
    (strict ``>``), consistent with Phase-3 cascade acceptance.

    The gate never drops the error metrics (rel_mse/cosine_sim/max_abs/snr_db)
    — only suppresses the ratio claim when error is over threshold.

    Parameters
    ----------
    payload : Any
        The compressed payload (bytes, ndarray, dict, or mixed nested structure).
    original_elements : int
        Element count of the original tensor (not byte count — ``dual_ratio``
        derives fp32/bf16 byte counts from it).
    rel_mse : float
        Relative mean squared error from ``end_to_end_error``.
    threshold : float, optional
        Error threshold; defaults to ``ERROR_GATE_THRESHOLD``.

    Returns
    -------
    Dict[str, Any]
        ratio_vs_bf16   : float or None (None if gated)
        ratio_vs_fp32   : float or None (None if gated)
        rel_mse         : float (always retained)
        gated           : bool (True if rel_mse > threshold)
        gate_reason     : str ("" if not gated, otherwise human-readable reason)
    """
    ratios = dual_ratio(original_elements, payload)
    gated = bool(rel_mse > threshold)  # STRICT >, never >=
    gate_reason = (
        f"rel_mse {rel_mse:.4f} > {threshold}" if gated else ""
    )
    return {
        "ratio_vs_bf16": ratios["ratio_vs_bf16"] if not gated else None,
        "ratio_vs_fp32": ratios["ratio_vs_fp32"] if not gated else None,
        "rel_mse": float(rel_mse),
        "gated": gated,
        "gate_reason": gate_reason,
    }

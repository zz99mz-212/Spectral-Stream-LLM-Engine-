"""DEPRECATED — Validation and quality metrics module.

This module is deprecated and has been moved to _archive/core/validation.py.
Use spectralstream.core.math_primitives.metrics for individual metric functions
or the CompressionIntelligenceEngine's built-in validation for end-to-end checks.

The archived copy is preserved for reference but will be removed in a future version.
"""

from __future__ import annotations


import warnings as _warnings

_warnings.warn(
    "core.validation is deprecated. Use spectralstream.core.math_primitives.metrics "
    "for metric functions, or CompressionIntelligenceEngine.validate for validation.",
    DeprecationWarning,
    stacklevel=2,
)

import json
import math
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    dct as _dct,
    cosine_similarity as _cosine_similarity,
    effective_rank as _effective_rank_inner,
    compute_mse as _compute_mse,
    compute_snr as _compute_snr,
    compute_psnr as _compute_psnr,
    compute_relative_error as _compute_relative_error,
)


@dataclass
class CompressionQuality:
    """Complete quality assessment of a compressed tensor."""

    method: str
    compression_ratio: float
    mse: float
    snr_db: float
    psnr_db: float
    max_relative_error: float
    mean_relative_error: float
    spectral_angle: float
    histogram_overlap: float
    kld_divergence: float
    effective_rank_preserved: float
    passes_threshold: bool


def _sanitize(*arrays: np.ndarray) -> List[np.ndarray]:
    result = []
    for a in arrays:
        arr = np.asarray(a, dtype=np.float64).ravel()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        result.append(arr)
    return result


def compute_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig, recon = _sanitize(original, reconstructed)
    return _compute_snr(orig, recon)


def compute_psnr(
    original: np.ndarray, reconstructed: np.ndarray, peak: float = None
) -> float:
    orig, recon = _sanitize(original, reconstructed)
    if peak is not None and peak > 0:
        mse = float(np.mean((orig - recon) ** 2)) + 1e-30
        return 10.0 * math.log10(peak**2 / mse)
    return _compute_psnr(orig, recon)


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig, recon = _sanitize(original, reconstructed)
    return _compute_mse(orig, recon)


def compute_max_relative_error(
    original: np.ndarray, reconstructed: np.ndarray
) -> float:
    orig, recon = _sanitize(original, reconstructed)
    denom = np.abs(orig) + 1e-30
    rel_err = np.abs(orig - recon) / denom
    return float(np.max(rel_err))


def compute_mean_relative_error(
    original: np.ndarray, reconstructed: np.ndarray
) -> float:
    orig, recon = _sanitize(original, reconstructed)
    denom = np.abs(orig) + 1e-30
    rel_err = np.abs(orig - recon) / denom
    return float(np.mean(rel_err))


def compute_spectral_angle(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig, recon = _sanitize(original, reconstructed)
    dct_o = _dct(orig)
    dct_r = _dct(recon)
    sim = _cosine_similarity(dct_o, dct_r)
    sim = float(np.clip(sim, -1.0, 1.0))
    return float(math.acos(sim))


def compute_histogram_overlap(
    original: np.ndarray, reconstructed: np.ndarray, bins: int = 100
) -> float:
    orig, recon = _sanitize(original, reconstructed)
    combined = np.concatenate([orig, recon])
    lo, hi = float(np.min(combined)), float(np.max(combined))
    if hi - lo < 1e-30:
        return 1.0
    n = len(orig)
    actual_bins = min(bins, max(2, n // 2))
    h1, _ = np.histogram(orig, bins=actual_bins, range=(lo, hi), density=False)
    h2, _ = np.histogram(recon, bins=actual_bins, range=(lo, hi), density=False)
    intersection = float(np.sum(np.minimum(h1, h2)))
    denom = float(max(np.sum(h1), np.sum(h2)))
    if denom == 0:
        return 1.0
    return intersection / denom


def compute_kld(
    original: np.ndarray, reconstructed: np.ndarray, bins: int = 100
) -> float:
    orig, recon = _sanitize(original, reconstructed)
    combined = np.concatenate([orig, recon])
    lo, hi = float(np.min(combined)), float(np.max(combined))
    if hi - lo < 1e-30:
        return 0.0
    n = len(orig)
    actual_bins = min(bins, max(2, n // 2))
    h_p, _ = np.histogram(orig, bins=actual_bins, range=(lo, hi), density=True)
    h_q, _ = np.histogram(recon, bins=actual_bins, range=(lo, hi), density=True)
    eps = 1e-30
    p = np.clip(h_p, eps, None)
    q = np.clip(h_q, eps, None)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def compute_effective_rank(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig_rank = _effective_rank_inner(original)
    recon_rank = _effective_rank_inner(reconstructed)
    if orig_rank <= 0:
        return 1.0
    return float(min(recon_rank / orig_rank, 1.0))


class CompressionValidator:
    """DEPRECATED — Validates that compression meets targets.

    Use CompressionIntelligenceEngine.validate() instead.
    """

    def __init__(
        self,
        target_min_ratio: float = 500.0,
        target_max_error: float = 0.01,
        target_throughput: float = 2000.0,
    ):
        self.target_min_ratio = target_min_ratio
        self.target_max_error = target_max_error
        self.target_throughput = target_throughput

    def validate_tensor(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        method_name: str,
        compressed_size: int = None,
    ) -> CompressionQuality:
        orig = np.asarray(original, dtype=np.float64)
        recon = np.asarray(reconstructed, dtype=np.float64)
        mse = compute_mse(orig, recon)
        if compressed_size is not None:
            ratio = orig.nbytes / max(compressed_size, 1)
        else:
            ratio = 1.0
        quality = CompressionQuality(
            method=method_name,
            compression_ratio=ratio,
            mse=mse,
            snr_db=compute_snr(orig, recon),
            psnr_db=compute_psnr(orig, recon),
            max_relative_error=compute_max_relative_error(orig, recon),
            mean_relative_error=compute_mean_relative_error(orig, recon),
            spectral_angle=compute_spectral_angle(orig, recon),
            histogram_overlap=compute_histogram_overlap(orig, recon),
            kld_divergence=compute_kld(orig, recon),
            effective_rank_preserved=compute_effective_rank(orig, recon),
            passes_threshold=False,
        )
        quality.passes_threshold = self.assert_passes(quality)
        return quality

    def validate_batch(
        self, tensors: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, CompressionQuality]:
        results = {}
        for name, (orig, recon) in tensors.items():
            results[name] = self.validate_tensor(orig, recon, name)
        return results

    def assert_passes(
        self,
        quality: CompressionQuality,
        target_error: float = None,
    ) -> bool:
        if target_error is None:
            target_error = self.target_max_error
        ratio_ok = quality.compression_ratio >= self.target_min_ratio
        error_ok = quality.max_relative_error <= target_error
        return bool(ratio_ok and error_ok)

    def summary_report(self, qualities: Dict[str, CompressionQuality]) -> str:
        lines = ["Compression Validation Report", "=" * 50]
        passes = sum(1 for q in qualities.values() if q.passes_threshold)
        total = len(qualities)
        lines.append(f"Passed: {passes}/{total}")
        lines.append("")
        for name, q in qualities.items():
            status = "PASS" if q.passes_threshold else "FAIL"
            lines.append(
                f"[{status}] {name}: ratio={q.compression_ratio:.1f}x, "
                f"SNR={q.snr_db:.1f}dB, max_rel_err={q.max_relative_error:.6f}"
            )
        return "\n".join(lines)


class PerplexityValidator:
    """DEPRECATED — Validates model quality by measuring perplexity.

    Use CompressionIntelligenceEngine.validate() instead.
    """

    def __init__(self, reference_model=None, tokenizer=None):
        self.reference_model = reference_model
        self.tokenizer = tokenizer

    def compute_perplexity(self, text: str, max_length: int = 2048) -> float:
        if self.reference_model is None or self.tokenizer is None:
            return self._proxy_perplexity(text)
        try:
            import torch

            enc = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
            input_ids = enc["input_ids"]
            with torch.no_grad():
                outputs = self.reference_model(input_ids)
                logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return float(torch.exp(loss).item())
        except Exception:
            return self._proxy_perplexity(text)

    def _proxy_perplexity(self, text: str, ngram_n: int = 3) -> float:
        from collections import Counter

        words = text.lower().split()
        if not words:
            return 100.0
        if len(words) < ngram_n:
            return float(max(len(set(words)), 1))
        ngrams = Counter()
        for i in range(len(words) - ngram_n + 1):
            ngrams[tuple(words[i : i + ngram_n])] += 1
        total_ngrams = sum(ngrams.values())
        if total_ngrams == 0:
            return 100.0
        log_prob = 0.0
        count = 0
        for i in range(len(words) - ngram_n):
            context = tuple(words[i : i + ngram_n - 1])
            target = words[i + ngram_n - 1]
            context_ngrams = sum(v for k, v in ngrams.items() if k[:-1] == context)
            target_count = ngrams.get(tuple(list(context) + [target]), 0)
            if context_ngrams > 0 and target_count > 0:
                log_prob += math.log(target_count / context_ngrams)
                count += 1
        if count == 0:
            return 100.0
        avg_log_prob = log_prob / count
        perplexity = math.exp(-avg_log_prob)
        return float(min(max(perplexity, 1.0), 100000.0))

    def validate_compressed(
        self,
        compressed_model,
        calibration_texts: List[str],
    ) -> Dict[str, Any]:
        ref_ppls = []
        comp_ppls = []
        for text in calibration_texts:
            ref_ppls.append(self.compute_perplexity(text))
            try:
                old_model = self.reference_model
                self.reference_model = compressed_model
                comp_ppls.append(self.compute_perplexity(text))
                self.reference_model = old_model
            except Exception:
                comp_ppls.append(float("nan"))
        ref_avg = float(np.nanmean(ref_ppls)) if ref_ppls else float("nan")
        comp_avg = float(np.nanmean(comp_ppls)) if comp_ppls else float("nan")
        if ref_avg > 0 and not math.isnan(comp_avg):
            increase = (comp_avg - ref_avg) / ref_avg * 100
        else:
            increase = float("nan")
        return {
            "reference_ppl": ref_avg,
            "compressed_ppl": comp_avg,
            "ppl_increase_pct": increase,
            "calibration_samples": len(calibration_texts),
        }

    def validate_compressed_tensor(
        self,
        original_tensor: np.ndarray,
        compressed_data: Any,
        layer_name: str,
        calibration_hidden_states: np.ndarray = None,
    ) -> Dict[str, float]:
        orig = np.asarray(original_tensor, dtype=np.float64)
        decomp = np.asarray(compressed_data, dtype=np.float64)
        if orig.shape != decomp.shape:
            if decomp.size == orig.size:
                decomp = decomp.reshape(orig.shape)
            else:
                return {"error": 1.0, "snr_db": 0.0}
        result = {
            "mse": compute_mse(orig, decomp),
            "snr_db": compute_snr(orig, decomp),
            "max_relative_error": compute_max_relative_error(orig, decomp),
        }
        if calibration_hidden_states is not None:
            cal = np.asarray(calibration_hidden_states, dtype=np.float64)
            if cal.ndim >= 2:
                mul = lambda a, b: (a @ b.T) if b.ndim == 2 else (a @ b)
                out_ref = mul(cal, orig)
                out_comp = mul(cal, decomp)
                result["hidden_state_mse"] = float(np.mean((out_ref - out_comp) ** 2))
                result["hidden_state_snr"] = compute_snr(out_ref, out_comp)
            else:
                result["hidden_state_mse"] = float("nan")
                result["hidden_state_snr"] = float("nan")
        return result


class BenchmarkRunner:
    """DEPRECATED — Runs compression benchmarks and generates reports."""

    def __init__(self, validator: CompressionValidator = None):
        self.validator = validator or CompressionValidator()

    def run_method_comparison(
        self,
        tensor: np.ndarray,
        methods: Dict[str, Callable[[np.ndarray], np.ndarray]],
    ) -> Dict[str, CompressionQuality]:
        results = {}
        for name, reconstruct_fn in methods.items():
            reconstructed = np.asarray(reconstruct_fn(tensor), dtype=np.float64)
            results[name] = self.validator.validate_tensor(
                tensor,
                reconstructed,
                name,
            )
        return results

    def run_ratio_sweep(
        self,
        tensor: np.ndarray,
        method: str,
        ratios: List[float],
        compressor: Callable[[np.ndarray, float], np.ndarray],
    ) -> Dict[float, CompressionQuality]:
        results = {}
        for ratio in ratios:
            reconstructed = np.asarray(compressor(tensor, ratio), dtype=np.float64)
            quality = self.validator.validate_tensor(
                tensor,
                reconstructed,
                f"{method}_r{ratio:.0f}",
            )
            results[ratio] = quality
        return results

    def run_rate_distortion(
        self,
        tensor: np.ndarray,
        methods: Dict[str, List[Tuple[float, Callable]]],
    ) -> Dict[str, List[Dict]]:
        results = {}
        for name, rate_fns in methods.items():
            points = []
            for rate, reconstruct_fn in rate_fns:
                reconstructed = np.asarray(reconstruct_fn(tensor), dtype=np.float64)
                mse = compute_mse(tensor, reconstructed)
                snr = compute_snr(tensor, reconstructed)
                points.append(
                    {
                        "rate": rate,
                        "mse": mse,
                        "snr_db": snr,
                        "compression_ratio": rate,
                    }
                )
            results[name] = points
        return results

    def generate_report(self, results: Any) -> str:
        if isinstance(results, dict) and all(
            isinstance(v, CompressionQuality) for v in results.values()
        ):
            return self.validator.summary_report(results)
        return json.dumps(results, indent=2, default=str)

    def format_markdown_table(self, results: Dict[str, CompressionQuality]) -> str:
        lines = [
            "| Method | Ratio | MSE | SNR (dB) | PSNR (dB) | MaxRelErr | SpecAngle | HistOverlap | KLD | EffRank |",
            "|--------|-------|-----|----------|-----------|-----------|-----------|-------------|-----|---------|",
        ]
        for name, q in results.items():
            lines.append(
                f"| {name} | {q.compression_ratio:.1f}x "
                f"| {q.mse:.2e} | {q.snr_db:.1f} | {q.psnr_db:.1f} "
                f"| {q.max_relative_error:.2e} | {q.spectral_angle:.4f} "
                f"| {q.histogram_overlap:.4f} | {q.kld_divergence:.4f} "
                f"| {q.effective_rank_preserved:.4f} |"
            )
        return "\n".join(lines)

    def save_json(self, results: Any, path: str):
        if isinstance(results, dict) and all(
            isinstance(v, CompressionQuality) for v in results.values()
        ):
            data = {k: asdict(v) for k, v in results.items()}
        elif isinstance(results, CompressionQuality):
            data = asdict(results)
        else:
            data = results
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def validate_lossless(original: np.ndarray, decompressed: np.ndarray) -> bool:
    if original.dtype != decompressed.dtype:
        return False
    if original.shape != decompressed.shape:
        return False
    if original.nbytes != decompressed.nbytes:
        return False
    return bool(np.array_equal(original, decompressed))


def kolmogorov_smirnov_test(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig, recon = _sanitize(original, reconstructed)
    if len(orig) < 2 or len(recon) < 2:
        return 1.0
    combined = np.concatenate([orig, recon])
    combined.sort()
    n1, n2 = len(orig), len(recon)
    cdf1 = np.searchsorted(orig, combined, side="right") / n1
    cdf2 = np.searchsorted(recon, combined, side="right") / n2
    d_stat = float(np.max(np.abs(cdf1 - cdf2)))
    ne = n1 * n2 / (n1 + n2)
    try:
        p_value = 2.0 * math.exp(-2.0 * d_stat**2 * ne)
    except (OverflowError, ValueError):
        p_value = 0.0
    return float(min(max(p_value, 0.0), 1.0))


def wasserstein_distance(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig, recon = _sanitize(original, reconstructed)
    orig.sort()
    recon.sort()
    n = min(len(orig), len(recon))
    if n < 2:
        return 0.0
    orig_resampled = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(orig)),
        orig,
    )
    recon_resampled = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(recon)),
        recon,
    )
    return float(np.mean(np.abs(orig_resampled - recon_resampled)))


def correlation_coefficient(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig, recon = _sanitize(original, reconstructed)
    if len(orig) < 2:
        return 1.0
    o_mean = orig - np.mean(orig)
    r_mean = recon - np.mean(recon)
    num = float(np.dot(o_mean, r_mean))
    denom = float(np.linalg.norm(o_mean) * np.linalg.norm(r_mean))
    if denom < 1e-30:
        return 0.0
    return float(np.clip(num / denom, -1.0, 1.0))


__all__ = [
    "CompressionQuality",
    "compute_snr",
    "compute_psnr",
    "compute_mse",
    "compute_max_relative_error",
    "compute_mean_relative_error",
    "compute_spectral_angle",
    "compute_histogram_overlap",
    "compute_kld",
    "compute_effective_rank",
    "CompressionValidator",
    "PerplexityValidator",
    "BenchmarkRunner",
    "validate_lossless",
    "kolmogorov_smirnov_test",
    "wasserstein_distance",
    "correlation_coefficient",
]

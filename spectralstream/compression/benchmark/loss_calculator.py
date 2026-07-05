from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TensorLossMetrics:
    mse: float = 0.0
    mae: float = 0.0
    snr_db: float = 0.0
    psnr_db: float = 0.0
    relative_error: float = 0.0
    cosine_similarity: float = 0.0
    max_abs_error: float = 0.0
    kl_divergence: float = 0.0
    wasserstein_distance: float = 0.0
    ks_statistic: float = 0.0
    ks_p_value: float = 0.0
    anderson_darling_statistic: float = 0.0
    correlation_coefficient: float = 0.0
    histogram_overlap: float = 0.0
    effective_rank_ratio: float = 0.0
    bit_error_rate: float = 0.0
    bit_precision_achieved: float = 0.0

    outlier_preservation_ratio: float = 0.0
    outlier_false_positive_rate: float = 0.0
    outlier_false_negative_rate: float = 0.0

    num_elements: int = 0
    original_bytes: int = 0
    compressed_bytes: int = 0
    compression_ratio: float = 1.0


@dataclass
class LayerLossMetrics:
    layer_name: str = ""
    tensor_count: int = 0
    weighted_mse: float = 0.0
    weighted_relative_error: float = 0.0
    avg_snr_db: float = 0.0
    tensor_metrics: Dict[str, TensorLossMetrics] = field(default_factory=dict)


@dataclass
class ModelLossMetrics:
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    overall_ratio: float = 1.0
    weighted_mse: float = 0.0
    weighted_relative_error: float = 0.0
    avg_snr_db: float = 0.0
    avg_psnr_db: float = 0.0
    avg_cosine_similarity: float = 0.0
    avg_bit_precision: float = 0.0
    tensor_count: int = 0
    layer_count: int = 0
    tensor_metrics: Dict[str, TensorLossMetrics] = field(default_factory=dict)
    layer_metrics: Dict[str, LayerLossMetrics] = field(default_factory=dict)
    estimated_perplexity_impact: float = 0.0


def _sanitize(*arrays: np.ndarray) -> List[np.ndarray]:
    result = []
    for a in arrays:
        arr = np.asarray(a, dtype=np.float64).ravel()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        result.append(arr)
    if len(result) > 1:
        min_len = min(len(r) for r in result)
        result = [r[:min_len] for r in result]
    return result


def _match_shape(o: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(o.size, r.size)
    return o.ravel()[:n], r.ravel()[:n]


def _extract_layer(name: str) -> str:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                int(parts[i + 1])
                return ".".join(parts[: i + 2])
            except ValueError:
                continue
    return name


class LossCalculator:
    def __init__(self, outlier_std_threshold: float = 3.0):
        self._outlier_std_threshold = outlier_std_threshold

    def compute_tensor_metrics(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        original_bytes: int = 0,
        compressed_bytes: int = 0,
    ) -> TensorLossMetrics:
        o, r = _match_shape(original, reconstructed)
        o_f64 = o.astype(np.float64)
        r_f64 = r.astype(np.float64)
        diff = o_f64 - r_f64
        n = len(o_f64)

        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))
        var_o = float(np.var(o_f64)) + 1e-30
        signal_power = float(np.mean(o_f64**2))
        noise_power = mse + 1e-30

        snr_db = (
            10.0 * math.log10(signal_power / noise_power)
            if noise_power > 0
            else float("inf")
        )
        max_val = float(np.max(np.abs(o_f64)))
        psnr_db = (
            10.0 * math.log10(max_val**2 / mse)
            if mse > 0 and max_val > 1e-30
            else float("inf")
        )
        relative_error = math.sqrt(mse / var_o) if var_o > 0 else math.sqrt(mse)

        o_norm = float(np.linalg.norm(o_f64))
        r_norm = float(np.linalg.norm(r_f64))
        dot = float(np.dot(o_f64, r_f64))
        cos_sim = (
            float(np.clip(dot / (o_norm * r_norm + 1e-30), -1.0, 1.0))
            if o_norm > 1e-30 and r_norm > 1e-30
            else 1.0
        )
        max_abs = float(np.max(np.abs(diff)))

        combined = np.concatenate([o_f64, r_f64])
        lo, hi = float(np.min(combined)), float(np.max(combined))
        actual_bins = min(100, max(2, n // 100))

        if hi - lo > 1e-30 and actual_bins >= 2:
            h_p, edges = np.histogram(
                o_f64, bins=actual_bins, range=(lo, hi), density=True
            )
            h_q, _ = np.histogram(r_f64, bins=actual_bins, range=(lo, hi), density=True)
            eps = 1e-10
            p = np.clip(h_p, eps, None)
            q = np.clip(h_q, eps, None)
            p = p / np.sum(p)
            q = q / np.sum(q)
            kl_div = float(np.sum(p * np.log(p / q)))
            hi_int = float(
                np.sum(
                    np.minimum(
                        np.histogram(
                            o_f64, bins=actual_bins, range=(lo, hi), density=False
                        )[0],
                        np.histogram(
                            r_f64, bins=actual_bins, range=(lo, hi), density=False
                        )[0],
                    )
                )
            )
            total_hist = float(
                max(
                    np.sum(
                        np.histogram(
                            o_f64, bins=actual_bins, range=(lo, hi), density=False
                        )[0]
                    ),
                    1,
                )
            )
            hist_overlap = hi_int / total_hist
            wass_dist = float(np.mean(np.abs(np.sort(o_f64) - np.sort(r_f64))))
        else:
            kl_div = 0.0
            hist_overlap = 1.0
            wass_dist = 0.0

        o_sorted = np.sort(o_f64)
        r_sorted = np.sort(r_f64)
        ks_stat = float(
            np.max(
                np.abs(
                    np.searchsorted(o_sorted, r_sorted, side="right") / n
                    - np.searchsorted(r_sorted, o_sorted, side="right") / n
                )
            )
        )
        n_eff = n
        ks_p = 2.0 * math.exp(-2.0 * (ks_stat**2) * n_eff) if n_eff > 0 else 1.0

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from scipy import stats as scipy_stats

                ad_result = scipy_stats.anderson_ksamp([o_f64, r_f64])
                ad_stat = float(ad_result.statistic)
        except Exception:
            ad_stat = 0.0

        corr = float(np.corrcoef(o_f64, r_f64)[0, 1]) if n > 1 else 0.0

        U_o, S_o, _ = (
            np.linalg.svd(
                o_f64.reshape(-1, 1)
                if o_f64.ndim == 1
                else o_f64[: min(n, 10000)].reshape(-1, min(100, int(math.isqrt(n)))),
                full_matrices=False,
            )
            if n > 1
            else (None, np.array([1.0]), None)
        )
        if S_o is not None and len(S_o) > 1:
            S_o = S_o[: min(len(S_o), 100)]
            S_norm = S_o / (S_o[0] + 1e-30)
            energy = float(np.sum(S_norm**2) / len(S_norm))
            eff_rank = float(np.sum(S_norm > 0.01) / len(S_norm))
        else:
            energy = 1.0
            eff_rank = 1.0

        bit_error = float(
            np.mean(
                np.abs((o_f64 > 0).astype(np.float64) - (r_f64 > 0).astype(np.float64))
            )
        )

        target_precision = 32.0
        mse_bits = -math.log2(mse + 1e-30) / 2.0 if mse > 0 else target_precision
        bit_precision = min(target_precision, max(0.0, mse_bits))

        outlier_mask_o = np.abs(
            o_f64 - float(np.mean(o_f64))
        ) > self._outlier_std_threshold * float(np.std(o_f64))
        outlier_mask_r = np.abs(
            r_f64 - float(np.mean(r_f64))
        ) > self._outlier_std_threshold * float(np.std(r_f64))
        n_out_o = int(np.sum(outlier_mask_o))
        if n_out_o > 0:
            preserved = int(np.sum(outlier_mask_o & outlier_mask_r))
            precision = preserved / n_out_o
            false_neg = (
                (
                    n_out_o
                    - int(
                        np.sum(
                            outlier_mask_o & outlier_mask_r[:, np.newaxis]
                            if r_f64.ndim > 1
                            else outlier_mask_o
                        )
                    )
                )
                / n_out_o
                if False
                else int(np.sum(outlier_mask_o & ~outlier_mask_r)) / n_out_o
            )
            false_neg_rate = int(np.sum(outlier_mask_o & ~outlier_mask_r)) / n_out_o
            n_pred = int(np.sum(outlier_mask_r))
            false_pos_rate = (
                int(np.sum(~outlier_mask_o & outlier_mask_r)) / max(n_pred, 1)
                if n_pred > 0
                else 0.0
            )
        else:
            precision = 1.0
            false_neg_rate = 0.0
            false_pos_rate = 0.0

        ratio = (
            original_bytes / max(compressed_bytes, 1) if compressed_bytes > 0 else 1.0
        )

        return TensorLossMetrics(
            mse=mse,
            mae=mae,
            snr_db=snr_db,
            psnr_db=psnr_db,
            relative_error=relative_error,
            cosine_similarity=cos_sim,
            max_abs_error=max_abs,
            kl_divergence=kl_div,
            wasserstein_distance=wass_dist,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            anderson_darling_statistic=ad_stat,
            correlation_coefficient=corr,
            histogram_overlap=hist_overlap,
            effective_rank_ratio=eff_rank,
            bit_error_rate=bit_error,
            bit_precision_achieved=bit_precision,
            outlier_preservation_ratio=precision,
            outlier_false_positive_rate=false_pos_rate,
            outlier_false_negative_rate=false_neg_rate,
            num_elements=n,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=ratio,
        )

    def compute_layer_metrics(
        self, tensors: Dict[str, Tuple[np.ndarray, np.ndarray, int, int]]
    ) -> LayerLossMetrics:
        total_weight = 0.0
        weighted_mse = 0.0
        weighted_rel = 0.0
        snrs: List[float] = []
        tensor_metrics: Dict[str, TensorLossMetrics] = {}

        for name, (orig, recon, orig_bytes, comp_bytes) in tensors.items():
            tm = self.compute_tensor_metrics(orig, recon, orig_bytes, comp_bytes)
            tensor_metrics[name] = tm
            w = float(orig.size)
            total_weight += w
            weighted_mse += tm.mse * w
            weighted_rel += tm.relative_error * w
            if tm.snr_db != float("inf"):
                snrs.append(tm.snr_db)

        layer_name = _extract_layer(list(tensors.keys())[0]) if tensors else ""

        return LayerLossMetrics(
            layer_name=layer_name,
            tensor_count=len(tensors),
            weighted_mse=weighted_mse / max(total_weight, 1.0),
            weighted_relative_error=weighted_rel / max(total_weight, 1.0),
            avg_snr_db=float(np.mean(snrs)) if snrs else 0.0,
            tensor_metrics=tensor_metrics,
        )

    def compute_model_metrics(
        self,
        tensor_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        compressed_sizes: Optional[Dict[str, int]] = None,
    ) -> ModelLossMetrics:
        total_w = 0.0
        w_mse = 0.0
        w_rel = 0.0
        snrs: List[float] = []
        psnrs: List[float] = []
        coss: List[float] = []
        precs: List[float] = []
        tensor_count = 0
        total_orig = 0
        total_comp = 0

        tensor_metrics: Dict[str, TensorLossMetrics] = {}
        layers: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, int, int]]] = {}

        for name, (orig, recon) in tensor_pairs.items():
            orig_b = orig.nbytes
            comp_b = compressed_sizes.get(name, 0) if compressed_sizes else 0
            tm = self.compute_tensor_metrics(orig, recon, orig_b, comp_b)
            tensor_metrics[name] = tm
            tensor_count += 1
            total_orig += orig_b
            total_comp += comp_b

            w = float(orig.size)
            total_w += w
            w_mse += tm.mse * w
            w_rel += tm.relative_error * w
            if tm.snr_db != float("inf"):
                snrs.append(tm.snr_db)
            if tm.psnr_db != float("inf"):
                psnrs.append(tm.psnr_db)
            coss.append(tm.cosine_similarity)
            precs.append(tm.bit_precision_achieved)

            layer_name = _extract_layer(name)
            layers.setdefault(layer_name, {})[name] = (orig, recon, orig_b, comp_b)

        overall_ratio = total_orig / max(total_comp, 1)
        weighted_mse = w_mse / max(total_w, 1.0)
        weighted_rel = w_rel / max(total_w, 1.0)

        perp_impact = self._estimate_perplexity_impact(weighted_rel)

        layer_metrics: Dict[str, LayerLossMetrics] = {}
        for lname, l_tensors in layers.items():
            layer_metrics[lname] = self.compute_layer_metrics(l_tensors)

        return ModelLossMetrics(
            total_original_bytes=total_orig,
            total_compressed_bytes=total_comp,
            overall_ratio=overall_ratio,
            weighted_mse=weighted_mse,
            weighted_relative_error=weighted_rel,
            avg_snr_db=float(np.mean(snrs)) if snrs else 0.0,
            avg_psnr_db=float(np.mean(psnrs)) if psnrs else 0.0,
            avg_cosine_similarity=float(np.mean(coss)),
            avg_bit_precision=float(np.mean(precs)),
            tensor_count=tensor_count,
            layer_count=len(layer_metrics),
            tensor_metrics=tensor_metrics,
            layer_metrics=layer_metrics,
            estimated_perplexity_impact=perp_impact,
        )

    def _estimate_perplexity_impact(self, weighted_rel: float) -> float:
        if weighted_rel < 1e-6:
            return 0.0
        return min(100.0, 10.0 * math.sqrt(weighted_rel) + 50.0 * weighted_rel)

    def compare_methods(
        self,
        results: Dict[str, TensorLossMetrics],
    ) -> Dict[str, Any]:
        if not results:
            return {"rankings": [], "best_method": None, "worst_method": None}

        scored = []
        for method, m in results.items():
            score = self._composite_score(m)
            scored.append((score, method, m))
        scored.sort(key=lambda x: -x[0])

        return {
            "rankings": [
                {
                    "method": m,
                    "score": s,
                    "mse": met.mse,
                    "snr": met.snr_db,
                    "rel_error": met.relative_error,
                    "cosine": met.cosine_similarity,
                    "ratio": met.compression_ratio,
                }
                for s, m, met in scored
            ],
            "best_method": scored[0][1] if scored else None,
            "worst_method": scored[-1][1] if scored else None,
            "score_range": (scored[-1][0], scored[0][0]) if scored else (0, 0),
        }

    def _composite_score(self, m: TensorLossMetrics) -> float:
        rel = max(0.0, 1.0 - min(m.relative_error * 10, 10.0))
        snr = min(1.0, max(0.0, m.snr_db / 60.0)) if m.snr_db != float("inf") else 1.0
        cos = max(0.0, (m.cosine_similarity + 1.0) / 2.0)
        hist = min(1.0, max(0.0, m.histogram_overlap))
        corr = max(0.0, (m.correlation_coefficient + 1.0) / 2.0)
        rank = min(1.0, max(0.0, m.effective_rank_ratio))
        ratio_factor = min(1.0, m.compression_ratio / 5000.0)
        outlier = max(0.0, m.outlier_preservation_ratio)
        bits = m.bit_precision_achieved / 32.0

        return float(
            rel * 0.12
            + snr * 0.10
            + cos * 0.10
            + hist * 0.08
            + corr * 0.08
            + rank * 0.06
            + ratio_factor * 0.20
            + outlier * 0.12
            + bits * 0.14
        )

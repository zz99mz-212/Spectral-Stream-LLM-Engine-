"""Helper functions for tensor profiling and analysis."""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._constants import MAX_PROFILE_SAMPLES

# ── Bypass decision constants (mirrored from world_model.method_oracle) ────
BYPASS_HIGH_CONFIDENCE = "bypass_high_confidence"
BYPASS_MEDIUM_CONFIDENCE = "bypass_medium_confidence"
TEST_FULL = "test_full"


def _classify_by_name(name: str) -> str:
    if not name:
        return "weight"
    nl = name.lower()
    if any(k in nl for k in ("embed", "tok_embeddings", "wte")):
        return "embedding"
    if any(k in nl for k in ("attn_q", "q_proj", "wq", "query")):
        return "attention_q"
    if any(k in nl for k in ("attn_k", "k_proj", "wk", "key")):
        return "attention_k"
    if any(k in nl for k in ("attn_v", "v_proj", "wv", "value")):
        return "attention_v"
    if any(k in nl for k in ("attn_o", "o_proj", "wo", "out")):
        return "attention_o"
    if any(k in nl for k in ("qkv",)):
        return "qkv_fused"
    if any(k in nl for k in ("ffn_gate", "gate_proj", "w1", "fc_gate")):
        return "ffn_gate"
    if any(k in nl for k in ("ffn_up", "up_proj", "w3")):
        return "ffn_up"
    if any(k in nl for k in ("ffn_down", "down_proj", "w2")):
        return "ffn_down"
    if any(k in nl for k in ("ffn", "mlp", "expert")):
        return "ffn_gate"
    if any(k in nl for k in ("norm", "ln_", "rms")):
        return "norm"
    if any(k in nl for k in ("output", "lm_head", "head")):
        return "output"
    return "weight"


def _classify_by_name_simple(name: str) -> str:
    if not name:
        return "weight"
    nl = name.lower()
    if any(k in nl for k in ("embed", "tok_embeddings", "wte")):
        return "embedding"
    if any(k in nl for k in ("attn", "q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if "qkv" in nl:
        return "qkv_fused"
    if any(k in nl for k in ("ffn", "gate", "up_proj", "down_proj", "mlp")):
        return "ffn"
    if any(k in nl for k in ("norm", "ln_", "rms")):
        return "norm_bias"
    return "weight"


def _safe_bytes(data: Any) -> int:
    if isinstance(data, np.ndarray):
        return data.nbytes
    if isinstance(data, dict):
        return sum(_safe_bytes(v) for v in data.values())
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    return len(data) if isinstance(data, (bytes, bytearray)) else 8


def _compute_metrics(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    """Compute quality metrics using the authoritative QualityAssessor.

    Returns a dict with ALL metrics (mse, rmse, mae, nmse, snr_db, psnr_db,
    relative_error, cosine_similarity, max_abs_error, ssim, spectral_angle,
    histogram_overlap, kld_divergence, wasserstein_distance, ks_statistic,
    ks_p_value, correlation_coefficient, effective_rank_ratio, bit_error_rate).

    Backward compatible: retains all original keys.
    """
    from spectralstream.core.math_primitives.quality import QualityAssessor

    qa = QualityAssessor()
    quality = qa.assess(orig, recon)
    return quality.to_dict()


def _compute_ratio(original_nbytes: int, compressed_data: bytes) -> float:
    return original_nbytes / max(len(compressed_data), 1)


def _sample_flat(
    tensor: np.ndarray, max_samples: int = MAX_PROFILE_SAMPLES
) -> np.ndarray:
    flat = tensor.ravel()
    if flat.size <= max_samples:
        return flat.astype(np.float32)
    idx = np.random.choice(flat.size, max_samples, replace=False)
    result = flat[idx].astype(np.float32)
    return result


def _grade_error(err: float) -> str:
    if err < 0.0002:
        return "S"
    if err < 0.001:
        return "A"
    if err < 0.005:
        return "B"
    if err < 0.01:
        return "C"
    if err < 0.05:
        return "D"
    return "F"


def _metrics_summary(quality: "CompressionQuality") -> str:
    """Human-readable one-line summary from a CompressionQuality object.

    Parameters
    ----------
    quality : CompressionQuality
        Fully populated quality assessment.

    Returns
    -------
    str
        Compact summary: ``Grade S  Score 0.9876  MSE 1.23e-05  SNR 48.2 dB  RelErr 0.02%``
    """
    try:
        grade = quality.grade()
        score = quality.composite_score()
        return (
            f"Grade {grade}  Score {score:.4f}  "
            f"MSE {quality.mse:.6e}  SNR {quality.snr_db:.2f} dB  "
            f"RelErr {quality.relative_error * 100:.4f}%  "
            f"SSIM {quality.ssim:.4f}  CosSim {quality.cosine_similarity:.4f}"
        )
    except Exception:
        return "Grade ?  Score 0.0  MSE N/A  SNR N/A"


def _bootstrap_error(
    errors: np.ndarray, n_resamples: int = 1000
) -> Tuple[float, float]:
    if len(errors) < 2:
        return float(np.mean(errors)), 0.0
    means = np.zeros(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = np.random.choice(len(errors), len(errors), replace=True)
        means[i] = np.mean(errors[idx])
    return float(np.mean(means)), float(np.std(means))


def _estimate_noise_floor(tensor: np.ndarray, n_bins: int = 100) -> float:
    flat = tensor.ravel()
    if flat.size < 16:
        return 0.0
    sample = flat[: min(len(flat), 10000)]
    try:
        hist, _ = np.histogram(sample.astype(np.float64), bins=n_bins)
        hist = hist.astype(np.float64)
        hist /= max(hist.sum(), 1e-30)
        entropy = -np.sum(hist * np.log2(hist + 1e-30))
        max_entropy = np.log2(n_bins)
        del hist, sample
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
    except Exception:
        return 0.0


def _estimate_entropy_rate(flat: np.ndarray, order: int = 1) -> float:
    if flat.size < order + 10:
        return 0.0
    try:
        sample = flat[: min(len(flat), 5000)]
        percentiles = np.percentile(sample, np.linspace(0, 100, 17))
        quantized = np.digitize(sample, percentiles) - 1
        del percentiles, sample
        n_states = 16
        trans = np.zeros((n_states, n_states), dtype=np.float64)
        for i in range(len(quantized) - order):
            state = quantized[i]
            nxt = quantized[i + 1]
            trans[state, nxt] += 1.0
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        probs = trans / row_sums
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.sum(probs * np.log2(probs + 1e-30), axis=1)
        stationary = row_sums.ravel() / max(row_sums.sum(), 1.0)
        del trans, probs, quantized
        result = float(np.sum(stationary * h))
        del stationary, row_sums
        return result
    except Exception:
        return 0.0


def _toeplitz_score(tensor: np.ndarray) -> float:
    if tensor.ndim != 2 or min(tensor.shape) < 4:
        return 0.0
    sub = tensor[:32, :32]
    nd = min(sub.shape) - 1
    diag_stds = []
    diag_means = []
    for k in range(-nd, nd + 1):
        d = np.diag(sub, k)
        if len(d) > 1:
            diag_stds.append(float(np.std(d)))
            diag_means.append(float(np.mean(np.abs(d))) + 1e-10)
    if not diag_means:
        return 0.0
    cv = np.mean(np.array(diag_stds) / np.array(diag_means))
    return max(0.0, 1.0 - float(cv))


def _circulant_score(tensor: np.ndarray) -> float:
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1] or tensor.shape[0] < 4:
        return 0.0
    sub = tensor[:32, :32]
    first_col = sub[:, 0]
    n = min(len(first_col), sub.shape[1])
    scores = np.zeros(n, dtype=np.float64)
    for j in range(n):
        expected = np.roll(first_col, j)
        actual = sub[:, j]
        corr = np.corrcoef(expected, actual)
        scores[j] = float(np.abs(corr[0, 1]))
        del corr
    result = float(np.mean(scores)) if n > 0 else 0.0
    del scores
    return result


def _block_diagonal_score(tensor: np.ndarray, block_size: int = 16) -> float:
    if tensor.ndim != 2 or min(tensor.shape) < block_size:
        return 0.0
    m, n = tensor.shape
    off_diag_energy = 0.0
    total_energy = 0.0
    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            block = tensor[i : i + block_size, j : j + block_size]
            block_f = block.astype(np.float32)
            energy = float(np.sum(block_f**2))
            total_energy += energy
            if i != j:
                off_diag_energy += energy
    if total_energy < 1e-30:
        return 0.0
    return max(0.0, 1.0 - off_diag_energy / total_energy)


def _hierarchical_structure_score(tensor: np.ndarray) -> float:
    if tensor.ndim != 2 or min(tensor.shape) < 8:
        return 0.0
    sub = tensor[:64, :64]
    h = min(sub.shape) // 2
    if h < 2:
        return 0.0
    q1 = sub[:h, :h].ravel()
    q2 = sub[:h, h : 2 * h].ravel()
    q3 = sub[h : 2 * h, :h].ravel()
    q4 = sub[h : 2 * h, h : 2 * h].ravel()
    c12 = (
        float(np.abs(np.corrcoef(q1, q2)[0, 1])) if len(q1) > 1 and len(q2) > 1 else 0.0
    )
    c34 = (
        float(np.abs(np.corrcoef(q3, q4)[0, 1])) if len(q3) > 1 and len(q4) > 1 else 0.0
    )
    return float(np.mean([c12, c34]))


def _mutual_information_blocks(tensor: np.ndarray, n_blocks: int = 4) -> float:
    if tensor.ndim != 2 or min(tensor.shape) < n_blocks:
        return 0.0
    flat = tensor.ravel()
    n = len(flat)
    block_size = n // n_blocks
    if block_size < 2:
        return 0.0
    mi_sum = 0.0
    count = 0
    for i in range(n_blocks):
        a = flat[i * block_size : (i + 1) * block_size]
        for j in range(i + 1, n_blocks):
            b = flat[j * block_size : (j + 1) * block_size]
            n_min = min(len(a), len(b))
            if n_min < 4:
                continue
            ca, cb = a[:n_min], b[:n_min]
            r = float(np.abs(np.corrcoef(ca, cb)[0, 1]))
            if r > 0.99:
                r = 0.99
            if r < 1e-10:
                continue
            mi = -0.5 * math.log1p(-r * r)
            mi_sum += mi
            count += 1
    return mi_sum / max(count, 1)


def _kolmogorov_estimate(flat: np.ndarray) -> float:
    if flat.size < 32:
        return 0.0
    sample = flat[: min(len(flat), 1000)]
    percentiles = np.percentile(sample, np.linspace(0, 100, 9))
    quantized = np.digitize(sample, percentiles) - 1
    del percentiles
    s = "".join(chr(ord("a") + int(x)) for x in quantized)
    del quantized
    n = len(s)
    lib: set = set()
    w = ""
    for c in s:
        if w + c in lib:
            w += c
        else:
            lib.add(w + c)
            w = ""
    if n == 0:
        return 0.0
    complexity = len(lib) / n
    return min(float(complexity), 1.0)


def _structured_nm_score(tensor: np.ndarray, n: int = 2, m: int = 4) -> float:
    if tensor.ndim != 2:
        return 0.0
    flat = tensor.ravel()
    n_groups = len(flat) // m
    if n_groups < 1:
        return 0.0
    groups = flat[: n_groups * m].reshape(-1, m)
    abs_g = np.abs(groups)
    threshold = np.sort(abs_g, axis=1)[:, -n]
    nz_ratio = float(np.mean(np.sum(abs_g >= threshold[:, np.newaxis], axis=1) <= n))
    return nz_ratio


def _block_sparsity_score(tensor: np.ndarray, block_size: int = 16) -> float:
    if tensor.ndim != 2 or min(tensor.shape) < block_size:
        return 0.0
    m, n = tensor.shape
    blocks_m = m // block_size
    blocks_n = n // block_size
    if blocks_m < 1 or blocks_n < 1:
        return 0.0
    block_norms = np.zeros((blocks_m, blocks_n), dtype=np.float32)
    for i in range(blocks_m):
        for j in range(blocks_n):
            b = tensor[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            block_norms[i, j] = float(np.linalg.norm(b))
    threshold = np.percentile(block_norms, 50)
    result = float(np.mean(block_norms < threshold))
    del block_norms
    return result


def _unstructured_sparsity_score(tensor: np.ndarray) -> float:
    flat = tensor.ravel()
    if flat.size == 0:
        return 0.0
    abs_v = np.abs(flat)
    threshold = np.percentile(abs_v, 50)
    result = float(np.mean(abs_v < threshold))
    return result


def _nm_sparsity_score(tensor: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    n2m4 = _structured_nm_score(tensor, 2, 4)
    n4m8 = _structured_nm_score(tensor, 4, 8)
    block = _block_sparsity_score(tensor)
    unstruct = _unstructured_sparsity_score(tensor)
    details = {"2:4": n2m4, "4:8": n4m8, "block": block, "unstructured": unstruct}
    result = float(np.mean([n2m4, n4m8, block, unstruct]))
    return result, details


def compute_tensor_size(shape: Tuple[int, ...], dtype: Any) -> int:
    """Compute the memory size of a tensor from its shape and dtype."""
    try:
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)
        return int(np.prod(shape)) * dtype.itemsize
    except Exception:
        return 0


def compute_compression_ratio(original: Any, compressed: Any) -> float:
    """Compute compression ratio from original and compressed data."""
    orig_bytes = original.nbytes if isinstance(original, np.ndarray) else len(original)
    comp_bytes = (
        compressed.nbytes if isinstance(compressed, np.ndarray) else len(compressed)
    )
    return orig_bytes / max(comp_bytes, 1)


def compute_error_metrics(
    original: np.ndarray, reconstructed: np.ndarray
) -> Dict[str, float]:
    """Compute error metrics between original and reconstructed tensors."""
    from spectralstream.core.math_primitives.quality import QualityAssessor

    qa = QualityAssessor()
    quality = qa.assess(original, reconstructed)
    return quality.to_dict()


def _sensitivity_weight(tensor_type: str, sensitivity_map: Dict[str, float]) -> float:
    """Look up sensitivity weight for a tensor type."""
    if not tensor_type:
        return 0.5
    return sensitivity_map.get(tensor_type, 0.5)


def _method_compatibility_score(method_name: str, profile: Any) -> float:
    """Score (0-1+) how compatible a method is with a tensor profile."""
    name_lower = method_name.lower()
    if isinstance(profile, dict):
        nbytes = profile.get("nbytes", 0)
        e_rank = profile.get("effective_rank", 0.5)
        energy = profile.get("energy_concentration", 0.5)
        nm_sparsity = profile.get("nm_sparsity_score", 0.0)
    else:
        nbytes = getattr(profile, "nbytes", 0)
        e_rank = getattr(profile, "effective_rank", 0.5)
        energy = getattr(profile, "energy_concentration", 0.5)
        nm_sparsity = getattr(profile, "nm_sparsity_score", 0.0)
    if nbytes < 1024 and "svd" in name_lower:
        return 0.0
    if (
        e_rank < 0.3
        and energy > 0.8
        and ("svd" in name_lower or "low_rank" in name_lower)
    ):
        return 1.5
    if energy > 0.7 and ("dct" in name_lower or "spectral" in name_lower):
        return 1.4
    if nm_sparsity > 0.4 and "sparsity" in name_lower:
        return 1.3
    return 1.0


def _select_methods(
    profile: Any,
    error_budget: float = 0.01,
    target_ratio: float = 5000.0,
    available_methods: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """DEPRECATED: Use MethodOracle.select() from world_model.method_oracle instead.

    Kept as fast fallback for legacy compatibility.
    Select candidate compression methods based on tier and profile compatibility.

    Returns list of method dicts with 'instance' and 'params' keys, sorted by score.
    """
    from ._tier_common import get_tier, tier_score

    if available_methods is None:
        # Use the lazy METHOD_CLASSES dict — only instantiates methods
        # that are actually accessed in the scoring loop below.
        try:
            from spectralstream.compression.methods import METHOD_CLASSES, _load_extra

            _load_extra()  # Ensure lazy sections are populated
            # Build lazy method list: instances created on first access only
            available_methods = []
            for name, cls in METHOD_CLASSES.items():
                try:
                    inst = cls() if isinstance(cls, type) else cls
                except Exception:
                    continue
                available_methods.append({"instance": inst, "params": {}, "name": name})
        except Exception:
            from ._methods import METHOD_REGISTRY

            all_methods = dict(METHOD_REGISTRY)
            available_methods = [
                {"instance": inst, "params": {}, "name": name}
                for name, inst in all_methods.items()
            ]

    scored: List[Tuple[Dict[str, Any], float]] = []
    for m in available_methods:
        if not isinstance(m, dict):
            continue
        inst = m.get("instance")
        if inst is None:
            continue
        name = m.get("name", getattr(inst, "name", "unknown"))
        cat = m.get("category", getattr(inst, "category", "quantization"))
        try:
            tier = get_tier(name, cat)
            ts = tier_score(tier)
        except Exception:
            ts = 0.3
        compat = _method_compatibility_score(name, profile)
        if compat <= 0.0:
            continue
        score = ts * 0.6 + compat * 0.4
        scored.append((m, score))

    scored.sort(key=lambda x: -x[1])
    return [s[0] for s in scored[:10]]


def _enrich_meta(
    meta: dict,
    tensor: np.ndarray,
    method_name: str = "",
    compressed_data: bytes = b"",
    reconstructed: Optional[np.ndarray] = None,
) -> dict:
    """Add original_shape, method info, compression_ratio, and error metrics to metadata."""
    meta["original_shape"] = list(tensor.shape)
    if method_name:
        meta["method"] = method_name
    if compressed_data:
        meta["compression_ratio"] = tensor.nbytes / max(len(compressed_data), 1)
    if reconstructed is not None:
        var = float(np.var(tensor))
        mse = float(np.mean((tensor.ravel() - reconstructed.ravel()) ** 2))
        meta["relative_error"] = mse / var if var > 0 else mse
        meta["snr_db"] = 10.0 * math.log10(var / mse) if mse > 0 else 100.0
    return meta


def compress_tensor_with_validation(
    tensor: np.ndarray,
    profile: Any,
    methods: List[Dict[str, Any]],
    error_budget: float = 0.01,
    skip_validation: bool = False,
    bypass_decision: Optional[str] = None,
) -> Tuple[bytes, dict, float, float]:
    """Try methods in order, return first that meets error budget.

    When *skip_validation* is True, only the first method is tried and
    its result is returned immediately without checking error budget.
    This is used by the grouping optimizer for non-representative tensors.

    When *bypass_decision* is provided, the testing strategy changes:
      - BYPASS_HIGH_CONFIDENCE  — use top-1 method directly, skip testing
      - BYPASS_MEDIUM_CONFIDENCE — test only top-3 methods
      - TEST_FULL / None         — test all candidates (original behavior)

    Returns (compressed_data, metadata, ratio, relative_error).
    Metadata always includes original_shape and method for decompress.
    """
    # ── Bypass path: high confidence — use top-1 method directly ──
    if bypass_decision == BYPASS_HIGH_CONFIDENCE and methods:
        m = methods[0]
        if isinstance(m, dict):
            inst = m.get("instance")
            if inst is not None:
                mname = m.get("name", getattr(inst, "name", "unknown"))
                params = m.get("params", {})
                try:
                    data, meta = inst.compress(tensor, **params)
                    recon = inst.decompress(data, meta)
                    if recon.shape != tensor.shape:
                        recon = recon.reshape(tensor.shape)
                    metrics = _compute_metrics(tensor, recon)
                    error = metrics["relative_error"]
                    ratio = _compute_ratio(tensor.nbytes, data)
                    _enrich_meta(meta, tensor, mname, data, recon)
                    return data, meta, ratio, error
                except Exception:
                    pass

    # ── Bypass path: medium confidence — test only top 3 ──
    if bypass_decision == BYPASS_MEDIUM_CONFIDENCE and methods:
        methods = methods[:3]

    # ── Skip-validation fast path (grouping optimizer) ──
    if skip_validation and methods:
        m = methods[0]
        if isinstance(m, dict):
            inst = m.get("instance")
            if inst is not None:
                mname = m.get("name", getattr(inst, "name", "unknown"))
                params = m.get("params", {})
                try:
                    data, meta = inst.compress(tensor, **params)
                    recon = inst.decompress(data, meta)
                    if recon.shape != tensor.shape:
                        recon = recon.reshape(tensor.shape)
                    metrics = _compute_metrics(tensor, recon)
                    error = metrics["relative_error"]
                    ratio = _compute_ratio(tensor.nbytes, data)
                    _enrich_meta(meta, tensor, mname, data, recon)
                    return data, meta, ratio, error
                except Exception:
                    pass

    # ── Full validation path ──
    for m in methods:
        if not isinstance(m, dict):
            continue
        inst = m.get("instance")
        if inst is None:
            continue
        mname = m.get("name", getattr(inst, "name", "unknown"))
        params = m.get("params", {})
        try:
            data, meta = inst.compress(tensor, **params)
            recon = inst.decompress(data, meta)
            if recon.shape != tensor.shape:
                recon = recon.reshape(tensor.shape)
            metrics = _compute_metrics(tensor, recon)
            error = metrics["relative_error"]
            ratio = _compute_ratio(tensor.nbytes, data)
            _enrich_meta(meta, tensor, mname, data, recon)
            if error <= error_budget:
                return data, meta, ratio, error
        except Exception:
            continue

    best_error = 1.0
    best_result: Optional[Tuple[bytes, dict, float, float]] = None
    for m in methods:
        if not isinstance(m, dict):
            continue
        inst = m.get("instance")
        if inst is None:
            continue
        mname = m.get("name", getattr(inst, "name", "unknown"))
        params = m.get("params", {})
        try:
            data, meta = inst.compress(tensor, **params)
            recon = inst.decompress(data, meta)
            if recon.shape != tensor.shape:
                recon = recon.reshape(tensor.shape)
            metrics = _compute_metrics(tensor, recon)
            error = metrics["relative_error"]
            ratio = _compute_ratio(tensor.nbytes, data)
            _enrich_meta(meta, tensor, mname, data, recon)
            if error < best_error:
                best_error = error
                best_result = (data, meta, ratio, error)
        except Exception:
            continue

    if best_result is not None:
        return best_result

    # Absolute last resort: block_int8 passthrough
    from ._methods import _BlockINT8

    try:
        inst = _BlockINT8()
        data, meta = inst.compress(tensor)
        meta["method"] = "block_int8"
        recon = inst.decompress(data, meta)
        _enrich_meta(meta, tensor, "block_int8", data, recon)
        ratio = _compute_ratio(tensor.nbytes, data)
        return data, meta, ratio, error_budget
    except Exception:
        pass

    return (
        b"",
        {"method": "passthrough", "original_shape": list(tensor.shape)},
        1.0,
        1.0,
    )


def _build_report(stats: Dict[str, Any]) -> dict:
    """Format compression statistics into a report dict.

    Computes overall_ratio from total_orig_bytes / total_compressed_bytes
    if not explicitly provided.
    """
    total_orig = stats.get("total_orig_bytes", 0)
    total_comp = stats.get("total_compressed_bytes", 0)
    overall = stats.get("overall_ratio", total_orig / max(total_comp, 1))
    avg = stats.get("average_ratio", overall)
    return {
        "tensors": stats.get("tensors", []),
        "total_orig_bytes": total_orig,
        "total_compressed_bytes": total_comp,
        "overall_ratio": overall,
        "average_ratio": avg,
        "avg_error": stats.get("avg_error", 0.0),
        "max_error": stats.get("max_error", 0.0),
        "min_error": stats.get("min_error", 0.0),
        "num_tensors": stats.get("num_tensors", len(stats.get("tensors", []))),
        "method_distribution": stats.get("method_distribution", {}),
        "time_seconds": stats.get("time_seconds", 0.0),
        "failures": stats.get("failures", []),
        "weighted_error": stats.get("weighted_error", 0.0),
        "per_layer_error": stats.get("per_layer_error", {}),
        "tensor_errors": stats.get("tensor_errors", {}),
        "tensor_ratios": stats.get("tensor_ratios", {}),
        "tensor_methods": stats.get("tensor_methods", {}),
    }

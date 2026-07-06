"""CompressionProfiler — statistical, spectral, structural, information, sparsity analysis."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    compression_quality,
    dct,
    detect_native_dtype,
    effective_rank,
    energy_concentration,
    ensure_float32,
    spectral_entropy as _spectral_entropy,
)

from ._constants import MAX_PROFILE_SAMPLES
from ._dataclasses import CalibrationData, TensorProfile
from ._helpers import (
    _block_diagonal_score,
    _circulant_score,
    _classify_by_name_simple,
    _compute_metrics,
    _hierarchical_structure_score,
    _kolmogorov_estimate,
    _mutual_information_blocks,
    _nm_sparsity_score,
    _sample_flat,
    _toeplitz_score,
    _estimate_entropy_rate,
    _estimate_noise_floor as _basic_noise_floor,
)

# Enhanced noise floor detection via NoiseAwareCompressor
try:
    from spectralstream.compression.methods.novel._archive_integration import (
        NoiseFloorProbe as _NoiseFloorProbe,
    )

    _enhanced_noise_probe = _NoiseFloorProbe()
except Exception:
    _enhanced_noise_probe = None

from ._methods import _BlockINT8
from ._sensitivity import _get_sensitivity

logger = logging.getLogger(__name__)


class CompressionProfiler:
    """Profiles tensors: statistical + spectral + structural + information + sensitivity + sparsity."""

    def __init__(
        self,
        max_samples: int = MAX_PROFILE_SAMPLES,
        enable_spectral: bool = True,
        enable_structural: bool = True,
        enable_information: bool = True,
    ) -> None:
        self.max_samples = max_samples
        self.enable_spectral = enable_spectral
        self.enable_structural = enable_structural
        self.enable_information = enable_information
        # ── Lazy profiling cache ───────────────────────────────────────
        self._lazy_enabled = False
        self._cache_limit = 500
        self._full_profile_cache: Dict[Tuple, TensorProfile] = {}

    def enable_lazy_profiling(self, cache_size_limit: int = 500) -> None:
        """Enable lazy profiling with bounded cache.

        When enabled, profile_tensor_lazy() caches expensive spectral and
        structural analysis results keyed by group signature (shape, dtype,
        tensor_type). Subsequent tensors with the same signature skip
        SVD/DCT/structural analysis and reuse cached values.

        Parameters
        ----------
        cache_size_limit : int
            Maximum number of cache entries (default: 500).
        """
        self._lazy_enabled = True
        self._cache_limit = cache_size_limit
        logger.debug("Lazy profiling enabled (cache limit=%d)", cache_size_limit)

    def clear_profile_cache(self) -> None:
        """Clear all cached spectral/structural profiles."""
        self._full_profile_cache.clear()

    def profile_tensor_lazy(
        self,
        tensor: np.ndarray,
        name: str = "",
        group_signature: Optional[Tuple] = None,
    ) -> TensorProfile:
        """Profile with lazy spectral analysis.

        Always runs lightweight statistical and sparsity analysis (~1ms).
        Only runs expensive spectral (SVD, DCT) and structural analysis
        for the *first* tensor with a given group_signature. Subsequent
        tensors with the same signature copy cached spectral/structural fields.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to profile.
        name : str
            Tensor name (for logging and classification).
        group_signature : tuple, optional
            Cache key, typically ``(shape, dtype, tensor_type)``. All tensors
            sharing the same signature are assumed to have identical spectral
            properties, so only the first one runs full analysis.

        Returns
        -------
        TensorProfile
            Fully populated profile (all fields set).
        """
        tensor = np.asarray(tensor)
        native_dtype = detect_native_dtype(tensor)
        flat = ensure_float32(tensor).ravel().astype(np.float64)
        n = flat.size
        p = TensorProfile(
            name=name,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            native_dtype=native_dtype,
            n_elements=n,
            nbytes=tensor.nbytes,
        )
        if n == 0:
            return p
        sample = _sample_flat(tensor, self.max_samples)

        # ── Always: fast analysis (statistical + sparsity) ──
        self._statistical_analysis(p, sample, flat)
        self._sparsity_analysis(p, tensor)

        # ── Conditionally: expensive analysis (spectral + structural) ──
        use_cache = (
            self._lazy_enabled
            and group_signature is not None
            and group_signature in self._full_profile_cache
        )

        if use_cache:
            cached = self._full_profile_cache[group_signature]
            p.effective_rank = cached.effective_rank
            p.spectral_decay_rate = cached.spectral_decay_rate
            p.energy_concentration = cached.energy_concentration
            p.spectral_entropy = cached.spectral_entropy
            p.toeplitz_score = cached.toeplitz_score
            p.circulant_score = cached.circulant_score
            p.block_diagonal_score = cached.block_diagonal_score
            p.hierarchical_score = cached.hierarchical_score
        else:
            if self.enable_spectral:
                self._spectral_analysis(p, sample, tensor)
            if self.enable_structural and tensor.ndim == 2:
                self._structural_analysis(p, tensor)

            # Cache for future tensors with same signature
            if self._lazy_enabled and group_signature is not None:
                if len(self._full_profile_cache) < self._cache_limit:
                    cache_entry = TensorProfile(
                        effective_rank=p.effective_rank,
                        spectral_decay_rate=p.spectral_decay_rate,
                        energy_concentration=p.energy_concentration,
                        spectral_entropy=p.spectral_entropy,
                        toeplitz_score=p.toeplitz_score,
                        circulant_score=p.circulant_score,
                        block_diagonal_score=p.block_diagonal_score,
                        hierarchical_score=p.hierarchical_score,
                    )
                    self._full_profile_cache[group_signature] = cache_entry

        # ── Always: information analysis (lightweight) ──
        if self.enable_information:
            self._information_analysis(p, sample, tensor)

        # ── Always: classification and recommendations ──
        p.name_sensitivity = _get_sensitivity(name) if name else 0.5
        p.sensitivity = p.name_sensitivity
        p.sensitivity = min(max(p.sensitivity, 0.1), 1.0)
        p.sensitivity_category = self._categorize_sensitivity(p.sensitivity)
        p.tensor_type = self._classify_tensor_type(p, name)
        p.recommended_bits, p.recommended_methods = self._recommend(p)
        p.optimal_bits = p.recommended_bits

        logger.debug(
            "Profile (lazy) '%s': type=%s, sens=%.3f, rank=%.1f, energy=%.3f, cached=%s",
            name,
            p.tensor_type,
            p.sensitivity,
            p.effective_rank,
            p.energy_concentration,
            use_cache,
        )
        return p

    def profile_tensor(
        self,
        tensor: np.ndarray,
        name: str = "",
        calibration: Optional[CalibrationData] = None,
    ) -> TensorProfile:
        tensor = np.asarray(tensor)
        native_dtype = detect_native_dtype(tensor)
        flat = ensure_float32(tensor).ravel().astype(np.float64)
        n = flat.size
        p = TensorProfile(
            name=name,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            native_dtype=native_dtype,
            n_elements=n,
            nbytes=tensor.nbytes,
        )
        if n == 0:
            return p
        sample = _sample_flat(tensor, self.max_samples)
        self._statistical_analysis(p, sample, flat)
        if self.enable_spectral:
            self._spectral_analysis(p, sample, tensor)
        if self.enable_structural and tensor.ndim == 2:
            self._structural_analysis(p, tensor)
        if self.enable_information:
            self._information_analysis(p, sample, tensor)
        self._sparsity_analysis(p, tensor)
        p.name_sensitivity = _get_sensitivity(name) if name else 0.5
        p.sensitivity = p.name_sensitivity
        if calibration and calibration.fisher_info and name in calibration.fisher_info:
            p.gradient_sensitivity = float(np.mean(calibration.fisher_info[name]))
            p.sensitivity = max(
                p.sensitivity, float(np.mean(calibration.fisher_info[name]) / 10.0)
            )
        p.sensitivity = min(max(p.sensitivity, 0.1), 1.0)
        p.sensitivity_category = self._categorize_sensitivity(p.sensitivity)
        p.tensor_type = self._classify_tensor_type(p, name)
        p.recommended_bits, p.recommended_methods = self._recommend(p)
        p.optimal_bits = p.recommended_bits
        logger.debug(
            "Profile '%s': type=%s, sens=%.3f, rank=%.1f, energy=%.3f, entropy=%.3f",
            name,
            p.tensor_type,
            p.sensitivity,
            p.effective_rank,
            p.energy_concentration,
            p.spectral_entropy,
        )
        return p

    def _statistical_analysis(
        self, p: TensorProfile, sample: np.ndarray, flat: np.ndarray
    ) -> None:
        p.mean = float(np.mean(sample))
        p.std = float(np.std(sample))
        p.min_val = float(np.min(flat))
        p.max_val = float(np.max(flat))
        p.dynamic_range = p.max_val - p.min_val
        if p.std > 1e-10 and len(sample) > 3:
            c = (sample - p.mean) / p.std
            p.kurtosis = float(np.mean(c**4) - 3.0)
            p.skewness = float(np.mean(c**3))
            p.outlier_ratio = float(np.mean(np.abs(c) > 3.0))
        else:
            p.kurtosis = 0.0
            p.skewness = 0.0
            p.outlier_ratio = 0.0

    def _spectral_analysis(
        self, p: TensorProfile, sample: np.ndarray, tensor: np.ndarray
    ) -> None:
        if tensor.ndim >= 2 and all(s > 1 for s in tensor.shape[:2]):
            # Subsample large tensors BEFORE float64 conversion to avoid OOM
            max_svd_elements = 10_000  # absolute max for SVD profiling
            flat_cols = tensor[
                0
            ].size  # total elements per "row" (product of remaining dims)
            nrows, ncols = tensor.shape[0], flat_cols
            if nrows * ncols > max_svd_elements:
                ratio = (max_svd_elements / (nrows * ncols)) ** 0.5
                sub_rows = max(2, min(nrows, int(nrows * ratio)))
                sub_cols = max(2, min(ncols, int(ncols * ratio)))
                # Slice first dim fully, then subsample the flattened row
                t_slice = tensor[:sub_rows]
                t_flat = t_slice.reshape(sub_rows, -1)[:, :sub_cols]
                mat = np.asarray(t_flat, dtype=np.float64)
                del t_slice, t_flat
                nrows, ncols = mat.shape
            else:
                mat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)
                nrows, ncols = mat.shape
            try:
                p.effective_rank = effective_rank(mat, max_samples=128)
                m1, m2 = min(nrows, 64), min(ncols, 64)
                sv = np.linalg.svd(mat[:m1, :m2], compute_uv=False)
                sv_norm = sv / (sv[0] + 1e-10)
                if len(sv_norm) > 2:
                    top_s = sv_norm[sv_norm > 1e-10]
                    if len(top_s) > 2:
                        lsv = np.log(top_s[: min(20, len(top_s))] + 1e-30)
                        nf = len(lsv)
                        xs = np.arange(nf, dtype=np.float64)
                        A = np.vstack([xs, np.ones(nf)]).T
                        try:
                            coeffs = np.linalg.lstsq(A, lsv, rcond=None)[0]
                            p.spectral_decay_rate = max(0.0, float(-coeffs[0]))
                        except np.linalg.LinAlgError:
                            pass
            except np.linalg.LinAlgError:
                pass
        if len(sample) >= 8:
            try:
                s_len = min(len(sample), 2048)
                pw = dct(sample[:s_len]) ** 2
                total = float(np.sum(pw))
                if total > 1e-30:
                    sorted_pw = np.sort(pw)[::-1]
                    cum = np.cumsum(sorted_pw) / total
                    n_keep = int(np.searchsorted(cum, 0.9)) + 1
                    p.energy_concentration = n_keep / max(len(sorted_pw), 1)
            except Exception:
                p.energy_concentration = 0.0
        if len(sample) >= 4:
            try:
                p.spectral_entropy = _spectral_entropy(sample[: min(len(sample), 4096)])
            except Exception:
                p.spectral_entropy = 0.0

    def _structural_analysis(self, p: TensorProfile, tensor: np.ndarray) -> None:
        if min(tensor.shape) >= 4:
            p.toeplitz_score = _toeplitz_score(tensor)
            p.circulant_score = (
                _circulant_score(tensor) if tensor.shape[0] == tensor.shape[1] else 0.0
            )
            p.block_diagonal_score = _block_diagonal_score(tensor)
            p.hierarchical_score = _hierarchical_structure_score(tensor)

    def _information_analysis(
        self, p: TensorProfile, sample: np.ndarray, tensor: np.ndarray
    ) -> None:
        p.entropy_rate = _estimate_entropy_rate(sample)
        p.mutual_information = _mutual_information_blocks(tensor)
        p.kolmogorov_complexity = _kolmogorov_estimate(sample)
        if _enhanced_noise_probe is not None:
            try:
                p.noise_floor = float(
                    _enhanced_noise_probe.estimate_noise_floor(tensor)
                )
            except Exception:
                p.noise_floor = _basic_noise_floor(tensor)
        else:
            p.noise_floor = _basic_noise_floor(tensor)

    @staticmethod
    def _categorize_sensitivity_static(sensitivity: float) -> str:
        if sensitivity >= 0.8:
            return "CRITICAL"
        if sensitivity >= 0.6:
            return "HIGH"
        if sensitivity >= 0.4:
            return "MEDIUM"
        return "LOW"

    def _sparsity_analysis(self, p: TensorProfile, tensor: np.ndarray) -> None:
        if tensor.ndim >= 2:
            avg_sparsity, details = _nm_sparsity_score(tensor)
            p.nm_sparsity_score = avg_sparsity
            p.sparsity_details = details
            p.block_sparsity_score = details.get("block", 0.0)
            p.unstructured_sparsity_score = details.get("unstructured", 0.0)

    def _categorize_sensitivity(self, sensitivity: float) -> str:
        if sensitivity >= 0.8:
            return "CRITICAL"
        if sensitivity >= 0.6:
            return "HIGH"
        if sensitivity >= 0.4:
            return "MEDIUM"
        return "LOW"

    def _classify_tensor_type(self, p: TensorProfile, name: str) -> str:
        if p.nbytes < 1024:
            return "norm_bias"
        if p.shape and p.shape[0] > 10000 and p.n_elements > 1_000_000:
            return "embedding"
        t = _classify_by_name_simple(name)
        if t == "norm":
            return "norm_bias"
        return t

    def _recommend(self, p: TensorProfile) -> Tuple[int, List[str]]:
        recommended: List[str] = []
        bits = 8
        if p.nbytes < 4096:
            return 16, ["passthrough"]
        if p.noise_floor > 0.6:
            recommended.append("delta_int4")
        if p.outlier_ratio > 0.3:
            recommended.append("sparsity_int4")
        if p.effective_rank < 16 and p.energy_concentration > 0.8:
            recommended.append("hadamard_int8")
            recommended.append("block_int8")
        elif p.energy_concentration > 0.7:
            recommended.append("block_int8")
            if p.spectral_entropy < 0.5:
                recommended.append("hadamard_int8")
        elif p.energy_concentration > 0.5:
            recommended.append("block_int4")
            recommended.append("block_int8")
        else:
            recommended.append("block_int8")
        if p.toeplitz_score > 0.8:
            recommended.insert(0, "block_int8")
        if p.circulant_score > 0.8:
            recommended.insert(0, "block_int8")
        if p.hierarchical_score > 0.7:
            recommended.insert(0, "block_int8")
        if "sparsity_int4" not in recommended and p.nm_sparsity_score > 0.5:
            recommended.insert(0, "sparsity_int4")
        if not recommended:
            recommended.append("block_int8")
        return bits, recommended

    def sensitivity_analysis(self, tensor: np.ndarray) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for bits in [2, 3, 4, 5, 6, 8]:
            try:
                ctx = _BlockINT8()
                cd, meta = ctx.compress(tensor, block_size=128)
                results[f"int8_{bits}bit"] = _compute_metrics(
                    tensor, ctx.decompress(cd, meta).reshape(tensor.shape)
                )["relative_error"]
            except Exception:
                results[f"int8_{bits}bit"] = 1.0
        return results

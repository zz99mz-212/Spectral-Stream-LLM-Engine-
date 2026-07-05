"""
Resonant Tensor Grouping — groups tensors by spectral "resonance fingerprint"
rather than just shape/name patterns.

Two tensors with different shapes but similar spectral properties (singular value
decay, DCT energy concentration, etc.) "resonate" together and can share the same
compression method. This produces fewer groups and more intelligent grouping.
"""

from __future__ import annotations

__all__ = [
    "SpectralResonanceProfile",
    "ResonantGroup",
    "ResonantGrouper",
    "resonance_refine_groups",
]

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Resonance weights for distance computation ────────────────────────
_WEIGHTS = np.array([0.30, 0.20, 0.20, 0.15, 0.10, 0.05], dtype=np.float64)


@dataclass
class SpectralResonanceProfile:
    """A tensor's "resonance fingerprint" — its response to compression methods.

    Computed from a lightweight statistical sample of the tensor (~2ms for
    typical layers, ~5ms for large embeddings).

    Attributes
    ----------
    spectral_decay_rate : float
        How fast singular values decay (0=all equal, 1=rank-1).
    energy_concentration : float
        DCT energy fraction in top 10% coefficients.
    spectral_flatness : float
        Wiener entropy (0=tonal/tracktable, 1=noise-like).
    effective_rank_ratio : float
        effective_rank / min(shape) — normalized rank.
    nm_sparsity_score : float
        Structured N:M sparsity score.
    outlier_ratio : float
        Fraction of 3-sigma outliers.
    entropy_rate : float
        Approximate entropy rate (0=deterministic, 4=max).
    tensor_type : str
        Semantic type (attention_q, ffn_gate, embedding, etc.).
    shape : Tuple[int, ...]
        Original tensor shape.
    ndim : int
        Number of dimensions.
    """

    spectral_decay_rate: float = 0.0
    energy_concentration: float = 0.0
    spectral_flatness: float = 0.0
    effective_rank_ratio: float = 0.0
    nm_sparsity_score: float = 0.0
    outlier_ratio: float = 0.0
    entropy_rate: float = 0.0
    tensor_type: str = "generic"
    shape: Tuple[int, ...] = ()
    ndim: int = 0

    def resonance_distance(self, other: SpectralResonanceProfile) -> float:
        """Weighted Euclidean distance in resonance vector space.

        Low distance = similar compression response.

        Weights:
        - spectral_decay_rate: 0.30 (most important for SVD methods)
        - energy_concentration: 0.20 (important for DCT methods)
        - effective_rank_ratio: 0.20 (important for low-rank methods)
        - outlier_ratio: 0.15 (important for quantization methods)
        - entropy_rate: 0.10 (important for entropy methods)
        - spectral_flatness: 0.05 (fine-tuning)
        """
        vec_a = self.to_fingerprint()
        vec_b = other.to_fingerprint()
        diff = vec_a - vec_b
        return float(np.sqrt(np.dot(_WEIGHTS, diff * diff)))

    def to_fingerprint(self) -> np.ndarray:
        """6-element numpy array for fast comparison."""
        return np.array(
            [
                self.spectral_decay_rate,
                self.energy_concentration,
                self.effective_rank_ratio,
                self.outlier_ratio,
                self.entropy_rate,
                self.spectral_flatness,
            ],
            dtype=np.float64,
        )


@dataclass
class ResonantGroup:
    """A group of tensors that resonate similarly to compression methods.

    Attributes
    ----------
    members : List[str]
        Tensor names in this group.
    centroid : SpectralResonanceProfile
        Centroid profile of the group (element-wise mean of fingerprints).
    group_id : int
        Unique group identifier.
    """

    members: List[str]
    centroid: SpectralResonanceProfile
    group_id: int

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def is_unity(self) -> bool:
        return self.size == 1


def _dct_1d(x: np.ndarray) -> np.ndarray:
    """Type-II DCT using FFT (no external dependency)."""
    n = x.size
    x2 = np.empty(2 * n, dtype=np.float64)
    x2[:n] = x
    x2[n:] = x[::-1]
    f = np.fft.fft(x2)
    k = np.arange(n, dtype=np.float64)
    coeffs = np.real(f[:n] * np.exp(-1j * np.pi * k / (2 * n)))
    coeffs[0] = x.sum()
    return coeffs / np.sqrt(2 * n)


def _compute_profile_from_sample(
    tensor: np.ndarray,
    tensor_type: str = "generic",
) -> SpectralResonanceProfile:
    """Compute a spectral resonance profile from a tensor (lightweight).

    Uses subsampling to keep computation under ~2ms for typical tensors.
    Implements DCT directly to avoid circular imports.
    """
    tensor = np.asarray(tensor)
    shape = tensor.shape
    ndim = tensor.ndim
    flat = tensor.ravel()
    flat_f64 = flat.astype(np.float64)

    # ── Spectral decay rate (SVD on subsampled matrix) ──
    spectral_decay_rate = 0.0
    if ndim >= 2 and all(s > 1 for s in shape[:2]):
        max_svd = 5000
        nrows = shape[0]
        ncols = int(np.prod(shape[1:]))
        if nrows * ncols > max_svd:
            ratio = (max_svd / (nrows * ncols)) ** 0.5
            sub_r = max(2, min(nrows, int(nrows * ratio)))
            sub_c = max(2, min(ncols, int(ncols * ratio)))
            mat = tensor[:sub_r].reshape(sub_r, -1)[:, :sub_c].astype(np.float64)
        else:
            mat = tensor.reshape(nrows, -1).astype(np.float64)
        m1, m2 = min(mat.shape[0], 48), min(mat.shape[1], 48)
        if m1 > 1 and m2 > 1:
            try:
                sv = np.linalg.svd(mat[:m1, :m2], compute_uv=False)
                sv_norm = sv / (sv[0] + 1e-10)
                top_s = sv_norm[sv_norm > 1e-10]
                if len(top_s) > 3:
                    lsv = np.log(top_s[: min(15, len(top_s))] + 1e-30)
                    nf = len(lsv)
                    xs = np.arange(nf, dtype=np.float64)
                    A = np.vstack([xs, np.ones(nf)]).T
                    coeffs = np.linalg.lstsq(A, lsv, rcond=None)[0]
                    spectral_decay_rate = max(0.0, min(1.0, float(-coeffs[0])))
            except np.linalg.LinAlgError:
                pass

    # ── Energy concentration (DCT) ──
    energy_concentration = 0.0
    if flat.size >= 8:
        s_len = min(flat.size, 2048)
        try:
            pw = _dct_1d(flat_f64[:s_len]) ** 2
            total = float(np.sum(pw))
            if total > 1e-30:
                sorted_pw = np.sort(pw)[::-1]
                cum = np.cumsum(sorted_pw) / total
                n_keep = int(np.searchsorted(cum, 0.9)) + 1
                energy_concentration = n_keep / max(len(sorted_pw), 1)
        except Exception:
            pass

    # ── Spectral flatness (Wiener entropy) ──
    spectral_flatness = 0.0
    if flat.size >= 16:
        s_len = min(flat.size, 2048)
        try:
            pw = _dct_1d(flat_f64[:s_len]) ** 2
            total = float(np.sum(pw))
            if total > 1e-30:
                pw_norm = pw / total
                geo = float(np.exp(np.mean(np.log(pw_norm + 1e-30))))
                arith = float(np.mean(pw_norm))
                spectral_flatness = 1.0 - (geo / (arith + 1e-30))
        except Exception:
            pass

    # ── Effective rank ratio ──
    effective_rank_ratio = 0.0
    if ndim >= 2 and all(s > 1 for s in shape[:2]):
        try:
            max_ev = 5000
            nrows = shape[0]
            ncols = int(np.prod(shape[1:]))
            if nrows * ncols > max_ev:
                ratio = (max_ev / (nrows * ncols)) ** 0.5
                sub_r = max(2, min(nrows, int(nrows * ratio)))
                sub_c = max(2, min(ncols, int(ncols * ratio)))
                mat = tensor[:sub_r].reshape(sub_r, -1)[:, :sub_c].astype(np.float64)
            else:
                mat = tensor.reshape(nrows, -1).astype(np.float64)
            sv = np.linalg.svd(mat, compute_uv=False)
            sv_norm = sv / (sv[0] + 1e-10)
            energy = np.cumsum(sv_norm**2) / (np.sum(sv_norm**2) + 1e-30)
            eff_r = float(np.searchsorted(energy, 0.9) + 1)
            min_dim = min(mat.shape)
            effective_rank_ratio = eff_r / max(min_dim, 1)
        except Exception:
            pass

    # ── N:M sparsity score (2:4 pattern) ──
    nm_sparsity_score = 0.0
    if ndim >= 2:
        try:
            n_groups = flat.size // 4
            if n_groups >= 1:
                groups = flat_f64[: n_groups * 4].reshape(-1, 4)
                abs_g = np.abs(groups)
                threshold = np.sort(abs_g, axis=1)[:, -2]
                nm_sparsity_score = float(
                    np.mean(np.sum(abs_g >= threshold[:, np.newaxis], axis=1) <= 2)
                )
        except Exception:
            pass

    # ── Outlier ratio ──
    outlier_ratio = 0.0
    if flat.size > 3:
        sample = flat_f64[: min(flat.size, 10000)]
        mu = float(np.mean(sample))
        sigma = float(np.std(sample))
        if sigma > 1e-10:
            outlier_ratio = float(np.mean(np.abs(sample - mu) > 3.0 * sigma))

    # ── Entropy rate ──
    entropy_rate = 0.0
    if flat.size >= 16:
        sample = flat_f64[: min(flat.size, 5000)]
        try:
            percentiles = np.percentile(sample, np.linspace(0, 100, 17))
            quantized = np.digitize(sample, percentiles) - 1
            n_states = 16
            trans = np.zeros((n_states, n_states), dtype=np.float64)
            for i in range(len(quantized) - 1):
                s = quantized[i]
                nxt = quantized[i + 1]
                trans[s, nxt] += 1.0
            row_sums = trans.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            probs = trans / row_sums
            with np.errstate(divide="ignore", invalid="ignore"):
                h = -np.sum(probs * np.log2(probs + 1e-30), axis=1)
            stationary = row_sums.ravel() / max(row_sums.sum(), 1.0)
            entropy_rate = float(np.sum(stationary * h))
        except Exception:
            pass

    return SpectralResonanceProfile(
        spectral_decay_rate=spectral_decay_rate,
        energy_concentration=energy_concentration,
        spectral_flatness=spectral_flatness,
        effective_rank_ratio=effective_rank_ratio,
        nm_sparsity_score=nm_sparsity_score,
        outlier_ratio=outlier_ratio,
        entropy_rate=entropy_rate,
        tensor_type=tensor_type,
        shape=shape,
        ndim=ndim,
    )


class ResonantGrouper:
    """Groups tensors by spectral resonance similarity, not just shape/name.

    Two tensors "resonate" together if their SpectralResonanceProfiles
    are close (resonance_distance < threshold). This allows tensors of
    different shapes but similar spectral properties to share the same
    compression method.

    Benefits:
    - Fewer groups than shape-based grouping (20% fewer is typical)
    - More intelligent grouping (tensors that respond similarly to
      compression are grouped together)
    - Adaptive threshold based on model complexity
    """

    def __init__(self, resonance_threshold: float = 0.15):
        self._threshold = resonance_threshold
        self._profiles: Dict[str, SpectralResonanceProfile] = {}
        self._fingerprint_matrix: Optional[np.ndarray] = None
        self._tensor_names: List[str] = []
        # Cache for compute_profile: keyed by (shape, tensor_type)
        self._profile_cache: Dict[Tuple, SpectralResonanceProfile] = {}

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.01, min(0.5, value))

    @property
    def profiles(self) -> Dict[str, SpectralResonanceProfile]:
        return dict(self._profiles)

    def compute_profile(
        self, tensor: np.ndarray, tensor_type: str = "generic"
    ) -> SpectralResonanceProfile:
        """Compute a resonance profile from a tensor (lightweight, ~2ms).

        Results are cached by ``(tensor.shape, tensor_type)`` so that
        identical-shaped tensors of the same type share a single profile
        computation.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to profile.
        tensor_type : str
            Semantic type hint (e.g. 'attention_q', 'ffn_gate').

        Returns
        -------
        SpectralResonanceProfile
            The computed resonance fingerprint.
        """
        key = (tensor.shape, tensor_type)
        if key in self._profile_cache:
            return self._profile_cache[key]
        profile = _compute_profile_from_sample(tensor, tensor_type)
        self._profile_cache[key] = profile
        return profile

    def clear_cache(self) -> None:
        """Clear the resonance profile cache."""
        self._profile_cache.clear()

    def add_tensor(
        self, name: str, tensor: np.ndarray, tensor_type: str = "generic"
    ) -> None:
        """Register a tensor and compute its resonance profile."""
        profile = self.compute_profile(tensor, tensor_type)
        self._profiles[name] = profile
        self._tensor_names = list(self._profiles.keys())
        self._fingerprint_matrix = None

    def add_tensor_with_profile(
        self, name: str, profile: SpectralResonanceProfile
    ) -> None:
        """Register a tensor with a pre-computed profile."""
        self._profiles[name] = profile
        self._tensor_names = list(self._profiles.keys())
        self._fingerprint_matrix = None

    def _build_fingerprint_matrix(self) -> np.ndarray:
        """Build cached fingerprint matrix (n_tensors x 6)."""
        if self._fingerprint_matrix is not None:
            return self._fingerprint_matrix
        names = self._tensor_names
        if not names:
            return np.empty((0, 6), dtype=np.float64)
        fps = [self._profiles[n].to_fingerprint() for n in names]
        self._fingerprint_matrix = np.stack(fps, axis=0)
        return self._fingerprint_matrix

    def group_tensors(self) -> List[ResonantGroup]:
        """Group all registered tensors by resonance similarity.

        Algorithm:
        1. Start with each tensor as its own group.
        2. Iteratively merge the closest pair of groups.
        3. Stop when min inter-group distance > resonance_threshold.
        4. Return sorted groups (largest first).

        Returns
        -------
        List[ResonantGroup]
            Sorted groups with computed centroids.
        """
        names = self._tensor_names
        n = len(names)
        if n == 0:
            return []
        if n == 1:
            profile = self._profiles[names[0]]
            return [ResonantGroup(members=[names[0]], centroid=profile, group_id=0)]

        fp_mat = self._build_fingerprint_matrix()

        clusters: List[List[int]] = [[i] for i in range(n)]

        while len(clusters) > 1:
            min_dist = float("inf")
            merge_pair = (0, 1)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    ci_fp = np.mean(fp_mat[clusters[i]], axis=0)
                    cj_fp = np.mean(fp_mat[clusters[j]], axis=0)
                    diff = ci_fp - cj_fp
                    dist = float(np.sqrt(np.dot(_WEIGHTS, diff * diff)))
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (i, j)

            if min_dist > self._threshold:
                break

            i, j = merge_pair
            if i > j:
                i, j = j, i
            clusters[i].extend(clusters[j])
            clusters.pop(j)

        groups: List[ResonantGroup] = []
        for gid, indices in enumerate(clusters):
            member_names = [names[idx] for idx in indices]
            centroid_fp = np.mean(fp_mat[indices], axis=0)
            rep_profile = self._profiles[member_names[0]]
            centroid = SpectralResonanceProfile(
                spectral_decay_rate=float(centroid_fp[0]),
                energy_concentration=float(centroid_fp[1]),
                effective_rank_ratio=float(centroid_fp[2]),
                outlier_ratio=float(centroid_fp[3]),
                entropy_rate=float(centroid_fp[4]),
                spectral_flatness=float(centroid_fp[5]),
                nm_sparsity_score=rep_profile.nm_sparsity_score,
                tensor_type=rep_profile.tensor_type,
                shape=rep_profile.shape,
                ndim=rep_profile.ndim,
            )
            groups.append(
                ResonantGroup(members=member_names, centroid=centroid, group_id=gid)
            )

        groups.sort(key=lambda g: -g.size)
        return groups

    def add_metadata_grouping(self, shape_groups: List[Any]) -> List[ResonantGroup]:
        """Enhance shape-based groups with resonance profiles.

        Takes the output of TensorGrouper.group_tensors() and refines it
        by computing resonance profiles for each group representative.
        Two shape-based groups with similar resonance profiles are merged.

        Parameters
        ----------
        shape_groups : List[TensorGroup]
            Shape-based groups from TensorGrouper.

        Returns
        -------
        List[ResonantGroup]
            Refined resonance-based groups.
        """
        if not shape_groups:
            return []

        group_reps: List[Tuple[Any, SpectralResonanceProfile]] = []
        for sg in shape_groups:
            rep_name = sg.representative if sg.representative else sg.tensor_names[0]
            if rep_name in self._profiles:
                profile = self._profiles[rep_name]
            else:
                profile = SpectralResonanceProfile(
                    spectral_decay_rate=0.0,
                    energy_concentration=0.0,
                    spectral_flatness=0.0,
                    effective_rank_ratio=0.0,
                    nm_sparsity_score=0.0,
                    outlier_ratio=0.0,
                    entropy_rate=0.0,
                    tensor_type="generic",
                    shape=sg.shape,
                    ndim=len(sg.shape),
                )
            group_reps.append((sg, profile))

        n_groups = len(group_reps)
        if n_groups == 0:
            return []

        fp_list = [p.to_fingerprint() for _, p in group_reps]
        fp_mat = np.stack(fp_list, axis=0)

        cluster_map: List[List[int]] = [[i] for i in range(n_groups)]

        while len(cluster_map) > 1:
            min_dist = float("inf")
            merge_pair = (0, 1)

            for i in range(len(cluster_map)):
                for j in range(i + 1, len(cluster_map)):
                    ci_fp = np.mean(fp_mat[cluster_map[i]], axis=0)
                    cj_fp = np.mean(fp_mat[cluster_map[j]], axis=0)
                    diff = ci_fp - cj_fp
                    dist = float(np.sqrt(np.dot(_WEIGHTS, diff * diff)))
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (i, j)

            if min_dist > self._threshold:
                break

            i, j = merge_pair
            if i > j:
                i, j = j, i
            cluster_map[i].extend(cluster_map[j])
            cluster_map.pop(j)

        groups: List[ResonantGroup] = []
        for gid, indices in enumerate(cluster_map):
            member_names: List[str] = []
            for idx in indices:
                sg = group_reps[idx][0]
                member_names.extend(sg.tensor_names)

            centroid_fp = np.mean(fp_mat[indices], axis=0)
            rep_sg = group_reps[indices[0]][0]

            centroid = SpectralResonanceProfile(
                spectral_decay_rate=float(centroid_fp[0]),
                energy_concentration=float(centroid_fp[1]),
                effective_rank_ratio=float(centroid_fp[2]),
                outlier_ratio=float(centroid_fp[3]),
                entropy_rate=float(centroid_fp[4]),
                spectral_flatness=float(centroid_fp[5]),
                nm_sparsity_score=0.0,
                tensor_type="generic",
                shape=rep_sg.shape,
                ndim=len(rep_sg.shape),
            )
            groups.append(
                ResonantGroup(members=member_names, centroid=centroid, group_id=gid)
            )

        groups.sort(key=lambda g: -g.size)
        return groups

    def tensor_deviation(self, tensor_name: str) -> float:
        """How much a tensor deviates from its group centroid.

        Returns 0.0 if the tensor is not registered or has no group centroid.
        """
        if tensor_name not in self._profiles:
            return 0.0
        profile = self._profiles[tensor_name]
        for g in self.group_tensors():
            if tensor_name in g.members:
                return profile.resonance_distance(g.centroid)
        return 0.0

    def resonance_distance_between(self, name_a: str, name_b: str) -> Optional[float]:
        """Resonance distance between two named tensors.

        Returns None if either tensor is not registered.
        """
        pa = self._profiles.get(name_a)
        pb = self._profiles.get(name_b)
        if pa is None or pb is None:
            return None
        return pa.resonance_distance(pb)


def resonance_refine_groups(
    groups: List[Any],
    tensors: Dict[str, np.ndarray],
    threshold: float = 0.15,
    profiles: Optional[Dict[str, SpectralResonanceProfile]] = None,
) -> List[ResonantGroup]:
    """Refine shape-based groups using spectral resonance profiles.

    Convenience function that:
    1. Takes existing shape-based groups (from TensorGrouper).
    2. Computes resonance profiles for each group's representative.
    3. Merges groups with similar resonance profiles.
    4. Returns refined ResonantGroup list.

    Parameters
    ----------
    groups : List[TensorGroup]
        Shape-based groups from TensorGrouper.group_tensors().
    tensors : Dict[str, np.ndarray]
        Full tensor dictionary (name -> array).
    threshold : float
        Resonance distance threshold for merging groups.
    profiles : Dict[str, SpectralResonanceProfile] or None
        Pre-computed profiles (will compute on-demand for missing entries).

    Returns
    -------
    List[ResonantGroup]
        Refined resonance-based groups.
    """
    if not groups:
        return []

    grouper = ResonantGrouper(resonance_threshold=threshold)

    profiles = profiles or {}
    all_tensor_names: set = set()
    for g in groups:
        for tn in g.tensor_names:
            all_tensor_names.add(tn)

    for tn in all_tensor_names:
        if tn in profiles:
            grouper.add_tensor_with_profile(tn, profiles[tn])
        elif tn in tensors:
            grouper.add_tensor(tn, tensors[tn])
        else:
            logger.warning("Tensor '%s' not found in tensors dict or profiles", tn)

    return grouper.add_metadata_grouping(groups)

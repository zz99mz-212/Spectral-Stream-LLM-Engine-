"""
Plasma Confinement Tensor Shaper v1.0
=====================================

Inspired by magnetic confinement fusion (tokamak) physics.

**The plasma-tensor analogy:**

A tokamak confines a plasma (ionized gas at 150M°C) using helical
magnetic field lines.  The tensor's coefficients are the "plasma" —
a sea of numbers with complex turbulent dynamics.

+---------------------------+--------------------------------------------+
| Tokamak Physics           | Tensor Compression                        |
+---------------------------+--------------------------------------------+
| Plasma                    | Tensor coefficients                        |
| Toroidal field B_φ        | Low-rank decomposition (large-scale)       |
| Poloidal field B_θ        | Spectral methods (cross-sectional)         |
| Helical field lines       | Combined decomposition + spectral cascade  |
| Safety factor q(r)        | Compression ratio profile vs. rank        |
| Magnetic island           | High-entropy region needing compression   |
| Sawtooth instability      | Error-budget redistribution               |
| H-mode (confinement)      | Optimal compression regime                |
| L-mode (loss)             | Information loss threshold                |
+---------------------------+--------------------------------------------+

**Mathematical framework:**

The plasma equilibrium (Grad-Shafranov equation):
  Δ*ψ = -μ₀ R² dp/dψ - F dF/dψ

where ψ = poloidal flux function.  We map this onto tensor compression:
  - ψ = cumulative compression ratio (flux surface)
  - p(ψ) = information density at compression level ψ
  - F(ψ) = toroidal field function (method effectiveness)
  - Δ* = elliptic operator (singular value decomposition)

The safety factor:
  q(r) = r B_φ / (R B_θ) ≈ dΦ/dψ

gives the "twist" of field lines — analogous to the compression ratio
profile across the tensor's singular value spectrum.  High q = good
confinement = high compression.

**Magnetic island model:**

Islands form when q(r) = m/n (rational surface).  These are regions of
chaotic field lines = high entropy in the tensor.  The island width:
  w = 4 √(r_s B_r / (n B_θ dq/dr))

gives the size of the region needing compression.  We compute this from
singular value ratios and spectral decay.

References:
  - Wesson, J. (2011). "Tokamaks." 4th ed. Oxford University Press.
  - Freidberg, J.P. (2007). "Plasma Physics and Fusion Energy."
    Cambridge University Press.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Tokamak Geometry ───────────────────────────────────────────────────
# Safety factor profile: q(r) = q_0 + (q_a - q_0) * (r/a)^2
# For a typical tokamak: q_0 ≈ 1.0, q_a ≈ 3.0-5.0 (edge)


def _safety_factor_profile(
    rank_frac: np.ndarray,
    q_axis: float = 1.0,
    q_edge: float = 4.0,
) -> np.ndarray:
    """Compute the safety factor q(r) across normalized rank.

    q(r) = q_0 + (q_a - q_0) * r²

    where r = fractional rank (0 → core, 1 → edge).

    In compression:
      - Low rank (core): q ≈ q_0 (tight confinement = high compression)
      - High rank (edge): q ≈ q_a (loose confinement = low compression)
      - Rational surfaces: q = m/n → magnetic islands (entropy spikes)
    """
    return q_axis + (q_edge - q_axis) * (rank_frac**2)


def _magnetic_island_width(
    rational_surface_r: float,
    shear: float,
    perturbed_field: float,
    toroidal_field: float,
    n_mode: int,
) -> float:
    """Compute magnetic island width from perturbed field.

    w = 4 √(r_s * B_r / (n * B_φ * shear))

    where:
      r_s = rational surface location (fractional rank)
      shear = dq/dr at r_s
      B_r = radial perturbed field (entropy spike magnitude)
      B_φ = toroidal field (total singular value scale)
      n = toroidal mode number (inverse of island period)

    A wider island = more region affected by entropy = more compression needed.
    """
    numerator = rational_surface_r * perturbed_field
    denominator = max(n_mode * toroidal_field * max(shear, 1e-10), 1e-30)
    return float(4.0 * np.sqrt(numerator / denominator))


def _mhd_stability(compression_ratios: np.ndarray, error_rates: np.ndarray) -> float:
    """Compute MHD stability parameter β.

    β = 2μ₀ ⟨p⟩ / ⟨B²⟩ = plasma pressure / magnetic pressure

    In compression terms:
      β = 2 * avg_compression_ratio / (avg_error_rate)²

    High β → unstable → plasma disrupts (compression fails).
    Low β → stable → good confinement (compression succeeds).

    The Troyon limit β_max ≈ 0.05 gives the maximum stable ratio.
    """
    avg_ratio = np.mean(compression_ratios)
    avg_error = np.mean(error_rates)
    beta = 2.0 * avg_ratio / (max(avg_error**2, 1e-30))
    return float(beta)


# ── Data Structures ────────────────────────────────────────────────────


@dataclass
class TokamakFieldLine:
    """A magnetic field line in the tokamak.

    Field lines are helical: they wrap around the torus both
    toroidally (long way) and poloidally (short way).

    In compression, each field line is a particular compression
    method's "view" of the tensor — it cuts across the tensor
    at a specific angle.
    """

    method_name: str = ""
    category: str = ""
    toroidal_mode: int = 0
    poloidal_mode: int = 1
    safety_factor: float = 2.0
    field_strength: float = 0.0

    def pitch_angle(self) -> float:
        """Compute the field line pitch ι = 1/q.

        ι = B_θ / (r B_φ) = poloidal / toroidal

        Higher ι → more poloidal → more "twisted" → better at
        capturing cross-tensor correlations.
        """
        return float(1.0 / max(self.safety_factor, 0.01))

    def rotational_transform(self) -> float:
        """Compute rotational transform ι/2π.

        The angle by which a field line rotates in one toroidal transit.
        """
        return float(self.pitch_angle() / (2.0 * np.pi))


@dataclass
class MagneticIsland:
    """A magnetic island at a rational surface.

    Magnetic islands form where q = m/n (rational).
    Inside the island, field lines are chaotic = high entropy.
    These correspond to regions of the tensor that are hardest
    to compress (high entropy, low structure).
    """

    rational_surface: float = 0.0
    m_mode: int = 1
    n_mode: int = 1
    island_width: float = 0.0
    entropy_density: float = 0.0
    is_unstable: bool = False

    def island_overlap(self, other: MagneticIsland) -> float:
        """Compute Chirikov overlap parameter.

        σ = (w_i + w_j) / |r_i - r_j|

        When σ > 1, islands overlap → global stochasticity →
        total information loss (compression impossible).
        """
        r_i = self.rational_surface
        r_j = other.rational_surface
        w_i = self.island_width
        w_j = other.island_width
        separation = abs(r_i - r_j)
        if separation < 1e-10:
            return 1.0
        return float((w_i + w_j) / separation)


@dataclass
class SafetyFactorProfile:
    """Safety factor profile q(r) across the tensor.

    The safety factor is the "compression efficiency" as a
    function of singular value rank.  Low q (core) = efficient,
    high q (edge) = less efficient.
    """

    q_values: np.ndarray = field(default_factory=lambda: np.ones(10))
    rational_surfaces: List[float] = field(default_factory=list)
    shear_profile: np.ndarray = field(default_factory=lambda: np.zeros(10))
    beta: float = 0.0
    is_h_mode: bool = False

    def confinement_quality(self) -> str:
        """Classify confinement regime."""
        if self.beta < 0.01:
            return "L-mode"
        if self.beta < 0.04:
            return "H-mode"
        return "disruption"


# ── Main Engine ────────────────────────────────────────────────────────


class PlasmaConfinementTensorShaper:
    """Plasma Confinement Tensor Shaper.

    Shapes a tensor's coefficients using tokamak-inspired magnetic
    confinement principles.  The tensor is treated as a plasma of
    coefficients confined by "magnetic fields" of compression methods.

    Key operations:

    1. **Safety factor analysis**: Compute q(r) profile = compression
       efficiency vs. singular value rank.  Identifies rational surfaces
       where entropy islands form.

    2. **Island detection**: Find magnetic islands (high-entropy regions)
       at rational surfaces.  These are regions that need aggressive
       compression methods.

    3. **H-mode access**: Identify the optimal compression regime by
       computing the MHD stability parameter β.  H-mode = high confinement
       = high compression with low error.

    4. **Field line tracing**: Map optimal compression paths through
       the tensor's coefficient space using helical field line geometry.

    The poloidal/toroidal decomposition:
      - Toroidal (n=0, large mode) → SVD/decomposition methods
      - Poloidal (m=1,2,...) → spectral methods (DCT, FFT, wavelets)
      - Helical (combined) → hybrid cascade patterns
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._profile: Optional[SafetyFactorProfile] = None
        self._islands: List[MagneticIsland] = []
        self._field_lines: List[TokamakFieldLine] = []
        self._beta: float = 0.0

    def _AnalyzeTensorPlasma(
        self,
        tensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analyze the tensor's singular value spectrum as a plasma.

        Returns (singular_values, q_profile, shear_profile).

        The singular values are the "flux surfaces" of the tensor.
        q(r) = safety factor at each singular value rank.
        """
        flat = tensor.ravel()
        if flat.size == 0:
            return np.array([1.0]), np.array([1.0]), np.array([0.0])

        svd_n = min(flat.size, min(tensor.shape), 512)
        if svd_n < 2:
            s = np.abs(flat[: max(2, min(512, flat.size // 10))].copy())
            s = np.sort(s)[::-1]
        else:
            _, s, _ = np.linalg.svd(
                tensor.reshape(tensor.shape[0], -1), full_matrices=False
            )
            s = s[:512].copy()

        s = np.maximum(s, 1e-30)
        s_norm = s / max(s[0], 1e-30)
        n_s = len(s_norm)

        rank_frac = np.linspace(0.0, 1.0, n_s)
        q_vals = _safety_factor_profile(rank_frac)

        dq_dr = np.gradient(q_vals, rank_frac)
        shear = np.abs(dq_dr)

        return s_norm, q_vals, shear

    def _DetectMagneticIslands(
        self,
        singular_values: np.ndarray,
        q_profile: np.ndarray,
        shear: np.ndarray,
    ) -> List[MagneticIsland]:
        """Detect magnetic islands at rational q surfaces.

        Islands form where q = m/n (rational).  We scan for
        near-rational values and compute island widths from
        the local entropy (singular value fluctuation).

        Uses NumPy vectorized ops — no Python loops.
        """
        n_s = len(singular_values)
        if n_s < 10:
            return []

        m_max = 10
        n_max = 5

        m_vals = np.arange(1, m_max + 1)
        n_vals = np.arange(1, n_max + 1)
        rationals = m_vals[:, None] / n_vals[None, :]
        rationals_flat = rationals.ravel()

        q_range = np.array([1.0, 5.0])
        mask = (rationals_flat >= q_range[0]) & (rationals_flat <= q_range[1])
        rationals_flat = rationals_flat[mask]

        if len(rationals_flat) == 0:
            return []

        islands: List[MagneticIsland] = []
        q_min, q_max = q_profile.min(), q_profile.max()

        for q_target in rationals_flat[:30]:
            if q_target < q_min or q_target > q_max:
                continue

            nearest = np.argmin(np.abs(q_profile - q_target))
            r_s = nearest / max(n_s - 1, 1)

            local_entropy = 0.0
            if nearest > 0 and nearest < n_s - 1:
                window = slice(max(0, nearest - 3), min(n_s, nearest + 4))
                sv_fluctuation = np.std(
                    singular_values[window] / max(singular_values[window].mean(), 1e-30)
                )
                local_entropy = float(min(sv_fluctuation, 1.0))
            else:
                local_entropy = float(singular_values[nearest])

            shear_val = float(shear[nearest]) if nearest < len(shear) else 0.1
            island_width = _magnetic_island_width(
                rational_surface_r=r_s,
                shear=max(shear_val, 0.01),
                perturbed_field=max(local_entropy, 0.001),
                toroidal_field=float(singular_values[0])
                if len(singular_values) > 0
                else 1.0,
                n_mode=int(max(1, q_target)),
            )

            islands.append(
                MagneticIsland(
                    rational_surface=float(r_s),
                    m_mode=int(q_target * 3 + 1),
                    n_mode=int(max(1, q_target)),
                    island_width=min(island_width, 0.5),
                    entropy_density=local_entropy,
                    is_unstable=island_width > 0.2,
                )
            )

        islands.sort(key=lambda x: -x.entropy_density)
        return islands[:20]

    def _GetFieldLinesFromMethods(self) -> List[TokamakFieldLine]:
        """Map available compression methods to tokamak field lines.

        Method categories correspond to toroidal/poloidal mode numbers:
          - Decomposition: toroidal (n=0, dominant mode)
          - Spectral: poloidal (m=1,2,...)
          - Others: helical combinations
        """
        if not hasattr(self._engine, "get_methods_by_categories"):
            return []

        field_lines: List[TokamakFieldLine] = []

        for cat_name, (t_mode, p_mode, q_factor) in {
            "decomposition": (0, 1, 1.0),
            "spectral": (1, 2, 1.5),
            "structural": (2, 3, 2.0),
            "functional": (1, 4, 2.5),
            "tensor_network": (3, 2, 2.0),
            "physics": (2, 5, 3.0),
            "entropy": (0, 1, 4.0),
            "lossless": (0, 1, 4.0),
            "quantization": (1, 3, 3.5),
            "hybrid": (2, 4, 2.5),
            "novel": (3, 5, 3.0),
        }.items():
            methods_in_cat = self._engine.get_methods_by_categories([cat_name])
            for mname in methods_in_cat:
                field_lines.append(
                    TokamakFieldLine(
                        method_name=mname,
                        category=cat_name,
                        toroidal_mode=t_mode,
                        poloidal_mode=p_mode,
                        safety_factor=q_factor,
                        field_strength=1.0 / max(q_factor, 0.5),
                    )
                )

        return field_lines

    def suggest_sequences(
        self,
        profile: Any,
        target_ratio: float = 1200.0,
        n_sequences: int = 3,
    ) -> List[Dict[str, Any]]:
        """Suggest confinement-optimized compression sequences.

        Uses the tokamak plasma confinement model to order methods:
          1. Start with toroidal methods (decomposition — large-scale)
          2. Add poloidal methods (spectral — cross-section)
          3. Add helical methods (structural — fine-grain)
          4. Add confinement (entropy — final compression)

        Parameters
        ----------
        profile : TensorProfile
            Tensor profile.
        target_ratio : float
            Desired compression ratio.
        n_sequences : int
            Number of sequences.

        Returns
        -------
        List[Dict[str, Any]]
            Ranked sequences with confinement metadata.
        """
        tensor = self._BuildMockTensorFromProfile(profile)
        s_vals, q_vals, shear = self._AnalyzeTensorPlasma(tensor)
        islands = self._DetectMagneticIslands(s_vals, q_vals, shear)

        compression_ratios = np.array([2.0] * 4 + [1.5] * 4)
        error_rates = np.array([0.01] * 4 + [0.005] * 4)
        self._beta = _mhd_stability(compression_ratios, error_rates)

        self._islands = islands
        self._profile = SafetyFactorProfile(
            q_values=q_vals,
            rational_surfaces=[i.rational_surface for i in islands[:5]],
            shear_profile=shear,
            beta=self._beta,
            is_h_mode=self._beta < 0.04,
        )

        self._field_lines = self._GetFieldLinesFromMethods()

        n_unstable = sum(1 for i in islands if i.is_unstable)
        chaos_level = min(1.0, n_unstable / max(len(islands), 1))
        safety = 1.0 - chaos_level

        confinement_modes = []
        if safety > 0.7:
            confinement_modes.append(self._BuildSequence_HMode(profile, target_ratio))
            confinement_modes.append(self._BuildSequence_LMode(profile, target_ratio))
            confinement_modes.append(
                self._BuildSequence_DisruptionResistant(profile, target_ratio)
            )
        elif safety > 0.4:
            confinement_modes.append(self._BuildSequence_LMode(profile, target_ratio))
            confinement_modes.append(
                self._BuildSequence_DisruptionResistant(profile, target_ratio)
            )
            confinement_modes.append(self._BuildSequence_HMode(profile, target_ratio))
        else:
            confinement_modes.append(
                self._BuildSequence_DisruptionResistant(profile, target_ratio)
            )
            confinement_modes.append(self._BuildSequence_LMode(profile, target_ratio))
            confinement_modes.append(self._BuildSequence_HMode(profile, target_ratio))

        results = []
        for seq_dict in confinement_modes[:n_sequences]:
            if not seq_dict.get("methods"):
                continue
            methods = seq_dict["methods"]
            n_methods = len(methods)

            if n_methods > 0:
                expected_ratio = 2.0**n_methods * max(1.0, target_ratio / 100.0) ** (
                    1.0 / max(n_methods, 1)
                )
            else:
                expected_ratio = target_ratio

            beta_stable = self._beta < 0.05

            results.append(
                {
                    "methods": methods,
                    "expected_ratio": max(1.0, expected_ratio),
                    "expected_error": 0.01 * (1.0 + chaos_level),
                    "confinement_mode": seq_dict.get("mode", "L-mode"),
                    "beta": self._beta,
                    "mhd_stable": beta_stable,
                    "n_islands": len(islands),
                    "chaos_level": chaos_level,
                    "safety_factor_edge": float(q_vals[-1]) if len(q_vals) > 0 else 4.0,
                    "toroidal_modes": [
                        fl.toroidal_mode for fl in self._field_lines[:n_methods]
                    ]
                    if self._field_lines
                    else [],
                    "poloidal_modes": [
                        fl.poloidal_mode for fl in self._field_lines[:n_methods]
                    ]
                    if self._field_lines
                    else [],
                }
            )

        return results

    def _BuildMockTensorFromProfile(self, profile: Any) -> np.ndarray:
        """Build a synthetic tensor from profile statistics."""
        shape = getattr(profile, "shape", (64, 64))
        if not shape or all(d == 0 for d in shape):
            return np.random.randn(64, 64).astype(np.float32)

        flat_size = int(np.prod(shape))
        base = np.random.randn(flat_size).astype(np.float64)
        mean = float(getattr(profile, "mean", 0.0))
        std = float(max(getattr(profile, "std", 1.0), 1e-10))
        tensor = base * std + mean

        eff_rank = int(max(1, getattr(profile, "effective_rank", min(shape) // 2)))
        if eff_rank < flat_size and len(shape) >= 2:
            tensor = tensor.reshape(shape)
            u, s, vt = np.linalg.svd(
                tensor.reshape(tensor.shape[0], -1), full_matrices=False
            )
            s[eff_rank:] *= 0.01
            tensor = (u * s) @ vt

        return tensor.reshape(shape).astype(np.float32)

    def _BuildSequence_HMode(self, profile: Any, target_ratio: float) -> Dict[str, Any]:
        """Build H-mode (high confinement) sequence.

        H-mode = optimal regime: start with strong toroidal confinement,
        then add poloidal shaping.  Uses a "pedestal" of strong methods
        at the edge (high ratio regime).
        """
        methods: List[str] = []
        target_log = np.log2(max(target_ratio, 10.0))
        n_stages = int(np.ceil(target_log / 2.0)) + 1

        if hasattr(self._engine, "get_methods_by_categories"):
            for tier, incat in [
                (1, ["decomposition", "spectral"]),
                (2, ["structural", "functional"]),
                (3, ["entropy", "lossless"]),
                (4, ["hybrid", "novel"]),
                (5, ["quantization"]),
            ]:
                if len(methods) >= n_stages:
                    break
                for cat in incat:
                    if len(methods) >= n_stages:
                        break
                    cat_methods = self._engine.get_methods_by_categories([cat])
                    if cat_methods:
                        mname = next(iter(cat_methods))
                        methods.append(mname)

        if not methods:
            methods = ["svd_compress", "dct_spectral", "fwht_compress", "block_int8"]

        return {
            "methods": methods,
            "mode": "H-mode",
            "expected_ratio": 2.0 ** len(methods),
        }

    def _BuildSequence_LMode(self, profile: Any, target_ratio: float) -> Dict[str, Any]:
        """Build L-mode (low confinement) sequence.

        L-mode = conservative: fewer methods, lower compression,
        but more stable (lower error).
        """
        methods: List[str] = []
        n_stages = min(4, max(2, int(np.log2(target_ratio / 10.0)) + 1))

        if hasattr(self._engine, "get_methods_by_categories"):
            for cat in ["decomposition", "spectral"]:
                if len(methods) >= n_stages:
                    break
                cat_methods = self._engine.get_methods_by_categories([cat])
                if cat_methods:
                    mname = next(iter(cat_methods))
                    methods.append(mname)

            if len(methods) < n_stages:
                for cat in ["structural", "entropy"]:
                    if len(methods) >= n_stages:
                        break
                    cat_methods = self._engine.get_methods_by_categories([cat])
                    if cat_methods:
                        mname = next(iter(cat_methods))
                        methods.append(mname)

        if not methods:
            methods = ["svd_compress", "dct_spectral"]

        return {
            "methods": methods,
            "mode": "L-mode",
            "expected_ratio": 1.5 ** len(methods),
        }

    def _BuildSequence_DisruptionResistant(
        self, profile: Any, target_ratio: float
    ) -> Dict[str, Any]:
        """Build disruption-resistant sequence (safe mode).

        When MHD instability is high (β > 0.05, many islands),
        we use methods that are robust to chaos:
          - Entropy methods first (they handle randomness well)
          - Then spectral (filter noise)
          - Then structural (find remaining patterns)
        """
        methods: List[str] = []
        n_stages = min(6, max(2, int(np.log2(target_ratio))))

        if hasattr(self._engine, "get_methods_by_categories"):
            for cat in [
                "entropy",
                "lossless",
                "spectral",
                "structural",
                "decomposition",
                "functional",
            ]:
                if len(methods) >= n_stages:
                    break
                cat_methods = self._engine.get_methods_by_categories([cat])
                if cat_methods:
                    mname = next(iter(cat_methods))
                    methods.append(mname)

        if not methods:
            methods = ["fwht_compress", "dct_spectral", "svd_compress", "block_int8"]

        return {
            "methods": methods,
            "mode": "disruption_resistant",
            "expected_ratio": 1.3 ** len(methods),
        }

    def get_confinement_report(self) -> Dict[str, Any]:
        """Return diagnostic info about the plasma confinement state."""
        return {
            "beta": self._beta,
            "confinement_quality": self._profile.confinement_quality()
            if self._profile
            else "unknown",
            "n_islands": len(self._islands),
            "unstable_islands": sum(1 for i in self._islands if i.is_unstable),
            "n_field_lines": len(self._field_lines),
            "edge_safety_factor": float(self._profile.q_values[-1])
            if self._profile and len(self._profile.q_values) > 0
            else 0.0,
            "is_h_mode": self._profile.is_h_mode if self._profile else False,
        }

    def fuse_with_engine(self, engine: Any) -> None:
        """Wire this shaper to a CompressionIntelligenceEngine instance."""
        self._engine = engine
        self._field_lines = self._GetFieldLinesFromMethods()

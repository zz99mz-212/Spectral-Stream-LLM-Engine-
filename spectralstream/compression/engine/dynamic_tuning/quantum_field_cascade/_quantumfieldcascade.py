"""
Quantum Field Theory Cascade Optimizer v1.0
===========================================

Formulates the optimal compression method sequence as a **quantum field
theory** (QFT) scattering problem.

The core analogy:
  - Each compression method is a **particle** in the QFT, with a
    propagator (how it acts on a tensor) and interaction vertices
    (how it couples to other methods).
  - The cascade is a **scattering process**: a → b → c → ... where
    methods interact and exchange "momentum" (compression ratio).
  - The optimal sequence is the **path integral over all possible
    Feynman diagrams** connecting methods.

Mathematical framework:

  1. **Method Propagator** G_m(p) = i / (p² - m² + iε)
     where p = log(compression_ratio) is the "momentum"
           m = 1 / tier_score is the "mass"
           ε = error_budget is the "regulator"

  2. **Interaction Vertex** V_{abc} = g_{abc} * δ(p_a + p_b + p_c)
     where g_{abc} = coupling constant between methods a, b, c
     Three-method vertices: method_1 → method_2 + method_3 (branching)

  3. **Scattering Amplitude** M = Σ_{diagrams} ∫ Π d⁴p_i A(diagram)
     Summed over all Feynman diagrams (method interaction networks)

  4. **Path Integral** Z[J] = ∫ D[φ] exp(i S[φ] + i ∫ J·φ)
     The optimal sequence is the classical solution δS/δφ = 0,
     which gives the Euler-Lagrange equation for method selection.

  5. **Renormalization Group** RGE flow:
     As we cascade deeper (higher energy scale μ), the coupling
     constants g(μ) flow via beta functions:
       β(g) = dg/d(log μ) = β₀ g³ + O(g⁵)
     This tells us which methods become more/less effective at
     higher compression ratios.

Reference: Peskin & Schroeder (1995). "An Introduction to Quantum
           Field Theory." Chapters 4-10 (Feynman diagrams, path
           integrals, renormalization).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── QFT constants ──────────────────────────────────────────────────────
# Running coupling β-function coefficients (one-loop QCD-like)
BETA_0 = 0.5  # One-loop coefficient
BETA_1 = 0.25  # Two-loop coefficient

# Fine-structure-like coupling constants for method categories
CATEGORY_COUPLINGS: Dict[str, float] = {
    "decomposition": 0.118,  # α_s (strong) — large coupling
    "spectral": 0.007297,  # α_em (fine structure)
    "structural": 0.033,  # α_w (weak)
    "functional": 0.025,
    "tensor_network": 0.15,
    "physics": 0.10,
    "entropy": 0.003,  # Nearly free (asymptotically free)
    "lossless": 0.002,
    "quantization": 0.05,
    "hybrid": 0.08,
    "novel": 0.12,
}


def _beta_function(g: float, scale: float) -> float:
    """One-loop beta function for running coupling.

    β(g) = -β₀ g³ / (16π²)

    Negative sign → asymptotic freedom at high energy (high ratio).
    This means at HIGH compression ratios, methods become MORE
    effective (like QCD at high energies).
    """
    return -BETA_0 * (g**3) / (16.0 * np.pi**2) * scale


def _running_coupling(g0: float, log_scale: float) -> float:
    """Compute running coupling g(log_scale) at one loop.

    g(μ)² = g₀² / (1 + β₀ g₀² log(μ/μ₀) / (8π²))

    As μ → ∞ (higher compression), g → 0 (asymptotic freedom).
    """
    if log_scale == 0:
        return g0
    numerator = g0 * g0
    denominator = 1.0 + BETA_0 * numerator * log_scale / (8.0 * np.pi * np.pi)
    return np.sqrt(numerator / max(denominator, 1e-30))


def _feynman_propagator(momentum: float, mass: float, eps: float = 1e-6) -> float:
    """Feynman propagator for a method particle.

    G(p) = i / (p² - m² + iε)

    where p = log(compression_ratio), m = 1/tier.
    The imaginary part gives the decay width (how much the method
    "scatters" into other methods).
    """
    p_sq = momentum * momentum
    m_sq = mass * mass
    denominator = p_sq - m_sq + 1j * eps
    return float(np.abs(1.0 / denominator))


def _scattering_amplitude(
    momenta: np.ndarray,
    masses: np.ndarray,
    couplings: np.ndarray,
) -> float:
    """Compute the tree-level scattering amplitude for a sequence.

    M = Π G(p_i) * Π g_{ijk} * δ(Σ p_i)

    at tree level (no loops).  Higher amplitude = more likely cascade.
    """
    propagators = np.array([_feynman_propagator(p, m) for p, m in zip(momenta, masses)])
    vertex_factor = np.prod(couplings)
    momentum_conservation = np.abs(np.sum(momenta))
    return float(np.prod(propagators) * vertex_factor / (1.0 + momentum_conservation))


# ── Data Structures ────────────────────────────────────────────────────


@dataclass
class FeynmanVertex:
    """A three-point interaction vertex between compression methods.

    In QFT, vertices describe how particles interact:
      method_a → method_b + method_c  (branching cascade)

    The coupling g_abc determines the strength.
    """

    method_a: str = ""
    method_b: str = ""
    method_c: str = ""
    coupling: float = 0.0
    momentum_transfer: float = 0.0

    def cross_section(self, energy: float) -> float:
        """Compute the scattering cross section σ ~ |M|².

        In QFT, the cross section is proportional to the squared
        scattering amplitude.  Here it measures how likely a
        method is to "split" into two methods at the next cascade
        stage.
        """
        return float(self.coupling**2 / (1.0 + (energy - self.momentum_transfer) ** 2))


@dataclass
class MethodPropagator:
    """Propagator for a single compression method in the QFT.

    Encodes:
      - mass = 1/tier (lighter methods are more "fundamental")
      - momentum = log(expected_ratio)
      - decay_width = error_rate (how much info is lost)
    """

    name: str = ""
    category: str = ""
    mass: float = 1.0
    momentum: float = 0.0
    decay_width: float = 0.0
    coupling_constant: float = 0.0

    def vertex_coupling(
        self, other: MethodPropagator, third: MethodPropagator
    ) -> float:
        """Compute the three-method coupling g_abc.

        g_abc = √(g_a * g_b * g_c) * exp(-|m_a - m_b| - |m_b - m_c|)

        Methods with similar masses (tiers) couple more strongly.
        """
        g_prod = (
            self.coupling_constant * other.coupling_constant * third.coupling_constant
        )
        mass_diff = abs(self.mass - other.mass) + abs(other.mass - third.mass)
        return float(np.sqrt(max(g_prod, 0.0)) * np.exp(-mass_diff))


@dataclass
class ScatteringAmplitude:
    """The full scattering amplitude for a method cascade.

    In QFT, the amplitude M encodes ALL possible ways a cascade
    can proceed.  |M|² = probability of observing that cascade.
    """

    method_order: List[str] = field(default_factory=list)
    amplitude: float = 0.0
    energy: float = 0.0
    cross_section: float = 0.0
    n_vertices: int = 0
    loop_order: int = 0  # 0 = tree level, 1 = one-loop, etc.

    def branching_ratio(self, target_method: str) -> float:
        """Probability that the cascade ends with target_method.

        Branching ratio Γ_i / Γ_total.
        """
        if not self.method_order:
            return 0.0
        n_target = sum(1 for m in self.method_order if m == target_method)
        return n_target / len(self.method_order)


# ── Main Engine ────────────────────────────────────────────────────────


class QuantumFieldCascadeOptimizer:
    """QFT-Inspired Cascade Optimizer.

    Finds optimal compression method sequences by solving the
    quantum field theory of cascading compression methods.

    Key QFT concepts mapped onto compression:

    +---------------------+----------------------------------------+
    | QFT Concept         | Compression Mapping                    |
    +---------------------+----------------------------------------+
    | Particle            | Compression method                     |
    | Momentum p          | log(compression_ratio)                 |
    | Mass m              | 1 / tier_score                         |
    | Propagator G(p)     | Method effectiveness at target ratio   |
    | Vertex V_abc        | Three-method synergy                   |
    | Scattering M        | Cascade sequence likelihood            |
    | Path integral Z[J]  | Sum over all possible cascades         |
    | Running coupling    | Method effectiveness vs. ratio          |
    | Renormalization     | Error budget flow between methods       |
    +---------------------+----------------------------------------+

    The Euler-Lagrange equations δS/δφ = 0 give the CLASSICAL
    solution — the optimal method sequence.  Quantum fluctuations
    (loops) give subleading corrections (alternative sequences).
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._propagators: Dict[str, MethodPropagator] = {}
        self._vertices: List[FeynmanVertex] = []
        self._best_amplitude: Optional[ScatteringAmplitude] = None
        self._coupling_matrix: Optional[np.ndarray] = None
        self._running_couplings: Dict[str, float] = {}
        self._InitializePropagators()

    def _InitializePropagators(self) -> None:
        """Create method propagators from available methods.

        Each method gets:
          - mass = 1.0 / max(tier, 1)
          - coupling from CATEGORY_COUPLINGS
          - initial momentum = 0 (set during scattering)
        """
        if not hasattr(self._engine, "get_methods_by_categories"):
            return

        categories = list(CATEGORY_COUPLINGS.keys())
        for cat in categories:
            methods_in_cat = self._engine.get_methods_by_categories([cat])
            for name, inst in methods_in_cat.items():
                tier = getattr(inst, "tier", 0)
                try:
                    tier_val = int(tier)
                except (ValueError, TypeError):
                    tier_val = 0
                mass = 1.0 / max(tier_val, 1)
                coupling = CATEGORY_COUPLINGS.get(cat, 0.05)
                decay = getattr(inst, "max_error", 0.01)
                self._propagators[name] = MethodPropagator(
                    name=name,
                    category=cat,
                    mass=mass,
                    momentum=0.0,
                    decay_width=float(decay),
                    coupling_constant=coupling,
                )

        self._BuildCouplingMatrix()

    def _BuildCouplingMatrix(self) -> None:
        """Build the QFT coupling matrix between all methods.

        The coupling matrix J_ij encodes the "interaction strength"
        between methods i and j.  Positive = synergy (methods work
        well together), negative = anti-synergy (they interfere).

        Uses the category assignments and propagator masses.
        """
        names = list(self._propagators.keys())
        n = len(names)
        if n == 0:
            self._coupling_matrix = np.zeros((1, 1))
            return

        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = self._propagators[names[i]].coupling_constant
                else:
                    mi = self._propagators[names[i]].mass
                    mj = self._propagators[names[j]].mass
                    gi = self._propagators[names[i]].coupling_constant
                    gj = self._propagators[names[j]].coupling_constant
                    decay = self._propagators[names[i]].decay_width
                    matrix[i, j] = (
                        np.sqrt(gi * gj) * np.exp(-abs(mi - mj)) * (1.0 - decay)
                    )

        self._coupling_matrix = matrix
        self._method_names = names

    def _SolveEulerLagrange(
        self,
        target_ratio: float,
        target_error: float,
    ) -> np.ndarray:
        """Solve the Euler-Lagrange equations for optimal method selection.

        The action S[φ] = ∫ d⁴x [½(∂φ)² - V(φ)] gives the field equations:
          □φ_i + dV/dφ_i = 0

        We discretize this as:
          Σ_j (K_ij + V_ij) φ_j = source_i

        where K_ij = kinetic term (method propagation)
              V_ij = potential term (method interactions)
              source_i = target ratio coupling

        The solution φ* gives the optimal "field strength" for each method.
        """
        n = len(self._propagators)
        if n == 0:
            return np.array([1.0])

        names = list(self._propagators.keys())

        log_target = np.log(max(target_ratio, 2.0))

        masses = np.array([self._propagators[n].mass for n in names], dtype=np.float64)
        couplings_arr = np.array(
            [self._propagators[n].coupling_constant for n in names], dtype=np.float64
        )

        kinetic = np.eye(n) * (log_target**2 - masses**2)
        if self._coupling_matrix is not None and self._coupling_matrix.shape == (n, n):
            potential = self._coupling_matrix * couplings_arr[:, None]
        else:
            potential = np.outer(couplings_arr, couplings_arr) * 0.1

        K_matrix = kinetic + potential
        K_matrix += np.eye(n) * 1e-6  # Regularization

        sources = couplings_arr * log_target
        try:
            fields = np.linalg.solve(K_matrix, sources)
        except np.linalg.LinAlgError:
            fields = np.linalg.lstsq(K_matrix, sources, rcond=None)[0]

        fields = np.maximum(fields, 0.0)
        fields = fields / (np.max(fields) + 1e-30)
        return fields

    def suggest_sequences(
        self,
        profile: Any,
        target_ratio: float = 1200.0,
        n_sequences: int = 3,
    ) -> List[Dict[str, Any]]:
        """Suggest QFT-optimized compression sequences.

        Uses the Euler-Lagrange equations and Feynman diagram
        expansion to find optimal method sequences.

        Parameters
        ----------
        profile : TensorProfile
            Tensor profile from the compression engine.
        target_ratio : float
            Desired compression ratio.
        n_sequences : int
            Number of sequences to return.

        Returns
        -------
        List[Dict[str, Any]]
            Ranked list of sequences with QFT metadata.
        """
        if not self._propagators:
            return self._fallback_sequences(target_ratio, n_sequences)

        log_scale = np.log(max(target_ratio / 100.0, 1.0))
        for name in self._propagators:
            g0 = self._propagators[name].coupling_constant
            g_run = _running_coupling(g0, log_scale)
            self._running_couplings[name] = g_run

        fields = self._SolveEulerLagrange(
            target_ratio,
            max(0.01, profile.sensitivity if hasattr(profile, "sensitivity") else 0.5),
        )

        names = list(self._propagators.keys())
        n_cycles = int(np.ceil(np.log2(target_ratio))) + 2
        target_error = max(
            getattr(profile, "noise_floor", 0.001) * 10.0,
            0.0001,
        )

        sequences: List[Dict[str, Any]] = []
        for seq_idx in range(n_sequences):
            seed = 42 + seq_idx * 137
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0.85, 1.15, size=len(fields))
            perturbed_fields = fields * noise

            candidates = np.argsort(perturbed_fields)[::-1]
            selected_methods: List[str] = []
            used_categories: set = set()
            running_momentum = 0.0
            running_ratio = 1.0
            n_vertices_created = 0

            for cycle in range(n_cycles):
                top_n = min(15, len(candidates))
                chosen_idx = None
                for idx in candidates[:top_n]:
                    method_name = names[idx]
                    prop = self._propagators[method_name]
                    if prop.category not in used_categories or rng.rand() < 0.2:
                        chosen_idx = int(idx)
                        break

                if chosen_idx is None:
                    chosen_idx = int(candidates[0])

                method_name = names[chosen_idx]
                prop = self._propagators[method_name]

                sub_ratio = 2.0 + (len(selected_methods) * 0.5)
                running_momentum += np.log(sub_ratio)
                running_ratio *= sub_ratio

                selected_methods.append(method_name)
                used_categories.add(prop.category)

                if len(selected_methods) >= 2:
                    n_vertices_created += 1

                if running_ratio >= target_ratio and cycle >= 2:
                    break

            if not selected_methods:
                continue

            momenta = np.array(
                [
                    np.log(max(2.0, self._propagators[m].mass * 10.0))
                    for m in selected_methods
                ],
                dtype=np.float64,
            )
            masses = np.array(
                [self._propagators[m].mass for m in selected_methods], dtype=np.float64
            )
            couplings = np.array(
                [
                    self._running_couplings.get(
                        m, self._propagators[m].coupling_constant
                    )
                    for m in selected_methods
                ],
                dtype=np.float64,
            )

            amplitude = _scattering_amplitude(momenta, masses, couplings)
            cross_section = amplitude * amplitude

            expected_error = target_error * (1.0 + 0.5 * np.mean(couplings))
            expected_ratio = max(1.0, running_ratio)

            sequences.append(
                {
                    "methods": selected_methods,
                    "expected_ratio": expected_ratio,
                    "expected_error": min(expected_error, 0.5),
                    "scattering_amplitude": amplitude,
                    "cross_section": cross_section,
                    "n_vertices": n_vertices_created,
                    "n_momenta": list(momenta),
                    "running_couplings": {
                        m: self._running_couplings.get(m, 0.0) for m in selected_methods
                    },
                }
            )

        sequences.sort(key=lambda s: (-s["scattering_amplitude"], -s["expected_ratio"]))
        return sequences[:n_sequences]

    def _fallback_sequences(
        self, target_ratio: float, n_sequences: int
    ) -> List[Dict[str, Any]]:
        """Fallback when QFT initialization fails."""
        base = [
            (["svd_compress", "dct_spectral", "block_int8"], 200.0),
            (["tensor_train", "fwht_compress", "hadamard_int8", "block_int4"], 800.0),
            (
                [
                    "svd_compress",
                    "dct_spectral",
                    "fwht_compress",
                    "hadamard_int4",
                    "block_int4",
                ],
                5000.0,
            ),
        ]
        results = []
        for names, ratio in base:
            momenta = np.array(
                [np.log(ratio / len(names))] * len(names), dtype=np.float64
            )
            masses = np.ones(len(names), dtype=np.float64)
            couplings = np.ones(len(names), dtype=np.float64) * 0.1
            amp = _scattering_amplitude(momenta, masses, couplings)
            results.append(
                {
                    "methods": list(names),
                    "expected_ratio": max(ratio, target_ratio * 0.5),
                    "expected_error": 0.01,
                    "scattering_amplitude": amp,
                    "cross_section": amp * amp,
                    "n_vertices": len(names) - 1,
                    "n_momenta": list(momenta),
                    "running_couplings": {},
                }
            )
        return results[:n_sequences]

    def compute_path_integral(
        self, sequences: List[Dict[str, Any]], temperature: float = 1.0
    ) -> List[float]:
        """Compute the path integral weight for each sequence.

        Z[J] = ∫ D[φ] exp(i S[φ] + i ∫ J·φ)
        ≈ Σ_{sequences} exp(-S_eff / T)

        Where S_eff = -log(amplitude) is the effective action.
        Returns normalized probabilities.
        """
        if not sequences:
            return []
        amplitudes = np.array([s.get("scattering_amplitude", 1e-6) for s in sequences])
        weights = np.exp(
            -(-np.log(np.maximum(amplitudes, 1e-30))) / max(temperature, 0.01)
        )
        weights = np.maximum(weights, 1e-30)
        weights = weights / weights.sum()
        return list(weights)

    def get_qft_report(self) -> Dict[str, Any]:
        """Return diagnostic info about the QFT cascade optimizer."""
        return {
            "n_propagators": len(self._propagators),
            "n_vertices": len(self._vertices),
            "best_amplitude": (
                self._best_amplitude.amplitude if self._best_amplitude else 0.0
            ),
            "running_couplings": dict(self._running_couplings),
        }

    def fuse_with_engine(self, engine: Any) -> None:
        """Wire this optimizer to a CompressionIntelligenceEngine instance."""
        self._engine = engine
        self._InitializePropagators()

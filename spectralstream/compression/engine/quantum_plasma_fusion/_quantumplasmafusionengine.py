"""
Quantum-Plasma Fusion Engine v3 — Quantum Annealing with Tunneling & Tokamak Cascade

Core architecture — three systems working in concert:

  1. QUANTUM ANNEALING METHOD SELECTOR (Ising model):
     Each compression method category is a spin in an Ising system.
     The Hamiltonian H = -Σ h_i s_i - Σ J_ij s_i s_j + λ_n Σ (s_i - 4)² + λ_r R(s)
     encodes ratio potential, error, tier, and inter-method compatibility.

  2. QUANTUM TUNNELING EXPLORER:
     After standard SA converges, checks if solutions are trapped in local
     optima (low Hamming diversity).  Applies random group spin flips
     (tunneling events) to escape minima — analogous to quantum tunneling
     through energy barriers.

  3. PLASMA PHYSICS CASCADE OPTIMIZER (Tokamak confinement):
     Methods are tokamak magnetic field coils arranged helically around the
     plasma torus (tensor).  Ordering principle:
       - Toroidal field  → decomposition (large-scale confinement)
       - Poloidal field  → spectral/transform (cross-sectional)
       - Helical field   → quantization + structural (combined confinement)
       - Confinement     → entropy (final compression)

  LHC/CERN Analogy:
    Methods = particles    Tensor = beam line
    Hamiltonian = magnetic lattice   Annealing = beam optimization
    Tunneling = quantum effects in superconducting magnets
    Cascade = particle trajectory through the accelerator chain
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._dataclasses import TensorProfile
from ..method_tiers import get_method_tier, tier_score
from ._classic import (
    QuantumStatePreparer,
    QuantumMethodSelector,
    PlasmaWavePropagator,
    SpectralFusionAnalyzer,
)

logger = logging.getLogger(__name__)


# ── Category System ────────────────────────────────────────────────────
# 10 parent categories for the Ising model, with sub-category aliases.

CATEGORY_PARENT: Dict[str, str] = {
    "decomposition": "decomposition",
    "breakthrough_decomposition": "decomposition",
    "spectral": "spectral",
    "breakthrough_signal": "spectral",
    "structural": "structural",
    "functional": "functional",
    "tensor_network": "tensor_network",
    "physics": "physics",
    "revolutionary_gauge": "physics",
    "revolutionary_topological": "physics",
    "novel": "novel",
    "novel_fractal": "novel",
    "breakthrough_math": "novel",
    "breakthrough_info": "novel",
    "entropy": "entropy",
    "lossless": "entropy",
    "quantization": "quantization",
    "transform_quant": "quantization",
    "sparsity_quant": "quantization",
    "delta_quant": "quantization",
    "hybrid": "hybrid",
    "breakthrough_hybrid": "hybrid",
    "cascade": "hybrid",
}

PARENT_CATEGORIES: List[str] = [
    "decomposition",
    "spectral",
    "structural",
    "physics",
    "quantization",
    "entropy",
    "hybrid",
    "functional",
    "tensor_network",
    "novel",
]

CATEGORY_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(PARENT_CATEGORIES)
}

N_CATEGORIES = len(PARENT_CATEGORIES)


# ── Ising Coupling Matrix ──────────────────────────────────────────────
# J[i][j] = coupling between parent categories i and j.
# Positive = ferromagnetic (good to stack), Negative = antiferromagnetic (avoid).
# Diagonal is self-repulsive to prevent same-category duplication.

COUPLING_MATRIX: np.ndarray = np.array(
    [
        # decomp  spec   struc  phys  quant  entr  hybr  func  tnw    novel
        [-0.25, 0.35, 0.25, 0.10, -0.10, 0.20, 0.05, 0.25, 0.30, 0.15],
        [0.35, -0.25, 0.20, 0.10, -0.05, 0.25, 0.10, 0.30, 0.25, 0.20],
        [0.25, 0.20, -0.25, 0.25, -0.15, 0.15, 0.05, 0.20, 0.20, 0.10],
        [0.10, 0.10, 0.25, -0.25, -0.20, 0.10, 0.00, 0.15, 0.30, 0.25],
        [-0.10, -0.05, -0.15, -0.20, -0.25, 0.15, 0.10, -0.10, -0.15, -0.20],
        [0.20, 0.25, 0.15, 0.10, 0.15, -0.25, 0.05, 0.15, 0.15, 0.10],
        [0.05, 0.10, 0.05, 0.00, 0.10, 0.05, -0.25, 0.05, 0.05, 0.05],
        [0.25, 0.30, 0.20, 0.15, -0.10, 0.15, 0.05, -0.25, 0.25, 0.15],
        [0.30, 0.25, 0.20, 0.30, -0.15, 0.15, 0.05, 0.25, -0.25, 0.20],
        [0.15, 0.20, 0.10, 0.25, -0.20, 0.10, 0.05, 0.15, 0.20, -0.25],
    ],
    dtype=np.float64,
)


# ── Baseline Stage Contributions ───────────────────────────────────────
# Typical per-stage compression ratio and error by parent category.

BASE_RATIOS: Dict[str, float] = {
    "decomposition": 50.0,
    "spectral": 5.0,
    "functional": 4.0,
    "tensor_network": 30.0,
    "novel": 10.0,
    "structural": 3.0,
    "physics": 4.0,
    "quantization": 8.0,
    "entropy": 2.0,
    "hybrid": 5.0,
}

BASE_ERRORS: Dict[str, float] = {
    "decomposition": 0.005,
    "spectral": 0.002,
    "functional": 0.003,
    "tensor_network": 0.004,
    "novel": 0.006,
    "structural": 0.003,
    "physics": 0.004,
    "quantization": 0.01,
    "entropy": 0.0,
    "hybrid": 0.005,
}

# ── Tokamak Cascade Order ──────────────────────────────────────────────
# In a tokamak, the helical magnetic field is the sum of:
#   Toroidal (large axis) + Poloidal (cross-section) = Helical confinement
#
# Methods mapped to tokamak coil types:
#   Toroidal coils   → decomposition, tensor_network  (large-scale structure)
#   Poloidal coils   → spectral, functional           (cross-sectional detail)
#   Helical windings → structural, physics, novel     (combined confinement)
#   Divertor         → quantization, hybrid           (exhaust / fine-tune)
#   Blanket          → entropy                        (final insulation)

TOKAMAK_CASCADE_ORDER: List[str] = [
    # Phase 1 — Toroidal field: large-scale structural confinement
    "decomposition",
    "tensor_network",
    # Phase 2 — Poloidal field: cross-sectional detail
    "spectral",
    "functional",
    # Phase 3 — Helical windings: combined confinement / exotic modes
    "structural",
    "novel",
    "physics",
    # Phase 4 — Divertor: exhaust / quantization
    "hybrid",
    "quantization",
    # Phase 5 — Blanket: final entropy confinement
    "entropy",
]

TOKAMAK_PHASE_MAP: Dict[str, int] = {
    name: i for i, name in enumerate(TOKAMAK_CASCADE_ORDER)
}


@dataclass
class AnnealingResult:
    energy: float
    spin_config: np.ndarray
    selected_categories: List[str]
    final_temperature: float


@dataclass
class TunnelEvent:
    """A quantum tunneling event — group spin flip to escape local optima."""

    energy_before: float
    energy_after: float
    n_spins_flipped: int
    accepted: bool


class QuantumPlasmaFusionEngine:
    """
    Production-grade quantum annealing engine for optimal compression
    method sequence selection, augmented with quantum tunneling and
    tokamak-inspired plasma cascade ordering.

    Core innovation: Method selection as Ising model ground-state search:

        H(s) = -Σ h_i s_i - Σ J_ij s_i s_j + λ_n Σ(s_i - 4)² + λ_r R(s) + λ_e E(s)

    where:
    - s_i ∈ {0, 1}: whether parent category i is in the sequence
    - h_i: local field (ratio contribution + profile match of category i)
    - J_ij: coupling (compatibility) between categories
    - λ_n (s_i - 4)²: quadratic sparsity penalty (prefer 4-stage cascades)
    - R(s): shortfall penalty for not meeting target ratio
    - E(s): error accumulation penalty
    """

    def __init__(self):
        self.state_preparer = QuantumStatePreparer()
        self.method_selector = QuantumMethodSelector()
        self.plasma = PlasmaWavePropagator()
        self.spectral = SpectralFusionAnalyzer()
        self._engine: Any = None
        self._multiplicative_stacking: Any = None
        self._stacking_patterns: Any = None
        # Quantum tunneling diagnostics
        self._tunnel_events: List[TunnelEvent] = []

    def fuse_with_engine(self, engine: Any) -> "QuantumPlasmaFusionEngine":
        self._engine = engine
        self._init_multiplicative_stacking()
        return self

    def _init_multiplicative_stacking(self) -> None:
        if self._engine is None:
            return
        try:
            from ..dynamic_tuning.multiplicative_stacking import (
                MultiplicativeStackingEngine,
            )

            self._multiplicative_stacking = MultiplicativeStackingEngine(self._engine)
            self._stacking_patterns = self._multiplicative_stacking.STACKING_PATTERNS
        except Exception as exc:
            logger.debug("MultiplicativeStackingEngine init skipped: %s", exc)
            self._multiplicative_stacking = None
            self._stacking_patterns = None

    # ═══════════════════════════════════════════════════════════════════
    #  PRIMARY API
    # ═══════════════════════════════════════════════════════════════════

    def suggest_sequences(
        self,
        profile: TensorProfile,
        target_ratio: float = 1200.0,
        n_sequences: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find optimal compression method sequences via quantum annealing
        with quantum tunneling escape and tokamak cascade ordering.

        Parameters
        ----------
        profile : TensorProfile
            Tensor profile from CompressionIntelligenceEngine.profile_tensor().
        target_ratio : float
            Desired compression ratio.
        n_sequences : int
            Number of alternative sequences to return.

        Returns
        -------
        list of dict
            Each dict: name, methods, categories, energy,
            expected_ratio, expected_error, n_stages.
        """
        self._tunnel_events = []
        analysis = self._analyze_profile(profile)
        h = self._compute_local_fields(analysis, target_ratio)
        J = self._build_coupling_matrix(analysis)

        h_mean = float(np.mean(np.abs(h)))
        sparsity_lambda = max(h_mean * 1.5, 1.0)

        # --- Phase 1: Standard simulated annealing -----------------------
        results = self._anneal(
            h,
            J,
            sparsity_lambda,
            n_restarts=n_sequences * 6,
            n_steps=3000,
        )

        # --- Phase 2: Quantum tunneling escape from local optima ---------
        results = self._quantum_tunnel(results, h, J, sparsity_lambda)

        # --- Phase 3: Tokamak cascade ordering ---------------------------
        sequences = []
        seen_signatures: set = set()

        for result in results:
            seq = self._decode_sequence(result, profile, target_ratio, analysis)
            if seq is None:
                continue

            sig = tuple(seq["methods"])
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            sequences.append(seq)

            if len(sequences) >= n_sequences * 2:
                break

        # --- Phase 4: Enhance with MultiplicativeStackingEngine ----------
        sequences = self._enhance_with_stacking(sequences, profile, target_ratio)

        sequences.sort(key=lambda s: s["energy"])
        return sequences[:n_sequences]

    # ═══════════════════════════════════════════════════════════════════
    #  QUANTUM TUNNELING EXPLORATION
    # ═══════════════════════════════════════════════════════════════════
    #
    #  Principle: When SA converges, measure the Hamming distance between
    #  unique spin configurations.  If diversity is low (mean distance < 3),
    #  the system is trapped in a local minimum.  "Tunnel" out by flipping
    #  correlated groups of spins — analogous to a quantum particle tunneling
    #  through a potential barrier rather than climbing over it.
    #
    #  Implementation:
    #    1. Compute pairwise Hamming distances among unique spin configs.
    #    2. If mean Hamming < tunnel_threshold, apply N_tunnel tunneling
    #       events: each event flips a random subset of spins (size drawn
    #       from Poisson(λ=2)), accepts if energy decreases or with
    #       Boltzmann probability at an elevated (reheat) temperature.
    #    3. Add any accepted tunneled states to the result pool.

    def _quantum_tunnel(
        self,
        results: List[AnnealingResult],
        h: np.ndarray,
        J: np.ndarray,
        sparsity_lambda: float = 0.05,
        tunnel_threshold: float = 3.0,
        n_tunnel_attempts: int = 30,
    ) -> List[AnnealingResult]:
        if len(results) < 2:
            return results

        configs = np.array([r.spin_config for r in results])
        n = configs.shape[1]

        hamming = np.sum(configs[:, None, :] != configs[None, :, :], axis=-1)
        triu = np.triu_indices(len(configs), k=1)
        mean_hamming = float(np.mean(hamming[triu]))

        if mean_hamming >= tunnel_threshold:
            return results

        n_tunnel = min(
            n_tunnel_attempts,
            max(1, int(n_tunnel_attempts * (1.0 - mean_hamming / tunnel_threshold))),
        )

        for _ in range(n_tunnel):
            parent_rng = np.random.RandomState(int(mean_hamming * 1000 + _ * 137 + 7))
            parent_idx = parent_rng.randint(len(results))
            spin = results[parent_idx].spin_config.copy()

            n_flip = max(1, int(parent_rng.poisson(2.0)))
            flip_idx = parent_rng.choice(n, size=min(n_flip, n), replace=False)
            for fi in flip_idx:
                spin[fi] = 1.0 - spin[fi]

            if float(np.sum(spin)) == 0:
                spin[parent_rng.randint(n)] = 1.0

            energy_before = results[parent_idx].energy
            energy_after = self._ising_energy(
                spin, h, J, sparsity_lambda=sparsity_lambda
            )
            delta_e = energy_after - energy_before
            T_tunnel = 0.5

            accepted = delta_e < 0 or parent_rng.rand() < np.exp(
                -delta_e / max(T_tunnel, 1e-10)
            )

            self._tunnel_events.append(
                TunnelEvent(
                    energy_before=energy_before,
                    energy_after=energy_after,
                    n_spins_flipped=len(flip_idx),
                    accepted=accepted,
                )
            )

            if accepted:
                selected_cats = [
                    name for name, idx in CATEGORY_INDEX.items() if spin[idx] > 0.5
                ]
                results.append(
                    AnnealingResult(
                        energy=energy_after,
                        spin_config=spin.copy(),
                        selected_categories=selected_cats,
                        final_temperature=0.0,
                    )
                )

        results.sort(key=lambda r: r.energy)
        unique_results, seen = [], set()
        for r in results:
            sig = tuple(int(s) for s in r.spin_config)
            if sig in seen:
                continue
            seen.add(sig)
            unique_results.append(r)

        return unique_results

    # ═══════════════════════════════════════════════════════════════════
    #  TOKAMAK PLASMA CASCADE OPTIMIZER
    # ═══════════════════════════════════════════════════════════════════
    #
    #  Principle: In a tokamak fusion reactor, plasma confinement is
    #  achieved by a HELICAL magnetic field — the SUM of toroidal and
    #  poloidal components creates a rotating transform that confines
    #  particles along nested flux surfaces.
    #
    #  Mapping to compression:
    #    Toroidal field coils  → Decomposition / TensorNetwork
    #      (large-scale structure, like the main toroidal field)
    #    Poloidal field coils  → Spectral / Functional
    #      (cross-sectional detail, fine-scale structure)
    #    Helical windings      → Structural / Novel / Physics
    #      (combination of both, exotic confinement modes)
    #    Divertor plates       → Quantization / Hybrid
    #      (exhaust, edge-localized modes, fine-tuning)
    #    Blanket / Shield      → Entropy
    #      (final neutron shielding, lossless wrap)
    #
    #  The cascade ordering follows the tokamak assembly sequence:
    #  toroidal → poloidal → helical → divertor → blanket.
    #  Each stage compresses the residual of the previous one, exactly
    #  as each magnetic coil builds on the confinement of the previous.

    @staticmethod
    def _tokamak_order(categories: List[str]) -> List[str]:
        """Order categories by the tokamak cascade sequence.

        The helical transform of the tokamak means each method type
        builds on the residual confinement of the previous:
          Toroidal → Poloidal → Helical → Divertor → Blanket
        """
        ordered = sorted(
            categories,
            key=lambda c: TOKAMAK_PHASE_MAP.get(c, 999),
        )
        return ordered

    def _plasma_cascade_predict(
        self,
        ordered_categories: List[str],
        analysis: Dict[str, Any],
        target_ratio: float,
    ) -> Tuple[float, float]:
        """Predict the multiplicative ratio and additive error for a
        tokamak-ordered cascade, using diminishing-returns physics.

        In a tokamak, each additional coil system contributes less to
        confinement (the β_limit).  Similarly, each additional compression
        stage contributes less to the ratio (diminishing returns).
        """
        profile = analysis.get("profile")
        shape = getattr(profile, "shape", None) if profile else None
        if shape and isinstance(shape, (tuple, list)) and len(shape) >= 2:
            min_dim = min(shape[0], shape[-1])
        else:
            min_dim = max(getattr(profile, "n_elements", 512), 1)
        er_norm = analysis["effective_rank"] / max(min_dim, 1)
        er_norm = min(max(er_norm, 0.01), 1.0)

        predicted_ratio = 1.0
        predicted_error = 0.0

        for i, cat in enumerate(ordered_categories):
            cat_idx = TOKAMAK_PHASE_MAP.get(cat, 999)
            ratio = BASE_RATIOS.get(cat, 5.0)
            error = BASE_ERRORS.get(cat, 0.005)

            if cat in ("decomposition", "tensor_network"):
                ratio *= 1.0 / max(er_norm, 0.01)

            # Diminishing returns: each subsequent stage in the
            # tokamak cascade has less residual to work with.
            phase_factor = 0.85 ** (i % 5)
            position_harmonic = 1.0 / (1.0 + 0.15 * i)
            effective_factor = phase_factor * position_harmonic

            ratio = max(ratio * effective_factor, 1.2)
            error *= 1.0 + 0.2 * i

            predicted_ratio *= ratio
            predicted_error += error

        return predicted_ratio, predicted_error

    # ═══════════════════════════════════════════════════════════════════
    #  PROFILE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    def _analyze_profile(self, profile: TensorProfile) -> Dict[str, Any]:
        er = getattr(profile, "effective_rank", 0.5)
        ec = getattr(profile, "energy_concentration", 0.5)
        se = getattr(profile, "spectral_entropy", 0.5)
        sdr = getattr(profile, "spectral_decay_rate", 0.5)
        noise = getattr(profile, "noise_floor", 0.01)

        analysis: Dict[str, Any] = {
            "profile": profile,
            "effective_rank": er,
            "spectral_decay_rate": sdr,
            "energy_concentration": ec,
            "spectral_entropy": se,
            "entropy_rate": getattr(profile, "entropy_rate", 0.5),
            "noise_floor": noise,
            "sensitivity": getattr(profile, "sensitivity", 0.5),
            "von_neumann_entropy": min(1.0, se * (1.0 + er)),
            "purity": 1.0 - min(1.0, ec * 0.5),
            "schmidt_rank": max(1, int(er * 100)),
            "energy_gap": max(0.01, 1.0 - sdr),
            "alfven_mode": min(1.0, max(0.0, ec)),
            "acoustic_mode": min(1.0, max(0.0, 1.0 - ec)),
            "whistler_mode": min(1.0, max(0.0, se)),
            "decay_rate": sdr,
            "dct_efficiency": min(1.0, max(0.0, ec)),
            "fft_low_freq_ratio": min(1.0, max(0.0, ec * 0.8)),
        }

        qstate = type(
            "QState",
            (),
            {
                "von_neumann_entropy": analysis["von_neumann_entropy"],
                "purity": analysis["purity"],
                "schmidt_rank": analysis["schmidt_rank"],
                "entanglement_entropy": analysis["von_neumann_entropy"] * 0.5,
                "energy_gap": analysis["energy_gap"],
                "ground_state_energy": 0.05,
            },
        )()

        analysis["method_overlaps"] = QuantumMethodSelector.compute_method_overlaps(
            qstate
        )
        return analysis

    # ═══════════════════════════════════════════════════════════════════
    #  ISING MODEL CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════

    def _compute_local_fields(
        self, analysis: Dict[str, Any], target_ratio: float
    ) -> np.ndarray:
        h = np.zeros(N_CATEGORIES, dtype=np.float64)
        overlaps = analysis["method_overlaps"]
        profile = analysis.get("profile")
        shape = getattr(profile, "shape", None) if profile else None
        if shape and isinstance(shape, (tuple, list)) and len(shape) >= 2:
            min_dim = min(shape[0], shape[-1])
        else:
            min_dim = max(getattr(profile, "n_elements", 512) if profile else 512, 1)
        er = min(max(analysis["effective_rank"] / max(min_dim, 1), 0.01), 1.0)
        ec = analysis["energy_concentration"]
        se = analysis["spectral_entropy"]
        noise = analysis["noise_floor"]

        for parent, idx in CATEGORY_INDEX.items():
            overlap = overlaps.get(parent, {}).get("overlap", 0.5)

            if parent == "decomposition":
                ratio_potential = min(1.0, 1.0 / max(er, 0.01))
                error_factor = 1.0 - noise
            elif parent == "spectral":
                ratio_potential = min(1.0, ec * 2.0)
                error_factor = 0.9 - noise * 0.5
            elif parent == "quantization":
                ratio_potential = min(1.0, 32.0 / max(target_ratio, 1.0))
                error_factor = 0.3
            elif parent == "entropy":
                ratio_potential = 0.3
                error_factor = 1.0
            elif parent == "structural":
                ratio_potential = 0.4 * (1.0 - er)
                error_factor = 0.8
            elif parent == "physics":
                ratio_potential = 0.3 * (1.0 - er)
                error_factor = 0.7
            elif parent == "hybrid":
                ratio_potential = 0.5
                error_factor = 0.6
            elif parent == "functional":
                ratio_potential = 0.3 * se
                error_factor = 0.8
            elif parent == "tensor_network":
                ratio_potential = 0.6 * (1.0 - er)
                error_factor = 0.7
            elif parent == "novel":
                ratio_potential = 0.4
                error_factor = 0.6
            else:
                ratio_potential = 0.3
                error_factor = 0.6

            tier = get_method_tier(parent, parent)
            tier_factor = tier_score(tier) / 10.0

            # Tokamak phase boost: earlier phases (toroidal) get a
            # higher local field because they contribute more fundamental
            # compression (large-scale structure).
            phase_order = TOKAMAK_PHASE_MAP.get(parent, 999)
            phase_boost = 1.0 + max(0.0, (9 - phase_order) * 0.03)

            h[idx] = (
                overlap * 0.20
                + ratio_potential * 0.40
                + error_factor * 0.25
                + tier_factor * 0.15
            ) * phase_boost

        ratio_scale = np.log2(max(target_ratio, 10.0)) / 10.0
        if ratio_scale > 0.5:
            decomp_scale = min(2.0, 1.0 + ratio_scale)
            spec_scale = min(1.5, 1.0 + ratio_scale * 0.5)
            h[CATEGORY_INDEX["decomposition"]] *= decomp_scale
            h[CATEGORY_INDEX["spectral"]] *= spec_scale
            h[CATEGORY_INDEX["tensor_network"]] *= min(1.5, 1.0 + ratio_scale * 0.4)

        return h

    def _build_coupling_matrix(self, analysis: Dict[str, Any]) -> np.ndarray:
        J = COUPLING_MATRIX.copy()
        profile = analysis.get("profile")
        shape = getattr(profile, "shape", None) if profile else None
        if shape and isinstance(shape, (tuple, list)) and len(shape) >= 2:
            min_dim = min(shape[0], shape[-1])
        else:
            min_dim = max(getattr(profile, "n_elements", 512) if profile else 512, 1)
        er = min(max(analysis["effective_rank"] / max(min_dim, 1), 0.01), 1.0)
        se = analysis["spectral_entropy"]
        noise = analysis["noise_floor"]

        if er < 0.3:
            J[0, :] *= 1.2
            J[:, 0] *= 1.2
            J[0, 0] = -0.5

        if se > 0.6:
            J[1, :] *= 1.15
            J[:, 1] *= 1.15
            J[1, 1] = -0.5

        if noise > 0.05:
            J[4, :] = np.where(J[4, :] < 0, J[4, :] * 0.5, J[4, :])
            J[:, 4] = np.where(J[:, 4] < 0, J[:, 4] * 0.5, J[:, 4])

        # Tokamak-aware coupling enhancement:
        # Phases close in the tokamak sequence get slightly boosted coupling.
        for i, name_i in enumerate(PARENT_CATEGORIES):
            for j, name_j in enumerate(PARENT_CATEGORIES):
                if i == j:
                    continue
                phase_i = TOKAMAK_PHASE_MAP.get(name_i, 999)
                phase_j = TOKAMAK_PHASE_MAP.get(name_j, 999)
                phase_dist = abs(phase_i - phase_j)
                if 1 <= phase_dist <= 2 and J[i, j] > 0:
                    J[i, j] *= 1.1

        J = (J + J.T) / 2.0
        np.fill_diagonal(J, np.diagonal(J) * 0.5)
        return J

    # ═══════════════════════════════════════════════════════════════════
    #  HAMILTONIAN ENERGY
    # ═══════════════════════════════════════════════════════════════════

    def _ising_energy(
        self,
        spin: np.ndarray,
        h: np.ndarray,
        J: np.ndarray,
        sparsity_lambda: float = 0.0,
        target_ratio: float = 1200.0,
        penalty_scale: float = 0.0,
    ) -> float:
        field_term = -np.dot(h, spin)
        coupling_term = -0.5 * np.dot(spin, np.dot(J, spin))
        n_selected = float(np.sum(spin))
        sparsity_term = sparsity_lambda * (n_selected - 4.0) ** 2

        penalty_term = 0.0
        if penalty_scale > 0:
            selected = [name for name, idx in CATEGORY_INDEX.items() if spin[idx] > 0.5]
            if selected:
                predicted_ratio = 1.0
                for cat in selected:
                    base = BASE_RATIOS.get(cat, 5.0)
                    predicted_ratio *= base
                if predicted_ratio < target_ratio:
                    penalty_term = penalty_scale * (
                        1.0 - predicted_ratio / target_ratio
                    )

        return float(field_term + coupling_term + sparsity_term + penalty_term)

    # ═══════════════════════════════════════════════════════════════════
    #  SIMULATED ANNEALING
    # ═══════════════════════════════════════════════════════════════════

    def _anneal(
        self,
        h: np.ndarray,
        J: np.ndarray,
        sparsity_lambda: float = 0.05,
        n_restarts: int = 20,
        n_steps: int = 3000,
        t_min: float = 0.01,
        t_max: float = 10.0,
    ) -> List[AnnealingResult]:
        n = len(h)
        results: List[AnnealingResult] = []

        # Adaptive cooling schedule based on the number of categories.
        # More categories need slower cooling.
        n_schedule = max(1, n - 6)
        t_max_adaptive = max(1.0, t_max * (1.0 + 0.1 * n_schedule))

        for restart in range(n_restarts):
            seed = restart * 137 + 7
            rng = np.random.RandomState(seed)
            spin = rng.randint(0, 2, size=n).astype(np.float64)

            if np.sum(spin) == 0:
                spin[rng.randint(n)] = 1.0

            current_energy = self._ising_energy(
                spin, h, J, sparsity_lambda=sparsity_lambda
            )

            for step in range(n_steps):
                frac = step / n_steps
                T = t_max_adaptive * (t_min / t_max_adaptive) ** frac

                i = rng.randint(n)
                spin[i] = 1.0 - spin[i]
                new_energy = self._ising_energy(
                    spin, h, J, sparsity_lambda=sparsity_lambda
                )
                delta_e = new_energy - current_energy

                if delta_e < 0 or rng.rand() < np.exp(-delta_e / max(T, 1e-10)):
                    current_energy = new_energy
                else:
                    spin[i] = 1.0 - spin[i]

            # Reheat + fast-cool for diversity
            for _ in range(200):
                i = rng.randint(n)
                spin[i] = 1.0 - spin[i]
                new_energy = self._ising_energy(
                    spin, h, J, sparsity_lambda=sparsity_lambda
                )
                delta_e = new_energy - current_energy
                T_reheat = t_max_adaptive * 0.3
                if delta_e < 0 or rng.rand() < np.exp(-delta_e / max(T_reheat, 1e-10)):
                    current_energy = new_energy
                else:
                    spin[i] = 1.0 - spin[i]

            final_energy = self._ising_energy(
                spin, h, J, sparsity_lambda=sparsity_lambda
            )
            selected_cats = [
                name for name, idx in CATEGORY_INDEX.items() if spin[idx] > 0.5
            ]

            results.append(
                AnnealingResult(
                    energy=final_energy,
                    spin_config=spin.copy(),
                    selected_categories=selected_cats,
                    final_temperature=T,
                )
            )

        results.sort(key=lambda r: r.energy)
        unique_results, seen = [], set()
        for r in results:
            sig = tuple(int(s) for s in r.spin_config)
            if sig in seen:
                continue
            seen.add(sig)
            unique_results.append(r)

        return unique_results

    # ═══════════════════════════════════════════════════════════════════
    #  SEQUENCE DECODING (Tokamak-Ordered)
    # ═══════════════════════════════════════════════════════════════════

    def _decode_sequence(
        self,
        result: AnnealingResult,
        profile: TensorProfile,
        target_ratio: float,
        analysis: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not result.selected_categories:
            return None

        # Tokamak cascade ordering instead of tier-only ordering.
        # This arranges methods in the helical confinement sequence:
        #   Toroidal → Poloidal → Helical → Divertor → Blanket
        ordered = self._tokamak_order(result.selected_categories)

        methods_list: List[str] = []
        if self._engine is not None:
            for parent in ordered:
                method = self._pick_best_method(parent, analysis)
                if method:
                    methods_list.append(method)
        else:
            methods_list = ordered[:]

        if not methods_list:
            return None

        # Use tokamak cascade prediction for expected ratio/error.
        expected_ratio, expected_error = self._plasma_cascade_predict(
            ordered, analysis, target_ratio
        )

        energy = (
            result.energy
            + max(0.0, 1.0 - expected_ratio / max(target_ratio, 1.0)) * 5.0
        )

        # Name encodes the tokamak phase sequence.
        name = "QPF_Tokamak_" + "_".join(c[:4] for c in ordered)

        return {
            "name": name,
            "methods": methods_list,
            "categories": ordered,
            "energy": energy,
            "expected_ratio": expected_ratio,
            "expected_error": expected_error,
            "n_stages": len(ordered),
        }

    def _pick_best_method(
        self,
        parent_category: str,
        analysis: Dict[str, Any],
    ) -> Optional[str]:
        if self._engine is None:
            return None

        sub_cats = {sc for sc, pc in CATEGORY_PARENT.items() if pc == parent_category}

        overlap = (
            analysis.get("method_overlaps", {})
            .get(parent_category, {})
            .get("overlap", 0.5)
        )
        methods = self._engine._methods

        candidates: List[Tuple[str, float]] = []
        for name, inst in methods.items():
            cat = getattr(inst, "category", "").lower()
            if cat in sub_cats:
                tier = get_method_tier(name, cat)
                tf = tier_score(tier) / 10.0
                score = overlap * 0.6 + tf * 0.4
                candidates.append((name, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    # ═══════════════════════════════════════════════════════════════════
    #  MULTIPLICATIVE STACKING ENHANCEMENT
    # ═══════════════════════════════════════════════════════════════════
    #
    #  After annealing selects categories and tokamak ordering arranges
    #  them, feed the candidate sequences into MultiplicativeStackingEngine
    #  for refinement.  The stacking engine provides accurate expected
    #  ratios/errors based on real method parameters, not just base estimates.

    def _enhance_with_stacking(
        self,
        sequences: List[Dict[str, Any]],
        profile: TensorProfile,
        target_ratio: float,
    ) -> List[Dict[str, Any]]:
        if self._multiplicative_stacking is None:
            return sequences

        try:
            pattern_names = list(self._stacking_patterns.keys())

            enhanced: List[Dict[str, Any]] = []
            seen_sigs: set = set()

            for seq in sequences:
                categories = seq.get("categories", [])
                if not categories:
                    enhanced.append(seq)
                    continue

                method_types = []
                for cat in categories:
                    mt = self._category_to_method_type(cat)
                    if mt:
                        method_types.append(mt)

                if not method_types:
                    enhanced.append(seq)
                    continue

                # Check if any stacking pattern matches our method types.
                best_pattern = None
                best_match_ratio = 0.0
                for pname in pattern_names:
                    pattern = self._stacking_patterns[pname]
                    ptypes = [s["method_type"] for s in pattern.get("stages", [])]
                    match = sum(1 for mt in method_types if mt in ptypes) / max(
                        len(method_types), 1
                    )
                    if match > best_match_ratio:
                        best_match_ratio = match
                        best_pattern = pname

                if best_pattern and best_match_ratio >= 0.5:
                    pattern = self._stacking_patterns[best_pattern]
                    enhanced_seq = {
                        **seq,
                        "name": f"QPF_Stacked_{best_pattern}",
                        "expected_ratio": pattern.get(
                            "expected_ratio", seq["expected_ratio"]
                        ),
                        "expected_error": pattern.get(
                            "expected_error", seq["expected_error"]
                        ),
                        "stacking_pattern": best_pattern,
                    }
                else:
                    enhanced_seq = seq

                sig = tuple(enhanced_seq["methods"])
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    enhanced.append(enhanced_seq)

            return enhanced if enhanced else sequences

        except Exception as exc:
            logger.debug("Stacking enhancement skipped: %s", exc)
            return sequences

    @staticmethod
    def _category_to_method_type(category: str) -> Optional[str]:
        mapping = {
            "decomposition": "decomposition",
            "tensor_network": "decomposition",
            "spectral": "spectral",
            "functional": "spectral",
            "structural": "structural",
            "physics": "structural",
            "novel": "structural",
            "quantization": "quantization",
            "hybrid": "quantization",
            "entropy": "entropy",
        }
        return mapping.get(category)

    # ═══════════════════════════════════════════════════════════════════
    #  DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════

    @property
    def tunnel_events(self) -> List[TunnelEvent]:
        return list(self._tunnel_events)

    @property
    def n_tunnel_events(self) -> int:
        return len(self._tunnel_events)

    @property
    def tunnel_acceptance_rate(self) -> float:
        if not self._tunnel_events:
            return 0.0
        return sum(1 for e in self._tunnel_events if e.accepted) / len(
            self._tunnel_events
        )

    # ═══════════════════════════════════════════════════════════════════
    #  BACKWARD-COMPATIBLE API (classic fusion engine)
    # ═══════════════════════════════════════════════════════════════════

    def analyze_tensor(self, tensor: np.ndarray, name: str = "") -> Dict[str, Any]:
        state = self.state_preparer.estimate_state_fast(tensor)
        method_overlaps = self.method_selector.compute_method_overlaps(state)
        plasma_modes = self.plasma.compute_plasma_dispersion(tensor)
        spectral_fingerprint = self.spectral.full_spectral_analysis(tensor)

        return {
            "name": name,
            "shape": tensor.shape,
            "nbytes": tensor.nbytes,
            "quantum": {
                "von_neumann_entropy": state.von_neumann_entropy,
                "purity": state.purity,
                "schmidt_rank": state.schmidt_rank,
                "entanglement_entropy": state.entanglement_entropy,
                "energy_gap": state.energy_gap,
                "ground_state_energy": state.ground_state_energy,
            },
            "method_overlaps": method_overlaps,
            "plasma": plasma_modes,
            "spectral": spectral_fingerprint,
            "compressibility_score": self._compute_fused_score(
                state, plasma_modes, spectral_fingerprint
            ),
            "recommended_categories": self.method_selector.select_top_categories(
                state, 3
            ),
        }

    def _compute_fused_score(self, state, plasma: Dict, spectral: Dict) -> float:
        score = 0.0
        if state.von_neumann_entropy > 0:
            score += 0.3 * min(1.0, state.von_neumann_entropy / 5.0)
        dct_eff = spectral.get("dct_efficiency", 0.5)
        score += 0.3 * dct_eff
        alfven = plasma.get("alfven_mode", 0.5)
        score += 0.2 * alfven
        score += 0.1 * min(1.0, state.energy_gap * 10)
        score += 0.1 * (1.0 - state.purity)
        return min(score, 1.0)

    def rank_methods(
        self,
        tensor: np.ndarray,
        available_methods: Dict[str, Dict],
        top_k: int = 15,
    ) -> List[Tuple[str, float, str]]:
        analysis = self.analyze_tensor(tensor)
        overlaps = analysis["method_overlaps"]
        ranked = []

        for method_name, info in available_methods.items():
            category = info.get("category", "quantization")
            cat_overlap = overlaps.get(category, {}).get("overlap", 0.5)
            plasma_modes = analysis["plasma"]

            if "spectral" in category or "transform" in category:
                plasma_match = plasma_modes.get("alfven_mode", 0.5)
            elif "decomposition" in category or "low_rank" in category:
                plasma_match = plasma_modes.get("acoustic_mode", 0.5)
            else:
                plasma_match = plasma_modes.get("whistler_mode", 0.5)

            spectral_eff = analysis["spectral"].get("dct_efficiency", 0.5)
            score = cat_overlap * 0.5 + plasma_match * 0.3 + spectral_eff * 0.2

            tier = get_method_tier(method_name, category)
            tb = tier.value if hasattr(tier, "value") else tier
            tier_bonus = {1: 1.5, 2: 1.3, 3: 1.1, 4: 1.0, 5: 0.5}
            score *= tier_bonus.get(tb, 1.0)

            ranked.append((method_name, score, category))

        ranked.sort(key=lambda x: -x[1])
        return ranked[:top_k]


def fuse_with_engine(engine: Any) -> QuantumPlasmaFusionEngine:
    """Module-level hook for backward compatibility."""
    qpfe = QuantumPlasmaFusionEngine()
    qpfe.fuse_with_engine(engine)
    logger.info("QuantumPlasmaFusionEngine (v3 — tunneling + tokamak) integrated")
    return qpfe

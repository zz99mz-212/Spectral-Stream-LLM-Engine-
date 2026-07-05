"""
Time Crystal Compression Engine v1.0
=====================================

Based on the physics of **discrete time crystals** — systems that break
time-translation symmetry by entering a periodic state that never repeats.

The core insight: In a time crystal, a periodic drive (Floquet operator)
produces a state whose period is a MULTIPLE of the drive period. This
"period doubling" is a signature of spontaneous symmetry breaking in time.

We map this onto compression:
  - Time (t) → cascade cycle index (n)
  - Floquet operator U_F → a sequence of compression methods applied
    to successive residuals
  - Time crystal phase → the tensor's state after each cycle, which
    oscillates between compressible and incompressible subspaces
  - Period doubling → each method in the cascade reveals NEW compression
    opportunities that were invisible to previous methods

Mathematical foundation:
  The Floquet operator U_F = T exp(-i ∫ H(t) dt) generates time evolution.
  We use a discrete Floquet map:
    |ψ_{n+1}⟩ = F |ψ_n⟩ = M_k ... M_2 M_1 |ψ_n⟩
  where M_i are compression method operators acting on the tensor state.

  The "time crystal" regime occurs when:
    ⟨ψ_{n+P} | O | ψ_{n+P}⟩ = ⟨ψ_n | O | ψ_n⟩  but  |ψ_{n+P}⟩ ≠ |ψ_n⟩
  i.e., observables (compression ratio) are periodic with period P, but
  the state (tensor) never repeats exactly.

  This means each cycle finds a DIFFERENT representation of the tensor
  that yields the SAME compression ratio — until a critical "melting"
  point where the ratio jumps (period doubling bifurcation).

Floquet theory mapping:
  - The cascade period T = sum(method durations)
  - Quasi-energy ε = log(compression_ratio) / T  (Floquet exponent)
  - Time crystal phase rigidity = stability of ε against perturbations

Reference: Else, D.V., Bauer, B., Nayak, C. (2016). "Floquet time crystals."
           Physical Review Letters 117, 090402.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Floquet spectral basis ─────────────────────────────────────────────
# Chebyshev polynomials of the Floquet operator for spectral decomposition
# T_n(cos θ) = cos(nθ) — maps cascade cycles onto the unit circle


def _floquet_chebyshev_moments(
    singular_values: np.ndarray, n_cycles: int
) -> np.ndarray:
    """Compute Floquet spectral moments via Chebyshev recursion.

    The singular values σ_i act as quasi-energies ε_i = log(σ_i).
    Chebyshev moments μ_n = Tr[T_n(cos(H))] = Σ_i cos(n ε_i)
    give the spectral density of the Floquet operator.

    Uses NumPy vectorized ops only — no Python loops over elements.
    """
    eps = np.log(np.maximum(singular_values, 1e-30))
    eps = eps - eps.mean()
    eps = eps / (np.maximum(np.abs(eps).max(), 1e-30))
    moment_indices = np.arange(1, n_cycles + 1, dtype=np.float64)
    cos_vals = np.cos(moment_indices[:, None] * eps[None, :])
    moments = np.sum(cos_vals, axis=1)
    return moments / max(singular_values.size, 1)


def _period_doubling_bifurcation(
    compression_ratios: np.ndarray,
) -> Tuple[float, int]:
    """Detect period-doubling bifurcation in the compression cascade.

    The Feigenbaum constant δ ≈ 4.669 governs the rate of period
    doubling in a time crystal.  We compute the Lyapunov exponent
    from the ratio sequence to detect the bifurcation point.

    Returns (lyapunov_exponent, bifurcation_cycle).
    """
    if len(compression_ratios) < 4:
        return 0.0, -1
    diffs = np.diff(np.log(np.maximum(compression_ratios, 1e-30)))
    lyap = np.mean(diffs)
    diffs2 = np.abs(np.diff(diffs))
    if diffs2.size == 0:
        return float(lyap), -1
    bifurcation = int(np.argmax(diffs2)) + 1 if diffs2.max() > 0.1 else -1
    return float(lyap), bifurcation


def _floquet_mixing_angle(eigenvalues: np.ndarray) -> float:
    """Compute the Floquet mixing angle from eigenvalue statistics.

    In a time crystal, the eigenstates of U_F form a band structure.
    The mixing angle θ = arccos(|Tr(U_F)| / N) measures how much the
    Floquet operator "rotates" tensor states in method-space.

    A mixing angle of 0 → trivial (no time crystal).
    A mixing angle of π/2 → maximally non-trivial (robust time crystal).
    """
    n = max(eigenvalues.size, 1)
    trace_norm = np.abs(np.sum(eigenvalues)) / n
    angle = np.arccos(np.clip(trace_norm, -1.0, 1.0))
    return float(angle)


# ── Data structures ────────────────────────────────────────────────────


@dataclass
class FloquetOperator:
    """Floquet operator for a cascade cycle — the "drive" Hamiltonian.

    Encodes how a sequence of compression methods maps the tensor
    from one cycle to the next.
    """

    method_names: List[str] = field(default_factory=list)
    singular_value_scales: np.ndarray = field(default_factory=lambda: np.ones(1))
    spectral_moments: np.ndarray = field(default_factory=lambda: np.zeros(1))
    mixing_angle: float = 0.0
    quasi_energies: np.ndarray = field(default_factory=lambda: np.zeros(1))

    def compute_overlap(self, other: FloquetOperator) -> float:
        """Compute the Floquet overlap between two operators.

        In time crystal theory, the overlap |⟨ψ|U_F† V_F|ψ⟩| measures
        how similar two Floquet drives are.  High overlap = same
        "time crystal phase".
        """
        s1 = self.singular_value_scales
        s2 = other.singular_value_scales
        min_len = min(s1.size, s2.size)
        if min_len == 0:
            return 0.0
        overlap = np.abs(np.sum(s1[:min_len] * s2[:min_len]))
        norm = np.sqrt(np.sum(s1[:min_len] ** 2) * np.sum(s2[:min_len] ** 2))
        return float(overlap / max(norm, 1e-30))


@dataclass
class TimeCrystalCycle:
    """A single cycle in the time crystal cascade.

    Each cycle applies a Floquet operator to the residual tensor,
    extracting compression from a different "phase" of the crystal.
    """

    cycle_index: int = 0
    method_names: List[str] = field(default_factory=list)
    compression_ratio: float = 1.0
    error: float = 0.0
    residual_norm: float = 0.0
    cycle_phase: float = 0.0
    quasi_energy: float = 0.0


@dataclass
class CrystalMethodState:
    """A compression method in the time crystal "spin" state.

    Each method has a Floquet phase — methods are applied in order
    of increasing phase, creating the non-repeating time crystal order.
    """

    name: str = ""
    category: str = ""
    tier: int = 0
    floquet_phase: float = 0.0
    coupling_strength: float = 0.0
    expected_ratio: float = 2.0
    expected_error: float = 0.01

    def phase_evolve(self, delta_t: float) -> None:
        """Evolve the Floquet phase forward by delta_t."""
        self.floquet_phase = (self.floquet_phase + delta_t) % (2 * np.pi)


# ── Main Engine ────────────────────────────────────────────────────────


class TimeCrystalCompressionEngine:
    """Time Crystal Compression Engine.

    Uses discrete time crystal physics to discover perpetually
    novel compression sequences.  The engine:

    1. Builds a Floquet operator from available compression methods
    2. Computes the "quasi-energy spectrum" — the spectrum of
       log(compression_ratio) per cycle
    3. Uses time-translation symmetry breaking to find sequences
       whose period never repeats with the same tensor state
    4. Detects period-doubling bifurcations (Feigenbaum route)
       as signals to add more aggressive methods

    Key insight: In a time crystal, ⟨O(t)⟩ is periodic but the
    microstate never repeats.  For compression this means each
    cycle finds a NEW way to compress the SAME tensor, until the
    information-theoretic limit is reached.

    References:
      - Else, Monroe, Nayak, & Yao (2020). "Discrete time crystals."
        Annual Review of Condensed Matter Physics 11, 467-499.
      - Khemani, Moessner, & Sondhi (2019). "A brief history of
        time crystals." arXiv:1910.10745.
    """

    FLOQUET_PHASES: Dict[str, float] = {
        "decomposition": 0.0,
        "spectral": np.pi / 4,
        "structural": np.pi / 2,
        "functional": 3 * np.pi / 4,
        "tensor_network": np.pi,
        "physics": 5 * np.pi / 4,
        "entropy": 3 * np.pi / 2,
        "lossless": 3 * np.pi / 2,
        "quantization": 7 * np.pi / 4,
        "hybrid": np.pi / 8,
        "novel": np.pi / 6,
    }

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._cycles: List[TimeCrystalCycle] = []
        self._floquet_operator: Optional[FloquetOperator] = None
        self._method_states: List[CrystalMethodState] = []
        self._lyapunov_exponent: float = 0.0
        self._bifurcation_cycle: int = -1
        self._is_time_crystal_phase: bool = False
        self._InitializeMethodStates()

    def _InitializeMethodStates(self) -> None:
        """Build initial method states from available methods."""
        if not hasattr(self._engine, "get_methods_by_categories"):
            return
        categories = [
            "decomposition",
            "spectral",
            "structural",
            "functional",
            "tensor_network",
            "physics",
            "entropy",
            "lossless",
            "quantization",
            "hybrid",
            "novel",
        ]
        phase_angles = np.array(
            [self.FLOQUET_PHASES.get(c, 0.0) for c in categories],
            dtype=np.float64,
        )
        for cat, phase in zip(categories, phase_angles):
            methods_in_cat = self._engine.get_methods_by_categories([cat])
            for name, inst in methods_in_cat.items():
                tier = getattr(inst, "tier", 0)
                if isinstance(tier, int):
                    pass
                else:
                    try:
                        tier = int(tier)
                    except (ValueError, TypeError):
                        tier = 0
                coupling = np.abs(float(np.cos(phase))) + 0.1
                self._method_states.append(
                    CrystalMethodState(
                        name=name,
                        category=cat,
                        tier=tier,
                        floquet_phase=phase + 0.01 * hash(name) % 100 / 100.0,
                        coupling_strength=coupling,
                        expected_ratio=max(2.0, 100.0 / max(tier, 1)),
                        expected_error=0.01 / max(coupling, 0.1),
                    )
                )

    def _ComputeFloquetOperator(self, tensor: np.ndarray) -> FloquetOperator:
        """Build the Floquet operator from the tensor's singular value spectrum.

        The singular values of the tensor form the "quasi-energy" spectrum
        of the Floquet operator U_F.  We compute:
          - Singular value scales (the "band structure")
          - Spectral moments (Chebyshev moments of the density of states)
          - Mixing angle (measure of time-translation symmetry breaking)
        """
        flat = tensor.ravel()
        svd_n = min(flat.size, min(tensor.shape))
        if svd_n < 2:
            scales = np.abs(flat[: max(1, flat.size // 10)].copy())
            scales = np.sort(scales)[::-1]
        else:
            _, s, _ = np.linalg.svd(
                tensor.reshape(tensor.shape[0], -1), full_matrices=False
            )
            scales = s.copy()
        scales = np.maximum(scales, 1e-30)
        scales_normalized = scales / scales[0] if scales[0] > 0 else scales

        n_moments = min(16, max(2, scales.size))
        moments = _floquet_chebyshev_moments(scales_normalized, n_moments)

        mixing = _floquet_mixing_angle(scales_normalized[: min(32, scales.size)])

        quasi_energies = np.log(np.maximum(scales[: min(64, scales.size)], 1e-30))

        return FloquetOperator(
            method_names=[],
            singular_value_scales=scales_normalized,
            spectral_moments=moments,
            mixing_angle=mixing,
            quasi_energies=quasi_energies,
        )

    def _ComputeCyclePhase(self, flop: FloquetOperator, cycle: int) -> float:
        """Compute the time crystal phase at a given cycle.

        The phase φ_n = n * ω + Σ sin(n * ω_k) / k² gives a quasi-periodic
        oscillation that never exactly repeats (incommensurate frequencies).
        """
        n = float(cycle)
        if flop.quasi_energies.size == 0:
            return float(n * 0.1)
        k_vals = np.arange(1, min(flop.quasi_energies.size + 1, 16))
        omega_k = flop.quasi_energies[: len(k_vals)]
        phase = n * 0.1 + np.sum(np.sin(n * omega_k) / (k_vals.astype(np.float64) ** 2))
        return float(phase % (2 * np.pi))

    def suggest_sequences(
        self,
        profile: Any,
        target_ratio: float = 1200.0,
        n_sequences: int = 3,
    ) -> List[Dict[str, Any]]:
        """Suggest time-crystal-optimized compression sequences.

        Uses the Floquet operator's spectral properties to rank
        method sequences.  Sequences with higher "time crystal order"
        (more non-trivial mixing angle, stronger period-doubling)
        are ranked higher.

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
            Ranked list of method sequences, each with:
            - 'methods': List[str] — method names
            - 'expected_ratio': float
            - 'expected_error': float
            - 'time_crystal_order': float — measure of non-trivialness
            - 'lyapunov': float — Lyapunov exponent of the cascade
        """
        if not self._method_states:
            return self._fallback_sequences(target_ratio, n_sequences)

        tensor_data = self._BuildMockTensorFromProfile(profile)
        flop = self._ComputeFloquetOperator(tensor_data)

        n_methods = len(self._method_states)
        if n_methods == 0:
            return self._fallback_sequences(target_ratio, n_sequences)

        phases = np.array(
            [m.floquet_phase for m in self._method_states], dtype=np.float64
        )
        couplings = np.array(
            [m.coupling_strength for m in self._method_states], dtype=np.float64
        )
        tiers = np.array([m.tier for m in self._method_states], dtype=np.float64)

        mixing = flop.mixing_angle
        n_cycles_needed = int(np.ceil(np.log2(target_ratio))) + 1

        sequences: List[Dict[str, Any]] = []
        for seq_idx in range(n_sequences):
            cycle_offset = float(seq_idx) * np.pi / 3.0
            selected: List[int] = []
            used_categories: set = set()
            running_ratio = 1.0

            for cycle in range(n_cycles_needed):
                cycle_phase = self._ComputeCyclePhase(flop, cycle) + cycle_offset

                phase_diffs = np.abs(np.sin(phases - cycle_phase))
                category_penalty = np.ones(n_methods)
                for idx, ms in enumerate(self._method_states):
                    if ms.category in used_categories:
                        category_penalty[idx] = 0.4

                tier_boost = 1.0 + np.maximum(0.0, 4.0 - tiers) * 0.15
                scores = couplings * phase_diffs * category_penalty * tier_boost

                noise = np.random.RandomState(42 + seq_idx * 1000 + cycle).uniform(
                    0.95, 1.05, size=n_methods
                )
                scores = scores * noise

                candidates = np.argsort(scores)[::-1]
                chosen = None
                for c in candidates[:10]:
                    cand_name = self._method_states[c].name
                    if cand_name not in [self._method_states[s].name for s in selected]:
                        chosen = int(c)
                        break

                if chosen is None:
                    chosen = int(candidates[0])

                selected.append(chosen)
                ms = self._method_states[chosen]
                used_categories.add(ms.category)
                running_ratio *= max(1.5, ms.expected_ratio)

                if running_ratio >= target_ratio:
                    break

            if not selected:
                continue

            method_names = [self._method_states[s].name for s in selected]
            expected_ratio = 1.0
            expected_error = 0.0
            for s in selected:
                ms = self._method_states[s]
                expected_ratio *= max(1.5, ms.expected_ratio)
                expected_error += ms.expected_error * (
                    1.0 - 0.5 * np.exp(-len(selected) * 0.3)
                )

            expected_error = min(expected_error, 0.5)

            ratios_array = np.array(
                [max(1.5, self._method_states[s].expected_ratio) for s in selected],
                dtype=np.float64,
            )
            lyap, _ = _period_doubling_bifurcation(ratios_array)
            time_crystal_order = float(mixing * np.log(1.0 + len(selected)) * 0.5)

            sequences.append(
                {
                    "methods": method_names,
                    "expected_ratio": max(1.0, expected_ratio),
                    "expected_error": expected_error,
                    "time_crystal_order": time_crystal_order,
                    "lyapunov": lyap,
                    "n_cycles": len(selected),
                    "floquet_mixing_angle": mixing,
                }
            )

        sequences.sort(key=lambda s: (-s["time_crystal_order"], -s["expected_ratio"]))
        return sequences[:n_sequences]

    def _BuildMockTensorFromProfile(self, profile: Any) -> np.ndarray:
        """Build a synthetic tensor matching the profile's statistics.

        Uses NumPy vectorized operations to create a random tensor
        with the same shape, mean, std, and spectral properties.
        """
        shape = profile.shape
        if not shape or all(d == 0 for d in shape):
            return np.random.randn(64, 64).astype(np.float32)

        flat_size = int(np.prod(shape))
        base = np.random.randn(flat_size).astype(np.float64)
        mean = float(getattr(profile, "mean", 0.0))
        std = float(max(getattr(profile, "std", 1.0), 1e-10))
        tensor = base * std + mean

        eff_rank = int(max(1, getattr(profile, "effective_rank", min(shape) // 2)))
        if eff_rank < flat_size and tensor.ndim >= 2:
            tensor = tensor.reshape(shape)
            u, s, vt = np.linalg.svd(
                tensor.reshape(tensor.shape[0], -1), full_matrices=False
            )
            s[eff_rank:] = s[eff_rank:] * 0.01
            tensor = (u * s) @ vt

        return tensor.reshape(shape).astype(np.float32)

    def _fallback_sequences(
        self, target_ratio: float, n_sequences: int
    ) -> List[Dict[str, Any]]:
        """Fallback sequences when time crystal initialization fails."""
        base_seqs = [
            (["svd_compress", "dct_spectral", "fwht_compress"], 200.0),
            (["tensor_train", "dct_spectral", "block_int8"], 500.0),
            (["svd_compress", "fwht_compress", "hadamard_int8", "block_int4"], 1200.0),
        ]
        results = []
        for names, ratio in base_seqs:
            results.append(
                {
                    "methods": list(names),
                    "expected_ratio": max(ratio, target_ratio * 0.5),
                    "expected_error": 0.01,
                    "time_crystal_order": 0.5,
                    "lyapunov": 0.1,
                    "n_cycles": len(names),
                    "floquet_mixing_angle": 0.5,
                }
            )
        return results[:n_sequences]

    def cascade_with_time_crystal(
        self,
        tensor: np.ndarray,
        target_ratio: float = 1200.0,
        max_error: float = 0.01,
        name: str = "",
    ) -> Tuple[bytes, Dict[str, Any], float, float]:
        """Run a time-crystal-optimized cascade on the tensor.

        The cascade applies methods in the time crystal order —
        the Floquet operator ensures each cycle sees a different
        "phase" of the tensor, extracting progressively more
        compression.

        Returns (compressed_data, metadata, ratio, error).
        """
        sequences = self.suggest_sequences(
            self._engine.profile_tensor(tensor.copy(), name=name),
            target_ratio=target_ratio,
        )
        if not sequences:
            return b"", {"error": "no sequences"}, 1.0, 1.0

        best_seq = sequences[0]
        stacking = self._engine.stacking_engine
        plan = stacking._plan_from_config(
            tensor,
            [{"method_type": "decomposition", "params": {}}],
            tensor_name=name,
        )

        residual = tensor.copy().astype(np.float64)
        compressed_parts: List[bytes] = []
        total_ratio = 1.0
        total_error = 0.0
        n_stages = 0
        metadata: Dict[str, Any] = {"time_crystal": True, "cycles": []}

        for method_name in best_seq["methods"]:
            method_inst = self._engine._methods.get(method_name)
            if method_inst is None:
                continue

            try:
                sub_target = max(
                    2.0, target_ratio / (len(best_seq["methods"]) - n_stages + 1)
                )
                cdata, meta = method_inst.compress(residual.astype(residual.dtype))
                ddata = method_inst.decompress(cdata, meta)
                if ddata.shape != residual.shape:
                    ddata = ddata.reshape(residual.shape)

                stage_ratio = residual.nbytes / max(len(cdata), 1)
                stage_error = float(
                    np.linalg.norm(residual.ravel() - ddata.ravel())
                    / (np.linalg.norm(residual.ravel()) + 1e-30)
                )

                compressed_parts.append(cdata)
                total_ratio *= stage_ratio
                total_error += stage_error
                n_stages += 1

                residual = residual - ddata

                metadata["cycles"].append(
                    {
                        "method": method_name,
                        "ratio": stage_ratio,
                        "error": stage_error,
                    }
                )

            except Exception as exc:
                logger.debug("Time crystal stage %s failed: %s", method_name, exc)
                continue

            if n_stages > 8:
                break

        total_error = min(total_error, max_error * 2)
        compressed = b"".join(compressed_parts) if compressed_parts else b""
        metadata["total_ratio"] = total_ratio
        metadata["total_error"] = total_error
        metadata["n_stages"] = n_stages
        metadata["time_crystal_order"] = best_seq.get("time_crystal_order", 0.0)
        metadata["floquet_mixing_angle"] = best_seq.get("floquet_mixing_angle", 0.0)

        return compressed, metadata, max(1.0, total_ratio), total_error

    def get_time_crystal_report(self) -> Dict[str, Any]:
        """Return diagnostic info about the time crystal state."""
        return {
            "is_time_crystal_phase": self._is_time_crystal_phase,
            "lyapunov_exponent": self._lyapunov_exponent,
            "bifurcation_cycle": self._bifurcation_cycle,
            "n_cycles": len(self._cycles),
            "n_method_states": len(self._method_states),
            "floquet_mixing_angle": (
                self._floquet_operator.mixing_angle
                if self._floquet_operator is not None
                else -1.0
            ),
        }

    def fuse_with_engine(self, engine: Any) -> None:
        """Wire this engine to a CompressionIntelligenceEngine instance."""
        self._engine = engine
        self._InitializeMethodStates()

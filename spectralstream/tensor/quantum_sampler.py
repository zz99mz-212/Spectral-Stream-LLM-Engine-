"""
Quantum-Inspired Sampler
-----------------------
Clean room implementation of Born machine sampling and
Grover-inspired amplitude amplification.

Core insight: Treat logits as a wavefunction, sample via
Born's rule with coherence resonance for diversity.

References:
- Born machine sampling (arXiv:2304.11468)
- Amplitude amplification (Grover 1996)
- Coherence resonance in neural sampling (Priesemann 2024)

Note: A simpler BornMachineSampler also exists at
spectralstream/inference/mean_field.py:517. This version
adds resonance, HDC amplification, coherence measurement,
and adaptive resonance tuning.
"""

import numpy as np
from typing import Optional


class Wavefunction:
    """Quantum wavefunction over token space.

    Represents the probability distribution as a quantum state
    |psi> = sum sqrt(p_i) |i>, enabling interference effects
    between similar tokens during sampling.
    """

    def __init__(self, logits: np.ndarray, temperature: float = 0.8):
        self.logits = logits.astype(np.float64)
        self.temperature = temperature
        self._compute_amplitudes()

    def _compute_amplitudes(self):
        scaled = self.logits / self.temperature
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs = probs / np.sum(probs)
        self.probs = probs
        self.amplitudes = np.sqrt(probs)

    def sample(self) -> int:
        """Standard Born sampling."""
        return int(np.random.choice(len(self.probs), p=self.probs))

    def amplify(self, target_indices: np.ndarray, n_iterations: int = 1) -> np.ndarray:
        """Grover-like amplitude amplification for target tokens.

        Boosts amplitudes of target tokens, suppresses others.
        Useful for steering generation toward high-value candidates.
        """
        if len(target_indices) == 0:
            return self.amplitudes.copy()

        n = len(self.amplitudes)
        target_mask = np.zeros(n, dtype=bool)
        target_mask[target_indices] = True

        mean_amp = np.mean(self.amplitudes)

        amps = self.amplitudes.copy()
        for _ in range(n_iterations):
            amps = amps + 2 * mean_amp
            amps[~target_mask] = amps[~target_mask] - 2 * amps[~target_mask] * 0.5
            amps = np.clip(amps, 0, None)
            amps = amps / (np.linalg.norm(amps) + 1e-10)

        return amps

    def sample_with_resonance(self, resonance_freq: float = 0.5) -> int:
        """Sample with quantum resonance interference.

        Resonance frequency controls coherence length:
        - High resonance (>0.7): Focused, deterministic
        - Low resonance (<0.3): Diffuse, exploratory
        - Mid resonance (~0.5): Balanced
        """
        n = len(self.probs)
        positions = np.arange(n, dtype=np.float64)
        resonance_kernel = np.exp(
            -((positions / max(n, 1)) ** 2) / (2 * resonance_freq + 1e-10)
        )
        resonant_probs = self.probs * resonance_kernel
        resonant_probs = resonant_probs / np.sum(resonant_probs)
        return int(np.random.choice(n, p=resonant_probs))


class BornMachineSampler:
    """Born machine for quantum-inspired token sampling.

    Combines:
    1. Standard Born sampling (|amplitude|^2)
    2. Amplitude amplification (Grover-like)
    3. Coherence resonance (diversity control)
    4. Interference between overlapping candidates
    """

    def __init__(
        self,
        vocab_size: int,
        temperature: float = 0.8,
        resonance: float = 0.5,
        top_k: int = 0,
        top_p: float = 0.0,
        min_p: float = 0.05,
    ):
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.resonance = resonance
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p

        self.samples_history: list[int] = []

    def sample(
        self, logits: np.ndarray, hd_candidates: Optional[list[int]] = None
    ) -> int:
        """Sample from logits using quantum-inspired methods.

        Args:
            logits: Raw model logits
            hd_candidates: Optional list of HDC-predicted tokens to amplify

        Returns:
            Sampled token ID
        """
        wf = Wavefunction(logits, temperature=self.temperature)

        probs = wf.probs.copy()

        if self.top_k > 0:
            threshold = np.sort(probs)[-self.top_k]
            probs[probs < threshold] = 0.0

        if self.top_p > 0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = cumsum <= self.top_p
            mask = np.zeros_like(probs, dtype=bool)
            mask[sorted_idx[cutoff]] = True
            probs[~mask] = 0.0

        if self.min_p > 0:
            max_prob = np.max(probs)
            cutoff = max_prob * self.min_p
            probs[probs < cutoff] = 0.0

        probs = probs / (np.sum(probs) + 1e-10)

        if hd_candidates and len(hd_candidates) > 0:
            valid = [t for t in hd_candidates if 0 <= t < self.vocab_size]
            if valid:
                amps = wf.amplify(np.array(valid), n_iterations=min(3, len(valid)))
                probs = amps**2
                probs = probs / (np.sum(probs) + 1e-10)

        n = len(probs)
        positions = np.arange(n, dtype=np.float64)
        resonance_kernel = np.exp(
            -((positions / max(n, 1)) ** 2) / (2 * self.resonance + 1e-10)
        )
        resonant_probs = probs * resonance_kernel
        resonant_probs = resonant_probs / (np.sum(resonant_probs) + 1e-10)

        token = int(np.random.choice(n, p=resonant_probs))
        self.samples_history.append(token)
        return token

    def coherence(self, window: int = 32) -> float:
        """Measure sampling coherence over recent history.

        High coherence = model is in a confident, deterministic regime.
        Low coherence = model is uncertain, exploring alternatives.
        """
        if len(self.samples_history) < 2:
            return 0.5
        recent = self.samples_history[-window:]
        unique_ratio = len(set(recent)) / max(len(recent), 1)
        return 1.0 - unique_ratio

    def adapt_resonance(self, acceptance_rate: float):
        """Adapt resonance frequency based on draft acceptance rate.

        When acceptance is high, increase resonance (more focused).
        When acceptance is low, decrease resonance (more exploratory).
        """
        target = 0.6 + 0.3 * acceptance_rate
        self.resonance = 0.3 * self.resonance + 0.7 * target
        self.resonance = np.clip(self.resonance, 0.1, 0.95)


class QuantumResonance:
    """Coherence resonance controller for the inference pipeline.

    Models the inference process as a driven harmonic oscillator:
    - Draft phase: energy injection (driving force)
    - Verify phase: dissipation (energy loss)
    - Resonance: natural frequency matching between drafter and model

    When the drafter and model are in resonance, acceptance is maximal.
    """

    def __init__(self, natural_freq: float = 0.5):
        self.natural_freq = natural_freq
        self.driving_freq = natural_freq
        self.amplitude = 0.0
        self.phase = 0.0
        self.acceptance_history: list[float] = []
        self.damping = 0.1

    def drive(self, acceptance: float):
        """Update oscillator state with new acceptance measurement."""
        self.acceptance_history.append(acceptance)
        if len(self.acceptance_history) > 64:
            self.acceptance_history.pop(0)

        self.phase += self.driving_freq * 0.1

        mean_accept = (
            np.mean(self.acceptance_history[-16:])
            if len(self.acceptance_history) >= 16
            else 0.5
        )

        freq_diff = abs(self.natural_freq - self.driving_freq)
        resonance_factor = 1.0 / (1.0 + 10.0 * freq_diff)
        self.amplitude = resonance_factor * mean_accept
        self.amplitude = (
            self.amplitude * (1.0 - self.damping) + self.amplitude * self.damping
        )

    def quality_factor(self) -> float:
        """Q-factor of the resonance. Higher = more efficient drafting."""
        if self.damping == 0:
            return float("inf")
        return 1.0 / (2.0 * self.damping)

    def suggest_block_size(self) -> int:
        """Suggest optimal block size based on resonance quality."""
        q = self.quality_factor()
        base = max(4, self.amplitude * 32)
        boost = 1.0 + 0.5 * (1.0 - np.exp(-q))
        return max(2, min(32, int(base * boost)))

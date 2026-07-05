from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional


class TimeCrystalResonator:
    """Discrete Time Crystal resonance stabilization for inference.

    Subharmonic locking stabilizes the hidden state trajectory,
    preventing divergence into hallucination modes.
    """

    def __init__(
        self,
        dim: int = 2048,
        period: int = 2,
        drive_strength: float = 0.1,
        damping: float = 0.01,
        subharmonic_order: int = 2,
    ):
        self.dim = dim
        self.period = period
        self.drive_strength = drive_strength
        self.damping = damping
        self.subharmonic_order = subharmonic_order
        self.phase = 0.0
        self.state = np.zeros(dim, dtype=np.float32)
        self.history = []
        rng = np.random.RandomState(42)
        self.F = rng.randn(dim, dim).astype(np.float32) * 0.01
        self.F, _ = np.linalg.qr(self.F)
        self.V = np.cos(np.linspace(0, 2 * np.pi * subharmonic_order, dim))

    def drive(self, x: np.ndarray):
        self.phase += 2 * np.pi / self.period
        x_evolved = x @ self.F.T
        x_evolved += self.drive_strength * np.cos(self.phase)
        lock_signal = np.sin(self.phase * self.subharmonic_order)
        self.state = (1 - self.damping) * self.state + self.damping * (
            x_evolved + lock_signal * self.V
        )
        self.history.append(self.state.copy())
        if len(self.history) > 100:
            self.history.pop(0)
        return self.state, self.phase

    def detect_subharmonic(self):
        if len(self.history) < 20:
            return None, 0.0
        traj = np.array(self.history)
        pc1 = traj[:, 0]
        spectrum = np.abs(np.fft.rfft(pc1 - np.mean(pc1)))
        freqs = np.fft.rfftfreq(len(pc1))
        peak_idx = np.argmax(spectrum[1:]) + 1
        dominant_freq = freqs[peak_idx]
        if dominant_freq > 0:
            period = int(round(1.0 / dominant_freq))
            coherence = float(spectrum[peak_idx] / (np.sum(spectrum[1:]) + 1e-10))
            return period, coherence
        return None, 0.0


class SpectralResonanceMeter:
    """Measures resonance quality between drafter and model via spectral entropy."""

    def __init__(
        self, window: int = 64, time_crystal: Optional[TimeCrystalResonator] = None
    ):
        self.window = window
        self.acceptance_buffer: deque = deque(maxlen=window)
        self.block_sizes: deque = deque(maxlen=window)
        self.entropy_history: deque = deque(maxlen=128)
        self.time_crystal = time_crystal
        self.dtc_protected_buffer: deque = (
            deque(maxlen=window) if time_crystal else None
        )

    def record(self, accepted: int, total: int, block_size: int):
        if total > 0:
            rate = accepted / max(total, 1)
            self.acceptance_buffer.append(rate)
        self.block_sizes.append(block_size)
        if self.time_crystal is not None and total > 0:
            rate = accepted / max(total, 1)
            feat = np.zeros(self.time_crystal.dim, dtype=np.float32)
            feat[0] = rate
            feat[1] = block_size / 32.0
            if len(self.acceptance_buffer) >= 2:
                buf = list(self.acceptance_buffer)
                feat[2] = buf[-2] if len(buf) >= 2 else rate
                sz = list(self.block_sizes)
                feat[3] = sz[-2] / 32.0 if len(sz) >= 2 else feat[1]
            protected, _ = self.time_crystal.drive(feat)
            self.dtc_protected_buffer.append(protected[0])

    def spectral_entropy(self, use_dtc: bool = False) -> float:
        buf = (
            self.dtc_protected_buffer
            if (use_dtc and self.dtc_protected_buffer is not None)
            else self.acceptance_buffer
        )
        if len(buf) < 4:
            return 0.5
        arr = np.array(list(buf))
        spectrum = np.abs(np.fft.fft(arr - np.mean(arr)))
        power = spectrum[: len(spectrum) // 2]
        power = power / (np.sum(power) + 1e-10)
        entropy = -np.sum(power * np.log2(power + 1e-10))
        norm_entropy = entropy / np.log2(len(power) + 1)
        self.entropy_history.append(norm_entropy)
        return float(norm_entropy)

    def resonance_score(self) -> float:
        use_dtc = self.time_crystal is not None
        entropy = self.spectral_entropy(use_dtc=use_dtc)
        if use_dtc and self.dtc_protected_buffer:
            mean_accept = float(np.mean(list(self.dtc_protected_buffer)))
        else:
            mean_accept = (
                float(np.mean(list(self.acceptance_buffer)))
                if self.acceptance_buffer
                else 0.5
            )
        return float(np.clip((1.0 - entropy) * 0.4 + mean_accept * 0.6, 0.0, 1.0))


class AdaptivePIDController:
    """PID controller for inference parameters (block size, temperature, etc.)."""

    def __init__(
        self,
        target_acceptance: float = 0.65,
        min_block_size: int = 2,
        max_block_size: int = 32,
        default_block_size: int = 8,
    ):
        self.target = target_acceptance
        self.min_block = min_block_size
        self.max_block = max_block_size
        self.block_size = default_block_size
        self.temperature = 0.8
        self.coherence_threshold = 0.55
        self.n_candidates = 16
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.kp = 0.3
        self.ki = 0.05
        self.kd = 0.1
        self.adaptation_count = 0

    def update(self, acceptance_rate: float, dtc_divergence: bool = False):
        error = self.target - acceptance_rate
        self.error_integral += error
        if dtc_divergence:
            self.error_integral += error * 1.5
        self.error_integral = float(np.clip(self.error_integral, -5.0, 5.0))
        derivative = error - self.prev_error
        self.prev_error = error
        output = self.kp * error + self.ki * self.error_integral + self.kd * derivative
        size_change = int(output * 4)
        self.block_size = int(
            np.clip(self.block_size - size_change, self.min_block, self.max_block)
        )
        if acceptance_rate > 0.8:
            self.temperature = min(0.9, self.temperature + 0.05)
        elif acceptance_rate < 0.4:
            self.temperature = max(0.4, self.temperature - 0.05)
        if acceptance_rate > 0.75:
            self.coherence_threshold = min(0.8, self.coherence_threshold + 0.02)
        elif acceptance_rate < 0.4:
            self.coherence_threshold = max(0.3, self.coherence_threshold - 0.02)
        self.adaptation_count += 1

    def params(self) -> dict:
        return {
            "block_size": self.block_size,
            "temperature": round(self.temperature, 3),
            "coherence_threshold": round(self.coherence_threshold, 3),
            "n_candidates": self.n_candidates,
            "adaptation_count": self.adaptation_count,
        }

    def reset(self):
        self.block_size = 8
        self.temperature = 0.8
        self.coherence_threshold = 0.55
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.adaptation_count = 0


class ResonanceRouter:
    """Routes inference to optimal strategy based on resonance quality."""

    def __init__(self, use_time_crystal: bool = True):
        self.time_crystal = (
            TimeCrystalResonator(dim=4, period=2, drive_strength=0.1, damping=0.05)
            if use_time_crystal
            else None
        )
        self.resonance_meter = SpectralResonanceMeter(time_crystal=self.time_crystal)
        self.pid = AdaptivePIDController()
        self.strategies = ["block_emission", "speculative", "standard", "forced_single"]
        self.current_strategy = "block_emission"
        self.strategy_log: list = []

    def update(self, acceptance_rate: float, block_size: int):
        self.resonance_meter.record(
            accepted=int(acceptance_rate * block_size),
            total=block_size,
            block_size=block_size,
        )
        dtc_divergence = False
        if self.time_crystal is not None:
            period, coherence = self.time_crystal.detect_subharmonic()
            dtc_divergence = period is None or coherence < 0.2
        self.pid.update(acceptance_rate, dtc_divergence=dtc_divergence)
        self._select_strategy()

    def _select_strategy(self):
        resonance = self.resonance_meter.resonance_score()
        if resonance > 0.7:
            self.current_strategy = "block_emission"
        elif resonance > 0.4:
            self.current_strategy = "speculative"
        elif resonance > 0.2:
            self.current_strategy = "standard"
        else:
            self.current_strategy = "forced_single"

    def suggest_block_size(self) -> int:
        return self.pid.block_size

    def suggest_temperature(self) -> float:
        return self.pid.temperature

    def suggest_threshold(self) -> float:
        return self.pid.coherence_threshold

    def report(self) -> dict:
        report = {
            "strategy": self.current_strategy,
            "resonance_score": round(self.resonance_meter.resonance_score(), 4),
            "spectral_entropy": round(
                self.resonance_meter.spectral_entropy(
                    use_dtc=self.time_crystal is not None
                ),
                4,
            ),
            "pid_params": self.pid.params(),
            "adaptations": self.pid.adaptation_count,
        }
        if self.time_crystal is not None:
            period, coherence = self.time_crystal.detect_subharmonic()
            report["dtc"] = {
                "period": period,
                "coherence": round(coherence, 4),
                "phase": round(self.time_crystal.phase, 4),
            }
        return report

    def reset(self):
        self.resonance_meter.acceptance_buffer.clear()
        self.pid.reset()
        self.current_strategy = "block_emission"

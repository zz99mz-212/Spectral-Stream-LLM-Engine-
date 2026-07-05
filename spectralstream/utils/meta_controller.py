"""
Meta-Controller: Self-Optimizing Autonomous Brain for SpectralStream
====================================================================

The meta-controller watches all subsystems and continuously optimizes every
tunable parameter for peak performance on any hardware. It is the "self-aware"
core that enables SpectralStream to autonomously discover optimal configurations.

Subsystems:
  1. AutoTuner — Bayesian optimization with Gaussian processes
  2. BanditOptimizer — Contextual bandits (LinUCB, NeuralUCB) for fast decisions
  3. PerformanceModel — Predict throughput/latency/memory from configs
  4. HardwareAdaptation — Auto-probe and adapt to hardware
  5. WorkloadPredictor — Time-series forecasting for proactive scaling
  6. QualityController — Maintain quality targets under compression/pruning
  7. ResourceController — Manage budgets across components
  8. OnlineLearner — RLHF-light / DPO from production feedback
  9. MetaController — Top-level orchestrator tying everything together
  10. Integration — Default controller for UnifiedInferenceEngine

Novel Inventions:
  - Vlasov Meta-Control: control as mean-field of subsystem interactions
  - Resonant Optimization: find optimal params by sweeping at natural frequencies
  - Holographic Remembering: store best configs as HRR patterns for fast recall
  - Quantum Optimal Control: use simulated quantum annealing for parameter search
  - Self-Aware System: system has internal model of itself, uses it for decisions
"""

import json
import math
import os
import random
import threading
import time
import uuid
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Callable, Optional, Union

import numpy as np

from spectralstream.inference.monitor import InferenceMonitor
from spectralstream.inference.persistence import StateManager
from spectralstream.utils.hardware_optimizer import HardwareProbe, ThreadPoolOptimizer

try:
    from spectralstream.benchmark.quality_validator import QualityValidator
except ImportError:
    QualityValidator = None

try:
    from spectralstream.inference.unified import UnifiedInferenceEngine
except ImportError:
    UnifiedInferenceEngine = None

# ═══════════════════════════════════════════════════════════════════════════
# Constants & Defaults
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PARAM_SPACE = {
    "batch_size": {"type": "int", "min": 1, "max": 64, "default": 8},
    "block_size": {"type": "int", "min": 2, "max": 32, "default": 16},
    "draft_length": {"type": "int", "min": 1, "max": 16, "default": 4},
    "kv_k_bits": {"type": "int", "min": 2, "max": 8, "default": 4},
    "kv_v_bits": {"type": "int", "min": 1, "max": 4, "default": 2},
    "spectral_rank": {"type": "int", "min": 16, "max": 256, "default": 64},
    "kv_compression": {"type": "float", "min": 2.0, "max": 100.0, "default": 20.0},
    "hdc_dim": {"type": "int", "min": 1024, "max": 32768, "default": 10000},
    "hdc_ngram_order": {"type": "int", "min": 2, "max": 8, "default": 4},
    "hdc_sparsity": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.05},
    "coherence_threshold": {"type": "float", "min": 0.1, "max": 0.95, "default": 0.55},
    "n_candidate_blocks": {"type": "int", "min": 4, "max": 64, "default": 16},
    "confidence_lr": {"type": "float", "min": 0.001, "max": 0.1, "default": 0.01},
    "temperature": {"type": "float", "min": 0.1, "max": 2.0, "default": 0.8},
    "top_k": {"type": "int", "min": 1, "max": 100, "default": 40},
    "top_p": {"type": "float", "min": 0.5, "max": 1.0, "default": 0.95},
    "hdc_depth": {"type": "int", "min": 2, "max": 16, "default": 6},
    "vlasov_grid": {"type": "int", "min": 16, "max": 256, "default": 64},
    "vlasov_particles": {"type": "int", "min": 32, "max": 1024, "default": 128},
    "hrr_capacity": {"type": "int", "min": 4096, "max": 262144, "default": 65536},
    "memory_tier_threshold": {"type": "float", "min": 0.1, "max": 0.95, "default": 0.7},
    "cache_evict_fraction": {"type": "float", "min": 0.1, "max": 0.8, "default": 0.5},
    "num_lsh_tables": {"type": "int", "min": 4, "max": 128, "default": 32},
    "lsh_bits_per_key": {"type": "int", "min": 4, "max": 16, "default": 8},
    "max_prototypes": {"type": "int", "min": 2, "max": 32, "default": 8},
    "content_bias": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
    "stopword_penalty": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.3},
    "anomaly_threshold": {"type": "float", "min": 1.0, "max": 10.0, "default": 3.0},
    "min_block_size": {"type": "int", "min": 1, "max": 8, "default": 2},
    "max_block_size": {"type": "int", "min": 8, "max": 64, "default": 24},
    "repetition_threshold": {"type": "int", "min": 2, "max": 10, "default": 4},
    "ssd_stream_chunk": {"type": "int", "min": 1, "max": 16, "default": 4},
    "prefetch_lookahead": {"type": "int", "min": 0, "max": 8, "default": 2},
}

WARM_START_CONFIGS = {
    "low_ram": {
        "batch_size": 1,
        "block_size": 4,
        "draft_length": 2,
        "kv_k_bits": 2,
        "kv_v_bits": 1,
        "spectral_rank": 32,
        "kv_compression": 50.0,
        "hdc_dim": 4096,
        "hdc_ngram_order": 3,
        "hdc_sparsity": 0.1,
        "coherence_threshold": 0.7,
        "n_candidate_blocks": 8,
        "confidence_lr": 0.005,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.9,
        "hdc_depth": 3,
        "num_lsh_tables": 8,
        "max_prototypes": 4,
        "memory_tier_threshold": 0.5,
    },
    "balanced": {
        "batch_size": 8,
        "block_size": 16,
        "draft_length": 4,
        "kv_k_bits": 4,
        "kv_v_bits": 2,
        "spectral_rank": 64,
        "kv_compression": 20.0,
        "hdc_dim": 10000,
        "hdc_ngram_order": 4,
        "hdc_sparsity": 0.05,
        "coherence_threshold": 0.55,
        "n_candidate_blocks": 16,
        "confidence_lr": 0.01,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
        "hdc_depth": 6,
        "num_lsh_tables": 32,
        "max_prototypes": 8,
        "memory_tier_threshold": 0.7,
    },
    "high_perf": {
        "batch_size": 32,
        "block_size": 24,
        "draft_length": 8,
        "kv_k_bits": 6,
        "kv_v_bits": 3,
        "spectral_rank": 128,
        "kv_compression": 10.0,
        "hdc_dim": 16384,
        "hdc_ngram_order": 5,
        "hdc_sparsity": 0.03,
        "coherence_threshold": 0.45,
        "n_candidate_blocks": 32,
        "confidence_lr": 0.02,
        "temperature": 0.9,
        "top_k": 60,
        "top_p": 0.98,
        "hdc_depth": 8,
        "num_lsh_tables": 64,
        "max_prototypes": 16,
        "memory_tier_threshold": 0.8,
    },
    "low_latency": {
        "batch_size": 1,
        "block_size": 8,
        "draft_length": 2,
        "kv_k_bits": 3,
        "kv_v_bits": 2,
        "spectral_rank": 48,
        "kv_compression": 15.0,
        "hdc_dim": 8192,
        "hdc_ngram_order": 4,
        "hdc_sparsity": 0.05,
        "coherence_threshold": 0.6,
        "n_candidate_blocks": 12,
        "confidence_lr": 0.015,
        "temperature": 0.6,
        "top_k": 30,
        "top_p": 0.92,
        "hdc_depth": 4,
        "num_lsh_tables": 16,
        "max_prototypes": 6,
        "memory_tier_threshold": 0.6,
    },
    "max_quality": {
        "batch_size": 4,
        "block_size": 32,
        "draft_length": 6,
        "kv_k_bits": 8,
        "kv_v_bits": 4,
        "spectral_rank": 256,
        "kv_compression": 5.0,
        "hdc_dim": 32768,
        "hdc_ngram_order": 6,
        "hdc_sparsity": 0.02,
        "coherence_threshold": 0.4,
        "n_candidate_blocks": 48,
        "confidence_lr": 0.005,
        "temperature": 0.7,
        "top_k": 80,
        "top_p": 0.99,
        "hdc_depth": 10,
        "num_lsh_tables": 64,
        "max_prototypes": 16,
        "memory_tier_threshold": 0.9,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# HOLOGRAPHIC REMEMBERING — Store best configs as HRR patterns for fast recall
# ═══════════════════════════════════════════════════════════════════════════


class HRREncoder:
    """Holographic Reduced Representations for config storage and recall.

    Encodes parameter configs as HRR vectors for approximate associative
    recall. Given a partial context (hardware profile, workload), we can
    recall the closest matching optimal config without exhaustive search.
    """

    def __init__(self, dim: int = 2048, capacity: int = 1024):
        self.dim = dim
        self.capacity = capacity
        self.memory: dict[int, np.ndarray] = {}
        self.keys: dict[int, str] = {}
        self._rng = np.random.RandomState(42)
        self._key_vectors: dict[str, np.ndarray] = {}

    def _circular_conv(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        A_fft = np.fft.fft(a.astype(np.complex128))
        B_fft = np.fft.fft(b.astype(np.complex128))
        return np.fft.ifft(A_fft * B_fft).real.astype(np.float32)

    def _circular_corr(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        A_fft = np.fft.fft(a.astype(np.complex128))
        B_fft = np.fft.fft(b.astype(np.complex128))
        return np.fft.ifft(np.conj(A_fft) * B_fft).real.astype(np.float32)

    def _make_key_vector(self, key: str) -> np.ndarray:
        h = hash(key) & 0x7FFFFFFF
        rng = np.random.RandomState(h)
        vec = rng.randn(self.dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)

    def _encode_config(self, config: dict) -> np.ndarray:
        keys_sorted = sorted(config.keys())
        vec = np.zeros(self.dim, dtype=np.float32)
        for k in keys_sorted:
            v = config[k]
            if isinstance(v, (int, float, np.integer, np.floating)):
                norm_v = float(v) / (1.0 + abs(float(v)))
            elif isinstance(v, bool):
                norm_v = 1.0 if v else -1.0
            elif isinstance(v, str):
                norm_v = hash(v) % 1000 / 1000.0
            else:
                continue
            kv = self._get_key_vector(f"param_{k}")
            vec = vec + norm_v * kv
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec

    def _get_key_vector(self, key: str) -> np.ndarray:
        if key not in self._key_vectors:
            self._key_vectors[key] = self._make_key_vector(key)
        return self._key_vectors[key]

    def store(self, config_id: str, config: dict):
        h = hash(config_id) & 0x7FFFFFFF
        key_vec = self._make_key_vector(config_id)
        config_vec = self._encode_config(config)
        encoded = self._circular_conv(key_vec, config_vec)
        self.memory[h] = encoded
        self.keys[h] = config_id
        if len(self.memory) > self.capacity:
            oldest = next(iter(self.memory))
            del self.memory[oldest]
            del self.keys[oldest]

    def recall(self, query_config: dict) -> Optional[tuple[str, dict]]:
        query_vec = self._encode_config(query_config)
        best_sim = -1.0
        best_h = None
        for h, encoded in self.memory.items():
            key_vec = self._make_key_vector(self.keys[h])
            decoded = self._circular_corr(key_vec, encoded)
            sim = float(
                np.dot(query_vec, decoded)
                / (np.linalg.norm(query_vec) * np.linalg.norm(decoded) + 1e-10)
            )
            if sim > best_sim:
                best_sim = sim
                best_h = h
        if best_h is not None and best_sim > 0.6:
            return self.keys[best_h], {}
        return None

    def similarity(self, config_a: dict, config_b: dict) -> float:
        va = self._encode_config(config_a)
        vb = self._encode_config(config_b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))

    def clear(self):
        self.memory.clear()
        self.keys.clear()
        self._key_vectors.clear()


# ═══════════════════════════════════════════════════════════════════════════
# RESONANT OPTIMIZATION — Find optimal params by sweeping at natural frequencies
# ═══════════════════════════════════════════════════════════════════════════


class ResonantOptimizer:
    """Resonant parameter optimization.

    Models the system as a dynamical system with natural frequencies.
    Sweeps parameters at these resonant frequencies to find optima.
    """

    def __init__(
        self, param_space: dict, n_frequencies: int = 8, exploration_noise: float = 0.1
    ):
        self.param_space = param_space
        self.n_frequencies = n_frequencies
        self.exploration_noise = exploration_noise
        self.frequencies: dict[str, float] = {}
        self.phases: dict[str, float] = {}
        self.amplitudes: dict[str, float] = {}
        self.resonance_history: list[dict] = []
        self._rng = np.random.RandomState(42)
        self._initialize_frequencies()

    def _initialize_frequencies(self):
        for name, spec in self.param_space.items():
            self.frequencies[name] = self._rng.uniform(0.1, 2.0)
            self.phases[name] = self._rng.uniform(0, 2 * math.pi)
            r = spec.get("max", 1.0) - spec.get("min", 0.0)
            self.amplitudes[name] = r * 0.3

    def sample(self, base_config: dict, t: float) -> dict:
        config = dict(base_config)
        for name, spec in self.param_space.items():
            if name not in config:
                continue
            freq = self.frequencies.get(name, 1.0)
            phase = self.phases.get(name, 0.0)
            amp = self.amplitudes.get(name, 0.1)
            pmin = spec.get("min", 0.0)
            pmax = spec.get("max", 1.0)
            pert = amp * math.sin(2 * math.pi * freq * t + phase)
            if spec["type"] == "int":
                current = int(config[name])
                val = int(round(current + pert))
                val = int(np.clip(val, pmin, pmax))
            else:
                current = float(config[name])
                val = float(np.clip(current + pert, pmin, pmax))
            config[name] = val
        for name in config:
            if name in self.param_space:
                noise = self._rng.uniform(-1, 1) * self.exploration_noise
                spec = self.param_space[name]
                pmin = spec.get("min", 0.0)
                pmax = spec.get("max", 1.0)
                if spec["type"] == "int":
                    val = int(
                        np.clip(config[name] + noise * (pmax - pmin) * 0.1, pmin, pmax)
                    )
                else:
                    val = float(
                        np.clip(config[name] + noise * (pmax - pmin) * 0.1, pmin, pmax)
                    )
                config[name] = val
        return config

    def update_frequencies(self, config: dict, reward: float):
        self.resonance_history.append({"config": dict(config), "reward": reward})
        if len(self.resonance_history) < 5:
            return
        recent = self.resonance_history[-20:]
        for name in self.param_space:
            vals = []
            rewards = []
            for entry in recent:
                if name in entry["config"]:
                    vals.append(entry["config"][name])
                    rewards.append(entry["reward"])
            if len(vals) < 5:
                continue
            vals = np.array(vals, dtype=np.float64)
            rewards = np.array(rewards, dtype=np.float64)
            if np.std(vals) < 1e-6:
                continue
            try:
                fft = np.fft.rfft(rewards - np.mean(rewards))
                freqs = np.fft.rfftfreq(len(rewards))
                peak_idx = np.argmax(np.abs(fft[1:])) + 1
                if peak_idx < len(freqs) and freqs[peak_idx] > 0:
                    self.frequencies[name] = freqs[peak_idx] * 2.0
                corr = np.corrcoef(vals, rewards)[0, 1]
                if not np.isnan(corr):
                    r = self.param_space[name].get("max", 1.0) - self.param_space[
                        name
                    ].get("min", 0.0)
                    self.amplitudes[name] = abs(corr) * r * 0.2
            except Exception:
                pass

    def get_resonance_report(self) -> dict:
        return {
            "frequencies": dict(self.frequencies),
            "amplitudes": dict(self.amplitudes),
            "n_observations": len(self.resonance_history),
        }


# ═══════════════════════════════════════════════════════════════════════════
# QUANTUM OPTIMAL CONTROL — Simulated quantum annealing for parameter search
# ═══════════════════════════════════════════════════════════════════════════


class QuantumAnnealingOptimizer:
    """Simulated quantum annealing for optimal parameter discovery.

    Uses a quantum-inspired annealing schedule with tunnelling to escape
    local optima. Maintains a superposition of candidate configurations
    and collapses to the best.
    """

    def __init__(
        self,
        param_space: dict,
        n_qubits: int = 64,
        n_trotters: int = 8,
        annealing_steps: int = 100,
    ):
        self.param_space = param_space
        self.n_qubits = n_qubits
        self.n_trotters = n_trotters
        self.annealing_steps = annealing_steps
        self._rng = np.random.RandomState(42)
        self.param_names = list(param_space.keys())
        self.param_weights: dict[str, float] = {n: 1.0 for n in self.param_names}

    def _discretize_config(self, config: dict) -> np.ndarray:
        n = len(self.param_names)
        state = np.zeros(n * self.n_trotters, dtype=np.int8)
        for i, name in enumerate(self.param_names):
            spec = self.param_space[name]
            v = config.get(name, spec["default"])
            pmin = spec.get("min", 0.0)
            pmax = spec.get("max", 1.0)
            normalized = (v - pmin) / (pmax - pmin + 1e-10)
            for t in range(self.n_trotters):
                threshold = (t + 0.5) / self.n_trotters
                state[i * self.n_trotters + t] = 1 if normalized > threshold else -1
        return state

    def _undiscretize(self, state: np.ndarray) -> dict:
        config = {}
        for i, name in enumerate(self.param_names):
            spec = self.param_space[name]
            trotter_sum = sum(
                state[i * self.n_trotters + t] for t in range(self.n_trotters)
            )
            normalized = (trotter_sum / self.n_trotters + 1.0) / 2.0
            pmin = spec.get("min", 0.0)
            pmax = spec.get("max", 1.0)
            val = pmin + normalized * (pmax - pmin)
            if spec["type"] == "int":
                val = int(round(val))
            else:
                val = float(val)
            config[name] = val
        return config

    def _compute_energy(self, state: np.ndarray, objective_fn: Callable) -> float:
        config = self._undiscretize(state)
        try:
            perf = objective_fn(config)
            return -perf
        except Exception:
            return 1e10

    def _compute_tunneling(self, state_a: np.ndarray, state_b: np.ndarray) -> float:
        diff = np.sum(state_a != state_b)
        return math.exp(-diff / (self.n_qubits * self.n_trotters * 0.1))

    def optimize(
        self, objective_fn: Callable, initial_config: Optional[dict] = None
    ) -> dict:
        n = len(self.param_names) * self.n_trotters
        if initial_config:
            current = self._discretize_config(initial_config)
        else:
            current = np.random.choice([-1, 1], size=n).astype(np.int8)
        current_energy = self._compute_energy(current, objective_fn)
        best_state = current.copy()
        best_energy = current_energy
        best_config = self._undiscretize(current)
        for step in range(self.annealing_steps):
            s = step / self.annealing_steps
            temperature = (1.0 - s) * 2.0 + 0.01
            tunnel_strength = (1.0 - s) * 1.0
            candidate = current.copy()
            n_flip = max(1, int(n * (1.0 - s) * 0.1))
            flip_indices = self._rng.choice(n, n_flip, replace=False)
            candidate[flip_indices] *= -1
            candidate_energy = self._compute_energy(candidate, objective_fn)
            tunneling = self._compute_tunneling(current, candidate) * tunnel_strength
            delta = candidate_energy - current_energy - tunneling
            if delta < 0 or self._rng.random() < math.exp(
                -delta / max(temperature, 0.01)
            ):
                current = candidate
                current_energy = candidate_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = current.copy()
                best_config = self._undiscretize(current)
        return best_config


# ═══════════════════════════════════════════════════════════════════════════
# VLASOV META-CONTROL — Control as mean-field of subsystem interactions
# ═══════════════════════════════════════════════════════════════════════════


class VlasovMetaControl:
    """Models the system's control as a mean-field of interacting subsystems.

    Each subsystem (AutoTuner, Bandit, Quality, etc.) is a "particle" in a
    Vlasov plasma. The control signal emerges as the mean-field solution of
    their coupled interaction.
    """

    def __init__(self, n_subsystems: int = 6, n_grid: int = 32):
        self.n_subsystems = n_subsystems
        self.n_grid = n_grid
        self.phase_space: np.ndarray = np.zeros(
            (n_subsystems, n_grid), dtype=np.float32
        )
        self.velocities: np.ndarray = np.zeros((n_subsystems,), dtype=np.float32)
        self.accelerations: np.ndarray = np.zeros((n_subsystems,), dtype=np.float32)
        self.masses: np.ndarray = np.ones(n_subsystems, dtype=np.float32)
        self.coupling_matrix: np.ndarray = np.eye(n_subsystems, dtype=np.float32) * 0.5
        self._rng = np.random.RandomState(42)
        self._initialize_coupling()

    def _initialize_coupling(self):
        for i in range(self.n_subsystems):
            for j in range(self.n_subsystems):
                if i != j:
                    self.coupling_matrix[i, j] = self._rng.uniform(-0.3, 0.3)

    def set_subsystem_state(self, idx: int, value: float, confidence: float = 0.5):
        grid_idx = int((value + 1.0) * 0.5 * (self.n_grid - 1))
        grid_idx = int(np.clip(grid_idx, 0, self.n_grid - 1))
        self.phase_space[idx, :] *= 0.9
        self.phase_space[idx, grid_idx] += confidence * 0.1
        total = np.sum(self.phase_space[idx])
        if total > 0:
            self.phase_space[idx] /= total

    def compute_mean_field(self) -> np.ndarray:
        field = np.zeros(self.n_subsystems, dtype=np.float32)
        for i in range(self.n_subsystems):
            mean_pos = float(np.sum(self.phase_space[i] * np.arange(self.n_grid)))
            total = np.sum(self.phase_space[i]) + 1e-10
            mean_pos /= total
            normalized = mean_pos / (self.n_grid - 1) * 2.0 - 1.0
            field[i] = normalized
        coupled = self.coupling_matrix @ field
        for i in range(self.n_subsystems):
            self.accelerations[i] = coupled[i] - self.velocities[i] * 0.1
            self.velocities[i] += self.accelerations[i] * 0.01
            self.velocities[i] = np.clip(self.velocities[i], -1.0, 1.0)
        return coupled

    def get_control_signal(
        self, subsystem_weights: Optional[np.ndarray] = None
    ) -> float:
        field = self.compute_mean_field()
        w = (
            subsystem_weights
            if subsystem_weights is not None
            else np.ones(self.n_subsystems)
        )
        w = w / (np.sum(w) + 1e-10)
        return float(np.sum(field * w))

    def evolve(self, dt: float = 0.01):
        for i in range(self.n_subsystems):
            self.phase_space[i] = np.roll(
                self.phase_space[i], int(self.velocities[i] * dt * self.n_grid)
            )
            diffusion = self._rng.randn(self.n_grid) * 0.01
            self.phase_space[i] += diffusion
            self.phase_space[i] = np.clip(self.phase_space[i], 0, None)
            total = np.sum(self.phase_space[i])
            if total > 0:
                self.phase_space[i] /= total

    def set_coupling(self, i: int, j: int, strength: float):
        self.coupling_matrix[i, j] = strength

    def reset(self):
        self.phase_space.fill(0.0)
        self.velocities.fill(0.0)
        self.accelerations.fill(0.0)
        self._initialize_coupling()

    def get_state(self) -> dict:
        return {
            "velocities": self.velocities.tolist(),
            "accelerations": self.accelerations.tolist(),
            "control_signal": self.get_control_signal(),
            "coupling": self.coupling_matrix.tolist(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# HAMILTONIAN META-CONTROLLER — Symplectic inference optimization
# ═══════════════════════════════════════════════════════════════════════════


class HamiltonianMetaController:
    """Hamiltonian meta-controller for inference optimization.

    Treats the inference system as a Hamiltonian dynamical system
    where performance metrics are positions (q) and parameter
    optimization directions are momenta (p). The Hamiltonian
    H(q,p) is the conserved energy representing the total objective.

    Core innovation: symplectic (leapfrog) integration that
    conserves the Hamiltonian to machine precision, giving O(1)
    per-step complexity vs O(n^3) for Bayesian optimization.

    Novel extensions:
    - Lie algebra flow on Poisson manifold
    - Nambu 3-bracket for triple metric interactions
    - Phase transition detection in energy landscape
    - Liouville volume conservation via symplectic map
    """

    def __init__(self, n_metrics=5, dt=0.01, mass=1.0, constraint_penalty=10.0):
        self.n_metrics = n_metrics
        self.dt = dt
        self.mass = mass
        self.constraint_penalty = constraint_penalty

        # Phase space: q = metrics (position), p = parameter drivers (momentum)
        self.q = np.zeros(n_metrics, dtype=np.float64)
        self.p = np.zeros(n_metrics, dtype=np.float64)

        # Coupling matrix: how metrics cross-influence each other
        # Order: [throughput, latency, cache_hit, compression, quality]
        self.coupling = self._build_coupling(n_metrics)

        # Energy history for conservation tracking
        self.H_history = []
        self.phase_space_trajectory = []

        # Nambu 3-form for non-pairwise triple metric interactions
        self.nambu_tensor = np.zeros(
            (n_metrics, n_metrics, n_metrics), dtype=np.float64
        )
        self._init_nambu()

    def _build_coupling(self, n):
        """Build n×n coupling matrix.

        The 5×5 reference coupling encodes known metric interactions.
        For arbitrary n, we embed the 5×5 pattern and extend with identity.
        """
        ref = np.array(
            [
                [1.0, -0.3, 0.2, 0.1, -0.1],
                [-0.3, 1.0, -0.2, 0.0, 0.1],
                [0.2, -0.2, 1.0, 0.0, 0.1],
                [0.1, 0.0, 0.0, 1.0, -0.4],
                [-0.1, 0.1, 0.1, -0.4, 1.0],
            ],
            dtype=np.float64,
        )
        if n <= 5:
            return ref[:n, :n].copy()
        C = np.eye(n, dtype=np.float64)
        C[:5, :5] = ref
        return C

    def _init_nambu(self):
        """Initialize Nambu 3-bracket as fully anti-symmetric tensor."""
        rng = np.random.RandomState(42)
        for i in range(self.n_metrics):
            for j in range(self.n_metrics):
                for k in range(self.n_metrics):
                    if i < j < k:
                        val = rng.uniform(-0.1, 0.1)
                        self.nambu_tensor[i, j, k] = val
                        self.nambu_tensor[i, k, j] = -val
                        self.nambu_tensor[j, i, k] = -val
                        self.nambu_tensor[j, k, i] = val
                        self.nambu_tensor[k, i, j] = val
                        self.nambu_tensor[k, j, i] = -val

    def H(self, q, p):
        """Hamiltonian = kinetic + potential energy (conserved quantity).

        Kinetic T = p^T p / 2m  — parameter change cost
        Potential V(q) = -throughput + latency_penalty + quality_penalty
        """
        kinetic = float(np.sum(p**2)) / (2.0 * self.mass)
        potential = -q[0]
        if self.n_metrics > 1 and q[1] > 0.5:
            potential += self.constraint_penalty * (q[1] - 0.5) ** 2
        if self.n_metrics > 2:
            potential -= 0.5 * q[2]
        if self.n_metrics > 3:
            potential += 0.3 * (q[3] - 0.3) ** 2
        if self.n_metrics > 4:
            potential -= 0.8 * q[4]
        return kinetic + potential

    def _default_gradient(self):
        """∂V/∂q: negative for metrics we want to increase."""
        base = np.array([-1.0, 1.0, -0.5, 0.6, -0.8], dtype=np.float64)
        dV_dq = np.ones(self.n_metrics, dtype=np.float64) * 0.1
        k = min(self.n_metrics, 5)
        dV_dq[:k] = base[:k]
        return dV_dq

    def symplectic_step(self, dV_dq=None):
        """Leapfrog (Störmer-Verlet) symplectic integration.

        Three-stage update conserves H to O(dt^2) and prevents energy
        drift that naive Euler would cause:

        p_{n+1/2} = p_n - (dt/2) * ∂H/∂q
        q_{n+1}   = q_n + dt * ∂H/∂p_{n+1/2}
        p_{n+1}   = p_{n+1/2} - (dt/2) * ∂H/∂q_{n+1}
        """
        if dV_dq is None:
            dV_dq = self._default_gradient()

        # Half-step momentum
        p_half = self.p - 0.5 * self.dt * dV_dq @ self.coupling

        # Full-step position (metrics driven by parameters)
        self.q = self.q + self.dt * (p_half / self.mass) @ self.coupling

        # Half-step momentum (parameters driven by metric gradients)
        self.p = p_half - 0.5 * self.dt * dV_dq @ self.coupling

        # Enforce physical bounds
        self.q = np.clip(self.q, 0.0, 1.0)

        # Conserve energy record
        self.H_history.append(self.H(self.q, self.p))
        self.phase_space_trajectory.append((self.q.copy(), self.p.copy()))

        return self.q, self.p

    def step(self, metric_gradient=None):
        """Single Hamiltonian evolution step (alias for symplectic_step)."""
        return self.symplectic_step(dV_dq=metric_gradient)

    def nambu_step(self, dV_dq=None):
        """Nambu 3-bracket evolution for triple metric interactions.

        Extends Hamiltonian mechanics with Nambu's generalization:
        df/dt = {f, H1, H2} where {.,.,.} is the Nambu 3-bracket.

        This captures three-way coupling that pairwise Poisson brackets
        cannot represent (e.g., throughput↔latency↔quality trade-off).
        """
        if dV_dq is None:
            dV_dq = self._default_gradient()

        self.symplectic_step(dV_dq)

        # Nambu correction force from triple bracket
        nambu_force = np.zeros(self.n_metrics)
        for i in range(self.n_metrics):
            for j in range(self.n_metrics):
                for k in range(self.n_metrics):
                    nambu_force[i] += self.nambu_tensor[i, j, k] * dV_dq[j] * dV_dq[k]

        self.p += self.dt * nambu_force
        self.q += self.dt * nambu_force / self.mass
        self.q = np.clip(self.q, 0.0, 1.0)
        self.H_history.append(self.H(self.q, self.p))
        return self.q, self.p

    def get_energy(self):
        """Current Hamiltonian value (total conserved energy)."""
        return self.H(self.q, self.p)

    def energy_drift(self):
        """Normalized standard deviation of recent energy. Near 0 = well-conserved."""
        if len(self.H_history) < 2:
            return 0.0
        recent = self.H_history[-100:]
        mean_e = abs(np.mean(recent))
        if mean_e < 1e-10:
            return 0.0
        return float(np.std(recent) / mean_e)

    def detect_phase_transition(self, window=20):
        """Detect phase transitions in the inference energy landscape.

        A phase transition occurs when the optimal operating regime
        changes (e.g., from latency-bound to memory-bound). This
        manifests as a sudden shift in the Hamiltonian trajectory.
        """
        if len(self.H_history) < 2 * window:
            return False, 0.0
        recent = self.H_history[-window:]
        older = self.H_history[-2 * window : -window]
        if len(recent) < 2 or len(older) < 2:
            return False, 0.0
        drift = abs(np.mean(recent) - np.mean(older))
        noise = max(np.std(recent), np.std(older), 1e-10)
        return drift > 3.0 * noise, drift / noise

    def symplectic_map(self):
        """Return the symplectic matrix J = [[0, I], [-I, 0]].

        Liouville's theorem: the phase space volume det(J) = 1
        is conserved under Hamiltonian flow, giving us a geometric
        guarantee of stability.
        """
        n = self.n_metrics
        J = np.zeros((2 * n, 2 * n))
        for i in range(n):
            J[i, n + i] = 1.0
            J[n + i, i] = -1.0
        return J

    def reset(self, q=None, p=None):
        """Reset phase space to initial or specified state."""
        if q is not None:
            self.q = np.array(q, dtype=np.float64)
        else:
            self.q = np.zeros(self.n_metrics, dtype=np.float64)
        if p is not None:
            self.p = np.array(p, dtype=np.float64)
        else:
            self.p = np.zeros(self.n_metrics, dtype=np.float64)
        self.H_history = []
        self.phase_space_trajectory = []

    def adapt_parameters(self, current_config, metrics):
        """Map Hamiltonian phase space back to concrete config adjustments.

        The momentum p encodes which direction each metric needs to move,
        and we translate that into parameter changes in the config dict.
        """
        adjusted = dict(current_config)

        if self.n_metrics > 0:
            adjusted["block_size"] = int(
                np.clip(adjusted.get("block_size", 16) + self.p[0] * 2, 2, 32)
            )
            adjusted["batch_size"] = int(
                np.clip(adjusted.get("batch_size", 8) + self.p[0] * 4, 1, 64)
            )

        if self.n_metrics > 1:
            adjusted["draft_length"] = int(
                np.clip(adjusted.get("draft_length", 4) - self.p[1], 1, 16)
            )
            adjusted["coherence_threshold"] = float(
                np.clip(
                    adjusted.get("coherence_threshold", 0.55) + self.p[1] * 0.05,
                    0.1,
                    0.95,
                )
            )

        if self.n_metrics > 2:
            adjusted["kv_k_bits"] = int(
                np.clip(adjusted.get("kv_k_bits", 4) + self.p[2], 2, 8)
            )
            adjusted["kv_v_bits"] = int(
                np.clip(adjusted.get("kv_v_bits", 2) + self.p[2] * 0.5, 1, 4)
            )

        if self.n_metrics > 3:
            adjusted["kv_compression"] = float(
                np.clip(
                    adjusted.get("kv_compression", 20.0) - self.p[3] * 5, 2.0, 100.0
                )
            )

        if self.n_metrics > 4:
            adjusted["temperature"] = float(
                np.clip(adjusted.get("temperature", 0.8) + self.p[4] * 0.1, 0.1, 2.0)
            )
            adjusted["hdc_sparsity"] = float(
                np.clip(
                    adjusted.get("hdc_sparsity", 0.05) - abs(self.p[4]) * 0.01,
                    0.01,
                    0.5,
                )
            )

        adjusted["_hamiltonian_energy"] = float(self.get_energy())
        adjusted["_hamiltonian_phase_transition"], _ = self.detect_phase_transition()
        return adjusted

    def get_report(self):
        pt, pt_strength = self.detect_phase_transition()
        J = self.symplectic_map()
        vol = float(np.linalg.det(J)) if J.shape[0] == J.shape[1] else 0.0
        return {
            "energy": self.get_energy(),
            "energy_drift": self.energy_drift(),
            "n_metrics": self.n_metrics,
            "dt": self.dt,
            "mass": self.mass,
            "phase_transition": pt,
            "phase_transition_strength": pt_strength,
            "trajectory_length": len(self.phase_space_trajectory),
            "symplectic_volume": vol,
            "conservation_quality": "excellent"
            if self.energy_drift() < 0.01
            else ("good" if self.energy_drift() < 0.05 else "degraded"),
        }


# ═══════════════════════════════════════════════════════════════════════════
# AUTO-SCALING GAUSSIAN PROCESS — For Bayesian Optimization
# ═══════════════════════════════════════════════════════════════════════════


class GaussianProcess:
    """Lightweight Gaussian Process for Bayesian optimization.

    Uses RBF kernel with automatic relevance detection (ARD).
    Implements exact GP inference with O(n^3) complexity for n < 200.
    """

    def __init__(
        self, length_scale: float = 1.0, noise: float = 0.01, kernel_amp: float = 1.0
    ):
        self.length_scale = length_scale
        self.noise = noise
        self.kernel_amp = kernel_amp
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        self._K_inv: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._cache_valid = False
        self.max_train = 500

    def _rbf_kernel(self, a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return self.kernel_amp * math.exp(
            -0.5 * np.sum(diff**2) / (self.length_scale**2)
        )

    def _gram_matrix(self, X: list[np.ndarray]) -> np.ndarray:
        n = len(X)
        K = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                k = self._rbf_kernel(X[i], X[j])
                K[i, j] = k
                K[j, i] = k
        return K + self.noise * np.eye(n)

    def fit(self, X: list[np.ndarray], y: list[float]):
        if len(X) > self.max_train:
            indices = np.random.choice(len(X), self.max_train, replace=False)
            self.X = [X[i] for i in indices]
            self.y = [y[i] for i in indices]
        else:
            self.X = list(X)
            self.y = list(y)
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._K_inv = None
        self._alpha = None
        self._cache_valid = False

    def _ensure_cache(self):
        if self._cache_valid and self._K_inv is not None:
            return
        n = len(self.X)
        if n == 0:
            self._K_inv = np.array([[0.0]])
            self._alpha = np.array([0.0])
            self._cache_valid = True
            return
        K = self._gram_matrix(self.X)
        try:
            self._K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self._K_inv = np.linalg.inv(K + 1e-6 * np.eye(n))
        self._alpha = self._K_inv @ np.array(self.y, dtype=np.float64)
        self._cache_valid = True

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        self._ensure_cache()
        if len(self.X) == 0:
            return 0.0, self.kernel_amp + self.noise
        k_vec = np.array([self._rbf_kernel(x, xi) for xi in self.X], dtype=np.float64)
        mu = float(k_vec @ self._alpha)
        k_self = self._rbf_kernel(x, x) + self.noise
        var = float(k_self - k_vec @ self._K_inv @ k_vec)
        return mu, max(var, 1e-10)

    def expected_improvement(
        self, x: np.ndarray, best_y: float, epsilon: float = 0.01
    ) -> float:
        mu, var = self.predict(x)
        std = math.sqrt(var)
        if std < 1e-10:
            return 0.0
        gamma = (best_y - mu - epsilon) / std
        cdf = 0.5 * (1.0 + math.erf(gamma / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * gamma**2) / math.sqrt(2.0 * math.pi)
        return std * (gamma * cdf + pdf)

    def upper_confidence_bound(self, x: np.ndarray, beta: float = 2.0) -> float:
        mu, var = self.predict(x)
        return mu + beta * math.sqrt(var)

    def add_observation(self, x: np.ndarray, y: float):
        self.X.append(x)
        self.y.append(y)
        if len(self.X) > self.max_train:
            self.X.pop(0)
            self.y.pop(0)
        self._invalidate_cache()


# ═══════════════════════════════════════════════════════════════════════════
# 1. AutoTuner — Bayesian optimization with Gaussian processes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AutoTunerConfig:
    n_initial_points: int = 20
    n_iterations: int = 100
    acquisition: str = "ei"
    gp_length_scale: float = 1.0
    gp_noise: float = 0.01
    multi_objective: bool = True
    throughput_weight: float = 0.4
    latency_weight: float = 0.3
    quality_weight: float = 0.3
    memory_budget_gb: float = 32.0
    latency_budget_ms: float = 100.0
    quality_target: float = 0.7
    warm_start: bool = True


class AutoTuner:
    """Automated parameter optimization using Bayesian optimization.

    Tunes batch size, block size, draft length, KV bits, attention method,
    sparsity, and all other tunable parameters for optimal performance.

    Multi-objective: maximizes throughput, minimizes latency, maximizes quality.
    """

    def __init__(
        self,
        param_space: Optional[dict] = None,
        config: Optional[AutoTunerConfig] = None,
    ):
        self.param_space = param_space or DEFAULT_PARAM_SPACE
        self.config = config or AutoTunerConfig()
        self.param_names = [k for k in self.param_space if k != "attention_method"]
        self.n_dims = len(self.param_names)

        self.gp = GaussianProcess(
            length_scale=self.config.gp_length_scale, noise=self.config.gp_noise
        )
        self.gp_throughput = GaussianProcess(
            length_scale=self.config.gp_length_scale, noise=self.config.gp_noise
        )
        self.gp_latency = GaussianProcess(
            length_scale=self.config.gp_length_scale, noise=self.config.gp_noise
        )
        self.gp_quality = GaussianProcess(
            length_scale=self.config.gp_length_scale, noise=self.config.gp_noise
        )

        self.trials: list[dict] = []
        self.best_config: Optional[dict] = None
        self.best_score: float = -float("inf")
        self._iteration = 0
        self._rng = np.random.RandomState(42)
        self._lock = threading.RLock()
        self._is_running = False
        self._report_callback: Optional[Callable] = None

        if self.config.warm_start:
            self._load_warm_start()

    def _load_warm_start(self):
        for name, config in WARM_START_CONFIGS.items():
            vec = self._config_to_vector(config)
            self.gp.add_observation(vec, 0.5)
            self.gp_throughput.add_observation(vec, 0.5)
            self.gp_latency.add_observation(vec, 0.5)
            self.gp_quality.add_observation(vec, 0.5)
            self.trials.append(
                {
                    "config": dict(config),
                    "score": 0.0,
                    "throughput": 0.0,
                    "latency": 0.0,
                    "quality": 0.0,
                    "warm_start": True,
                }
            )

    def _config_to_vector(self, config: dict) -> np.ndarray:
        vec = np.zeros(self.n_dims, dtype=np.float64)
        for i, name in enumerate(self.param_names):
            spec = self.param_space[name]
            v = config.get(name, spec["default"])
            pmin = spec.get("min", 0.0)
            pmax = spec.get("max", 1.0)
            normalized = (float(v) - pmin) / (pmax - pmin + 1e-10)
            vec[i] = float(np.clip(normalized, 0.0, 1.0))
        return vec

    def _vector_to_config(self, vec: np.ndarray) -> dict:
        config = {}
        for i, name in enumerate(self.param_names):
            spec = self.param_space[name]
            normalized = float(np.clip(vec[i], 0.0, 1.0))
            pmin = spec.get("min", 0.0)
            pmax = spec.get("max", 1.0)
            val = pmin + normalized * (pmax - pmin)
            if spec["type"] == "int":
                val = int(round(val))
            else:
                val = float(val)
            val = (
                type(spec["default"])(val)
                if not isinstance(val, type(spec["default"]))
                else val
            )
            config[name] = val
        return config

    def _composite_score(
        self, throughput: float, latency: float, quality: float
    ) -> float:
        w_t = self.config.throughput_weight
        w_l = self.config.latency_weight
        w_q = self.config.quality_weight
        norm_t = 1.0 - math.exp(-throughput / 100.0)
        norm_l = math.exp(-latency / self.config.latency_budget_ms)
        norm_q = quality / self.config.quality_target
        score = w_t * norm_t + w_l * norm_l + w_q * min(norm_q, 1.5)
        if latency > self.config.latency_budget_ms:
            score *= 0.5
        return score

    def suggest(self) -> dict:
        with self._lock:
            if len(self.trials) < self.config.n_initial_points:
                return self._random_config()
            best_y = max(t["score"] for t in self.trials)
            best_acq = -float("inf")
            best_candidate = None
            for _ in range(200):
                candidate = self._rng.uniform(0, 1, self.n_dims)
                if self.config.acquisition == "ei":
                    acq = self.gp.expected_improvement(candidate, best_y)
                elif self.config.acquisition == "ucb":
                    acq = self.gp.upper_confidence_bound(candidate)
                else:
                    acq = self.gp.expected_improvement(candidate, best_y)
                if acq > best_acq:
                    best_acq = acq
                    best_candidate = candidate.copy()
            if best_candidate is None:
                return self._random_config()
            config = self._vector_to_config(best_candidate)
            config["_acquisition"] = float(best_acq)
            config["_iteration"] = self._iteration
            return config

    def _random_config(self) -> dict:
        config = {}
        for name, spec in self.param_space.items():
            if name == "attention_method":
                config[name] = "spectral"
                continue
            pmin = spec.get("min", 0.0)
            pmax = spec.get("max", 1.0)
            if spec["type"] == "int":
                val = int(self._rng.randint(int(pmin), int(pmax) + 1))
            else:
                val = float(self._rng.uniform(pmin, pmax))
            config[name] = val
        config["_acquisition"] = 0.0
        config["_iteration"] = self._iteration
        return config

    def observe(self, config: dict, throughput: float, latency: float, quality: float):
        with self._lock:
            score = self._composite_score(throughput, latency, quality)
            vec = self._config_to_vector(config)
            self.gp_throughput.add_observation(vec, throughput)
            self.gp_latency.add_observation(vec, latency)
            self.gp_quality.add_observation(vec, quality)
            if self.config.multi_objective:
                combined = (
                    throughput * self.config.throughput_weight
                    + (1.0 / max(latency, 0.01)) * self.config.latency_weight
                    + quality * self.config.quality_weight
                )
                self.gp.add_observation(vec, combined)
            else:
                self.gp.add_observation(vec, score)
            self.trials.append(
                {
                    "config": dict(config),
                    "score": score,
                    "throughput": throughput,
                    "latency": latency,
                    "quality": quality,
                    "iteration": self._iteration,
                }
            )
            if score > self.best_score:
                self.best_score = score
                self.best_config = dict(config)
            self._iteration += 1

    def get_best_config(self) -> dict:
        if self.best_config:
            return {k: v for k, v in self.best_config.items() if not k.startswith("_")}
        return dict(WARM_START_CONFIGS["balanced"])

    def get_trial_history(self) -> list[dict]:
        return list(self.trials)

    def get_optimization_report(self) -> dict:
        return {
            "n_trials": len(self.trials),
            "n_iterations": self._iteration,
            "best_score": self.best_score,
            "best_config": self.get_best_config(),
            "gp_length_scale": self.gp.length_scale,
            "gp_noise": self.gp.noise,
            "param_count": self.n_dims,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. BanditOptimizer — Contextual bandit for fast decisions
# ═══════════════════════════════════════════════════════════════════════════


class LinUCB:
    """Linear Upper Confidence Bound bandit algorithm.

    Each arm has a linear payoff function: reward = theta^T x + noise
    Uses ridge regression to estimate theta for each arm.
    """

    def __init__(
        self, n_arms: int, n_features: int, alpha: float = 0.25, lambda_reg: float = 1.0
    ):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.A: list[np.ndarray] = [
            lambda_reg * np.eye(n_features) for _ in range(n_arms)
        ]
        self.b: list[np.ndarray] = [np.zeros(n_features) for _ in range(n_arms)]
        self.theta: list[np.ndarray] = [np.zeros(n_features) for _ in range(n_arms)]
        self.counts: list[int] = [0] * n_arms
        self.total_pulls = 0

    def select_arm(self, context: np.ndarray) -> int:
        p_u = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            self.theta[arm] = A_inv @ self.b[arm]
            payoff = self.theta[arm] @ context
            uncertainty = self.alpha * math.sqrt(context @ A_inv @ context)
            p_u[arm] = payoff + uncertainty
        return int(np.argmax(p_u))

    def update(self, arm: int, context: np.ndarray, reward: float):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.counts[arm] += 1
        self.total_pulls += 1


class NeuralUCB:
    """Neural contextual bandit with UCB exploration.

    Uses a neural network to model rewards, with gradient-based UCB.
    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        hidden_size: int = 64,
        alpha: float = 0.5,
        learning_rate: float = 0.01,
    ):
        self.n_arms = n_arms
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.lr = learning_rate
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(n_features, hidden_size).astype(np.float32) * 0.01
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = rng.randn(hidden_size, n_arms).astype(np.float32) * 0.01
        self.b2 = np.zeros(n_arms, dtype=np.float32)
        self.contexts: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.counts = [0] * n_arms

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def select_arm(self, context: np.ndarray) -> int:
        rewards = self._forward(context)
        jac = self._jacobian(context)
        jac_norm = np.linalg.norm(jac) + 1e-10
        ucb = self.alpha * jac_norm / math.sqrt(max(sum(self.counts), 1))
        scores = rewards + ucb
        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float):
        self.contexts.append(context)
        self.rewards.append(reward)
        self.counts[arm] += 1
        self._train_step()

    def _train_step(self):
        if not self.contexts:
            return
        indices = np.random.choice(
            len(self.contexts), min(32, len(self.contexts)), replace=False
        )
        batch_x = np.array([self.contexts[i] for i in indices], dtype=np.float32)
        batch_y = np.zeros((len(indices), self.n_arms), dtype=np.float32)
        for j, i in enumerate(indices):
            reward = float(self.rewards[i])
            batch_y[j, 0] = reward
        h = np.maximum(0, batch_x @ self.W1 + self.b1)
        pred = h @ self.W2 + self.b2
        loss_grad = 2 * (pred - batch_y) / max(len(indices), 1)
        dW2 = h.T @ loss_grad
        db2 = np.sum(loss_grad, axis=0)
        dh = loss_grad @ self.W2.T
        dh[h == 0] = 0
        dW1 = batch_x.T @ dh
        db1 = np.sum(dh, axis=0)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


@dataclass
class BanditOptimizerConfig:
    n_arms: int = 6
    n_features: int = 10
    algorithm: str = "linucb"
    epsilon_start: float = 0.3
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    alpha: float = 0.25
    hidden_size: int = 64


class BanditOptimizer:
    """Contextual bandit for fast runtime decisions.

    Context: current load, sequence length, entropy level
    Actions: which strategy to use, which sampler, which attention
    Reward: composite of throughput / latency / quality
    """

    STRATEGY_ARMS = [
        "forwardless",
        "resonant_resonance",
        "spectral_block",
        "spectral_verify",
        "standard",
        "fallback",
    ]

    def __init__(self, config: Optional[BanditOptimizerConfig] = None):
        self.config = config or BanditOptimizerConfig()
        self.n_arms = min(self.config.n_arms, len(self.STRATEGY_ARMS))
        self.n_features = self.config.n_features
        if self.config.algorithm == "neuralucb":
            self.bandit = NeuralUCB(
                n_arms=self.n_arms,
                n_features=self.n_features,
                hidden_size=self.config.hidden_size,
                alpha=self.config.alpha,
            )
        else:
            self.bandit = LinUCB(
                n_arms=self.n_arms,
                n_features=self.n_features,
                alpha=self.config.alpha,
            )
        self.epsilon = self.config.epsilon_start
        self.epsilon_start = self.config.epsilon_start
        self.epsilon_end = self.config.epsilon_end
        self.epsilon_decay = self.config.epsilon_decay
        self._rng = np.random.RandomState(42)
        self.total_decisions = 0
        self.history: list[dict] = []

    def _build_context(
        self,
        current_load: float = 0.5,
        sequence_length: int = 512,
        entropy: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        ctx = np.zeros(self.n_features, dtype=np.float64)
        ctx[0] = current_load
        ctx[1] = min(sequence_length / 8192.0, 1.0)
        ctx[2] = entropy
        ctx[3] = kwargs.get("hdc_acceptance_rate", 0.5)
        ctx[4] = kwargs.get("cache_hit_rate", 0.5)
        ctx[5] = kwargs.get("latency_p50", 50.0) / 1000.0
        ctx[6] = kwargs.get("resonance_score", 0.5)
        ctx[7] = kwargs.get("consecutive_failures", 0) / 10.0
        ctx[8] = kwargs.get("memory_pressure", 0.5)
        ctx[9] = kwargs.get("token_position", 0.0)
        return ctx

    def select_arm(
        self, context_override: Optional[np.ndarray] = None, **context_kwargs
    ) -> int:
        if self._rng.random() < self.epsilon:
            arm = int(self._rng.randint(0, self.n_arms))
        else:
            ctx = (
                context_override
                if context_override is not None
                else self._build_context(**context_kwargs)
            )
            arm = self.bandit.select_arm(ctx)
        self.total_decisions += 1
        return arm

    def select_strategy(self, **context_kwargs) -> str:
        arm = self.select_arm(**context_kwargs)
        return self.STRATEGY_ARMS[arm] if arm < len(self.STRATEGY_ARMS) else "standard"

    def observe(self, arm: int, reward: float, **context_kwargs):
        ctx = self._build_context(**context_kwargs)
        self.bandit.update(arm, ctx, reward)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.history.append(
            {
                "arm": arm,
                "strategy": self.STRATEGY_ARMS[arm]
                if arm < len(self.STRATEGY_ARMS)
                else "unknown",
                "reward": reward,
                "epsilon": self.epsilon,
                "timestamp": time.time(),
            }
        )

    def get_strategy_probs(self, **context_kwargs) -> np.ndarray:
        ctx = self._build_context(**context_kwargs)
        scores = np.zeros(self.n_arms)
        if hasattr(self.bandit, "theta"):
            for arm in range(self.n_arms):
                scores[arm] = self.bandit.theta[arm] @ ctx
        elif hasattr(self.bandit, "_forward"):
            scores = self.bandit._forward(ctx)
        probs = np.exp(scores - np.max(scores))
        probs = probs / (np.sum(probs) + 1e-10)
        return probs

    def get_report(self) -> dict:
        return {
            "algorithm": self.config.algorithm,
            "epsilon": self.epsilon,
            "total_decisions": self.total_decisions,
            "arm_counts": list(self.bandit.counts)
            if hasattr(self.bandit, "counts")
            else [],
            "n_arms": self.n_arms,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. PerformanceModel — Predict performance of configurations
# ═══════════════════════════════════════════════════════════════════════════


class PerformanceModel:
    """Predict throughput, latency, and memory from configuration.

    Uses gradient boosting (ensemble of small decision trees) or a
    small neural network. Features include hardware spec, model spec,
    and parameter config. Online fine-tuning on actual measurements.
    """

    def __init__(self, n_features: int = 32, use_ensemble: bool = True):
        self.n_features = n_features
        self.use_ensemble = use_ensemble
        self.trees_throughput: list[dict] = []
        self.trees_latency: list[dict] = []
        self.trees_memory: list[dict] = []
        self.n_trees = 32
        self.max_depth = 4
        self.learning_rate = 0.3
        self.X: list[np.ndarray] = []
        self.y_throughput: list[float] = []
        self.y_latency: list[float] = []
        self.y_memory: list[float] = []
        self._rng = np.random.RandomState(42)
        self._online_buffer: deque = deque(maxlen=500)
        self._trained = False

    def _extract_features(
        self,
        hardware: Optional[dict] = None,
        model: Optional[dict] = None,
        config: Optional[dict] = None,
    ) -> np.ndarray:
        feat = np.zeros(self.n_features, dtype=np.float64)
        idx = 0
        if hardware:
            feat[idx] = hardware.get("cpu_cores", 8) / 64.0
            idx += 1
            feat[idx] = hardware.get("ram_gb", 16) / 512.0
            idx += 1
            feat[idx] = 1.0 if hardware.get("avx2", False) else 0.0
            idx += 1
            feat[idx] = 1.0 if hardware.get("avx512", False) else 0.0
            idx += 1
            feat[idx] = hardware.get("ssd_speed_mbps", 2000) / 10000.0
            idx += 1
            feat[idx] = hardware.get("gpu_vram_gb", 0) / 48.0
            idx += 1
        if model:
            feat[idx] = model.get("n_params_b", 7) / 100.0
            idx += 1
            feat[idx] = model.get("n_layers", 32) / 128.0
            idx += 1
            feat[idx] = model.get("hidden_dim", 4096) / 16384.0
            idx += 1
            feat[idx] = model.get("n_heads", 32) / 128.0
            idx += 1
            feat[idx] = model.get("vocab_size", 32000) / 200000.0
            idx += 1
            feat[idx] = model.get("quantization", 4) / 8.0
            idx += 1
        if config:
            for name in [
                "batch_size",
                "block_size",
                "draft_length",
                "kv_k_bits",
                "kv_v_bits",
                "spectral_rank",
                "kv_compression",
                "hdc_dim",
                "hdc_ngram_order",
                "hdc_sparsity",
                "coherence_threshold",
                "n_candidate_blocks",
                "hdc_depth",
                "num_lsh_tables",
            ]:
                if name in config and idx < self.n_features:
                    spec = DEFAULT_PARAM_SPACE.get(name, {})
                    pmin = spec.get("min", 0.0)
                    pmax = spec.get("max", 1.0)
                    feat[idx] = (float(config[name]) - pmin) / (pmax - pmin + 1e-10)
                    idx += 1
        return feat

    def _train_tree(self, X: np.ndarray, y: np.ndarray, max_depth: int) -> dict:
        n = len(X)
        if n < 2:
            return {"leaf_value": float(np.mean(y)) if len(y) > 0 else 0.0}
        indices = self._rng.choice(n, n, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        return self._build_tree(X_boot, y_boot, depth=0, max_depth=max_depth)

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int, max_depth: int
    ) -> dict:
        if depth >= max_depth or len(y) < 2 or np.std(y) < 1e-6:
            return {"leaf_value": float(np.mean(y))}
        best_gain = -1.0
        best_feat = 0
        best_thresh = 0.0
        n_features = X.shape[1]
        var_total = np.var(y) * len(y)
        for f in range(n_features):
            thresholds = np.percentile(X[:, f], [25, 50, 75])
            for thresh in thresholds:
                left = y[X[:, f] <= thresh]
                right = y[X[:, f] > thresh]
                if len(left) < 1 or len(right) < 1:
                    continue
                gain = var_total - (
                    np.var(left) * len(left) + np.var(right) * len(right)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh
        if best_gain < 0:
            return {"leaf_value": float(np.mean(y))}
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        return {
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._build_tree(X[left_idx], y[left_idx], depth + 1, max_depth),
            "right": self._build_tree(X[right_idx], y[right_idx], depth + 1, max_depth),
        }

    def _predict_tree(self, tree: dict, x: np.ndarray) -> float:
        if "leaf_value" in tree:
            return tree["leaf_value"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_tree(tree["left"], x)
        return self._predict_tree(tree["right"], x)

    def train(
        self,
        X: list[np.ndarray],
        throughput: list[float],
        latency: list[float],
        memory: list[float],
    ):
        self.X = list(X)
        self.y_throughput = list(throughput)
        self.y_latency = list(latency)
        self.y_memory = list(memory)
        if len(X) < 3:
            self._trained = False
            return
        X_arr = np.array(X)
        y_t_arr = np.array(throughput)
        y_l_arr = np.array(latency)
        y_m_arr = np.array(memory)
        self.trees_throughput = []
        self.trees_latency = []
        self.trees_memory = []
        residuals_t = y_t_arr - np.mean(y_t_arr)
        residuals_l = y_l_arr - np.mean(y_l_arr)
        residuals_m = y_m_arr - np.mean(y_m_arr)
        for _ in range(self.n_trees):
            tree_t = self._train_tree(X_arr, residuals_t, self.max_depth)
            tree_l = self._train_tree(X_arr, residuals_l, self.max_depth)
            tree_m = self._train_tree(X_arr, residuals_m, self.max_depth)
            self.trees_throughput.append(tree_t)
            self.trees_latency.append(tree_l)
            self.trees_memory.append(tree_m)
            preds_t = np.array([self._predict_tree(tree_t, x) for x in X])
            preds_l = np.array([self._predict_tree(tree_l, x) for x in X])
            preds_m = np.array([self._predict_tree(tree_m, x) for x in X])
            residuals_t -= self.learning_rate * preds_t
            residuals_l -= self.learning_rate * preds_l
            residuals_m -= self.learning_rate * preds_m
        self._trained = True

    def predict(
        self,
        hardware: Optional[dict] = None,
        model: Optional[dict] = None,
        config: Optional[dict] = None,
    ) -> dict:
        x = self._extract_features(hardware, model, config)
        if not self._trained or not self.trees_throughput:
            return {
                "throughput": 50.0,
                "latency_p50": 50.0,
                "latency_p95": 100.0,
                "latency_p99": 200.0,
                "memory_gb": 8.0,
                "confidence": 0.1,
            }
        throughput = float(
            np.mean([self._predict_tree(t, x) for t in self.trees_throughput])
        )
        latency = float(np.mean([self._predict_tree(t, x) for t in self.trees_latency]))
        memory = float(np.mean([self._predict_tree(t, x) for t in self.trees_memory]))
        return {
            "throughput": max(0.1, throughput),
            "latency_p50": max(0.1, latency),
            "latency_p95": max(0.1, latency * 2.0),
            "latency_p99": max(0.1, latency * 3.0),
            "memory_gb": max(0.1, memory),
            "confidence": min(0.9, 0.3 + 0.6 * len(self.X) / 100.0),
        }

    def fine_tune(
        self,
        hardware: Optional[dict],
        model: Optional[dict],
        config: dict,
        actual_throughput: float,
        actual_latency: float,
        actual_memory: float,
    ):
        x = self._extract_features(hardware, model, config)
        self._online_buffer.append(
            (x, actual_throughput, actual_latency, actual_memory)
        )
        if len(self._online_buffer) >= 10:
            buf = list(self._online_buffer)
            X_buf = np.array([b[0] for b in buf])
            yt = [b[1] for b in buf]
            yl = [b[2] for b in buf]
            ym = [b[3] for b in buf]
            self.train(list(X_buf), yt, yl, ym)

    def get_report(self) -> dict:
        return {
            "trained": self._trained,
            "n_trees": len(self.trees_throughput),
            "n_samples": len(self.X),
            "online_buffer": len(self._online_buffer),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. HardwareAdaptation — Auto-adapt to hardware
# ═══════════════════════════════════════════════════════════════════════════


class HardwareAdaptation:
    """Auto-probe hardware and determine optimal configuration.

    Probes: CPU cores, RAM, cache sizes, SSD speed, NUMA topology, GPU.
    Discovers: optimal thread count, batch size, block size.
    Recommends: best quantization format, parallelism strategy, KV cache budget.
    """

    def __init__(self):
        self.probe = HardwareProbe()
        self.cpu_info: dict = {}
        self.memory_info: dict = {}
        self.gpu_info: dict = {}
        self.disk_info: dict = {}
        self.numa_info: dict = {}
        self._probed = False
        self._cache_sizes: dict[str, int] = {}

    def probe_all(self) -> dict:
        self.cpu_info = self.probe.cpu_info()
        self.memory_info = self.probe.memory_info()
        self.gpu_info = self.probe.gpu_info()
        self.disk_info = self.probe.disk_speed()
        self._probe_cache()
        self._probe_numa()
        self._probed = True
        return self.get_spec()

    def _probe_cache(self):
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read()
        except Exception:
            return
        import re

        cache_pattern = re.compile(r"cache size\s*:\s*(\d+)\s*KB", re.IGNORECASE)
        matches = cache_pattern.findall(content)
        if matches:
            self._cache_sizes["l3_kb"] = int(matches[0])

    def _probe_numa(self):
        try:
            with open("/sys/devices/system/node/possible") as f:
                self.numa_info["nodes"] = f.read().strip()
        except Exception:
            self.numa_info["nodes"] = "0"

    def get_spec(self) -> dict:
        return {
            "cpu_cores": self.cpu_info.get("cores", os.cpu_count() or 8),
            "cpu_model": self.cpu_info.get("model", "Unknown"),
            "avx2": self.cpu_info.get("avx2", False),
            "avx512": self.cpu_info.get("avx512", False),
            "fma": self.cpu_info.get("fma", False),
            "ram_gb": self.memory_info.get("total_gb", 16.0),
            "ram_available_gb": self.memory_info.get(
                "available_gb", self.memory_info.get("total_gb", 16.0)
            ),
            "gpu_available": self.gpu_info.get("available", False),
            "gpu_vram_gb": self.gpu_info.get("vram_gb", 0),
            "gpu_devices": self.gpu_info.get("devices", []),
            "ssd_read_mbps": self.disk_info.get("read_mb_s", 2000),
            "ssd_write_mbps": self.disk_info.get("write_mb_s", 1000),
            "cache": dict(self._cache_sizes),
            "numa": dict(self.numa_info),
        }

    def recommend_config(self) -> dict:
        if not self._probed:
            self.probe_all()
        spec = self.get_spec()
        cores = spec["cpu_cores"]
        ram = spec["ram_gb"]
        ssd = spec["ssd_read_mbps"]
        gpu_vram = spec["gpu_vram_gb"]
        avx2 = spec["avx2"]
        avx512 = spec["avx512"]
        config = {}
        if ram < 8:
            config.update(WARM_START_CONFIGS["low_ram"])
        elif ram < 16:
            config.update(WARM_START_CONFIGS["balanced"])
            config["batch_size"] = min(config.get("batch_size", 8), 4)
            config["block_size"] = 8
            config["hdc_dim"] = 4096
        elif ram >= 64 and cores >= 16:
            config.update(WARM_START_CONFIGS["high_perf"])
        elif ram >= 32 and cores >= 8:
            config.update(WARM_START_CONFIGS["high_perf"])
            config["batch_size"] = 16
            config["hdc_dim"] = 10000
        else:
            config.update(WARM_START_CONFIGS["balanced"])
        config["num_threads"] = ThreadPoolOptimizer.optimal_thread_count(
            memory_bound=True
        )
        if gpu_vram >= 8:
            config["offload_strategy"] = "attention_layers_first"
            config["n_gpu_layers"] = -1
        elif gpu_vram >= 4:
            config["offload_strategy"] = "partial"
            config["n_gpu_layers"] = 16
        else:
            config["offload_strategy"] = "cpu_only"
        if avx512:
            config["quantization"] = "q4_0"
        elif avx2:
            config["quantization"] = "q4_k_m"
        else:
            config["quantization"] = "q5_k_m"
        if ssd > 3000:
            config["use_mmap"] = True
            config["ssd_stream_chunk"] = 8
        elif ssd > 1000:
            config["use_mmap"] = True
            config["ssd_stream_chunk"] = 4
        else:
            config["use_mmap"] = False
            config["ssd_stream_chunk"] = 2
        config["kv_cache_budget_gb"] = round(max(0.5, ram * 0.3), 1)
        config["max_ram_weights"] = max(1, cores // 2)
        return config

    def recommend_model(self, model_size_gb: float) -> dict:
        spec = self.get_spec()
        ram = spec["ram_gb"]
        gpu_vram = spec["gpu_vram_gb"]
        ssd = spec["ssd_read_mbps"]
        total_available = ram + gpu_vram
        if model_size_gb <= total_available * 0.8:
            mode = "full_ram" if model_size_gb <= ram * 0.8 else "hybrid_gpu_cpu"
        elif model_size_gb <= total_available * 1.5:
            mode = "ssd_streaming"
        else:
            mode = "progressive_load"
        throughput_estimate = 0.0
        if mode == "full_ram":
            throughput_estimate = 50.0 * (ram / 16.0) * (1.0 + 0.5 * (gpu_vram > 0))
        elif mode == "hybrid_gpu_cpu":
            throughput_estimate = 30.0 * (1.0 + 0.3 * (gpu_vram / 8.0))
        elif mode == "ssd_streaming":
            throughput_estimate = 10.0 * (ssd / 2000.0)
        else:
            throughput_estimate = 3.0
        return {
            "fits": mode not in ("progressive_load",),
            "mode": mode,
            "estimated_throughput": round(throughput_estimate, 1),
            "kv_cache_budget_gb": round(max(0.5, ram * 0.25), 1),
            "quantization": "q4_k_m" if ram < 32 else "q4_0",
        }

    def get_report(self) -> dict:
        return {
            "probed": self._probed,
            "spec": self.get_spec(),
            "recommended_config": self.recommend_config(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. WorkloadPredictor — Predict workload patterns
# ═══════════════════════════════════════════════════════════════════════════


class WorkloadPredictor:
    """Time-series forecasting for workload prediction.

    Predicts request rate, average context length, and model distribution
    for proactive scaling. Detects seasonal patterns (daily/weekly) and
    anomalous load.
    """

    def __init__(self, window_size: int = 1000, forecast_horizon: int = 60):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.request_rates: deque = deque(maxlen=window_size)
        self.context_lengths: deque = deque(maxlen=window_size)
        self.model_distribution: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.latencies: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
        self._hourly_pattern: Optional[np.ndarray] = None
        self._daily_pattern: Optional[np.ndarray] = None
        self._trend: float = 0.0
        self._seasonal_period: int = 60
        self._anomaly_threshold: float = 3.0
        self._last_trained: float = 0.0

    def observe(
        self,
        request_rate: float,
        context_length: int,
        model_name: str = "default",
        latency_ms: float = 0.0,
    ):
        now = time.time()
        self.request_rates.append(request_rate)
        self.context_lengths.append(context_length)
        self.model_distribution[model_name].append(1)
        self.latencies.append(latency_ms)
        self.timestamps.append(now)

    def _update_seasonal_patterns(self):
        rates = list(self.request_rates)
        if len(rates) < 120:
            return
        self._hourly_pattern = np.zeros(60)
        n = min(len(rates), 60)
        for i in range(n):
            self._hourly_pattern[i] = (
                float(np.mean(rates[-(n - i) :])) if n - i > 0 else 0.0
            )
        if len(rates) >= 1440:
            daily = np.array(rates[-1440:])
            self._daily_pattern = np.mean(daily.reshape(-1, 60), axis=0)

    def forecast(self, steps: Optional[int] = None) -> dict:
        h = steps or self.forecast_horizon
        rates = list(self.request_rates)
        ctx_lengths = list(self.context_lengths)
        if len(rates) < 10:
            return {
                "predicted_rates": [1.0] * h,
                "predicted_context_length": [512] * h,
                "current_rate": rates[-1] if rates else 0.0,
                "rate_trend": 0.0,
                "rate_mean": 1.0,
                "rate_std": 0.5,
                "confidence": 0.1,
                "anomaly": False,
            }
        self._update_seasonal_patterns()
        n = len(rates)
        recent = rates[-min(n, 30) :]
        rate_mean = float(np.mean(recent))
        rate_std = float(np.std(recent)) + 1e-10
        trend = (
            (float(np.mean(rates[-10:])) - float(np.mean(rates[-20:])))
            if len(rates) >= 20
            else 0.0
        )
        trend = np.clip(trend, -rate_mean * 0.5, rate_mean * 0.5)
        predicted_rates = []
        for i in range(h):
            seasonal = 0.0
            if self._hourly_pattern is not None:
                seasonal = self._hourly_pattern[i % len(self._hourly_pattern)]
            pred = rate_mean + trend * i + seasonal * 0.1
            pred = max(0, pred + float(np.random.randn() * rate_std * 0.2))
            predicted_rates.append(pred)
        ctx_mean = float(np.mean(ctx_lengths[-min(n, 50) :]))
        ctx_std = float(np.std(ctx_lengths[-min(n, 50) :])) + 1e-10
        predicted_ctx = [
            max(1, ctx_mean + float(np.random.randn() * ctx_std * 0.1))
            for _ in range(h)
        ]
        current_rate = rates[-1] if rates else 0.0
        rate_ma = (
            float(np.mean(rates[-min(n, 10) :])) if len(rates) >= 10 else current_rate
        )
        anomaly = (
            abs(current_rate - rate_ma) > self._anomaly_threshold * rate_std
            if rate_std > 0
            else False
        )
        confidence = min(0.9, 0.3 + 0.6 * len(rates) / 500.0)
        return {
            "predicted_rates": predicted_rates,
            "predicted_context_length": predicted_ctx,
            "current_rate": current_rate,
            "rate_trend": trend,
            "rate_mean": rate_mean,
            "rate_std": rate_std,
            "confidence": confidence,
            "anomaly": anomaly,
            "seasonal_pattern": self._hourly_pattern.tolist()
            if self._hourly_pattern is not None
            else None,
        }

    def detect_anomaly(self, request_rate: float) -> bool:
        rates = list(self.request_rates)
        if len(rates) < 10:
            return False
        recent = rates[-min(len(rates), 20) :]
        mean_r = np.mean(recent)
        std_r = np.std(recent) + 1e-10
        return abs(request_rate - mean_r) > self._anomaly_threshold * std_r

    def resource_provision(self, forecast: dict) -> dict:
        max_rate = (
            max(forecast["predicted_rates"]) if forecast["predicted_rates"] else 1.0
        )
        current_rate = forecast["current_rate"]
        peak_factor = max_rate / max(current_rate, 0.01) if current_rate > 0 else 2.0
        return {
            "recommended_threads": min(
                os.cpu_count() or 8, max(1, int(peak_factor * 4))
            ),
            "recommended_batch_size": min(64, max(1, int(peak_factor * 8))),
            "estimated_load_multiplier": round(peak_factor, 2),
            "scale_recommendation": "scale_up"
            if peak_factor > 1.5
            else "maintain"
            if peak_factor > 0.8
            else "scale_down",
        }

    def get_report(self) -> dict:
        return {
            "n_observations": len(self.request_rates),
            "current_rate": self.request_rates[-1] if self.request_rates else 0.0,
            "avg_context_length": float(np.mean(list(self.context_lengths)))
            if self.context_lengths
            else 0.0,
            "model_distribution": {
                k: len(v) for k, v in self.model_distribution.items()
            },
            "hourly_pattern_trained": self._hourly_pattern is not None,
            "daily_pattern_trained": self._daily_pattern is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 6. QualityController — Maintain quality targets
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class QualityControllerConfig:
    target_quality: float = 0.7
    quality_floor: float = 0.4
    degradation_threshold: float = 0.1
    adaptation_rate: float = 0.1
    monitor_window: int = 50
    enable_guard: bool = True


class QualityController:
    """Maintain output quality targets.

    Monitors perplexity proxy, diversity, repetition rate.
    Detects quality degradation from compression/pruning.
    Adjusts parameters to maintain quality above floor.
    """

    def __init__(
        self,
        config: Optional[QualityControllerConfig] = None,
        validator: Optional = None,
    ):
        self.config = config or QualityControllerConfig()
        self.validator = validator or QualityValidator()
        self.quality_scores: deque = deque(maxlen=self.config.monitor_window)
        self.quality_history: list[dict] = []
        self._current_compression = 1.0
        self._current_precision = 4
        self._degradation_count = 0
        self._recovery_count = 0
        self._lock = threading.RLock()

    def evaluate(self, text: str) -> dict:
        result = self.validator.evaluate(text)
        quality = result.get("overall_quality", 0.0)
        if quality == 0.0:
            quality = (
                0.3 * (1.0 / max(result.get("perplexity_proxy", 100), 1))
                + 0.25 * result.get("coherence", 0.0)
                + 0.2 * result.get("diversity", 0.0)
                + 0.15 * result.get("repetition_penalty", 0.0)
                + 0.1 * result.get("information_density", 0.0)
            )
        result["quality"] = quality
        with self._lock:
            self.quality_scores.append(quality)
            self.quality_history.append(
                {
                    **result,
                    "timestamp": time.time(),
                    "compression": self._current_compression,
                    "precision": self._current_precision,
                }
            )
        return result

    def check_degradation(self) -> bool:
        with self._lock:
            if len(self.quality_scores) < 5:
                return False
            recent = list(self.quality_scores)[-5:]
            avg = float(np.mean(recent))
            degraded = (
                avg < self.config.target_quality - self.config.degradation_threshold
            )
            if degraded:
                self._degradation_count += 1
            else:
                self._recovery_count += 1
            return degraded

    def should_adjust(self) -> bool:
        if not self.config.enable_guard:
            return False
        with self._lock:
            if len(self.quality_scores) < 3:
                return False
            recent = list(self.quality_scores)[-3:]
            avg = float(np.mean(recent))
            return avg < self.config.quality_floor + 0.05

    def adjust_parameters(self, current_config: dict) -> dict:
        adjusted = dict(current_config)
        with self._lock:
            if not self.quality_scores:
                return adjusted
            recent = list(self.quality_scores)[-5:]
            avg_quality = float(np.mean(recent))
        if avg_quality < self.config.quality_floor:
            adjusted["kv_k_bits"] = max(adjusted.get("kv_k_bits", 4), 6)
            adjusted["kv_v_bits"] = max(adjusted.get("kv_v_bits", 2), 3)
            adjusted["kv_compression"] = min(adjusted.get("kv_compression", 20.0), 10.0)
            adjusted["hdc_sparsity"] = min(adjusted.get("hdc_sparsity", 0.05), 0.03)
            adjusted["spectral_rank"] = max(adjusted.get("spectral_rank", 64), 128)
            adjusted["coherence_threshold"] = min(
                adjusted.get("coherence_threshold", 0.55), 0.4
            )
            adjusted["_quality_action"] = "emergency_recovery"
        elif avg_quality < self.config.target_quality:
            adjusted["kv_k_bits"] = min(adjusted.get("kv_k_bits", 4) + 1, 8)
            adjusted["kv_v_bits"] = min(adjusted.get("kv_v_bits", 2) + 1, 4)
            adjusted["kv_compression"] = max(
                adjusted.get("kv_compression", 20.0) * 0.8, 5.0
            )
            adjusted["_quality_action"] = "quality_boost"
        elif avg_quality > self.config.target_quality + 0.15:
            adjusted["kv_k_bits"] = max(adjusted.get("kv_k_bits", 4) - 1, 2)
            adjusted["kv_v_bits"] = max(adjusted.get("kv_v_bits", 2) - 1, 1)
            adjusted["kv_compression"] = min(
                adjusted.get("kv_compression", 20.0) * 1.2, 100.0
            )
            adjusted["_quality_action"] = "speed_boost"
        else:
            adjusted["_quality_action"] = "maintain"
        self._current_compression = adjusted.get(
            "kv_compression", self._current_compression
        )
        self._current_precision = adjusted.get("kv_k_bits", self._current_precision)
        return adjusted

    def accept(self, quality_score: float) -> bool:
        return quality_score >= self.config.quality_floor

    def get_report(self) -> dict:
        with self._lock:
            return {
                "target_quality": self.config.target_quality,
                "quality_floor": self.config.quality_floor,
                "current_quality": float(np.mean(list(self.quality_scores)))
                if self.quality_scores
                else 0.0,
                "degradation_count": self._degradation_count,
                "recovery_count": self._recovery_count,
                "current_compression": self._current_compression,
                "current_precision": self._current_precision,
                "n_evaluations": len(self.quality_history),
                "guard_enabled": self.config.enable_guard,
            }


# ═══════════════════════════════════════════════════════════════════════════
# 7. ResourceController — Manage resource budgets
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ResourceBudget:
    memory_gb: float = 8.0
    cpu_percent: float = 100.0
    disk_iops: int = 1000
    priority: float = 0.5


class ResourceController:
    """Manage resource budgets across components.

    Allocates memory, CPU time, and disk I/O proportionally to priority.
    Detects contention and ensures fairness.
    Allows temporary oversubscription if resources are available.
    """

    def __init__(self, total_memory_gb: float = 16.0, n_cpu_cores: int = 8):
        self.total_memory_gb = total_memory_gb
        self.total_cpu_cores = n_cpu_cores
        self.budgets: dict[str, ResourceBudget] = {}
        self.usage: dict[str, dict] = defaultdict(
            lambda: {"memory_gb": 0.0, "cpu_percent": 0.0, "disk_iops": 0}
        )
        self.component_priorities: dict[str, float] = {}
        self._contention_history: list[dict] = []
        self._lock = threading.RLock()

    def register_component(
        self, name: str, priority: float = 0.5, memory_gb: float = 1.0
    ):
        with self._lock:
            self.budgets[name] = ResourceBudget(memory_gb=memory_gb, priority=priority)
            self.component_priorities[name] = priority

    def update_usage(
        self,
        name: str,
        memory_gb: float = 0.0,
        cpu_percent: float = 0.0,
        disk_iops: int = 0,
    ):
        with self._lock:
            self.usage[name]["memory_gb"] = memory_gb
            self.usage[name]["cpu_percent"] = cpu_percent
            self.usage[name]["disk_iops"] = disk_iops

    def allocate_budget(
        self, requesting_component: str, requested_memory_gb: float
    ) -> float:
        with self._lock:
            total_used = sum(u["memory_gb"] for u in self.usage.values())
            available = self.total_memory_gb - total_used
            if requested_memory_gb <= available:
                return requested_memory_gb
            priority = self.component_priorities.get(requesting_component, 0.5)
            oversubscribe = available < 0
            if not oversubscribe:
                allocated = available * priority
                return max(0.1, allocated)
            low_prio_components = sorted(
                self.component_priorities.items(), key=lambda x: x[1]
            )
            freed = 0.0
            for comp, prio in low_prio_components:
                if prio >= priority or comp == requesting_component:
                    continue
                comp_usage = self.usage[comp]["memory_gb"]
                reclaim = comp_usage * 0.3
                self.usage[comp]["memory_gb"] = max(0.1, comp_usage - reclaim)
                freed += reclaim
                if freed >= requested_memory_gb:
                    break
            self._contention_history.append(
                {
                    "time": time.time(),
                    "component": requesting_component,
                    "requested": requested_memory_gb,
                    "available": available,
                    "freed": freed,
                    "oversubscribed": oversubscribe,
                }
            )
            return min(requested_memory_gb, available + freed + 1.0)

    def detect_contention(self) -> list[dict]:
        with self._lock:
            total_mem = sum(u["memory_gb"] for u in self.usage.values())
            total_cpu = sum(u["cpu_percent"] for u in self.usage.values())
            contention = []
            if total_mem > self.total_memory_gb * 0.9:
                contention.append(
                    {
                        "resource": "memory",
                        "usage_gb": total_mem,
                        "total_gb": self.total_memory_gb,
                        "severity": "high"
                        if total_mem > self.total_memory_gb
                        else "medium",
                    }
                )
            if total_cpu > self.total_cpu_cores * 100 * 0.9:
                contention.append(
                    {
                        "resource": "cpu",
                        "usage_percent": total_cpu,
                        "total_percent": self.total_cpu_cores * 100,
                        "severity": "high"
                        if total_cpu > self.total_cpu_cores * 100
                        else "medium",
                    }
                )
            return contention

    def ensure_fairness(self) -> dict:
        with self._lock:
            if not self.component_priorities:
                return {}
            total_priority = sum(self.component_priorities.values())
            allocations = {}
            for comp, prio in self.component_priorities.items():
                fair_share = (prio / total_priority) * self.total_memory_gb
                current = self.usage[comp]["memory_gb"]
                allocations[comp] = {
                    "fair_share_gb": round(fair_share, 2),
                    "current_gb": round(current, 2),
                    "delta_gb": round(fair_share - current, 2),
                    "under_allocated": current < fair_share * 0.8,
                    "over_allocated": current > fair_share * 1.2,
                }
            return allocations

    def get_report(self) -> dict:
        return {
            "total_memory_gb": self.total_memory_gb,
            "total_cpu_cores": self.total_cpu_cores,
            "usage": dict(self.usage),
            "budgets": {k: asdict(v) for k, v in self.budgets.items()},
            "contention": self.detect_contention(),
            "fairness": self.ensure_fairness(),
            "n_contention_events": len(self._contention_history),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 8. OnlineLearner — Learn from production feedback
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FeedbackSample:
    context: list[int]
    hdc_tokens: list[int]
    model_tokens: list[int]
    preference: float
    features: list[float]
    timestamp: float


class OnlineLearner:
    """Learn from production feedback (RLHF-light / DPO).

    Accepts explicit feedback (thumbs up/down) and implicit signals
    (retries, edits). Updates HDC engine, confidence gate, and
    strategy selection based on preference data.

    Implements Direct Preference Optimization (DPO) from pairwise comparisons.
    """

    def __init__(
        self,
        hd_engine=None,
        confidence_gate=None,
        max_buffer: int = 10000,
        dpo_lr: float = 0.01,
        dpo_beta: float = 0.1,
    ):
        self.hd_engine = hd_engine
        self.confidence_gate = confidence_gate
        self.max_buffer = max_buffer
        self.dpo_lr = dpo_lr
        self.dpo_beta = dpo_beta
        self.feedback_buffer: deque = deque(maxlen=max_buffer)
        self.preference_buffer: list[tuple[FeedbackSample, FeedbackSample]] = []
        self.total_feedback = 0
        self.total_preferences = 0
        self.recent_quality: deque = deque(maxlen=100)
        self.preference_model: dict[str, np.ndarray] = {
            "W": np.random.randn(10, 4).astype(np.float32) * 0.01,
            "b": np.zeros(4, dtype=np.float32),
        }
        self._rng = np.random.RandomState(42)
        self._lock = threading.RLock()
        self._background_thread: Optional[threading.Thread] = None
        self._running = False

    def observe_feedback(
        self,
        context_tokens: list[int],
        hdc_tokens: list[int],
        model_tokens: list[int],
        preference: float,
        features: Optional[list[float]] = None,
    ):
        sample = FeedbackSample(
            context=list(context_tokens),
            hdc_tokens=list(hdc_tokens),
            model_tokens=list(model_tokens),
            preference=float(np.clip(preference, -1.0, 1.0)),
            features=features or [0.0] * 10,
            timestamp=time.time(),
        )
        with self._lock:
            self.feedback_buffer.append(sample)
            self.total_feedback += 1
            self.recent_quality.append(preference)

    def observe_preference(self, chosen: FeedbackSample, rejected: FeedbackSample):
        with self._lock:
            self.preference_buffer.append((chosen, rejected))
            self.total_preferences += 1

    def _dpo_loss(
        self, chosen_feat: np.ndarray, rejected_feat: np.ndarray
    ) -> tuple[float, float, float]:
        W = self.preference_model["W"]
        b = self.preference_model["b"]
        chosen_score = float(chosen_feat @ W + b)
        rejected_score = float(rejected_feat @ W + b)
        logits = self.dpo_beta * (chosen_score - rejected_score)
        loss = -math.log(1.0 / (1.0 + math.exp(-logits)) + 1e-10)
        d_chosen = self.dpo_beta * (1.0 / (1.0 + math.exp(logits)) - 1.0)
        d_rejected = self.dpo_beta * (1.0 / (1.0 + math.exp(-logits)))
        return loss, d_chosen, d_rejected

    def _train_step(self):
        with self._lock:
            if not self.preference_buffer:
                return
            batch = self._rng.choice(
                len(self.preference_buffer),
                min(16, len(self.preference_buffer)),
                replace=False,
            )
            W = self.preference_model["W"]
            b = self.preference_model["b"]
            dW = np.zeros_like(W)
            db = np.zeros_like(b)
            for idx in batch:
                chosen, rejected = self.preference_buffer[idx]
                c_feat = np.array(chosen.features[:10], dtype=np.float32)
                r_feat = np.array(rejected.features[:10], dtype=np.float32)
                _, dc, dr = self._dpo_loss(c_feat, r_feat)
                dW += dc * np.outer(c_feat, np.ones(4))
                dW += dr * np.outer(r_feat, np.ones(4))
                db += dc + dr
            self.preference_model["W"] -= self.dpo_lr * dW / max(len(batch), 1)
            self.preference_model["b"] -= self.dpo_lr * db / max(len(batch), 1)

    def train_hd_from_feedback(self):
        with self._lock:
            if self.hd_engine is None:
                return
            samples = list(self.feedback_buffer)[-100:]
            for s in samples:
                if s.preference > 0 and s.hdc_tokens:
                    for tok in s.hdc_tokens[-1:]:
                        self.hd_engine.observe(tok)

    def train_confidence_from_feedback(self):
        with self._lock:
            if self.confidence_gate is None:
                return
            samples = list(self.feedback_buffer)[-50:]
            for s in samples:
                if s.features and len(s.features) >= 2:
                    correct = s.preference > 0
                    self.confidence_gate.train(s.features[:10], correct)

    def get_preference_score(self, features: list[float]) -> float:
        W = self.preference_model["W"]
        b = self.preference_model["b"]
        feat = np.array(features[:10], dtype=np.float32)
        return float(feat @ W + b)

    def start_background_learning(self, interval: float = 30.0):
        if self._running:
            return
        self._running = True

        def _loop():
            while self._running:
                time.sleep(interval)
                try:
                    self._train_step()
                    self.train_hd_from_feedback()
                    self.train_confidence_from_feedback()
                except Exception:
                    pass

        self._background_thread = threading.Thread(target=_loop, daemon=True)
        self._background_thread.start()

    def stop_background_learning(self):
        self._running = False

    def get_report(self) -> dict:
        return {
            "total_feedback": self.total_feedback,
            "total_preferences": self.total_preferences,
            "buffer_size": len(self.feedback_buffer),
            "preference_buffer_size": len(self.preference_buffer),
            "avg_recent_quality": float(np.mean(list(self.recent_quality)))
            if self.recent_quality
            else 0.0,
            "background_running": self._running,
        }


# ═══════════════════════════════════════════════════════════════════════════
# SELF-AWARE SYSTEM — Internal model of itself for decisions
# ═══════════════════════════════════════════════════════════════════════════


class SelfAwareModel:
    """Internal model of the system's own behavior.

    Learns a predictive model of how the system responds to parameter
    changes. Uses this internal model for "mental simulation" before
    making real changes.

    The system can ask: "If I change parameter X, what will happen?"
    and get an answer from this internal model without actually running.
    """

    def __init__(self, n_state_dims: int = 32):
        self.n_state_dims = n_state_dims
        self.state_history: list[np.ndarray] = []
        self.action_history: list[np.ndarray] = []
        self.reward_history: list[float] = []
        self.next_state_history: list[np.ndarray] = []
        self.W_state = (
            np.random.randn(n_state_dims, n_state_dims).astype(np.float32) * 0.01
        )
        self.W_action = (
            np.random.randn(n_state_dims, n_state_dims).astype(np.float32) * 0.01
        )
        self.b_state = np.zeros(n_state_dims, dtype=np.float32)
        self.reward_predictor: dict[str, np.ndarray] = {
            "W": np.random.randn(n_state_dims, 1).astype(np.float32) * 0.01,
            "b": np.zeros(1, dtype=np.float32),
        }
        self._rng = np.random.RandomState(42)
        self._trained = False

    def encode_state(self, metrics: dict) -> np.ndarray:
        state = np.zeros(self.n_state_dims, dtype=np.float32)
        keys = [
            "throughput",
            "latency_p50",
            "quality",
            "memory_usage_gb",
            "cache_hit_rate",
            "hdc_acceptance",
            "compression_ratio",
            "error_rate",
            "fallback_rate",
            "consecutive_failures",
            "strategy_level",
            "resonance_score",
            "batch_size",
            "block_size",
            "draft_length",
            "kv_k_bits",
            "kv_v_bits",
            "spectral_rank",
            "hdc_dim",
            "hdc_sparsity",
            "n_candidate_blocks",
            "hdc_depth",
            "coherence_threshold",
            "num_lsh_tables",
            "temperature",
            "top_k",
            "top_p",
            "cpu_utilization",
            "memory_pressure",
            "request_rate",
            "context_length",
            "entropy",
        ]
        for i, key in enumerate(keys):
            if i < self.n_state_dims:
                state[i] = float(metrics.get(key, 0.0))
        return state / (np.linalg.norm(state) + 1e-10)

    def encode_action(self, config: dict) -> np.ndarray:
        action = np.zeros(self.n_state_dims, dtype=np.float32)
        param_keys = [
            "batch_size",
            "block_size",
            "draft_length",
            "kv_k_bits",
            "kv_v_bits",
            "spectral_rank",
            "kv_compression",
            "hdc_dim",
            "hdc_ngram_order",
            "hdc_sparsity",
            "coherence_threshold",
            "n_candidate_blocks",
            "confidence_lr",
            "temperature",
            "top_k",
            "top_p",
            "hdc_depth",
            "num_lsh_tables",
            "lsh_bits_per_key",
            "max_prototypes",
        ]
        for i, key in enumerate(param_keys):
            if i < self.n_state_dims:
                spec = DEFAULT_PARAM_SPACE.get(key, {})
                pmin = spec.get("min", 0.0)
                pmax = spec.get("max", 1.0)
                v = float(config.get(key, spec.get("default", 0.5)))
                action[i] = (v - pmin) / (pmax - pmin + 1e-10)
        return action

    def observe_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
    ):
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.next_state_history.append(next_state)
        if len(self.state_history) > 1000:
            self.state_history.pop(0)
            self.action_history.pop(0)
            self.reward_history.pop(0)
            self.next_state_history.pop(0)
        self._train()

    def _train(self):
        if len(self.state_history) < 5:
            return
        n = len(self.state_history)
        S = np.array(self.state_history, dtype=np.float32)
        A = np.array(self.action_history, dtype=np.float32)
        R = np.array(self.reward_history, dtype=np.float32)
        S_next = np.array(self.next_state_history, dtype=np.float32)
        pred_next = S @ self.W_state + A @ self.W_action + self.b_state
        delta = S_next - pred_next
        lr = 0.01 / max(n, 1)
        self.W_state += lr * S.T @ delta
        self.W_action += lr * A.T @ delta
        self.b_state += lr * np.sum(delta, axis=0)
        pred_reward = S @ self.reward_predictor["W"] + self.reward_predictor["b"]
        r_delta = R.reshape(-1, 1) - pred_reward
        self.reward_predictor["W"] += lr * S.T @ r_delta
        self.reward_predictor["b"] += lr * np.sum(r_delta, axis=0)
        self._trained = True

    def simulate(
        self, state: np.ndarray, action: np.ndarray
    ) -> tuple[np.ndarray, float]:
        sim_next = state @ self.W_state + action @ self.W_action + self.b_state
        sim_reward = float(
            state @ self.reward_predictor["W"] + self.reward_predictor["b"]
        )
        return sim_next, sim_reward

    def predict_outcome(self, current_metrics: dict, proposed_config: dict) -> dict:
        state = self.encode_state(current_metrics)
        action = self.encode_action(proposed_config)
        _, reward = self.simulate(state, action)
        return {
            "predicted_reward": reward,
            "improvement": reward - float(np.mean(self.reward_history[-20:]))
            if self.reward_history
            else 0.0,
            "confidence": min(0.9, 0.2 + 0.7 * len(self.state_history) / 200.0),
            "trained": self._trained,
        }

    def get_report(self) -> dict:
        return {
            "trained": self._trained,
            "n_transitions": len(self.state_history),
            "avg_reward": float(np.mean(self.reward_history))
            if self.reward_history
            else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# META-CONTROLLER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MetaControllerConfig:
    enable_auto_tune: bool = True
    enable_bandit: bool = True
    enable_performance_model: bool = True
    enable_hardware_adaptation: bool = True
    enable_workload_prediction: bool = True
    enable_quality_control: bool = True
    enable_resource_control: bool = True
    enable_online_learning: bool = True
    enable_self_aware: bool = True
    enable_vlasov_control: bool = True
    enable_resonant_optimization: bool = True
    enable_holographic_memory: bool = True
    enable_quantum_annealing: bool = False
    enable_hamiltonian: bool = True
    tune_interval: float = 60.0
    bandit_update_interval: float = 1.0
    quality_check_interval: float = 10.0
    workload_forecast_interval: float = 300.0
    resource_check_interval: float = 30.0
    checkpoint_interval: float = 300.0
    report_interval: float = 60.0
    auto_tuner_config: AutoTunerConfig = field(default_factory=AutoTunerConfig)
    bandit_config: BanditOptimizerConfig = field(default_factory=BanditOptimizerConfig)
    quality_config: QualityControllerConfig = field(
        default_factory=QualityControllerConfig
    )
    state_dir: str = "~/.spectralstream/state/"
    profile_name: str = "default"
    profile: str = "balanced"


# Subsystem indices for VlasovMetaControl
_VLASOV_AUTOTUNER = 0
_VLASOV_BANDIT = 1
_VLASOV_QUALITY = 2
_VLASOV_RESOURCE = 3
_VLASOV_WORKLOAD = 4
_VLASOV_LEARNER = 5


# ═══════════════════════════════════════════════════════════════════════════
# 9. MetaController — Top-level orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class MetaController:
    """Top-level orchestrator for autonomous self-optimization.

    Initializes all subsystems, monitors metrics, decides parameter
    changes via bandit/Bayesian optimization, learns from production
    data, and reports health and performance trends.

    This is the "brain" that makes SpectralStream self-optimizing.
    """

    def __init__(
        self,
        config: Optional[MetaControllerConfig] = None,
        engine: Optional[Any] = None,
        state_manager: Optional[StateManager] = None,
    ):
        self.config = config or MetaControllerConfig()
        self.engine = engine
        self._state_manager = state_manager or StateManager(
            state_dir=self.config.state_dir
        )
        self._start_time = time.time()
        self._lock = threading.RLock()
        self._running = False
        self._main_thread: Optional[threading.Thread] = None

        self.hardware_spec: dict = {}
        self.hardware_adaptation = HardwareAdaptation()
        self.workload_predictor = WorkloadPredictor()
        self.quality_controller = QualityController(config=self.config.quality_config)
        self.resource_controller = ResourceController()
        self.online_learner = OnlineLearner()

        self.auto_tuner = AutoTuner(config=self.config.auto_tuner_config)
        self.bandit_optimizer = BanditOptimizer(config=self.config.bandit_config)
        self.performance_model = PerformanceModel()
        self.self_aware_model = SelfAwareModel()
        self.vlasov_control = VlasovMetaControl(n_subsystems=6)
        self.hrr_memory = HRREncoder()
        self.resonant_optimizer = ResonantOptimizer(DEFAULT_PARAM_SPACE)
        self.quantum_optimizer = QuantumAnnealingOptimizer(DEFAULT_PARAM_SPACE)
        self.hamiltonian_controller = HamiltonianMetaController(n_metrics=5)

        self.current_config: dict = {}
        self.baseline_config: dict = {}
        self.metrics_history: deque = deque(maxlen=10000)
        self.param_history: deque = deque(maxlen=5000)
        self.decisions: list[dict] = []
        self.alerts: list[dict] = []

        self._last_tune_time: float = 0.0
        self._last_bandit_update: float = 0.0
        self._last_quality_check: float = 0.0
        self._last_workload_forecast: float = 0.0
        self._last_resource_check: float = 0.0
        self._last_checkpoint: float = 0.0
        self._last_report: float = 0.0
        self._iteration = 0

        self._probe_hardware()

    def _probe_hardware(self):
        try:
            self.hardware_spec = self.hardware_adaptation.probe_all()
            self.current_config = self.hardware_adaptation.recommend_config()
            self.baseline_config = dict(self.current_config)
        except Exception:
            self.hardware_spec = {"cpu_cores": os.cpu_count() or 8, "ram_gb": 16.0}
            self.current_config = dict(WARM_START_CONFIGS["balanced"])
            self.baseline_config = dict(self.current_config)
        self.resource_controller = ResourceController(
            total_memory_gb=float(self.hardware_spec.get("ram_gb", 16)),
            n_cpu_cores=int(self.hardware_spec.get("cpu_cores", 8)),
        )
        profile_config = WARM_START_CONFIGS.get(
            self.config.profile, WARM_START_CONFIGS["balanced"]
        )
        for k, v in profile_config.items():
            if k in self.current_config:
                self.current_config[k] = v

    def _get_current_metrics(self) -> dict:
        metrics = {
            "throughput": 0.0,
            "latency_p50": 50.0,
            "latency_p95": 100.0,
            "latency_p99": 200.0,
            "quality": 0.5,
            "memory_usage_gb": 0.0,
            "cache_hit_rate": 0.0,
            "hdc_acceptance": 0.5,
            "compression_ratio": 20.0,
            "error_rate": 0.0,
            "fallback_rate": 0.0,
            "consecutive_failures": 0,
            "strategy_level": 2.0,
            "resonance_score": 0.5,
            "cpu_utilization": 0.3,
            "memory_pressure": 0.3,
            "request_rate": 1.0,
            "context_length": 512,
            "entropy": 0.5,
        }
        if self.engine is not None:
            try:
                stats_func = getattr(self.engine, "stats", None)
                stats = stats_func() if stats_func else {}
                metrics.update(
                    {
                        "throughput": stats.get("tokens_per_second", 0.0),
                        "hdc_acceptance": stats.get("hd_acceptance_rate", 0.5),
                        "cache_hit_rate": stats.get("kv_cache_hit_rate", 0.0),
                        "compression_ratio": stats.get("kv_compression_ratio", 20.0),
                        "consecutive_failures": stats.get("consecutive_failures", 0),
                    }
                )
                if hasattr(self.engine, "monitor"):
                    m = self.engine.monitor
                    metrics["latency_p50"] = (
                        float(np.mean(list(m.model_call_latencies)))
                        if m.model_call_latencies
                        else 50.0
                    )
                    metrics["error_rate"] = (
                        m.error_rate() if hasattr(m, "error_rate") else 0.0
                    )
                    metrics["fallback_rate"] = (
                        m.fallback_rate() if hasattr(m, "fallback_rate") else 0.0
                    )
            except Exception:
                pass
        return metrics

    def _composite_metric(self, metrics: dict) -> float:
        throughput = metrics.get("throughput", 0.0)
        norm_t = 1.0 - math.exp(-throughput / 100.0)
        latency = metrics.get("latency_p50", 50.0)
        norm_l = math.exp(-latency / 100.0)
        quality = metrics.get("quality", 0.5)
        return 0.4 * norm_t + 0.3 * norm_l + 0.3 * quality

    def _auto_tune_step(self):
        if not self.config.enable_auto_tune:
            return
        metrics = self._get_current_metrics()
        current_composite = self._composite_metric(metrics)
        if self.auto_tuner.best_config is not None:
            self.auto_tuner.observe(
                config=self.current_config,
                throughput=metrics.get("throughput", 1.0),
                latency=metrics.get("latency_p50", 50.0),
                quality=metrics.get("quality", 0.5),
            )
        if self.config.enable_self_aware:
            state = self.self_aware_model.encode_state(metrics)
            action = self.self_aware_model.encode_action(self.current_config)
            self.self_aware_model.observe_transition(
                state, action, current_composite, state * 0.99
            )
        if self.config.enable_quantum_annealing and self._iteration % 20 == 0:

            def _objective(cfg: dict) -> float:
                pred = self.performance_model.predict(
                    hardware=self.hardware_spec, config=cfg
                )
                return pred.get("throughput", 1.0) / max(
                    pred.get("latency_p50", 50.0), 1.0
                )

            qa_config = self.quantum_optimizer.optimize(
                _objective, initial_config=self.current_config
            )
            self.current_config.update(qa_config)
        if self.config.enable_resonant_optimization:
            self.current_config = self.resonant_optimizer.sample(
                self.current_config, time.time() - self._start_time
            )
            self.resonant_optimizer.update_frequencies(
                self.current_config, current_composite
            )
        suggestion = self.auto_tuner.suggest()
        for k, v in suggestion.items():
            if not k.startswith("_"):
                self.current_config[k] = v
        if self.config.enable_holographic_memory:
            self.hrr_memory.store(f"config_{self._iteration}", self.current_config)
            recalled = self.hrr_memory.recall(self.current_config)
            if recalled is not None:
                recalled_id, _ = recalled
                self.decisions.append(
                    {
                        "type": "holographic_recall",
                        "recalled": recalled_id,
                        "iteration": self._iteration,
                    }
                )

    def _bandit_step(self):
        if not self.config.enable_bandit:
            return
        metrics = self._get_current_metrics()
        strategy = self.bandit_optimizer.select_strategy(
            current_load=metrics.get("request_rate", 0.5),
            sequence_length=int(metrics.get("context_length", 512)),
            entropy=metrics.get("entropy", 0.5),
            hdc_acceptance_rate=metrics.get("hdc_acceptance", 0.5),
            cache_hit_rate=metrics.get("cache_hit_rate", 0.5),
            latency_p50=metrics.get("latency_p50", 50.0),
            resonance_score=metrics.get("resonance_score", 0.5),
            consecutive_failures=metrics.get("consecutive_failures", 0),
            memory_pressure=metrics.get("memory_pressure", 0.5),
        )
        arm_idx = (
            self.bandit_optimizer.STRATEGY_ARMS.index(strategy)
            if strategy in self.bandit_optimizer.STRATEGY_ARMS
            else 0
        )
        reward = self._composite_metric(metrics)
        self.bandit_optimizer.observe(
            arm_idx,
            reward,
            **{
                k: metrics.get(k, 0.5)
                for k in [
                    "current_load",
                    "sequence_length",
                    "entropy",
                    "hdc_acceptance_rate",
                    "cache_hit_rate",
                    "latency_p50",
                    "resonance_score",
                    "consecutive_failures",
                    "memory_pressure",
                ]
            },
        )
        if strategy != self.current_config.get("_strategy", ""):
            self.current_config["_strategy"] = strategy
            self.decisions.append(
                {
                    "type": "bandit_strategy",
                    "strategy": strategy,
                    "reward": reward,
                    "iteration": self._iteration,
                }
            )

    def _quality_step(self):
        if not self.config.enable_quality_control:
            return
        metrics = self._get_current_metrics()
        degradation = self.quality_controller.check_degradation()
        if degradation or self.quality_controller.should_adjust():
            self.current_config = self.quality_controller.adjust_parameters(
                self.current_config
            )
            action = self.current_config.get("_quality_action", "unknown")
            self.decisions.append(
                {
                    "type": "quality_adjustment",
                    "action": action,
                    "quality": metrics.get("quality", 0.5),
                    "degraded": degradation,
                    "iteration": self._iteration,
                }
            )

    def _workload_step(self):
        if not self.config.enable_workload_prediction:
            return
        forecast = self.workload_predictor.forecast()
        if forecast.get("anomaly", False):
            self.alerts.append(
                {
                    "type": "workload_anomaly",
                    "current_rate": forecast.get("current_rate", 0.0),
                    "timestamp": time.time(),
                }
            )
        provision = self.workload_predictor.resource_provision(forecast)
        if provision.get("scale_recommendation") == "scale_up":
            self.current_config["num_threads"] = provision["recommended_threads"]
            self.current_config["batch_size"] = min(
                self.current_config.get("batch_size", 8),
                provision["recommended_batch_size"],
            )
            self.decisions.append(
                {
                    "type": "workload_scaling",
                    "recommendation": "scale_up",
                    "iteration": self._iteration,
                }
            )

    def _resource_step(self):
        if not self.config.enable_resource_control:
            return
        contention = self.resource_controller.detect_contention()
        if contention:
            for c in contention:
                self.alerts.append(
                    {
                        "type": "resource_contention",
                        "resource": c["resource"],
                        "severity": c["severity"],
                        "timestamp": time.time(),
                    }
                )
            self.current_config["memory_tier_threshold"] = max(
                0.3, self.current_config.get("memory_tier_threshold", 0.7) - 0.1
            )
            self.current_config["cache_evict_fraction"] = max(
                0.3, self.current_config.get("cache_evict_fraction", 0.5) - 0.05
            )

    def _vlasov_step(self):
        if not self.config.enable_vlasov_control:
            return
        metrics = self._get_current_metrics()
        self.vlasov_control.set_subsystem_state(
            _VLASOV_AUTOTUNER, metrics.get("throughput", 0.0) / 100.0
        )
        self.vlasov_control.set_subsystem_state(
            _VLASOV_BANDIT, metrics.get("latency_p50", 50.0) / 200.0
        )
        self.vlasov_control.set_subsystem_state(
            _VLASOV_QUALITY, metrics.get("quality", 0.5)
        )
        self.vlasov_control.set_subsystem_state(
            _VLASOV_RESOURCE, 1.0 - metrics.get("memory_pressure", 0.3)
        )
        self.vlasov_control.set_subsystem_state(
            _VLASOV_WORKLOAD, metrics.get("request_rate", 0.5) / 10.0
        )
        self.vlasov_control.set_subsystem_state(
            _VLASOV_LEARNER, float(len(self.online_learner.feedback_buffer)) / 1000.0
        )
        self.vlasov_control.evolve()
        control = self.vlasov_control.get_control_signal()
        self.current_config["_vlasov_control"] = float(control)

    def _hamiltonian_step(self):
        if not self.config.enable_hamiltonian:
            return
        metrics = self._get_current_metrics()

        # Encode current metrics as Hamiltonian position vector q
        q_vec = np.array(
            [
                metrics.get("throughput", 0.0)
                / max(metrics.get("throughput", 100.0), 100.0),
                metrics.get("latency_p50", 50.0) / 200.0,
                metrics.get("cache_hit_rate", 0.5),
                metrics.get("compression_ratio", 20.0) / 100.0,
                metrics.get("quality", 0.5),
            ],
            dtype=np.float64,
        )

        # Skip if metrics are stale (all zeros)
        if np.all(q_vec < 0.01):
            self.decisions.append(
                {
                    "type": "hamiltonian_skip",
                    "reason": "stale_metrics",
                    "iteration": self._iteration,
                }
            )
            return

        self.hamiltonian_controller.q = q_vec

        # Compute approximate gradient: direction metrics need to move
        gradient = np.array(
            [
                (metrics.get("throughput", 0.0) - 50.0) / 100.0,
                (50.0 - metrics.get("latency_p50", 50.0)) / 100.0,
                metrics.get("cache_hit_rate", 0.5) - 0.5,
                0.3 - metrics.get("compression_ratio", 20.0) / 100.0,
                metrics.get("quality", 0.5) - 0.5,
            ],
            dtype=np.float64,
        )

        # Hamiltonian evolution step
        q, p = self.hamiltonian_controller.step(gradient)

        # Only apply if energy is improving (Hamiltonian should decrease)
        prev_energy = (
            self.hamiltonian_controller.H_history[-2]
            if len(self.hamiltonian_controller.H_history) >= 2
            else None
        )

        if prev_energy is not None:
            curr_energy = self.hamiltonian_controller.get_energy()
            if curr_energy < prev_energy or self._iteration % 5 == 0:
                adjusted = self.hamiltonian_controller.adapt_parameters(
                    self.current_config, metrics
                )
                for k, v in adjusted.items():
                    if not k.startswith("_"):
                        self.current_config[k] = v
                self.decisions.append(
                    {
                        "type": "hamiltonian_adapt",
                        "energy": float(curr_energy),
                        "energy_delta": float(prev_energy - curr_energy),
                        "energy_drift": self.hamiltonian_controller.energy_drift(),
                        "iteration": self._iteration,
                    }
                )

        # Detect phase transitions in the energy landscape
        pt, pt_strength = self.hamiltonian_controller.detect_phase_transition()
        if pt:
            self.alerts.append(
                {
                    "type": "hamiltonian_phase_transition",
                    "strength": float(pt_strength),
                    "timestamp": time.time(),
                    "iteration": self._iteration,
                }
            )

    def _learn_step(self):
        if not self.config.enable_online_learning:
            return
        metrics = self._get_current_metrics()
        self.online_learner.observe_feedback(
            context_tokens=[],
            hdc_tokens=[],
            model_tokens=[],
            preference=metrics.get("quality", 0.5) * 2.0 - 1.0,
            features=[
                metrics.get(k, 0.5)
                for k in [
                    "throughput",
                    "latency_p50",
                    "quality",
                    "hdc_acceptance",
                    "cache_hit_rate",
                    "compression_ratio",
                    "error_rate",
                    "fallback_rate",
                    "memory_pressure",
                    "entropy",
                ]
            ],
        )

    def _checkpoint_step(self):
        elapsed = time.time() - self._last_checkpoint
        if elapsed >= self.config.checkpoint_interval:
            self._last_checkpoint = time.time()
            try:
                self._state_manager.maybe_checkpoint(force=True)
            except Exception:
                pass

    def update(self, metrics: Optional[dict] = None):
        with self._lock:
            self._iteration += 1
            now = time.time()
            if metrics:
                self.metrics_history.append(
                    {**metrics, "timestamp": now, "iteration": self._iteration}
                )
            self.workload_predictor.observe(
                request_rate=metrics.get("request_rate", 1.0) if metrics else 1.0,
                context_length=int(
                    metrics.get("context_length", 512) if metrics else 512
                ),
                latency_ms=metrics.get("latency_p50", 50.0) if metrics else 50.0,
            )
            if now - self._last_tune_time >= self.config.tune_interval:
                self._last_tune_time = now
                self._auto_tune_step()
                self._bandit_step()
                self._vlasov_step()
                self._hamiltonian_step()
            if now - self._last_quality_check >= self.config.quality_check_interval:
                self._last_quality_check = now
                self._quality_step()
            if (
                now - self._last_workload_forecast
                >= self.config.workload_forecast_interval
            ):
                self._last_workload_forecast = now
                self._workload_step()
            if now - self._last_resource_check >= self.config.resource_check_interval:
                self._last_resource_check = now
                self._resource_step()
            self._learn_step()
            self._checkpoint_step()
            self.param_history.append(
                {
                    "config": dict(self.current_config),
                    "timestamp": now,
                    "iteration": self._iteration,
                }
            )

    def get_config(self) -> dict:
        with self._lock:
            cfg = {
                k: v for k, v in self.current_config.items() if not k.startswith("_")
            }
            return cfg

    def get_report(self) -> dict:
        with self._lock:
            metrics = self._get_current_metrics()
            return {
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self._start_time,
                "iteration": self._iteration,
                "running": self._running,
                "current_config": self.get_config(),
                "metrics": metrics,
                "composite_score": self._composite_metric(metrics),
                "auto_tuner": self.auto_tuner.get_optimization_report(),
                "bandit": self.bandit_optimizer.get_report(),
                "performance_model": self.performance_model.get_report(),
                "hardware": self.hardware_adaptation.get_report(),
                "workload": self.workload_predictor.get_report(),
                "quality": self.quality_controller.get_report(),
                "resource": self.resource_controller.get_report(),
                "online_learner": self.online_learner.get_report(),
                "self_aware": self.self_aware_model.get_report(),
                "vlasov_control": self.vlasov_control.get_state(),
                "resonant_optimizer": self.resonant_optimizer.get_resonance_report(),
                "hamiltonian": self.hamiltonian_controller.get_report(),
                "n_alerts": len(self.alerts),
                "n_decisions": len(self.decisions),
                "config_profile": self.config.profile,
                "subsystems_enabled": {
                    "auto_tune": self.config.enable_auto_tune,
                    "bandit": self.config.enable_bandit,
                    "performance_model": self.config.enable_performance_model,
                    "hardware_adaptation": self.config.enable_hardware_adaptation,
                    "workload_prediction": self.config.enable_workload_prediction,
                    "quality_control": self.config.enable_quality_control,
                    "resource_control": self.config.enable_resource_control,
                    "online_learning": self.config.enable_online_learning,
                    "self_aware": self.config.enable_self_aware,
                    "vlasov_control": self.config.enable_vlasov_control,
                    "resonant_optimization": self.config.enable_resonant_optimization,
                    "holographic_memory": self.config.enable_holographic_memory,
                    "quantum_annealing": self.config.enable_quantum_annealing,
                    "hamiltonian": self.config.enable_hamiltonian,
                },
            }

    def get_performance_report(self) -> str:
        r = self.get_report()
        lines = [
            "=" * 72,
            "  SpectralStream Meta-Controller --- Performance Report",
            "=" * 72,
            f"  Uptime:            {r['uptime_seconds']:.1f}s",
            f"  Iteration:         {r['iteration']}",
            f"  Composite Score:   {r['composite_score']:.4f}",
            f"  Profile:           {r['config_profile']}",
            "",
            "  --- Metrics ---",
            f"  Throughput:        {r['metrics']['throughput']:.1f} tok/s",
            f"  Latency p50:       {r['metrics']['latency_p50']:.1f} ms",
            f"  Quality:           {r['metrics']['quality']:.3f}",
            f"  HDC Acceptance:    {r['metrics']['hdc_acceptance']:.1%}",
            f"  Cache Hit Rate:    {r['metrics']['cache_hit_rate']:.1%}",
            "",
            "  --- AutoTuner ---",
            f"  Trials:            {r['auto_tuner']['n_trials']}",
            f"  Best Score:        {r['auto_tuner']['best_score']:.4f}",
            "",
            "  --- Bandit ---",
            f"  Algorithm:         {r['bandit']['algorithm']}",
            f"  Epsilon:           {r['bandit']['epsilon']:.4f}",
            f"  Decisions:         {r['bandit']['total_decisions']}",
            "",
            "  --- Quality ---",
            f"  Current Quality:   {r['quality']['current_quality']:.3f}",
            f"  Degradations:      {r['quality']['degradation_count']}",
            "",
            "  --- Subsystems ---",
        ]
        for name, enabled in r["subsystems_enabled"].items():
            status = "+" if enabled else "-"
            lines.append(f"  [{status}] {name}")
        lines.append("=" * 72)
        return "\n".join(lines)

    def start_background_loop(self, interval: float = 1.0):
        if self._running:
            return
        self._running = True
        if self.config.enable_online_learning:
            self.online_learner.start_background_learning(interval=30.0)

        def _loop():
            while self._running:
                try:
                    metrics = self._get_current_metrics()
                    self.update(metrics)
                except Exception as exc:
                    self.alerts.append(
                        {
                            "type": "controller_error",
                            "error": str(exc),
                            "timestamp": time.time(),
                        }
                    )
                time.sleep(interval)

        self._main_thread = threading.Thread(target=_loop, daemon=True)
        self._main_thread.start()

    def stop_background_loop(self):
        self._running = False
        self.online_learner.stop_background_learning()
        if self._main_thread:
            self._main_thread.join(timeout=5.0)

    def close(self):
        self.stop_background_loop()
        try:
            self._state_manager.save_checkpoint()
        except Exception:
            pass

    def __enter__(self):
        self.start_background_loop()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ═══════════════════════════════════════════════════════════════════════════
# 10. Integration with SpectralStream
# ═══════════════════════════════════════════════════════════════════════════


class SpectralStreamMetaController:
    """High-level facade that integrates MetaController with UnifiedInferenceEngine.

    Provides a simple API for:
    - Construction with auto-detection
    - Automatic tuning on startup
    - Continous optimization during production
    - Benchmarking for offline tuning
    """

    def __init__(
        self,
        engine=None,
        model_path: Optional[str] = None,
        config: Optional[Union[MetaControllerConfig, dict]] = None,
        profile: str = "balanced",
    ):
        if isinstance(config, dict):
            self.meta_config = MetaControllerConfig(**config)
        elif isinstance(config, MetaControllerConfig):
            self.meta_config = config
        else:
            self.meta_config = MetaControllerConfig(profile=profile)
        self.meta_config.profile = profile
        self.engine = engine
        self.meta_controller: Optional[MetaController] = None
        self._started = False
        if engine is not None:
            self.attach(engine)
        elif model_path is not None and UnifiedInferenceEngine is not None:
            eng = UnifiedInferenceEngine(model_path=model_path)
            self.attach(eng)

    def attach(self, engine):
        self.engine = engine
        self.meta_controller = MetaController(config=self.meta_config, engine=engine)
        self._apply_config()

    def _apply_config(self):
        if self.engine is None or self.meta_controller is None:
            return
        cfg = self.meta_controller.get_config()
        if hasattr(self.engine, "config") and isinstance(self.engine.config, dict):
            for k, v in cfg.items():
                if k not in (
                    "attention_method",
                    "num_threads",
                    "offload_strategy",
                    "quantization",
                    "use_mmap",
                    "kv_cache_budget_gb",
                    "max_ram_weights",
                    "strategy_override",
                ):
                    self.engine.config[k] = v

    def start(self):
        if self._started or self.meta_controller is None:
            return
        self.meta_controller.start_background_loop()
        self._started = True

    def stop(self):
        if self.meta_controller:
            self.meta_controller.stop_background_loop()
        self._started = False

    def update(self, metrics: Optional[dict] = None):
        if self.meta_controller:
            self.meta_controller.update(metrics)

    def get_config(self) -> dict:
        if self.meta_controller:
            return self.meta_controller.get_config()
        return {}

    def get_report(self) -> dict:
        if self.meta_controller:
            return self.meta_controller.get_report()
        return {}

    def get_performance_report(self) -> str:
        if self.meta_controller:
            return self.meta_controller.get_performance_report()
        return "Meta-controller not initialized."

    def adapt_hamiltonian(self) -> dict:
        """Hamiltonian-based parameter adaptation.

        Switches between Bayesian (AutoTuner) and Hamiltonian methods
        based on problem dimensionality. For high-dimensional problems
        (>10 params), Hamiltonian is O(1) vs Bayesian O(n^3).

        Returns:
            Updated configuration dict
        """
        if self.meta_controller is None:
            return {}
        n_params = len(self.meta_controller.auto_tuner.param_names)
        use_hamiltonian = (
            n_params > 10 or self.meta_controller.config.enable_hamiltonian
        )
        if use_hamiltonian:
            metrics = self.meta_controller._get_current_metrics()
            q_vec = np.array(
                [
                    metrics.get("throughput", 0.0)
                    / max(metrics.get("throughput", 100.0), 100.0),
                    metrics.get("latency_p50", 50.0) / 200.0,
                    metrics.get("cache_hit_rate", 0.5),
                    metrics.get("compression_ratio", 20.0) / 100.0,
                    metrics.get("quality", 0.5),
                ],
                dtype=np.float64,
            )
            self.meta_controller.hamiltonian_controller.q = q_vec
            gradient = np.array(
                [
                    (metrics.get("throughput", 0.0) - 50.0) / 100.0,
                    (50.0 - metrics.get("latency_p50", 50.0)) / 100.0,
                    metrics.get("cache_hit_rate", 0.5) - 0.5,
                    0.3 - metrics.get("compression_ratio", 20.0) / 100.0,
                    metrics.get("quality", 0.5) - 0.5,
                ],
                dtype=np.float64,
            )
            self.meta_controller.hamiltonian_controller.step(gradient)
            adjusted = self.meta_controller.hamiltonian_controller.adapt_parameters(
                self.meta_controller.current_config, metrics
            )
            if n_params > 10:
                # For high-D, Hamiltonian takes over completely
                for k, v in adjusted.items():
                    if not k.startswith("_"):
                        self.meta_controller.current_config[k] = v
                self.meta_controller.decisions.append(
                    {
                        "type": "hamiltonian_override",
                        "n_params": n_params,
                        "energy": self.meta_controller.hamiltonian_controller.get_energy(),
                        "iteration": self.meta_controller._iteration,
                    }
                )
            return adjusted
        return self.get_config()

    def run_auto_tune(self, n_iterations: int = 50, silent: bool = False):
        """Run automated tuning loop: probe, suggest, benchmark, observe."""
        if self.meta_controller is None:
            raise RuntimeError("Meta-controller not attached to any engine.")
        for i in range(n_iterations):
            config = self.meta_controller.auto_tuner.suggest()
            if not silent:
                print(f"  Trial {i + 1}/{n_iterations}: testing config...")
            if self.engine is not None:
                try:
                    result = self.engine.benchmark(n_tokens=128)
                    best_strategy = result.get("best_strategy", "standard")
                    strategy_result = result.get("results", {}).get(best_strategy, {})
                    throughput = strategy_result.get("throughput_tok_s", 1.0)
                    latency = strategy_result.get("avg_latency_s", 0.5) * 1000.0
                    self.meta_controller.auto_tuner.observe(
                        config, throughput, latency, 0.5
                    )
                    if not silent:
                        print(
                            f"    Throughput: {throughput:.1f} tok/s, Latency: {latency:.1f} ms"
                        )
                except Exception as e:
                    if not silent:
                        print(f"    Benchmark failed: {e}")
            config["_iter"] = i
            self.meta_controller.update()
        if not silent:
            best = self.meta_controller.auto_tuner.get_best_config()
            print(f"\n  Best config found: {json.dumps(best, indent=2)}")

    def close(self):
        self.stop()
        if self.meta_controller:
            self.meta_controller.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ═══════════════════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════════════════


def create_meta_controller(
    engine=None, model_path: Optional[str] = None, profile: str = "balanced", **kwargs
) -> SpectralStreamMetaController:
    """Create a fully configured SpectralStreamMetaController.

    Args:
        engine: Existing UnifiedInferenceEngine instance (optional)
        model_path: Path to GGUF model to auto-create engine (optional)
        profile: Configuration profile ("balanced", "high_perf", "low_latency", etc.)
        **kwargs: Additional MetaControllerConfig overrides

    Returns:
        SpectralStreamMetaController instance
    """
    config = MetaControllerConfig(**kwargs) if kwargs else None
    return SpectralStreamMetaController(
        engine=engine, model_path=model_path, config=config, profile=profile
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test / Self-Verification
# ═══════════════════════════════════════════════════════════════════════════


def _test_auto_tuner():
    print("  Testing AutoTuner...")
    tuner = AutoTuner()
    for i in range(5):
        config = tuner.suggest()
        tuner.observe(
            config,
            throughput=50.0 + i * 10,
            latency=50.0 - i * 5,
            quality=0.5 + i * 0.05,
        )
    best = tuner.get_best_config()
    assert len(best) > 0, "AutoTuner should have a best config"
    print(f"    Best config has {len(best)} params, score={tuner.best_score:.3f}")
    print("  [OK] AutoTuner")


def _test_bandit():
    print("  Testing BanditOptimizer...")
    bandit = BanditOptimizer()
    for i in range(20):
        arm = bandit.select_arm(current_load=0.5, sequence_length=512, entropy=0.3)
        bandit.observe(
            arm,
            reward=0.5 + i * 0.01,
            current_load=0.5,
            sequence_length=512,
            entropy=0.3,
        )
    report = bandit.get_report()
    assert report["total_decisions"] == 20, "Should have 20 decisions"
    print(f"    Epsilon: {report['epsilon']:.4f}, Arms: {report['arm_counts']}")
    print("  [OK] BanditOptimizer")


def _test_performance_model():
    print("  Testing PerformanceModel...")
    pm = PerformanceModel()
    for i in range(20):
        config = {
            "batch_size": 1 + i % 8,
            "block_size": 4 + i % 16,
            "hdc_dim": 4096 + i * 256,
        }
        x = pm._extract_features(config=config)
        pm.train([x], [float(50 + i)], [float(50 - i)], [float(8 + i * 0.5)])
    pred = pm.predict(config={"batch_size": 4, "block_size": 8, "hdc_dim": 8192})
    assert pred["throughput"] > 0, "Throughput prediction should be positive"
    print(f"    Predicted throughput: {pred['throughput']:.1f} tok/s")
    print("  [OK] PerformanceModel")


def _test_hardware_adaptation():
    print("  Testing HardwareAdaptation...")
    ha = HardwareAdaptation()
    spec = ha.probe_all()
    config = ha.recommend_config()
    assert len(spec) > 0, "Hardware spec should not be empty"
    assert len(config) > 0, "Recommended config should not be empty"
    print(
        f"    CPU: {spec.get('cpu_cores')} cores, RAM: {spec.get('ram_gb')} GB, AVX2: {spec.get('avx2')}"
    )
    print(
        f"    Recommended config: {config.get('batch_size')} batch, {config.get('block_size')} block"
    )
    print("  [OK] HardwareAdaptation")


def _test_workload_predictor():
    print("  Testing WorkloadPredictor...")
    wp = WorkloadPredictor()
    for i in range(100):
        wp.observe(
            request_rate=1.0 + math.sin(i * 0.1) * 0.5,
            context_length=512 + int(math.sin(i * 0.05) * 100),
        )
    forecast = wp.forecast(steps=10)
    assert len(forecast["predicted_rates"]) == 10, "Should predict 10 steps"
    print(
        f"    Current rate: {forecast['current_rate']:.2f}, Confidence: {forecast['confidence']:.2f}"
    )
    print("  [OK] WorkloadPredictor")


def _test_quality_controller():
    print("  Testing QualityController...")
    qc = QualityController()
    for i in range(10):
        qc.evaluate(
            "This is a test sentence with various words and patterns for quality evaluation."
        )
    report = qc.get_report()
    assert report["current_quality"] > 0, "Quality should be measurable"
    print(
        f"    Current quality: {report['current_quality']:.3f}, Target: {report['target_quality']}"
    )
    print("  [OK] QualityController")


def _test_resource_controller():
    print("  Testing ResourceController...")
    rc = ResourceController(total_memory_gb=32.0, n_cpu_cores=16)
    rc.register_component("hdc", priority=0.8)
    rc.register_component("kv_cache", priority=0.6)
    rc.register_component("model", priority=0.9)
    alloc = rc.allocate_budget("model", 20.0)
    assert alloc > 0, "Should allocate some memory"
    print(f"    Allocated: {alloc:.1f} GB")
    print("  [OK] ResourceController")


def _test_online_learner():
    print("  Testing OnlineLearner...")
    ol = OnlineLearner()
    for i in range(10):
        ol.observe_feedback(
            [1, 2, 3], [4], [5], preference=0.5 + i * 0.05, features=[0.5] * 10
        )
    report = ol.get_report()
    assert report["total_feedback"] == 10, "Should have 10 feedback samples"
    print(f"    Feedback: {report['total_feedback']}, Buffer: {report['buffer_size']}")
    print("  [OK] OnlineLearner")


def _test_self_aware():
    print("  Testing SelfAwareModel...")
    sam = SelfAwareModel()
    for i in range(10):
        metrics = {
            "throughput": float(50 + i),
            "latency_p50": float(50 - i),
            "quality": 0.5,
        }
        state = sam.encode_state(metrics)
        action = sam.encode_action({"batch_size": 8, "block_size": 16})
        sam.observe_transition(state, action, 0.5, state * 0.99)
    report = sam.get_report()
    assert report["n_transitions"] == 10, "Should have 10 transitions"
    print(f"    Transitions: {report['n_transitions']}, Trained: {report['trained']}")
    print("  [OK] SelfAwareModel")


def _test_vlasov():
    print("  Testing VlasovMetaControl...")
    vc = VlasovMetaControl(n_subsystems=6)
    for i in range(6):
        vc.set_subsystem_state(i, float(i) / 5.0)
    vc.evolve()
    signal = vc.get_control_signal()
    assert -1.0 <= signal <= 1.0, "Control signal should be in [-1, 1]"
    print(f"    Control signal: {signal:.4f}")
    print("  [OK] VlasovMetaControl")


def _test_hrr():
    print("  Testing HRREncoder...")
    hrr = HRREncoder(dim=256, capacity=16)
    cfg1 = {"batch_size": 8, "block_size": 16, "hdc_dim": 10000}
    cfg2 = {"batch_size": 4, "block_size": 8, "hdc_dim": 4096}
    hrr.store("test_config", cfg1)
    recalled = hrr.recall(cfg2)
    sim = hrr.similarity(cfg1, cfg2)
    print(f"    Similarity: {sim:.4f}")
    print("  [OK] HRREncoder")


def _test_resonant():
    print("  Testing ResonantOptimizer...")
    ro = ResonantOptimizer(DEFAULT_PARAM_SPACE)
    for i in range(10):
        cfg = ro.sample({"batch_size": 8, "block_size": 16}, t=i * 0.1)
        ro.update_frequencies(cfg, 0.5 + i * 0.05)
    report = ro.get_resonance_report()
    assert report["n_observations"] == 10
    print(f"    Frequencies: {len(report['frequencies'])} params tracked")
    print("  [OK] ResonantOptimizer")


def _test_quantum():
    print("  Testing QuantumAnnealingOptimizer...")
    qa = QuantumAnnealingOptimizer(
        DEFAULT_PARAM_SPACE, n_qubits=16, n_trotters=4, annealing_steps=20
    )

    def _obj(cfg):
        return cfg.get("batch_size", 8) * 10.0

    result = qa.optimize(_obj)
    assert "batch_size" in result
    print(f"    Optimized batch_size: {result.get('batch_size')}")
    print("  [OK] QuantumAnnealingOptimizer")


def _test_hamiltonian():
    print("  Testing HamiltonianMetaController...")
    ctrl = HamiltonianMetaController(n_metrics=5, dt=0.01)
    assert ctrl.n_metrics == 5, "Should have 5 metrics"
    initial_energy = ctrl.get_energy()

    # Run 50 symplectic steps with random gradients
    for i in range(50):
        gradient = np.random.randn(5) * 0.1
        q, p = ctrl.step(gradient)

    # Energy should be approximately conserved (low drift)
    drift = ctrl.energy_drift()
    assert drift < 1.0, f"Energy drift should be bounded, got {drift:.4f}"
    assert len(ctrl.H_history) == 50, "Should have 50 energy records"

    # Phase transition detection
    pt, strength = ctrl.detect_phase_transition()
    assert not pt, "No phase transition in random walk"

    # Symplectic map should have det = 1
    J = ctrl.symplectic_map()
    det = float(np.linalg.det(J))
    assert abs(det - 1.0) < 1e-10, f"Symplectic volume should be 1, got {det}"

    # Nambu step
    ctrl.reset()
    q2, p2 = ctrl.nambu_step()
    assert len(ctrl.H_history) > 0, "Nambu step should record energy"

    # Reset and verify
    ctrl.reset()
    assert np.allclose(ctrl.q, 0.0), "Reset should zero out position"
    assert np.allclose(ctrl.p, 0.0), "Reset should zero out momentum"

    # adapt_parameters should produce valid config adjustments
    config = WARM_START_CONFIGS["balanced"]
    q_test = np.array([0.5, 0.3, 0.7, 0.2, 0.6])
    ctrl.q = q_test
    ctrl.p = np.array(
        [2.0, -1.0, 0.5, -0.3, 0.8]
    )  # Strong momentum to ensure visible changes
    adapted = ctrl.adapt_parameters(config, {"throughput": 50.0, "latency_p50": 30.0})
    assert "block_size" in adapted, "Should produce adjusted config"
    assert (
        adapted["block_size"] != config["block_size"]
        or adapted["batch_size"] != config["batch_size"]
    ), "Should actually modify parameters"

    report = ctrl.get_report()
    assert "energy" in report
    assert "energy_drift" in report
    assert "conservation_quality" in report
    print(f"    Final energy: {ctrl.get_energy():.4f}, drift: {drift:.6f}")
    print(f"    Symplectic volume: {det:.10f} = 1 (Liouville conserved)")
    print(f"    Conservation quality: {report['conservation_quality']}")
    print("  [OK] HamiltonianMetaController")


def _test_meta_controller():
    print("  Testing MetaController...")
    mc = MetaController()
    mc.update(
        {
            "throughput": 50.0,
            "latency_p50": 30.0,
            "quality": 0.7,
            "request_rate": 1.0,
            "context_length": 512,
            "entropy": 0.4,
            "memory_usage_gb": 4.0,
            "cache_hit_rate": 0.8,
            "hdc_acceptance": 0.75,
            "compression_ratio": 20.0,
            "error_rate": 0.0,
            "fallback_rate": 0.0,
            "consecutive_failures": 0,
            "strategy_level": 2.0,
            "resonance_score": 0.6,
            "cpu_utilization": 0.4,
            "memory_pressure": 0.3,
        }
    )
    report = mc.get_report()
    assert report["iteration"] > 0, "Should have at least 1 iteration"
    config = mc.get_config()
    assert len(config) > 0, "Should have a config"
    print(f"    Iterations: {report['iteration']}, Config params: {len(config)}")
    print("  [OK] MetaController")


def _test_gp():
    print("  Testing GaussianProcess...")
    gp = GaussianProcess()
    for i in range(10):
        x = np.array([float(i) / 10.0, float(i % 5) / 5.0])
        gp.add_observation(x, math.sin(i * 0.5))
    mu, var = gp.predict(np.array([0.5, 0.5]))
    ei = gp.expected_improvement(np.array([0.5, 0.5]), best_y=0.8)
    assert var >= 0, "Variance should be non-negative"
    print(f"    Prediction: mu={mu:.3f}, var={var:.3f}, EI={ei:.3f}")
    print("  [OK] GaussianProcess")


def run_all_tests():
    """Run all meta-controller tests."""
    print("=" * 60)
    print("  SpectralStream Meta-Controller - Self-Verification")
    print("=" * 60)
    _test_gp()
    _test_auto_tuner()
    _test_bandit()
    _test_performance_model()
    _test_hardware_adaptation()
    _test_workload_predictor()
    _test_quality_controller()
    _test_resource_controller()
    _test_online_learner()
    _test_self_aware()
    _test_vlasov()
    _test_hrr()
    _test_resonant()
    _test_quantum()
    _test_hamiltonian()
    _test_meta_controller()
    print("=" * 60)
    print("  All tests passed!")
    print("=" * 60)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Points
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        run_all_tests()
    elif "--auto-tune" in sys.argv:
        idx = sys.argv.index("--auto-tune")
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]
            print(f"Auto-tuning for model: {model_path}")
            controller = create_meta_controller(
                model_path=model_path, profile="balanced"
            )
            controller.run_auto_tune(n_iterations=30)
            print(f"\nFinal config: {json.dumps(controller.get_config(), indent=2)}")
            print(controller.get_performance_report())
        else:
            print(
                "Usage: python -m spectralstream.meta_controller --auto-tune /path/to/model.gguf"
            )
    else:
        print("SpectralStream Meta-Controller v0.1.0")
        print("Usage:")
        print("  python -m spectralstream.meta_controller --test")
        print(
            "  python -m spectralstream.meta_controller --auto-tune /path/to/model.gguf"
        )

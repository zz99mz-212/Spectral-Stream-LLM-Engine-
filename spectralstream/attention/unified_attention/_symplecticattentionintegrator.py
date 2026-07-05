from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import (
    BAND_HIGH,
    BAND_LOW,
    BAND_NORMAL,
    DCTRotator,
    HadamardRotator,
    WaveletTransform,
    apply_spectral_kernel,
    band_limit,
    dct,
    fft,
    fftfreq,
    gibbs_softmax,
    idct,
    ifft,
    next_power_of_two,
    softmax,
    spectral_entropy,
    yukawa_kernel_1d,
)


class SymplecticAttentionIntegrator:
    """Symplectic integrator for attention dynamics using leapfrog scheme.

    Treats attention as Hamiltonian dynamics:
        H(q, p) = T(p) + V(q)
        T(p) = 1/2 |p|^2          (kinetic — token velocity)
        V(q) = Phi(q)              (potential — mean-field energy)

    The leapfrog (Störmer-Verlet) step:
        p^{n+1/2} = p^n - (dt/2) . grad_V(q^n)
        q^{n+1}   = q^n + dt . p^{n+1/2}
        p^{n+1}   = p^{n+1/2} - (dt/2) . grad_V(q^{n+1})

    Properties:
        - Symplectic: preserves phase-space volume dq ^ dp
        - Time-reversible
        - O(dt^3) local error, O(dt^2) global error
        - Conserves energy exponentially well for long times
    """

    def __init__(
        self,
        dt: float = 0.1,
        n_substeps: int = 1,
        hamiltonian_monitor: bool = False,
    ):
        self.dt = dt
        self.n_substeps = n_substeps
        self.hamiltonian_monitor = hamiltonian_monitor
        self._energy_history: List[float] = []

    def _potential_energy(self, x: np.ndarray, attn_output: np.ndarray) -> float:
        return -float(np.sum(x * attn_output)) / max(x.shape[0], 1)

    def _kinetic_energy(self, momentum: np.ndarray) -> float:
        return 0.5 * float(np.mean(np.sum(momentum ** 2, axis=-1)))

    def total_energy(
        self,
        x: np.ndarray,
        momentum: np.ndarray,
        attn_output: np.ndarray,
    ) -> float:
        return self._kinetic_energy(momentum) + self._potential_energy(x, attn_output)

    def leapfrog_step(
        self,
        x: np.ndarray,
        momentum: np.ndarray,
        force_fn: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        eps = self.dt / max(self.n_substeps, 1)

        for _ in range(max(self.n_substeps, 1)):
            grad_v = force_fn(x)
            momentum_half = momentum - 0.5 * eps * grad_v
            x = x + eps * momentum_half
            grad_v_new = force_fn(x)
            momentum = momentum_half - 0.5 * eps * grad_v_new

        return x, momentum

    def integrate_layer(
        self,
        x: np.ndarray,
        momentum: Optional[np.ndarray] = None,
        attn_layer: Optional[Callable] = None,
        force_fn: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, d = x.shape

        if momentum is None:
            momentum = np.zeros_like(x)

        if force_fn is not None:
            force_func = force_fn
        elif attn_layer is not None:
            def force_func(q):
                return attn_layer(q)
        else:
            raise ValueError("Must provide either attn_layer or force_fn")

        x_new, momentum_new = self.leapfrog_step(x, momentum, force_func)

        if self.hamiltonian_monitor:
            with np.errstate(all='ignore'):
                attn_out = force_func(x)
                e = self.total_energy(x_new, momentum_new, attn_out)
                self._energy_history.append(e)

        return x_new, momentum_new

    def reset(self):
        self._energy_history.clear()

    def energy_conservation_error(self) -> float:
        if len(self._energy_history) < 2:
            return 0.0
        energies = np.array(self._energy_history)
        mean_e = np.mean(energies)
        if abs(mean_e) < 1e-30:
            return 0.0
        return float(np.std(energies) / abs(mean_e))

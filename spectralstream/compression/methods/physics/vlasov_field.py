from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "vlasov_field"

__all__ = ["VlasovConfig", "VlasovField", "METHOD_NAME"]


@dataclass
class VlasovConfig:
    n_particles: int = 64
    n_grid: int = 32
    n_steps: int = 50
    dt: float = 0.01
    coupling: float = 0.1


class VlasovField:
    METHOD_NAME = METHOD_NAME

    def __init__(self, config: Optional[VlasovConfig] = None):
        self.config = config or VlasovConfig()

    def _compute_field(
        self, positions: np.ndarray, grid: np.ndarray, charges: np.ndarray
    ) -> np.ndarray:
        n_grid = len(grid)
        field = np.zeros(n_grid, dtype=np.float64)
        for i, (pos, q) in enumerate(zip(positions, charges)):
            field += q * np.exp(-((grid - pos) ** 2) / 2.0)
        return field

    def _particle_step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        field: np.ndarray,
        grid: np.ndarray,
        dt: float,
        coupling: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        forces = np.interp(positions, grid, -np.gradient(field))
        new_velocities = velocities + dt * coupling * forces
        new_positions = positions + dt * new_velocities
        return new_positions, new_velocities

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        n_particles = kwargs.get("n_particles", self.config.n_particles)
        n_grid = kwargs.get("n_grid", self.config.n_grid)
        n_steps = kwargs.get("n_steps", self.config.n_steps)
        orig_shape = tensor.shape

        mat = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
        m, n = mat.shape

        all_positions = []
        all_velocities = []
        all_charges = []

        for i in range(m):
            signal = mat[i]
            rng = np.random.RandomState(42 + i)
            positions = np.sort(rng.uniform(0, n, n_particles))
            velocities = rng.randn(n_particles) * 0.1
            charges = np.interp(positions, np.arange(n), signal)
            charges = charges / (np.max(np.abs(charges)) + 1e-10)

            grid = np.linspace(0, n, n_grid)
            for _ in range(n_steps):
                field = self._compute_field(positions, grid, charges)
                positions, velocities = self._particle_step(
                    positions,
                    velocities,
                    field,
                    grid,
                    self.config.dt,
                    self.config.coupling,
                )
                positions = np.clip(positions, 0, n - 1)

            all_positions.append(positions.astype(np.float32))
            all_velocities.append(velocities.astype(np.float32))
            all_charges.append(charges.astype(np.float32))

        data_out = {
            "positions": all_positions,
            "velocities": all_velocities,
            "charges": all_charges,
            "n_grid": n_grid,
        }
        meta = {"orig_shape": orig_shape, "method": METHOD_NAME}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        n = metadata["orig_shape"][-1]
        n_grid = data["n_grid"]
        grid = np.linspace(0, n, n_grid)
        result = np.zeros((len(data["positions"]), n), dtype=np.float64)

        for i, (pos, charges) in enumerate(zip(data["positions"], data["charges"])):
            result[i] = np.interp(np.arange(n), pos, charges, left=0, right=0)

        return result.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        n_particles = kwargs.get("n_particles", self.config.n_particles)
        orig = tensor.nbytes
        comp = tensor.shape[0] * n_particles * 12
        return comp / max(orig, 1)

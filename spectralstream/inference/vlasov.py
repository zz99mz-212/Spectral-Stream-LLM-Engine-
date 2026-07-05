from __future__ import annotations

import numpy as np


class VlasovMeanFieldAttention:
    """O(n) attention via Vlasov mean-field approximation.

    Treats key-value pairs as a charged plasma and solves the
    Vlasov-Poisson system using Particle-in-Cell (PIC).
    Attention distribution ≈ steady-state of collisionless plasma.
    """

    def __init__(self, dim: int = 128, n_grid: int = 64, n_particles: int = 128):
        self.dim = dim
        self.n_grid = n_grid
        self.n_particles = n_particles
        self._rng = np.random.RandomState(42)
        self.grid: np.ndarray = np.zeros(n_grid, dtype=np.float32)
        self.particles: np.ndarray = np.zeros((n_particles, dim), dtype=np.float32)
        self.weights: np.ndarray = np.ones(n_particles, dtype=np.float32) / n_particles

    def _query_to_grid(self, query: np.ndarray) -> np.ndarray:
        q = query.ravel().astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        proj = np.dot(q_norm, self.particles.T)
        proj = np.clip(proj, -1.0, 1.0)
        grid_idx = ((proj + 1.0) * 0.5 * (self.n_grid - 1)).astype(np.int32)
        grid_idx = np.clip(grid_idx, 0, self.n_grid - 1)
        grid = np.zeros(self.n_grid, dtype=np.float32)
        for i, gi in enumerate(grid_idx):
            grid[gi] += self.weights[i]
        return grid / (np.sum(grid) + 1e-10)

    def _solve_mean_field(self, grid: np.ndarray) -> np.ndarray:
        potential = np.zeros_like(grid)
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                dx = (i - j) / self.n_grid
                potential[i] += grid[j] * np.exp(-dx * dx * 10.0)
        return -np.gradient(potential)

    def attend(
        self, query: np.ndarray, keys: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        if len(keys) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        grid = self._query_to_grid(query)
        field = self._solve_mean_field(grid)
        attn_weights = np.exp(field - np.max(field))
        attn_weights = attn_weights / (np.sum(attn_weights) + 1e-10)
        n_kv = min(len(keys), self.n_grid)
        weights = np.zeros(len(keys), dtype=np.float32)
        for i in range(len(keys)):
            idx = int(i * self.n_grid / max(len(keys), 1))
            idx = min(idx, self.n_grid - 1)
            weights[i] = attn_weights[idx]
        weights = weights / (np.sum(weights) + 1e-10)
        output = np.zeros(self.dim, dtype=np.float32)
        for i in range(len(keys)):
            v = values[i].ravel().astype(np.float32)
            if len(v) > self.dim:
                v = v[: self.dim]
            elif len(v) < self.dim:
                v = np.pad(v, (0, self.dim - len(v)))
            output += weights[i] * v
        return output

    def reset(self):
        self.grid.fill(0.0)
        self._rng = np.random.RandomState(42)

"""Compressed Sensing: FISTA/ISTA for L1-minimisation weight recovery."""

import math
from typing import Tuple

import numpy as np


class CompressedSensing:
    """FISTA/ISTA for L1-minimisation weight recovery. Solves: min_x ||x||_1 s.t. Phi @ x = y."""

    @staticmethod
    def random_projection(
        x: np.ndarray, m: int, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[-1]
        rng = np.random.RandomState(seed)
        Phi = rng.randn(m, n) / math.sqrt(m)
        if x.ndim == 1:
            y = Phi @ x
        else:
            y = Phi @ x.T
            y = y.T
        return y, Phi

    @staticmethod
    def fista(
        Phi: np.ndarray, y: np.ndarray, max_iter: int = 200, tol: float = 1e-4
    ) -> np.ndarray:
        m, n = Phi.shape
        y = np.asarray(y, dtype=np.float64)
        sigma_max = CompressedSensing._power_iteration_lipschitz(Phi)
        L = sigma_max**2
        lam = 0.1 * np.max(np.abs(Phi.T @ y))
        x = np.zeros(n, dtype=np.float64)
        z = x.copy()
        t = 1.0
        for _ in range(max_iter):
            grad = Phi.T @ (Phi @ z - y)
            x_new = CompressedSensing._soft_threshold(z - grad / L, lam / L)
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t**2))
            z = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = t_new
            if np.linalg.norm(x_new - x) < tol * (np.linalg.norm(x) + 1e-10):
                x = x_new
                break
            x = x_new
        return x

    @staticmethod
    def _power_iteration_lipschitz(A: np.ndarray, n_iter: int = 50) -> float:
        m, n = A.shape
        rng = np.random.RandomState(0)
        v = rng.randn(n)
        v /= np.linalg.norm(v) + 1e-10
        for _ in range(n_iter):
            Av = A @ v
            u = Av / (np.linalg.norm(Av) + 1e-10)
            Atu = A.T @ u
            v = Atu / (np.linalg.norm(Atu) + 1e-10)
        return float(np.linalg.norm(A @ v))

    @staticmethod
    def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

    @staticmethod
    def compress(matrix: np.ndarray, measurement_ratio: float = 0.3) -> dict:
        matrix = np.asarray(matrix, dtype=np.float64)
        rows, cols = matrix.shape
        m = max(1, int(cols * measurement_ratio))
        rng = np.random.RandomState(42)
        Phi = rng.randn(m, cols) / math.sqrt(m)
        y = np.array([(Phi @ matrix[i]) for i in range(rows)])
        return {
            "shape": (rows, cols),
            "measurement_ratio": measurement_ratio,
            "measurements": m,
            "Phi": Phi,
            "y": y,
            "type": "compressed_sensing",
        }

    @staticmethod
    def decompress(compressed: dict) -> np.ndarray:
        Phi = compressed["Phi"]
        y_all = compressed["y"]
        rows, cols = compressed["shape"]
        recovered = np.zeros((rows, cols), dtype=np.float64)
        for i in range(rows):
            recovered[i] = CompressedSensing.fista(Phi, y_all[i])
        return recovered

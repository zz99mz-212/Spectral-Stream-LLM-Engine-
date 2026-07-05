from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


class COCONUTEngine:
    """Continuous Chain of Thought — latent reasoning before token emission.

    Explores in latent space via a learned MLP transition before projecting
    to vocabulary.  Gives the model "thinking time" before committing to tokens.
    """

    def __init__(
        self,
        d_model: int = 2048,
        max_steps: int = 16,
        entropy_threshold: float = 0.5,
        exploration_noise: float = 0.01,
    ):
        self.d_model = d_model
        self.max_steps = max_steps
        self.entropy_threshold = entropy_threshold
        self.exploration_noise = exploration_noise

        self.W1 = np.random.randn(d_model, d_model * 4).astype(np.float32) * 0.01
        self.b1 = np.zeros(d_model * 4, dtype=np.float32)
        self.W2 = np.random.randn(d_model * 4, d_model).astype(np.float32) * 0.01
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def _transition(self, h: np.ndarray) -> np.ndarray:
        hidden = h @ self.W1 + self.b1
        hidden = np.maximum(hidden, 0)
        delta = hidden @ self.W2 + self.b2
        return (
            h
            + delta
            + np.random.randn(*h.shape).astype(np.float32) * self.exploration_noise
        )

    def _latent_entropy(self, h: np.ndarray) -> float:
        p = np.abs(h) / (np.sum(np.abs(h)) + 1e-30)
        entropy = -np.sum(p * np.log(p + 1e-30))
        return float(entropy / np.log(self.d_model))

    def explore(
        self,
        h_0: np.ndarray,
    ) -> Tuple[np.ndarray, int, List[np.ndarray]]:
        h = h_0.copy()
        trajectory = [h.copy()]
        n_steps = 0
        for step in range(self.max_steps):
            h = self._transition(h)
            trajectory.append(h.copy())
            n_steps = step + 1
            if self._latent_entropy(h) < self.entropy_threshold:
                break
        return h, n_steps, trajectory

    def fuse_multiple_paths(
        self,
        h_0: np.ndarray,
        n_paths: int = 4,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float]]]:
        paths: List[Tuple[np.ndarray, float]] = []
        for _ in range(n_paths):
            h_final, _n_steps, _traj = self.explore(h_0)
            confidence = 1.0 - self._latent_entropy(h_final)
            paths.append((h_final, confidence))
        total_conf = sum(c for _, c in paths) + 1e-30
        fused = sum(h * c / total_conf for h, c in paths)
        return fused, paths


def coconut_action(
    confidence: float,
    high_thresh: float = 0.75,
    low_thresh: float = 0.35,
) -> str:
    if confidence >= high_thresh:
        return "skip"
    elif confidence >= low_thresh:
        return "single"
    return "multi"


def integrate_coconut(
    engine: COCONUTEngine,
    h: np.ndarray,
    confidence: float,
    high_thresh: float = 0.75,
    low_thresh: float = 0.35,
    n_paths: int = 4,
) -> Tuple[np.ndarray, int, str]:
    action = coconut_action(confidence, high_thresh, low_thresh)
    if action == "skip":
        return h, 0, action
    if action == "single":
        h_out, n_steps, _traj = engine.explore(h)
        return h_out, n_steps, action
    h_out, _paths = engine.fuse_multiple_paths(h, n_paths=n_paths)
    return h_out, engine.max_steps, action

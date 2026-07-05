from __future__ import annotations

import numpy as np
from typing import List


class SpectralEntropyScorer:
    """Scores token sequences by spectral entropy of their trajectory."""

    def __init__(self, hidden_dim: int = 512):
        self.hidden_dim = hidden_dim
        self.entropy_history: list = []

    def score_sequence(
        self,
        candidate_tokens: list,
        context_embeddings: np.ndarray,
        candidate_embeddings: List[np.ndarray],
    ) -> float:
        n = len(candidate_tokens)
        if n < 2:
            return 0.5
        sim_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                ei = candidate_embeddings[i].ravel()
                ej = candidate_embeddings[j].ravel()
                sim = float(np.dot(ei, ej))
                norm = float(np.linalg.norm(ei) * np.linalg.norm(ej) + 1e-10)
                sim_matrix[i, j] = sim / norm
                sim_matrix[j, i] = sim / norm
        eigenvalues = np.abs(np.linalg.eigvalsh(sim_matrix))
        power = eigenvalues / (np.sum(eigenvalues) + 1e-10)
        spectral_entropy = -np.sum(power * np.log2(power + 1e-10))
        normalized_entropy = spectral_entropy / np.log2(n)
        coherence = 1.0 - normalized_entropy
        length_bonus = np.log1p(n) / np.log1p(32)
        self.entropy_history.append(normalized_entropy)
        return float(coherence * (0.7 + 0.3 * length_bonus))


class HopfieldEnergyScorer:
    """Scores candidates using modern Hopfield network energy."""

    def __init__(self, beta: float = 8.0):
        self.beta = beta
        self.stored_patterns: list = []

    def store(self, pattern: np.ndarray):
        self.stored_patterns.append(pattern.copy())

    def energy(self, state: np.ndarray) -> float:
        if not self.stored_patterns:
            return 0.0
        patterns = np.stack(self.stored_patterns, axis=0)
        similarities = patterns @ state.ravel()
        logsumexp = float(np.log(np.sum(np.exp(self.beta * similarities)) + 1e-10))
        return -1.0 / self.beta * logsumexp

    def score(self, hidden_states: List[np.ndarray]) -> float:
        if len(hidden_states) < 2 or not self.stored_patterns:
            return 0.5
        energies = [self.energy(h) for h in hidden_states]
        return float(1.0 / (1.0 + np.exp(energies[-1] - energies[0])))


class AttractorScoringEnsemble:
    """Ensemble of spectral entropy + Hopfield energy + self-consistency."""

    def __init__(self, hidden_dim: int = 512):
        self.spectral = SpectralEntropyScorer(hidden_dim)
        self.hopfield = HopfieldEnergyScorer(beta=8.0)
        self.hidden_dim = hidden_dim

    def score_candidates(
        self,
        candidates: List[list],
        context_embeddings: np.ndarray,
        hidden_states_per_candidate: List[List[np.ndarray]],
    ) -> list:
        if not candidates:
            return []
        scores = []
        for i, (tokens, hidden_states) in enumerate(
            zip(candidates, hidden_states_per_candidate)
        ):
            if len(hidden_states) < 2:
                scores.append((i, 0.3))
                continue
            spect_score = self.spectral.score_sequence(
                tokens, context_embeddings, [h.ravel() for h in hidden_states]
            )
            hopf_score = self.hopfield.score(hidden_states)
            consistency = 1.0
            if len(hidden_states) > 1:
                last_hidden = hidden_states[-1].ravel()
                similarities = []
                for j, other_hidden in enumerate(
                    [hs[-1] for hs in hidden_states_per_candidate]
                ):
                    if i != j:
                        s = float(np.dot(last_hidden, other_hidden.ravel()))
                        n = float(
                            np.linalg.norm(last_hidden)
                            * np.linalg.norm(other_hidden.ravel())
                            + 1e-10
                        )
                        similarities.append(s / n)
                if similarities:
                    consistency = float(np.mean(similarities))
            combined = 0.4 * spect_score + 0.3 * hopf_score + 0.3 * consistency
            scores.append((i, combined))
        scores.sort(key=lambda x: -x[1])
        return scores

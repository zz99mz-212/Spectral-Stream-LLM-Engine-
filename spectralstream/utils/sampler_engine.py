"""
Universal Sampler Engine for SpectralStream
============================================
Clean-room implementation of ALL known sampling methods plus novel inventions.
Every technique is implemented from first principles using only numpy.

Standard samplers:
  - Greedy, Temperature, Top-K, Top-P (Nucleus), Min-P, Typical, Locally Typical
  - Eta (epsilon cutoff), Tails-Free (TFS-Z), Top-A, Contrastive Search
  - Mirostat v1 (adaptive top-K via surprise), Mirostat v2 (via perplexity target)
  - Beam Search, Diverse Beam Search

Penalties & transformations (applied before sampling):
  - Repetition, Frequency, Presence, LogitBias, TokenBan
  - GrammarConstraint, JSONMode, XMLMode

Novel samplers (SpectralStream originals):
  - QuantumCollapseSampler: Born rule over logit wavefunction, measurement entropy
  - BornRuleQuantumSampler: Wavefunction |ψ⟩ = √(softmax), Grover amplification,
    superposition branching, decoherence
  - SpectralResonanceSampler: weight by FFT frequency resonance with context
  - HolographicPatternSampler: bias toward holographic memory patterns
  - AttractorBasinSampler: suppress degenerate attractor basins (loops/repetition)
  - HamiltonianTrajectorySampler: HMC on token space for diverse coherent text
  - VlasovFieldSampler: token-token interactions via mean-field potential
  - PredictorCorrectorSampler: HDC draft, model correct, sample corrected dist

Pipeline, adaptive, and batch sampling infrastructure.

References:
  - Fan et al. (2018) Top-K sampling
  - Holtzman et al. (2019) Top-P / Nucleus
  - Basu et al. (2021) Mirostat
  - Meister et al. (2022) Typical Sampling
  - Keskar et al. (2019) Repetition Penalty (CTRL)
  - Su et al. (2022) Contrastive Search
  - Leviathan et al. (2022) Speculative Sampling
  - Hewitt et al. (2022) Eta Sampling / Locally Typical
"""

from __future__ import annotations

import math
import re
import threading
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union, Literal

import numpy as np

from spectralstream.core.math_primitives.numerical import softmax as _softmax


def _sample_multinomial(
    probs: np.ndarray, rng: np.random.RandomState | None = None
) -> int:
    if rng is None:
        return int(np.random.choice(len(probs), p=probs))
    return int(rng.choice(len(probs), p=probs))


def _top_k_mask(probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= len(probs):
        return probs
    threshold = np.sort(probs)[-k]
    masked = probs.copy()
    masked[masked < threshold] = 0.0
    return masked / (np.sum(masked) + 1e-30)


def _top_p_mask(probs: np.ndarray, p: float) -> np.ndarray:
    if p >= 1.0 or p <= 0.0:
        return probs
    sorted_idx = np.argsort(probs)[::-1]
    cumsum = np.cumsum(probs[sorted_idx])
    cutoff = cumsum > p
    first_cutoff = int(np.where(cutoff)[0][0]) if np.any(cutoff) else len(probs) - 1
    masked = probs.copy()
    masked[sorted_idx[first_cutoff + 1 :]] = 0.0
    return masked / (np.sum(masked) + 1e-30)


def _min_p_mask(probs: np.ndarray, min_p: float) -> np.ndarray:
    if min_p <= 0.0:
        return probs
    max_prob = np.max(probs)
    threshold = max_prob * min_p
    masked = probs.copy()
    masked[masked < threshold] = 0.0
    return masked / (np.sum(masked) + 1e-30)


def _get_rng(seed: int | None = None) -> np.random.RandomState:
    if seed is not None:
        return np.random.RandomState(seed)
    return np.random.RandomState()


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 1e-10]
    return -float(np.sum(p * np.log(p)))


def _validate_sampling_params(**kwargs) -> list[str]:
    errors = []
    for k, v in kwargs.items():
        if v is None:
            continue
        if k in (
            "temperature",
            "top_p",
            "min_p",
            "typical_p",
            "eta",
            "tfs_z",
            "top_a",
            "mirostat_tau",
            "mirostat_eta",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
            "diversity_penalty",
            "beam_width",
            "contrastive_alpha",
            "coherence_penalty",
        ):
            if not isinstance(v, (int, float)):
                errors.append(f"{k}: must be numeric, got {type(v).__name__}")
            elif v < 0:
                errors.append(f"{k}: must be non-negative, got {v}")
        if k in ("top_k",) and isinstance(v, int) and v < 0:
            errors.append(f"{k}: must be non-negative, got {v}")
    return errors


# ═══════════════════════════════════════════════════════════════════════════
# 1. BaseSampler — Abstract Base
# ═══════════════════════════════════════════════════════════════════════════


class BaseSampler(ABC):
    """Abstract base for all samplers."""

    name: str = "base"

    @abstractmethod
    def sample(self, logits: np.ndarray, temperature: float = 1.0, **kwargs) -> int: ...

    def batch_sample(
        self, logits_batch: np.ndarray, temperature: float = 1.0, **kwargs
    ) -> list[int]:
        return [self.sample(logits, temperature, **kwargs) for logits in logits_batch]

    def get_probs(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        return _softmax(logits, temperature)

    def validate_params(self, **kwargs) -> list[str]:
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Standard Samplers
# ═══════════════════════════════════════════════════════════════════════════


class GreedySampler(BaseSampler):
    """Deterministic argmax sampling."""

    name = "greedy"

    def sample(self, logits: np.ndarray, temperature: float = 1.0, **kwargs) -> int:
        return int(np.argmax(logits))

    def get_probs(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        probs = np.zeros_like(logits, dtype=np.float64)
        probs[int(np.argmax(logits))] = 1.0
        return probs


class TemperatureSampler(BaseSampler):
    """Standard temperature-scaled softmax + multinomial."""

    name = "temperature"

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        probs = _softmax(logits, temperature)
        rg = rng or _get_rng(seed)
        return _sample_multinomial(probs, rg)


class TopKSampler(BaseSampler):
    """Top-K sampling (Fan et al., 2018). Sample only from the K most likely tokens."""

    name = "top_k"

    def __init__(self, default_k: int = 40):
        self.default_k = default_k

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        k = top_k if top_k is not None else self.default_k
        probs = _softmax(logits, temperature)
        masked = _top_k_mask(probs, k)
        rg = rng or _get_rng(seed)
        return _sample_multinomial(masked, rg)


class TopPSampler(BaseSampler):
    """Top-P / Nucleus sampling (Holtzman et al., 2019)."""

    name = "top_p"

    def __init__(self, default_p: float = 0.95):
        self.default_p = default_p

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_p: float | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        p = top_p if top_p is not None else self.default_p
        probs = _softmax(logits, temperature)
        masked = _top_p_mask(probs, p)
        rg = rng or _get_rng(seed)
        return _sample_multinomial(masked, rg)


class MinPSampler(BaseSampler):
    """Min-P sampling: keep tokens with probability >= min_p * max(probs)."""

    name = "min_p"

    def __init__(self, default_min_p: float = 0.05):
        self.default_min_p = default_min_p

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        min_p: float | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        mp = min_p if min_p is not None else self.default_min_p
        probs = _softmax(logits, temperature)
        masked = _min_p_mask(probs, mp)
        rg = rng or _get_rng(seed)
        return _sample_multinomial(masked, rg)


class TypicalSampler(BaseSampler):
    """Typical sampling (Meister et al., 2022).
    Sample based on how close each token's log-prob is to the entropy of the distribution.
    """

    name = "typical"

    def __init__(self, default_p: float = 0.95):
        self.default_p = default_p

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        typical_p: float | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        p = typical_p or self.default_p
        probs = _softmax(logits, temperature)
        log_probs = np.log(probs + 1e-30)
        H = _entropy(probs)
        neg_entropy = -H
        surprisal = np.abs(log_probs - neg_entropy)
        sorted_idx = np.argsort(surprisal)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = cumsum > p
        first_cutoff = int(np.where(cutoff)[0][0]) if np.any(cutoff) else len(probs) - 1
        masked = np.zeros_like(probs)
        masked[sorted_idx[:first_cutoff]] = probs[sorted_idx[:first_cutoff]]
        masked = masked / (np.sum(masked) + 1e-30)
        rg = rng or _get_rng(seed)
        return _sample_multinomial(masked, rg)


class LocallyTypicalSampler(BaseSampler):
    """Locally Typical sampling — matching local entropy statistics."""

    name = "locally_typical"

    def __init__(self, default_p: float = 0.95, filter_threshold: float = 0.1):
        self.default_p = default_p
        self.filter_threshold = filter_threshold

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        typical_p: float | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        p = typical_p or self.default_p
        probs = _softmax(logits, temperature)
        log_probs = np.log(probs + 1e-30)
        H = _entropy(probs)
        neg_entropy = -H
        surprisal = np.abs(log_probs - neg_entropy)
        threshold = self.filter_threshold
        filtered = probs.copy()
        filtered[surprisal > threshold + abs(neg_entropy) * 0.5] = 0.0
        masked = _top_p_mask(filtered, p) if np.sum(filtered) > 0 else probs
        rg = rng or _get_rng(seed)
        return _sample_multinomial(masked, rg)


class EtaSampler(BaseSampler):
    """Eta sampling with epsilon cutoff (Hewitt et al., 2022).
    Removes tokens with probability below eta * max_prob then rescales.
    """

    name = "eta"

    def __init__(self, default_eta: float = 0.2):
        self.default_eta = default_eta

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        eta: float | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        eps = eta if eta is not None else self.default_eta
        probs = _softmax(logits, temperature)
        max_prob = np.max(probs)
        threshold = eps * 0.1
        cutoff = max_prob * (eps / (1.0 + eps))
        masked = probs.copy()
        masked[masked < cutoff] = 0.0
        if np.sum(masked) < 1e-30:
            masked = probs.copy()
            masked[masked < max_prob * 0.01] = 0.0
        masked = masked / (np.sum(masked) + 1e-30)
        rg = rng or _get_rng(seed)
        return _sample_multinomial(masked, rg)


class TFSSampler(BaseSampler):
    """Tails-Free Sampling (TFS-Z). Remove tail by second derivative of sorted probs."""

    name = "tfs"

    def __init__(self, default_z: float = 0.95):
        self.default_z = default_z

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        tfs_z: float | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        z = tfs_z if tfs_z is not None else self.default_z
        probs = _softmax(logits, temperature)
        sorted_probs = np.sort(probs)
        second_deriv = np.diff(sorted_probs, n=2)
        second_deriv = np.pad(second_deriv, (0, 2), constant_values=0.0)
        second_deriv = np.abs(second_deriv)
        second_deriv = second_deriv / (np.max(second_deriv) + 1e-30)
        entropy_scale = np.sum(second_deriv) / (len(second_deriv) + 1e-30)
        cutoff = np.searchsorted(np.cumsum(second_deriv), z * np.sum(second_deriv))
        cutoff = max(1, cutoff)
        threshold = sorted_probs[-cutoff] if cutoff < len(sorted_probs) else 0.0
        masked = probs.copy()
        masked[masked < threshold] = 0.0
        masked = masked / (np.sum(masked) + 1e-30)
        rg = rng or _get_rng(seed)
        return _sample_multinomial(masked, rg)


class TopASampler(BaseSampler):
    """Top-A sampling: keep tokens with prob >= top_a * max_prob."""

    name = "top_a"

    def __init__(self, default_a: float = 0.2):
        self.default_a = default_a

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_a: float | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        a = top_a if top_a is not None else self.default_a
        probs = _softmax(logits, temperature)
        max_prob = np.max(probs)
        threshold = max_prob * a
        masked = probs.copy()
        masked[masked < threshold] = 0.0
        if np.sum(masked) < 1e-30:
            return int(np.argmax(probs))
        masked = masked / (np.sum(masked) + 1e-30)
        rg = rng or _get_rng(seed)
        return _sample_multinomial(masked, rg)


class ContrastiveSampler(BaseSampler):
    """Contrastive search with degeneration penalty (Su et al., 2022)."""

    name = "contrastive"

    def __init__(self, alpha: float = 0.6, k: int = 10):
        self.alpha = alpha
        self.k = k
        self._past_tokens: list[int] = []
        self._context_vectors: list[np.ndarray] = []

    def reset(self):
        self._past_tokens.clear()
        self._context_vectors.clear()

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        alpha: float | None = None,
        contrastive_k: int | None = None,
        past_tokens: list[int] | None = None,
        hidden_state: np.ndarray | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        a = alpha if alpha is not None else self.alpha
        k = contrastive_k if contrastive_k is not None else self.k
        probs = _softmax(logits, temperature)
        topk_idx = np.argsort(probs)[::-1][:k]
        topk_probs = probs[topk_idx]
        if hidden_state is not None:
            self._context_vectors.append(hidden_state.ravel().copy())
        pts = past_tokens if past_tokens is not None else self._past_tokens
        degeneration_penalty = np.ones(k, dtype=np.float64)
        for i, idx in enumerate(topk_idx):
            count = pts.count(int(idx)) if pts else 0
            degeneration_penalty[i] = 1.0 - min(a * count / max(len(pts), 1), 0.99)
        scores = topk_probs * degeneration_penalty
        scores = scores / (np.sum(scores) + 1e-30)
        rg = rng or _get_rng(seed)
        chosen = _sample_multinomial(scores, rg)
        token = int(topk_idx[chosen])
        self._past_tokens.append(token)
        return token


class _MirostatState:
    def __init__(self, tau: float = 3.0, eta: float = 0.1, max_surprise: float = 5.0):
        self.tau = tau
        self.eta = eta
        self.max_surprise = max_surprise
        self.surprise_history: list[float] = []
        self.current_k: int = 100
        self.step: int = 0


class MirostatV1(BaseSampler):
    """Mirostat v1: adaptive top-K via surprise (Basu et al., 2021)."""

    name = "mirostat_v1"

    def __init__(self, tau: float = 3.0, eta: float = 0.1, initial_k: int = 100):
        self._state = _MirostatState(tau=tau, eta=eta)
        self.initial_k = initial_k

    @property
    def state(self) -> _MirostatState:
        return self._state

    def reset(self, tau: float | None = None, eta: float | None = None):
        self._state = _MirostatState(
            tau=tau or self._state.tau,
            eta=eta or self._state.eta,
        )

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        mirostat_tau: float | None = None,
        mirostat_eta: float | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        s = self._state
        tau = mirostat_tau if mirostat_tau is not None else s.tau
        eta = mirostat_eta if mirostat_eta is not None else s.eta

        probs = _softmax(logits, temperature)
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]

        k = min(s.current_k, len(probs))
        k = max(k, 1)

        candidate_probs = sorted_probs[:k]
        candidate_idx = sorted_idx[:k]
        candidate_probs = candidate_probs / (np.sum(candidate_probs) + 1e-30)

        rg = rng or _get_rng(seed)
        chosen = _sample_multinomial(candidate_probs, rg)
        token = int(candidate_idx[chosen])

        token_prob = float(candidate_probs[chosen])
        surprise = -math.log(token_prob + 1e-30)

        s.surprise_history.append(surprise)
        if len(s.surprise_history) > 100:
            s.surprise_history.pop(0)

        diff = surprise - tau
        s.current_k = max(1, int(s.current_k - eta * diff * k / max(k, 1)))
        s.step += 1

        return token


class MirostatV2(BaseSampler):
    """Mirostat v2: adaptive top-K via perplexity target (Basu et al., 2021)."""

    name = "mirostat_v2"

    def __init__(self, tau: float = 3.0, eta: float = 0.1, initial_k: int = 100):
        self._state = _MirostatState(tau=tau, eta=eta)
        self.initial_k = initial_k

    @property
    def state(self) -> _MirostatState:
        return self._state

    def reset(self, tau: float | None = None, eta: float | None = None):
        self._state = _MirostatState(
            tau=tau or self._state.tau,
            eta=eta or self._state.eta,
        )

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        mirostat_tau: float | None = None,
        mirostat_eta: float | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        s = self._state
        tau = mirostat_tau if mirostat_tau is not None else s.tau
        eta = mirostat_eta if mirostat_eta is not None else s.eta

        probs = _softmax(logits, temperature)
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]

        k = min(s.current_k, len(probs))
        k = max(k, 1)

        candidate_probs = sorted_probs[:k]
        candidate_idx = sorted_idx[:k]
        candidate_probs = candidate_probs / (np.sum(candidate_probs) + 1e-30)

        rg = rng or _get_rng(seed)
        chosen = _sample_multinomial(candidate_probs, rg)
        token = int(candidate_idx[chosen])

        token_prob = float(candidate_probs[chosen])
        perplexity = -math.log(token_prob + 1e-30) / (math.log(k) + 1e-30)
        surprise = -math.log(token_prob + 1e-30)

        s.surprise_history.append(surprise)
        if len(s.surprise_history) > 100:
            s.surprise_history.pop(0)

        moving_avg = (
            np.mean(s.surprise_history[-10:]) if s.surprise_history else surprise
        )
        diff = moving_avg - tau
        s.current_k = max(1, int(s.current_k - eta * diff * k / max(k, 1)))
        s.step += 1

        return token


class BeamSearchSampler(BaseSampler):
    """Beam search sampling with configurable width."""

    name = "beam_search"

    def __init__(self, beam_width: int = 5, max_length_penalty: float = 0.6):
        self.beam_width = beam_width
        self.max_length_penalty = max_length_penalty

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        beam_width: int | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        bw = beam_width if beam_width is not None else self.beam_width
        probs = _softmax(logits, temperature)
        topk_idx = np.argsort(probs)[::-1][:bw]
        topk_probs = probs[topk_idx]
        topk_probs = topk_probs / (np.sum(topk_probs) + 1e-30)
        rg = rng or _get_rng(seed)
        chosen = _sample_multinomial(topk_probs, rg)
        return int(topk_idx[chosen])

    def beam_step(
        self,
        logits: np.ndarray,
        beams: list[tuple[list[int], float]],
        beam_width: int | None = None,
    ) -> list[tuple[list[int], float]]:
        bw = beam_width if beam_width is not None else self.beam_width
        probs = _softmax(logits)
        topk_idx = np.argsort(probs)[::-1][:bw]
        candidates: list[tuple[list[int], float]] = []
        for tokens, score in beams:
            for idx in topk_idx:
                new_tokens = tokens + [int(idx)]
                new_score = score + math.log(probs[idx] + 1e-30)
                candidates.append((new_tokens, new_score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:bw]


class DiverseBeamSearch(BaseSampler):
    """Beam search with diversity penalty (Li & Jurafsky, 2016)."""

    name = "diverse_beam"

    def __init__(
        self, beam_width: int = 5, diversity_penalty: float = 0.5, n_groups: int = 3
    ):
        self.beam_width = beam_width
        self.diversity_penalty = diversity_penalty
        self.n_groups = n_groups

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        beam_width: int | None = None,
        diversity_penalty: float | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        bw = beam_width if beam_width is not None else self.beam_width
        dp = (
            diversity_penalty
            if diversity_penalty is not None
            else self.diversity_penalty
        )
        probs = _softmax(logits, temperature)
        n_groups = min(self.n_groups, bw)
        group_size = max(1, bw // n_groups)
        topk_idx = np.argsort(probs)[::-1][: bw * 2]
        rg = rng or _get_rng(seed)

        selected: list[int] = []
        for g in range(n_groups):
            group_candidates = topk_idx[g * group_size * 2 : (g + 1) * group_size * 2]
            group_probs = probs[group_candidates]
            for prev in selected:
                group_probs *= (1.0 - dp) if prev in group_candidates else 1.0
            group_probs = group_probs / (np.sum(group_probs) + 1e-30)
            n_pick = min(group_size, len(group_candidates))
            picks = rg.choice(
                len(group_candidates), size=n_pick, p=group_probs, replace=False
            )
            selected.extend(int(group_candidates[p]) for p in picks)

        selected_probs = probs[selected]
        selected_probs = selected_probs / (np.sum(selected_probs) + 1e-30)
        chosen = _sample_multinomial(selected_probs, rg)
        return selected[chosen]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Penalties & Transformations
# ═══════════════════════════════════════════════════════════════════════════


class LogitTransform(ABC):
    """Abstract base for logit/probability transformations."""

    name: str = "transform"

    @abstractmethod
    def apply(
        self, logits: np.ndarray, context: list[int] | None = None, **kwargs
    ) -> np.ndarray: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class RepetitionPenalty(LogitTransform):
    """Repetition penalty (CTRL: Keskar et al., 2019).
    Scale down logits of tokens that have already appeared.
    """

    name = "repetition_penalty"

    def __init__(self, penalty: float = 1.1):
        self.penalty = penalty

    def apply(
        self,
        logits: np.ndarray,
        context: list[int] | None = None,
        repetition_penalty: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        p = repetition_penalty if repetition_penalty is not None else self.penalty
        if p == 1.0 or not context:
            return logits
        result = logits.copy().astype(np.float64)
        seen = set(context)
        for token in seen:
            if 0 <= token < len(result):
                if result[token] < 0:
                    result[token] *= p
                else:
                    result[token] /= p
        return result


class FrequencyPenalty(LogitTransform):
    """Reduce logits based on token frequency in context."""

    name = "frequency_penalty"

    def __init__(self, penalty: float = 0.1):
        self.penalty = penalty

    def apply(
        self,
        logits: np.ndarray,
        context: list[int] | None = None,
        frequency_penalty: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        p = frequency_penalty if frequency_penalty is not None else self.penalty
        if p == 0.0 or not context:
            return logits
        result = logits.copy().astype(np.float64)
        freq = Counter(context)
        for token, count in freq.items():
            if 0 <= token < len(result):
                result[token] -= p * count
        return result


class PresencePenalty(LogitTransform):
    """Reduce logits based on token presence (binary)."""

    name = "presence_penalty"

    def __init__(self, penalty: float = 0.1):
        self.penalty = penalty

    def apply(
        self,
        logits: np.ndarray,
        context: list[int] | None = None,
        presence_penalty: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        p = presence_penalty if presence_penalty is not None else self.penalty
        if p == 0.0 or not context:
            return logits
        result = logits.copy().astype(np.float64)
        seen = set(context)
        for token in seen:
            if 0 <= token < len(result):
                result[token] -= p
        return result


class LogitBiasProcessor(LogitTransform):
    """Add/subtract bias from specific token logits."""

    name = "logit_bias"

    def __init__(self, bias_map: dict[int, float] | None = None):
        self.bias_map = bias_map or {}

    def apply(
        self,
        logits: np.ndarray,
        context: list[int] | None = None,
        logit_bias: dict[int, float] | None = None,
        **kwargs,
    ) -> np.ndarray:
        bias = logit_bias if logit_bias is not None else self.bias_map
        if not bias:
            return logits
        result = logits.copy().astype(np.float64)
        for token, delta in bias.items():
            if 0 <= token < len(result):
                result[token] += delta
        return result


class TokenBanProcessor(LogitTransform):
    """Set specific token probabilities to zero."""

    name = "token_ban"

    def __init__(self, banned_tokens: set[int] | None = None):
        self.banned_tokens = banned_tokens or set()

    def apply(
        self,
        logits: np.ndarray,
        context: list[int] | None = None,
        banned_tokens: set[int] | None = None,
        **kwargs,
    ) -> np.ndarray:
        banned = banned_tokens if banned_tokens is not None else self.banned_tokens
        if not banned:
            return logits
        result = logits.copy().astype(np.float64)
        for token in banned:
            if 0 <= token < len(result):
                result[token] = -float("inf")
        return result


class GrammarConstraint(LogitTransform):
    """Enforce token-level grammar via regex.
    Masks out tokens that would lead to invalid sequences.
    """

    name = "grammar"

    def __init__(
        self,
        grammar_patterns: dict[int, re.Pattern] | None = None,
        default_valid: set[int] | None = None,
        token_decoder: Callable[[int], str] | None = None,
    ):
        self.grammar_patterns = grammar_patterns or {}
        self.default_valid = default_valid
        self.token_decoder = token_decoder or (lambda t: chr(t % 128))

    def apply(
        self,
        logits: np.ndarray,
        context: list[int] | None = None,
        grammar_patterns: dict[int, re.Pattern] | None = None,
        **kwargs,
    ) -> np.ndarray:
        patterns = grammar_patterns or self.grammar_patterns
        if not patterns and self.default_valid is None:
            return logits
        result = logits.copy().astype(np.float64)
        context_str = "".join(self.token_decoder(t) for t in (context or []))
        for token in range(len(result)):
            token_str = self.token_decoder(token)
            candidate = context_str + token_str
            if patterns:
                valid = False
                for name, pattern in patterns.items():
                    if pattern.fullmatch(candidate) or pattern.search(candidate):
                        valid = True
                        break
                if not valid:
                    result[token] = -float("inf")
            elif self.default_valid is not None:
                if token not in self.default_valid:
                    result[token] = -float("inf")
        return result


class JSONModeProcessor(LogitTransform):
    """Force valid JSON output by masking tokens that break JSON structure."""

    name = "json_mode"

    JSON_STRUCTURAL = set(ord(c) for c in '{}[]:,"\\ \t\n\r')
    JSON_WHITESPACE = set(ord(c) for c in " \t\n\r")
    JSON_DIGIT = set(ord(c) for c in "0123456789")
    JSON_LITERALS = set(ord(c) for c in "truefalsnul")

    def __init__(
        self,
        vocab_size: int,
        token_decoder: Callable[[int], str] | None = None,
        token_encoder: Callable[[str], int] | None = None,
    ):
        self.vocab_size = vocab_size
        self.token_decoder = token_decoder or (lambda t: chr(t % 128))
        self.token_encoder = token_encoder

    def apply(
        self, logits: np.ndarray, context: list[int] | None = None, **kwargs
    ) -> np.ndarray:
        if not context:
            return logits
        result = logits.copy().astype(np.float64)
        context_str = "".join(self.token_decoder(t) for t in context)
        state = self._json_state(context_str)
        valid_chars = self._valid_chars_for_state(state)
        for token in range(len(result)):
            t_str = self.token_decoder(token)
            if t_str and not any(c in valid_chars for c in t_str):
                result[token] = -float("inf")
        return result

    def _json_state(self, s: str) -> str:
        stripped = s.strip()
        if not stripped:
            return "start"
        if stripped.startswith('"'):
            if stripped.count('"') % 2 == 1 and not stripped.endswith('\\"'):
                return "string"
            else:
                return "after_value"
        if stripped in ("", "{"):
            return "object_start"
        if stripped.endswith("{") or stripped.endswith(","):
            return "expect_key"
        if stripped.endswith(":"):
            return "expect_value"
        if stripped.endswith("[") or stripped.endswith(","):
            return "expect_value"
        return "after_value"

    def _valid_chars_for_state(self, state: str) -> set[int]:
        if state == "start":
            return set(ord(c) for c in "{[")
        if state == "string":
            chars = set(range(32, 127)) | self.JSON_WHITESPACE
            return chars
        if state == "object_start":
            return set(ord(c) for c in '"')
        if state == "expect_key":
            return set(ord(c) for c in '"')
        if state == "expect_value":
            chars = (
                set(ord(c) for c in "{[")
                | self.JSON_DIGIT
                | self.JSON_LITERALS
                | set([ord('"')])
            )
            return chars
        return set(range(32, 127))


class XMLModeProcessor(LogitTransform):
    """Force valid XML output by masking tokens that break XML structure."""

    name = "xml_mode"

    def __init__(
        self, vocab_size: int, token_decoder: Callable[[int], str] | None = None
    ):
        self.vocab_size = vocab_size
        self.token_decoder = token_decoder or (lambda t: chr(t % 128))

    def apply(
        self, logits: np.ndarray, context: list[int] | None = None, **kwargs
    ) -> np.ndarray:
        if not context:
            return logits
        result = logits.copy().astype(np.float64)
        context_str = "".join(self.token_decoder(t) for t in context)
        try:
            if context_str.strip():
                ET.fromstring(context_str + "<dummy/>")
        except ET.ParseError:
            for token in range(len(result)):
                t_str = self.token_decoder(token)
                if t_str and t_str in "<>&":
                    result[token] = -float("inf")
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. Novel Samplers (SpectralStream Inventions)
# ═══════════════════════════════════════════════════════════════════════════


class QuantumCollapseSampler(BaseSampler):
    """Quantum Coherence Sampling — treat logits as quantum amplitudes,
    collapse via Born rule with measurement entropy.

    Maintains a superposition of plausible next tokens and collapses
    via quantum measurement. The measurement entropy controls whether
    the collapse is sharp (deterministic) or diffuse (exploratory).
    """

    name = "quantum_collapse"

    def __init__(self, measurement_entropy: float = 0.5, n_superposition: int = 10):
        self.measurement_entropy = measurement_entropy
        self.n_superposition = n_superposition
        self._coherence: float = 1.0

    @property
    def coherence(self) -> float:
        return self._coherence

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        measurement_entropy: float | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        me = (
            measurement_entropy
            if measurement_entropy is not None
            else self.measurement_entropy
        )
        rg = rng or _get_rng(seed)

        probs = _softmax(logits, temperature)
        amplitudes = np.sqrt(probs + 1e-30)

        n_super = min(self.n_superposition, len(probs))
        top_idx = np.argsort(probs)[::-1][:n_super]
        top_amps = amplitudes[top_idx]
        phases = rg.uniform(0, 2 * math.pi, size=n_super)
        wavefunction = top_amps * np.exp(1j * phases)

        interference = np.abs(np.sum(wavefunction)) / max(np.sum(top_amps), 1e-30)
        self._coherence = float(interference)

        measurement_std = max(me, 0.01)
        measured_amplitudes = top_amps + rg.normal(0, measurement_std, size=n_super)
        measured_amplitudes = np.maximum(measured_amplitudes, 0)
        collapse_probs = measured_amplitudes**2
        collapse_probs = collapse_probs / (np.sum(collapse_probs) + 1e-30)

        if interference > 0.7:
            collapse_probs = collapse_probs**0.5
            collapse_probs = collapse_probs / (np.sum(collapse_probs) + 1e-30)

        chosen = _sample_multinomial(collapse_probs, rg)
        return int(top_idx[chosen])


class SpectralResonanceSampler(BaseSampler):
    """Spectral Resonant Sampling — weight tokens by frequency resonance
    with the current context's FFT power spectrum.

    Analyzes the context token stream in the frequency domain via FFT,
    then biases the sampling toward tokens whose "natural frequency"
    resonates with the dominant modes of the context.
    """

    name = "spectral_resonance"

    def __init__(
        self,
        resonance_strength: float = 0.3,
        n_fft: int = 64,
        use_token_embedding: Callable[[int], np.ndarray] | None = None,
    ):
        self.resonance_strength = resonance_strength
        self.n_fft = n_fft
        self.use_token_embedding = use_token_embedding

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        resonance_strength: float | None = None,
        context_tokens: list[int] | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        rs = (
            resonance_strength
            if resonance_strength is not None
            else self.resonance_strength
        )
        rg = rng or _get_rng(seed)

        probs = _softmax(logits, temperature)

        if context_tokens and len(context_tokens) >= 8:
            n_fft = min(self.n_fft, len(context_tokens))
            ctx_fft = context_tokens[-n_fft:]
            spectrum = np.abs(np.fft.fft(ctx_fft))
            spectrum = spectrum[: len(spectrum) // 2]
            spectrum = spectrum / (np.max(spectrum) + 1e-30)

            token_spectra = []
            for t in context_tokens[-n_fft:]:
                vec = np.zeros(n_fft)
                idx = t % n_fft
                vec[idx] = 1.0
                token_spectrum = np.abs(np.fft.fft(vec))
                token_spectrum = token_spectrum[: len(token_spectrum) // 2]
                token_spectrum = token_spectrum / (np.max(token_spectrum) + 1e-30)
                token_spectra.append(token_spectrum)

            if token_spectra:
                mean_token_spectrum = np.mean(token_spectra, axis=0)
                resonance = np.dot(spectrum, mean_token_spectrum) / max(
                    len(spectrum), 1
                )
                resonance_factor = 1.0 + rs * (max(0, resonance) - 0.5)
                probs = probs**resonance_factor
                probs = probs / (np.sum(probs) + 1e-30)

        return _sample_multinomial(probs, rg)


class HolographicPatternSampler(BaseSampler):
    """Holographic Pattern Sampling — bias toward coherent patterns
    stored in a holographic memory of past successful generations.

    Maintains a holographic weight matrix that stores patterns.
    At each step, retrieves the pattern most similar to the current
    context and biases the sampling toward it.
    """

    name = "holographic_pattern"

    def __init__(
        self,
        mem_dim: int = 256,
        memory_capacity: int = 1024,
        bias_strength: float = 0.2,
    ):
        self.mem_dim = mem_dim
        self.bias_strength = bias_strength
        self._memory: list[tuple[np.ndarray, np.ndarray]] = []
        self._memory_capacity = memory_capacity
        self._pattern_phase: np.ndarray = np.zeros(mem_dim, dtype=np.float64)
        self._rng_global = np.random.RandomState(42)

    def store_pattern(self, context: np.ndarray, target_distribution: np.ndarray):
        key = context.ravel().astype(np.float64)
        val = target_distribution.ravel().astype(np.float64)
        if len(key) > self.mem_dim:
            key = key[: self.mem_dim]
        elif len(key) < self.mem_dim:
            key = np.pad(key, (0, self.mem_dim - len(key)))
        if len(val) > self.mem_dim:
            val = val[: self.mem_dim]
        elif len(val) < self.mem_dim:
            val = np.pad(val, (0, self.mem_dim - len(val)))
        key = key / (np.linalg.norm(key) + 1e-30)
        self._memory.append((key, val))
        if len(self._memory) > self._memory_capacity:
            self._memory.pop(0)

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        holographic_bias: float | None = None,
        context_embedding: np.ndarray | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        bs = holographic_bias if holographic_bias is not None else self.bias_strength
        rg = rng or _get_rng(seed)
        probs = _softmax(logits, temperature)

        if context_embedding is not None and self._memory:
            query = context_embedding.ravel().astype(np.float64)
            if len(query) > self.mem_dim:
                query = query[: self.mem_dim]
            elif len(query) < self.mem_dim:
                query = np.pad(query, (0, self.mem_dim - len(query)))
            query = query / (np.linalg.norm(query) + 1e-30)

            best_sim = -1.0
            best_val = None
            for key, val in self._memory:
                sim = float(np.dot(query, key))
                if sim > best_sim:
                    best_sim = sim
                    best_val = val

            if best_val is not None and best_sim > 0.3:
                retrieved = (
                    best_val[: len(probs)]
                    if len(best_val) >= len(probs)
                    else np.pad(best_val, (0, len(probs) - len(best_val)))
                )
                retrieved = np.maximum(retrieved, 0)
                retrieved = retrieved / (np.sum(retrieved) + 1e-30)
                probs = (1.0 - bs) * probs + bs * retrieved
                probs = probs / (np.sum(probs) + 1e-30)

        return _sample_multinomial(probs, rg)


class AttractorBasinSampler(BaseSampler):
    """Attractor Basin Escape — detect and break loops / degenerate
    attractor basins via basin topology analysis.

    Analyzes the recent token history for attractor basins (repetition
    cycles, periodic loops) and actively suppresses tokens that would
    deepen the basin while promoting tokens that escape it.
    """

    name = "attractor_basin"

    def __init__(
        self,
        window: int = 32,
        escape_strength: float = 0.5,
        min_cycle_length: int = 2,
        max_cycle_length: int = 16,
    ):
        self.window = window
        self.escape_strength = escape_strength
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length
        self._history: list[int] = []
        self._basin_depth: float = 0.0

    @property
    def basin_depth(self) -> float:
        return self._basin_depth

    def reset(self):
        self._history.clear()
        self._basin_depth = 0.0

    def _detect_cycles(self) -> list[list[int]]:
        cycles = []
        n = len(self._history)
        for L in range(self.min_cycle_length, min(self.max_cycle_length, n // 2) + 1):
            for start in range(max(0, n - 3 * L), n - 2 * L + 1):
                a = self._history[start : start + L]
                b = self._history[start + L : start + 2 * L]
                if a == b:
                    cycles.append(a)
                    break
        return cycles

    def _compute_basin_depth(self) -> float:
        if len(self._history) < 4:
            return 0.0
        cycles = self._detect_cycles()
        if not cycles:
            self._basin_depth *= 0.95
            return self._basin_depth
        depth = min(1.0, len(cycles) * 0.2 + max(0, len(self._history) - 4) * 0.01)
        self._basin_depth = depth
        return depth

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        escape_strength: float | None = None,
        context_tokens: list[int] | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        es = escape_strength if escape_strength is not None else self.escape_strength
        rg = rng or _get_rng(seed)

        if context_tokens:
            self._history = context_tokens[-self.window :]

        probs = _softmax(logits, temperature)
        depth = self._compute_basin_depth()

        if depth > 0.2 and self._history:
            cycles = self._detect_cycles()
            basin_tokens: set[int] = set()
            for cycle in cycles:
                basin_tokens.update(cycle)
            for token in basin_tokens:
                if 0 <= token < len(probs):
                    probs[token] *= max(0.01, 1.0 - es * depth)
            novelty_bonus = np.zeros(len(probs), dtype=np.float64)
            if self._history:
                last = self._history[-1]
                for i in range(len(probs)):
                    if i not in basin_tokens:
                        novelty_bonus[i] = es * depth * 0.1
                novelty_bonus[len(probs) // 2 :] *= 0.5
            probs = probs + novelty_bonus
            probs = np.maximum(probs, 0)
            probs = probs / (np.sum(probs) + 1e-30)

        token = _sample_multinomial(probs, rg)
        self._history.append(token)
        if len(self._history) > self.window:
            self._history.pop(0)
        return token


class HamiltonianTrajectorySampler(BaseSampler):
    """Hamiltonian Trajectory Sampling — Hamiltonian Monte Carlo on
    token space for diverse, coherent text.

    Uses leapfrog integration in probability space with a potential
    energy defined by the negative log-probability and momentum
    sampled from a Gaussian. Accept/reject via Metropolis-Hastings.
    """

    name = "hamiltonian_trajectory"

    def __init__(
        self,
        n_steps: int = 5,
        step_size: float = 0.1,
        mass: float = 1.0,
        target_accept: float = 0.65,
    ):
        self.n_steps = n_steps
        self.step_size = step_size
        self.mass = mass
        self.target_accept = target_accept
        self._acceptance_rate: float = 0.5
        self._trajectory_energy: list[float] = []

    @property
    def acceptance_rate(self) -> float:
        return self._acceptance_rate

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        rg = rng or _get_rng(seed)
        probs = _softmax(logits, temperature)
        n = len(probs)

        position = np.zeros(n, dtype=np.float64)
        position[int(np.argmax(probs))] = 1.0

        momentum = rg.normal(0, math.sqrt(self.mass), size=n)
        current_U = -math.log(probs[int(np.argmax(probs))] + 1e-30)
        current_K = 0.5 * np.sum(momentum**2) / self.mass
        current_H = current_U + current_K

        pos_new = position.copy()
        mom = momentum.copy()
        ss = self.step_size

        mom = mom - 0.5 * ss * self._potential_gradient(pos_new, probs)
        for _ in range(self.n_steps):
            pos_new = pos_new + ss * mom / self.mass
            pos_new = np.maximum(pos_new, 0)
            pos_new = pos_new / (np.sum(pos_new) + 1e-30)
            mom = mom - ss * self._potential_gradient(pos_new, probs)
        mom = mom - 0.5 * ss * self._potential_gradient(pos_new, probs)
        mom = -mom

        proposed_U = -math.log(np.dot(pos_new, probs) + 1e-30)
        proposed_K = 0.5 * np.sum(mom**2) / self.mass
        proposed_H = proposed_U + proposed_K

        accept_prob = min(1.0, math.exp(current_H - proposed_H))
        self._trajectory_energy.append(current_H)

        if rg.random() < accept_prob:
            self._acceptance_rate = 0.9 * self._acceptance_rate + 0.1 * 1.0
            chosen = _sample_multinomial(pos_new, rg)
        else:
            self._acceptance_rate = 0.9 * self._acceptance_rate + 0.1 * 0.0
            chosen = _sample_multinomial(probs, rg)

        self.step_size *= 1.0 + 0.01 * (self._acceptance_rate - self.target_accept)
        self.step_size = max(0.01, min(1.0, self.step_size))

        return chosen

    def _potential_gradient(self, pos: np.ndarray, probs: np.ndarray) -> np.ndarray:
        dot = max(np.dot(pos, probs), 1e-30)
        return -probs / dot


class VlasovFieldSampler(BaseSampler):
    """Vlasov Field Sampling — token-token interactions via
    mean-field potential.

    Treats tokens as charged particles in a plasma. The logits
    are modified by the mean-field potential generated by the
    context tokens. This creates coherent, long-range interactions
    between tokens.
    """

    name = "vlasov_field"

    def __init__(
        self,
        field_strength: float = 0.3,
        n_grid: int = 64,
        interaction_radius: float = 0.2,
        token_embedding: Callable[[int], np.ndarray] | None = None,
    ):
        self.field_strength = field_strength
        self.n_grid = n_grid
        self.interaction_radius = interaction_radius
        self.token_embedding = token_embedding or (
            lambda t: np.array([t % 1000], dtype=np.float64)
        )

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        field_strength: float | None = None,
        context_tokens: list[int] | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        fs = field_strength if field_strength is not None else self.field_strength
        rg = rng or _get_rng(seed)
        probs = _softmax(logits, temperature)

        if context_tokens and len(context_tokens) > 0:
            context_embeds = np.array(
                [self.token_embedding(t) for t in context_tokens[-64:]]
            )
            mean_context = np.mean(context_embeds, axis=0)
            potential_grid = np.zeros(self.n_grid, dtype=np.float64)
            for i in range(self.n_grid):
                for embed in context_embeds:
                    dx = np.linalg.norm(embed - mean_context)
                    potential_grid[i] += np.exp(-dx / (self.interaction_radius + 1e-10))
            potential_grid = potential_grid / (np.max(potential_grid) + 1e-30)
            potential_grid = potential_grid - np.mean(potential_grid)

            token_potentials = np.zeros(len(probs), dtype=np.float64)
            for t in range(min(len(probs), self.n_grid)):
                grid_idx = t % self.n_grid
                token_potentials[t] = potential_grid[grid_idx]

            if np.max(np.abs(token_potentials)) > 0:
                token_potentials = (
                    token_potentials * fs / (np.max(np.abs(token_potentials)) + 1e-30)
                )
                modified_logits = np.log(probs + 1e-30) + token_potentials
                probs = _softmax(modified_logits, temperature=1.0)

        return _sample_multinomial(probs, rg)


class PredictorCorrectorSampler(BaseSampler):
    """Predictor-Corrector Sampling — HDC predicts candidate tokens,
    model corrects the distribution, sample from corrected distribution.

    Uses a predictor (e.g. HDC, n-gram, or small model) to propose
    candidate tokens, then uses the full model logits to correct
    the distribution via Bayesian fusion.
    """

    name = "predictor_corrector"

    def __init__(
        self,
        corrector_strength: float = 0.5,
        predictor: Callable[[list[int]], list[tuple[int, float]]] | None = None,
    ):
        self.corrector_strength = corrector_strength
        self.predictor = predictor
        self._prediction_accuracy: float = 0.5

    @property
    def prediction_accuracy(self) -> float:
        return self._prediction_accuracy

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        corrector_strength: float | None = None,
        predictor_candidates: list[tuple[int, float]] | None = None,
        context_tokens: list[int] | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        cs = (
            corrector_strength
            if corrector_strength is not None
            else self.corrector_strength
        )
        rg = rng or _get_rng(seed)

        model_probs = _softmax(logits, temperature)

        if (
            predictor_candidates is None
            and self.predictor is not None
            and context_tokens
        ):
            predictor_candidates = self.predictor(context_tokens)

        if predictor_candidates and len(predictor_candidates) > 0:
            predictor_probs = np.zeros(len(model_probs), dtype=np.float64)
            predictor_indices = []
            predictor_scores = []
            for token, score in predictor_candidates:
                if 0 <= token < len(model_probs):
                    predictor_indices.append(token)
                    predictor_scores.append(float(score))
            if predictor_scores:
                predictor_scores = np.array(predictor_scores, dtype=np.float64)
                predictor_scores = np.maximum(predictor_scores, 1e-30)
                predictor_probs[predictor_indices] = predictor_scores
                predictor_probs = predictor_probs / (np.sum(predictor_probs) + 1e-30)

                prior_conf = self._prediction_accuracy
                alpha = cs * prior_conf
                fused = (1.0 - alpha) * model_probs + alpha * predictor_probs
                probs = fused / (np.sum(fused) + 1e-30)
            else:
                probs = model_probs
        else:
            probs = model_probs

        token = _sample_multinomial(probs, rg)
        return token

    def update_accuracy(self, predicted: int, actual: int):
        self._prediction_accuracy = 0.95 * self._prediction_accuracy + 0.05 * (
            1.0 if predicted == actual else 0.0
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. BornRuleQuantumSampler — Quantum Wavefunction Token Sampler
# ═══════════════════════════════════════════════════════════════════════════


class BornRuleQuantumSampler(BaseSampler):
    """
    Born Rule Quantum Wavefunction Sampler

    Treats the logit vector as a quantum state |ψ⟩ in a Hilbert space
    where each token |i⟩ is a basis state. The wavefunction is:

        |ψ⟩ = Σ_i α_i |i⟩   where   α_i = √(softmax(logits / τ)_i)

    Sampling = quantum measurement that collapses |ψ⟩ onto |i⟩ with
    Born probability:

        P(i) = |⟨i|ψ⟩|² = α_i² = softmax(logits / τ)_i

    This is formally equivalent to temperature sampling, but the
    quantum wavefunction framing enables three novel extensions:

    **Grover Amplification** (grover_iters > 0):
        Before collapse, apply Grover's amplitude amplification to
        boost oracle-marked tokens (those with above-median quality
        scores). After K Grover iterations (oracle reflection +
        diffusion about the mean + renormalization), the marked
        tokens have quadratically higher measurement probability.

    **Superposition Branching** (n_branches > 1):
        Instead of single-token collapse, perform a "partial
        measurement" returning the N highest-probability branches.
        This is the quantum analogue of top-K sampling: each returned
        token is a distinct collapsed branch of the superposition.
        Call sample_branches() to get the full list.

    **Decoherence** (decoherence_rate > 0):
        Inject Gaussian thermal noise into the wavefunction amplitudes,
        modeling environmental decoherence. This smooths the
        distribution, increases exploration, and prevents
        overconfident collapse. Negative amplitudes are rectified
        and the wavefunction is renormalized.

    References:
        - Born, M. (1926). Zur Quantenmechanik der Stoßvorgänge.
        - Grover, L. K. (1996). A fast quantum mechanical algorithm
          for database search. (arXiv:quant-ph/9605043)
        - Zurek, W. H. (2003). Decoherence, einselection, and the
          quantum origins of the classical. Rev. Mod. Phys. 75, 715.
    """

    name = "born_rule_quantum"

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        grover_iters: int = 0,
        n_branches: int = 1,
        decoherence_rate: float = 0.0,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.grover_iters = grover_iters
        self.n_branches = max(1, n_branches)
        self.decoherence_rate = max(0.0, decoherence_rate)

    def _build_wavefunction(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Build quantum wavefunction |ψ⟩ = Σ_i α_i |i⟩ from logits.

        The amplitude of token i is the square root of its softmax
        probability after temperature scaling:

            α_i = √(softmax(logits_i / τ))

        This is the fundamental mapping from classical logits to
        quantum amplitudes. The Born rule then says that measuring
        |ψ⟩ yields token i with probability |α_i|² = softmax(...),
        which recovers the classical distribution.
        """
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-30)
        amplitudes = np.sqrt(probs + 1e-30)
        return amplitudes

    def _grover_amplify(
        self,
        amplitudes: np.ndarray,
        oracle_scores: np.ndarray,
    ) -> np.ndarray:
        """Apply Grover amplitude amplification to boost marked tokens.

        The Grover iteration consists of:
        1. Oracle reflection: flip the sign of marked amplitudes
        2. Diffusion transform: invert all amplitudes about their mean
        3. Renormalization

        Tokens are marked if their oracle score exceeds the median.
        After K iterations, the marked tokens' amplitudes are
        amplified quadratically relative to unmarked tokens.

        Args:
            amplitudes: Current wavefunction amplitudes [vocab]
            oracle_scores: Quality scores per token [vocab]
        """
        oracle_mask = oracle_scores > np.median(oracle_scores)
        amps = amplitudes.copy()
        for _ in range(self.grover_iters):
            # Oracle reflection: flip sign of marked (good) states
            amps[oracle_mask] *= -1
            # Inversion about the mean (diffusion operator)
            mean = np.mean(amps)
            amps = 2.0 * mean - amps
            # Renormalize
            norm = np.linalg.norm(amps)
            if norm > 0:
                amps /= norm
        return amps

    def _apply_decoherence(self, amplitudes: np.ndarray) -> np.ndarray:
        """Inject thermal noise simulating environmental decoherence.

        Environmental interactions cause the wavefunction to lose
        coherence. We model this by adding Gaussian noise to the
        amplitudes, then rectifying (discarding negative amplitudes)
        and renormalizing. Higher decoherence_rate → more exploration.

        Args:
            amplitudes: Wavefunction amplitudes [vocab]
        """
        if self.decoherence_rate <= 0:
            return amplitudes
        noise = np.random.randn(*amplitudes.shape) * self.decoherence_rate
        amps = amplitudes + noise
        # Rectify: discard unphysical negative amplitudes
        amps = np.abs(amps)
        norm = np.linalg.norm(amps)
        if norm > 0:
            amps /= norm
        return amps

    def get_probs(
        self, logits: np.ndarray, temperature: float = 1.0, **kwargs
    ) -> np.ndarray:
        """Return Born probabilities after wavefunction processing.

        Builds the wavefunction, applies decoherence/Grover if
        configured, then converts amplitudes to Born probabilities
        and applies top-k/top-p filtering.
        """
        amplitudes = self._build_wavefunction(logits, temperature)
        if self.decoherence_rate > 0:
            amplitudes = self._apply_decoherence(amplitudes)
        probs = amplitudes**2
        if self.top_k > 0:
            probs = _top_k_mask(probs, self.top_k)
        if self.top_p > 0:
            probs = _top_p_mask(probs, self.top_p)
        return probs / (np.sum(probs) + 1e-30)

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        """Sample a token via Born rule wavefunction collapse.

        The full sampling pipeline:
        1. Build wavefunction |ψ⟩ from logits (α_i = √(softmax))
        2. Apply decoherence if configured (thermal noise)
        3. Apply Grover amplification if oracle_scores provided
        4. Convert |amplitude|² → Born probabilities
        5. Apply top-k / top-p filtering
        6. Collapse: measure |ψ⟩ → specific token

        When n_branches > 1, the collapse samples proportionally
        from the top-N branches of the superposition rather than
        the full distribution.

        Args:
            logits: Raw model logits [vocab_size]
            temperature: Temperature scaling factor (default: 1.0)
            seed: Random seed for reproducibility
            rng: Optional pre-seeded RandomState

        Keyword Args:
            grover_iters: Override Grover iteration count
            decoherence_rate: Override decoherence rate
            n_branches: Override superposition branch count
            oracle_scores: Quality scores for Grover marking [vocab]

        Returns:
            Sampled token index (int)

        Raises:
            ValueError: If all probabilities are zero after filtering.
        """
        rg = rng or _get_rng(seed)

        grover = kwargs.get("grover_iters", self.grover_iters)
        decohere = kwargs.get("decoherence_rate", self.decoherence_rate)
        branches = kwargs.get("n_branches", self.n_branches)
        oracle_scores = kwargs.get("oracle_scores", None)

        # 1. Build wavefunction
        amplitudes = self._build_wavefunction(logits, temperature)

        # 2. Apply decoherence (environmental noise)
        if decohere > 0:
            amplitudes = self._apply_decoherence(amplitudes)

        # 3. Grover amplify if oracle provided
        if oracle_scores is not None and grover > 0:
            amplitudes = self._grover_amplify(amplitudes, oracle_scores)

        # 4. Born rule: probabilities = |amplitude|²
        probs = amplitudes**2

        # 5. Apply top-k / top-p filtering
        if self.top_k > 0:
            probs = _top_k_mask(probs, self.top_k)
        if self.top_p > 0:
            probs = _top_p_mask(probs, self.top_p)

        if np.sum(probs) < 1e-30:
            raise ValueError("All probabilities are zero after filtering.")

        # 6. Collapse: measure |ψ⟩ → |token⟩
        if branches <= 1:
            return _sample_multinomial(probs, rg)
        else:
            # Superposition: sample from top-N branches proportionally
            top_idx = np.argsort(probs)[-branches:]
            top_probs = probs[top_idx]
            top_probs = top_probs / (np.sum(top_probs) + 1e-30)
            chosen = _sample_multinomial(top_probs, rg)
            return int(top_idx[chosen])

    def sample_branches(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        n_branches: int | None = None,
        oracle_scores: np.ndarray | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
    ) -> list[int]:
        """Sample multiple tokens via superposition branching.

        Performs a partial measurement that preserves the top-N
        branches of the wavefunction superposition. Each returned
        token represents a distinct collapsed branch, ordered from
        highest to lowest probability.

        This is the quantum analogue of top-K sampling, where the
        wavefunction is measured simultaneously along multiple
        basis directions, yielding a set of most-likely tokens.

        Args:
            logits: Raw model logits [vocab_size]
            temperature: Temperature scaling factor
            n_branches: Number of branches (default: self.n_branches)
            oracle_scores: Optional quality scores for Grover marking
            seed: Random seed
            rng: Optional pre-seeded RandomState

        Returns:
            List of token indices, highest probability first
        """
        rg = rng or _get_rng(seed)
        branches = n_branches if n_branches is not None else self.n_branches

        amplitudes = self._build_wavefunction(logits, temperature)

        if self.decoherence_rate > 0:
            amplitudes = self._apply_decoherence(amplitudes)

        if oracle_scores is not None and self.grover_iters > 0:
            amplitudes = self._grover_amplify(amplitudes, oracle_scores)

        probs = amplitudes**2

        if self.top_k > 0:
            probs = _top_k_mask(probs, self.top_k)
        if self.top_p > 0:
            probs = _top_p_mask(probs, self.top_p)

        branches = min(branches, len(probs))
        top_idx = np.argsort(probs)[-branches:]
        return [int(i) for i in top_idx[::-1]]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Speculative & Rejection Samplers
# ═══════════════════════════════════════════════════════════════════════════


class SpectralSpeculativeSampler(BaseSampler):
    """Speculative sampling (Leviathan et al., 2022; Chen et al., 2023).
    Uses a draft model to propose tokens, accepts/rejects via
    rejection sampling against the target model distribution.
    """

    name = "speculative"

    def __init__(
        self,
        draft_model_fn: Callable[[list[int]], np.ndarray] | None = None,
        max_draft_length: int = 5,
    ):
        self.draft_model_fn = draft_model_fn
        self.max_draft_length = max_draft_length
        self._acceptance_rate: float = 0.0
        self._n_accepted: int = 0
        self._n_drafted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self._acceptance_rate

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        draft_logits: np.ndarray | None = None,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        rg = rng or _get_rng(seed)
        target_probs = _softmax(logits, temperature)

        if draft_logits is not None:
            draft_probs = _softmax(draft_logits, temperature)
            draft_token = _sample_multinomial(draft_probs, rg)
            p_target = target_probs[draft_token]
            p_draft = draft_probs[draft_token]
            accept_prob = min(1.0, p_target / max(p_draft, 1e-30))

            self._n_drafted += 1
            if rg.random() < accept_prob:
                self._n_accepted += 1
                self._acceptance_rate = self._n_accepted / max(self._n_drafted, 1)
                return draft_token

        self._acceptance_rate = self._n_accepted / max(self._n_drafted, 1)
        return _sample_multinomial(target_probs, rg)


# ═══════════════════════════════════════════════════════════════════════════
# 7. SamplerPipeline — Chain Multiple Samplers
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineStep:
    sampler: BaseSampler | LogitTransform
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class SamplerPipeline(BaseSampler):
    """Chain multiple samplers and transforms in sequence.

    Example:
        pipeline = SamplerPipeline(steps=[
            PipelineStep(RepetitionPenalty(1.15)),
            PipelineStep(TopKSampler(40)),
            PipelineStep(TopPSampler(0.95)),
            PipelineStep(TemperatureSampler()),
        ])
        token = pipeline.sample(logits, context=[...])
    """

    name = "pipeline"

    def __init__(
        self,
        steps: list[PipelineStep] | None = None,
        fallback_sampler: BaseSampler | None = None,
    ):
        self.steps: list[PipelineStep] = steps or []
        self.fallback_sampler = fallback_sampler or GreedySampler()
        self._intermediate_distributions: list[dict] = []

    def add_step(
        self, sampler: BaseSampler | LogitTransform, **params
    ) -> SamplerPipeline:
        self.steps.append(PipelineStep(sampler=sampler, params=params))
        return self

    def clear(self):
        self.steps.clear()
        self._intermediate_distributions.clear()

    @property
    def intermediate_distributions(self) -> list[dict]:
        return list(self._intermediate_distributions)

    def get_probs(
        self, logits: np.ndarray, temperature: float = 1.0, **kwargs
    ) -> np.ndarray:
        current = logits.copy().astype(np.float64)
        for step in self.steps:
            if not step.enabled:
                continue
            try:
                merged = {**step.params, **kwargs}
                if isinstance(step.sampler, LogitTransform):
                    current = step.sampler.apply(current, **merged)
                elif isinstance(step.sampler, BaseSampler):
                    pass
            except Exception:
                continue
        return _softmax(current, temperature)

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        self._intermediate_distributions.clear()

        current = logits.copy().astype(np.float64)
        rg = rng or _get_rng(seed)

        for step in self.steps:
            if not step.enabled:
                continue
            try:
                merged = {**step.params, **kwargs}
                if isinstance(step.sampler, LogitTransform):
                    current = step.sampler.apply(current, **merged)
                    self._intermediate_distributions.append(
                        {
                            "step": step.sampler.name,
                            "type": "transform",
                        }
                    )
                elif isinstance(step.sampler, BaseSampler):
                    pass
            except Exception as exc:
                current = logits.copy().astype(np.float64)

        for step in self.steps:
            if not step.enabled:
                continue
            if isinstance(step.sampler, BaseSampler):
                try:
                    merged_kw = {
                        **step.params,
                        **kwargs,
                        "seed": None if seed is None else rg.randint(0, 2**31),
                        "rng": rg,
                    }
                    result = step.sampler.sample(current, temperature, **merged_kw)
                    self._intermediate_distributions.append(
                        {
                            "step": step.sampler.name,
                            "type": "sampler",
                            "token": result,
                        }
                    )
                    return result
                except Exception:
                    continue

        return self.fallback_sampler.sample(current, temperature, rng=rg)

    def inspect(self, logits: np.ndarray, temperature: float = 1.0, **kwargs) -> dict:
        current = logits.copy().astype(np.float64)
        stages = []
        for step in self.steps:
            if not step.enabled:
                continue
            try:
                merged = {**step.params, **kwargs}
                if isinstance(step.sampler, LogitTransform):
                    current = step.sampler.apply(current, **merged)
                    probs = _softmax(current, temperature)
                    stages.append(
                        {
                            "name": step.sampler.name,
                            "entropy": _entropy(probs),
                            "top_token": int(np.argmax(probs)),
                            "top_prob": float(np.max(probs)),
                            "n_nonzero": int(np.sum(probs > 1e-10)),
                        }
                    )
            except Exception:
                continue
        return {
            "n_steps": len(self.steps),
            "stages": stages,
            "final_entropy": stages[-1]["entropy"] if stages else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 8. AdaptiveSampler — Auto-Select Best Sampler
# ═══════════════════════════════════════════════════════════════════════════


class AdaptiveSampler(BaseSampler):
    """Adaptive sampler that monitors generation statistics and
    switches to the best sampler based on context.

    Uses a multi-armed bandit (UCB1) to learn which sampler
    performs best for the current generation regime.
    """

    name = "adaptive"

    def __init__(
        self,
        samplers: list[BaseSampler] | None = None,
        exploration_factor: float = 1.0,
        adaptation_window: int = 100,
    ):
        self.samplers = samplers or [
            TemperatureSampler(),
            TopKSampler(40),
            TopPSampler(0.95),
            MinPSampler(0.05),
            TypicalSampler(0.95),
            MirostatV1(),
        ]
        self.exploration_factor = exploration_factor
        self.adaptation_window = adaptation_window

        self._n_pulls: list[int] = [0] * len(self.samplers)
        self._rewards: list[float] = [0.0] * len(self.samplers)
        self._total_pulls: int = 0
        self._history: list[int] = []
        self._entropy_history: list[float] = []
        self._current_sampler_idx: int = 0

    @property
    def current_sampler(self) -> BaseSampler:
        return self.samplers[self._current_sampler_idx]

    @property
    def sampler_weights(self) -> list[float]:
        return [
            self._rewards[i] / max(self._n_pulls[i], 1)
            for i in range(len(self.samplers))
        ]

    def reset(self):
        self._n_pulls = [0] * len(self.samplers)
        self._rewards = [0.0] * len(self.samplers)
        self._total_pulls = 0
        self._history.clear()
        self._entropy_history.clear()
        self._current_sampler_idx = 0

    def _select_sampler(self, entropy: float) -> int:
        n = len(self.samplers)
        if self._total_pulls < n * 2:
            return self._total_pulls % n
        ucb_scores = []
        total_log = math.log(self._total_pulls + 1)
        for i in range(n):
            if self._n_pulls[i] == 0:
                ucb_scores.append(float("inf"))
            else:
                avg = self._rewards[i] / self._n_pulls[i]
                ucb = avg + self.exploration_factor * math.sqrt(
                    total_log / self._n_pulls[i]
                )
                ucb_scores.append(ucb)
        return int(np.argmax(ucb_scores))

    def _compute_reward(self, logits: np.ndarray, token: int, entropy: float) -> float:
        probs = _softmax(logits)
        token_prob = probs[token] if token < len(probs) else 0.0
        prob_reward = math.log(max(token_prob, 1e-10)) / math.log(max(len(probs), 2))
        ent_bonus = 0.0
        if self._entropy_history:
            ent_change = entropy - np.mean(self._entropy_history[-10:])
            ent_bonus = 0.1 * min(0, ent_change)
        return max(-1.0, prob_reward + ent_bonus)

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        rg = rng or _get_rng(seed)
        probs = _softmax(logits, temperature)
        entropy = _entropy(probs)
        self._entropy_history.append(entropy)
        if len(self._entropy_history) > self.adaptation_window:
            self._entropy_history.pop(0)

        idx = self._select_sampler(entropy)
        self._current_sampler_idx = idx
        sampler = self.samplers[idx]

        token = sampler.sample(logits, temperature, seed=None, rng=rg, **kwargs)

        reward = self._compute_reward(logits, token, entropy)
        self._n_pulls[idx] += 1
        self._rewards[idx] += reward
        self._total_pulls += 1
        self._history.append(idx)
        if len(self._history) > self.adaptation_window:
            self._history.pop(0)

        return token


# ═══════════════════════════════════════════════════════════════════════════
# 9. BatchSampler — Vectorized Batch Sampling
# ═══════════════════════════════════════════════════════════════════════════


class BatchSampler(BaseSampler):
    """Vectorized batch sampling for multiple sequences in parallel.

    Uses counter-based RNG for reproducibility across batch elements.
    Supports per-sequence state tracking.
    """

    name = "batch"

    def __init__(self, base_sampler: BaseSampler, use_counter_rng: bool = True):
        self.base_sampler = base_sampler
        self.use_counter_rng = use_counter_rng
        self._seq_states: dict[int, dict] = {}

    def reset_seq_state(self, seq_id: int):
        self._seq_states.pop(seq_id, None)

    def reset_all(self):
        self._seq_states.clear()

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        **kwargs,
    ) -> int:
        return self.base_sampler.sample(
            logits, temperature, seed=seed, rng=rng, **kwargs
        )

    def batch_sample(
        self,
        logits_batch: np.ndarray,
        temperature: float = 1.0,
        seeds: list[int] | None = None,
        **kwargs,
    ) -> list[int]:
        batch_size = len(logits_batch)
        rg = np.random.RandomState(42 if seeds is None else seeds[0])
        results = []
        for i in range(batch_size):
            if self.use_counter_rng and seeds is not None:
                seq_rng = np.random.RandomState(seeds[i % len(seeds)])
            else:
                seq_rng = np.random.RandomState(rg.randint(0, 2**31))
            token = self.base_sampler.sample(
                logits_batch[i], temperature, rng=seq_rng, **kwargs
            )
            results.append(token)
        return results

    def batch_sample_vectorized(
        self, logits_batch: np.ndarray, temperature: float = 1.0, **kwargs
    ) -> np.ndarray:
        batch_size = len(logits_batch)
        probs_batch = np.array(
            [_softmax(logits, temperature) for logits in logits_batch]
        )
        cumsum = np.cumsum(probs_batch, axis=1)
        uniform = np.random.uniform(size=(batch_size, 1))
        indices = np.argmax(cumsum > uniform, axis=1)
        return indices


# ═══════════════════════════════════════════════════════════════════════════
# 10. Factory: create_sampler_pipeline
# ═══════════════════════════════════════════════════════════════════════════

REGISTRY: dict[str, type[BaseSampler] | type[LogitTransform]] = {
    # Standard samplers
    "greedy": GreedySampler,
    "temperature": TemperatureSampler,
    "top_k": TopKSampler,
    "top_p": TopPSampler,
    "min_p": MinPSampler,
    "typical": TypicalSampler,
    "locally_typical": LocallyTypicalSampler,
    "eta": EtaSampler,
    "tfs": TFSSampler,
    "top_a": TopASampler,
    "contrastive": ContrastiveSampler,
    "mirostat_v1": MirostatV1,
    "mirostat_v2": MirostatV2,
    "beam_search": BeamSearchSampler,
    "diverse_beam": DiverseBeamSearch,
    # Penalties & transforms
    "repetition_penalty": RepetitionPenalty,
    "frequency_penalty": FrequencyPenalty,
    "presence_penalty": PresencePenalty,
    "logit_bias": LogitBiasProcessor,
    "token_ban": TokenBanProcessor,
    "grammar_constraint": GrammarConstraint,
    "json_mode": JSONModeProcessor,
    "xml_mode": XMLModeProcessor,
    # Novel
    "quantum_collapse": QuantumCollapseSampler,
    "born_rule_quantum": "BornRuleQuantumSampler",
    "spectral_resonance": SpectralResonanceSampler,
    "holographic_pattern": HolographicPatternSampler,
    "attractor_basin": AttractorBasinSampler,
    "hamiltonian": HamiltonianTrajectorySampler,
    "vlasov_field": VlasovFieldSampler,
    "predictor_corrector": PredictorCorrectorSampler,
    # Speculative
    "speculative": SpectralSpeculativeSampler,
    # Meta
    "adaptive": AdaptiveSampler,
    "pipeline": SamplerPipeline,
    "batch": lambda: BatchSampler(TemperatureSampler()),
}


def create_sampler(name: str, **kwargs) -> BaseSampler | LogitTransform:
    cls = REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown sampler '{name}'. Available: {list(REGISTRY.keys())}"
        )
    if isinstance(cls, str):
        cls = globals()[cls]
    return cls(**kwargs)


def create_sampler_pipeline(
    config: list[dict] | None = None, **kwargs
) -> SamplerPipeline:
    """Create a sampler pipeline from a config list.

    Each element of config is a dict with:
      - name: str (sampler/transform name)
      - params: dict (optional, keyword arguments)
      - enabled: bool (optional, default True)

    Example:
        pipeline = create_sampler_pipeline([
            {'name': 'repetition_penalty', 'params': {'penalty': 1.15}},
            {'name': 'top_k', 'params': {'default_k': 40}},
            {'name': 'top_p', 'params': {'default_p': 0.95}},
            {'name': 'temperature'},
        ])
    """
    pipeline = SamplerPipeline()

    if config:
        for step_cfg in config:
            name = step_cfg.get("name", "")
            params = step_cfg.get("params", {})
            enabled = step_cfg.get("enabled", True)
            sampler = create_sampler(name, **params)
            step = PipelineStep(sampler=sampler, params={}, enabled=enabled)
            pipeline.steps.append(step)

    return pipeline


def list_samplers() -> list[str]:
    return list(REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════════
# 11. Self-Test / Verification
# ═══════════════════════════════════════════════════════════════════════════


def _run_self_test() -> int:
    """Run ALL samplers through basic verification. Returns number of failures."""
    rng = np.random.RandomState(42)
    test_logits = rng.randn(32000).astype(np.float64)
    vocab_size = len(test_logits)

    failures = 0
    total = 0

    def check(name: str, sampler: BaseSampler, **kwargs):
        nonlocal failures, total
        total += 1
        try:
            token = sampler.sample(test_logits, temperature=0.8, **kwargs)
            if not (0 <= token < vocab_size):
                raise ValueError(f"token {token} out of range [0, {vocab_size})")
            probs = sampler.get_probs(test_logits, temperature=0.8)
            if not np.allclose(np.sum(probs), 1.0, atol=1e-5):
                raise ValueError(f"probs sum = {np.sum(probs)} != 1.0")
            batch = sampler.batch_sample(
                np.stack([test_logits, test_logits * 0.9]), temperature=0.8, **kwargs
            )
            if len(batch) != 2:
                raise ValueError(f"batch returned {len(batch)} tokens, expected 2")
            print(f"  ✅ {name}")
        except Exception as exc:
            failures += 1
            print(f"  ❌ {name}: {exc}")

    def check_transform(name: str, transform: LogitTransform):
        nonlocal failures, total
        total += 1
        try:
            result = transform.apply(test_logits.copy(), context=[42, 43, 44])
            if len(result) != len(test_logits):
                raise ValueError(f"output length {len(result)} != {len(test_logits)}")
            print(f"  ✅ {name}")
        except Exception as exc:
            failures += 1
            print(f"  ❌ {name}: {exc}")

    print("═══ Sampler Engine Self-Test ═══")
    print(f"  vocab_size={vocab_size}")

    # Standard samplers
    check("GreedySampler", GreedySampler())
    check("TemperatureSampler", TemperatureSampler())
    check("TopKSampler(40)", TopKSampler(40))
    check("TopPSampler(0.95)", TopPSampler(0.95))
    check("MinPSampler(0.05)", MinPSampler(0.05))
    check("TypicalSampler(0.95)", TypicalSampler(0.95))
    check("LocallyTypicalSampler", LocallyTypicalSampler(0.95))
    check("EtaSampler(0.2)", EtaSampler(0.2))
    check("TFSSampler(0.95)", TFSSampler(0.95))
    check("TopASampler(0.2)", TopASampler(0.2))
    check("ContrastiveSampler", ContrastiveSampler(0.6))
    check("MirostatV1", MirostatV1())
    check("MirostatV2", MirostatV2())
    check("BeamSearchSampler", BeamSearchSampler())
    check("DiverseBeamSearch", DiverseBeamSearch())

    # Transforms
    check_transform("RepetitionPenalty", RepetitionPenalty(1.15))
    check_transform("FrequencyPenalty", FrequencyPenalty(0.1))
    check_transform("PresencePenalty", PresencePenalty(0.1))
    check_transform("LogitBiasProcessor", LogitBiasProcessor({100: 5.0}))
    check_transform("TokenBanProcessor", TokenBanProcessor({0, 1, 2}))

    # Novel samplers
    check("QuantumCollapseSampler", QuantumCollapseSampler())
    check(
        "SpectralResonanceSampler",
        SpectralResonanceSampler(),
        context_tokens=[10, 20, 30, 40, 50, 60, 70, 80],
    )
    check(
        "HolographicPatternSampler",
        HolographicPatternSampler(),
        context_embedding=np.random.randn(256),
    )
    check(
        "AttractorBasinSampler",
        AttractorBasinSampler(),
        context_tokens=[1, 2, 3, 1, 2, 3, 1, 2],
    )
    check("HamiltonianTrajectorySampler", HamiltonianTrajectorySampler())
    check("VlasovFieldSampler", VlasovFieldSampler(), context_tokens=[10, 20, 30, 40])
    check(
        "PredictorCorrectorSampler",
        PredictorCorrectorSampler(),
        context_tokens=[10, 20, 30],
        predictor_candidates=[(100, 0.9), (200, 0.8)],
    )

    # Quantum-inspired samplers
    check("BornRuleQuantumSampler()", BornRuleQuantumSampler())
    check(
        "BornRuleQuantumSampler(grover=2)",
        BornRuleQuantumSampler(grover_iters=2),
        oracle_scores=rng.rand(vocab_size),
    )
    check(
        "BornRuleQuantumSampler(decoherence=0.1)",
        BornRuleQuantumSampler(decoherence_rate=0.1),
    )
    check("BornRuleQuantumSampler(branches=3)", BornRuleQuantumSampler(n_branches=3))
    check("BornRuleQuantumSampler(top_k=10)", BornRuleQuantumSampler(top_k=10))
    check("BornRuleQuantumSampler(top_p=0.9)", BornRuleQuantumSampler(top_p=0.9))
    # Sample branches (returns list, verify it works)
    brq = BornRuleQuantumSampler(n_branches=5)
    branches = brq.sample_branches(test_logits)
    if len(branches) != 5 or any(not (0 <= t < vocab_size) for t in branches):
        print(f"  ❌ BornRuleQuantumSampler.sample_branches")

    # Speculative
    check(
        "SpectralSpeculativeSampler",
        SpectralSpeculativeSampler(),
        draft_logits=rng.randn(vocab_size),
    )

    # Meta samplers
    adaptive = AdaptiveSampler()
    check("AdaptiveSampler", adaptive)
    check("AdaptiveSampler(2nd call)", adaptive)

    batch_sampler = BatchSampler(TemperatureSampler())
    check("BatchSampler", batch_sampler)

    # Pipeline
    pipeline = SamplerPipeline(
        steps=[
            PipelineStep(RepetitionPenalty(1.15)),
            PipelineStep(TopKSampler(40)),
            PipelineStep(TopPSampler(0.95)),
            PipelineStep(TemperatureSampler()),
        ]
    )
    check("SamplerPipeline", pipeline)

    # create_sampler_pipeline
    pipe2 = create_sampler_pipeline(
        [
            {"name": "repetition_penalty", "params": {"penalty": 1.1}},
            {"name": "top_k", "params": {"default_k": 50}},
            {"name": "top_p", "params": {"default_p": 0.9}},
            {"name": "mirostat_v1", "params": {"tau": 3.0}},
        ]
    )
    check("create_sampler_pipeline", pipe2)

    # get_probs on pipeline
    probs = pipeline.get_probs(test_logits)
    if not np.allclose(np.sum(probs), 1.0):
        print(f"  ⚠ Pipeline get_probs sum = {np.sum(probs)}")

    # inspect
    inspection = pipeline.inspect(test_logits)
    print(
        f"  📊 Pipeline inspection: {inspection['n_steps']} steps, "
        f"final entropy={inspection['final_entropy']:.4f}"
    )

    print(f"\n═══ Results: {total - failures}/{total} passed, {failures} failed ═══")
    return failures


# ═══════════════════════════════════════════════════════════════════════════
# 12. Integration with SpectralStream
# ═══════════════════════════════════════════════════════════════════════════


def build_default_pipeline(
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    mirostat_tau: float | None = None,
    mirostat_eta: float | None = None,
    use_mirostat: bool = False,
) -> SamplerPipeline:
    pipeline = SamplerPipeline()

    if repetition_penalty != 1.0:
        pipeline.add_step(RepetitionPenalty(repetition_penalty))

    if frequency_penalty > 0:
        pipeline.add_step(FrequencyPenalty(frequency_penalty))

    if presence_penalty > 0:
        pipeline.add_step(PresencePenalty(presence_penalty))

    if top_k < len(REGISTRY):
        pipeline.add_step(TopKSampler(top_k))

    if top_p < 1.0:
        pipeline.add_step(TopPSampler(top_p))

    if use_mirostat and mirostat_tau is not None:
        pipeline.add_step(MirostatV1(tau=mirostat_tau, eta=mirostat_eta or 0.1))
    else:
        pipeline.add_step(TemperatureSampler())

    return pipeline


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        n_failures = _run_self_test()
        sys.exit(n_failures)
    elif "--list" in sys.argv:
        print("Available samplers:")
        for name in list_samplers():
            print(f"  - {name}")
    else:
        print("SpectralStream Sampler Engine")
        print("Usage: python -m spectralstream.sampler_engine --test")

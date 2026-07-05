from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _ensure_2d(t: np.ndarray) -> Tuple[np.ndarray, tuple]:
    if t.ndim == 1:
        return t.reshape(1, -1), t.shape
    if t.ndim > 2:
        orig = t.shape
        return t.reshape(orig[0], -1), orig
    return t, t.shape

def _restore_shape(t: np.ndarray, orig_shape: tuple) -> np.ndarray:
    return t.reshape(orig_shape) if t.shape != orig_shape else t

def _safe_bytes(data: Any) -> int:
    if isinstance(data, np.ndarray):
        return data.nbytes
    if isinstance(data, dict):
        return sum(_safe_bytes(v) for v in data.values()) + sum(_safe_bytes(k) for k in data.keys())
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    if isinstance(data, (int, float, np.integer, np.floating)):
        return 8
    if isinstance(data, str):
        return len(data)
    return 0

def _generate_monomials(n_vars: int, degree: int) -> list:
    """Generate all monomials of given degree in n_vars variables."""
    if degree == 0:
        return [()]
    if degree == 1:
        return [(i,) for i in range(n_vars)]
    result = []
    for i in range(n_vars):
        for rest in _generate_monomials(n_vars, degree - 1):
            if len(rest) == 0 or i >= rest[0]:
                result.append((i,) + rest)
    return result[:50]  # limit for efficiency

def _register_all():
    classes = [
        # Quantum Mechanics
        QuantumStateCompression,
        QuantumEntanglementCompression,
        QuantumTunnelingOptimizer,
        DensityMatrixCompression,
        QuantumErrorCorrectionCompression,
        # Plasma Physics
        VlasovDistributionCompression,
        PlasmaOscillationDecomposition,
        MHDWaveCompression,
        DebyeShieldingCompression,
        PlasmaTurbulenceDecomposition,
        # Information Theory
        RateDistortionOptimalCompression,
        MutualInformationCompression,
        KolmogorovComplexityApproximation,
        FisherInformationWeighting,
        EntropyRateCompression,
        # Advanced Mathematics
        ManifoldLearningCompression,
        OptimalTransportCompression,
        CategoryTheoryCompression,
        AlgebraicGeometryCompression,
        TopologicalDataCompression,
        # Hybrid
        ResonanceCompression,
        HarmonicOscillatorDecomposition,
        FourierNeuralOperatorCompression,
        WaveletScatteringTransform,
        NeuralODECompression,
    ]
    for cls in classes:
        inst = cls()
        ALL_METHODS[inst.name] = inst

class BenchmarkResult:
    method_name: str
    category: str
    compression_ratio: float
    snr_db: float
    rel_error: float
    mae: float
    max_error: float
    cosine_similarity: float
    time_ms: float
    mse: float

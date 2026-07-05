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

class CompressionBenchmark:
    """Benchmark all cutting-edge methods on various matrix types."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def run_single(self, method: CompressionMethod, tensor: np.ndarray) -> Optional[BenchmarkResult]:
        """Run a single method and return benchmark result."""
        try:
            t0 = time.time()
            comp, meta = method.compress(tensor)
            tc = (time.time() - t0) * 1000

            t0 = time.time()
            recon = method.decompress(comp, meta)
            td = (time.time() - t0) * 1000

            errors = method.estimate_error(tensor)
            orig_bytes = tensor.nbytes
            comp_bytes = _safe_bytes(comp) + _safe_bytes(meta)
            ratio = max(comp_bytes / max(orig_bytes, 1), 1e-6)

            return BenchmarkResult(
                method_name=method.name,
                category=method.category,
                compression_ratio=ratio,
                snr_db=errors["snr_db"],
                rel_error=errors["rel_error"],
                mae=errors["mae"],
                max_error=errors["max_error"],
                cosine_similarity=errors["cosine_similarity"],
                time_ms=tc + td,
                mse=errors["mse"],
            )
        except Exception as e:
            print(f"  {method.name}: FAILED ({e})")
            return None

    def run_all(self, tensor: np.ndarray) -> List[BenchmarkResult]:
        """Run all methods on a tensor."""
        self.results = []
        for name, method in ALL_METHODS.items():
            result = self.run_single(method, tensor)
            if result is not None:
                self.results.append(result)
        return self.results

    def summary_table(self, top_n: int = 25) -> str:
        """Print summary table sorted by relative error."""
        sorted_results = sorted(self.results, key=lambda r: r.rel_error)
        lines = [
            f"{'Method':<35} {'Category':<22} {'Ratio':>8} {'SNR(dB)':>10} {'RelErr':>10} {'CosSim':>8} {'Time(ms)':>10}",
            "-" * 113,
        ]
        for r in sorted_results[:top_n]:
            lines.append(
                f"{r.method_name:<35} {r.category:<22} {r.compression_ratio:>8.4f} "
                f"{r.snr_db:>10.2f} {r.rel_error:>10.6f} {r.cosine_similarity:>8.4f} {r.time_ms:>10.1f}"
            )
        return "\n".join(lines)

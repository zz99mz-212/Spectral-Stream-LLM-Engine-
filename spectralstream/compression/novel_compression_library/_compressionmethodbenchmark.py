from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import dct, idct, fwht, zigzag_indices, next_power_of_two


def _ensure_2d(t):
    if t.ndim == 1:
        return t.reshape(1, -1), t.shape
    if t.ndim > 2:
        orig = t.shape
        return t.reshape(orig[0], -1), orig
    return t, t.shape

def _restore_shape(t, orig_shape):
    return t.reshape(orig_shape) if t.shape != orig_shape else t

def _safe_bytes(data):
    if isinstance(data, np.ndarray):
        return data.nbytes
    if isinstance(data, dict):
        return sum(_safe_bytes(v) for v in data.values())
    if isinstance(data, (list, tuple)):
        return sum(_safe_bytes(x) for x in data)
    return 8

def _register_all():
    classes = [
        TTSVD, TTOR, TRSVD, CPALS, TuckerSVD, BlockTucker, HierarchicalTucker,
        TensorNetwork, ButterflyFactorization, KroneckerProduct, BlockDiagonal,
        CirculantApprox, ToeplitzApprox, HankelApprox, LoTR,
        LloydMaxQuant, AdaptiveScalarQuant, ProductQuantization, ResidualVectorQuant,
        AdditiveCodebookQuant, E8LatticeQuant, LatticeQuantAnchored, MixedPrecisionQuant,
        HessianAwareQuant, FisherInfoQuant, GPTQLayerQuant, AWQActivationAware,
        BinaryQuant, TernaryQuant, NF4Quant,
        DCTSpectral, DCT2DBlock, WaveletThreshold, FWHTCompress, RandomizedHadamard,
        WinogradTransform, NTTCompress, RandomRotationQuant, ButterflySparseTransform,
        SparseRandomProjection,
        StructuredSparsity, BlockSparsity, UnstructuredPruning, SparseGPT, WandaPruning,
        DynamicNMSparsity, ChannelPruning, GroupLasso, AdaptiveSparsityAlloc, SparseQuantizeCombined,
        HuffmanCoding, RANS, TANS, ArithmeticCoding, LZ77Entropy,
        VlasovMeanField, HolographicPhaseEncoding, QuantumTensorNetwork, TimeCrystalPhase,
        PlasmaFieldDecomposition, SpectralDensityEstimation, InformationBottleneck,
        RateDistortionOptimal, FisherRaoCompression, SymplecticWeightEvolution,
        LandauZenerSampling, BoltzmannEncoding, MaxEntropyCompression,
        CrossLayerDelta, HierarchicalClusteredPQ,
    ]
    for cls in classes:
        inst = cls()
        ALL_METHODS[inst.name] = inst

class CompressionMethodBenchmark:
    """Benchmark all methods and find Pareto-optimal solutions."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.results = []

    def run_all(self, methods=None, **kw):
        if methods is None:
            methods = list(ALL_METHODS.keys())
        self.results = []
        for name in methods:
            if name not in ALL_METHODS:
                continue
            method = ALL_METHODS[name]
            try:
                t0 = time.time()
                comp, meta = method.compress(self.tensor, **kw)
                tc = (time.time() - t0) * 1000
                t0 = time.time()
                recon = method.decompress(comp, meta)
                td = (time.time() - t0) * 1000
                errors = method.estimate_error(self.tensor, **kw)
                orig = self.tensor.nbytes
                comp_bytes = _safe_bytes(comp) + _safe_bytes(meta)
                ratio = max(comp_bytes / max(orig, 1), 1e-6)
                self.results.append(BenchmarkResult(
                    method_name=name, category=method.category,
                    compression_ratio=ratio, snr_db=errors["snr_db"],
                    rel_error=errors["rel_error"], mae=errors["mae"],
                    max_error=errors["max_error"], cosine_similarity=errors["cosine_similarity"],
                    time_ms=tc+td, mse=errors["mse"],
                ))
            except Exception as e:
                pass
        return self.results

    def sorted_by_ratio(self, ascending=True):
        return sorted(self.results, key=lambda r: r.compression_ratio, reverse=not ascending)

    def sorted_by_error(self, key="rel_error", ascending=True):
        km = {"rel_error": lambda r: r.rel_error, "snr": lambda r: -r.snr_db,
              "mse": lambda r: r.mse, "mae": lambda r: r.mae}
        return sorted(self.results, key=km.get(key, lambda r: r.rel_error), reverse=ascending)

    def pareto_optimal(self):
        """Find Pareto-optimal methods (best ratio for each error level)."""
        if not self.results:
            return []
        pts = [(r.compression_ratio, r.rel_error, r) for r in self.results]
        pts.sort(key=lambda x: x[0])
        pareto = []
        best_err = float('inf')
        for ratio, err, r in pts:
            if err < best_err:
                pareto.append(r)
                best_err = err
        return pareto

    def recommend(self, target_ratio=None, target_snr=None):
        """Recommend best method for target ratio or SNR."""
        candidates = self.results
        if target_ratio:
            candidates = [r for r in candidates if r.compression_ratio <= target_ratio]
        if target_snr:
            candidates = [r for r in candidates if r.snr_db >= target_snr]
        if not candidates:
            return None
        # Best balance: highest SNR among lowest ratio
        return min(candidates, key=lambda r: r.compression_ratio / (r.snr_db + 1e-10))

    def summary_table(self, top_n=20):
        """Print a summary table."""
        sorted_results = sorted(self.results, key=lambda r: r.rel_error)[:top_n]
        lines = [f"{'Method':<30} {'Category':<22} {'Ratio':>8} {'SNR(dB)':>10} {'RelErr':>10} {'CosSim':>8} {'Time(ms)':>10}"]
        lines.append("-" * 110)
        for r in sorted_results:
            lines.append(f"{r.method_name:<30} {r.category:<22} {r.compression_ratio:>8.4f} {r.snr_db:>10.2f} {r.rel_error:>10.6f} {r.cosine_similarity:>8.4f} {r.time_ms:>10.1f}")
        return "\n".join(lines)

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

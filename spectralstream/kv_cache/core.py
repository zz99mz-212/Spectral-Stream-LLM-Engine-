from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-30

_E8_GENERATOR: np.ndarray = np.array(
    [
        [2, -1, -1, -1, -1, -1, -1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=np.float64,
)

_E8_VORONOI: np.ndarray = np.array(
    [
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


# FUTURE: E8 lattice quantization — implemented but not yet wired into CacheCompressor.
# These functions provide 8D lattice-based quantization for KV cache entries.
# To activate: add "e8_lattice" → _e8_quantize_batch mapping in compressor._METHOD_ALIASES.


def _e8_quantize_batch(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    orig_shape = vectors.shape
    flat = vectors.reshape(-1, 8)
    G = _E8_GENERATOR @ _E8_VORONOI
    Ginv = np.linalg.inv(G)
    x = flat @ Ginv.T
    x_round = np.round(x)
    x_floor = np.floor(x)
    x_ceil = np.ceil(x)
    d_round = np.sum((x - x_round) ** 2, axis=1)
    d_floor = np.sum((x - x_floor) ** 2, axis=1)
    d_ceil = np.sum((x - x_ceil) ** 2, axis=1)
    mask = np.stack(
        [
            d_round <= np.minimum(d_floor, d_ceil),
            (d_floor <= d_ceil) & (d_floor < d_round),
            (d_ceil < d_floor) & (d_ceil < d_round),
        ],
        axis=1,
    )
    best = np.where(mask[:, 0:1], x_round, np.where(mask[:, 1:2], x_floor, x_ceil))
    quantized = (best @ G.T).astype(np.float32)
    return quantized.reshape(orig_shape)


@dataclass
class QualityMetrics:
    mse: float = 0.0
    snr: float = 0.0
    psnr: float = 0.0
    relative_error: float = 0.0
    compression_ratio: float = 1.0
    method: str = "none"
    bits_per_element: float = 32.0
    entropy: float = 0.0
    timestamp: float = 0.0

    def score(self) -> float:
        return (
            math.log10(self.snr + 1.0) * 0.3
            + math.log10(self.psnr + 1.0) * 0.2
            - math.log10(self.mse + 1e-30) * 0.2
            + math.log10(self.compression_ratio + 1.0) * 0.2
            + self.entropy * 0.1
        )


@dataclass
class CacheMetrics:
    total_stored: int = 0
    total_bytes_orig: int = 0
    total_bytes_compressed: int = 0
    hit_rate: float = 0.0
    avg_compression_ratio: float = 1.0
    eviction_count: int = 0
    method_used: str = "none"
    per_layer_ratios: dict = field(default_factory=dict)


@dataclass
class KVCacheConfig:
    max_seq_len: int = 131072
    num_layers: int = 35
    num_heads: int = 1
    head_dim: int = 256
    hidden_size: int = 1536
    cache_dtype: str = "float16"
    compression_method: str = "none"
    quantize_bits: int = 8
    hadamard_rotate: bool = True
    spectral_keep_fraction: float = 0.5
    wavelet_name: str = "haar"
    wavelet_level: int = 3
    svd_rank: int = 32
    tensor_train_rank: int = 16
    vq_codebook_size: int = 256
    pq_subvectors: int = 8
    pq_subbits: int = 8
    residual_vq_stages: int = 3
    random_projection_dim: int = 64
    chebyshev_degree: int = 16
    e8_scaling: float = 1.0
    eviction_policy: str = "spectral"
    window_size: int = 4096
    heavy_hitter_frac: float = 0.1
    cache_size_limit_gb: float = 4.0
    use_holographic: bool = False
    use_predictive: bool = False
    use_resonance: bool = False
    use_kalman: bool = False
    use_siren: bool = False
    use_fourier_feature: bool = False
    enable_tiering: bool = True
    ssd_cache_path: str = "/tmp/kv_cache_ssd"
    auto_tune: bool = True
    adaptive_compression: bool = True
    progressive_compression: bool = True
    simulated_annealing_eviction: bool = False
    cache_coherence_monitoring: bool = False
    quality_tracking: bool = True
    prefetch_enabled: bool = True
    prefetch_window: int = 32
    annealing_temp: float = 1.0
    annealing_cooling_rate: float = 0.995
    annealing_min_temp: float = 0.01
    compression_pressure: float = 0.0
    hit_rate_window: int = 1000
    method_entropy_threshold: float = 3.0


@dataclass
class KVCacheEntry:
    key: np.ndarray
    value: np.ndarray
    position: int
    layer_idx: int
    score: float = 0.0
    compressed: bool = False
    checksum: int = 0
    quality: Optional[QualityMetrics] = None
    compressed_size: int = 0

    def __post_init__(self):
        self.checksum = hash(str(self.position) + str(self.layer_idx))

    def byte_size(self) -> int:
        if self.quality is not None and self.compressed_size > 0:
            return min(self.key.nbytes + self.value.nbytes, self.compressed_size)
        return self.key.nbytes + self.value.nbytes


# DEPRECATED: Many methods listed here are experimental stubs.
# Only block_int8/4, hadamard_int8/4, dct_sparse have real implementations.
class KVCacheMethod(IntEnum):
    NONE = 0
    HADAMARD_QUANTIZE = 1
    SPECTRAL_COMPRESS = 2
    QUANTILE_QUANTIZE = 3
    FWHT_INT8 = 4
    FWHT_INT4 = 5
    DCT_SPARSE = 6
    WAVELET_COMPRESS = 7
    SVD_COMPRESS = 8
    TENSOR_TRAIN_COMPRESS = 9
    VQ_COMPRESS = 10
    PRODUCT_QUANTIZATION = 11
    RESIDUAL_VQ = 12
    DELTA_ENCODING = 13
    PREDICTIVE_CODING = 14
    HOLOGRAPHIC_COMPRESS = 15
    VLASOV_COMPRESS = 16
    QUANTUM_STATE_COMPRESS = 17
    TIMECRYSTAL_COMPRESS = 18
    CONTEXT_ADAPTIVE = 19
    CROSS_LAYER_DELTA = 20
    HEAVY_HITTER_PROTECT = 21
    FREQUENCY_DOMAIN = 22
    RANDOM_PROJECTION = 23
    LOW_RANK_APPROX = 24
    LLOYD_MAX_COMPRESS = 25
    E8_LATTICE_COMPRESS = 26
    ADAPTIVE_BITWIDTH = 27
    SPARSE_ATTENTION_MAPS = 28
    ENTROPY_BASED_SELECTION = 29
    KALMAN_FILTER_PREDICT = 30
    SIREN_ENCODING = 31
    FOURIER_FEATURE = 32
    CHEBYSHEV_APPROX = 33


def _generate_random_hd_vector(dim: int, seed: Optional[int] = None) -> np.ndarray:
    from spectralstream.core.math_primitives import generate_random_hd_vector as _hd

    return _hd(dim, seed)

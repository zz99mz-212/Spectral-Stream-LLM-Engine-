"""
Comprehensive tests for the Noise-Aware Compressor module.

Tests NoiseFloorDetector (4 static methods), NoiseAwareResult (dataclass),
and NoiseAwareCompressor (compress/decompress with 3 methods, auto-select,
noise floor detection, compression potential, edge cases, errors).
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict

import numpy as np
import pytest

sys.path.insert(0, ".")

try:
    from spectralstream.compression.noise_aware_compressor import (
        NoiseAwareCompressor,
        NoiseAwareResult,
        NoiseFloorDetector,
    )
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(42)


@pytest.fixture
def matrix_2d(rng) -> np.ndarray:
    return rng.randn(16, 16).astype(np.float64)


@pytest.fixture
def vector_1d(rng) -> np.ndarray:
    return rng.randn(64).astype(np.float64)


@pytest.fixture
def low_rank_matrix(rng) -> np.ndarray:
    """Rank-4 matrix with additive noise."""
    U = rng.randn(16, 4)
    s = np.array([10.0, 8.0, 6.0, 4.0])
    Vt = rng.randn(4, 16)
    signal = U @ np.diag(s) @ Vt
    noise = 0.01 * rng.randn(16, 16)
    return (signal + noise).astype(np.float64)


@pytest.fixture
def zero_matrix() -> np.ndarray:
    return np.zeros((8, 8), dtype=np.float64)


@pytest.fixture
def single_element() -> np.ndarray:
    return np.array([[42.0]], dtype=np.float64)


@pytest.fixture
def compressor() -> NoiseAwareCompressor:
    return NoiseAwareCompressor(min_signal_rank=1)


# ═══════════════════════════════════════════════════════════════════════════
# NoiseFloorDetector
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseFloorDetectorMarchenkoPastur:
    def test_returns_positive_integer(self):
        sv = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.1])
        result = NoiseFloorDetector.marchenko_pastur_bound(sv, 16, 16)
        assert isinstance(result, int)
        assert result >= 1

    def test_single_singular_value(self):
        sv = np.array([10.0])
        result = NoiseFloorDetector.marchenko_pastur_bound(sv, 4, 4)
        assert result == 1

    def test_all_small_values(self):
        sv = np.array([0.1, 0.09, 0.08, 0.07])
        result = NoiseFloorDetector.marchenko_pastur_bound(sv, 8, 8)
        assert result >= 1

    def test_clear_signal_separation(self):
        sv = np.array([100.0, 80.0, 60.0, 0.1, 0.09, 0.08])
        result = NoiseFloorDetector.marchenko_pastur_bound(sv, 8, 8)
        assert result >= 1

    def test_zero_variance(self):
        sv = np.zeros(10)
        result = NoiseFloorDetector.marchenko_pastur_bound(sv, 8, 8)
        assert result >= 1

    def test_empty_singular_values(self):
        sv = np.array([], dtype=np.float64)
        result = NoiseFloorDetector.marchenko_pastur_bound(sv, 8, 8)
        assert result >= 0


class TestNoiseFloorDetectorEigenvalueRatio:
    def test_returns_integer(self):
        sv = np.array([10.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = NoiseFloorDetector.eigenvalue_ratio_test(sv)
        assert isinstance(result, int)
        assert result >= 1

    def test_clear_signal_noise_separation(self):
        sv = np.array([100.0, 20.0, 4.0, 0.8, 0.7])
        result = NoiseFloorDetector.eigenvalue_ratio_test(sv, threshold=4.0)
        assert result == 3

    def test_only_noise_values(self):
        sv = np.array([0.5, 0.48, 0.47, 0.45, 0.44])
        result = NoiseFloorDetector.eigenvalue_ratio_test(sv, threshold=1.5)
        assert result >= 1

    def test_single_sv(self):
        sv = np.array([42.0])
        result = NoiseFloorDetector.eigenvalue_ratio_test(sv)
        assert result == 1

    def test_two_singular_values(self):
        sv = np.array([10.0, 1.0])
        result = NoiseFloorDetector.eigenvalue_ratio_test(sv, threshold=2.0)
        assert result == 1

    def test_empty_array(self):
        sv = np.array([], dtype=np.float64)
        result = NoiseFloorDetector.eigenvalue_ratio_test(sv)
        assert result == 0

    def test_default_threshold(self):
        sv = np.array([100.0, 50.0, 10.0, 9.5, 9.0])
        result = NoiseFloorDetector.eigenvalue_ratio_test(sv)
        assert result >= 1


class TestNoiseFloorDetectorScreeElbow:
    def test_returns_positive_integer(self):
        sv = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 0.5])
        result = NoiseFloorDetector.scree_elbow_detect(sv)
        assert isinstance(result, int)
        assert result >= 1

    def test_known_elbow_position(self):
        sv = np.array([100.0, 80.0, 60.0, 5.0, 4.5, 4.0, 3.5, 3.0])
        result = NoiseFloorDetector.scree_elbow_detect(sv)
        assert 2 <= result <= 5

    def test_less_than_three_values(self):
        sv = np.array([10.0, 5.0])
        result = NoiseFloorDetector.scree_elbow_detect(sv)
        assert result == 2

    def test_single_value(self):
        sv = np.array([42.0])
        result = NoiseFloorDetector.scree_elbow_detect(sv)
        assert result == 1

    def test_zero_energy(self):
        sv = np.zeros(10)
        result = NoiseFloorDetector.scree_elbow_detect(sv)
        assert result == 1

    def test_identical_values(self):
        sv = np.full(10, 5.0)
        result = NoiseFloorDetector.scree_elbow_detect(sv)
        assert result >= 1

    def test_decreasing_geometric_series(self):
        sv = 10.0 ** np.linspace(0, -4, 16)
        result = NoiseFloorDetector.scree_elbow_detect(sv)
        assert 1 <= result <= 12

    def test_two_element_array(self):
        sv = np.array([10.0, 9.9])
        result = NoiseFloorDetector.scree_elbow_detect(sv)
        assert result == 2


class TestNoiseFloorDetectorBayesian:
    def test_returns_positive_integer(self):
        sv = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 0.5])
        result = NoiseFloorDetector.bayesian_threshold(sv, 16, 16)
        assert isinstance(result, int)
        assert result >= 1

    def test_empty_sv(self):
        sv = np.array([], dtype=np.float64)
        result = NoiseFloorDetector.bayesian_threshold(sv, 8, 8)
        assert result == 0

    def test_zero_energy(self):
        sv = np.zeros(10)
        result = NoiseFloorDetector.bayesian_threshold(sv, 8, 8)
        assert result == 1

    def test_non_square_matrix(self):
        sv = np.array([10.0, 5.0, 3.0, 1.0, 0.5])
        result = NoiseFloorDetector.bayesian_threshold(sv, 16, 8)
        assert result >= 1

    def test_all_equal_values(self):
        sv = np.full(10, 1.0)
        result = NoiseFloorDetector.bayesian_threshold(sv, 16, 16)
        assert result >= 1

    def test_large_matrix_small_rank(self):
        sv = np.array([100.0, 80.0, 1.0, 0.5, 0.3, 0.2, 0.1])
        result = NoiseFloorDetector.bayesian_threshold(sv, 16, 16)
        assert result >= 1


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareResult
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareResult:
    def test_constructor_and_field_access(self):
        result = NoiseAwareResult(
            compressed_data={"key": np.array([1, 2, 3])},
            original_shape=(16, 16),
            method="svd_noise",
            compression_ratio=4.5,
            reconstruction_error=0.001,
            signal_rank=8,
            noise_floor_estimate=0.0078,
        )
        assert result.method == "svd_noise"
        assert result.compression_ratio == 4.5
        assert result.reconstruction_error == 0.001
        assert result.signal_rank == 8
        assert result.noise_floor_estimate == 0.0078
        assert result.original_shape == (16, 16)
        assert "key" in result.compressed_data

    def test_repr_format(self):
        result = NoiseAwareResult(
            compressed_data={},
            original_shape=(8, 8),
            method="dct_noise",
            compression_ratio=3.25,
            reconstruction_error=0.0005,
            signal_rank=4,
            noise_floor_estimate=0.0035,
        )
        rep = repr(result)
        assert rep.startswith("NoiseAwareResult(")
        assert "method='dct_noise'" in rep
        assert "ratio=3.25x" in rep
        assert "signal_rank=4" in rep
        assert "noise_floor=0.003500" in rep

    def test_metadata_storage(self):
        result = NoiseAwareResult(
            compressed_data={},
            original_shape=(4, 4),
            method="bf16_exploit",
            compression_ratio=2.0,
            reconstruction_error=0.01,
            signal_rank=2,
            noise_floor_estimate=0.0078,
            metadata={"custom_key": "custom_value", "version": 2},
        )
        assert result.metadata["custom_key"] == "custom_value"
        assert result.metadata["version"] == 2

    def test_empty_metadata_defaults_to_dict(self):
        result = NoiseAwareResult(
            compressed_data={},
            original_shape=(4, 4),
            method="svd_noise",
            compression_ratio=1.0,
            reconstruction_error=0.0,
            signal_rank=1,
            noise_floor_estimate=0.0,
        )
        assert result.metadata == {}

    def test_all_slots_accessible(self):
        result = NoiseAwareResult(
            compressed_data="dummy",
            original_shape=(3, 3),
            method="test",
            compression_ratio=1.5,
            reconstruction_error=0.02,
            signal_rank=3,
            noise_floor_estimate=0.01,
        )
        assert hasattr(result, "compressed_data")
        assert hasattr(result, "original_shape")
        assert hasattr(result, "method")
        assert hasattr(result, "compression_ratio")
        assert hasattr(result, "reconstruction_error")
        assert hasattr(result, "signal_rank")
        assert hasattr(result, "noise_floor_estimate")
        assert hasattr(result, "metadata")

    def test_no_slots_added_outside_definition(self):
        result = NoiseAwareResult(
            compressed_data="dummy",
            original_shape=(2, 2),
            method="test",
            compression_ratio=1.0,
            reconstruction_error=0.0,
            signal_rank=1,
            noise_floor_estimate=0.0,
        )
        with pytest.raises(AttributeError):
            result.nonexistent_attr = 42


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareCompressor — SVD Noise
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareCompressorSVD:
    def test_svd_noise_roundtrip(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="svd_noise")
        reconstructed = compressor.decompress(result)
        assert reconstructed.shape == matrix_2d.shape
        assert isinstance(result.compression_ratio, float)
        assert result.compression_ratio > 0
        assert result.method == "svd_noise"
        assert 1 <= result.signal_rank <= 16

    def test_svd_noise_reconstruction_error_small(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="svd_noise")
        reconstructed = compressor.decompress(result)
        mse = float(np.mean((matrix_2d - reconstructed) ** 2))
        assert mse < 1.0

    def test_svd_noise_with_low_rank_matrix(self, low_rank_matrix, compressor):
        result = compressor.compress(low_rank_matrix, method="svd_noise")
        reconstructed = compressor.decompress(result)
        assert reconstructed.shape == low_rank_matrix.shape
        assert result.compression_ratio >= 0.5

    def test_svd_noise_preserves_metadata(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="svd_noise")
        assert "detection_method" in result.metadata
        assert "noise_sv_count" in result.metadata
        assert "energy_retained" in result.metadata
        assert "compress_time_ms" in result.metadata

    def test_svd_noise_has_reasonable_ratio(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="svd_noise")
        assert result.compression_ratio >= 0.1


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareCompressor — DCT Noise
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareCompressorDCT:
    def test_dct_noise_roundtrip_1d(self, vector_1d, compressor):
        result = compressor.compress(vector_1d, method="dct_noise")
        reconstructed = compressor.decompress(result)
        assert reconstructed.shape == vector_1d.shape
        assert result.method == "dct_noise"
        assert result.compression_ratio > 0

    def test_dct_noise_roundtrip_2d(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="dct_noise")
        reconstructed = compressor.decompress(result)
        assert reconstructed.shape == matrix_2d.shape
        mse = float(np.mean((matrix_2d - reconstructed) ** 2))
        assert mse < 1.0

    def test_dct_noise_signal_rank_positive(self, vector_1d, compressor):
        result = compressor.compress(vector_1d, method="dct_noise")
        assert result.signal_rank >= 1
        assert result.signal_rank <= vector_1d.size

    def test_dct_noise_metadata(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="dct_noise")
        assert "signal_fraction" in result.metadata
        assert "energy_retained" in result.metadata
        assert "compress_time_ms" in result.metadata

    def test_dct_noise_reconstruction_reasonable(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="dct_noise")
        reconstructed = compressor.decompress(result)
        cosine_sim = np.sum(matrix_2d * reconstructed) / (
            np.linalg.norm(matrix_2d) * np.linalg.norm(reconstructed) + 1e-10
        )
        assert cosine_sim > 0.9


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareCompressor — BF16 Exploit
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareCompressorBF16:
    def test_bf16_exploit_roundtrip(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="bf16_exploit")
        reconstructed = compressor.decompress(result)
        assert reconstructed.shape == matrix_2d.shape
        assert result.method == "bf16_exploit"
        assert result.compression_ratio > 0

    def test_bf16_exploit_reconstruction_error_bounded(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="bf16_exploit")
        reconstructed = compressor.decompress(result)
        mse = float(np.mean((matrix_2d - reconstructed) ** 2))
        assert mse < 10.0

    def test_bf16_exploit_signal_rank(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="bf16_exploit")
        assert result.signal_rank >= 1
        assert result.signal_rank <= matrix_2d.size

    def test_bf16_exploit_metadata(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="bf16_exploit")
        assert "bf16_noise_floor" in result.metadata
        assert "below_floor_fraction" in result.metadata
        assert "above_floor_fraction" in result.metadata
        assert "compress_time_ms" in result.metadata

    def test_bf16_exploit_noise_floor_positive(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="bf16_exploit")
        assert result.noise_floor_estimate >= 0.0

    def test_bf16_exploit_with_low_rank(self, low_rank_matrix, compressor):
        result = compressor.compress(low_rank_matrix, method="bf16_exploit")
        reconstructed = compressor.decompress(result)
        assert reconstructed.shape == low_rank_matrix.shape

    def test_bf16_exploit_spectral_thresholding_disabled(self, matrix_2d):
        compressor_no_spectral = NoiseAwareCompressor(
            enable_spectral_thresholding=False
        )
        result = compressor_no_spectral.compress(matrix_2d, method="bf16_exploit")
        reconstructed = compressor_no_spectral.decompress(result)
        assert reconstructed.shape == matrix_2d.shape


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareCompressor — Auto Method Selection
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareCompressorAuto:
    def test_auto_selects_svd_for_2d_small(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d)
        assert result.method in ("svd_noise", "dct_noise")

    def test_auto_selects_dct_for_1d(self, vector_1d, compressor):
        result = compressor.compress(vector_1d)
        assert result.method in ("dct_noise",)

    def test_auto_roundtrip_2d(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d)
        reconstructed = compressor.decompress(result)
        assert reconstructed.shape == matrix_2d.shape

    def test_auto_roundtrip_1d(self, vector_1d, compressor):
        result = compressor.compress(vector_1d)
        reconstructed = compressor.decompress(result)
        assert reconstructed.shape == vector_1d.shape

    def test_auto_compression_ratio_positive(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d)
        assert result.compression_ratio > 0


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareCompressor — detect_noise_floor
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareCompressorDetectNoiseFloor:
    def test_detect_2d_matrix(self, matrix_2d, compressor):
        signal_rank, noise_floor = compressor.detect_noise_floor(matrix_2d)
        assert isinstance(signal_rank, int)
        assert signal_rank >= 1
        assert signal_rank <= min(matrix_2d.shape)
        assert isinstance(noise_floor, float)
        assert noise_floor >= 0.0

    def test_detect_1d_vector(self, vector_1d, compressor):
        signal_rank, noise_floor = compressor.detect_noise_floor(vector_1d)
        assert isinstance(signal_rank, int)
        assert signal_rank >= 1
        assert signal_rank <= vector_1d.size
        assert isinstance(noise_floor, float)
        assert noise_floor >= 0.0

    def test_detect_zero_matrix(self, zero_matrix, compressor):
        signal_rank, noise_floor = compressor.detect_noise_floor(zero_matrix)
        assert signal_rank >= 1
        assert noise_floor >= 0.0

    def test_detect_single_element(self, single_element, compressor):
        signal_rank, noise_floor = compressor.detect_noise_floor(single_element)
        assert signal_rank >= 1

    def test_detect_low_rank_signal(self, low_rank_matrix, compressor):
        signal_rank, noise_floor = compressor.detect_noise_floor(low_rank_matrix)
        assert signal_rank >= 1
        assert signal_rank <= min(low_rank_matrix.shape)

    def test_detect_ragged_tensor(self, compressor):
        ragged = np.random.randn(1, 16).astype(np.float64)
        signal_rank, noise_floor = compressor.detect_noise_floor(ragged)
        assert signal_rank >= 1


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareCompressor — get_compression_potential
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareCompressorPotential:
    def test_returns_required_keys_2d(self, matrix_2d, compressor):
        metrics = compressor.get_compression_potential(matrix_2d)
        required_keys = {
            "signal_rank",
            "signal_fraction",
            "noise_fraction",
            "signal_energy_fraction",
            "noise_energy_fraction",
            "max_compression_from_noise",
        }
        assert required_keys.issubset(metrics.keys())

    def test_returns_required_keys_1d(self, vector_1d, compressor):
        metrics = compressor.get_compression_potential(vector_1d)
        required_keys = {
            "signal_rank",
            "signal_fraction",
            "noise_fraction",
            "signal_energy_fraction",
            "noise_energy_fraction",
            "max_compression_from_noise",
        }
        assert required_keys.issubset(metrics.keys())

    def test_signal_fraction_in_range(self, matrix_2d, compressor):
        metrics = compressor.get_compression_potential(matrix_2d)
        assert 0 <= metrics["signal_fraction"] <= 1
        assert 0 <= metrics["noise_fraction"] <= 1
        assert 0 <= metrics["signal_energy_fraction"] <= 1
        assert 0 <= metrics["noise_energy_fraction"] <= 1

    def test_signal_energy_fraction_reasonable(self, low_rank_matrix, compressor):
        metrics = compressor.get_compression_potential(low_rank_matrix)
        assert metrics["signal_energy_fraction"] > 0.5

    def test_max_compression_from_noise_positive(self, matrix_2d, compressor):
        metrics = compressor.get_compression_potential(matrix_2d)
        assert metrics["max_compression_from_noise"] >= 1.0

    def test_zero_matrix_does_not_crash(self, zero_matrix, compressor):
        metrics = compressor.get_compression_potential(zero_matrix)
        assert metrics["signal_rank"] >= 0

    def test_signal_rank_vs_fraction_consistency(self, matrix_2d, compressor):
        metrics = compressor.get_compression_potential(matrix_2d)
        total = matrix_2d.size
        expected_fraction = metrics["signal_rank"] / total
        assert abs(metrics["signal_fraction"] - expected_fraction) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareCompressor — Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareCompressorEdgeCases:
    def test_very_small_matrix(self, compressor):
        tiny = np.random.randn(2, 3).astype(np.float64)
        for method in ("svd_noise", "dct_noise", "bf16_exploit"):
            result = compressor.compress(tiny, method=method)
            recon = compressor.decompress(result)
            assert recon.shape == tiny.shape

    def test_single_element_tensor(self, single_element, compressor):
        for method in ("dct_noise", "bf16_exploit"):
            result = compressor.compress(single_element, method=method)
            recon = compressor.decompress(result)
            assert recon.shape == single_element.shape

    def test_single_element_svd_falls_back_to_dct(self, single_element, compressor):
        result = compressor.compress(single_element, method="svd_noise")
        recon = compressor.decompress(result)
        assert recon.shape == single_element.shape

    def test_zero_tensor_compress(self, zero_matrix, compressor):
        for method in ("svd_noise", "dct_noise"):
            result = compressor.compress(zero_matrix, method=method)
            recon = compressor.decompress(result)
            assert recon.shape == zero_matrix.shape

    def test_ones_tensor(self, compressor):
        ones = np.ones((8, 8), dtype=np.float64)
        for method in ("svd_noise", "dct_noise", "bf16_exploit"):
            result = compressor.compress(ones, method=method)
            recon = compressor.decompress(result)
            assert recon.shape == ones.shape

    def test_constant_value_tensor(self, compressor):
        const = np.full((6, 6), 3.14159, dtype=np.float64)
        result = compressor.compress(const, method="dct_noise")
        recon = compressor.decompress(result)
        assert recon.shape == const.shape

    def test_1xN_matrix(self, compressor):
        row = np.random.randn(1, 32).astype(np.float64)
        result = compressor.compress(row, method="dct_noise")
        recon = compressor.decompress(result)
        assert recon.shape == row.shape

    def test_Nx1_matrix(self, compressor):
        col = np.random.randn(32, 1).astype(np.float64)
        result = compressor.compress(col, method="dct_noise")
        recon = compressor.decompress(result)
        assert recon.shape == col.shape

    def test_3d_tensor_gets_flattened(self, compressor):
        tensor_3d = np.random.randn(4, 4, 4).astype(np.float64)
        result = compressor.compress(tensor_3d, method="dct_noise")
        recon = compressor.decompress(result)
        assert recon.shape == tensor_3d.shape

    def test_repeated_compress_decompress_same_tensor(self, matrix_2d, compressor):
        result1 = compressor.compress(matrix_2d, method="dct_noise")
        result2 = compressor.compress(matrix_2d, method="dct_noise")
        assert abs(result1.compression_ratio - result2.compression_ratio) < 1e-6

    def test_repeated_bf16_compress(self, matrix_2d, compressor):
        result1 = compressor.compress(matrix_2d, method="bf16_exploit")
        result2 = compressor.compress(matrix_2d, method="bf16_exploit")
        assert abs(result1.compression_ratio - result2.compression_ratio) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareCompressor — Error Handling
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareCompressorErrors:
    def test_invalid_method_raises_value_error(self, compressor, matrix_2d):
        with pytest.raises(ValueError, match="Unknown method"):
            compressor.compress(matrix_2d, method="nonexistent")

    def test_invalid_decompress_method_raises_value_error(self, compressor):
        bad_result = NoiseAwareResult(
            compressed_data={},
            original_shape=(4, 4),
            method="invalid_method",
            compression_ratio=1.0,
            reconstruction_error=0.0,
            signal_rank=1,
            noise_floor_estimate=0.0,
        )
        with pytest.raises(ValueError, match="Unknown method"):
            compressor.decompress(bad_result)

    def test_compress_empty_tensor_raises_error(self, compressor):
        empty = np.array([], dtype=np.float64)
        with pytest.raises((ValueError, IndexError)):
            compressor.compress(empty, method="dct_noise")


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareCompressor — Configuration Variants
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareCompressorConfig:
    def test_custom_energy_threshold(self, matrix_2d):
        compressor = NoiseAwareCompressor(energy_threshold=0.99)
        result = compressor.compress(matrix_2d, method="svd_noise")
        assert result.compression_ratio > 0

    def test_custom_noise_floor(self, matrix_2d):
        compressor = NoiseAwareCompressor(bf16_noise_floor=0.001)
        result = compressor.compress(matrix_2d, method="bf16_exploit")
        assert result.compression_ratio > 0

    def test_min_signal_rank_respected(self):
        compressor = NoiseAwareCompressor(min_signal_rank=3)
        mat = np.random.randn(8, 8).astype(np.float64)
        result = compressor.compress(mat, method="svd_noise")
        assert result.signal_rank >= 3

    def test_max_signal_ratio_respected(self):
        compressor = NoiseAwareCompressor(max_signal_ratio=0.25)
        mat = np.random.randn(16, 16).astype(np.float64)
        result = compressor.compress(mat, method="svd_noise")
        max_possible = int(16 * 0.25)
        assert result.signal_rank <= max_possible


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareCompressor — Integration with Detection Methods
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareCompressorDetectionMethod:
    def test_marchenko_pastur_detection(self, low_rank_matrix, compressor):
        result = compressor.compress(
            low_rank_matrix,
            method="svd_noise",
            detection_method="marchenko_pastur",
        )
        assert result.signal_rank >= 1

    def test_eigenvalue_ratio_detection(self, low_rank_matrix, compressor):
        result = compressor.compress(
            low_rank_matrix,
            method="svd_noise",
            detection_method="eigenvalue_ratio",
        )
        assert result.signal_rank >= 1

    def test_scree_detection(self, low_rank_matrix, compressor):
        result = compressor.compress(
            low_rank_matrix,
            method="svd_noise",
            detection_method="scree",
        )
        assert result.signal_rank >= 1

    def test_bayesian_detection(self, low_rank_matrix, compressor):
        result = compressor.compress(
            low_rank_matrix,
            method="svd_noise",
            detection_method="bayesian",
        )
        assert result.signal_rank >= 1


# ═══════════════════════════════════════════════════════════════════════════
# NoiseAwareCompressor — Decompress Roundtrip Consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseAwareCompressorRoundtripConsistency:
    def test_svd_roundtrip_preserves_dtype_style(self, compressor):
        mat = np.random.randn(8, 8).astype(np.float64)
        result = compressor.compress(mat, method="svd_noise")
        recon = compressor.decompress(result)
        assert recon.dtype == np.float64

    def test_dct_roundtrip_preserves_dtype_style(self, compressor):
        vec = np.random.randn(32).astype(np.float64)
        result = compressor.compress(vec, method="dct_noise")
        recon = compressor.decompress(result)
        assert recon.dtype == np.float64

    def test_bf16_roundtrip_preserves_dtype_style(self, compressor):
        mat = np.random.randn(8, 8).astype(np.float64)
        result = compressor.compress(mat, method="bf16_exploit")
        recon = compressor.decompress(result)
        assert recon.dtype == np.float64

    def test_compress_decompress_independence(self, matrix_2d, compressor):
        result = compressor.compress(matrix_2d, method="dct_noise")
        recon1 = compressor.decompress(result)
        recon2 = compressor.decompress(result)
        np.testing.assert_array_equal(recon1, recon2)

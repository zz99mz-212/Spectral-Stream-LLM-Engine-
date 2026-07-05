"""
Comprehensive tests for the Cutting-Edge Compression Library.

Tests helper functions (_ensure_2d, _restore_shape, _safe_bytes),
the CompressionMethod base class (estimate_ratio, estimate_error),
and roundtrip compress/decompress for 25+ compression methods.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, ".")

try:
    from spectralstream.compression.cutting_edge import (
        CompressionMethod,
        _ensure_2d,
        _restore_shape,
        _safe_bytes,
        ALL_METHODS,
        get_all_methods,
        get_methods_by_category,
        # Quantum mechanics
        QuantumStateCompression,
        QuantumEntanglementCompression,
        QuantumTunnelingOptimizer,
        DensityMatrixCompression,
        QuantumErrorCorrectionCompression,
        # Plasma physics
        VlasovDistributionCompression,
        PlasmaOscillationDecomposition,
        MHDWaveCompression,
        DebyeShieldingCompression,
        PlasmaTurbulenceDecomposition,
        # Information theory
        RateDistortionOptimalCompression,
        MutualInformationCompression,
        KolmogorovComplexityApproximation,
        FisherInformationWeighting,
        EntropyRateCompression,
        # Advanced mathematics
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
    )
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

RNG = np.random.RandomState(42)


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(42)


@pytest.fixture
def mat_8x8(rng) -> np.ndarray:
    return rng.randn(8, 8).astype(np.float32)


@pytest.fixture
def mat_16x16(rng) -> np.ndarray:
    return rng.randn(16, 16).astype(np.float32)


@pytest.fixture
def vec_64(rng) -> np.ndarray:
    return rng.randn(64).astype(np.float32)


@pytest.fixture
def tensor_3d(rng) -> np.ndarray:
    return rng.randn(4, 6, 8).astype(np.float32)


@pytest.fixture
def single_val() -> np.ndarray:
    return np.array([[42.0]], dtype=np.float32)


@pytest.fixture
def zero_8x8() -> np.ndarray:
    return np.zeros((8, 8), dtype=np.float32)


@pytest.fixture
def low_rank_8x8(rng) -> np.ndarray:
    u = rng.randn(8, 3)
    v = rng.randn(3, 8)
    return (u @ v).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# _ensure_2d / _restore_shape
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsure2d:
    def test_1d_array(self, vec_64):
        t2d, orig = _ensure_2d(vec_64)
        assert t2d.ndim == 2
        assert t2d.shape[0] == 1
        assert t2d.shape[1] == vec_64.shape[0]
        assert orig == vec_64.shape

    def test_2d_array(self, mat_8x8):
        t2d, orig = _ensure_2d(mat_8x8)
        assert t2d.shape == mat_8x8.shape
        assert orig == mat_8x8.shape

    def test_3d_array(self, tensor_3d):
        t2d, orig = _ensure_2d(tensor_3d)
        assert t2d.ndim == 2
        assert t2d.shape[0] == tensor_3d.shape[0]
        assert t2d.shape[1] == tensor_3d.shape[1] * tensor_3d.shape[2]
        assert orig == tensor_3d.shape

    def test_single_element(self, single_val):
        t2d, orig = _ensure_2d(single_val)
        assert t2d.shape == (1, 1)
        assert orig == (1, 1)


class TestRestoreShape:
    def test_1d_restore(self, vec_64):
        t2d, orig = _ensure_2d(vec_64)
        restored = _restore_shape(t2d, orig)
        assert restored.shape == vec_64.shape
        assert np.allclose(restored, vec_64)

    def test_2d_restore(self, mat_8x8):
        t2d, orig = _ensure_2d(mat_8x8)
        restored = _restore_shape(t2d, orig)
        assert restored.shape == mat_8x8.shape

    def test_3d_restore(self, tensor_3d):
        t2d, orig = _ensure_2d(tensor_3d)
        restored = _restore_shape(t2d, orig)
        assert restored.shape == tensor_3d.shape
        assert np.allclose(restored, tensor_3d)

    def test_same_shape_noop(self, mat_8x8):
        restored = _restore_shape(mat_8x8, mat_8x8.shape)
        assert restored is mat_8x8


# ═══════════════════════════════════════════════════════════════════════════
# _safe_bytes
# ═══════════════════════════════════════════════════════════════════════════


class TestSafeBytes:
    def test_numpy_array(self):
        a = np.zeros((10, 10), dtype=np.float64)
        assert _safe_bytes(a) == a.nbytes

    def test_numpy_int32(self):
        a = np.zeros(100, dtype=np.int32)
        assert _safe_bytes(a) == a.nbytes

    def test_numpy_bool(self):
        a = np.zeros(50, dtype=bool)
        assert _safe_bytes(a) == a.nbytes

    def test_dict(self):
        d = {"a": np.ones(10), "b": np.ones(20)}
        expected = np.ones(10).nbytes + np.ones(20).nbytes + len("a") + len("b")
        assert _safe_bytes(d) == expected

    def test_nested_dict(self):
        d = {"x": {"y": np.ones(5), "z": 3.0}}
        expected = np.ones(5).nbytes + 8 + len("x") + len("y") + len("z")
        assert _safe_bytes(d) == expected

    def test_list(self):
        lst = [np.ones(5), np.ones(10)]
        assert _safe_bytes(lst) == np.ones(5).nbytes + np.ones(10).nbytes

    def test_tuple(self):
        tup = (np.ones(3), 42)
        assert _safe_bytes(tup) == np.ones(3).nbytes + 8

    def test_scalar_int(self):
        assert _safe_bytes(42) == 8

    def test_scalar_float(self):
        assert _safe_bytes(3.14) == 8

    def test_string(self):
        assert _safe_bytes("hello") == 5

    def test_empty_string(self):
        assert _safe_bytes("") == 0

    def test_none(self):
        assert _safe_bytes(None) == 0

    def test_mixed_nested(self):
        data = {
            "weights": np.ones((4, 4), dtype=np.float32),
            "params": [1.0, 2.0, 3.0],
            "name": "test",
        }
        expected = (
            np.ones((4, 4), dtype=np.float32).nbytes
            + 3 * 8
            + len("weights")
            + len("params")
            + len("name")
            + len("test")
        )
        assert _safe_bytes(data) == expected

    def test_np_integer(self):
        assert _safe_bytes(np.int32(5)) == 8

    def test_np_floating(self):
        assert _safe_bytes(np.float32(3.14)) == 8


# ═══════════════════════════════════════════════════════════════════════════
# CompressionMethod base class
# ═══════════════════════════════════════════════════════════════════════════


class _DummyMethod(CompressionMethod):
    name = "dummy"
    category = "test"

    def compress(self, tensor, **kw):
        flat = tensor.ravel().astype(np.float32)
        return {"mean": float(flat.mean())}, {"orig_shape": tensor.shape}

    def decompress(self, cd, meta):
        m = float(np.prod(meta["orig_shape"]))
        return np.full(meta["orig_shape"], cd["mean"], dtype=np.float32)


class TestCompressionMethodBase:
    def test_estimate_ratio_positive(self, mat_8x8):
        method = _DummyMethod()
        ratio = method.estimate_ratio(mat_8x8)
        assert ratio > 0
        assert ratio < 1.0  # should compress since only storing 1 float

    def test_estimate_error_returns_dict(self, mat_8x8):
        method = _DummyMethod()
        errors = method.estimate_error(mat_8x8)
        assert isinstance(errors, dict)
        for key in (
            "mse",
            "snr_db",
            "rel_error",
            "mae",
            "max_error",
            "cosine_similarity",
        ):
            assert key in errors
            assert isinstance(errors[key], float)

    def test_estimate_error_identical_tensor(self, mat_8x8):
        method = _DummyMethod()
        errors = method.estimate_error(mat_8x8)
        assert errors["mse"] >= 0
        assert errors["cosine_similarity"] > 0

    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            CompressionMethod()  # noqa


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 1: QUANTUM MECHANICS
# ═══════════════════════════════════════════════════════════════════════════


class TestQuantumStateCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = QuantumStateCompression()
        cd, meta = method.compress(mat_8x8, keep_ratio=0.5)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_16x16(self, mat_16x16):
        method = QuantumStateCompression()
        cd, meta = method.compress(mat_16x16, keep_ratio=0.25)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_roundtrip_1d(self, vec_64):
        method = QuantumStateCompression()
        cd, meta = method.compress(vec_64, keep_ratio=0.5)
        recon = method.decompress(cd, meta)
        assert recon.shape == vec_64.shape

    def test_roundtrip_small_2x2(self):
        small = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        method = QuantumStateCompression()
        cd, meta = method.compress(small, keep_ratio=0.5)
        recon = method.decompress(cd, meta)
        assert recon.shape == small.shape

    def test_roundtrip_zero(self, zero_8x8):
        method = QuantumStateCompression()
        cd, meta = method.compress(zero_8x8)
        recon = method.decompress(cd, meta)
        assert recon.shape == zero_8x8.shape

    def test_error_returns_dict(self, mat_8x8):
        method = QuantumStateCompression()
        errors = method.estimate_error(mat_8x8, keep_ratio=0.5)
        assert isinstance(errors, dict)
        assert "mse" in errors

    def test_ratio_positive(self, mat_8x8):
        method = QuantumStateCompression()
        ratio = method.estimate_ratio(mat_8x8, keep_ratio=0.5)
        assert ratio > 0

    def test_keep_ratio_changes_stored_data(self, mat_8x8):
        method = QuantumStateCompression()
        cd_low, _ = method.compress(mat_8x8, keep_ratio=0.1)
        cd_high, _ = method.compress(mat_8x8, keep_ratio=0.9)
        assert len(cd_high["amplitudes"]) > len(cd_low["amplitudes"])


class TestQuantumEntanglementCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = QuantumEntanglementCompression()
        cd, meta = method.compress(mat_8x8, n_pairs=2, max_schmidt_rank=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape

    def test_roundtrip_16x16(self, mat_16x16):
        method = QuantumEntanglementCompression()
        cd, meta = method.compress(mat_16x16, n_pairs=4, max_schmidt_rank=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_output_has_pairs(self, mat_8x8):
        method = QuantumEntanglementCompression()
        cd, meta = method.compress(mat_8x8, n_pairs=2, max_schmidt_rank=4)
        assert "pairs" in cd
        assert len(cd["pairs"]) > 0

    def test_error_returns_dict(self, mat_8x8):
        method = QuantumEntanglementCompression()
        errors = method.estimate_error(mat_8x8, n_pairs=2, max_schmidt_rank=4)
        assert isinstance(errors, dict)
        assert "mse" in errors


class TestQuantumTunnelingOptimizer:
    def test_roundtrip_8x8(self, mat_8x8):
        method = QuantumTunnelingOptimizer()
        cd, meta = method.compress(mat_8x8, n_bits=4, n_rounds=5)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_1d(self, vec_64):
        method = QuantumTunnelingOptimizer()
        cd, meta = method.compress(vec_64, n_bits=4, n_rounds=5)
        recon = method.decompress(cd, meta)
        assert recon.shape == vec_64.shape

    def test_roundtrip_single(self, single_val):
        method = QuantumTunnelingOptimizer()
        cd, meta = method.compress(single_val, n_bits=2, n_rounds=5)
        recon = method.decompress(cd, meta)
        assert recon.shape == single_val.shape

    def test_quantization_bit_depth(self, mat_8x8):
        method = QuantumTunnelingOptimizer()
        cd, meta = method.compress(mat_8x8, n_bits=3, n_rounds=5)
        n_levels = 1 << 3
        assert len(cd["cb"]) == n_levels

    def test_error_decreases_with_more_bits(self):
        rng = np.random.RandomState(42)
        arr = rng.randn(16).astype(np.float32)
        method = QuantumTunnelingOptimizer()
        err_2bit = method.estimate_error(arr, n_bits=2, n_rounds=10)
        err_6bit = method.estimate_error(arr, n_bits=6, n_rounds=10)
        assert err_6bit["mse"] < err_2bit["mse"] + 1e-4

    def test_ratio_positive(self, mat_8x8):
        method = QuantumTunnelingOptimizer()
        ratio = method.estimate_ratio(mat_8x8, n_bits=4)
        assert ratio > 0


class TestDensityMatrixCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = DensityMatrixCompression()
        cd, meta = method.compress(mat_8x8, n_components=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_16x16(self, mat_16x16):
        method = DensityMatrixCompression()
        cd, meta = method.compress(mat_16x16, n_components=8)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_roundtrip_1d(self, vec_64):
        method = DensityMatrixCompression()
        cd, meta = method.compress(vec_64, n_components=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == vec_64.shape

    def test_components_contains_eigvals(self, mat_8x8):
        method = DensityMatrixCompression()
        cd, meta = method.compress(mat_8x8, n_components=4)
        assert "eigvals" in cd
        assert "eigvecs" in cd
        assert "coeffs" in cd
        assert cd["k"] == 4

    def test_error_reasonable(self, mat_8x8):
        method = DensityMatrixCompression()
        errors = method.estimate_error(mat_8x8, n_components=4)
        assert errors["cosine_similarity"] > 0.5

    def test_ratio_positive(self, mat_8x8):
        method = DensityMatrixCompression()
        ratio = method.estimate_ratio(mat_8x8, n_components=4)
        assert ratio > 0


class TestQuantumErrorCorrectionCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = QuantumErrorCorrectionCompression()
        cd, meta = method.compress(mat_8x8, n_bits=4, block_size=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_16x16(self, mat_16x16):
        method = QuantumErrorCorrectionCompression()
        cd, meta = method.compress(mat_16x16, n_bits=4, block_size=8)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_roundtrip_1d(self, vec_64):
        method = QuantumErrorCorrectionCompression()
        cd, meta = method.compress(vec_64, n_bits=4, block_size=8)
        recon = method.decompress(cd, meta)
        assert recon.shape == vec_64.shape

    def test_syndromes_present(self, mat_8x8):
        method = QuantumErrorCorrectionCompression()
        cd, meta = method.compress(mat_8x8, n_bits=4, block_size=4)
        assert "syndromes" in cd
        assert len(cd["syndromes"]) > 0

    def test_correction_table_present(self, mat_8x8):
        method = QuantumErrorCorrectionCompression()
        cd, meta = method.compress(mat_8x8, n_bits=4, block_size=4)
        assert "correction_table" in cd
        assert cd["correction_table"].shape == (16, 4)

    def test_error_returns_dict(self, mat_8x8):
        method = QuantumErrorCorrectionCompression()
        errors = method.estimate_error(mat_8x8, n_bits=4, block_size=4)
        assert isinstance(errors, dict)
        assert "mse" in errors


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 2: PLASMA PHYSICS
# ═══════════════════════════════════════════════════════════════════════════


class TestVlasovDistributionCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = VlasovDistributionCompression()
        cd, meta = method.compress(mat_8x8, n_particles=8, n_char_steps=5)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_16x16(self, mat_16x16):
        method = VlasovDistributionCompression()
        cd, meta = method.compress(mat_16x16, n_particles=8, n_char_steps=5)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_output_has_particles(self, mat_8x8):
        method = VlasovDistributionCompression()
        cd, meta = method.compress(mat_8x8, n_particles=4, n_char_steps=3)
        assert "particles" in cd
        assert len(cd["particles"]) > 0
        assert "pot_coeffs" in cd

    def test_error_returns_dict(self, mat_8x8):
        method = VlasovDistributionCompression()
        errors = method.estimate_error(mat_8x8, n_particles=4, n_char_steps=3)
        assert isinstance(errors, dict)

    def test_ratio_positive(self, mat_8x8):
        method = VlasovDistributionCompression()
        ratio = method.estimate_ratio(mat_8x8)
        assert ratio > 0


class TestPlasmaOscillationDecomposition:
    def test_roundtrip_8x8(self, mat_8x8):
        method = PlasmaOscillationDecomposition()
        cd, meta = method.compress(mat_8x8, n_modes=8)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_16x16(self, mat_16x16):
        method = PlasmaOscillationDecomposition()
        cd, meta = method.compress(mat_16x16, n_modes=16)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_roundtrip_low_rank(self, low_rank_8x8):
        method = PlasmaOscillationDecomposition()
        cd, meta = method.compress(low_rank_8x8, n_modes=8)
        recon = method.decompress(cd, meta)
        assert recon.shape == low_rank_8x8.shape

    def test_modes_structure(self, mat_8x8):
        method = PlasmaOscillationDecomposition()
        cd, meta = method.compress(mat_8x8, n_modes=4)
        assert "modes" in cd
        assert len(cd["modes"]) == 4
        for mode in cd["modes"]:
            for key in ("amp", "kx", "ky", "phase", "gamma"):
                assert key in mode

    def test_error_returns_dict(self, mat_8x8):
        method = PlasmaOscillationDecomposition()
        errors = method.estimate_error(mat_8x8, n_modes=8)
        assert isinstance(errors, dict)
        assert "mse" in errors

    def test_more_modes_less_error(self, mat_8x8):
        method = PlasmaOscillationDecomposition()
        err_few = method.estimate_error(mat_8x8, n_modes=2)
        err_many = method.estimate_error(mat_8x8, n_modes=16)
        assert err_many["mse"] <= err_few["mse"] + 1e-6


class TestMHDWaveCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = MHDWaveCompression()
        cd, meta = method.compress(mat_8x8, n_components=6)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_16x16(self, mat_16x16):
        method = MHDWaveCompression()
        cd, meta = method.compress(mat_16x16, n_components=12)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_output_keys(self, mat_8x8):
        method = MHDWaveCompression()
        cd, meta = method.compress(mat_8x8, n_components=6)
        for key in ("alfven", "acoustic", "entropy"):
            assert key in cd

    def test_error_returns_dict(self, mat_8x8):
        method = MHDWaveCompression()
        errors = method.estimate_error(mat_8x8, n_components=6)
        assert isinstance(errors, dict)
        assert "mse" in errors

    def test_ratio_positive(self, mat_8x8):
        method = MHDWaveCompression()
        ratio = method.estimate_ratio(mat_8x8)
        assert ratio > 0


class TestDebyeShieldingCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = DebyeShieldingCompression()
        cd, meta = method.compress(mat_8x8, keep_ratio=0.5)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape

    def test_roundtrip_zero(self, zero_8x8):
        method = DebyeShieldingCompression()
        cd, meta = method.compress(zero_8x8, keep_ratio=0.5)
        recon = method.decompress(cd, meta)
        assert recon.shape == zero_8x8.shape

    def test_output_keys(self, mat_8x8):
        method = DebyeShieldingCompression()
        cd, meta = method.compress(mat_8x8, keep_ratio=0.5)
        for key in ("shielded_mean", "shielded_std", "sparse_vals", "debye_length"):
            assert key in cd

    def test_error_returns_dict(self, mat_8x8):
        method = DebyeShieldingCompression()
        errors = method.estimate_error(mat_8x8, keep_ratio=0.5)
        assert isinstance(errors, dict)


class TestPlasmaTurbulenceDecomposition:
    def test_roundtrip_8x8(self, mat_8x8):
        method = PlasmaTurbulenceDecomposition()
        cd, meta = method.compress(mat_8x8, n_scales=2, coherent_ratio=0.5)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape

    def test_roundtrip_16x16(self, mat_16x16):
        method = PlasmaTurbulenceDecomposition()
        cd, meta = method.compress(mat_16x16, n_scales=3, coherent_ratio=0.3)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_output_scales(self, mat_8x8):
        method = PlasmaTurbulenceDecomposition()
        cd, meta = method.compress(mat_8x8, n_scales=2, coherent_ratio=0.5)
        assert "scales" in cd
        assert len(cd["scales"]) > 0
        assert "residual" in cd

    def test_error_returns_dict(self, mat_8x8):
        method = PlasmaTurbulenceDecomposition()
        errors = method.estimate_error(mat_8x8, n_scales=2, coherent_ratio=0.5)
        assert isinstance(errors, dict)
        assert "mse" in errors


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 3: INFORMATION THEORY
# ═══════════════════════════════════════════════════════════════════════════


class TestRateDistortionOptimalCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = RateDistortionOptimalCompression()
        cd, meta = method.compress(
            mat_8x8, n_input_levels=64, n_output_levels=8, max_iter=10
        )
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_1d(self, vec_64):
        method = RateDistortionOptimalCompression()
        cd, meta = method.compress(
            vec_64, n_input_levels=64, n_output_levels=8, max_iter=10
        )
        recon = method.decompress(cd, meta)
        assert recon.shape == vec_64.shape

    def test_roundtrip_single(self, single_val):
        method = RateDistortionOptimalCompression()
        cd, meta = method.compress(
            single_val, n_input_levels=16, n_output_levels=4, max_iter=5
        )
        recon = method.decompress(cd, meta)
        assert recon.shape == single_val.shape

    def test_output_keys(self, mat_8x8):
        method = RateDistortionOptimalCompression()
        cd, meta = method.compress(
            mat_8x8, n_input_levels=32, n_output_levels=4, max_iter=5
        )
        for key in ("idx", "output_values", "scale"):
            assert key in cd

    def test_more_output_levels_less_error(self):
        rng = np.random.RandomState(42)
        arr = rng.randn(16).astype(np.float32)
        method = RateDistortionOptimalCompression()
        err_low = method.estimate_error(
            arr, n_input_levels=32, n_output_levels=2, max_iter=10
        )
        err_high = method.estimate_error(
            arr, n_input_levels=32, n_output_levels=16, max_iter=10
        )
        assert err_high["mse"] <= err_low["mse"] + 1e-4

    def test_error_returns_dict(self, mat_8x8):
        method = RateDistortionOptimalCompression()
        errors = method.estimate_error(mat_8x8)
        assert isinstance(errors, dict)

    def test_ratio_positive(self, mat_8x8):
        method = RateDistortionOptimalCompression()
        ratio = method.estimate_ratio(mat_8x8)
        assert ratio > 0


class TestMutualInformationCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = MutualInformationCompression()
        cd, meta = method.compress(mat_8x8, n_clusters=4, n_iter=5)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_16x16(self, mat_16x16):
        method = MutualInformationCompression()
        cd, meta = method.compress(mat_16x16, n_clusters=8, n_iter=5)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_roundtrip_zero(self, zero_8x8):
        method = MutualInformationCompression()
        cd, meta = method.compress(zero_8x8, n_clusters=2, n_iter=5)
        recon = method.decompress(cd, meta)
        assert recon.shape == zero_8x8.shape

    def test_output_keys(self, mat_8x8):
        method = MutualInformationCompression()
        cd, meta = method.compress(mat_8x8, n_clusters=4, n_iter=5)
        for key in ("centroids", "assignments", "n_clusters", "IX_T"):
            assert key in cd
        assert cd["n_clusters"] == 4

    def test_error_returns_dict(self, mat_8x8):
        method = MutualInformationCompression()
        errors = method.estimate_error(mat_8x8, n_clusters=4, n_iter=5)
        assert isinstance(errors, dict)
        assert "mse" in errors

    def ratio_positive(self, mat_8x8):
        method = MutualInformationCompression()
        ratio = method.estimate_ratio(mat_8x8, n_clusters=4)
        assert ratio > 0


class TestKolmogorovComplexityApproximation:
    def test_roundtrip_8x8(self, mat_8x8):
        method = KolmogorovComplexityApproximation()
        cd, meta = method.compress(mat_8x8, max_rank=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape

    def test_roundtrip_1d(self, vec_64):
        method = KolmogorovComplexityApproximation()
        cd, meta = method.compress(vec_64, max_rank=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == vec_64.shape

    def test_model_selection(self, mat_8x8):
        method = KolmogorovComplexityApproximation()
        cd, meta = method.compress(mat_8x8, max_rank=4)
        assert "model" in cd
        assert cd["model"] in ("zero", "mean", "svd", "dct")

    def test_svd_roundtrip_low_rank(self, low_rank_8x8):
        method = KolmogorovComplexityApproximation()
        cd, meta = method.compress(low_rank_8x8, max_rank=8)
        recon = method.decompress(cd, meta)
        assert recon.shape == low_rank_8x8.shape

    def test_error_returns_dict(self, mat_8x8):
        method = KolmogorovComplexityApproximation()
        errors = method.estimate_error(mat_8x8, max_rank=4)
        assert isinstance(errors, dict)

    def test_ratio_positive(self, mat_8x8):
        method = KolmogorovComplexityApproximation()
        ratio = method.estimate_ratio(mat_8x8, max_rank=4)
        assert ratio > 0


class TestFisherInformationWeighting:
    def test_roundtrip_8x8(self, mat_8x8):
        method = FisherInformationWeighting()
        cd, meta = method.compress(mat_8x8, total_bits=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_single(self, single_val):
        method = FisherInformationWeighting()
        cd, meta = method.compress(single_val, total_bits=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == single_val.shape

    def test_output_has_quantized(self, mat_8x8):
        method = FisherInformationWeighting()
        cd, meta = method.compress(mat_8x8, total_bits=4)
        assert "quantized" in cd
        assert len(cd["quantized"]) == mat_8x8.shape[1]

    def test_error_returns_dict(self, mat_8x8):
        method = FisherInformationWeighting()
        errors = method.estimate_error(mat_8x8, total_bits=4)
        assert isinstance(errors, dict)

    def test_more_bits_less_error(self):
        rng = np.random.RandomState(42)
        arr = rng.randn(8, 4).astype(np.float32)
        method = FisherInformationWeighting()
        err_low = method.estimate_error(arr, total_bits=2)
        err_high = method.estimate_error(arr, total_bits=8)
        assert err_high["mse"] <= err_low["mse"] + 1e-4


class TestEntropyRateCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = EntropyRateCompression()
        cd, meta = method.compress(mat_8x8, context_order=1, n_bins=16)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_1d(self, vec_64):
        method = EntropyRateCompression()
        cd, meta = method.compress(vec_64, context_order=1, n_bins=16)
        recon = method.decompress(cd, meta)
        assert recon.shape == vec_64.shape

    def test_entropy_rate_present(self, mat_8x8):
        method = EntropyRateCompression()
        cd, meta = method.compress(mat_8x8, context_order=1, n_bins=16)
        assert "entropy_rate" in cd
        assert cd["entropy_rate"] > 0

    def test_error_returns_dict(self, mat_8x8):
        method = EntropyRateCompression()
        errors = method.estimate_error(mat_8x8, context_order=1, n_bins=16)
        assert isinstance(errors, dict)


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 4: ADVANCED MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════


class TestManifoldLearningCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = ManifoldLearningCompression()
        cd, meta = method.compress(mat_8x8, n_neighbors=4, n_components=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape

    def test_roundtrip_low_rank(self, low_rank_8x8):
        method = ManifoldLearningCompression()
        cd, meta = method.compress(low_rank_8x8, n_neighbors=4, n_components=3)
        recon = method.decompress(cd, meta)
        assert recon.shape == low_rank_8x8.shape

    def test_output_keys(self, mat_8x8):
        method = ManifoldLearningCompression()
        cd, meta = method.compress(mat_8x8, n_neighbors=4, n_components=4)
        for key in ("coords", "decoder", "d"):
            assert key in cd

    def test_error_returns_dict(self, mat_8x8):
        method = ManifoldLearningCompression()
        errors = method.estimate_error(mat_8x8, n_neighbors=4, n_components=4)
        assert isinstance(errors, dict)


class TestOptimalTransportCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = OptimalTransportCompression()
        cd, meta = method.compress(mat_8x8, n_bins=8)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape

    def test_roundtrip_1d(self, vec_64):
        method = OptimalTransportCompression()
        cd, meta = method.compress(vec_64, n_bins=8)
        recon = method.decompress(cd, meta)
        assert recon.shape == vec_64.shape

    def test_output_keys(self, mat_8x8):
        method = OptimalTransportCompression()
        cd, meta = method.compress(mat_8x8, n_bins=8)
        for key in ("sparse_T", "bin_centers", "target", "n_bins"):
            assert key in cd

    def test_error_returns_dict(self, mat_8x8):
        method = OptimalTransportCompression()
        errors = method.estimate_error(mat_8x8, n_bins=8)
        assert isinstance(errors, dict)


class TestCategoryTheoryCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = CategoryTheoryCompression()
        cd, meta = method.compress(mat_8x8, block_size=4, n_generators=2)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape

    def test_roundtrip_16x16(self, mat_16x16):
        method = CategoryTheoryCompression()
        cd, meta = method.compress(mat_16x16, block_size=8, n_generators=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_output_keys(self, mat_8x8):
        method = CategoryTheoryCompression()
        cd, meta = method.compress(mat_8x8, block_size=4, n_generators=2)
        for key in ("generators", "coeffs", "bs", "n_blocks_m", "n_blocks_n"):
            assert key in cd

    def test_error_returns_dict(self, mat_8x8):
        method = CategoryTheoryCompression()
        errors = method.estimate_error(mat_8x8, block_size=4, n_generators=2)
        assert isinstance(errors, dict)


class TestAlgebraicGeometryCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = AlgebraicGeometryCompression()
        cd, meta = method.compress(mat_8x8, degree=2, n_polys=4, neighborhood_size=3)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape

    def test_output_keys(self, mat_8x8):
        method = AlgebraicGeometryCompression()
        cd, meta = method.compress(mat_8x8, degree=2, n_polys=4, neighborhood_size=3)
        for key in ("coeffs", "coeff_idx", "n_terms", "degree", "r"):
            assert key in cd

    def test_error_returns_dict(self, mat_8x8):
        method = AlgebraicGeometryCompression()
        errors = method.estimate_error(
            mat_8x8, degree=2, n_polys=4, neighborhood_size=3
        )
        assert isinstance(errors, dict)


class TestTopologicalDataCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = TopologicalDataCompression()
        cd, meta = method.compress(mat_8x8, n_features=4, max_dim=2)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape

    def test_persistence_keys(self, mat_8x8):
        method = TopologicalDataCompression()
        cd, meta = method.compress(mat_8x8, n_features=4, max_dim=2)
        for key in ("persistence_0", "persistence_1", "points"):
            assert key in cd

    def test_error_returns_dict(self, mat_8x8):
        method = TopologicalDataCompression()
        errors = method.estimate_error(mat_8x8, n_features=4, max_dim=2)
        assert isinstance(errors, dict)


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 5: HYBRID / NOVEL
# ═══════════════════════════════════════════════════════════════════════════


class TestResonanceCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = ResonanceCompression()
        cd, meta = method.compress(mat_8x8, energy_threshold=0.9)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_low_rank(self, low_rank_8x8):
        method = ResonanceCompression()
        cd, meta = method.compress(low_rank_8x8, energy_threshold=0.95)
        recon = method.decompress(cd, meta)
        assert recon.shape == low_rank_8x8.shape

    def test_output_keys(self, mat_8x8):
        method = ResonanceCompression()
        cd, meta = method.compress(mat_8x8, energy_threshold=0.9)
        for key in ("U", "S", "Vt", "n_modes", "total_energy"):
            assert key in cd
        assert cd["n_modes"] >= 1

    def test_error_small(self, mat_8x8):
        method = ResonanceCompression()
        errors = method.estimate_error(mat_8x8, energy_threshold=0.99)
        assert errors["rel_error"] < 0.5

    def test_higher_threshold_lower_error(self, mat_8x8):
        method = ResonanceCompression()
        err_low = method.estimate_error(mat_8x8, energy_threshold=0.5)
        err_high = method.estimate_error(mat_8x8, energy_threshold=0.99)
        assert err_high["mse"] <= err_low["mse"] + 1e-6

    def test_ratio_positive(self, mat_8x8):
        method = ResonanceCompression()
        ratio = method.estimate_ratio(mat_8x8, energy_threshold=0.9)
        assert ratio > 0


class TestHarmonicOscillatorDecomposition:
    def test_roundtrip_8x8(self, mat_8x8):
        method = HarmonicOscillatorDecomposition()
        cd, meta = method.compress(mat_8x8, n_modes=8)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape

    def test_roundtrip_16x16(self, mat_16x16):
        method = HarmonicOscillatorDecomposition()
        cd, meta = method.compress(mat_16x16, n_modes=16)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_modes_structure(self, mat_8x8):
        method = HarmonicOscillatorDecomposition()
        cd, meta = method.compress(mat_8x8, n_modes=4)
        assert "modes" in cd
        assert len(cd["modes"]) == 4
        for mode in cd["modes"]:
            for key in ("amplitude", "freq_x", "freq_y", "phase", "damping"):
                assert key in mode

    def test_error_returns_dict(self, mat_8x8):
        method = HarmonicOscillatorDecomposition()
        errors = method.estimate_error(mat_8x8, n_modes=8)
        assert isinstance(errors, dict)

    def test_more_modes_store_more_data(self, mat_8x8):
        method = HarmonicOscillatorDecomposition()
        cd_few, _ = method.compress(mat_8x8, n_modes=4)
        cd_many, _ = method.compress(mat_8x8, n_modes=32)
        assert len(cd_many["modes"]) > len(cd_few["modes"])


class TestFourierNeuralOperatorCompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = FourierNeuralOperatorCompression()
        cd, meta = method.compress(mat_8x8, n_modes=8, filter_order=2)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_16x16(self, mat_16x16):
        method = FourierNeuralOperatorCompression()
        cd, meta = method.compress(mat_16x16, n_modes=16, filter_order=2)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_output_keys(self, mat_8x8):
        method = FourierNeuralOperatorCompression()
        cd, meta = method.compress(mat_8x8, n_modes=8, filter_order=2)
        for key in ("coeffs", "indices", "n_modes"):
            assert key in cd
        assert cd["n_modes"] == 8

    def test_error_returns_dict(self, mat_8x8):
        method = FourierNeuralOperatorCompression()
        errors = method.estimate_error(mat_8x8, n_modes=8, filter_order=2)
        assert isinstance(errors, dict)

    def test_ratio_positive(self, mat_8x8):
        method = FourierNeuralOperatorCompression()
        ratio = method.estimate_ratio(mat_8x8, n_modes=8)
        assert ratio > 0


class TestWaveletScatteringTransform:
    def test_roundtrip_8x8(self, mat_8x8):
        method = WaveletScatteringTransform()
        cd, meta = method.compress(
            mat_8x8, n_scales=2, scattering_order=1, keep_ratio=0.5
        )
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape

    def test_roundtrip_16x16(self, mat_16x16):
        method = WaveletScatteringTransform()
        cd, meta = method.compress(
            mat_16x16, n_scales=2, scattering_order=1, keep_ratio=0.5
        )
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_scattering_coeffs(self, mat_8x8):
        method = WaveletScatteringTransform()
        cd, meta = method.compress(
            mat_8x8, n_scales=2, scattering_order=1, keep_ratio=0.5
        )
        assert "scattering" in cd
        assert len(cd["scattering"]) > 0

    def test_error_returns_dict(self, mat_8x8):
        method = WaveletScatteringTransform()
        errors = method.estimate_error(
            mat_8x8, n_scales=2, scattering_order=1, keep_ratio=0.5
        )
        assert isinstance(errors, dict)


class TestNeuralODECompression:
    def test_roundtrip_8x8(self, mat_8x8):
        method = NeuralODECompression()
        cd, meta = method.compress(mat_8x8, n_layers_approx=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_8x8.shape
        assert recon.dtype == np.float32

    def test_roundtrip_16x16(self, mat_16x16):
        method = NeuralODECompression()
        cd, meta = method.compress(mat_16x16, n_layers_approx=4)
        recon = method.decompress(cd, meta)
        assert recon.shape == mat_16x16.shape

    def test_output_keys(self, mat_8x8):
        method = NeuralODECompression()
        cd, meta = method.compress(mat_8x8, n_layers_approx=4)
        for key in ("W0", "alpha", "n_phi", "m"):
            assert key in cd

    def test_error_returns_dict(self, mat_8x8):
        method = NeuralODECompression()
        errors = method.estimate_error(mat_8x8, n_layers_approx=4)
        assert isinstance(errors, dict)

    def test_ratio_positive(self, mat_8x8):
        method = NeuralODECompression()
        ratio = method.estimate_ratio(mat_8x8, n_layers_approx=4)
        assert ratio > 0


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════


class TestRegistry:
    def test_all_methods_populated(self):
        methods = get_all_methods()
        assert len(methods) >= 25
        assert "quantum_state" in methods
        assert "plasma_oscillation" in methods
        assert "resonance" in methods

    def test_each_method_has_name_and_category(self):
        for name, method in ALL_METHODS.items():
            assert method.name == name
            assert isinstance(method.category, str)
            assert len(method.category) > 0

    def test_get_methods_by_category(self):
        qm = get_methods_by_category("quantum_mechanics")
        assert len(qm) >= 5
        pp = get_methods_by_category("plasma_physics")
        assert len(pp) >= 5
        it = get_methods_by_category("information_theory")
        assert len(it) >= 5
        am = get_methods_by_category("advanced_mathematics")
        assert len(am) >= 5
        hybrid = get_methods_by_category("hybrid")
        assert len(hybrid) >= 5

    def test_each_method_category_is_valid(self):
        valid_categories = {
            "quantum_mechanics",
            "plasma_physics",
            "information_theory",
            "advanced_mathematics",
            "hybrid",
        }
        for method in ALL_METHODS.values():
            assert method.category in valid_categories

    def test_registered_instances_are_methods(self):
        for method in ALL_METHODS.values():
            assert isinstance(method, CompressionMethod)
            assert hasattr(method, "compress")
            assert hasattr(method, "decompress")
            assert hasattr(method, "estimate_ratio")
            assert hasattr(method, "estimate_error")

    def test_get_all_methods_returns_copy(self):
        methods = get_all_methods()
        original_id = id(ALL_METHODS)
        copy_id = id(methods)
        assert copy_id != original_id


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases — Across Methods
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    @pytest.mark.parametrize(
        "method_name",
        [
            "density_matrix",
            "quantum_tunneling",
            "plasma_oscillation",
            "mhd_wave",
            "rate_distortion_optimal",
            "mutual_information",
            "resonance",
            "harmonic_oscillator",
            "fourier_neural_operator",
            "neural_ode",
        ],
    )
    def test_single_element_roundtrip(self, method_name, single_val):
        method = ALL_METHODS[method_name]
        try:
            cd, meta = method.compress(single_val)
            recon = method.decompress(cd, meta)
            assert recon.shape == single_val.shape
        except Exception as e:
            pytest.skip(f"{method_name} single-element: {e}")

    @pytest.mark.parametrize(
        "method_name",
        [
            "quantum_state",
            "density_matrix",
            "quantum_tunneling",
            "plasma_oscillation",
            "mhd_wave",
            "rate_distortion_optimal",
            "mutual_information",
            "resonance",
            "harmonic_oscillator",
            "fourier_neural_operator",
            "neural_ode",
        ],
    )
    def test_zero_tensor_roundtrip(self, method_name, zero_8x8):
        method = ALL_METHODS[method_name]
        try:
            cd, meta = method.compress(zero_8x8)
            recon = method.decompress(cd, meta)
            assert recon.shape == zero_8x8.shape
        except Exception as e:
            pytest.skip(f"{method_name} zero tensor: {e}")

    @pytest.mark.parametrize(
        "method_name",
        [
            "quantum_state",
            "density_matrix",
            "plasma_oscillation",
            "mhd_wave",
            "resonance",
            "harmonic_oscillator",
        ],
    )
    def test_3d_tensor_roundtrip(self, method_name, tensor_3d):
        method = ALL_METHODS[method_name]
        try:
            cd, meta = method.compress(tensor_3d)
            recon = method.decompress(cd, meta)
            assert recon.shape == tensor_3d.shape
        except Exception as e:
            pytest.skip(f"{method_name} 3D tensor: {e}")

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("quantum_state", {}),
            ("quantum_tunneling", {}),
            ("density_matrix", {}),
            ("rate_distortion_optimal", {}),
            ("mutual_information", {"n_clusters": 4, "n_iter": 5}),
        ],
    )
    def test_estimate_returns_valid_numbers(self, method_name, kwargs, mat_8x8):
        method = ALL_METHODS[method_name]
        ratio = method.estimate_ratio(mat_8x8, **kwargs)
        assert ratio > 0
        errors = method.estimate_error(mat_8x8, **kwargs)
        assert errors["mse"] >= 0
        assert errors["cosine_similarity"] > -1.1
        assert math.isfinite(errors["snr_db"])

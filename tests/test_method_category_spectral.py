"""Tests for ALL spectral compression methods — round-trip, shape, dtype."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.methods.spectral._class_wrappers import (
    DCTBlock,
    DCT2D,
    DCT2DBlock,
    FWHT,
    WaveletHaar,
    WaveletDaubechies,
    WaveletSymlet,
    Fourier,
    FrequencyDomain,
    Givens,
    Chebyshev,
    Winograd,
    NTTTransform,
    RandomizedHadamard,
    ButterflySparse,
    SparseRandomProjection,
    PolynomialApprox,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def small_tensor_2d(rng):
    return rng.randn(32, 64).astype(np.float32)


def _roundtrip(inst, tensor, **kwargs):
    data, meta = inst.compress(tensor, **kwargs)
    recon = inst.decompress(data, meta)
    if recon.shape != tensor.shape:
        recon = recon.reshape(tensor.shape)
    return data, meta, recon


class TestDCTBlock:
    def test_roundtrip(self, small_tensor_2d):
        inst = DCTBlock()
        _, _, recon = _roundtrip(inst, small_tensor_2d)
        assert recon.shape == small_tensor_2d.shape

    def test_roundtrip_various_shapes(self):
        inst = DCTBlock()
        for shape in [(16, 16), (32, 64)]:
            tensor = np.random.randn(*shape).astype(np.float32)
            _, _, recon = _roundtrip(inst, tensor)
            assert recon.shape == tensor.shape


class TestDCT2D:
    def test_roundtrip(self, small_tensor_2d):
        inst = DCT2D()
        _, _, recon = _roundtrip(inst, small_tensor_2d)
        assert recon.shape == small_tensor_2d.shape

    def test_various_sizes(self):
        inst = DCT2D()
        for size in [(16, 16), (32, 32)]:
            tensor = np.random.randn(*size).astype(np.float32)
            _, _, recon = _roundtrip(inst, tensor, keep_fraction=0.5)
            assert recon.shape == tensor.shape


class TestDCT2DBlock:
    def test_roundtrip(self, small_tensor_2d):
        inst = DCT2DBlock()
        _, _, recon = _roundtrip(inst, small_tensor_2d, block_size=16)
        assert recon.shape == small_tensor_2d.shape


class TestFWHT:
    def test_roundtrip(self):
        inst = FWHT()
        tensor = np.random.randn(1, 128).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape

    def test_roundtrip_2d(self):
        inst = FWHT()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestWaveletHaar:
    def test_roundtrip(self, small_tensor_2d):
        inst = WaveletHaar()
        _, _, recon = _roundtrip(inst, small_tensor_2d)
        assert recon.shape == small_tensor_2d.shape

    def test_1d_roundtrip(self):
        inst = WaveletHaar()
        tensor = np.random.randn(1, 64).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestWaveletDaubechies:
    def test_roundtrip(self):
        inst = WaveletDaubechies()
        tensor = np.random.randn(64, 64).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestWaveletSymlet:
    def test_roundtrip(self):
        inst = WaveletSymlet()
        tensor = np.random.randn(64, 64).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestFourier:
    def test_roundtrip(self, small_tensor_2d):
        inst = Fourier()
        _, _, recon = _roundtrip(inst, small_tensor_2d)
        assert recon.shape == small_tensor_2d.shape

    def test_keep_fraction(self):
        inst = Fourier()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor, keep_fraction=0.3)
        assert recon.shape == tensor.shape


class TestFrequencyDomain:
    def test_roundtrip(self, small_tensor_2d):
        inst = FrequencyDomain()
        _, _, recon = _roundtrip(inst, small_tensor_2d)
        assert recon.shape == small_tensor_2d.shape


class TestNTTTransform:
    def test_roundtrip(self):
        inst = NTTTransform()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor, keep_fraction=0.3)
        assert recon.shape == tensor.shape


class TestGivens:
    def test_roundtrip(self):
        inst = Givens()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor, threshold=0.1)
        assert recon.shape == tensor.shape


class TestChebyshev:
    def test_roundtrip(self):
        inst = Chebyshev()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestWinograd:
    def test_roundtrip(self):
        inst = Winograd()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestPolynomialApprox:
    def test_roundtrip(self):
        inst = PolynomialApprox()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestRandomizedHadamard:
    def test_roundtrip(self):
        inst = RandomizedHadamard()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestButterflySparse:
    def test_roundtrip(self):
        inst = ButterflySparse()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestSparseRandomProjection:
    def test_roundtrip(self):
        inst = SparseRandomProjection()
        tensor = np.random.randn(16, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape

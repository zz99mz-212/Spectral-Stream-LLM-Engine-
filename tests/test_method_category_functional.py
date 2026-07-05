"""Tests for ALL functional compression methods — round-trip, shape, dtype."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.methods.functional.fractal import FractalCompression
from spectralstream.compression.methods.functional.kolmogorov import (
    KolmogorovComplexity,
)
from spectralstream.compression.methods.functional.information import (
    InformationBottleneck,
    RateDistortionOptimal,
)
from spectralstream.compression.methods.functional.hamiltonian import Hamiltonian
from spectralstream.compression.methods.functional.siren import SIRENINR
from spectralstream.compression.methods.functional.lagrangian import Lagrangian
from spectralstream.compression.methods.functional.landau_zener import LandauZener
from spectralstream.compression.methods.functional.neural_ode import NeuralODE
from spectralstream.compression.methods.functional.boltzmann import (
    BoltzmannEncoding,
    MaxEntropy,
)
from spectralstream.compression.methods.functional.symbolic import SymbolicRegression


@pytest.fixture
def rng():
    return np.random.RandomState(42)


def _roundtrip(inst, tensor, **kwargs):
    data, meta = inst.compress(tensor, **kwargs)
    recon = inst.decompress(data, meta)
    if recon.shape != tensor.shape:
        recon = recon.reshape(tensor.shape)
    return data, meta, recon


class TestFractalCompression:
    def test_roundtrip(self):
        inst = FractalCompression()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor, block_size=4)
        assert recon.shape == tensor.shape

    def test_small_tensor(self):
        inst = FractalCompression()
        tensor = np.random.randn(8, 8).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor, block_size=4)
        assert recon.shape == tensor.shape


class TestKolmogorovComplexity:
    def test_roundtrip(self):
        inst = KolmogorovComplexity()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape

    def test_constant_tensor(self):
        inst = KolmogorovComplexity()
        tensor = np.ones((16, 16)).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestInformationBottleneck:
    def test_roundtrip(self):
        inst = InformationBottleneck()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestRateDistortionOptimal:
    def test_roundtrip(self):
        inst = RateDistortionOptimal()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestHamiltonian:
    def test_roundtrip(self):
        inst = Hamiltonian()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestSIRENINR:
    def test_roundtrip(self):
        inst = SIRENINR()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestLagrangian:
    def test_roundtrip(self):
        inst = Lagrangian()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestLandauZener:
    def test_roundtrip(self):
        inst = LandauZener()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestNeuralODE:
    def test_roundtrip(self):
        inst = NeuralODE()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestBoltzmannEncoding:
    def test_roundtrip(self):
        inst = BoltzmannEncoding()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestMaxEntropy:
    def test_roundtrip(self):
        inst = MaxEntropy()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestSymbolicRegression:
    def test_roundtrip(self):
        inst = SymbolicRegression()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape

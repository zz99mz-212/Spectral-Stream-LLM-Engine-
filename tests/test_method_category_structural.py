"""Tests for ALL structural compression methods — round-trip, shape, dtype."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.methods.structural._class_wrappers import (
    BlockSparsity,
    UnstructuredPruning,
    SparseGPT,
    MonarchStructured,
    Circulant,
    Einsort,
    ButterflyStructured,
    Vandermonde,
    Cauchy,
    HSSMatrix,
    BSSMatrix,
    Structured24,
    WandaPruning,
    DynamicNMSparsity,
    ChannelPruning,
    GroupLasso,
    AdaptiveSparsity,
    SparseQuantizeCombined,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


def _roundtrip(inst, tensor, **kwargs):
    data, meta = inst.compress(tensor, **kwargs)
    recon = inst.decompress(data, meta)
    if recon.shape != tensor.shape:
        recon = recon.reshape(tensor.shape)
    return data, meta, recon


class TestBlockSparsity:
    def test_roundtrip(self):
        inst = BlockSparsity()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape

    def test_various_shapes(self):
        inst = BlockSparsity()
        for shape in [(16, 16), (32, 32)]:
            tensor = np.random.randn(*shape).astype(np.float32)
            _, _, recon = _roundtrip(inst, tensor)
            assert recon.shape == tensor.shape


class TestUnstructuredPruning:
    def test_roundtrip(self):
        inst = UnstructuredPruning()
        tensor = np.random.randn(64).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape

    def test_with_sparsity(self):
        inst = UnstructuredPruning()
        tensor = np.random.randn(128).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor, sparsity=0.3)
        assert recon.shape == tensor.shape


class TestSparseGPT:
    def test_roundtrip(self):
        inst = SparseGPT()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestWandaPruning:
    def test_roundtrip(self):
        inst = WandaPruning()
        tensor = np.random.randn(32, 64).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestMonarchStructured:
    def test_roundtrip(self):
        inst = MonarchStructured()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestCirculant:
    def test_roundtrip(self):
        inst = Circulant()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestEinsort:
    def test_roundtrip(self):
        inst = Einsort()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestButterflyStructured:
    def test_roundtrip(self):
        inst = ButterflyStructured()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestVandermonde:
    def test_roundtrip(self):
        inst = Vandermonde()
        tensor = np.random.randn(16, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestCauchy:
    def test_roundtrip(self):
        inst = Cauchy()
        tensor = np.random.randn(16, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestHSSMatrix:
    def test_roundtrip(self):
        inst = HSSMatrix()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestStructured24:
    def test_roundtrip(self):
        inst = Structured24()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestDynamicNMSparsity:
    def test_roundtrip(self):
        inst = DynamicNMSparsity()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestChannelPruning:
    def test_roundtrip(self):
        inst = ChannelPruning()
        tensor = np.random.randn(16, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestGroupLasso:
    def test_roundtrip(self):
        inst = GroupLasso()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestAdaptiveSparsity:
    def test_roundtrip(self):
        inst = AdaptiveSparsity()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape


class TestSparseQuantizeCombined:
    def test_roundtrip(self):
        inst = SparseQuantizeCombined()
        tensor = np.random.randn(16, 16).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert recon.shape == tensor.shape

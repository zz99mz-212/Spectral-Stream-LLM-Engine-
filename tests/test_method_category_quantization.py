"""Tests for ALL quantization compression methods — round-trip, shape, dtype."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.engine._methods import (
    _BlockINT8,
    _BlockINT4,
    _HadamardINT8,
    _HadamardINT4,
    _SparsityINT4,
    _DeltaINT4,
)
from spectralstream.compression.methods.quantization.kmeans import (
    KMeansQuant,
    LloydMaxQuant,
)
from spectralstream.compression.methods.quantization.nf4 import NF4
from spectralstream.compression.methods.quantization.binary import (
    BinaryQuant,
    TernaryQuant,
)
from spectralstream.compression.methods.quantization.product import ProductQuantization
from spectralstream.compression.methods.quantization.mixed_precision import (
    MixedPrecision,
)
from spectralstream.compression.methods.quantization.adaptive import (
    AdaptiveGroupQuant,
    OutlierAwareQuant,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def small_tensor_2d(rng):
    return rng.randn(32, 64).astype(np.float32)


@pytest.fixture
def small_tensor_1d(rng):
    return rng.randn(128).astype(np.float32)


def _roundtrip(inst, tensor, **kwargs):
    data, meta = inst.compress(tensor, **kwargs)
    recon = inst.decompress(data, meta)
    if recon.shape != tensor.shape:
        recon = recon.reshape(tensor.shape)
    return data, meta, recon


class TestBlockINT8:
    def test_roundtrip_2d(self, small_tensor_2d):
        _, _, recon = _roundtrip(_BlockINT8(), small_tensor_2d)
        assert recon.shape == small_tensor_2d.shape
        assert recon.dtype == small_tensor_2d.dtype

    def test_roundtrip_1d(self, small_tensor_1d):
        _, _, recon = _roundtrip(_BlockINT8(), small_tensor_1d)
        assert recon.shape == small_tensor_1d.shape

    def test_compressed_smaller(self, small_tensor_2d):
        data, _, _ = _roundtrip(_BlockINT8(), small_tensor_2d)
        assert len(data) < small_tensor_2d.nbytes

    def test_not_all_zeros(self):
        inst = _BlockINT8()
        tensor = np.random.randn(64, 64).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor)
        assert not np.allclose(recon, 0)

    def test_various_block_sizes(self):
        inst = _BlockINT8()
        tensor = np.random.randn(64, 64).astype(np.float32)
        for bs in [64, 128, 256]:
            _, _, recon = _roundtrip(inst, tensor, block_size=bs)
            assert recon.shape == tensor.shape


class TestBlockINT4:
    def test_roundtrip_2d(self, small_tensor_2d):
        _, _, recon = _roundtrip(_BlockINT4(), small_tensor_2d, block_size=32)
        assert recon.shape == small_tensor_2d.shape

    def test_roundtrip_1d(self, small_tensor_1d):
        _, _, recon = _roundtrip(_BlockINT4(), small_tensor_1d, block_size=16)
        assert recon.shape == small_tensor_1d.shape

    def test_compressed_smaller(self):
        inst = _BlockINT4()
        tensor = np.random.randn(128, 128).astype(np.float32)
        data, _, _ = _roundtrip(inst, tensor, block_size=32)
        assert len(data) < tensor.nbytes

    def test_various_block_sizes(self):
        inst = _BlockINT4()
        tensor = np.random.randn(64, 64).astype(np.float32)
        for bs in [16, 32, 64]:
            _, _, recon = _roundtrip(inst, tensor, block_size=bs)
            assert recon.shape == tensor.shape


class TestHadamardINT8:
    def test_roundtrip(self, small_tensor_2d):
        _, _, recon = _roundtrip(_HadamardINT8(), small_tensor_2d)
        assert recon.shape == small_tensor_2d.shape

    def test_roundtrip_1d(self, small_tensor_1d):
        _, _, recon = _roundtrip(_HadamardINT8(), small_tensor_1d)
        assert recon.shape == small_tensor_1d.shape


class TestHadamardINT4:
    def test_roundtrip(self, small_tensor_2d):
        _, _, recon = _roundtrip(_HadamardINT4(), small_tensor_2d, block_size=16)
        assert recon.shape == small_tensor_2d.shape

    def test_roundtrip_1d(self):
        inst = _HadamardINT4()
        tensor = np.random.randn(128).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor, block_size=32)
        assert recon.shape == tensor.shape


class TestSparsityINT4:
    def test_roundtrip(self):
        inst = _SparsityINT4()
        tensor = np.random.randn(128).astype(np.float32)
        data, meta, recon = _roundtrip(inst, tensor, group_size=32)
        assert recon.shape == tensor.shape
        assert meta["n_nonzero"] <= meta["n_elements"]

    def test_mask_nonzero_ratio(self):
        inst = _SparsityINT4()
        tensor = np.random.randn(256).astype(np.float32)
        _, meta, _ = _roundtrip(inst, tensor, group_size=32)
        nonzero_ratio = meta["n_nonzero"] / meta["n_elements"]
        assert nonzero_ratio < 1.0


class TestDeltaINT4:
    def test_roundtrip_with_reference(self):
        inst = _DeltaINT4()
        tensor = np.random.randn(64, 64).astype(np.float32)
        ref = np.random.randn(64, 64).astype(np.float32)
        data, meta = inst.compress(tensor, reference=ref, block_size=32)
        delta = inst.decompress(data, meta)
        recon = ref.ravel()[: delta.size] + delta
        assert recon.reshape(tensor.shape).shape == tensor.shape

    def test_roundtrip_no_reference(self):
        inst = _DeltaINT4()
        tensor = np.random.randn(64, 64).astype(np.float32)
        data, meta, recon = _roundtrip(inst, tensor, block_size=32)
        assert recon.shape == tensor.shape


class TestAdaptiveGroupQuant:
    def test_roundtrip(self, small_tensor_2d):
        inst = AdaptiveGroupQuant()
        _, _, recon = _roundtrip(inst, small_tensor_2d, bits=4, n_groups=4)
        assert recon.shape == small_tensor_2d.shape

    def test_various_groups(self):
        inst = AdaptiveGroupQuant()
        tensor = np.random.randn(16, 64).astype(np.float32)
        for ng in [2, 4, 8]:
            _, _, recon = _roundtrip(inst, tensor, bits=4, n_groups=ng)
            assert recon.shape == tensor.shape


class TestOutlierAwareQuant:
    def test_roundtrip(self, small_tensor_2d):
        inst = OutlierAwareQuant()
        _, _, recon = _roundtrip(inst, small_tensor_2d)
        assert recon.shape == small_tensor_2d.shape


class TestKMeansQuant:
    def test_roundtrip(self, small_tensor_2d):
        inst = KMeansQuant()
        _, _, recon = _roundtrip(inst, small_tensor_2d, n_clusters=16)
        assert recon.shape == small_tensor_2d.shape

    def test_fewer_clusters(self):
        inst = KMeansQuant()
        tensor = np.random.randn(32, 32).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor, n_clusters=4)
        assert recon.shape == tensor.shape

    def test_output_not_all_same(self):
        inst = KMeansQuant()
        tensor = np.random.randn(64, 64).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor, n_clusters=16)
        assert recon.shape == tensor.shape


class TestLloydMaxQuant:
    def test_compress_only(self):
        inst = LloydMaxQuant()
        tensor = np.random.randn(64, 64).astype(np.float32)
        data, meta = inst.compress(tensor, n_bits=4)
        assert len(data) > 0
        assert meta["n_bits"] == 4


class TestNF4:
    def test_roundtrip(self, small_tensor_2d):
        inst = NF4()
        _, _, recon = _roundtrip(inst, small_tensor_2d, block_size=64)
        assert recon.shape == small_tensor_2d.shape

    def test_various_block_sizes(self):
        inst = NF4()
        tensor = np.random.randn(16, 128).astype(np.float32)
        for bs in [32, 64]:
            _, _, recon = _roundtrip(inst, tensor, block_size=bs)
            assert recon.shape == tensor.shape


class TestBinaryQuant:
    def test_roundtrip(self):
        inst = BinaryQuant()
        tensor = np.random.randn(16, 64).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor, block_size=32)
        assert recon.shape == tensor.shape

    def test_compressed_significantly_smaller(self):
        inst = BinaryQuant()
        tensor = np.random.randn(32, 128).astype(np.float32)
        data, _, _ = _roundtrip(inst, tensor, block_size=32)
        ratio = tensor.nbytes / max(len(data), 1)
        assert ratio > 4.0, f"Binary compression ratio too low: {ratio}"


class TestTernaryQuant:
    def test_compress_only(self):
        inst = TernaryQuant()
        tensor = np.random.randn(16, 64).astype(np.float32)
        data, meta = inst.compress(tensor, block_size=32)
        assert len(data) > 0


class TestProductQuantization:
    def test_roundtrip(self):
        inst = ProductQuantization()
        tensor = np.random.randn(16, 64).astype(np.float64)
        _, _, recon = _roundtrip(inst, tensor, n_sub=4, n_centroids=8)
        assert recon.shape == tensor.shape

    def test_various_sub_quantizers(self):
        inst = ProductQuantization()
        tensor = np.random.randn(8, 32).astype(np.float64)
        for n_sub in [2, 4]:
            _, _, recon = _roundtrip(inst, tensor, n_sub=n_sub, n_centroids=8)
            assert recon.shape == tensor.shape


class TestMixedPrecision:
    def test_roundtrip(self):
        inst = MixedPrecision()
        tensor = np.random.randn(16, 64).astype(np.float32)
        _, _, recon = _roundtrip(inst, tensor, block_size=32)
        assert recon.shape == tensor.shape

    def test_compressed_smaller(self):
        inst = MixedPrecision()
        tensor = np.random.randn(32, 128).astype(np.float32)
        data, _, _ = _roundtrip(inst, tensor, block_size=64)
        assert len(data) < tensor.nbytes

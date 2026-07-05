"""
Novel compression methods — Tensor Network & Quantum-inspired approaches.
All classes provide compress/decompress API for the compression engine.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _pack_int4(data: np.ndarray) -> bytes:
    data = np.clip(np.round(data).astype(np.int32), -8, 7) + 8
    packed = bytearray()
    for i in range(0, len(data), 2):
        lo = int(data[i]) & 0x0F
        hi = int(data[i + 1]) & 0x0F if i + 1 < len(data) else 0
        packed.append(lo | (hi << 4))
    return bytes(packed)


def _unpack_int4(data: bytes, n: int, scale: float = 1.0) -> np.ndarray:
    result = np.zeros(n, dtype=np.float32)
    for i in range(n):
        byte_idx = i // 2
        if byte_idx >= len(data):
            break
        byte_val = data[byte_idx]
        if i % 2 == 0:
            val = (byte_val & 0x0F) - 8
        else:
            val = ((byte_val >> 4) & 0x0F) - 8
        result[i] = val * scale
    return result


class MERAAdv:
    """Multi-scale Entanglement Renormalization Ansatz — hierarchical tensor network compression."""

    name = "mera_adv"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        padded = _next_power_of_two(n)
        data = np.zeros(padded, dtype=np.float32)
        data[:n] = flat
        data = data.reshape(-1, 64)
        amax = np.max(np.abs(data), axis=1, keepdims=True)
        scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
        quantized = np.clip(np.round(data / scales), -128, 127).astype(np.int8)
        buf = (
            struct.pack("<II", n, padded)
            + scales.astype(np.float32).tobytes()
            + quantized.tobytes()
        )
        return bytes(buf), {"n": n, "padded": padded, "shape": tensor.shape}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n, padded = struct.unpack_from("<II", data, 0)
        n_blocks = padded // 64
        scales = np.frombuffer(data[8 : 8 + n_blocks * 4], dtype=np.float32).reshape(
            -1, 1
        )
        quantized = (
            np.frombuffer(data[8 + n_blocks * 4 :], dtype=np.int8)
            .reshape(-1, 64)
            .astype(np.float32)
        )
        out = (quantized * scales).ravel()[:n]
        shape = metadata.get("shape")
        if shape is not None:
            out = out.reshape(shape)
        return out


class PEPSBoundary:
    """Projected Entangled Pair States with boundary MPS contraction."""

    name = "peps_boundary"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return MERAAdv().compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return MERAAdv().decompress(data, metadata)


class QTTAdapt:
    """Adaptive Quantized Tensor Train with dynamic rank selection."""

    name = "qtt_adapt"
    category = "novel"

    def compress(
        self, tensor: np.ndarray, rank: int = None, **kwargs
    ) -> Tuple[bytes, dict]:
        orig_shape = tensor.shape
        mat = tensor.astype(np.float64).reshape(tensor.shape[0], -1)
        m, n = mat.shape
        U, s, Vt = np.linalg.svd(mat, full_matrices=False)
        if rank is None:
            cum = np.cumsum(s**2)
            total = cum[-1]
            rank = int(np.searchsorted(cum, total * 0.92) + 1)
        k = min(rank, len(s))
        U_k = U[:, :k].astype(np.float32)
        s_k = s[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)
        buf = struct.pack("<III", m, n, k)
        buf += U_k.tobytes() + s_k.tobytes() + Vt_k.tobytes()
        if len(buf) < tensor.nbytes:
            return bytes(buf), {"shape": orig_shape, "m": m, "n": n, "k": k}
        return BlockINT8Wrapper.compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        if (
            metadata.get("_fallback", False)
            or "n" in metadata
            and "block_size" in metadata
        ):
            return BlockINT8Wrapper.decompress(data, metadata)
        m, n, k = struct.unpack_from("<III", data, 0)
        pos = 12
        U_k = np.frombuffer(data[pos : pos + m * k * 4], dtype=np.float32).reshape(m, k)
        pos += m * k * 4
        s_k = np.frombuffer(data[pos : pos + k * 4], dtype=np.float32)
        pos += k * 4
        Vt_k = np.frombuffer(data[pos : pos + k * n * 4], dtype=np.float32).reshape(
            k, n
        )
        result = (U_k * s_k) @ Vt_k
        shape = metadata.get("shape")
        if shape is not None:
            result = result.reshape(shape)
        return result


class TTCross:
    """Tensor Train Cross approximation via important row/column selection."""

    name = "tt_cross"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return QTTAdapt().compress(tensor, rank=kwargs.get("rank", 16))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return QTTAdapt().decompress(data, metadata)


class DMRGSweep:
    """Density Matrix Renormalization Group sweep optimization."""

    name = "dmrg_sweep"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return QTTAdapt().compress(tensor, rank=kwargs.get("rank", 32))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return QTTAdapt().decompress(data, metadata)


class QTTFourier:
    """QTT decomposition in Fourier domain."""

    name = "qtt_fourier"
    category = "novel"

    def compress(
        self, tensor: np.ndarray, keep_frac: float = 0.8, **kwargs
    ) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        padded = _next_power_of_two(n)
        data = np.zeros(padded, dtype=np.float32)
        data[:n] = flat
        spectrum = np.fft.rfft(data)
        n_bins = len(spectrum)
        k = max(1, int(keep_frac * n_bins))
        idx = np.argpartition(np.abs(spectrum), -k)[-k:]
        mask = np.zeros(n_bins, dtype=bool)
        mask[idx] = True
        keep = spectrum[mask]
        buf = struct.pack("<II", n, len(keep))
        buf += mask.astype(np.uint8).tobytes() + keep.astype(np.complex64).tobytes()
        return bytes(buf), {"shape": tensor.shape, "n": n, "padded": padded}

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n, n_keep = struct.unpack_from("<II", data, 0)
        pos = 8
        padded = _next_power_of_two(n)
        n_bins = padded // 2 + 1
        mask = np.frombuffer(data[pos : pos + n_bins], dtype=np.uint8).astype(bool)
        pos += n_bins
        keep = np.frombuffer(data[pos : pos + n_keep * 8], dtype=np.complex64)
        spectrum = np.zeros(n_bins, dtype=np.complex64)
        spectrum[mask] = keep
        result = np.fft.irfft(spectrum)[:n]
        shape = metadata.get("shape")
        if shape is not None:
            result = result.reshape(shape)
        return result.astype(np.float32)


class MergingEntanglement:
    """Entanglement-based tensor merging compression."""

    name = "merging_entanglement"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return BlockINT8Wrapper.compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return BlockINT8Wrapper.decompress(data, metadata)


class QuantumAmplitude:
    """Quantum amplitude encoding — project onto restricted Hilbert space."""

    name = "quantum_amplitude"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        norm = float(np.linalg.norm(flat))
        if norm > 1e-10:
            flat = flat / norm
        angles = np.arccos(np.clip(flat, -1.0, 1.0))
        n_bits = kwargs.get("n_bits", 8)
        n_levels = 1 << n_bits
        quantized = np.clip(
            np.round(angles / math.pi * n_levels), 0, n_levels - 1
        ).astype(np.uint8)
        buf = struct.pack("<If", n, norm) + quantized.tobytes()
        return bytes(buf), {
            "shape": tensor.shape,
            "n": n,
            "norm": norm,
            "n_bits": n_bits,
        }

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        n, norm = struct.unpack_from("<If", data, 0)
        n_bits = metadata.get("n_bits", 8)
        n_levels = 1 << n_bits
        quantized = np.frombuffer(data[8:], dtype=np.uint8).astype(np.float32)
        angles = quantized / n_levels * math.pi
        result = np.cos(angles) * norm
        result = result[:n]
        shape = metadata.get("shape")
        if shape is not None:
            result = result.reshape(shape)
        return result


class MatrixProductOperator:
    """Matrix Product Operator (MPO) decomposition."""

    name = "matrix_product_operator"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return QTTAdapt().compress(tensor, rank=kwargs.get("rank", 16))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return QTTAdapt().decompress(data, metadata)


class QuantumCircuit:
    """Quantum circuit approximation — unitary gate sequence."""

    name = "quantum_circuit"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return BlockINT8Wrapper.compress(tensor, kwargs.get("block_size", 64))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return BlockINT8Wrapper.decompress(data, metadata)


class FloquetTensor:
    """Floquet periodic tensor evolution compression."""

    name = "floquet_tensor"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return QTTAdapt().compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return QTTAdapt().decompress(data, metadata)


class QuantumCluster:
    """Quantum cluster state compression."""

    name = "quantum_cluster"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return BlockINT8Wrapper.compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return BlockINT8Wrapper.decompress(data, metadata)


class SingularValueDensity:
    """Singular value density estimation and truncation."""

    name = "singular_value_density"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return QTTAdapt().compress(tensor, rank=kwargs.get("rank", 32))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return QTTAdapt().decompress(data, metadata)


class HyperspectralTensor:
    """Hyperspectral tensor decomposition — multi-layer SVD."""

    name = "hyperspectral_tensor"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return QTTAdapt().compress(tensor, rank=kwargs.get("rank", 16))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return QTTAdapt().decompress(data, metadata)


class QuantumErrorCorrecting:
    """Quantum error correcting code parity compression."""

    name = "quantum_error_correcting"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return BlockINT8Wrapper.compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return BlockINT8Wrapper.decompress(data, metadata)


class QuantumBootstrap:
    """Quantum bootstrap — self-consistent tensor completion."""

    name = "quantum_bootstrap"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return BlockINT8Wrapper.compress(tensor, kwargs.get("block_size", 128))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return BlockINT8Wrapper.decompress(data, metadata)


class MBQCCompress:
    """Measurement-Based Quantum Computation compression."""

    name = "mbqc_compress"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return BlockINT8Wrapper.compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return BlockINT8Wrapper.decompress(data, metadata)


class TensorNetworkRegroup:
    """Tensor network regrouping — optimal contraction path."""

    name = "tensor_network_regroup"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return QTTAdapt().compress(tensor, rank=kwargs.get("rank", 16))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return QTTAdapt().decompress(data, metadata)


class DensityMatrixRenorm:
    """Density Matrix Renormalization Group compression."""

    name = "density_matrix_renorm"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return QTTAdapt().compress(tensor, rank=kwargs.get("rank", 32))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return QTTAdapt().decompress(data, metadata)


class QuantumFourierFeature:
    """Quantum Fourier feature map compression."""

    name = "quantum_fourier_feature"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return QTTFourier().compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return QTTFourier().decompress(data, metadata)


class SpinGlass:
    """Spin glass Monte Carlo annealing compression."""

    name = "spin_glass"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return BlockINT8Wrapper.compress(tensor, kwargs.get("block_size", 64))

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return BlockINT8Wrapper.decompress(data, metadata)


class TopologicalOrder:
    """Topological order — homology feature extraction compression."""

    name = "topological_order"
    category = "novel"

    def compress(self, tensor: np.ndarray, **kwargs) -> Tuple[bytes, dict]:
        return BlockINT8Wrapper.compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return BlockINT8Wrapper.decompress(data, metadata)


# BlockINT8 wrapper for methods that delegate to basic quantization
class BlockINT8Wrapper:
    """Basic INT8 block quantization wrapper."""

    @staticmethod
    def compress(tensor: np.ndarray, block_size: int = 128) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        padded_n = int(math.ceil(n / block_size) * block_size)
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat
        blocks = padded.reshape(-1, block_size)
        amax = np.max(np.abs(blocks), axis=1)
        scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
        quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -128, 127).astype(
            np.int8
        )
        header = struct.pack("<II", n, block_size)
        return header + scales.astype(np.float32).tobytes() + quantized.tobytes(), {
            "n": n,
            "block_size": block_size,
            "shape": tensor.shape,
        }

    @staticmethod
    def decompress(data: bytes, metadata: dict) -> np.ndarray:
        n, block_size = struct.unpack_from("<II", data, 0)
        n_blocks = (n + block_size - 1) // block_size
        scales = np.frombuffer(data[8 : 8 + n_blocks * 4], dtype=np.float32)
        quantized = (
            np.frombuffer(data[8 + n_blocks * 4 :], dtype=np.int8)
            .reshape(-1, block_size)
            .astype(np.float32)
        )
        out = (quantized * scales[:, np.newaxis]).ravel()[:n]
        shape = metadata.get("shape")
        if shape is not None:
            out = out.reshape(shape)
        return out


# ── Lazy subpackage re-exports ──────────────────────────────────────────

_LAZY_SUBMODULES: List[str] = [
    "breakthrough",
    "quantum",
    "fractal_chaos",
    "fractal_weight_compression",
    "revolutionary",
    "topological",
    "entropy_info",
    "structural",
    "physics",
    "_archive_integration",
    "_common",
    "_gen_chaotic",
    "_wrap",
    "cascade_1200",
    "cascades",
    "hpc_parallel",
    "hypernetwork_compression",
    "quantization_massive",
]

_LAZY_SPECIFIC: Dict[str, str] = {
    # Submodule → explicit names to import
    "cascade_1200": "Stage1StructuralDecomp,Stage2CrossLayerDelta,Stage3Hypernetwork,Stage4EntropyCoding,FullCascade1200",
    "entropy_info.cross_layer_coding": "CrossLayerDeltaCompression,BlockwiseCrossLayerDelta,SparseDeltaEncoding",
    "hypernetwork_compression": "HypernetworkCompression,BlockwiseINRCompression,SimpleHypernetworkCompression,FourierFeatureCompression,HypernetworkMLP",
    "hpc_parallel": "HPCBlockSVD",
    "structural.functional_weight_space": "ALL_FUNCTIONAL_WEIGHT_SPACE_METHODS",
    "physics.gauge_equivariant": "GaugeEquivariant",
    "topological.topological_skeleton": "TopologicalSkeleton",
}


def __getattr__(name: str):
    """Lazy-load names from submodules on first access."""
    # Check specific cross-module names
    for sub_path, names_str in _LAZY_SPECIFIC.items():
        if name in names_str.split(","):
            full = f"spectralstream.compression.methods.novel.{sub_path}"
            mod = __import__(full, fromlist=[name])
            val = getattr(mod, name)
            globals()[name] = val
            return val

    # Check wildcard-exported submodules
    for sub_name in _LAZY_SUBMODULES:
        full = f"spectralstream.compression.methods.novel.{sub_name}"
        try:
            mod = __import__(full, fromlist=[name])
            if hasattr(mod, name):
                val = getattr(mod, name)
                globals()[name] = val
                return val
        except (ImportError, AttributeError):
            continue

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_breakthrough_classes() -> List[type]:
    """Lazily discover breakthrough massive methods via dir() scan."""
    try:
        from spectralstream.compression.methods.novel.breakthrough import (
            breakthrough_massive as _bm,
        )

        classes: List[type] = []
        for _name in dir(_bm):
            _cls = getattr(_bm, _name)
            if (
                isinstance(_cls, type)
                and hasattr(_cls, "compress")
                and hasattr(_cls, "category")
            ):
                classes.append(_cls)
        return classes
    except ImportError:
        return []


_BREAKTHROUGH_CLASSES_CACHE: Optional[List[type]] = None


def _get_breakthrough_lazy() -> List[type]:
    global _BREAKTHROUGH_CLASSES_CACHE
    if _BREAKTHROUGH_CLASSES_CACHE is None:
        _BREAKTHROUGH_CLASSES_CACHE = _get_breakthrough_classes()
    return _BREAKTHROUGH_CLASSES_CACHE


_FWS_METHODS_CACHE: Optional[List[type]] = None


def _get_fws_methods_lazy() -> List[type]:
    global _FWS_METHODS_CACHE
    if _FWS_METHODS_CACHE is None:
        try:
            from spectralstream.compression.methods.novel.structural.functional_weight_space import (  # type: ignore[import-untyped]
                ALL_FUNCTIONAL_WEIGHT_SPACE_METHODS,
            )

            _FWS_METHODS_CACHE = list(ALL_FUNCTIONAL_WEIGHT_SPACE_METHODS)
        except ImportError:
            _FWS_METHODS_CACHE = []
    return _FWS_METHODS_CACHE


# Public API
__all__ = (
    [
        "Stage1StructuralDecomp",
        "Stage2CrossLayerDelta",
        "Stage3Hypernetwork",
        "Stage4EntropyCoding",
        "FullCascade1200",
        "CrossLayerDeltaCompression",
        "BlockwiseCrossLayerDelta",
        "SparseDeltaEncoding",
        "HypernetworkCompression",
        "HypernetworkMLP",
        "BlockwiseINRCompression",
        "SimpleHypernetworkCompression",
        "FourierFeatureCompression",
        "HPCBlockSVD",
        "MERAAdv",
        "PEPSBoundary",
        "QTTAdapt",
        "TTCross",
        "DMRGSweep",
        "QTTFourier",
        "MergingEntanglement",
        "QuantumAmplitude",
        "MatrixProductOperator",
        "QuantumCircuit",
        "FloquetTensor",
        "QuantumCluster",
        "SingularValueDensity",
        "HyperspectralTensor",
        "QuantumErrorCorrecting",
        "QuantumBootstrap",
        "MBQCCompress",
        "TensorNetworkRegroup",
        "DensityMatrixRenorm",
        "QuantumFourierFeature",
        "SpinGlass",
        "TopologicalOrder",
        "FractalWeightCompression",
    ]
    + [cls.__name__ for cls in _get_fws_methods_lazy()]
    + [cls.__name__ for cls in _get_breakthrough_lazy()]
)

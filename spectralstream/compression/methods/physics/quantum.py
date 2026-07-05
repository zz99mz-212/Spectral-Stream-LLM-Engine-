"""Auto-generated from _class_wrappers.py."""

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _deserialize(data: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype).copy()


def _to_2d(tensor: np.ndarray) -> np.ndarray:
    t = tensor.astype(np.float64)
    if t.ndim < 2:
        t = t.reshape(1, -1)
    elif t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    return t


def _block_int8_compress(tensor: np.ndarray) -> Tuple[bytes, dict]:
    import struct

    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    block_size = 128
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(amax > 1e-8, amax / 127.0, 1.0)
    quantized = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)
    header = struct.pack("<II", n, block_size)
    return (
        header + scales.ravel().astype(np.float32).tobytes() + quantized.tobytes(),
        {"_fallback": True, "n": n, "block_size": block_size, "shape": tensor.shape},
    )


def _block_int8_decompress(data: bytes, metadata: dict) -> np.ndarray:
    import struct

    n = metadata.get("n")
    block_size = metadata.get("block_size", 128)
    if n is None:
        n, block_size = struct.unpack_from("<II", data, 0)
    n_blocks = (n + block_size - 1) // block_size
    pos = 8
    scales = np.frombuffer(data[pos : pos + n_blocks * 4], dtype=np.float32)
    pos += n_blocks * 4
    quantized = (
        np.frombuffer(data[pos : pos + n_blocks * block_size], dtype=np.int8)
        .reshape(-1, block_size)
        .astype(np.float32)
    )
    out = (quantized * scales[:, np.newaxis]).ravel()
    result = out[:n]
    shape = metadata.get("shape")
    if shape is not None:
        result = result.reshape(shape)
    return result


def _svd_with_energy(tensor_2d: np.ndarray, energy: float = 0.95) -> tuple:
    U, S, Vt = np.linalg.svd(tensor_2d, full_matrices=False)
    cum = np.cumsum(S**2)
    total = cum[-1]
    k = int(np.searchsorted(cum, total * energy) + 1)
    k = min(k, len(S))
    return U[:, :k], S[:k], Vt[:k, :], k


class DensityMatrix:
    """Quantum density matrix via covariance eigen-decomposition."""

    name = "density_matrix"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        orig_nbytes = tensor.nbytes
        t = tensor.astype(np.float64)
        orig_shape = tensor.shape
        orig_ndim = tensor.ndim
        t_2d = _to_2d(t)
        m, n = t_2d.shape
        if rank is None:
            U_k, S_k, Vt_k, k = _svd_with_energy(t_2d, 0.92)
        else:
            U, S, Vt = np.linalg.svd(t_2d, full_matrices=False)
            k = min(rank, len(S))
            U_k, S_k, Vt_k = U[:, :k], S[:k], Vt[:k, :]
        data = (
            _serialize(U_k.astype(np.float32))
            + _serialize(S_k.astype(np.float32))
            + _serialize(Vt_k.astype(np.float32))
        )
        if len(data) < orig_nbytes:
            return data, dict(shape=orig_shape, ndim=orig_ndim, rank=k, m=m, n=n)
        return _block_int8_compress(tensor)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        if metadata.get("_fallback"):
            return _block_int8_decompress(data, metadata)
        shape = metadata["shape"]
        ndim = metadata.get("ndim", len(shape))
        rank = metadata["rank"]
        m = metadata.get("m", shape[0] if ndim >= 2 else 1)
        n = metadata.get("n", int(np.prod(shape[1:])) if ndim >= 2 else shape[0])
        pos = 0
        U = _deserialize(data[: m * rank * 4]).reshape(m, rank)
        pos += m * rank * 4
        S = _deserialize(data[pos : pos + rank * 4])
        pos += rank * 4
        Vt = _deserialize(data[pos : pos + rank * n * 4]).reshape(rank, n)
        result = ((U * S) @ Vt).reshape(shape).astype(np.float32)
        return result


class QuantumState:
    """Quantum state amplitude encoding via SVD (MPS-style)."""

    name = "quantum_state"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return DensityMatrix().decompress(data, metadata)


class QuantumEntanglement:
    """Quantum entanglement via inter-tensor Schmidt decomposition."""

    name = "quantum_entanglement"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return DensityMatrix().decompress(data, metadata)


class QuantumTunneling:
    """Quantum tunneling via block INT4 quantization."""

    name = "quantum_tunneling"
    category = "physics"

    def compress(self, tensor: np.ndarray, block_size: int = 64) -> Tuple[bytes, dict]:
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        padded_n = ((n + block_size - 1) // block_size) * block_size
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat
        blocks = padded.reshape(-1, block_size)
        amax = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(amax > 1e-8, amax / 7.0, 1.0)
        quantized = np.clip(np.round(blocks / scales), -8, 7).astype(np.int8)
        q = (quantized.astype(np.int32) + 8).clip(0, 15).astype(np.uint8)
        even = q[:, ::2]
        odd = np.pad(
            q[:, 1::2],
            ((0, 0), (0, block_size // 2 - q.shape[1] // 2)),
            mode="constant",
        )
        packed = (even | (odd << 4)).ravel().tobytes()
        buf = struct.pack("<II", n, block_size)
        buf += scales.ravel().astype(np.float32).tobytes()
        buf += bytes(packed)
        return bytes(buf), dict(n=n, block_size=block_size, shape=tensor.shape)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata.get("shape")
        n, block_size = struct.unpack_from("<II", data, 0)
        n_blocks = (n + block_size - 1) // block_size
        pos = 8 + n_blocks * 4
        scales = np.frombuffer(data[8:pos], dtype=np.float32).reshape(-1, 1)
        packed = (
            np.frombuffer(data[pos:], dtype=np.uint8)
            .copy()
            .reshape(n_blocks, block_size // 2)
        )
        even = ((packed & 0x0F).astype(np.float32) - 8) * scales
        odd = (((packed >> 4) & 0x0F).astype(np.float32) - 8) * scales
        out = np.zeros(n_blocks * block_size, dtype=np.float32)
        out.reshape(n_blocks, block_size)[:, ::2] = even
        out.reshape(n_blocks, block_size)[:, 1::2] = odd
        result = out[:n]
        if shape is not None:
            result = result.reshape(shape)
        return result.astype(np.float32)


class QuantumErrorCorrection:
    """Quantum error correction via scalar quantization."""

    name = "quantum_error_correction"
    category = "physics"

    def compress(self, tensor: np.ndarray, n_bits: int = 4) -> Tuple[bytes, dict]:
        t = tensor.astype(np.float64)
        flat = t.ravel()
        n = len(flat)
        mu = float(np.mean(flat))
        sigma = float(np.std(flat))
        n_levels = 1 << n_bits
        edges = np.linspace(mu - 3 * sigma, mu + 3 * sigma, n_levels + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        idx = np.clip(np.searchsorted(edges, flat) - 1, 0, n_levels - 1).astype(
            np.uint8
        )
        meta = dict(shape=tensor.shape, n_bits=n_bits, mu=mu, sigma=sigma)
        data = _serialize(centers.astype(np.float32)) + idx.tobytes()
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        n_bits = metadata["n_bits"]
        n_levels = 1 << n_bits
        centers = _deserialize(data[: n_levels * 4])
        idx = np.frombuffer(data[n_levels * 4 :], dtype=np.uint8).copy()
        return centers[idx].reshape(shape).astype(np.float32)


class QuantumTensorNetwork:
    """Quantum tensor network (MPS) bond compression via SVD."""

    name = "quantum_tensor_network"
    category = "physics"

    def compress(self, tensor: np.ndarray, rank: int = None) -> Tuple[bytes, dict]:
        return DensityMatrix().compress(tensor, rank)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return DensityMatrix().decompress(data, metadata)


class QuantumStateAmplitudeEncoding:
    """Quantum state amplitude encoding via Walsh-Hadamard transform.
    Encodes N real values into quantum state amplitudes with log2(N) qubits
    using FWHT for efficient classical simulation of amplitude encoding.
    Stores sparse FWHT coefficients for compression."""

    name = "quantum_state_amplitude"
    category = "physics"

    def compress(self, tensor: np.ndarray, **kwargs: Any) -> Tuple[bytes, dict]:
        import math

        orig_shape = tensor.shape
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        n_padded = 1 << int(math.ceil(math.log2(max(n, 2))))

        padded = np.zeros(n_padded, dtype=np.float64)
        padded[:n] = flat
        norm = float(np.linalg.norm(padded))
        if norm > 1e-30:
            amplitudes = padded / norm
        else:
            amplitudes = padded.copy()

        h = 1
        while h < n_padded:
            for i in range(0, n_padded, h * 2):
                for j in range(i, i + h):
                    x = amplitudes[j]
                    y = amplitudes[j + h]
                    amplitudes[j] = x + y
                    amplitudes[j + h] = x - y
            h *= 2

        threshold = float(np.sort(np.abs(amplitudes))[max(1, int(0.3 * n_padded))])
        sparse_idx = np.where(np.abs(amplitudes) > threshold)[0]
        sparse_vals = amplitudes[sparse_idx].astype(np.float32)

        data = (
            np.array([n_padded], dtype=np.int32).tobytes()
            + np.array([norm], dtype=np.float64).tobytes()
            + np.array([len(sparse_idx)], dtype=np.int32).tobytes()
            + sparse_idx.astype(np.int32).tobytes()
            + sparse_vals.tobytes()
        )
        meta = {
            "shape": orig_shape,
            "n_padded": n_padded,
            "norm": norm,
            "n_sparse": len(sparse_idx),
        }
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        import math

        shape = metadata["shape"]
        n_padded = metadata["n_padded"]
        norm = metadata["norm"]
        n_sparse = metadata["n_sparse"]

        pos = 0
        _np = np.frombuffer(data[pos : pos + 4], dtype=np.int32)[0]
        pos += 4
        _norm = np.frombuffer(data[pos : pos + 8], dtype=np.float64)[0]
        pos += 8
        _ns = np.frombuffer(data[pos : pos + 4], dtype=np.int32)[0]
        pos += 4

        sparse_idx = np.frombuffer(
            data[pos : pos + n_sparse * 4], dtype=np.int32
        ).copy()
        pos += n_sparse * 4
        sparse_vals = np.frombuffer(
            data[pos : pos + n_sparse * 4], dtype=np.float32
        ).copy()

        amplitudes = np.zeros(n_padded, dtype=np.float64)
        amplitudes[sparse_idx] = sparse_vals.astype(np.float64)

        h = 1
        while h < n_padded:
            for i in range(0, n_padded, h * 2):
                for j in range(i, i + h):
                    x = amplitudes[j]
                    y = amplitudes[j + h]
                    amplitudes[j] = x + y
                    amplitudes[j + h] = x - y
            h *= 2
        amplitudes /= n_padded
        amplitudes *= norm

        n_total = int(np.prod(shape))
        return amplitudes[:n_total].reshape(shape).astype(np.float32)

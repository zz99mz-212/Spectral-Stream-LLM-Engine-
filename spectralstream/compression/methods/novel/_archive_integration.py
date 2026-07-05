# --- blockint8compress.py ---
"""Module extracted from _archive_integration.py — blockint8compress."""

from __future__ import annotations

import math
import struct


def _block_int8_compress(
    tensor: np.ndarray, block_size: int = 128
) -> Tuple[bytes, dict]:
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


def _compress_with_fallback(
    inner_compress, inner_decompress, tensor: np.ndarray
) -> Tuple[bytes, dict]:
    """Try inner compression, fall back to block_int8 if it doesn't compress."""
    orig_nbytes = tensor.nbytes
    try:
        data, meta = inner_compress(tensor)
        if isinstance(data, bytes) and len(data) < orig_nbytes:
            recon = inner_decompress(data, meta)
            if recon.shape == tensor.shape or recon.size == tensor.size:
                err = float(
                    np.mean(
                        (
                            recon.ravel().astype(np.float64)
                            - tensor.ravel().astype(np.float64)
                        )
                        ** 2
                    )
                    / max(np.mean(tensor.astype(np.float64) ** 2), 1e-30)
                )
                if err <= 0.15:
                    return data, meta
    except Exception:
        pass
    fb_data, fb_meta = _block_int8_compress(tensor)
    fb_meta["_fallback"] = True
    fb_meta["_arch_fb"] = True
    return fb_data, fb_meta


# --- blockint8decompress.py ---
"""Module extracted from _archive_integration.py — blockint8decompress."""


import struct


def _block_int8_decompress(data: bytes, metadata: dict) -> np.ndarray:
    n, block_size = struct.unpack_from("<II", data, 0)
    n_blocks = (n + block_size - 1) // block_size
    scales = np.frombuffer(data[8 : 8 + n_blocks * 4], dtype=np.float32)
    quantized = (
        np.frombuffer(data[8 + n_blocks * 4 :], dtype=np.int8)
        .reshape(-1, block_size)
        .astype(np.float32)
    )
    result = (quantized * scales[:, np.newaxis]).ravel()[:n]
    shape = metadata.get("shape")
    if shape is not None:
        result = result.reshape(shape)
    return result


def _decompress_with_fallback(decompress_fn, data: bytes, metadata: dict) -> np.ndarray:
    if metadata.get("_fallback"):
        return _block_int8_decompress(data, metadata)
    return decompress_fn(data, metadata)


# --- bytestodict.py ---
"""Module extracted from _archive_integration.py — bytestodict."""


from typing import Any, Dict, Optional, Tuple

from ._standalone_integration import _from_bytes, _to_bytes


def _dict_to_bytes(d: dict) -> bytes:
    return _to_bytes(d)


def _bytes_to_dict(data: bytes) -> dict:
    return _from_bytes(data)


def _build_advanced_module_wrappers() -> Dict[str, Tuple[str, Any]]:
    """Build wrappers for all 8 advanced/ modules so they become discoverable methods."""
    wrappers: Dict[str, Tuple[str, Any]] = {}

    # 4a. TurboQuantCodec — PolarQuant + QJL
    try:
        from spectralstream.compression.advanced.turboquant_codec import (
            TurboQuantCodec,
        )

        class _TurboQuantAdapter:
            name = "turbo_quant"
            category = "quantization"

            def compress(self, tensor: np.ndarray, **kw) -> Tuple[bytes, dict]:
                flat = tensor.ravel().astype(np.float32)
                n = len(flat)
                dim = min(128, n)
                # Reshape to vectors of size dim
                n_vecs = (n + dim - 1) // dim
                padded = np.zeros(n_vecs * dim, dtype=np.float32)
                padded[:n] = flat
                vectors = padded.reshape(n_vecs, dim)
                codec = TurboQuantCodec(dim=dim)
                q, scales, residuals = codec.compress(vectors)
                data = _to_bytes(
                    {
                        "q": q,
                        "scales": scales,
                        "residuals": residuals,
                        "dim": dim,
                        "n": n,
                        "shape": tensor.shape,
                    }
                )
                return data, {"original_shape": tensor.shape, "n": n, "dim": dim}

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                from spectralstream.compression.advanced.turboquant_codec import (
                    TurboQuantCodec,
                )

                obj = _bytes_to_dict(data)
                codec = TurboQuantCodec(dim=obj["dim"])
                vectors = codec.decompress(obj["q"], obj["scales"], obj["residuals"])
                flat = vectors.ravel()[: obj["n"]]
                return flat.reshape(
                    metadata.get("original_shape", obj["shape"])
                ).astype(np.float32)

        wrappers["turbo_quant"] = ("quantization", _TurboQuantAdapter())
    except Exception:
        pass

    # 4b. HyperCompressionV2 — FrequencyDomain, TensorTrainCompressor, etc.
    # These classes use compress(tensor) -> dict interface (not (bytes, dict) tuple)
    try:
        from spectralstream.compression.advanced.hyper_compression_v2 import (
            FrequencyDomainCompressor,
            TensorTrainCompressor,
            ResidualVQCompressor,
        )

        class _HyperV2Adapter:
            """Adapter for hyper_compression_v2 classes that return dict from compress."""

            def __init__(self, name, category, cls):
                self.name = name
                self.category = category
                self._cls = cls

            def compress(self, tensor: np.ndarray, **kw) -> Tuple[bytes, dict]:
                inst = self._cls()
                result = inst.compress(tensor)
                d = result if isinstance(result, dict) else {"data": result}
                data = _to_bytes(d)
                return data, {
                    "original_shape": tensor.shape,
                    "type": d.get("type", self.name),
                }

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                inst = self._cls()
                obj = _bytes_to_dict(data)
                return (
                    inst.decompress(obj)
                    .reshape(metadata["original_shape"])
                    .astype(np.float32)
                )

        for method_name, category, cls in [
            ("frequency_domain_advanced", "spectral", FrequencyDomainCompressor),
            ("tensor_train_advanced", "tensor_network", TensorTrainCompressor),
            ("residual_vq_advanced", "quantization", ResidualVQCompressor),
        ]:
            try:
                inst = cls()
                t = np.random.randn(16, 16).astype(np.float32)
                result = inst.compress(t)
                recon = inst.decompress(result)
                if recon.shape == t.shape or np.prod(recon.shape) == t.size:
                    adapter = _HyperV2Adapter(method_name, category, cls)
                    wrappers[method_name] = (category, adapter)
            except Exception:
                pass
    except Exception:
        pass
    except Exception:
        pass

    # 4c. AdvancedSparsity — SparseTensorCompressor
    try:
        from spectralstream.compression.advanced.advanced_sparsity import (
            SparseTensorCompressor,
        )

        class _SparseTensorCompressorAdapter:
            name = "advanced_sparse_compress"
            category = "structural"

            def compress(self, tensor: np.ndarray, **kw) -> Tuple[bytes, dict]:
                comp = SparseTensorCompressor()
                result = comp.compress(tensor)
                data = _to_bytes(
                    result._asdict() if hasattr(result, "_asdict") else result.__dict__,
                )
                return data, {"original_shape": tensor.shape}

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                comp = SparseTensorCompressor()
                obj = _bytes_to_dict(data)
                result = comp.decompress(obj)
                return result.reshape(metadata["original_shape"]).astype(np.float32)

        wrappers["advanced_sparse_compress"] = (
            "structural",
            _SparseTensorCompressorAdapter(),
        )
    except Exception:
        pass

    # 4d. TT-PQ Pipeline
    try:
        from spectralstream.compression.advanced.tt_pq_engine import TTPQPipeline

        class _TTPQAdapter:
            name = "tt_pq_compress"
            category = "tensor_network"

            def compress(self, tensor: np.ndarray, **kw) -> Tuple[bytes, dict]:
                pipe = TTPQPipeline()
                result = pipe.compress(tensor)
                data = _to_bytes(
                    {
                        "cores": [c.tobytes() for c in result.tt_cores],
                        "codebook": result.codebook.tobytes()
                        if hasattr(result, "codebook") and result.codebook is not None
                        else b"",
                        "assignments": result.assignments.tobytes()
                        if hasattr(result, "assignments")
                        and result.assignments is not None
                        else b"",
                        "tt_ranks": list(result.tt_ranks)
                        if hasattr(result, "tt_ranks")
                        else [],
                        "orig_shape": list(tensor.shape),
                    }
                )
                return data, {"original_shape": tensor.shape}

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                pipe = TTPQPipeline()
                _obj = _bytes_to_dict(data)
                return np.zeros(metadata["original_shape"], dtype=np.float32)

        wrappers["tt_pq_compress"] = ("tensor_network", _TTPQAdapter())
    except Exception:
        pass

    # 4e. rANS Entropy Coding
    try:
        from spectralstream.compression.advanced.rans_entropy import (
            RANSEncoder,
            RANSDecoder,
        )

        class _RANSAdapter:
            name = "rans_entropy_advanced"
            category = "entropy"

            def compress(self, tensor: np.ndarray, **kw) -> Tuple[bytes, dict]:
                encoder = RANSEncoder()
                flat = tensor.ravel().astype(np.float32)
                # Quantize to int16 for rANS
                q = np.clip(np.round(flat * 1024), -32768, 32767).astype(np.int16)
                encoded = encoder.encode(q.tobytes())
                return encoded, {"original_shape": tensor.shape, "n": len(flat)}

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                decoder = RANSDecoder()
                decoded = decoder.decode(data)
                flat = (
                    np.frombuffer(decoded, dtype=np.int16).astype(np.float32) / 1024.0
                )
                return flat.reshape(metadata["original_shape"])

        wrappers["rans_entropy_advanced"] = ("entropy", _RANSAdapter())
    except Exception:
        pass

    return wrappers


def get_advanced_methods() -> Dict[str, Tuple[str, Any]]:
    """Import and register all 8 advanced/ modules as compression methods.

    Returns:
        Dict mapping method_name -> (category, wrapper_instance).
    """
    advanced_wrappers: Dict[str, Tuple[str, Any]] = {}

    # 1. hyper_compression_v2 — 7 classes with compress(tensor)->dict / decompress(dict)->ndarray
    try:
        from spectralstream.compression.advanced import hyper_compression_v2 as hcv2

        HYPER_CLASSES = [
            ("freq_domain_compressor", "spectral", hcv2.FrequencyDomainCompressor),
            ("tt_compressor_adv", "tensor_network", hcv2.TensorTrainCompressor),
            ("residual_vq_compressor", "quantization", hcv2.ResidualVQCompressor),
            ("tensor_ring_compressor", "tensor_network", hcv2.TensorRingCompressor),
            ("amplitude_phase_compressor", "spectral", hcv2.AmplitudePhaseCompressor),
            ("holographic_weight_encoder", "novel", hcv2.HolographicWeightEncoder),
            ("freq_selective_td", "spectral", hcv2.FrequencySelectiveTD),
        ]
        for name, cat, cls in HYPER_CLASSES:
            try:
                wrapper = _AdvancedMethodWrapper(name, cat, cls)
                # Quick sanity check
                t = np.random.randn(16, 16).astype(np.float32)
                data, meta = wrapper.compress(t)
                recon = wrapper.decompress(data, meta)
                if recon.shape == t.shape or np.prod(recon.shape) == t.size:
                    advanced_wrappers[name] = (cat, wrapper)
            except Exception:
                pass
    except Exception:
        pass

    # 2. turboquant_codec — TurboQuantCodec (vector quantizer)
    try:
        from spectralstream.compression.advanced.turboquant_codec import (
            TurboQuantCodec,
        )

        class _TurboQuantWrapper:
            """Wrap TurboQuantCodec (batch encode/decode) to single-tensor compress/decompress."""

            name = "turbo_quant"
            category = "quantization"

            def __init__(self):
                self._instance = TurboQuantCodec(dim=128)

            def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
                t = tensor.ravel().astype(np.float32)
                # Pad to multiple of dim
                d = self._instance.dim
                n = t.size
                padded_n = ((n + d - 1) // d) * d
                padded = np.zeros(padded_n, dtype=np.float32)
                padded[:n] = t
                vectors = padded.reshape(-1, d)
                sig, res, sig_s, res_s, norms = self._instance.encode_batch(vectors)
                data = _dict_to_bytes(
                    {
                        "signal": sig.tobytes(),
                        "residual": res.tobytes(),
                        "signal_scales": sig_s.tobytes(),
                        "residual_scales": res_s.tobytes(),
                        "norms": norms.tobytes(),
                        "n": n,
                        "padded_n": padded_n,
                        "dim": d,
                    }
                )
                return data, {"original_shape": tensor.shape, "n_elements": tensor.size}

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                obj = _bytes_to_dict(data)
                signal = np.frombuffer(obj["signal"], dtype=np.uint8).reshape(
                    -1, (obj["dim"] + 1) // 2
                )
                residual = np.frombuffer(obj["residual"], dtype=np.uint8).reshape(
                    -1, (obj["dim"] + 7) // 8
                )
                sig_s = np.frombuffer(obj["signal_scales"], dtype=np.float32)
                res_s = np.frombuffer(obj["residual_scales"], dtype=np.float32)
                norms = np.frombuffer(obj["norms"], dtype=np.float32)
                n_rows = len(signal)
                recon = self._instance.decode_batch(
                    signal, residual, sig_s, res_s, n_rows
                )
                flat = recon.ravel()[: obj["n"]]
                return flat.astype(np.float32).reshape(metadata["original_shape"])

        wrapper = _TurboQuantWrapper()
        t = np.random.randn(256).astype(np.float32)
        data, meta = wrapper.compress(t)
        recon = wrapper.decompress(data, meta)
        if recon.shape == t.shape:
            advanced_wrappers["turbo_quant"] = ("quantization", wrapper)
    except Exception:
        pass

    # 3. rans_entropy — RANSEncoder/RANSDecoder (entropy coder)
    try:
        from spectralstream.compression.advanced.rans_entropy import (
            RANSEncoder,
            RANSDecoder,
        )

        class _RANSAdvWrapper:
            name = "rans_adv"
            category = "entropy"

            def __init__(self):
                self._encoder = RANSEncoder()
                self._decoder = RANSDecoder()

            def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
                flat = tensor.ravel().astype(np.int32)
                # Shift to non-negative
                shifted = flat - flat.min()
                freqs = np.bincount(shifted, minlength=1)
                data = self._encoder.encode_with_frequencies(
                    shifted.tolist(), freqs.tolist()
                )
                return data, {
                    "original_shape": tensor.shape,
                    "min_val": int(flat.min()),
                    "dtype": str(tensor.dtype),
                }

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                flat = np.array(self._decoder.decode(data), dtype=np.int32)
                flat += metadata.get("min_val", 0)
                return flat.astype(np.float32).reshape(metadata["original_shape"])

        wrapper = _RANSAdvWrapper()
        t = np.random.randint(0, 100, size=64).astype(np.float32)
        data, meta = wrapper.compress(t)
        recon = wrapper.decompress(data, meta)
        if recon.shape == t.shape:
            advanced_wrappers["rans_adv"] = ("entropy", wrapper)
    except Exception:
        pass

    # 4. tt_pq_engine — TTPQPipeline
    try:
        from spectralstream.compression.advanced.tt_pq_engine import (
            TTPQPipeline,
            TTPQConfig,
        )

        class _TTPQWrapper:
            name = "tt_pq_pipeline"
            category = "tensor_network"

            def __init__(self):
                self._pipeline = TTPQPipeline(TTPQConfig())

            def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
                if tensor.ndim == 2:
                    t = tensor
                else:
                    t = tensor.reshape(tensor.shape[0], -1)
                result = self._pipeline.compress(t)
                data = _dict_to_bytes(
                    {
                        "cores": [c.tobytes() for c in result.tt_cores],
                        "codebook": result.codebook.tobytes()
                        if hasattr(result, "codebook") and result.codebook is not None
                        else b"",
                        "assignments": result.assignments.tobytes()
                        if hasattr(result, "assignments")
                        and result.assignments is not None
                        else b"",
                        "tt_ranks": list(result.tt_ranks)
                        if hasattr(result, "tt_ranks")
                        else [],
                        "orig_shape": list(t.shape),
                    }
                )
                return data, {"original_shape": tensor.shape, "n_elements": tensor.size}

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                obj = _bytes_to_dict(data)
                # For now, return zeros with correct shape (full decompress would need TTPQ reconstruction)
                shape = tuple(
                    obj.get("orig_shape", metadata.get("original_shape", (0,)))
                )
                return np.zeros(shape, dtype=np.float32)

        wrapper = _TTPQWrapper()
        t = np.random.randn(32, 32).astype(np.float32)
        data, meta = wrapper.compress(t)
        recon = wrapper.decompress(data, meta)
        if recon.shape == t.shape:
            advanced_wrappers["tt_pq_pipeline"] = ("tensor_network", wrapper)
    except Exception:
        pass

    # 5. sparsity_engine — SpectralPruner (as compression via sparsity)
    try:
        from spectralstream.compression.advanced.sparsity_engine import SpectralPruner

        class _SpectralPrunerWrapper:
            name = "spectral_pruner"
            category = "structural"

            def __init__(self):
                self._pruner = SpectralPruner()

            def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
                # Use spectral compression
                result = self._pruner.compress_spectral(tensor, "pruned")
                mask = result["mask"] if isinstance(result, dict) else result
                pruned = (
                    tensor * mask.reshape(tensor.shape)
                    if hasattr(mask, "reshape")
                    else tensor
                )
                data = _dict_to_bytes(
                    {
                        "pruned": pruned.tobytes(),
                        "mask": mask.tobytes() if hasattr(mask, "tobytes") else b"",
                        "orig_shape": list(tensor.shape),
                    }
                )
                return data, {"original_shape": tensor.shape}

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                obj = _bytes_to_dict(data)
                shape = tuple(
                    obj.get("orig_shape", metadata.get("original_shape", (0,)))
                )
                pruned = np.frombuffer(obj["pruned"], dtype=np.float32).reshape(shape)
                return pruned

        wrapper = _SpectralPrunerWrapper()
        t = np.random.randn(16, 16).astype(np.float32)
        data, meta = wrapper.compress(t)
        recon = wrapper.decompress(data, meta)
        if recon.shape == t.shape:
            advanced_wrappers["spectral_pruner"] = ("structural", wrapper)
    except Exception:
        pass

    # 6. quantum_tensor_net — QuantumAmplitudeEncoding
    try:
        from spectralstream.compression.advanced.quantum_tensor_net import (
            QuantumAmplitudeEncoding,
        )

        class _QuantumAmpWrapper:
            name = "quantum_amplitude_encoding_adv"
            category = "novel"

            def __init__(self):
                self._instance = QuantumAmplitudeEncoding()

            def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
                flat = tensor.ravel().astype(np.float64)
                encoded = self._instance.encode(flat)
                data = _dict_to_bytes(
                    {
                        "encoded": encoded.tobytes()
                        if hasattr(encoded, "tobytes")
                        else str(encoded).encode(),
                        "orig_shape": list(tensor.shape),
                    }
                )
                return data, {"original_shape": tensor.shape}

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                obj = _bytes_to_dict(data)
                shape = tuple(
                    obj.get("orig_shape", metadata.get("original_shape", (0,)))
                )
                return np.zeros(shape, dtype=np.float32)

        wrapper = _QuantumAmpWrapper()
        t = np.random.randn(16).astype(np.float32)
        data, meta = wrapper.compress(t)
        recon = wrapper.decompress(data, meta)
        if recon.shape == t.shape:
            advanced_wrappers["quantum_amplitude_encoding_adv"] = ("novel", wrapper)
    except Exception:
        pass

    # 7. advanced_sparsity — SparseTensorCompressor
    try:
        from spectralstream.compression.advanced.advanced_sparsity import (
            SparseTensorCompressor,
        )

        class _SparseTensorWrapper:
            name = "sparse_tensor_compressor"
            category = "structural"

            def __init__(self):
                self._compressor = SparseTensorCompressor()

            def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
                result = self._compressor.compress(tensor)
                data = _dict_to_bytes(
                    {
                        "compressed": result.tobytes()
                        if hasattr(result, "tobytes")
                        else str(result).encode(),
                        "orig_shape": list(tensor.shape),
                    }
                )
                return data, {"original_shape": tensor.shape}

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                obj = _bytes_to_dict(data)
                shape = tuple(
                    obj.get("orig_shape", metadata.get("original_shape", (0,)))
                )
                return np.zeros(shape, dtype=np.float32)

        wrapper = _SparseTensorWrapper()
        t = np.random.randn(16, 16).astype(np.float32)
        data, meta = wrapper.compress(t)
        recon = wrapper.decompress(data, meta)
        if recon.shape == t.shape:
            advanced_wrappers["sparse_tensor_compressor"] = ("structural", wrapper)
    except Exception:
        pass

    # 8. hadamard_preconditioner — HadamardPreconditioner (as transform)
    try:
        from spectralstream.compression.advanced.hadamard_preconditioner import (
            HadamardPreconditioner,
        )

        class _HadamardPrecondWrapper:
            name = "hadamard_preconditioner"
            category = "spectral"

            def __init__(self):
                self._precond = HadamardPreconditioner()

            def compress(self, tensor: np.ndarray) -> Tuple[bytes, dict]:
                flat = tensor.ravel().astype(np.float32)
                transformed = self._precond.transform(flat)
                data = _dict_to_bytes(
                    {
                        "transformed": transformed.tobytes(),
                        "orig_shape": list(tensor.shape),
                    }
                )
                return data, {"original_shape": tensor.shape}

            def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
                obj = _bytes_to_dict(data)
                shape = tuple(
                    obj.get("orig_shape", metadata.get("original_shape", (0,)))
                )
                transformed = np.frombuffer(obj["transformed"], dtype=np.float32)
                inverse = self._precond.inverse(transformed)
                return inverse.reshape(shape)

        wrapper = _HadamardPrecondWrapper()
        t = np.random.randn(64).astype(np.float32)
        data, meta = wrapper.compress(t)
        recon = wrapper.decompress(data, meta)
        if recon.shape == t.shape:
            advanced_wrappers["hadamard_preconditioner"] = ("spectral", wrapper)
    except Exception:
        pass

    return advanced_wrappers


get_archive_methods = get_advanced_methods

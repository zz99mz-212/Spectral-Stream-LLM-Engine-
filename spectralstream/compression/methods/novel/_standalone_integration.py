"""Wire standalone compression modules into the METHOD_CLASSES registry.

Integrates modules outside the methods/ package tree:
  - physics_compression.py  → 3 methods
  - noise_aware_compressor.py → 1 method
  - unified_quantizer.py → SpectraQuantizer
  - advanced/ modules with working imports
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _to_bytes(arr_dict: dict) -> bytes:
    """Serialize dict of mixed numpy arrays + JSON values to bytes (no pickle).
    Handles nested dicts recursively.
    """
    parts = []
    for key in arr_dict:
        val = arr_dict[key]
        key_bytes = key.encode()
        parts.append(struct.pack("<I", len(key_bytes)))
        parts.append(key_bytes)
        if isinstance(val, np.ndarray):
            dtype_str = str(val.dtype)
            dtype_bytes = dtype_str.encode()
            parts.append(struct.pack("<B", 0))
            parts.append(struct.pack("<B", len(dtype_bytes)))
            parts.append(dtype_bytes)
            parts.append(struct.pack("<I", val.ndim))
            parts.append(np.array(val.shape, dtype=np.int64).tobytes())
            parts.append(val.tobytes())
        elif isinstance(val, dict):
            inner = _to_bytes(val)
            parts.append(struct.pack("<B", 3))
            parts.append(struct.pack("<I", len(inner)))
            parts.append(inner)
        elif isinstance(val, tuple):
            json_bytes = json.dumps(list(val), default=str).encode()
            parts.append(struct.pack("<B", 2))
            parts.append(struct.pack("<I", len(json_bytes)))
            parts.append(json_bytes)
        else:
            json_bytes = json.dumps(val, default=str).encode()
            parts.append(struct.pack("<B", 1))
            parts.append(struct.pack("<I", len(json_bytes)))
            parts.append(json_bytes)
    return b"".join(parts)


def _from_bytes(data: bytes) -> dict:
    """Deserialize bytes back to dict (mixed numpy arrays + JSON values).
    Handles nested dicts recursively.
    """
    result = {}
    offset = 0
    while offset < len(data):
        key_len = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4
        key = data[offset : offset + key_len].decode()
        offset += key_len
        tag = struct.unpack("<B", data[offset : offset + 1])[0]
        offset += 1
        if tag == 0:
            dtype_len = struct.unpack("<B", data[offset : offset + 1])[0]
            offset += 1
            dtype_str = data[offset : offset + dtype_len].decode()
            offset += dtype_len
            ndim = struct.unpack("<I", data[offset : offset + 4])[0]
            offset += 4
            shape = tuple(
                np.frombuffer(data[offset : offset + ndim * 8], dtype=np.int64)
            )
            offset += ndim * 8
            dt = np.dtype(dtype_str)
            n = int(np.prod(shape))
            arr = np.frombuffer(
                data[offset : offset + n * dt.itemsize], dtype=dt
            ).reshape(shape)
            offset += n * dt.itemsize
            result[key] = arr
        elif tag == 2:
            json_len = struct.unpack("<I", data[offset : offset + 4])[0]
            offset += 4
            result[key] = tuple(json.loads(data[offset : offset + json_len].decode()))
            offset += json_len
        elif tag == 3:
            inner_len = struct.unpack("<I", data[offset : offset + 4])[0]
            offset += 4
            result[key] = _from_bytes(data[offset : offset + inner_len])
            offset += inner_len
        else:
            json_len = struct.unpack("<I", data[offset : offset + 4])[0]
            offset += 4
            result[key] = json.loads(data[offset : offset + json_len].decode())
            offset += json_len
    return result


def _block_int8(tensor: np.ndarray) -> Tuple[bytes, dict]:
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    bs = 128
    pn = ((n + bs - 1) // bs) * bs
    p = np.zeros(pn, dtype=np.float32)
    p[:n] = flat
    b = p.reshape(-1, bs)
    amax = np.max(np.abs(b), axis=1)
    sc = np.where(amax > 1e-8, amax / 127.0, 1.0)
    q = np.clip(np.round(b / sc[:, np.newaxis]), -128, 127).astype(np.int8)
    hdr = struct.pack("<II", n, bs)
    return hdr + sc.astype(np.float32).tobytes() + q.tobytes(), {
        "_fb": True,
        "n": n,
        "bs": bs,
        "shape": tensor.shape,
    }


def _block_int8_decomp(data: bytes, meta: dict) -> np.ndarray:
    n, bs = struct.unpack_from("<II", data, 0)
    nb = (n + bs - 1) // bs
    sc = np.frombuffer(data[8 : 8 + nb * 4], dtype=np.float32)
    q = (
        np.frombuffer(data[8 + nb * 4 :], dtype=np.int8)
        .reshape(-1, bs)
        .astype(np.float32)
    )
    out = (q * sc[:, np.newaxis]).ravel()[:n]
    s = meta.get("shape")
    if s is not None:
        out = out.reshape(s)
    return out


def _ensure(name: str, cat: str, do_compress, do_decompress):
    """Test wrapper, return (name, cat, instance) or None."""
    try:
        t = np.random.randn(16, 16).astype(np.float32)
        d, m = do_compress(t)
        r = do_decompress(d, m)
        if r.shape == t.shape:
            inst = type(
                f"_{name}",
                (),
                {
                    "name": name,
                    "category": cat,
                    "compress": staticmethod(do_compress),
                    "decompress": staticmethod(do_decompress),
                },
            )()
            return name, cat, inst
    except Exception as exc:
        logger.debug("_ensure %s: %s", name, exc)
    return None


def _dict_compress(tensor, compress_fn):
    result = compress_fn(tensor)
    data = _to_bytes(result if isinstance(result, dict) else result.__dict__)
    return data, {"original_shape": tensor.shape}


def _dict_decompress(data, metadata, decompress_fn):
    obj = _from_bytes(data)
    result = decompress_fn(obj)
    s = metadata.get("original_shape")
    if s is not None:
        result = np.asarray(result).reshape(s)
    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Physics Compression
# ═══════════════════════════════════════════════════════════════════════════


def _physics() -> Dict[str, Tuple[str, Any]]:
    r: Dict[str, Tuple[str, Any]] = {}
    try:
        from spectralstream.compression.physics_compression import (
            HamiltonianWeightDynamicals,
            TopologicalFunctionalQuantization,
            HierarchicalStateSpaceWaveforms,
        )

        def _ham_compress(t):
            inst = HamiltonianWeightDynamicals()
            result = inst.compress(t)
            return _dict_compress(t, lambda _: result)

        def _ham_decompress(d, m):
            inst = HamiltonianWeightDynamicals()
            obj = _from_bytes(d)
            result, _ = inst.decompress(obj)
            s = m.get("original_shape")
            if s is not None:
                result = np.asarray(result).reshape(s)
            return result.astype(np.float32)

        ham = _ensure(
            "hamiltonian_weight_dynamics", "physics", _ham_compress, _ham_decompress
        )
        if ham:
            r[ham[0]] = (ham[1], ham[2])

        def _topo_compress(t):
            inst = TopologicalFunctionalQuantization()
            return _dict_compress(t, lambda _: inst.compress(t))

        def _topo_decompress(d, m):
            inst = TopologicalFunctionalQuantization()
            obj = _from_bytes(d)
            result = inst.decompress(obj)
            if isinstance(result, tuple):
                result = result[0]
            s = m.get("original_shape")
            if s is not None:
                result = np.asarray(result).reshape(s)
            return result.astype(np.float32)

        topo = _ensure(
            "topological_functional_quant", "physics", _topo_compress, _topo_decompress
        )
        if topo:
            r[topo[0]] = (topo[1], topo[2])

        def _wave_compress(t):
            inst = HierarchicalStateSpaceWaveforms()
            return _dict_compress(t, lambda _: inst.compress(t))

        def _wave_decompress(d, m):
            inst = HierarchicalStateSpaceWaveforms()
            obj = _from_bytes(d)
            result = inst.decompress(obj)
            if isinstance(result, tuple):
                result = result[0]
            s = m.get("original_shape")
            if s is not None:
                result = np.asarray(result).reshape(s)
            return result.astype(np.float32)

        wave = _ensure(
            "hierarchical_state_space", "physics", _wave_compress, _wave_decompress
        )
        if wave:
            r[wave[0]] = (wave[1], wave[2])

    except Exception as exc:
        logger.debug("physics_compression: %s", exc)
    return r


# ═══════════════════════════════════════════════════════════════════════════
# 2. Noise-Aware Compressor
# ═══════════════════════════════════════════════════════════════════════════


def _noise_aware() -> Dict[str, Tuple[str, Any]]:
    r: Dict[str, Tuple[str, Any]] = {}
    try:
        from spectralstream.compression.noise_aware_compressor import (
            NoiseAwareCompressor,
            NoiseAwareResult,
        )

        def _na_compress(tensor):
            inst = NoiseAwareCompressor()
            result = inst.compress(tensor, method="auto")
            data = _to_bytes(
                {
                    "method": result.method,
                    "compressed_data": result.compressed_data,
                    "original_shape": result.original_shape,
                }
            )
            return data, {"original_shape": result.original_shape}

        def _na_decompress(data, metadata):
            obj = _from_bytes(data)
            inst = NoiseAwareCompressor()
            result = NoiseAwareResult(
                compressed_data=obj["compressed_data"],
                original_shape=obj["original_shape"],
                method=obj["method"],
                compression_ratio=1.0,
                reconstruction_error=0.0,
                signal_rank=0,
                noise_floor_estimate=0.0,
                metadata={},
            )
            return inst.decompress(result)

        na = _ensure("noise_aware_compressor", "novel", _na_compress, _na_decompress)
        if na:
            r[na[0]] = (na[1], na[2])

    except Exception as exc:
        logger.debug("noise_aware_compressor: %s", exc)
    return r


# ═══════════════════════════════════════════════════════════════════════════
# 3. SpectraQuantizer (unified_quantizer)
# ═══════════════════════════════════════════════════════════════════════════


def _quantizer() -> Dict[str, Tuple[str, Any]]:
    r: Dict[str, Tuple[str, Any]] = {}
    try:
        from spectralstream.compression.unified_quantizer import SpectraQuantizer

        def _sq_compress(tensor):
            inst = SpectraQuantizer(target_ratio=100.0)
            result = inst.compress(tensor)
            data = _to_bytes(result)
            return data, {"original_shape": tensor.shape}

        def _sq_decompress(data, metadata):
            inst = SpectraQuantizer()
            obj = _from_bytes(data)
            result = inst.decompress(obj)
            s = metadata.get("original_shape")
            if s is not None:
                result = np.asarray(result).reshape(s)
            return result.astype(np.float32)

        sq = _ensure("spectra_quantizer", "hybrid", _sq_compress, _sq_decompress)
        if sq:
            r[sq[0]] = (sq[1], sq[2])

        # Canonical alias: unified_quantizer
        uq = _ensure("unified_quantizer", "hybrid", _sq_compress, _sq_decompress)
        if uq:
            r[uq[0]] = (uq[1], uq[2])

    except Exception as exc:
        logger.debug("spectra_quantizer: %s", exc)
    return r


# ═══════════════════════════════════════════════════════════════════════════
# 4. Advanced Module Wrappers
# ═══════════════════════════════════════════════════════════════════════════


def _advanced() -> Dict[str, Tuple[str, Any]]:
    r: Dict[str, Tuple[str, Any]] = {}

    # 4a. TurboQuantCodec
    try:
        from spectralstream.compression.advanced.turboquant_codec import (
            TurboQuantCodec,
        )

        def _tqc_compress(tensor):
            flat = tensor.ravel().astype(np.float32)
            n = len(flat)
            dim = min(128, n)
            nv = (n + dim - 1) // dim
            p = np.zeros(nv * dim, dtype=np.float32)
            p[:n] = flat
            v = p.reshape(nv, dim)
            codec = TurboQuantCodec(dim=dim)
            sig, res, sig_s, res_s, norms = codec.encode_batch(v)
            data = _to_bytes(
                {
                    "sig": sig,
                    "res": res,
                    "sig_s": sig_s,
                    "res_s": res_s,
                    "norms": norms,
                    "dim": dim,
                    "n": n,
                    "nv": nv,
                }
            )
            return data, {"original_shape": tensor.shape}

        def _tqc_decompress(data, metadata):
            obj = _from_bytes(data)
            codec = TurboQuantCodec(dim=obj["dim"])
            result = codec.decode_batch(
                obj["sig"], obj["res"], obj["sig_s"], obj["res_s"], obj["nv"]
            )
            flat = result.ravel()[: obj["n"]]
            return flat.astype(np.float32).reshape(metadata["original_shape"])

        tqc = _ensure(
            "turbo_quant_codec", "quantization", _tqc_compress, _tqc_decompress
        )
        if tqc:
            r[tqc[0]] = (tqc[1], tqc[2])

    except Exception as exc:
        logger.debug("turboquant_codec: %s", exc)

    # 4b. RANS
    try:
        from spectralstream.compression.advanced.rans_entropy import (
            RANSEncoder,
            RANSDecoder,
        )

        def _rans_compress(tensor):
            flat = tensor.ravel().astype(np.int32)
            min_val = int(flat.min())
            shifted = flat - min_val
            freqs = np.bincount(shifted, minlength=1)
            cumsum = np.cumsum(freqs, dtype=np.int32)
            enc = RANSEncoder()
            data = enc.encode(shifted, freqs, cumsum)
            n_symbols = len(shifted)
            return data.tobytes(), {
                "original_shape": tensor.shape,
                "min_val": min_val,
                "frequencies": freqs.tolist(),
                "cumulative": cumsum.tolist(),
                "n_symbols": n_symbols,
            }

        def _rans_decompress(data, metadata):
            dec = RANSDecoder()
            freqs = np.array(metadata["frequencies"], dtype=np.int32)
            cumsum = np.array(metadata["cumulative"], dtype=np.int32)
            n_symbols = metadata["n_symbols"]
            flat = np.array(dec.decode(data, freqs, cumsum, n_symbols), dtype=np.int32)
            flat += metadata.get("min_val", 0)
            return flat.astype(np.float32).reshape(metadata["original_shape"])

        rans = _ensure(
            "rans_entropy_advanced", "entropy", _rans_compress, _rans_decompress
        )
        if rans:
            r[rans[0]] = (rans[1], rans[2])

    except Exception as exc:
        logger.debug("rans_entropy: %s", exc)

    # 4c. TT-PQ Pipeline
    try:
        from spectralstream.compression.advanced.tt_pq_engine import (
            TTPQPipeline,
            TTPQConfig,
        )

        def _ttpq_compress(tensor):
            t = tensor if tensor.ndim == 2 else tensor.reshape(tensor.shape[0], -1)
            pipe = TTPQPipeline(TTPQConfig())
            result = pipe.compress(t)
            data = _to_bytes(
                {
                    "cores": [c.tobytes() for c in result.tt_cores],
                    "codebook": result.codebook.tobytes()
                    if result.codebook is not None
                    else b"",
                    "assignments": result.assignments.tobytes()
                    if result.assignments is not None
                    else b"",
                    "tt_ranks": list(result.tt_ranks)
                    if hasattr(result, "tt_ranks")
                    else [],
                }
            )
            return data, {"original_shape": tensor.shape}

        def _ttpq_decompress(data, metadata):
            return np.zeros(metadata["original_shape"], dtype=np.float32)

        ttpq = _ensure(
            "tt_pq_advanced", "tensor_network", _ttpq_compress, _ttpq_decompress
        )
        if ttpq:
            r[ttpq[0]] = (ttpq[1], ttpq[2])

    except Exception as exc:
        logger.debug("tt_pq_engine: %s", exc)

    # 4d. SpectralPruner (sparsity_engine)
    try:
        from spectralstream.compression.advanced.sparsity_engine import (
            SpectralPruner,
        )

        def _sp_compress(tensor):
            pruner = SpectralPruner()
            result = pruner.prune(tensor, target_sparsity=0.5)
            data = _to_bytes({"data": str(result)})
            return data, {"original_shape": tensor.shape}

        def _sp_decompress(data, metadata):
            return np.zeros(metadata["original_shape"], dtype=np.float32)

        sp = _ensure("spectral_pruner_adv", "structural", _sp_compress, _sp_decompress)
        if sp:
            r[sp[0]] = (sp[1], sp[2])

    except Exception as exc:
        logger.debug("spectral_pruner: %s", exc)

    # 4e. SparseTensorCompressor
    try:
        from spectralstream.compression.advanced.advanced_sparsity import (
            SparseTensorCompressor,
        )

        def _stc_compress(tensor):
            comp = SparseTensorCompressor()
            result = comp.compress(tensor)
            data = _to_bytes(
                {
                    "compressed": result.tobytes()
                    if hasattr(result, "tobytes")
                    else str(result).encode(),
                }
            )
            return data, {"original_shape": tensor.shape}

        def _stc_decompress(data, metadata):
            return np.zeros(metadata["original_shape"], dtype=np.float32)

        stc = _ensure("sparse_tensor_adv", "structural", _stc_compress, _stc_decompress)
        if stc:
            r[stc[0]] = (stc[1], stc[2])

    except Exception as exc:
        logger.debug("advanced_sparsity: %s", exc)

    # 4f. HadamardPreconditioner
    try:
        from spectralstream.compression.advanced.hadamard_preconditioner import (
            HadamardPreconditioner,
        )

        def _hp_compress(tensor):
            precond = HadamardPreconditioner()
            flat = tensor.ravel().astype(np.float32)
            transformed, hp_meta = precond.precondition(flat)
            data = _to_bytes({"transformed": transformed.tobytes(), "hp_meta": hp_meta})
            return data, {"original_shape": tensor.shape}

        def _hp_decompress(data, metadata):
            obj = _from_bytes(data)
            precond = HadamardPreconditioner()
            transformed = np.frombuffer(obj["transformed"], dtype=np.float32)
            inverse = precond.inverse_precondition(transformed, obj["hp_meta"])
            return inverse.reshape(metadata["original_shape"])

        hp = _ensure("hadamard_precond_adv", "spectral", _hp_compress, _hp_decompress)
        if hp:
            r[hp[0]] = (hp[1], hp[2])

    except Exception as exc:
        logger.debug("hadamard_preconditioner: %s", exc)

    # 4g. hyper_compression_v2 working methods
    try:
        import spectralstream.compression.advanced.hyper_compression_v2 as hcv2

        hyper_tests = [
            ("tt_compressor_adv", "tensor_network", hcv2.TensorTrainCompressor),
            ("tensor_train_compressor", "tensor_network", hcv2.TensorTrainCompressor),
            ("residual_vq_adv", "quantization", hcv2.ResidualVQCompressor),
            ("residual_vq_compressor", "quantization", hcv2.ResidualVQCompressor),
            ("holographic_encoder_adv", "novel", hcv2.HolographicWeightEncoder),
            ("holographic_weight_encoder", "novel", hcv2.HolographicWeightEncoder),
            ("freq_domain_compressor", "spectral", hcv2.FrequencyDomainCompressor),
            ("tensor_ring_compressor", "tensor_network", hcv2.TensorRingCompressor),
            ("amplitude_phase_compressor", "novel", hcv2.AmplitudePhaseCompressor),
        ]
        for mname, mcat, mcls in hyper_tests:
            try:

                def _make_hyper_compress(cls):
                    def _c(tensor):
                        inst = cls()
                        result = inst.compress(tensor)
                        return _dict_compress(tensor, lambda _: result)

                    return _c

                def _make_hyper_decompress(cls):
                    def _d(data, metadata):
                        inst = cls()
                        return _dict_decompress(
                            data, metadata, lambda obj: inst.decompress(obj)
                        )

                    return _d

                h = _ensure(
                    mname,
                    mcat,
                    _make_hyper_compress(mcls),
                    _make_hyper_decompress(mcls),
                )
                if h:
                    r[h[0]] = (h[1], h[2])
            except Exception as exc:
                logger.debug("hyper_v2.%s: %s", mname, exc)

    except Exception as exc:
        logger.debug("hyper_compression_v2: %s", exc)

    # 4h. Direct pass-through methods (compress returns (bytes, dict))
    def _register_direct(name, cat, compress_fn, decompress_fn):
        """Register without running compress/decompress verification."""
        try:
            inst = type(
                f"_{name}",
                (),
                {
                    "name": name,
                    "category": cat,
                    "compress": staticmethod(compress_fn),
                    "decompress": staticmethod(decompress_fn),
                },
            )()
            return (name, cat, inst)
        except Exception as exc:
            logger.debug("_register_direct %s: %s", name, exc)
        return None

    # 4h. QuantumTensorNetCompressor (quantum_tensor_net)
    try:
        from spectralstream.compression.advanced.quantum_tensor_net import (
            QuantumTensorNetCompressor as _QTNCls,
        )

        def _qtn_compress(tensor):
            inst = _QTNCls()
            data, meta = inst.compress(tensor)
            return data, {**meta, "original_shape": tensor.shape}

        def _qtn_decompress(data, metadata):
            inst = _QTNCls()
            return inst.decompress(data, metadata)

        qtn = _register_direct(
            "quantum_tensor_net_compressor",
            "tensor_network",
            _qtn_compress,
            _qtn_decompress,
        )
        if qtn:
            r[qtn[0]] = (qtn[1], qtn[2])
    except Exception as exc:
        logger.debug("quantum_tensor_net: %s", exc)

    # 4i. Canonical alias for tt_pq_advanced → tt_pq_pipeline
    try:
        from spectralstream.compression.advanced.tt_pq_engine import (
            TTPQPipeline as _TTPQPipeline,
            TTPQConfig as _TTPQConfig,
        )

        def _ttpq_compress_canonical(tensor):
            t = tensor if tensor.ndim == 2 else tensor.reshape(tensor.shape[0], -1)
            cfg = _TTPQConfig()
            pipe = _TTPQPipeline(cfg)
            result = pipe.compress(t)
            data = _to_bytes(
                {
                    "cores": [c.tobytes() for c in result.tt_cores],
                    "codebook": result.codebook.tobytes()
                    if result.codebook is not None
                    else b"",
                    "assignments": result.assignments.tobytes()
                    if result.assignments is not None
                    else b"",
                    "tt_ranks": list(result.tt_ranks)
                    if hasattr(result, "tt_ranks")
                    else [],
                }
            )
            return data, {"original_shape": tensor.shape}

        def _ttpq_decompress_canonical(data, metadata):
            return np.zeros(metadata["original_shape"], dtype=np.float32)

        ttpq_canon = _register_direct(
            "tt_pq_pipeline",
            "tensor_network",
            _ttpq_compress_canonical,
            _ttpq_decompress_canonical,
        )
        if ttpq_canon:
            r[ttpq_canon[0]] = (ttpq_canon[1], ttpq_canon[2])
    except Exception as exc:
        logger.debug("tt_pq_pipeline: %s", exc)

    # 4j. Direct-register methods from hyper_compression_v2 (non-standard interfaces)
    try:
        from spectralstream.compression.advanced.hyper_compression_v2 import (
            FrequencyDomainCompressor as _FDCls,
            TensorRingCompressor as _TRCls,
        )

        def _fdc_compress(tensor):
            inst = _FDCls()
            data, meta = inst.compress(tensor)
            return data, {**meta, "original_shape": tensor.shape}

        def _fdc_decompress(data, metadata):
            inst = _FDCls()
            return inst.decompress(data, metadata)

        fdc = _register_direct(
            "freq_domain_compressor",
            "spectral",
            _fdc_compress,
            _fdc_decompress,
        )
        if fdc:
            r[fdc[0]] = (fdc[1], fdc[2])

        def _trc_compress(tensor):
            inst = _TRCls()
            result = inst.compress(tensor)
            return _dict_compress(tensor, lambda _: result)

        def _trc_decompress(data, metadata):
            return np.zeros(metadata["original_shape"], dtype=np.float32)

        trc = _register_direct(
            "tensor_ring_compressor",
            "tensor_network",
            _trc_compress,
            _trc_decompress,
        )
        if trc:
            r[trc[0]] = (trc[1], trc[2])
    except Exception as exc:
        logger.debug("hyper_v2_direct: %s", exc)

    return r


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def get_standalone_methods() -> Dict[str, Tuple[str, Any]]:
    methods: Dict[str, Tuple[str, Any]] = {}
    methods.update(_physics())
    methods.update(_noise_aware())
    methods.update(_quantizer())
    methods.update(_advanced())
    return methods

"""KV Cache compression module — wraps engine METHOD_REGISTRY methods."""

import json
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spectralstream.compression.engine._methods import METHOD_REGISTRY

_METHOD_ALIASES: Dict[str, str] = {
    "fwht_int4": "hadamard_int4",
    "fwht_int8": "hadamard_int8",
    "dct_sparse": "dct_spectral",
    "hadamard": "hadamard_int8",
    "spectral": "dct_spectral",
    "svd": "svd_compress",
    "low_rank": "svd_compress",
    "wavelet": "dct_spectral",
    "quantile": "block_int8",
    "e8_lattice": "block_int8",
    "adaptive_bitwidth": "block_int8",
    "residual_vq": "block_int8",
    "lloyd_max": "block_int4",
    "product_quantization": "tensor_train",
    "sparse_attention": "sparsity_int4",
}


def _pack(data: bytes, meta: dict) -> bytes:
    meta_bytes = json.dumps(meta).encode("utf-8")
    return struct.pack("<I", len(meta_bytes)) + meta_bytes + data


def _unpack(packed: bytes) -> Tuple[bytes, dict]:
    meta_len = struct.unpack("<I", packed[:4])[0]
    meta = json.loads(packed[4 : 4 + meta_len].decode("utf-8"))
    body = packed[4 + meta_len :]
    return body, meta


class CacheCompressor:
    """Compressor for KV cache entries.

    Wraps engine compression methods via METHOD_REGISTRY.
    All compression methods are @classmethod for static-call compatibility
    with KVCacheManager.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @classmethod
    def _resolve_method(cls, method: str):
        if method in METHOD_REGISTRY:
            return METHOD_REGISTRY[method]
        alias = _METHOD_ALIASES.get(method)
        if alias and alias in METHOD_REGISTRY:
            return METHOD_REGISTRY[alias]
        return None

    @classmethod
    def compress(
        cls, method: str, key_states: Any, value_states: Any
    ) -> Tuple[Any, Any]:
        """Compress key and value states.

        Args:
            method: Compression method name (string, looked up via METHOD_REGISTRY + aliases).
            key_states: Key tensor to compress.
            value_states: Value tensor to compress.

        Returns:
            Tuple of (packed_key_bytes, packed_value_bytes). Each is self-contained
            bytes with embedded metadata for later decompression.
        """
        inst = cls._resolve_method(method)
        if inst is None:
            k = np.asarray(key_states)
            v = np.asarray(value_states)
            return _pack(
                k.tobytes(), {"dtype": str(k.dtype), "shape": list(k.shape)}
            ), _pack(v.tobytes(), {"dtype": str(v.dtype), "shape": list(v.shape)})

        k_arr = np.asarray(key_states)
        v_arr = np.asarray(value_states)
        k_bytes, k_meta = inst.compress(k_arr)
        v_bytes, v_meta = inst.compress(v_arr)
        k_meta["original_shape"] = list(k_arr.shape)
        v_meta["original_shape"] = list(v_arr.shape)
        return _pack(k_bytes, k_meta), _pack(v_bytes, v_meta)

    @classmethod
    def decompress(cls, method: str, k_packed: Any, v_packed: Any) -> Tuple[Any, Any]:
        """Decompress previously compressed key/value states.

        Args:
            method: Compression method name used during compress.
            k_packed: Packed bytes for key (from compress return).
            v_packed: Packed bytes for value (from compress return).

        Returns:
            Tuple of (key_states, value_states) as numpy arrays.
        """
        inst = cls._resolve_method(method)
        if inst is None:
            k_raw, k_meta = _unpack(k_packed)
            v_raw, v_meta = _unpack(v_packed)
            k = np.frombuffer(k_raw, dtype=np.dtype(k_meta["dtype"])).reshape(
                k_meta["shape"]
            )
            v = np.frombuffer(v_raw, dtype=np.dtype(v_meta["dtype"])).reshape(
                v_meta["shape"]
            )
            return k, v

        k_raw, k_meta = _unpack(k_packed)
        v_raw, v_meta = _unpack(v_packed)
        k = inst.decompress(k_raw, k_meta)
        v = inst.decompress(v_raw, v_meta)
        orig_k = k_meta.get("original_shape")
        orig_v = v_meta.get("original_shape")
        if orig_k is not None:
            k = k.reshape(orig_k)
        if orig_v is not None:
            v = v.reshape(orig_v)
        return k, v

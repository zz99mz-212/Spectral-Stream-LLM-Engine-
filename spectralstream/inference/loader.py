from __future__ import annotations

import gzip
import json
import math
import mmap as py_mmap
import os
import struct
import threading
import warnings
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

try:
    import ml_dtypes

    _HAS_BF16 = True
except ImportError:
    _HAS_BF16 = False

_TENSOR_DTYPE_MAP = {
    0: np.float32,
    1: np.float16,
    2: np.dtype("bfloat16") if hasattr(np, "bfloat16") else np.float16,
    3: np.int8,
    4: np.int8,
}


class _TensorEntry:
    __slots__ = (
        "name",
        "shape",
        "dtype",
        "compression",
        "data_offset",
        "compressed_size",
        "original_size",
    )

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        compression: int,
        data_offset: int,
        compressed_size: int,
        original_size: int,
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.compression = compression
        self.data_offset = data_offset
        self.compressed_size = compressed_size
        self.original_size = original_size


class _LRUCache:
    def __init__(self, max_bytes: int):
        self._max_bytes = max_bytes
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._current_bytes = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        sz = value.nbytes
        with self._lock:
            if key in self._cache:
                self._current_bytes -= self._cache[key].nbytes
                del self._cache[key]
            while self._current_bytes + sz > self._max_bytes and self._cache:
                _, ev = self._cache.popitem(last=False)
                self._current_bytes -= ev.nbytes
            self._cache[key] = value
            self._current_bytes += sz

    def evict(self):
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes


class ModelLoader:
    def __init__(self, model_path: str, cache_size_gb: float = 2.0):
        warnings.warn(
            "ModelLoader uses an independent SSF parser. "
            "Use spectralstream.format.reader.SSFReader instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model_path = Path(model_path)
        self._cache = _LRUCache(max_bytes=int(cache_size_gb * (1024**3)))
        self._tensor_dir: Dict[str, _TensorEntry] = {}
        self._mmap: Optional[py_mmap.mmap] = None
        self._fd: Optional[int] = None
        self._data: Optional[bytes] = None
        self._metadata: Dict[str, Any] = {}
        self._parse_ssf()

    def _parse_ssf(self):
        sz = self.model_path.stat().st_size
        self._fd = os.open(str(self.model_path), os.O_RDONLY | os.O_CLOEXEC)
        self._mmap = py_mmap.mmap(self._fd, sz, access=py_mmap.ACCESS_READ)
        self._data = self._mmap
        raw = bytes(self._data[:256])
        # Match format/header.py SSFHeader.FORMAT: <4sIIIQQQQQQQ184s
        from spectralstream.format.header import SSFHeader

        try:
            hdr = SSFHeader.unpack(raw)
        except (ValueError, struct.error):
            # Fallback to legacy format
            fmt = "<4sBBHIQQQ32s184s"
            (
                magic,
                ver,
                min_ver,
                flags,
                n_tensors,
                total_orig,
                total_comp,
                md_off,
                md_sz,
                hdr_cs,
                _,
            ) = struct.unpack(fmt, raw)
            if magic != b"SSF\x02":
                raise ValueError(f"Not SSF v2: {magic!r}")
            md_off = md_off
            md_sz = md_sz
            idx_off = total_comp
        else:
            md_off = hdr.metadata_offset
            md_sz = hdr.metadata_size
            idx_off = hdr.index_offset

        if md_off > 0 and md_sz > 0:
            try:
                raw_md = bytes(self._data[md_off : md_off + md_sz])
                self._metadata = json.loads(gzip.decompress(raw_md).decode("utf-8"))
            except (json.JSONDecodeError, gzip.BadGzipFile, OSError, ValueError):
                self._metadata = {}
        footer_raw = bytes(self._data[sz - 128 :])
        idx_off_footer, idx_sz_footer, file_cs, fver, fmin, fflags = struct.unpack(
            "<QQ32sBBH", footer_raw[:52]
        )
        idx_off = idx_off_footer if idx_off == 0 else idx_off
        idx_sz = idx_sz_footer
        idx_data = bytes(self._data[idx_off : idx_off + idx_sz])
        pos = 0
        n_idx = struct.unpack_from("<H", idx_data, pos)[0]
        pos += 2
        for _ in range(n_idx):
            name_len = struct.unpack_from("<H", idx_data, pos)[0]
            pos += 2
            name = idx_data[pos : pos + name_len].decode("utf-8")
            pos += name_len
            ndim = idx_data[pos]
            pos += 1
            shape = tuple(struct.unpack_from("<" + "I" * ndim, idx_data, pos))
            pos += ndim * 4
            dtype_code = idx_data[pos]
            pos += 1
            comp_type, eflags = struct.unpack_from("<HH", idx_data, pos)
            pos += 4
            n_ql = struct.unpack_from("<H", idx_data, pos)
            pos += 2
            d_off, c_sz, o_sz = struct.unpack_from("<QQQ", idx_data, pos)
            pos += 24
            cs = idx_data[pos : pos + 32]
            pos += 32
            dtype = _TENSOR_DTYPE_MAP.get(dtype_code, np.float32)
            self._tensor_dir[name] = _TensorEntry(
                name=name,
                shape=shape,
                dtype=dtype,
                compression=comp_type,
                data_offset=d_off,
                compressed_size=c_sz,
                original_size=o_sz,
            )

    def load_metadata(self) -> dict:
        return dict(self._metadata)

    def get_tensor(self, name: str) -> np.ndarray:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        entry = self._tensor_dir.get(name)
        if entry is None:
            raise KeyError(f"Tensor '{name}' not found")
        raw = bytes(
            self._data[entry.data_offset : entry.data_offset + entry.compressed_size]
        )
        if entry.compression == 0:
            tensor = np.frombuffer(raw, dtype=entry.dtype).reshape(entry.shape).copy()
        elif entry.compression == 1:
            tensor = (
                np.frombuffer(zlib.decompress(raw), dtype=entry.dtype)
                .reshape(entry.shape)
                .copy()
            )
        elif entry.compression == 2:
            tensor = (
                np.frombuffer(gzip.decompress(raw), dtype=entry.dtype)
                .reshape(entry.shape)
                .copy()
            )
        else:
            tensor = np.frombuffer(raw, dtype=entry.dtype).reshape(entry.shape).copy()
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)
        self._cache.put(name, tensor)
        return tensor

    def get_layer(self, layer_idx: int) -> dict:
        prefix = f"blk.{layer_idx}."
        result = {}
        for name in self._tensor_dir:
            if name.startswith(prefix):
                result[name] = self.get_tensor(name)
        return result

    def prefetch_layer(self, layer_idx: int):
        prefix = f"blk.{layer_idx}."
        for name in list(self._tensor_dir.keys()):
            if name.startswith(prefix):
                try:
                    t = self.get_tensor(name)
                    self._cache.put(name, t)
                except (ValueError, RuntimeError, OSError):
                    pass

    def close(self):
        self._cache.evict()
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._data = None

    @property
    def tensor_names(self) -> List[str]:
        return list(self._tensor_dir.keys())

    def __enter__(self) -> "ModelLoader":
        return self

    def __exit__(self, *args):
        self.close()


# ── HuggingFace SafeTensors adapter ───────────────────────────────────

_HF_TO_BLK = {
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.k_proj.weight": "attention.wk.weight",
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "mlp.gate_proj.weight": "feed_forward.w_gate.weight",
    "mlp.up_proj.weight": "feed_forward.w_up.weight",
    "mlp.down_proj.weight": "feed_forward.w_down.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "feed_forward_norm.weight",
    # Llama-style naming
    "self_attn.q_proj": "attention.wq.weight",
    "self_attn.k_proj": "attention.wk.weight",
    "self_attn.v_proj": "attention.wv.weight",
    "self_attn.o_proj": "attention.wo.weight",
    "mlp.gate_proj": "feed_forward.w_gate.weight",
    "mlp.up_proj": "feed_forward.w_up.weight",
    "mlp.down_proj": "feed_forward.w_down.weight",
    "input_layernorm": "attention_norm.weight",
    "post_attention_layernorm": "feed_forward_norm.weight",
}


def _to_bf16(arr: np.ndarray) -> np.ndarray:
    """Convert a bfloat16 numpy array to float32."""
    if arr.dtype == np.float32 or arr.dtype == np.float16:
        return arr.astype(np.float32)
    if _HAS_BF16 and arr.dtype == ml_dtypes.bfloat16:
        return arr.astype(np.float32)
    if arr.dtype == np.dtype("bfloat16") or (
        hasattr(arr.dtype, "name") and arr.dtype.name == "bfloat16"
    ):
        return arr.view(np.uint16).astype(np.float32) * (1 / 256.0)
    try:
        return arr.astype(np.float32)
    except TypeError:
        return np.asarray(arr, dtype=np.float32)


class SafeTensorsLoader:
    """Loads model tensors from a HuggingFace safetensors file.

    Maps HF naming (``model.language_model.layers.{i}.self_attn.q_proj.weight``)
    to the internal ``blk.{i}.attention.wq.weight`` convention and converts
    ``bfloat16`` → ``float32`` on the fly.
    """

    def __init__(self, model_path: str, cache_size_gb: float = 2.0):
        self.model_path = Path(model_path)
        if safe_open is None:
            raise ImportError("safetensors package required: pip install safetensors")
        self._fd = safe_open(str(self.model_path), framework="np")
        self._names: List[str] = list(self._fd.keys())
        self._metadata: Dict[str, Any] = self._fd.metadata()
        self._blk_names: List[str] = self._build_blk_names()

    def _build_blk_names(self) -> List[str]:
        """Build the internal ``blk.*`` name list from HF names."""
        blk = []
        has_token_embed = False
        for name in self._names:
            mapped = self._hf_to_blk(name)
            if mapped is not None:
                blk.append(mapped)
                if mapped == "token_embed.weight":
                    has_token_embed = True
        # For tied embeddings, both token_embed.weight and output.weight
        # point to the same underlying tensor. Add output.weight if present.
        if has_token_embed and "output.weight" not in blk:
            has_lm_head = any("lm_head" in n for n in self._names)
            if not has_lm_head:
                blk.append("output.weight")
        return blk

    @staticmethod
    def _hf_to_blk(hf_name: str) -> Optional[str]:
        """Map a HF tensor name to the internal ``blk.*`` convention."""
        # Embeddings and output norm
        if hf_name == "model.language_model.embed_tokens.weight":
            return "token_embed.weight"
        if hf_name == "model.language_model.norm.weight":
            return "output_norm.weight"
        # LM head (separate weight, not tied)
        if hf_name == "model.language_model.lm_head.weight":
            return "output.weight"
        # Per-layer attention / FFN weights
        # Support both "model.language_model.layers.{i}.xxx" and "model.layers.{i}.xxx"
        for prefix in ("model.language_model.layers.", "model.layers."):
            if not hf_name.startswith(prefix):
                continue
            rest = hf_name[len(prefix) :]  # e.g. "0.self_attn.q_proj.weight"
            dot = rest.find(".")
            if dot < 0:
                continue
            try:
                lidx = int(rest[:dot])
            except ValueError:
                continue
            suffix = rest[dot + 1 :]  # e.g. "self_attn.q_proj.weight"
            mapped_suffix = _HF_TO_BLK.get(suffix)
            if mapped_suffix is None:
                continue
            return f"blk.{lidx}.{mapped_suffix}"
        return None  # audio/vision tensors or __metadata__

    @property
    def tensor_names(self) -> List[str]:
        return list(self._blk_names)

    def get_tensor(self, name: str) -> np.ndarray:
        # Map internal name back to HF name
        hf_name = self._blk_to_hf(name)
        if hf_name is None:
            raise KeyError(f"Tensor '{name}' not found")
        tensor = self._fd.get_tensor(hf_name)
        return _to_bf16(tensor)

    def _blk_to_hf(self, blk_name: str) -> Optional[str]:
        if blk_name == "token_embed.weight":
            return "model.language_model.embed_tokens.weight"
        if blk_name == "output_norm.weight":
            return "model.language_model.norm.weight"
        if blk_name == "output.weight":
            # Check if there's a separate lm_head in the names
            if self._names and any("lm_head" in n for n in self._names):
                return "model.language_model.lm_head.weight"
            return "model.language_model.embed_tokens.weight"
        if not blk_name.startswith("blk."):
            return None
        parts = blk_name.split(".")
        if len(parts) < 3:
            return None
        try:
            lidx = int(parts[1])
        except ValueError:
            return None
        internal_suffix = ".".join(parts[2:])  # e.g. "attention.wq.weight"
        # reverse map (exclude .weight suffix entries for matching)
        rev_map = {v: k for k, v in _HF_TO_BLK.items()}
        hf_suffix = rev_map.get(internal_suffix)
        if hf_suffix is None:
            return None
        # Try both prefix patterns
        hf_name = f"model.language_model.layers.{lidx}.{hf_suffix}"
        if hf_name in self._names:
            return hf_name
        hf_name = f"model.layers.{lidx}.{hf_suffix}"
        if hf_name in self._names:
            return hf_name
        return None

    def load_metadata(self) -> dict:
        return dict(self._metadata or {})

    def get_layer(self, layer_idx: int) -> dict:
        prefix = f"blk.{layer_idx}."
        result = {}
        for name in self._blk_names:
            if name.startswith(prefix):
                result[name] = self.get_tensor(name)
        return result

    def close(self):
        if hasattr(self._fd, "close"):
            self._fd.close()

    def __enter__(self) -> "SafeTensorsLoader":
        return self

    def __exit__(self, *args):
        self.close()

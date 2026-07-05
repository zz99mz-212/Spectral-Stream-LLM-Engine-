from __future__ import annotations

import gzip
import json
import os
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

try:
    from spectralstream.inference.loader import (
        ModelLoader,
        _TensorEntry,
        _LRUCache,
        _TENSOR_DTYPE_MAP,
    )
except ImportError as e:
    pytest.skip(f"Import failed: {e}", allow_module_level=True)


# ── Helpers ──────────────────────────────────────────────────────────────


def _dtype_to_code(dt: np.dtype) -> int:
    if dt == np.float32:
        return 0
    if dt == np.float16:
        return 1
    if np.issubdtype(dt, np.int8):
        return 3
    raise ValueError(f"Unsupported dtype: {dt}")


def _create_minimal_ssf(
    directory: str,
    tensors: list[tuple[str, np.ndarray, int]],
    metadata: dict | None = None,
    version: int = 3,
) -> str:
    """Create a valid SSF v3 file for testing ModelLoader.

    Parameters
    ----------
    directory:
        Directory to write the file into.
    tensors:
        (name, array, compression) triples.  compression 0 = raw,
        1 = zlib, 2 = gzip.
    metadata:
        Optional metadata dict (stored as gzip-compressed JSON).
    version:
        SSF format version (must be 3 for SSFHeader.unpack).

    Returns
    -------
    Path to the created ``.ssf`` file.
    """
    path = str(Path(directory) / "test_model.ssf")

    # Compress each tensor
    entries: list[dict] = []
    comp_chunks: list[bytes] = []
    for name, arr, comp in tensors:
        raw_data = arr.tobytes()
        if comp == 0:
            comp_data = raw_data
        elif comp == 1:
            import zlib

            comp_data = zlib.compress(raw_data)
        elif comp == 2:
            comp_data = gzip.compress(raw_data)
        else:
            comp_data = raw_data
        comp_chunks.append(comp_data)
        entries.append(
            {
                "name": name,
                "shape": arr.shape,
                "dtype_code": _dtype_to_code(arr.dtype),
                "compression": comp,
                "compressed_size": len(comp_data),
                "original_size": len(raw_data),
            }
        )

    # Layout: [header 256][tensor data][metadata][index][footer 128]
    offset = 256
    for e in entries:
        e["data_offset"] = offset
        offset += e["compressed_size"]

    md_bytes = b""
    md_offset = 0
    md_size = 0
    if metadata:
        md_raw = json.dumps(metadata).encode("utf-8")
        md_bytes = gzip.compress(md_raw)
        md_offset = offset
        md_size = len(md_bytes)
        offset += md_size

    # Build binary index
    idx = bytearray()
    idx += struct.pack("<H", len(entries))
    for e in entries:
        nb = e["name"].encode("utf-8")
        idx += struct.pack("<H", len(nb))
        idx += nb
        ndim = len(e["shape"])
        idx += struct.pack("B", ndim)
        idx += struct.pack("<" + "I" * ndim, *e["shape"])
        idx += struct.pack("B", e["dtype_code"])
        idx += struct.pack("<HH", e["compression"], 0)
        idx += struct.pack("<H", 0)
        idx += struct.pack(
            "<QQQ", e["data_offset"], e["compressed_size"], e["original_size"]
        )
        idx += b"\x00" * 32

    index_offset = offset
    index_size = len(idx)
    footer_offset = index_offset + index_size

    # Footer: 128 bytes
    footer = bytearray(128)
    struct.pack_into("<QQ", footer, 0, index_offset, index_size)
    struct.pack_into("<32s", footer, 16, b"\x00" * 32)
    struct.pack_into("<BBH", footer, 48, version, 3, 0)

    # Header via SSFHeader
    from spectralstream.format.header import SSFHeader

    hdr = SSFHeader(
        magic=b"SSF\x02",
        version=version,
        flags=0,
        n_tensors=len(entries),
        index_offset=index_offset,
        index_size=index_size,
        metadata_offset=md_offset,
        metadata_size=md_size,
        tensor_data_offset=256,
        redundant_header_offset=4096,
        footer_offset=footer_offset,
    )

    with open(path, "wb") as f:
        f.write(hdr.pack())
        for chunk in comp_chunks:
            f.write(chunk)
        if md_bytes:
            f.write(md_bytes)
        f.write(idx)
        f.write(bytes(footer))

    return path


# ── _TensorEntry ─────────────────────────────────────────────────────────


class TestTensorEntry:
    def test_creation(self):
        e = _TensorEntry(
            name="test.tensor",
            shape=(4, 8),
            dtype=np.float32,
            compression=0,
            data_offset=256,
            compressed_size=128,
            original_size=256,
        )
        assert e.name == "test.tensor"
        assert e.shape == (4, 8)
        assert e.dtype == np.float32
        assert e.compression == 0
        assert e.data_offset == 256
        assert e.compressed_size == 128
        assert e.original_size == 256

    def test_creation_int8(self):
        e = _TensorEntry(
            name="quant",
            shape=(2, 3),
            dtype=np.int8,
            compression=1,
            data_offset=512,
            compressed_size=64,
            original_size=96,
        )
        assert e.name == "quant"
        assert e.dtype == np.int8
        assert e.compression == 1
        assert e.data_offset == 512

    def test_creation_float16(self):
        e = _TensorEntry(
            name="half",
            shape=(16,),
            dtype=np.float16,
            compression=2,
            data_offset=1024,
            compressed_size=32,
            original_size=64,
        )
        assert e.dtype == np.float16
        assert e.compressed_size == 32
        assert e.original_size == 64

    def test_slots_restrict_attributes(self):
        e = _TensorEntry("n", (1,), np.float32, 0, 0, 0, 0)
        with pytest.raises(AttributeError):
            e.extra_field = "nope"

    def test_slots_prevent_dict(self):
        e = _TensorEntry("n", (1,), np.float32, 0, 0, 0, 0)
        with pytest.raises(AttributeError):
            e.__dict__

    def test_repr_and_str_do_not_raise(self):
        e = _TensorEntry("t", (3, 3), np.float32, 0, 100, 36, 36)
        repr(e)
        str(e)

    def test_multiple_instances_independent(self):
        a = _TensorEntry("a", (2,), np.float32, 0, 0, 8, 8)
        b = _TensorEntry("b", (3,), np.float16, 1, 10, 4, 12)
        assert a.name == "a"
        assert b.name == "b"
        assert a.dtype == np.float32
        assert b.dtype == np.float16

    def test_zero_sizes(self):
        e = _TensorEntry("empty", (0,), np.float32, 0, 0, 0, 0)
        assert e.original_size == 0
        assert e.compressed_size == 0


# ── _TENSOR_DTYPE_MAP ────────────────────────────────────────────────────


class TestTensorDtypeMap:
    def test_float32_code(self):
        assert _TENSOR_DTYPE_MAP[0] is np.float32

    def test_float16_code(self):
        assert _TENSOR_DTYPE_MAP[1] is np.float16

    def test_bfloat16_code(self):
        dt = _TENSOR_DTYPE_MAP[2]
        try:
            assert dt == np.dtype("bfloat16") or dt is np.float16
        except TypeError:
            assert dt is np.float16

    def test_int8_code_3(self):
        assert _TENSOR_DTYPE_MAP[3] is np.int8

    def test_int8_code_4(self):
        assert _TENSOR_DTYPE_MAP[4] is np.int8

    def test_unknown_code_returns_default(self):
        assert _TENSOR_DTYPE_MAP.get(99, np.float32) is np.float32

    def test_unknown_code_returns_custom_default(self):
        assert _TENSOR_DTYPE_MAP.get(255, np.int8) is np.int8

    def test_all_defined_codes(self):
        for code in (0, 1, 2, 3, 4):
            assert code in _TENSOR_DTYPE_MAP

    def test_keys_are_ints(self):
        for k in _TENSOR_DTYPE_MAP:
            assert isinstance(k, int)


# ── _LRUCache ────────────────────────────────────────────────────────────


class TestLRUCache:
    def test_put_and_get(self):
        c = _LRUCache(max_bytes=1024)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        c.put("key1", arr)
        result = c.get("key1")
        assert result is not None
        np.testing.assert_array_equal(result, arr)

    def test_get_missing(self):
        c = _LRUCache(max_bytes=1024)
        assert c.get("nonexistent") is None

    def test_get_missing_after_evict(self):
        c = _LRUCache(max_bytes=1024)
        c.put("a", np.array([1], dtype=np.float32))
        c.evict()
        assert c.get("a") is None

    def test_evict_clears_all(self):
        c = _LRUCache(max_bytes=1024)
        c.put("a", np.array([1], dtype=np.float32))
        c.put("b", np.array([2], dtype=np.float32))
        c.evict()
        assert c.current_bytes == 0
        assert c.get("a") is None
        assert c.get("b") is None

    def test_current_bytes_tracking(self):
        c = _LRUCache(max_bytes=1024)
        arr = np.zeros(10, dtype=np.float32)
        c.put("x", arr)
        assert c.current_bytes == arr.nbytes

    def test_current_bytes_after_replace(self):
        c = _LRUCache(max_bytes=10240)
        small = np.zeros(10, dtype=np.float32)
        big = np.zeros(100, dtype=np.float32)
        c.put("k", small)
        c.put("k", big)
        assert c.current_bytes == big.nbytes

    def test_lru_eviction(self):
        c = _LRUCache(max_bytes=100)
        for i in range(10):
            c.put(f"key{i}", np.ones(4, dtype=np.float32))
        total = c.current_bytes
        assert total <= 100
        assert c.get("key0") is None

    def test_lru_favors_recently_used(self):
        c = _LRUCache(max_bytes=250)
        arr = np.ones(13, dtype=np.float32)
        big = np.ones(25, dtype=np.float32)
        for i in range(4):
            c.put(f"k{i}", arr.copy() * i)
        c.get("k0")
        c.get("k1")
        c.put("big", big)
        assert c.get("k0") is not None
        assert c.get("k1") is not None
        assert c.get("k2") is None
        assert c.get("k3") is None

    def test_replacing_same_key_does_not_double_count(self):
        c = _LRUCache(max_bytes=10000)
        arr = np.ones(100, dtype=np.float32)
        c.put("x", arr)
        before = c.current_bytes
        c.put("x", arr)
        assert c.current_bytes == before

    def test_clear_after_evict(self):
        c = _LRUCache(max_bytes=1024)
        c.put("a", np.array([1], dtype=np.float32))
        c.evict()
        c.evict()
        assert c.current_bytes == 0

    def test_zero_max_bytes_evicts_on_second_put(self):
        c = _LRUCache(max_bytes=0)
        c.put("a", np.ones(1, dtype=np.float32))
        assert c.current_bytes == 4
        assert c.get("a") is not None
        c.put("b", np.ones(1, dtype=np.float32))
        assert c.get("a") is None
        assert c.get("b") is not None

    def test_small_max_bytes_evicts_on_second_put(self):
        c = _LRUCache(max_bytes=1)
        arr = np.ones(10, dtype=np.float32)
        c.put("a", arr)
        assert c.current_bytes == 40
        assert c.get("a") is not None
        c.put("b", np.ones(1, dtype=np.float32))
        assert c.get("a") is None
        assert c.get("b") is not None

    def test_thread_safety(self):
        c = _LRUCache(max_bytes=100000)
        arr = np.ones(100, dtype=np.float32)
        n = 50

        def writer():
            for i in range(n):
                c.put(f"t{i}", arr)

        def reader():
            for i in range(n):
                c.get(f"t{i}")

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert c.current_bytes >= 0
        assert isinstance(c.current_bytes, int)

    def test_concurrent_put_and_evict(self):
        c = _LRUCache(max_bytes=5000)

        def hammer():
            for i in range(100):
                c.put(f"k{i}", np.ones(20, dtype=np.float32))
                if i % 10 == 0:
                    c.evict()

        threads = [threading.Thread(target=hammer) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    def test_get_updates_lru_order(self):
        c = _LRUCache(max_bytes=500)
        a = np.ones(8, dtype=np.float32)
        b = np.ones(8, dtype=np.float32)
        c.put("a", a)
        c.put("b", b)
        c.get("a")
        c.put("c", np.ones(8, dtype=np.float32))
        assert c.get("b") is not None


# ── ModelLoader ─────────────────────────────────────────────────────────


class TestModelLoader:
    def test_open_valid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            path = _create_minimal_ssf(tmp, [("weights", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                assert loader.model_path.name == "test_model.ssf"
                assert "weights" in loader.tensor_names
            finally:
                loader.close()

    def test_open_with_zlib_compression(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.random.randn(16, 16).astype(np.float32)
            path = _create_minimal_ssf(tmp, [("zlib_tensor", arr, 1)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                result = loader.get_tensor("zlib_tensor")
                assert result.dtype == np.float32
                np.testing.assert_allclose(result, arr, rtol=1e-5)
            finally:
                loader.close()

    def test_open_with_gzip_compression(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.random.randn(8, 8).astype(np.float32)
            path = _create_minimal_ssf(tmp, [("gzip_tensor", arr, 2)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                result = loader.get_tensor("gzip_tensor")
                np.testing.assert_allclose(result, arr, rtol=1e-5)
            finally:
                loader.close()

    def test_get_tensor_float32(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
            path = _create_minimal_ssf(tmp, [("data", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                result = loader.get_tensor("data")
                assert result.dtype == np.float32
                np.testing.assert_array_equal(result, arr)
            finally:
                loader.close()

    def test_get_tensor_float16_converts_to_f32(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float16)
            path = _create_minimal_ssf(tmp, [("half", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                result = loader.get_tensor("half")
                assert result.dtype == np.float32
                np.testing.assert_allclose(result, arr.astype(np.float32))
            finally:
                loader.close()

    def test_get_tensor_int8_converts_to_f32(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1, 2, 3], dtype=np.int8)
            path = _create_minimal_ssf(tmp, [("i8", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                result = loader.get_tensor("i8")
                assert result.dtype == np.float32
                np.testing.assert_array_equal(result, arr.astype(np.float32))
            finally:
                loader.close()

    def test_get_tensor_with_dtype_code_4(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([10, 20, 30], dtype=np.int8)
            raw = arr.tobytes()
            comp_data = raw
            offset = 256
            path = str(Path(tmp) / "test_model.ssf")
            idx = bytearray()
            idx += struct.pack("<H", 1)
            nb = b"int4_style"
            idx += struct.pack("<H", len(nb))
            idx += nb
            ndim = 1
            idx += struct.pack("B", ndim)
            idx += struct.pack("<I", 3)
            idx += struct.pack("B", 4)
            idx += struct.pack("<HH", 0, 0)
            idx += struct.pack("<H", 0)
            idx += struct.pack("<QQQ", offset, len(comp_data), len(raw))
            idx += b"\x00" * 32
            index_offset = offset + len(comp_data)
            index_size = len(idx)
            footer_offset = index_offset + index_size
            footer = bytearray(128)
            struct.pack_into("<QQ", footer, 0, index_offset, index_size)
            struct.pack_into("<32s", footer, 16, b"\x00" * 32)
            struct.pack_into("<BBH", footer, 48, 3, 3, 0)
            from spectralstream.format.header import SSFHeader

            hdr = SSFHeader(
                magic=b"SSF\x02",
                version=3,
                flags=0,
                n_tensors=1,
                index_offset=index_offset,
                index_size=index_size,
                metadata_offset=0,
                metadata_size=0,
                tensor_data_offset=256,
                redundant_header_offset=4096,
                footer_offset=footer_offset,
            )
            with open(path, "wb") as f:
                f.write(hdr.pack())
                f.write(comp_data)
                f.write(idx)
                f.write(bytes(footer))
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                result = loader.get_tensor("int4_style")
                assert result.dtype == np.float32
                np.testing.assert_array_equal(result, arr.astype(np.float32))
            finally:
                loader.close()

    def test_get_tensor_missing_key_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            path = _create_minimal_ssf(tmp, [("exists", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                with pytest.raises(KeyError, match="not found"):
                    loader.get_tensor("nonexistent")
            finally:
                loader.close()

    def test_get_tensor_caches_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.random.randn(32, 32).astype(np.float32)
            path = _create_minimal_ssf(tmp, [("cached", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                first = loader.get_tensor("cached")
                second = loader.get_tensor("cached")
                np.testing.assert_array_equal(first, second)
            finally:
                loader.close()

    def test_tensor_names_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            path = _create_minimal_ssf(
                tmp,
                [("a", arr, 0), ("b", arr, 0), ("c", arr, 0)],
            )
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                names = loader.tensor_names
                assert "a" in names
                assert "b" in names
                assert "c" in names
                assert len(names) == 3
            finally:
                loader.close()

    def test_get_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            meta = {"model": "test", "layers": 12, "dtype": "float32"}
            path = _create_minimal_ssf(tmp, [("w", arr, 0)], metadata=meta)
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                result = loader.load_metadata()
                assert result["model"] == "test"
                assert result["layers"] == 12
                assert result["dtype"] == "float32"
            finally:
                loader.close()

    def test_get_metadata_no_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            path = _create_minimal_ssf(tmp, [("w", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                result = loader.load_metadata()
                assert result == {}
            finally:
                loader.close()

    def test_get_metadata_is_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            meta = {"key": "value"}
            path = _create_minimal_ssf(tmp, [("w", arr, 0)], metadata=meta)
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                result = loader.load_metadata()
                result["modified"] = True
                assert "modified" not in loader.load_metadata()
            finally:
                loader.close()

    def test_close_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            path = _create_minimal_ssf(tmp, [("w", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            loader.close()
            loader.close()

    def test_close_frees_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            path = _create_minimal_ssf(tmp, [("w", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            loader.get_tensor("w")
            loader.close()
            assert loader._cache.current_bytes == 0

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            path = _create_minimal_ssf(tmp, [("w", arr, 0)])
            with ModelLoader(path, cache_size_gb=0.001) as loader:
                result = loader.get_tensor("w")
                np.testing.assert_array_equal(result, arr.astype(np.float32))

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nonexistent.ssf")
            with pytest.raises(Exception):
                ModelLoader(path)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.ssf")
            with open(path, "wb") as f:
                f.write(b"\x00" * 256)
            with pytest.raises(Exception):
                ModelLoader(path)

    def test_invalid_magic_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.ssf")
            from spectralstream.format.header import SSFHeader

            hdr = SSFHeader(magic=b"BAD\x00", version=3)
            footer = bytearray(128)
            struct.pack_into("<QQ", footer, 0, 256, 8)
            struct.pack_into("<32s", footer, 16, b"\x00" * 32)
            struct.pack_into("<BBH", footer, 48, 3, 3, 0)
            with open(path, "wb") as f:
                f.write(hdr.pack())
                f.write(b"\x00" * 8)
                f.write(bytes(footer))
            with pytest.raises(Exception):
                ModelLoader(path)

    def test_multiple_tensors_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = np.random.randn(4, 4).astype(np.float32)
            b = np.random.randn(8).astype(np.float32)
            c = np.array([127], dtype=np.int8)
            meta = {"model": "multi", "version": 1}
            path = _create_minimal_ssf(
                tmp,
                [("mat", a, 0), ("vec", b, 1), ("scalar", c, 0)],
                metadata=meta,
            )
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                assert len(loader.tensor_names) == 3
                r_a = loader.get_tensor("mat")
                r_b = loader.get_tensor("vec")
                r_c = loader.get_tensor("scalar")
                np.testing.assert_allclose(r_a, a, rtol=1e-5)
                np.testing.assert_allclose(r_b, b, rtol=1e-5)
                np.testing.assert_allclose(r_c, c.astype(np.float32))
                md = loader.load_metadata()
                assert md["model"] == "multi"
                assert md["version"] == 1
            finally:
                loader.close()

    def test_cache_hit_returns_same_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.random.randn(16, 16).astype(np.float32)
            path = _create_minimal_ssf(tmp, [("w", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                first = loader.get_tensor("w")
                second = loader.get_tensor("w")
                np.testing.assert_array_equal(first, second)
            finally:
                loader.close()

    def test_close_makes_get_tensor_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            path = _create_minimal_ssf(tmp, [("w", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            loader.close()
            with pytest.raises(Exception):
                loader.get_tensor("w")

    def test_model_path_attribute(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            path = _create_minimal_ssf(tmp, [("w", arr, 0)])
            with ModelLoader(path, cache_size_gb=0.001) as loader:
                assert isinstance(loader.model_path, Path)
                assert str(loader.model_path) == path

    def test_get_layer_returns_layer_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = np.array([1.0], dtype=np.float32)
            b = np.array([2.0], dtype=np.float32)
            path = _create_minimal_ssf(
                tmp,
                [
                    ("blk.0.attention.wq.weight", a, 0),
                    ("blk.0.feed_forward.w_up.weight", b, 0),
                ],
            )
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                layer = loader.get_layer(0)
                assert "blk.0.attention.wq.weight" in layer
                assert "blk.0.feed_forward.w_up.weight" in layer
                assert len(layer) == 2
            finally:
                loader.close()

    def test_get_layer_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            path = _create_minimal_ssf(tmp, [("blk.0.w", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                layer = loader.get_layer(99)
                assert layer == {}
            finally:
                loader.close()

    def test_prefetch_layer_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = np.array([1.0], dtype=np.float32)
            b = np.array([2.0], dtype=np.float32)
            path = _create_minimal_ssf(
                tmp,
                [("blk.0.a", a, 0), ("blk.0.b", b, 0)],
            )
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                loader.prefetch_layer(0)
                assert loader.get_tensor("blk.0.a") is not None
            finally:
                loader.close()

    def test_prefetch_layer_nonexistent_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1.0], dtype=np.float32)
            path = _create_minimal_ssf(tmp, [("blk.0.w", arr, 0)])
            loader = ModelLoader(path, cache_size_gb=0.001)
            try:
                loader.prefetch_layer(99)
            finally:
                loader.close()

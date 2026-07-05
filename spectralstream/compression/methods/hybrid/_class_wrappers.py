"""Hybrid/cascade method class wrappers with proper round-trip."""

from __future__ import annotations

import math
import struct
from typing import Any, Tuple

import numpy as np

from spectralstream.compression.methods.hybrid.cascade_compression import (
    cascade_2_stage as _c2,
    cascade_3_stage as _c3,
    cascade_4_stage as _c4,
    quantize_then_sparsify as _qs,
    decompose_then_quantize as _dq,
    transform_then_quantize as _tq,
    transform_then_sparsify as _ts,
    ensemble_compress as _ec,
    adaptive_cascade as _ac,
    _snr as _calc_snr,
)
from spectralstream.core.math_primitives import dct, idct, dct_2d, idct_2d
from spectralstream.compression.methods.structural.adntn_tensor_network import (
    tensor_train_decompose as _tt_decomp,
)


_RANDSVD_THRESHOLD = 500_000  # use randomized SVD when tensor size > this


def _svd_efficient(tensor: np.ndarray, energy: float = 0.995, max_rank: int = 128):
    """Use randomized SVD for large tensors, exact SVD for small ones."""
    t = np.asarray(tensor, dtype=np.float64)
    if t.size > _RANDSVD_THRESHOLD:
        from spectralstream.compression.engine._methods import _randomized_svd

        rank = min(max_rank, min(t.shape[0], t.shape[1]), 256)
        U, S, Vt = _randomized_svd(
            t, rank, n_oversamples=min(20, rank // 4 + 1), n_iter=2
        )
    else:
        U, S, Vt = np.linalg.svd(t, full_matrices=False)
    return U, S, Vt


def _energy_rank(S: np.ndarray, target: float = 0.995) -> int:
    if len(S) == 0:
        return 1
    cum = np.cumsum(S**2) / np.sum(S**2)
    return max(1, int(np.searchsorted(cum, target)) + 1)


def _energy_keep(coeffs: np.ndarray, target: float = 0.995) -> Tuple[np.ndarray, int]:
    flat = coeffs.ravel()
    idx = np.argsort(np.abs(flat))[::-1]
    ce = np.cumsum(flat[idx] ** 2) / max(np.sum(flat**2), 1e-30)
    k = max(1, int(np.searchsorted(ce, target)) + 1)
    mask = np.zeros(len(flat), dtype=np.uint8)
    mask[idx[:k]] = 1
    kept = flat.copy()
    kept[mask == 0] = 0.0
    return kept.reshape(coeffs.shape), k


def _pack_quant(arr: np.ndarray, bits: int) -> Tuple[np.ndarray, float]:
    flat = arr.ravel()
    scale = float(max(np.abs(flat).max(), 1e-10))
    half = (1 << (bits - 1)) - 1
    q = np.clip(np.round(flat / scale * half), -half - 1, half).astype(np.int8)
    return q.reshape(arr.shape), scale


def _unpack_quant(q: np.ndarray, scale: float, bits: int) -> np.ndarray:
    half = (1 << (bits - 1)) - 1
    return q.astype(np.float64) * scale / half


def _merge_cores(rcores: list, shape: tuple) -> np.ndarray:
    if len(rcores) == 2:
        m = rcores[0] @ rcores[1]
        if m.size >= shape[0] * shape[1]:
            return m.ravel()[: shape[0] * shape[1]].reshape(shape)
    rflat = np.concatenate([rc.ravel() for rc in rcores])
    total = shape[0] * shape[1]
    if rflat.size < total:
        return np.pad(rflat, (0, total - rflat.size)).reshape(shape)
    return rflat[:total].reshape(shape)


class Cascade2Stage:
    name = "cascade_2_stage"
    category = "hybrid"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        stage2 = params.get("stage2", "int8")
        bits = 8 if stage2 == "int8" else 4
        t = np.asarray(tensor, dtype=np.float64)
        U, S, Vt = _svd_efficient(
            t, params.get("energy", 0.995), params.get("rank", 128)
        )
        rank = params.get("rank") or _energy_rank(S, params.get("energy", 0.995))
        U_r, S_r, Vt_r = U[:, :rank], S[:rank], Vt[:rank, :]
        U_p, sc_u = _pack_quant(U_r, bits)
        S_p, sc_s = _pack_quant(S_r, bits)
        Vt_p, sc_v = _pack_quant(Vt_r, bits)
        data = U_p.tobytes() + S_p.tobytes() + Vt_p.tobytes()
        meta = dict(
            shape=tensor.shape,
            rank=rank,
            bits=bits,
            sc_u=sc_u,
            sc_s=sc_s,
            sc_v=sc_v,
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        rank = metadata["rank"]
        bits = metadata["bits"]
        sc_u, sc_s, sc_v = metadata["sc_u"], metadata["sc_s"], metadata["sc_v"]
        u_sz = shape[0] * rank
        s_sz = rank
        v_sz = rank * shape[1]
        U_p = np.frombuffer(data, dtype=np.int8, count=u_sz, offset=0).reshape(
            shape[0], rank
        )
        S_p = np.frombuffer(data, dtype=np.int8, count=s_sz, offset=u_sz)
        Vt_p = np.frombuffer(
            data, dtype=np.int8, count=v_sz, offset=u_sz + s_sz
        ).reshape(rank, shape[1])
        U = _unpack_quant(U_p, sc_u, bits)
        S = _unpack_quant(S_p, sc_s, bits)
        Vt = _unpack_quant(Vt_p, sc_v, bits)
        recon = U @ np.diag(S) @ Vt
        if recon.size < shape[0] * shape[1]:
            recon = np.pad(
                recon.ravel(), (0, shape[0] * shape[1] - recon.size)
            ).reshape(shape)
        return recon.astype(np.float32)


class Cascade3Stage:
    name = "cascade_3_stage"
    category = "hybrid"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        stage3 = params.get("stage3", "int8")
        bits = 8 if stage3 == "int8" else 4
        t = np.asarray(tensor, dtype=np.float64)
        U, S, Vt = _svd_efficient(t, params.get("energy", 0.995))
        tt_rank = _energy_rank(S, params.get("energy", 0.995))
        breakeven = t.size / (tt_rank * (t.shape[0] + t.shape[1]))
        if breakeven > 1.0 and tt_rank <= min(t.shape):
            cores, _ = _tt_decomp(t, rank=tt_rank)
            packed = []
            shapes = []
            scales = []
            for c in cores:
                ds = dct(c.ravel())
                v_q, sc = _pack_quant(ds, bits)
                packed.append(v_q.tobytes())
                shapes.append(c.shape)
                scales.append(sc)
            meta = dict(
                shape=tensor.shape,
                bits=bits,
                n_cores=len(packed),
                core_shapes=shapes,
                core_scales=scales,
                dense=True,
            )
            return b"".join(packed), meta
        coeffs, nk = _energy_keep(dct_2d(t), 0.995)
        coeffs[np.abs(coeffs) < 1e-10] = 0.0
        nnz_mask = np.abs(coeffs.ravel()) > 1e-10
        nnz = int(np.count_nonzero(nnz_mask))
        vals = coeffs.ravel()[nnz_mask]
        v_q, sc = _pack_quant(vals, bits)
        mask = nnz_mask.astype(np.uint8)
        data = struct.pack("!I", nnz) + mask.tobytes() + v_q.tobytes()
        meta = dict(
            shape=tensor.shape,
            bits=bits,
            n_cores=0,
            core_shapes=[],
            core_scales=[],
            direct_dct=True,
            scale=sc,
            nnz=nnz,
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bits = metadata["bits"]
        if metadata.get("direct_dct", False):
            nnz = metadata["nnz"]
            scale = metadata["scale"]
            offset = 4
            mask = np.frombuffer(
                data, dtype=np.uint8, count=shape[0] * shape[1], offset=offset
            )
            offset += shape[0] * shape[1]
            vals_raw = np.frombuffer(data, dtype=np.int8, count=nnz, offset=offset)
            vals = _unpack_quant(vals_raw, scale, bits)
            coeffs = np.zeros(shape[0] * shape[1], dtype=np.float64)
            coeffs[mask.astype(bool)] = vals
            recon = idct_2d(coeffs.reshape(shape))
            return (
                recon.ravel()[: shape[0] * shape[1]].reshape(shape).astype(np.float32)
            )
        n_cores = metadata["n_cores"]
        core_shapes = metadata["core_shapes"]
        core_scales = metadata["core_scales"]
        dense = metadata.get("dense", False)
        offset = 0
        rcores = []
        for i in range(n_cores):
            cs = core_shapes[i]
            sz = int(cs[0] * cs[1]) if len(cs) == 2 else int(cs[0])
            if dense:
                raw_i = np.frombuffer(data, dtype=np.int8, count=sz, offset=offset)
                offset += sz
                dc = _unpack_quant(raw_i, core_scales[i], bits)
            else:
                nnz_i = struct.unpack_from("!I", data, offset)[0]
                offset += 4
                mask_i = np.frombuffer(data, dtype=np.uint8, count=sz, offset=offset)
                offset += sz
                raw_i = np.frombuffer(data, dtype=np.int8, count=nnz_i, offset=offset)
                offset += nnz_i
                dc = np.zeros(sz, dtype=np.float64)
                dc[mask_i.astype(bool)] = _unpack_quant(raw_i, core_scales[i], bits)
            ict = idct(dc)
            rcores.append(ict.reshape(cs))
        return _merge_cores(rcores, shape).astype(np.float32)


class Cascade4Stage:
    name = "cascade_4_stage"
    category = "hybrid"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        stage3 = params.get("stage3", "int8")
        bits = 8 if stage3 == "int8" else 4
        t = np.asarray(tensor, dtype=np.float64)
        U, S, Vt = _svd_efficient(t, params.get("energy", 0.995))
        tt_rank = _energy_rank(S, params.get("energy", 0.995))
        breakeven = t.size / (tt_rank * (t.shape[0] + t.shape[1]))
        if breakeven > 1.0 and tt_rank <= min(t.shape):
            cores, _ = _tt_decomp(t, rank=tt_rank)
            packed = []
            shapes = []
            scales = []
            for c in cores:
                ds = dct(c.ravel())
                v_q, sc = _pack_quant(ds, bits)
                packed.append(v_q.tobytes())
                shapes.append(c.shape)
                scales.append(sc)
            meta = dict(
                shape=tensor.shape,
                bits=bits,
                n_cores=len(packed),
                core_shapes=shapes,
                core_scales=scales,
                dense=True,
            )
            return b"".join(packed), meta
        coeffs, nk = _energy_keep(dct_2d(t), 0.995)
        coeffs[np.abs(coeffs) < 1e-10] = 0.0
        nnz_mask = np.abs(coeffs.ravel()) > 1e-10
        nnz = int(np.count_nonzero(nnz_mask))
        vals = coeffs.ravel()[nnz_mask]
        v_q, sc = _pack_quant(vals, bits)
        mask = nnz_mask.astype(np.uint8)
        data = struct.pack("!I", nnz) + mask.tobytes() + v_q.tobytes()
        meta = dict(
            shape=tensor.shape,
            bits=bits,
            n_cores=0,
            core_shapes=[],
            core_scales=[],
            direct_dct=True,
            scale=sc,
            nnz=nnz,
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return Cascade3Stage.decompress(self, data, metadata)


class QuantizeThenSparsify:
    name = "quantize_then_sparsify"
    category = "hybrid"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        quant_bits = params.get("quant_bits", 8)
        t = np.asarray(tensor, dtype=np.float64)
        bits = quant_bits
        q_recon, _, q_scale = _quantize_inline(t, bits)
        flat = q_recon.ravel()
        idx = np.argsort(np.abs(flat))[::-1]
        ce = np.cumsum(flat[idx] ** 2) / max(np.sum(flat**2), 1e-30)
        nk = max(1, int(np.searchsorted(ce, 0.995)) + 1)
        mask = np.zeros(len(flat), dtype=np.uint8)
        mask[idx[:nk]] = 1
        vals = flat[mask.astype(bool)]
        v_q, _ = _pack_quant(vals, bits)
        data = struct.pack("!I", nk) + mask.tobytes() + v_q.tobytes()
        meta = dict(shape=tensor.shape, bits=bits, scale=q_scale, nnz=nk)
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bits = metadata["bits"]
        scale = metadata["scale"]
        nnz = metadata["nnz"]
        offset = 4
        n = shape[0] * shape[1]
        mask = np.frombuffer(data, dtype=np.uint8, count=n, offset=offset)
        offset += n
        vals_raw = np.frombuffer(data, dtype=np.int8, count=nnz, offset=offset)
        vals = _unpack_quant(vals_raw, scale, bits)
        recon = np.zeros(n, dtype=np.float64)
        recon[mask.astype(bool)] = vals
        return recon.reshape(shape).astype(np.float32)


def _quantize_inline(t: np.ndarray, bits: int) -> Tuple[np.ndarray, float, float]:
    flat = t.ravel()
    scale = float(max(np.abs(flat).max(), 1e-10))
    half = (1 << (bits - 1)) - 1
    quant = np.clip(np.round(flat / scale * half), -half - 1, half).astype(np.int8)
    recon = quant.astype(np.float64) * scale / half
    return recon.reshape(t.shape), 32.0 / bits, scale


class DecomposeThenQuantize:
    name = "decompose_then_quantize"
    category = "hybrid"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        method = params.get("method", "svd")
        quant_bits = params.get("quant_bits", 8)
        bits = quant_bits
        t = np.asarray(tensor, dtype=np.float64)
        if method == "svd":
            U, S, Vt = _svd_efficient(t, params.get("energy", 0.995))
            rank = _energy_rank(S, params.get("energy", 0.995))
            U_r, S_r, Vt_r = U[:, :rank], S[:rank], Vt[:rank, :]
            U_p, sc_u = _pack_quant(U_r, bits)
            S_p, sc_s = _pack_quant(S_r, bits)
            Vt_p, sc_v = _pack_quant(Vt_r, bits)
            data = U_p.tobytes() + S_p.tobytes() + Vt_p.tobytes()
            meta = dict(
                shape=tensor.shape,
                method="svd",
                rank=rank,
                bits=bits,
                u_sc=sc_u,
                s_sc=sc_s,
                v_sc=sc_v,
            )
        elif method == "tt":
            U_s, S_s, _ = _svd_efficient(t, params.get("energy", 0.995))
            tt_rank = _energy_rank(S_s, params.get("energy", 0.995))
            breakeven = t.size / (tt_rank * (t.shape[0] + t.shape[1]))
            if breakeven > 1.0 and tt_rank <= min(t.shape):
                cores, _ = _tt_decomp(t, rank=tt_rank)
            else:
                cores, _ = _tt_decomp(t, rank=max(2, min(t.shape) // 4))
            packed = []
            scales = []
            shapes = []
            for c in cores:
                qc, _, cs = _quantize_inline(c, bits)
                q_p, _ = _pack_quant(qc, bits)
                packed.append(q_p.tobytes())
                scales.append(cs)
                shapes.append(c.shape)
            data = b"".join(packed)
            meta = dict(
                shape=tensor.shape,
                method="tt",
                rank=tt_rank,
                bits=bits,
                n_cores=len(packed),
                core_scales=scales,
                core_shapes=shapes,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bits = metadata["bits"]
        if metadata["method"] == "svd":
            rank = metadata["rank"]
            u_sc, s_sc, v_sc = metadata["u_sc"], metadata["s_sc"], metadata["v_sc"]
            u_sz = shape[0] * rank
            s_sz = rank
            v_sz = rank * shape[1]
            U_p = np.frombuffer(data, dtype=np.int8, count=u_sz, offset=0).reshape(
                shape[0], rank
            )
            S_p = np.frombuffer(data, dtype=np.int8, count=s_sz, offset=u_sz)
            Vt_p = np.frombuffer(
                data, dtype=np.int8, count=v_sz, offset=u_sz + s_sz
            ).reshape(rank, shape[1])
            U = _unpack_quant(U_p, u_sc, bits)
            S = _unpack_quant(S_p, s_sc, bits)
            Vt = _unpack_quant(Vt_p, v_sc, bits)
            recon = U @ np.diag(S) @ Vt
        else:
            core_scales = metadata["core_scales"]
            core_shapes = metadata["core_shapes"]
            n_cores = len(core_shapes)
            offset = 0
            rcores = []
            for i in range(n_cores):
                cs = core_shapes[i]
                sz = int(cs[0] * cs[1]) if len(cs) == 2 else int(cs[0])
                qc = np.frombuffer(
                    data, dtype=np.int8, count=sz, offset=offset
                ).reshape(cs)
                offset += sz
                rc = _unpack_quant(qc, core_scales[i], bits)
                rcores.append(rc)
            if len(rcores) == 2:
                recon = rcores[0] @ rcores[1]
            else:
                recon = rcores[0]
        if recon.size < shape[0] * shape[1]:
            recon = np.pad(
                recon.ravel(), (0, shape[0] * shape[1] - recon.size)
            ).reshape(shape)
        return recon.astype(np.float32)


class TransformThenQuantize:
    name = "transform_then_quantize"
    category = "hybrid"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        transform = params.get("transform", "dct")
        quant_bits = params.get("quant_bits", 8)
        t = np.asarray(tensor, dtype=np.float64)
        coeffs = dct_2d(t) if transform == "dct" else np.abs(np.fft.fft2(t))
        kept, nk = _energy_keep(coeffs, params.get("energy", 0.995))
        kept[np.abs(kept) < 1e-10] = 0.0
        q_recon, _, q_scale = _quantize_inline(kept, quant_bits)
        qc = q_recon.reshape(coeffs.shape)
        nnz_mask = np.abs(qc.ravel()) > 1e-10
        nnz = int(np.count_nonzero(nnz_mask))
        vals = qc.ravel()[nnz_mask]
        v_q, _ = _pack_quant(vals, quant_bits)
        mask = nnz_mask.astype(np.uint8)
        data = struct.pack("!I", nnz) + mask.tobytes() + v_q.tobytes()
        meta = dict(
            shape=tensor.shape,
            transform=transform,
            bits=quant_bits,
            nnz=nnz,
            scale=q_scale,
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        transform = metadata["transform"]
        bits = metadata["bits"]
        nnz = metadata["nnz"]
        scale = metadata["scale"]
        offset = 4
        n = shape[0] * shape[1]
        mask = np.frombuffer(data, dtype=np.uint8, count=n, offset=offset)
        offset += n
        vals_raw = np.frombuffer(data, dtype=np.int8, count=nnz, offset=offset)
        vals = _unpack_quant(vals_raw, scale, bits)
        coeffs = np.zeros(n, dtype=np.float64)
        coeffs[mask.astype(bool)] = vals
        coeffs = coeffs.reshape(shape)
        if transform == "dct":
            recon = idct_2d(coeffs)
        else:
            recon = coeffs
        return recon.ravel()[:n].reshape(shape).astype(np.float32)


class TransformThenSparsify:
    name = "transform_then_sparsify"
    category = "hybrid"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        transform = params.get("transform", "dct")
        keep = params.get("keep", 0.5)
        t = np.asarray(tensor, dtype=np.float64)
        coeffs = dct_2d(t) if transform == "dct" else np.abs(np.fft.fft2(t))
        flat = coeffs.ravel()
        idx = np.argsort(np.abs(flat))[::-1]
        ce = np.cumsum(flat[idx] ** 2) / max(np.sum(flat**2), 1e-30)
        nk = max(1, int(np.searchsorted(ce, 0.995)) + 1)
        mask = np.zeros(len(flat), dtype=np.uint8)
        mask[idx[:nk]] = 1
        bits = 8
        vals = flat[mask.astype(bool)]
        v_q, scale = _pack_quant(vals, bits)
        data = struct.pack("!I", nk) + mask.tobytes() + v_q.tobytes()
        meta = dict(
            shape=tensor.shape,
            transform=transform,
            bits=bits,
            nnz=nk,
            scale=scale,
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        transform = metadata["transform"]
        bits = metadata["bits"]
        nnz = metadata["nnz"]
        scale = metadata["scale"]
        offset = 4
        n = shape[0] * shape[1]
        mask = np.frombuffer(data, dtype=np.uint8, count=n, offset=offset)
        offset += n
        vals_raw = np.frombuffer(data, dtype=np.int8, count=nnz, offset=offset)
        vals = _unpack_quant(vals_raw, scale, bits)
        coeffs = np.zeros(n, dtype=np.float64)
        coeffs[mask.astype(bool)] = vals
        coeffs = coeffs.reshape(shape)
        if transform == "dct":
            recon = idct_2d(coeffs)
        else:
            recon = coeffs
        return recon.ravel()[:n].reshape(shape).astype(np.float32)


class DecomposeThenTransform:
    """TT decompose → DCT each core → energy sparsify → quantize (int8)."""

    name = "decompose_then_transform"
    category = "hybrid"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        bits = params.get("bits", 8)
        energy = params.get("energy", 0.95)
        t = np.asarray(tensor, dtype=np.float64)
        U_s, S_s, _ = _svd_efficient(t, params.get("energy", 0.995))
        rank = _energy_rank(S_s, params.get("energy", 0.995))
        tt_rank = min(rank, min(t.shape))
        cores, _ = _tt_decomp(t, rank=tt_rank)
        packed = []
        orig_shapes = []
        scales = []
        nnz_list = []
        for c in cores:
            orig_shapes.append(c.shape)
            cd = dct(c.ravel())
            kept, nk = _energy_keep(cd, energy)
            kept[np.abs(kept) < 1e-10] = 0.0
            nnz_mask = np.abs(kept.ravel()) > 1e-10
            nnz = int(np.count_nonzero(nnz_mask))
            vals = kept.ravel()[nnz_mask]
            v_q, sc = _pack_quant(vals, bits)
            mask = nnz_mask.astype(np.uint8)
            import struct as _struct

            chunk = _struct.pack("!I", nnz) + mask.tobytes() + v_q.tobytes()
            packed.append(chunk)
            scales.append(sc)
            nnz_list.append(nnz)
        data = b"".join(packed)
        meta = dict(
            shape=tensor.shape,
            bits=bits,
            n_cores=len(packed),
            orig_shapes=orig_shapes,
            core_scales=scales,
            nnz_list=nnz_list,
            energy=energy,
            dct_transform=True,
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata["shape"]
        bits = metadata["bits"]
        core_scales = metadata["core_scales"]
        orig_shapes = metadata.get("orig_shapes", [])
        nnz_list = metadata.get("nnz_list", [])
        n_cores = len(orig_shapes) if orig_shapes else metadata.get("n_cores", 0)
        offset = 0
        rcores = []
        import struct as _struct

        for i in range(n_cores):
            if metadata.get("dct_transform", False):
                os = orig_shapes[i]
                n_total = int(np.prod(os))
                n_total = max(n_total, 1)
                nnz_hdr = _struct.unpack_from("!I", data, offset)[0]
                offset += 4
                mask = np.frombuffer(data, dtype=np.uint8, count=n_total, offset=offset)
                offset += n_total
                vals_raw = np.frombuffer(
                    data, dtype=np.int8, count=nnz_hdr, offset=offset
                )
                offset += nnz_hdr
                vals = _unpack_quant(
                    vals_raw, core_scales[i] if i < len(core_scales) else 1.0, bits
                )
                cd_recon = np.zeros(n_total, dtype=np.float64)
                cd_recon[mask.astype(bool)] = vals
                ict = idct(cd_recon)
                rcores.append(ict.reshape(os))
            else:
                core_shapes = metadata.get("core_shapes", [])
                cs = core_shapes[i] if i < len(core_shapes) else (1,)
                sz = int(cs[0] * cs[1]) if len(cs) == 2 else int(cs[0])
                qc = np.frombuffer(
                    data, dtype=np.int8, count=sz, offset=offset
                ).reshape(cs)
                offset += sz
                rc = _unpack_quant(
                    qc, core_scales[i] if i < len(core_scales) else 1.0, bits
                )
                rcores.append(rc)
        recon = _merge_cores(rcores, shape)
        return recon.astype(np.float32)


class AllMethodsEnsemble:
    name = "all_methods_ensemble"
    category = "hybrid"

    def compress(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        methods = [
            ("c2", Cascade2Stage(), {"stage2": "int8"}),
            ("c3", Cascade3Stage(), {"stage3": "int8"}),
            ("c4", Cascade4Stage(), {"stage3": "int8"}),
            ("qs", QuantizeThenSparsify(), {"quant_bits": 8}),
            ("dq", DecomposeThenQuantize(), {"method": "svd", "quant_bits": 8}),
            ("tq", TransformThenQuantize(), {"transform": "dct", "quant_bits": 8}),
            ("ts", TransformThenSparsify(), {"transform": "dct"}),
        ]
        t = np.asarray(tensor, dtype=np.float64)
        candidates = []
        for name, inst, p in methods:
            try:
                d, m = inst.compress(tensor, **p)
                ratio = len(tensor.tobytes()) / max(len(d), 1)
                recon = inst.decompress(d, m)
                snr = _calc_snr(t, recon)
                score = snr * math.log(max(ratio, 1.0) + 1.0)
                candidates.append((score, name, d, m, ratio, snr))
            except Exception:
                continue
        if not candidates:
            data = tensor.astype(np.float32).tobytes()
            return data, dict(shape=tensor.shape, best_method="raw")
        scores_arr = np.array([c[0] for c in candidates])
        best = candidates[int(np.argmax(scores_arr))]
        meta = dict(
            shape=tensor.shape,
            best_method=best[1],
            inner_meta=best[3],
            ratio=best[4],
            snr=best[5],
        )
        return best[2], meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        best = metadata["best_method"]
        inner_meta = metadata["inner_meta"]
        if best == "raw":
            return (
                np.frombuffer(data, dtype=np.float32).copy().reshape(metadata["shape"])
            )
        dispatch = {
            "c2": Cascade2Stage,
            "c3": Cascade3Stage,
            "c4": Cascade4Stage,
            "qs": QuantizeThenSparsify,
            "dq": DecomposeThenQuantize,
            "tq": TransformThenQuantize,
            "ts": TransformThenSparsify,
        }
        return dispatch[best]().decompress(data, inner_meta)

"""
Hadamard-transformed quantization - rotate via FWHT before quantizing.
Rotating spreads outliers evenly across values, reducing quantization error.
"""

from typing import Dict, Tuple
import numpy as np
from spectralstream.core.math_primitives import fwht, ifwht, next_power_of_two


def _snr(orig, recon):
    o = orig.ravel().astype(np.float64)
    r = recon.ravel().astype(np.float64)
    mse = np.mean((o - r) ** 2)
    return float(10.0 * np.log10(np.var(o) / (mse + 1e-30)))


def _pad_fwht(tensor):
    """Flatten and pad last dim to power of 2, always returns a copy."""
    t = np.asarray(tensor, dtype=np.float32)
    s = t.shape
    f = t.reshape(-1, s[-1])
    nr, nc = f.shape
    n = next_power_of_two(nc)
    if n > nc:
        fp = np.pad(f, ((0, 0), (0, n - nc)))
    else:
        fp = f.copy()
    return fp, s, nr, nc, n


def _unpad_fwht(d, s, nc, n):
    """Trim FWHT padding and restore shape."""
    if n > nc:
        d = d[:, :nc]
    return d.reshape(s)


def hadamard_int8(tensor: np.ndarray, block_size: int = 32) -> Tuple[Dict, float, float]:
    """FWHT rotate + INT8 quant. ~4x, better outlier handling."""
    t = np.asarray(tensor, dtype=np.float32)
    fp, s, nr, nc, n = _pad_fwht(t)
    r = fwht(fp, normalize=True)
    p = (block_size - r.shape[1] % block_size) % block_size
    rp = np.pad(r, ((0, 0), (0, p))) if p else r
    nb = rp.shape[1] // block_size
    b = rp.reshape(nr, nb, block_size)
    sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
    q = np.round(b / sc * 127.0).clip(-128, 127).astype(np.int8)
    comp = {'q': q, 'sc': sc.astype(np.float32), 'shape': s, 'bs': block_size,
            'pad_fwht': n - nc, 'pad_quant': p, 'method': 'hadamard_int8'}
    ratio = float(t.nbytes / (q.nbytes + sc.nbytes))
    d = sc * (q.astype(np.float32) / 127.0)
    d = d.reshape(rp.shape)
    if p:
        d = d[:, :-p]
    d = ifwht(d, normalize=True)
    rv = _unpad_fwht(d, s, nc, n)
    return comp, ratio, _snr(t, rv)


def hadamard_int4(tensor: np.ndarray, block_size: int = 32) -> Tuple[Dict, float, float]:
    """FWHT rotate + INT4 quant. ~8x."""
    t = np.asarray(tensor, dtype=np.float32)
    fp, s, nr, nc, n = _pad_fwht(t)
    r = fwht(fp, normalize=True)
    p = (block_size - r.shape[1] % block_size) % block_size
    rp = np.pad(r, ((0, 0), (0, p))) if p else r
    nb = rp.shape[1] // block_size
    b = rp.reshape(nr, nb, block_size)
    sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
    q = np.round(b / sc * 7.0).clip(-7, 7).astype(np.int8)
    uv = (q + 8).astype(np.uint8)
    pairs = uv.reshape(nr, nb, block_size // 2, 2)
    packed = (pairs[..., 0].astype(np.uint8) << 4) | pairs[..., 1].astype(np.uint8)
    comp = {'q': packed, 'sc': sc.astype(np.float32), 'shape': s, 'bs': block_size,
            'pad_fwht': n - nc, 'pad_quant': p, 'method': 'hadamard_int4'}
    ratio = float(t.nbytes / (packed.nbytes + sc.nbytes))
    uv_e = (packed.astype(np.uint16) >> 4).astype(np.uint8)
    uv_o = (packed & 0x0F).astype(np.uint8)
    uv_d = np.empty((nr, nb, block_size), dtype=np.uint8)
    uv_d[..., 0::2] = uv_e
    uv_d[..., 1::2] = uv_o
    d = sc * ((uv_d.astype(np.int8) - 8).astype(np.float32) / 7.0)
    d = d.reshape(rp.shape)
    if p:
        d = d[:, :-p]
    d = ifwht(d, normalize=True)
    rv = _unpad_fwht(d, s, nc, n)
    return comp, ratio, _snr(t, rv)


def hadamard_quant_entropy(tensor: np.ndarray, block_size: int = 32) -> Tuple[Dict, float, float]:
    """FWHT + quant + entropy code. ~8-16x combined."""
    t = np.asarray(tensor, dtype=np.float32)
    fp, s, nr, nc, n = _pad_fwht(t)
    r = fwht(fp, normalize=True)
    p = (block_size - r.shape[1] % block_size) % block_size
    rp = np.pad(r, ((0, 0), (0, p))) if p else r
    nb = rp.shape[1] // block_size
    b = rp.reshape(nr, nb, block_size)
    sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
    q = np.round(b / sc * 7.0).clip(-7, 7).astype(np.int8)
    flat_q = q.ravel()
    mask = flat_q != 0
    nonzero_vals = flat_q[mask]
    mask_bytes = np.packbits(mask)
    uv = (nonzero_vals + 8).astype(np.uint8)
    pairs = uv.reshape(-1, 2) if len(uv) % 2 == 0 else np.pad(uv, (0, 1))[:len(uv) + (len(uv) % 2)].reshape(-1, 2)
    if len(uv) >= 2:
        pairs = uv[:len(uv) - (len(uv) % 2)].reshape(-1, 2)
        packed_vals = (pairs[..., 0].astype(np.uint8) << 4) | pairs[..., 1].astype(np.uint8)
        last_byte = np.array([uv[-1] << 4], dtype=np.uint8) if len(uv) % 2 else np.array([], dtype=np.uint8)
    else:
        packed_vals = np.array([], dtype=np.uint8)
        last_byte = np.array([uv[0] << 4], dtype=np.uint8) if len(uv) == 1 else np.array([], dtype=np.uint8)
    comp = {'sc': sc.astype(np.float32), 'mask_bytes': mask_bytes, 'nonzero': packed_vals,
            'last_byte': last_byte, 'n_nonzero': len(nonzero_vals), 'n_total': len(flat_q),
            'shape': s, 'bs': block_size, 'pad_fwht': n - nc, 'pad_quant': p, 'method': 'hadamard_quant_entropy'}
    comp_bytes = sum(v.nbytes for v in [sc, mask_bytes, packed_vals, last_byte] if isinstance(v, np.ndarray) and len(v) > 0)
    ratio = float(t.nbytes / max(comp_bytes, 1))
    mask_full = np.unpackbits(mask_bytes)[:len(flat_q)]
    q_rec = np.zeros(len(flat_q), dtype=np.int8)
    full_vals = np.concatenate([packed_vals.view(dtype=np.uint8), last_byte]) if len(last_byte) else packed_vals.view(dtype=np.uint8)
    if len(full_vals) > 0:
        uv_e = (full_vals.astype(np.uint16) >> 4).astype(np.uint8)
        uv_o = (full_vals & 0x0F).astype(np.uint8)
        unpacked = np.empty(len(full_vals) * 2, dtype=np.uint8)
        unpacked[0::2] = uv_e
        unpacked[1::2] = uv_o
        unpacked = unpacked[:len(nonzero_vals)]
        q_rec[mask] = unpacked.astype(np.int8) - 8
    q_rec = q_rec.reshape(q.shape)
    d = sc * (q_rec.astype(np.float32) / 7.0)
    d = d.reshape(rp.shape)
    if p:
        d = d[:, :-p]
    d = ifwht(d, normalize=True)
    rv = _unpad_fwht(d, s, nc, n)
    return comp, ratio, _snr(t, rv)


if __name__ == '__main__':
    t = np.random.randn(256, 256).astype(np.float32)
    data, ratio, snr = hadamard_int8(t)
    print(f"HadamardInt8: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = hadamard_int4(t)
    print(f"HadamardInt4: {ratio:.2f}x, SNR={snr:.1f}dB")
    data, ratio, snr = hadamard_quant_entropy(t)
    print(f"HadamardQuantEntropy: {ratio:.2f}x, SNR={snr:.1f}dB")

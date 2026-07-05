"""
Block quantization methods - divide tensor into blocks, quantize each independently.
All methods return: (compressed: dict, ratio: float, snr_db: float)
"""

from typing import Dict, Tuple
import numpy as np


def _pad_blocks(tensor, block_size):
    """Reshape to 2D, pad columns for block alignment."""
    t = np.asarray(tensor, dtype=np.float32)
    s = t.shape
    f = t.reshape(-1, s[-1])
    nr, nc = f.shape
    p = (block_size - nc % block_size) % block_size
    if p:
        f = np.pad(f, ((0, 0), (0, p)))
    nb = f.shape[1] // block_size
    b = f.reshape(nr, nb, block_size)
    return b, s, nr, nb, p


def _deblock(b, s, p):
    """Inverse of _pad_blocks."""
    d = b.reshape(-1, b.shape[-2] * b.shape[-1]) if b.ndim == 3 else b
    if p:
        d = d[:, :-p]
    return d.reshape(s)


def _snr(orig, recon):
    o = orig.ravel().astype(np.float64)
    r = recon.ravel().astype(np.float64)
    mse = np.mean((o - r) ** 2)
    return float(10.0 * np.log10(np.var(o) / (mse + 1e-30)))


def block_int8(tensor: np.ndarray, block_size: int = 32) -> Tuple[Dict, float, float]:
    """Block-wise INT8 quantization. ~4x with <1% error."""
    t = np.asarray(tensor, dtype=np.float32)
    b, s, nr, nb, p = _pad_blocks(t, block_size)
    sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
    q = np.round(b / sc * 127.0).clip(-128, 127).astype(np.int8)
    comp = {'q': q, 'sc': sc.astype(np.float32), 'shape': s, 'bs': block_size, 'p': p, 'method': 'block_int8'}
    ratio = float(t.nbytes / (q.nbytes + sc.nbytes))
    d = sc * (q.astype(np.float32) / 127.0)
    r = _deblock(d, s, p)
    return comp, ratio, _snr(t, r)


def block_int4(tensor: np.ndarray, block_size: int = 32) -> Tuple[Dict, float, float]:
    """Block-wise INT4 quantization. ~8x at 5-15% error."""
    t = np.asarray(tensor, dtype=np.float32)
    b, s, nr, nb, p = _pad_blocks(t, block_size)
    sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
    q = np.round(b / sc * 7.0).clip(-7, 7).astype(np.int8)
    uv = (q + 8).astype(np.uint8)
    pairs = uv.reshape(nr, nb, block_size // 2, 2)
    packed = (pairs[..., 0].astype(np.uint8) << 4) | pairs[..., 1].astype(np.uint8)
    comp = {'q': packed, 'sc': sc.astype(np.float32), 'shape': s, 'bs': block_size, 'p': p, 'method': 'block_int4'}
    ratio = float(t.nbytes / (packed.nbytes + sc.nbytes))
    uv_e = (packed.astype(np.uint16) >> 4).astype(np.uint8)
    uv_o = (packed & 0x0F).astype(np.uint8)
    uv_d = np.empty((nr, nb, block_size), dtype=np.uint8)
    uv_d[..., 0::2] = uv_e
    uv_d[..., 1::2] = uv_o
    d = sc * ((uv_d.astype(np.int8) - 8).astype(np.float32) / 7.0)
    r = _deblock(d, s, p)
    return comp, ratio, _snr(t, r)


def block_int2(tensor: np.ndarray, block_size: int = 32) -> Tuple[Dict, float, float]:
    """Block-wise INT2 quantization. ~16x at 40-60% error."""
    t = np.asarray(tensor, dtype=np.float32)
    b, s, nr, nb, p = _pad_blocks(t, block_size)
    sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
    norm = np.clip(b / sc, -1.0, 1.0)
    q = np.round((norm + 1.0) / 2.0 * 3.0).clip(0, 3).astype(np.uint8)
    quads = q.reshape(nr, nb, block_size // 4, 4)
    packed = (quads[..., 0].astype(np.uint8) << 6) | (quads[..., 1].astype(np.uint8) << 4) | \
             (quads[..., 2].astype(np.uint8) << 2) | quads[..., 3].astype(np.uint8)
    comp = {'q': packed, 'sc': sc.astype(np.float32), 'shape': s, 'bs': block_size, 'p': p, 'method': 'block_int2'}
    ratio = float(t.nbytes / (packed.nbytes + sc.nbytes))
    v0 = (packed >> 6) & 0x03
    v1 = (packed >> 4) & 0x03
    v2 = (packed >> 2) & 0x03
    v3 = packed & 0x03
    uv_d = np.empty((nr, nb, block_size), dtype=np.uint8)
    uv_d[..., 0::4] = v0
    uv_d[..., 1::4] = v1
    uv_d[..., 2::4] = v2
    uv_d[..., 3::4] = v3
    d = sc * (uv_d.astype(np.float32) / 3.0 * 2.0 - 1.0)
    r = _deblock(d, s, p)
    return comp, ratio, _snr(t, r)


def block_binary(tensor: np.ndarray, block_size: int = 32) -> Tuple[Dict, float, float]:
    """Block-wise binary quantization. ~32x, very high error."""
    t = np.asarray(tensor, dtype=np.float32)
    b, s, nr, nb, p = _pad_blocks(t, block_size)
    sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
    bits = (b >= 0).astype(np.uint8)
    octets = bits.reshape(nr, nb, block_size // 8, 8)
    packed = sum(octets[..., i].astype(np.uint8) << (7 - i) for i in range(8))
    comp = {'q': packed, 'sc': sc.astype(np.float32), 'shape': s, 'bs': block_size, 'p': p, 'method': 'block_binary'}
    ratio = float(t.nbytes / (packed.nbytes + sc.nbytes))
    bits_d = np.unpackbits(packed).reshape(nr, nb, block_size).astype(np.float32)
    d = sc * (bits_d * 2.0 - 1.0)
    r = _deblock(d, s, p)
    return comp, ratio, _snr(t, r)


def block_ternary(tensor: np.ndarray, block_size: int = 32) -> Tuple[Dict, float, float]:
    """Block-wise ternary quantization (values: -1, 0, +1). ~16x."""
    t = np.asarray(tensor, dtype=np.float32)
    b, s, nr, nb, p = _pad_blocks(t, block_size)
    sc = np.maximum(np.max(np.abs(b), axis=2, keepdims=True), 1e-10)
    norm = np.clip(b / sc, -1.0, 1.0)
    eps = 0.05
    tval = np.zeros_like(norm, dtype=np.int8)
    tval[norm > eps] = 1
    tval[norm < -eps] = -1
    uv = (tval + 1).astype(np.uint8)
    quads = uv.reshape(nr, nb, block_size // 4, 4)
    packed = (quads[..., 0].astype(np.uint8) << 6) | (quads[..., 1].astype(np.uint8) << 4) | \
             (quads[..., 2].astype(np.uint8) << 2) | quads[..., 3].astype(np.uint8)
    comp = {'q': packed, 'sc': sc.astype(np.float32), 'shape': s, 'bs': block_size, 'p': p, 'method': 'block_ternary'}
    ratio = float(t.nbytes / (packed.nbytes + sc.nbytes))
    v0 = (packed >> 6) & 0x03
    v1 = (packed >> 4) & 0x03
    v2 = (packed >> 2) & 0x03
    v3 = packed & 0x03
    uv_d = np.empty((nr, nb, block_size), dtype=np.uint8)
    uv_d[..., 0::4] = v0
    uv_d[..., 1::4] = v1
    uv_d[..., 2::4] = v2
    uv_d[..., 3::4] = v3
    d = sc * (uv_d.astype(np.int8) - 1).astype(np.float32)
    r = _deblock(d, s, p)
    return comp, ratio, _snr(t, r)


if __name__ == '__main__':
    t = np.random.randn(256, 256).astype(np.float32)
    for fn, nm in [(block_int8, 'BlockInt8'), (block_int4, 'BlockInt4'), (block_int2, 'BlockInt2'),
                   (block_binary, 'BlockBinary'), (block_ternary, 'BlockTernary')]:
        data, ratio, snr = fn(t)
        print(f"{nm}: {ratio:.2f}x, SNR={snr:.1f}dB")

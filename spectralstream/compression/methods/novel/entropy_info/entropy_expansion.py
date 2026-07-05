# --- fromsymbols.py ---
"""Module extracted from entropy_expansion.py — fromsymbols."""

from __future__ import annotations


def _block_dequantize(symbols, scales, n, block_size):
    n_blocks = len(scales)
    blocks = symbols.reshape(-1, block_size).astype(np.float32)
    out = (blocks / scales[:, np.newaxis]).ravel()
    return out[:n]
def _from_symbols(symbols, scales, n, block_size):
    n_blocks = len(scales)
    expected = n_blocks * block_size
    if len(symbols) < expected:
        symbols = np.pad(
            symbols.ravel(),
            (0, expected - len(symbols)),
            mode="constant",
            constant_values=0,
        )
    elif len(symbols) > expected:
        symbols = symbols.ravel()[:expected]
    return _block_dequantize(symbols, scales, n, block_size)
# --- huffwalk.py ---
"""Module extracted from entropy_expansion.py — huffwalk."""


import heapq

def _build_huffman_tree(freqs):
    heap = [_HuffNode(f, s) for s, f in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        heapq.heappush(heap, _HuffNode(lo.freq + hi.freq, left=lo, right=hi))
    return heap[0] if heap else None
def _huff_walk(node, prefix, codes):
    if node.sym is not None:
        codes[node.sym] = prefix if prefix else "0"
        return
    if node.left:
        _huff_walk(node.left, prefix + "0", codes)
    if node.right:
        _huff_walk(node.right, prefix + "1", codes)
def _build_huffman_codes(freqs):
    if not freqs:
        return {}
    root = _build_huffman_tree(freqs)
    codes = {}
    _huff_walk(root, "", codes)
    return codes
# --- tosymbols.py ---
"""Module extracted from entropy_expansion.py — tosymbols."""



def _block_quantize(tensor, block_size=128):
    flat = tensor.ravel().astype(np.float32)
    n = len(flat)
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(-1, block_size)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(amax > 1e-8, 127.0 / amax, 1.0)
    quantized = np.clip(np.round(blocks * scales), -128, 127).astype(np.int8)
    return quantized.ravel(), scales.ravel(), n, block_size
def _to_symbols(tensor, block_size=128):
    return _block_quantize(tensor, block_size)
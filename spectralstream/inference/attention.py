from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class Gemma4RMSNorm:
    def __init__(self, weight: np.ndarray, eps: float = 1e-6):
        self.weight = weight.astype(np.float32)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        rsqrt = np.float32(1.0) / np.sqrt(var + self.eps)
        return (x * rsqrt) * (np.float32(1.0) + self.weight)


class Gemma4Attention:
    def __init__(self, config, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.NUM_ATTENTION_HEADS
        self.num_kv_heads = config.NUM_KEY_VALUE_HEADS
        self.head_dim = config.HEAD_DIM
        self.head_group_size = config.head_group_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.is_sliding = config.is_sliding_window_layer(layer_idx)
        self.softcap = config.ATTENTION_SOFTCAP

    def _rope(
        self, q: np.ndarray, k: np.ndarray, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        theta = float(self.config.ROPE_THETA)
        head_dim = self.head_dim
        half = head_dim // 2
        freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
        angles = np.outer(positions.astype(np.float32), freqs)
        cos = np.cos(angles).astype(np.float32)
        sin = np.sin(angles).astype(np.float32)

        def _rotate(x: np.ndarray) -> np.ndarray:
            n, d = x.shape
            n_h = d // head_dim
            x_r = x.reshape(n, n_h, head_dim)
            x1 = x_r[:, :, :half]
            x2 = x_r[:, :, half:]
            out = np.empty_like(x_r)
            out[:, :, :half] = x1 * cos[:, np.newaxis, :] - x2 * sin[:, np.newaxis, :]
            out[:, :, half:] = x1 * sin[:, np.newaxis, :] + x2 * cos[:, np.newaxis, :]
            return out.reshape(n, d)

        return _rotate(q), _rotate(k)

    def precompute_freqs_cis(
        self, max_seq_len: int, theta: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        theta = theta or float(self.config.ROPE_THETA)
        head_dim = self.head_dim
        half = head_dim // 2
        freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
        positions = np.arange(max_seq_len, dtype=np.float32)
        angles = np.outer(positions, freqs)
        return {
            "cos": np.cos(angles).astype(np.float32),
            "sin": np.sin(angles).astype(np.float32),
        }

    def _rope_with_freqs(
        self,
        q: np.ndarray,
        k: np.ndarray,
        positions: np.ndarray,
        freqs_cis: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        head_dim = self.head_dim
        half = head_dim // 2
        cos = freqs_cis["cos"][positions]
        sin = freqs_cis["sin"][positions]

        def _rotate(x: np.ndarray) -> np.ndarray:
            n, d = x.shape
            n_h = d // head_dim
            x_r = x.reshape(n, n_h, head_dim)
            x1 = x_r[:, :, :half]
            x2 = x_r[:, :, half:]
            out = np.empty_like(x_r)
            out[:, :, :half] = x1 * cos[:, np.newaxis, :] - x2 * sin[:, np.newaxis, :]
            out[:, :, half:] = x1 * sin[:, np.newaxis, :] + x2 * cos[:, np.newaxis, :]
            return out.reshape(n, d)

        return _rotate(q), _rotate(k)

    def __call__(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        positions: Optional[np.ndarray] = None,
        freqs_cis: Optional[Dict[str, np.ndarray]] = None,
        mask: Optional[np.ndarray] = None,
        kv_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        n_tokens = q.shape[0]
        if positions is None:
            positions = np.arange(n_tokens, dtype=np.int32)

        if freqs_cis is not None:
            q, k = self._rope_with_freqs(q, k, positions, freqs_cis)
        else:
            q, k = self._rope(q, k, positions)

        n_kv = n_tokens
        if kv_cache is not None:
            k_prev = kv_cache.get(f"k_{self.layer_idx}")
            v_prev = kv_cache.get(f"v_{self.layer_idx}")
            if k_prev is not None:
                k = np.concatenate([k_prev, k], axis=0)
                v = np.concatenate([v_prev, v], axis=0)
            kv_cache[f"k_{self.layer_idx}"] = k
            kv_cache[f"v_{self.layer_idx}"] = v
            n_kv = k.shape[0]

        q_heads = q.reshape(n_tokens, self.num_heads, self.head_dim)
        k_heads = k.reshape(n_kv, self.num_kv_heads, self.head_dim)
        v_heads = v.reshape(n_kv, self.num_kv_heads, self.head_dim)

        is_sliding = self.is_sliding
        window = self.config.SLIDING_WINDOW if is_sliding else n_kv
        k_start = max(0, n_kv - window)
        head_group = self.head_group_size
        scale = self.scale
        softcap = self.softcap

        out = np.zeros((n_tokens, self.num_heads, self.head_dim), dtype=np.float32)
        for g in range(self.num_kv_heads):
            h_start = g * head_group
            h_end = min(h_start + head_group, self.num_heads)
            qg = q_heads[:, h_start:h_end]
            kg = k_heads[k_start:, g : g + 1]
            vg = v_heads[k_start:, g : g + 1]
            attn = np.matmul(qg, kg.transpose(0, 2, 1)) * scale
            if softcap > 0:
                attn = np.tanh(attn / softcap) * softcap
            if mask is not None:
                attn = attn + mask
            else:
                causal = np.triu(
                    np.full(
                        (n_tokens, attn.shape[-1]), -np.float32("inf"), dtype=np.float32
                    ),
                    k=1,
                )
                attn = (
                    attn + causal[:, np.newaxis, :] if attn.ndim == 3 else attn + causal
                )
            w = np.exp(attn - attn.max(axis=-1, keepdims=True))
            w = w / (w.sum(axis=-1, keepdims=True) + 1e-30)
            out[:, h_start:h_end] = np.matmul(w, vg)
        return out.reshape(n_tokens, -1).astype(np.float32)

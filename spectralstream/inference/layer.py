from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spectralstream.inference.attention import Gemma4Attention, Gemma4RMSNorm
from spectralstream.inference.ffn import Gemma4FFN, _gelu_tanh


class TransformerLayer:
    def __init__(
        self,
        config,
        layer_idx: int,
        attn_norm_w: np.ndarray,
        ffn_norm_w: np.ndarray,
        wq: np.ndarray,
        wk: np.ndarray,
        wv: np.ndarray,
        wo: np.ndarray,
        w_gate: np.ndarray,
        w_up: np.ndarray,
        w_down: np.ndarray,
        freqs_cis: Optional[Dict[str, np.ndarray]] = None,
        causal_mask: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = Gemma4RMSNorm(attn_norm_w, config.NORM_EPS)
        self.ffn_norm = Gemma4RMSNorm(ffn_norm_w, config.NORM_EPS)
        self.wq = wq.astype(np.float32)
        self.wk = wk.astype(np.float32)
        self.wv = wv.astype(np.float32)
        self.wo = wo.astype(np.float32)
        self.attn = Gemma4Attention(config, layer_idx)
        self.ffn = Gemma4FFN(w_gate, w_up, w_down)
        self._freqs_cis = freqs_cis
        self._causal_mask = causal_mask

    def __call__(
        self,
        x: np.ndarray,
        freqs_cis: Optional[Dict[str, np.ndarray]] = None,
        mask: Optional[np.ndarray] = None,
        kv_cache: Optional[Dict[str, np.ndarray]] = None,
        positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        h = self.attn_norm(x)
        q = h @ self.wq
        k = h @ self.wk
        v = h @ self.wv
        n_tokens = x.shape[0]
        if positions is None:
            positions = np.arange(n_tokens, dtype=np.int32)

        freqs = freqs_cis if freqs_cis is not None else self._freqs_cis
        attn_mask = mask if mask is not None else self._causal_mask

        attn_out = self.attn(q, k, v, positions, freqs, attn_mask, kv_cache)
        x = x + attn_out @ self.wo

        h2 = self.ffn_norm(x)
        gate = h2 @ self.ffn.w_gate
        up = h2 @ self.ffn.w_up
        hidden = _gelu_tanh(gate) * up
        ffn_out = hidden @ self.ffn.w_down
        x = x + ffn_out
        return x

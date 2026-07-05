"""
Gemma 4 Configuration — constants and adapter for Gemma 4 E2B/E4B models.
Extracted from archive/v1.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

GEMMA4_E2B_CONFIG = {
    "n_layers": 35,
    "d_model": 1536,
    "n_heads": 8,
    "n_kv_heads": 1,
    "head_dim": 192,
    "ff_dim": 12288,
    "vocab_size": 262144,
    "max_seq_len": 131072,
    "norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "attention_softcap": 50.0,
    "logit_softcap": 30.0,
    "embedding_scale": True,
    "weight_tying": True,
    "activation": "gelu_pytorch_tanh",
    "norm_type": "gemma4_rmsnorm",
    "use_p_rope": True,
    "sliding_window": 512,
    "shared_kv_layers": 20,
    "ple_dim": 256,
    "rope_dim": 512,
    "rope_dim_swa": 256,
}

GEMMA4_E4B_CONFIG = {
    "n_layers": 42,
    "d_model": 2560,
    "n_heads": 8,
    "n_kv_heads": 2,
    "head_dim": 320,
    "ff_dim": 10240,
    "vocab_size": 262144,
    "max_seq_len": 131072,
    "norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "attention_softcap": 50.0,
    "logit_softcap": 30.0,
    "embedding_scale": True,
    "weight_tying": True,
    "activation": "gelu_pytorch_tanh",
    "norm_type": "gemma4_rmsnorm",
    "use_p_rope": True,
    "sliding_window": 512,
    "shared_kv_layers": 18,
    "ple_dim": 256,
    "rope_dim": 512,
    "rope_dim_swa": 256,
}

ARCH_CONFIG_MAP = {
    "gemma-4-e2b": GEMMA4_E2B_CONFIG,
    "gemma-4-e4b": GEMMA4_E4B_CONFIG,
    "gemma-4-e2b-it": GEMMA4_E2B_CONFIG,
    "gemma-4-e4b-it": GEMMA4_E4B_CONFIG,
}


def detect_gemma4_variant(model_path: str) -> str:
    """Detect Gemma 4 variant by filename heuristic."""
    name = Path(model_path).name.lower()
    if "e2b" in name:
        return "e2b"
    if "e4b" in name:
        return "e4b"
    raise ValueError(f"Cannot detect Gemma 4 variant from {model_path}")


def get_gemma4_config(model_path: str) -> dict:
    """Get configuration for a Gemma 4 model (filename-based heuristic)."""
    variant = detect_gemma4_variant(model_path)
    if variant == "e2b":
        return dict(GEMMA4_E2B_CONFIG)
    return dict(GEMMA4_E4B_CONFIG)


def extract_gguf_config(model_path: str) -> dict:
    """Extract Gemma4 config from GGUF metadata (requires gguf library)."""
    try:
        from gguf import GGUFReader
    except ImportError:
        return get_gemma4_config(model_path)
    reader = GGUFReader(model_path)
    fields = reader.fields

    def _get_val(key, default=None):
        f = fields.get(key)
        if f is None:
            return default
        try:
            val = f.parts[-1]
            if hasattr(val, "dtype"):
                if val.ndim == 0:
                    return val.item()
                return int(val) if val.dtype.kind in ("i", "u") else float(val)
            return val
        except Exception:
            return default

    n_layers = _get_val("gemma4.block_count", 0)
    d_model = _get_val("gemma4.embedding_length", 0)
    n_heads = _get_val("gemma4.attention.head_count", 0)
    n_kv_heads = _get_val("gemma4.attention.head_count_kv", n_heads)
    head_dim = d_model // n_heads if n_heads > 0 else 0
    ff_dim = _get_val("gemma4.feed_forward_length", 0)
    max_seq_len = _get_val("gemma4.context_length", 131072)
    norm_eps = _get_val("gemma4.attention.layer_norm_rms_epsilon", 1e-6)
    rope_theta = _get_val("gemma4.rope.freq_base", 1000000.0)
    logit_softcap = _get_val("gemma4.final_logit_softcapping", 30.0)
    sliding_window = _get_val("gemma4.attention.sliding_window", 512)
    shared_kv_layers = _get_val("gemma4.attention.shared_kv_layers", 4)
    ple_dim = _get_val("gemma4.embedding_length_per_layer_input", 0)
    rope_dim = _get_val("gemma4.rope.dimension_count", 0)
    rope_dim_swa = _get_val("gemma4.rope.dimension_count_swa", 0)
    tok_field = fields.get("tokenizer.ggml.tokens")
    vocab_size = len(tok_field.parts[-1]) if tok_field is not None else 262144
    return {
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "ff_dim": ff_dim,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "norm_eps": norm_eps,
        "rope_theta": rope_theta,
        "attention_softcap": 50.0,
        "logit_softcap": logit_softcap,
        "embedding_scale": True,
        "weight_tying": True,
        "activation": "gelu_pytorch_tanh",
        "norm_type": "gemma4_rmsnorm",
        "use_p_rope": True,
        "sliding_window": sliding_window,
        "shared_kv_layers": shared_kv_layers,
        "ple_dim": ple_dim,
        "rope_dim": rope_dim,
        "rope_dim_swa": rope_dim_swa,
    }


def gemma4_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    rsqrt = 1.0 / np.sqrt(variance + eps)
    return (x * rsqrt).astype(np.float32) * (1.0 + weight.astype(np.float32))


def gemma4_attention_softcap(attn: np.ndarray, cap: float = 50.0) -> np.ndarray:
    return np.tanh(attn / cap) * cap


def gemma4_logit_softcap(logits: np.ndarray, cap: float = 30.0) -> np.ndarray:
    return np.tanh(logits / cap) * cap


def gemma4_embed_scale(embed: np.ndarray, d_model: int) -> np.ndarray:
    return embed * np.sqrt(d_model).astype(embed.dtype)


def create_hd_config_for_gemma4(config: dict, hd_dim: int = 8192) -> dict:
    return {
        "hd_dim": hd_dim,
        "vocab_size": config["vocab_size"],
        "n_ngram": 4,
        "n_candidates": min(64, config["vocab_size"] // 4096),
        "coherence_threshold": 0.45,
        "block_size": 8,
        "max_blocks_per_cycle": 4,
        "temperature": 0.8,
    }


def create_kv_config_for_gemma4(config: dict) -> dict:
    head_dim = config["head_dim"]
    return {
        "dim": head_dim,
        "max_size": config["max_seq_len"] // 8,
        "k_bits": 4,
        "v_bits": 2,
        "hadamard_dim": 256,
        "use_spectral": True,
    }


def create_engine_config_for_gemma4(config: dict) -> dict:
    return {
        "hidden_dim": config["d_model"],
        "vocab_size": config["vocab_size"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "block_size": 8,
        "hd_dim": 8192,
        "kv_cache_size": config["max_seq_len"] // 8,
        "coherence_threshold": 0.45,
        "n_candidate_blocks": 16,
    }

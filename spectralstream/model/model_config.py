"""
Model Configuration — architecture-agnostic config reader for any LLM.
Reads from GGUF metadata, safetensors config.json, SSF metadata, or JSON.
"""

from __future__ import annotations

import json
import gzip
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


GGUF_FIELD_MAP: Dict[str, str] = {
    "general.architecture": "architecture",
    "block_count": "n_layers",
    "embedding_length": "d_model",
    "feed_forward_length": "ff_dim",
    "context_length": "max_seq_len",
    "attention.head_count": "n_heads",
    "attention.head_count_kv": "n_kv_heads",
    "attention.layer_norm_rms_epsilon": "norm_eps",
    "attention.sliding_window": "sliding_window",
    "rope.freq_base": "rope_theta",
    "final_logit_softcapping": "logit_softcap",
    "attention.key_length": "head_dim",
    "attention.value_length": "head_dim",
}

GGUF_ARCH_ALIASES: Dict[str, str] = {
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gemma4": "gemma4",
    "llama": "llama",
    "mistral": "mistral",
    "mixtral": "mixtral",
    "qwen2": "qwen2",
    "qwen2.5": "qwen2.5",
    "phi3": "phi3",
    "falcon": "falcon",
    "deepseek2": "deepseek2",
    "starcoder2": "starcoder2",
    "stablelm": "stablelm",
    "mpt": "mpt",
    "gpt2": "gpt2",
    "gptj": "gptj",
    "gpt_neox": "gpt_neox",
    "dbrx": "dbrx",
    "bert": "bert",
    "nomic-bert": "nomic-bert",
    "roberta": "roberta",
}


def _gguf_key(arch: str, field: str) -> str:
    """Build the GGUF metadata key for a given architecture and field."""
    return f"{arch}.{field}"


def _read_gguf_metadata(path: str) -> Dict[str, Any]:
    """Read GGUF metadata fields into a flat dict."""
    try:
        from gguf import GGUFReader
    except ImportError:
        return {}
    try:
        reader = GGUFReader(path)
    except Exception:
        return {}
    fields = reader.fields

    def _get_val(key: str, default=None):
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

    result: Dict[str, Any] = {}
    arch = _get_val("general.architecture", "")
    if arch:
        result["architecture"] = arch

    arch_key = arch if arch else "gguf"
    for gguf_field, our_key in GGUF_FIELD_MAP.items():
        if "." in gguf_field:
            full_key = _gguf_key(arch_key, gguf_field)
        else:
            full_key = gguf_field
        val = _get_val(full_key)
        if val is not None and val != 0:
            result[our_key] = val

    if "d_model" in result and "n_heads" in result and "head_dim" not in result:
        d_model = result["d_model"]
        n_heads = result["n_heads"]
        if n_heads > 0:
            result["head_dim"] = d_model // n_heads

    tok_field = fields.get("tokenizer.ggml.tokens")
    if tok_field is not None:
        try:
            result["vocab_size"] = len(tok_field.parts[-1])
        except Exception:
            pass

    return result


def _read_safetensors_config(path: str) -> Dict[str, Any]:
    """Read config from safetensors companion config.json."""
    p = Path(path)
    for candidate in (
        p.parent / "config.json",
        p.with_suffix(".json"),
        Path(str(p) + "_config.json"),
    ):
        if candidate.exists():
            try:
                with open(candidate) as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _read_ssf_metadata(path: str) -> Dict[str, Any]:
    """Read config from SSF v2 binary format."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"SSF\x02":
                return {}
            f.seek(0)
            data = f.read()
        fmt = "<4sBBHIQQQ32s184s"
        hdr = struct.unpack(fmt, data[:256])
        md_off, md_sz = hdr[6], hdr[7]
        if md_off > 0 and md_sz > 0:
            raw = gzip.decompress(data[md_off : md_off + md_sz])
            return json.loads(raw.decode("utf-8"))
    except Exception:
        pass
    return {}


def _read_json_config(path: str) -> Dict[str, Any]:
    """Read config from a plain JSON file."""
    p = Path(path)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def build_model_config(path: str) -> Dict[str, Any]:
    """Build a model configuration dict from any supported format.

    Detection priority:
    1. GGUF metadata (if `gguf` library available)
    2. Companion config.json (safetensors / PyTorch)
    3. SSF v2 binary metadata
    4. Plain JSON file
    5. Empty dict (fallback)
    """
    config: Dict[str, Any] = {}

    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".gguf":
        config = _read_gguf_metadata(str(p))
    elif ext == ".ssf":
        md = _read_ssf_metadata(str(p))
        c = md.get("config", {})
        if c:
            return c

    if not config:
        config = _read_safetensors_config(str(p))

    if not config:
        config = _read_json_config(str(p))

    if not config and ext == ".gguf":
        config = _read_gguf_metadata(str(p))

    return config


def extract_config_value(config: Dict[str, Any], *keys: str, default=None):
    """Extract a value from config trying multiple key names."""
    for key in keys:
        val = config.get(key)
        if val is not None:
            return val
        # camelCase variants
        alt = key.replace("_", "")
        val = config.get(alt)
        if val is not None:
            return val
    return default


def detect_architecture(config: Dict[str, Any]) -> str:
    """Detect model architecture from config."""
    arch = config.get("architecture", "").lower()
    if arch:
        return arch

    model_type = config.get("model_type", "").lower()
    if model_type:
        return model_type

    for key in config:
        for arch_name in GGUF_ARCH_ALIASES:
            if arch_name in key.lower():
                return arch_name

    return "transformer"

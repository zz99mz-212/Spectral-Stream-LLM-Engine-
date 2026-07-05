from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


ARCH_PATTERNS: Dict[str, List[str]] = {
    "gemma": ["blk.0.attention.wq.weight", "token_embed.weight"],
    "llama": ["model.layers.0.self_attn.q_proj.weight", "lm_head.weight"],
    "mistral": [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ],
    "qwen": [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ],
    "mixtral": ["model.layers.0.block_sparse_moe.gate.weight"],
    "phi": ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.fc1.weight"],
    "falcon": ["transformer.h.0.self_attention.query_key_value.weight"],
    "deepseek": [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ],
}

ARCH_PREFIX_MAP: Dict[str, str] = {
    "gemma": "blk.",
    "llama": "model.layers.",
    "mistral": "model.layers.",
    "qwen": "model.layers.",
    "mixtral": "model.layers.",
    "phi": "model.layers.",
    "falcon": "transformer.h.",
    "deepseek": "model.layers.",
}


@dataclass
class GenericModelConfig:
    hidden_size: int = 1536
    num_layers: int = 35
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    vocab_size: int = 262144
    ffn_size: int = 6144
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_seq_len: int = 131072
    sliding_window: int = 512
    attention_type: str = "gqa"
    activation: str = "gelu"
    expert_count: int = 0
    architecture: str = "gemma"
    tensor_prefix: str = "blk."
    logit_softcap: float = 30.0
    attention_softcap: float = 50.0
    num_shared_kv_layers: int = 20
    global_head_dim: int = 512

    @classmethod
    def from_dict(cls, cfg: dict) -> GenericModelConfig:
        mapping = {
            "hidden_size": "hidden_size",
            "num_hidden_layers": "num_layers",
            "num_attention_heads": "num_heads",
            "num_key_value_heads": "num_kv_heads",
            "head_dim": "head_dim",
            "vocab_size": "vocab_size",
            "intermediate_size": "ffn_size",
            "rms_norm_eps": "norm_eps",
            "rope_theta": "rope_theta",
            "max_position_embeddings": "max_seq_len",
            "sliding_window": "sliding_window",
            "num_experts": "expert_count",
            "logit_softcap": "logit_softcap",
            "attention_softcap": "attention_softcap",
        }
        kw: Dict[str, Any] = {}
        for src, dst in mapping.items():
            if src in cfg:
                kw[dst] = cfg[src]
        rope_params = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
        if isinstance(rope_params, dict):
            if "theta" in rope_params:
                kw["rope_theta"] = rope_params["theta"]
            if "rope_theta" in rope_params:
                kw["rope_theta"] = rope_params["rope_theta"]

        hidden_act = cfg.get("hidden_act", cfg.get("activation_function", ""))
        if hidden_act:
            kw["activation"] = hidden_act

        arch = cls._detect_architecture_from_keys(set(cfg.keys()))
        kw["architecture"] = arch
        kw["tensor_prefix"] = ARCH_PREFIX_MAP.get(arch, "blk.")
        return cls(**kw)

    @classmethod
    def from_tensor_names(cls, names: List[str]) -> GenericModelConfig:
        cfg = cls()
        arch = cls._detect_architecture_from_names(names)
        cfg.architecture = arch
        cfg.tensor_prefix = ARCH_PREFIX_MAP.get(arch, "blk.")
        cfg._infer_from_tensors(names)
        return cfg

    @classmethod
    def from_ssf_path(cls, path: str) -> GenericModelConfig:
        base = path.rsplit(".", 1)[0] if "." in path else path
        for candidate in (
            base + ".json",
            base + "_config.json",
            str(Path(path).parent / "config.json"),
        ):
            p = Path(candidate)
            if p.exists():
                try:
                    with open(p) as f:
                        meta = json.load(f)
                    tc = meta.get("text_config", meta)
                    return cls.from_dict(tc)
                except Exception:
                    pass
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
            if magic == b"SSF\x02":
                from spectralstream.format.header import SSFHeader
                import gzip
                import struct

                f.seek(0)
                data = f.read()
                hdr = SSFHeader.unpack(data[:256])
                if hdr.metadata_offset > 0 and hdr.metadata_size > 0:
                    raw_md = bytes(
                        data[
                            hdr.metadata_offset : hdr.metadata_offset
                            + hdr.metadata_size
                        ]
                    )
                    md = json.loads(gzip.decompress(raw_md).decode("utf-8"))
                    c = md.get("config", {})
                    if c:
                        return cls.from_dict(c)
        except Exception:
            pass
        return cls()

    def _infer_from_tensors(self, names: List[str]) -> None:
        prefix = self.tensor_prefix
        layer_indices: set = set()
        has_gate = False
        for name in names:
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == prefix.rstrip("."):
                try:
                    lidx = int(parts[1])
                    layer_indices.add(lidx)
                except ValueError:
                    pass
            if "gate" in name or "w_gate" in name:
                has_gate = True
        if layer_indices:
            self.num_layers = max(layer_indices) + 1
        for name in names:
            if "wq.weight" in name or "q_proj.weight" in name:
                tensor = self._peek_shape(name, names)
                if tensor and len(tensor) >= 2:
                    self.num_heads = tensor[0] // self.head_dim
                break
            if "wk.weight" in name or "k_proj.weight" in name:
                tensor = self._peek_shape(name, names)
                if tensor and len(tensor) >= 2:
                    self.num_kv_heads = tensor[0] // self.head_dim
                break
        for name in names:
            if "w_gate" in name or "gate_proj" in name:
                t = self._peek_shape(name, names)
                if t and len(t) >= 1:
                    self.ffn_size = t[0]
                break
        for name in names:
            if "token_embed.weight" in name or "embed_tokens.weight" in name:
                t = self._peek_shape(name, names)
                if t and len(t) >= 2:
                    self.vocab_size = t[0]
                    self.hidden_size = t[1]
                break
        if "expert" in str(names) or "moe" in str(names).lower():
            self.expert_count = 8

    @staticmethod
    def _peek_shape(name: str, all_names: List[str]) -> Optional[tuple]:
        return None

    @staticmethod
    def _detect_architecture_from_names(names: List[str]) -> str:
        name_set = set(names)
        for arch, patterns in ARCH_PATTERNS.items():
            if any(p in name_set for p in patterns):
                return arch
        for name in names:
            if name.startswith("blk."):
                return "gemma"
            if name.startswith("model.layers."):
                continue
        name_str = " ".join(names).lower()
        if "blk." in name_str:
            return "gemma"
        if "model.layers." in name_str:
            return "llama"
        return "gemma"

    @staticmethod
    def _detect_architecture_from_keys(keys: set) -> str:
        text = " ".join(str(k).lower() for k in keys)
        if "gemma" in text:
            return "gemma"
        if "mistral" in text:
            return "mistral"
        if "qwen" in text:
            return "qwen"
        if "mixtral" in text:
            return "mixtral"
        if "falcon" in text:
            return "falcon"
        if "phi" in text:
            return "phi"
        if "deepseek" in text:
            return "deepseek"
        if "llama" in text:
            return "llama"
        return "gemma"

    def to_gemma_config(self):
        from spectralstream.inference.config import Gemma4Config

        return Gemma4Config(
            hidden_size=self.hidden_size,
            intermediate_size=self.ffn_size,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            num_hidden_layers=self.num_layers,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_seq_len,
            sliding_window=self.sliding_window,
            head_dim=self.head_dim,
            global_head_dim=self.global_head_dim,
            rms_norm_eps=self.norm_eps,
            rope_theta=self.rope_theta,
            attention_softcap=self.attention_softcap,
            logit_softcap=self.logit_softcap,
        )

    @property
    def HEAD_DIM(self) -> int:
        return self.head_dim

    @property
    def HIDDEN_SIZE(self) -> int:
        return self.hidden_size

    @property
    def NUM_HIDDEN_LAYERS(self) -> int:
        return self.num_layers

    @property
    def NUM_ATTENTION_HEADS(self) -> int:
        return self.num_heads

    @property
    def NUM_KEY_VALUE_HEADS(self) -> int:
        return self.num_kv_heads

    @property
    def VOCAB_SIZE(self) -> int:
        return self.vocab_size

    @property
    def MAX_POSITION_EMBEDDINGS(self) -> int:
        return self.max_seq_len

    @property
    def NORM_EPS(self) -> float:
        return self.norm_eps

    @property
    def ROPE_THETA(self) -> float:
        return self.rope_theta

    @property
    def LOGIT_SOFTCAP(self) -> float:
        return self.logit_softcap

    @property
    def head_group_size(self) -> int:
        return self.num_heads // max(self.num_kv_heads, 1)

    def __getattr__(self, name: str):
        compat = {
            "INTERMEDIATE_SIZE": "ffn_size",
            "SLIDING_WINDOW": "sliding_window",
            "ATTENTION_SOFTCAP": "attention_softcap",
            "NUM_SHARED_KV_LAYERS": "num_shared_kv_layers",
            "GLOBAL_HEAD_DIM": "global_head_dim",
        }
        if name in compat:
            return getattr(self, compat[name])
        raise AttributeError(f"GenericModelConfig has no attribute '{name}'")

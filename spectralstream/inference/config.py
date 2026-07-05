from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path


class Gemma4Config:
    HIDDEN_SIZE = 1536
    INTERMEDIATE_SIZE = 6144
    NUM_ATTENTION_HEADS = 8
    NUM_KEY_VALUE_HEADS = 1
    NUM_HIDDEN_LAYERS = 35
    VOCAB_SIZE = 262144
    MAX_POSITION_EMBEDDINGS = 131072
    SLIDING_WINDOW = 512
    NUM_SHARED_KV_LAYERS = 20
    HEAD_DIM = 256
    GLOBAL_HEAD_DIM = 512
    NORM_EPS = 1e-6
    ROPE_THETA = 1000000.0
    ATTENTION_SOFTCAP = 50.0
    LOGIT_SOFTCAP = 30.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k.upper(), v)

    @classmethod
    def from_ssf(cls, model_path: str) -> "Gemma4Config":
        cfg = cls()
        # First try loading from a companion config.json
        base = model_path.rsplit(".", 1)[0] if "." in model_path else model_path
        config_json_path = None
        for candidate in (
            base + ".json",
            base + "_config.json",
            str(Path(model_path).parent / "config.json"),
        ):
            p = Path(candidate)
            if p.exists():
                config_json_path = str(p)
                break

        if config_json_path is not None:
            try:
                with open(config_json_path) as f:
                    meta = json.load(f)
                tc = meta.get("text_config", meta)
                mapping = {
                    "hidden_size": "HIDDEN_SIZE",
                    "intermediate_size": "INTERMEDIATE_SIZE",
                    "num_attention_heads": "NUM_ATTENTION_HEADS",
                    "num_key_value_heads": "NUM_KEY_VALUE_HEADS",
                    "num_hidden_layers": "NUM_HIDDEN_LAYERS",
                    "vocab_size": "VOCAB_SIZE",
                    "max_position_embeddings": "MAX_POSITION_EMBEDDINGS",
                    "sliding_window": "SLIDING_WINDOW",
                    "head_dim": "HEAD_DIM",
                    "global_head_dim": "GLOBAL_HEAD_DIM",
                    "rms_norm_eps": "NORM_EPS",
                    "final_logit_softcapping": "LOGIT_SOFTCAP",
                }
                for src, dst in mapping.items():
                    if src in tc:
                        setattr(cfg, dst, tc[src])
                # RoPE theta from attention config
                rope_params = tc.get("rope_parameters", {})
                for attn_type in ("full_attention", "sliding_attention"):
                    rp = rope_params.get(attn_type, {})
                    if "rope_theta" in rp:
                        cfg.ROPE_THETA = rp["rope_theta"]
                        break
                # Attention softcap
                if "attention_logit_cap" in meta.get("audio_config", {}):
                    cfg.ATTENTION_SOFTCAP = meta["audio_config"]["attention_logit_cap"]
                # Shared KV layers
                if "num_kv_shared_layers" in tc:
                    cfg.NUM_SHARED_KV_LAYERS = tc["num_kv_shared_layers"]
                return cfg
            except Exception:
                pass

        # Fallback: try SSF metadata
        try:
            with open(model_path, "rb") as f:
                magic = f.read(4)
                if magic != b"SSF\x02":
                    return cfg
                f.seek(0)
                data = f.read()
            h = cls._parse_ssf_header(data)
            md = cls._parse_ssf_metadata(data, h)
            if md:
                c = md.get("config", {})
                for k, v in c.items():
                    key = k.upper()
                    if hasattr(cfg, key):
                        setattr(cfg, key, v)
        except Exception:
            pass
        return cfg

    @staticmethod
    def _parse_ssf_header(data: bytes) -> dict:
        fmt = "<4sBBHIQQQ32s184s"
        (
            magic,
            ver,
            min_ver,
            flags,
            n_tensors,
            total_orig,
            total_comp,
            md_off,
            md_sz,
            cs,
            _,
        ) = struct.unpack(fmt, data[:256])
        return {
            "magic": magic,
            "n_tensors": n_tensors,
            "md_off": md_off,
            "md_sz": md_sz,
        }

    @staticmethod
    def _parse_ssf_metadata(data: bytes, h: dict) -> dict:
        off, sz = h.get("md_off", 0), h.get("md_sz", 0)
        if off > 0 and sz > 0:
            try:
                return json.loads(gzip.decompress(data[off : off + sz]).decode("utf-8"))
            except Exception:
                pass
        return {}

    @property
    def head_group_size(self) -> int:
        return self.NUM_ATTENTION_HEADS // max(self.NUM_KEY_VALUE_HEADS, 1)

    def is_sliding_window_layer(self, layer_idx: int) -> bool:
        idx = layer_idx % 6
        return idx not in (4,)

    def is_shared_kv_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.NUM_HIDDEN_LAYERS - self.NUM_SHARED_KV_LAYERS

    def shared_kv_source(self, layer_idx: int) -> int:
        if not self.is_shared_kv_layer(layer_idx):
            return layer_idx
        offset = layer_idx - (self.NUM_HIDDEN_LAYERS - self.NUM_SHARED_KV_LAYERS)
        return offset % (self.NUM_HIDDEN_LAYERS - self.NUM_SHARED_KV_LAYERS)

"""Backward-compat re-export — Gemma 4 config migrated to model.gemma4_config."""

from spectralstream.model.gemma4_config import (
    GEMMA4_E2B_CONFIG,
    GEMMA4_E4B_CONFIG,
    extract_gguf_config,
    detect_gemma4_variant,
    get_gemma4_config,
    gemma4_rmsnorm,
    gemma4_attention_softcap,
    gemma4_logit_softcap,
    gemma4_embed_scale,
    create_hd_config_for_gemma4,
    create_kv_config_for_gemma4,
    create_engine_config_for_gemma4,
)

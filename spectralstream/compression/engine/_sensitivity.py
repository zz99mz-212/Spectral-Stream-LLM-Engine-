"""Layer sensitivity maps and helper.

Uses compact dict lookups — O(1) per name, memory proportional to entry count.
"""

from typing import Dict

LAYER_SENSITIVITY: Dict[str, float] = {
    "embed": 1.0,
    "tok_embeddings": 1.0,
    "wte": 1.0,
    "attn_q": 1.0,
    "q_proj": 1.0,
    "wq": 1.0,
    "query": 1.0,
    "attn_k": 0.92,
    "k_proj": 0.92,
    "wk": 0.92,
    "key": 0.92,
    "attn_v": 0.88,
    "v_proj": 0.88,
    "wv": 0.88,
    "value": 0.88,
    "attn_o": 1.0,
    "o_proj": 1.0,
    "wo": 1.0,
    "attn_norm": 0.7,
    "ln_1": 0.7,
    "layernorm": 0.7,
    "ffn_gate": 0.55,
    "gate_proj": 0.55,
    "w1": 0.55,
    "ffn_up": 0.60,
    "up_proj": 0.60,
    "w3": 0.60,
    "ffn_down": 0.65,
    "down_proj": 0.65,
    "w2": 0.65,
    "ffn_norm": 0.50,
    "ln_2": 0.50,
    "norm": 0.50,
    "final_norm": 0.50,
    "rms_norm": 0.50,
    "output": 1.0,
    "lm_head": 1.0,
    "head": 1.0,
}

SUBLEVEL_SENSITIVITY: Dict[str, float] = {
    "qkv": 0.95,
    "rotary": 0.60,
    "bias": 0.95,
    "gamma": 0.50,
    "beta": 0.50,
}


def _get_sensitivity(name: str) -> float:
    name_lower = name.lower()
    sens = 0.5
    for key, val in LAYER_SENSITIVITY.items():
        if key in name_lower:
            sens = max(sens, val)
    for key, val in SUBLEVEL_SENSITIVITY.items():
        if key in name_lower:
            sens = max(sens, val)
    return min(sens, 1.0)

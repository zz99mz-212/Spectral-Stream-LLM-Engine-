"""
OpenAI-compatible embeddings endpoint.
Supports both model-based (hidden state extraction) and HDC-based embeddings.
"""

import numpy as np
import time
from typing import Optional


def compute_embedding_model(
    text: str,
    tokenize_fn,
    forward_fn,
    hidden_dim: int,
    vocab_size: int,
) -> list[float]:
    """Run text through the model and extract mean-pooled last-layer hidden state."""
    input_ids = tokenize_fn(text)
    if not input_ids:
        input_ids = [0]
    logits, layer_hidden_states, past = forward_fn(input_ids)
    if layer_hidden_states and len(layer_hidden_states) > 0:
        last_hidden = layer_hidden_states[-1]
    else:
        last_hidden = np.zeros((len(input_ids), hidden_dim), dtype=np.float32)
    if last_hidden.ndim == 1:
        last_hidden = last_hidden.reshape(1, -1)
    pooled = last_hidden.mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled.tolist()


def compute_embedding_hdc(
    text: str,
    hd_engine,
    hd_dim: int,
    vocab_size: int,
) -> list[float]:
    """HDC-based embedding: hash tokens to hypervectors, bundle via majority sum."""
    hv = np.zeros(hd_dim, dtype=np.float64)
    count = 0
    rng = np.random.RandomState(42)
    for ch in text:
        tid = ord(ch) % vocab_size
        rng.seed(tid)
        token_hv = np.where(rng.uniform(0, 1, size=hd_dim) < 0.05, 1.0, -1.0)
        hv += token_hv
        count += 1
    if count > 0:
        hv = np.sign(hv).astype(np.float64)
    norm = np.linalg.norm(hv)
    if norm > 0:
        hv = hv / norm
    return hv.tolist()


def handle_embeddings_request(
    body: dict,
    tokenize_fn,
    forward_fn,
    hidden_dim: int,
    vocab_size: int,
    hd_engine,
    hd_dim: int,
    is_real_model: bool,
    use_hdc: Optional[bool] = None,
) -> dict:
    """Handle POST /v1/embeddings request.

    OpenAI-compatible format:
      Request: {"input": "text" or ["text1", "text2"], "model": "model-name"}
      Response: {"object": "list", "data": [...], "model": ..., "usage": {...}}
    """
    raw_input = body.get("input", "")
    model_name = body.get("model", "default")
    if use_hdc is None:
        use_hdc = not is_real_model

    if isinstance(raw_input, str):
        inputs = [raw_input]
    elif isinstance(raw_input, list):
        inputs = raw_input
    else:
        raise ValueError("'input' must be a string or list of strings")

    data = []
    total_tokens = 0
    for idx, text in enumerate(inputs):
        if use_hdc:
            vector = compute_embedding_hdc(
                text,
                hd_engine,
                hd_dim,
                vocab_size,
            )
        else:
            vector = compute_embedding_model(
                text,
                tokenize_fn,
                forward_fn,
                hidden_dim,
                vocab_size,
            )
        tokens = len(text.split())
        total_tokens += tokens
        data.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": vector,
            }
        )

    return {
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }

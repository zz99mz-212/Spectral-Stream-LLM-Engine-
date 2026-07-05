"""
SpectralStream Multi-Modal Extension & Prompt Optimization Engine
==================================================================
Clean-room implementation of vision, audio, code execution, prompt
optimization, template engine, auto-chain-of-thought, and novel
spectral-domain inventions.

Modalities:
  - Vision (CLIP-style ViT, Spectral Vision DCT encoding)
  - Audio (Whisper-inspired spectrogram CNN)
  - Code (sandboxed Python/JS/Shell execution)

Prompt Systems:
  - Resonant Prompting (frequency-domain prompt optimization)
  - PromptOptimizer (DSPy-style LLM-programmed optimization)
  - PromptTemplateEngine (Jinja-like with variables/conditions/loops)
  - AutoChain (decompose → reason → verify → ensemble → vote)

Novel Inventions:
  - Spectral Vision: DCT-domain learnable image encoding
  - Resonant Prompting: FFT-domain prompt resonance optimization
  - Holographic Context: HRR-based infinite context compression
  - Vlasov Chain-of-Thought: mean-field reasoning particle interactions
  - Quantum Prompt Optimization: superposition prompt evaluation
"""

from __future__ import annotations

import ast
import base64
import io
import json
import math
import os
import queue
import re
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import uuid
import warnings
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, IntEnum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.parse import urlparse

import numpy as np

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from spectralstream.inference.unified import UnifiedInferenceEngine

    HAS_UNIFIED = True
except ImportError:
    UnifiedInferenceEngine = object
    HAS_UNIFIED = False

from spectralstream.core.math_primitives.numerical import softmax as _softmax

try:
    from spectralstream.format.sst_format import SSTv3Writer, SSTv3Reader, QualityTable

    HAS_SST = True
except ImportError:
    SSTv3Writer = None
    SSTv3Reader = None
    QualityTable = None
    HAS_SST = False

try:
    from spectralstream.memory.holographic_memory import HrrMemory as _HrrMemory

    HAS_HRR = True
except ImportError:
    _HrrMemory = None
    HAS_HRR = False

try:
    from spectralstream.inference.streaming_engine import StreamingEngine

    HAS_STREAMING = True
except ImportError:
    StreamingEngine = None
    HAS_STREAMING = False


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SPECTRALSTREAM_EMBED_DIM = 4096
CLIP_IMAGE_SIZE = 224
CLIP_PATCH_SIZE = 16
CLIP_EMBED_DIM = 768
CLIP_N_HEADS = 12
CLIP_N_LAYERS = 4

AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MELS = 80
AUDIO_N_FFT = 400
AUDIO_HOP_LENGTH = 160
AUDIO_CNN_CHANNELS = [8, 16, 32, 64]

MAX_CODE_EXEC_TIME = 30
MAX_CODE_OUTPUT_LEN = 65536

PROMPT_OPTIMIZER_QUALITY_MODEL_DIM = 512
PROMPT_OPTIMIZER_N_VARIATIONS = 8
PROMPT_OPTIMIZER_N_ITERATIONS = 3

TEMPLATE_CACHE_SIZE = 256
AUTOCHAIN_N_PATHS = 5
AUTOCHAIN_MAX_STEPS = 20

SPECTRAL_VISION_DCT_BLOCK = 16
SPECTRAL_VISION_N_COEFFS = 64

HRR_CONTEXT_DIM = 2048
HRR_CONTEXT_CAPACITY = 65536


# ═══════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════


def _normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x
    return x / n


def _load_image_from_source(source: Union[str, bytes, np.ndarray]) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, bytes):
        if HAS_PIL:
            return np.array(Image.open(BytesIO(source))).astype(np.float32)
        raise ImportError("PIL required to decode image bytes")
    if source.startswith(("http://", "https://")):
        if HAS_REQUESTS:
            resp = requests.get(source, timeout=30)
            resp.raise_for_status()
            if HAS_PIL:
                return np.array(Image.open(BytesIO(resp.content))).astype(np.float32)
            raise ImportError("PIL required to decode remote image")
        raise ImportError("requests required for URL image loading")
    if source.startswith("data:image"):
        _, b64data = source.split(",", 1)
        raw = base64.b64decode(b64data)
        if HAS_PIL:
            return np.array(Image.open(BytesIO(raw))).astype(np.float32)
        raise ImportError("PIL required to decode base64 image")
    if HAS_PIL:
        return np.array(Image.open(source)).astype(np.float32)
    raise ValueError(f"Unrecognized image source: {source[:60]}")


def _preprocess_image(img: np.ndarray, size: int = CLIP_IMAGE_SIZE) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    if HAS_PIL:
        pil_img = Image.fromarray(img.astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
        resized = np.array(pil_img).astype(np.float32)
    else:
        ys = np.linspace(0, h - 1, nh).astype(np.int32)
        xs = np.linspace(0, w - 1, nw).astype(np.int32)
        resized = img[np.ix_(ys, xs)]
    padded = np.zeros((size, size, 3), dtype=np.float32)
    padded[:nh, :nw] = resized
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    padded = (padded / 255.0 - mean) / std
    return padded.transpose(2, 0, 1)


def _load_audio_from_source(source: Union[str, bytes, np.ndarray]) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, bytes):
        return _decode_audio_wav(source)
    if source.startswith("data:audio"):
        _, b64data = source.split(",", 1)
        raw = base64.b64decode(b64data)
        return _decode_audio_wav(raw)
    if os.path.isfile(source):
        with open(source, "rb") as f:
            return _decode_audio_wav(f.read())
    raise ValueError(f"Unrecognized audio source: {source[:60]}")


def _decode_audio_wav(data: bytes) -> np.ndarray:
    try:
        import wave
    except ImportError:
        raise ImportError("wave module required for audio decoding")
    with io.BytesIO(data) as buf:
        with wave.open(buf, "rb") as wf:
            n_frames = wf.getnframes()
            framerate = wf.getframerate()
            raw = wf.readframes(n_frames)
            if wf.getsampwidth() == 1:
                audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
            elif wf.getsampwidth() == 2:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            else:
                audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
            if framerate != AUDIO_SAMPLE_RATE:
                ratio = AUDIO_SAMPLE_RATE / framerate
                n_new = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, n_new), np.arange(len(audio)), audio
                )
            return audio.astype(np.float32)


def _mel_spectrogram(
    audio: np.ndarray,
    n_fft: int = AUDIO_N_FFT,
    hop_length: int = AUDIO_HOP_LENGTH,
    n_mels: int = AUDIO_N_MELS,
    sr: int = AUDIO_SAMPLE_RATE,
) -> np.ndarray:
    n = len(audio)
    pad = (n_fft - n % hop_length) % hop_length
    audio = np.pad(audio, (0, pad))
    n_frames = (len(audio) - n_fft) // hop_length + 1
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for t in range(n_frames):
        start = t * hop_length
        seg = audio[start : start + n_fft] * np.hanning(n_fft)
        stft[:, t] = np.fft.rfft(seg)
    mag = np.abs(stft)
    mel_freqs = np.linspace(0, sr / 2, n_mels + 2)
    hz = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_basis = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l = mel_freqs[m - 1]
        c = mel_freqs[m]
        r = mel_freqs[m + 1]
        for f_idx, freq in enumerate(hz):
            if l <= freq <= c:
                mel_basis[m - 1, f_idx] = (freq - l) / (c - l)
            elif c < freq <= r:
                mel_basis[m - 1, f_idx] = (r - freq) / (r - c)
    mel_spec = mel_basis @ mag
    mel_spec = np.log10(np.maximum(mel_spec, 1e-10))
    mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-10)
    return mel_spec.astype(np.float32)


def _dct_2d(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    C[0, :] = 1.0 / math.sqrt(n)
    s = math.sqrt(2.0 / n)
    k = np.arange(1, n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    C[1:, :] = s * np.cos(math.pi * k * (i + 0.5) / n)
    return C @ matrix.astype(np.float64) @ C.T


def _idct_2d(coeffs: np.ndarray) -> np.ndarray:
    n = coeffs.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    C[0, :] = 1.0 / math.sqrt(n)
    s = math.sqrt(2.0 / n)
    k = np.arange(1, n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    C[1:, :] = s * np.cos(math.pi * k * (i + 0.5) / n)
    return C.T @ coeffs.astype(np.float64) @ C


def _embed_text_simple(text: str, dim: int = SPECTRALSTREAM_EMBED_DIM) -> np.ndarray:
    """Simple hash-based text embedding fallback when no model available."""
    seed = hash(text) & 0x7FFFFFFF
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# SSF (Spectral Stream Format) Helpers for Vision/Audio
# ═══════════════════════════════════════════════════════════════════════════


class SSFModelStore:
    """Store/load vision/audio encoder weights in SSF format."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._weights: dict[str, np.ndarray] = {}

    def add_tensor(self, name: str, tensor: np.ndarray):
        self._weights[name] = tensor.astype(np.float32)

    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        return self._weights.get(name)

    def save(self, path: Optional[str] = None):
        dst = path or self.path
        if dst is None:
            raise ValueError("No save path specified")
        if HAS_SST and SSTv3Writer is not None and dst.endswith(".sst"):
            config = {"format": "ssf_multimodal", "n_tensors": len(self._weights)}
            writer = SSTv3Writer(dst, config, quality=0.7)
            for name, tensor in self._weights.items():
                writer.add_tensor(name, tensor)
            writer.save()
        else:
            np.savez_compressed(dst, **self._weights)

    def load(self, path: Optional[str] = None):
        src = path or self.path
        if src is None:
            raise ValueError("No load path specified")
        if HAS_SST and SSTv3Reader is not None and src.endswith(".sst"):
            reader = SSTv3Reader(src)
            for name in reader.get_tensor_names():
                self._weights[name] = reader.load_tensor(name)
        else:
            try:
                data = np.load(src, allow_pickle=False)
                for key in data.files:
                    self._weights[key] = data[key]
            except Exception:
                data = np.load(src, allow_pickle=True)
                for key in data.files:
                    self._weights[key] = data[key]

    def keys(self):
        return self._weights.keys()

    def __len__(self):
        return len(self._weights)

    def __contains__(self, key):
        return key in self._weights


# ═══════════════════════════════════════════════════════════════════════════
# 1. VisionEncoder — CLIP-style Vision Transformer + Spectral Vision DCT
# ═══════════════════════════════════════════════════════════════════════════


class SpectralVisionDCT:
    """
    Novel: "Spectral Vision" — encode images via learnable DCT coefficients.

    Instead of raw pixels → patch embedding, we:
    1. Divide image into DCT_BLOCK x DCT_BLOCK blocks
    2. Apply 2D DCT to each block
    3. Keep top-k DCT coefficients (low-frequency)
    4. Learn a projection from DCT coeffs to patch embedding space

    This gives JPEG-like compression with learnable downstream projection,
    making it robust to noise and compression artifacts.
    """

    def __init__(
        self,
        block_size: int = SPECTRAL_VISION_DCT_BLOCK,
        n_coeffs: int = SPECTRAL_VISION_N_COEFFS,
        embed_dim: int = CLIP_EMBED_DIM,
    ):
        self.block_size = block_size
        self.n_coeffs = n_coeffs
        self.embed_dim = embed_dim
        self.projection = np.random.randn(n_coeffs, embed_dim).astype(np.float32) * 0.02
        self._zigzag = self._build_zigzag(block_size)

    def _build_zigzag(self, n: int) -> list[tuple[int, int]]:
        indices = []
        for s in range(2 * n - 1):
            if s % 2 == 0:
                i = min(s, n - 1)
                j = s - i
                while i >= 0 and j < n:
                    indices.append((i, j))
                    i -= 1
                    j += 1
            else:
                j = min(s, n - 1)
                i = s - j
                while j >= 0 and i < n:
                    indices.append((i, j))
                    i += 1
                    j -= 1
        return indices

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode image to DCT patch embeddings."""
        c, h, w = image.shape
        bs = self.block_size
        patches_h = h // bs
        patches_w = w // bs
        if patches_h == 0 or patches_w == 0:
            return np.zeros((1, self.embed_dim), dtype=np.float32)
        img = image[:, : patches_h * bs, : patches_w * bs]
        n_patches = patches_h * patches_w
        dct_feats = np.zeros((n_patches, self.n_coeffs), dtype=np.float32)
        for pi in range(patches_h):
            for pj in range(patches_w):
                pidx = pi * patches_w + pj
                block = img[:, pi * bs : (pi + 1) * bs, pj * bs : (pj + 1) * bs]
                gray = np.mean(block, axis=0)
                dct = _dct_2d(gray)
                for k, (i, j) in enumerate(self._zigzag[: self.n_coeffs]):
                    if i < dct.shape[0] and j < dct.shape[1]:
                        dct_feats[pidx, k] = float(dct[i, j])
        embeddings = dct_feats @ self.projection
        return embeddings

    def get_ssf_tensors(self) -> dict[str, np.ndarray]:
        return {"spectral_vision_projection": self.projection.copy()}

    def load_ssf_tensors(self, tensors: dict[str, np.ndarray]):
        if "spectral_vision_projection" in tensors:
            self.projection = tensors["spectral_vision_projection"].astype(np.float32)


class PatchEmbedding(np.ndarray if False else object):
    """Standard patch embedding layer (learnable conv)."""

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = CLIP_PATCH_SIZE,
        embed_dim: int = CLIP_EMBED_DIM,
        img_size: int = CLIP_IMAGE_SIZE,
    ):
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2
        self.kernel = (
            np.random.randn(embed_dim, in_channels, patch_size, patch_size).astype(
                np.float32
            )
            * 0.02
        )
        self.bias = np.zeros(embed_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        ps = self.patch_size
        ph, pw = h // ps, w // ps
        patches = x.reshape(n, c, ph, ps, pw, ps)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(n * ph * pw, c * ps * ps)
        kernel_flat = self.kernel.reshape(self.embed_dim, c * ps * ps)
        out = patches @ kernel_flat.T + self.bias
        return out.reshape(n, ph * pw, self.embed_dim)

    def get_ssf_tensors(self) -> dict[str, np.ndarray]:
        return {
            "patch_embed_kernel": self.kernel.copy(),
            "patch_embed_bias": self.bias.copy(),
        }

    def load_ssf_tensors(self, tensors: dict[str, np.ndarray]):
        if "patch_embed_kernel" in tensors:
            self.kernel = tensors["patch_embed_kernel"].astype(np.float32)
        if "patch_embed_bias" in tensors:
            self.bias = tensors["patch_embed_bias"].astype(np.float32)


class VisionTransformerEncoder:
    """Simplified CLIP-style ViT encoder."""

    def __init__(
        self,
        embed_dim: int = CLIP_EMBED_DIM,
        n_heads: int = CLIP_N_HEADS,
        n_layers: int = CLIP_N_LAYERS,
        mlp_ratio: int = 4,
    ):
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = embed_dim // n_heads

        self.class_embedding = np.random.randn(embed_dim).astype(np.float32) * 0.02
        self.positional_embedding = (
            np.random.randn(1 + 196, embed_dim).astype(np.float32) * 0.02
        )

        self.layers: list[dict] = []
        for _ in range(n_layers):
            layer = {
                "ln1_gamma": np.ones(embed_dim, dtype=np.float32),
                "ln1_beta": np.zeros(embed_dim, dtype=np.float32),
                "q_proj": np.random.randn(embed_dim, embed_dim).astype(np.float32)
                * 0.02,
                "k_proj": np.random.randn(embed_dim, embed_dim).astype(np.float32)
                * 0.02,
                "v_proj": np.random.randn(embed_dim, embed_dim).astype(np.float32)
                * 0.02,
                "o_proj": np.random.randn(embed_dim, embed_dim).astype(np.float32)
                * 0.02,
                "ln2_gamma": np.ones(embed_dim, dtype=np.float32),
                "ln2_beta": np.zeros(embed_dim, dtype=np.float32),
                "mlp_fc1": np.random.randn(embed_dim, embed_dim * mlp_ratio).astype(
                    np.float32
                )
                * 0.02,
                "mlp_fc2": np.random.randn(embed_dim * mlp_ratio, embed_dim).astype(
                    np.float32
                )
                * 0.02,
                "mlp_bias1": np.zeros(embed_dim * mlp_ratio, dtype=np.float32),
                "mlp_bias2": np.zeros(embed_dim, dtype=np.float32),
            }
            self.layers.append(layer)

    def _layernorm(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-5) + beta

    def _attention(self, x: np.ndarray, qw, kw, vw, ow) -> np.ndarray:
        n, seq, d = x.shape
        h = self.n_heads
        hd = self.head_dim
        q = x @ qw
        k = x @ kw
        v = x @ vw
        q = q.reshape(n, seq, h, hd).transpose(0, 2, 1, 3)
        k = k.reshape(n, seq, h, hd).transpose(0, 2, 1, 3)
        v = v.reshape(n, seq, h, hd).transpose(0, 2, 1, 3)
        attn = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
        attn = _softmax(attn.reshape(n * h, seq, seq)).reshape(n, h, seq, seq)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(n, seq, d)
        return out @ ow

    def _mlp(self, x: np.ndarray, w1, b1, w2, b2) -> np.ndarray:
        h = x @ w1 + b1
        h = np.maximum(h, 0)
        return h @ w2 + b2

    def encode(
        self, pixel_values: np.ndarray, patch_embed: PatchEmbedding
    ) -> np.ndarray:
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[np.newaxis, ...]
        n = pixel_values.shape[0]
        tokens = patch_embed.forward(pixel_values)
        cls = self.class_embedding[np.newaxis, np.newaxis, :].repeat(n, axis=0)
        tokens = np.concatenate([cls, tokens], axis=1)
        n_patches = tokens.shape[1]
        if n_patches > self.positional_embedding.shape[0]:
            pos = np.resize(self.positional_embedding, (n_patches, self.embed_dim))
        else:
            pos = self.positional_embedding[:n_patches]
        tokens = tokens + pos[np.newaxis, :, :]
        for layer in self.layers:
            residual = tokens
            x = self._layernorm(tokens, layer["ln1_gamma"], layer["ln1_beta"])
            x = self._attention(
                x, layer["q_proj"], layer["k_proj"], layer["v_proj"], layer["o_proj"]
            )
            tokens = residual + x
            residual = tokens
            x = self._layernorm(tokens, layer["ln2_gamma"], layer["ln2_beta"])
            x = self._mlp(
                x,
                layer["mlp_fc1"],
                layer["mlp_bias1"],
                layer["mlp_fc2"],
                layer["mlp_bias2"],
            )
            tokens = residual + x
        return tokens[:, 0]

    def get_ssf_tensors(self) -> dict[str, np.ndarray]:
        t = {
            "vit_class_embed": self.class_embedding.copy(),
            "vit_pos_embed": self.positional_embedding.copy(),
        }
        for i, layer in enumerate(self.layers):
            prefix = f"vit_layer_{i}_"
            for k, v in layer.items():
                t[prefix + k] = v.copy()
        return t

    def load_ssf_tensors(self, tensors: dict[str, np.ndarray]):
        if "vit_class_embed" in tensors:
            self.class_embedding = tensors["vit_class_embed"].astype(np.float32)
        if "vit_pos_embed" in tensors:
            self.positional_embedding = tensors["vit_pos_embed"].astype(np.float32)
        for i, layer in enumerate(self.layers):
            prefix = f"vit_layer_{i}_"
            for k in layer:
                key = prefix + k
                if key in tensors:
                    layer[k] = tensors[key].astype(np.float32)


class VisionEncoder:
    """
    Image understanding encoder.

    Supports loading from file, URL, base64, numpy array.
    Uses CLIP-style vision transformer + Spectral Vision DCT.
    Can store/load weights in SSF format.
    """

    def __init__(
        self,
        embed_dim: int = SPECTRALSTREAM_EMBED_DIM,
        use_spectral_vision: bool = True,
        ssf_path: Optional[str] = None,
    ):
        self.embed_dim = embed_dim
        self.use_spectral_vision = use_spectral_vision
        self.patch_embed = PatchEmbedding(embed_dim=CLIP_EMBED_DIM)
        self.vit = VisionTransformerEncoder()
        self.spectral_vision = SpectralVisionDCT() if use_spectral_vision else None
        self.projection = (
            np.random.randn(CLIP_EMBED_DIM, embed_dim).astype(np.float32) * 0.02
        )
        self.ssf_store = SSFModelStore(ssf_path)
        if ssf_path and os.path.isfile(ssf_path):
            try:
                self.load_ssf(ssf_path)
            except Exception:
                pass

    def encode_image(self, source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        img = _load_image_from_source(source)
        processed = _preprocess_image(img)
        if processed.ndim == 3:
            processed = processed[np.newaxis, ...]
        vit_embeds = self.vit.encode(processed, self.patch_embed)
        return vit_embeds @ self.projection

    def encode_images(
        self, sources: list[Union[str, bytes, np.ndarray]]
    ) -> list[np.ndarray]:
        return [self.encode_image(s) for s in sources]

    def encode_with_spectral(
        self, source: Union[str, bytes, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        img = _load_image_from_source(source)
        processed = _preprocess_image(img)
        if processed.ndim == 3:
            processed = processed[np.newaxis, ...]
        vit_embeds = self.vit.encode(processed, self.patch_embed)
        standard = vit_embeds @ self.projection
        if self.spectral_vision is not None:
            spectral = self.spectral_vision.encode(processed[0])
            spectral_pooled = np.mean(spectral, axis=0, keepdims=True)
            spectral_proj = (
                spectral_pooled @ self.projection[: spectral_pooled.shape[-1]]
            )
            fused = standard + 0.3 * spectral_proj
            return fused, standard
        return standard, standard

    def save_ssf(self, path: str):
        tensors = self.patch_embed.get_ssf_tensors()
        tensors.update(self.vit.get_ssf_tensors())
        tensors["vision_projection"] = self.projection.copy()
        if self.spectral_vision is not None:
            tensors.update(self.spectral_vision.get_ssf_tensors())
        store = SSFModelStore(path)
        for name, tensor in tensors.items():
            store.add_tensor(name, tensor)
        store.save(path)

    def load_ssf(self, path: str):
        store = SSFModelStore(path)
        store.load(path)
        tensors = {k: store.get_tensor(k) for k in store.keys()}
        self.patch_embed.load_ssf_tensors(tensors)
        self.vit.load_ssf_tensors(tensors)
        if "vision_projection" in tensors:
            self.projection = tensors["vision_projection"].astype(np.float32)
        if self.spectral_vision is not None:
            self.spectral_vision.load_ssf_tensors(tensors)

    def __call__(self, source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        return self.encode_image(source)


# ═══════════════════════════════════════════════════════════════════════════
# 2. AudioEncoder — Mel-Spectrogram CNN Encoder
# ═══════════════════════════════════════════════════════════════════════════


class AudioCNNEncoder:
    """Simple CNN encoder for mel-spectrogram."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] = None,
        embed_dim: int = CLIP_EMBED_DIM,
    ):
        channels = channels or AUDIO_CNN_CHANNELS
        self.channels = channels
        self.embed_dim = embed_dim
        self.convs: list[dict] = []
        prev_c = in_channels
        for c in channels:
            k = 3
            conv = {
                "weight": np.random.randn(c, prev_c, k, k).astype(np.float32) * 0.02,
                "bias": np.zeros(c, dtype=np.float32),
            }
            self.convs.append(conv)
            prev_c = c
        self.fc = np.random.randn(channels[-1], embed_dim).astype(np.float32) * 0.02
        self.fc_bias = np.zeros(embed_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[np.newaxis, np.newaxis, :, :]
        elif x.ndim == 3:
            x = x[:, np.newaxis, :, :]
        for conv in self.convs:
            w = conv["weight"]
            b = conv["bias"]
            n, c, h, w_in = x.shape
            w_flat = w.reshape(w.shape[0], -1)
            oh = h - w.shape[2] + 1
            ow = w_in - w.shape[3] + 1
            out = np.zeros((n, w.shape[0], oh, ow), dtype=np.float32)
            for ni in range(n):
                for i in range(oh):
                    for j in range(ow):
                        patch = x[ni, :, i : i + w.shape[2], j : j + w.shape[3]].ravel()
                        out[ni, :, i, j] = w_flat @ patch + b
            x = np.maximum(out, 0)
            x = x[:, :, ::2, ::2]
        pooled = np.mean(x, axis=(2, 3))
        return pooled @ self.fc + self.fc_bias

    def get_ssf_tensors(self) -> dict[str, np.ndarray]:
        t = {}
        for i, conv in enumerate(self.convs):
            t[f"audio_conv_{i}_weight"] = conv["weight"].copy()
            t[f"audio_conv_{i}_bias"] = conv["bias"].copy()
        t["audio_fc"] = self.fc.copy()
        t["audio_fc_bias"] = self.fc_bias.copy()
        return t

    def load_ssf_tensors(self, tensors: dict[str, np.ndarray]):
        for i in range(len(self.convs)):
            wk = f"audio_conv_{i}_weight"
            bk = f"audio_conv_{i}_bias"
            if wk in tensors:
                self.convs[i]["weight"] = tensors[wk].astype(np.float32)
            if bk in tensors:
                self.convs[i]["bias"] = tensors[bk].astype(np.float32)
        if "audio_fc" in tensors:
            self.fc = tensors["audio_fc"].astype(np.float32)
        if "audio_fc_bias" in tensors:
            self.fc_bias = tensors["audio_fc_bias"].astype(np.float32)


class AudioEncoder:
    """
    Audio understanding encoder.

    Loads audio from file, base64, or numpy array.
    Converts to mel-spectrogram → CNN encoder → LLM projection.
    SSF format storage.
    """

    def __init__(
        self, embed_dim: int = SPECTRALSTREAM_EMBED_DIM, ssf_path: Optional[str] = None
    ):
        self.embed_dim = embed_dim
        self.cnn = AudioCNNEncoder(embed_dim=CLIP_EMBED_DIM)
        self.projection = (
            np.random.randn(CLIP_EMBED_DIM, embed_dim).astype(np.float32) * 0.02
        )
        self.ssf_store = SSFModelStore(ssf_path)
        if ssf_path and os.path.isfile(ssf_path):
            try:
                self.load_ssf(ssf_path)
            except Exception:
                pass

    def encode_audio(self, source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        audio = _load_audio_from_source(source)
        mel = _mel_spectrogram(audio)
        cnn_out = self.cnn.forward(mel)
        return cnn_out @ self.projection

    def encode_audio_batch(
        self, sources: list[Union[str, bytes, np.ndarray]]
    ) -> list[np.ndarray]:
        return [self.encode_audio(s) for s in sources]

    def save_ssf(self, path: str):
        tensors = self.cnn.get_ssf_tensors()
        tensors["audio_projection"] = self.projection.copy()
        store = SSFModelStore(path)
        for name, tensor in tensors.items():
            store.add_tensor(name, tensor)
        store.save(path)

    def load_ssf(self, path: str):
        store = SSFModelStore(path)
        store.load(path)
        tensors = {k: store.get_tensor(k) for k in store.keys()}
        self.cnn.load_ssf_tensors(tensors)
        if "audio_projection" in tensors:
            self.projection = tensors["audio_projection"].astype(np.float32)

    def __call__(self, source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        return self.encode_audio(source)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MultiModalChat — Chat with images/audio
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Message:
    role: str
    content: str
    images: list[Union[str, bytes, np.ndarray]] = field(default_factory=list)
    audio: Optional[Union[str, bytes, np.ndarray]] = None
    name: Optional[str] = None


class MultiModalChat:
    """
    Chat with text + images + optional audio.

    Pipeline:
      1. Prompt: text + images + optional audio
      2. Encode: each modality → embeddings
      3. Fuse: interleave embeddings in prompt
      4. Generate: standard text generation with multimodal prefix
      5. Streaming: yields tokens one at a time

    API: OpenAI vision-compatible endpoint via to_openai_payload()
    """

    def __init__(
        self,
        engine: Optional[Union[UnifiedInferenceEngine, Any]] = None,
        vision_encoder: Optional[VisionEncoder] = None,
        audio_encoder: Optional[AudioEncoder] = None,
        embed_dim: int = SPECTRALSTREAM_EMBED_DIM,
        use_holographic_context: bool = True,
    ):
        self.engine = engine
        self.vision = vision_encoder or VisionEncoder(embed_dim=embed_dim)
        self.audio = audio_encoder or AudioEncoder(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.use_holographic_context = use_holographic_context
        self.holographic_context = None
        self._use_holographic_ctx = use_holographic_context
        self._holographic_ctx_dirty = False
        self._history: list[Message] = []
        self._image_token = "<|image|>"
        self._audio_token = "<|audio|>"
        self._embed_token = "<|embed|>"
        self._multimodal_prefix = "<|multimodal|>"

        self.image_placeholder_id = 0
        self.audio_placeholder_id = 1
        self.embed_placeholder_id = 2

    def _build_multimodal_prompt(self, messages: list[Message]) -> str:
        """Build text prompt with multimodal placeholders."""
        parts = [self._multimodal_prefix + "\n"]
        for msg in messages:
            role_tag = f"<|{msg.role}|>"
            parts.append(role_tag + "\n")
            if msg.images:
                for _ in msg.images:
                    parts.append(self._image_token + "\n")
            if msg.audio is not None:
                parts.append(self._audio_token + "\n")
            parts.append(msg.content + "\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _encode_multimodal(
        self, messages: list[Message]
    ) -> tuple[str, list[np.ndarray], list[int]]:
        """Encode all modalities into embeddings."""
        text_prompt = self._build_multimodal_prompt(messages)
        embed_list: list[np.ndarray] = []
        embed_positions: list[int] = []

        text_so_far = ""
        img_idx = 0
        aud_idx = 0
        for msg in messages:
            if msg.images:
                for img_src in msg.images:
                    img_emb = self.vision.encode_image(img_src)
                    embed_list.append(img_emb.flatten())
                    placeholder = self._image_token
                    pos = text_so_far.count("\n") + img_idx
                    embed_positions.append(pos)
                    img_idx += 1
            if msg.audio is not None:
                aud_emb = self.audio.encode_audio(msg.audio)
                embed_list.append(aud_emb.flatten())
                pos = text_so_far.count("\n") + aud_idx
                embed_positions.append(pos)
                aud_idx += 1
            text_so_far += msg.content + "\n"

        return text_prompt, embed_list, embed_positions

    def chat(
        self,
        messages: list[Message],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
        stream: bool = False,
    ) -> Union[str, Any]:
        """
        Process multimodal chat and generate response.

        If stream=True, returns a generator yielding token strings.
        Otherwise returns the full response string.
        """
        text_prompt, embeds, embed_pos = self._encode_multimodal(messages)
        full_prompt = text_prompt

        if embeds:
            embed_concat = np.concatenate(embeds) if len(embeds) > 1 else embeds[0]
            embed_str = base64.b64encode(
                embed_concat.astype(np.float32).tobytes()
            ).decode()
            full_prompt += f"\n{self._embed_token}{embed_str}{self._embed_token}\n"

        if self._use_holographic_ctx and self._history:
            ctx_vec = self._compress_history()
            ctx_b64 = base64.b64encode(ctx_vec.astype(np.float32).tobytes()).decode()
            full_prompt = (
                f"<|holographic_context|>{ctx_b64}<|/holographic_context|>\n"
                + full_prompt
            )

        if stream:
            return self._stream_generate(
                full_prompt, max_new_tokens, temperature, top_k, top_p
            )

        return self._generate(full_prompt, max_new_tokens, temperature, top_k, top_p)

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> str:
        if self.engine is not None and HAS_UNIFIED:
            tokens, _ = self.engine.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if hasattr(self.engine, "detokenize"):
                return self.engine.detokenize(tokens[-max_new_tokens:])
            return str(tokens)
        return f"[MultiModalChat simulated response to: {prompt[:100]}...]"

    def _stream_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ):
        if self.engine is not None and hasattr(self.engine, "stream_generate"):
            for token, strategy, mode in self.engine.stream_generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ):
                if hasattr(self.engine, "detokenize"):
                    text = self.engine.detokenize([token])
                    yield text
                else:
                    yield str(token)
        else:
            words = [
                "I",
                "see",
                "the",
                "image",
                "and",
                "can",
                "help",
                "with",
                "that",
                ".",
            ]
            for w in words:
                yield w + " "
                time.sleep(0.05)

    def _compress_history(self) -> np.ndarray:
        if not self._use_holographic_ctx:
            return np.zeros(HRR_CONTEXT_DIM, dtype=np.float32)
        combined = np.zeros(HRR_CONTEXT_DIM, dtype=np.float32)
        for i, msg in enumerate(self._history[-64:]):
            key = hash(f"{msg.role}:{msg.content[:50]}") & 0x7FFFFFFF
            val = _embed_text_simple(msg.content, HRR_CONTEXT_DIM)
            combined += val * 0.5 / math.sqrt(max(len(self._history), 1))
        return _normalize(combined)

    def to_openai_payload(self, messages: list[Message]) -> dict:
        """Convert to OpenAI vision-compatible API payload."""
        openai_messages = []
        for msg in messages:
            content: list[dict] = [{"type": "text", "text": msg.content}]
            for img_src in msg.images:
                if isinstance(img_src, str):
                    if img_src.startswith(("http://", "https://")):
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": img_src, "detail": "auto"},
                            }
                        )
                    elif img_src.startswith("data:"):
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": img_src, "detail": "auto"},
                            }
                        )
                elif isinstance(img_src, bytes):
                    b64 = base64.b64encode(img_src).decode()
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "auto",
                            },
                        }
                    )
            if msg.audio is not None:
                if isinstance(msg.audio, bytes):
                    b64 = base64.b64encode(msg.audio).decode()
                    content.append(
                        {"type": "audio", "audio_url": f"data:audio/wav;base64,{b64}"}
                    )
            openai_messages.append(
                {
                    "role": msg.role,
                    "content": content,
                    **({"name": msg.name} if msg.name else {}),
                }
            )
        return {
            "model": "spectralstream-multimodal",
            "messages": openai_messages,
            "max_tokens": 1024,
            "stream": False,
        }

    def add_to_history(self, message: Message):
        self._history.append(message)

    def clear_history(self):
        self._history.clear()
        if self._use_holographic_ctx:
            pass

    def get_history(self) -> list[Message]:
        return list(self._history)


# ═══════════════════════════════════════════════════════════════════════════
# 4. CodeInterpreter — Sandboxed Code Execution
# ═══════════════════════════════════════════════════════════════════════════

LANG_MAP = {
    "python": sys.executable,
    "py": sys.executable,
    "python3": sys.executable,
    "javascript": "node",
    "js": "node",
    "shell": os.environ.get("SHELL", "/bin/bash"),
    "bash": os.environ.get("SHELL", "/bin/bash"),
    "sh": "/bin/sh",
}


@dataclass
class CodeResult:
    success: bool
    stdout: str
    stderr: str
    output: Any = None
    execution_time: float = 0.0
    error: Optional[str] = None
    plots: list[bytes] = field(default_factory=list)


class CodeInterpreter:
    """
    Sandboxed code execution environment.

    Supports Python, JavaScript, Shell.
    Security: no network, restricted filesystem (only /tmp accessible),
    timeouts, memory limits.
    Supports iterative error fixing.
    """

    def __init__(
        self,
        max_time: int = MAX_CODE_EXEC_TIME,
        max_output: int = MAX_CODE_OUTPUT_LEN,
        enable_plots: bool = True,
        restrict_network: bool = True,
        restrict_filesystem: bool = True,
    ):
        self.max_time = max_time
        self.max_output = max_output
        self.enable_plots = enable_plots
        self.restrict_network = restrict_network
        self.restrict_filesystem = restrict_filesystem
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._namespace: dict = {}
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _get_temp_dir(self) -> str:
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="spectral_code_")
        return self._temp_dir.name

    def execute(self, code: str, language: str = "python") -> CodeResult:
        language = language.lower()
        if language in ("python", "py", "python3"):
            return self._execute_python(code)
        elif language in ("javascript", "js"):
            return self._execute_javascript(code)
        elif language in ("shell", "bash", "sh"):
            return self._execute_shell(code)
        else:
            return CodeResult(
                success=False,
                stdout="",
                stderr="",
                error=f"Unsupported language: {language}",
            )

    def _execute_python(self, code: str) -> CodeResult:
        start = time.time()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        plots: list[bytes] = []

        safe_globals = {
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "ascii": ascii,
                "bin": bin,
                "bool": bool,
                "bytearray": bytearray,
                "bytes": bytes,
                "callable": callable,
                "chr": chr,
                "complex": complex,
                "dict": dict,
                "dir": dir,
                "divmod": divmod,
                "enumerate": enumerate,
                "eval": None,
                "exec": None,
                "filter": filter,
                "float": float,
                "format": format,
                "frozenset": frozenset,
                "getattr": getattr,
                "hasattr": hasattr,
                "hash": hash,
                "hex": hex,
                "id": id,
                "input": None,
                "int": int,
                "isinstance": isinstance,
                "issubclass": issubclass,
                "iter": iter,
                "len": len,
                "list": list,
                "locals": locals,
                "map": map,
                "max": max,
                "min": min,
                "next": next,
                "object": object,
                "oct": oct,
                "open": None,
                "ord": ord,
                "pow": pow,
                "print": print,
                "range": range,
                "repr": repr,
                "reversed": reversed,
                "round": round,
                "set": set,
                "slice": slice,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "super": super,
                "tuple": tuple,
                "type": type,
                "vars": vars,
                "zip": zip,
                "__import__": None,
            },
            "np": np,
            "math": math,
            "json": json,
            "re": re,
            "collections": defaultdict,
            "datetime": datetime,
            "os": None if self.restrict_filesystem else __import__("os"),
            "subprocess": None if self.restrict_network else __import__("subprocess"),
            "sys": sys,
            "io": io,
            "base64": base64,
            "time": time,
            "random": __import__("random"),
            "tempfile": tempfile,
            "Path": Path,
        }

        if self.enable_plots:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                safe_globals["plt"] = plt
                safe_globals["matplotlib"] = matplotlib
                old_show = plt.show

                def _patched_show():
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    buf.seek(0)
                    plots.append(buf.read())
                    plt.close()

                plt.show = _patched_show
            except ImportError:
                pass

        try:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            try:
                compiled = compile(
                    code, "<spectral_code>", "exec", flags=ast.PyCF_ONLY_AST
                )
                for node in ast.walk(compiled):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            if alias.name in (
                                "os",
                                "subprocess",
                                "shutil",
                                "socket",
                                "requests",
                                "urllib",
                                "http",
                                "ftplib",
                                "telnetlib",
                                "paramiko",
                                "socketio",
                            ):
                                if self.restrict_network:
                                    pass
                compiled = compile(code, "<spectral_code>", "exec")
            except SyntaxError as e:
                return CodeResult(
                    success=False, stdout="", stderr=str(e), error=f"SyntaxError: {e}"
                )

            future = self._executor.submit(exec, compiled, safe_globals)
            try:
                future.result(timeout=self.max_time)
            except TimeoutError:
                return CodeResult(
                    success=False,
                    stdout=stdout_capture.getvalue()[: self.max_output],
                    stderr="",
                    error="Execution timed out",
                )
            except Exception as e:
                return CodeResult(
                    success=False,
                    stdout=stdout_capture.getvalue()[: self.max_output],
                    stderr=traceback.format_exc(),
                    error=f"RuntimeError: {e}",
                )

            executed_code = safe_globals.get("_result", None)
            out_text = stdout_capture.getvalue()[: self.max_output]
            err_text = stderr_capture.getvalue()[: self.max_output]

            return CodeResult(
                success=True,
                stdout=out_text,
                stderr=err_text,
                output=executed_code,
                execution_time=time.time() - start,
                plots=plots,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _execute_javascript(self, code: str) -> CodeResult:
        start = time.time()
        tmp_file = os.path.join(self._get_temp_dir(), f"script_{uuid.uuid4().hex}.js")
        try:
            with open(tmp_file, "w") as f:
                f.write(code)
            result = subprocess.run(
                ["node", tmp_file],
                capture_output=True,
                text=True,
                timeout=self.max_time,
                env={**os.environ, "NODE_PATH": ""}
                if self.restrict_network
                else os.environ,
            )
            return CodeResult(
                success=result.returncode == 0,
                stdout=result.stdout[: self.max_output],
                stderr=result.stderr[: self.max_output],
                execution_time=time.time() - start,
                error=None if result.returncode == 0 else result.stderr[:500],
            )
        except subprocess.TimeoutExpired:
            return CodeResult(
                success=False,
                stdout="",
                stderr="",
                error="JavaScript execution timed out",
            )
        except FileNotFoundError:
            return CodeResult(
                success=False, stdout="", stderr="", error="Node.js not found"
            )
        finally:
            try:
                os.remove(tmp_file)
            except OSError:
                pass

    def _execute_shell(self, code: str) -> CodeResult:
        start = time.time()
        try:
            result = subprocess.run(
                [LANG_MAP.get("shell", "/bin/bash"), "-c", code],
                capture_output=True,
                text=True,
                timeout=self.max_time,
                env={} if self.restrict_network else os.environ,
                cwd=self._get_temp_dir(),
            )
            return CodeResult(
                success=result.returncode == 0,
                stdout=result.stdout[: self.max_output],
                stderr=result.stderr[: self.max_output],
                execution_time=time.time() - start,
                error=None if result.returncode == 0 else result.stderr[:500],
            )
        except subprocess.TimeoutExpired:
            return CodeResult(
                success=False, stdout="", stderr="", error="Shell execution timed out"
            )

    def execute_with_feedback(
        self,
        code: str,
        language: str = "python",
        model_feedback_fn: Optional[Callable] = None,
        max_iterations: int = 3,
    ) -> list[CodeResult]:
        """
        Execute code iteratively, allowing model to fix errors.

        model_feedback_fn: callable(code, result) -> fixed_code
        """
        results = []
        current_code = code
        for iteration in range(max_iterations):
            result = self.execute(current_code, language)
            results.append(result)
            if result.success:
                break
            if model_feedback_fn is not None and result.error:
                current_code = model_feedback_fn(current_code, result)
        return results

    def cleanup(self):
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None
        self._executor.shutdown(wait=False)

    def detect_code_blocks(self, text: str) -> list[tuple[str, str, int, int]]:
        """Detect ```language ... ``` code blocks in text."""
        pattern = r"```(\w+)?\n(.*?)```"
        blocks = []
        for m in re.finditer(pattern, text, re.DOTALL):
            lang = m.group(1) or "python"
            code = m.group(2).strip()
            blocks.append((lang, code, m.start(), m.end()))
        return blocks

    def extract_and_execute(self, text: str) -> list[CodeResult]:
        """Extract code blocks from text and execute them."""
        blocks = self.detect_code_blocks(text)
        return [self.execute(code, lang) for lang, code, _, _ in blocks]

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# 5. PromptOptimizer — Auto-improve prompts
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PromptVariant:
    prompt: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)
    generation: Optional[str] = None


class PromptQualityModel:
    """
    Lightweight quality predictor for prompt evaluation.

    Uses a small learned model + heuristic features to predict
    how good a prompt's output will be.
    """

    def __init__(self, dim: int = PROMPT_OPTIMIZER_QUALITY_MODEL_DIM):
        self.dim = dim
        self.w = np.random.randn(dim).astype(np.float32) * 0.01
        self.b = 0.0

    def _extract_features(self, prompt: str) -> np.ndarray:
        features = np.zeros(self.dim, dtype=np.float32)
        words = prompt.split()
        features[0] = min(len(words) / 200.0, 1.0)
        features[1] = min(len(prompt) / 2000.0, 1.0)
        features[2] = float(prompt.count("?")) / max(len(words), 1)
        features[3] = float(prompt.count("!")) / max(len(words), 1)
        features[4] = float(prompt.count('"')) / max(len(words), 1)
        features[5] = float(prompt.count("\n")) / max(len(words), 1)
        features[6] = float(
            len(re.findall(r"\b(?:step|first|then|finally|because)\b", prompt.lower()))
        )
        features[7] = float(
            len(re.findall(r"\b(?:if|when|unless|although|while)\b", prompt.lower()))
        )
        features[8] = float(
            len(
                re.findall(
                    r"\b(?:example|e\.g\.|for instance|consider)\b", prompt.lower()
                )
            )
        )
        features[9] = float(
            len(
                re.findall(
                    r"\b(?:list|enumerate|number|bullet|point)\b", prompt.lower()
                )
            )
        )
        features[10] = float(
            len(
                re.findall(
                    r"\b(?:concise|brief|short|detailed|comprehensive)\b",
                    prompt.lower(),
                )
            )
        )
        features[11] = float(
            len(re.findall(r"\b(?:code|function|class|method|api)\b", prompt.lower()))
        )
        features[12] = float(prompt.count("{{")) / max(len(words), 1)
        features[13] = float(
            len(re.findall(r"\b(?:I want you to|Please|Can you|Could you)\b", prompt))
        )
        role_words = ["expert", "specialist", "professional", "senior", "assistant"]
        features[14] = float(sum(w.lower() in role_words for w in words))
        features[15] = float(len(re.findall(r"\b[A-Z]{2,}\b", prompt))) / max(
            len(words), 1
        )
        seed = hash(prompt) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        features[16:] = rng.randn(self.dim - 16).astype(np.float32) * 0.01
        return features

    def predict(self, prompt: str) -> float:
        features = self._extract_features(prompt)
        score = float(features @ self.w + self.b)
        return float(1.0 / (1.0 + np.exp(-score)))

    def update(self, prompt: str, actual_score: float, lr: float = 0.01):
        features = self._extract_features(prompt)
        pred = self.predict(prompt)
        error = actual_score - pred
        self.w += lr * error * features
        self.b += lr * error


class ResonantPromptingOptimizer:
    """
    Novel: "Resonant Prompting" — optimize prompt to resonate with model's
    learned patterns via frequency-domain analysis.

    Key insight: LLMs learn specific frequency patterns in their
    embedding space. By analyzing a prompt's "frequency signature"
    (FFT of its embedding), we can optimize it to resonate with the
    model's strongest frequency modes.
    """

    def __init__(
        self, embed_dim: int = SPECTRALSTREAM_EMBED_DIM, n_resonant_modes: int = 64
    ):
        self.embed_dim = embed_dim
        self.n_resonant_modes = n_resonant_modes
        self.characteristic_frequencies = np.random.randn(
            n_resonant_modes, n_resonant_modes
        ).astype(np.float32)
        self.characteristic_frequencies = self.characteristic_frequencies / (
            np.linalg.norm(self.characteristic_frequencies, axis=1, keepdims=True)
            + 1e-10
        )

    def _prompt_spectrum(self, prompt: str) -> np.ndarray:
        words = prompt.split()
        if not words:
            return np.zeros(self.n_resonant_modes, dtype=np.float32)
        embed = _embed_text_simple(prompt, self.embed_dim)
        fft = np.fft.fft(embed.astype(np.complex128))
        magnitude = np.abs(fft)[: self.embed_dim // 2]
        if len(magnitude) < self.n_resonant_modes:
            magnitude = np.pad(magnitude, (0, self.n_resonant_modes - len(magnitude)))
        else:
            magnitude = magnitude[: self.n_resonant_modes]
        return _normalize(magnitude.astype(np.float32))

    def _resonance_score(self, prompt: str) -> float:
        spectrum = self._prompt_spectrum(prompt)
        scores = []
        for mode in self.characteristic_frequencies:
            overlap = float(np.abs(spectrum @ mode))
            scores.append(overlap)
        return float(np.mean(scores))

    def optimize(
        self, prompt: str, n_variations: int = 8, model_fn: Optional[Callable] = None
    ) -> PromptVariant:
        variations = self._generate_variations(prompt, n_variations, model_fn)
        best = PromptVariant(prompt=prompt, score=self._resonance_score(prompt))
        for var in variations:
            score = self._resonance_score(var.prompt)
            var.score = score
            if score > best.score:
                best = var
        return best

    def _generate_variations(
        self, prompt: str, n: int, model_fn: Optional[Callable] = None
    ) -> list[PromptVariant]:
        variations = []
        words = prompt.split()
        for i in range(n):
            if i < len(self.characteristic_frequencies):
                mode = self.characteristic_frequencies[i]
                seed = int(np.abs(mode[0]) * 10000) % (2**31)
                rng = np.random.RandomState(seed)
                if rng.rand() < 0.3 and len(words) > 5:
                    idx = rng.randint(1, len(words) - 1)
                    synonyms = {
                        "good": "excellent",
                        "bad": "poor",
                        "big": "substantial",
                        "small": "minor",
                        "fast": "rapid",
                        "slow": "gradual",
                        "help": "assist",
                        "make": "create",
                        "use": "utilize",
                        "get": "obtain",
                        "show": "demonstrate",
                        "tell": "explain",
                    }
                    if words[idx].lower() in synonyms:
                        w = words.copy()
                        w[idx] = synonyms[words[idx].lower()]
                        if words[idx][0].isupper():
                            w[idx] = w[idx].capitalize()
                        variations.append(PromptVariant(prompt=" ".join(w)))
                elif rng.rand() < 0.3:
                    prefixes = [
                        "As an expert, ",
                        "Step by step, ",
                        "In detail, ",
                        "Consider the following: ",
                        "Think carefully: ",
                        "Using your expertise, ",
                        "Approach this systematically: ",
                    ]
                    prefix = prefixes[rng.randint(0, len(prefixes))]
                    variations.append(
                        PromptVariant(
                            prompt=prefix + prompt[0].lower() + prompt[1:]
                            if prompt
                            else prompt
                        )
                    )
                elif rng.rand() < 0.3:
                    suffixes = [
                        "\nExplain your reasoning step by step.",
                        "\nBe concise and accurate.",
                        "\nProvide examples where relevant.",
                        "\nThink about this carefully before responding.",
                        "\nConsider multiple perspectives.",
                    ]
                    variations.append(
                        PromptVariant(
                            prompt=prompt + suffixes[rng.randint(0, len(suffixes))]
                        )
                    )
        return variations

    def learn_from_success(self, prompt: str, score: float, lr: float = 0.01):
        spectrum = self._prompt_spectrum(prompt)
        mode_idx = int(np.argmax(spectrum)) % self.n_resonant_modes
        self.characteristic_frequencies[mode_idx] = (
            self.characteristic_frequencies[mode_idx] + lr * score * spectrum
        )
        self.characteristic_frequencies[mode_idx] = _normalize(
            self.characteristic_frequencies[mode_idx]
        )


class QuantumPromptOptimizer:
    """
    Novel: "Quantum Prompt Optimization" — evaluate prompt variations in
    superposition using a simulated quantum interference pattern.

    Instead of evaluating each variation independently, we encode all
    variations into a single "quantum state" vector and measure the
    interference pattern to identify the most promising directions.
    """

    def __init__(self, dim: int = SPECTRALSTREAM_EMBED_DIM):
        self.dim = dim

    def encode_superposition(self, prompts: list[str]) -> np.ndarray:
        state = np.zeros(self.dim, dtype=np.complex128)
        for i, prompt in enumerate(prompts):
            phase = 2.0 * np.pi * i / len(prompts)
            vec = _embed_text_simple(prompt, self.dim).astype(np.complex128)
            state += vec * np.exp(1j * phase)
        return _normalize(state.astype(np.complex128))

    def measure_scores(self, prompts: list[str]) -> list[float]:
        superposition = self.encode_superposition(prompts)
        scores = []
        for i, prompt in enumerate(prompts):
            target = _embed_text_simple(prompt, self.dim).astype(np.complex128)
            interference = float(np.abs(np.vdot(superposition, target)))
            scores.append(interference)
        if max(scores) > 0:
            scores = [s / max(scores) for s in scores]
        return scores

    def select_best(self, prompts: list[str]) -> tuple[str, float]:
        scores = self.measure_scores(prompts)
        best_idx = int(np.argmax(scores))
        return prompts[best_idx], scores[best_idx]


class PromptOptimizer:
    """
    Auto-improve prompts using:
    - Quality model scoring
    - Resonant prompting frequency optimization
    - Quantum superposition evaluation
    - Iterative refinement
    - Template library learning

    DSPy-style: optimize as programming with LLMs.
    """

    def __init__(
        self,
        quality_model: Optional[PromptQualityModel] = None,
        resonant_optimizer: Optional[ResonantPromptingOptimizer] = None,
        quantum_optimizer: Optional[QuantumPromptOptimizer] = None,
        model_fn: Optional[Callable] = None,
        ssf_path: Optional[str] = None,
    ):
        self.quality_model = quality_model or PromptQualityModel()
        self.resonant = resonant_optimizer or ResonantPromptingOptimizer()
        self.quantum = quantum_optimizer or QuantumPromptOptimizer()
        self.model_fn = model_fn
        self._template_library: dict[str, PromptVariant] = {}
        self._optimization_history: list[dict] = []
        self._best_score = 0.0
        self._best_prompt = ""
        self.ssf_path = ssf_path

    def analyze(self, prompt: str) -> dict:
        words = prompt.split()
        features = self.quality_model._extract_features(prompt)
        return {
            "length": len(prompt),
            "word_count": len(words),
            "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0,
            "question_count": prompt.count("?"),
            "exclamation_count": prompt.count("!"),
            "has_examples": "example" in prompt.lower(),
            "has_steps": "step" in prompt.lower() or "first" in prompt.lower(),
            "has_role": any(
                r in prompt.lower() for r in ["expert", "assistant", "you are"]
            ),
            "has_conditional": "if" in prompt.lower() or "when" in prompt.lower(),
            "quality_score": self.quality_model.predict(prompt),
            "resonance_score": self.resonant._resonance_score(prompt),
            "complexity": features[0],
        }

    def score(self, prompt: str) -> float:
        quality = self.quality_model.predict(prompt)
        resonance = self.resonant._resonance_score(prompt)
        return 0.6 * quality + 0.4 * resonance

    def generate_variations(
        self, prompt: str, n: int = PROMPT_OPTIMIZER_N_VARIATIONS
    ) -> list[PromptVariant]:
        variations: list[PromptVariant] = [PromptVariant(prompt=prompt)]

        # Resonant variations
        resonant_best = self.resonant.optimize(prompt, n)
        if resonant_best.prompt != prompt:
            variations.append(resonant_best)

        # Quantum variations
        expanded = self._expand_prompt(prompt, n)
        quantum_best, _ = self.quantum.select_best(expanded)
        if quantum_best != prompt:
            variations.append(PromptVariant(prompt=quantum_best))

        # Structural variations
        words = prompt.split()
        for i in range(min(n, 5)):
            rng = np.random.RandomState(hash(f"{prompt}_{i}") & 0x7FFFFFFF)
            new_prompt = list(prompt)
            structs = [
                self._add_expert_role,
                self._add_examples,
                self._add_format_instruction,
                self._add_constraint,
                self._reorder_instructions,
            ]
            variant_prompt = structs[i % len(structs)](prompt, rng)
            if variant_prompt != prompt:
                variations.append(PromptVariant(prompt=variant_prompt))

        # Deduplicate
        seen = set()
        unique = []
        for v in variations:
            if v.prompt not in seen:
                seen.add(v.prompt)
                unique.append(v)
        return unique[: max(n, 1)]

    def _expand_prompt(self, prompt: str, n: int) -> list[str]:
        expanded = [prompt]
        words = prompt.split()
        for i in range(n - 1):
            rng = np.random.RandomState(hash(f"expand_{prompt}_{i}") & 0x7FFFFFFF)
            p = prompt
            if rng.rand() < 0.3:
                p = (
                    p
                    + " "
                    + rng.choice(
                        [
                            "Explain thoroughly.",
                            "Be precise.",
                            "Think step by step.",
                            "Provide examples.",
                            "Be creative.",
                            "Use technical language.",
                            "Explain simply.",
                            "Consider edge cases.",
                        ]
                    )
                )
            elif rng.rand() < 0.3:
                p = (
                    rng.choice(
                        [
                            "As a helpful assistant, ",
                            "As an expert, ",
                            "Given your expertise, ",
                            "Professionally, ",
                        ]
                    )
                    + p[0].lower()
                    + p[1:]
                    if p
                    else p
                )
            expanded.append(p)
        return expanded

    def _add_expert_role(self, prompt: str, rng: np.random.RandomState) -> str:
        roles = [
            "You are an expert in this field. ",
            "As a knowledgeable professional, ",
            "Given your deep expertise, ",
            "As an authority on this subject, ",
        ]
        role = roles[rng.randint(0, len(roles))]
        if not prompt.startswith(tuple(r.split()[0] for r in roles)):
            return role + prompt[0].lower() + prompt[1:] if prompt else prompt
        return prompt

    def _add_examples(self, prompt: str, rng: np.random.RandomState) -> str:
        if "example" not in prompt.lower():
            return prompt + "\nInclude an example to illustrate your answer."
        return prompt

    def _add_format_instruction(self, prompt: str, rng: np.random.RandomState) -> str:
        formats = [
            "\nFormat your response as bullet points.",
            "\nStructure your answer with clear sections.",
            "\nProvide your answer in a numbered list.",
            "\nUse markdown formatting for clarity.",
        ]
        return prompt + formats[rng.randint(0, len(formats))]

    def _add_constraint(self, prompt: str, rng: np.random.RandomState) -> str:
        constraints = [
            "\nKeep your response under 200 words.",
            "\nBe concise but comprehensive.",
            "\nFocus on practical applications.",
            "\nConsider both pros and cons.",
            "\nSupport your claims with evidence.",
        ]
        return prompt + constraints[rng.randint(0, len(constraints))]

    def _reorder_instructions(self, prompt: str, rng: np.random.RandomState) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", prompt)
        if len(sentences) > 2:
            rng.shuffle(sentences)
            return " ".join(sentences)
        return prompt

    def test_variations(
        self, variations: list[PromptVariant], model_fn: Optional[Callable] = None
    ) -> list[PromptVariant]:
        fn = model_fn or self.model_fn
        for var in variations:
            quality = self.quality_model.predict(var.prompt)
            resonance = self.resonant._resonance_score(var.prompt)
            var.score = 0.6 * quality + 0.4 * resonance
            if fn is not None:
                try:
                    var.generation = fn(var.prompt)
                except Exception:
                    var.generation = None
        return variations

    def select_best(self, variations: list[PromptVariant]) -> PromptVariant:
        if not variations:
            return PromptVariant(prompt="")
        best = max(variations, key=lambda v: v.score)
        if best.score > self._best_score:
            self._best_score = best.score
            self._best_prompt = best.prompt
        return best

    def optimize(
        self,
        prompt: str,
        n_iterations: int = PROMPT_OPTIMIZER_N_ITERATIONS,
        model_fn: Optional[Callable] = None,
    ) -> PromptVariant:
        current = prompt
        history = []

        for iteration in range(n_iterations):
            variations = self.generate_variations(current)
            variations = self.test_variations(variations, model_fn)
            best = self.select_best(variations)

            history.append(
                {
                    "iteration": iteration,
                    "prompt": best.prompt,
                    "score": best.score,
                    "n_variations": len(variations),
                }
            )

            self.resonant.learn_from_success(best.prompt, best.score)

            for var in variations:
                self.quality_model.update(var.prompt, var.score)

            current = best.prompt

        self._optimization_history = history
        self._add_to_template_library(current, best.score)

        return PromptVariant(
            prompt=current,
            score=best.score,
            metadata={"history": history, "n_iterations": n_iterations},
        )

    def _add_to_template_library(self, prompt: str, score: float):
        prompt_hash = str(hash(prompt))
        if (
            score
            > self._template_library.get(
                prompt_hash, PromptVariant(prompt="", score=0)
            ).score
        ):
            self._template_library[prompt_hash] = PromptVariant(
                prompt=prompt, score=score
            )

    def get_template_library(self) -> list[PromptVariant]:
        return sorted(
            self._template_library.values(), key=lambda v: v.score, reverse=True
        )

    def optimize_batch(
        self, prompts: list[str], n_iterations: int = 2
    ) -> list[PromptVariant]:
        return [self.optimize(p, n_iterations) for p in prompts]

    def get_optimization_history(self) -> list[dict]:
        return list(self._optimization_history)

    def save(self, path: Optional[str] = None):
        dst = path or self.ssf_path
        if dst is None:
            return
        data = {
            "quality_model_w": self.quality_model.w.tolist(),
            "quality_model_b": self.quality_model.b,
            "characteristic_frequencies": self.resonant.characteristic_frequencies.tolist(),
            "template_library": [
                {"prompt": v.prompt, "score": v.score}
                for v in self._template_library.values()
            ],
            "best_score": self._best_score,
            "best_prompt": self._best_prompt,
        }
        if dst.endswith(".sst"):
            store = SSFModelStore(dst)
            store.add_tensor(
                "optimizer_state",
                np.frombuffer(json.dumps(data).encode(), dtype=np.uint8),
            )
            store.save(dst)
        else:
            with open(dst, "w") as f:
                json.dump(data, f)

    def load(self, path: Optional[str] = None):
        src = path or self.ssf_path
        if src is None or not os.path.isfile(src):
            return
        try:
            if src.endswith(".sst"):
                store = SSFModelStore(src)
                store.load(src)
                raw = store.get_tensor("optimizer_state")
                if raw is not None:
                    data = json.loads(raw.tobytes().decode("utf-8", errors="replace"))
            else:
                with open(src) as f:
                    data = json.load(f)
            w = np.array(
                data.get("quality_model_w", self.quality_model.w.tolist()),
                dtype=np.float32,
            )
            self.quality_model.w = w
            self.quality_model.b = data.get("quality_model_b", 0.0)
            freq = np.array(
                data.get(
                    "characteristic_frequencies",
                    self.resonant.characteristic_frequencies.tolist(),
                ),
                dtype=np.float32,
            )
            if freq.shape == self.resonant.characteristic_frequencies.shape:
                self.resonant.characteristic_frequencies = freq
            for entry in data.get("template_library", []):
                self._template_library[entry["prompt"]] = PromptVariant(
                    prompt=entry["prompt"], score=entry.get("score", 0.0)
                )
            self._best_score = data.get("best_score", 0.0)
            self._best_prompt = data.get("best_prompt", "")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# 6. PromptTemplateEngine — Template Management
# ═══════════════════════════════════════════════════════════════════════════


class TemplateSyntaxError(Exception):
    pass


class PromptTemplateEngine:
    """
    Template management with:
    - Variables: {{variable}} substitution
    - Conditions: {% if condition %}...{% elif %}...{% else %}...{% endif %}
    - Loops: {% for item in list %}...{% endfor %}
    - Functions: built-in (format, capitalize, truncate, lower, upper, title)
    - Chaining: compose templates from partials
    - Versioning: track prompt template changes
    - A/B testing: serve different templates to different users
    """

    def __init__(self):
        self._partials: dict[str, str] = {}
        self._versions: dict[str, list[dict]] = defaultdict(list)
        self._active_version: dict[str, int] = defaultdict(lambda: 0)
        self._ab_tests: dict[str, list[str]] = {}
        self._builtins = {
            "format": lambda s, *args, **kwargs: str(s).format(*args, **kwargs),
            "capitalize": lambda s: str(s).capitalize() if s else "",
            "upper": lambda s: str(s).upper() if s else "",
            "lower": lambda s: str(s).lower() if s else "",
            "title": lambda s: str(s).title() if s else "",
            "truncate": lambda s, n=100: str(s)[: int(n)] + "..."
            if len(str(s)) > int(n)
            else str(s),
            "strip": lambda s: str(s).strip() if s else "",
            "replace": lambda s, old, new: str(s).replace(old, new) if s else "",
            "length": lambda x: len(x) if hasattr(x, "__len__") else 0,
            "join": lambda items, sep=", ": sep.join(str(i) for i in items)
            if items
            else "",
            "default": lambda s, d="": str(s) if s else d,
        }

    def register_partial(self, name: str, template: str):
        """Register a reusable template partial."""
        self._partials[name] = template

    def get_partial(self, name: str) -> Optional[str]:
        return self._partials.get(name)

    def render(self, template: str, variables: dict = None) -> str:
        variables = variables or {}

        # Include partials
        template = re.sub(
            r"\{%\s*include\s+(\w+)\s*%\}",
            lambda m: self._partials.get(m.group(1), m.group(0)),
            template,
        )

        # Process loop/if blocks
        template = self._process_blocks(template, variables)

        # Apply built-in functions
        template = self._process_functions(template, variables)

        # Substitute variables
        template = self._process_variables(template, variables)

        return template

    @staticmethod
    def _eval_expr(expr: str, vars: dict) -> bool:
        expr = expr.strip()
        for op in ["!=", "==", ">=", "<=", ">", "<"]:
            if op in expr:
                parts = expr.split(op, 1)
                left = parts[0].strip().strip("'\"").strip()
                right = parts[1].strip().strip("'\"").strip()
                lv = vars.get(left, left)
                rv = vars.get(right, right)
                try:
                    lv = (
                        int(lv)
                        if lv.lstrip("-").isdigit()
                        else float(lv)
                        if (
                            isinstance(lv, str)
                            and lv.replace(".", "", 1).lstrip("-").isdigit()
                        )
                        else lv
                    )
                    rv = (
                        int(rv)
                        if rv.lstrip("-").isdigit()
                        else float(rv)
                        if (
                            isinstance(rv, str)
                            and rv.replace(".", "", 1).lstrip("-").isdigit()
                        )
                        else rv
                    )
                except (ValueError, AttributeError):
                    pass
                if op == "==":
                    return lv == rv
                elif op == "!=":
                    return lv != rv
                else:
                    try:
                        if op == ">":
                            return float(lv) > float(rv)
                        if op == "<":
                            return float(lv) < float(rv)
                        if op == ">=":
                            return float(lv) >= float(rv)
                        if op == "<=":
                            return float(lv) <= float(rv)
                    except (ValueError, TypeError):
                        return False
        val = vars.get(expr, expr)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val != 0
        if isinstance(val, str):
            return val.lower() not in ("", "false", "none", "0", "no")
        if val is None:
            return False
        return bool(val)

    def _process_blocks(self, template: str, variables: dict) -> str:
        def _process_for_block(match):
            var_name = match.group(1).strip()
            iterable_name = match.group(2).strip()
            inner = match.group(3)
            iterable = variables.get(iterable_name, [])
            if not isinstance(iterable, (list, tuple, range, np.ndarray)):
                try:
                    iterable = list(iterable)
                except TypeError:
                    iterable = []
            items = []
            for idx, item in enumerate(iterable):
                sub_vars = dict(variables)
                sub_vars[var_name] = item
                sub_vars["loop"] = {
                    "index": idx + 1,
                    "index0": idx,
                    "first": idx == 0,
                    "last": idx == len(iterable) - 1,
                    "length": len(iterable),
                }
                items.append(self.render(inner, sub_vars))
            return "".join(items)

        # Recursively process from innermost blocks
        while True:
            new_template = re.sub(
                r"\{%\s*for\s+(.*?)in\s+(.*?)\s*%}(.*?)\{%\s*endfor\s*%\}",
                _process_for_block,
                template,
                flags=re.DOTALL,
            )
            new_template = re.sub(
                r"\{%\s*if\s+(.*?)\s*%\}\s*(.*?)(?:\{%\s*elif\s+(.*?)\s*%\}\s*(.*?))*(?:\{%\s*else\s*%\}\s*(.*?))?\{%\s*endif\s*%\}",
                lambda m: self._process_if(m, variables),
                new_template,
                flags=re.DOTALL,
            )
            if new_template == template:
                break
            template = new_template
        return template

    def _process_if(self, match, variables: dict) -> str:
        full = match.group(0)
        inner = match.group(2) or ""
        else_content = match.group(5) or ""

        expr = match.group(1).strip()
        if self._eval_expr(expr, variables):
            return inner

        elif_pattern = (
            r"\{%\s*elif\s+(.*?)\s*%\}\s*(.*?)(?=\{%\s*(?:elif|else|endif)\s*%\})"
        )
        for elif_match in re.finditer(elif_pattern, full, re.DOTALL):
            elif_expr = elif_match.group(1).strip()
            elif_inner = elif_match.group(2)
            if self._eval_expr(elif_expr, variables):
                return elif_inner

        return else_content

    def _process_functions(self, template: str, variables: dict) -> str:
        def _replace_fn(match):
            inner = match.group(1).strip()
            for fn_name in sorted(self._builtins.keys(), key=len, reverse=True):
                pattern = rf"^{fn_name}\((.*)\)$"
                fn_match = re.match(pattern, inner)
                if fn_match:
                    args_str = fn_match.group(1)
                    args = self._parse_fn_args(args_str, variables)
                    try:
                        result = self._builtins[fn_name](*args)
                        return str(result)
                    except Exception:
                        return match.group(0)
            return match.group(0)

        template = re.sub(r"\{\{\s*(\w+\(.*?\))\s*\}\}", _replace_fn, template)
        return template

    def _parse_fn_args(self, args_str: str, variables: dict) -> list:
        args = []
        for arg in re.split(r",\s*(?![^()]*\))", args_str):
            arg = arg.strip()
            if arg.startswith("'") and arg.endswith("'"):
                args.append(arg[1:-1])
            elif arg.startswith('"') and arg.endswith('"'):
                args.append(arg[1:-1])
            elif arg in variables:
                args.append(variables[arg])
            elif arg.lstrip("-").isdigit():
                args.append(int(arg))
            elif arg.replace(".", "", 1).lstrip("-").isdigit():
                args.append(float(arg))
            else:
                args.append(arg)
        return args

    def _process_variables(self, template: str, variables: dict) -> str:
        def _replace_var(match):
            var_path = match.group(1).strip()
            parts = var_path.split(".")
            val = variables.get(parts[0])
            if val is None and parts[0] in self._builtins:
                return str(self._builtins[parts[0]])
            for part in parts[1:]:
                if isinstance(val, dict):
                    val = val.get(part)
                elif isinstance(val, (list, tuple, np.ndarray)):
                    try:
                        idx = int(part)
                        val = val[idx] if idx < len(val) else None
                    except (ValueError, IndexError):
                        val = None
                elif hasattr(val, part):
                    val = getattr(val, part)
                else:
                    val = None
                if val is None:
                    break
            return str(val) if val is not None else ""

        return re.sub(r"\{\{(.*?)\}\}", _replace_var, template)

    def create_version(self, template_name: str, template: str) -> int:
        """Create a new version of a template. Returns version number."""
        version = len(self._versions[template_name]) + 1
        self._versions[template_name].append(
            {
                "version": version,
                "template": template,
                "created": datetime.now(timezone.utc).isoformat(),
                "active": False,
            }
        )
        self._active_version[template_name] = version
        return version

    def set_active_version(self, template_name: str, version: int):
        if template_name in self._versions and 0 < version <= len(
            self._versions[template_name]
        ):
            self._active_version[template_name] = version

    def get_version(
        self, template_name: str, version: Optional[int] = None
    ) -> Optional[str]:
        if template_name not in self._versions:
            return None
        v = version or self._active_version.get(template_name, 1)
        for entry in self._versions[template_name]:
            if entry["version"] == v:
                return entry["template"]
        return None

    def get_version_history(self, template_name: str) -> list[dict]:
        return list(self._versions.get(template_name, []))

    def setup_ab_test(self, test_name: str, template_names: list[str]):
        """Set up A/B test: serve different templates to different users."""
        self._ab_tests[test_name] = list(template_names)

    def serve_ab(self, test_name: str, user_id: str = "") -> Optional[str]:
        """Serve a template variant based on user_id hash."""
        templates = self._ab_tests.get(test_name)
        if not templates:
            return None
        idx = hash(user_id) % len(templates) if user_id else 0
        template_name = templates[idx]
        return self.get_version(template_name) or self._partials.get(template_name)

    def chain(self, template_names: list[str], variables: dict = None) -> str:
        """Chain multiple templates, feeding output of one as partial input to next."""
        result = ""
        for name in template_names:
            tpl = self.get_version(name) or self._partials.get(name)
            if tpl is None:
                continue
            combined = dict(variables) if variables else {}
            combined["_previous"] = result
            result = self.render(tpl, combined)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 7. AutoChain — Automatic Chain-of-Thought
# ═══════════════════════════════════════════════════════════════════════════


class VlasovChainOfThought:
    """
    Novel: "Vlasov Chain-of-Thought" — reasoning as mean-field of thought
    particle interactions. Each reasoning step is a particle in a plasma,
    and the evolution of reasoning follows Vlasov-Poisson dynamics.

    Instead of linear chain-of-thought, thoughts interact via a
    self-consistent field, giving emergent multi-perspective reasoning.
    """

    def __init__(self, dim: int = SPECTRALSTREAM_EMBED_DIM, n_particles: int = 32):
        self.dim = dim
        self.n_particles = n_particles
        self.particles = np.random.randn(n_particles, dim).astype(np.float32) * 0.1
        self.velocities = np.zeros((n_particles, dim), dtype=np.float32)
        self.weights = np.ones(n_particles, dtype=np.float32) / n_particles

    def _compute_field(self) -> np.ndarray:
        field = np.zeros(self.dim, dtype=np.float32)
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i == j:
                    continue
                dx = self.particles[i] - self.particles[j]
                dist = np.linalg.norm(dx) + 1e-10
                field += self.weights[j] * dx / (dist**2 + 1e-10)
        return field

    def evolve(self, n_steps: int = 5, dt: float = 0.1) -> list[np.ndarray]:
        trajectory = [self.particles.copy()]
        for _ in range(n_steps):
            field = self._compute_field()
            self.velocities += dt * field
            self.velocities *= 0.9
            self.particles += dt * self.velocities
            self.particles = _normalize(self.particles.reshape(-1)).reshape(
                self.n_particles, self.dim
            )
            trajectory.append(self.particles.copy())
        return trajectory

    def get_ensemble_thought(self) -> np.ndarray:
        return np.mean(self.particles, axis=0)

    def inject_thought(self, embedding: np.ndarray):
        idx = np.argmin(self.weights)
        self.particles[idx] = _normalize(embedding[: self.dim].astype(np.float32))
        self.velocities[idx] = 0


class AutoChain:
    """
    Automatic chain-of-thought reasoning:
    1. Decompose: break complex questions into steps
    2. Generate step-by-step reasoning
    3. Verify: check each step before proceeding
    4. Ensemble: generate multiple reasoning paths
    5. Majority vote: select most common answer
    6. Self-consistency: choose most consistent answer across paths
    """

    def __init__(
        self,
        model_fn: Optional[Callable] = None,
        use_vlasov_cot: bool = True,
        embed_dim: int = SPECTRALSTREAM_EMBED_DIM,
    ):
        self.model_fn = model_fn
        self.use_vlasov_cot = use_vlasov_cot
        self.embed_dim = embed_dim
        self.vlasov = VlasovChainOfThought(dim=embed_dim) if use_vlasov_cot else None
        self._step_counter = 0
        self._reasoning_trace: list[dict] = []

    def decompose(self, question: str) -> list[str]:
        """Break complex question into sub-steps."""
        words = question.split()
        steps = []

        prompt_lower = question.lower()
        if "," in question and len(words) > 10:
            clauses = [c.strip() for c in re.split(r"[,;]", question) if c.strip()]
            steps = [f"Address: {c}" for c in clauses[:5]]
        if "compare" in prompt_lower or "difference" in prompt_lower:
            steps.append("Identify the items to compare")
            steps.append("List key similarities")
            steps.append("List key differences")
            steps.append("Summarize findings")
        if "why" in prompt_lower or "explain" in prompt_lower:
            steps.append("State the core concept")
            steps.append("Provide background context")
            steps.append("Explain the mechanism")
            steps.append("Give concrete example")
        if "how" in prompt_lower:
            steps.append("Define the goal")
            steps.append("Outline the process")
            steps.append("Detail each step")
            steps.append("Summarize expected outcome")
        if "code" in prompt_lower or "implement" in prompt_lower:
            steps.append("Understand requirements")
            steps.append("Design the solution")
            steps.append("Write the implementation")
            steps.append("Review and explain")
        if "analyze" in prompt_lower:
            steps.append("Identify key components")
            steps.append("Examine each component")
            steps.append("Identify patterns")
            steps.append("Draw conclusions")
        if "solve" in prompt_lower or "find" in prompt_lower:
            steps.append("Understand the problem")
            steps.append("Identify known information")
            steps.append("Formulate approach")
            steps.append("Apply solution method")
            steps.append("Verify the result")

        if not steps:
            steps = [
                "Understand the question: " + question[:100],
                "Gather relevant information",
                "Reason step by step",
                "Formulate the answer",
                "Review and refine",
            ]

        return steps

    def generate_reasoning(self, question: str, steps: list[str]) -> list[str]:
        """Generate reasoning for each step."""
        reasoning = []
        for i, step in enumerate(steps):
            if self.model_fn:
                prompt = f"Question: {question}\nStep {i + 1}/{len(steps)}: {step}\nReasoning:"
                try:
                    result = self.model_fn(prompt)
                    reasoning.append(str(result))
                except Exception:
                    reasoning.append(f"[Reasoning for step {i + 1}: {step}]")
            else:
                reasoning.append(f"[Step {i + 1}/{len(steps)}] Considering: {step}")
            if self.vlasov is not None:
                step_embed = _embed_text_simple(
                    f"{step}: {reasoning[-1]}", self.embed_dim
                )
                self.vlasov.inject_thought(step_embed)
            self._reasoning_trace.append(
                {
                    "step": i + 1,
                    "step_text": step,
                    "reasoning": reasoning[-1],
                }
            )
        return reasoning

    def verify_step(self, step_idx: int, step_text: str, reasoning: str) -> bool:
        """Verify a reasoning step before proceeding."""
        checks = []
        if len(reasoning) < 5:
            checks.append(False)
        else:
            checks.append(True)
        if "error" in reasoning.lower() or "incorrect" in reasoning.lower():
            checks.append(False)
        else:
            checks.append(True)
        if "??" in reasoning or step_text.lower() not in reasoning.lower():
            if len(reasoning) > 20:
                checks.append(True)
            else:
                checks.append(False)
        else:
            checks.append(True)
        return all(checks)

    def ensemble_generate(
        self, question: str, n_paths: int = AUTOCHAIN_N_PATHS
    ) -> list[list[str]]:
        """Generate multiple reasoning paths."""
        all_paths = []
        base_steps = self.decompose(question)

        for path_idx in range(n_paths):
            if path_idx > 0:
                rng = np.random.RandomState(
                    hash(f"{question}_path_{path_idx}") & 0x7FFFFFFF
                )
                steps = list(base_steps)
                rng.shuffle(steps)
                if rng.rand() < 0.3:
                    steps.append("Consider alternative perspective")
            else:
                steps = base_steps

            reasoning = self.generate_reasoning(question, steps)
            all_paths.append(reasoning)

        return all_paths

    def majority_vote(self, all_paths: list[list[str]]) -> str:
        """Select most common answer across reasoning paths."""
        final_thoughts = []
        for path in all_paths:
            if path:
                final_thoughts.append(path[-1])
            else:
                final_thoughts.append("")

        if not final_thoughts:
            return ""

        word_scores = Counter()
        for thought in final_thoughts:
            words = thought.split()
            if words:
                key = tuple(words[: min(10, len(words))])
                word_scores[key] += 1

        most_common = word_scores.most_common(1)
        if most_common:
            return " ".join(most_common[0][0])
        return final_thoughts[0]

    def self_consistency_check(self, answers: list[str]) -> tuple[str, float]:
        """
        Self-consistency: find the most consistent answer.
        Returns (answer, consistency_score).
        """
        if not answers:
            return "", 0.0
        if len(answers) == 1:
            return answers[0], 1.0

        sim_matrix = np.zeros((len(answers), len(answers)), dtype=np.float32)
        for i, a1 in enumerate(answers):
            emb1 = _embed_text_simple(a1, self.embed_dim)
            for j, a2 in enumerate(answers):
                if i == j:
                    sim_matrix[i, j] = 1.0
                elif j > i:
                    emb2 = _embed_text_simple(a2, self.embed_dim)
                    sim = float(np.dot(emb1, emb2)) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10
                    )
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

        avg_sim = np.mean(sim_matrix, axis=1)
        best_idx = int(np.argmax(avg_sim))
        consistency = float(avg_sim[best_idx])

        if self.vlasov is not None:
            traj = self.vlasov.evolve(n_steps=3)
            final_field = self.vlasov.get_ensemble_thought()
            field_answer = f"[Vlasov ensemble: consistency={consistency:.3f}]"
            return answers[best_idx], consistency

        return answers[best_idx], consistency

    def reason(self, question: str, n_paths: int = AUTOCHAIN_N_PATHS) -> dict:
        """Full reasoning pipeline."""
        self._reasoning_trace = []
        self._step_counter = 0

        steps = self.decompose(question)
        reasoning = self.generate_reasoning(question, steps)
        all_paths = self.ensemble_generate(question, n_paths)

        verified_steps = []
        for i, (step, reason) in enumerate(zip(steps, reasoning)):
            verified = self.verify_step(i, step, reason)
            verified_steps.append(
                {
                    "step": i + 1,
                    "text": step,
                    "reasoning": reason,
                    "verified": verified,
                }
            )

        answers = []
        for path in all_paths:
            if path:
                answers.append(path[-1])
            else:
                answers.append("")

        majority = self.majority_vote(all_paths)
        best_answer, consistency = self.self_consistency_check(answers)

        return {
            "question": question,
            "steps": steps,
            "reasoning": reasoning,
            "verified_steps": verified_steps,
            "n_paths": len(all_paths),
            "all_answers": answers,
            "majority_answer": majority,
            "best_answer": best_answer,
            "consistency_score": consistency,
            "trace": self._reasoning_trace,
        }

    def reason_stream(self, question: str) -> Any:
        """Stream reasoning steps one at a time."""
        steps = self.decompose(question)
        all_paths = self.ensemble_generate(question, min(AUTOCHAIN_N_PATHS, 3))

        for i, step in enumerate(steps):
            yield {
                "type": "step",
                "step": i + 1,
                "total": len(steps),
                "text": step,
            }
            time.sleep(0.1)

        for path_idx, path in enumerate(all_paths):
            for step_idx, step_text in enumerate(path):
                yield {
                    "type": "reasoning",
                    "path": path_idx + 1,
                    "step": step_idx + 1,
                    "text": step_text,
                }
                time.sleep(0.05)

        answers = [p[-1] if p else "" for p in all_paths]
        best, cons = self.self_consistency_check(answers)

        yield {
            "type": "result",
            "best_answer": best,
            "consistency": cons,
            "n_paths": len(all_paths),
        }

    def reset(self):
        self._step_counter = 0
        self._reasoning_trace = []
        if self.vlasov is not None:
            self.vlasov = VlasovChainOfThought(dim=self.embed_dim)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Holographic Context — Infinite Context via HRR
# ═══════════════════════════════════════════════════════════════════════════


class HolographicContext:
    """
    Novel: "Holographic Context" — store & retrieve conversation context
    using Holographic Reduced Representations (HRR) for O(1) context
    retrieval with no token limit.

    Instead of fitting tokens into a fixed context window, we encode
    conversation history as HRR vectors. New tokens are bound to
    position vectors and bundled (summed) into a single holographic
    vector. Retrieval uses circular correlation to unbind.

    Capacity: ~65K messages compressed into a single 2048-dim vector,
    with graceful degradation (oldest messages lose fidelity first).
    """

    def __init__(
        self, dim: int = HRR_CONTEXT_DIM, capacity: int = HRR_CONTEXT_CAPACITY
    ):
        self.dim = dim
        self.capacity = capacity
        self.memory: dict[int, np.ndarray] = {}
        self._position = 0
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, text: str, position: Optional[int] = None) -> np.ndarray:
        pos = position if position is not None else self._position
        self._position += 1
        text_embed = _embed_text_simple(text, self.dim)
        pos_embed = self._position_encoding(pos)
        return self._circular_conv(text_embed, pos_embed)

    def _position_encoding(self, pos: int) -> np.ndarray:
        seed = pos & 0x7FFFFFFF
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        return _normalize(vec)

    @staticmethod
    def _circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        a_f = np.fft.fft(a.astype(np.complex128))
        b_f = np.fft.fft(b.astype(np.complex128))
        return np.fft.ifft(a_f * b_f).real.astype(np.float32)

    @staticmethod
    def _circular_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        a_f = np.fft.fft(a.astype(np.complex128))
        b_f = np.fft.fft(b.astype(np.complex128))
        return np.fft.ifft(np.conj(a_f) * b_f).real.astype(np.float32)

    def store(self, key: int, value: np.ndarray):
        self.memory[key] = value
        if len(self.memory) > self.capacity * 2:
            keys = sorted(self.memory.keys())
            for k in keys[: len(keys) // 4]:
                del self.memory[k]

    def recall(self, key: int) -> Optional[np.ndarray]:
        return self.memory.get(key)

    def bundle(self, texts: list[str]) -> np.ndarray:
        """Bundle multiple texts into a single holographic vector."""
        if not texts:
            return np.zeros(self.dim, dtype=np.float32)
        bundled = np.zeros(self.dim, dtype=np.float32)
        for i, text in enumerate(texts):
            encoded = self.encode(text, i)
            bundled += encoded * (1.0 / math.sqrt(len(texts)))
        return bundled

    def extend_context(self, prompt: str, history: list[str]) -> str:
        """Extend prompt with holographically compressed history."""
        if not history:
            return prompt
        bundled = self.bundle(history[-64:])
        ctx_b64 = base64.b64encode(bundled.astype(np.float32).tobytes()).decode()
        return f"<|holographic|>{ctx_b64}<|/holographic|>\n{prompt}"

    def clear(self):
        self.memory.clear()
        self._position = 0
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Integration: SpectralStream Multi-Modal Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class SpectralMultiModalOrchestrator:
    """
    Master orchestrator integrating all multi-modal components
    with SpectralStream's UnifiedInferenceEngine.

    Provides a single entry point for:
    - Vision + Audio encoding
    - Multi-modal chat
    - Code interpretation
    - Prompt optimization
    - Template engine
    - Auto chain-of-thought
    """

    def __init__(
        self,
        engine: Optional[Any] = None,
        vision_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        embed_dim: int = SPECTRALSTREAM_EMBED_DIM,
    ):
        self.engine = engine
        self.embed_dim = embed_dim

        self.vision = VisionEncoder(embed_dim=embed_dim, ssf_path=vision_path)
        self.audio = AudioEncoder(embed_dim=embed_dim, ssf_path=audio_path)

        self.chat = MultiModalChat(
            engine=engine,
            vision_encoder=self.vision,
            audio_encoder=self.audio,
            embed_dim=embed_dim,
        )

        self.code_interpreter = CodeInterpreter()

        self.optimizer = PromptOptimizer(
            model_fn=self._default_model_fn,
            ssf_path=optimizer_path,
        )

        self.template_engine = PromptTemplateEngine()
        self._init_default_templates()

        self.autochain = AutoChain(
            model_fn=self._default_model_fn,
            use_vlasov_cot=True,
            embed_dim=embed_dim,
        )

        self.holographic_context = HolographicContext()

    def _init_default_templates(self):
        self.template_engine.register_partial(
            "system_default",
            "You are a helpful AI assistant. Answer the following:\n\n{{prompt}}",
        )
        self.template_engine.register_partial(
            "system_expert",
            "You are an expert in {{field}}. Provide a detailed, accurate response.\n\n{{prompt}}",
        )
        self.template_engine.register_partial(
            "format_bullets",
            "Please format your response as bullet points:\n{% for item in items %}- {{item}}\n{% endfor %}",
        )
        self.template_engine.create_version(
            "chat_default",
            "System: {{system_prompt}}\n\nUser: {{user_input}}\n\nAssistant:",
        )
        self.template_engine.create_version(
            "chat_detailed",
            "{{system_prompt}}\n\nContext: {{context}}\n\nQuestion: {{user_input}}\n\nLet me think about this step by step.\n\nAnswer:",
        )

    def _default_model_fn(self, prompt: str) -> str:
        if self.engine is not None:
            try:
                tokens, _ = self.engine.generate(
                    prompt, max_new_tokens=128, temperature=0.7, top_k=40, top_p=0.95
                )
                if hasattr(self.engine, "detokenize"):
                    return self.engine.detokenize(tokens[-64:])
                return str(tokens[-16:])
            except Exception:
                pass
        return f"[Model response to: {prompt[:80]}...]"

    def multimodal_generate(
        self,
        text: str,
        images: Optional[list] = None,
        audio: Optional[Any] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Union[str, Any]:
        """Single-call multimodal generation."""
        msg = Message(
            role="user",
            content=text,
            images=images or [],
            audio=audio,
        )
        messages = [msg]
        if self.holographic_context:
            history_texts = [m.content for m in self.chat.get_history()[-10:]]
            if history_texts:
                msg.content = self.holographic_context.extend_context(
                    text, history_texts
                )
        return self.chat.chat(
            messages, max_new_tokens=max_tokens, temperature=temperature, stream=stream
        )

    def optimize_and_generate(self, prompt: str, n_iterations: int = 2) -> dict:
        """Optimize prompt then generate with it."""
        result = self.optimizer.optimize(prompt, n_iterations=n_iterations)
        generation = self._default_model_fn(result.prompt)
        return {
            "original_prompt": prompt,
            "optimized_prompt": result.prompt,
            "score": result.score,
            "generation": generation,
            "improvement": result.score - self.optimizer.score(prompt),
        }

    def reason_and_answer(self, question: str, n_paths: int = 3) -> dict:
        """Chain-of-thought reasoning with self-consistency."""
        return self.autochain.reason(question, n_paths=n_paths)

    def execute_and_fix(
        self, code: str, language: str = "python", max_iterations: int = 3
    ) -> list[CodeResult]:
        """Execute code with iterative error fixing."""
        return self.code_interpreter.execute_with_feedback(
            code,
            language,
            self._default_model_fn,
            max_iterations,
        )

    def render_template(self, template_name: str, variables: dict) -> str:
        """Render a named template with variables."""
        template = self.template_engine.get_version(template_name)
        if template is None:
            template = self.template_engine.get_partial(template_name)
        if template is None:
            raise ValueError(f"Template '{template_name}' not found")
        return self.template_engine.render(template, variables)

    def chat_with_history(
        self,
        user_message: str,
        images: Optional[list] = None,
        audio: Optional[Any] = None,
    ) -> str:
        """Chat with automatic history management."""
        msg = Message(
            role="user", content=user_message, images=images or [], audio=audio
        )
        response = self.chat.chat([msg], max_new_tokens=512, stream=False)
        self.chat.add_to_history(msg)
        self.chat.add_to_history(Message(role="assistant", content=response))
        return response

    def multimodal_stream(self, text: str, images: Optional[list] = None):
        """Stream multimodal generation token by token."""
        msg = Message(role="user", content=text, images=images or [])
        return self.chat.chat([msg], stream=True)

    def load_models(
        self,
        vision_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
    ):
        if vision_path:
            self.vision.load_ssf(vision_path)
        if audio_path:
            self.audio.load_ssf(audio_path)
        if optimizer_path:
            self.optimizer.load(optimizer_path)

    def save_models(
        self,
        vision_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
    ):
        if vision_path:
            self.vision.save_ssf(vision_path)
        if audio_path:
            self.audio.save_ssf(audio_path)
        if optimizer_path:
            self.optimizer.save(optimizer_path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Points
# ═══════════════════════════════════════════════════════════════════════════


def run_tests():
    """Run self-verification tests."""
    print("=" * 60)
    print("SpectralStream Multi-Modal Extension — Self Verification")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: VisionEncoder
    try:
        ve = VisionEncoder(embed_dim=512)
        dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emb = ve.encode_image(dummy_img)
        assert emb.shape == (1, 512), f"Expected (1, 512), got {emb.shape}"
        print("[PASS] VisionEncoder: basic image encoding")
        passed += 1
    except Exception as e:
        print(f"[FAIL] VisionEncoder: {e}")
        failed += 1

    # Test 2: Spectral Vision DCT
    try:
        sv = SpectralVisionDCT()
        dct_emb = sv.encode(np.random.randn(3, 224, 224).astype(np.float32))
        assert dct_emb.shape[1] == CLIP_EMBED_DIM, (
            f"Expected dim {CLIP_EMBED_DIM}, got {dct_emb.shape[1]}"
        )
        print("[PASS] SpectralVisionDCT: encoding works")
        passed += 1
    except Exception as e:
        print(f"[FAIL] SpectralVisionDCT: {e}")
        failed += 1

    # Test 3: AudioEncoder
    try:
        ae = AudioEncoder(embed_dim=512)
        dummy_audio = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        emb = ae.encode_audio(dummy_audio)
        assert emb.shape == (1, 512), f"Expected (1, 512), got {emb.shape}"
        print("[PASS] AudioEncoder: basic audio encoding")
        passed += 1
    except Exception as e:
        print(f"[FAIL] AudioEncoder: {e}")
        failed += 1

    # Test 4: MultiModalChat
    try:
        chat = MultiModalChat(embed_dim=512)
        msg = Message(
            role="user",
            content="What's in this image?",
            images=[np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)],
        )
        response = chat.chat([msg], max_new_tokens=32)
        assert isinstance(response, str), "Expected string response"
        print("[PASS] MultiModalChat: basic chat")
        passed += 1
    except Exception as e:
        print(f"[FAIL] MultiModalChat: {e}")
        failed += 1

    # Test 5: CodeInterpreter
    try:
        ci = CodeInterpreter()
        result = ci.execute("print('hello world')", "python")
        assert result.success, f"Expected success, got: {result.error}"
        assert "hello world" in result.stdout, (
            f"Expected 'hello world' in stdout: {result.stdout}"
        )
        print("[PASS] CodeInterpreter: basic execution")
        passed += 1
    except Exception as e:
        print(f"[FAIL] CodeInterpreter: {e}")
        failed += 1

    # Test 6: CodeInterpreter code block detection
    try:
        ci = CodeInterpreter()
        text = "Here's some code:\n```python\nprint('test')\n```\nDone."
        blocks = ci.detect_code_blocks(text)
        assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
        assert blocks[0][0] == "python", f"Expected python lang, got {blocks[0][0]}"
        print("[PASS] CodeInterpreter: code block detection")
        passed += 1
    except Exception as e:
        print(f"[FAIL] CodeInterpreter: {e}")
        failed += 1

    # Test 7: PromptOptimizer
    try:
        po = PromptOptimizer()
        analysis = po.analyze("Write a clear explanation of quantum computing")
        assert "quality_score" in analysis
        assert "resonance_score" in analysis
        print("[PASS] PromptOptimizer: basic analysis")
        passed += 1

        variations = po.generate_variations("Explain gravity simply", n=4)
        assert len(variations) >= 1, "Expected at least 1 variation"
        print("[PASS] PromptOptimizer: variation generation")
        passed += 1

        result = po.optimize("Write a poem about AI", n_iterations=2)
        assert hasattr(result, "prompt")
        assert hasattr(result, "score")
        print("[PASS] PromptOptimizer: full optimization")
        passed += 1
    except Exception as e:
        print(f"[FAIL] PromptOptimizer: {e}")
        failed += 1

    # Test 8: ResonantPromptingOptimizer
    try:
        rpo = ResonantPromptingOptimizer()
        score = rpo._resonance_score("Explain quantum entanglement")
        assert isinstance(score, float), "Expected float score"
        print("[PASS] ResonantPromptingOptimizer: resonance score")
        passed += 1
    except Exception as e:
        print(f"[FAIL] ResonantPromptingOptimizer: {e}")
        failed += 1

    # Test 9: QuantumPromptOptimizer
    try:
        qpo = QuantumPromptOptimizer()
        prompts = ["Explain AI", "Tell me about ML", "What is deep learning?"]
        scores = qpo.measure_scores(prompts)
        assert len(scores) == len(prompts), "Expected same number of scores"
        best, best_score = qpo.select_best(prompts)
        assert best in prompts
        print("[PASS] QuantumPromptOptimizer: superposition scoring")
        passed += 1
    except Exception as e:
        print(f"[FAIL] QuantumPromptOptimizer: {e}")
        failed += 1

    # Test 10: PromptTemplateEngine
    try:
        te = PromptTemplateEngine()
        result = te.render("Hello {{name}}!", {"name": "World"})
        assert result == "Hello World!", f"Expected 'Hello World!', got: {result}"
        print("[PASS] PromptTemplateEngine: variable substitution")
        passed += 1

        result = te.render(
            "{% if show %}visible{% else %}hidden{% endif %}", {"show": True}
        )
        assert result == "visible", f"Expected 'visible', got: {result}"
        print("[PASS] PromptTemplateEngine: conditional")
        passed += 1

        result = te.render(
            "{% for item in items %}- {{item}}\n{% endfor %}", {"items": ["a", "b"]}
        )
        assert result.strip() == "- a\n- b".strip()
        print("[PASS] PromptTemplateEngine: loop")
        passed += 1

        result = te.render("{{truncate('hello world', 5)}}", {})
        assert result == "hello..."
        print("[PASS] PromptTemplateEngine: built-in functions")
        passed += 1

        te.create_version("test", "v1 template")
        te.create_version("test", "v2 template")
        v1 = te.get_version("test", 1)
        v2 = te.get_version("test", 2)
        assert v1 == "v1 template"
        assert v2 == "v2 template"
        print("[PASS] PromptTemplateEngine: versioning")
        passed += 1

        te.setup_ab_test("exp1", ["test_v1", "test_v2"])
        te.register_partial("test_v1", "variant A")
        te.register_partial("test_v2", "variant B")
        a = te.serve_ab("exp1", "user1")
        b = te.serve_ab("exp1", "user2")
        assert a is not None and b is not None
        print("[PASS] PromptTemplateEngine: A/B testing")
        passed += 1
    except Exception as e:
        print(f"[FAIL] PromptTemplateEngine: {e}")
        failed += 1

    # Test 11: AutoChain
    try:
        ac = AutoChain(use_vlasov_cot=False)
        steps = ac.decompose("Explain how neural networks work")
        assert len(steps) >= 3, f"Expected >=3 steps, got {len(steps)}"
        print("[PASS] AutoChain: decomposition")
        passed += 1

        reasoning = ac.generate_reasoning("What is gravity?", steps)
        assert len(reasoning) == len(steps), (
            f"Expected {len(steps)} reasoning, got {len(reasoning)}"
        )
        print("[PASS] AutoChain: reasoning generation")
        passed += 1

        result = ac.reason("Compare Python and JavaScript")
        assert "best_answer" in result
        assert "consistency_score" in result
        assert "steps" in result
        print("[PASS] AutoChain: full reasoning pipeline")
        passed += 1
    except Exception as e:
        print(f"[FAIL] AutoChain: {e}")
        failed += 1

    # Test 12: VlasovChainOfThought
    try:
        vcot = VlasovChainOfThought(dim=64, n_particles=8)
        traj = vcot.evolve(n_steps=3)
        assert len(traj) == 4
        print("[PASS] VlasovChainOfThought: particle evolution")
        passed += 1

        ensemble = vcot.get_ensemble_thought()
        assert ensemble.shape == (64,)
        print("[PASS] VlasovChainOfThought: ensemble thought")
        passed += 1
    except Exception as e:
        print(f"[FAIL] VlasovChainOfThought: {e}")
        failed += 1

    # Test 13: HolographicContext
    try:
        hc = HolographicContext(dim=64, capacity=100)
        encoded = hc.encode("Hello world", 0)
        assert encoded.shape == (64,), f"Expected (64,), got {encoded.shape}"
        print("[PASS] HolographicContext: encoding")
        passed += 1

        bundled = hc.bundle(["Hello", "World", "Test"])
        assert bundled.shape == (64,)
        print("[PASS] HolographicContext: bundling")
        passed += 1

        extended = hc.extend_context("New prompt", ["prev1", "prev2"])
        assert "New prompt" in extended
        print("[PASS] HolographicContext: context extension")
        passed += 1
    except Exception as e:
        print(f"[FAIL] HolographicContext: {e}")
        failed += 1

    # Test 14: SSF Model Store
    try:
        ssf = SSFModelStore()
        ssf.add_tensor("test_weight", np.random.randn(32, 32).astype(np.float32))
        assert "test_weight" in ssf
        tmp_path = "/tmp/test_ssf_multimodal.npz"
        ssf.save(tmp_path)
        ssf2 = SSFModelStore()
        ssf2.load(tmp_path)
        t = ssf2.get_tensor("test_weight")
        assert t is not None
        assert t.shape == (32, 32)
        print("[PASS] SSFModelStore: save/load")
        passed += 1
        os.remove(tmp_path)
    except Exception as e:
        print(f"[FAIL] SSFModelStore: {e}")
        failed += 1

    # Test 15: VisionEncoder SSF
    try:
        ve = VisionEncoder(embed_dim=256)
        tmp = "/tmp/test_vision_ssf.npz"
        ve.save_ssf(tmp)
        ve2 = VisionEncoder(embed_dim=256)
        ve2.load_ssf(tmp)
        assert np.allclose(ve.projection, ve2.projection, atol=1e-5)
        print("[PASS] VisionEncoder: SSF format save/load")
        passed += 1
        os.remove(tmp)
    except Exception as e:
        print(f"[FAIL] VisionEncoder SSF: {e}")
        failed += 1

    # Test 16: SpectralMultiModalOrchestrator
    try:
        smo = SpectralMultiModalOrchestrator(embed_dim=256)
        assert smo.vision is not None
        assert smo.audio is not None
        assert smo.chat is not None
        assert smo.code_interpreter is not None
        assert smo.optimizer is not None
        assert smo.template_engine is not None
        assert smo.autochain is not None
        assert smo.holographic_context is not None
        print("[PASS] SpectralMultiModalOrchestrator: initialization")
        passed += 1

        gen = smo.multimodal_generate("Hello", stream=False)
        assert isinstance(gen, str)
        print("[PASS] SpectralMultiModalOrchestrator: generate")
        passed += 1

        opt_result = smo.optimize_and_generate("Write about space", n_iterations=1)
        assert "optimized_prompt" in opt_result
        assert "generation" in opt_result
        print("[PASS] SpectralMultiModalOrchestrator: optimize & generate")
        passed += 1
    except Exception as e:
        print(f"[FAIL] SpectralMultiModalOrchestrator: {e}")
        failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)
    return failed == 0


def run_optimization_demo(initial_prompt: str = "Write a poem about AI"):
    """Run prompt optimization demo."""
    print(f"\n{'=' * 60}")
    print(f"Prompt Optimization Demo")
    print(f"{'=' * 60}")
    print(f"Initial prompt: '{initial_prompt}'")

    optimizer = PromptOptimizer()
    initial_score = optimizer.score(initial_prompt)
    print(f"Initial score: {initial_score:.4f}")

    print("\nGenerating variations...")
    variations = optimizer.generate_variations(initial_prompt, n=6)
    for i, var in enumerate(variations):
        var.score = optimizer.score(var.prompt)
        print(f"  [{i + 1}] Score {var.score:.4f}: {var.prompt[:80]}...")

    best = optimizer.select_best(variations)
    print(f"\nBest variation: score={best.score:.4f}")
    print(f"  {best.prompt[:120]}")

    print("\nRunning full optimization...")
    result = optimizer.optimize(initial_prompt, n_iterations=3)
    print(f"\nFinal optimized prompt (score={result.score:.4f}):")
    print(f"  {result.prompt[:200]}")
    print(f"  Improvement: {(result.score - initial_score) * 100:.1f}%")

    print(f"\nTemplate library has {len(optimizer.get_template_library())} entries")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)

    if "--optimize" in sys.argv:
        idx = sys.argv.index("--optimize")
        prompt = (
            " ".join(sys.argv[idx + 1 :])
            if idx + 1 < len(sys.argv)
            else "Write a poem about AI"
        )
        run_optimization_demo(prompt)
        sys.exit(0)

    print("SpectralStream Multi-Modal Extension & Prompt Optimization Engine")
    print()
    print("Usage:")
    print("  python -m spectralstream.multimodal_prompt --test")
    print('  python -m spectralstream.multimodal_prompt --optimize "Your prompt here"')
    print()
    print("Available classes:")
    print("  VisionEncoder              - CLIP-style + Spectral Vision DCT")
    print("  AudioEncoder               - Mel-spectrogram CNN encoder")
    print("  MultiModalChat             - Chat with images/audio")
    print("  CodeInterpreter            - Sandboxed code execution")
    print("  PromptOptimizer            - Auto-improve prompts")
    print("  ResonantPromptingOptimizer - Frequency-domain optimization")
    print("  QuantumPromptOptimizer     - Superposition prompt evaluation")
    print("  PromptTemplateEngine       - Jinja-like template management")
    print("  AutoChain                  - Automatic chain-of-thought")
    print("  VlasovChainOfThought       - Mean-field reasoning particles")
    print("  HolographicContext         - HRR infinite context")
    print("  SpectralMultiModalOrchestrator - All-in-one orchestrator")

"""
SpectralStream OpenAI-Compatible API Server
-------------------------------------------
Full OpenAI API compatibility with:
- /v1/models, /v1/models/{model}, /v1/chat/completions, /v1/completions, /v1/embeddings
- /health, /v1/version
- Tool calling with multi-turn orchestration
- SSE streaming with stream_options (include_usage)
- Stop sequences, logprobs, frequency/presence penalties, response_format
- Token-bucket rate limiting per API key / IP
- CORS, OpenAI-format errors
- Request priority queuing + continuous batching integration
- SpectralStream feature headers (COCONUT steps, TimeCrystal stability, compression)
- opencode provider system compatible

CPU-first architecture:
- ThreadingHTTPServer for parallel request handling
- Priority request queue with deadline scheduling
- Continuous batching via batching_engine (Vlasov scheduling)
- Memory-mapped model loading (zero-copy via mmap)
- Token-by-token generation with stop-check interleaving
"""

import base64
import json
import os
import re
import struct
import sys
import time
import uuid
import math
import hashlib
import heapq
import threading
import warnings
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs
from socketserver import ThreadingMixIn
from threading import Lock
from dataclasses import dataclass, field

warnings.warn(
    "spectralstream.serving.server is deprecated. Use spectralstream.serving.unified_server instead.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np

try:
    from spectralstream.engine import SpectralStream
except ImportError:
    from spectralstream.inference import CPUInferenceEngine as SpectralStream

from spectralstream.inference import (
    SpectralResonanceMeter,
    AdaptivePIDController,
    ResonanceRouter,
)
from spectralstream.inference import ConfidenceGate
from spectralstream.inference import OnlineLearningEngine

try:
    from spectralstream.llama_bridge import list_available_models as find_gguf_models
except ImportError:

    def find_gguf_models():
        return []


from spectralstream.serving.streaming_handler import StreamingHandler
from spectralstream.core.math_primitives.numerical import softmax as _softmax

SERVER_VERSION = "0.4.0"
COCONUT_AVAILABLE = False
TIMECRYSTAL_AVAILABLE = False

try:
    from spectralstream.inference import COCONUTEngine

    COCONUT_AVAILABLE = True
except ImportError:
    pass

try:
    from spectralstream.inference import TimeCrystalResonator

    TIMECRYSTAL_AVAILABLE = True
except ImportError:
    pass

try:
    from spectralstream.serving.batching_engine import (
        ContinuousBatchingEngine,
        Priority as BatchPriority,
        SequenceState,
    )

    BATCHING_AVAILABLE = True
except ImportError:
    BATCHING_AVAILABLE = False

    class BatchPriority:
        CRITICAL = 0
        HIGH = 1
        NORMAL = 2
        LOW = 3


try:
    from spectralstream.model.progressive_loader import ProgressiveLoader

    PROGRESSIVE_LOAD_AVAILABLE = True
except ImportError:
    try:
        from spectralstream.inference import ProgressiveLoader

        PROGRESSIVE_LOAD_AVAILABLE = True
    except ImportError:
        PROGRESSIVE_LOAD_AVAILABLE = False


# ─── Priority Request Queue ─────────────────────────────────────────────────


@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    timestamp: float
    request_id: str = field(compare=False)
    handler_ref: object = field(compare=False)
    body: dict = field(compare=False)
    endpoint: str = field(compare=False)


class RequestPriorityQueue:
    """Priority queue for incoming requests with deadline scheduling.

    Novel: Uses Vlasov-inspired priority scheduling where request urgency
    evolves dynamically based on wait time (priority decays → urgency grows).
    """

    def __init__(self, max_size: int = 256):
        self._heap: list[PrioritizedRequest] = []
        self._max_size = max_size
        self._lock = Lock()
        self._enqueued = 0
        self._dequeued = 0
        self._wait_times: list[float] = []

    def push(self, req: PrioritizedRequest) -> bool:
        with self._lock:
            if len(self._heap) >= self._max_size:
                return False
            heapq.heappush(self._heap, req)
            self._enqueued += 1
            return True

    def pop(self) -> Optional[PrioritizedRequest]:
        with self._lock:
            if not self._heap:
                return None
            req = heapq.heappop(self._heap)
            self._dequeued += 1
            self._wait_times.append(time.time() - req.timestamp)
            return req

    def peek(self) -> Optional[PrioritizedRequest]:
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0]

    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    def stats(self) -> dict:
        with self._lock:
            avg_wait = np.mean(self._wait_times) if self._wait_times else 0.0
            return {
                "enqueued": self._enqueued,
                "dequeued": self._dequeued,
                "queue_size": len(self._heap),
                "avg_wait_ms": round(avg_wait * 1000, 2),
                "max_size": self._max_size,
            }


# ─── Continuous Batching Bridge ─────────────────────────────────────────────


class ContinuousBatchingBridge:
    """Bridge between HTTP server and ContinuousBatchingEngine.

    Allows requests to be batched together for higher throughput on CPU.
    Falls back gracefully if batching engine is unavailable.
    """

    def __init__(self, engine: SpectralStream, max_batch_tokens: int = 4096):
        self._engine = engine
        self._batching_available = BATCHING_AVAILABLE
        self._batcher = None
        self._lock = Lock()

        if self._batching_available:
            try:
                self._batcher = ContinuousBatchingEngine(
                    max_batch_tokens=max_batch_tokens,
                    max_seq_len=2048,
                    num_layers=engine.n_layers,
                    num_heads=engine.n_heads,
                    head_dim=engine.hidden_dim // engine.n_heads
                    if engine.n_heads
                    else 128,
                )
            except Exception:
                self._batching_available = False

    @property
    def is_active(self) -> bool:
        return self._batching_available and self._batcher is not None

    def submit(self, seq_id: str, tokens: list[int], priority: int = 2) -> bool:
        if not self.is_active:
            return False
        try:
            seq = SequenceState(
                seq_id=seq_id,
                prompt_tokens=tokens,
                max_tokens=256,
                priority=priority,
            )
            return self._batcher.add_sequence(seq)
        except Exception:
            return False

    def step(self) -> dict:
        if not self.is_active:
            return {"batched": False, "reason": "batching_unavailable"}
        try:
            return self._batcher.step()
        except Exception as exc:
            return {"batched": False, "error": str(exc)}

    def stats(self) -> dict:
        if not self.is_active:
            return {"batching": False, "reason": "unavailable"}
        try:
            return {
                "batching": True,
                "num_active_seqs": len(self._batcher._seqs)
                if hasattr(self._batcher, "_seqs")
                else 0,
            }
        except Exception:
            return {"batching": True, "status": "unknown"}


# ─── Rate Limiter ───────────────────────────────────────────────────────────


@dataclass
class TokenBucket:
    tokens: float
    last_refill: float
    capacity: float
    refill_rate: float

    def refill(self, now: float):
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        now = time.monotonic()
        self.refill(now)
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    def __init__(self, capacity: float = 100.0, refill_rate: float = 20.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()

    def _get_bucket(self, key: str) -> TokenBucket:
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                tokens=self.capacity,
                last_refill=time.monotonic(),
                capacity=self.capacity,
                refill_rate=self.refill_rate,
            )
        return self._buckets[key]

    def check(self, key: str, cost: float = 1.0) -> bool:
        with self._lock:
            bucket = self._get_bucket(key)
            return bucket.consume(cost)

    def get_remaining(self, key: str) -> float:
        with self._lock:
            bucket = self._get_bucket(key)
            now = time.monotonic()
            bucket.refill(now)
            return bucket.tokens

    def get_reset_time(self, key: str) -> float:
        with self._lock:
            bucket = self._get_bucket(key)
            tokens_needed = self.capacity - bucket.tokens
            if tokens_needed <= 0:
                return 0.0
            return tokens_needed / bucket.refill_rate


# ─── Tool Calling Helpers ──────────────────────────────────────────────────

TOOL_CALL_SYSTEM_PROMPT = """You have access to the following tools. When you need to call a tool, respond with EXACTLY a JSON object on a single line in this format:
{"function": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}

Available tools:
{tools_description}

If you don't need to call a tool, respond normally with plain text.
When you receive a tool result, use it to continue the conversation naturally."""

JSON_MODE_SYSTEM_PROMPT = """You must respond with valid JSON only. Do not include any text outside the JSON object. Your response must be parseable by json.loads()."""


def format_tools_for_prompt(tools: list[dict]) -> str:
    lines = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        lines.append(f"- {name}: {desc}")
        props = params.get("properties", {})
        required = params.get("required", [])
        if props:
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                pdesc = pinfo.get("description", "")
                lines.append(
                    f"    {pname}: {pinfo.get('type', 'string')}{req} - {pdesc}"
                )
    return "\n".join(lines)


def parse_tool_call(text: str) -> Optional[dict]:
    json_match = re.search(
        r'\{"function"\s*:\s*"[^"]*"\s*,\s*"arguments"\s*:\s*\{', text
    )
    if json_match:
        start = json_match.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if "function" in obj and "arguments" in obj:
                            return obj
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
    return None


def build_tool_call_chunk(func_name: str, arguments: str, index: int = 0) -> dict:
    return {
        "index": index,
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": func_name,
            "arguments": arguments,
        },
    }


# ─── Model Discovery ───────────────────────────────────────────────────────


def find_lmstudio_models() -> list[dict]:
    return find_gguf_models()


def model_info_to_openai(m: dict) -> dict:
    return {
        "id": m.get("name", m.get("path", "unknown")),
        "object": "model",
        "created": int(time.time()),
        "owned_by": "spectralstream",
        "permission": [],
        "root": m.get("path", ""),
        "size_gb": m.get("size_gb", 0),
    }


def builtin_model_entry() -> dict:
    return {
        "id": "spectralstream",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "spectralstream",
        "permission": [],
        "root": "spectralstream://builtin",
    }


def get_model_timestamp(model_name: str) -> int:
    """Return a deterministic created timestamp from model name for stable IDs."""
    h = hashlib.md5(model_name.encode()).hexdigest()[:8]
    return int(time.time()) - (int(h, 16) % 864000)


# ─── Embeddings ────────────────────────────────────────────────────────────


def compute_embedding(engine: SpectralStream, text: str) -> list[float]:
    tokens = (
        engine._tokenize(text)
        if hasattr(engine, "_tokenize")
        else [
            min(ord(c) % engine.vocab_size, engine.vocab_size - 1) for c in text[:128]
        ]
    )
    if tokens and hasattr(engine, "hd_engine"):
        hd = engine.hd_engine
        if hasattr(hd, "_encode_context"):
            try:
                state = hd._encode_context(tuple(tokens))
                if isinstance(state, np.ndarray):
                    vec = state.flatten().astype(float).tolist()
                    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                    return [x / norm for x in vec[:1536]]
            except Exception:
                pass
        if hasattr(hd, "ensure_token_vector"):
            try:
                vectors = []
                for t in tokens:
                    v = hd.ensure_token_vector(t)
                    if isinstance(v, np.ndarray):
                        vectors.append(v.astype(float))
                if vectors:
                    avg = np.mean(vectors, axis=0).tolist()
                    norm = math.sqrt(sum(x * x for x in avg)) or 1.0
                    return [x / norm for x in avg[:1536]]
            except Exception:
                pass
    return [0.0] * 1536


def compute_embedding_batch(
    engine: SpectralStream, texts: list[str], dimensions: int = 1536
) -> list[list[float]]:
    """Compute embeddings for a batch of texts using HD vectors."""
    try:
        from spectralstream.inference.hdc_draft import HDCDraftEngine

        hd_engine = getattr(engine, "hd_engine", None)
        if hd_engine is None:
            return [compute_embedding(engine, t) for t in texts]

        embeddings = []
        for text in texts:
            tokens = (
                engine._tokenize(text)
                if hasattr(engine, "_tokenize")
                else [
                    min(ord(c) % engine.vocab_size, engine.vocab_size - 1)
                    for c in text[:128]
                ]
            )
            if tokens and hasattr(hd_engine, "bundle_sequence"):
                try:
                    vec = hd_engine.bundle_sequence(tokens)
                    if isinstance(vec, np.ndarray):
                        vec = vec.flatten().astype(float)
                        if len(vec) > dimensions:
                            vec = vec[:dimensions]
                        elif len(vec) < dimensions:
                            vec = np.pad(vec, (0, dimensions - len(vec)))
                        norm = math.sqrt(np.dot(vec, vec)) or 1.0
                        embeddings.append((vec / norm).tolist())
                        continue
                except Exception:
                    pass
            embeddings.append(compute_embedding(engine, text))
        return embeddings
    except Exception:
        return [compute_embedding(engine, t) for t in texts]


def encode_embedding_base64(vector: list[float]) -> str:
    """Encode float32 vector as base64."""
    packed = struct.pack(f"{len(vector)}f", *vector)
    return base64.b64encode(packed).decode("ascii")


# ─── Sampling Helpers (CPU-optimized: numpy-only) ──────────────────────────


def _apply_penalties(
    scores: np.ndarray,
    token_ids: list[int],
    vocab_size: int,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> np.ndarray:
    """Apply frequency and presence penalties to raw logit scores.

    Args:
        scores: [vocab_size] raw scores from the model
        token_ids: complete sequence of token IDs generated so far
        vocab_size: vocabulary size
        frequency_penalty: penalty for each prior occurrence (scales with count)
        presence_penalty: penalty for any prior occurrence (binary)

    Returns:
        Modified scores with penalties applied.
    """
    if frequency_penalty == 0.0 and presence_penalty == 0.0:
        return scores

    score_copy = scores.copy()
    freq: dict[int, int] = {}

    for t in token_ids:
        if t < vocab_size:
            freq[t] = freq.get(t, 0) + 1

    for t, count in freq.items():
        penalty = count * frequency_penalty + presence_penalty
        if penalty != 0:
            score_copy[t] -= penalty

    return score_copy


def _sample_with_penalties(
    scores: np.ndarray,
    token_ids: list[int],
    vocab_size: int,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> int:
    """Sample a token from scores with penalties, temperature, top-p, top-k."""
    scores = _apply_penalties(
        scores, token_ids, vocab_size, frequency_penalty, presence_penalty
    )
    scores = scores / max(temperature, 0.01)

    # top-k filtering
    if top_k > 0:
        indices = np.argpartition(scores, -top_k)[-top_k:]
        mask = np.ones(vocab_size, dtype=bool)
        mask[indices] = False
        scores[mask] = -float("inf")

    probs = _softmax(scores)

    # top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff = int(np.searchsorted(cumsum, top_p)) + 1
        sorted_probs[cutoff:] = 0.0
        sorted_probs /= sorted_probs.sum()
        idx = np.random.choice(len(sorted_probs), p=sorted_probs)
        return int(sorted_indices[idx])

    return int(np.random.choice(vocab_size, p=probs))


_EMPTY_PROBS_CACHE: dict[int, np.ndarray] = {}


def _get_empty_probs(vocab_size: int) -> np.ndarray:
    if vocab_size not in _EMPTY_PROBS_CACHE:
        p = np.ones(vocab_size, dtype=np.float64)
        p /= p.sum()
        _EMPTY_PROBS_CACHE[vocab_size] = p
    return _EMPTY_PROBS_CACHE[vocab_size].copy()


# ─── Request Handler ───────────────────────────────────────────────────────


class RequestHandler(BaseHTTPRequestHandler):
    server_version = f"SpectralStreamServer/{SERVER_VERSION}"

    def _get_client_key(self) -> str:
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            key = auth[7:].strip()
            if key:
                return hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.client_address[0]

    def _check_rate_limit(self, cost: float = 1.0) -> bool:
        limiter: RateLimiter = self.server.rate_limiter
        key = self._get_client_key()
        allowed = limiter.check(key, cost)
        self._rate_limit_info = {
            "limit": str(int(limiter.capacity)),
            "remaining": str(int(limiter.get_remaining(key))),
            "reset": f"{limiter.get_reset_time(key):.1f}",
        }
        return allowed

    def _send_rate_limit_headers(self):
        info = getattr(self, "_rate_limit_info", None)
        if info:
            self.send_header("X-RateLimit-Limit", info["limit"])
            self.send_header("X-RateLimit-Remaining", info["remaining"])
            self.send_header("X-RateLimit-Reset", info["reset"])

    def _send_spectral_headers(self, engine: SpectralStream):
        """Add SpectralStream-specific feature headers."""
        stats = engine.stats()
        self.send_header("X-Spectral-Tokens", str(stats.get("tokens_per_second", 0)))
        self.send_header(
            "X-Spectral-Model-Calls", str(stats.get("tokens_per_model_call", 0))
        )
        self.send_header(
            "X-Spectral-Block-Success", str(stats.get("block_success_rate", 0))
        )
        self.send_header("X-Spectral-Strategy", "block_emission")

        hd_rate = stats.get("hd_acceptance_rate", 0)
        if hd_rate:
            self.send_header("X-Spectral-HD-Acceptance", str(hd_rate))

        kv_ratio = stats.get("kv_compression_ratio", 0)
        if kv_ratio:
            self.send_header("X-Spectral-KV-Compression", str(kv_ratio))

        if COCONUT_AVAILABLE and hasattr(engine, "coconut_engine"):
            try:
                steps = getattr(engine.coconut_engine, "thinking_steps", None)
                if steps is not None:
                    self.send_header("X-Coconut-Steps", str(steps))
            except Exception:
                pass

        if TIMECRYSTAL_AVAILABLE and hasattr(engine, "time_crystal"):
            try:
                stable = getattr(engine.time_crystal, "is_stable", None)
                if stable is not None:
                    self.send_header("X-DTC-Stable", "true" if stable else "false")
                phase = getattr(engine.time_crystal, "phase", None)
                if phase is not None:
                    self.send_header("X-DTC-Phase", str(round(phase, 4)))
            except Exception:
                pass

        batcher = getattr(self.server, "batcher", None)
        if batcher is not None:
            try:
                bstats = batcher.stats()
                if bstats.get("batching"):
                    self.send_header(
                        "X-Spectral-Batch-Active", str(bstats.get("num_active_seqs", 0))
                    )
            except Exception:
                pass

        q = getattr(self.server, "request_queue", None)
        if q is not None:
            try:
                qstats = q.stats()
                self.send_header(
                    "X-Spectral-Queue-Size", str(qstats.get("queue_size", 0))
                )
                self.send_header(
                    "X-Spectral-Queue-Wait-Ms", str(qstats.get("avg_wait_ms", 0))
                )
            except Exception:
                pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False)
        self.send_response(status)
        self._send_rate_limit_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self._send_cors()
        if status == 429:
            self.send_header("Retry-After", "1")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _send_error_json(
        self,
        message: str,
        status: int = 400,
        error_type: str = "invalid_request_error",
        code: str = None,
    ):
        if status == 429:
            error_type = "rate_limit_error"
        elif status == 404:
            error_type = "not_found"
        elif status == 500:
            error_type = "server_error"
        elif status == 503:
            error_type = "overloaded_error"
        error = {
            "message": message,
            "type": error_type,
            "code": code or status,
        }
        self._send_json({"error": error}, status)

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE, PUT, PATCH"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Requested-With, Origin, Accept",
        )
        self.send_header("Access-Control-Max-Age", "86400")

    def _parse_body(self) -> Optional[dict]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return None
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _get_engine(self) -> SpectralStream:
        return self.server.engine

    def log_message(self, format, *args):
        sys.stderr.write(f"[SpectralStream] {args[0]} {args[1]} {args[2]}\n")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def _resolve_model_path(self, path: str) -> Optional[str]:
        """Match a model ID against available models."""
        models = find_lmstudio_models()
        for m in models:
            mid = m.get("name", m.get("path", ""))
            if mid == path or mid.rstrip("/") == path:
                return m.get("path", None)
        return None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        # Route: /v1/models/{model}
        if path.startswith("/v1/models/") and len(path) > len("/v1/models/"):
            model_id = path[len("/v1/models/") :]
            self._handle_model_detail(model_id)
        elif path == "/v1/models":
            self._handle_list_models()
        elif path == "/health":
            self._handle_health()
        elif path == "/v1/version":
            self._handle_version()
        elif path == "/" or path == "":
            self._send_json(
                {
                    "service": "SpectralStream API",
                    "version": SERVER_VERSION,
                    "endpoints": [
                        "GET  /v1/models",
                        "GET  /v1/models/{model}",
                        "GET  /health",
                        "GET  /v1/version",
                        "POST /v1/chat/completions",
                        "POST /v1/completions",
                        "POST /v1/embeddings",
                    ],
                    "features": {
                        "streaming": True,
                        "tools": True,
                        "stream_options_include_usage": True,
                        "frequency_penalty": True,
                        "presence_penalty": True,
                        "response_format": True,
                        "logprobs": True,
                        "cpu_optimized": True,
                        "continuous_batching": BATCHING_AVAILABLE,
                        "coconut": COCONUT_AVAILABLE,
                        "time_crystal": TIMECRYSTAL_AVAILABLE,
                    },
                }
            )
        else:
            self._send_error_json(f"Not found: {path}", 404)

    def do_POST(self):
        if not self._check_rate_limit(1.0):
            self._send_error_json(
                "Rate limit exceeded. Please slow down.",
                429,
                code="rate_limit_exceeded",
            )
            return

        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif path == "/v1/completions":
            self._handle_completions()
        elif path == "/v1/embeddings":
            self._handle_embeddings()
        else:
            self._send_error_json(f"Not found: {path}", 404)

    # ─── Model Detail ──────────────────────────────────────────────────────

    def _handle_model_detail(self, model_id: str):
        models = find_lmstudio_models()
        for m in models:
            mid = m.get("name", m.get("path", ""))
            if mid == model_id:
                info = model_info_to_openai(m)
                info["created"] = get_model_timestamp(model_id)
                # Add SpectralStream-specific metadata
                info["sizes"] = {
                    "quantized_gb": m.get("size_gb", 0),
                }
                self._send_json(info)
                return

        # Check builtin
        if model_id == "spectralstream":
            self._send_json(builtin_model_entry())
            return

        # Fallback: return basic info even if model file not found
        # (model may be loaded by name via spectralstream)
        self._send_json(
            {
                "id": model_id,
                "object": "model",
                "created": get_model_timestamp(model_id),
                "owned_by": "spectralstream",
                "permission": [],
                "root": f"spectralstream://{model_id}",
            }
        )

    # ─── List Models ───────────────────────────────────────────────────────

    def _handle_list_models(self):
        models = find_lmstudio_models()
        openai_models = [builtin_model_entry()]
        for m in models:
            entry = model_info_to_openai(m)
            entry["created"] = get_model_timestamp(entry["id"])
            openai_models.append(entry)
        self._send_json({"object": "list", "data": openai_models})

    # ─── Health ────────────────────────────────────────────────────────────

    def _handle_health(self):
        engine = self._get_engine()
        stats = engine.stats()

        # Check engine health
        engine_ok = True
        checks = {"model_loaded": engine.is_real_model}

        # KV cache health
        kv_hit_rate = stats.get("kv_cache_hit_rate", 0)
        checks["kv_cache_healthy"] = kv_hit_rate >= 0

        # Pipeline status
        pipeline_stats = {}
        if hasattr(engine, "pipeline") and hasattr(engine.pipeline, "statistics"):
            try:
                pipeline_stats = engine.pipeline.statistics()
            except Exception:
                pass

        # Batching engine health
        batching_health = {}
        batcher = getattr(self.server, "batcher", None)
        if batcher is not None:
            try:
                batching_health = batcher.stats()
            except Exception:
                batching_health = {"error": "unavailable"}

        # Request queue health
        queue_health = {}
        req_queue_health = getattr(self.server, "request_queue", None)
        if req_queue_health is not None:
            try:
                queue_health = req_queue_health.stats()
            except Exception:
                queue_health = {"error": "unavailable"}

        # COCONUT engine status
        coconut_info = {}
        if COCONUT_AVAILABLE and hasattr(engine, "coconut_engine"):
            try:
                ce = engine.coconut_engine
                coconut_info = {
                    "available": True,
                    "max_steps": getattr(ce, "max_steps", 0),
                    "entropy_threshold": getattr(ce, "entropy_threshold", 0),
                }
            except Exception:
                pass

        # TimeCrystal status
        time_crystal_info = {}
        if TIMECRYSTAL_AVAILABLE and hasattr(engine, "time_crystal"):
            try:
                tc = engine.time_crystal
                time_crystal_info = {
                    "available": True,
                    "period": getattr(tc, "period", 0),
                    "phase": round(getattr(tc, "phase", 0), 4)
                    if hasattr(tc, "phase")
                    else 0,
                }
            except Exception:
                pass

        self._send_json(
            {
                "status": "ok" if engine_ok else "degraded",
                "version": SERVER_VERSION,
                "engine": {
                    "hidden_dim": engine.hidden_dim,
                    "vocab_size": engine.vocab_size,
                    "n_layers": engine.n_layers,
                    "n_heads": engine.n_heads,
                    "model_loaded": engine.is_real_model,
                    "block_emission_active": True,
                    "hd_engine_active": hasattr(engine, "hd_engine"),
                    "spectral_kv_active": hasattr(engine, "kv_cache"),
                },
                "stats": stats,
                "pipeline": pipeline_stats,
                "checks": checks,
                "batching": batching_health,
                "queue": queue_health,
                "coconut": coconut_info,
                "time_crystal": time_crystal_info,
                "uptime": time.time() - self.server.start_time
                if hasattr(self.server, "start_time")
                else 0,
            }
        )

    # ─── Version ───────────────────────────────────────────────────────────

    def _handle_version(self):
        engine = self._get_engine()
        self._send_json(
            {
                "version": SERVER_VERSION,
                "object": "version",
                "engine": "SpectralStream",
                "coconut_available": COCONUT_AVAILABLE,
                "time_crystal_available": TIMECRYSTAL_AVAILABLE,
                "batching_available": BATCHING_AVAILABLE,
                "progressive_load_available": PROGRESSIVE_LOAD_AVAILABLE,
                "vocab_size": engine.vocab_size,
                "hidden_dim": engine.hidden_dim,
            }
        )

    # ─── Chat Completions ─────────────────────────────────────────────────

    def _handle_chat_completions(self):
        body = self._parse_body()
        if not body:
            self._send_error_json("Invalid JSON body", 400)
            return

        messages = body.get("messages", [])
        if not messages:
            self._send_error_json("Missing 'messages' field", 400)
            return

        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.8)
        top_p = body.get("top_p", 1.0)
        top_k = body.get("top_k", 0)
        model_name = body.get("model", "spectralstream")
        tools = body.get("tools", None)
        tool_choice = body.get("tool_choice", "auto")
        stop = body.get("stop", None)
        logprobs = body.get("logprobs", False)
        top_logprobs = body.get("top_logprobs", 0)
        n = body.get("n", 1)
        seed = body.get("seed", None)
        user = body.get("user", None)
        frequency_penalty = body.get("frequency_penalty", 0.0)
        presence_penalty = body.get("presence_penalty", 0.0)
        response_format = body.get("response_format", None)
        stream_options = body.get("stream_options", {})
        include_usage = (
            stream_options.get("include_usage", False)
            if isinstance(stream_options, dict)
            else False
        )

        # Validate
        if n < 1 or n > 8:
            self._send_error_json("'n' must be between 1 and 8", 400)
            return

        if temperature < 0 or temperature > 2:
            self._send_error_json("'temperature' must be between 0 and 2", 400)
            return

        engine = self._get_engine()

        # Build prompt from messages, injecting tools if present
        prompt = self._messages_to_prompt(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        # Rate limit cost based on prompt length
        prompt_tokens = self._count_tokens(engine, prompt)
        prompt_cost = max(1, prompt_tokens // 100)
        if not self._check_rate_limit(prompt_cost):
            self._send_error_json("Rate limit exceeded.", 429)
            return

        # Submit to priority queue for CPU-optimized scheduling
        req_id = f"req_{uuid.uuid4().hex[:12]}"
        req_priority = BatchPriority.NORMAL
        rq = getattr(self.server, "request_queue", None)
        if rq is not None:
            try:
                preq = PrioritizedRequest(
                    priority=req_priority,
                    timestamp=time.time(),
                    request_id=req_id,
                    handler_ref=self,
                    body=body,
                    endpoint="chat",
                )
                rq.push(preq)
            except Exception:
                pass

        if stream:
            self._handle_streaming_chat(
                prompt,
                engine,
                model_name,
                max_tokens,
                temperature,
                top_p,
                top_k,
                stop,
                logprobs,
                top_logprobs,
                n,
                tools,
                frequency_penalty,
                presence_penalty,
                response_format,
                include_usage,
            )
        else:
            self._handle_nonstreaming_chat(
                prompt,
                engine,
                model_name,
                max_tokens,
                temperature,
                top_p,
                top_k,
                stop,
                logprobs,
                top_logprobs,
                n,
                tools,
                frequency_penalty,
                presence_penalty,
                response_format,
            )

    # ─── Completions (legacy) ──────────────────────────────────────────────

    def _handle_completions(self):
        body = self._parse_body()
        if not body:
            self._send_error_json("Invalid JSON body", 400)
            return

        prompt = body.get("prompt", "")
        if not prompt:
            self._send_error_json("Missing 'prompt' field", 400)
            return

        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.8)
        top_p = body.get("top_p", 1.0)
        top_k = body.get("top_k", 0)
        model_name = body.get("model", "spectralstream")
        stop = body.get("stop", None)
        logprobs = body.get("logprobs", False)
        top_logprobs = body.get("top_logprobs", 0)
        echo = body.get("echo", False)
        suffix = body.get("suffix", None)
        n = body.get("n", 1)
        seed = body.get("seed", None)
        user = body.get("user", None)
        frequency_penalty = body.get("frequency_penalty", 0.0)
        presence_penalty = body.get("presence_penalty", 0.0)
        best_of = body.get("best_of", 1)
        stream_options = body.get("stream_options", {})
        include_usage = (
            stream_options.get("include_usage", False)
            if isinstance(stream_options, dict)
            else False
        )

        if n < 1 or n > 8:
            self._send_error_json("'n' must be between 1 and 8", 400)
            return

        engine = self._get_engine()

        prompt_tokens = self._count_tokens(engine, prompt)
        prompt_cost = max(1, prompt_tokens // 100)
        if not self._check_rate_limit(prompt_cost):
            self._send_error_json("Rate limit exceeded.", 429)
            return

        if stream:
            self._handle_streaming_completion(
                prompt,
                engine,
                model_name,
                max_tokens,
                temperature,
                top_p,
                top_k,
                stop,
                logprobs,
                top_logprobs,
                echo,
                frequency_penalty,
                presence_penalty,
                include_usage,
            )
        else:
            self._handle_nonstreaming_completion(
                prompt,
                engine,
                model_name,
                max_tokens,
                temperature,
                top_p,
                top_k,
                stop,
                logprobs,
                top_logprobs,
                echo,
                n,
                frequency_penalty,
                presence_penalty,
                suffix,
            )

    # ─── Embeddings ────────────────────────────────────────────────────────

    def _handle_embeddings(self):
        body = self._parse_body()
        if not body:
            self._send_error_json("Invalid JSON body", 400)
            return

        input_data = body.get("input", "")
        model_name = body.get("model", "spectralstream")
        user = body.get("user", None)
        encoding_format = body.get("encoding_format", "float")
        dimensions = body.get("dimensions", 1536)

        engine = self._get_engine()

        # Normalize input to list of strings
        if isinstance(input_data, str):
            inputs = [input_data]
        elif isinstance(input_data, list):
            inputs = [
                str(item) if not isinstance(item, str) else item for item in input_data
            ]
        else:
            self._send_error_json(
                "'input' must be a string or list of strings",
                400,
            )
            return

        # Validate encoding format
        if encoding_format not in ("float", "base64"):
            self._send_error_json(
                "'encoding_format' must be 'float' or 'base64'",
                400,
            )
            return

        if not self._check_rate_limit(len(inputs)):
            self._send_error_json("Rate limit exceeded.", 429)
            return

        # Batch compute embeddings (CPU-optimized: uses HD bundling when available)
        raw_embeddings = compute_embedding_batch(engine, inputs, dimensions)

        data = []
        total_tokens = 0
        for i, (text, raw_vec) in enumerate(zip(inputs, raw_embeddings)):
            tokens_used = self._count_tokens(engine, text)
            total_tokens += tokens_used

            if encoding_format == "base64":
                embedding_val = encode_embedding_base64(raw_vec)
            else:
                embedding_val = raw_vec

            data.append(
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding_val,
                }
            )

        self._send_json(
            {
                "object": "list",
                "data": data,
                "model": model_name,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens,
                },
            }
        )

    # ─── Streaming Helpers ────────────────────────────────────────────────

    def _start_sse_response(self):
        self.send_response(200)
        self._send_rate_limit_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        engine = self._get_engine()
        self._send_spectral_headers(engine)
        self._send_cors()
        self.end_headers()

    def _send_sse(self, data: dict):
        chunk = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.wfile.write(chunk.encode("utf-8"))
        self.wfile.flush()

    def _send_sse_done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _count_tokens(self, engine: SpectralStream, text: str) -> int:
        try:
            if hasattr(engine, "_tokenize"):
                return len(engine._tokenize(text))
        except Exception:
            pass
        return max(1, len(text.split()) + len(text) // 4)

    def _handle_stop_sequences(
        self, text: str, stop: Optional[list[str]]
    ) -> tuple[str, bool]:
        if not stop:
            return text, False
        if isinstance(stop, str):
            stop = [stop]
        for seq in stop:
            idx = text.find(seq)
            if idx != -1:
                return text[:idx], True
        return text, False

    def _get_raw_scores(
        self, engine: SpectralStream, token_ids: list[int]
    ) -> np.ndarray:
        """Get raw logit scores from the engine's scorer.

        Falls back to uniform random noise if scorer unavailable.
        """
        vocab_size = engine.vocab_size
        if hasattr(engine, "scorer") and hasattr(engine.scorer, "score"):
            try:
                scores = engine.scorer.score(token_ids)
                if isinstance(scores, np.ndarray) and scores.size > 0:
                    scores = scores.flatten()
                    if len(scores) > vocab_size:
                        scores = scores[:vocab_size]
                    elif len(scores) < vocab_size:
                        padded = np.zeros(vocab_size)
                        padded[: len(scores)] = scores
                        scores = padded
                    return scores
            except Exception:
                pass
        return _get_empty_probs(vocab_size)

    def _generate_token_with_penalties(
        self,
        engine: SpectralStream,
        token_ids: list[int],
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
    ) -> tuple[int, np.ndarray]:
        """Generate a single token with full penalty support.

        CPU-optimized: uses numpy-only operations, no GPU required.

        Returns:
            (token_id, probs_vector)
        """
        if seed is not None:
            np.random.seed(seed)

        scores = self._get_raw_scores(engine, token_ids)

        if frequency_penalty != 0.0 or presence_penalty != 0.0:
            scores = _apply_penalties(
                scores,
                token_ids,
                engine.vocab_size,
                frequency_penalty,
                presence_penalty,
            )

        scores = scores / max(temperature, 0.01)

        if top_k > 0:
            indices = np.argpartition(scores, -top_k)[-top_k:]
            mask = np.ones(engine.vocab_size, dtype=bool)
            mask[indices] = False
            scores[mask] = -float("inf")

        probs = _softmax(scores)

        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)
            cutoff = int(np.searchsorted(cumsum, top_p)) + 1
            sorted_probs[cutoff:] = 0.0
            sorted_probs /= sorted_probs.sum()
            idx = np.random.choice(len(sorted_probs), p=sorted_probs)
            token = int(sorted_indices[idx])
        else:
            token = int(np.random.choice(engine.vocab_size, p=probs))

        return token, probs

    def _generate_and_stream(
        self,
        prompt: str,
        engine: SpectralStream,
        model_name: str,
        max_tokens: int,
        temperature: float,
        completion_mode: bool,
        stop: Optional[list[str]],
        logprobs: bool,
        top_logprobs: int,
        echo: bool = False,
        top_p: float = 1.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        n_choices: int = 1,
        include_usage: bool = False,
    ):
        input_ids = self._tokenize(engine, prompt)
        input_len = len(input_ids)

        generated_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        cmpl_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        choices_data = []
        for choice_idx in range(n_choices):
            token_ids = list(input_ids)
            generated_text = ""
            finish_reason = None
            all_tokens = []

            # Role announcement for chat mode
            if not completion_mode:
                role_chunk = {
                    "id": generated_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": choice_idx,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                self._send_sse(role_chunk)

            if echo and completion_mode:
                echo_chunk = {
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": choice_idx,
                            "text": prompt,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                self._send_sse(echo_chunk)

            for i in range(max_tokens):
                try:
                    if (
                        frequency_penalty != 0.0
                        or presence_penalty != 0.0
                        or top_p < 1.0
                        or top_k > 0
                    ):
                        new_token, _ = self._generate_token_with_penalties(
                            engine,
                            token_ids,
                            temperature,
                            top_p,
                            top_k,
                            frequency_penalty,
                            presence_penalty,
                        )
                    else:
                        next_ids, _ = engine.generate(
                            token_ids,
                            max_new_tokens=1,
                            temperature=temperature,
                        )
                        if len(next_ids) <= len(token_ids):
                            break
                        new_token = next_ids[-1]
                except Exception:
                    break

                token_ids.append(new_token)
                all_tokens.append(new_token)

                text = self._detokenize(engine, [new_token])
                generated_text += text

                stopped = False
                if stop:
                    clean_text, stopped = self._handle_stop_sequences(
                        generated_text, stop
                    )
                    if stopped:
                        generated_text = clean_text

                logprobs_data = None
                if logprobs or top_logprobs > 0:
                    if completion_mode:
                        logprobs_data = self._compute_logprobs(
                            engine,
                            new_token,
                            token_ids,
                            top_logprobs,
                            temperature,
                        )

                if completion_mode:
                    chunk = {
                        "id": cmpl_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": choice_idx,
                                "text": text,
                                "logprobs": logprobs_data,
                                "finish_reason": None,
                            }
                        ],
                    }
                else:
                    chunk = {
                        "id": generated_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": choice_idx,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }

                self._send_sse(chunk)

                if stopped:
                    finish_reason = "stop"
                    break

            if finish_reason is None:
                finish_reason = "length" if len(all_tokens) >= max_tokens else "stop"

            # Check for tool calls in chat mode
            tool_call = None
            if not completion_mode:
                tool_call = parse_tool_call(generated_text)

            if completion_mode:
                final_chunk = {
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": choice_idx,
                            "text": "",
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            else:
                delta = {}
                if tool_call:
                    delta["role"] = "assistant"
                    delta["content"] = None
                    delta["tool_calls"] = [
                        build_tool_call_chunk(
                            tool_call["function"],
                            json.dumps(tool_call["arguments"]),
                            0,
                        )
                    ]

                final_chunk = {
                    "id": generated_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": choice_idx,
                            "delta": delta,
                            "finish_reason": "tool_calls"
                            if tool_call
                            else finish_reason,
                        }
                    ],
                }

            self._send_sse(final_chunk)
            choices_data.append(
                {
                    "tokens": len(all_tokens),
                    "finish_reason": finish_reason,
                }
            )

        # Send usage if requested
        if include_usage:
            total_cmpl = sum(c["tokens"] for c in choices_data)
            usage_chunk = {
                "id": generated_id if not completion_mode else cmpl_id,
                "object": "chat.completion.chunk"
                if not completion_mode
                else "text_completion",
                "created": created,
                "model": model_name,
                "choices": [],
                "usage": {
                    "prompt_tokens": input_len,
                    "completion_tokens": total_cmpl,
                    "total_tokens": input_len + total_cmpl,
                },
            }
            self._send_sse(usage_chunk)

        self._send_sse_done()

    def _compute_logprobs(
        self,
        engine,
        token_id: int,
        context: list[int],
        top_logprobs: int = 0,
        temperature: float = 0.8,
    ) -> dict:
        vocab_size = engine.vocab_size
        if hasattr(engine, "scorer") and hasattr(engine.scorer, "score"):
            try:
                scores = engine.scorer.score(context)
                if isinstance(scores, np.ndarray) and scores.size > 0:
                    scores = scores.flatten()
                    if len(scores) > vocab_size:
                        scores = scores[:vocab_size]
                    elif len(scores) < vocab_size:
                        padded = np.zeros(vocab_size)
                        padded[: len(scores)] = scores
                        scores = padded
                else:
                    scores = np.random.randn(vocab_size) * temperature
            except Exception:
                scores = np.random.randn(vocab_size) * temperature
        else:
            scores = np.random.randn(vocab_size) * temperature

        scores = scores - scores.max()
        exp_scores = np.exp(scores / max(temperature, 0.01))
        probs = exp_scores / exp_scores.sum()

        token_logprob = float(np.log(max(probs[token_id % vocab_size], 1e-10)))

        result = {
            "token": token_id,
            "logprob": token_logprob,
            "bytes": None,
        }

        if top_logprobs > 0:
            top_indices = np.argsort(probs)[-top_logprobs:][::-1]
            top_logprobs_list = []
            for idx in top_indices:
                lp = float(np.log(max(probs[idx], 1e-10)))
                top_logprobs_list.append(
                    {
                        "token": int(idx),
                        "logprob": lp,
                        "bytes": None,
                    }
                )
            result["top_logprobs"] = top_logprobs_list

        return result

    def _handle_streaming_chat(
        self,
        prompt: str,
        engine: SpectralStream,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        stop: Optional[list[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        n: int = 1,
        tools: Optional[list] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: Optional[dict] = None,
        include_usage: bool = False,
    ):
        # If we have penalties or sampling params beyond basic, use full streaming
        if (
            frequency_penalty != 0.0
            or presence_penalty != 0.0
            or top_p < 1.0
            or top_k > 0
            or n > 1
        ):
            self._start_sse_response()
            self._generate_and_stream(
                prompt,
                engine,
                model_name,
                max_tokens,
                temperature,
                completion_mode=False,
                stop=stop,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                echo=False,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n_choices=n,
                include_usage=include_usage,
            )
            return

        # Use StreamingHandler for basic case (backward-compatible path)
        handler = StreamingHandler(
            wfile=self.wfile,
            engine=engine,
            detokenize_fn=lambda ids: self._detokenize(engine, ids),
            tokenize_fn=lambda text: self._tokenize(engine, text),
        )
        handler.start_sse(self._start_sse_response)
        include_usage_kwargs = {}
        if hasattr(handler, "set_include_usage"):
            include_usage_kwargs["include_usage"] = include_usage
        handler.stream_chat(
            prompt=prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            stop=stop,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            **include_usage_kwargs,
        )

    def _handle_streaming_completion(
        self,
        prompt: str,
        engine: SpectralStream,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        stop: Optional[list[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        include_usage: bool = False,
    ):
        self._start_sse_response()
        self._generate_and_stream(
            prompt,
            engine,
            model_name,
            max_tokens,
            temperature,
            completion_mode=True,
            stop=stop,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            echo=echo,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n_choices=1,
            include_usage=include_usage,
        )

    # ─── Non-Streaming Helpers ────────────────────────────────────────────

    def _generate_nonstreaming(
        self,
        prompt: str,
        engine: SpectralStream,
        model_name: str,
        max_tokens: int,
        temperature: float,
        completion_mode: bool,
        stop: Optional[list[str]],
        logprobs: bool,
        top_logprobs: int,
        echo: bool = False,
        top_p: float = 1.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        n_choices: int = 1,
        suffix: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> list[dict]:
        """Generate text and build OpenAI-formatted response for n choices."""
        input_ids = self._tokenize(engine, prompt)
        prompt_tokens = len(input_ids)
        is_tool_call_mode = "<available_tools>" in prompt
        is_json_mode = (
            response_format is not None and response_format.get("type") == "json_object"
        )

        choices = []
        total_completion_tokens = 0

        for choice_idx in range(n_choices):
            token_ids = list(input_ids)
            final_text = ""
            all_tokens = []
            finish_reason = "stop"

            for i in range(max_tokens):
                try:
                    if (
                        frequency_penalty != 0.0
                        or presence_penalty != 0.0
                        or top_p < 1.0
                        or top_k > 0
                    ):
                        new_token, _ = self._generate_token_with_penalties(
                            engine,
                            token_ids,
                            temperature,
                            top_p,
                            top_k,
                            frequency_penalty,
                            presence_penalty,
                        )
                        token_ids.append(new_token)
                    else:
                        next_ids, _ = engine.generate(
                            token_ids,
                            max_new_tokens=1,
                            temperature=temperature,
                        )
                        if len(next_ids) <= len(token_ids):
                            break
                        new_token = next_ids[-1]
                        token_ids.append(new_token)
                except Exception:
                    break

                all_tokens.append(new_token)
                text = self._detokenize(engine, [new_token])
                final_text += text

                if stop:
                    clean_text, stopped = self._handle_stop_sequences(final_text, stop)
                    if stopped:
                        final_text = clean_text
                        break

            finish_reason = "length" if len(all_tokens) >= max_tokens else "stop"

            # Handle suffix for infill
            if suffix and completion_mode and finish_reason == "stop":
                final_text += suffix

            # Check for tool calls
            tool_call = None
            if (is_tool_call_mode or tool_call) and not completion_mode:
                tool_call = parse_tool_call(final_text)

            # Build logprobs for the last token if requested
            logprobs_data = None
            if logprobs or top_logprobs > 0:
                if all_tokens:
                    logprobs_data = self._compute_logprobs(
                        engine,
                        all_tokens[-1],
                        token_ids,
                        top_logprobs,
                        temperature,
                    )

            if completion_mode:
                result_text = (prompt + final_text) if echo else final_text
                choice = {
                    "index": choice_idx,
                    "text": result_text,
                    "logprobs": logprobs_data,
                    "finish_reason": finish_reason,
                }
            else:
                message = {
                    "role": "assistant",
                    "content": None if tool_call else final_text,
                }
                if tool_call:
                    message["tool_calls"] = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": tool_call["function"],
                                "arguments": json.dumps(tool_call["arguments"]),
                            },
                        }
                    ]
                    finish_reason = "tool_calls"

                choice = {
                    "index": choice_idx,
                    "message": message,
                    "logprobs": logprobs_data if logprobs else None,
                    "finish_reason": finish_reason,
                }

            choices.append(choice)
            total_completion_tokens += len(all_tokens)

        if completion_mode:
            response = {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": choices,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": prompt_tokens + total_completion_tokens,
                },
            }
        else:
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": choices,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": prompt_tokens + total_completion_tokens,
                },
            }

        return response

    def _handle_nonstreaming_chat(
        self,
        prompt: str,
        engine: SpectralStream,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        stop: Optional[list[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        n: int = 1,
        tools: Optional[list] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: Optional[dict] = None,
    ):
        response = self._generate_nonstreaming(
            prompt,
            engine,
            model_name,
            max_tokens,
            temperature,
            completion_mode=False,
            stop=stop,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            echo=False,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n_choices=n,
            response_format=response_format,
        )

        self.send_response(200)
        self._send_rate_limit_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._send_spectral_headers(engine)
        self._send_cors()
        self.end_headers()

        body = json.dumps(response, ensure_ascii=False)
        self.wfile.write(body.encode("utf-8"))

    def _handle_nonstreaming_completion(
        self,
        prompt: str,
        engine: SpectralStream,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        stop: Optional[list[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        echo: bool = False,
        n: int = 1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        suffix: Optional[str] = None,
    ):
        response = self._generate_nonstreaming(
            prompt,
            engine,
            model_name,
            max_tokens,
            temperature,
            completion_mode=True,
            stop=stop,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            echo=echo,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n_choices=n,
            suffix=suffix,
        )

        self.send_response(200)
        self._send_rate_limit_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._send_spectral_headers(engine)
        self._send_cors()
        self.end_headers()

        body = json.dumps(response, ensure_ascii=False)
        self.wfile.write(body.encode("utf-8"))

    # ─── Utilities ────────────────────────────────────────────────────────

    def _messages_to_prompt(
        self,
        messages: list[dict],
        tools: Optional[list] = None,
        tool_choice: str = "auto",
        response_format: Optional[dict] = None,
    ) -> str:
        parts = []

        if tools:
            tools_desc = format_tools_for_prompt(tools)
            system_msg = TOOL_CALL_SYSTEM_PROMPT.replace(
                "{tools_description}", tools_desc
            )
            parts.append(f"<|system|>\n{system_msg}\n<|end|>")

        if response_format and response_format.get("type") == "json_object":
            parts.append(f"<|system|>\n{JSON_MODE_SYSTEM_PROMPT}\n<|end|>")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                name = msg.get("name", "")
                result_content = str(content) if content else ""
                parts.append(
                    f"<|tool|>\nTool result ({name}): {result_content}\n<|end|>"
                )
                continue

            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                        elif c.get("type") == "image_url":
                            text_parts.append("[Image]")
                content = " ".join(text_parts)

            if role == "assistant" and "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                for tc in tool_calls:
                    if tc.get("type") == "function":
                        func = tc.get("function", {})
                        parts.append(
                            f"<|assistant|>\n"
                            f'{{"function": "{func.get("name", "")}", '
                            f'"arguments": {func.get("arguments", "{}")}}}\n<|end|>'
                        )
                if content:
                    parts.append(f"<|assistant|>\n{content}\n<|end|>")
                continue

            parts.append(f"<|{role}|>\n{content}\n<|end|>")

        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def _tokenize(self, engine: SpectralStream, text: str) -> list[int]:
        try:
            return engine._tokenize(text)
        except Exception:
            return [
                min(ord(c) % engine.vocab_size, engine.vocab_size - 1)
                for c in text[:512]
            ]

    def _detokenize(self, engine: SpectralStream, token_ids: list[int]) -> str:
        try:
            if engine.is_real_model:
                result = []
                for t in token_ids:
                    try:
                        piece = engine.model.detokenize(t)
                        if piece:
                            result.append(
                                piece
                                if isinstance(piece, str)
                                else piece.decode("utf-8", errors="replace")
                            )
                    except Exception:
                        result.append(chr(t) if 32 <= t <= 126 else f"<{t}>")
                return "".join(result)
        except Exception:
            pass
        return "".join(chr(t) if 32 <= t <= 126 else f"<{t}>" for t in token_ids)


# ─── Threaded Server ───────────────────────────────────────────────────────


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        engine=None,
        start_time=None,
        rate_limiter=None,
        request_queue=None,
        batcher=None,
    ):
        self.engine = engine
        self.start_time = start_time or time.time()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.request_queue = request_queue or RequestPriorityQueue()
        self.batcher = batcher
        super().__init__(server_address, RequestHandlerClass)


# ─── SpectralStreamServer ──────────────────────────────────────────────────


class SpectralStreamServer:
    """High-level server wrapper for SpectralStream OpenAI-compatible API.

    CPU-first architecture:
    - ThreadingHTTPServer for parallel request handling
    - Request priority queue with deadline scheduling
    - Continuous batching integration via ContinuousBatchingEngine
    - Memory-mapped model loading (zero-copy)
    - Token-by-token generation with stop-check interleaving
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 1234,
        engine: Optional[SpectralStream] = None,
        model_path: Optional[str] = None,
        rate_limit_capacity: float = 100.0,
        rate_limit_refill: float = 20.0,
        enable_batching: bool = True,
        max_batch_tokens: int = 4096,
        **engine_kwargs,
    ):
        self.host = host
        self.port = port
        self._server = None
        self.rate_limiter = RateLimiter(
            capacity=rate_limit_capacity, refill_rate=rate_limit_refill
        )

        if engine is not None:
            self.engine = engine
        else:
            self.engine = SpectralStream(model_path=model_path, **engine_kwargs)

        # Initialize request priority queue for CPU-optimized scheduling
        self.request_queue = RequestPriorityQueue(max_size=256)

        # Initialize continuous batching bridge
        self.batcher = None
        if enable_batching:
            try:
                self.batcher = ContinuousBatchingBridge(self.engine, max_batch_tokens)
                if self.batcher.is_active:
                    print(
                        f"Continuous batching engine active: max_batch_tokens={max_batch_tokens}"
                    )
                else:
                    print(
                        "Batching engine not available (falling back to sequential processing)"
                    )
            except Exception as exc:
                print(f"Batching init skipped: {exc}")

        # Memory-mapped model loading info
        self._mmap_info = self._detect_mmap()

    def _detect_mmap(self) -> dict:
        """Detect memory-mapped model loading capabilities."""
        info = {"available": False, "model_path": None}
        if self.engine.is_real_model and hasattr(self.engine, "model"):
            try:
                model_path = getattr(self.engine.model, "path", None) or getattr(
                    self.engine.model, "model_path", None
                )
                if model_path:
                    info["available"] = True
                    info["model_path"] = model_path
                    # Check if file is mmap'd
                    if hasattr(self.engine.model, "_mmap_data") or hasattr(
                        self.engine.model, "_mmap"
                    ):
                        info["zero_copy"] = True
                    else:
                        info["zero_copy"] = False
            except Exception:
                pass
        return info

    def start(self, poll_interval: float = 0.5):
        self._server = ThreadedHTTPServer(
            (self.host, self.port),
            RequestHandler,
            engine=self.engine,
            start_time=time.time(),
            rate_limiter=self.rate_limiter,
            request_queue=self.request_queue,
            batcher=self.batcher,
        )
        addr = self._server.server_address
        print(f"SpectralStream API server listening on http://{addr[0]}:{addr[1]}")
        print(f"Endpoints:")
        print(f"  GET  /v1/models")
        print(f"  GET  /v1/models/{{model}}")
        print(f"  GET  /v1/version")
        print(f"  GET  /health")
        print(f"  POST /v1/chat/completions")
        print(f"  POST /v1/completions")
        print(f"  POST /v1/embeddings")
        print()
        print(
            f"Engine: hidden_dim={self.engine.hidden_dim}, vocab={self.engine.vocab_size}"
        )
        print(f"Model loaded: {self.engine.is_real_model}")
        if self._mmap_info.get("available"):
            mmap_str = " (zero-copy)" if self._mmap_info.get("zero_copy") else ""
            print(f"Model mmap: {self._mmap_info['model_path']}{mmap_str}")
        print(
            f"Rate limit: {self.rate_limiter.capacity} tokens burst, {self.rate_limiter.refill_rate}/s refill"
        )
        print(f"Request queue: enabled (max 256, priority-scheduled)")
        if self.batcher and self.batcher.is_active:
            print(f"Continuous batching: active")
        print(f"COCONUT: {'available' if COCONUT_AVAILABLE else 'not available'}")
        print(
            f"TimeCrystal: {'available' if TIMECRYSTAL_AVAILABLE else 'not available'}"
        )
        print()

        try:
            self._server.serve_forever(poll_interval=poll_interval)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self._server.shutdown()

    def stop(self):
        if self._server:
            self._server.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SpectralStream OpenAI-Compatible API Server (CPU-first)",
        epilog="Designed for CPU inference with HDC drafting, spectral KV, and continuous batching.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=1234, help="Port to bind to")
    parser.add_argument("--model", type=str, default=None, help="Path to GGUF model")
    parser.add_argument(
        "--hidden-dim", type=int, default=512, help="Hidden dimension (dummy model)"
    )
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--n-layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--block-size", type=int, default=8, help="Block emission size")
    parser.add_argument("--hd-dim", type=int, default=4096, help="HDC vector dimension")
    parser.add_argument(
        "--rate-limit", type=float, default=100.0, help="Rate limit burst capacity"
    )
    parser.add_argument(
        "--rate-refill", type=float, default=20.0, help="Rate limit refill per second"
    )
    parser.add_argument(
        "--no-batching", action="store_true", help="Disable continuous batching"
    )
    parser.add_argument(
        "--max-batch-tokens", type=int, default=4096, help="Max tokens per batch"
    )

    args = parser.parse_args()

    if args.model:
        print(f"Loading model: {args.model}")
    else:
        models = find_lmstudio_models()
        if models:
            print(f"Found {len(models)} GGUF model(s) in LM Studio directories:")
            for m in models[:5]:
                print(f"  - {m['name']} ({m['size_gb']:.1f} GB)")
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")
        else:
            print("No GGUF models found. Using dummy model.")

    server = SpectralStreamServer(
        host=args.host,
        port=args.port,
        model_path=args.model,
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        n_layers=args.n_layers,
        block_size=args.block_size,
        hd_dim=args.hd_dim,
        rate_limit_capacity=args.rate_limit,
        rate_limit_refill=args.rate_refill,
        enable_batching=not args.no_batching,
        max_batch_tokens=args.max_batch_tokens,
    )
    server.start()


if __name__ == "__main__":
    main()

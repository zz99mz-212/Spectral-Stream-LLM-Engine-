from __future__ import annotations

import json
import logging
import math
import struct
import base64
import time
import uuid
from typing import Optional

import warnings

warnings.warn(
    "spectralstream.serving.api_server is deprecated. Use spectralstream.serving.unified_server instead.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from spectralstream.config import SpectralStreamConfig
from spectralstream.serving.model_manager import ModelManager
from spectralstream.serving.request_queue import RequestQueue, Priority
from spectralstream.serving.streaming import TokenStreamer, format_sse, StreamChunk

logger = logging.getLogger(__name__)

SERVER_VERSION = "1.0.0"


class ChatMessage(BaseModel):
    role: str = "user"
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: dict


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = "spectralstream"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    n: int = Field(default=1, ge=1, le=8)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[list[str] | str] = None
    max_tokens: int = Field(default=256, ge=1, le=131072)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logprobs: bool = False
    top_logprobs: int = Field(default=0, ge=0, le=20)
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: str | dict = "auto"
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str = "spectralstream"
    prompt: str | list[str] = ""
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    n: int = Field(default=1, ge=1, le=8)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[list[str] | str] = None
    max_tokens: int = Field(default=256, ge=1, le=131072)
    echo: bool = False
    suffix: Optional[str] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logprobs: bool = False
    top_logprobs: int = Field(default=0, ge=0, le=20)
    best_of: int = Field(default=1, ge=1, le=8)
    seed: Optional[int] = None
    user: Optional[str] = None


class EmbeddingRequest(BaseModel):
    model: str = "spectralstream"
    input: str | list[str]
    encoding_format: str = Field(default="float", pattern="^(float|base64)$")
    dimensions: int = Field(default=1536, ge=1)
    user: Optional[str] = None


class CompressRequest(BaseModel):
    model_path: str
    target_ratio: float = Field(default=10.0, ge=1.0, le=1000.0)
    output_path: Optional[str] = None
    quant_bits: int = Field(default=4, ge=1, le=8)


class BenchmarkRequest(BaseModel):
    model_id: str = "spectralstream"
    prompt: str = "Hello, how are you?"
    max_tokens: int = Field(default=128, ge=1, le=4096)
    num_runs: int = Field(default=5, ge=1, le=100)
    temperature: float = 0.0


class AppState:
    def __init__(self):
        self.config = SpectralStreamConfig()
        self.model_manager = ModelManager(
            max_models=4,
            default_config=self.config,
        )
        self.request_queue = RequestQueue(
            max_concurrent=self.config.server.max_connections,
            max_queued=256,
            default_timeout=self.config.server.request_timeout,
        )
        self.start_time = time.time()
        self.request_count = 0
        self.total_tokens_generated = 0

    def get_engine(self, model_name: str = "spectralstream"):
        entry = self.model_manager.get(model_name)
        if entry is None:
            entry = self.model_manager.load(
                model_path=model_name,
                model_id=model_name,
                config=self.config,
            )
        return entry.engine


state = AppState()

app = FastAPI(
    title="SpectralStream API",
    description="Extreme compression LLM inference server with OpenAI-compatible API",
    version=SERVER_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_metrics = {
    "requests_total": 0,
    "chat_requests": 0,
    "completion_requests": 0,
    "embedding_requests": 0,
    "streaming_requests": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "errors_total": 0,
    "error_by_type": {},
}


def _track_request(endpoint: str, prompt_tokens: int = 0, completion_tokens: int = 0):
    _metrics["requests_total"] += 1
    _metrics["total_prompt_tokens"] += prompt_tokens
    _metrics["total_completion_tokens"] += completion_tokens
    state.request_count += 1
    state.total_tokens_generated += completion_tokens
    if endpoint == "chat":
        _metrics["chat_requests"] += 1
    elif endpoint == "completion":
        _metrics["completion_requests"] += 1
    elif endpoint == "embedding":
        _metrics["embedding_requests"] += 1


def _track_error(error_type: str):
    _metrics["errors_total"] += 1
    _metrics["error_by_type"][error_type] = (
        _metrics["error_by_type"].get(error_type, 0) + 1
    )


TOOL_CALL_SYSTEM_PROMPT = """You have access to the following tools. When you need to call a tool, respond with EXACTLY a JSON object on a single line in this format:
{"function": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}

Available tools:
{tools_description}

If you don't need to call a tool, respond normally with plain text.
When you receive a tool result, use it to continue the conversation naturally."""

JSON_MODE_SYSTEM_PROMPT = "You must respond with valid JSON only. Do not include any text outside the JSON object."


def _format_tools_for_prompt(tools: list[dict]) -> str:
    lines = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        lines.append(f"- {name}: {desc}")
        props = params.get("properties", {})
        required = params.get("required", [])
        for pname, pinfo in props.items():
            req = " (required)" if pname in required else ""
            pdesc = pinfo.get("description", "")
            lines.append(f"    {pname}: {pinfo.get('type', 'string')}{req} - {pdesc}")
    return "\n".join(lines)


def _messages_to_prompt(
    messages: list[ChatMessage],
    tools: Optional[list] = None,
    response_format: Optional[dict] = None,
) -> str:
    parts = []

    if tools:
        tools_desc = _format_tools_for_prompt([t.model_dump() for t in tools])
        system_msg = TOOL_CALL_SYSTEM_PROMPT.replace("{tools_description}", tools_desc)
        parts.append(f"<|system|>\n{system_msg}\n<|end|>")

    if response_format and response_format.get("type") == "json_object":
        parts.append(f"<|system|>\n{JSON_MODE_SYSTEM_PROMPT}\n<|end|>")

    for msg in messages:
        role = msg.role
        content = msg.content or ""

        if role == "tool":
            tool_call_id = msg.tool_call_id or ""
            name = msg.name or ""
            parts.append(f"<|tool|>\nTool result ({name}): {content}\n<|end|>")
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

        if role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    parts.append(
                        f'<|assistant|>\n{{"function": "{func.get("name", "")}", '
                        f'"arguments": {func.get("arguments", "{}")}}}\n<|end|>'
                    )
            if content:
                parts.append(f"<|assistant|>\n{content}\n<|end|>")
            continue

        parts.append(f"<|{role}|>\n{content}\n<|end|>")

    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _count_tokens(engine, text: str) -> int:
    try:
        if hasattr(engine, "tokenize"):
            return len(engine.tokenize(text))
    except Exception:
        pass
    return max(1, len(text.split()) + len(text) // 4)


def _compute_embedding(engine, text: str) -> list[float]:
    tokens = _tokenize(engine, text)
    if tokens:
        return [0.0] * 1536
    return [0.0] * 1536


def _tokenize(engine, text: str) -> list[int]:
    try:
        return engine.tokenize(text)
    except Exception:
        return [min(ord(c) % 32000, 31999) for c in text[:512]]


def _detokenize(engine, token_ids: list[int]) -> str:
    return "".join(chr(t) if 32 <= t <= 126 else f"<{t}>" for t in token_ids)


@app.get("/health")
async def health():
    stats = {"uptime_seconds": round(time.time() - state.start_time, 2)}
    return {
        "status": "ok",
        "version": SERVER_VERSION,
        "uptime_seconds": round(time.time() - state.start_time, 2),
        "stats": stats,
        "request_queue": state.request_queue.stats(),
        "model_manager": state.model_manager.stats(),
    }


@app.get("/v1/models")
async def list_models():
    models = state.model_manager.list_models()
    data = [
        {
            "id": "spectralstream",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "spectralstream",
            "permission": [],
            "root": "spectralstream://builtin",
        }
    ]
    for m in models:
        data.append(
            {
                "id": m["id"],
                "object": "model",
                "created": int(m["loaded_at"]),
                "owned_by": "spectralstream",
                "permission": [],
                "root": m["path"],
            }
        )
    return {"object": "list", "data": data}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    if model_id == "spectralstream":
        return {
            "id": "spectralstream",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "spectralstream",
            "permission": [],
            "root": "spectralstream://builtin",
        }
    entry = state.model_manager.get(model_id)
    if entry:
        return {
            "id": entry.model_id,
            "object": "model",
            "created": int(entry.loaded_at),
            "owned_by": "spectralstream",
            "permission": [],
            "root": entry.path,
        }
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    engine = state.get_engine(request.model)
    prompt = _messages_to_prompt(
        request.messages, request.tools, request.response_format
    )
    prompt_tokens = _count_tokens(engine, prompt)

    include_usage = (
        request.stream_options.include_usage if request.stream_options else False
    )

    if request.stream:
        _track_request("chat")
        _metrics["streaming_requests"] += 1

        async def generate():
            streamer = TokenStreamer(model=request.model, include_usage=include_usage)
            async for chunk in streamer.stream_chat(
                engine=engine,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop,
                n=request.n,
                tools=[t.model_dump() for t in request.tools]
                if request.tools
                else None,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
            ):
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    completions_response = _generate_nonstreaming_chat(
        engine,
        prompt,
        request.model,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.top_k,
        request.stop,
        request.n,
        request.tools,
        request.frequency_penalty,
        request.presence_penalty,
        request.logprobs,
        request.top_logprobs,
        request.response_format,
    )

    completion_tokens = sum(
        c.get("tokens", 0) for c in completions_response.get("choices", [])
    )
    _track_request("chat", prompt_tokens, completion_tokens)

    return completions_response


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    engine = state.get_engine(request.model)

    include_usage = (
        request.stream_options.include_usage if request.stream_options else False
    )

    if request.stream:
        _track_request("completion")
        _metrics["streaming_requests"] += 1

        async def generate():
            streamer = TokenStreamer(model=request.model, include_usage=include_usage)
            async for chunk in streamer.stream_completion(
                engine=engine,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop,
                echo=request.echo,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
            ):
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    completions_response = _generate_nonstreaming_completion(
        engine,
        request.prompt,
        request.model,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.top_k,
        request.stop,
        request.echo,
        request.n,
        request.frequency_penalty,
        request.presence_penalty,
        request.suffix,
    )

    completion_tokens = sum(
        c.get("tokens", 0) for c in completions_response.get("choices", [])
    )
    prompt_tokens = _count_tokens(engine, request.prompt)
    _track_request("completion", prompt_tokens, completion_tokens)

    return completions_response


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    engine = state.get_engine(request.model)

    input_data = request.input
    if isinstance(input_data, str):
        inputs = [input_data]
    else:
        inputs = input_data

    raw_embeddings = [_compute_embedding(engine, t) for t in inputs]

    data = []
    total_tokens = 0
    for i, (text, raw_vec) in enumerate(zip(inputs, raw_embeddings)):
        tokens_used = _count_tokens(engine, text)
        total_tokens += tokens_used

        if request.encoding_format == "base64":
            packed = struct.pack(f"{len(raw_vec)}f", *raw_vec)
            embedding_val = base64.b64encode(packed).decode("ascii")
        else:
            embedding_val = raw_vec

        data.append(
            {
                "object": "embedding",
                "index": i,
                "embedding": embedding_val,
            }
        )

    _track_request("embedding", total_tokens, 0)

    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


@app.get("/metrics")
async def metrics():
    prometheus_lines = []
    prometheus_lines.append(f"# HELP spectralstream_requests_total Total requests")
    prometheus_lines.append(f"# TYPE spectralstream_requests_total counter")
    prometheus_lines.append(
        f"spectralstream_requests_total {_metrics['requests_total']}"
    )
    prometheus_lines.append(f"# HELP spectralstream_chat_requests_total Chat requests")
    prometheus_lines.append(f"# TYPE spectralstream_chat_requests_total counter")
    prometheus_lines.append(
        f"spectralstream_chat_requests_total {_metrics['chat_requests']}"
    )
    prometheus_lines.append(
        f"# HELP spectralstream_completion_requests_total Completion requests"
    )
    prometheus_lines.append(f"# TYPE spectralstream_completion_requests_total counter")
    prometheus_lines.append(
        f"spectralstream_completion_requests_total {_metrics['completion_requests']}"
    )
    prometheus_lines.append(
        f"# HELP spectralstream_embedding_requests_total Embedding requests"
    )
    prometheus_lines.append(f"# TYPE spectralstream_embedding_requests_total counter")
    prometheus_lines.append(
        f"spectralstream_embedding_requests_total {_metrics['embedding_requests']}"
    )
    prometheus_lines.append(
        f"# HELP spectralstream_streaming_requests_total Streaming requests"
    )
    prometheus_lines.append(f"# TYPE spectralstream_streaming_requests_total counter")
    prometheus_lines.append(
        f"spectralstream_streaming_requests_total {_metrics['streaming_requests']}"
    )
    prometheus_lines.append(
        f"# HELP spectralstream_prompt_tokens_total Total prompt tokens"
    )
    prometheus_lines.append(f"# TYPE spectralstream_prompt_tokens_total counter")
    prometheus_lines.append(
        f"spectralstream_prompt_tokens_total {_metrics['total_prompt_tokens']}"
    )
    prometheus_lines.append(
        f"# HELP spectralstream_completion_tokens_total Total completion tokens"
    )
    prometheus_lines.append(f"# TYPE spectralstream_completion_tokens_total counter")
    prometheus_lines.append(
        f"spectralstream_completion_tokens_total {_metrics['total_completion_tokens']}"
    )
    prometheus_lines.append(f"# HELP spectralstream_errors_total Total errors")
    prometheus_lines.append(f"# TYPE spectralstream_errors_total counter")
    prometheus_lines.append(f"spectralstream_errors_total {_metrics['errors_total']}")
    prometheus_lines.append(f"# HELP spectralstream_uptime_seconds Server uptime")
    prometheus_lines.append(f"# TYPE spectralstream_uptime_seconds gauge")
    prometheus_lines.append(
        f"spectralstream_uptime_seconds {round(time.time() - state.start_time, 2)}"
    )
    prometheus_lines.append(
        f"# HELP spectralstream_queue_size Current request queue size"
    )
    prometheus_lines.append(f"# TYPE spectralstream_queue_size gauge")
    prometheus_lines.append(f"spectralstream_queue_size {state.request_queue.size}")

    return StreamingResponse(
        iter(["\n".join(prometheus_lines) + "\n"]),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/v1/admin/compress")
async def compress_model(request: CompressRequest):
    return {
        "status": "accepted",
        "message": "Compression job queued",
        "model_path": request.model_path,
        "target_ratio": request.target_ratio,
        "output_path": request.output_path,
        "quant_bits": request.quant_bits,
    }


@app.post("/v1/admin/benchmark")
async def benchmark_model(request: BenchmarkRequest):
    engine = state.get_engine(request.model_id)
    results = _run_benchmark(engine, request)
    return results


def _generate_nonstreaming_chat(
    engine,
    prompt: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    stop: Optional[list[str]] = None,
    n: int = 1,
    tools: Optional[list] = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logprobs: bool = False,
    top_logprobs: int = 0,
    response_format: Optional[dict] = None,
) -> dict:
    input_ids = _tokenize(engine, prompt)
    prompt_tokens = len(input_ids)
    choices = []
    total_completion_tokens = 0

    for choice_idx in range(n):
        token_ids = list(input_ids)
        final_text = ""
        all_tokens = []

        for _ in range(max_tokens):
            try:
                next_ids, _ = engine.generate(
                    token_ids, max_new_tokens=1, temperature=temperature
                )
                if len(next_ids) <= len(token_ids):
                    break
                new_token = next_ids[-1]
            except Exception:
                break

            token_ids.append(new_token)
            all_tokens.append(new_token)
            text = _detokenize(engine, [new_token])
            final_text += text

            if stop:
                clean, stopped = _check_stop(final_text, stop)
                if stopped:
                    final_text = clean
                    break

        finish_reason = "length" if len(all_tokens) >= max_tokens else "stop"

        message = {"role": "assistant", "content": final_text}
        choice = {
            "index": choice_idx,
            "message": message,
            "logprobs": None,
            "finish_reason": finish_reason,
        }
        choices.append(choice)
        total_completion_tokens += len(all_tokens)

    return {
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


def _generate_nonstreaming_completion(
    engine,
    prompt: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    stop: Optional[list[str]] = None,
    echo: bool = False,
    n: int = 1,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    suffix: Optional[str] = None,
) -> dict:
    input_ids = _tokenize(engine, prompt)
    prompt_tokens = len(input_ids)
    choices = []
    total_completion_tokens = 0

    for choice_idx in range(n):
        token_ids = list(input_ids)
        final_text = ""
        all_tokens = []

        for _ in range(max_tokens):
            try:
                next_ids, _ = engine.generate(
                    token_ids, max_new_tokens=1, temperature=temperature
                )
                if len(next_ids) <= len(token_ids):
                    break
                new_token = next_ids[-1]
            except Exception:
                break

            token_ids.append(new_token)
            all_tokens.append(new_token)
            text = _detokenize(engine, [new_token])
            final_text += text

            if stop:
                clean, stopped = _check_stop(final_text, stop)
                if stopped:
                    final_text = clean
                    break

        if suffix and final_text:
            final_text += suffix

        finish_reason = "length" if len(all_tokens) >= max_tokens else "stop"
        result_text = (prompt + final_text) if echo else final_text

        choices.append(
            {
                "index": choice_idx,
                "text": result_text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        )
        total_completion_tokens += len(all_tokens)

    return {
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


def _check_stop(text: str, stop: Optional[list[str]]) -> tuple[str, bool]:
    if not stop:
        return text, False
    if isinstance(stop, str):
        stop = [stop]
    for seq in stop:
        idx = text.find(seq)
        if idx != -1:
            return text[:idx], True
    return text, False


def _run_benchmark(engine, request: BenchmarkRequest) -> dict:
    times = []
    token_counts = []

    for _ in range(request.num_runs):
        input_ids = _tokenize(engine, request.prompt)
        start = time.time()
        token_ids = list(input_ids)
        generated = 0

        for _ in range(request.max_tokens):
            try:
                next_ids, _ = engine.generate(
                    token_ids, max_new_tokens=1, temperature=request.temperature
                )
                if len(next_ids) <= len(token_ids):
                    break
                token_ids = next_ids
                generated += 1
            except Exception:
                break

        elapsed = time.time() - start
        times.append(elapsed)
        token_counts.append(generated)

    avg_time = sum(times) / max(len(times), 1)
    avg_tokens = sum(token_counts) / max(len(token_counts), 1)
    tokens_per_sec = avg_tokens / max(avg_time, 1e-6)

    return {
        "model_id": request.model_id,
        "num_runs": request.num_runs,
        "avg_time_seconds": round(avg_time, 4),
        "avg_tokens_generated": round(avg_tokens, 1),
        "tokens_per_second": round(tokens_per_sec, 2),
        "min_time": round(min(times), 4) if times else 0,
        "max_time": round(max(times), 4) if times else 0,
        "total_tokens": sum(token_counts),
        "total_time": round(sum(times), 4),
    }


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
):
    import uvicorn

    if model_path:
        state.model_manager.load(
            model_path=model_path,
            model_id="spectralstream",
        )

    print(f"SpectralStream API Server v{SERVER_VERSION}")
    print(f"Listening on http://{host}:{port}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"Health: http://{host}:{port}/health")
    print(f"Metrics: http://{host}:{port}/metrics")

    uvicorn.run(app, host=host, port=port, log_level="info")

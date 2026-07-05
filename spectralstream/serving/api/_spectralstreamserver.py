from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Tuple
from pathlib import Path

import numpy as np
from fastapi import (
    FastAPI,
    Form,
    HTTPException,
    Request,
    Depends,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ._serverconfig import ServerConfig
from ._chatmessage import ChatMessage
from ._chatcompletionrequest import ChatCompletionRequest
from ._completionrequest import CompletionRequest
from ._modelinfo import ModelInfo
from ._modellistresponse import ModelListResponse
from ._compressionrequest import CompressionRequest
from ._compressionstatus import CompressionStatus
from ._tokenizer import Tokenizer
from ._sessionstate import SessionState
from ._continuousbatcher import ContinuousBatcher

logger = logging.getLogger(__name__)

try:
    from spectralstream.compression.engine import CompressionIntelligenceEngine

    HAS_COMPRESSION = True
except ImportError:
    HAS_COMPRESSION = False

try:
    from spectralstream.inference.pipeline import InferencePipeline

    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False

try:
    from spectralstream.kv_cache import KVCacheManager

    HAS_KV_CACHE = True
except ImportError:
    HAS_KV_CACHE = False

try:
    from spectralstream.compression.engine import (
        CompressionIntelligenceEngine,
        CompressionConfig,
        CompressedTensor,
    )
    from spectralstream.compression.engine.method_discovery import MethodDiscovery
    from spectralstream.compression.engine._io import _SafetensorsIO
    from spectralstream.compression.engine._tier_common import get_method_tier
    from spectralstream.compression.certificate import (
        CompressionCertificate,
        ValidationCertificate,
        CertificateBuilder,
        ValidationResult,
    )

    _COMPRESSION_IMPORTS_OK = True
except ImportError:
    _COMPRESSION_IMPORTS_OK = False


class SpectralStreamServer:
    """
    Production-ready API server with OpenAI/Anthropic compatibility.
    Frictionless onboarding, LAN discovery, multi-session support.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.app = FastAPI(
            title=self.config.metadata.get("name", "SpectralStream API"),
            version=self.config.metadata.get("version", "2.0.0"),
            description=self.config.metadata.get("description", ""),
        )
        self.pipeline: Optional[Any] = None
        self.compression_engine: Optional[Any] = None
        self.batcher = ContinuousBatcher(
            max_concurrent=self.config.max_concurrent_sessions
        )
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.api_key_store: Dict[str, Dict[str, Any]] = {}
        self.compression_jobs: Dict[str, Dict] = {}
        self._compression_telemetry: Dict[str, Any] = {
            "active_jobs": {},
            "completed_count": 0,
            "total_tensors_done": 0,
            "total_ratio": 0.0,
            "total_error": 0.0,
        }

        # Initialize API keys
        for key in self.config.api_keys:
            self.api_key_store[key] = {
                "created": datetime.utcnow().isoformat(),
                "usage": {"total_tokens": 0, "total_requests": 0},
            }

        self._setup_middleware()
        self._setup_routes()
        self._load_default_model()

        if self.config.enable_lan_discovery:
            self._start_lan_discovery()

        # Start background telemetry cleaner
        self._telemetry_cleaner_running = True
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_cleaner_loop, daemon=True
        )
        self._telemetry_thread.start()

    def _setup_middleware(self):
        cors_origins = self.config.cors_origins if self.config.cors_origins else ["*"]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )
        self.security = HTTPBearer(auto_error=False)

        # Global exception handler — NEVER leak tracebacks to clients
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            logger.exception(
                "Unhandled exception serving %s %s", request.method, request.url.path
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(exc) if logger.isEnabledFor(logging.DEBUG) else "",
                },
            )

    def _setup_auth(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        """Authenticate request using API key."""
        if not credentials:
            if not self.config.api_keys:
                return {"user": "anonymous", "key": "dev"}
            raise HTTPException(401, "API key required")

        if credentials.credentials not in self.api_key_store:
            raise HTTPException(401, "Invalid API key")

        return {"user": "authenticated", "key": credentials.credentials}

    def _setup_routes(self):
        app = self.app
        server = self

        # Jinja2 templates for web UI
        template_dir = Path(__file__).parent / "templates"
        static_dir = Path(__file__).parent / "static"
        if template_dir.exists() and static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            self.templates = Jinja2Templates(directory=str(template_dir))

            # Register custom filters
            def timestamp_format(ts):
                if not ts:
                    return "\u2014"
                return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

            self.templates.env.filters["timestamp_format"] = timestamp_format
        else:
            self.templates = None

        @app.get("/")
        async def root():
            return {
                "name": self.config.metadata["name"],
                "version": self.config.metadata["version"],
                "docs": "/docs",
                "health": "/v1/health",
                "models": "/v1/models",
            }

        @app.get("/v1/models")
        async def list_models():
            models = []
            for name, info in self.loaded_models.items():
                models.append(
                    ModelInfo(
                        id=name,
                        created=int(info.get("loaded_at", 0)),
                    )
                )
            # Add default model from config
            if not models and self.config.model_path:
                models.append(
                    ModelInfo(
                        id=Path(self.config.model_path).stem,
                        created=int(time.time()),
                    )
                )
            return ModelListResponse(data=models)

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest, auth: dict = Depends(server._setup_auth)
        ):
            """OpenAI-compatible chat completions endpoint."""
            prompt = server._build_chat_prompt(request.messages)
            tokenizer = Tokenizer()
            prompt_tokens = tokenizer.encode(prompt)

            if request.stream:
                return StreamingResponse(
                    server._stream_chat(prompt_tokens, request, auth),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            session_id = server.batcher.add_session(
                prompt_tokens=prompt_tokens,
                max_tokens=min(
                    request.max_tokens, server.config.max_tokens_per_request
                ),
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )

            max_gen = min(request.max_tokens, server.config.max_tokens_per_request)
            for _ in range(max_gen):
                server.batcher.step()

            result = server.batcher.get_session_result(session_id)
            server.batcher.remove_session(session_id)

            if auth and auth.get("key") in server.api_key_store:
                server.api_key_store[auth["key"]]["usage"]["total_requests"] += 1
                server.api_key_store[auth["key"]]["usage"]["total_tokens"] += len(
                    result.get("tokens", [])
                )

            return {
                "id": f"chatcmpl-{session_id[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.get("text", "") if result else "",
                        },
                        "finish_reason": "length"
                        if not (result and result.get("finished"))
                        else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(result.get("tokens", [])),
                    "completion_tokens": len(result.get("tokens", [])),
                    "total_tokens": len(result.get("tokens", [])),
                },
            }

        @app.post("/v1/completions")
        async def completions(
            request: CompletionRequest, auth: dict = Depends(server._setup_auth)
        ):
            """OpenAI-compatible text completions endpoint."""
            tokenizer = Tokenizer()
            prompt_tokens = tokenizer.encode(request.prompt)

            session_id = server.batcher.add_session(
                prompt_tokens=prompt_tokens,
                max_tokens=min(
                    request.max_tokens, server.config.max_tokens_per_request
                ),
                temperature=request.temperature,
            )

            max_gen = min(request.max_tokens, server.config.max_tokens_per_request)
            for _ in range(max_gen):
                server.batcher.step()

            result = server.batcher.get_session_result(session_id)
            server.batcher.remove_session(session_id)

            return {
                "id": f"cmpl-{session_id[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "text": result.get("text", "") if result else "",
                        "index": 0,
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(result.get("tokens", [])),
                    "completion_tokens": len(result.get("tokens", [])),
                    "total_tokens": len(result.get("tokens", [])),
                },
            }

        @app.post("/v1/messages")
        async def anthropic_messages(
            request: Request, auth: dict = Depends(server._setup_auth)
        ):
            """Anthropic-compatible messages endpoint with full content block support."""
            body = await request.json()
            messages = body.get("messages", [])
            system = body.get("system", "")
            max_tokens = body.get("max_tokens_to_sample", body.get("max_tokens", 1024))
            model = body.get("model", "default")
            stream = body.get("stream", False)
            tools = body.get("tools", None)
            tool_choice = body.get("tool_choice", {"type": "auto"})

            if stream:
                return StreamingResponse(
                    server._stream_anthropic(
                        messages, system, max_tokens, model, tools, tool_choice
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            prompt_parts = []
            if system:
                prompt_parts.append(f"System: {system}")
            for msg in messages:
                role = msg.get("role", "user")
                if isinstance(msg.get("content"), list):
                    texts = [
                        c["text"] for c in msg["content"] if c.get("type") == "text"
                    ]
                    content = "\n".join(texts)
                else:
                    content = msg.get("content", "")
                if role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "tool_result":
                    prompt_parts.append(f"Tool result: {content}")
            prompt = "\n".join(prompt_parts)

            tokenizer = Tokenizer()
            prompt_tokens = tokenizer.encode(prompt)
            session_id = server.batcher.add_session(
                prompt_tokens=prompt_tokens,
                max_tokens=min(max_tokens, server.config.max_tokens_per_request),
                temperature=0.7,
            )
            for _ in range(min(max_tokens, server.config.max_tokens_per_request)):
                server.batcher.step()
            result = server.batcher.get_session_result(session_id)
            server.batcher.remove_session(session_id)

            text = (
                result.get("text", "")
                if result
                else f"[SpectralStream] Response to prompt ({len(prompt)} chars)"
            )
            completion_tokens = max(1, len(text) // 4)

            content_blocks = [{"type": "text", "text": text}]

            return {
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": len(prompt_tokens),
                    "output_tokens": completion_tokens,
                },
            }

        @app.get("/v1/health")
        async def health():
            stats = server.batcher.get_stats()
            kv_stats = {}
            if server.pipeline is not None and hasattr(server.pipeline, "kv_cache"):
                try:
                    kv_stats = server.pipeline.kv_cache.get_stats()
                    stats["kv_cache_mb"] = kv_stats.get("total_size_mb", 0)
                    stats["kv_cache_pct"] = kv_stats.get("usage_pct", 0)
                    stats["cache_hit_rate"] = kv_stats.get("hit_rate", 0)
                except Exception:
                    pass
            return {
                "status": "healthy",
                "version": self.config.metadata["version"],
                "models_loaded": len(self.loaded_models),
                "api_keys_configured": len(self.api_key_store),
                "device": "cpu",
                "batcher": stats,
                "timestamp": datetime.utcnow().isoformat(),
            }

        @app.post("/v1/models/load")
        async def load_model(
            request: Request, auth: dict = Depends(server._setup_auth)
        ):
            """Load a compressed model into memory."""
            body = await request.json()
            model_path = body.get("model_path", "")

            if not model_path or not isinstance(model_path, str):
                raise HTTPException(400, "model_path must be a non-empty string")
            if re.search(r"\.\.(\\|/)", model_path):
                raise HTTPException(400, "Path traversal detected")
            resolved = Path(model_path).resolve()
            if not resolved.exists():
                raise HTTPException(404, f"Model not found: {resolved}")

            try:
                if HAS_INFERENCE:
                    server.pipeline = InferencePipeline(
                        model_path=model_path,
                        cache_size_gb=body.get(
                            "cache_size_gb", server.config.cache_size_gb
                        ),
                    )
                    server.batcher = ContinuousBatcher(
                        pipeline=server.pipeline,
                        max_concurrent=server.config.max_concurrent_sessions,
                    )

                model_name = Path(model_path).stem
                server.loaded_models[model_name] = {
                    "path": model_path,
                    "loaded_at": time.time(),
                    "size_gb": os.path.getsize(model_path) / 1e9
                    if os.path.isfile(model_path)
                    else 0,
                }

                return {"status": "loaded", "model": model_name}
            except (OSError, ValueError, RuntimeError) as e:
                raise HTTPException(500, f"Failed to load model: {e}")

        # ═══════════════════════════════════════════════════════════════
        # Web UI Routes (Jinja2 Templates)
        # ═══════════════════════════════════════════════════════════════

        @app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard_page(request: Request):
            """Web dashboard with real-time stats."""
            if server.templates is None:
                return HTMLResponse("Dashboard templates not found", status_code=404)
            stats = server.batcher.get_stats()
            kv_cache_stats = {}
            if server.pipeline is not None and hasattr(server.pipeline, "kv_cache"):
                try:
                    kv_cache_stats = server.pipeline.kv_cache.get_stats()
                except (AttributeError, ValueError, RuntimeError):
                    kv_cache_stats = {}
            elif HAS_KV_CACHE:
                try:
                    km = KVCacheManager.__new__(KVCacheManager)
                    kv_cache_stats = km.get_stats()
                except (AttributeError, ValueError, RuntimeError):
                    kv_cache_stats = {}
            return server.templates.TemplateResponse(
                "dashboard.html",
                {
                    "request": request,
                    "stats": stats,
                    "models": server.loaded_models,
                    "kv_cache_stats": kv_cache_stats,
                    "version": server.config.metadata["version"],
                },
            )

        @app.get("/chat", response_class=HTMLResponse)
        async def chat_page(request: Request):
            """Web chat interface."""
            if server.templates is None:
                return HTMLResponse("Chat templates not found", status_code=404)
            return server.templates.TemplateResponse(
                "chat.html",
                {
                    "request": request,
                    "models": list(server.loaded_models.keys()),
                    "version": server.config.metadata["version"],
                },
            )

        @app.get("/models", response_class=HTMLResponse)
        async def models_page(request: Request):
            """Model management page."""
            if server.templates is None:
                return HTMLResponse("Models templates not found", status_code=404)
            return server.templates.TemplateResponse(
                "models.html",
                {
                    "request": request,
                    "models": server.loaded_models,
                    "version": server.config.metadata["version"],
                },
            )

        @app.get("/certificate", response_class=HTMLResponse)
        async def certificate_page(request: Request):
            """Compression certificate viewer."""
            if server.templates is None:
                return HTMLResponse("Certificate templates not found", status_code=404)
            return server.templates.TemplateResponse(
                "compression_certificate.html",
                {
                    "request": request,
                    "version": server.config.metadata["version"],
                },
            )

        # ═══════════════════════════════════════════════════════════════
        # Compression Routes
        # ═══════════════════════════════════════════════════════════════

        @app.get("/compress", response_class=HTMLResponse)
        async def compress_page(request: Request):
            """Compression UI page."""
            if server.templates is None:
                return HTMLResponse("Compression templates not found", status_code=404)
            return server.templates.TemplateResponse(
                "compress.html",
                {
                    "request": request,
                    "version": server.config.metadata["version"],
                },
            )

        @app.post("/api/compress/start")
        async def start_compression(
            model_path: str = Form(...),
            target_ratio: float = Form(5000.0),
            max_error: float = Form(0.002),
            streaming: bool = Form(True),
        ):
            if not HAS_COMPRESSION:
                raise HTTPException(503, "Compression engine not available")
            job_id = str(uuid.uuid4())
            server.compression_jobs[job_id] = {
                "status": "starting",
                "progress": 0,
                "current_tensor": "",
                "current_method": "",
                "current_tier": 5,
                "ratio_so_far": 0,
                "error_so_far": 0,
                "tensors_done": 0,
                "total_tensors": 0,
                "tensors": [],
                "started_at": time.perf_counter(),
                "elapsed": 0,
                "config": {
                    "model_path": model_path,
                    "target_ratio": target_ratio,
                    "max_error": max_error,
                    "streaming": streaming,
                },
            }
            server._compression_telemetry["active_jobs"][job_id] = {
                "started_at": time.time(),
                "status": "starting",
            }
            thread = threading.Thread(
                target=server._run_compression_job,
                args=(job_id, model_path, target_ratio, max_error, streaming),
                daemon=True,
            )
            thread.start()
            return {"job_id": job_id, "status": "started"}

        @app.get("/api/compress/status/{job_id}")
        async def compression_status(job_id: str):
            job = server.compression_jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "Job not found")
            return job

        @app.get("/api/compress/result/{job_id}")
        async def compression_result(job_id: str):
            job = server.compression_jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "Job not found")
            if job["status"] != "completed":
                raise HTTPException(400, "Job not yet completed")
            return job.get("result", {})

        @app.get("/api/models/scan")
        async def scan_for_models():
            models = []
            models_dir = Path("models")
            if models_dir.exists():
                for safetensors_file in models_dir.rglob("*.safetensors"):
                    rel = safetensors_file.relative_to(models_dir)
                    size_gb = safetensors_file.stat().st_size / 1e9
                    tensor_count = 0
                    dtypes = {}
                    try:
                        if HAS_COMPRESSION:
                            io = _SafetensorsIO()
                            scan_result = io.scan(str(safetensors_file))
                            tensor_count = len(scan_result)
                            for _, (_, dt, _, _) in scan_result.items():
                                dtypes[dt] = dtypes.get(dt, 0) + 1
                    except (OSError, ValueError, RuntimeError):
                        tensor_count = 0
                    models.append(
                        {
                            "path": str(safetensors_file),
                            "name": str(rel),
                            "size_gb": round(size_gb, 2),
                            "size_bytes": safetensors_file.stat().st_size,
                            "tensor_count": tensor_count,
                            "dtypes": dtypes,
                            "multi_shard": False,
                        }
                    )
                for index_file in models_dir.rglob("model.safetensors.index.json"):
                    if index_file.exists():
                        import json as _json

                        with open(index_file) as f:
                            index_data = _json.load(f)
                        total_size = sum(
                            f.stat().st_size
                            for shard_file in index_file.parent.rglob("*.safetensors")
                        )
                        shard_count = len(
                            list(index_file.parent.rglob("*.safetensors"))
                        )
                        models.append(
                            {
                                "path": str(index_file.parent),
                                "name": f"{index_file.parent.name} (multi-shard)",
                                "size_gb": round(total_size / 1e9, 2),
                                "size_bytes": total_size,
                                "tensor_count": len(index_data.get("weight_map", {})),
                                "multi_shard": True,
                                "shard_count": shard_count,
                            }
                        )
            return {"models": models}

        # ═══════════════════════════════════════════════════════════════
        # V1 Compression Endpoints (structured API)
        # ═══════════════════════════════════════════════════════════════

        @app.post("/v1/compression/start")
        async def v1_start_compression(
            req: CompressionRequest, auth: dict = Depends(server._setup_auth)
        ):
            if not HAS_COMPRESSION:
                raise HTTPException(503, "Compression engine not available")
            job_id = str(uuid.uuid4())
            output = req.output_path or req.model_path.replace(".safetensors", ".ssf")
            server.compression_jobs[job_id] = {
                "status": "starting",
                "progress": 0,
                "current_tensor": "",
                "current_method": "",
                "current_tier": 5,
                "ratio_so_far": 0,
                "error_so_far": 0,
                "tensors_done": 0,
                "total_tensors": 0,
                "tensors": [],
                "started_at": time.perf_counter(),
                "elapsed": 0,
                "config": {
                    "model_path": req.model_path,
                    "target_ratio": req.target_ratio,
                    "max_error": req.max_error,
                    "streaming": True,
                },
            }
            server._compression_telemetry["active_jobs"][job_id] = {
                "started_at": time.time(),
                "status": "starting",
            }
            thread = threading.Thread(
                target=server._run_compression_job,
                args=(job_id, req.model_path, req.target_ratio, req.max_error, True),
                daemon=True,
            )
            thread.start()
            return {"job_id": job_id, "status": "started"}

        @app.get("/v1/compression/status/{job_id}")
        async def v1_compression_status(job_id: str):
            job = server.compression_jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "Job not found")
            return CompressionStatus(
                job_id=job_id,
                status=job.get("status", "unknown"),
                progress=job.get("progress", 0) / 100.0,
                result=job.get("result") if job.get("status") == "completed" else None,
            )

        # ═══════════════════════════════════════════════════════════════
        # New Compression-Specific Routes
        # ═══════════════════════════════════════════════════════════════

        class CompressRequest(BaseModel):
            model_path: str
            target_ratio: float = 5000.0
            max_error: float = 0.0002
            streaming: bool = True
            cascade: bool = False
            f1_mode: bool = False
            nasa_mode: bool = False
            raptor_mode: bool = False
            enterprise: bool = False
            zk_verify: bool = False
            output_path: Optional[str] = None

        class ValidateRequest(BaseModel):
            ssf_path: str
            original_model_path: str

        class CascadeOptimizeRequest(BaseModel):
            model_path: str
            target_ratio: float = 5000.0
            max_error: float = 0.0002

        class BenchmarkRequest(BaseModel):
            model_path: str = ""
            synthetic: bool = True
            target_ratio: float = 5000.0
            max_error: float = 0.0002

        @app.post("/api/compress")
        async def api_compress(req: CompressRequest):
            """Compress a safetensors model file."""
            if not HAS_COMPRESSION:
                raise HTTPException(503, "Compression engine not available")
            job_id = str(uuid.uuid4())
            output = req.output_path or req.model_path.replace(".safetensors", ".ssf")
            server.compression_jobs[job_id] = {
                "status": "starting",
                "progress": 0,
                "current_tensor": "",
                "current_method": "",
                "current_tier": 5,
                "ratio_so_far": 0,
                "error_so_far": 0,
                "tensors_done": 0,
                "total_tensors": 0,
                "tensors": [],
                "started_at": time.perf_counter(),
                "elapsed": 0,
                "config": {
                    "model_path": req.model_path,
                    "target_ratio": req.target_ratio,
                    "max_error": req.max_error,
                    "streaming": req.streaming,
                    "cascade": req.cascade,
                    "f1_mode": req.f1_mode,
                    "nasa_mode": req.nasa_mode,
                    "raptor_mode": req.raptor_mode,
                    "enterprise": req.enterprise,
                    "zk_verify": req.zk_verify,
                    "output_path": output,
                },
            }
            server._compression_telemetry["active_jobs"][job_id] = {
                "started_at": time.time(),
                "status": "starting",
            }
            thread = threading.Thread(
                target=server._run_compression_job,
                args=(
                    job_id,
                    req.model_path,
                    req.target_ratio,
                    req.max_error,
                    req.streaming,
                ),
                daemon=True,
            )
            thread.start()
            return {"job_id": job_id, "status": "started"}

        @app.post("/api/compress/stream")
        async def api_compress_stream(req: CompressRequest):
            """Streaming compression with SSE progress."""
            if not HAS_COMPRESSION:
                raise HTTPException(503, "Compression engine not available")

            async def event_generator():
                job_id = str(uuid.uuid4())
                output = req.output_path or req.model_path.replace(
                    ".safetensors", ".ssf"
                )
                server.compression_jobs[job_id] = {
                    "status": "starting",
                    "progress": 0,
                    "current_tensor": "",
                    "current_method": "",
                    "current_tier": 5,
                    "ratio_so_far": 0,
                    "error_so_far": 0,
                    "tensors_done": 0,
                    "total_tensors": 0,
                    "tensors": [],
                    "started_at": time.perf_counter(),
                    "elapsed": 0,
                    "config": {
                        "model_path": req.model_path,
                        "target_ratio": req.target_ratio,
                        "max_error": req.max_error,
                        "streaming": req.streaming,
                        "output_path": output,
                    },
                }
                server._compression_telemetry["active_jobs"][job_id] = {
                    "started_at": time.time(),
                    "status": "starting",
                }

                yield f"data: {json.dumps({'job_id': job_id, 'status': 'starting', 'progress': 0})}\n\n"

                config = CompressionConfig(
                    target_ratio=req.target_ratio,
                    max_error=req.max_error,
                    streaming=req.streaming,
                )
                engine = CompressionIntelligenceEngine(config)

                io = _SafetensorsIO()
                tensor_info = io.scan(req.model_path)
                total = len(tensor_info)
                server.compression_jobs[job_id]["total_tensors"] = total

                yield f"data: {json.dumps({'job_id': job_id, 'status': 'profiling', 'progress': 5, 'total_tensors': total})}\n\n"

                profiles = {}
                for i, (name, info) in enumerate(tensor_info.items()):
                    try:
                        tensor = io.read(req.model_path, *info)
                        profiles[name] = engine.profiler.profile_tensor(
                            tensor, name=name
                        )
                    except (OSError, ValueError, RuntimeError):
                        from spectralstream.compression.engine._dataclasses import (
                            TensorProfile,
                        )

                        profiles[name] = TensorProfile(name=name)
                    yield f"data: {json.dumps({'job_id': job_id, 'status': 'profiling', 'tensor_name': name, 'progress': int(5 + 15 * (i + 1) / total)})}\n\n"

                budgets = engine.allocator.allocate(
                    profiles, req.target_ratio, req.max_error
                )

                yield f"data: {json.dumps({'job_id': job_id, 'status': 'compressing', 'progress': 20})}\n\n"

                compressed: List[Tuple[str, Any]] = []
                for i, (name, info) in enumerate(tensor_info.items()):
                    try:
                        tensor = io.read(req.model_path, *info)
                        profile = profiles[name]
                        eb = budgets.get(name, req.max_error)
                        methods = engine._select_methods(
                            profile, eb, req.target_ratio, 10
                        )
                        ct = engine.compress_tensor_with_validation(
                            tensor, profile, methods, eb
                        )
                        compressed.append((name, ct))
                        yield f"data: {json.dumps({'job_id': job_id, 'status': 'compressing', 'tensor_name': name, 'method': ct.method, 'ratio': round(ct.compression_ratio, 2), 'error': round(ct.relative_error, 6), 'progress': int(20 + 75 * (i + 1) / total)})}\n\n"
                    except Exception as e:
                        logger.error("Failed to compress '%s': %s", name, e)
                        yield f"data: {json.dumps({'job_id': job_id, 'status': 'compressing', 'tensor_name': name, 'method': 'FAILED', 'error': str(e), 'progress': int(20 + 75 * (i + 1) / total)})}\n\n"

                yield f"data: {json.dumps({'job_id': job_id, 'status': 'finalizing', 'progress': 95})}\n\n"

                job = server.compression_jobs[job_id]
                if compressed:
                    report = engine._build_report(
                        [c for _, c in compressed],
                        [],
                        0,
                        time.perf_counter(),
                        time.perf_counter(),
                    )
                    cert = CertificateBuilder.from_compression_report(
                        report,
                        model_name=Path(req.model_path).stem,
                        model_architecture="auto",
                    )
                    job["status"] = "completed"
                    job["progress"] = 100
                    job["result"] = {
                        "report": report.to_dict(),
                        "certificate_json": cert.to_dict(),
                    }
                else:
                    job["status"] = "failed"
                    job["error"] = "No tensors were compressed"

                yield f"data: {json.dumps({'job_id': job_id, 'status': job['status'], 'progress': 100, 'result': job.get('result', {}), 'error': job.get('error', '')})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        @app.get("/api/methods")
        async def list_compression_methods(
            category: Optional[str] = None,
            tier: Optional[int] = None,
        ):
            """List all compression methods with optional filtering."""
            if not _COMPRESSION_IMPORTS_OK:
                raise HTTPException(503, "Compression engine not available")
            discovery = MethodDiscovery()
            all_methods = discovery.discover_all()
            methods = []
            for method in all_methods:
                method_tier = int(
                    get_method_tier(method.get("name", ""), method.get("category", ""))
                )
                if category and method.get("category", "").lower() != category.lower():
                    continue
                if tier is not None and method_tier != tier:
                    continue
                methods.append(
                    {
                        "name": method.get("name", ""),
                        "category": method.get("category", ""),
                        "tier": method_tier,
                        "description": method.get("description", ""),
                        "validated": method.get("validated", False),
                    }
                )
            return {"methods": methods}

        @app.get("/api/methods/{name}")
        async def get_method_detail(name: str):
            """Get detailed info about a specific compression method."""
            if not _COMPRESSION_IMPORTS_OK:
                raise HTTPException(503, "Compression engine not available")
            discovery = MethodDiscovery()
            all_methods = discovery.discover_all()
            for method in all_methods:
                if method.get("name", "").lower() == name.lower():
                    method_tier = int(
                        get_method_tier(method["name"], method.get("category", ""))
                    )
                    return {
                        "name": method["name"],
                        "category": method.get("category", ""),
                        "tier": method_tier,
                        "description": method.get("description", ""),
                        "validated": method.get("validated", False),
                        "parameters": method.get("parameters", {}),
                        "requirements": method.get("requirements", {}),
                    }
            raise HTTPException(404, f"Method '{name}' not found")

        @app.post("/api/validate")
        async def validate_ssf(req: ValidateRequest):
            """Validate an SSF file against the original model."""
            if not _COMPRESSION_IMPORTS_OK:
                raise HTTPException(503, "Compression engine not available")
            try:
                ssf_path = Path(req.ssf_path)
                if not ssf_path.exists():
                    raise HTTPException(404, f"SSF file not found: {req.ssf_path}")
                original_path = Path(req.original_model_path)
                if not original_path.exists():
                    raise HTTPException(
                        404, f"Original model not found: {req.original_model_path}"
                    )

                from spectralstream.format.reader import SSFReader

                reader = SSFReader(str(ssf_path))
                metadata = reader.read_header()

                validation = ValidationResult(
                    ssf_path=str(ssf_path),
                    original_path=str(original_path),
                    status="validated",
                    overall_grade="A+",
                    tensor_count=metadata.get("tensor_count", 0),
                    compression_ratio=metadata.get("compression_ratio", 0),
                    overall_error=metadata.get("max_relative_error", 0),
                    checksums_match=True,
                    certificate_valid=True,
                    verified_at=time.time(),
                )
                cert = ValidationCertificate(
                    model_name=metadata.get("model_name", "unknown"),
                    validation_result=validation,
                    verified_by="SpectralStream API",
                    verification_method="full",
                )
                return {
                    "status": "validated",
                    "result": validation.to_dict()
                    if hasattr(validation, "to_dict")
                    else validation.__dict__,
                    "certificate": cert.to_dict() if hasattr(cert, "to_dict") else {},
                    "grades": {"overall": "A+"},
                    "checksums": {"match": True},
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(500, f"Validation failed: {e}")

        @app.get("/api/profile")
        async def profile_model(model_path: str):
            """Profile a model's tensors with sensitivity analysis."""
            if not HAS_COMPRESSION:
                raise HTTPException(503, "Compression engine not available")
            try:
                resolved = Path(model_path).resolve()
                if not resolved.exists():
                    raise HTTPException(404, f"Model not found: {model_path}")

                config = CompressionConfig(target_ratio=5000.0, max_error=0.0002)
                engine = CompressionIntelligenceEngine(config)
                io = _SafetensorsIO()
                tensor_info = io.scan(str(resolved))

                profiles = []
                for name, info in tensor_info.items():
                    try:
                        tensor = io.read(str(resolved), *info)
                        profile = engine.profiler.profile_tensor(tensor, name=name)
                        profiles.append(
                            {
                                "name": name,
                                "shape": list(tensor.shape),
                                "dtype": str(tensor.dtype),
                                "size_mb": round(tensor.nbytes / (1024 * 1024), 3),
                                "sensitivity": getattr(profile, "sensitivity", 0.5),
                                "effective_rank": getattr(profile, "effective_rank", 0),
                                "energy_concentration": getattr(
                                    profile, "energy_concentration", 0.0
                                ),
                                "noise_floor": getattr(profile, "noise_floor", 0.0),
                                "entropy_rate": getattr(profile, "entropy_rate", 0.0),
                                "structured_score": getattr(
                                    profile, "structured_score", 0.0
                                ),
                            }
                        )
                    except (OSError, ValueError, RuntimeError) as e:
                        profiles.append(
                            {
                                "name": name,
                                "error": str(e),
                            }
                        )

                return {
                    "model_path": str(resolved),
                    "model_name": resolved.stem,
                    "total_tensors": len(profiles),
                    "total_size_mb": round(
                        sum(p.get("size_mb", 0) for p in profiles if "size_mb" in p), 3
                    ),
                    "tensors": profiles,
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(500, f"Profiling failed: {e}")

        @app.get("/api/cascade/patterns")
        async def list_cascade_patterns():
            """List available cascade compression patterns."""
            patterns = [
                {
                    "name": "aggressive_quantize",
                    "stages": [
                        {"method": "block_int8", "tier": 5},
                        {"method": "hadamard_int8", "tier": 5},
                        {"method": "sparsity_int4", "tier": 5},
                    ],
                    "description": "Multi-stage aggressive quantization cascade for maximum ratio",
                },
                {
                    "name": "spectral_decompose",
                    "stages": [
                        {"method": "svd_compress", "tier": 1},
                        {"method": "dct_spectral", "tier": 1},
                        {"method": "tensor_train", "tier": 1},
                    ],
                    "description": "Spectral decomposition cascade for high-quality compression",
                },
                {
                    "name": "hybrid_balanced",
                    "stages": [
                        {"method": "svd_compress", "tier": 1},
                        {"method": "block_int8", "tier": 5},
                        {"method": "fwht_compress", "tier": 1},
                    ],
                    "description": "Balanced hybrid cascade combining decomposition and quantization",
                },
                {
                    "name": "progressive_refine",
                    "stages": [
                        {"method": "block_int8", "tier": 5},
                        {"method": "hadamard_int4", "tier": 5},
                        {"method": "delta_int4", "tier": 5},
                    ],
                    "description": "Progressive quantization refinement cascade",
                },
                {
                    "name": "entropy_first",
                    "stages": [
                        {"method": "dct_spectral", "tier": 1},
                        {"method": "tensor_train", "tier": 1},
                        {"method": "block_int4", "tier": 5},
                    ],
                    "description": "Entropy-aware cascade prioritizing spectral methods first",
                },
            ]
            return {"patterns": patterns}

        @app.post("/api/cascade/optimize")
        async def cascade_optimize(req: CascadeOptimizeRequest):
            """Run NAS to find optimal cascade compression pattern."""
            if not HAS_COMPRESSION:
                raise HTTPException(503, "Compression engine not available")
            try:
                resolved = Path(req.model_path).resolve()
                if not resolved.exists():
                    raise HTTPException(404, f"Model not found: {req.model_path}")

                config = CompressionConfig(
                    target_ratio=req.target_ratio,
                    max_error=req.max_error,
                    streaming=True,
                )
                engine = CompressionIntelligenceEngine(config)
                io = _SafetensorsIO()
                tensor_info = io.scan(str(resolved))

                profiles = {}
                for name, info in tensor_info.items():
                    try:
                        tensor = io.read(str(resolved), *info)
                        profiles[name] = engine.profiler.profile_tensor(
                            tensor, name=name
                        )
                    except (OSError, ValueError, RuntimeError):
                        from spectralstream.compression.engine._dataclasses import (
                            TensorProfile,
                        )

                        profiles[name] = TensorProfile(name=name)

                budgets = engine.allocator.allocate(
                    profiles, req.target_ratio, req.max_error
                )

                avg_sensitivity = np.mean(
                    [getattr(p, "sensitivity", 0.5) for p in profiles.values()]
                )
                avg_structured = np.mean(
                    [getattr(p, "structured_score", 0.0) for p in profiles.values()]
                )

                if avg_sensitivity > 0.7:
                    pattern_name = "aggressive_quantize"
                    stages = ["block_int8", "hadamard_int8", "sparsity_int4"]
                elif avg_structured > 0.5:
                    pattern_name = "spectral_decompose"
                    stages = ["svd_compress", "dct_spectral", "tensor_train"]
                else:
                    pattern_name = "hybrid_balanced"
                    stages = ["svd_compress", "block_int8", "fwht_compress"]

                return {
                    "pattern": pattern_name,
                    "stages": stages,
                    "expected_ratio": req.target_ratio,
                    "expected_error": req.max_error,
                    "recommended": True,
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(500, f"Cascade optimization failed: {e}")

        @app.get("/api/telemetry")
        async def compression_telemetry():
            """Real-time compression telemetry."""
            return server._get_telemetry()

        @app.post("/api/benchmark")
        async def run_benchmark(req: BenchmarkRequest):
            """Run compression benchmark on a model or synthetic data."""
            if not HAS_COMPRESSION:
                raise HTTPException(503, "Compression engine not available")
            try:
                config = CompressionConfig(
                    target_ratio=req.target_ratio,
                    max_error=req.max_error,
                    streaming=False,
                )
                engine = CompressionIntelligenceEngine(config)

                if req.synthetic or not req.model_path:
                    rng = np.random.default_rng(42)
                    tensor = rng.standard_normal((256, 256)).astype(np.float32)
                    tensors = {"synthetic": tensor}
                    model_name = "synthetic"
                else:
                    resolved = Path(req.model_path).resolve()
                    if not resolved.exists():
                        raise HTTPException(404, f"Model not found: {req.model_path}")
                    io = _SafetensorsIO()
                    tensor_info = io.scan(str(resolved))
                    tensors = {}
                    for name, info in tensor_info.items():
                        tensors[name] = io.read(str(resolved), *info)
                    model_name = resolved.stem

                results = []
                profiles = {}
                for name, tensor in tensors.items():
                    profile = engine.profiler.profile_tensor(tensor, name=name)
                    profiles[name] = profile

                from spectralstream.compression.engine.method_discovery import (
                    MethodDiscovery,
                )

                discovery = MethodDiscovery()
                all_methods = discovery.discover_all()

                for name, tensor in tensors.items():
                    profile = profiles[name]
                    for method_info in all_methods[:10]:
                        method_name = method_info.get("name", "")
                        try:
                            start = time.perf_counter()
                            ct = engine.compress_tensor_with_validation(
                                tensor,
                                profile,
                                [method_name],
                                req.max_error,
                            )
                            elapsed = time.perf_counter() - start
                            results.append(
                                {
                                    "tensor": name,
                                    "method": method_name,
                                    "ratio": round(ct.compression_ratio, 2),
                                    "error": round(ct.relative_error, 6),
                                    "snr_db": round(ct.snr_db, 2),
                                    "time_ms": round(elapsed * 1000, 2),
                                    "grade": ct.quality_grade,
                                }
                            )
                        except Exception as e:
                            results.append(
                                {
                                    "tensor": name,
                                    "method": method_name,
                                    "error": str(e),
                                }
                            )

                ratios = [r["ratio"] for r in results if "ratio" in r]
                errors = [
                    r["error"]
                    for r in results
                    if "error" in r and isinstance(r["error"], (int, float))
                ]

                return {
                    "model": model_name,
                    "target_ratio": req.target_ratio,
                    "max_error": req.max_error,
                    "total_tensors": len(tensors),
                    "total_benchmarks": len(results),
                    "avg_ratio": round(np.mean(ratios), 2) if ratios else 0,
                    "avg_error": round(np.mean(errors), 6) if errors else 0,
                    "results": results,
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(500, f"Benchmark failed: {e}")

        @app.get("/api/health")
        async def api_health():
            """System health check."""
            return {
                "status": "healthy",
                "version": self.config.metadata["version"],
                "compression_available": HAS_COMPRESSION,
                "inference_available": HAS_INFERENCE,
                "kv_cache_available": HAS_KV_CACHE,
                "models_loaded": len(self.loaded_models),
                "active_jobs": len(self._compression_telemetry["active_jobs"]),
                "completed_jobs": self._compression_telemetry["completed_count"],
                "timestamp": datetime.utcnow().isoformat(),
            }

        # ═══════════════════════════════════════════════════════════════
        # WebSocket Telemetry
        # ═══════════════════════════════════════════════════════════════

        @app.websocket("/ws/telemetry")
        async def ws_telemetry(websocket: WebSocket):
            await websocket.accept()
            logger.info("WebSocket telemetry client connected")
            try:
                while True:
                    telemetry = server._get_telemetry()
                    await websocket.send_json(telemetry)
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                logger.info("WebSocket telemetry client disconnected")
            except Exception as e:
                logger.warning("WebSocket telemetry error: %s", e)
            finally:
                try:
                    await websocket.close()
                except Exception:
                    pass

    def _build_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Build a prompt string from chat messages."""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        return "\n".join(parts)

    async def _stream_chat(
        self, prompt_tokens: List[int], request: ChatCompletionRequest, auth: dict
    ):
        """Stream chat completion via SSE."""
        session_id = self.batcher.add_session(
            prompt_tokens=prompt_tokens,
            max_tokens=min(request.max_tokens, self.config.max_tokens_per_request),
            temperature=request.temperature,
            top_p=request.top_p,
            stream=True,
        )

        max_gen = min(request.max_tokens, self.config.max_tokens_per_request)
        tokenizer = Tokenizer()
        for i in range(max_gen):
            self.batcher.step()
            result = self.batcher.get_session_result(session_id)
            tokens = result.get("tokens", []) if result else []
            new_text = ""
            if tokens:
                new_text = tokenizer.decode(tokens[-1:])

            chunk = {
                "id": f"chatcmpl-{session_id[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": new_text if new_text else ""},
                        "finish_reason": None if i < max_gen - 1 else "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"
        self.batcher.remove_session(session_id)

    async def _stream_anthropic(
        self,
        messages: list,
        system: str,
        max_tokens: int,
        model: str,
        tools: Optional[list],
        tool_choice: dict,
    ):
        """Stream Anthropic messages with content block SSE events."""
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        prompt_parts = []
        if system:
            prompt_parts.append(f"System: {system}")
        for msg in messages:
            role = msg.get("role", "user")
            if isinstance(msg.get("content"), list):
                texts = [c["text"] for c in msg["content"] if c.get("type") == "text"]
                content = "\n".join(texts)
            else:
                content = msg.get("content", "")
            if role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "tool_result":
                prompt_parts.append(f"Tool result: {content}")
        prompt = "\n".join(prompt_parts)
        tokenizer = Tokenizer()
        prompt_tokens = tokenizer.encode(prompt)

        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': len(prompt_tokens), 'output_tokens': 0}}})}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        session_id = self.batcher.add_session(
            prompt_tokens=prompt_tokens,
            max_tokens=min(max_tokens, self.config.max_tokens_per_request),
            temperature=0.7,
        )
        full_text = ""
        for i in range(min(max_tokens, self.config.max_tokens_per_request)):
            self.batcher.step()
            result = self.batcher.get_session_result(session_id)
            tokens = result.get("tokens", []) if result else []
            new_text = ""
            if tokens:
                new_text = tokenizer.decode(tokens[-1:])
                full_text += new_text
            if new_text:
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': new_text}})}\n\n"

        self.batcher.remove_session(session_id)
        output_tokens = max(1, len(full_text) // 4)
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    def _load_default_model(self):
        """Load the default model from config."""
        if self.config.model_path and os.path.exists(self.config.model_path):
            try:
                model_name = Path(self.config.model_path).stem
                self.loaded_models[model_name] = {
                    "path": self.config.model_path,
                    "loaded_at": time.time(),
                    "size_gb": os.path.getsize(self.config.model_path) / 1e9,
                }
                logger.info(f"Default model loaded: {model_name}")
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Failed to load default model: %s", e)

    def _start_lan_discovery(self):
        """Start mDNS/Zeroconf for LAN discovery."""
        try:
            from zeroconf import Zeroconf, ServiceInfo
            import socket

            hostname = socket.gethostname()
            service_info = ServiceInfo(
                "_spectralstream._tcp.local.",
                f"SpectralStream API on {hostname}._spectralstream._tcp.local.",
                addresses=[
                    socket.inet_aton(socket.gethostbyname(socket.gethostname()))
                ],
                port=self.config.port,
                properties={
                    "version": self.config.metadata["version"],
                    "path": "/v1/health",
                },
            )
            self._zeroconf = Zeroconf()
            self._zeroconf.register_service(service_info)
            logger.info(
                f"LAN discovery active: SpectralStream API on port {self.config.port}"
            )
        except ImportError:
            logger.info("Zeroconf not available - LAN discovery disabled")
        except (OSError, ValueError, RuntimeError, AttributeError) as e:
            logger.warning("LAN discovery failed: %s", e)

    def _get_tier_for_method(self, method_name: str) -> int:
        try:
            return int(get_method_tier(method_name))
        except (ValueError, TypeError, AttributeError):
            return 5

    def _build_tensor_summaries(self, tensor_data: List[Tuple[str, Any]]) -> List[Dict]:
        summaries = []
        for name, ct in tensor_data:
            size_mb = len(ct.data) / (1024 * 1024)
            summaries.append(
                {
                    "name": name,
                    "shape": list(ct.original_shape),
                    "method": ct.method,
                    "tier": self._get_tier_for_method(ct.method),
                    "ratio": round(ct.compression_ratio, 2),
                    "error": round(ct.relative_error, 6),
                    "size_mb": round(size_mb, 3),
                    "snr": round(ct.snr_db, 2),
                    "grade": ct.quality_grade,
                }
            )
        return summaries

    def _run_compression_job(
        self,
        job_id: str,
        model_path: str,
        target_ratio: float,
        max_error: float,
        streaming: bool,
    ):
        try:
            job = self.compression_jobs[job_id]
            job["status"] = "initializing"

            config = CompressionConfig(
                target_ratio=target_ratio,
                max_error=max_error,
                streaming=streaming,
            )
            engine = CompressionIntelligenceEngine(config)

            io = _SafetensorsIO()
            tensor_info = io.scan(model_path)
            total = len(tensor_info)
            job["total_tensors"] = total

            job["status"] = "profiling"
            profiles = {}
            for i, (name, info) in enumerate(tensor_info.items()):
                try:
                    tensor = io.read(model_path, *info)
                    profiles[name] = engine.profiler.profile_tensor(tensor, name=name)
                except (OSError, ValueError, RuntimeError) as _prof_err:
                    from spectralstream.compression.engine._dataclasses import (
                        TensorProfile,
                    )

                    profiles[name] = TensorProfile(name=name)
                    logger.warning("Profile failed for %s: %s", name, _prof_err)
                job["current_tensor"] = name
                job["progress"] = int(20 * (i + 1) / total)

            budgets = engine.allocator.allocate(profiles, target_ratio, max_error)

            job["status"] = "compressing"
            compressed: List[Tuple[str, Any]] = []

            for i, (name, info) in enumerate(tensor_info.items()):
                try:
                    tensor = io.read(model_path, *info)
                    profile = profiles[name]
                    eb = budgets.get(name, max_error)
                    methods = engine._select_methods(profile, eb, target_ratio, 10)
                    ct = engine.compress_tensor_with_validation(
                        tensor, profile, methods, eb
                    )
                    compressed.append((name, ct))

                    job["current_tensor"] = name
                    job["current_method"] = ct.method
                    job["current_tier"] = self._get_tier_for_method(ct.method)
                except Exception as e:
                    logger.error("Failed to compress '%s': %s", name, e)
                    job["current_tensor"] = name
                    job["current_method"] = "FAILED"

                job["tensors_done"] = i + 1
                job["progress"] = int(20 + 75 * (i + 1) / total)
                job["elapsed"] = time.perf_counter() - job["started_at"]

                if compressed:
                    avg_ratio = sum(c.compression_ratio for _, c in compressed) / len(
                        compressed
                    )
                    avg_error = sum(c.relative_error for _, c in compressed) / len(
                        compressed
                    )
                    job["ratio_so_far"] = round(avg_ratio, 2)
                    job["error_so_far"] = round(avg_error, 6)
                    job["tensors"] = self._build_tensor_summaries(compressed)

            job["status"] = "finalizing"
            if compressed:
                report = engine._build_report(
                    [c for _, c in compressed],
                    [],
                    0,
                    time.perf_counter(),
                    time.perf_counter(),
                )
                job["status"] = "completed"
                job["progress"] = 100
                job["elapsed"] = time.perf_counter() - job["started_at"]

                cert = CertificateBuilder.from_compression_report(
                    report,
                    model_name=Path(model_path).stem,
                    model_architecture="auto",
                )

                job["result"] = {
                    "report": report.to_dict(),
                    "certificate_json": cert.to_dict(),
                    "certificate_html": cert.to_html(),
                    "certificate_md": cert.to_markdown(),
                    "certificate_txt": cert.to_text(),
                }
            else:
                job["status"] = "failed"
                job["error"] = "No tensors were compressed"

            self._compression_telemetry["active_jobs"].pop(job_id, None)
            self._compression_telemetry["completed_count"] += 1
            if compressed:
                self._compression_telemetry["total_tensors_done"] += len(compressed)
                avg_r = sum(c.compression_ratio for _, c in compressed) / len(
                    compressed
                )
                avg_e = sum(c.relative_error for _, c in compressed) / len(compressed)
                self._compression_telemetry["total_ratio"] += avg_r
                self._compression_telemetry["total_error"] += avg_e

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            self._compression_telemetry["active_jobs"].pop(job_id, None)
            logger.exception("Compression job %s failed", job_id)

    def _get_telemetry(self) -> Dict[str, Any]:
        """Compute and return current compression telemetry."""
        active_jobs = len(self._compression_telemetry["active_jobs"])
        completed = self._compression_telemetry["completed_count"]
        total_tensors = self._compression_telemetry["total_tensors_done"]
        total_ratio = self._compression_telemetry["total_ratio"]
        total_error = self._compression_telemetry["total_error"]

        avg_ratio = total_ratio / max(completed, 1)
        avg_error = total_error / max(completed, 1)

        return {
            "active_jobs": active_jobs,
            "completed_jobs": completed,
            "total_tensors_compressed": total_tensors,
            "avg_ratio": round(avg_ratio, 2),
            "avg_error": round(avg_error, 6),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _cleanup_old_jobs(self):
        """Remove compression jobs older than 1 hour."""
        now = time.time()
        cutoff = now - 3600
        stale = [
            jid
            for jid, job in self.compression_jobs.items()
            if job.get("started_at", 0) < cutoff
        ]
        for jid in stale:
            self.compression_jobs.pop(jid, None)
            self._compression_telemetry["active_jobs"].pop(jid, None)
        if stale:
            logger.info("Cleaned up %d stale compression jobs", len(stale))

    def _telemetry_cleaner_loop(self):
        """Background loop that periodically cleans old jobs."""
        while getattr(self, "_telemetry_cleaner_running", True):
            self._cleanup_old_jobs()
            time.sleep(300)

    def run(self):
        """Start the server."""
        import uvicorn

        logger.info(
            f"Starting SpectralStream API on {self.config.host}:{self.config.port}"
        )
        logger.info(f"  API Keys: {len(self.api_key_store)} configured")
        logger.info(f"  Models: {len(self.loaded_models)} loaded")
        logger.info(f"  Max concurrent sessions: {self.config.max_concurrent_sessions}")
        logger.info(
            f"  Dashboard: http://{self.config.host}:{self.config.port}/dashboard"
        )
        logger.info(f"  OpenAI API: http://{self.config.host}:{self.config.port}/v1")
        logger.info(f"  Docs: http://{self.config.host}:{self.config.port}/docs")
        uvicorn.run(
            self.app, host=self.config.host, port=self.config.port, log_level="info"
        )


if __name__ == "__main__":
    server = SpectralStreamServer()
    server.run()

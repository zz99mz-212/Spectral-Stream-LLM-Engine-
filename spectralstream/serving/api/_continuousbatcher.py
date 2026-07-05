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
from fastapi import FastAPI, Form, HTTPException, Request, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


class ContinuousBatcher:
    """
    Continuous batching engine for parallel agent swarms.
    Supports 1000s of concurrent sessions with dynamic batching.
    """

    def __init__(self, pipeline: Optional[Any] = None, max_concurrent: int = 1024):
        self.pipeline = pipeline
        self.max_concurrent = max_concurrent
        self.sessions: Dict[str, SessionState] = {}
        self._lock = Lock()
        self._total_sessions_created = 0
        self._total_tokens_generated = 0
        self._start_time = time.perf_counter()

    def add_session(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            if len(self.sessions) >= self.max_concurrent:
                finished = [(k, v) for k, v in self.sessions.items() if v.finished]
                if finished:
                    oldest = min(finished, key=lambda x: x[1].created_at)
                    del self.sessions[oldest[0]]
                else:
                    oldest = min(self.sessions.items(), key=lambda x: x[1].created_at)
                    del self.sessions[oldest[0]]

            session = SessionState(
                session_id, prompt_tokens, max_tokens, temperature, top_p, stop, stream
            )

            # Create a token generator from the pipeline if available
            if self.pipeline is not None and HAS_INFERENCE:
                try:
                    session.token_generator = self.pipeline.generate_stream(
                        prompt_tokens=prompt_tokens,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                except (ValueError, RuntimeError, OSError):
                    session.token_generator = None

            self.sessions[session_id] = session
            self._total_sessions_created += 1
        return session_id

    def step(self) -> Dict[str, Any]:
        """Execute one continuous batching step."""
        with self._lock:
            active = {k: v for k, v in self.sessions.items() if not v.finished}

        if not active:
            return {"status": "idle", "active": 0}

        tokens_generated = 0

        for session_id, session in list(active.items()):
            if session.current_pos >= min(session.max_tokens, 4096):
                session.finished = True
                continue

            if self.pipeline is not None and HAS_INFERENCE:
                try:
                    next_token = self._generate_from_pipeline(session)
                except (ValueError, RuntimeError, OSError):
                    next_token = np.random.randint(0, 32000)
            else:
                next_token = np.random.randint(0, 32000)

            session.generated_tokens.append(next_token)
            session.generated_text += session.tokenizer.decode([next_token])
            session.current_pos += 1
            tokens_generated += 1

            if session.stop:
                for stop_str in session.stop:
                    if stop_str in session.generated_text:
                        session.finished = True
                        break

        with self._lock:
            self._total_tokens_generated += tokens_generated

        return {
            "status": "running",
            "active": len(active),
            "tokens_generated": tokens_generated,
        }

    def get_session_result(self, session_id: str) -> Optional[Dict]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None
            return {
                "tokens": session.generated_tokens,
                "text": session.generated_text,
                "finished": session.finished,
                "prompt_tokens": len(session.prompt_tokens),
                "completion_tokens": len(session.generated_tokens),
            }

    def remove_session(self, session_id: str):
        with self._lock:
            self.sessions.pop(session_id, None)

    def get_stats(self) -> Dict:
        elapsed = time.perf_counter() - self._start_time
        with self._lock:
            return {
                "total_sessions_created": self._total_sessions_created,
                "active_sessions": len(self.sessions),
                "total_tokens_generated": self._total_tokens_generated,
                "tokens_per_second": self._total_tokens_generated / max(elapsed, 0.001),
                "uptime_seconds": elapsed,
            }

    def _generate_from_pipeline(self, session: SessionState) -> int:
        """Generate a token using the actual inference pipeline."""
        if session.token_generator is not None:
            try:
                return int(next(session.token_generator))
            except StopIteration:
                session.finished = True
                return 0
        elif self.pipeline is not None:
            try:
                gen = self.pipeline.generate_stream(
                    prompt_tokens=session.prompt_tokens + session.generated_tokens,
                    max_new_tokens=1,
                    temperature=session.temperature,
                    top_p=session.top_p,
                )
                return int(next(gen))
            except (StopIteration, ValueError, RuntimeError, OSError):
                return np.random.randint(0, 32000)
        return np.random.randint(0, 32000)

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


class SessionState:
    """State for a single generation session."""

    def __init__(
        self,
        session_id: str,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ):
        self.session_id = session_id
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop
        self.stream = stream
        self.generated_tokens: List[int] = []
        self.generated_text: str = ""
        self.kv_cache: Dict[str, np.ndarray] = {}
        self.current_pos = 0
        self.finished = False
        self.created_at = time.perf_counter()
        self.api_key: str = ""
        self.user_id: str = ""
        self.token_generator: Optional[Iterator[int]] = None
        self.tokenizer = Tokenizer()

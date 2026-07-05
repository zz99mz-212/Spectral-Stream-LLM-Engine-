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


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: List[str] = field(default_factory=list)
    model_path: Optional[str] = None
    max_concurrent_sessions: int = 1024
    max_tokens_per_request: int = 4096
    enable_lan_discovery: bool = True
    enable_dashboard: bool = True
    cache_size_gb: float = 4.0
    kv_cache_size_gb: float = 8.0
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "SpectralStream API",
            "version": "2.0.0",
            "description": "Mind-bending LLM inference engine",
            "author": "SpectralStream R&D",
        }
    )

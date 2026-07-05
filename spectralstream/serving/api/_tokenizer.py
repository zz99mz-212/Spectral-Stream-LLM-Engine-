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


class Tokenizer:
    """Minimal byte-level tokenizer for encoding/decoding text."""

    def encode(self, text: str) -> List[int]:
        return [b for b in text.encode("utf-8")]

    def decode(self, tokens: List[int]) -> str:
        return bytes([max(0, min(t, 255)) for t in tokens]).decode(
            "utf-8", errors="replace"
        )

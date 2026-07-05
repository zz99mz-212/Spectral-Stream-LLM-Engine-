from __future__ import annotations

import hashlib
import logging
import mmap
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from spectralstream.config import SpectralStreamConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    model_id: str
    engine: object
    path: str
    loaded_at: float
    last_used: float
    size_bytes: int = 0
    device: str = "cpu"
    config: Optional[SpectralStreamConfig] = None
    use_count: int = 0


class ModelManager:
    def __init__(
        self,
        max_models: int = 4,
        max_memory_gb: float = 0.0,
        default_config: Optional[SpectralStreamConfig] = None,
    ):
        self._models: dict[str, ModelEntry] = {}
        self._lock = threading.RLock()
        self._max_models = max_models
        self._max_memory_bytes = (
            int(max_memory_gb * 1024**3) if max_memory_gb > 0 else 0
        )
        self._default_config = default_config or SpectralStreamConfig()
        self._total_loaded_bytes = 0

    def load(
        self,
        model_path: str,
        model_id: Optional[str] = None,
        device: str = "cpu",
        config: Optional[SpectralStreamConfig] = None,
        **engine_kwargs,
    ) -> ModelEntry:
        resolved_path = str(Path(model_path).expanduser().resolve())
        if model_id is None:
            model_id = Path(resolved_path).stem

        with self._lock:
            if model_id in self._models:
                entry = self._models[model_id]
                entry.last_used = time.time()
                entry.use_count += 1
                logger.info("model_cache_hit model_id=%s", model_id)
                return entry

            self._evict_if_needed()

            cfg = config or self._default_config
            try:
                from spectralstream.inference import CPUInferenceEngine

                engine = CPUInferenceEngine(model_path=resolved_path, **engine_kwargs)
            except ImportError:
                engine = object()

            size_bytes = self._estimate_model_size(resolved_path)
            entry = ModelEntry(
                model_id=model_id,
                engine=engine,
                path=resolved_path,
                loaded_at=time.time(),
                last_used=time.time(),
                size_bytes=size_bytes,
                device=device,
                config=cfg,
            )

            self._models[model_id] = entry
            self._total_loaded_bytes += size_bytes
            logger.info(
                "model_loaded model_id=%s path=%s size_mb=%.1f device=%s",
                model_id,
                resolved_path,
                size_bytes / 1024**2,
                device,
            )
            return entry

    def get(self, model_id: str) -> Optional[ModelEntry]:
        with self._lock:
            entry = self._models.get(model_id)
            if entry is not None:
                entry.last_used = time.time()
                entry.use_count += 1
            return entry

    def unload(self, model_id: str) -> bool:
        with self._lock:
            entry = self._models.pop(model_id, None)
            if entry is None:
                return False
            self._total_loaded_bytes -= entry.size_bytes
            logger.info("model_unloaded model_id=%s", model_id)
            return True

    def hot_swap(self, model_id: str, new_path: str, **engine_kwargs) -> ModelEntry:
        with self._lock:
            old_entry = self._models.get(model_id)
            old_path = old_entry.path if old_entry else None

            if old_entry:
                self._models.pop(model_id, None)
                self._total_loaded_bytes -= old_entry.size_bytes

            try:
                new_entry = self.load(new_path, model_id=model_id, **engine_kwargs)
                logger.info(
                    "hot_swap_complete model_id=%s old=%s new=%s",
                    model_id,
                    old_path,
                    new_path,
                )
                return new_entry
            except Exception:
                if old_entry:
                    self._models[model_id] = old_entry
                    self._total_loaded_bytes += old_entry.size_bytes
                raise

    def list_models(self) -> list[dict]:
        with self._lock:
            result = []
            for entry in self._models.values():
                result.append(
                    {
                        "id": entry.model_id,
                        "path": entry.path,
                        "device": entry.device,
                        "size_bytes": entry.size_bytes,
                        "loaded_at": entry.loaded_at,
                        "last_used": entry.last_used,
                        "use_count": entry.use_count,
                    }
                )
            return result

    def stats(self) -> dict:
        with self._lock:
            return {
                "loaded_count": len(self._models),
                "max_models": self._max_models,
                "total_loaded_bytes": self._total_loaded_bytes,
                "total_loaded_gb": round(self._total_loaded_bytes / 1024**3, 2),
                "models": [
                    {
                        "id": e.model_id,
                        "size_bytes": e.size_bytes,
                        "use_count": e.use_count,
                    }
                    for e in self._models.values()
                ],
            }

    def _evict_if_needed(self) -> None:
        while len(self._models) >= self._max_models:
            lru_id = min(self._models, key=lambda k: self._models[k].last_used)
            evicted = self._models.pop(lru_id)
            self._total_loaded_bytes -= evicted.size_bytes
            logger.info(
                "model_evicted model_id=%s use_count=%d", lru_id, evicted.use_count
            )

            if self._max_memory_bytes > 0:
                while (
                    self._total_loaded_bytes > self._max_memory_bytes and self._models
                ):
                    lru_id = min(self._models, key=lambda k: self._models[k].last_used)
                    evicted = self._models.pop(lru_id)
                    self._total_loaded_bytes -= evicted.size_bytes
                    logger.info("model_evicted_memory model_id=%s", lru_id)

    @staticmethod
    def _estimate_model_size(path: str) -> int:
        try:
            p = Path(path)
            if p.is_file():
                return p.stat().st_size
            if p.is_dir():
                return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        except OSError:
            pass
        return 0

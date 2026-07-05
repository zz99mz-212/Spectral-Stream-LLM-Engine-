from __future__ import annotations

import gc
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spectralstream.inference.model_config import GenericModelConfig
from spectralstream.format.reader import SSFReader


class UnifiedModelLoader:
    """Single interface for loading tensors from SSF, safetensors, or GGUF.

    Delegates to the correct backend based on file extension and provides
    memoization, architecture auto-detection, and a unified ``load_tensor(name)``
    API.
    """

    def __init__(
        self,
        model_path: str,
        cache_size_gb: float = 2.0,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.cache_size_gb = cache_size_gb
        self.verbose = verbose
        self._lock = threading.Lock()

        self._ssf_reader: Optional[SSFReader] = None
        self._safetensors_loader: Any = None
        self._model_loader: Any = None
        self._gguf_backend: Any = None
        self._tensor_cache: Dict[str, np.ndarray] = {}
        self._tensor_names_cache: Optional[List[str]] = None
        self._model_config: Optional[GenericModelConfig] = None
        self._backend_type: str = "none"

        self._init_backend()

    def _init_backend(self) -> None:
        path = self.model_path

        if path.endswith(".ssf"):
            self._try_ssf(path) or self._try_legacy(path) or self._try_safetensors(path)
        elif path.endswith(".safetensors"):
            self._try_safetensors(path) or self._try_legacy(path)
        elif path.endswith(".gguf"):
            self._try_gguf(path)
        else:
            self._try_safetensors(path) or self._try_legacy(path) or self._try_ssf(path)

        if self._backend_type == "none":
            raise RuntimeError(
                f"UnifiedModelLoader: could not load model from '{path}'. "
                f"Tried SSF, safetensors, and legacy formats."
            )

        self._model_config = self._build_config()

    def _try_ssf(self, path: str) -> bool:
        try:
            self._ssf_reader = SSFReader(path, cache_size=32)
            self._backend_type = "ssf"
            if self.verbose:
                print(f"[UnifiedModelLoader] Loaded SSF: {path}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"[UnifiedModelLoader] SSF failed: {e}")
            return False

    def _try_safetensors(self, path: str) -> bool:
        base = path.rsplit(".", 1)[0] if "." in path else path
        st_path = path if path.endswith(".safetensors") else base + ".safetensors"
        if not os.path.exists(st_path):
            st_path = os.path.join(os.path.dirname(path), "model.safetensors")
            if not os.path.exists(st_path):
                return False
        try:
            from spectralstream.inference.loader import SafeTensorsLoader

            self._safetensors_loader = SafeTensorsLoader(st_path)
            self._backend_type = "safetensors"
            if self.verbose:
                print(f"[UnifiedModelLoader] Loaded safetensors: {st_path}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"[UnifiedModelLoader] safetensors failed: {e}")
            return False

    def _try_legacy(self, path: str) -> bool:
        try:
            from spectralstream.inference.loader import ModelLoader

            self._model_loader = ModelLoader(path, self.cache_size_gb)
            self._backend_type = "legacy"
            if self.verbose:
                print(f"[UnifiedModelLoader] Loaded legacy SSF: {path}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"[UnifiedModelLoader] legacy failed: {e}")
            return False

    def _try_gguf(self, path: str) -> bool:
        try:
            from spectralstream.inference.weight_loader import WeightLoader

            self._gguf_backend = WeightLoader(path)
            self._backend_type = "gguf"
            if self.verbose:
                print(f"[UnifiedModelLoader] Loaded GGUF: {path}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"[UnifiedModelLoader] GGUF failed: {e}")
            return False

    def _build_config(self) -> GenericModelConfig:
        names = self.tensor_names
        cfg = GenericModelConfig.from_tensor_names(names)
        if self._backend_type == "ssf" and self._ssf_reader is not None:
            try:
                md = self._ssf_reader.metadata
                c = md.get("config", {})
                if c:
                    cfg = GenericModelConfig.from_dict(c)
            except Exception:
                pass
        elif self._backend_type == "legacy" and self._model_loader is not None:
            try:
                md = self._model_loader.load_metadata()
                c = md.get("config", {})
                if c:
                    cfg = GenericModelConfig.from_dict(c)
            except Exception:
                pass
        elif (
            self._backend_type == "safetensors" and self._safetensors_loader is not None
        ):
            try:
                md = self._safetensors_loader.load_metadata()
                if md:
                    cfg = GenericModelConfig.from_dict(md)
            except Exception:
                pass

        cfg = cfg or GenericModelConfig.from_ssf_path(self.model_path)
        if cfg is None or cfg.hidden_size == 1536:
            cfg = GenericModelConfig.from_tensor_names(names)
        return cfg

    @property
    def model_config(self) -> GenericModelConfig:
        if self._model_config is None:
            self._model_config = self._build_config()
        return self._model_config

    @property
    def tensor_names(self) -> List[str]:
        if self._tensor_names_cache is not None:
            return self._tensor_names_cache
        names: List[str] = []
        if self._ssf_reader is not None:
            names = self._ssf_reader.tensor_names()
        elif self._safetensors_loader is not None:
            names = self._safetensors_loader.tensor_names
        elif self._model_loader is not None:
            names = self._model_loader.tensor_names
        elif self._gguf_backend is not None:
            names = (
                self._gguf_backend.tensor_names
                if hasattr(self._gguf_backend, "tensor_names")
                else []
            )
        self._tensor_names_cache = names
        return names

    def load_tensor(self, name: str) -> np.ndarray:
        with self._lock:
            if name in self._tensor_cache:
                return self._tensor_cache[name]
            tensor = self._load_tensor_impl(name)
            self._tensor_cache[name] = tensor
            return tensor

    def get_tensor(self, name: str) -> np.ndarray:
        return self.load_tensor(name)

    def _load_tensor_impl(self, name: str) -> np.ndarray:
        if self._ssf_reader is not None:
            return self._ssf_reader.get_tensor(name)
        if self._safetensors_loader is not None:
            return self._safetensors_loader.get_tensor(name)
        if self._model_loader is not None:
            return self._model_loader.get_tensor(name)
        if self._gguf_backend is not None and hasattr(self._gguf_backend, "get_tensor"):
            return self._gguf_backend.get_tensor(name)
        raise KeyError(f"Tensor '{name}' not found and no backend loaded")

    def get_layer_weights(self, layer_idx: int) -> Optional[Dict[str, np.ndarray]]:
        prefix = f"{self.model_config.tensor_prefix}{layer_idx}."
        names = self.tensor_names
        result: Dict[str, np.ndarray] = {}
        for n in names:
            if n.startswith(prefix):
                result[n] = self.load_tensor(n)
        return result if result else None

    def prefetch_layer(self, layer_idx: int) -> None:
        weights = self.get_layer_weights(layer_idx)
        if weights:
            with self._lock:
                for name, tensor in weights.items():
                    self._tensor_cache[name] = tensor

    def clear_cache(self) -> None:
        with self._lock:
            self._tensor_cache.clear()
        gc.collect()

    def close(self) -> None:
        self.clear_cache()
        if self._ssf_reader is not None:
            self._ssf_reader.close()
            self._ssf_reader = None
        if self._safetensors_loader is not None:
            self._safetensors_loader.close()
            self._safetensors_loader = None
        if self._model_loader is not None:
            self._model_loader.close()
            self._model_loader = None
        if self._gguf_backend is not None:
            close_fn = getattr(self._gguf_backend, "close", None)
            if callable(close_fn):
                close_fn()
            self._gguf_backend = None

    def __enter__(self) -> UnifiedModelLoader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"UnifiedModelLoader(path={self.model_path}, "
            f"backend={self._backend_type}, "
            f"arch={self.model_config.architecture})"
        )

import json
import os
import socket
import subprocess
import sys
import time
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from typing import Optional


def _find_models() -> list[dict]:
    home = Path.home()
    search_paths = [
        home / ".lmstudio" / "models",
        home / ".lmstudio" / "models" / "huggingface",
        home / "lmstudio" / "models",
        Path("/usr/local/share/lmstudio/models"),
    ]
    models = []
    for base in search_paths:
        if not base.exists():
            continue
        for f in base.rglob("*.gguf"):
            size_gb = f.stat().st_size / (1024**3) if f.stat().st_size > 0 else 0
            models.append(
                {
                    "path": str(f),
                    "name": f.stem,
                    "size_gb": round(size_gb, 2),
                    "parent": str(f.parent.name),
                    "modified": f.stat().st_mtime,
                }
            )
    return models


def _http_request(
    host: str,
    port: int,
    method: str,
    path: str,
    body: Optional[dict] = None,
    timeout: float = 5.0,
) -> Optional[dict]:
    try:
        conn = HTTPConnection(host, port, timeout=timeout)
        headers = {"Content-Type": "application/json"}
        raw_body = json.dumps(body).encode("utf-8") if body else None
        conn.request(method, path, body=raw_body, headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        conn.close()
        if resp.status >= 200 and resp.status < 300 and data:
            return json.loads(data)
        return None
    except (
        ConnectionRefusedError,
        socket.timeout,
        OSError,
        HTTPException,
        json.JSONDecodeError,
    ):
        return None


class LMStudioManager:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1234,
        lmstudio_binary: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self._binary = lmstudio_binary or self._find_binary()
        self._process: Optional[subprocess.Popen] = None

    @staticmethod
    def _find_binary() -> Optional[str]:
        candidates = [
            "LM Studio",
            "lm-studio",
            "lmstudio",
            "/Applications/LM Studio.app/Contents/MacOS/LM Studio",
            os.path.expanduser("~/.lmstudio/LM Studio"),
            os.path.expanduser("~/lmstudio/LM Studio"),
        ]
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        which = os.environ.get("PATH", "")
        for d in which.split(os.pathsep):
            for c in ["LM Studio", "lm-studio", "lmstudio"]:
                p = os.path.join(d, c)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
        return None

    def is_running(self) -> bool:
        return _http_request(self.host, self.port, "GET", "/v1/models") is not None

    def wait_for_start(self, timeout: float = 30.0, poll_interval: float = 1.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return True
            time.sleep(poll_interval)
        return False

    def start(self, wait: bool = True, timeout: float = 30.0) -> bool:
        if self.is_running():
            return True
        if self._binary:
            try:
                self._process = subprocess.Popen(
                    [self._binary],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        if wait:
            return self.wait_for_start(timeout=timeout)
        return self.is_running()

    def stop(self) -> bool:
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
                self._process = None
                return True
            except Exception:
                try:
                    self._process.kill()
                    self._process = None
                    return True
                except Exception:
                    pass
        return False

    def restart(self, wait: bool = True, timeout: float = 30.0) -> bool:
        self.stop()
        time.sleep(1)
        return self.start(wait=wait, timeout=timeout)

    def get_loaded_model(self) -> Optional[dict]:
        data = _http_request(self.host, self.port, "GET", "/v1/models")
        if data and "data" in data and data["data"]:
            return data["data"][0]
        return None

    def list_available_models(self) -> list[dict]:
        return _find_models()

    def load_model(self, model_path: str) -> bool:
        result = _http_request(
            self.host,
            self.port,
            "POST",
            "/v1/models/load",
            body={"model": model_path},
            timeout=30.0,
        )
        if result is not None:
            time.sleep(2)
            return True
        return False

    def get_server_status(self) -> dict:
        health = _http_request(self.host, self.port, "GET", "/v1/health")
        if health:
            return health
        models = _http_request(self.host, self.port, "GET", "/v1/models")
        return {
            "status": "ok" if models else "unreachable",
            "running": models is not None,
            "model_count": len(models.get("data", [])) if models else 0,
        }

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.port}"


class LMStudioAPIProxy:
    def __init__(self, orchestrator, lmstudio_url: str = "http://127.0.0.1:1234"):
        self.orchestrator = orchestrator
        self.lmstudio_url = lmstudio_url.rstrip("/")
        self._parse_addr()

    def _parse_addr(self):
        parts = (
            self.lmstudio_url.replace("http://", "").replace("https://", "").split(":")
        )
        self._host = parts[0]
        self._port = int(parts[1]) if len(parts) > 1 else 1234

    def _lmstudio_chat(self, messages: list, **kwargs) -> Optional[dict]:
        body = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.8),
            "stream": False,
        }
        if "model" in kwargs:
            body["model"] = kwargs["model"]
        return _http_request(
            self._host,
            self._port,
            "POST",
            "/v1/chat/completions",
            body=body,
            timeout=60,
        )

    def _lmstudio_completion(self, prompt: str, **kwargs) -> Optional[dict]:
        body = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.8),
            "stream": False,
        }
        if "model" in kwargs:
            body["model"] = kwargs["model"]
        return _http_request(
            self._host, self._port, "POST", "/v1/completions", body=body, timeout=60
        )

    def _lmstudio_embeddings(self, input_data, **kwargs) -> Optional[dict]:
        body = {"input": input_data}
        if isinstance(input_data, str):
            body["input"] = [input_data]
        if "model" in kwargs:
            body["model"] = kwargs["model"]
        return _http_request(
            self._host, self._port, "POST", "/v1/embeddings", body=body, timeout=30
        )

    def chat_completion(self, messages: list, **kwargs) -> dict:
        accelerator = getattr(self.orchestrator, "accelerator", None)
        confidence = None
        accelerated_text = None
        fallback_used = False

        if accelerator is not None:
            prompt_str = (
                self._messages_to_prompt(messages)
                if hasattr(self, "_messages_to_prompt")
                else str(messages)
            )
            try:
                result = accelerator.generate(
                    prompt_str,
                    max_new_tokens=kwargs.get("max_tokens", 256),
                    temperature=kwargs.get("temperature", 0.8),
                )
                if isinstance(result, tuple):
                    accelerated_text = result[0]
                    confidence = 0.8
                elif isinstance(result, dict):
                    accelerated_text = result.get("text", "")
                    confidence = result.get("confidence", 0.5)
            except Exception:
                pass

        if accelerated_text is None or (confidence is not None and confidence < 0.5):
            lm_result = self._lmstudio_chat(messages, **kwargs)
            fallback_used = True
            if lm_result is not None:
                choices = lm_result.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    text = msg.get("content", "")
                    if accelerated_text and text:
                        self._learn_correction(accelerated_text, text)
                    return {
                        "id": lm_result.get("id", ""),
                        "object": "chat.completion",
                        "created": lm_result.get("created", int(time.time())),
                        "model": lm_result.get("model", "lm-studio"),
                        "choices": choices,
                        "usage": lm_result.get("usage", {}),
                        "accelerated": False,
                        "fallback": True,
                    }
            return self._empty_chat_response("Acceleration and fallback both failed")

        return {
            "id": f"chatcmpl-acc-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "spectralstream-accelerated",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": accelerated_text
                        if isinstance(accelerated_text, str)
                        else str(accelerated_text),
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(accelerated_text.split())
                if isinstance(accelerated_text, str)
                else 0,
                "total_tokens": 0,
            },
            "accelerated": True,
            "confidence": confidence,
            "fallback": False,
        }

    def text_completion(self, prompt: str, **kwargs) -> dict:
        accelerator = getattr(self.orchestrator, "accelerator", None)
        accelerated_text = None
        fallback_used = False

        if accelerator is not None:
            try:
                result = accelerator.generate(
                    prompt,
                    max_new_tokens=kwargs.get("max_tokens", 256),
                    temperature=kwargs.get("temperature", 0.8),
                )
                if isinstance(result, tuple):
                    accelerated_text = result[0]
                elif isinstance(result, dict):
                    accelerated_text = result.get("text", "")
            except Exception:
                pass

        if accelerated_text is None:
            lm_result = self._lmstudio_completion(prompt, **kwargs)
            fallback_used = True
            if lm_result is not None:
                choices = lm_result.get("choices", [])
                return {
                    "id": lm_result.get("id", ""),
                    "object": "text_completion",
                    "created": lm_result.get("created", int(time.time())),
                    "model": lm_result.get("model", "lm-studio"),
                    "choices": choices,
                    "usage": lm_result.get("usage", {}),
                    "accelerated": False,
                    "fallback": True,
                }
            return self._empty_completion_response()

        return {
            "id": f"cmpl-acc-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "spectralstream-accelerated",
            "choices": [
                {
                    "index": 0,
                    "text": accelerated_text
                    if isinstance(accelerated_text, str)
                    else str(accelerated_text),
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(accelerated_text.split())
                if isinstance(accelerated_text, str)
                else 0,
                "total_tokens": 0,
            },
            "accelerated": True,
            "fallback": False,
        }

    def embeddings(self, input_data, **kwargs) -> dict:
        lm_result = self._lmstudio_embeddings(input_data, **kwargs)
        if lm_result is not None:
            return lm_result
        return {
            "object": "list",
            "data": [],
            "model": kwargs.get("model", "lm-studio"),
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    def _learn_correction(self, draft: str, correct: str):
        learner = getattr(self.orchestrator, "online_learning", None)
        if learner is None:
            return
        if not draft or not correct:
            return
        pairs = list(
            zip(
                [ord(c) % 32000 for c in draft[:64]],
                [ord(c) % 32000 for c in correct[:64]],
            )
        )
        if pairs:
            try:
                learner.ingest_batch(pairs)
            except Exception:
                pass

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text_parts.append(c.get("text", ""))
                content = " ".join(text_parts)
            parts.append(f"<|{role}|>\n{content}\n<|end|>")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    @staticmethod
    def _empty_chat_response(reason: str = "") -> dict:
        return {
            "id": "chatcmpl-empty",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "none",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"[Error: {reason}]"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "accelerated": False,
            "fallback": False,
        }

    @staticmethod
    def _empty_completion_response(reason: str = "") -> dict:
        return {
            "id": "cmpl-empty",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "none",
            "choices": [
                {
                    "index": 0,
                    "text": f"[Error: {reason}]",
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "accelerated": False,
            "fallback": False,
        }


class ModelHotReloader:
    def __init__(
        self,
        hd_engine,
        confidence_gate,
        lmstudio: LMStudioManager,
        poll_interval: float = 5.0,
    ):
        self.hd_engine = hd_engine
        self.confidence_gate = confidence_gate
        self.lmstudio = lmstudio
        self.poll_interval = poll_interval
        self._last_model_id: Optional[str] = None
        self._last_check: float = 0.0
        self._running = False

    def check_for_changes(self) -> bool:
        model_info = self.lmstudio.get_loaded_model()
        if model_info is None:
            return False

        current_id = model_info.get("id", "") or model_info.get("root", "")
        if not current_id:
            return False

        if self._last_model_id is None:
            self._last_model_id = current_id
            return False

        if current_id != self._last_model_id:
            self._last_model_id = current_id
            return True

        return False

    def reload_engine(self, new_model_path: str) -> bool:
        try:
            new_vocab = self._detect_vocab_size(new_model_path)
            if new_vocab and hasattr(self.hd_engine, "reset_vocab"):
                self.hd_engine.reset_vocab(new_vocab)
            if hasattr(self.hd_engine, "reset"):
                self.hd_engine.reset()
            self._last_model_id = str(Path(new_model_path).stem)
            return True
        except Exception:
            return False

    def start_polling(self):
        self._running = True
        while self._running:
            time.sleep(self.poll_interval)
            try:
                if self.check_for_changes():
                    model = self.lmstudio.get_loaded_model()
                    if model:
                        model_root = model.get("root", "") or model.get("id", "")
                        if model_root:
                            self.reload_engine(model_root)
            except Exception:
                pass

    def stop_polling(self):
        self._running = False

    @staticmethod
    def _detect_vocab_size(model_path: str) -> Optional[int]:
        try:
            from gguf import GGUFReader

            r = GGUFReader(model_path, "r")
            tok_field = r.fields.get("tokenizer.ggml.tokens")
            if tok_field is not None:
                data = tok_field.parts[-1]
                return len(data)
            return None
        except Exception:
            return None

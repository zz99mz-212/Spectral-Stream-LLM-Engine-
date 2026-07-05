from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    id: str
    object: str
    created: int
    model: str
    choices: list[dict]
    usage: Optional[dict] = None


@dataclass
class StreamState:
    request_id: str
    model: str
    created: int
    tokens_generated: int = 0
    prompt_tokens: int = 0
    finished: bool = False
    finish_reason: Optional[str] = None
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def completion_id(self) -> str:
        return self.request_id


class TokenStreamer:
    def __init__(
        self,
        model: str = "spectralstream",
        include_usage: bool = False,
    ):
        self.model = model
        self.include_usage = include_usage

    async def stream_chat(
        self,
        engine: Any,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 0,
        stop: Optional[list[str]] = None,
        n: int = 1,
        tools: Optional[list] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> AsyncIterator[str]:
        state = StreamState(
            request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=self.model,
            created=int(time.time()),
        )

        loop = asyncio.get_event_loop()

        for choice_idx in range(n):
            role_sent = False

            token_ids = await loop.run_in_executor(
                None, lambda: self._tokenize(engine, prompt)
            )
            state.prompt_tokens = len(token_ids)

            for i in range(max_tokens):
                if state.abort_event.is_set():
                    break

                new_token = await loop.run_in_executor(
                    None,
                    lambda ids=token_ids: self._generate_token(
                        engine,
                        ids,
                        temperature,
                        top_p,
                        top_k,
                        frequency_penalty,
                        presence_penalty,
                    ),
                )

                if new_token is None:
                    break

                token_ids.append(new_token)
                text = await loop.run_in_executor(
                    None, lambda t=new_token: self._detokenize(engine, [t])
                )
                state.tokens_generated += 1

                if not role_sent and not tools:
                    yield format_sse(
                        StreamChunk(
                            id=state.completion_id,
                            object="chat.completion.chunk",
                            created=state.created,
                            model=self.model,
                            choices=[
                                {
                                    "index": choice_idx,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None,
                                }
                            ],
                        )
                    )
                    role_sent = True

                stopped = False
                if stop:
                    text, stopped = self._check_stop(text, stop)

                yield format_sse(
                    StreamChunk(
                        id=state.completion_id,
                        object="chat.completion.chunk",
                        created=state.created,
                        model=self.model,
                        choices=[
                            {
                                "index": choice_idx,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    )
                )

                if stopped:
                    state.finish_reason = "stop"
                    break

            if state.finish_reason is None:
                state.finish_reason = (
                    "length" if state.tokens_generated >= max_tokens else "stop"
                )

            yield format_sse(
                StreamChunk(
                    id=state.completion_id,
                    object="chat.completion.chunk",
                    created=state.created,
                    model=self.model,
                    choices=[
                        {
                            "index": choice_idx,
                            "delta": {},
                            "finish_reason": state.finish_reason,
                        }
                    ],
                )
            )

        if self.include_usage:
            yield format_sse(
                StreamChunk(
                    id=state.completion_id,
                    object="chat.completion.chunk",
                    created=state.created,
                    model=self.model,
                    choices=[],
                    usage={
                        "prompt_tokens": state.prompt_tokens,
                        "completion_tokens": state.tokens_generated,
                        "total_tokens": state.prompt_tokens + state.tokens_generated,
                    },
                )
            )

        yield "data: [DONE]\n\n"

    async def stream_completion(
        self,
        engine: Any,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 0,
        stop: Optional[list[str]] = None,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> AsyncIterator[str]:
        state = StreamState(
            request_id=f"cmpl-{uuid.uuid4().hex[:12]}",
            model=self.model,
            created=int(time.time()),
        )

        loop = asyncio.get_event_loop()

        token_ids = await loop.run_in_executor(
            None, lambda: self._tokenize(engine, prompt)
        )
        state.prompt_tokens = len(token_ids)

        if echo:
            yield format_sse(
                {
                    "id": state.completion_id,
                    "object": "text_completion",
                    "created": state.created,
                    "model": self.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": prompt,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )

        generated_text = ""
        for i in range(max_tokens):
            if state.abort_event.is_set():
                break

            new_token = await loop.run_in_executor(
                None,
                lambda ids=token_ids: self._generate_token(
                    engine,
                    ids,
                    temperature,
                    top_p,
                    top_k,
                    frequency_penalty,
                    presence_penalty,
                ),
            )

            if new_token is None:
                break

            token_ids.append(new_token)
            text = await loop.run_in_executor(
                None, lambda t=new_token: self._detokenize(engine, [t])
            )
            generated_text += text
            state.tokens_generated += 1

            stopped = False
            if stop:
                clean, stopped = self._check_stop(generated_text, stop)
                if stopped:
                    text = clean[len(generated_text) - len(text) :]

            yield format_sse(
                {
                    "id": state.completion_id,
                    "object": "text_completion",
                    "created": state.created,
                    "model": self.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )

            if stopped:
                state.finish_reason = "stop"
                break

        if state.finish_reason is None:
            state.finish_reason = (
                "length" if state.tokens_generated >= max_tokens else "stop"
            )

        yield format_sse(
            {
                "id": state.completion_id,
                "object": "text_completion",
                "created": state.created,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "logprobs": None,
                        "finish_reason": state.finish_reason,
                    }
                ],
            }
        )

        if self.include_usage:
            yield format_sse(
                {
                    "id": state.completion_id,
                    "object": "text_completion",
                    "created": state.created,
                    "model": self.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": state.prompt_tokens,
                        "completion_tokens": state.tokens_generated,
                        "total_tokens": state.prompt_tokens + state.tokens_generated,
                    },
                }
            )

        yield "data: [DONE]\n\n"

    def _tokenize(self, engine: Any, text: str) -> list[int]:
        try:
            return engine.tokenize(text)
        except Exception:
            return [min(ord(c) % 32000, 31999) for c in text[:512]]

    def _detokenize(self, engine: Any, token_ids: list[int]) -> str:
        return "".join(chr(t) if 32 <= t <= 126 else f"<{t}>" for t in token_ids)

    def _generate_token(
        self,
        engine: Any,
        token_ids: list[int],
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> Optional[int]:
        try:
            next_ids, _ = engine.generate(
                token_ids, max_new_tokens=1, temperature=temperature
            )
            if len(next_ids) > len(token_ids):
                return next_ids[-1]
            return None
        except Exception:
            return None

    @staticmethod
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


def format_sse(data: dict | StreamChunk | str) -> str:
    if isinstance(data, StreamChunk):
        payload = {
            "id": data.id,
            "object": data.object,
            "created": data.created,
            "model": data.model,
            "choices": data.choices,
        }
        if data.usage is not None:
            payload["usage"] = data.usage
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    if isinstance(data, str):
        return data if data.startswith("data:") else f"data: {data}\n\n"
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

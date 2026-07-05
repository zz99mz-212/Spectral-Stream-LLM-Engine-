import json
import re
import time
import uuid
from typing import Optional, Callable


_RE_TOOL_CALL_START = re.compile(r'\{"function"\s*:\s*"')
_RE_TOOL_CALL_OPEN = re.compile(
    r'\{\s*"function"\s*:\s*"(?P<name>[^"]*)"\s*,\s*"arguments"\s*:\s*\{'
)


def find_tool_call_start(text: str) -> Optional[tuple[int, str]]:
    m = _RE_TOOL_CALL_OPEN.search(text)
    if not m:
        return None
    start = m.start()
    name = m.group("name")
    return start, name


def parse_tool_call(text: str) -> Optional[dict]:
    m = _RE_TOOL_CALL_START.search(text)
    if not m:
        return None
    start = m.start()
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    if "function" in obj and "arguments" in obj:
                        return obj
                except (json.JSONDecodeError, ValueError):
                    pass
                break
    return None


def build_tool_call_chunk(func_name: str, arguments: str, index: int = 0) -> dict:
    return {
        "index": index,
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": func_name,
            "arguments": arguments,
        },
    }


class StreamingHandler:
    def __init__(
        self,
        wfile,
        engine,
        detokenize_fn: Callable[[list[int]], str],
        tokenize_fn: Callable[[str], list[int]],
    ):
        self.wfile = wfile
        self.engine = engine
        self._detokenize = detokenize_fn
        self._tokenize = tokenize_fn

        self.generated_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.cmpl_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        self.created = int(time.time())
        self.model_name = ""

    def start_sse(self, send_headers_fn: Callable):
        send_headers_fn()

    def _send(self, data: dict):
        chunk = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.wfile.write(chunk.encode("utf-8"))
        self.wfile.flush()

    def _done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _error(self, msg: str):
        self._send(
            {
                "error": {"message": msg, "type": "stream_error", "code": 500},
            }
        )
        self._done()

    def _chat_chunk(self, delta: dict, finish_reason: Optional[str] = None) -> dict:
        return {
            "id": self.generated_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_name,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason},
            ],
        }

    def _usage_chunk(self, prompt_tokens: int, completion_tokens: int) -> dict:
        return {
            "id": self.generated_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_name,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def stream_chat(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        tools: Optional[list] = None,
        stop: Optional[list[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
    ):
        self.model_name = model_name
        try:
            self._stream_chat_tokens(
                prompt, max_tokens, temperature, tools, stop, logprobs, top_logprobs
            )
        except Exception as exc:
            self._error(str(exc))

    def stream_completion(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        echo: bool = False,
    ):
        self.model_name = model_name
        try:
            self._stream_completion_tokens(
                prompt, max_tokens, temperature, stop, logprobs, top_logprobs, echo
            )
        except Exception as exc:
            self._error(str(exc))

    def _stream_chat_tokens(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: Optional[list],
        stop: Optional[list[str]],
        logprobs: bool,
        top_logprobs: int,
    ):
        token_ids = list(self._tokenize(prompt))
        input_len = len(token_ids)

        self._send(self._chat_chunk({"role": "assistant"}))

        text_buffer = ""
        tool_call_name = None
        tool_call_args_start = 0
        tool_call_active = False
        content_before_tool = ""

        for _ in range(max_tokens):
            tokens, _ = self.engine.generate(
                token_ids,
                max_new_tokens=1,
                temperature=temperature,
            )
            if len(tokens) <= len(token_ids):
                break
            new_id = tokens[-1]
            token_ids.append(new_id)
            text = self._detokenize([new_id])
            text_buffer += text

            if stop:
                clean_text, stopped = _check_stop(text_buffer, stop)
                if stopped:
                    text_buffer = clean_text
                    break

            if not tool_call_active:
                result = find_tool_call_start(text_buffer)
                if result is not None:
                    start_idx, name = result
                    tool_call_active = True
                    tool_call_name = name
                    tool_call_args_start = len(text_buffer)

                    content_prefix = text_buffer[:start_idx].rstrip(",")
                    if content_prefix:
                        content_before_tool = content_prefix
                        self._send(self._chat_chunk({"content": content_prefix}))

                    self._send(
                        self._chat_chunk(
                            {
                                "tool_calls": [
                                    build_tool_call_chunk(tool_call_name, "", 0)
                                ],
                            }
                        )
                    )

                    name_end = start_idx + text[start_idx:].find(name) + len(name)
                    after_name = text_buffer[name_end:]
                    brace_pos = after_name.find("{")
                    if brace_pos >= 0:
                        args_text = after_name[brace_pos:]
                        if args_text:
                            self._send(
                                self._chat_chunk(
                                    {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {"arguments": args_text},
                                            }
                                        ],
                                    }
                                )
                            )
                    continue

            if tool_call_active:
                if text:
                    self._send(
                        self._chat_chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": text},
                                    }
                                ],
                            }
                        )
                    )
            else:
                if text:
                    self._send(self._chat_chunk({"content": text}))

        completion_len = len(token_ids) - input_len
        if tool_call_active:
            finish = "tool_calls"
        elif completion_len >= max_tokens:
            finish = "length"
        else:
            finish = "stop"

        self._send(self._chat_chunk({}, finish_reason=finish))
        self._send(self._usage_chunk(input_len, completion_len))
        self._done()

    def _stream_completion_tokens(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]],
        logprobs: bool,
        top_logprobs: int,
        echo: bool,
    ):
        token_ids = list(self._tokenize(prompt))
        input_len = len(token_ids)

        if echo:
            self._send(
                {
                    "id": self.cmpl_id,
                    "object": "text_completion",
                    "created": self.created,
                    "model": self.model_name,
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

        text_buffer = ""
        for _ in range(max_tokens):
            tokens, _ = self.engine.generate(
                token_ids,
                max_new_tokens=1,
                temperature=temperature,
            )
            if len(tokens) <= len(token_ids):
                break
            new_id = tokens[-1]
            token_ids.append(new_id)
            text = self._detokenize([new_id])
            text_buffer += text

            if stop:
                _, stopped = _check_stop(text_buffer, stop)
                if stopped:
                    break

            self._send(
                {
                    "id": self.cmpl_id,
                    "object": "text_completion",
                    "created": self.created,
                    "model": self.model_name,
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

        completion_len = len(token_ids) - input_len
        finish = "length" if completion_len >= max_tokens else "stop"
        self._send(
            {
                "id": self.cmpl_id,
                "object": "text_completion",
                "created": self.created,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "logprobs": None,
                        "finish_reason": finish,
                    }
                ],
                "usage": {
                    "prompt_tokens": input_len,
                    "completion_tokens": completion_len,
                    "total_tokens": input_len + completion_len,
                },
            }
        )
        self._done()


def _check_stop(text: str, stop: list[str]) -> tuple[str, bool]:
    if isinstance(stop, str):
        stop = [stop]
    for seq in stop:
        idx = text.find(seq)
        if idx != -1:
            return text[:idx], True
    return text, False

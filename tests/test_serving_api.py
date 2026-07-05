import sys

import pytest

pytest.importorskip("fastapi")

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.serving.api import (
        ServerConfig,
        ChatMessage,
        ChatCompletionRequest,
        CompletionRequest,
        ModelInfo,
        CompressionRequest,
        ContinuousBatcher,
        Tokenizer,
        SessionState,
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestServerConfig:
    def test_default_config(self):
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.max_concurrent_sessions == 1024
        assert cfg.enable_dashboard is True

    def test_custom_config(self):
        cfg = ServerConfig(host="127.0.0.1", port=9000, max_concurrent_sessions=64)
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9000
        assert cfg.max_concurrent_sessions == 64


class TestTokenizer:
    def test_encode_decode(self):
        tokenizer = Tokenizer()
        text = "Hello, World!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_encode_empty(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.encode("")
        assert tokens == []

    def test_decode_empty(self):
        tokenizer = Tokenizer()
        assert tokenizer.decode([]) == ""

    def test_decode_clamped(self):
        tokenizer = Tokenizer()
        result = tokenizer.decode([72, 101, 108, 108, 111])
        assert "Hello" in result


class TestSessionState:
    def test_initialization(self):
        session = SessionState(
            session_id="test-123",
            prompt_tokens=[1, 2, 3],
            max_tokens=100,
            temperature=0.7,
            top_p=0.95,
        )
        assert session.session_id == "test-123"
        assert session.prompt_tokens == [1, 2, 3]
        assert session.max_tokens == 100
        assert session.finished is False

    def test_default_stop(self):
        session = SessionState(
            session_id="s1",
            prompt_tokens=[1],
            max_tokens=10,
            temperature=0.5,
            top_p=0.9,
        )
        assert session.stop is None

    def test_with_stop_and_stream(self):
        session = SessionState(
            session_id="s2",
            prompt_tokens=[1, 2],
            max_tokens=50,
            temperature=0.8,
            top_p=0.9,
            stop=["\n"],
            stream=True,
        )
        assert session.stop == ["\n"]
        assert session.stream is True


class TestContinuousBatcher:
    def test_add_session(self):
        batcher = ContinuousBatcher(max_concurrent=10)
        sid = batcher.add_session(
            prompt_tokens=[1, 2, 3],
            max_tokens=10,
            temperature=0.7,
            top_p=0.95,
        )
        assert sid is not None
        assert len(sid) > 0

    def test_step_idle(self):
        batcher = ContinuousBatcher(max_concurrent=10)
        result = batcher.step()
        assert result["status"] == "idle"
        assert result["active"] == 0

    def test_step_active(self):
        batcher = ContinuousBatcher(max_concurrent=10)
        batcher.add_session(
            prompt_tokens=[1, 2, 3],
            max_tokens=10,
            temperature=0.7,
            top_p=0.95,
        )
        result = batcher.step()
        assert result["status"] == "running"
        assert result["tokens_generated"] >= 1

    def test_get_session_result(self):
        batcher = ContinuousBatcher(max_concurrent=10)
        sid = batcher.add_session(
            prompt_tokens=[1, 2, 3],
            max_tokens=5,
            temperature=0.7,
            top_p=0.95,
        )
        for _ in range(3):
            batcher.step()
        result = batcher.get_session_result(sid)
        assert result is not None
        assert "tokens" in result
        assert "text" in result
        assert len(result["tokens"]) > 0

    def test_remove_session(self):
        batcher = ContinuousBatcher(max_concurrent=10)
        sid = batcher.add_session(
            prompt_tokens=[1, 2, 3],
            max_tokens=5,
            temperature=0.7,
            top_p=0.95,
        )
        batcher.remove_session(sid)
        assert batcher.get_session_result(sid) is None

    def test_get_stats(self):
        batcher = ContinuousBatcher(max_concurrent=10)
        batcher.add_session(
            prompt_tokens=[1, 2, 3],
            max_tokens=5,
            temperature=0.7,
            top_p=0.95,
        )
        batcher.step()
        stats = batcher.get_stats()
        assert stats["total_sessions_created"] >= 1
        assert stats["total_tokens_generated"] >= 0
        assert "uptime_seconds" in stats

    def test_max_concurrent_eviction(self):
        batcher = ContinuousBatcher(max_concurrent=2)
        for i in range(3):
            batcher.add_session(
                prompt_tokens=[i],
                max_tokens=100,
                temperature=0.7,
                top_p=0.95,
            )
        assert len(batcher.sessions) <= 2


class TestModelInfo:
    def test_model_info_creation(self):
        info = ModelInfo(id="test-model", created=1234567890)
        assert info.id == "test-model"
        assert info.object == "model"
        assert info.owned_by == "spectralstream"


class TestChatCompletionRequest:
    def test_default_request(self):
        msg = ChatMessage(role="user", content="Hello")
        req = ChatCompletionRequest(messages=[msg])
        assert req.model == "default"
        assert req.temperature == 0.7
        assert req.max_tokens == 1024
        assert req.stream is False

    def test_custom_request(self):
        msgs = [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="Hi"),
        ]
        req = ChatCompletionRequest(
            model="gemma-4",
            messages=msgs,
            temperature=0.9,
            max_tokens=2048,
            stream=True,
        )
        assert req.model == "gemma-4"
        assert req.temperature == 0.9
        assert len(req.messages) == 2


class TestCompressionRequest:
    def test_default_request(self):
        req = CompressionRequest(model_path="/tmp/model.safetensors")
        assert req.target_ratio == 5000.0
        assert req.max_error == 0.0002
        assert req.output_path is None

    def test_custom_request(self):
        req = CompressionRequest(
            model_path="/tmp/model.safetensors",
            target_ratio=1000.0,
            max_error=0.001,
            output_path="/tmp/out.ssf",
        )
        assert req.target_ratio == 1000.0
        assert req.output_path == "/tmp/out.ssf"

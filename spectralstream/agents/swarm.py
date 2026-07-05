import heapq
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np


class Priority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=True)
class AgentRequest:
    priority: int
    timestamp: float = field(compare=False)
    agent_id: str = field(compare=False)
    prompt: str = field(compare=False)
    max_tokens: int = field(compare=False, default=256)
    callback: Optional[Callable] = field(compare=False, default=None)
    context: dict = field(compare=False, default_factory=dict)


class RateLimiter:
    def __init__(self, tokens_per_minute: int = 100000):
        self.capacity = tokens_per_minute
        self.tokens = float(tokens_per_minute)
        self.refill_rate = tokens_per_minute / 60.0
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def reset(self):
        with self.lock:
            self.tokens = float(self.capacity)
            self.last_refill = time.time()


class ContinuousBatcher:
    def __init__(self, max_batch_size: int = 16, max_wait_ms: float = 50.0):
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self._queue: list[AgentRequest] = []
        self.lock = threading.Lock()
        self.batch_count = 0
        self.avg_batch_size = 0.0

    def submit(self, request: AgentRequest):
        with self.lock:
            heapq.heappush(self._queue, request)

    def get_batch(self) -> list[AgentRequest]:
        with self.lock:
            if not self._queue:
                return []

        start = time.time()
        while True:
            with self.lock:
                qsize = len(self._queue)
            if qsize >= self.max_batch_size:
                break
            if time.time() - start >= self.max_wait:
                break
            time.sleep(0.001)

        with self.lock:
            batch = []
            while self._queue and len(batch) < self.max_batch_size:
                batch.append(heapq.heappop(self._queue))

            if batch:
                self.batch_count += 1
                self.avg_batch_size = self.avg_batch_size * 0.9 + len(batch) * 0.1

            return batch

    @property
    def pending(self) -> int:
        with self.lock:
            return len(self._queue)

    def clear(self):
        with self.lock:
            self._queue.clear()

    def __len__(self) -> int:
        with self.lock:
            return len(self._queue)


class BatchHDCVerifier:
    def __init__(self, model_fn: Optional[Callable] = None):
        self.model_fn = model_fn
        self.total_verified = 0
        self.total_accepted = 0
        self.batch_verify_count = 0

    def verify_batch(self, agents: list[dict], hd_engine) -> list[list[int]]:
        self.batch_verify_count += 1
        results = []

        for agent in agents:
            context = agent.get("context", [])
            draft = agent.get("draft", [])

            if not draft:
                results.append([])
                continue

            hd_scores = []
            for i, token in enumerate(draft):
                sim = self._hdc_similarity(hd_engine, context + draft[:i], token)
                hd_scores.append(sim)

            accepted = []
            for i, (token, score) in enumerate(zip(draft, hd_scores)):
                if score > 0.3:
                    accepted.append(token)
                    self.total_accepted += 1
                else:
                    break

            self.total_verified += len(draft)
            results.append(accepted)

        return results

    def _hdc_similarity(self, hd_engine, context: list[int], token: int) -> float:
        try:
            if hasattr(hd_engine, "hd") and hasattr(hd_engine.hd, "predict_next"):
                candidates = hd_engine.hd.predict_next(tuple(context[-16:]))
                for t, score in candidates:
                    if t == token:
                        return score
            return 0.5
        except Exception:
            return 0.5

    def verify_with_model(
        self, agents: list[dict], model_fn: Callable
    ) -> list[list[int]]:
        if not agents:
            return []

        total_draft_tokens = sum(len(a.get("draft", [])) for a in agents)
        if total_draft_tokens == 0:
            return [[] for _ in agents]

        combined_input = []
        agent_spans = []
        for agent in agents:
            ctx = agent.get("context", [])
            draft = agent.get("draft", [])
            span_start = len(combined_input)
            combined_input.extend(ctx)
            combined_input.extend(draft)
            agent_spans.append((span_start, len(ctx), len(draft)))

        try:
            logits, _, _ = model_fn(combined_input, past=None)
        except Exception:
            return [[] for _ in agents]

        results = []
        for span_start, ctx_len, draft_len in agent_spans:
            draft = []
            for i in range(draft_len):
                pos = span_start + ctx_len + i
                if pos < len(logits):
                    step_logits = (
                        logits[pos]
                        if hasattr(logits, "ndim") and logits.ndim > 1
                        else logits
                    )
                    token_id = combined_input[span_start + ctx_len + i]
                    probs = self._softmax(step_logits)
                    if token_id < len(probs) and probs[token_id] > 0.01:
                        draft.append(token_id)
                        self.total_accepted += 1
                    else:
                        break
                else:
                    break
            self.total_verified += draft_len
            results.append(draft)

        return results

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        if hasattr(logits, "ndim") and logits.ndim > 1:
            logits = logits[-1]
        logits = np.asarray(logits, dtype=np.float64)
        max_l = np.max(logits)
        exp = np.exp(logits - max_l)
        return exp / np.sum(exp)

    def acceptance_rate(self) -> float:
        return self.total_accepted / max(self.total_verified, 1)

    def reset(self):
        self.total_verified = 0
        self.total_accepted = 0
        self.batch_verify_count = 0


class AgentSwarmEngine:
    def __init__(self, orchestrator, max_agents: int = 32, target_tok_s: int = 5000):
        self.orchestrator = orchestrator
        self.max_agents = max_agents
        self.target_tok_s = target_tok_s

        self.batcher = ContinuousBatcher(max_batch_size=16)
        self.rate_limiter = RateLimiter(tokens_per_minute=target_tok_s * 60)
        self.verifier = BatchHDCVerifier()

        self.agents: dict[str, dict] = {}
        self.running = False
        self.stats = {
            "total_tokens": 0,
            "total_time": 0.0,
            "batches_processed": 0,
            "hdc_accepted": 0,
            "model_calls": 0,
        }
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def register_agent(self, agent_id: str, prompt: str):
        context_tokens = self.orchestrator.tokenize(prompt)
        with self._lock:
            self.agents[agent_id] = {
                "id": agent_id,
                "prompt": prompt,
                "context": context_tokens,
                "generated": [],
                "hdc_confidence": 0.8,
                "state": "active",
                "tokens_generated": 0,
            }

    def unregister_agent(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                return True
            return False

    def submit_request(
        self,
        agent_id: str,
        max_tokens: int = 256,
        callback: Optional[Callable] = None,
        priority: Priority = Priority.NORMAL,
    ) -> bool:
        with self._lock:
            if agent_id not in self.agents:
                return False
            prompt = self.agents[agent_id]["prompt"]

        req = AgentRequest(
            priority=priority.value,
            timestamp=time.time(),
            agent_id=agent_id,
            prompt=prompt,
            max_tokens=max_tokens,
            callback=callback,
        )
        self.batcher.submit(req)
        return True

    def _process_batch(self, batch: list[AgentRequest]) -> list[dict]:
        results = []

        drafts = []
        for req in batch:
            with self._lock:
                agent = self.agents.get(req.agent_id)
            if not agent:
                continue

            context = agent["context"]
            hd_engine = self.orchestrator.hd_engine

            draft_block = []
            if hasattr(hd_engine, "draft_block"):
                blocks = hd_engine.draft_block(block_size=min(16, req.max_tokens))
                if blocks and blocks[0]:
                    draft_block = blocks[0]

            if not draft_block and hasattr(hd_engine.hd, "generate_block"):
                draft_block = hd_engine.hd.generate_block(
                    tuple(context[-64:]),
                    block_size=min(8, req.max_tokens),
                    temperature=0.8,
                )

            drafts.append(
                {
                    "agent_id": req.agent_id,
                    "context": context,
                    "draft": draft_block,
                    "request": req,
                }
            )

        verified_blocks = self.verifier.verify_batch(
            drafts, self.orchestrator.hd_engine
        )

        for draft_info, verified in zip(drafts, verified_blocks):
            with self._lock:
                agent = self.agents.get(draft_info["agent_id"])
            if not agent:
                continue

            req = draft_info["request"]

            if verified:
                with self._lock:
                    agent["generated"].extend(verified)
                    agent["tokens_generated"] += len(verified)
                    agent["context"] = agent["context"] + verified
                self.stats["hdc_accepted"] += len(verified)

                for token in verified:
                    if hasattr(self.orchestrator, "hd_engine"):
                        self.orchestrator.hd_engine.observe(token)

                results.append(
                    {
                        "agent_id": draft_info["agent_id"],
                        "tokens": verified,
                        "accepted": True,
                        "hdc": True,
                    }
                )
            else:
                if self.rate_limiter.acquire(1):
                    try:
                        model_result = self.orchestrator.generate(
                            agent["prompt"],
                            max_new_tokens=min(4, req.max_tokens),
                            temperature=0.8,
                        )
                        full_output = (
                            model_result[0]
                            if isinstance(model_result, tuple)
                            else model_result.get("tokens", [])
                        )
                    except Exception:
                        full_output = []

                    with self._lock:
                        new_tokens = full_output[len(agent["context"]) :]
                        agent["generated"].extend(new_tokens)
                        agent["tokens_generated"] += len(new_tokens)
                        agent["context"] = agent["context"] + new_tokens
                    self.stats["model_calls"] += 1

                    for token in new_tokens:
                        if hasattr(self.orchestrator, "hd_engine"):
                            self.orchestrator.hd_engine.observe(token)

                    results.append(
                        {
                            "agent_id": draft_info["agent_id"],
                            "tokens": new_tokens,
                            "accepted": True,
                            "hdc": False,
                        }
                    )

                    if req.callback and new_tokens:
                        req.callback(new_tokens)

        self.stats["batches_processed"] += 1
        return results

    def start(self):
        if self.running:
            return
        self.running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        self.running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None

    def _worker_loop(self):
        while self.running:
            batch = self.batcher.get_batch()
            if batch:
                t0 = time.time()
                results = self._process_batch(batch)
                elapsed = time.time() - t0
                total_tokens = sum(len(r["tokens"]) for r in results)
                self.stats["total_tokens"] += total_tokens
                self.stats["total_time"] += elapsed
            else:
                time.sleep(0.001)

    def get_throughput(self) -> float:
        elapsed = self.stats["total_time"]
        return self.stats["total_tokens"] / max(elapsed, 0.001)

    def get_agent_stats(self, agent_id: str) -> Optional[dict]:
        with self._lock:
            agent = self.agents.get(agent_id)
            if agent is None:
                return None
            return {
                "agent_id": agent["id"],
                "tokens_generated": agent["tokens_generated"],
                "state": agent["state"],
                "hdc_confidence": agent["hdc_confidence"],
                "context_length": len(agent["context"]),
            }

    def get_stats(self) -> dict:
        active = 0
        with self._lock:
            active = sum(1 for a in self.agents.values() if a["state"] == "active")
        return {
            "tokens_per_second": round(self.get_throughput(), 1),
            "batches_processed": self.stats["batches_processed"],
            "total_tokens": self.stats["total_tokens"],
            "hdc_acceptance": round(self.verifier.acceptance_rate(), 3),
            "model_calls": self.stats["model_calls"],
            "active_agents": active,
            "registered_agents": len(self.agents),
            "pending_requests": self.batcher.pending,
            "avg_batch_size": round(self.batcher.avg_batch_size, 1),
            "target": f"{self.target_tok_s} tok/s",
        }

    def reset(self):
        self.stop()
        self.batcher.clear()
        self.rate_limiter.reset()
        self.verifier.reset()
        with self._lock:
            self.agents.clear()
        self.stats = {
            "total_tokens": 0,
            "total_time": 0.0,
            "batches_processed": 0,
            "hdc_accepted": 0,
            "model_calls": 0,
        }

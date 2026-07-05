from __future__ import annotations

import numpy as np
from collections import deque

from spectralstream.inference.hdc_engine import HDCDraftEngine
from spectralstream.inference.confidence_gate import ConfidenceGate


class OnlineLearningEngine:
    """Continuously learns from model corrections during inference.

    Trains both the HDC engine (n-gram memory) and the confidence gate
    (logistic regression) from live HDC-vs-model disagreement signals.
    """

    def __init__(
        self,
        hd_engine: HDCDraftEngine,
        confidence_gate: ConfidenceGate,
        max_buffer: int = 10000,
    ):
        self.hd = hd_engine
        self.gate = confidence_gate
        self.max_buffer = max_buffer
        self.correction_buffer = deque(maxlen=max_buffer)
        self.acceptance_buffer = deque(maxlen=max_buffer)
        self.total_corrections = 0
        self.total_acceptances = 0
        self.recent_accuracy = deque(maxlen=100)

    def observe_correction(
        self,
        context_tokens: list,
        hdc_predicted: int,
        model_token: int,
        features: list,
    ):
        self.hd.observe(model_token)
        self.gate.train(features, hdc_was_correct=False)
        self.correction_buffer.append(
            (context_tokens, hdc_predicted, model_token, features)
        )
        self.total_corrections += 1
        self.recent_accuracy.append(0)

    def observe_acceptance(
        self, context_tokens: list, accepted_token: int, features: list
    ):
        self.hd.observe(accepted_token)
        self.gate.train(features, hdc_was_correct=True)
        self.acceptance_buffer.append((context_tokens, accepted_token, features))
        self.total_acceptances += 1
        self.recent_accuracy.append(1)

    def get_stats(self) -> dict:
        total = self.total_corrections + self.total_acceptances
        return {
            "total_corrections": self.total_corrections,
            "total_acceptances": self.total_acceptances,
            "acceptance_rate": self.total_acceptances / max(total, 1),
            "gate_accuracy": self.gate.accuracy(),
            "recent_hdc_accuracy": float(np.mean(list(self.recent_accuracy)))
            if self.recent_accuracy
            else 0.0,
            "gate_updates": self.gate.update_count,
        }

from __future__ import annotations

import numpy as np
from collections import deque


class ConfidenceGate:
    """Online logistic regression classifier for HDC confidence estimation.

    Adaptive threshold maintains FPR < 15%. Trained continuously from
    model corrections during inference.
    """

    def __init__(self, n_features: int = 10, learning_rate: float = 0.01):
        self.n_features = n_features
        self.base_lr = learning_rate
        self.lr = learning_rate
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0
        self.train_buffer = deque(maxlen=1000)
        self.neg_preds = deque(maxlen=500)
        self.update_count = 0
        self.content_weights: dict = {}
        self.total_train = 0
        self.total_correct_pred = 0

    def _sigmoid(self, z: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -100.0, 100.0))))

    def predict(self, features: list) -> float:
        return self._sigmoid(float(np.dot(self.weights, features)) + self.bias)

    def train(self, features: list, hdc_was_correct: bool):
        label = 1.0 if hdc_was_correct else 0.0
        x = np.array(features, dtype=np.float32)
        pred = self._sigmoid(float(np.dot(self.weights, x)) + self.bias)
        error = pred - label
        self.weights -= self.lr * error * x
        self.bias -= self.lr * error
        self.train_buffer.append((features, label))
        if not hdc_was_correct:
            self.neg_preds.append(pred)
        self.update_count += 1
        self.lr = self.base_lr / (1.0 + 0.001 * self.update_count)
        pred_label = 1.0 if pred >= 0.5 else 0.0
        if pred_label == label:
            self.total_correct_pred += 1
        self.total_train += 1

    def should_fallback(self, features: list) -> bool:
        return self.predict(features) < self._adaptive_threshold()

    def _adaptive_threshold(self) -> float:
        if len(self.neg_preds) < 10:
            return 0.5
        sorted_preds = sorted(self.neg_preds)
        idx = min(int(len(sorted_preds) * 0.85), len(sorted_preds) - 1)
        return float(np.clip(sorted_preds[idx], 0.3, 0.95))

    def accuracy(self) -> float:
        return self.total_correct_pred / max(self.total_train, 1)

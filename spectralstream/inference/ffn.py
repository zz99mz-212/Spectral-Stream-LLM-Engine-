import numpy as np


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


class Gemma4FFN:
    def __init__(
        self, gate_weight: np.ndarray, up_weight: np.ndarray, down_weight: np.ndarray
    ):
        self.w_gate = gate_weight.astype(np.float32)
        self.w_up = up_weight.astype(np.float32)
        self.w_down = down_weight.astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        gate = x.astype(np.float32) @ self.w_gate
        up = x.astype(np.float32) @ self.w_up
        hidden = _gelu_tanh(gate) * up
        return hidden @ self.w_down

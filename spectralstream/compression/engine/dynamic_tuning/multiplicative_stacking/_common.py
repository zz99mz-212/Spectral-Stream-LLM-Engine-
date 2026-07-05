import numpy as np


def _decomp_error_gradient(ratio: float) -> float:
    return -0.05 * np.exp(-0.05 * ratio)


def _spectral_error_gradient(ratio: float) -> float:
    return -1.0 / (ratio * ratio + 1e-30)


def _structural_error_gradient(ratio: float) -> float:
    return -1.0 / (ratio * ratio + 1e-30)


def _quant_error_gradient(ratio: float) -> float:
    return -2.0 / (ratio * ratio * ratio + 1e-30)


def _entropy_error_gradient(ratio: float) -> float:
    return 0.0


ERROR_GRADIENT_MAP = {
    "decomposition": _decomp_error_gradient,
    "spectral": _spectral_error_gradient,
    "structural": _structural_error_gradient,
    "quantization": _quant_error_gradient,
    "entropy": _entropy_error_gradient,
}

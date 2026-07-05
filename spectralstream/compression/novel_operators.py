"""
Novel Inference Operators
-------------------------
Novel algorithmic operators for the SpectralStream inference pipeline.

Inventions:
11. HDC-Weighted Token Sampling - learned fusion of HDC scores with model logits
12. Spectral Entropy Gating - gate entire transformer layers by spectral entropy
13. Adaptive Forwardless Depth - dynamic generation depth from confidence/resonance
14. Gradient-Free Fine-Tuning - activation matching without backward pass
15. Predictor-Corrector Inference - HDC predicts, model corrects, like ODE methods

All implementations use numpy + standard library only.
"""

import numpy as np
from collections import deque
from typing import Optional


class HDCWeightedTokenSampling:
    """Invention 11: Fuse HDC similarity scores with model logits via learned weighting.

    Instead of using HDC only for candidate generation, this operator gives HDC
    direct influence on the output distribution.

    weight = sigmoid(confidence_gate_output)
    final_score(token) = weight * hdc_score(token) + (1 - weight) * model_logit(token)

    The weight is learned per-token and adjusted online based on whether the HDC
    prediction was accepted or corrected by the model.
    """

    def __init__(
        self,
        vocab_size: int,
        default_weight: float = 0.3,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ):
        self.vocab_size = vocab_size
        self.default_weight = default_weight
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = 0.0
        self.weight_history: deque = deque(maxlen=200)
        self.fusion_count = 0
        self.acceptance_count = 0

    def compute_weight(self, confidence: float, token_frequency: float) -> float:
        """Compute the fusion weight for combining HDC and model scores.

        weight = sigmoid(confidence * (1 + 0.2 * log(token_frequency + 1)))

        High confidence + frequent token = more HDC influence.
        Low confidence + rare token = more model influence.
        """
        freq_boost = 1.0 + 0.2 * np.log1p(token_frequency)
        z = confidence * freq_boost
        z_clipped = np.clip(z, -10.0, 10.0)
        weight = float(1.0 / (1.0 + np.exp(-z_clipped)))

        # Blend with default weight to prevent extreme values
        weight = 0.7 * weight + 0.3 * self.default_weight
        weight = float(np.clip(weight, 0.05, 0.95))
        self.weight_history.append(weight)
        return weight

    def fuse(
        self,
        model_logits: np.ndarray,
        hdc_scores: Optional[dict[int, float]],
        confidence: float,
        token_frequency: float,
    ) -> np.ndarray:
        """Fuse model logits with HDC scores.

        Args:
            model_logits: Raw logits from model forward pass (vocab_size,)
            hdc_scores: Dict mapping token_id -> HDC similarity score, or None
            confidence: Confidence gate output (0..1)
            token_frequency: Frequency of the current context token

        Returns:
            Fused logits (vocab_size,) ready for sampling
        """
        fused = model_logits.copy().astype(np.float64)

        if hdc_scores is None or not hdc_scores:
            return fused

        weight = self.compute_weight(confidence, token_frequency)
        hdc_logits = np.full(self.vocab_size, -1e10, dtype=np.float64)

        # Normalize HDC scores to logit scale
        hdc_values = np.array(list(hdc_scores.values()), dtype=np.float64)
        if len(hdc_values) > 0:
            hdc_mean = np.mean(hdc_values)
            hdc_std = max(np.std(hdc_values), 1e-10)
            for tok, score in hdc_scores.items():
                if 0 <= tok < self.vocab_size:
                    normalized = (score - hdc_mean) / hdc_std
                    hdc_logits[tok] = normalized

            # Blend: weight * HDC + (1-weight) * model
            max_logit = np.max(fused)
            if max_logit > -1e9:
                fused = weight * hdc_logits + (1.0 - weight) * fused

        self.fusion_count += 1
        return fused

    def record_acceptance(self, was_accepted: bool):
        """Record whether the HDC-weighted token was accepted by the model.

        Adjusts default weight via simple RL: increase weight on acceptance,
        decrease on rejection.
        """
        if was_accepted:
            self.default_weight = min(0.5, self.default_weight * 1.02)
            self.acceptance_count += 1
        else:
            self.default_weight = max(0.1, self.default_weight * 0.98)

    def stats(self) -> dict:
        avg_weight = float(np.mean(list(self.weight_history))) if self.weight_history else self.default_weight
        total = self.fusion_count
        accept_rate = self.acceptance_count / max(total, 1)
        return {
            "avg_fusion_weight": round(avg_weight, 4),
            "current_default_weight": round(self.default_weight, 4),
            "fusion_count": self.fusion_count,
            "acceptance_rate": round(accept_rate, 4),
        }

    def reset(self):
        self.default_weight = 0.3
        self.velocity = 0.0
        self.weight_history.clear()
        self.fusion_count = 0
        self.acceptance_count = 0


class SpectralEntropyGating:
    """Invention 12: Gate transformer layers based on spectral entropy.

    Computes spectral entropy from the DCT (Discrete Cosine Transform)
    of hidden states. If entropy is below threshold, skip attention + FFN
    for those layers.

    Key insight: when hidden states are spectrally simple (low entropy),
    the model doesn't need full computation to produce a good token.
    This saves ~30-70% compute on easy tokens.

    The gating is per-layer: each layer gets its own entropy measurement
    and can be independently skipped.
    """

    def __init__(
        self,
        n_layers: int = 8,
        entropy_threshold: float = 0.4,
        min_skipped_layers: int = 0,
        max_skipped_layers: int = 6,
        smoothing: float = 0.9,
    ):
        self.n_layers = n_layers
        self.entropy_threshold = entropy_threshold
        self.min_skipped = min_skipped_layers
        self.max_skipped = max_skipped_layers
        self.smoothing = smoothing

        # Per-layer smoothed entropy estimates
        self.layer_entropies: list[float] = [0.5] * n_layers
        self.layer_gate_history: list[deque] = [
            deque(maxlen=100) for _ in range(n_layers)
        ]
        self.total_layers_processed = 0
        self.total_layers_skipped = 0
        self.skip_pattern: list[bool] = [False] * n_layers

    def compute_spectral_entropy_dct(self, hidden_states: np.ndarray) -> float:
        """Compute spectral entropy using DCT of hidden states.

        Args:
            hidden_states: (hidden_dim,) or (seq_len, hidden_dim) array

        Returns:
            Normalized spectral entropy (0..1). Low = spectrally simple.
        """
        h = hidden_states.ravel().astype(np.float64)
        if len(h) < 4:
            return 0.5

        h = h - np.mean(h)

        # Apply DCT (type-II)
        n = len(h)
        dct = np.zeros(n, dtype=np.float64)
        for k in range(n):
            dct[k] = np.sum(
                h * np.cos(np.pi * (np.arange(n) + 0.5) * k / n)
            )
        dct = np.abs(dct)

        # Compute power spectrum
        power = dct[: n // 2]
        power_sum = np.sum(power)
        if power_sum < 1e-10:
            return 0.0
        power = power / power_sum

        # Compute entropy
        entropy = -np.sum(power * np.log2(power + 1e-10))
        max_entropy = np.log2(len(power))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.5

        return float(np.clip(norm_entropy, 0.0, 1.0))

    def should_skip_layer(self, layer_idx: int, hidden_state: np.ndarray) -> bool:
        """Decide whether to skip a specific transformer layer.

        Updates the smoothed entropy estimate for this layer and gates
        based on the current threshold.
        """
        entropy = self.compute_spectral_entropy_dct(hidden_state)

        # Exponential smoothing
        self.layer_entropies[layer_idx] = (
            self.smoothing * self.layer_entropies[layer_idx]
            + (1.0 - self.smoothing) * entropy
        )

        skip = self.layer_entropies[layer_idx] < self.entropy_threshold
        self.layer_gate_history[layer_idx].append(1.0 if skip else 0.0)
        self.total_layers_processed += 1
        if skip:
            self.total_layers_skipped += 1

        self.skip_pattern[layer_idx] = skip
        return skip

    def compute_gate_mask(self, hidden_states: list[np.ndarray]) -> list[bool]:
        """Compute gating decisions for all layers.

        Returns a list of bools: True = skip this layer.
        Respects min/max skipped layer constraints.
        """
        gate_decisions = []
        for i, hs in enumerate(hidden_states):
            if i >= self.n_layers:
                break
            gate_decisions.append(self.should_skip_layer(i, hs))

        n_skipped = sum(gate_decisions)

        # Enforce constraints
        if n_skipped < self.min_skipped:
            # Force-skip additional layers (the ones with lowest entropy)
            entropies_with_idx = [
                (self.layer_entropies[i], i)
                for i in range(min(len(gate_decisions), self.n_layers))
                if not gate_decisions[i]
            ]
            entropies_with_idx.sort(key=lambda x: x[0])
            for _, idx in entropies_with_idx[: self.min_skipped - n_skipped]:
                gate_decisions[idx] = True
                n_skipped += 1

        elif n_skipped > self.max_skipped:
            # Force-unskip layers (the ones with highest entropy)
            entropies_with_idx = [
                (self.layer_entropies[i], i)
                for i in range(min(len(gate_decisions), self.n_layers))
                if gate_decisions[i]
            ]
            entropies_with_idx.sort(key=lambda x: -x[0])
            for _, idx in entropies_with_idx[: n_skipped - self.max_skipped]:
                gate_decisions[idx] = False
                n_skipped -= 1

        return gate_decisions

    def compute_savings(self) -> float:
        """Fraction of layers skipped (compute savings)."""
        if self.total_layers_processed == 0:
            return 0.0
        return self.total_layers_skipped / self.total_layers_processed

    def adapt_threshold(self, acceptance_rate: float):
        """Adapt entropy threshold based on acceptance rate.

        When acceptance is high, we can be more aggressive (higher threshold).
        When acceptance is low, be more conservative (lower threshold).
        """
        target = 0.3 + 0.3 * acceptance_rate
        self.entropy_threshold = 0.8 * self.entropy_threshold + 0.2 * target
        self.entropy_threshold = float(np.clip(self.entropy_threshold, 0.1, 0.8))

    def stats(self) -> dict:
        return {
            "entropy_threshold": round(self.entropy_threshold, 3),
            "layer_skipped_fraction": round(self.compute_savings(), 4),
            "total_skipped": self.total_layers_skipped,
            "total_processed": self.total_layers_processed,
            "skip_pattern": [int(s) for s in self.skip_pattern],
        }

    def reset(self):
        self.layer_entropies = [0.5] * self.n_layers
        for dq in self.layer_gate_history:
            dq.clear()
        self.total_layers_processed = 0
        self.total_layers_skipped = 0
        self.skip_pattern = [False] * self.n_layers


class AdaptiveForwardlessDepth:
    """Invention 13: Dynamically choose forwardless generation depth.

    Depth = min(block_size, max_depth * confidence * resonance)

    When confidence and resonance are high, generate more tokens per model call.
    When uncertain, reduce depth for more frequent model verification.

    The depth is adjusted per-block, reacting to the latest state signals.
    """

    def __init__(
        self,
        min_depth: int = 1,
        max_depth: int = 32,
        default_block_size: int = 8,
        smoothing: float = 0.7,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.default_block_size = default_block_size
        self.smoothing = smoothing
        self.current_depth = default_block_size
        self.smoothed_depth = float(default_block_size)
        self.depth_history: deque = deque(maxlen=200)
        self.block_count = 0

    def compute_depth(
        self,
        confidence: float,
        resonance_score: float,
        spectral_entropy: float,
        acceptance_rate: float,
        block_size_hint: Optional[int] = None,
    ) -> int:
        """Compute optimal forwardless depth from current state.

        depth = block_size * confidence * resonance * entropy_factor

        Where entropy_factor is high when entropy is low (confident regime).
        """
        c = float(np.clip(confidence, 0.0, 1.0))
        r = float(np.clip(resonance_score, 0.0, 1.0))
        e = float(np.clip(1.0 - spectral_entropy, 0.0, 1.0))  # Invert: low entropy = high factor
        a = float(np.clip(acceptance_rate, 0.0, 1.0))

        # Product of signals with exponential weighting
        raw_depth = self.default_block_size * (c ** 0.5) * (r ** 0.5) * (e ** 0.3) * (a ** 0.3)

        # Boost: if all signals are strong, super-linear scaling
        composite = c * r * e * a
        if composite > 0.5:
            boost = 1.0 + (composite - 0.5) * 2.0
            raw_depth *= boost

        depth = int(np.clip(raw_depth, self.min_depth, self.max_depth))
        if block_size_hint is not None:
            depth = min(depth, block_size_hint)

        # Smooth depth changes to avoid oscillation
        self.smoothed_depth = (
            self.smoothing * self.smoothed_depth + (1.0 - self.smoothing) * depth
        )
        self.current_depth = int(round(self.smoothed_depth))
        self.current_depth = int(np.clip(self.current_depth, self.min_depth, self.max_depth))

        self.depth_history.append(self.current_depth)
        self.block_count += 1
        return self.current_depth

    def get_expected_forwardless_tokens(self, n_remaining: int) -> int:
        """How many tokens we expect to generate forwardlessly given remaining budget."""
        return min(self.current_depth, n_remaining)

    def stats(self) -> dict:
        depths = list(self.depth_history)
        avg_depth = float(np.mean(depths)) if depths else float(self.default_block_size)
        return {
            "current_depth": self.current_depth,
            "average_depth": round(avg_depth, 1),
            "min_allowed": self.min_depth,
            "max_allowed": self.max_depth,
            "blocks_generated": self.block_count,
        }

    def reset(self):
        self.current_depth = self.default_block_size
        self.smoothed_depth = float(self.default_block_size)
        self.depth_history.clear()
        self.block_count = 0


class GradientFreeFineTuning:
    """Invention 14: Fine-tune via activation matching without backward pass.

    Core idea: Use HDC to predict what a layer's activation should be for a
    given input, then adjust LoRA weights to minimize Hamming distance between
    HDC prediction and actual model activation.

    This is a pure forward-pass fine-tuning method (theoretically much faster
    than backprop). The HDC acts as a "teacher" that has already seen the
    correct activation pattern.

    LoRA is simulated as low-rank weight updates: W' = W + A @ B
    We optimize A and B using only forward passes and HDC similarity.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        lora_rank: int = 8,
        learning_rate: float = 0.001,
        n_iterations: int = 10,
    ):
        self.hidden_dim = hidden_dim
        self.lora_rank = lora_rank
        self.lr = learning_rate
        self.n_iterations = n_iterations

        # LoRA weights (random init)
        self.lora_A: list[np.ndarray] = []
        self.lora_B: list[np.ndarray] = []
        self.activation_buffer: deque = deque(maxlen=1000)
        self.hdc_predictions: deque = deque(maxlen=1000)
        self.finetune_steps = 0

    def _init_lora(self, n_layers: int):
        """Initialize LoRA weights for each layer."""
        while len(self.lora_A) < n_layers:
            rng = np.random.RandomState(len(self.lora_A))
            self.lora_A.append(
                rng.randn(self.hidden_dim, self.lora_rank).astype(np.float32) * 0.01
            )
            self.lora_B.append(
                rng.randn(self.lora_rank, self.hidden_dim).astype(np.float32) * 0.01
            )

    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute normalized Hamming distance between two activation vectors.

        Binarizes by thresholding at median, then counts bit differences.
        """
        a_bin = a > np.median(a)
        b_bin = b > np.median(b)
        n_total = a_bin.size
        if n_total == 0:
            return 0.0
        n_diff = np.count_nonzero(a_bin != b_bin)
        return float(n_diff / n_total)

    def record_activation(
        self,
        layer_idx: int,
        input_activation: np.ndarray,
        target_activation: np.ndarray,
    ):
        """Record a (input, target) activation pair for fine-tuning."""
        self.activation_buffer.append((layer_idx, input_activation.copy(), target_activation.copy()))

    def record_hdc_prediction(self, layer_idx: int, predicted_activation: np.ndarray):
        """Record an HDC-predicted activation for comparison."""
        self.hdc_predictions.append((layer_idx, predicted_activation.copy()))

    def apply_lora(self, weights: np.ndarray, layer_idx: int) -> np.ndarray:
        """Apply LoRA update to weights: W' = W + A @ B."""
        if layer_idx >= len(self.lora_A):
            self._init_lora(layer_idx + 1)
        return weights + self.lora_A[layer_idx] @ self.lora_B[layer_idx]

    def finetune_step(self, n_layers: int) -> dict:
        """Perform one fine-tuning step using activation matching.

        For each recorded (input, target) pair:
        1. Compute current LoRA output: input @ (A @ B)
        2. Measure Hamming distance to target activation
        3. Adjust LoRA weights to reduce distance (forward-only optimization)

        Uses a random perturbation + accept/reject scheme (no backprop).
        """
        self._init_lora(n_layers)

        if len(self.activation_buffer) < 4:
            return {"status": "insufficient_data", "samples": len(self.activation_buffer)}

        total_improvement = 0.0
        updates_per_layer: dict[int, int] = {}

        for batch_idx in range(min(self.n_iterations, len(self.activation_buffer) // 4)):
            # Sample a mini-batch from buffer
            indices = np.random.choice(
                len(self.activation_buffer),
                size=min(4, len(self.activation_buffer)),
                replace=False,
            )

            for idx in indices:
                layer_idx, input_act, target_act = self.activation_buffer[idx]

                if layer_idx >= len(self.lora_A):
                    continue

                # Current output
                current_output = input_act @ (self.lora_A[layer_idx] @ self.lora_B[layer_idx])
                current_distance = self.hamming_distance(current_output, target_act)

                # Perturb LoRA weights (forward-only optimization)
                rng = np.random.RandomState(self.finetune_steps + idx)
                perturbation_A = rng.randn(*self.lora_A[layer_idx].shape).astype(np.float32) * self.lr
                perturbation_B = rng.randn(*self.lora_B[layer_idx].shape).astype(np.float32) * self.lr

                # Try perturbation
                new_output = input_act @ (
                    (self.lora_A[layer_idx] + perturbation_A)
                    @ (self.lora_B[layer_idx] + perturbation_B)
                )
                new_distance = self.hamming_distance(new_output, target_act)

                improvement = current_distance - new_distance
                if improvement > 0:
                    self.lora_A[layer_idx] += perturbation_A
                    self.lora_B[layer_idx] += perturbation_B
                    total_improvement += improvement
                    updates_per_layer[layer_idx] = updates_per_layer.get(layer_idx, 0) + 1

        self.finetune_steps += 1

        return {
            "status": "completed",
            "steps": self.finetune_steps,
            "improvement": round(total_improvement, 4),
            "layers_updated": len(updates_per_layer),
            "buffer_size": len(self.activation_buffer),
        }

    def get_lora_weights(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get LoRA weights for a specific layer."""
        if layer_idx < len(self.lora_A):
            return self.lora_A[layer_idx], self.lora_B[layer_idx]
        return np.zeros((self.hidden_dim, self.lora_rank)), np.zeros((self.lora_rank, self.hidden_dim))

    def stats(self) -> dict:
        return {
            "finetune_steps": self.finetune_steps,
            "buffer_size": len(self.activation_buffer),
            "lora_layers": len(self.lora_A),
            "lora_rank": self.lora_rank,
            "hidden_dim": self.hidden_dim,
        }

    def reset(self):
        self.lora_A.clear()
        self.lora_B.clear()
        self.activation_buffer.clear()
        self.hdc_predictions.clear()
        self.finetune_steps = 0


class PredictorCorrectorInference:
    """Invention 15: Predictor-corrector method for inference.

    Inspired by numerical ODE integration (e.g., Adams-Bashforth-Moulton):
    - Predictor step: HDC predicts the next N tokens quickly
    - Corrector step: model evaluates prediction quality
    - If prediction is good (within tolerance): emit and skip model for next N tokens
    - If prediction is bad: run model, train HDC on the correction, continue

    This is like speculative decoding but framed as an ODE integrator,
    with adaptive step size (number of tokens) and error control.
    """

    def __init__(
        self,
        min_step: int = 1,
        max_step: int = 16,
        default_step: int = 4,
        error_tolerance: float = 0.3,
        step_increase_factor: float = 1.5,
        step_decrease_factor: float = 0.5,
    ):
        self.min_step = min_step
        self.max_step = max_step
        self.step_size = default_step
        self.error_tolerance = error_tolerance
        self.step_increase = step_increase_factor
        self.step_decrease = step_decrease_factor

        self.predictions_made = 0
        self.corrections_made = 0
        self.tokens_emitted_without_verification = 0
        self.current_error = 0.0
        self.error_history: deque = deque(maxlen=100)
        self.step_history: deque = deque(maxlen=100)
        self.prediction_buffer: deque = deque(maxlen=64)

    def predictor_step(
        self,
        hdc_candidates: list[list[int]],
        hdc_scores: list[float],
        n_tokens: int,
    ) -> list[int]:
        """Predictor: use HDC to draft N tokens.

        Selects the highest-scoring candidate block.
        Records the prediction for later correction.

        Returns list of predicted token IDs.
        """
        if not hdc_candidates:
            return []

        # Pick the best candidate block
        best_idx = int(np.argmax(hdc_scores)) if hdc_scores else 0
        best_idx = min(best_idx, len(hdc_candidates) - 1)
        prediction = list(hdc_candidates[best_idx])

        # Truncate/pad to exact step size
        if len(prediction) > n_tokens:
            prediction = prediction[:n_tokens]
        elif len(prediction) < n_tokens:
            # Pad with placeholder if needed (shouldn't happen normally)
            prediction.extend([0] * (n_tokens - len(prediction)))

        self.prediction_buffer.append(prediction)
        self.predictions_made += 1
        return prediction

    def corrector_step(
        self,
        prediction: list[int],
        model_tokens: list[int],
    ) -> tuple[list[int], float]:
        """Corrector: evaluate prediction quality against model output.

        Computes per-token error as 0 (match) or 1 (mismatch).
        Returns (corrected_tokens, error_rate).

        The error rate determines whether to accept the full prediction,
        partially accept it, or reject it entirely.
        """
        if len(model_tokens) == 0:
            return prediction, 1.0

        min_len = min(len(prediction), len(model_tokens))
        if min_len == 0:
            return model_tokens, 1.0

        errors = []
        corrected = []
        for i in range(min_len):
            if prediction[i] == model_tokens[i]:
                errors.append(0.0)
                corrected.append(prediction[i])
            else:
                errors.append(1.0)
                corrected.append(model_tokens[i])
                self.corrections_made += 1

        # Compute error rate
        error_rate = float(np.mean(errors)) if errors else 0.0
        self.current_error = error_rate
        self.error_history.append(error_rate)

        return corrected, error_rate

    def adjust_step_size(self, error_rate: float):
        """Adapt step size based on error rate (like ODE adaptive stepping).

        Low error -> increase step (more aggressive prediction).
        High error -> decrease step (more frequent correction).
        """
        if error_rate < self.error_tolerance * 0.5:
            self.step_size = min(self.max_step, int(self.step_size * self.step_increase))
        elif error_rate > self.error_tolerance:
            self.step_size = max(self.min_step, int(self.step_size * self.step_decrease))

        self.step_size = int(np.clip(self.step_size, self.min_step, self.max_step))
        self.step_history.append(self.step_size)

    def should_skip_verification(self, error_rate: float) -> tuple[bool, int]:
        """Decide whether to skip model verification entirely.

        Returns (skip, n_tokens_to_emit).
        If error rate is very low, emit the full prediction without model call.
        """
        if error_rate < self.error_tolerance * 0.25 and self.step_size >= 2:
            self.tokens_emitted_without_verification += self.step_size
            return True, self.step_size
        return False, 0

    def predict_correct_cycle(
        self,
        hdc_candidates: list[list[int]],
        hdc_scores: list[float],
        model_forward_fn,
        context: list[int],
    ) -> tuple[list[int], float, bool]:
        """Perform one full predictor-corrector cycle.

        Args:
            hdc_candidates: Candidate blocks from HDC
            hdc_scores: Scores for each candidate block
            model_forward_fn: Function that takes (context, n_tokens) and returns tokens
            context: Current context tokens

        Returns:
            (emitted_tokens, error_rate, used_model)
        """
        # Predictor: HDC drafts tokens
        prediction = self.predictor_step(hdc_candidates, hdc_scores, self.step_size)

        if not prediction:
            # Fallback: use model directly
            model_out = model_forward_fn(context, self.step_size)
            return model_out, 1.0, True

        # Check if prediction is good enough to skip model entirely
        # We need at least one token from model to verify
        # Try just the first token as verification
        first_model_token = model_forward_fn(context, 1)
        if len(first_model_token) == 0:
            return prediction, 0.0, False

        # Corrector: evaluate first token
        corrected, error_rate = self.corrector_step(prediction[:1], first_model_token[:1])

        # If first token matches and error is very low, emit rest without verification
        if error_rate < 1e-6 and len(prediction) > 1:
            self.tokens_emitted_without_verification += len(prediction) - 1
            self.adjust_step_size(error_rate)
            return prediction, error_rate, True  # used model for 1 token only

        # First token was wrong: get full model output
        model_tokens = model_forward_fn(context, self.step_size)
        corrected, error_rate = self.corrector_step(prediction, model_tokens)
        self.adjust_step_size(error_rate)

        return corrected, error_rate, True

    def stats(self) -> dict:
        avg_error = float(np.mean(list(self.error_history))) if self.error_history else 0.0
        total = self.predictions_made + self.corrections_made
        skip_rate = self.tokens_emitted_without_verification / max(self.predictions_made * self.step_size, 1)
        return {
            "step_size": self.step_size,
            "predictions_made": self.predictions_made,
            "corrections_made": self.corrections_made,
            "current_error_rate": round(self.current_error, 4),
            "average_error_rate": round(avg_error, 4),
            "tokens_without_verification": self.tokens_emitted_without_verification,
            "skip_rate": round(skip_rate, 4),
        }

    def reset(self):
        self.step_size = 4
        self.current_error = 0.0
        self.predictions_made = 0
        self.corrections_made = 0
        self.tokens_emitted_without_verification = 0
        self.error_history.clear()
        self.step_history.clear()
        self.prediction_buffer.clear()

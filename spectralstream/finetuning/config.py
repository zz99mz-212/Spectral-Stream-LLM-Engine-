"""
Training Configuration for SpectralStream Fine-Tuning
=====================================================
All configurable parameters for fine-tuning compressed models.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning compressed/quantized models."""

    model_path: str = ""
    output_path: str = ""
    dataset_source: str = ""
    dataset_format: str = "auto"  # auto, csv, json, jsonl, text, chatml, code

    # Training hyperparameters
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation: int = 4
    max_seq_length: int = 2048
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # LoRA
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list] = None  # auto-detect if None

    # Quantization
    maintain_compression: bool = True
    target_ratio: float = 5000.0
    max_error: float = 0.0002
    quantization_bits: int = 4

    # Hardware
    device: str = "cpu"
    num_workers: int = 4
    mixed_precision: bool = False  # CPU doesn't support FP16
    seed: int = 42

    # Logging & checkpointing
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 200
    checkpoint_dir: str = "checkpoints"

    def validate(self) -> list:
        """Validate configuration, returning list of warnings."""
        warnings = []
        if not self.model_path:
            warnings.append("model_path is required")
        if not self.dataset_source:
            warnings.append("dataset_source is required")
        if self.epochs < 1:
            warnings.append("epochs must be >= 1")
        if self.learning_rate <= 0:
            warnings.append("learning_rate must be positive")
        if self.batch_size < 1:
            warnings.append("batch_size must be >= 1")
        if self.max_seq_length < 32:
            warnings.append("max_seq_length must be >= 32")
        if self.lora_rank < 1:
            warnings.append("lora_rank must be >= 1")
        if self.lora_alpha < 1:
            warnings.append("lora_alpha must be >= 1")
        if self.use_lora and self.lora_rank > min(256, self.max_seq_length):
            warnings.append("lora_rank is unusually large")
        return warnings

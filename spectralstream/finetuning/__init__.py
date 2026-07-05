"""Fine-tuning package for SpectralStream."""

from .engine import FinetuningEngine, FinetuningConfig
from .engine import LoRAAdapter as EngineLoRAAdapter  # legacy
from .config import TrainingConfig
from .dataset_loader import DatasetLoader, SimpleTokenizer
from .lora_adapter import LoRAAdapterV2, CompressedLoRA
from .trainer import CompressedModelTrainer, TinyTransformerModel
from .gguf_finetune import GGUFEditEngine, GGUFAdapterManager
from .dataset_streamer import DatasetStreamer, stream_dataset
from .intelligence_engine import (
    FineTuningIntelligenceEngine,
    FineTuningIntelligenceConfig,
    LoRAAdapter,
)

__all__ = [
    "FinetuningEngine",
    "FinetuningConfig",
    "EngineLoRAAdapter",
    "TrainingConfig",
    "DatasetLoader",
    "SimpleTokenizer",
    "LoRAAdapterV2",
    "CompressedLoRA",
    "CompressedModelTrainer",
    "TinyTransformerModel",
    "GGUFEditEngine",
    "GGUFAdapterManager",
    "DatasetStreamer",
    "stream_dataset",
    "FineTuningIntelligenceEngine",
    "FineTuningIntelligenceConfig",
    "LoRAAdapter",
]

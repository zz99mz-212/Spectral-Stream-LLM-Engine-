"""SpectralStream - Next-generation neural compression & inference engine."""

__version__ = "1.0.0"

from spectralstream.orchestrator import SpectralOrchestrator
from spectralstream.logging_config import (
    setup_logging,
    get_logger,
    CompressionLogger,
    PerformanceTimer,
    timed_operation,
    AuditLogger,
)
from spectralstream.audit import CompressionAudit, InferenceAudit, AuditTrail
from spectralstream.supreme_quant_engine import (
    SupremeQuantEngine,
    CompressionPipeline,
    CompressedWeight,
    CompressionBudget,
)

__all__ = [
    "SpectralOrchestrator",
    "setup_logging",
    "get_logger",
    "CompressionLogger",
    "PerformanceTimer",
    "timed_operation",
    "AuditLogger",
    "CompressionAudit",
    "InferenceAudit",
    "AuditTrail",
    "SupremeQuantEngine",
    "CompressionPipeline",
    "CompressedWeight",
    "CompressionBudget",
]

"""
Central Configuration for SpectralStream

All configurable parameters in one place.
Loaded from:
1. YAML/JSON config file (if present)
2. Environment variables
3. Command-line arguments
4. Sensible defaults

Supports:
- Model-specific overrides (Gemma 4 E2B vs E4B)
- Hardware-specific tuning (CPU cores, RAM, SSD speed)
- Strategy configuration (when to use each inference approach)
- Online learning parameters
- Monitoring settings
"""

import json
import os
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class HDCConfig:
    dim: int = 10000
    ngram_order: int = 4
    sparsity: float = 0.05
    max_prototypes: int = 8
    num_lsh_tables: int = 32
    lsh_bits_per_key: int = 8


@dataclass
class SpectralConfig:
    kv_compression: float = 20.0
    k_bits: int = 4
    v_bits: int = 2
    spectral_rank: int = 64
    use_vlasov: bool = False
    use_turboquant: bool = True


@dataclass
class ConfidenceGateConfig:
    learning_rate: float = 0.01
    n_features: int = 10
    target_fpr: float = 0.15
    adaptive_threshold: bool = True


@dataclass
class BlockEmissionConfig:
    min_block_size: int = 2
    max_block_size: int = 24
    n_candidates: int = 16
    coherence_threshold: float = 0.55
    use_pid_control: bool = True


@dataclass
class OnlineLearningConfig:
    max_buffer: int = 10000
    batch_size: int = 32
    save_every: int = 1000


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 1234
    lmstudio_url: str = "http://127.0.0.1:1234"
    max_connections: int = 4
    request_timeout: int = 300


@dataclass
class MonitoringConfig:
    window_size: int = 100
    enable_prometheus: bool = False
    prometheus_port: int = 9090
    console_report_interval: int = 60
    log_file: Optional[str] = None


@dataclass
class PersistenceConfig:
    state_dir: str = "~/.spectralstream/state/"
    checkpoint_interval: int = 300
    auto_save: bool = True
    auto_load: bool = True
    max_checkpoints: int = 10


@dataclass
class HardwareConfig:
    cpu_cores: int = field(default_factory=lambda: os.cpu_count() or 4)
    ram_gb: float = field(default_factory=lambda: _estimate_ram_gb())
    ssd_speed_mbps: int = 2000


def _estimate_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 * 1024)
        except (OSError, ValueError, IOError):
            pass
        return 16.0


_ENV_MAP = {
    "SS_HDC_DIM": ("hdc", "dim", int),
    "SS_HDC_NGRAM_ORDER": ("hdc", "ngram_order", int),
    "SS_KV_COMPRESSION": ("spectral", "kv_compression", float),
    "SS_K_BITS": ("spectral", "k_bits", int),
    "SS_V_BITS": ("spectral", "v_bits", int),
    "SS_SPECTRAL_RANK": ("spectral", "spectral_rank", int),
    "SS_GATE_LR": ("confidence", "learning_rate", float),
    "SS_TARGET_FPR": ("confidence", "target_fpr", float),
    "SS_MIN_BLOCK": ("block_emission", "min_block_size", int),
    "SS_MAX_BLOCK": ("block_emission", "max_block_size", int),
    "SS_CANDIDATES": ("block_emission", "n_candidates", int),
    "SS_COHERENCE": ("block_emission", "coherence_threshold", float),
    "SS_MAX_BUFFER": ("online_learning", "max_buffer", int),
    "SS_BATCH_SIZE": ("online_learning", "batch_size", int),
    "SS_HOST": ("server", "host", str),
    "SS_PORT": ("server", "port", int),
    "SS_LMSTUDIO_URL": ("server", "lmstudio_url", str),
    "SS_MAX_CONNECTIONS": ("server", "max_connections", int),
    "SS_TIMEOUT": ("server", "request_timeout", int),
    "SS_STATE_DIR": ("persistence", "state_dir", str),
    "SS_CHECKPOINT_INTERVAL": ("persistence", "checkpoint_interval", int),
}

_MODEL_OVERRIDES = {
    "gemma-4-2b": {
        "hdc": {"dim": 4096, "ngram_order": 3},
        "block_emission": {"min_block_size": 2, "max_block_size": 16},
    },
    "gemma-4-9b": {
        "hdc": {"dim": 8192, "ngram_order": 4},
        "block_emission": {"min_block_size": 2, "max_block_size": 20},
    },
    "gemma-4-27b": {
        "hdc": {"dim": 10000, "ngram_order": 4},
        "block_emission": {"min_block_size": 2, "max_block_size": 24},
    },
    "llama-3.2-1b": {
        "hdc": {"dim": 4096, "ngram_order": 3},
        "block_emission": {"min_block_size": 2, "max_block_size": 16},
    },
    "llama-3.2-3b": {
        "hdc": {"dim": 8192, "ngram_order": 4},
        "block_emission": {"min_block_size": 2, "max_block_size": 20},
    },
    "llama-3.1-8b": {
        "hdc": {"dim": 10000, "ngram_order": 4},
        "block_emission": {"min_block_size": 4, "max_block_size": 24},
    },
    "qwen2.5-0.5b": {
        "hdc": {"dim": 4096, "ngram_order": 3},
    },
    "qwen2.5-1.5b": {
        "hdc": {"dim": 8192, "ngram_order": 3},
    },
    "qwen2.5-7b": {
        "hdc": {"dim": 10000, "ngram_order": 4},
    },
    "qwen2.5-14b": {
        "hdc": {"dim": 10000, "ngram_order": 4},
        "block_emission": {"min_block_size": 4, "max_block_size": 32},
    },
    "deepseek-coder-1.3b": {
        "hdc": {"dim": 4096},
        "spectral": {"spectral_rank": 32},
    },
    "deepseek-coder-6.7b": {
        "hdc": {"dim": 10000},
        "spectral": {"spectral_rank": 64},
    },
    "deepseek-r1-7b": {
        "hdc": {"dim": 10000, "ngram_order": 5},
        "spectral": {"spectral_rank": 64},
    },
}


@dataclass
class SpectralStreamConfig:
    hdc: HDCConfig = field(default_factory=HDCConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    confidence: ConfidenceGateConfig = field(default_factory=ConfidenceGateConfig)
    block_emission: BlockEmissionConfig = field(default_factory=BlockEmissionConfig)
    online_learning: OnlineLearningConfig = field(default_factory=OnlineLearningConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @classmethod
    def from_file(cls, path: str) -> "SpectralStreamConfig":
        """Load config from JSON file, merging with defaults."""
        cfg = cls()
        try:
            with open(path) as f:
                raw = json.load(f)
            cfg._merge(raw)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load config from {path}: {exc}")
        return cfg

    @classmethod
    def from_env(cls) -> "SpectralStreamConfig":
        """Load config from environment variables (SS_ prefix)."""
        cfg = cls()
        for env_key, (section, attr, typ) in _ENV_MAP.items():
            val = os.environ.get(env_key)
            if val is not None:
                section_cfg = getattr(cfg, section)
                try:
                    setattr(section_cfg, attr, typ(val))
                except (ValueError, TypeError):
                    pass
        return cfg

    @classmethod
    def load(cls, path: Optional[str] = None) -> "SpectralStreamConfig":
        """Load config from: env vars -> file -> defaults (priority: env > file > defaults)."""
        if path:
            cfg = cls.from_file(path)
        else:
            cfg = cls()

        for env_key, (section, attr, typ) in _ENV_MAP.items():
            val = os.environ.get(env_key)
            if val is not None:
                section_cfg = getattr(cfg, section)
                try:
                    setattr(section_cfg, attr, typ(val))
                except (ValueError, TypeError):
                    pass
        return cfg

    def to_file(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def to_dict(self) -> dict:
        return asdict(self)

    def _merge(self, raw: dict, overwrite: bool = False):
        """Recursive merge of raw dict into config dataclasses."""
        for section_name, section_cfg in [
            ("hdc", self.hdc),
            ("spectral", self.spectral),
            ("confidence", self.confidence),
            ("block_emission", self.block_emission),
            ("online_learning", self.online_learning),
            ("server", self.server),
            ("monitoring", self.monitoring),
            ("persistence", self.persistence),
            ("hardware", self.hardware),
        ]:
            raw_section = raw.get(section_name, {})
            for key, value in raw_section.items():
                if hasattr(section_cfg, key):
                    current = getattr(section_cfg, key)
                    if isinstance(current, dict) and isinstance(value, dict):
                        if overwrite:
                            setattr(section_cfg, key, value)
                        else:
                            merged = {**value, **current}
                            setattr(section_cfg, key, merged)
                    else:
                        setattr(section_cfg, key, value)

    def for_model(self, model_name: str) -> "SpectralStreamConfig":
        """Get model-specific configuration copy with overrides applied."""
        cfg = copy.deepcopy(self)
        lower_name = model_name.lower()

        matches = []
        for key in _MODEL_OVERRIDES:
            if key in lower_name:
                matches.append(key)

        if not matches:
            return cfg

        matches.sort(key=len, reverse=True)
        best = matches[0]
        overrides = _MODEL_OVERRIDES[best]

        for section_name, section_overrides in overrides.items():
            section_cfg = getattr(cfg, section_name, None)
            if section_cfg is None:
                continue
            for attr, val in section_overrides.items():
                if hasattr(section_cfg, attr):
                    setattr(section_cfg, attr, val)

        return cfg

    def for_hardware(self) -> "SpectralStreamConfig":
        """Apply hardware-specific tuning."""
        cfg = copy.deepcopy(self)
        ram = cfg.hardware.ram_gb
        cores = cfg.hardware.cpu_cores

        if ram < 8:
            cfg.hdc.dim = min(cfg.hdc.dim, 4096)
            cfg.hdc.num_lsh_tables = min(cfg.hdc.num_lsh_tables, 8)
            cfg.spectral.spectral_rank = min(cfg.spectral.spectral_rank, 32)
            cfg.server.max_connections = max(cfg.server.max_connections, 1)
        elif ram < 16:
            cfg.hdc.dim = min(cfg.hdc.dim, 8192)
            cfg.hdc.num_lsh_tables = min(cfg.hdc.num_lsh_tables, 16)

        if cores <= 2:
            cfg.hdc.num_lsh_tables = min(cfg.hdc.num_lsh_tables, 4)
            cfg.server.max_connections = 1
        elif cores <= 4:
            cfg.hdc.num_lsh_tables = min(cfg.hdc.num_lsh_tables, 8)
            cfg.server.max_connections = min(cfg.server.max_connections, 2)

        return cfg

    def validate(self) -> list[str]:
        """Validate configuration, returning list of warnings/errors."""
        warnings = []
        if self.hdc.dim <= 0:
            warnings.append("HDC dim must be positive")
        if self.hdc.ngram_order < 2:
            warnings.append("HDC ngram_order should be >= 2")
        if not (0 < self.hdc.sparsity < 1):
            warnings.append("HDC sparsity should be between 0 and 1")
        if self.hdc.max_prototypes <= 0:
            warnings.append("HDC max_prototypes must be positive")
        if self.spectral.kv_compression <= 0:
            warnings.append("Spectral KV compression must be positive")
        if self.confidence.learning_rate <= 0:
            warnings.append("Confidence gate learning rate must be positive")
        if self.confidence.target_fpr <= 0 or self.confidence.target_fpr >= 1:
            warnings.append("Confidence gate target_fpr should be in (0, 1)")
        if self.block_emission.min_block_size < 1:
            warnings.append("Block emission min_block_size must be >= 1")
        if self.block_emission.max_block_size < self.block_emission.min_block_size:
            warnings.append("Block emission max_block_size must be >= min_block_size")
        if self.server.port < 1 or self.server.port > 65535:
            warnings.append("Server port must be between 1 and 65535")
        if self.server.request_timeout <= 0:
            warnings.append("Server request_timeout must be positive")
        return warnings

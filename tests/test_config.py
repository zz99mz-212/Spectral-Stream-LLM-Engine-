import sys

sys.path.insert(0, ".")
import json
import os
import tempfile
import copy

try:
    from spectralstream.config import (
        SpectralStreamConfig,
        HDCConfig,
        SpectralConfig,
        ConfidenceGateConfig,
        BlockEmissionConfig,
        HardwareConfig,
        ServerConfig,
        MonitoringConfig,
        PersistenceConfig,
        OnlineLearningConfig,
    )
except ImportError:
    pass


def test_default_config_creation():
    cfg = SpectralStreamConfig()
    assert isinstance(cfg.hdc, HDCConfig)
    assert isinstance(cfg.spectral, SpectralConfig)
    assert isinstance(cfg.confidence, ConfidenceGateConfig)
    assert isinstance(cfg.block_emission, BlockEmissionConfig)
    assert isinstance(cfg.online_learning, OnlineLearningConfig)
    assert isinstance(cfg.server, ServerConfig)
    assert isinstance(cfg.monitoring, MonitoringConfig)
    assert isinstance(cfg.persistence, PersistenceConfig)
    assert isinstance(cfg.hardware, HardwareConfig)


def test_default_field_access():
    cfg = SpectralStreamConfig()
    assert cfg.hdc.dim == 10000
    assert cfg.hdc.ngram_order == 4
    assert cfg.hdc.sparsity == 0.05
    assert cfg.hdc.max_prototypes == 8
    assert cfg.hdc.num_lsh_tables == 32
    assert cfg.hdc.lsh_bits_per_key == 8
    assert cfg.spectral.kv_compression == 20.0
    assert cfg.spectral.k_bits == 4
    assert cfg.spectral.v_bits == 2
    assert cfg.spectral.spectral_rank == 64
    assert cfg.spectral.use_vlasov is False
    assert cfg.spectral.use_turboquant is True
    assert cfg.confidence.learning_rate == 0.01
    assert cfg.confidence.n_features == 10
    assert cfg.confidence.target_fpr == 0.15
    assert cfg.confidence.adaptive_threshold is True
    assert cfg.block_emission.min_block_size == 2
    assert cfg.block_emission.max_block_size == 24
    assert cfg.block_emission.n_candidates == 16
    assert cfg.block_emission.coherence_threshold == 0.55
    assert cfg.block_emission.use_pid_control is True
    assert cfg.online_learning.max_buffer == 10000
    assert cfg.online_learning.batch_size == 32
    assert cfg.online_learning.save_every == 1000
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 1234
    assert cfg.server.lmstudio_url == "http://127.0.0.1:1234"
    assert cfg.server.max_connections == 4
    assert cfg.server.request_timeout == 300
    assert cfg.persistence.state_dir == "~/.spectralstream/state/"
    assert cfg.persistence.checkpoint_interval == 300
    assert cfg.persistence.auto_save is True
    assert cfg.persistence.auto_load is True
    assert cfg.persistence.max_checkpoints == 10


def test_nested_config_dataclasses():
    hdc = HDCConfig(dim=8192, ngram_order=3, sparsity=0.1)
    assert hdc.dim == 8192
    assert hdc.ngram_order == 3
    assert hdc.sparsity == 0.1
    spectral = SpectralConfig(kv_compression=10.0, k_bits=8, spectral_rank=128)
    assert spectral.kv_compression == 10.0
    assert spectral.k_bits == 8
    assert spectral.spectral_rank == 128


def test_to_dict_roundtrip():
    cfg = SpectralStreamConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert "hdc" in d
    assert "spectral" in d
    assert "confidence" in d
    assert "block_emission" in d
    assert "online_learning" in d
    assert "server" in d
    assert "monitoring" in d
    assert "persistence" in d
    assert "hardware" in d
    assert d["hdc"]["dim"] == 10000
    assert d["spectral"]["k_bits"] == 4
    assert d["server"]["port"] == 1234


def test_to_dict_json_serializable():
    cfg = SpectralStreamConfig()
    d = cfg.to_dict()
    dumped = json.dumps(d, default=str)
    loaded = json.loads(dumped)
    assert loaded["hdc"]["dim"] == 10000
    assert loaded["server"]["host"] == "127.0.0.1"


def test_validate_valid_default():
    cfg = SpectralStreamConfig()
    warnings = cfg.validate()
    assert isinstance(warnings, list)
    assert len(warnings) == 0


def test_validate_invalid_hdc_dim():
    cfg = SpectralStreamConfig()
    cfg.hdc.dim = 0
    warnings = cfg.validate()
    assert "HDC dim must be positive" in warnings


def test_validate_invalid_ngram_order():
    cfg = SpectralStreamConfig()
    cfg.hdc.ngram_order = 1
    warnings = cfg.validate()
    assert "HDC ngram_order should be >= 2" in warnings


def test_validate_invalid_sparsity():
    cfg = SpectralStreamConfig()
    cfg.hdc.sparsity = 0
    warnings = cfg.validate()
    assert "HDC sparsity should be between 0 and 1" in warnings


def test_validate_negative_max_prototypes():
    cfg = SpectralStreamConfig()
    cfg.hdc.max_prototypes = -1
    warnings = cfg.validate()
    assert "HDC max_prototypes must be positive" in warnings


def test_validate_invalid_spectral_kv_compression():
    cfg = SpectralStreamConfig()
    cfg.spectral.kv_compression = 0
    warnings = cfg.validate()
    assert "Spectral KV compression must be positive" in warnings


def test_validate_invalid_learning_rate():
    cfg = SpectralStreamConfig()
    cfg.confidence.learning_rate = 0
    warnings = cfg.validate()
    assert "Confidence gate learning rate must be positive" in warnings


def test_validate_invalid_target_fpr():
    cfg = SpectralStreamConfig()
    cfg.confidence.target_fpr = 0
    warnings = cfg.validate()
    assert "Confidence gate target_fpr should be in (0, 1)" in warnings


def test_validate_invalid_target_fpr_above_one():
    cfg = SpectralStreamConfig()
    cfg.confidence.target_fpr = 1.0
    warnings = cfg.validate()
    assert "Confidence gate target_fpr should be in (0, 1)" in warnings


def test_validate_invalid_min_block_size():
    cfg = SpectralStreamConfig()
    cfg.block_emission.min_block_size = 0
    warnings = cfg.validate()
    assert "Block emission min_block_size must be >= 1" in warnings


def test_validate_max_block_below_min():
    cfg = SpectralStreamConfig()
    cfg.block_emission.min_block_size = 10
    cfg.block_emission.max_block_size = 5
    warnings = cfg.validate()
    assert "Block emission max_block_size must be >= min_block_size" in warnings


def test_validate_invalid_port_low():
    cfg = SpectralStreamConfig()
    cfg.server.port = 0
    warnings = cfg.validate()
    assert "Server port must be between 1 and 65535" in warnings


def test_validate_invalid_port_high():
    cfg = SpectralStreamConfig()
    cfg.server.port = 65536
    warnings = cfg.validate()
    assert "Server port must be between 1 and 65535" in warnings


def test_validate_invalid_timeout():
    cfg = SpectralStreamConfig()
    cfg.server.request_timeout = 0
    warnings = cfg.validate()
    assert "Server request_timeout must be positive" in warnings


def test_validate_multiple_errors():
    cfg = SpectralStreamConfig()
    cfg.hdc.dim = 0
    cfg.hdc.sparsity = 0
    cfg.server.port = 0
    warnings = cfg.validate()
    assert len(warnings) >= 3
    assert "HDC dim must be positive" in warnings
    assert "HDC sparsity should be between 0 and 1" in warnings
    assert "Server port must be between 1 and 65535" in warnings


def test_for_model_gemma_4_2b():
    cfg = SpectralStreamConfig()
    model_cfg = cfg.for_model("gemma-4-2b")
    assert model_cfg.hdc.dim == 4096
    assert model_cfg.hdc.ngram_order == 3
    assert model_cfg.block_emission.max_block_size == 16
    assert model_cfg.spectral.spectral_rank == 64


def test_for_model_llama_3_2_1b():
    cfg = SpectralStreamConfig()
    model_cfg = cfg.for_model("llama-3.2-1b")
    assert model_cfg.hdc.dim == 4096
    assert model_cfg.hdc.ngram_order == 3
    assert model_cfg.block_emission.max_block_size == 16


def test_for_model_unknown_returns_copy():
    cfg = SpectralStreamConfig()
    cfg.hdc.dim = 999
    model_cfg = cfg.for_model("nonexistent-model-v99")
    assert model_cfg.hdc.dim == 999
    assert model_cfg is not cfg


def test_for_model_does_not_mutate_original():
    cfg = SpectralStreamConfig()
    model_cfg = cfg.for_model("gemma-4-2b")
    assert model_cfg.hdc.dim == 4096
    assert cfg.hdc.dim == 10000


def test_for_model_deepseek_coder():
    cfg = SpectralStreamConfig()
    model_cfg = cfg.for_model("deepseek-coder-1.3b")
    assert model_cfg.hdc.dim == 4096
    assert model_cfg.spectral.spectral_rank == 32


def test_for_model_qwen():
    cfg = SpectralStreamConfig()
    model_cfg = cfg.for_model("qwen2.5-7b")
    assert model_cfg.hdc.dim == 10000
    assert model_cfg.hdc.ngram_order == 4


def test_for_hardware_low_ram():
    cfg = SpectralStreamConfig()
    cfg.hardware.ram_gb = 4
    cfg.hardware.cpu_cores = 4
    hw_cfg = cfg.for_hardware()
    assert hw_cfg.hdc.dim <= 4096
    assert hw_cfg.hdc.num_lsh_tables <= 8
    assert hw_cfg.spectral.spectral_rank <= 32


def test_for_hardware_medium_ram():
    cfg = SpectralStreamConfig()
    cfg.hardware.ram_gb = 10
    cfg.hardware.cpu_cores = 4
    hw_cfg = cfg.for_hardware()
    assert hw_cfg.hdc.dim <= 8192
    assert hw_cfg.hdc.num_lsh_tables <= 8


def test_for_hardware_high_ram():
    cfg = SpectralStreamConfig()
    cfg.hardware.ram_gb = 32
    cfg.hardware.cpu_cores = 8
    hw_cfg = cfg.for_hardware()
    assert hw_cfg.hdc.dim == 10000


def test_for_hardware_few_cores():
    cfg = SpectralStreamConfig()
    cfg.hardware.ram_gb = 32
    cfg.hardware.cpu_cores = 2
    hw_cfg = cfg.for_hardware()
    assert hw_cfg.hdc.num_lsh_tables <= 4
    assert hw_cfg.server.max_connections == 1


def test_for_hardware_does_not_mutate_original():
    cfg = SpectralStreamConfig()
    cfg.hardware.ram_gb = 4
    cfg.hardware.cpu_cores = 2
    hw_cfg = cfg.for_hardware()
    assert hw_cfg.hdc.dim <= 4096
    assert cfg.hdc.dim == 10000


def test_from_file_roundtrip():
    cfg = SpectralStreamConfig()
    cfg.hdc.dim = 7777
    cfg.spectral.k_bits = 16
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json.dumps(cfg.to_dict()))
        tmp_path = f.name
    try:
        loaded = SpectralStreamConfig.from_file(tmp_path)
        assert loaded.hdc.dim == 7777
        assert loaded.spectral.k_bits == 16
    finally:
        os.unlink(tmp_path)


def test_from_file_nonexistent():
    try:
        SpectralStreamConfig.from_file("/tmp/nonexistent_config_file.json")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_from_file_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json")
        tmp_path = f.name
    try:
        SpectralStreamConfig.from_file(tmp_path)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    finally:
        os.unlink(tmp_path)


def test_from_file_partial_override():
    partial = {"hdc": {"dim": 5555}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json.dumps(partial))
        tmp_path = f.name
    try:
        loaded = SpectralStreamConfig.from_file(tmp_path)
        assert loaded.hdc.dim == 5555
        assert loaded.spectral.k_bits == 4
        assert loaded.server.port == 1234
    finally:
        os.unlink(tmp_path)


def test_from_env():
    env_vars = {
        "SS_HDC_DIM": "4096",
        "SS_K_BITS": "8",
        "SS_PORT": "8080",
        "SS_KV_COMPRESSION": "15.5",
    }
    original = {k: os.environ.get(k) for k in env_vars}
    for k, v in env_vars.items():
        if v is not None:
            os.environ[k] = v
    try:
        cfg = SpectralStreamConfig.from_env()
        assert cfg.hdc.dim == 4096
        assert cfg.spectral.k_bits == 8
        assert cfg.server.port == 8080
        assert cfg.spectral.kv_compression == 15.5
    finally:
        for k in env_vars:
            if k in os.environ:
                if original[k] is not None:
                    os.environ[k] = original[k]
                else:
                    del os.environ[k]


def test_from_env_ignores_unset():
    for k in ["SS_HDC_DIM", "SS_K_BITS"]:
        os.environ.pop(k, None)
    cfg = SpectralStreamConfig.from_env()
    assert cfg.hdc.dim == 10000
    assert cfg.spectral.k_bits == 4


def test_load_with_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json.dumps({"hdc": {"dim": 3333}}))
        tmp_path = f.name
    try:
        cfg = SpectralStreamConfig.load(tmp_path)
        assert cfg.hdc.dim == 3333
    finally:
        os.unlink(tmp_path)


def test_load_with_env_override():
    os.environ["SS_HDC_DIM"] = "2222"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json.dumps({"hdc": {"dim": 1111}}))
        tmp_path = f.name
    try:
        cfg = SpectralStreamConfig.load(tmp_path)
        assert cfg.hdc.dim == 2222
    finally:
        os.environ.pop("SS_HDC_DIM", None)
        os.unlink(tmp_path)


def test_load_without_file():
    cfg = SpectralStreamConfig.load()
    assert cfg.hdc.dim == 10000


def test_merge_without_overwrite():
    cfg = SpectralStreamConfig()
    cfg._merge({"hdc": {"dim": 5000, "ngram_order": 6}})
    assert cfg.hdc.dim == 5000
    assert cfg.hdc.ngram_order == 6


def test_merge_with_overwrite():
    cfg = SpectralStreamConfig()
    cfg.hdc.dim = 100
    cfg._merge({"hdc": {"dim": 5000}}, overwrite=True)
    assert cfg.hdc.dim == 5000


def test_merge_only_existing_attrs():
    cfg = SpectralStreamConfig()
    cfg._merge({"hdc": {"nonexistent_attr": 999}})
    assert not hasattr(cfg.hdc, "nonexistent_attr")


def test_merge_unknown_section():
    cfg = SpectralStreamConfig()
    original_dim = cfg.hdc.dim
    cfg._merge({"unknown_section": {"dim": 999}})
    assert cfg.hdc.dim == original_dim


def test_hardware_config_default_cpu_cores():
    hw = HardwareConfig()
    assert hw.cpu_cores > 0
    assert isinstance(hw.cpu_cores, int)


def test_hardware_config_default_ssd_speed():
    hw = HardwareConfig()
    assert hw.ssd_speed_mbps == 2000


def test_hardware_config_custom_values():
    hw = HardwareConfig(cpu_cores=8, ram_gb=64.0, ssd_speed_mbps=5000)
    assert hw.cpu_cores == 8
    assert hw.ram_gb == 64.0
    assert hw.ssd_speed_mbps == 5000


def test_server_config_port_bounds_valid():
    server = ServerConfig(port=1)
    cfg = SpectralStreamConfig()
    cfg.server = server
    assert len(cfg.validate()) == 0
    server.port = 65535
    assert len(cfg.validate()) == 0


def test_server_config_port_bounds_invalid():
    server = ServerConfig(port=0)
    cfg = SpectralStreamConfig()
    cfg.server = server
    warnings = cfg.validate()
    assert "Server port must be between 1 and 65535" in warnings


def test_server_config_port_negative():
    server = ServerConfig(port=-1)
    cfg = SpectralStreamConfig()
    cfg.server = server
    warnings = cfg.validate()
    assert "Server port must be between 1 and 65535" in warnings


def test_to_file_save_roundtrip():
    cfg = SpectralStreamConfig()
    cfg.hdc.dim = 4444
    cfg.server.port = 9999
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        cfg.to_file(tmp_path)
        with open(tmp_path) as f:
            data = json.load(f)
        assert data["hdc"]["dim"] == 4444
        assert data["server"]["port"] == 9999
        loaded = SpectralStreamConfig.from_file(tmp_path)
        assert loaded.hdc.dim == 4444
        assert loaded.server.port == 9999
    finally:
        os.unlink(tmp_path)


def test_copy_deeply_independent():
    cfg = SpectralStreamConfig()
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.hdc.dim = 123
    assert cfg.hdc.dim == 10000

"""
Target Model Configurations for SpectralStream.

Defines configurations and download instructions for state-of-the-art models
that can run on our hardware (62GB RAM, Ryzen 2700X, RX 6600).

Hardware Budget:
- RAM: ~48GB available after OS
- GPU: RX 6600 8GB (Vulkan, no CUDA)
- SSD: 729GB free NVMe
- CPU: 8C/16T Zen+ (AVX2)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


TARGET_MODEL_REGISTRY: dict[str, dict] = {}


def _register(name: str, config: dict) -> dict:
    TARGET_MODEL_REGISTRY[name] = config
    return config


DEEPSEEK_V4_FLASH_CONFIG = _register(
    "deepseek-v4-flash",
    {
        "name": "DeepSeek V4 Flash",
        "hf_repo": "bartowski/DeepSeek-V4-Flash-GGUF",
        "total_params": 284_000_000_000,
        "active_params": 13_000_000_000,
        "architecture": "moe",
        "n_layers": 64,
        "d_model": 4096,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 100_000,
        "max_seq_len": 1_000_000,
        "context_length": 1_000_000,
        "quant_types": ["q4_k_m", "q4_k_s", "iq4_xs", "q3_k_m", "q2_k", "iq1_m"],
        "recommended_quant": "IQ4_XS",
        "ssd_required": True,
        "ssd_compressed_size_gb": {"q4_k_m": 160, "iq4_xs": 100, "iq1_m": 40},
        "hf_filename_pattern": "*IQ4_XS*.gguf",
    },
)

GLM_5_1_CONFIG = _register(
    "glm-5.1",
    {
        "name": "GLM 5.1",
        "hf_repo": "unsloth/GLM-5.1-GGUF",
        "total_params": 754_000_000_000,
        "active_params": 50_000_000_000,
        "architecture": "moe_dsa",
        "n_layers": 80,
        "d_model": 8192,
        "n_heads": 64,
        "n_kv_heads": 8,
        "vocab_size": 150_000,
        "max_seq_len": 262_144,
        "context_length": 262_144,
        "quant_types": ["q4_k_m", "q3_k_m", "q2_k"],
        "recommended_quant": "Q2_K",
        "ssd_required": True,
        "note": "754B model requires ~420GB at Q4. Only Q2_K (~250GB) is feasible with 729GB SSD + 48GB RAM.",
        "hf_filename_pattern": "*Q2_K*.gguf",
    },
)

QWEN3_6_35B_A3B_CONFIG = _register(
    "qwen3.6-35b-a3b",
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_repo": "bartowski/Qwen3.6-35B-A3B-GGUF",
        "total_params": 35_000_000_000,
        "active_params": 3_500_000_000,
        "architecture": "moe",
        "n_layers": 32,
        "d_model": 5120,
        "n_heads": 40,
        "n_kv_heads": 8,
        "vocab_size": 152_064,
        "max_seq_len": 131_072,
        "context_length": 131_072,
        "quant_types": ["q4_k_m", "q4_k_s", "q3_k_m", "q2_k"],
        "recommended_quant": "Q4_K_M",
        "ssd_required": False,
        "ram_size_gb": {"q4_k_m": 21},
    },
)

GEMMA4_E2B_REGISTRY = _register(
    "gemma-4-e2b",
    {
        "name": "Gemma 4 E2B",
        "hf_repo": "lmstudio-community/gemma-4-E2B-it-GGUF",
        "total_params": 5_100_000_000,
        "active_params": 2_300_000_000,
        "architecture": "gemma4",
        "n_layers": 35,
        "d_model": 1536,
        "n_heads": 8,
        "n_kv_heads": 1,
        "vocab_size": 262_144,
        "max_seq_len": 131_072,
        "context_length": 131_072,
        "quant_types": ["q4_k_m"],
        "recommended_quant": "Q4_K_M",
        "ssd_required": False,
        "ram_size_gb": {"q4_k_m": 2.9},
    },
)

GEMMA4_E4B_REGISTRY = _register(
    "gemma-4-e4b",
    {
        "name": "Gemma 4 E4B",
        "hf_repo": "lmstudio-community/gemma-4-E4B-it-GGUF",
        "total_params": 8_000_000_000,
        "active_params": 4_500_000_000,
        "architecture": "gemma4",
        "n_layers": 42,
        "d_model": 2560,
        "n_heads": 8,
        "n_kv_heads": 2,
        "vocab_size": 262_144,
        "max_seq_len": 131_072,
        "context_length": 131_072,
        "quant_types": ["q4_k_m"],
        "recommended_quant": "Q4_K_M",
        "ssd_required": False,
        "ram_size_gb": {"q4_k_m": 4.6},
    },
)


def list_target_models() -> list[dict]:
    return list(TARGET_MODEL_REGISTRY.values())


def get_target_config(model_key: str) -> Optional[dict]:
    for key, cfg in TARGET_MODEL_REGISTRY.items():
        if (
            key == model_key
            or cfg["name"].lower().replace(" ", "-") == model_key.lower()
        ):
            return dict(cfg)
    return None


def find_model_fuzzy(name_fragment: str) -> Optional[dict]:
    frag = name_fragment.lower()
    for key, cfg in TARGET_MODEL_REGISTRY.items():
        if frag in key.lower() or frag in cfg["name"].lower():
            return dict(cfg)
    return None


def probe_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 16.0


def probe_ssd_free_gb(path: str = "/") -> float:
    try:
        st = os.statvfs(path)
        return (st.f_frsize * st.f_bavail) / (1024**3)
    except Exception:
        return 10.0


def probe_ssd_speed_mbps(path: str = "/tmp/ssd_speed_test") -> float:
    try:
        tmp = Path(path)
        data = os.urandom(256 * 1024 * 1024)
        tmp.write_bytes(data)
        start = os.times().elapsed if hasattr(os.times(), "elapsed") else 0
        _ = tmp.read_bytes()
        elapsed = os.times().elapsed if hasattr(os.times(), "elapsed") else 1.0
        tmp.unlink()
        elapsed = max(elapsed, 0.001)
        return (256.0 / elapsed) * 1000.0
    except Exception:
        return 2000.0


def probe_cpu_cores() -> int:
    return os.cpu_count() or 4


def probe_gpu_vram_gb() -> float:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "Total Memory" in line:
                parts = line.split()
                for p in parts:
                    try:
                        val = float(p.replace("GB", "").replace("MB", ""))
                        if "MB" in p:
                            return val / 1024.0
                        return val
                    except ValueError:
                        continue
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0]) / 1024.0
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "VRAM" in line and "size" in line.lower():
                import re

                m = re.search(r"(\d+)\s*[MG]i?B", line)
                if m:
                    val = int(m.group(1))
                    if "G" in m.group(0).upper():
                        return float(val)
                    return val / 1024.0
    except Exception:
        pass
    return 0.0


@dataclass
class HardwareProfile:
    ram_gb: float = 48.0
    ssd_free_gb: float = 729.0
    ssd_speed_mbps: float = 2000.0
    cpu_cores: int = 8
    gpu_vram_gb: float = 8.0
    gpu_available: bool = False

    @classmethod
    def probe(cls) -> HardwareProfile:
        ram = probe_ram_gb()
        usable_ram = max(ram - 14.0, 4.0)
        ssd_free = probe_ssd_free_gb()
        ssd_speed = probe_ssd_speed_mbps()
        cores = probe_cpu_cores()
        gpu_vram = probe_gpu_vram_gb()
        return cls(
            ram_gb=usable_ram,
            ssd_free_gb=ssd_free,
            ssd_speed_mbps=ssd_speed,
            cpu_cores=cores,
            gpu_vram_gb=gpu_vram,
            gpu_available=gpu_vram > 1.0,
        )


@dataclass
class ModelFitResult:
    model_key: str
    model_name: str
    fits: bool
    strategy: str
    ram_needed_gb: float
    ssd_needed_gb: float
    ram_available_gb: float
    ssd_available_gb: float
    expected_tok_s: float


def compute_model_fit(
    model_key: str,
    quant: str = "q4_k_m",
    hardware: Optional[HardwareProfile] = None,
) -> ModelFitResult:
    hw = hardware or HardwareProfile.probe()
    config = get_target_config(model_key)
    if config is None:
        return ModelFitResult(
            model_key=model_key,
            model_name=model_key,
            fits=False,
            strategy="unknown_model",
            ram_needed_gb=0,
            ssd_needed_gb=0,
            ram_available_gb=hw.ram_gb,
            ssd_available_gb=hw.ssd_free_gb,
            expected_tok_s=0,
        )

    total_params = config.get("total_params", 0)
    active_params = config.get("active_params", total_params)

    # Estimate RAM needed (in GB) for active parameters at fp32
    ram_for_active = active_params * 4 / (1024**3)
    ram_for_full = total_params * 4 / (1024**3)

    # SSD needed for compressed model
    ssd_sizes = config.get("ssd_compressed_size_gb", {})
    ssd_needed = ssd_sizes.get(quant, ram_for_full * 0.25)

    strategies = []
    if ram_for_active <= hw.ram_gb * 0.85:
        strategies.append("ram")
    if ssd_needed <= hw.ssd_free_gb * 0.85:
        strategies.append("ssd_stream")

    if strategies:
        best = strategies[0]
        tok_s = {
            "ram": max(1.0, 50.0 / (active_params / 1e9 + 0.5)),
            "ssd_stream": max(0.1, 5.0 / (active_params / 1e9 + 0.5)),
        }.get(best, 0.5)
        return ModelFitResult(
            model_key=model_key,
            model_name=config["name"],
            fits=True,
            strategy=best,
            ram_needed_gb=ram_for_active,
            ssd_needed_gb=ssd_needed,
            ram_available_gb=hw.ram_gb,
            ssd_available_gb=hw.ssd_free_gb,
            expected_tok_s=round(tok_s, 2),
        )

    return ModelFitResult(
        model_key=model_key,
        model_name=config["name"],
        fits=False,
        strategy="no_strategy",
        ram_needed_gb=ram_for_active,
        ssd_needed_gb=ssd_needed,
        ram_available_gb=hw.ram_gb,
        ssd_available_gb=hw.ssd_free_gb,
        expected_tok_s=0,
    )


@dataclass
class DownloadResult:
    success: bool
    path: str
    message: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "path": self.path,
            "message": self.message,
            "error": self.error,
        }


_download_progress: dict[str, dict] = {}
_download_progress_lock = threading.Lock()


def download_model(
    model_key: str,
    output_dir: str = ".",
    quant: str = "q4_k_m",
    hf_token: Optional[str] = None,
) -> DownloadResult:
    config = get_target_config(model_key)
    if config is None:
        return DownloadResult(
            success=False,
            path="",
            message=f"Unknown model: {model_key}",
            error="model_not_found",
        )

    hf_repo = config.get("hf_repo", "")
    pattern = config.get("hf_filename_pattern", f"*{quant}*.gguf")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dl_id = f"{model_key}_{quant}"
    with _download_progress_lock:
        _download_progress[dl_id] = {
            "status": "starting",
            "progress": 0.0,
            "speed_mbps": 0.0,
        }

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        import fnmatch

        repo_files = []
        try:
            from huggingface_hub import list_repo_files

            repo_files = list_repo_files(hf_repo, token=hf_token)
        except Exception:
            pass

        matching = fnmatch.filter(repo_files, pattern) if repo_files else []
        if matching:
            target_file = matching[0]
        else:
            target_file = pattern.replace("*", "")

        with _download_progress_lock:
            _download_progress[dl_id]["status"] = "downloading"

        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename=target_file,
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
            token=hf_token,
            resume=True,
        )

        with _download_progress_lock:
            _download_progress[dl_id]["status"] = "done"
            _download_progress[dl_id]["progress"] = 100.0

        return DownloadResult(
            success=True,
            path=downloaded,
            message=f"Downloaded {config['name']} ({quant}) to {downloaded}",
        )
    except Exception as e:
        with _download_progress_lock:
            _download_progress[dl_id]["status"] = "error"
        return DownloadResult(
            success=False,
            path="",
            message=f"Download failed: {e}",
            error=str(e),
        )


def get_download_progress(dl_id: str) -> Optional[dict]:
    with _download_progress_lock:
        return _download_progress.get(dl_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model target management")
    parser.add_argument("action", choices=["list", "fit", "download"])
    parser.add_argument("--model", help="Model key or name fragment")
    parser.add_argument("--quant", default="q4_k_m", help="Quantization type")
    parser.add_argument("--output-dir", default=".", help="Download directory")
    args = parser.parse_args()

    if args.action == "list":
        for cfg in list_target_models():
            print(
                f"{cfg['name']:40s} {cfg['architecture']:15s} "
                f"{cfg['total_params'] / 1e9:.0f}B param  "
                f"repo: {cfg['hf_repo']}"
            )

    elif args.action == "fit":
        hw = HardwareProfile.probe()
        print(
            f"Hardware: {hw.ram_gb:.0f}GB RAM, {hw.ssd_free_gb:.0f}GB SSD, "
            f"{hw.cpu_cores}C, {hw.gpu_vram_gb:.0f}GB GPU"
        )
        print()
        for cfg in list_target_models():
            result = compute_model_fit(cfg.get("hf_repo", ""), args.quant, hw)
            status = "✓" if result.fits else "✗"
            print(
                f"{status} {cfg['name']:35s} "
                f"RAM: {result.ram_needed_gb:>5.0f}GB "
                f"SSD: {result.ssd_needed_gb:>5.0f}GB "
                f"T/s: {result.expected_tok_s:>5.1f}  [{result.strategy}]"
            )

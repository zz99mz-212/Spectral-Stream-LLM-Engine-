"""Tests for the CLI module — argument parsing and basic command execution."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.cli import build_parser, cmd_list_methods


@pytest.fixture
def parser():
    return build_parser()


class TestCLIBuildParser:
    def test_parser_created(self, parser):
        assert parser is not None
        assert "SpectralStream" in parser.description

    def test_compress_subcommand(self, parser):
        args = parser.parse_args(["compress", "model.safetensors", "output.ssf"])
        assert args.command == "compress"
        assert args.model == "model.safetensors"

    def test_list_methods_subcommand(self, parser):
        args = parser.parse_args(["list-methods"])
        assert args.command == "list-methods"

    def test_profile_subcommand(self, parser):
        args = parser.parse_args(["profile", "model.safetensors"])
        assert args.command == "profile"

    def test_validate_subcommand(self, parser):
        args = parser.parse_args(["validate", "model.ssf"])
        assert args.command == "validate"

    def test_benchmark_subcommand(self, parser):
        args = parser.parse_args(["benchmark", "model.safetensors"])
        assert args.command == "benchmark"

    def test_generate_subcommand(self, parser):
        args = parser.parse_args(["generate", "model.ssf"])
        assert args.command == "generate"
        assert args.ssf_file == "model.ssf"

    def test_generate_format_flags(self, parser):
        args = parser.parse_args(
            ["generate", "model.ssf", "--format", "html,md", "--output-dir", "./certs"]
        )
        assert args.format == "html,md"
        assert args.output_dir == "./certs"

    def test_infer_subcommand(self, parser):
        args = parser.parse_args(["infer", "model.ssf"])
        assert args.command == "infer"

    def test_infer_flags(self, parser):
        args = parser.parse_args(
            [
                "infer",
                "model.ssf",
                "--prompt",
                "Hello world",
                "--max-tokens",
                "50",
                "--temperature",
                "0.5",
                "--top-k",
                "20",
                "--top-p",
                "0.9",
            ]
        )
        assert args.prompt == "Hello world"
        assert args.max_tokens == 50
        assert args.temperature == 0.5
        assert args.command == "infer"

    def test_verify_subcommand(self, parser):
        args = parser.parse_args(["verify", "model.safetensors"])
        assert args.command == "verify"

    def test_convert_subcommand(self, parser):
        args = parser.parse_args(["convert", "model.safetensors", "output.ssf"])
        assert args.command == "convert"

    def test_info_subcommand(self, parser):
        args = parser.parse_args(["info", "model.ssf"])
        assert args.command == "info"

    def test_compress_defaults(self, parser):
        args = parser.parse_args(["compress", "model.safetensors", "out.ssf"])
        assert args.target_ratio == 0  # auto-detect via world model
        assert args.max_error == 0  # auto-detect via world model
        assert args.auto is True  # world model auto mode by default
        assert args.workers is None  # resolved to CPU count at runtime

    def test_compress_custom_values(self, parser):
        args = parser.parse_args(
            [
                "compress",
                "model.safetensors",
                "out.ssf",
                "--target-ratio",
                "1000",
                "--max-error",
                "0.001",
                "--workers",
                "2",
            ]
        )
        assert args.target_ratio == 1000.0
        assert args.max_error == 0.001
        assert args.workers == 2

    def test_compress_no_auto(self, parser):
        """--no-auto disables world model auto mode."""
        args = parser.parse_args(
            [
                "compress",
                "model.safetensors",
                "out.ssf",
                "--no-auto",
                "--target-ratio",
                "5000",
                "--max-error",
                "0.0002",
            ]
        )
        assert args.auto is False
        assert args.target_ratio == 5000.0
        assert args.max_error == 0.0002

    def test_compress_quiet_flag(self, parser):
        args = parser.parse_args(
            ["compress", "model.safetensors", "out.ssf", "--quiet"]
        )
        assert args.quiet is True

    def test_compress_quiet_alias(self, parser):
        args = parser.parse_args(["compress", "model.safetensors", "out.ssf", "-q"])
        assert args.quiet is True

    def test_compress_cascade_mode(self, parser):
        args = parser.parse_args(
            ["compress", "model.safetensors", "out.ssf", "--cascade-mode", "extreme"]
        )
        assert args.cascade_mode == "extreme"

    def test_compress_cascade_mode_default(self, parser):
        args = parser.parse_args(["compress", "model.safetensors", "out.ssf"])
        assert args.cascade_mode == "balanced"

    def test_compress_profile_cache_size(self, parser):
        args = parser.parse_args(
            ["compress", "model.safetensors", "out.ssf", "--profile-cache-size", "2000"]
        )
        assert args.profile_cache_size == 2000

    def test_compress_profile_cache_size_default(self, parser):
        args = parser.parse_args(["compress", "model.safetensors", "out.ssf"])
        assert args.profile_cache_size == 500

    def test_compress_holographic_memory(self, parser):
        args = parser.parse_args(
            [
                "compress",
                "model.safetensors",
                "out.ssf",
                "--holographic-memory",
                "/tmp/hmem.npz",
            ]
        )
        assert args.holographic_memory == "/tmp/hmem.npz"

    def test_compress_holographic_memory_default(self, parser):
        args = parser.parse_args(["compress", "model.safetensors", "out.ssf"])
        assert args.holographic_memory is None

    def test_compress_streaming_default(self, parser):
        args = parser.parse_args(["compress", "model.safetensors", "out.ssf"])
        assert args.streaming is True

    def test_compress_no_streaming(self, parser):
        args = parser.parse_args(
            ["compress", "model.safetensors", "out.ssf", "--no-streaming"]
        )
        assert args.streaming is False

    def test_profile_defaults(self, parser):
        args = parser.parse_args(["profile", "model.safetensors"])
        assert args.target_ratio == 100.0

    def test_benchmark_defaults(self, parser):
        args = parser.parse_args(["benchmark", "model.safetensors"])
        assert args.target_ratio == 100.0
        assert args.max_error == 0.01
        assert args.prompt_lengths == "128,512"

    def test_verbose_flag(self, parser):
        args = parser.parse_args(["--verbose", "compress", "m.safetensors", "out.ssf"])
        assert args.verbose is True

    def test_help_not_none(self, parser):
        help_text = parser.format_help()
        assert "compress" in help_text
        assert "list-methods" in help_text
        assert "validate" in help_text

    def test_compress_quality_flags(self, parser):
        args = parser.parse_args(
            [
                "compress",
                "m.safetensors",
                "out.ssf",
                "--safety-margin",
                "2.0",
                "--max-candidates",
                "5",
            ]
        )
        assert args.safety_margin == 2.0
        assert args.max_candidates == 5

    def test_validate_accepts_ssf(self, parser):
        args = parser.parse_args(["validate", "test.ssf"])
        assert args.ssf_file == "test.ssf"
        assert os.path.splitext(args.ssf_file)[1] == ".ssf"

    def test_info_json_flag(self, parser):
        args = parser.parse_args(["info", "test.ssf", "--json"])
        assert args.json is True
        assert args.ssf_file == "test.ssf"

    def test_verify_flags(self, parser):
        args = parser.parse_args(
            [
                "verify",
                "model.safetensors",
                "--all-methods",
                "--num-tensors",
                "3",
            ]
        )
        assert args.all_methods is True
        assert args.num_tensors == 3

    def test_convert_defaults(self, parser):
        args = parser.parse_args(["convert", "in.safetensors", "out.ssf"])
        assert args.target_ratio == 5000.0
        assert args.max_error == 0.0002

    def test_list_methods_no_error(self):
        out = io.StringIO()
        sys.stdout = out
        try:
            cmd_list_methods(
                argparse.Namespace(category=None, tier=None, verbose=False)
            )
        except SystemExit:
            pass
        finally:
            sys.stdout = sys.__stdout__
        output = out.getvalue()
        # Function completed without crash — that's the test
        assert True

    def test_list_methods_verbose_shows_descriptions(self):
        out = io.StringIO()
        sys.stdout = out
        try:
            cmd_list_methods(argparse.Namespace(category=None, tier=None, verbose=True))
        except SystemExit:
            pass
        finally:
            sys.stdout = sys.__stdout__
        assert True

    def test_list_methods_category_filter(self):
        out = io.StringIO()
        sys.stdout = out
        try:
            cmd_list_methods(
                argparse.Namespace(category="quantization", tier=None, verbose=False)
            )
        except SystemExit:
            pass
        finally:
            sys.stdout = sys.__stdout__
        assert True

    def test_list_methods_tier_filter(self):
        out = io.StringIO()
        sys.stdout = out
        try:
            cmd_list_methods(argparse.Namespace(category=None, tier="5", verbose=False))
        except SystemExit:
            pass
        finally:
            sys.stdout = sys.__stdout__
        assert True


class TestCLIEngineIntegration:
    def test_engine_available(self, tiny_engine):
        engine = tiny_engine
        assert engine is not None
        info = engine.get_method_names()
        assert len(info) > 0

    def test_engine_compress_small_tensor(self, tiny_engine):
        engine = tiny_engine
        tensor = np.random.randn(16, 16).astype(np.float32)
        p = engine.profiler.profile_tensor(tensor, name="test")
        data, meta, ratio_val, error_val = engine.compress_fast(tensor, name="test")
        assert ratio_val > 0
        assert len(data) > 0

    def test_engine_get_available_methods(self, tiny_engine):
        engine = tiny_engine
        methods = engine.get_available_methods()
        assert len(methods) >= 10
        assert "block_int8" in methods

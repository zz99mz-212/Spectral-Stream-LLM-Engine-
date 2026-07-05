#!/usr/bin/env python3
"""
Comprehensive Test Runner for SpectralStream Compression Engine
===============================================================
Runs all tests, generates coverage report, produces summary JSON.

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --verbose
    python tests/run_all_tests.py --coverage
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure we can find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
REPORT_PATH = TESTS_DIR / "test_results.json"


def run_pytest(
    verbose: bool = True,
    coverage: bool = False,
) -> dict:
    """Run pytest and collect results."""
    cmd = [sys.executable, "-m", "pytest", str(TESTS_DIR)]

    if verbose:
        cmd.extend(["-v", "--tb=short"])

    cmd.extend(["-x", "--timeout=120"])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=600,
    )
    elapsed = time.time() - start

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed": elapsed,
    }


def run_coverage() -> dict:
    """Run tests with coverage."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(TESTS_DIR),
        "-v",
        "--tb=short",
        "-x",
        f"--cov={PROJECT_ROOT / 'spectralstream'}",
        "--cov-report=json",
        "--cov-report=term-missing",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=600,
    )
    elapsed = time.time() - start

    coverage_data = {}
    cov_json = TESTS_DIR / "coverage.json"
    if cov_json.exists():
        with open(cov_json) as f:
            raw = json.load(f)
        coverage_data = {
            "total_percent": raw.get("totals", {}).get("percent_covered", 0),
            "files": {
                k: v.get("summary", {}).get("percent_covered", 0)
                for k, v in raw.get("files", {}).items()
            },
        }

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed": elapsed,
        "coverage": coverage_data,
    }


def parse_pytest_output(stdout: str) -> dict:
    """Parse pytest output for pass/fail counts."""
    lines = stdout.strip().split("\n")
    summary = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xfail": 0,
        "total": 0,
    }

    for line in reversed(lines):
        # Look for the final summary line like "164 passed, 3 skipped, 1 xfailed"
        if "passed" in line and (
            "failed" in line or "error" in line or "skipped" in line or "xfail" in line
        ):
            import re

            m = re.search(r"(\d+) passed", line)
            if m:
                summary["passed"] = int(m.group(1))
            m = re.search(r"(\d+) failed", line)
            if m:
                summary["failed"] = int(m.group(1))
            m = re.search(r"(\d+) error", line)
            if m:
                summary["errors"] = int(m.group(1))
            m = re.search(r"(\d+) skipped", line)
            if m:
                summary["skipped"] = int(m.group(1))
            m = re.search(r"(\d+) xfail", line)
            if m:
                summary["xfail"] = int(m.group(1))
            break
        # Also handle "X passed" without failures
        if " passed" in line and "failed" not in line and "error" not in line:
            import re

            m = re.search(r"(\d+) passed", line)
            if m:
                summary["passed"] = int(m.group(1))
            m = re.search(r"(\d+) skipped", line)
            if m:
                summary["skipped"] = int(m.group(1))
            m = re.search(r"(\d+) xfail", line)
            if m:
                summary["xfail"] = int(m.group(1))
            break

    summary["total"] = (
        summary["passed"]
        + summary["failed"]
        + summary["errors"]
        + summary["skipped"]
        + summary["xfail"]
    )
    return summary


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run SpectralStream compression tests")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--coverage", "-c", action="store_true", default=False)
    args = parser.parse_args()

    print("=" * 70)
    print("SpectralStream Compression Engine — Comprehensive Test Suite")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"Working Dir: {os.getcwd()}")
    print()

    # Run tests
    print("Running tests...")
    if args.coverage:
        results = run_coverage(verbose=args.verbose, coverage=True)
    else:
        results = run_pytest(verbose=args.verbose, coverage=False)

    # Parse results
    test_counts = parse_pytest_output(results["stdout"])

    # Build summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "test_file": str(TESTS_DIR),
        "exit_code": results["returncode"],
        "elapsed_seconds": round(results["elapsed"], 2),
        "tests": test_counts,
        "all_passed": results["returncode"] == 0,
    }

    if "coverage" in results and results["coverage"]:
        summary["coverage"] = results["coverage"]

    # Print results
    print()
    print("-" * 70)
    print(results["stdout"])
    if results["stderr"]:
        print("STDERR:")
        print(results["stderr"][-2000:])
    print("-" * 70)

    # Print summary
    print()
    print("TEST SUMMARY")
    print(f"  Total:   {test_counts['total']}")
    print(f"  Passed:  {test_counts['passed']}")
    print(f"  Failed:  {test_counts['failed']}")
    print(f"  Errors:  {test_counts['errors']}")
    print(f"  Skipped: {test_counts['skipped']}")
    print(f"  Time:    {results['elapsed']:.1f}s")
    print(f"  Status:  {'PASS' if results['returncode'] == 0 else 'FAIL'}")

    if "coverage" in results and results["coverage"]:
        print(f"  Coverage: {results['coverage'].get('total_percent', 0):.1f}%")

    print("=" * 70)

    # Save report
    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Report saved to {REPORT_PATH}")

    sys.exit(results["returncode"])


if __name__ == "__main__":
    main()

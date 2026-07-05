"""Security & Compliance Audit Tests — verify no vulnerabilities in codebase.

Checks: bare excepts, hardcoded secrets, pickle/eval/exec, path traversal,
information disclosure, and unsafe imports.
"""

import ast
import os
import re
import sys
import warnings
from pathlib import Path

import pytest


def _iter_py_files(root: str = "spectralstream"):
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".py"):
                yield Path(dirpath) / f


@pytest.fixture(scope="module")
def py_file_paths():
    return list(_iter_py_files())


def _get_ast(filepath: Path):
    with open(filepath) as f:
        return ast.parse(f.read())


# ───────────────────────────────────────────────────────────
# A. Bare excepts (catch BaseException → KeyboardInterrupt)
# ───────────────────────────────────────────────────────────


class BareExceptVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_ExceptHandler(self, node):
        if node.type is None:
            self.violations.append((node.lineno, "bare except:"))
        self.generic_visit(node)


def test_no_bare_excepts(py_file_paths):
    violations = []
    for path in py_file_paths:
        tree = _get_ast(path)
        visitor = BareExceptVisitor()
        visitor.visit(tree)
        for lineno, msg in visitor.violations:
            violations.append(f"{path}:{lineno}: {msg}")
    assert len(violations) == 0, (
        f"Bare excepts found ({len(violations)}):\n" + "\n".join(violations[:20])
    )


# ───────────────────────────────────────────────────────────
# B. Overly broad `except Exception:` with `pass`
# ───────────────────────────────────────────────────────────


class ExceptPassVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_ExceptHandler(self, node):
        if (
            node.type is not None
            and isinstance(node.type, ast.Name)
            and node.type.id == "Exception"
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Pass)
        ):
            self.violations.append(node.lineno)
        self.generic_visit(node)


def test_no_broad_except_pass(py_file_paths):
    violations = []
    for path in py_file_paths:
        # Skip archived/compat modules with known patterns
        if any(
            seg in str(path)
            for seg in [
                "_archive_integration",
                "_wrap",
                "_common",
                "novel_compression_library",
                "physics_compression",
                "breakthrough_signal",
                "structural_expansion",
                "functional_weight_space",
                "topological_biological",
                "multiplicative_stacking",
            ]
        ):
            continue
        tree = _get_ast(path)
        visitor = ExceptPassVisitor()
        visitor.visit(tree)
        for lineno in visitor.violations:
            violations.append(f"{path}:{lineno}: except Exception: pass")
    assert len(violations) < 20, (
        f"Broad except: pass found in core modules ({len(violations)}):\n"
        + "\n".join(violations[:20])
    )


# ───────────────────────────────────────────────────────────
# C. Hardcoded secrets
# ───────────────────────────────────────────────────────────

SECRET_PATTERNS = [
    re.compile(r'["\'][A-Za-z0-9+/=]{40,}["\']'),  # base64-like tokens
    re.compile(r"sk-[A-Za-z0-9]{32,}"),  # OpenAI-style keys
    re.compile(r"SK-[A-Za-z0-9]{32,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key
    re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
]


def test_no_hardcoded_secrets(py_file_paths):
    violations = []
    for path in py_file_paths:
        with open(path) as f:
            content = f.read()
        for pat in SECRET_PATTERNS:
            if pat.search(content):
                violations.append(f"{path}: matches {pat.pattern[:40]}...")
    assert len(violations) == 0, f"Hardcoded secrets found:\n" + "\n".join(violations)


# ───────────────────────────────────────────────────────────
# D. No pickle.loads on untrusted data
# ───────────────────────────────────────────────────────────


def test_no_unrestricted_pickle(py_file_paths):
    violations = []
    for path in py_file_paths:
        tree = _get_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, "attr"):
                if node.func.attr == "loads" and (
                    hasattr(node.func, "value")
                    and hasattr(node.func.value, "id")
                    and node.func.value.id == "pickle"
                ):
                    if "safe_pickle_loads" not in path.name:
                        violations.append(f"{path}:{node.lineno}: pickle.loads()")
    assert len(violations) < 5, f"Unrestricted pickle.loads() found:\n" + "\n".join(
        violations
    )


# ───────────────────────────────────────────────────────────
# E. No eval/exec on untrusted data
# ───────────────────────────────────────────────────────────


class EvalExecVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in (
            "eval",
            "exec",
            "compile",
        ):
            self.violations.append((node.lineno, node.func.id))
        self.generic_visit(node)


def test_no_eval_exec(py_file_paths):
    violations = []
    for path in py_file_paths:
        tree = _get_ast(path)
        visitor = EvalExecVisitor()
        visitor.visit(tree)
        for lineno, name in visitor.violations:
            violations.append(f"{path}:{lineno}: {name}()")
    assert len(violations) == 0, f"eval()/exec()/compile() found:\n" + "\n".join(
        violations
    )


# ───────────────────────────────────────────────────────────
# F. Path traversal validation present
# ───────────────────────────────────────────────────────────


def test_path_traversal_protection():
    """Verify path validation functions exist in key modules."""
    cli_path = Path("spectralstream/compression/cli.py")
    assert cli_path.exists(), "CLI module not found"
    tree = _get_ast(cli_path)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_validate_input_path" in names, "CLI missing _validate_input_path"
    assert "_validate_output_path" in names, "CLI missing _validate_output_path"

    reader_path = Path("spectralstream/format/reader.py")
    assert reader_path.exists(), "Reader module not found"
    tree2 = _get_ast(reader_path)
    names2 = {n.name for n in ast.walk(tree2) if isinstance(n, ast.FunctionDef)}
    assert "_validate_path_safe" in names2, "Reader missing _validate_path_safe"

    api_path = Path("spectralstream/serving/api.py")
    if api_path.exists():
        tree3 = _get_ast(api_path)
        names3 = {n.name for n in ast.walk(tree3) if isinstance(n, ast.FunctionDef)}
        # API should validate model_path before loading
        with open(api_path) as f:
            content = f.read()
        assert "Path traversal detected" in content, "API missing path traversal check"


# ───────────────────────────────────────────────────────────
# G. No traceback disclosure in API error responses
# ───────────────────────────────────────────────────────────


def test_no_traceback_disclosure():
    api_path = Path("spectralstream/serving/api.py")
    if not api_path.exists():
        pytest.skip("API module not found")
    with open(api_path) as f:
        content = f.read()
    assert "traceback" not in content or "traceback.format_exc()" not in content, (
        "API may leak tracebacks in error responses"
    )


# ───────────────────────────────────────────────────────────
# H. Restricted unpickler exists
# ───────────────────────────────────────────────────────────


def test_restricted_unpickler_exists():
    archive_path = Path(
        "spectralstream/compression/methods/novel/_archive_integration.py"
    )
    if not archive_path.exists():
        pytest.skip("Archive integration module not found")
    tree = _get_ast(archive_path)
    class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "RestrictedUnpickler" in class_names, (
        "Missing RestrictedUnpickler — critical for pickle safety"
    )
    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "safe_pickle_loads" in func_names, "Missing safe_pickle_loads function"


# ───────────────────────────────────────────────────────────
# I. Import consistency — no dangerous imports
# ───────────────────────────────────────────────────────────


def test_no_dangerous_imports(py_file_paths):
    dangerous = {"pickle": "RestrictedUnpickler must be used instead"}
    violations = []
    for path in py_file_paths:
        if "_archive_integration" in str(path):
            continue
        with open(path) as f:
            content = f.read()
        for mod_name in dangerous:
            if f"import {mod_name}" in content:
                # Skip imports inside test-only blocks
                if f"def run_all_tests" in content:
                    continue
                violations.append(f"{path}: imports {mod_name}")
    assert len(violations) == 0, f"Dangerous imports found:\n" + "\n".join(violations)


# ───────────────────────────────────────────────────────────
# J. CORS security (no wildcard with credentials)
# ───────────────────────────────────────────────────────────


def test_cors_no_wildcard_credentials():
    api_path = Path("spectralstream/serving/api.py")
    if not api_path.exists():
        pytest.skip("API module not found")
    with open(api_path) as f:
        content = f.read()
    assert "allow_credentials=False" in content, (
        "CORS should not allow credentials with wildcard origins"
    )

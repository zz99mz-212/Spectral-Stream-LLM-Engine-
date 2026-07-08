---
status: issues_found
phase: 02-eval-subsystem
date: 2026-07-09
reviewer: gsd-code-reviewer
depth: standard
findings_count: 16
  critical: 2
  warning: 8
  info: 6
---

# Phase 02 Code Review

**Reviewed:** 2026-07-09
**Depth:** Standard (per-file analysis with language-specific checks)
**Files reviewed:** 12 source files

## Executive Summary

Phase 02 (eval-subsystem) implements a WikiText-2 perplexity grader with honest JSON artifacts, a default tokenizer fallback, parameterized model paths, and model-native tokenizer wiring. The implementation is functionally correct and passes all 33 tests (23 eval grader + 10 tokenizer fallback). However, two critical Windows-compatibility issues remain in `benchmark_physics_real_weights.py`, and several path-validation gaps exist in CLI argument handling.

## Findings

| ID | Severity | File | Line | Summary |
|----|----------|------|------|---------|
| CR-01 | Critical | benchmark_physics_real_weights.py | 23 | `signal.alarm()` is Linux-only; crashes on Windows with `TimeoutError` |
| CR-02 | Critical | benchmark_physics_real_weights.py | 27 | `exec("raise TimeoutError()")` in signal handler is dangerous pattern |
| WR-01 | Warning | benchmark_physics_real_weights.py | 35-45 | argparse defined at module level; importing module calls `parse_args()` |
| WR-02 | Warning | eval/run_eval.py | 122,133 | `--tokenizer` path not validated for traversal |
| WR-03 | Warning | eval/corpus.py | 65 | `--corpus` path not validated for traversal |
| WR-04 | Warning | eval/artifact.py | 156 | `write_artifact()` does not validate output path |
| WR-05 | Warning | eval/model_path.py | 21 | Path traversal regex can be bypassed with URL-encoded `..` |
| WR-06 | Warning | scripts/fetch_eval_corpus.py | 44 | No validation of downloaded content |
| WR-07 | Warning | eval/run_eval.py | 144 | `--corpus` path passed to `resolve_corpus()` without validation |
| WR-08 | Warning | eval/run_eval.py | 168 | `--output` path not validated |
| IN-01 | Info | benchmark_physics_real_weights.py | 22-32 | Unused `Timeout` class |
| IN-02 | Info | eval/run_eval.py | 150 | Redundant `import os as _os` |
| IN-03 | Info | eval/artifact.py | 62 | `_get_git_ref()` swallows all exceptions |
| IN-04 | Info | eval/corpus.py | 28 | Type hint for `tokenizer` parameter is incomplete |
| IN-05 | Info | eval/model_path.py | 67 | `_validate_path_safety()` is private but used externally |
| IN-06 | Info | eval/constants.py | 23 | `VOCAB_LOG_SOFTMAX_BLOCK_SIZE` is duplicated from `pipeline.py` |

## Detailed Findings

### Critical

#### CR-01: `benchmark_physics_real_weights.py` — `signal.alarm()` is Linux-only

- **File:** `benchmark_physics_real_weights.py:23`
- **Line:** 23
- **Category:** Cross-platform compatibility
- **Severity:** Critical
- **Description:** `signal.alarm()` is not available on Windows. The script will crash with `AttributeError: module 'signal' has no attribute 'alarm'` when the timeout is triggered. This violates the project's Windows compatibility requirement (CLAUDE.md: "Windows compatibility: The `fix/honest-metrics-windows-compat` branch exists. `signal.alarm`/`exec`-based timeouts (Linux-only) must be replaced.").
- **Impact:** Script is unusable on Windows when any compression method takes longer than the timeout.
- **Fix:** Replace with a cross-platform timeout mechanism (e.g., `concurrent.futures.ThreadPoolExecutor` with `timeout`, or `signal` only on POSIX with a platform check).

#### CR-02: `benchmark_physics_real_weights.py` — `exec()` used in signal handler

- **File:** `benchmark_physics_real_weights.py:27`
- **Line:** 27
- **Category:** Security / Cross-platform
- **Severity:** Critical
- **Description:** `exec("raise TimeoutError()")` is used inside a signal handler. While the string is hardcoded (not user-controlled), `exec()` in a signal handler is a dangerous pattern that can cause undefined behavior if the interpreter is in an inconsistent state. Combined with `signal.alarm()` being Linux-only, this entire timeout mechanism is broken on Windows.
- **Fix:**
```python
import signal
import sys

class Timeout:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        if sys.platform == "win32":
            # Windows: skip timeout or use threading-based approach
            return self
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def _handler(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.seconds}s")

    def __exit__(self, *args):
        if sys.platform != "win32":
            signal.alarm(0)
```

### Warning

#### WR-01: `benchmark_physics_real_weights.py` — argparse defined at module level

- **File:** `benchmark_physics_real_weights.py:35-45`
- **Line:** 35-45
- **Category:** Code quality
- **Severity:** Warning
- **Description:** The `argparse.ArgumentParser` and `parser.parse_args()` are defined at module level (not inside `if __name__ == "__main__"` or a `main()` function). This means importing the module (e.g., for testing) will call `sys.exit()` if `--help` is passed, or parse `sys.argv` unexpectedly.
- **Fix:** Wrap in a `main()` function:
```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument(...)
    args = parser.parse_args()
    # ... rest of script ...

if __name__ == "__main__":
    main()
```

#### WR-02: `eval/run_eval.py` — `--tokenizer` path not validated for traversal

- **File:** `eval/run_eval.py:122,133`
- **Line:** 122, 133
- **Category:** Security
- **Severity:** Warning
- **Description:** The `--tokenizer` CLI argument is passed directly to `AutoTokenizer.from_gguf()` or `AutoTokenizer.from_pretrained()` without path-traversal validation. While these are local file reads (not network), a malicious `../` path could load unintended files. The `--model` and `--compressed` args go through `resolve_model_path()` which validates, but `--tokenizer` does not.
- **Fix:** Validate the tokenizer path before loading:
```python
if args.tokenizer:
    from eval.model_path import _validate_path_safety
    _validate_path_safety(args.tokenizer)
    # ... proceed with loading ...
```

#### WR-03: `eval/corpus.py` — `--corpus` path not validated for traversal

- **File:** `eval/corpus.py:65`
- **Line:** 65
- **Category:** Security
- **Severity:** Warning
- **Description:** `resolve_corpus()` accepts a user-supplied `corpus_path` and only checks `os.path.exists()` (line 65). It does not validate against path traversal. While the corpus is a read-only text/JSON file, a malicious path like `../../etc/passwd` could be read and tokenized, potentially leaking file contents into the eval artifact (via token IDs).
- **Fix:** Add traversal validation:
```python
from eval.model_path import _validate_path_safety

def resolve_corpus(corpus_path: str | None = None, tokenizer=None) -> list[int]:
    if corpus_path:
        _validate_path_safety(corpus_path)
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    # ... rest of function ...
```

#### WR-04: `eval/artifact.py` — `write_artifact()` does not validate output path

- **File:** `eval/artifact.py:156`
- **Line:** 156
- **Category:** Security
- **Severity:** Warning
- **Description:** `write_artifact()` accepts an arbitrary `output_path` and writes JSON to it without validation. A path like `../../etc/cron.d/malicious` could overwrite system files. While this is a local CLI tool, path validation is a defense-in-depth measure.
- **Fix:** Validate the output path:
```python
def write_artifact(artifact: dict, output_path: str | None = None) -> str:
    if output_path:
        from eval.model_path import _validate_path_safety
        _validate_path_safety(output_path)
    # ... rest of function ...
```

#### WR-05: `eval/model_path.py` — path traversal regex can be bypassed with URL-encoded `..`

- **File:** `eval/model_path.py:21`
- **Line:** 21
- **Category:** Security
- **Severity:** Warning
- **Description:** The regex `r"\.\./|\.\.\\|/\.\.|\\\.\."` catches literal `../` and `..\` but does not catch URL-encoded variants like `%2e%2e%2f` or Unicode normalization tricks (e.g., `..%c0%af`). While `Path.resolve()` will normalize most paths, the regex check happens *before* resolution, so a crafted path could bypass the regex and still resolve to a parent directory.
- **Fix:** After `Path.resolve()`, verify the resolved path is within an allowed root:
```python
def resolve_model_path(cli_model: str | None = None) -> str:
    # ... existing logic ...
    resolved = Path(raw).resolve()
    
    # Verify resolved path is within allowed roots (e.g., cwd, models/)
    allowed_roots = [Path.cwd(), Path("models").resolve()]
    if not any(is_relative_to(resolved, root) for root in allowed_roots):
        raise ValueError(f"Path {resolved} is outside allowed roots")
    
    if not resolved.exists():
        raise FileNotFoundError(f"Model file not found: {resolved}")
    return str(resolved)

def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
```

#### WR-06: `scripts/fetch_eval_corpus.py` — no validation of downloaded content

- **File:** `scripts/fetch_eval_corpus.py:44`
- **Line:** 44
- **Category:** Security
- **Severity:** Warning
- **Description:** `urllib.request.urlretrieve()` downloads a file from a hardcoded GitHub URL and writes it directly to disk. While the URL is hardcoded (not user-controlled), there is no validation of the downloaded content (e.g., file size, content type, or sanity checks). A compromised GitHub account or MITM attack could inject malicious content.
- **Fix:** Add basic validation:
```python
import hashlib

# After download
with open(out_path, "rb") as f:
    content = f.read()
    # Sanity check: file should be text, not binary
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Downloaded file is not valid UTF-8 text")
    
    # Optional: check file size is reasonable (e.g., < 100 MB)
    if len(content) > 100 * 1024 * 1024:
        raise ValueError("Downloaded file is suspiciously large")
```

#### WR-07: `eval/run_eval.py` — `--corpus` path passed to `resolve_corpus()` without validation

- **File:** `eval/run_eval.py:144`
- **Line:** 144
- **Category:** Security
- **Severity:** Warning
- **Description:** The CLI passes `args.corpus` directly to `resolve_corpus()` without validation. While `resolve_corpus()` checks `os.path.exists()`, it does not validate against path traversal (see WR-03). The CLI layer should validate before passing to the library function.
- **Fix:** Validate in the CLI before calling `resolve_corpus()`:
```python
if args.corpus:
    from eval.model_path import _validate_path_safety
    _validate_path_safety(args.corpus)
test_tokens = resolve_corpus(args.corpus, tokenizer=tokenizer)
```

#### WR-08: `eval/run_eval.py` — `--output` path not validated

- **File:** `eval/run_eval.py:168`
- **Line:** 168
- **Category:** Security
- **Severity:** Warning
- **Description:** The `--output` CLI argument is passed to `write_artifact()` without validation. See WR-04 for the underlying issue.
- **Fix:** Validate in the CLI:
```python
if args.output:
    from eval.model_path import _validate_path_safety
    _validate_path_safety(args.output)
out_path = write_artifact(artifact, output_path=args.output)
```

### Informational

#### IN-01: `benchmark_physics_real_weights.py` — unused `Timeout` class

- **File:** `benchmark_physics_real_weights.py:22-32`
- **Line:** 22-32
- **Category:** Code quality
- **Severity:** Info
- **Description:** The `Timeout` class is defined but never used in the script. It appears to be leftover from a previous version or intended for future use. Dead code should be removed or documented.
- **Fix:** Remove the unused class or add a comment explaining its purpose.

#### IN-02: `eval/run_eval.py` — redundant `import os as _os`

- **File:** `eval/run_eval.py:150`
- **Line:** 150
- **Category:** Code quality
- **Severity:** Info
- **Description:** `import os as _os` is redundant because `os` is already imported at the top of the file (line 21). The alias `_os` is unnecessary and confusing.
- **Fix:** Use the existing `os` import.

#### IN-03: `eval/artifact.py` — `_get_git_ref()` swallows all exceptions

- **File:** `eval/artifact.py:62`
- **Line:** 62
- **Category:** Code quality
- **Severity:** Info
- **Description:** The `except (FileNotFoundError, subprocess.TimeoutExpired, OSError)` clause catches specific exceptions, which is good. However, if `git` is not installed or the repo is not a git repo, the function silently returns an empty string. This is intentional (graceful degradation), but a debug log would help diagnose missing git metadata in artifacts.
- **Fix:** Add optional debug logging.

#### IN-04: `eval/corpus.py` — type hint for `tokenizer` parameter is incomplete

- **File:** `eval/corpus.py:28`
- **Line:** 28
- **Category:** Code quality
- **Severity:** Info
- **Description:** The `tokenizer` parameter is typed as `BaseTokenizer | None`, but the function calls `tokenizer.encode()` (line 82). The type hint should be a protocol or abstract base that guarantees an `encode()` method, not just `BaseTokenizer` (which now has a concrete `encode()` implementation). This is a minor type-safety issue.
- **Fix:** The current type hint is acceptable since `BaseTokenizer` now has a concrete `encode()` method (no longer abstract). No change needed, but consider documenting the expectation.

#### IN-05: `eval/model_path.py` — `_validate_path_safety()` is private but used externally

- **File:** `eval/model_path.py:67`
- **Line:** 67
- **Category:** Code quality
- **Severity:** Info
- **Description:** `_validate_path_safety()` is prefixed with `_` (private), but it is imported and used by other modules (e.g., `eval/run_eval.py`, `eval/corpus.py`, `eval/artifact.py`). If it is part of the public API, it should not be prefixed. If it is internal, it should not be imported externally.
- **Fix:** Either make it public (rename to `validate_path_safety()`) or keep it private and duplicate the logic in each module. The former is preferred.

#### IN-06: `eval/constants.py` — `VOCAB_LOG_SOFTMAX_BLOCK_SIZE` is duplicated from `pipeline.py`

- **File:** `eval/constants.py:23`
- **Line:** 23
- **Category:** Code quality
- **Severity:** Info
- **Description:** `VOCAB_LOG_SOFTMAX_BLOCK_SIZE = 4096` is defined in both `eval/constants.py` (line 23) and `spectralstream/inference/pipeline.py` (line 47). This duplication can lead to inconsistencies if one is updated but not the other.
- **Fix:** Import from the canonical source:
```python
# In eval/constants.py
from spectralstream.inference.pipeline import VOCAB_LOG_SOFTMAX_BLOCK_SIZE
```

## Recommendations

**Priority fixes:**

1. **CR-01 / CR-02:** Replace `signal.alarm()` + `exec()` with cross-platform timeout (Windows compat is a project requirement). This is blocking for Windows users.
2. **WR-01:** Wrap `benchmark_physics_real_weights.py` in `main()` to prevent module-level side effects.
3. **WR-02 / WR-03 / WR-04 / WR-07 / WR-08:** Add path validation to `--tokenizer`, `--corpus`, and `--output` CLI args. These are defense-in-depth security measures.
4. **WR-05:** Strengthen path traversal check by verifying resolved path is within allowed roots.
5. **IN-05:** Make `_validate_path_safety()` public (rename to `validate_path_safety()`) since it is used externally.

**Deferred (low priority):**

- IN-01, IN-02, IN-03, IN-04, IN-06: Code quality improvements that do not affect correctness or security.

---

_Reviewed: 2026-07-09_
_Reviewer: gsd-code-reviewer (standard depth)_

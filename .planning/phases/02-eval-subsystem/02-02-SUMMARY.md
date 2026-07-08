---
phase: 02-eval-subsystem
plan: 02
subsystem: tokenizer
tags: [base64, utf-8, byte-fallback, tokenizer, eval, EVAL-02]
requires: []
provides:
  - BaseTokenizer byte-identity fallback (never-raise encode/decode)
  - Default tokenizer (build_default_tokenizer) with vocab_size == 256 and round-trip guarantee
affects: [03-eval-execution, eval-subsystem verification]
tech-stack:
  added: []
  patterns:
    - "BaseTokenizer encodes via utf-8 bytes, decodes via bytes().decode(errors='replace')"
key-files:
  created:
    - tests/test_tokenizer_fallback.py
  modified:
    - spectralstream/utils/tokenizer_engine.py
key-decisions:
  - "BaseTokenizer.encode maps each UTF-8 byte to its integer value (never raises)"
  - "BaseTokenizer.decode reconstructs bytes from token ids and decodes as UTF-8 with errors='replace'"
  - "Subclasses (BPETokenizer, SentencePieceTokenizer, TiktokenTokenizer) override both methods — unchanged"
patterns-established:
  - "byte-identity fallback pattern: encode = list(text.encode('utf-8')), decode = bytes(t & 0xFF for t in token_ids).decode('utf-8', errors='replace')"
requirements-completed:
  - EVAL-02
coverage:
  - id: D1
    description: "BaseTokenizer.encode('hello') returns [104, 101, 108, 108, 111] and never raises NotImplementedError"
    requirement: EVAL-02
    verification:
      - kind: unit
        ref: "tests/test_tokenizer_fallback.py#test_base_tokenizer_encode_returns_byte_ids"
        status: pass
      - kind: unit
        ref: "tests/test_tokenizer_fallback.py#test_base_tokenizer_does_not_raise"
        status: pass
    human_judgment: false
  - id: D2
    description: "BaseTokenizer.decode round-trips ASCII, UTF-8 multibyte, and emoji text"
    requirement: EVAL-02
    verification:
      - kind: unit
        ref: "tests/test_tokenizer_fallback.py#test_base_tokenizer_roundtrip_ascii"
        status: pass
      - kind: unit
        ref: "tests/test_tokenizer_fallback.py#test_base_tokenizer_roundtrip_multibyte"
        status: pass
      - kind: unit
        ref: "tests/test_tokenizer_fallback.py#test_base_tokenizer_roundtrip_emoji"
        status: pass
    human_judgment: false
  - id: D3
    description: "build_default_tokenizer() has vocab_size == 256"
    requirement: EVAL-02
    verification:
      - kind: unit
        ref: "tests/test_tokenizer_fallback.py#test_default_tokenizer_vocab_size"
        status: pass
    human_judgment: false
  - id: D4
    description: "build_default_tokenizer() round-trips ASCII, UTF-8 multibyte, and emoji text"
    requirement: EVAL-02
    verification:
      - kind: unit
        ref: "tests/test_tokenizer_fallback.py#test_default_tokenizer_roundtrip_ascii"
        status: pass
      - kind: unit
        ref: "tests/test_tokenizer_fallback.py#test_default_tokenizer_roundtrip_multibyte"
        status: pass
      - kind: unit
        ref: "tests/test_tokenizer_fallback.py#test_default_tokenizer_roundtrip_emoji"
        status: pass
    human_judgment: false
duration: 18min
completed: 2026-07-09
status: complete
---

# Phase 02 Plan 02: BaseTokenizer fallback and default tokenizer round-trip (EVAL-02)

**Byte-identity fallback for BaseTokenizer.encode/decode replacing NotImplementedError, with 10-passing test suite proving round-trip correctness for ASCII, multibyte UTF-8, and emoji**

## Performance

- **Duration:** 18 min
- **Started:** 2026-07-09T00:41:00Z
- **Completed:** 2026-07-09T00:59:00Z
- **Tasks:** 3 (TDD: RED, GREEN, REFACTOR)
- **Files modified:** 2

## Accomplishments
- Replaced `BaseTokenizer.encode` / `decode` `raise NotImplementedError` with byte-identity fallback implementation: `encode` returns UTF-8 byte integer values via `list(text.encode("utf-8"))`; `decode` reconstructs text via `bytes(int(t) & 0xFF for t in token_ids).decode("utf-8", errors="replace")`
- Created `tests/test_tokenizer_fallback.py` with 10 passing tests covering BaseTokenizer encode/decode round-trips (ASCII, multibyte, emoji), default tokenizer vocab_size, and default tokenizer round-trips
- All 10 tests pass via `python -m pytest tests/test_tokenizer_fallback.py -x -q`
- Subclasses (`BPETokenizer`, `SentencePieceTokenizer`, `TiktokenTokenizer`, `SpectralTokenizer`) override both methods — their behavior is unchanged

## Task Commits

Each task was committed atomically following the TDD cycle:

1. **Task 1 (RED): Create failing tests** — `a5be606` (test)
2. **Task 2 (GREEN): Implement byte-identity fallback** — `c713ef2` (feat)
3. **Task 3 (REFACTOR): Add inline doc comments** — `c1524be` (refactor)

## Files Created/Modified
- `spectralstream/utils/tokenizer_engine.py` — Modified `BaseTokenizer.encode` and `decode` with byte-identity fallback; added inline documentation comments
- `tests/test_tokenizer_fallback.py` — Created with 10 tests (TestBaseTokenizerFallback: 6 tests, TestBuildDefaultTokenizer: 4 tests)

## Decisions Made
- Fallback implementation uses `list(text.encode("utf-8"))` for encode and `bytes(int(t) & 0xFF for t in token_ids).decode("utf-8", errors="replace")` for decode — matches the plan specification exactly
- `errors="replace"` in decode handles invalid byte sequences safely (per threat model T-02-02-01, accepted disposition)
- Subclass overrides remain entirely unchanged — BPETokenizer, SentencePieceTokenizer, TiktokenTokenizer, SpectralTokenizer all override both methods
- `build_default_tokenizer()` already caps at 256 byte-level tokens; no change needed

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness
- BaseTokenizer is now usable out of the box without raising `NotImplementedError` — eval subsystem (02-03) can call `encode`/`decode` on the base class directly
- Planning file `02-03-PLAN.md` already exists for the next plan (eval execution layer)

## Self-Check: PASSED

- [x] tests/test_tokenizer_fallback.py created
- [x] .planning/phases/02-eval-subsystem/02-02-SUMMARY.md created
- [x] a5be606 (RED commit) exists
- [x] c713ef2 (GREEN commit) exists
- [x] c1524be (REFACTOR commit) exists
- [x] All 10 tests passing
- [x] No NotImplementedError in BaseTokenizer.encode or decode

---
*Phase: 02-eval-subsystem*
*Plan: 02*
*Completed: 2026-07-09*

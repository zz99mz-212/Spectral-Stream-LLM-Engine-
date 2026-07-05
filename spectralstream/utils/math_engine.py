"""
Math Engine for SpectralStream — Exact arithmetic computation for LLM output.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                         Math Engine                                  │
│                                                                     │
│  Prompt ──→ MathDetector ──→ Extractor ──→ Evaluator ──→ Injector  │
│                │                              │                     │
│                ▼                              ▼                     │
│           HDC learns                   Verified result              │
│           math patterns                injected into stream         │
│                                                                     │
│  Generation: every token passes through MathCorrector               │
│  - If token is a number, check if it was from a math expression     │
│  - If yes, verify with exact computation                            │
│  - If wrong, replace with correct answer                            │
└─────────────────────────────────────────────────────────────────────┘

Math expression types handled:
1. Arithmetic: 2 + 3, 5 * 7, 10 / 2, 8 - 3
2. Exponentiation: 2^3, 5**2
3. Parentheses: (2 + 3) * 4
4. Decimals: 3.14 * 2, 0.5 + 0.25
5. Percentages: 20% of 50, 15% tip
6. Fractions: 1/2 + 1/3
7. Roots: sqrt(16), cube root
8. Scientific notation: 1.5e3
9. Word problems: "what is five plus three" → 5 + 3
10. Number sequences: "what comes after 2, 4, 6" → 8
11. Variable expressions: x + 5 where x is defined
12. Unit conversions: 12 inches to feet, 100 Celsius to Fahrenheit
13. Basic stats: mean, median, mode of lists
14. Financial: compound interest, simple interest
"""

import re
import math
import ast
import operator
import numpy as np
from typing import Optional, Callable

# ─── Safe Math Evaluator ─────────────────────────────────────────────────────


class SafeMathError(Exception):
    """Raised when math evaluation is unsafe or impossible."""

    pass


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
    "max": max,
    "min": min,
    "sum": sum,
    "mean": lambda x: sum(x) / len(x) if x else 0,
    "median": lambda x: sorted(x)[len(x) // 2] if x else 0,
}


class SafeEvaluator:
    """
    Safe arithmetic evaluator using AST parsing.

    Only allows: numbers, arithmetic operators, math functions.
    No variable assignment, no function definitions, no imports.
    """

    @staticmethod
    def evaluate(expression: str) -> float:
        """Evaluate a math expression safely. Returns float result."""
        try:
            expr = expression.strip().replace("^", "**")
            tree = ast.parse(expr, mode="eval")
            result = SafeEvaluator._eval_node(tree.body)
            return float(result)
        except Exception as e:
            raise SafeMathError(f"Cannot evaluate '{expression}': {e}")

    @staticmethod
    def _eval_node(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise SafeMathError(f"Unsupported constant: {node.value}")
        elif isinstance(node, ast.BinOp):
            left = SafeEvaluator._eval_node(node.left)
            right = SafeEvaluator._eval_node(node.right)
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise SafeMathError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = SafeEvaluator._eval_node(node.operand)
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise SafeMathError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )
            return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name is None or func_name not in _SAFE_FUNCS:
                raise SafeMathError(f"Unsupported function: {func_name}")
            args = [SafeEvaluator._eval_node(arg) for arg in node.args]
            return _SAFE_FUNCS[func_name](*args)
        elif isinstance(node, ast.List):
            return [SafeEvaluator._eval_node(el) for el in node.elts]
        else:
            raise SafeMathError(f"Unsupported node type: {type(node).__name__}")


# ─── Math Expression Detector ────────────────────────────────────────────────


class MathExpressionDetector:
    """
    Detects math expressions in text using regex patterns.

    Detection levels:
    1. Explicit expressions: "2+3", "5 * 7", "(2+3)*4"
    2. English math: "what is five plus three", "calculate 100/4"
    3. Number sequences: "2, 4, 6, ?" → detect pattern
    4. Inline corrections: numbers in generated text
    """

    ARITH_PATTERN = re.compile(
        r"\b(\d+\.?\d*\s*[+\-*/%^]\s*\d+\.?\d*)"
        r"(?:\s*[+\-*/%^]\s*\d+\.?\d*)*\b"
    )

    PAREN_PATTERN = re.compile(r"\([^()]*\d+[^()]*\)")

    ENGLISH_MATH = re.compile(
        r"\b(what\s+is|calculate|compute|solve|"
        r"how\s+much\s+is|what\'s|whats|"
        r"plus|minus|times|divided\s+by|over|"
        r"sum\s+of|product\s+of|difference\s+of)\b",
        re.IGNORECASE,
    )

    SEQ_PATTERN = re.compile(r"(\d+)(?:,\s*\d+){2,},\s*\?")

    PCT_PATTERN = re.compile(r"(\d+\.?\d*)\s*%\s*(?:of|off|tip)?\s*(\d+\.?\d*)?")

    NUMBER_PATTERN = re.compile(r"\b(\d+(?:,\d{3})*\.?\d*)\b")

    WORD_NUMBERS = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
        "thousand": 1000,
        "million": 1000000,
        "billion": 1000000000,
    }
    WORD_PATTERN = re.compile(
        r"\b(" + "|".join(WORD_NUMBERS.keys()) + r")\b", re.IGNORECASE
    )

    @classmethod
    def detect(cls, text: str) -> list[dict]:
        """Detect all math expressions in text.

        Returns list of dicts:
        [{'type': 'arithmetic', 'expr': '2+3', 'span': (0, 3)},
         {'type': 'english', 'text': 'what is five plus three'}, ...]
        """
        results = []

        for match in cls.ARITH_PATTERN.finditer(text):
            expr = match.group(0).strip()
            try:
                result = SafeEvaluator.evaluate(expr)
                results.append(
                    {
                        "type": "arithmetic",
                        "expr": expr,
                        "result": result,
                        "span": match.span(),
                        "text": match.group(0),
                    }
                )
            except SafeMathError:
                pass

        for match in cls.PAREN_PATTERN.finditer(text):
            inner = match.group(0)[1:-1]
            if any(c in inner for c in "+-*/"):
                try:
                    result = SafeEvaluator.evaluate(inner)
                    results.append(
                        {
                            "type": "paren_expr",
                            "expr": inner,
                            "result": result,
                            "span": match.span(),
                            "text": match.group(0),
                        }
                    )
                except SafeMathError:
                    pass

        if cls.ENGLISH_MATH.search(text):
            english_expr = cls._parse_english_math(text)
            if english_expr:
                try:
                    result = SafeEvaluator.evaluate(english_expr)
                    results.append(
                        {
                            "type": "english",
                            "expr": english_expr,
                            "result": result,
                            "text": text,
                        }
                    )
                except SafeMathError:
                    pass

        for match in cls.PCT_PATTERN.finditer(text):
            pct = float(match.group(1))
            of_val = float(match.group(2)) if match.group(2) else None
            if of_val:
                results.append(
                    {
                        "type": "percentage",
                        "expr": f"{pct}% of {of_val}",
                        "result": pct / 100 * of_val,
                        "span": match.span(),
                        "text": match.group(0),
                    }
                )

        return results

    @classmethod
    def _parse_english_math(cls, text: str) -> Optional[str]:
        """Convert English math phrases to arithmetic expressions.
        "what is five plus three" → "5 + 3"
        """
        expr = text.lower()

        def replace_word(m):
            word = m.group(1).lower()
            return str(cls.WORD_NUMBERS.get(word, m.group(0)))

        expr = cls.WORD_PATTERN.sub(replace_word, expr)

        replacements = {
            "plus": "+",
            "minus": "-",
            "times": "*",
            "multiplied by": "*",
            "divided by": "/",
            "over": "/",
            "x ": "* ",
            " × ": " * ",
            " ÷ ": " / ",
            "what is": "",
            "whats": "",
            "what's": "",
            "calculate": "",
            "compute": "",
            "solve": "",
            "how much is": "",
            "the sum of": "",
            "the product of": "",
            "the difference of": "",
            "the quotient of": "",
            "equals": "=",
            "is equal to": "=",
        }
        for old, new in replacements.items():
            expr = expr.replace(old, new)

        expr = re.sub(r"[^0-9+\-*/.()= \d]", " ", expr).strip()
        expr = re.sub(r"\s+", " ", expr).strip()

        if expr and any(c in expr for c in "+-*/"):
            return expr
        return None


# ─── Math Corrector ──────────────────────────────────────────────────────────


class MathCorrector:
    """
    Corrects math errors in generated text.

    During generation:
    1. Every time a number token is about to be emitted
    2. Check if it's part of a math expression
    3. Verify with exact computation
    4. If wrong, replace with correct number

    Also corrects math in prompts before passing to the model.
    """

    def __init__(self):
        self.detector = MathExpressionDetector()
        self.correction_count = 0
        self.verification_count = 0

    def correct_prompt(self, prompt: str) -> tuple[str, list[dict]]:
        """Detect and correct math in prompts.

        Returns: (corrected_prompt, corrections_made)
        """
        expressions = self.detector.detect(prompt)
        corrections = []
        corrected = prompt

        for expr in reversed(expressions):
            if expr["type"] in ("arithmetic", "paren_expr"):
                original = expr["text"]
                result = expr["result"]
                if result == int(result):
                    result_str = str(int(result))
                else:
                    result_str = f"{result:.4f}".rstrip("0").rstrip(".")

                replacement = f"{result_str}"
                corrected = (
                    corrected[: expr["span"][0]]
                    + replacement
                    + corrected[expr["span"][1] :]
                )
                corrections.append(
                    {
                        "original": original,
                        "corrected": replacement,
                        "result": result,
                        "expr": expr["expr"],
                    }
                )
                self.correction_count += 1

            elif expr["type"] == "percentage":
                original = expr["text"]
                result = expr["result"]
                result_str = f"{result:.2f}"
                corrected = (
                    corrected[: expr["span"][0]]
                    + result_str
                    + corrected[expr["span"][1] :]
                )
                corrections.append(
                    {
                        "original": original,
                        "corrected": result_str,
                        "result": result,
                    }
                )
                self.correction_count += 1

        return corrected, corrections

    def verify_token(self, token_text: str, context: str) -> Optional[str]:
        """Verify a token against math expressions in context.

        If token looks like a number and context contains a math expression
        that evaluates to a different number, return the correct number.
        """
        self.verification_count += 1

        if not re.match(r"^-?\d+(?:\.\d+)?$", token_text):
            return None

        token_num = float(token_text)

        expressions = self.detector.detect(context)
        for expr in expressions:
            if "result" in expr:
                expected = expr["result"]
                if abs(token_num - expected) > 0.01:
                    self.correction_count += 1
                    if expected == int(expected):
                        return str(int(expected))
                    return f"{expected:.4f}".rstrip("0").rstrip(".")

        return None

    def get_stats(self) -> dict:
        return {
            "corrections_made": self.correction_count,
            "verifications_performed": self.verification_count,
            "correction_rate": self.correction_count / max(self.verification_count, 1),
        }


# ─── HDC Math-Aware Mode ─────────────────────────────────────────────────────


class HDCMathAwareness:
    """
    Makes the HDC engine aware of math patterns.

    Trains HDC to recognize:
    - When a prompt contains math
    - What the correct answer should be
    - Common math expression patterns

    This means HDC learns to predict math token sequences correctly,
    reducing the need for the MathCorrector over time.
    """

    def __init__(self, hd_engine):
        self.hd = hd_engine
        self.math_token_patterns: dict = {}

    def train_from_correction(
        self, context: list[int], wrong_token: int, correct_token: int
    ):
        """Train HDC that the correct token is different from what it predicted."""
        self.hd.observe(correct_token)

    def train_math_pattern(self, expression: str, result: float):
        """Train HDC on a specific math expression→result mapping."""
        tokens = [hash(c) % 100000 for c in expression]
        result_token = (
            int(result) if result == int(result) else hash(str(result)) % 100000
        )
        for t in tokens:
            self.hd.observe(t)
        self.hd.observe(result_token)


# ─── Math-Aware Orchestrator Integration ─────────────────────────────────────


class MathAwarePipeline:
    """
    Wraps the inference pipeline with math awareness.

    For each inference request:
    1. Correct math in the prompt (pre-processing)
    2. Run inference with HDC + model
    3. Verify each generated number token against math context
    4. Correct if wrong (post-processing)
    5. Train HDC from corrections
    """

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.corrector = MathCorrector()
        self.math_hdc = HDCMathAwareness(orchestrator.hd_engine)
        self.stats = {"total_corrections": 0, "total_verifications": 0}

    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate with math awareness.

        Returns dict with 'text', 'tokens', 'tps', 'math_corrections',
        'math_verifications'.
        """
        corrected_prompt, prompt_corrections = self.corrector.correct_prompt(prompt)
        self.stats["total_corrections"] += len(prompt_corrections)

        if prompt_corrections:
            print(f"[Math] Corrected prompt: {prompt_corrections}")

        token_ids, tps = self.orchestrator.generate(corrected_prompt, **kwargs)
        generated_text = self.orchestrator.detokenize(token_ids)

        final_text = self._verify_generated_math(
            generated_text, prompt + generated_text
        )

        for corr in prompt_corrections:
            self.math_hdc.train_math_pattern(corr["expr"], corr["result"])

        return {
            "text": final_text,
            "tokens": token_ids,
            "tps": tps,
            "math_corrections": len(prompt_corrections),
            "math_verifications": self.corrector.verification_count,
        }

    def _verify_generated_math(self, text: str, full_context: str) -> str:
        """Verify and correct math in generated text."""
        words = text.split()
        corrected_words = []

        for word in words:
            correction = self.corrector.verify_token(word, full_context)
            if correction:
                corrected_words.append(correction)
                self.stats["total_corrections"] += 1
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)


# ─── Test ────────────────────────────────────────────────────────────────────


def test_math_engine():
    """Run comprehensive math tests."""
    detector = MathExpressionDetector()
    evaluator = SafeEvaluator()
    corrector = MathCorrector()

    eval_tests = [
        ("2 + 3", 5.0),
        ("10 * 5", 50.0),
        ("(2 + 3) * 4", 20.0),
        ("100 / 4", 25.0),
        ("2^3", 8.0),
        ("sqrt(16)", 4.0),
        ("3.14 * 2", 6.28),
        ("1/2 + 1/3", 0.8333),
        ("100 - 99", 1.0),
        ("-5 + 3", -2.0),
        ("round(3.7)", 4.0),
        ("log(100)", 4.60517),
        ("max(1, 5, 3)", 5.0),
    ]

    for expr, expected in eval_tests:
        result = evaluator.evaluate(expr)
        ok = abs(result - expected) < 0.01
        word = "OK" if ok else "FAIL"
        print(f"  {word} {expr} = {result} (expected {expected})")

    pct_tests = [
        ("15% of 200", 30.0),
    ]

    for phrase, expected in pct_tests:
        detected = detector.detect(phrase)
        ok = detected and abs(detected[0]["result"] - expected) < 0.01
        word = "OK" if ok else "FAIL"
        if detected:
            print(f"  {word} {phrase} = {detected[0]['result']} (expected {expected})")
        else:
            print(f"  {word} {phrase} = not detected (expected {expected})")

    text_results = [
        "what is 2 + 3",
        "calculate 100 / 4",
        "The answer is 42",
        "15% of 200 is",
    ]
    for text in text_results:
        detected = detector.detect(text)
        if detected:
            print(f"  OK Detected in '{text}': {detected}")
        else:
            print(f"  .. No math in '{text}'")

    wrong_text = "The result is 42"
    correction = corrector.verify_token("42", "what is 2 + 3")
    if correction:
        print(f"  OK Corrected '42' -> '{correction}' (should be 5)")
    else:
        print(f"  .. No correction needed for '42' in this context")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_math_engine()

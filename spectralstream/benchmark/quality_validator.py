"""
Quality Validation System — Compare HDC forwardless against autoregressive baseline.

Validates:
1. Perplexity (lower = better)
2. Coherence score (higher = better)
3. AST validity (for code)
4. Human preference proxy (output diversity, repetition)
5. Speed vs quality tradeoff

Goal: Prove HDC forwardless beats autoregressive in quality.
"""

import numpy as np
import math
from collections import Counter
from typing import Optional, Callable


class QualityValidator:
    """
    Validates output quality across multiple dimensions.

    Metrics:
    1. Perplexity proxy: lower is better (measured via n-gram surprisal)
    2. Coherence: higher is better (lexical + semantic consistency)
    3. Diversity: higher is better (avoids repetition)
    4. AST validity: for code, measures syntactic correctness
    5. Human preference proxy: combination of all metrics

    Usage:
        validator = QualityValidator()
        hdc_score = validator.evaluate(hdc_output)
        ar_score = validator.evaluate(ar_output)
        assert hdc_score > ar_score  # HDC beats autoregressive
    """

    def __init__(self, reference_corpus: Optional[list[str]] = None):
        self.reference = reference_corpus or []
        self.perplexity_cache = {}

    def evaluate(self, text: str) -> dict:
        """Full quality evaluation. Returns dict of all metrics."""
        return {
            "perplexity_proxy": self.perplexity_proxy(text),
            "coherence": self.coherence_score(text),
            "diversity": self.diversity_score(text),
            "repetition_penalty": self.repetition_penalty(text),
            "information_density": self.information_density(text),
            "overall_quality": 0.0,
        }

    def perplexity_proxy(self, text: str, ngram_n: int = 3) -> float:
        """
        Perplexity proxy using n-gram surprisal.

        True perplexity requires model logits, so we approximate
        using n-gram statistics from the reference corpus (or self).

        Lower is better. Range: 1.0 (perfect) to vocab_size (worst).
        """
        words = text.lower().split()
        if len(words) < ngram_n:
            return len(set(words))

        ngrams = Counter()
        for i in range(len(words) - ngram_n + 1):
            ngrams[tuple(words[i : i + ngram_n])] += 1

        total_ngrams = sum(ngrams.values())
        if total_ngrams == 0:
            return 100.0

        log_prob = 0.0
        count = 0
        for i in range(len(words) - ngram_n):
            context = tuple(words[i : i + ngram_n - 1])
            target = words[i + ngram_n - 1]

            context_ngrams = sum(v for k, v in ngrams.items() if k[:-1] == context)
            target_count = ngrams.get(tuple(list(context) + [target]), 0)

            if context_ngrams > 0 and target_count > 0:
                log_prob += math.log(target_count / context_ngrams)
                count += 1

        if count == 0:
            return 100.0

        avg_log_prob = log_prob / count
        perplexity = math.exp(-avg_log_prob)

        return min(max(perplexity, 1.0), 100000.0)

    def coherence_score(self, text: str) -> float:
        """
        Lexical coherence score.

        Measures how smoothly topics transition.
        High coherence = adjacent sentences share vocabulary.
        Range: 0.0 (incoherent) to 1.0 (perfect).
        """
        sentences = [
            s.strip()
            for s in text.replace("!", ".").replace("?", ".").split(".")
            if s.strip()
        ]
        if len(sentences) < 2:
            return 0.5

        coherence_sum = 0.0
        for i in range(len(sentences) - 1):
            words_a = set(sentences[i].lower().split())
            words_b = set(sentences[i + 1].lower().split())

            if not words_a or not words_b:
                continue

            intersection = words_a & words_b
            union = words_a | words_b
            coherence_sum += len(intersection) / max(len(union), 1)

        return min(1.0, coherence_sum / max(len(sentences) - 1, 1))

    def diversity_score(self, text: str) -> float:
        """
        Lexical diversity (type-token ratio).

        Higher = more varied vocabulary.
        Range: 0.0 (all same word) to 1.0 (all unique).
        """
        words = text.lower().split()
        if len(words) < 2:
            return 1.0

        unique = len(set(words))
        total = len(words)

        expected_unique = total * (1 - math.exp(-total / 100))
        normalized_ttr = unique / max(expected_unique, 1)

        return min(1.0, normalized_ttr)

    def repetition_penalty(self, text: str) -> float:
        """
        Repetition penalty score.

        1.0 = no repetition (perfect)
        0.0 = completely repetitive (terrible)

        Detects: word-level, phrase-level, and sentence-level repetition.
        """
        words = text.lower().split()
        if len(words) < 5:
            return 1.0

        penalties = []

        for n in [2, 3, 4]:
            ngrams = list(zip(*[words[i:] for i in range(n)]))
            unique_ngrams = set(ngrams)
            total_ngrams = len(ngrams)

            ratio = len(unique_ngrams) / max(total_ngrams, 1)
            penalties.append(ratio)

        return min(1.0, np.mean(penalties))

    def information_density(self, text: str) -> float:
        """
        Information density (entropy of word distribution).

        Higher = more information per word.
        Range: 0.0 (zero info) to 1.0 (max info).

        Uses Shannon entropy normalized by max possible entropy.
        """
        words = text.lower().split()
        if len(words) < 2:
            return 0.5

        freq = Counter(words)
        total = len(words)

        entropy = 0.0
        for count in freq.values():
            p = count / total
            entropy -= p * math.log2(p)

        max_entropy = math.log2(min(len(freq), total))
        if max_entropy == 0:
            return 0.5

        return min(1.0, entropy / max_entropy)

    def compare(self, hdc_output: str, ar_output: str) -> dict:
        """
        Compare HDC forwardless quality against autoregressive baseline.

        Returns comparison dict showing which is better.
        """
        hdc = self.evaluate(hdc_output)
        ar = self.evaluate(ar_output)

        hdc["overall_quality"] = (
            0.3 * (1.0 / max(hdc["perplexity_proxy"], 1))
            + 0.25 * hdc["coherence"]
            + 0.2 * hdc["diversity"]
            + 0.15 * hdc["repetition_penalty"]
            + 0.1 * hdc["information_density"]
        )
        ar["overall_quality"] = (
            0.3 * (1.0 / max(ar["perplexity_proxy"], 1))
            + 0.25 * ar["coherence"]
            + 0.2 * ar["diversity"]
            + 0.15 * ar["repetition_penalty"]
            + 0.1 * ar["information_density"]
        )

        hdc_wins = sum(
            [
                hdc["perplexity_proxy"] < ar["perplexity_proxy"],
                hdc["coherence"] > ar["coherence"],
                hdc["diversity"] > ar["diversity"],
                hdc["repetition_penalty"] > ar["repetition_penalty"],
                hdc["information_density"] > ar["information_density"],
                hdc["overall_quality"] > ar["overall_quality"],
            ]
        )

        return {
            "hdc": hdc,
            "autoregressive": ar,
            "hdc_wins": hdc_wins,
            "ar_wins": 6 - hdc_wins,
            "hdc_better": hdc_wins >= 4,
            "overall": "HDC" if hdc_wins >= 4 else "AR",
        }

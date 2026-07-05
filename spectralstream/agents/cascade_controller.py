import numpy as np
from collections import deque, defaultdict
from typing import Optional


class CascadeStrategySelector:
    STRATEGIES = [
        "forwardless",
        "block_emission",
        "speculative",
        "single_step",
        "rng_fallback",
    ]
    STRATEGY_COST = [0.0, 0.2, 0.5, 1.0, 0.0]

    def __init__(self, default_strategy: str = "block_emission"):
        self.current_level = self.STRATEGIES.index(default_strategy)
        self.strategy_history: deque = deque(maxlen=500)
        self.transition_log: list[tuple[int, str, str]] = []

    def select_strategy(
        self,
        confidence: float,
        resonance_score: float,
        spectral_entropy: float,
        recent_accuracy: float,
        hd_acceptance_rate: float,
    ) -> tuple[str, int]:
        composite = (
            0.30 * confidence
            + 0.25 * resonance_score
            + 0.15 * (1.0 - spectral_entropy)
            + 0.15 * recent_accuracy
            + 0.15 * hd_acceptance_rate
        )

        thresholds = [0.85, 0.65, 0.45, 0.25]
        target_level = 0
        for i, thr in enumerate(thresholds):
            if composite >= thr:
                target_level = i
                break
        else:
            target_level = len(self.STRATEGIES) - 1

        old_level = self.current_level
        if target_level < old_level:
            self.current_level = target_level
        elif target_level > old_level:
            self.current_level = min(target_level, old_level + 1)

        strategy = self.STRATEGIES[self.current_level]
        self.strategy_history.append(self.current_level)

        if old_level != self.current_level:
            self.transition_log.append(
                (len(self.transition_log), self.STRATEGIES[old_level], strategy)
            )

        return strategy, self.current_level

    def can_skip_verification(self, level: int) -> bool:
        return level <= 0

    def compute_expected_cost(self, strategy_level: int, n_tokens: int) -> float:
        return self.STRATEGY_COST[strategy_level] * n_tokens

    def stats(self) -> dict:
        strategy_distribution = {}
        for i, name in enumerate(self.STRATEGIES):
            count = self.strategy_history.count(i)
            if count > 0:
                strategy_distribution[name] = count / max(len(self.strategy_history), 1)
        return {
            "current_strategy": self.STRATEGIES[self.current_level],
            "current_level": self.current_level,
            "strategy_distribution": strategy_distribution,
            "total_transitions": len(self.transition_log),
            "expected_cost_ratio": self.STRATEGY_COST[self.current_level],
        }

    def reset(self):
        self.current_level = self.STRATEGIES.index("block_emission")
        self.strategy_history.clear()


class SelfHealingHDC:
    def __init__(
        self,
        entropy_threshold: float = 0.15,
        consecutive_threshold: int = 8,
        min_diversity: float = 0.1,
    ):
        self.entropy_threshold = entropy_threshold
        self.consecutive_threshold = consecutive_threshold
        self.min_diversity = min_diversity
        self.consecutive_low_entropy = 0
        self.healing_events = 0
        self.last_heal_type: Optional[str] = None
        self.output_history: deque = deque(maxlen=64)
        self.entropy_history: deque = deque(maxlen=128)
        self.current_seed_offset = 0

    def observe_output(self, token_id: int, entropy: float):
        self.output_history.append(token_id)
        self.entropy_history.append(entropy)

    def check_health(self) -> Optional[str]:
        if len(self.output_history) < 8:
            return None

        recent = list(self.output_history)[-self.consecutive_threshold :]
        recent_set = set(recent)

        if len(recent_set) == 1 and len(recent) >= 4:
            self.consecutive_low_entropy += 2
            return "repetition"

        if len(recent) >= 6:
            if len(recent[-2:]) == 2:
                pair = tuple(recent[-2:])
                if tuple(recent[-4:-2]) == pair and tuple(recent[-6:-4]) == pair:
                    self.consecutive_low_entropy += 2
                    return "cycling"

            if len(recent) >= 9:
                triple = tuple(recent[-3:])
                if tuple(recent[-6:-3]) == triple and tuple(recent[-9:-6]) == triple:
                    self.consecutive_low_entropy += 2
                    return "cycling"

        recent_entropy = list(self.entropy_history)[
            -min(16, len(self.entropy_history)) :
        ]
        if recent_entropy and np.mean(recent_entropy) < self.entropy_threshold:
            self.consecutive_low_entropy += 1
            if self.consecutive_low_entropy >= self.consecutive_threshold:
                return "low_entropy_plateau"
        else:
            self.consecutive_low_entropy = max(0, self.consecutive_low_entropy - 1)

        return None

    def heal(self, issue: str) -> dict:
        self.healing_events += 1
        self.last_heal_type = issue
        self.current_seed_offset += 1

        instructions = {
            "re_vectorize": False,
            "seed_offset": self.current_seed_offset,
            "temperature_boost": 0.0,
            "clear_prototypes": False,
            "noise_injection": 0.0,
        }

        if issue == "repetition":
            instructions["re_vectorize"] = True
            instructions["temperature_boost"] = 0.3
            instructions["clear_prototypes"] = True
            instructions["noise_injection"] = 0.1
        elif issue == "cycling":
            instructions["re_vectorize"] = True
            instructions["temperature_boost"] = 0.2
            instructions["noise_injection"] = 0.05
        elif issue == "low_entropy_plateau":
            instructions["temperature_boost"] = 0.15
            instructions["noise_injection"] = 0.08
            if self.healing_events > 3:
                instructions["clear_prototypes"] = True
        else:
            instructions["re_vectorize"] = True
            instructions["temperature_boost"] = 0.4
            instructions["clear_prototypes"] = True
            instructions["noise_injection"] = 0.15

        self.consecutive_low_entropy = 0
        return instructions

    def needs_healing(self) -> tuple[bool, Optional[str]]:
        issue = self.check_health()
        if issue is not None:
            return True, issue
        return False, None

    def stats(self) -> dict:
        return {
            "total_healing_events": self.healing_events,
            "last_heal_type": self.last_heal_type,
            "consecutive_low_entropy": self.consecutive_low_entropy,
            "current_seed_offset": self.current_seed_offset,
        }

    def reset(self):
        self.consecutive_low_entropy = 0
        self.healing_events = 0
        self.last_heal_type = None
        self.output_history.clear()
        self.entropy_history.clear()
        self.current_seed_offset = 0


class ResonanceAwareSpeculation:
    def __init__(
        self,
        min_spec_depth: int = 2,
        max_spec_depth: int = 16,
        min_candidates: int = 4,
        max_candidates: int = 32,
    ):
        self.min_spec_depth = min_spec_depth
        self.max_spec_depth = max_spec_depth
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.depth_history: deque = deque(maxlen=200)

    def compute_spec_depth(self, resonance_score: float) -> int:
        normalized = float(np.clip(resonance_score, 0.0, 1.0))
        exp_factor = normalized * normalized * (2.0 - normalized)
        depth = int(
            self.min_spec_depth
            + (self.max_spec_depth - self.min_spec_depth) * exp_factor
        )
        depth = int(np.clip(depth, self.min_spec_depth, self.max_spec_depth))
        self.depth_history.append(depth)
        return depth

    def compute_candidate_count(self, resonance_score: float) -> int:
        normalized = float(np.clip(resonance_score, 0.0, 1.0))
        candidate_factor = 1.0 - normalized * 0.7
        n = int(
            self.min_candidates
            + (self.max_candidates - self.min_candidates) * candidate_factor
        )
        return int(np.clip(n, self.min_candidates, self.max_candidates))

    def compute_verification_interval(self, resonance_score: float) -> int:
        normalized = float(np.clip(resonance_score, 0.0, 1.0))
        interval = max(1, int(normalized * 8))
        return interval

    def stats(self) -> dict:
        if not self.depth_history:
            avg_depth = 0.0
        else:
            avg_depth = float(np.mean(list(self.depth_history)))
        return {
            "avg_spec_depth": round(avg_depth, 1),
            "current_min_depth": self.min_spec_depth,
            "current_max_depth": self.max_spec_depth,
        }

    def reset(self):
        self.depth_history.clear()


class CrossContextMemory:
    def __init__(
        self,
        max_entries: int = 10000,
        hamming_threshold: float = 0.15,
        min_confidence: float = 0.4,
    ):
        self.max_entries = max_entries
        self.hamming_threshold = hamming_threshold
        self.min_confidence = min_confidence

        self.memory: dict[int, dict] = {}
        self.context_hashes: deque = deque(maxlen=max_entries)
        self.hits = 0
        self.misses = 0
        self.cache_saves = 0

    def _hash_context(self, context: tuple) -> int:
        return hash(context) & 0xFFFFFFFFFFFFFFFF

    def _hamming_distance_bits(self, a: tuple, b: tuple) -> float:
        if not a or not b:
            return 1.0

        max_len = max(len(a), len(b))
        min_len = min(len(a), len(b))

        if max_len == 0:
            return 0.0

        distance = 0.0
        for i in range(min_len):
            diff = a[i] ^ b[i]
            distance += bin(diff & 0xFFFFFFFF).count("1") / 32.0

        length_penalty = (max_len - min_len) / max_len
        distance += length_penalty * 0.5

        return distance / max_len

    def _find_similar_context(self, context: tuple) -> Optional[tuple[int, float]]:
        if not context:
            return None

        best_hash = None
        best_sim = float("inf")

        search_pool = list(self.context_hashes)
        max_search = min(len(search_pool), 200)
        if max_search < 1:
            return None

        indices = np.linspace(0, len(search_pool) - 1, max_search, dtype=int)
        for idx in indices:
            cached_hash = search_pool[idx]
            entry = self.memory.get(cached_hash)
            if entry is None:
                continue
            cached_context = entry.get("context", ())
            dist = self._hamming_distance_bits(context, cached_context)
            if dist < best_sim:
                best_sim = dist
                best_hash = cached_hash

        if best_hash is not None and best_sim <= self.hamming_threshold:
            return best_hash, best_sim
        return None

    def lookup(self, context: tuple) -> Optional[list[tuple[int, float]]]:
        h = self._hash_context(context)
        if h in self.memory:
            entry = self.memory[h]
            if entry["confidence"] >= self.min_confidence:
                self.hits += 1
                entry["hit_count"] += 1
                return list(entry["predictions"])

        result = self._find_similar_context(context)
        if result is not None:
            cached_hash, similarity = result
            entry = self.memory[cached_hash]
            if entry["confidence"] >= self.min_confidence:
                self.hits += 1
                entry["hit_count"] += 1
                return list(entry["predictions"])

        self.misses += 1
        return None

    def store(
        self,
        context: tuple,
        predictions: list[tuple[int, float]],
        confidence: float,
    ):
        if not predictions:
            return

        h = self._hash_context(context)
        if h not in self.memory and len(self.memory) >= self.max_entries:
            lru_hash = min(self.memory, key=lambda x: self.memory[x]["hit_count"])
            del self.memory[lru_hash]

        self.memory[h] = {
            "context": context,
            "predictions": predictions,
            "confidence": confidence,
            "hit_count": 0,
            "stored_at": len(self.context_hashes),
        }
        self.context_hashes.append(h)
        self.cache_saves += 1

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)

    def memory_usage(self) -> int:
        return len(self.memory)

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate(), 4),
            "cache_saves": self.cache_saves,
            "memory_entries": len(self.memory),
            "max_entries": self.max_entries,
        }

    def reset(self):
        self.hits = 0
        self.misses = 0

    def clear(self):
        self.memory.clear()
        self.context_hashes.clear()
        self.hits = 0
        self.misses = 0
        self.cache_saves = 0


class ProactiveAccuracyManager:
    TOKEN_TYPES = ["common", "rare", "number", "punctuation", "whitespace", "control"]

    def __init__(self, common_vocab_size: int = 1000):
        self.common_vocab_size = common_vocab_size
        self.type_counts: dict[str, int] = {t: 0 for t in self.TOKEN_TYPES}
        self.type_correct: dict[str, int] = {t: 0 for t in self.TOKEN_TYPES}
        self.type_confidences: dict[str, deque] = {
            t: deque(maxlen=500) for t in self.TOKEN_TYPES
        }
        self.token_to_type: dict[int, str] = {}
        self.calibration_biases: dict[str, float] = {t: 0.0 for t in self.TOKEN_TYPES}
        self.token_frequencies: defaultdict[int, int] = defaultdict(int)
        self.total_tokens_seen = 0

    def classify_token(
        self, token_id: int, token_text_hint: Optional[str] = None
    ) -> str:
        if token_id in self.token_to_type:
            return self.token_to_type[token_id]

        if token_text_hint:
            if any(c.isdigit() for c in token_text_hint):
                token_type = "number"
            elif any(c in ".,!?;:-'\"()[]{}<>" for c in token_text_hint):
                token_type = "punctuation"
            elif token_text_hint.strip() == "":
                token_type = "whitespace"
            elif token_id < self.common_vocab_size:
                token_type = "common"
            else:
                token_type = "rare"
        else:
            if token_id < 10:
                token_type = "control"
            elif token_id < 100:
                token_type = "whitespace" if token_id < 50 else "punctuation"
            elif token_id < self.common_vocab_size + 100:
                token_type = "common"
            else:
                token_type = "rare"

        self.token_to_type[token_id] = token_type
        return token_type

    def record_prediction(
        self,
        token_id: int,
        was_correct: bool,
        confidence: float,
        token_text_hint: Optional[str] = None,
    ):
        token_type = self.classify_token(token_id, token_text_hint)
        self.type_counts[token_type] += 1
        self.total_tokens_seen += 1
        self.token_frequencies[token_id] += 1

        if was_correct:
            self.type_correct[token_type] += 1

        self.type_confidences[token_type].append(1.0 if was_correct else 0.0)
        self._update_calibration(token_type)

    def _update_calibration(self, token_type: str):
        confs = list(self.type_confidences[token_type])
        if len(confs) < 20:
            return

        actual_acc = np.mean(confs)
        global_acc = sum(self.type_correct.values()) / max(
            sum(self.type_counts.values()), 1
        )
        self.calibration_biases[token_type] = float(
            np.clip(actual_acc - global_acc, -0.3, 0.3)
        )

    def get_calibration_bias(self, token_id: int) -> float:
        token_type = self.token_to_type.get(token_id, "rare")
        return self.calibration_biases.get(token_type, 0.0)

    def get_type_accuracy(self, token_type: str) -> float:
        if self.type_counts[token_type] == 0:
            return 0.0
        return self.type_correct[token_type] / self.type_counts[token_type]

    def get_best_types(self, top_n: int = 2) -> list[str]:
        accuracies = [(t, self.get_type_accuracy(t)) for t in self.TOKEN_TYPES]
        accuracies.sort(key=lambda x: -x[1])
        return [t for t, _ in accuracies[:top_n]]

    def get_worst_types(self, top_n: int = 2) -> list[str]:
        accuracies = [
            (t, self.get_type_accuracy(t))
            for t in self.TOKEN_TYPES
            if self.type_counts[t] > 10
        ]
        accuracies.sort(key=lambda x: x[1])
        return [t for t, _ in accuracies[:top_n]]

    def stats(self) -> dict:
        type_stats = {}
        for t in self.TOKEN_TYPES:
            if self.type_counts[t] > 0:
                type_stats[t] = {
                    "count": self.type_counts[t],
                    "accuracy": round(self.get_type_accuracy(t), 4),
                    "calibration_bias": round(self.calibration_biases[t], 4),
                }
        return {
            "total_tokens": self.total_tokens_seen,
            "overall_accuracy": round(
                sum(self.type_correct.values())
                / max(sum(self.type_counts.values()), 1),
                4,
            ),
            "per_type": type_stats,
            "best_types": self.get_best_types(),
            "worst_types": self.get_worst_types(),
        }

    def reset(self):
        for t in self.TOKEN_TYPES:
            self.type_counts[t] = 0
            self.type_correct[t] = 0
            self.type_confidences[t].clear()
            self.calibration_biases[t] = 0.0
        self.token_to_type.clear()
        self.token_frequencies.clear()
        self.total_tokens_seen = 0

"""
HDC Draft Engine — Hyperdimensional Computing token prediction.

Sparse hypervectors (5% density), XOR-permutation context encoding,
multi-prototype storage, entanglement-aware LSH, and spectral entropy scoring.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict, deque
from typing import Optional

_BYTE_SET_POSITIONS = tuple(
    tuple(i for i in range(8) if (bv >> i) & 1) for bv in range(256)
)


class HDCBundle:
    """Enhanced HDC bundle with sparse hypervectors, multi-prototype, LSH."""

    def __init__(self, dim: int = 10000, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.token_vectors: dict = {}
        self.prototypes: dict = {}
        self.max_prototypes = 8
        self.ngram_counts: defaultdict = defaultdict(int)
        self.min_order = 2
        self.max_order = 6
        self.stopword_ids: set = set()
        self.content_bias = 1.5
        self.stopword_penalty = 0.3
        self.prototype_match_counts: dict = {}
        self.prototype_counter = 0
        self.max_total_prototypes = 10000
        self.base_temperature = 0.8
        self.entropy_history: deque = deque(maxlen=32)
        self.context_state_cache: dict = {}
        self.cache_max_size = 256
        self.short_order_weight = 1.0
        self.long_order_weight = 0.7
        self.previous_hv = None
        self.hv_change_history: deque = deque(maxlen=64)
        self.anomaly_threshold = 0.3
        self._init_lsh()
        self.cache_hits = 0
        self.cache_misses = 0

    def _make_sparse_hv(self, seed_offset: int = 0) -> np.ndarray:
        rng = np.random.RandomState(self.seed + seed_offset)
        n_ones = max(1, self.dim // 20)
        hv = np.full(self.dim, -1, dtype=np.int8)
        positions = rng.choice(self.dim, size=n_ones, replace=False)
        hv[positions] = 1
        return hv

    def _permute(self, hv: np.ndarray, shift: int) -> np.ndarray:
        shift = shift % self.dim
        if shift == 0:
            return hv.copy()
        return np.concatenate([hv[-shift:], hv[:-shift]])

    def _encode_context(self, tokens: tuple) -> np.ndarray:
        if not tokens:
            return np.ones(self.dim, dtype=np.int8)
        cache_key = tokens
        if cache_key in self.context_state_cache:
            self.cache_hits += 1
            return self.context_state_cache[cache_key].copy()
        result = self._permute(self.ensure_token_vector(tokens[0]), 1)
        for i, tok in enumerate(tokens[1:], start=2):
            permuted = self._permute(self.ensure_token_vector(tok), i)
            result = result * permuted
        if len(self.context_state_cache) < self.cache_max_size:
            self.context_state_cache[cache_key] = result.copy()
        return result

    def ensure_token_vector(self, token_id: int) -> np.ndarray:
        if token_id not in self.token_vectors:
            self.token_vectors[token_id] = self._make_sparse_hv(seed_offset=token_id)
        return self.token_vectors[token_id]

    def _adapt_order(self, context_length: int):
        if context_length <= 2:
            self.min_order, self.max_order = 2, 3
        elif context_length <= 4:
            self.min_order, self.max_order = 2, 4
        elif context_length <= 8:
            self.min_order, self.max_order = 2, 5
        else:
            self.min_order, self.max_order = 2, 6

    def _bundle_majority(self, hv_list: list) -> np.ndarray:
        n = len(hv_list)
        if n == 0:
            return np.ones(self.dim, dtype=np.int8)
        if n <= 2:
            return hv_list[0].copy()
        binary = [((hv + 1) // 2).astype(np.uint8) for hv in hv_list]
        counts = np.zeros(self.dim, dtype=np.int32)
        for bv in binary:
            for byte_idx in range(0, (self.dim + 7) // 8):
                chunk = bv[byte_idx * 8 : (byte_idx + 1) * 8]
                if len(chunk) < 8:
                    chunk = np.pad(chunk, (0, 8 - len(chunk)), "constant")
                byte_val = 0
                for bit_i in range(8):
                    if chunk[bit_i]:
                        byte_val |= 1 << bit_i
                if byte_val:
                    for pos_in_byte in _BYTE_SET_POSITIONS[byte_val]:
                        global_pos = byte_idx * 8 + pos_in_byte
                        if global_pos < self.dim:
                            counts[global_pos] += 1
        half = n // 2
        first = binary[0]
        result_binary = np.zeros(self.dim, dtype=np.int8)
        for i in range(self.dim):
            if counts[i] > half:
                result_binary[i] = 1
            elif counts[i] == half and n % 2 == 0 and first[i]:
                result_binary[i] = 1
        return np.where(result_binary == 1, 1, -1).astype(np.int8)

    def _superposition_bundle(
        self, hv_list: list, phases: Optional[list] = None
    ) -> np.ndarray:
        n = len(hv_list)
        if n == 0:
            return np.ones(self.dim, dtype=np.int8)
        if n == 1:
            return hv_list[0].copy()
        if phases is None:
            phases = [1.0] * n
        weighted_sums = np.zeros(self.dim, dtype=np.float64)
        for idx, hv in enumerate(hv_list):
            binary = (hv + 1) // 2
            phase = phases[idx] if idx < len(phases) else 1.0
            weighted_sums += binary.astype(np.float64) * phase
        half = n / 2.0
        result_binary = np.where(weighted_sums >= half, 1, 0).astype(np.int8)
        return np.where(result_binary == 1, 1, -1).astype(np.int8)

    def _greedy_farthest_first(self, vectors: list, n_prototypes: int) -> list:
        if len(vectors) <= n_prototypes:
            return vectors[:]
        prototypes = [vectors[0]]
        remaining = list(range(1, len(vectors)))
        while len(prototypes) < n_prototypes and remaining:
            max_min_dist = -1.0
            farthest_idx = -1
            for idx in remaining:
                vec = vectors[idx]
                min_dist = min(
                    float(np.count_nonzero(vec != prot)) / self.dim
                    for prot in prototypes
                )
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_idx = idx
            if farthest_idx >= 0:
                prototypes.append(vectors[farthest_idx])
                remaining.remove(farthest_idx)
        return prototypes

    def _store_prototype(self, context: tuple, target_hv: np.ndarray):
        if context not in self.prototypes:
            self.prototypes[context] = []
        proto_list = self.prototypes[context]
        for i, (existing_hv, count, pid) in enumerate(proto_list):
            sim = float(np.count_nonzero(existing_hv == target_hv)) / self.dim
            if sim > 0.85:
                merged = self._superposition_bundle(
                    [existing_hv, target_hv], [float(count), 1.0]
                )
                proto_list[i] = (merged, count + 1, pid)
                return
        new_id = self.prototype_counter
        self.prototype_counter += 1
        proto_list.append((target_hv.copy(), 1, new_id))
        self.prototype_match_counts[new_id] = 0
        if len(proto_list) > self.max_prototypes:
            self._prune_prototypes(context)
        if len(self.prototype_match_counts) > self.max_total_prototypes:
            self._prune_global()

    def _prune_prototypes(self, context: tuple):
        if context not in self.prototypes:
            return
        proto_list = self.prototypes[context]
        if len(proto_list) <= self.max_prototypes:
            return
        proto_list.sort(
            key=lambda x: self.prototype_match_counts.get(x[2], 0), reverse=True
        )
        for _, _, pid in proto_list[self.max_prototypes :]:
            self.prototype_match_counts.pop(pid, None)
        self.prototypes[context] = proto_list[: self.max_prototypes]

    def _prune_global(self):
        if len(self.prototype_match_counts) <= self.max_total_prototypes:
            return
        for context, proto_list in sorted(
            self.prototypes.items(), key=lambda x: -len(x[1])
        )[:50]:
            if len(self.prototype_match_counts) <= self.max_total_prototypes:
                break
            self._prune_prototypes(context)

    def _init_lsh(self, n_tables: int = 8, n_bits: int = 12):
        self.lsh_tables = [{} for _ in range(n_tables)]
        self.lsh_projections = []
        overlap = max(1, int(n_bits * 0.15))
        for t in range(n_tables):
            rng = np.random.RandomState(self.seed + 1000 + t)
            proj = np.zeros((n_bits, self.dim), dtype=np.int8)
            for b in range(n_bits):
                n_ones = max(1, self.dim // 20)
                positions = rng.choice(self.dim, size=n_ones, replace=False)
                proj[b, positions] = 1
            if t > 0 and overlap > 0:
                for b in range(min(overlap, n_bits)):
                    proj[b] = self.lsh_projections[-1][b].copy()
            self.lsh_projections.append(proj)

    def _lsh_hash(self, hv: np.ndarray, table_idx: int) -> int:
        proj = self.lsh_projections[table_idx]
        binary = (hv + 1) // 2
        bits = (proj @ binary.astype(np.int32)) > 0
        hash_val = 0
        for b in bits:
            hash_val = (hash_val << 1) | int(b)
        return hash_val

    def _lsh_lookup(self, hv: np.ndarray) -> list:
        results = set()
        for t_idx, table in enumerate(self.lsh_tables):
            h = self._lsh_hash(hv, t_idx)
            if h in table:
                for item in table[h]:
                    results.add(item)
        return list(results)

    def _lsh_insert(self, key: tuple, hv: np.ndarray):
        for t_idx, table in enumerate(self.lsh_tables):
            h = self._lsh_hash(hv, t_idx)
            if h not in table:
                table[h] = []
            table[h].append(key)

    def _multi_scale_bundle(
        self, short_vectors: list, long_vectors: list
    ) -> np.ndarray:
        all_vectors = []
        phases = []
        for hv in short_vectors:
            all_vectors.append(hv)
            phases.append(self.short_order_weight)
        for hv in long_vectors:
            all_vectors.append(hv)
            phases.append(self.long_order_weight)
        if not all_vectors:
            return np.ones(self.dim, dtype=np.int8)
        return self._superposition_bundle(all_vectors, phases)

    def _track_change(self, current_hv: np.ndarray):
        if self.previous_hv is not None:
            change = float(np.count_nonzero(current_hv != self.previous_hv)) / self.dim
            self.hv_change_history.append(change)
        self.previous_hv = current_hv.copy()

    def anomaly_score(self) -> float:
        if len(self.hv_change_history) < 4:
            return 0.0
        recent = list(self.hv_change_history)[-8:]
        mean_change = float(np.mean(recent))
        std_change = float(np.std(recent)) + 1e-10
        return float(
            np.clip(abs(recent[-1] - mean_change) / (std_change * 3.0), 0.0, 1.0)
        )

    def _compute_entropy(self, scores: list) -> float:
        if not scores:
            return 0.0
        probs = np.array(scores, dtype=np.float64)
        probs = np.maximum(probs, 1e-10)
        probs = probs / np.sum(probs)
        return float(-np.sum(probs * np.log2(probs)) / (np.log2(len(probs) + 1)))

    def _adaptive_temperature(self, scores: list) -> float:
        entropy = self._compute_entropy(scores)
        self.entropy_history.append(entropy)
        avg_entropy = float(np.mean(list(self.entropy_history)))
        return float(
            np.clip(self.base_temperature * (0.6 + 0.8 * avg_entropy), 0.3, 1.5)
        )

    def set_stopwords(self, stopwords: set):
        self.stopword_ids = stopwords

    def _apply_content_bias(self, token_id: int, score: float) -> float:
        if token_id in self.stopword_ids:
            return score * self.stopword_penalty
        return score * self.content_bias

    def learn(self, sequence: list):
        self._adapt_order(len(sequence))
        for order in range(self.min_order, min(self.max_order, len(sequence)) + 1):
            for i in range(len(sequence) - order):
                context = tuple(sequence[i : i + order])
                target = sequence[i + order]
                context_hv = self._encode_context(context)
                target_hv = self.ensure_token_vector(target)
                encoded = context_hv * target_hv
                self._store_prototype(context, encoded)
                self._lsh_insert(context, encoded)
                self.ngram_counts[context] += 1
                self._track_change(encoded)

    def predict_next(self, context: tuple, n_candidates: int = 32) -> list:
        candidates = {}
        self._adapt_order(len(context))
        for order in range(self.max_order, self.min_order - 1, -1):
            if len(context) < order:
                continue
            ctx = context[-order:]
            context_hv = self._encode_context(ctx)
            target_protos = self.prototypes.get(ctx, [])
            if not target_protos:
                continue
            for proto_hv, count, pid in target_protos:
                predicted_hv = proto_hv * context_hv
                for tok, tok_hv in self.token_vectors.items():
                    sim = float(np.count_nonzero(predicted_hv == tok_hv)) / self.dim
                    if sim > 0.0:
                        weighted = self._apply_content_bias(tok, sim)
                        weighted *= 1.0 + 0.15 * order
                        weighted *= 1.0 + 0.05 * np.log1p(count)
                        candidates[tok] = max(candidates.get(tok, 0.0), weighted)
                self.prototype_match_counts[pid] = (
                    self.prototype_match_counts.get(pid, 0) + 1
                )
        if not candidates:
            self.cache_misses += 1
            most_common = sorted(self.ngram_counts.items(), key=lambda x: -x[1])
            for ctx_tup, cnt in most_common[:20]:
                if ctx_tup and ctx_tup[-1] not in candidates:
                    candidates[ctx_tup[-1]] = 0.3 * (1.0 + 0.1 * np.log1p(cnt))
        if candidates:
            scores = list(candidates.values())
            temp = self._adaptive_temperature(scores)
            if temp != 1.0:
                inv_temp = 1.0 / temp
                max_score = max(scores)
                candidates = {
                    tok: (s / max_score) ** inv_temp * max_score
                    for tok, s in candidates.items()
                }
        ranked = sorted(candidates.items(), key=lambda x: -x[1])
        return ranked[:n_candidates]

    def generate_block(
        self, context: tuple, block_size: int = 8, temperature: float = 0.8
    ) -> list:
        block = []
        current_context = list(context)
        for _ in range(block_size):
            candidates = self.predict_next(tuple(current_context), n_candidates=16)
            if not candidates:
                break
            scores = np.array([s for _, s in candidates], dtype=np.float64)
            scores = np.maximum(scores, 1e-10)
            scores = scores / np.max(scores)
            scores = scores ** (1.0 / max(temperature, 0.1))
            probs = scores / np.sum(scores)
            idx = int(np.random.choice(len(candidates), p=probs))
            token = candidates[idx][0]
            block.append(token)
            current_context.append(token)
        return block

    def generate_diverse_blocks(
        self, context: tuple, n_blocks: int = 8, min_block: int = 8, max_block: int = 24
    ) -> list:
        blocks = []
        for block_idx in range(n_blocks):
            bs = self.rng.randint(min_block, max_block + 1)
            temp = 0.5 + 0.8 * (block_idx / max(n_blocks - 1, 1))
            block = self.generate_block(context, block_size=bs, temperature=temp)
            if block:
                blocks.append(block)
        seen = set()
        unique = []
        for b in blocks:
            key = tuple(b)
            if key not in seen:
                seen.add(key)
                unique.append(b)
        return unique

    def reset_context_cache(self):
        self.context_state_cache.clear()


class NGramCascade:
    """Multi-level n-gram cascade with adaptive ordering."""

    def __init__(self, vocab_size: int, max_order: int = 6):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.counts = [
            defaultdict(lambda: defaultdict(int)) for _ in range(max_order + 1)
        ]
        self.total_counts = [defaultdict(int) for _ in range(max_order + 1)]

    def observe(self, sequence: list):
        for i in range(len(sequence)):
            self.counts[0][()][sequence[i]] += 1
            self.total_counts[0][()] += 1
            for order in range(1, min(self.max_order, i) + 1):
                ctx = tuple(sequence[i - order : i])
                self.counts[order][ctx][sequence[i]] += 1
                self.total_counts[order][ctx] += 1

    def predict(self, context: tuple, top_k: int = 32) -> list:
        candidates = {}
        for order in range(self.max_order, 0, -1):
            if len(context) < order:
                continue
            ctx = context[-order:]
            if ctx in self.counts[order]:
                total = self.total_counts[order][ctx]
                if total > 0:
                    for tok, cnt in self.counts[order][ctx].items():
                        candidates[tok] = max(
                            candidates.get(tok, 0.0),
                            (cnt / total) * (1.0 + 0.25 * order),
                        )
        if not candidates:
            total = self.total_counts[0][()]
            if total > 0:
                for tok, cnt in self.counts[0][()].items():
                    candidates[tok] = cnt / total
        ranked = sorted(candidates.items(), key=lambda x: -x[1])
        return ranked[:top_k]


class HDCDraftEngine:
    """Ultra-fast token drafting engine using hyperdimensional computing."""

    def __init__(
        self,
        vocab_size: int,
        hd_dim: int = 4096,
        max_order: int = 6,
        n_draft_candidates: int = 64,
        stopwords: Optional[set] = None,
    ):
        self.vocab_size = vocab_size
        self.n_draft_candidates = n_draft_candidates
        self.hd = HDCBundle(dim=hd_dim)
        self.ngram = NGramCascade(vocab_size, max_order=max_order)
        self.context_window: deque = deque(maxlen=128)
        self.draft_count = 0
        self.accept_count = 0
        if stopwords:
            self.hd.set_stopwords(stopwords)

    def observe(self, token_id: int):
        self.context_window.append(token_id)
        ctx = list(self.context_window)
        self.hd.learn(ctx)
        self.ngram.observe([token_id])

    def draft_block(self, block_size: int = 8) -> list:
        if len(self.context_window) < 1:
            return [[]]
        context = tuple(self.context_window)
        unique = self.hd.generate_diverse_blocks(
            context,
            n_blocks=self.n_draft_candidates,
            min_block=max(2, block_size - 4),
            max_block=block_size + 4,
        )
        if not unique:
            for _ in range(self.n_draft_candidates):
                draft = list(context)
                for _ in range(block_size):
                    nxt = self.hd.predict_next(tuple(draft), n_candidates=1)
                    if nxt:
                        draft.append(nxt[0][0])
                    else:
                        ng = self.ngram.predict(tuple(draft), top_k=1)
                        draft.append(
                            ng[0][0]
                            if ng
                            else int(np.random.randint(0, min(self.vocab_size, 10000)))
                        )
                candidate = draft[len(context) :]
                if tuple(candidate) not in {tuple(c) for c in unique}:
                    unique.append(candidate)
        return unique[: min(32, len(unique))]

    def acceptance_rate(self) -> float:
        return self.accept_count / max(self.draft_count, 1)

    def reset(self):
        self.context_window.clear()
        self.draft_count = 0
        self.accept_count = 0
        self.hd.reset_context_cache()

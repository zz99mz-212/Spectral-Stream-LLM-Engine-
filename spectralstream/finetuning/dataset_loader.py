"""
Multi-Source Dataset Loader for Fine-Tuning
=============================================
Handles loading datasets from various sources and formats.
"""

import csv
import json
import os
import re
from typing import Optional


class SimpleTokenizer:
    """Character-level tokenizer for CPU fine-tuning."""

    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.special_tokens = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._next_id = len(self.special_tokens)

    def train(self, texts):
        freq = {}
        for text in texts:
            for ch in text:
                freq[ch] = freq.get(ch, 0) + 1
        for ch, _ in sorted(freq.items(), key=lambda x: -x[1]):
            if self._next_id >= self.vocab_size:
                break
            if ch not in self.token_to_id:
                self.token_to_id[ch] = self._next_id
                self.id_to_token[self._next_id] = ch
                self._next_id += 1

    def encode(self, text, max_length=512):
        ids = [self.special_tokens["<bos>"]]
        for ch in text:
            ids.append(self.token_to_id.get(ch, self.special_tokens["<unk>"]))
            if len(ids) >= max_length - 1:
                break
        ids.append(self.special_tokens["<eos>"])
        return ids

    def decode(self, ids):
        return "".join(
            self.id_to_token.get(t, "<unk>") for t in ids if t not in (0, 1, 2)
        )

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "token_to_id": {str(k): v for k, v in self.token_to_id.items()},
                    "vocab_size": self.vocab_size,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.token_to_id = {}
        for k, v in data["token_to_id"].items():
            tok.token_to_id[int(k) if k.isdigit() else k] = v
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        nums = [k for k in tok.id_to_token if isinstance(k, int)]
        tok._next_id = max(nums) + 1 if nums else 4
        return tok

    @property
    def pad_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 2


class DatasetLoader:
    TEXT_FIELDS = [
        "text",
        "content",
        "question",
        "answer",
        "context",
        "input",
        "output",
        "prompt",
        "completion",
        "instruction",
    ]

    def __init__(self, source, fmt="auto", max_seq_length=512):
        self.source = source
        self.fmt = fmt
        self.max_seq_length = max_seq_length
        self.samples = []
        self.tokenizer = SimpleTokenizer()
        self.tokenized_samples = []
        if fmt == "auto":
            self.fmt = self._detect_format(source)
        self._load_dataset()
        self._build_tokenizer()
        self._tokenize()

    def _detect_format(self, source):
        if not os.path.exists(source) and "/" in source:
            return "huggingface"
        ext_map = {
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "jsonl",
            ".txt": "text",
            ".parquet": "parquet",
        }
        for ext, fmt in ext_map.items():
            if source.lower().endswith(ext):
                return fmt
        if os.path.exists(source):
            try:
                with open(source, "r", encoding="utf-8") as f:
                    first = f.readline().strip()
                    if first.startswith("{"):
                        try:
                            json.loads(first)
                            return "jsonl"
                        except json.JSONDecodeError:
                            return "json"
                    elif "," in first:
                        return "csv"
            except Exception:
                pass
        return "text"

    def _load_dataset(self):
        loaders = {
            "huggingface": self._load_huggingface,
            "csv": self._load_csv,
            "json": self._load_json,
            "jsonl": self._load_jsonl,
            "text": self._load_text,
            "parquet": self._load_parquet,
            "chatml": self._load_chatml,
        }
        loader = loaders.get(self.fmt)
        if loader is None:
            raise ValueError("Unsupported format: " + self.fmt)
        loader()

    def _extract_text(self, item):
        if isinstance(item, str):
            return item
        if not isinstance(item, dict):
            return str(item)
        for fn in self.TEXT_FIELDS:
            if fn in item and isinstance(item[fn], str) and item[fn].strip():
                return item[fn]
        parts = []
        for k, v in item.items():
            if isinstance(v, str) and v.strip():
                parts.append(k + ": " + v)
        return " ".join(parts) if parts else ""

    def _extract_qa_text(self, item):
        if "context" in item and "question" in item:
            parts = [
                "Context: " + str(item["context"]),
                "Question: " + str(item["question"]),
            ]
            if "answer" in item:
                ans = item["answer"]
                if isinstance(ans, list) and ans:
                    ans = ans[0]
                parts.append("Answer: " + str(ans))
            return " ".join(parts)
        if "question" in item and "answer" in item:
            ans = item["answer"]
            if isinstance(ans, list) and ans:
                ans = ans[0]
            return "Question: " + str(item["question"]) + " Answer: " + str(ans)
        return self._extract_text(item)

    def _load_huggingface(self):
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to load HuggingFace datasets. Install with: pip install huggingface_hub"
            )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            for fname in ["train.jsonl", "data.jsonl", "train.csv", "data.csv"]:
                try:
                    path = hf_hub_download(
                        repo_id=self.source,
                        filename=fname,
                        repo_type="dataset",
                        local_dir=tmp_dir,
                    )
                    self.source = path
                    if fname.endswith(".jsonl"):
                        self._load_jsonl()
                    else:
                        self._load_csv()
                    return
                except Exception:
                    continue
            try:
                files = list_repo_files(self.source, repo_type="dataset")
                for fn in files:
                    if fn.endswith((".jsonl", ".csv")):
                        path = hf_hub_download(
                            repo_id=self.source,
                            filename=fn,
                            repo_type="dataset",
                            local_dir=tmp_dir,
                        )
                        self.source = path
                        if fn.endswith(".jsonl"):
                            self._load_jsonl()
                        else:
                            self._load_csv()
                        return
            except Exception:
                pass
            raise ValueError("Could not load HuggingFace dataset: " + self.source)

    def _load_csv(self):
        with open(self.source, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = (
                    self._extract_qa_text(row)
                    if "question" in row
                    else self._extract_text(row)
                )
                if text:
                    self.samples.append(text)

    def _load_json(self):
        with open(self.source, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("data", data.get("items", data.get("rows", [data])))
            if not isinstance(items, list):
                items = [items]
        else:
            return
        for item in items:
            text = (
                self._extract_qa_text(item)
                if isinstance(item, dict) and "question" in item
                else self._extract_text(item)
            )
            if text:
                self.samples.append(text)

    def _load_jsonl(self):
        with open(self.source, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        text = (
                            self._extract_qa_text(item)
                            if "question" in item
                            else self._extract_text(item)
                        )
                    else:
                        text = str(item)
                except json.JSONDecodeError:
                    text = line
                if text:
                    self.samples.append(text)

    def _load_text(self):
        with open(self.source, "r", encoding="utf-8") as fp:
            content = fp.read()
        for para in content.split(chr(10) + chr(10)):
            para = para.strip()
            if para:
                self.samples.append(para)

    def _load_parquet(self):
        import pyarrow.parquet as pq

        table = pq.read_table(self.source)
        cols = table.column_names
        for i in range(len(table)):
            row = {c: table.column(c)[i].as_py() for c in cols}
            text = (
                self._extract_qa_text(row)
                if "question" in row
                else self._extract_text(row)
            )
            if text:
                self.samples.append(text)

    def _load_chatml(self):
        with open(self.source, "r", encoding="utf-8") as fp:
            content = fp.read()
        current_role = None
        current_text = []
        NL = chr(10)
        for line in content.split(NL):
            stripped = line.strip()
            if stripped.startswith(chr(119893)):
                if current_role and current_text:
                    self.samples.append(" ".join(current_text))
                current_role = (
                    stripped.split(None, 1)[0][1:] if stripped.split(None, 1) else None
                )
                current_text = []
            elif stripped.startswith(chr(119895)):
                if current_role and current_text:
                    self.samples.append(" ".join(current_text))
                current_role = None
                current_text = []
            elif current_role and stripped:
                current_text.append(stripped)
        if current_role and current_text:
            self.samples.append(" ".join(current_text))

    def _build_tokenizer(self):
        self.tokenizer.train(self.samples)

    def _tokenize(self):
        self.tokenized_samples = []
        for text in self.samples:
            ids = self.tokenizer.encode(text, self.max_seq_length)
            self.tokenized_samples.append(ids)

    def __len__(self):
        return len(self.tokenized_samples)

    def __getitem__(self, idx):
        ids = self.tokenized_samples[idx]
        input_ids = ids[:-1] if len(ids) > 1 else ids
        labels = ids[1:] if len(ids) > 1 else ids
        input_ids = input_ids + [0] * (self.max_seq_length - len(input_ids))
        labels = labels + [-100] * (self.max_seq_length - len(labels))
        return input_ids[: self.max_seq_length], labels[: self.max_seq_length]

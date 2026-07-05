"""
Memory-bounded streaming dataset loader with shuffle buffer.
Supports HuggingFace datasets (streaming), local text/JSON/JSONL files,
PDF (via PyMuPDF), and web scraping.
"""

import json
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np


class DatasetStreamer:
    """Memory-bounded streaming dataset loader with shuffle buffer.

    Streams samples from various sources without loading the entire
    dataset into RAM. A shuffle buffer maintains a window of samples
    for randomized iteration.

    Parameters
    ----------
    max_samples_in_ram : int
        Maximum samples to hold in memory at once (per chunk for HF).
    shuffle_buffer : int
        Size of the shuffle buffer for randomness.
    seed : int
        Random seed for shuffling.
    """

    def __init__(
        self,
        max_samples_in_ram: int = 1000,
        shuffle_buffer: int = 10000,
        seed: int = 42,
    ):
        self.max_samples_in_ram = max_samples_in_ram
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self._rng = random.Random(seed)
        self._buffer: List[Dict[str, Any]] = []
        self._source_iter: Optional[Iterator] = None
        self._sample_count = 0

    def from_huggingface(
        self,
        repo_id: str,
        split: str = "train",
        text_field: str = "text",
        **kwargs: Any,
    ) -> "DatasetStreamer":
        """Stream dataset from HuggingFace with chunked loading.

        Parameters
        ----------
        repo_id : str
            HuggingFace dataset repository ID (e.g., 'wikitext').
        split : str
            Dataset split to load ('train', 'validation', 'test').
        text_field : str
            Field name containing text data.
        **kwargs
            Additional arguments passed to ``load_dataset``.

        Returns
        -------
        self
        """
        try:
            from datasets import load_dataset, get_dataset_split_names
        except ImportError:
            raise ImportError(
                "Install 'datasets' for HuggingFace streaming: pip install datasets"
            )

        available = get_dataset_split_names(repo_id)
        if split not in available:
            split = available[0] if available else "train"

        dataset = load_dataset(
            repo_id,
            split=split,
            streaming=True,
            **kwargs,
        )
        self._source_iter = self._hf_iter(dataset, text_field)
        return self

    def from_text(
        self,
        path: str,
        encoding: str = "utf-8",
    ) -> "DatasetStreamer":
        """Stream text file line-by-line.

        Lines are grouped into samples by paragraph (double newline).
        """
        self._source_iter = self._text_iter(path, encoding)
        return self

    def from_jsonl(self, path: str, encoding: str = "utf-8") -> "DatasetStreamer":
        """Stream JSONL file line-by-line."""
        self._source_iter = self._jsonl_iter(path, encoding)
        return self

    def from_json(self, path: str, encoding: str = "utf-8") -> "DatasetStreamer":
        """Stream JSON array file (loaded in chunks)."""
        self._source_iter = self._json_iter(path, encoding)
        return self

    def from_pdf(
        self,
        path: str,
        chunk_size: int = 512,
    ) -> "DatasetStreamer":
        """Stream PDF file page-by-page via PyMuPDF.

        Parameters
        ----------
        path : str
            Path to PDF file.
        chunk_size : int
            Characters per chunk for long pages.
        """
        self._source_iter = self._pdf_iter(path, chunk_size)
        return self

    def from_web(
        self,
        url: str,
        chunk_size: int = 512,
    ) -> "DatasetStreamer":
        """Stream web page content by scraping.

        Uses newspaper3k or requests + BeautifulSoup as fallback.
        """
        self._source_iter = self._web_iter(url, chunk_size)
        return self

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self._stream()

    def __len__(self) -> int:
        return self._sample_count

    def _stream(self) -> Iterator[Dict[str, Any]]:
        """Core streaming loop with shuffle buffer."""
        buffer: List[Dict[str, Any]] = []

        for sample in self._source_iter or []:
            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer:
                self._rng.shuffle(buffer)
                while len(buffer) > self.shuffle_buffer // 2:
                    yield buffer.pop()
                    self._sample_count += 1

        self._rng.shuffle(buffer)
        while buffer:
            yield buffer.pop()
            self._sample_count += 1

    def _extract_text(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("text", "content", "input", "output", "prompt", "completion"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            parts = [str(v) for v in item.values() if isinstance(v, str) and v.strip()]
            return " ".join(parts) if parts else ""
        return str(item)

    def _hf_iter(self, dataset: Any, text_field: str) -> Iterator[Dict[str, Any]]:
        for i, example in enumerate(dataset):
            text = example.get(text_field, self._extract_text(example))
            if isinstance(text, str) and text.strip():
                yield {"text": text, "source": "huggingface", "id": i}
            if i >= self.max_samples_in_ram * 10:
                break

    def _text_iter(self, path: str, encoding: str) -> Iterator[Dict[str, Any]]:
        with open(path, "r", encoding=encoding) as f:
            content = f.read()
        paragraphs = content.split("\n\n")
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                yield {"text": para, "source": "text", "id": i}

    def _jsonl_iter(self, path: str, encoding: str) -> Iterator[Dict[str, Any]]:
        with open(path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        text = self._extract_text(item)
                        if text:
                            yield {"text": text, "source": "jsonl", "id": i, **item}
                    else:
                        yield {"text": str(item), "source": "jsonl", "id": i}
                except json.JSONDecodeError:
                    yield {"text": line, "source": "jsonl", "id": i}

    def _json_iter(self, path: str, encoding: str) -> Iterator[Dict[str, Any]]:
        with open(path, "r", encoding=encoding) as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        for i, item in enumerate(items):
            text = self._extract_text(item) if isinstance(item, dict) else str(item)
            if text:
                yield {"text": text, "source": "json", "id": i}

    def _pdf_iter(self, path: str, chunk_size: int) -> Iterator[Dict[str, Any]]:
        try:
            import fitz
        except ImportError:
            raise ImportError("Install PyMuPDF for PDF support: pip install pymupdf")
        doc = fitz.open(path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if not text.strip():
                continue
            for chunk_idx in range(0, len(text), chunk_size):
                chunk = text[chunk_idx : chunk_idx + chunk_size].strip()
                if chunk:
                    yield {
                        "text": chunk,
                        "source": "pdf",
                        "page": page_num,
                        "chunk": chunk_idx // chunk_size,
                    }
        doc.close()

    def _web_iter(self, url: str, chunk_size: int) -> Iterator[Dict[str, Any]]:
        try:
            from newspaper import Article

            article = Article(url)
            article.download()
            article.parse()
            text = article.text
        except ImportError:
            try:
                import requests
                from bs4 import BeautifulSoup

                resp = requests.get(url, timeout=30)
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
            except ImportError:
                raise ImportError(
                    "Install newspaper3k or requests+beautifulsoup4 for web scraping"
                )

        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size].strip()
            if chunk:
                yield {
                    "text": chunk,
                    "source": "web",
                    "url": url,
                    "chunk": i // chunk_size,
                }


def stream_dataset(
    source: str,
    max_samples_in_ram: int = 1000,
    shuffle_buffer: int = 10000,
    **kwargs: Any,
) -> DatasetStreamer:
    """Convenience function to create a streamer from a URI.

    URI schemes:
        hf://repo_id    -> HuggingFace dataset
        file://path     -> local file (auto-detect format)
        http(s)://url   -> web scraping
    """
    streamer = DatasetStreamer(
        max_samples_in_ram=max_samples_in_ram,
        shuffle_buffer=shuffle_buffer,
    )

    if source.startswith("hf://") or source.startswith("huggingface://"):
        repo_id = source.split("://", 1)[1]
        return streamer.from_huggingface(repo_id, **kwargs)

    if source.startswith("http://") or source.startswith("https://"):
        return streamer.from_web(source, **kwargs)

    if source.startswith("file://"):
        source = source[len("file://") :]

    if not os.path.exists(source):
        raise FileNotFoundError(f"Dataset source not found: {source}")

    ext = os.path.splitext(source)[1].lower()
    if ext == ".pdf":
        return streamer.from_pdf(source, **kwargs)
    if ext == ".jsonl":
        return streamer.from_jsonl(source)
    if ext == ".json":
        return streamer.from_json(source)
    return streamer.from_text(source)

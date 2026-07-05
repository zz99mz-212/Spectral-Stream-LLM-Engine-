from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from spectralstream.format.writer import SSFWriter
from spectralstream.format.reader import SSFReader


class SSFConverter:
    def __init__(self, output_path: str, metadata: Optional[dict] = None):
        self.output_path = Path(output_path)
        self._metadata = metadata or {}

    def from_gguf(self, gguf_path: str, **writer_kwargs: Any) -> dict:
        from spectralstream.format.gguf_parser_engine import GGUFParser, GGMLDequantizer

        parser = GGUFParser(gguf_path)
        parser.parse()
        meta = dict(self._metadata)
        meta.setdefault("source_format", "gguf")
        meta.setdefault("source_path", gguf_path)
        for k in (
            "general.architecture",
            "general.name",
            "llama.context_length",
            "llama.embedding_length",
            "llama.block_count",
            "llama.vocab_size",
        ):
            if k in parser.metadata:
                meta[k] = parser.metadata[k]
        with SSFWriter(str(self.output_path), metadata=meta, **writer_kwargs) as writer:
            for ti in parser.tensor_infos:
                name = ti["name"]
                raw = np.frombuffer(
                    parser._data,
                    dtype=np.uint8,
                    offset=parser.tensor_data_offset + ti["offset"],
                    count=ti["data_size"],
                ).copy()
                tensor = GGMLDequantizer.dequantize_fast(raw, ti["ggml_type"])
                writer.add_tensor(name, tensor)
        return {"output": str(self.output_path), "n_tensors": len(parser.tensor_infos)}

    def from_safetensors(self, st_path: str, **writer_kwargs: Any) -> dict:
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("safetensors not installed")
        meta = dict(self._metadata)
        meta.setdefault("source_format", "safetensors")
        meta.setdefault("source_path", st_path)
        with SSFWriter(str(self.output_path), metadata=meta, **writer_kwargs) as writer:
            with safe_open(st_path, framework="np") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    writer.add_tensor(name, tensor)
        return {
            "output": str(self.output_path),
            "n_tensors": len(list(safe_open(st_path, framework="np").keys())),
        }

    def from_numpy_dict(
        self, tensors: dict[str, np.ndarray], **writer_kwargs: Any
    ) -> dict:
        with SSFWriter(
            str(self.output_path), metadata=self._metadata, **writer_kwargs
        ) as writer:
            for name, tensor in tensors.items():
                writer.add_tensor(name, tensor)
        return {"output": str(self.output_path), "n_tensors": len(tensors)}

    def from_ssf(
        self,
        source_path: str,
        tensor_names: Optional[List[str]] = None,
        **writer_kwargs: Any,
    ) -> dict:
        reader = SSFReader(source_path)
        meta = dict(self._metadata)
        meta.setdefault("source_format", "ssf")
        meta.setdefault("source_path", source_path)
        names = tensor_names if tensor_names is not None else reader.tensor_names()
        rindex = reader._index
        with SSFWriter(str(self.output_path), metadata=meta, **writer_kwargs) as writer:
            for name in names:
                tensor = reader.get_tensor(name)
                e = rindex.get(name) if rindex is not None else None
                if e is not None:
                    writer.add_tensor(
                        name,
                        tensor,
                        method=e.compression_method,
                        params=e.compression_params,
                        quality_metrics=dict(e.quality_metrics),
                    )
                else:
                    writer.add_tensor(name, tensor)
        reader.close()
        return {"output": str(self.output_path), "n_tensors": len(names)}

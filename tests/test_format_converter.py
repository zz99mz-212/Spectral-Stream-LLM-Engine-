import sys
import os
import tempfile

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.format.converter import SSFConverter
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestSSFConverter:
    def test_init(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf") as f:
            converter = SSFConverter(f.name)
            assert str(converter.output_path) == f.name
            assert converter._metadata == {}

    def test_init_with_metadata(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf") as f:
            converter = SSFConverter(f.name, metadata={"author": "test"})
            assert converter._metadata["author"] == "test"

    def test_from_numpy_dict(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            out_path = f.name

        try:
            tensors = {
                "weight1": np.random.randn(4, 4).astype(np.float32),
                "bias1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            }
            converter = SSFConverter(out_path)
            result = converter.from_numpy_dict(tensors)
            assert result["n_tensors"] == 2
            assert result["output"] == out_path
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_from_numpy_dict_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            out_path = f.name

        try:
            converter = SSFConverter(out_path)
            result = converter.from_numpy_dict({})
            assert result["n_tensors"] == 0
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_from_numpy_dict_single(self):
        with tempfile.NamedTemporaryFile(suffix=".ssf", delete=False) as f:
            out_path = f.name

        try:
            tensors = {"single_weight": np.eye(3, dtype=np.float32)}
            converter = SSFConverter(out_path)
            result = converter.from_numpy_dict(tensors)
            assert result["n_tensors"] == 1

            from spectralstream.format.reader import SSFReader

            reader = SSFReader(out_path)
            names = reader.tensor_names()
            assert "single_weight" in names
            reader.close()
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

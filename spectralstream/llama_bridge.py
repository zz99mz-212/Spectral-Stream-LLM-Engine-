from typing import Optional
from pathlib import Path


def find_model_in_lmstudio(model_name: str) -> Optional[str]:
    home = Path.home()
    search_paths = [
        home / ".lmstudio" / "models",
        home / ".lmstudio" / "models" / "huggingface",
        home / "lmstudio" / "models",
        Path("/usr/local/share/lmstudio/models"),
    ]

    model_name = model_name.lower().replace(" ", "_")
    for base in search_paths:
        if not base.exists():
            continue
        for f in base.rglob("*.gguf"):
            if model_name in f.name.lower():
                return str(f)

    return None


def list_available_models() -> list[dict]:
    home = Path.home()
    search_paths = [
        home / ".lmstudio" / "models",
        home / ".lmstudio" / "models" / "huggingface",
        home / "lmstudio" / "models",
    ]
    models = []
    for base in search_paths:
        if not base.exists():
            continue
        for f in base.rglob("*.gguf"):
            size_gb = f.stat().st_size / (1024**3) if f.stat().st_size > 0 else 0
            models.append(
                {
                    "path": str(f),
                    "name": f.stem,
                    "size_gb": round(size_gb, 2),
                    "parent": str(f.parent.name),
                }
            )
    return models

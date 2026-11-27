from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def environment_info() -> dict:
    versions = {}
    for name in ("spatial-framework", "torch", "torch-geometric", "torch-sparse", "torch-scatter", "numpy", "scipy",
                 "pandas", "scanpy", "anndata", "scikit-learn", "scikit-misc", "matplotlib", "numba", "umap-learn", "igraph"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import torch
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": versions, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def capture_freeze(directory: Path) -> None:
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True)
    (directory / "pip-freeze.txt").write_text(result.stdout, encoding="utf-8")


def source_hashes() -> dict:
    root = Path(__file__).parent
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.suffix in (".py", ".R")}

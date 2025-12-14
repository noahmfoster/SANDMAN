from pathlib import Path
import os

def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / ".env").exists():
            return p
    raise RuntimeError("Could not find repo root (no .env found)")

REPO_ROOT = find_repo_root(Path(__file__).resolve())

DATA_DIR = (REPO_ROOT / os.getenv("DATA_DIR", "data")).resolve()
# MODEL_DIR = (REPO_ROOT / os.getenv("MODEL_DIR", "models")).resolve()

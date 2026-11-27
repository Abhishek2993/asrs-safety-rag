"""Project paths and default settings.

All paths can be overridden with environment variables so the package works
both from a source checkout and when installed elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("ASRS_RAG_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = Path(os.environ.get("ASRS_RAG_INDEX_DIR", DATA_DIR / "index"))
MODEL_DIR = Path(os.environ.get("ASRS_RAG_MODEL_DIR", DATA_DIR / "models"))
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 13


def default_data_dir() -> Path:
    """Return ``data/raw`` if it contains exports, otherwise the committed sample."""
    if RAW_DIR.exists() and any(RAW_DIR.glob("*.csv")):
        return RAW_DIR
    return SAMPLE_DIR

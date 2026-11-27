"""Text embedding backends.

* :class:`SentenceTransformerEmbedder` - local sentence-transformers model
  (default ``all-MiniLM-L6-v2``: 384-d, 22M parameters, fast on CPU, no API key).
* :class:`HashingEmbedder` - deterministic hashed bag-of-n-grams vectors. It
  needs no model download and is used in tests/CI and as an offline fallback.

All embedders return L2-normalised ``float32`` arrays so that inner product
equals cosine similarity.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from asrs_rag.config import DEFAULT_EMBEDDING_MODEL


class Embedder(Protocol):
    """Minimal interface shared by all embedding backends."""

    name: str

    @property
    def dim(self) -> int:  # pragma: no cover - protocol
        """Embedding dimensionality."""
        ...

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:  # pragma: no cover
        """Embed ``texts`` into an ``(n, dim)`` L2-normalised float32 array."""
        ...


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class SentenceTransformerEmbedder:
    """Wrapper around a local sentence-transformers model (loaded lazily)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu") -> None:
        self.name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """The underlying ``SentenceTransformer`` (imported on first use)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        """Embedding dimensionality reported by the model."""
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed texts; long inputs are truncated by the model (256 tokens for MiniLM)."""
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder:
    """Hashed word uni/bi-gram vectors: deterministic, dependency-light, no download."""

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hashing-{dim}"
        self._dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=dim,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm=None,
            stop_words="english",
        )

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._dim

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed texts (``batch_size`` is accepted for interface compatibility)."""
        x = self._vectorizer.transform(list(texts)).toarray()
        x = np.log1p(x)
        return _l2_normalise(x)


def get_embedder(name: str | None = None) -> Embedder:
    """Return an embedder by name: ``hashing[-DIM]`` or a sentence-transformers model id."""
    name = name or DEFAULT_EMBEDDING_MODEL
    if name.startswith("hashing"):
        dim = int(name.split("-")[1]) if "-" in name else 512
        return HashingEmbedder(dim)
    return SentenceTransformerEmbedder(name)

"""Chunking, embedding and the persistent hybrid index (FAISS + BM25).

Index layout on disk (``data/index/`` by default)::

    manifest.json      embedder name, chunking parameters, counts, build time
    reports.parquet    one row per report: metadata, cleaned text, labels
    chunks.parquet     one row per chunk: chunk_id, acn, text
    dense.faiss        exact inner-product FAISS index over chunk embeddings

BM25 is rebuilt from ``chunks.parquet`` when the index is loaded (a few seconds),
which avoids pickling and keeps the on-disk format portable.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from asrs_rag.bm25 import BM25
from asrs_rag.clean import metadata_line
from asrs_rag.embed import Embedder, get_embedder

log = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

#: Report columns persisted in the index.
REPORT_COLUMNS = [
    "acn",
    "date",
    "year",
    "month",
    "flight_phase",
    "aircraft_type",
    "aircraft_operator",
    "far_part",
    "reporter_function",
    "anomaly",
    "contributing_factors",
    "human_factors",
    "primary_problem",
    "labels",
    "synopsis",
    "narrative",
    "synopsis_clean",
    "narrative_clean",
]


@dataclass(frozen=True)
class ChunkConfig:
    """Chunking parameters (in words; MiniLM truncates at 256 word-pieces ~ 190 words)."""

    max_words: int = 160
    overlap_words: int = 40


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a light regex (good enough for narratives)."""
    parts: list[str] = []
    for para in re.split(r"\n\s*\n", text or ""):
        parts.extend(s.strip() for s in _SENTENCE_RE.split(para) if s.strip())
    return parts


def chunk_text(text: str, max_words: int = 160, overlap_words: int = 40) -> list[str]:
    """Greedily pack whole sentences into chunks of at most ``max_words`` words.

    Consecutive chunks share up to ``overlap_words`` words (whole trailing
    sentences) so that an event described across a chunk boundary is still
    retrievable. A single sentence longer than ``max_words`` is split on words.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be in [0, max_words)")

    sentences: list[list[str]] = []
    for sent in split_sentences(text):
        words = sent.split()
        for i in range(0, len(words), max_words):
            sentences.append(words[i : i + max_words])

    chunks: list[str] = []
    current: list[list[str]] = []
    count = 0
    for sent in sentences:
        if current and count + len(sent) > max_words:
            chunks.append(" ".join(w for s in current for w in s))
            # carry trailing sentences forward as overlap
            carry: list[list[str]] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) > overlap_words:
                    break
                carry.insert(0, prev)
                carried += len(prev)
            if carried + len(sent) > max_words:
                carry, carried = [], 0
            current, count = carry, carried
        current.append(sent)
        count += len(sent)
    if current:
        chunks.append(" ".join(w for s in current for w in s))
    return chunks


def make_chunks(reports: pd.DataFrame, config: ChunkConfig) -> pd.DataFrame:
    """Create chunk rows (``chunk_id``, ``acn``, ``text``) for prepared reports.

    Each chunk is prefixed with the report's metadata line so that it is
    interpretable on its own. The synopsis is kept together with the start of
    the narrative.
    """
    rows = []
    for rec in reports.to_dict("records"):
        header = metadata_line(rec)
        body = rec["narrative_clean"]
        if rec.get("synopsis_clean"):
            body = f"Synopsis: {rec['synopsis_clean']}\n\n{body}"
        for piece in chunk_text(body, config.max_words, config.overlap_words):
            rows.append({"acn": rec["acn"], "text": f"{header}\n{piece}"})
    chunks = pd.DataFrame(rows, columns=["acn", "text"])
    chunks.insert(0, "chunk_id", np.arange(len(chunks), dtype=np.int64))
    return chunks


class ReportIndex:
    """Persistent hybrid index over prepared ASRS reports."""

    def __init__(
        self,
        reports: pd.DataFrame,
        chunks: pd.DataFrame,
        dense: faiss.Index,
        embedder: Embedder,
        manifest: dict,
    ) -> None:
        self.reports = reports.reset_index(drop=True)
        self.chunks = chunks.reset_index(drop=True)
        self.dense = dense
        self.embedder = embedder
        self.manifest = manifest
        self.bm25 = BM25(self.chunks["text"].tolist())
        self._acn_to_row = {acn: i for i, acn in enumerate(self.reports["acn"])}
        self.chunk_acns = self.chunks["acn"].to_numpy()

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        reports: pd.DataFrame,
        embedder: Embedder | str | None = None,
        config: ChunkConfig | None = None,
        batch_size: int = 16,
    ) -> ReportIndex:
        """Chunk, embed and index prepared reports (see ``clean.prepare_corpus``)."""
        config = config or ChunkConfig()
        if embedder is None or isinstance(embedder, str):
            embedder = get_embedder(embedder)
        reports = reports[[c for c in REPORT_COLUMNS if c in reports.columns]].copy()
        chunks = make_chunks(reports, config)
        start = time.time()
        vectors = embedder.encode(chunks["text"].tolist(), batch_size=batch_size)
        dense = faiss.IndexFlatIP(vectors.shape[1])
        dense.add(vectors)
        manifest = {
            "embedder": embedder.name,
            "dim": int(vectors.shape[1]),
            "chunking": asdict(config),
            "n_reports": len(reports),
            "n_chunks": len(chunks),
            "embedding_seconds": round(time.time() - start, 1),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        log.info("Indexed %d reports as %d chunks", len(reports), len(chunks))
        return cls(reports, chunks, dense, embedder, manifest)

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        """Write the index to ``directory``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.reports.to_parquet(d / "reports.parquet", index=False)
        self.chunks.to_parquet(d / "chunks.parquet", index=False)
        faiss.write_index(self.dense, str(d / "dense.faiss"))
        (d / "manifest.json").write_text(json.dumps(self.manifest, indent=2))
        return d

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> ReportIndex:
        """Load an index written by :meth:`save`; the embedder must match the manifest."""
        d = Path(directory)
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(f"No index at {d}; run `python -m asrs_rag index` first")
        manifest = json.loads((d / "manifest.json").read_text())
        embedder = embedder or get_embedder(manifest["embedder"])
        if embedder.name != manifest["embedder"]:
            raise ValueError(
                f"Index built with {manifest['embedder']!r}, got embedder {embedder.name!r}"
            )
        reports = pd.read_parquet(d / "reports.parquet")
        for col in ("flight_phase", "anomaly", "contributing_factors", "human_factors", "labels"):
            if col in reports.columns:
                reports[col] = reports[col].map(lambda v: list(v) if v is not None else [])
        chunks = pd.read_parquet(d / "chunks.parquet")
        dense = faiss.read_index(str(d / "dense.faiss"))
        return cls(reports, chunks, dense, embedder, manifest)

    # ---------------------------------------------------------------- helpers
    def report(self, acn: str) -> dict:
        """Return one report as a dict."""
        return self.reports.iloc[self._acn_to_row[acn]].to_dict()

    def has(self, acn: str) -> bool:
        """Whether ``acn`` is in the index."""
        return acn in self._acn_to_row

    def chunk_mask(self, allowed_acns: set[str] | None) -> np.ndarray | None:
        """Boolean mask over chunks for a set of allowed ACNs (None = all)."""
        if allowed_acns is None:
            return None
        return np.isin(self.chunk_acns, list(allowed_acns))

"""Hybrid (dense + BM25) retrieval of similar ASRS reports with metadata filters.

Pipeline for one query narrative:

1. clean the query exactly like indexed text (abbreviation expansion);
2. **dense**: embed the query, search FAISS for the most similar chunks
   (cosine similarity), restricted to chunks that pass the metadata filters;
3. **sparse**: score every allowed chunk with BM25;
4. collapse chunks to reports, keeping each report's best chunk (max-pooling);
5. fuse the two report rankings with **Reciprocal Rank Fusion**
   ``score = sum_i w_i / (k + rank_i)`` (k = 60). RRF uses ranks, not raw
   scores, so cosine similarities and BM25 scores never need calibrating
   against each other.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import faiss
import numpy as np
import pandas as pd

from asrs_rag.bm25 import tokenize
from asrs_rag.clean import clean_text
from asrs_rag.index import ReportIndex

Mode = Literal["hybrid", "dense", "bm25"]


@dataclass
class Filters:
    """Metadata filters; ``None``/empty means "no constraint".

    Text filters are case-insensitive substring matches, so ``phases=["approach"]``
    matches both "Initial Approach" and "Final Approach".
    """

    year_from: int | None = None
    year_to: int | None = None
    phases: list[str] = field(default_factory=list)
    aircraft: str | None = None
    primary_problems: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    include_acns: set[str] | None = None
    exclude_acns: set[str] = field(default_factory=set)

    def is_empty(self) -> bool:
        """True when no filter is set."""
        return (
            self.year_from is None
            and self.year_to is None
            and not self.phases
            and not self.aircraft
            and not self.primary_problems
            and not self.labels
            and self.include_acns is None
            and not self.exclude_acns
        )

    def mask(self, reports: pd.DataFrame) -> np.ndarray:
        """Boolean mask of reports satisfying all filters."""
        m = np.ones(len(reports), dtype=bool)
        years = reports["year"].astype("float").to_numpy()
        if self.year_from is not None:
            m &= years >= self.year_from
        if self.year_to is not None:
            m &= years <= self.year_to
        if self.phases:
            wanted = [p.lower() for p in self.phases]
            m &= (
                reports["flight_phase"]
                .map(lambda ps: any(w in p.lower() for p in ps for w in wanted))
                .to_numpy(dtype=bool)
            )
        if self.aircraft:
            a = self.aircraft.lower()
            m &= reports["aircraft_type"].str.lower().str.contains(a, regex=False).to_numpy()
        if self.primary_problems:
            wanted = {p.lower() for p in self.primary_problems}
            m &= reports["primary_problem"].str.lower().isin(wanted).to_numpy()
        if self.labels:
            wanted = set(self.labels)
            m &= reports["labels"].map(lambda ls: bool(wanted & set(ls))).to_numpy(dtype=bool)
        if self.include_acns is not None:
            m &= reports["acn"].isin(self.include_acns).to_numpy()
        if self.exclude_acns:
            m &= ~reports["acn"].isin(self.exclude_acns).to_numpy()
        return m


@dataclass
class Hit:
    """One retrieved report."""

    acn: str
    score: float
    rank: int
    dense_score: float | None
    bm25_score: float | None
    dense_rank: int | None
    bm25_rank: int | None
    chunk_text: str
    matched_terms: list[str]
    report: dict

    def to_dict(self) -> dict:
        """Serialisable summary (without the full narrative)."""
        r = self.report
        return {
            "rank": self.rank,
            "acn": self.acn,
            "score": round(self.score, 5),
            "dense_score": None if self.dense_score is None else round(self.dense_score, 4),
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 3),
            "date": str(r.get("date"))[:7],
            "flight_phase": "; ".join(r.get("flight_phase", [])),
            "aircraft_type": r.get("aircraft_type"),
            "primary_problem": r.get("primary_problem"),
            "labels": list(r.get("labels", [])),
            "synopsis": r.get("synopsis"),
            "matched_terms": self.matched_terms,
        }


def reciprocal_rank_fusion(
    rankings: Iterable[list[str]], k: int = 60, weights: Iterable[float] | None = None
) -> dict[str, float]:
    """Fuse ranked lists of IDs: ``score(id) = sum_i w_i / (k + rank_i(id))`` (rank from 1)."""
    rankings = list(rankings)
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + w / (k + rank)
    return fused


def _collapse(chunk_ids: np.ndarray, scores: np.ndarray, chunk_acns: np.ndarray) -> list:
    """Max-pool chunk scores to reports; returns [(acn, score, chunk_id)] best first."""
    best: dict[str, tuple[float, int]] = {}
    for cid, s in zip(chunk_ids, scores, strict=True):
        acn = chunk_acns[cid]
        if acn not in best or s > best[acn][0]:
            best[acn] = (float(s), int(cid))
    return sorted(((a, s, c) for a, (s, c) in best.items()), key=lambda t: -t[1])


class HybridRetriever:
    """Retrieve similar reports from a :class:`ReportIndex`."""

    def __init__(
        self,
        index: ReportIndex,
        rrf_k: int = 60,
        candidates: int = 100,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.index = index
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.weights = weights

    def _allowed_chunks(self, filters: Filters | None) -> np.ndarray | None:
        if filters is None or filters.is_empty():
            return None
        allowed = set(self.index.reports.loc[filters.mask(self.index.reports), "acn"])
        return self.index.chunk_mask(allowed)

    def dense_search(self, query: str, n_chunks: int, chunk_mask: np.ndarray | None) -> list:
        """Top chunks by cosine similarity, collapsed to reports."""
        qv = self.index.embedder.encode([query])
        params = None
        if chunk_mask is not None:
            ids = np.flatnonzero(chunk_mask).astype(np.int64)
            if len(ids) == 0:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))
            n_chunks = min(n_chunks, len(ids))
        n_chunks = min(n_chunks, self.index.dense.ntotal)
        scores, ids = self.index.dense.search(qv, n_chunks, params=params)
        keep = ids[0] >= 0
        return _collapse(ids[0][keep], scores[0][keep], self.index.chunk_acns)

    def bm25_search(self, query: str, n_chunks: int, chunk_mask: np.ndarray | None) -> list:
        """Top chunks by BM25, collapsed to reports (zero-score chunks are dropped)."""
        scores = self.index.bm25.scores(query)
        if chunk_mask is not None:
            scores = np.where(chunk_mask, scores, -np.inf)
        n = min(n_chunks, len(scores))
        top = np.argpartition(-scores, n - 1)[:n] if n else np.array([], dtype=int)
        top = top[np.isfinite(scores[top]) & (scores[top] > 0)]
        top = top[np.argsort(-scores[top])]
        return _collapse(top, scores[top], self.index.chunk_acns)

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Filters | None = None,
        mode: Mode = "hybrid",
    ) -> list[Hit]:
        """Return the top-``k`` reports most similar to ``query``."""
        q = clean_text(query)
        mask = self._allowed_chunks(filters)
        n_chunks = self.candidates * 3  # several chunks per report
        dense = self.dense_search(q, n_chunks, mask) if mode in ("hybrid", "dense") else []
        sparse = self.bm25_search(q, n_chunks, mask) if mode in ("hybrid", "bm25") else []
        dense, sparse = dense[: self.candidates], sparse[: self.candidates]

        d_info = {a: (s, c, r) for r, (a, s, c) in enumerate(dense, start=1)}
        s_info = {a: (s, c, r) for r, (a, s, c) in enumerate(sparse, start=1)}
        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                [[a for a, _, _ in dense], [a for a, _, _ in sparse]],
                k=self.rrf_k,
                weights=self.weights,
            )
        elif mode == "dense":
            fused = {a: s for a, s, _ in dense}
        else:
            fused = {a: s for a, s, _ in sparse}
        ranked = sorted(fused.items(), key=lambda t: (-t[1], t[0]))[:k]

        # Highlights use the user's own words, not the inserted acronym expansions.
        q_terms = set(tokenize(query))
        hits = []
        for rank, (acn, score) in enumerate(ranked, start=1):
            d, s = d_info.get(acn), s_info.get(acn)
            chunk_id = (d or s)[1]
            if d and s and s[0] > 0 and mode == "hybrid" and s[2] < d[2]:
                chunk_id = s[1]  # show the chunk that ranked best
            text = self.index.chunks.at[chunk_id, "text"]
            hits.append(
                Hit(
                    acn=acn,
                    score=float(score),
                    rank=rank,
                    dense_score=d[0] if d else None,
                    bm25_score=s[0] if s else None,
                    dense_rank=d[2] if d else None,
                    bm25_rank=s[2] if s else None,
                    chunk_text=text,
                    matched_terms=sorted(q_terms & set(tokenize(text))),
                    report=self.index.report(acn),
                )
            )
        return hits

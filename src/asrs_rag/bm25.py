"""Okapi BM25 keyword scoring on a sparse term matrix.

For a query ``q`` and document ``d``::

    score(d, q) = sum_{t in q} idf(t) * tf(t,d) * (k1 + 1)
                                  / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

``k1`` controls term-frequency saturation and ``b`` document-length
normalisation; ``k1=1.5, b=0.75`` are the usual defaults. The idf is the
non-negative Lucene variant. All per-document term weights are precomputed
into one sparse matrix ``W`` so scoring a query is a single sparse
matrix-vector product, which is much faster than looping over documents.
"""

from __future__ import annotations

import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from asrs_rag.clean import DEIDENT_RE

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens without stop words or ASRS de-identification codes."""
    out = []
    for tok in _TOKEN_RE.findall(text or ""):
        if DEIDENT_RE.match(tok):
            continue
        low = tok.lower()
        if len(low) < 2 or low in ENGLISH_STOP_WORDS:
            continue
        out.append(low)
    return out


class BM25:
    """BM25 index over a list of texts."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, dtype=np.float32
        )
        tf = self.vectorizer.fit_transform(texts).tocsr()
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = doc_len.mean() if n_docs else 0.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # Precompute the BM25 weight of every (doc, term) pair.
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        norm = k1 * (1 - b + b * doc_len[rows] / (avgdl or 1.0))
        data = self.idf[tf.indices] * tf.data * (k1 + 1) / (tf.data + norm)
        self.weights = sparse.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape)

    def __len__(self) -> int:
        return self.weights.shape[0]

    def query_vector(self, query: str) -> sparse.csr_matrix:
        """Term-count vector of a query in the index vocabulary (unknown terms dropped)."""
        return self.vectorizer.transform([query])

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for ``query`` (query term counts are used once)."""
        q = self.query_vector(query)
        q.data[:] = 1.0  # each distinct query term contributes once
        return np.asarray((self.weights @ q.T).todense()).ravel()

"""Contributing-factor classification (multi-label).

Three approaches share one interface (``predict_proba`` / ``predict``):

* :class:`SupervisedClassifier` - TF-IDF *or* sentence embeddings + one
  logistic regression per label (one-vs-rest), ``class_weight="balanced"`` to
  counter label imbalance, and a per-label decision threshold tuned on
  out-of-fold predictions of the training set (never on the test set);
* :class:`NeighbourVoteClassifier` - labels voted by the retrieved similar
  reports (no training, no LLM);
* :class:`LLMClassifier` - zero-shot or few-shot prompting of an LLM, where the
  few-shot examples are the retrieved similar reports with their coded labels.

The input is always the *narrative only* (cleaned), because that is what a
new report contains; analyst-written synopses and coded fields are not used
as features.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

from asrs_rag.bm25 import tokenize
from asrs_rag.config import RANDOM_SEED
from asrs_rag.embed import Embedder, get_embedder
from asrs_rag.index import chunk_text
from asrs_rag.labels import LABEL_DESCRIPTIONS, LABELS

log = logging.getLogger(__name__)

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


@dataclass
class LabelScore:
    """A label decision with its confidence."""

    label: str
    probability: float
    threshold: float
    predicted: bool
    #: False when the probability is within ``margin`` of the threshold.
    confident: bool

    def to_dict(self) -> dict:
        """JSON-friendly representation."""
        return {
            "label": self.label,
            "probability": round(self.probability, 3),
            "threshold": round(self.threshold, 3),
            "predicted": self.predicted,
            "confident": self.confident,
        }


def best_thresholds(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Per-label threshold maximising F1 on (out-of-fold) probabilities."""
    out = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        t = y_true[:, j].astype(bool)
        if t.sum() == 0:
            continue
        best_f1 = -1.0
        for thr in THRESHOLD_GRID:
            p = proba[:, j] >= thr
            tp = (t & p).sum()
            f1 = 2 * tp / (t.sum() + p.sum()) if (t.sum() + p.sum()) else 0.0
            if f1 > best_f1:
                best_f1, out[j] = f1, thr
    return out


def scores_from_proba(
    proba: np.ndarray, thresholds: np.ndarray, margin: float = 0.1
) -> list[list[LabelScore]]:
    """Convert probabilities into :class:`LabelScore` lists (sorted by probability)."""
    result = []
    for row in np.atleast_2d(proba):
        items = [
            LabelScore(
                label=lab,
                probability=float(p),
                threshold=float(thr),
                predicted=bool(p >= thr),
                confident=bool(abs(p - thr) >= margin),
            )
            for lab, p, thr in zip(LABELS, row, thresholds, strict=True)
        ]
        result.append(sorted(items, key=lambda s: -s.probability))
    return result


class SupervisedClassifier:
    """One-vs-rest logistic regression over TF-IDF or embedding features."""

    def __init__(
        self,
        features: str = "tfidf",
        embedder: Embedder | str | None = None,
        C: float = 4.0,
        tune_thresholds: bool = True,
        seed: int = RANDOM_SEED,
    ) -> None:
        if features not in ("tfidf", "embedding"):
            raise ValueError("features must be 'tfidf' or 'embedding'")
        self.features = features
        self.embedder_name = embedder.name if hasattr(embedder, "name") else (embedder or None)
        self._embedder = embedder if hasattr(embedder, "encode") else None
        self.C = C
        self.tune_thresholds = tune_thresholds
        self.seed = seed
        self.vectorizer: TfidfVectorizer | None = None
        self.model: OneVsRestClassifier | None = None
        self.thresholds = np.full(len(LABELS), 0.5)

    # ---------------------------------------------------------------- features
    @property
    def embedder(self) -> Embedder:
        """Embedding model (loaded lazily, e.g. after unpickling)."""
        if self._embedder is None:
            self._embedder = get_embedder(self.embedder_name)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Mean of chunk embeddings, so long narratives are not truncated."""
        pieces, owner = [], []
        for i, t in enumerate(texts):
            chunks = chunk_text(t, 160, 40) or [""]
            pieces.extend(chunks)
            owner.extend([i] * len(chunks))
        vecs = self.embedder.encode(pieces, batch_size=16)
        out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
        np.add.at(out, np.asarray(owner), vecs)
        out /= np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-9)
        return out

    def _new_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
            max_features=100_000,
        )

    def _new_model(self) -> OneVsRestClassifier:
        return OneVsRestClassifier(
            LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000),
            n_jobs=1,
        )

    # ------------------------------------------------------------- train/use
    def fit(self, texts: list[str], y: np.ndarray, embeddings: np.ndarray | None = None):
        """Fit the model; ``embeddings`` may be passed to avoid re-encoding."""
        y = np.asarray(y)
        if self.features == "embedding":
            x = embeddings if embeddings is not None else self._embed(texts)
        else:
            self.vectorizer = self._new_vectorizer()
            x = self.vectorizer.fit_transform(texts)
        if self.tune_thresholds:
            oof = np.zeros(y.shape, dtype=float)
            for tr, va in KFold(5, shuffle=True, random_state=self.seed).split(x):
                if self.features == "tfidf":
                    vec = self._new_vectorizer()
                    xtr = vec.fit_transform([texts[i] for i in tr])
                    xva = vec.transform([texts[i] for i in va])
                else:
                    xtr, xva = x[tr], x[va]
                oof[va] = self._new_model().fit(xtr, y[tr]).predict_proba(xva)
            self.thresholds = best_thresholds(y, oof)
        self.model = self._new_model().fit(x, y)
        return self

    def _x(self, texts: list[str], embeddings: np.ndarray | None = None):
        if self.features == "embedding":
            return embeddings if embeddings is not None else self._embed(texts)
        assert self.vectorizer is not None, "model is not fitted"
        return self.vectorizer.transform(texts)

    def predict_proba(self, texts: list[str], embeddings: np.ndarray | None = None) -> np.ndarray:
        """Per-label probabilities ``(n, n_labels)``."""
        assert self.model is not None, "model is not fitted"
        return self.model.predict_proba(self._x(texts, embeddings))

    def predict(self, texts: list[str], embeddings: np.ndarray | None = None) -> np.ndarray:
        """Binary label matrix using the tuned thresholds."""
        return (self.predict_proba(texts, embeddings) >= self.thresholds).astype(int)

    def explain(self, label: str, top: int = 10) -> list[tuple[str, float]]:
        """Highest-weight TF-IDF terms for a label (TF-IDF models only)."""
        if self.vectorizer is None or self.model is None:
            return []
        j = LABELS.index(label)
        coef = self.model.estimators_[j].coef_.ravel()
        vocab = np.array(self.vectorizer.get_feature_names_out())
        idx = np.argsort(-coef)[:top]
        return [(str(vocab[i]), float(coef[i])) for i in idx]

    def save(self, path: str | Path) -> Path:
        """Persist with joblib (the embedding model itself is not pickled)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        emb, self._embedder = self._embedder, None
        try:
            joblib.dump(self, path)
        finally:
            self._embedder = emb
        return path

    @staticmethod
    def load(path: str | Path) -> SupervisedClassifier:
        """Load a model written by :meth:`save`."""
        return joblib.load(path)


class NeighbourVoteClassifier:
    """Predict labels from the coded labels of retrieved similar reports.

    ``score(label) = sum over top-k neighbours of w_r * [label in neighbour]``
    with rank weights ``w_r = 1/r`` normalised to sum to 1.
    """

    def __init__(self, retriever, k: int = 10, allowed_acns: set[str] | None = None) -> None:
        self.retriever = retriever
        self.k = k
        self.allowed_acns = allowed_acns
        self.thresholds = np.full(len(LABELS), 0.3)

    def _proba_one(self, text: str, exclude: set[str] | None = None) -> np.ndarray:
        from asrs_rag.retrieve import Filters

        f = Filters(include_acns=self.allowed_acns, exclude_acns=exclude or set())
        hits = self.retriever.search(text, k=self.k, filters=f)
        w = np.array([1.0 / h.rank for h in hits])
        w = w / w.sum() if len(w) else w
        out = np.zeros(len(LABELS))
        for wi, h in zip(w, hits, strict=True):
            for lab in h.report["labels"]:
                out[LABELS.index(lab)] += wi
        return out

    def predict_proba(self, texts: list[str], exclude: list[set[str]] | None = None) -> np.ndarray:
        """Vote shares for each label."""
        exclude = exclude or [set()] * len(texts)
        return np.vstack([self._proba_one(t, e) for t, e in zip(texts, exclude, strict=True)])

    def fit_thresholds(self, texts: list[str], y: np.ndarray, acns: list[str]) -> None:
        """Tune per-label thresholds leaving each training report out of its own vote."""
        proba = self.predict_proba(texts, [{a} for a in acns])
        self.thresholds = best_thresholds(np.asarray(y), proba)

    def predict(self, texts: list[str]) -> np.ndarray:
        """Binary label matrix."""
        return (self.predict_proba(texts) >= self.thresholds).astype(int)


# ---------------------------------------------------------------- LLM-based

SYSTEM_PROMPT = (
    "You are an aviation safety analyst coding NASA ASRS incident reports. "
    "Assign every contributing-factor label that the narrative supports. "
    "Use only the label names given. Answer with JSON only."
)


def _label_block() -> str:
    return "\n".join(f"- {lab}: {LABEL_DESCRIPTIONS[lab]}" for lab in LABELS)


def _truncate_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def build_classification_prompt(
    narrative: str, examples: list[dict] | None = None, max_words: int = 350
) -> str:
    """Prompt for zero-shot (no ``examples``) or few-shot classification.

    ``examples`` are retrieved similar reports: dicts with ``acn``, ``text`` and
    ``labels`` (their coded labels).
    """
    parts = ["Labels:", _label_block(), ""]
    if examples:
        parts.append("Similar past reports with their coded labels:")
        for ex in examples:
            parts.append(
                f"[ACN {ex['acn']}] {_truncate_words(ex['text'], 90)}\n"
                f"labels: {json.dumps(list(ex['labels']))}"
            )
        parts.append("")
    parts += [
        "New report narrative:",
        _truncate_words(narrative, max_words),
        "",
        'Return JSON: {"labels": [<label names>]}',
    ]
    return "\n".join(parts)


def parse_labels(response: str) -> list[str]:
    """Extract valid label names from an LLM response (JSON first, then a lenient scan)."""
    match = re.search(r"\{.*\}", response or "", flags=re.S)
    candidates: list[str] = []
    if match:
        try:
            obj = json.loads(match.group(0))
            raw = obj.get("labels", []) if isinstance(obj, dict) else []
            candidates = [str(x).strip().lower() for x in raw]
        except json.JSONDecodeError:
            candidates = []
    if not candidates:
        candidates = re.findall(r"[a-z_]+", (response or "").lower())
    seen = [c for c in candidates if c in LABELS]
    return [lab for lab in LABELS if lab in seen]


class LLMClassifier:
    """Zero-/few-shot LLM classification using retrieved reports as context."""

    def __init__(
        self,
        llm,
        retriever=None,
        n_examples: int = 4,
        allowed_acns: set[str] | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.n_examples = n_examples
        self.allowed_acns = allowed_acns

    def examples_for(self, narrative: str, exclude: set[str] | None = None) -> list[dict]:
        """Retrieve few-shot examples (empty list for zero-shot)."""
        if self.retriever is None or self.n_examples == 0:
            return []
        from asrs_rag.retrieve import Filters

        f = Filters(include_acns=self.allowed_acns, exclude_acns=exclude or set())
        hits = self.retriever.search(narrative, k=self.n_examples, filters=f)
        return [
            {
                "acn": h.acn,
                "text": h.report.get("synopsis") or h.report.get("narrative_clean", ""),
                "labels": list(h.report["labels"]),
            }
            for h in hits
        ]

    def classify(self, narrative: str, exclude: set[str] | None = None) -> dict:
        """Return ``{"labels": [...], "examples": [acn...], "raw": response}``."""
        examples = self.examples_for(narrative, exclude)
        prompt = build_classification_prompt(narrative, examples)
        raw = self.llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=60, json_mode=True)
        return {
            "labels": parse_labels(raw),
            "examples": [e["acn"] for e in examples],
            "raw": raw,
        }

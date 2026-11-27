"""Evaluation metrics implemented with NumPy so every number is easy to audit.

Multi-label classification
    Each report can have several labels. For label *j*, TP/FP/FN are counted
    over reports. **Micro** averaging pools TP/FP/FN over all labels (dominated
    by frequent labels); **macro** averaging computes F1 per label and takes the
    unweighted mean (every label counts equally, so rare labels such as
    *fatigue* matter).

Retrieval
    precision@k = relevant results in top k / k
    recall@k    = relevant results in top k / all relevant items
    MRR         = mean over queries of 1 / rank of the first relevant result
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2 * p * r, p + r)


def multilabel_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]
) -> dict[str, object]:
    """Micro/macro precision, recall and F1 plus per-label counts.

    Macro averages include every label (a label never predicted and never
    present scores F1 = 0 only if it has support; labels with zero support are
    excluded from macro averages).
    """
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch {yt.shape} vs {yp.shape}")
    tp = (yt & yp).sum(axis=0)
    fp = (~yt & yp).sum(axis=0)
    fn = (yt & ~yp).sum(axis=0)
    tn = (~yt & ~yp).sum(axis=0)

    per_label = {}
    for j, lab in enumerate(labels):
        p = _safe_div(tp[j], tp[j] + fp[j])
        r = _safe_div(tp[j], tp[j] + fn[j])
        per_label[lab] = {
            "precision": p,
            "recall": r,
            "f1": _f1(p, r),
            "support": int(tp[j] + fn[j]),
            "predicted": int(tp[j] + fp[j]),
            "tp": int(tp[j]),
            "fp": int(fp[j]),
            "fn": int(fn[j]),
            "tn": int(tn[j]),
        }

    micro_p = _safe_div(tp.sum(), tp.sum() + fp.sum())
    micro_r = _safe_div(tp.sum(), tp.sum() + fn.sum())
    supported = [m for m in per_label.values() if m["support"] > 0]
    macro_p = float(np.mean([m["precision"] for m in supported])) if supported else 0.0
    macro_r = float(np.mean([m["recall"] for m in supported])) if supported else 0.0
    macro_f1 = float(np.mean([m["f1"] for m in supported])) if supported else 0.0
    return {
        "n_samples": int(yt.shape[0]),
        "micro": {"precision": micro_p, "recall": micro_r, "f1": _f1(micro_p, micro_r)},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "subset_accuracy": float((yt == yp).all(axis=1).mean()) if len(yt) else 0.0,
        "hamming_loss": float((yt != yp).mean()) if yt.size else 0.0,
        "per_label": per_label,
    }


def confusion_summary(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], top: int = 8
) -> list[dict[str, object]]:
    """Most frequent "label swaps" within the same report.

    Multi-label problems have no single confusion matrix. This summary counts,
    for every report, each pair (missed label A, wrongly predicted label B),
    i.e. cases where the model probably attributed the event to B instead of A.
    """
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    pairs: Counter[tuple[str, str]] = Counter()
    for t, p in zip(yt, yp, strict=True):
        missed = [labels[j] for j in np.flatnonzero(t & ~p)]
        extra = [labels[j] for j in np.flatnonzero(~t & p)]
        for a in missed:
            for b in extra:
                pairs[(a, b)] += 1
    return [
        {"missed": a, "predicted_instead": b, "count": c} for (a, b), c in pairs.most_common(top)
    ]


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-``k`` retrieved items that are relevant (divides by k)."""
    rel = set(relevant)
    return sum(1 for r in retrieved[:k] if r in rel) / k


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of all relevant items found in the top ``k`` (0 if nothing is relevant)."""
    rel = set(relevant)
    return _safe_div(sum(1 for r in retrieved[:k] if r in rel), len(rel))


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant item (0 if none is retrieved)."""
    rel = set(relevant)
    for i, r in enumerate(retrieved, start=1):
        if r in rel:
            return 1.0 / i
    return 0.0

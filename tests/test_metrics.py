import numpy as np
import pytest
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

from asrs_rag.metrics import (
    confusion_summary,
    multilabel_report,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

LABELS = ["a", "b", "c"]
Y_TRUE = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]])
Y_PRED = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]])


def test_hand_computed_example():
    rep = multilabel_report(Y_TRUE, Y_PRED, LABELS)
    a = rep["per_label"]["a"]
    assert (a["tp"], a["fp"], a["fn"], a["tn"]) == (2, 0, 1, 2)
    assert a["precision"] == 1.0 and a["recall"] == pytest.approx(2 / 3)
    # micro: TP=4, FP=2, FN=3
    assert rep["micro"]["precision"] == pytest.approx(4 / 6)
    assert rep["micro"]["recall"] == pytest.approx(4 / 7)
    assert rep["subset_accuracy"] == pytest.approx(1 / 5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_scikit_learn(seed):
    rng = np.random.default_rng(seed)
    yt = rng.integers(0, 2, size=(50, 6))
    yp = rng.integers(0, 2, size=(50, 6))
    labels = [str(i) for i in range(6)]
    rep = multilabel_report(yt, yp, labels)
    for avg in ("micro", "macro"):
        assert rep[avg]["f1"] == pytest.approx(f1_score(yt, yp, average=avg, zero_division=0))
        assert rep[avg]["precision"] == pytest.approx(
            precision_score(yt, yp, average=avg, zero_division=0)
        )
        assert rep[avg]["recall"] == pytest.approx(
            recall_score(yt, yp, average=avg, zero_division=0)
        )
    assert rep["hamming_loss"] == pytest.approx(hamming_loss(yt, yp))


def test_label_without_support_is_excluded_from_macro():
    yt = np.array([[1, 0], [1, 0]])
    yp = np.array([[1, 0], [1, 0]])
    assert multilabel_report(yt, yp, ["x", "never"])["macro"]["f1"] == 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        multilabel_report(np.zeros((2, 3)), np.zeros((2, 2)), LABELS)


def test_confusion_summary_counts_label_swaps():
    swaps = confusion_summary(Y_TRUE, Y_PRED, LABELS)
    as_tuples = {(s["missed"], s["predicted_instead"]): s["count"] for s in swaps}
    assert as_tuples == {("a", "b"): 1}  # report 5: true a, predicted b


def test_retrieval_metrics():
    retrieved = ["x", "r1", "y", "r2", "z"]
    relevant = {"r1", "r2", "r3", "r4"}
    assert precision_at_k(retrieved, relevant, 1) == 0.0
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 4)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 2)
    assert recall_at_k(retrieved, set(), 5) == 0.0
    assert reciprocal_rank(["x"], relevant) == 0.0
    # fewer results than k still divides precision by k
    assert precision_at_k(["r1"], relevant, 5) == pytest.approx(1 / 5)

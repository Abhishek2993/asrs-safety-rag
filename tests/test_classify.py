"""Classifier tests on synthetic data (fast, no model downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from asrs_rag.classify import (
    LLMClassifier,
    NeighbourVoteClassifier,
    SupervisedClassifier,
    best_thresholds,
    build_classification_prompt,
    parse_labels,
    scores_from_proba,
)
from asrs_rag.embed import HashingEmbedder
from asrs_rag.labels import LABELS, to_matrix
from asrs_rag.retrieve import HybridRetriever

VOCAB = {
    "weather": "wind shear turbulence icing thunderstorm gusts visibility fog",
    "fatigue": "tired fatigue exhausted sleep duty day rest",
    "aircraft_equipment": "hydraulic pump failure engine warning fault inoperative",
    "communication": "readback misheard frequency congested clearance misunderstood",
}


def synthetic(n: int = 240, seed: int = 0):
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    keys = list(VOCAB)
    for _ in range(n):
        chosen = [k for k in keys if rng.random() < 0.35] or [keys[rng.integers(len(keys))]]
        words = []
        for k in chosen:
            words += list(rng.choice(VOCAB[k].split(), size=6))
        filler = ["the", "crew", "then", "continued", "normally", "after", "that"]
        words += list(rng.choice(filler, size=8))
        rng.shuffle(words)
        texts.append(" ".join(words))
        labels.append(chosen)
    return texts, to_matrix(labels)


@pytest.mark.parametrize("features", ["tfidf", "embedding"])
def test_supervised_classifier_learns_separable_labels(features):
    texts, y = synthetic()
    clf = SupervisedClassifier(features=features, embedder=HashingEmbedder(256))
    clf.fit(texts[:200], y[:200])
    pred = clf.predict(texts[200:])
    cols = [LABELS.index(k) for k in VOCAB]
    accuracy = (pred[:, cols] == y[200:, cols]).mean()
    assert accuracy > 0.9
    assert clf.predict_proba(texts[:3]).shape == (3, len(LABELS))


def test_supervised_save_load_roundtrip(tmp_path):
    texts, y = synthetic(120)
    clf = SupervisedClassifier(tune_thresholds=False).fit(texts, y)
    path = clf.save(tmp_path / "m.joblib")
    loaded = SupervisedClassifier.load(path)
    assert np.allclose(clf.predict_proba(texts[:5]), loaded.predict_proba(texts[:5]))
    top_terms = [t for t, _ in loaded.explain("fatigue", top=5)]
    assert set(top_terms) & set(VOCAB["fatigue"].split())


def test_invalid_feature_type():
    with pytest.raises(ValueError):
        SupervisedClassifier(features="bow")


def test_best_thresholds_picks_f1_optimal_cutoff():
    y = np.array([[1], [1], [0], [0]])
    proba = np.array([[0.4], [0.35], [0.2], [0.1]])
    thr = best_thresholds(y, proba)
    assert 0.2 < thr[0] <= 0.35


def test_scores_flag_low_confidence_near_threshold():
    proba = np.zeros((1, len(LABELS)))
    proba[0, 0], proba[0, 1] = 0.92, 0.55
    scores = {s.label: s for s in scores_from_proba(proba, np.full(len(LABELS), 0.5))[0]}
    assert scores[LABELS[0]].predicted and scores[LABELS[0]].confident
    assert scores[LABELS[1]].predicted and not scores[LABELS[1]].confident
    assert not scores[LABELS[2]].predicted


def test_neighbour_vote_classifier(toy_index):
    clf = NeighbourVoteClassifier(HybridRetriever(toy_index), k=2)
    proba = clf.predict_proba(["severe wind shear and gusts on final approach"])
    assert proba[0, LABELS.index("weather")] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"labels": ["weather", "fatigue"]}', ["fatigue", "weather"]),
        ('Sure! {"labels": ["Weather", "made_up"]}', ["weather"]),
        ("labels: procedure, human_factors", ["human_factors", "procedure"]),
        (
            '{"labels": ["organizational", "Communication Breakdown"]}',
            ["communication", "organisational"],
        ),
        ("{broken json", []),
        ("", []),
    ],
)
def test_parse_labels(response, expected):
    assert parse_labels(response) == expected


def test_prompt_contains_labels_examples_and_truncated_narrative():
    narrative = "word " * 1000
    ex = [{"acn": "123", "text": "Crew flew into icing.", "labels": ["weather"]}]
    prompt = build_classification_prompt(narrative, ex, max_words=50)
    assert all(lab in prompt for lab in LABELS)
    assert "[ACN 123]" in prompt and '["weather"]' in prompt
    assert prompt.count("word") <= 60


class FakeLLM:
    name = "fake"

    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt, system=None, max_tokens=512, json_mode=False):
        self.prompts.append(prompt)
        return self.response


def test_llm_classifier_few_shot_uses_retrieved_examples(toy_index):
    llm = FakeLLM('{"labels": ["weather"]}')
    clf = LLMClassifier(llm, HybridRetriever(toy_index), n_examples=2)
    out = clf.classify("We hit wind shear on final", exclude={"2000001"})
    assert out["labels"] == ["weather"]
    assert len(out["examples"]) == 2 and "2000001" not in out["examples"]
    assert "Similar past reports" in llm.prompts[0]


def test_llm_classifier_zero_shot_has_no_examples():
    llm = FakeLLM('{"labels": []}')
    out = LLMClassifier(llm, retriever=None).classify("Routine flight.")
    assert out == {"labels": [], "examples": [], "raw": '{"labels": []}'}
    assert "Similar past reports" not in llm.prompts[0]

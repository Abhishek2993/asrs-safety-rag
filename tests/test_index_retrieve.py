"""Chunking, BM25, fusion and retrieval ranking on a toy corpus (no model download)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from asrs_rag.bm25 import BM25, tokenize
from asrs_rag.embed import HashingEmbedder
from asrs_rag.index import ReportIndex, chunk_text, split_sentences
from asrs_rag.retrieve import Filters, HybridRetriever, reciprocal_rank_fusion

# ---------------------------------------------------------------- chunking


def test_split_sentences():
    assert split_sentences("One. Two! Three?\n\nFour.") == ["One.", "Two!", "Three?", "Four."]


def test_short_text_is_one_chunk():
    assert chunk_text("A short narrative. Nothing else.", max_words=50) == [
        "A short narrative. Nothing else."
    ]


def test_chunks_respect_max_words_and_overlap():
    text = " ".join(f"Sentence {i} has exactly six words." for i in range(30))
    chunks = chunk_text(text, max_words=30, overlap_words=12)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 30 for c in chunks)
    # consecutive chunks overlap by whole sentences
    for a, b in itertools.pairwise(chunks):
        last_sentence = a.split(". ")[-1]
        assert b.startswith(last_sentence.rstrip(".")) or last_sentence in b


def test_every_word_is_covered():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words) + ".", max_words=64, overlap_words=0)
    assert " ".join(chunks).split() == [*words[:-1], words[-1] + "."]
    assert all(len(c.split()) <= 64 for c in chunks)


def test_invalid_chunk_parameters():
    with pytest.raises(ValueError):
        chunk_text("x", max_words=10, overlap_words=10)
    with pytest.raises(ValueError):
        chunk_text("x", max_words=0)


# -------------------------------------------------------------------- BM25


def test_tokenize_drops_stopwords_and_deidentification_codes():
    assert tokenize("The aircraft at ZZZ1 departed RWY XXR") == ["aircraft", "departed", "rwy"]


def test_bm25_matches_hand_computation():
    docs = ["wind shear runway excursion", "wind", "engine failure"]
    assert [len(tokenize(d)) for d in docs] == [4, 1, 2]
    bm = BM25(docs, k1=1.5, b=0.75)
    n, avgdl = 3, (4 + 1 + 2) / 3
    idf_wind = math.log1p((n - 2 + 0.5) / (2 + 0.5))
    idf_shear = math.log1p((n - 1 + 0.5) / (1 + 0.5))

    def term(idf, tf, dl):
        return idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))

    expected0 = term(idf_wind, 1, 4) + term(idf_shear, 1, 4)
    expected1 = term(idf_wind, 1, 1)
    scores = bm.scores("wind shear")
    assert scores[0] == pytest.approx(expected0, rel=1e-5)
    assert scores[1] == pytest.approx(expected1, rel=1e-5)
    assert scores[2] == 0


def test_bm25_unknown_terms_score_zero():
    assert BM25(["alpha beta"]).scores("gamma").tolist() == [0.0]


# ------------------------------------------------------------------ fusion


def test_reciprocal_rank_fusion():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert max(fused, key=fused.get) == "b"
    assert fused["d"] == pytest.approx(1 / 63)


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(128)
    a, b = e.encode(["wind shear on final"]), e.encode(["wind shear on final"])
    assert np.allclose(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------- retrieval


@pytest.mark.parametrize("mode", ["hybrid", "dense", "bm25"])
def test_retrieval_ranks_the_relevant_report_first(toy_index, mode):
    r = HybridRetriever(toy_index)
    cases = {
        "We hit wind shear on short final and went around": "2000001",
        "Pilots were exhausted after a long duty day and missed a crossing restriction": "2000003",
        "We misunderstood the taxi instruction and crossed the hold short line": "2000004",
        "Oil pressure warning and engine shutdown in the climb": "2000006",
    }
    for query, expected in cases.items():
        hits = r.search(query, k=3, mode=mode)
        assert hits[0].acn == expected, (mode, query, [h.acn for h in hits])


def test_hits_carry_scores_metadata_and_highlights(toy_index):
    hit = HybridRetriever(toy_index).search("wind shear go-around on final", k=2)[0]
    assert hit.rank == 1 and hit.score > 0
    assert hit.dense_score is not None and hit.bm25_score is not None
    assert "shear" in hit.matched_terms
    d = hit.to_dict()
    assert d["primary_problem"] == "Weather" and d["labels"] == ["weather"]


def test_abbreviations_in_query_are_expanded(toy_index):
    hits = HybridRetriever(toy_index).search("ACFT ENG oil pressure low in CLB", k=1, mode="bm25")
    assert hits[0].acn == "2000006"


def test_filters_restrict_results(toy_index):
    r = HybridRetriever(toy_index)
    hits = r.search("weather", k=10, filters=Filters(phases=["approach"]))
    assert {h.acn for h in hits} <= {"2000001", "2000002"}
    hits = r.search("aircraft problem", k=10, filters=Filters(year_from=2022))
    assert {h.acn for h in hits} <= {"2000004", "2000005"}
    hits = r.search("engine", k=10, filters=Filters(aircraft="b737", labels=["aircraft_equipment"]))
    assert {h.acn for h in hits} == {"2000006"}
    hits = r.search("wind shear", k=10, filters=Filters(exclude_acns={"2000001"}))
    assert "2000001" not in {h.acn for h in hits}
    assert r.search("anything", filters=Filters(primary_problems=["Nonexistent"])) == []


def test_index_roundtrip(tmp_path, toy_index):
    toy_index.save(tmp_path)
    loaded = ReportIndex.load(tmp_path, HashingEmbedder(256))
    assert loaded.manifest["n_reports"] == 6
    assert loaded.report("2000005")["labels"] == ["aircraft_equipment"]
    q = "hydraulic pump deferred under the MEL"
    a = [h.acn for h in HybridRetriever(toy_index).search(q)]
    b = [h.acn for h in HybridRetriever(loaded).search(q)]
    assert a == b


def test_loading_with_wrong_embedder_fails(tmp_path, toy_index):
    toy_index.save(tmp_path)
    with pytest.raises(ValueError):
        ReportIndex.load(tmp_path, HashingEmbedder(64))

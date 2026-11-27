"""Evaluation harness. Every number in the README is produced by these functions.

* :func:`retrieval_proxy_eval` - precision@k / recall@k / MRR with "same ASRS
  primary problem" as an automatic relevance proxy, compared with chance.
* :func:`retrieval_labelled_eval` - the same metrics on hand-labelled queries
  (``data/eval/queries.jsonl``).
* :func:`classification_eval` - supervised (TF-IDF, embeddings) and
  neighbour-vote classifiers on a held-out split, plus a temporal split.
* :func:`llm_classification_eval` - zero-shot vs few-shot (RAG) LLM
  classification on a sample of the same held-out split.
* :func:`faithfulness_eval` - citation precision and statement coverage of
  LLM trend summaries.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from asrs_rag.cite import check_citations, enforce_citations
from asrs_rag.classify import LLMClassifier, NeighbourVoteClassifier, SupervisedClassifier
from asrs_rag.config import RANDOM_SEED
from asrs_rag.index import ReportIndex
from asrs_rag.labels import LABELS, to_matrix
from asrs_rag.metrics import (
    confusion_summary,
    multilabel_report,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from asrs_rag.retrieve import Filters, HybridRetriever
from asrs_rag.summarise import select_evidence, summarise

log = logging.getLogger(__name__)

MODES = ("bm25", "dense", "hybrid")
UNINFORMATIVE_PROBLEMS = {"", "Ambiguous"}


def save_json(obj: object, path: str | Path) -> Path:
    """Write JSON with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    return path


# ------------------------------------------------------------------ splitting


def make_split(acns: Sequence[str], test_frac: float = 0.2, seed: int = RANDOM_SEED) -> dict:
    """Random report-level train/test split (sorted ACN lists)."""
    acns = sorted(acns)
    rng = random.Random(seed)
    shuffled = acns[:]
    rng.shuffle(shuffled)
    n_test = round(len(acns) * test_frac)
    return {"seed": seed, "test": sorted(shuffled[:n_test]), "train": sorted(shuffled[n_test:])}


# ------------------------------------------------------------------ retrieval


def _aggregate(per_query: list[dict], ks: Sequence[int]) -> dict:
    out = {f"P@{k}": float(np.mean([q[f"P@{k}"] for q in per_query])) for k in ks}
    out |= {f"R@{k}": float(np.mean([q[f"R@{k}"] for q in per_query])) for k in ks}
    out["MRR"] = float(np.mean([q["RR"] for q in per_query]))
    return out


def retrieval_proxy_eval(
    index: ReportIndex,
    n_queries: int = 300,
    ks: Sequence[int] = (1, 5, 10),
    modes: Sequence[str] = MODES,
    seed: int = RANDOM_SEED,
) -> dict:
    """Leave-one-out retrieval with "same primary problem" as relevance.

    Each query is a report's narrative; the report itself is excluded from
    the results, and every other report with the same primary problem counts
    as relevant. Reports whose primary problem is empty or "Ambiguous" are
    not used as queries. ``chance P@k`` is the expected precision of a random
    ranking (= share of relevant reports in the collection).
    """
    reports = index.reports
    eligible = reports[~reports["primary_problem"].isin(UNINFORMATIVE_PROBLEMS)]
    sample = eligible.sample(n=min(n_queries, len(eligible)), random_state=seed)
    by_problem = reports.groupby("primary_problem")["acn"].apply(set).to_dict()
    retriever = HybridRetriever(index)
    max_k = max(ks)
    results: dict = {"n_queries": len(sample), "relevance": "same primary problem"}
    chance = [
        (len(by_problem[r["primary_problem"]]) - 1) / (len(reports) - 1)
        for r in sample.to_dict("records")
    ]
    results["chance"] = {f"P@{k}": float(np.mean(chance)) for k in ks}
    for mode in modes:
        per_query = []
        for r in sample.to_dict("records"):
            relevant = by_problem[r["primary_problem"]] - {r["acn"]}
            hits = retriever.search(
                r["narrative"], k=max_k, filters=Filters(exclude_acns={r["acn"]}), mode=mode
            )
            got = [h.acn for h in hits]
            q = {f"P@{k}": precision_at_k(got, relevant, k) for k in ks}
            q |= {f"R@{k}": recall_at_k(got, relevant, k) for k in ks}
            q["RR"] = reciprocal_rank(got, relevant)
            per_query.append(q)
        results[mode] = _aggregate(per_query, ks)
        log.info("proxy %s: %s", mode, results[mode])
    return results


def load_queries(path: str | Path) -> list[dict]:
    """Read hand-labelled queries (JSON lines)."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieval_labelled_eval(
    index: ReportIndex,
    queries: list[dict],
    ks: Sequence[int] = (1, 3, 5, 10),
    modes: Sequence[str] = MODES,
) -> dict:
    """P@k / R@k / MRR on hand-labelled queries (``relevant`` = list of ACNs)."""
    retriever = HybridRetriever(index)
    missing = sorted({a for q in queries for a in q["relevant"] if not index.has(a)})
    if missing:
        raise ValueError(f"Relevant ACNs not in index: {missing[:5]}...")
    results: dict = {
        "n_queries": len(queries),
        "mean_relevant_per_query": float(np.mean([len(q["relevant"]) for q in queries])),
    }
    per_mode_queries = {}
    for mode in modes:
        per_query = []
        for q in queries:
            got = [h.acn for h in retriever.search(q["query"], k=max(ks), mode=mode)]
            row = {f"P@{k}": precision_at_k(got, q["relevant"], k) for k in ks}
            row |= {f"R@{k}": recall_at_k(got, q["relevant"], k) for k in ks}
            row["RR"] = reciprocal_rank(got, q["relevant"])
            row["id"] = q["id"]
            per_query.append(row)
        results[mode] = _aggregate(per_query, ks)
        per_mode_queries[mode] = per_query
    results["per_query"] = per_mode_queries
    return results


# -------------------------------------------------------------- classification


def _report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rep = multilabel_report(y_true, y_pred, LABELS)
    rep["confusions"] = confusion_summary(y_true, y_pred, LABELS)
    return rep


def classification_eval(
    index: ReportIndex,
    split: dict,
    include_embedding: bool = True,
    knn_validation: int = 600,
    seed: int = RANDOM_SEED,
) -> tuple[dict, dict]:
    """Train on ``split['train']``, evaluate on ``split['test']``.

    Returns ``(results, models)``. Models: TF-IDF LR, embedding LR (optional),
    neighbour vote (thresholds tuned leave-one-out on a training subsample),
    plus a temporal check (train <= 2019, test >= 2020) for TF-IDF.
    """
    rep = index.reports.set_index("acn")
    train, test = rep.loc[split["train"]], rep.loc[split["test"]]
    y_train, y_test = to_matrix(train["labels"].tolist()), to_matrix(test["labels"].tolist())
    x_train, x_test = train["narrative_clean"].tolist(), test["narrative_clean"].tolist()
    results: dict = {
        "n_train": len(train),
        "n_test": len(test),
        "label_prevalence_test": {lab: float(y_test[:, j].mean()) for j, lab in enumerate(LABELS)},
    }
    models: dict = {}

    t0 = time.time()
    tfidf = SupervisedClassifier("tfidf").fit(x_train, y_train)
    results["tfidf_lr"] = _report(y_test, tfidf.predict(x_test))
    results["tfidf_lr"]["thresholds"] = dict(zip(LABELS, tfidf.thresholds.tolist(), strict=True))
    results["tfidf_lr"]["train_seconds"] = round(time.time() - t0, 1)
    results["tfidf_lr"]["top_terms"] = {lab: tfidf.explain(lab, 8) for lab in LABELS}
    models["tfidf_lr"] = tfidf
    log.info("tfidf micro-F1 %.3f", results["tfidf_lr"]["micro"]["f1"])

    if include_embedding:
        t0 = time.time()
        emb = SupervisedClassifier("embedding", embedder=index.embedder)
        e_train = emb._embed(x_train)
        e_test = emb._embed(x_test)
        emb.fit(x_train, y_train, embeddings=e_train)
        results["embedding_lr"] = _report(y_test, emb.predict(x_test, embeddings=e_test))
        results["embedding_lr"]["train_seconds"] = round(time.time() - t0, 1)
        models["embedding_lr"] = emb
        log.info("embedding micro-F1 %.3f", results["embedding_lr"]["micro"]["f1"])

    # Neighbour vote: tune thresholds on a training subsample, leave-one-out.
    knn = NeighbourVoteClassifier(HybridRetriever(index), k=10, allowed_acns=set(split["train"]))
    rng = random.Random(seed)
    val_acns = rng.sample(split["train"], min(knn_validation, len(split["train"])))
    val = rep.loc[val_acns]
    knn.fit_thresholds(val["narrative_clean"].tolist(), to_matrix(val["labels"].tolist()), val_acns)
    results["neighbour_vote"] = _report(y_test, knn.predict(x_test))
    models["neighbour_vote"] = knn

    # Temporal generalisation (concept drift): older reports -> newer reports.
    old = rep[rep["year"] <= 2019]
    new = rep[rep["year"] >= 2020]
    if len(old) and len(new):
        clf = SupervisedClassifier("tfidf").fit(
            old["narrative_clean"].tolist(), to_matrix(old["labels"].tolist())
        )
        results["tfidf_lr_temporal"] = _report(
            to_matrix(new["labels"].tolist()), clf.predict(new["narrative_clean"].tolist())
        )
        results["tfidf_lr_temporal"]["split"] = (
            f"train {int(old['year'].min())}-2019 (n={len(old)}), "
            f"test 2020-{int(new['year'].max())} (n={len(new)})"
        )
    return results, models


def llm_classification_eval(
    index: ReportIndex,
    split: dict,
    llm,
    n: int = 150,
    n_examples: int = 4,
    seed: int = RANDOM_SEED,
    predictions_path: str | Path | None = None,
    baselines: dict | None = None,
) -> dict:
    """Zero-shot and few-shot (retrieved examples) LLM classification.

    Runs on a seeded sample of ``n`` test reports. Baseline models passed in
    ``baselines`` are scored on the *same* sample for a like-for-like comparison.
    """
    rep = index.reports.set_index("acn")
    rng = random.Random(seed)
    acns = sorted(rng.sample(split["test"], min(n, len(split["test"]))))
    sample = rep.loc[acns]
    texts = sample["narrative_clean"].tolist()
    y_true = to_matrix(sample["labels"].tolist())
    retriever = HybridRetriever(index)
    allowed = set(split["train"])
    results: dict = {"n": len(acns), "llm": getattr(llm, "name", "llm"), "n_examples": n_examples}
    records = []
    for setting, shots in (("zero_shot", 0), ("few_shot", n_examples)):
        clf = LLMClassifier(llm, retriever if shots else None, shots, allowed)
        preds, t0 = [], time.time()
        for acn, text in zip(acns, texts, strict=True):
            out = clf.classify(text)
            preds.append(out["labels"])
            records.append(
                {"setting": setting, "acn": acn, **out, "gold": list(rep.at[acn, "labels"])}
            )
        results[setting] = _report(y_true, to_matrix(preds))
        results[setting]["seconds_per_report"] = round((time.time() - t0) / len(acns), 2)
        results[setting]["empty_predictions"] = sum(1 for p in preds if not p)
        log.info("%s micro-F1 %.3f", setting, results[setting]["micro"]["f1"])
    for name, model in (baselines or {}).items():
        results[f"{name}_same_sample"] = _report(y_true, model.predict(texts))
    if predictions_path:
        Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(predictions_path).open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return results


# --------------------------------------------------------------- faithfulness

#: Filters used for the summary faithfulness evaluation.
SUMMARY_FILTERS: list[dict] = [
    {"phases": ["approach"], "year_from": 2020, "year_to": 2022, "labels": ["weather"]},
    {"phases": ["approach"], "year_from": 2012, "year_to": 2022},
    {"phases": ["taxi"], "labels": ["communication"]},
    {"phases": ["cruise"], "labels": ["fatigue"]},
    {"labels": ["aircraft_equipment"], "phases": ["climb"]},
    {"phases": ["parked"], "labels": ["organisational"]},
    {"labels": ["atc_airspace"], "year_from": 2018},
    {"phases": ["landing"], "labels": ["airport_environment"]},
    {"phases": ["takeoff"], "labels": ["procedure"]},
    {"labels": ["fatigue"]},
]


#: A statement citing more reports than this is counted as "broad" (citation dumping).
BROAD_CITATIONS = 5
#: Statements whose lexical support is below this are flagged for review.
LOW_SUPPORT = 0.2


def _score_summary(index: ReportIndex, spec: dict, raw_text: str, n_evidence: int) -> dict:
    """Citation and support statistics for one generated summary text.

    The evidence set is recomputed deterministically from the filters, so
    stored LLM outputs can be re-scored without calling the LLM again.
    """
    evidence = select_evidence(index, Filters(**spec), n_evidence)
    allowed = {e["acn"] for e in evidence}
    sources = {e["acn"]: f"{e['synopsis']} {e['narrative']}" for e in evidence}
    check = check_citations(raw_text, allowed, sources).to_dict()
    enforced, removed = enforce_citations(raw_text, allowed)
    return {
        "filters": spec,
        **{
            k: check[k]
            for k in (
                "n_statements",
                "n_citations",
                "n_valid_citations",
                "citation_precision",
                "statement_coverage",
                "mean_support",
                "passed",
                "invalid_citations",
                "statements",
            )
        },
        "raw_text": raw_text,
        "enforced_text": enforced,
        "removed_statements": removed,
    }


def _aggregate_faithfulness(per_summary: list[dict]) -> dict:
    statements = [st for p in per_summary for st in p["statements"]]
    tot_cit = sum(p["n_citations"] for p in per_summary)
    tot_valid = sum(p["n_valid_citations"] for p in per_summary)
    supports = [st["support"] for st in statements if st["support"] is not None]
    n = len(statements)
    return {
        "n_summaries": len(per_summary),
        "n_statements": n,
        "citation_precision": tot_valid / tot_cit if tot_cit else 0.0,
        "statement_coverage": sum(st["n_valid"] > 0 for st in statements) / n if n else 0.0,
        "summaries_fully_grounded": sum(p["passed"] for p in per_summary),
        "mean_citations_per_statement": tot_cit / n if n else 0.0,
        "broad_statement_rate": (
            sum(st["n_citations"] > BROAD_CITATIONS for st in statements) / n if n else 0.0
        ),
        "mean_lexical_support": float(np.mean(supports)) if supports else None,
        "low_support_rate": (
            sum(sp < LOW_SUPPORT for sp in supports) / len(supports) if supports else None
        ),
        "thresholds": {"broad_citations": BROAD_CITATIONS, "low_support": LOW_SUPPORT},
        "after_enforcement": "100% of displayed statements carry >=1 valid citation "
        "(ungrounded statements are removed by cite.enforce_citations)",
    }


def faithfulness_eval(
    index: ReportIndex,
    llm,
    filter_specs: list[dict] | None = None,
    n_evidence: int = 10,
) -> dict:
    """Citation precision, coverage and support of summaries *before* enforcement."""
    filter_specs = filter_specs or SUMMARY_FILTERS
    per_summary, generator = [], None
    for spec in filter_specs:
        s = summarise(index, Filters(**spec), llm=llm, n_evidence=n_evidence)
        generator = s.generator
        per_summary.append(
            {**_score_summary(index, spec, s.raw_text, n_evidence), "n_reports": s.n_reports}
        )
    return {
        "generator": generator,
        "n_evidence": n_evidence,
        **_aggregate_faithfulness(per_summary),
        "per_summary": per_summary,
    }


def rescore_faithfulness(index: ReportIndex, results: dict) -> dict:
    """Recompute faithfulness metrics from stored raw summaries (no LLM calls)."""
    n_evidence = results.get("n_evidence", 10)
    per_summary = [
        {
            **_score_summary(index, p["filters"], p["raw_text"], n_evidence),
            "n_reports": p.get("n_reports"),
        }
        for p in results["per_summary"]
    ]
    return {
        "generator": results.get("generator"),
        "n_evidence": n_evidence,
        **_aggregate_faithfulness(per_summary),
        "per_summary": per_summary,
    }


def label_prevalence(reports: pd.DataFrame) -> dict[str, float]:
    """Share of reports carrying each label."""
    y = to_matrix(reports["labels"].tolist())
    return {lab: float(y[:, j].mean()) for j, lab in enumerate(LABELS)}

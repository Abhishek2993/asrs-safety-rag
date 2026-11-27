"""Command-line interface: ``python -m asrs_rag <command>``.

Commands
--------
index      load exports, clean, chunk, embed and save the hybrid index
query      retrieve the most similar past reports for a narrative
train      fit the supervised contributing-factor classifier on the indexed reports
classify   predict contributing factors for a narrative (supervised, neighbours, LLM)
summarise  trend statistics and a cited summary for a filtered set of reports
evaluate   run an evaluation and write results/<name>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from asrs_rag import __version__, config


def _load_corpus(data: list[str] | None):
    from asrs_rag.clean import prepare_corpus
    from asrs_rag.load import load_reports

    paths = data or [str(config.default_data_dir())]
    df = load_reports(*paths)
    return prepare_corpus(df)


def _filters_from_args(args: argparse.Namespace):
    from asrs_rag.retrieve import Filters

    return Filters(
        year_from=getattr(args, "year_from", None),
        year_to=getattr(args, "year_to", None),
        phases=[args.phase] if getattr(args, "phase", None) else [],
        aircraft=getattr(args, "aircraft", None),
        primary_problems=[args.primary_problem] if getattr(args, "primary_problem", None) else [],
        labels=[args.factor] if getattr(args, "factor", None) else [],
    )


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--phase", help="flight phase substring, e.g. 'approach'")
    p.add_argument("--from", dest="year_from", type=int, help="first year (inclusive)")
    p.add_argument("--to", dest="year_to", type=int, help="last year (inclusive)")
    p.add_argument("--aircraft", help="aircraft make/model substring, e.g. 'A320'")
    p.add_argument("--primary-problem", help="ASRS primary problem, e.g. 'Weather'")
    p.add_argument("--factor", help="analysis label, e.g. 'weather' (see labels.py)")


def cmd_index(args: argparse.Namespace) -> int:
    """Build and save the index."""
    from asrs_rag.index import ChunkConfig, ReportIndex

    corpus = _load_corpus(args.data)
    print(f"Loaded {len(corpus)} reports")
    idx = ReportIndex.build(corpus, args.embedder, ChunkConfig(args.max_words, args.overlap_words))
    out = idx.save(args.index_dir)
    print(json.dumps(idx.manifest, indent=2))
    print(f"Index written to {out}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Retrieve similar reports for a narrative."""
    from asrs_rag.index import ReportIndex
    from asrs_rag.retrieve import HybridRetriever

    text = args.narrative if args.narrative != "-" else sys.stdin.read()
    idx = ReportIndex.load(args.index_dir)
    hits = HybridRetriever(idx).search(
        text, k=args.k, filters=_filters_from_args(args), mode=args.mode
    )
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], indent=2, default=str))
        return 0
    if not hits:
        print("No reports match the filters.")
    for h in hits:
        d = h.to_dict()
        print(
            f"#{d['rank']}  ACN {d['acn']}  score={d['score']:.4f}  "
            f"(dense={d['dense_score']}, bm25={d['bm25_score']})"
        )
        print(
            f"    {d['date']} | {d['flight_phase']} | {d['aircraft_type']} | "
            f"primary problem: {d['primary_problem']} | labels: {', '.join(d['labels'])}"
        )
        print(f"    {d['synopsis']}")
        print(f"    matched terms: {', '.join(d['matched_terms'][:12])}\n")
    return 0


def _llm(args: argparse.Namespace):
    from asrs_rag.llm import get_llm

    return get_llm(args.llm, args.llm_model)


def _model_path(features: str) -> Path:
    return config.MODEL_DIR / f"{features}_lr.joblib"


def cmd_train(args: argparse.Namespace) -> int:
    """Fit the supervised classifier on every indexed report and save it."""
    from asrs_rag.classify import SupervisedClassifier
    from asrs_rag.index import ReportIndex
    from asrs_rag.labels import to_matrix

    idx = ReportIndex.load(args.index_dir)
    embedder = idx.embedder if args.features == "embedding" else None
    clf = SupervisedClassifier(args.features, embedder=embedder)
    clf.fit(idx.reports["narrative_clean"].tolist(), to_matrix(idx.reports["labels"].tolist()))
    path = clf.save(_model_path(args.features))
    print(f"Trained on {len(idx.reports)} reports; saved to {path}")
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    """Predict contributing factors for one narrative."""
    from asrs_rag.classify import (
        LLMClassifier,
        NeighbourVoteClassifier,
        SupervisedClassifier,
        scores_from_proba,
    )
    from asrs_rag.clean import clean_text
    from asrs_rag.index import ReportIndex
    from asrs_rag.labels import LABELS
    from asrs_rag.retrieve import HybridRetriever

    text = clean_text(args.narrative if args.narrative != "-" else sys.stdin.read())
    idx = ReportIndex.load(args.index_dir)
    out: dict = {}
    path = _model_path("tfidf")
    if path.exists():
        clf = SupervisedClassifier.load(path)
        scores = scores_from_proba(clf.predict_proba([text]), clf.thresholds)[0]
        out["supervised"] = [s.to_dict() for s in scores]
    else:
        out["supervised"] = f"no model at {path}; run `python -m asrs_rag train`"
    knn = NeighbourVoteClassifier(HybridRetriever(idx))
    out["neighbour_vote"] = dict(zip(LABELS, knn.predict_proba([text])[0].round(3), strict=True))
    llm = _llm(args)
    if llm is not None:
        out["llm_few_shot"] = LLMClassifier(llm, HybridRetriever(idx)).classify(text)
        out["llm_few_shot"]["backend"] = llm.name
    else:
        out["llm_few_shot"] = "no LLM configured (no-LLM mode)"
    print(json.dumps(out, indent=2, default=float))
    return 0


def cmd_summarise(args: argparse.Namespace) -> int:
    """Trend statistics and cited summary."""
    from asrs_rag.index import ReportIndex
    from asrs_rag.summarise import describe_filters, summarise

    idx = ReportIndex.load(args.index_dir)
    filters = _filters_from_args(args)
    s = summarise(idx, filters, llm=_llm(args), query=args.query, n_evidence=args.evidence)
    if args.json:
        print(json.dumps(s.to_dict(), indent=2, default=str))
        return 0
    print(f"Filter: {describe_filters(filters)}  ->  {s.n_reports} reports\n")
    print("Reports per year:", ", ".join(f"{y}: {n}" for y, n in s.counts_by_year.items()))
    top = sorted(s.label_counts.items(), key=lambda t: -t[1])[:6]
    print("Top contributing factors:", ", ".join(f"{k} ({v})" for k, v in top))
    print("Top primary problems:", ", ".join(f"{k} ({v})" for k, v in s.top_primary_problems))
    print(f"\nFindings (generator: {s.generator}):\n{s.text or '(none)'}")
    c = s.citation_check
    print(
        f"\nCitation check: {c.get('n_valid_citations', 0)}/{c.get('n_citations', 0)} citations "
        f"valid, statement coverage {c.get('statement_coverage', 0):.0%}, "
        f"{len(s.removed_statements)} ungrounded statement(s) removed."
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run one evaluation and write its JSON to results/."""
    from asrs_rag import eval as ev
    from asrs_rag.index import ReportIndex

    idx = ReportIndex.load(args.index_dir)
    out_dir = Path(args.out)
    split_path = out_dir / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
    else:
        split = ev.make_split(idx.reports["acn"].tolist())
        ev.save_json(split, split_path)

    if args.what == "retrieval":
        res = ev.retrieval_proxy_eval(idx, n_queries=args.n)
    elif args.what == "labelled":
        res = ev.retrieval_labelled_eval(idx, ev.load_queries(args.queries))
    elif args.what == "classification":
        res, models = ev.classification_eval(idx, split)
        for name in ("tfidf_lr", "embedding_lr"):
            if name in models:
                models[name].save(config.MODEL_DIR / f"{name}_eval.joblib")
    elif args.what == "llm":
        from asrs_rag.classify import SupervisedClassifier

        llm = _llm(args)
        if llm is None:
            print("No LLM available; set --llm (e.g. ollama, transformers, anthropic).")
            return 1
        baselines = {}
        for name in ("tfidf_lr", "embedding_lr"):
            p = config.MODEL_DIR / f"{name}_eval.joblib"
            if p.exists():
                m = SupervisedClassifier.load(p)
                if m.features == "embedding":
                    m._embedder = idx.embedder
                baselines[name] = m
        res = ev.llm_classification_eval(
            idx,
            split,
            llm,
            n=args.n,
            predictions_path=out_dir / "llm_predictions.jsonl",
            baselines=baselines,
        )
    else:  # faithfulness
        llm = _llm(args)
        res = ev.faithfulness_eval(idx, llm)
    res["index"] = idx.manifest
    path = ev.save_json(res, out_dir / f"{args.what}.json")
    print(f"Wrote {path}")
    summary = {
        k: v
        for k, v in res.items()
        if k in ("bm25", "dense", "hybrid", "chance", "citation_precision", "statement_coverage")
    }
    for key in (
        "tfidf_lr",
        "embedding_lr",
        "neighbour_vote",
        "zero_shot",
        "few_shot",
        "tfidf_lr_temporal",
    ):
        if key in res:
            summary[key] = {
                "micro_f1": res[key]["micro"]["f1"],
                "macro_f1": res[key]["macro"]["f1"],
            }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="asrs_rag", description="Retrieval-augmented analysis of NASA ASRS reports."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    parser.add_argument(
        "--llm",
        default=None,
        help="LLM backend: auto (default), ollama, transformers, openai, anthropic, none",
    )
    parser.add_argument("--llm-model", default=None, help="override the backend's model")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("index", help="build the hybrid index")
    p.add_argument("--data", nargs="+", help="CSV files/dirs (default: data/raw or sample)")
    p.add_argument(
        "--embedder",
        default=config.DEFAULT_EMBEDDING_MODEL,
        help="sentence-transformers model id or 'hashing' (no download)",
    )
    p.add_argument("--max-words", type=int, default=160)
    p.add_argument("--overlap-words", type=int, default=40)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("query", help="find similar past reports")
    p.add_argument("narrative", help="incident narrative ('-' reads stdin)")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    p.add_argument("--json", action="store_true")
    _add_filter_args(p)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("train", help="train the supervised classifier on indexed reports")
    p.add_argument("--features", choices=["tfidf", "embedding"], default="tfidf")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("classify", help="predict contributing factors for a narrative")
    p.add_argument("narrative", help="incident narrative ('-' reads stdin)")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("summarise", aliases=["summarize"], help="cited trend summary")
    _add_filter_args(p)
    p.add_argument("--query", help="focus the evidence on a theme (hybrid retrieval)")
    p.add_argument("--evidence", type=int, default=12, help="reports the summary may cite")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_summarise)

    p = sub.add_parser("evaluate", help="run an evaluation (writes results/*.json)")
    p.add_argument(
        "what", choices=["retrieval", "labelled", "classification", "llm", "faithfulness"]
    )
    p.add_argument("--n", type=int, default=300, help="queries (retrieval) / reports (llm)")
    p.add_argument("--queries", default=str(config.EVAL_DIR / "queries.jsonl"))
    p.add_argument("--out", default=str(config.RESULTS_DIR))
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m asrs_rag``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    # Keep third-party model-loading chatter out of CLI output.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)

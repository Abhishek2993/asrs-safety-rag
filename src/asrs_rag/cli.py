"""Command-line interface: ``python -m asrs_rag <command>``.

Commands
--------
index      load exports, clean, chunk, embed and save the hybrid index
query      retrieve the most similar past reports for a narrative
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


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="asrs_rag", description="Retrieval-augmented analysis of NASA ASRS reports."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
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

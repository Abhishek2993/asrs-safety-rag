"""Command-line interface (sub-commands are added as the pipeline grows)."""

from __future__ import annotations

import argparse

from asrs_rag import __version__


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="asrs_rag", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_subparsers(dest="command")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m asrs_rag``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0

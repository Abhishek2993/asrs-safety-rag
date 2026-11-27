"""Download ASRS reports from a public mirror and write them in ASRS export format.

The canonical source is ASRS Database Online (see README, "Getting the data").
Its query wizard cannot be scripted reliably, so for reproducible experiments
this script reads the public Hugging Face mirror ``elihoole/asrs-aviation-reports``
(ASRS Database Online records, 47,723 reports, flattened two-row header) and
writes them back as standard two-row-header CSV files, i.e. exactly the
format ``asrs_rag.load`` parses for your own exports.

Usage::

    python scripts/fetch_asrs_mirror.py --splits validation test --out data/raw
    python scripts/fetch_asrs_mirror.py --splits validation --sample 200 \
        --sample-out data/sample/asrs_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path

import requests

MIRROR = "https://huggingface.co/datasets/elihoole/asrs-aviation-reports/resolve/main"
SPLITS = {
    "train": "asrs-aviation-reports-train.jsonl",
    "validation": "asrs-aviation-reports-validation.jsonl",
    "test": "asrs-aviation-reports-test.jsonl",
}
MAX_ROWS_PER_FILE = 5000  # ASRS DBOL caps exports at 10,000; smaller files mimic batches
_SUFFIX = re.compile(r"\.\d+$")


def download(split: str, cache: Path) -> Path:
    """Download one split (cached)."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / SPLITS[split]
    if not path.exists():
        with requests.get(f"{MIRROR}/{SPLITS[split]}", stream=True, timeout=120) as r:
            r.raise_for_status()
            with path.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    return path


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, str]]:
    """Read records from JSONL files."""
    rows: list[dict[str, str]] = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def header_rows(keys: list[str]) -> tuple[list[str], list[str]]:
    """Rebuild the ASRS (group, field) header rows from flattened mirror keys."""
    groups, fields = [], []
    for key in keys:
        group, field = key.rsplit("_", 1)
        if field == "ACN":
            groups.append("")
        else:
            groups.append(_SUFFIX.sub("", group))
        fields.append(field)
    return groups, fields


def write_export(rows: list[dict[str, str]], path: Path) -> None:
    """Write records as an ASRS-style CSV (two header rows, trailing empty column)."""
    keys = list(rows[0].keys())
    groups, fields = header_rows(keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([*groups, ""])
        w.writerow([*fields, ""])
        for r in rows:
            w.writerow([*(r.get(k, "") or "" for k in keys), ""])


def stratified_sample(rows: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    """Pick ``n`` reports covering every analysis label, favouring readable lengths."""
    from asrs_rag.labels import LABELS, labels_for_report
    from asrs_rag.load import split_multi

    rng = random.Random(seed)
    pool = [
        r
        for r in rows
        if 60 <= len(r["Report 1_Narrative"].split()) <= 450 and r["Report 1.2_Synopsis"]
    ]
    rng.shuffle(pool)

    def labs(r: dict[str, str]) -> list[str]:
        hf = split_multi(r.get("Person 1.7_Human Factors", "")) + split_multi(
            r.get("Person 2.7_Human Factors", "")
        )
        return labels_for_report(
            split_multi(r["Assessments_Contributing Factors / Situations"]), hf
        )

    per_label = max(1, n // (2 * len(LABELS)))
    chosen: dict[str, dict[str, str]] = {}
    counts = {lab: 0 for lab in LABELS}
    for lab in sorted(LABELS, key=lambda x: sum(x in labs(r) for r in pool)):
        for r in pool:
            if counts[lab] >= per_label or len(chosen) >= n:
                break
            if lab in labs(r) and r["acn_num_ACN"] not in chosen:
                chosen[r["acn_num_ACN"]] = r
                for lab2 in labs(r):
                    counts[lab2] += 1
    for r in pool:
        if len(chosen) >= n:
            break
        chosen.setdefault(r["acn_num_ACN"], r)
    return sorted(chosen.values(), key=lambda r: (r["Time_Date"], r["acn_num_ACN"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", nargs="+", default=["validation", "test"], choices=SPLITS)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--out", type=Path, help="directory for full exports (e.g. data/raw)")
    parser.add_argument("--sample", type=int, default=0, help="also write a stratified sample")
    parser.add_argument("--sample-out", type=Path, default=Path("data/sample/asrs_sample.csv"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = read_jsonl(download(s, args.cache) for s in args.splits)
    print(f"Read {len(rows)} reports from splits {args.splits}")
    if args.out:
        for i in range(0, len(rows), MAX_ROWS_PER_FILE):
            path = args.out / f"asrs_export_{i // MAX_ROWS_PER_FILE + 1:02d}.csv"
            write_export(rows[i : i + MAX_ROWS_PER_FILE], path)
            print(f"Wrote {path}")
    if args.sample:
        sample = stratified_sample(rows, args.sample, args.seed)
        write_export(sample, args.sample_out)
        print(f"Wrote {len(sample)} reports to {args.sample_out}")


if __name__ == "__main__":
    main()

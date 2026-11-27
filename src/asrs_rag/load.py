"""Parse NASA ASRS Database Online CSV exports into a tidy DataFrame.

The ASRS export format
----------------------
ASRS Database Online exports a CSV with a *two-row header*:

* row 1 holds the **group** (``Time``, ``Place``, ``Aircraft 1``, ``Person 1``,
  ``Events``, ``Assessments``, ``Report 1`` ...). The group label is repeated
  for every column of the group in current exports; some tools leave all but
  the first cell blank, so blanks are forward-filled.
* row 2 holds the **field** (``Date``, ``Flight Phase``, ``Narrative`` ...).
  The first column is the report ID, ``ACN`` (ASRS accession number), whose
  group cell is empty.

Multi-valued cells (e.g. contributing factors) are separated by ``;``. Exports
are capped at 10,000 records, so a corpus is usually several files with
possible overlaps; :func:`load_reports` deduplicates by ACN.

The loader also accepts a *flattened* single-row header in which group and
field are joined by an underscore (``Aircraft 1.9_Flight Phase``), as produced
by ``pandas.read_csv`` on an export and found in public mirrors.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

#: (group, field) -> canonical column name. Groups are compared after removing
#: pandas' de-duplication suffixes (``Aircraft 1.9`` -> ``aircraft 1``).
FIELD_MAP: dict[tuple[str, str], str] = {
    ("time", "date"): "date_raw",
    ("time", "local time of day"): "local_time",
    ("place", "locale reference"): "locale",
    ("place", "state reference"): "state",
    ("environment", "flight conditions"): "flight_conditions",
    ("environment", "weather elements / visibility"): "weather_elements",
    ("environment", "light"): "light",
    ("aircraft 1", "aircraft operator"): "aircraft_operator",
    ("aircraft 1", "make model name"): "aircraft_type",
    ("aircraft 1", "operating under far part"): "far_part",
    ("aircraft 1", "flight plan"): "flight_plan",
    ("aircraft 1", "flight phase"): "flight_phase",
    ("aircraft 1", "airspace"): "airspace",
    ("component", "aircraft component"): "component",
    ("component", "problem"): "component_problem",
    ("person 1", "reporter organization"): "reporter_organization",
    ("person 1", "function"): "reporter_function",
    ("person 1", "human factors"): "human_factors_1",
    ("person 2", "human factors"): "human_factors_2",
    ("person 1", "communication breakdown"): "communication_breakdown",
    ("events", "anomaly"): "anomaly",
    ("events", "detector"): "detector",
    ("events", "when detected"): "when_detected",
    ("events", "result"): "result",
    ("assessments", "contributing factors / situations"): "contributing_factors",
    ("assessments", "primary problem"): "primary_problem",
    ("report 1", "narrative"): "narrative_1",
    ("report 1", "callback"): "callback_1",
    ("report 2", "narrative"): "narrative_2",
    ("report 2", "callback"): "callback_2",
    ("report 1", "synopsis"): "synopsis",
}

#: Canonical columns holding ``;``-separated lists.
LIST_COLUMNS = (
    "flight_phase",
    "anomaly",
    "contributing_factors",
    "human_factors",
    "weather_elements",
    "result",
)

#: Columns present in every DataFrame returned by :func:`load_reports`.
OUTPUT_COLUMNS = [
    "acn",
    "date",
    "year",
    "month",
    "local_time",
    "locale",
    "state",
    "flight_conditions",
    "weather_elements",
    "light",
    "aircraft_operator",
    "aircraft_type",
    "far_part",
    "flight_plan",
    "flight_phase",
    "airspace",
    "component",
    "component_problem",
    "reporter_organization",
    "reporter_function",
    "human_factors",
    "communication_breakdown",
    "anomaly",
    "detector",
    "when_detected",
    "result",
    "contributing_factors",
    "primary_problem",
    "synopsis",
    "narrative",
    "callback",
    "source_file",
]

_SUFFIX_RE = re.compile(r"\.\d+$")
_ACN_RE = re.compile(r"^\d{4,8}$")


class ASRSFormatError(ValueError):
    """Raised when a file does not look like an ASRS export."""


def _norm(text: str) -> str:
    """Lower-case, strip and collapse whitespace in a header cell."""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _read_text(path: Path) -> str:
    """Read a file trying UTF-8 (with BOM) first, then Windows-1252."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ASRSFormatError(f"Cannot decode {path}")  # pragma: no cover - latin-1 never fails


def _split_flat_header(cell: str) -> tuple[str, str]:
    """Split a flattened ``group_field`` header (``Aircraft 1.9_Flight Phase``)."""
    if "_" not in cell:
        return "", cell
    group, field = cell.rsplit("_", 1)
    return group, field


def _canonical_name(group: str, field: str) -> str | None:
    """Map an export (group, field) pair to a canonical column name."""
    field_n = _norm(field)
    if field_n == "acn":
        return "acn"
    group_n = _SUFFIX_RE.sub("", _norm(group))
    return FIELD_MAP.get((group_n, field_n))


def _locate_header(rows: list[list[str]]) -> tuple[int, bool]:
    """Find the field-name header row.

    Returns ``(index, flattened)`` where ``flattened`` is True for single-row
    ``group_field`` headers.
    """
    for i, row in enumerate(rows[:20]):
        cells = [_norm(c) for c in row]
        if "acn" in cells:
            return i, False
        if any(c == "acn" or c.endswith("_acn") for c in cells):
            return i, True
    raise ASRSFormatError("No 'ACN' column found in the first 20 rows")


def parse_export(source: str | Path | io.StringIO) -> pd.DataFrame:
    """Parse one ASRS export (path or text buffer) into canonical columns.

    Rows without a numeric ACN (blank lines, footers) are dropped. Columns not
    in :data:`FIELD_MAP` are ignored.
    """
    if isinstance(source, io.StringIO):
        text, name = source.getvalue(), "<buffer>"
    else:
        path = Path(source)
        text, name = _read_text(path), path.name
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ASRSFormatError(f"{name}: empty file")

    header_idx, flattened = _locate_header(rows)
    header = rows[header_idx]
    if flattened:
        pairs = [_split_flat_header(c) for c in header]
    else:
        groups = rows[header_idx - 1] if header_idx > 0 else [""] * len(header)
        groups = list(groups) + [""] * (len(header) - len(groups))
        filled, current = [], ""
        for i, (g, f) in enumerate(zip(groups, header, strict=False)):
            if g.strip():
                current = g.strip()
            # The ACN column has no group; do not inherit one from the left.
            filled.append("" if (_norm(f) == "acn" or (i == 0 and not g.strip())) else current)
        pairs = list(zip(filled, header, strict=False))

    columns: dict[int, str] = {}
    for idx, (group, field) in enumerate(pairs):
        canon = _canonical_name(group, field)
        if canon and canon not in columns.values():
            columns[idx] = canon
    if "acn" not in columns.values():
        raise ASRSFormatError(f"{name}: ACN column could not be mapped")

    records = []
    for row in rows[header_idx + 1 :]:
        rec = {canon: (row[i].strip() if i < len(row) else "") for i, canon in columns.items()}
        acn = rec.get("acn", "").split(".")[0]
        if not _ACN_RE.match(acn):
            continue  # blank line, footer or repeated header
        rec["acn"] = acn
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    df["source_file"] = name
    log.info("Parsed %d reports from %s", len(df), name)
    return df


def split_multi(value: object) -> list[str]:
    """Split an ASRS ``;``-separated cell into a list of trimmed values."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [v.strip() for v in value.split(";") if v.strip()]


def _parse_date(value: str) -> pd.Timestamp | None:
    """ASRS dates are ``YYYYMM``; return the first day of that month."""
    digits = re.sub(r"\D", "", value or "")
    if len(digits) >= 6:
        try:
            return pd.Timestamp(year=int(digits[:4]), month=int(digits[4:6]), day=1)
        except ValueError:
            return None
    return None


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw parsed columns into the canonical schema.

    * ``date`` is a timestamp (first of month), plus ``year``/``month`` ints;
    * narratives and callbacks from both reporters are concatenated;
    * human factors from both reporters are merged into one list;
    * multi-valued fields become Python lists.
    """
    out = df.copy()
    for col in FIELD_MAP.values():
        if col not in out.columns:
            out[col] = ""
    out = out.fillna("")

    dates = out["date_raw"].map(_parse_date)
    out["date"] = pd.to_datetime(dates)
    out["year"] = out["date"].dt.year.astype("Int64")
    out["month"] = out["date"].dt.month.astype("Int64")

    def join(a: str, b: str) -> str:
        return "\n\n".join(t for t in (a.strip(), b.strip()) if t)

    out["narrative"] = [
        join(a, b) for a, b in zip(out["narrative_1"], out["narrative_2"], strict=False)
    ]
    out["callback"] = [
        join(a, b) for a, b in zip(out["callback_1"], out["callback_2"], strict=False)
    ]
    out["human_factors"] = [
        ";".join(dict.fromkeys(split_multi(a) + split_multi(b)))
        for a, b in zip(out["human_factors_1"], out["human_factors_2"], strict=False)
    ]
    for col in LIST_COLUMNS:
        out[col] = out[col].map(split_multi)
    return out[OUTPUT_COLUMNS]


def _completeness(row: pd.Series) -> int:
    """Score used to keep the richest copy of a duplicated ACN."""
    filled = sum(bool(v) if not isinstance(v, list) else len(v) > 0 for v in row.values)
    return filled * 10_000 + len(row["narrative"])


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per ACN, preferring the most complete record."""
    if df.empty:
        return df
    scored = df.assign(_score=df.apply(_completeness, axis=1))
    scored = scored.sort_values(["acn", "_score"], ascending=[True, False])
    deduped = scored.drop_duplicates("acn", keep="first").drop(columns="_score")
    removed = len(df) - len(deduped)
    if removed:
        log.info("Removed %d duplicate reports (by ACN)", removed)
    return deduped.reset_index(drop=True)


def iter_export_files(paths: Iterable[str | Path]) -> list[Path]:
    """Expand directories into the CSV files they contain (sorted)."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(p.glob("*.csv")))
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    if not files:
        raise FileNotFoundError(f"No ASRS CSV exports found in {list(paths)}")
    return files


def load_reports(*paths: str | Path) -> pd.DataFrame:
    """Load, normalise and deduplicate ASRS exports from files or directories.

    Reports without a narrative are dropped because every downstream step
    (retrieval, classification, summarisation) is text based.
    """
    frames = [parse_export(f) for f in iter_export_files(paths)]
    df = normalise(pd.concat(frames, ignore_index=True))
    df = deduplicate(df)
    empty = df["narrative"].str.strip() == ""
    if empty.any():
        log.info("Dropped %d reports without a narrative", int(empty.sum()))
    df = df[~empty].sort_values(["date", "acn"], na_position="last").reset_index(drop=True)
    return df

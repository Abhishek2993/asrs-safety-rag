"""Text cleaning and document construction for ASRS reports."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from asrs_rag.abbreviations import ACRONYMS, SHORTHAND
from asrs_rag.labels import add_labels

_SHORTHAND_RE = re.compile(r"\b(" + "|".join(sorted(SHORTHAND, key=len, reverse=True)) + r")\b")
_ACRONYM_RE = re.compile(r"\b(" + "|".join(sorted(ACRONYMS, key=len, reverse=True)) + r")\b")
_FLIGHT_LEVEL_RE = re.compile(r"\bFL ?(\d{2,3})\b")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})

#: ASRS de-identification placeholders (``ZZZ``, ``ZZZ1``, ``XXR``, ``XYZ`` ...).
DEIDENT_RE = re.compile(r"^(?:Z{3,}\d*|X{1,4}[LRC]?\d*|XY\w?|YYY+\d*)$", re.IGNORECASE)


def normalise_whitespace(text: str) -> str:
    """Normalise unicode, quotes and whitespace while keeping paragraph breaks."""
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTES)
    paragraphs = [re.sub(r"[ \t\r\f\v]+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paragraphs = [re.sub(r"\s*\n\s*", " ", p) for p in paragraphs if p]
    return "\n\n".join(paragraphs)


def expand_abbreviations(text: str) -> str:
    """Expand ASRS abbreviations (see :mod:`asrs_rag.abbreviations`).

    * legacy shorthand is replaced everywhere (``ACFT`` -> ``aircraft``);
    * operational acronyms keep their form and gain an expansion at the first
      occurrence (``TCAS`` -> ``TCAS (traffic collision avoidance system)``);
    * flight levels are spelled out (``FL350`` -> ``flight level 350``).
    """
    text = _SHORTHAND_RE.sub(lambda m: SHORTHAND[m.group(1)], text)
    text = _FLIGHT_LEVEL_RE.sub(r"flight level \1", text)
    seen: set[str] = set()

    def _acronym(m: re.Match[str]) -> str:
        acr = m.group(1)
        if acr in seen:
            return acr
        seen.add(acr)
        # Do not add the expansion if the author already wrote it out.
        following = text[m.end() : m.end() + len(ACRONYMS[acr]) + 3].lower()
        if ACRONYMS[acr] in following:
            return acr
        return f"{acr} ({ACRONYMS[acr]})"

    return _ACRONYM_RE.sub(_acronym, text)


def clean_text(text: str) -> str:
    """Full cleaning pipeline used for indexing and classification."""
    return expand_abbreviations(normalise_whitespace(text))


def metadata_line(row: pd.Series | dict) -> str:
    """Short, label-free metadata header for a report document.

    Only fields a reporter would know at submission time are included. Coded
    analyst fields (primary problem, contributing factors, anomaly) are
    deliberately excluded so they cannot leak into retrieval or classification.
    """
    phase = "; ".join(row.get("flight_phase") or []) or "unknown"
    parts = [
        f"Flight phase: {phase}",
        f"Aircraft: {row.get('aircraft_type') or 'unknown'}",
        f"Operator: {row.get('aircraft_operator') or 'unknown'}",
    ]
    return " | ".join(parts)


def build_document(row: pd.Series | dict) -> str:
    """One retrieval document per report: metadata + synopsis + narrative."""
    synopsis = clean_text(row.get("synopsis", ""))
    narrative = clean_text(row.get("narrative", ""))
    parts = [metadata_line(row)]
    if synopsis:
        parts.append(f"Synopsis: {synopsis}")
    parts.append(f"Narrative: {narrative}")
    return "\n".join(parts)


def prepare_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned text, retrieval documents and analysis labels to loaded reports.

    Adds the columns ``narrative_clean``, ``synopsis_clean``, ``document`` and
    ``labels``.
    """
    out = add_labels(df)
    out["narrative_clean"] = out["narrative"].map(clean_text)
    out["synopsis_clean"] = out["synopsis"].map(clean_text)
    out["document"] = [build_document(r) for r in out.to_dict("records")]
    return out

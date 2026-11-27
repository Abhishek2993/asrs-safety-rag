"""Citation checking for generated text.

Requirement FR-6 says every generated statement must cite ASRS report IDs
(ACNs) and every cited ACN must be one of the reports actually given to the
generator. This module enforces that in three steps:

1. :func:`split_statements` - break generated text into statements
   (bullets, or sentences when there are no bullets);
2. :func:`check_citations` - extract ``[ACN 1234567]`` citations per statement
   and verify each against the allowed (retrieved) set; optionally measure
   lexical support of the statement by the cited reports' text;
3. :func:`enforce_citations` - drop statements with no valid citation and
   strip invalid citation tokens, so only grounded statements reach the user.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from asrs_rag.bm25 import tokenize

#: ``[ACN 1234567]``, ``ACN 1234567``, ``ACN#1234567``, ``[ACN 1234567, 7654321]``.
_ACN_BLOCK_RE = re.compile(r"\[?\s*ACNs?\s*[#:]?\s*((?:\d{5,8}\s*(?:,|;|and)?\s*)+)\]?", re.I)
_NUM_RE = re.compile(r"\d{5,8}")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def extract_acns(text: str) -> list[str]:
    """All ACNs cited in ``text`` (in order, de-duplicated)."""
    found: list[str] = []
    for block in _ACN_BLOCK_RE.findall(text or ""):
        for acn in _NUM_RE.findall(block):
            if acn not in found:
                found.append(acn)
    return found


def split_statements(text: str) -> list[str]:
    """Split generated text into statements.

    Bulleted/numbered lines are statements; otherwise sentences are used.
    Headings (lines ending with ':') and blank lines are ignored.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if any(_BULLET_RE.match(ln) for ln in lines):
        statements, current = [], ""
        for ln in lines:
            if _BULLET_RE.match(ln):
                if current:
                    statements.append(current)
                current = _BULLET_RE.sub("", ln)
            elif current and not ln.endswith(":"):
                current += " " + ln
        if current:
            statements.append(current)
        return statements
    prose = " ".join(ln for ln in lines if not ln.endswith(":"))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", prose) if s.strip()]


@dataclass
class StatementCheck:
    """Citation check result for one statement."""

    text: str
    citations: list[str]
    valid: list[str]
    invalid: list[str]
    support: float | None = None

    @property
    def grounded(self) -> bool:
        """At least one citation and no invalid ones."""
        return bool(self.valid) and not self.invalid


@dataclass
class CitationReport:
    """Aggregate citation statistics for a generated text."""

    statements: list[StatementCheck] = field(default_factory=list)

    @property
    def n_statements(self) -> int:
        return len(self.statements)

    @property
    def n_citations(self) -> int:
        return sum(len(s.citations) for s in self.statements)

    @property
    def n_valid(self) -> int:
        return sum(len(s.valid) for s in self.statements)

    @property
    def citation_precision(self) -> float:
        """Share of citations that point to a retrieved report (faithfulness rate)."""
        return self.n_valid / self.n_citations if self.n_citations else 0.0

    @property
    def statement_coverage(self) -> float:
        """Share of statements with at least one valid citation."""
        if not self.statements:
            return 0.0
        return sum(bool(s.valid) for s in self.statements) / self.n_statements

    @property
    def passed(self) -> bool:
        """True when every statement is grounded."""
        return bool(self.statements) and all(s.grounded for s in self.statements)

    def to_dict(self) -> dict:
        """JSON-friendly summary."""
        supports = [s.support for s in self.statements if s.support is not None]
        return {
            "n_statements": self.n_statements,
            "n_citations": self.n_citations,
            "n_valid_citations": self.n_valid,
            "citation_precision": round(self.citation_precision, 4),
            "statement_coverage": round(self.statement_coverage, 4),
            "mean_support": round(sum(supports) / len(supports), 4) if supports else None,
            "passed": self.passed,
            "invalid_citations": sorted({a for s in self.statements for a in s.invalid}),
            "uncited_statements": [s.text for s in self.statements if not s.citations],
            "statements": [
                {
                    "text": s.text,
                    "n_citations": len(s.citations),
                    "n_valid": len(s.valid),
                    "support": None if s.support is None else round(s.support, 4),
                }
                for s in self.statements
            ],
        }


def support_score(statement: str, sources: Iterable[str]) -> float:
    """Share of the statement's content words found in the cited reports' text.

    A cheap lexical proxy for "is this claim actually in the cited report?".
    Low values flag statements for human review; they do not prove an error.
    """
    words = {w for w in tokenize(_ACN_BLOCK_RE.sub(" ", statement)) if not w.isdigit()}
    if not words:
        return 0.0
    source_words = set()
    for s in sources:
        source_words |= set(tokenize(s))
    return len(words & source_words) / len(words)


def check_citations(
    text: str, allowed: Iterable[str], sources: Mapping[str, str] | None = None
) -> CitationReport:
    """Check every statement's citations against the allowed ACN set.

    ``sources`` maps ACN -> report text and enables :func:`support_score`.
    """
    allowed = set(allowed)
    report = CitationReport()
    for st in split_statements(text):
        cites = extract_acns(st)
        valid = [a for a in cites if a in allowed]
        support = None
        if sources is not None and valid:
            support = support_score(st, (sources[a] for a in valid if a in sources))
        report.statements.append(
            StatementCheck(
                text=st,
                citations=cites,
                valid=valid,
                invalid=[a for a in cites if a not in allowed],
                support=support,
            )
        )
    return report


def enforce_citations(text: str, allowed: Iterable[str]) -> tuple[str, list[str]]:
    """Keep only statements with a valid citation; remove invalid ACN references.

    Grounded statements are kept verbatim (citations stay where the author put
    them). In a statement that mixes valid and invalid citations, only the
    invalid ACNs are deleted from its citation blocks. Statements with no valid
    citation are dropped and returned in ``removed_statements``.

    Returns ``(clean_text, removed_statements)``; the clean text is a bullet list.
    """
    allowed = set(allowed)

    def _filter_block(m: re.Match[str]) -> str:
        acns = _NUM_RE.findall(m.group(0))
        valid = [a for a in acns if a in allowed]
        if len(valid) == len(acns):
            return m.group(0)
        if not valid:
            return ""
        return "[ACN " + ", ".join(valid) + "]"

    kept, removed = [], []
    for st in split_statements(text):
        if not any(a in allowed for a in extract_acns(st)):
            removed.append(st)
            continue
        cleaned = _ACN_BLOCK_RE.sub(_filter_block, st)
        cleaned = re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s{2,}", " ", cleaned)).strip()
        kept.append(f"- {cleaned}")
    return "\n".join(kept), removed

import pytest

from asrs_rag.cite import (
    check_citations,
    enforce_citations,
    extract_acns,
    split_statements,
    support_score,
)

ALLOWED = {"1111111", "2222222", "3333333"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Crews reported icing [ACN 1111111].", ["1111111"]),
        ("See [ACN 1111111, 2222222] and ACN#3333333.", ["1111111", "2222222", "3333333"]),
        ("ACNs 1111111 and 2222222 show this", ["1111111", "2222222"]),
        ("Climbed to 10000 feet with no citation", []),
    ],
)
def test_extract_acns(text, expected):
    assert extract_acns(text) == expected


def test_split_statements_bullets_and_prose():
    bullets = "Summary:\n- First point [ACN 1111111]\n  continues here.\n* Second [ACN 2222222]"
    assert split_statements(bullets) == [
        "First point [ACN 1111111] continues here.",
        "Second [ACN 2222222]",
    ]
    prose = "One claim [ACN 1111111]. Another claim without cite."
    assert split_statements(prose) == ["One claim [ACN 1111111].", "Another claim without cite."]


def test_faithful_summary_passes():
    text = "- Icing on approach recurred [ACN 1111111, 2222222]\n- Fatigue noted [ACN 3333333]"
    rep = check_citations(text, ALLOWED)
    assert rep.passed
    assert rep.citation_precision == 1.0 and rep.statement_coverage == 1.0


def test_wrong_and_missing_citations_are_detected():
    text = (
        "- Real pattern [ACN 1111111]\n"
        "- Hallucinated report [ACN 9999999]\n"
        "- Mixed [ACN 2222222, 8888888]\n"
        "- Uncited claim about engines"
    )
    rep = check_citations(text, ALLOWED)
    assert not rep.passed
    assert rep.n_citations == 4 and rep.n_valid == 2
    assert rep.citation_precision == pytest.approx(0.5)
    assert rep.statement_coverage == pytest.approx(2 / 4)
    d = rep.to_dict()
    assert d["invalid_citations"] == ["8888888", "9999999"]
    assert d["uncited_statements"] == ["Uncited claim about engines"]


def test_enforce_citations_removes_ungrounded_statements():
    text = "- Real [ACN 1111111]\n- Fake [ACN 9999999]\n- Mixed [ACN 2222222, 8888888]\n- None"
    clean, removed = enforce_citations(text, ALLOWED)
    assert clean == "- Real [ACN 1111111]\n- Mixed [ACN 2222222]"
    assert removed == ["Fake [ACN 9999999]", "None"]
    assert check_citations(clean, ALLOWED).passed


def test_support_score():
    src = ["The crew encountered severe icing during the approach."]
    assert support_score("Severe icing during approach [ACN 1111111]", src) == 1.0
    assert support_score("Engine fire on takeoff [ACN 1111111]", src) == 0.0


def test_check_with_sources_reports_support():
    rep = check_citations(
        "- Icing on approach [ACN 1111111]", ALLOWED, {"1111111": "icing approach"}
    )
    assert rep.statements[0].support == 1.0
    assert rep.to_dict()["mean_support"] == 1.0


def test_empty_text_fails_check():
    assert not check_citations("", ALLOWED).passed


def test_enforce_keeps_grounded_statements_verbatim():
    text = (
        "- For instance, [ACN 1111111] and [ACN 2222222] show icing; see also [ACN 9999999].\n"
        "- Inline cite ACN 3333333 stays put."
    )
    clean, removed = enforce_citations(text, ALLOWED)
    assert removed == []
    assert clean == (
        "- For instance, [ACN 1111111] and [ACN 2222222] show icing; see also.\n"
        "- Inline cite ACN 3333333 stays put."
    )


def test_report_lists_per_statement_details():
    rep = check_citations("- A [ACN 1111111, 2222222]\n- B", ALLOWED).to_dict()
    assert [s["n_citations"] for s in rep["statements"]] == [2, 0]

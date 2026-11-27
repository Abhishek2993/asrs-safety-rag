from pathlib import Path

import pytest

from asrs_rag.clean import (
    DEIDENT_RE,
    build_document,
    clean_text,
    expand_abbreviations,
    normalise_whitespace,
    prepare_corpus,
)
from asrs_rag.labels import LABELS, from_matrix, labels_for_report, to_matrix
from asrs_rag.load import load_reports

FIXTURE = Path(__file__).parent / "fixtures" / "asrs_export_fixture.csv"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ACFT WAS CLRD TO DSND", "aircraft WAS cleared TO descend"),
        ("THE CLRNC ALT WAS 5000 FT", "THE clearance altitude WAS 5000 feet"),
        ("Climbing through FL350 we", "Climbing through flight level 350 we"),
        ("RWY 27L and TWY B", "runway 27L and taxiway B"),
    ],
)
def test_shorthand_is_replaced(raw, expected):
    assert expand_abbreviations(raw) == expected


def test_acronym_is_expanded_once_and_kept():
    out = expand_abbreviations("ATC issued a turn. ATC then cleared us. TCAS RA followed.")
    assert out.count("air traffic control") == 1
    assert out.count("ATC") == 2
    assert "TCAS (traffic collision avoidance system) RA (resolution advisory)" in out


def test_existing_expansion_is_not_duplicated():
    out = expand_abbreviations("The QRH (quick reference handbook) was used.")
    assert out.count("quick reference handbook") == 1


def test_lowercase_and_embedded_tokens_are_untouched():
    text = "Set alt hold; the Star route; ALTITUDE; RAMP; MELT"
    assert expand_abbreviations(text) == text


def test_normalise_whitespace_keeps_paragraphs_and_fixes_quotes():
    text = "Line one\n  continues   here.\n\n“Quoted”  part"
    assert normalise_whitespace(text) == 'Line one continues here.\n\n"Quoted" part'


def test_deidentification_tokens():
    for tok in ("ZZZ", "ZZZ1", "ZZZZZ", "XXR", "XX", "XXXX"):
        assert DEIDENT_RE.match(tok)
    for tok in ("ZONE", "TAXI", "B737"):
        assert not DEIDENT_RE.match(tok)


def test_document_contains_metadata_synopsis_narrative_but_no_labels():
    df = prepare_corpus(load_reports(FIXTURE)).set_index("acn")
    doc = df.loc["1000001", "document"]
    assert doc.startswith("Flight phase: Final Approach | Aircraft: B737-800")
    assert "Synopsis: B737 crew" in doc
    assert "aircraft leveled at 2500 feet" in doc.lower()
    assert "Primary" not in doc and "Contributing" not in doc
    assert df.loc["1000001", "labels"] == ["human_factors", "fatigue", "weather"]


def test_build_document_without_synopsis():
    doc = build_document({"narrative": "ACFT landed.", "flight_phase": ["Landing"]})
    assert "Synopsis" not in doc and "Narrative: aircraft landed." in doc


def test_label_mapping_is_exact_match():
    assert labels_for_report(["Environment - Non Weather Related"], []) == ["airport_environment"]
    assert labels_for_report(["Weather", "MEL"], ["Fatigue"]) == [
        "fatigue",
        "weather",
        "aircraft_equipment",
    ]
    assert labels_for_report([], []) == []


def test_label_matrix_roundtrip():
    lists = [["weather", "fatigue"], [], ["procedure"]]
    y = to_matrix(lists)
    assert y.shape == (3, len(LABELS))
    assert from_matrix(y) == [["fatigue", "weather"], [], ["procedure"]]


def test_clean_text_combines_steps():
    assert clean_text("ACFT\n\nhit  WX") == "aircraft\n\nhit weather"

"""Loader tests on a fixture that mimics the ASRS Database Online export format."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd
import pytest

from asrs_rag.load import ASRSFormatError, load_reports, parse_export, split_multi

FIXTURE = Path(__file__).parent / "fixtures" / "asrs_export_fixture.csv"


def _fixture_rows() -> list[list[str]]:
    with FIXTURE.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _write(path: Path, rows: list[list[str]], encoding: str = "utf-8") -> Path:
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    path.write_bytes(buf.getvalue().encode(encoding))
    return path


def test_parse_export_maps_two_row_header():
    df = parse_export(FIXTURE)
    # Footer row is ignored; duplicate ACN is still present at this stage.
    assert list(df["acn"]) == ["1000001", "1000002", "1000001", "1000003"]
    for col in ("date_raw", "flight_phase", "contributing_factors", "primary_problem", "synopsis"):
        assert col in df.columns
    assert df.loc[0, "aircraft_type"] == "B737-800"


def test_load_reports_deduplicates_and_drops_empty_narratives():
    df = load_reports(FIXTURE)
    assert sorted(df["acn"]) == ["1000001", "1000002"]
    # The richer copy of ACN 1000001 (with synopsis) must be kept.
    row = df.set_index("acn").loc["1000001"]
    assert row["synopsis"].startswith("B737 crew descended")
    assert "FATIGUE" in row["narrative"]


def test_multi_valued_fields_become_lists():
    df = load_reports(FIXTURE).set_index("acn")
    assert df.loc["1000001", "contributing_factors"] == ["Human Factors", "Weather"]
    assert df.loc["1000001", "flight_phase"] == ["Final Approach"]
    # Human factors from Person 1 and Person 2 are merged.
    assert df.loc["1000002", "human_factors"] == ["Communication Breakdown", "Confusion"]


def test_narratives_from_both_reporters_are_joined_and_quotes_survive():
    narrative = load_reports(FIXTURE).set_index("acn").loc["1000002", "narrative"]
    assert 'via "B, C"' in narrative
    assert "Captain's report" in narrative
    assert "\n" in narrative  # embedded newline inside a quoted CSV cell


def test_dates_are_parsed_to_month_resolution():
    row = load_reports(FIXTURE).set_index("acn").loc["1000001"]
    assert row["date"] == pd.Timestamp("2022-03-01")
    assert row["year"] == 2022 and row["month"] == 3


def test_blank_group_cells_are_forward_filled(tmp_path):
    rows = _fixture_rows()
    groups = rows[0]
    blanked = [g if i == 0 or g != groups[i - 1] else "" for i, g in enumerate(groups)]
    path = _write(tmp_path / "blank_groups.csv", [blanked, *rows[1:]])
    df = load_reports(path).set_index("acn")
    assert df.loc["1000002", "primary_problem"] == "Airport"
    assert df.loc["1000002", "synopsis"].startswith("A320 crew")


def test_flattened_single_row_header(tmp_path):
    rows = _fixture_rows()
    counts: dict[str, int] = {}
    flat = []
    for g, f in zip(rows[0], rows[1], strict=True):
        if f == "ACN":
            flat.append("acn_num_ACN")
            continue
        n = counts.get(g, 0)
        counts[g] = n + 1
        flat.append(f"{g}{'.' + str(n) if n else ''}_{f}")
    path = _write(tmp_path / "flat.csv", [flat, *rows[2:]])
    df = load_reports(path)
    assert sorted(df["acn"]) == ["1000001", "1000002"]
    assert df.set_index("acn").loc["1000002", "flight_phase"] == ["Taxi"]


def test_windows_1252_encoding_and_leading_junk_rows(tmp_path):
    rows = _fixture_rows()
    rows[2][14] = rows[2][14] + " Crew noted 3° bank – unusual."
    path = _write(tmp_path / "cp1252.csv", [["ASRS Database Online export"], *rows], "cp1252")
    narrative = load_reports(path).set_index("acn").loc["1000001", "narrative"]
    assert "3° bank – unusual" in narrative


def test_directory_loading_merges_overlapping_exports(tmp_path):
    rows = _fixture_rows()
    _write(tmp_path / "a.csv", rows)
    _write(tmp_path / "b.csv", rows[:2] + rows[3:4])  # overlaps on ACN 1000002
    df = load_reports(tmp_path)
    assert sorted(df["acn"]) == ["1000001", "1000002"]


def test_missing_acn_column_raises(tmp_path):
    path = _write(tmp_path / "bad.csv", [["Date", "Narrative"], ["202201", "text"]])
    with pytest.raises(ASRSFormatError):
        parse_export(path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("A; B ;C", ["A", "B", "C"]), ("", []), (None, []), ("Single", ["Single"])],
)
def test_split_multi(value, expected):
    assert split_multi(value) == expected

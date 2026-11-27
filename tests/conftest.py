"""Shared fixtures: a toy corpus of prepared reports and a hashing-embedder index."""

from __future__ import annotations

import pandas as pd
import pytest

from asrs_rag.clean import prepare_corpus
from asrs_rag.embed import HashingEmbedder
from asrs_rag.index import ChunkConfig, ReportIndex

TOY = [
    # acn, year, phase, aircraft, primary problem, CF, HF, synopsis, narrative
    (
        "2000001",
        2019,
        "Final Approach",
        "B737-800",
        "Weather",
        "Weather",
        "",
        "Crew encountered wind shear on final approach and executed a go-around.",
        "On final approach we encountered severe wind shear with a 20 knot airspeed loss. "
        "The captain called go-around and we climbed out. Gusty winds were reported.",
    ),
    (
        "2000002",
        2020,
        "Initial Approach",
        "A320",
        "Weather",
        "Weather; Human Factors",
        "Workload",
        "Crew flew through moderate icing and turbulence during the approach.",
        "During the approach we entered icing conditions and moderate turbulence. "
        "Anti-ice was selected late because of high workload.",
    ),
    (
        "2000003",
        2021,
        "Cruise",
        "B737-800",
        "Human Factors",
        "Human Factors",
        "Fatigue",
        "Fatigued crew missed a crossing restriction after a long duty day.",
        "After a fourteen hour duty day both pilots were extremely tired. We missed the "
        "crossing restriction and ATC queried our altitude. Fatigue was the main factor.",
    ),
    (
        "2000004",
        2022,
        "Taxi",
        "CRJ200",
        "Human Factors",
        "Human Factors; Airport",
        "Communication Breakdown",
        "Crew misheard a taxi clearance and crossed a hold short line.",
        "Ground control instructed us to hold short of runway 27 but we heard cross runway "
        "27. The frequency was congested and the readback was not corrected.",
    ),
    (
        "2000005",
        2022,
        "Parked",
        "A320",
        "Aircraft",
        "Aircraft; MEL",
        "Troubleshooting",
        "Maintenance deferred a hydraulic pump under the MEL with incorrect paperwork.",
        "The number two hydraulic pump failed its test. Maintenance deferred the item using "
        "the MEL but the logbook entry referenced the wrong procedure.",
    ),
    (
        "2000006",
        2018,
        "Climb",
        "B737-800",
        "Aircraft",
        "Aircraft",
        "",
        "Engine oil pressure warning during climb led to an air turnback.",
        "Climbing through 15000 feet the engine oil pressure dropped and the warning "
        "illuminated. We ran the checklist, shut down the engine and returned to the airport.",
    ),
]


def toy_reports() -> pd.DataFrame:
    """Prepared toy reports in the canonical schema."""
    rows = []
    for acn, year, phase, ac, pp, cf, hf, syn, narr in TOY:
        rows.append(
            {
                "acn": acn,
                "date": pd.Timestamp(year=year, month=6, day=1),
                "year": year,
                "month": 6,
                "flight_phase": [phase],
                "aircraft_type": ac,
                "aircraft_operator": "Air Carrier",
                "far_part": "Part 121",
                "reporter_function": "Captain",
                "anomaly": [],
                "contributing_factors": [c.strip() for c in cf.split(";")],
                "human_factors": [h.strip() for h in hf.split(";") if h.strip()],
                "primary_problem": pp,
                "synopsis": syn,
                "narrative": narr,
            }
        )
    return prepare_corpus(pd.DataFrame(rows))


@pytest.fixture(scope="session")
def toy_corpus() -> pd.DataFrame:
    return toy_reports()


@pytest.fixture(scope="session")
def toy_index(toy_corpus) -> ReportIndex:
    return ReportIndex.build(
        toy_corpus, HashingEmbedder(256), ChunkConfig(max_words=40, overlap_words=10)
    )

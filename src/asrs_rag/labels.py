"""Contributing-factor label taxonomy derived from ASRS coded fields.

ASRS analysts code each report with *Contributing Factors / Situations*
(report level) and, per reporter, *Human Factors*. Those codes are fine
grained and unevenly used, so this project maps them onto nine analysis
labels. The mapping is exact-match on the coded values (never substring
matching, because e.g. ``"Environment - Non Weather Related"`` contains the
word *Weather*).

=====================  ==========================================================
Label                  ASRS source values
=====================  ==========================================================
human_factors          CF: Human Factors
fatigue                Person Human Factors: Fatigue
communication          Person Human Factors: Communication Breakdown
weather                CF: Weather
aircraft_equipment     CF: Aircraft, MEL, Incorrect / Not Installed / Unavailable
                       Part, Equipment / Tooling, Software and Automation
procedure              CF: Procedure
atc_airspace           CF: ATC Equipment / Nav Facility / Buildings, Airspace
                       Structure
organisational         CF: Company Policy, Manuals, Chart Or Publication,
                       Staffing, Logbook Entry
airport_environment    CF: Airport, Environment - Non Weather Related
=====================  ==========================================================
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

#: Contributing Factors / Situations values that map to each label.
CF_MAP: dict[str, frozenset[str]] = {
    "human_factors": frozenset({"Human Factors"}),
    "weather": frozenset({"Weather"}),
    "aircraft_equipment": frozenset(
        {
            "Aircraft",
            "MEL",
            "Incorrect / Not Installed / Unavailable Part",
            "Equipment / Tooling",
            "Software and Automation",
        }
    ),
    "procedure": frozenset({"Procedure"}),
    "atc_airspace": frozenset({"ATC Equipment / Nav Facility / Buildings", "Airspace Structure"}),
    "organisational": frozenset(
        {"Company Policy", "Manuals", "Chart Or Publication", "Staffing", "Logbook Entry"}
    ),
    "airport_environment": frozenset({"Airport", "Environment - Non Weather Related"}),
}

#: Person-level Human Factors values that map to each label.
HF_MAP: dict[str, frozenset[str]] = {
    "fatigue": frozenset({"Fatigue"}),
    "communication": frozenset({"Communication Breakdown"}),
}

#: Label order used everywhere (matrices, reports, figures).
LABELS: tuple[str, ...] = (
    "human_factors",
    "fatigue",
    "communication",
    "weather",
    "aircraft_equipment",
    "procedure",
    "atc_airspace",
    "organisational",
    "airport_environment",
)

#: One-line definitions shown to the LLM and in the app.
LABEL_DESCRIPTIONS: dict[str, str] = {
    "human_factors": "errors, situational awareness, distraction, workload, confusion or "
    "other human performance issues of crew, controllers or technicians",
    "fatigue": "tiredness, sleep loss, long duty periods or circadian disruption",
    "communication": "misunderstood, missed or ambiguous communication between crew, ATC, "
    "dispatch, maintenance or cabin (readback/hearback, frequency congestion, language)",
    "weather": "meteorological conditions: wind shear, turbulence, icing, thunderstorms, "
    "low visibility, gusts",
    "aircraft_equipment": "aircraft system, component, software/automation or maintenance "
    "equipment malfunction, deferred items (MEL) or wrong parts",
    "procedure": "a procedure was not followed, was inadequate or was applied incorrectly",
    "atc_airspace": "ATC equipment or facilities, navigation aids, or airspace design and "
    "structure",
    "organisational": "company policy, manuals, charts or publications, staffing, logbook "
    "and documentation issues",
    "airport_environment": "airport layout, signage, lighting, construction, wildlife, "
    "traffic or other non-weather environmental conditions",
}

assert set(LABEL_DESCRIPTIONS) == set(LABELS)


def labels_for_report(
    contributing_factors: Iterable[str], human_factors: Iterable[str]
) -> list[str]:
    """Return the analysis labels implied by a report's coded fields (in LABELS order)."""
    cf, hf = set(contributing_factors or ()), set(human_factors or ())
    found = {lab for lab, src in CF_MAP.items() if cf & src}
    found |= {lab for lab, src in HF_MAP.items() if hf & src}
    return [lab for lab in LABELS if lab in found]


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``labels`` list column."""
    out = df.copy()
    out["labels"] = [
        labels_for_report(cf, hf)
        for cf, hf in zip(out["contributing_factors"], out["human_factors"], strict=True)
    ]
    return out


def to_matrix(label_lists: Sequence[Sequence[str]], labels: Sequence[str] = LABELS) -> np.ndarray:
    """Convert label lists into a binary indicator matrix ``(n_reports, n_labels)``."""
    index = {lab: i for i, lab in enumerate(labels)}
    y = np.zeros((len(label_lists), len(labels)), dtype=np.int8)
    for r, labs in enumerate(label_lists):
        for lab in labs:
            if lab in index:
                y[r, index[lab]] = 1
    return y


def from_matrix(y: np.ndarray, labels: Sequence[str] = LABELS) -> list[list[str]]:
    """Inverse of :func:`to_matrix`."""
    return [[labels[j] for j in np.flatnonzero(row)] for row in np.asarray(y)]

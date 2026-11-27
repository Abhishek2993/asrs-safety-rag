"""A small, documented dictionary of ASRS abbreviations.

Two kinds of abbreviation appear in ASRS narratives:

1. **Legacy shorthand** (``SHORTHAND``) from the era when ASRS analysts
   re-typed narratives in capitals with contractions such as ``ACFT`` or
   ``CLRNC``. These carry no information beyond the word they stand for, so
   they are *replaced* by the full word.
2. **Operational acronyms** (``ACRONYMS``) such as ``TCAS`` or ``QRH``. Pilots
   and controllers search for these exact strings, so the acronym is *kept* and
   its expansion is inserted after the **first** occurrence only
   (``TCAS (traffic collision avoidance system)``). This lets embedding models
   see the meaning while BM25 keyword search still matches the acronym, and
   avoids inflating term counts by repeating the expansion.

Matching is case-sensitive on upper-case tokens with word boundaries, so
ordinary words (``alt``, ``Star``) are never touched. Sources: the ASRS
abbreviation list (ASRS Database Online help) and the FAA Pilot/Controller
Glossary. Entries were chosen from the most frequent upper-case tokens in the
2012-2022 corpus used for this project.
"""

from __future__ import annotations

SHORTHAND: dict[str, str] = {
    "ACFT": "aircraft",
    "ACFTS": "aircraft",
    "ALT": "altitude",
    "ALTS": "altitudes",
    "APCH": "approach",
    "ARPT": "airport",
    "CAPT": "captain",
    "CLB": "climb",
    "CLRD": "cleared",
    "CLRNC": "clearance",
    "CTLR": "controller",
    "CTLRS": "controllers",
    "CTR": "center",
    "DEP": "departure",
    "DSCNT": "descent",
    "DSND": "descend",
    "DSNDING": "descending",
    "EMER": "emergency",
    "ENG": "engine",
    "FLC": "flight crew",
    "FLT": "flight",
    "FREQ": "frequency",
    "FT": "feet",
    "HDG": "heading",
    "HYD": "hydraulic",
    "INTXN": "intersection",
    "KTS": "knots",
    "LBS": "pounds",
    "LNDG": "landing",
    "MAINT": "maintenance",
    "NM": "nautical miles",
    "PAX": "passengers",
    "PLT": "pilot",
    "PLTS": "pilots",
    "RPTR": "reporter",
    "RWY": "runway",
    "RWYS": "runways",
    "SPD": "speed",
    "TFC": "traffic",
    "TKOF": "takeoff",
    "TWR": "tower",
    "TWY": "taxiway",
    "VIS": "visibility",
    "WX": "weather",
    "XING": "crossing",
}

ACRONYMS: dict[str, str] = {
    "AGL": "above ground level",
    "AMM": "aircraft maintenance manual",
    "APU": "auxiliary power unit",
    "ARFF": "aircraft rescue and firefighting",
    "ATC": "air traffic control",
    "ATIS": "automatic terminal information service",
    "CA": "captain",
    "CIC": "controller in charge",
    "CRM": "crew resource management",
    "CTAF": "common traffic advisory frequency",
    "ECAM": "electronic centralized aircraft monitor",
    "EFB": "electronic flight bag",
    "EGPWS": "enhanced ground proximity warning system",
    "EICAS": "engine indicating and crew alerting system",
    "FA": "flight attendant",
    "FAF": "final approach fix",
    "FBO": "fixed base operator",
    "FLM": "front line manager",
    "FMC": "flight management computer",
    "FMS": "flight management system",
    "FO": "first officer",
    "GPWS": "ground proximity warning system",
    "IFR": "instrument flight rules",
    "ILS": "instrument landing system",
    "IMC": "instrument meteorological conditions",
    "LOA": "letter of agreement",
    "LOC": "localizer",
    "MEL": "minimum equipment list",
    "MSL": "mean sea level",
    "MVA": "minimum vectoring altitude",
    "NOTAM": "notice to air missions",
    "PAPI": "precision approach path indicator",
    "PDC": "pre-departure clearance",
    "PF": "pilot flying",
    "PIC": "pilot in command",
    "PM": "pilot monitoring",
    "QRH": "quick reference handbook",
    "RA": "resolution advisory",
    "RNAV": "area navigation",
    "RNP": "required navigation performance",
    "SID": "standard instrument departure",
    "SOP": "standard operating procedure",
    "STAR": "standard terminal arrival route",
    "TA": "traffic advisory",
    "TCAS": "traffic collision avoidance system",
    "TMU": "traffic management unit",
    "TOGA": "takeoff/go-around",
    "TRACON": "terminal radar approach control",
    "VFR": "visual flight rules",
    "VMC": "visual meteorological conditions",
    "VNAV": "vertical navigation",
}

assert not set(SHORTHAND) & set(ACRONYMS), "an abbreviation must be in one table only"

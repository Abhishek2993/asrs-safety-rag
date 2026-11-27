"""Streamlit front end: similar incidents, contributing factors and cited trends.

Run with ``streamlit run app/app.py`` after building an index
(``python -m asrs_rag index``) and, optionally, training the classifier
(``python -m asrs_rag train``).
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from asrs_rag import config
from asrs_rag.classify import (
    LLMClassifier,
    NeighbourVoteClassifier,
    SupervisedClassifier,
    scores_from_proba,
)
from asrs_rag.clean import clean_text
from asrs_rag.index import ReportIndex
from asrs_rag.labels import LABEL_DESCRIPTIONS, LABELS, to_matrix
from asrs_rag.llm import get_llm
from asrs_rag.retrieve import Filters, HybridRetriever
from asrs_rag.summarise import describe_filters, summarise

# Reference categorical palette (first three slots validate all-pairs).
BLUE, ORANGE, AQUA, GREY = "#257de0", "#f47230", "#00b894", "#9c9c9c"

EXAMPLE = (
    "On the RNAV arrival we were given a late descent and the captain, who was tired after "
    "a red-eye, reset the MCP altitude but the autopilot did not start down. ATC asked us to "
    "expedite and we crossed the fix 1,500 feet high, then had to go around because we "
    "were unstable on final in gusty winds."
)

st.set_page_config(page_title="ASRS Safety RAG", page_icon="✈️", layout="wide")


# ---------------------------------------------------------------- resources
def available_indexes() -> list[Path]:
    """Index directories that exist, full corpus first."""
    candidates = [config.INDEX_DIR, config.DATA_DIR / "index_sample"]
    return [p for p in candidates if (p / "manifest.json").exists()]


@st.cache_resource(show_spinner="Loading index ...")
def load_index(path: str) -> ReportIndex:
    return ReportIndex.load(path)


@st.cache_resource(show_spinner="Training the TF-IDF classifier on the indexed reports ...")
def load_classifier(index_path: str) -> SupervisedClassifier:
    """Use the saved model if present, otherwise fit one on the loaded index."""
    model_path = config.MODEL_DIR / "tfidf_lr.joblib"
    if model_path.exists() and Path(index_path) == config.INDEX_DIR:
        return SupervisedClassifier.load(model_path)
    idx = load_index(index_path)
    clf = SupervisedClassifier("tfidf")
    return clf.fit(idx.reports["narrative_clean"].tolist(), to_matrix(idx.reports["labels"]))


def highlight(text: str, terms: list[str]) -> str:
    """HTML-escape ``text`` and wrap matched query terms in <mark>."""
    safe = html.escape(text)
    if not terms:
        return safe
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, terms)) + r")\b", re.I)
    return pattern.sub(r"<mark>\1</mark>", safe)


def sidebar() -> tuple[ReportIndex | None, str, object]:
    st.sidebar.title("ASRS Safety RAG")
    indexes = available_indexes()
    if not indexes:
        st.sidebar.error("No index found. Run `python -m asrs_rag index` first.")
        return None, "", None
    labels = {
        str(p): f"{p.name} ({load_index(str(p)).manifest['n_reports']:,} reports)" for p in indexes
    }
    choice = st.sidebar.selectbox("Index", list(labels), format_func=labels.get)
    idx = load_index(choice)
    st.sidebar.caption(f"Embedder: `{idx.manifest['embedder']}`")

    backends = ["none", "auto", "ollama", "transformers", "openai", "anthropic"]
    default = os.environ.get("ASRS_RAG_LLM", "none")
    backend = st.sidebar.selectbox(
        "LLM backend",
        backends,
        index=backends.index(default) if default in backends else 0,
        help="'none' keeps everything deterministic: retrieval, the supervised classifier and "
        "template summaries still work. API keys are read from environment variables only.",
    )
    llm = None
    if backend != "none":
        try:
            llm = get_llm(backend)
        except Exception as exc:  # surfaced to the user, app keeps working
            st.sidebar.warning(f"LLM unavailable: {exc}")
    st.sidebar.caption(f"LLM: `{getattr(llm, 'name', 'none (no-LLM mode)')}`")
    st.sidebar.divider()
    st.sidebar.caption(
        "For safety analysis and learning only. ASRS reports are voluntary, unverified and "
        "de-identified; they are not a statistical sample and must not be used to assign "
        "blame or make operational decisions."
    )
    return idx, choice, llm


def filter_controls(idx: ReportIndex, key: str) -> Filters:
    """One row of filter controls."""
    years = idx.reports["year"].dropna().astype(int)
    phases = sorted({p for ps in idx.reports["flight_phase"] for p in ps})
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    yr = c1.slider(
        "Years",
        int(years.min()),
        int(years.max()),
        (int(years.min()), int(years.max())),
        key=f"{key}-years",
    )
    phase = c2.multiselect(
        "Flight phase contains",
        ["approach", "climb", "cruise", "descent", "landing", "taxi", "takeoff", "parked", *phases],
        key=f"{key}-phase",
    )
    factor = c3.multiselect("Contributing factor", LABELS, key=f"{key}-factor")
    aircraft = c4.text_input("Aircraft contains", key=f"{key}-ac")
    return Filters(
        year_from=yr[0], year_to=yr[1], phases=phase, labels=factor, aircraft=aircraft or None
    )


# ---------------------------------------------------------------- page: query
def factor_chart(scores) -> alt.Chart:
    df = pd.DataFrame(
        [
            {
                "factor": s.label,
                "probability": s.probability,
                "threshold": s.threshold,
                "decision": ("predicted" if s.predicted else "not predicted")
                + ("" if s.confident else " (uncertain)"),
            }
            for s in scores
        ]
    )
    order = df["factor"].tolist()
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=14)
        .encode(
            y=alt.Y("factor:N", sort=order, title=None),
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
            color=alt.Color(
                "decision:N",
                scale=alt.Scale(
                    domain=[
                        "predicted",
                        "predicted (uncertain)",
                        "not predicted",
                        "not predicted (uncertain)",
                    ],
                    range=[BLUE, "#84b9f3", "#d0d0d0", "#e6e6e6"],
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                "factor",
                alt.Tooltip("probability:Q", format=".2f"),
                alt.Tooltip("threshold:Q", format=".2f"),
                "decision",
            ],
        )
    )
    ticks = (
        alt.Chart(df)
        .mark_tick(color="#191919", thickness=2, size=18)
        .encode(
            y=alt.Y("factor:N", sort=order),
            x="threshold:Q",
            tooltip=[alt.Tooltip("threshold:Q", format=".2f")],
        )
    )
    return (bars + ticks).properties(height=28 * len(df))


def page_query(idx: ReportIndex, index_path: str, llm) -> None:
    st.header("Similar past incidents")
    narrative = st.text_area("Paste an incident narrative", EXAMPLE, height=140)
    c1, c2 = st.columns([1, 1])
    k = c1.slider("Number of reports", 3, 15, 5)
    mode = c2.radio("Retrieval", ["hybrid", "dense", "bm25"], horizontal=True)
    with st.expander("Filters", expanded=False):
        filters = filter_controls(idx, "q")
    if not narrative.strip():
        return

    retriever = HybridRetriever(idx)
    hits = retriever.search(narrative, k=k, filters=filters, mode=mode)
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader(f"Top {len(hits)} similar reports")
        if not hits:
            st.info("No reports match the filters.")
        for h in hits:
            d = h.to_dict()
            with st.container(border=True):
                factors = ", ".join(d["labels"]) or "none"
                st.markdown(
                    f"**ACN {h.acn}** &nbsp; `{d['date']}` &nbsp; {d['flight_phase']} · "
                    f"{d['aircraft_type']}  \nscore {d['score']:.4f} "
                    f"(dense {d['dense_score']}, BM25 {d['bm25_score']}) · primary problem: "
                    f"*{d['primary_problem'] or 'n/a'}* · factors: {factors}"
                )
                st.markdown(f"_{html.escape(d['synopsis'] or '')}_")
                st.markdown(
                    f"<div style='font-size:0.9em;line-height:1.5'>"
                    f"{highlight(h.chunk_text.split(chr(10), 1)[-1], h.matched_terms)}</div>",
                    unsafe_allow_html=True,
                )

    with right:
        st.subheader("Predicted contributing factors")
        clf = load_classifier(index_path)
        text = clean_text(narrative)
        scores = scores_from_proba(clf.predict_proba([text]), clf.thresholds)[0]
        st.altair_chart(factor_chart(scores), use_container_width=True)
        predicted = [s for s in scores if s.predicted]
        st.markdown(
            "**Supervised (TF-IDF + logistic regression):** "
            + (
                ", ".join(
                    f"{s.label} ({s.probability:.2f}{'' if s.confident else ', uncertain'})"
                    for s in predicted
                )
                or "no factor above threshold"
            )
        )
        st.caption(
            "Black tick = per-label decision threshold tuned on training data. "
            "'Uncertain' = within 0.10 of the threshold: review manually."
        )
        knn = NeighbourVoteClassifier(retriever, k=10)
        votes = knn.predict_proba([narrative])[0]
        st.markdown(
            "**Neighbour vote (top-10 similar reports):** "
            + ", ".join(
                f"{lab} {v:.0%}"
                for lab, v in sorted(zip(LABELS, votes, strict=True), key=lambda t: -t[1])
                if v >= 0.2
            )
        )
        if llm is not None and st.button("Ask the LLM (few-shot with retrieved examples)"):
            with st.spinner(f"Querying {llm.name} ..."):
                out = LLMClassifier(llm, retriever).classify(text)
            st.markdown(
                f"**LLM ({llm.name}):** {', '.join(out['labels']) or 'no labels'}  \n"
                f"examples used: {', '.join('ACN ' + a for a in out['examples'])}"
            )
        with st.expander("Label definitions"):
            for lab in LABELS:
                st.markdown(f"- **{lab}** - {LABEL_DESCRIPTIONS[lab]}")


# --------------------------------------------------------------- page: trends
def page_trends(idx: ReportIndex, llm) -> None:
    st.header("Trends across reports")
    filters = filter_controls(idx, "t")
    query = st.text_input("Optional theme to focus the cited evidence (e.g. 'unstable approach')")
    s = summarise(idx, filters, llm=llm, query=query or None, n_evidence=12)
    st.markdown(f"**{s.n_reports:,} reports** match: {describe_filters(filters)}")
    if not s.n_reports:
        return

    c1, c2 = st.columns(2, gap="large")
    years = pd.DataFrame(
        {"year": list(s.counts_by_year), "reports": list(s.counts_by_year.values())}
    )
    c1.altair_chart(
        alt.Chart(years, title="Reports per year")
        .mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2, color=BLUE)
        .encode(
            x=alt.X("year:O", title=None),
            y=alt.Y("reports:Q", title="Reports"),
            tooltip=["year", "reports"],
        )
        .properties(height=260),
        use_container_width=True,
    )
    shares = pd.DataFrame(
        {"factor": list(s.label_counts), "reports": list(s.label_counts.values())}
    ).assign(share=lambda d: d["reports"] / s.n_reports)
    c2.altair_chart(
        alt.Chart(shares, title="Share of reports coding each contributing factor")
        .mark_bar(cornerRadiusEnd=4, height=14, color=BLUE)
        .encode(
            y=alt.Y("factor:N", sort="-x", title=None),
            x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
            tooltip=["factor", "reports", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=260),
        use_container_width=True,
    )
    lab = st.selectbox("Factor over time", LABELS, index=LABELS.index("weather"))
    by_year = pd.DataFrame(
        [
            {"year": y, "share": counts[lab] / max(s.counts_by_year[y], 1)}
            for y, counts in s.label_by_year.items()
        ]
    )
    st.altair_chart(
        alt.Chart(by_year, title=f"Share of matching reports coded '{lab}'")
        .mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2, color=ORANGE)
        .encode(
            x=alt.X("year:O", title=None),
            y=alt.Y("share:Q", axis=alt.Axis(format="%"), title=None),
            tooltip=["year", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=220),
        use_container_width=True,
    )

    st.subheader("Findings (every statement cites report IDs)")
    st.caption(f"Generator: {s.generator}. Counts above are computed, not generated.")
    st.markdown(s.text or "_No grounded statements._")
    c = s.citation_check
    ok = c.get("passed")
    msg = (
        f"Citation check: {c.get('n_valid_citations', 0)}/{c.get('n_citations', 0)} citations "
        f"point to the evidence set; statement coverage {c.get('statement_coverage', 0):.0%}; "
        f"{len(s.removed_statements)} ungrounded statement(s) removed."
    )
    (st.success if ok else st.warning)(("✓ " if ok else "⚠ ") + msg)
    if s.removed_statements:
        with st.expander("Removed statements (no valid citation)"):
            for r in s.removed_statements:
                st.markdown(f"- {html.escape(r)}")
    with st.expander("Evidence reports the summary was allowed to cite"):
        st.dataframe(
            pd.DataFrame(s.evidence)[
                ["acn", "date", "flight_phase", "aircraft_type", "labels", "synopsis"]
            ],
            hide_index=True,
            use_container_width=True,
        )


def main() -> None:
    idx, index_path, llm = sidebar()
    if idx is None:
        st.stop()
    tab1, tab2 = st.tabs(["Similar incidents & factors", "Trends"])
    with tab1:
        page_query(idx, index_path, llm)
    with tab2:
        page_trends(idx, llm)


main()

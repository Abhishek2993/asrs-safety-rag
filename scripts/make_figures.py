"""Render README figures from results/*.json (every plotted number comes from a run).

Usage::

    python scripts/make_figures.py            # reads results/, writes docs/figures/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from asrs_rag.config import FIGURES_DIR, RESULTS_DIR
from asrs_rag.labels import LABELS

# Reference categorical palette, slots 1-3 (validated all-pairs for CVD).
SERIES = ["#257de0", "#f47230", "#00b894"]
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT,
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def load(name: str, sub: str = "") -> dict | None:
    p = RESULTS_DIR / sub / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def grouped_bars(ax, groups, series, values, colors, fmt="{:.2f}", horizontal=False):
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (s, c) in enumerate(zip(series, colors, strict=True)):
        pos = x + (i - (n - 1) / 2) * width
        vals = values[i]
        bars = (ax.barh if horizontal else ax.bar)(pos, vals, width * 0.9, color=c, label=s)
        for b, v in zip(bars, vals, strict=True):
            if horizontal:
                ax.text(
                    v + 0.01,
                    b.get_y() + b.get_height() / 2,
                    fmt.format(v),
                    va="center",
                    fontsize=7,
                    color=MUTED,
                )
            else:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.01,
                    fmt.format(v),
                    ha="center",
                    fontsize=7.5,
                    color=MUTED,
                )
    if horizontal:
        ax.set_yticks(x, groups)
        ax.invert_yaxis()
        ax.xaxis.grid(True, color=GRID)
    else:
        ax.set_xticks(x, groups)
        ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)


def fig_retrieval(out: Path) -> None:
    proxy, lab = load("retrieval"), load("labelled", "sample")
    if not proxy or not lab:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    modes = ["bm25", "dense", "hybrid"]
    names = ["BM25", "Dense (MiniLM)", "Hybrid (RRF)"]
    ks = [1, 5, 10]
    grouped_bars(
        axes[0],
        [f"P@{k}" for k in ks],
        names,
        [[proxy[m][f"P@{k}"] for k in ks] for m in modes],
        SERIES,
    )
    chance = proxy["chance"]["P@1"]
    axes[0].axhline(chance, color=MUTED, ls="--", lw=1)
    axes[0].text(2.45, chance + 0.015, f"chance {chance:.2f}", ha="right", fontsize=8, color=MUTED)
    axes[0].set_ylim(0, 1)
    axes[0].set_title(
        f"Proxy relevance: same primary problem\n"
        f"{proxy['n_queries']} queries, {proxy['index']['n_reports']:,} reports"
    )
    ks2 = [1, 3, 5, 10]
    grouped_bars(
        axes[1],
        [f"R@{k}" for k in ks2],
        names,
        [[lab[m][f"R@{k}"] for k in ks2] for m in modes],
        SERIES,
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title(
        f"Hand-labelled queries: recall@k\n{lab['n_queries']} queries, "
        f"{lab['index']['n_reports']} sample reports"
    )
    axes[1].legend(loc="upper left", fontsize=8)
    fig.savefig(out / "retrieval.png")
    plt.close(fig)


def fig_classification(out: Path) -> None:
    res = load("classification")
    if not res:
        return
    models = [
        ("tfidf_lr", "TF-IDF + LR"),
        ("embedding_lr", "Embeddings + LR"),
        ("neighbour_vote", "Neighbour vote"),
    ]
    models = [(k, n) for k, n in models if k in res]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    grouped_bars(
        axes[0],
        list(LABELS),
        [n for _, n in models],
        [[res[k]["per_label"][lab]["f1"] for lab in LABELS] for k, _ in models],
        SERIES[: len(models)],
        horizontal=True,
    )
    axes[0].set_xlim(0, 1.05)
    axes[0].set_title(f"Per-label F1 on held-out test split (n={res['n_test']:,})")
    axes[0].legend(loc="lower right", fontsize=8)
    prev = res["label_prevalence_test"]
    y = np.arange(len(LABELS))
    axes[1].barh(y, [prev[lab] for lab in LABELS], 0.6, color=SERIES[0])
    for i, lab in enumerate(LABELS):
        axes[1].text(prev[lab] + 0.01, i, f"{prev[lab]:.1%}", va="center", fontsize=8, color=MUTED)
    axes[1].set_yticks(y, LABELS)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(prev.values()) * 1.25)
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_title("Label prevalence (class imbalance)")
    fig.tight_layout()
    fig.savefig(out / "classification.png")
    plt.close(fig)


def fig_llm(out: Path) -> None:
    res = load("llm")
    if not res:
        return
    keys = [
        ("tfidf_lr_same_sample", "TF-IDF + LR"),
        ("zero_shot", "LLM zero-shot"),
        ("few_shot", "LLM few-shot (RAG)"),
    ]
    keys = [(k, n) for k, n in keys if k in res]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    grouped_bars(
        ax,
        ["micro-F1", "macro-F1"],
        [n for _, n in keys],
        [[res[k]["micro"]["f1"], res[k]["macro"]["f1"]] for k, _ in keys],
        SERIES[: len(keys)],
    )
    ax.set_ylim(0, 1)
    ax.set_title(f"Same {res['n']} test reports - {res['llm']}")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out / "llm_vs_supervised.png")
    plt.close(fig)


def fig_trend(out: Path) -> None:
    """Approach-phase reports per year and weather share (from the full index)."""
    from asrs_rag.config import INDEX_DIR
    from asrs_rag.index import ReportIndex
    from asrs_rag.retrieve import Filters
    from asrs_rag.summarise import compute_statistics

    if not (INDEX_DIR / "manifest.json").exists():
        return
    idx = ReportIndex.load(INDEX_DIR)
    sub = idx.reports[Filters(phases=["approach"]).mask(idx.reports)]
    stats = compute_statistics(sub)
    years = sorted(stats["counts_by_year"])
    share = [stats["label_by_year"][y]["weather"] / stats["counts_by_year"][y] for y in years]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(years, share, color=SERIES[0], lw=2, marker="o", ms=5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.grid(True, color=GRID)
    ax.set_ylim(0, max(share) * 1.3)
    ax.set_title(f"Approach-phase reports coded 'weather', by year (n={len(sub):,} reports)")
    for y, s, n in zip(years, share, [stats["counts_by_year"][y] for y in years], strict=True):
        ax.text(y, s + max(share) * 0.04, f"n={n}", ha="center", fontsize=7, color=MUTED)
    fig.savefig(out / "trend_weather_approach.png")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (fig_retrieval, fig_classification, fig_llm, fig_trend):
        fn(FIGURES_DIR)
    print(sorted(p.name for p in FIGURES_DIR.glob("*.png")))


if __name__ == "__main__":
    main()

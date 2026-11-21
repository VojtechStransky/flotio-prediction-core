"""Draws the charts in docs/figures from the evaluation reports in docs/results.

    python tools/plot_report.py

registry_*.json  -> who_curves.png, who_call.png, when_placement.png

Needs only matplotlib and numpy. The reports come from flotio_core.evaluate.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "docs" / "results", ROOT / "docs" / "figures"
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]  # one per data set, in the order they are passed


def style(ax, grid: str = "both") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=MUTED, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=10.5, color=INK, pad=8)


def percent_axis(ax, x: bool = True, y: bool = True) -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    if x:
        ax.xaxis.set_major_formatter(fmt)
    if y:
        ax.yaxis.set_major_formatter(fmt)


def legend(ax, **kw) -> None:
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, handlelength=1.6, **kw)


def who_curves(reports: list[dict], path: Path) -> None:
    """Ranking and probability quality of the 12-month classifier."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6), facecolor=SURFACE, gridspec_kw={"hspace": 0.36, "wspace": 0.2})
    roc, pr, gains, cal = axes.ravel()

    for r, colour in zip(reports, COLOURS, strict=False):
        w, c, t = r["who"], r["who"]["curves"], r["who"]["threshold"]
        roc.plot(
            c["roc"]["false_positive_rate"],
            c["roc"]["recall"],
            color=colour,
            linewidth=2,
            label=f"{r['name']}: buyer ranked higher {w['roc_auc']:.0%}",
        )
        roc.plot(
            t["false_positive_rate"],
            t["recall"],
            "o",
            color=colour,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        pr.plot(
            c["precision_recall"]["recall"],
            c["precision_recall"]["precision"],
            color=colour,
            linewidth=2,
            label=r["name"],
        )
        pr.plot(
            t["recall"], t["precision"], "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2
        )
        gains.plot(
            c["gains"]["called_share"],
            c["gains"]["buyers_found"],
            color=colour,
            linewidth=2,
            label=f"{r['name']}: top 20 % hold {w['top_20pct']['buyers_found']:.0%}",
        )
        rows = c["calibration"]
        cal.plot(
            [x["predicted"] for x in rows],
            [x["observed"] for x in rows],
            color=colour,
            linewidth=2,
            label=r["name"],
        )

    for ax in (roc, gains):
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label="no model")
    top = max(x["predicted"] for r in reports for x in r["who"]["curves"]["calibration"])
    top = max(top, max(x["observed"] for r in reports for x in r["who"]["curves"]["calibration"])) * 1.1
    cal.plot([0, top], [0, top], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label="perfect")

    title(roc, "ROC curve: buyers found against non-buyers called\n(dot = the threshold in use)")
    roc.set_xlabel("non-buyers called", color=SECONDARY, fontsize=9)
    roc.set_ylabel("buyers found (recall)", color=SECONDARY, fontsize=9)
    title(pr, "Precision and recall\n(dot = the threshold in use; right end = share of buyers)")
    pr.set_xlabel("buyers found (recall)", color=SECONDARY, fontsize=9)
    pr.set_ylabel("called firms that buy (precision)", color=SECONDARY, fontsize=9)
    pr.set_ylim(0, 1)
    title(gains, "Cumulative gains: calling firms in the model's order")
    gains.set_xlabel("share of firms called", color=SECONDARY, fontsize=9)
    gains.set_ylabel("share of all buyers found", color=SECONDARY, fontsize=9)
    title(cal, "Calibration: predicted chance against what happened\n(firms in ten equal groups)")
    cal.set_xlabel("predicted chance of a purchase", color=SECONDARY, fontsize=9)
    cal.set_ylabel("share that bought", color=SECONDARY, fontsize=9)

    legend(roc, loc="lower right")
    legend(pr, loc="upper right")
    legend(gains, loc="lower right")
    legend(cal, loc="upper left")
    for ax in axes.ravel():
        style(ax)
        percent_axis(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def who_call(reports: list[dict], path: Path) -> None:
    """The yes/no call at the threshold on the test year."""
    fig, bars = plt.subplots(figsize=(5.6, 4.3), facecolor=SURFACE)
    measures = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall")]
    width = 0.8 / len(reports)
    for i, (r, colour) in enumerate(zip(reports, COLOURS, strict=False)):
        values = [r["who"]["threshold"][k] for k, _ in measures]
        x = np.arange(len(measures)) + (i - (len(reports) - 1) / 2) * width
        bars.bar(x, values, width=width * 0.9, color=colour, label=r["name"])
        for xi, v in zip(x, values, strict=True):
            bars.text(xi, v + 0.012, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK)
    bars.set_xticks(range(len(measures)), [name for _, name in measures])
    bars.set_ylim(0, 1)
    title(bars, "The yes/no call on the test year, at the threshold in use")
    legend(bars, loc="upper right", ncols=1)
    style(bars, "y")
    percent_axis(bars, x=False)
    bars.tick_params(axis="x", labelsize=10, colors=INK)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def when_placement(reports: list[dict], path: Path) -> None:
    """The gate's 3-month window against the baselines; one small panel per data set."""
    rows = [
        (
            "A purchase falls in the chosen 3-month window (the gate)",
            lambda r: (
                [
                    ("Month model", r["when_gate"]["model_pct"] / 100),
                    ("Hand-written rule", r["when_gate"]["hand_rule_pct"] / 100),
                ]
                + ([("Random window", r["when_gate"]["random_pct"] / 100)] if "random_pct" in r["when_gate"] else [])
            ),
        ),
    ]
    fig, axes = plt.subplots(
        len(rows),
        len(reports),
        figsize=(10, 0.55 * max(len(rows[0][1](r)) for r in reports) + 0.9),
        facecolor=SURFACE,
        squeeze=False,
        gridspec_kw={"wspace": 0.08},
    )
    for i, (heading, pick) in enumerate(rows):
        limit = max(v for r in reports for _, v in pick(r)) * 1.22
        for j, (r, colour) in enumerate(zip(reports, COLOURS, strict=False)):
            ax, items = axes[i, j], pick(r)
            names, values = [n for n, _ in items], [v for _, v in items]
            y = np.arange(len(items))[::-1]
            ax.barh(y, values, height=0.68, color=[colour] + [MUTED] * (len(items) - 1))
            for yi, v in zip(y, values, strict=True):
                ax.text(v + limit * 0.015, yi, f"{v:.1%}", va="center", fontsize=9, color=INK)
            ax.set_xlim(0, limit)
            ax.set_yticks(y, names if j == 0 else [""] * len(items))
            style(ax, "x")
            percent_axis(ax, y=False)
            ax.tick_params(axis="y", labelsize=9.5, colors=INK)
            ax.set_title(r["name"], loc="left", fontsize=10, color=SECONDARY, pad=4)
        axes[i, 0].annotate(
            heading,
            (0, 1),
            xycoords="axes fraction",
            xytext=(0, 24),
            textcoords="offset points",
            fontsize=10.5,
            color=INK,
            ha="left",
            va="bottom",
            annotation_clip=False,
        )
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def draw(reports: list[dict], prefix: str) -> None:
    reports = [r for r in reports if "who" in r]
    if not reports:
        return
    who_curves(reports, FIGURES / f"{prefix}who_curves.png")
    who_call(reports, FIGURES / f"{prefix}who_call.png")
    if all("when" in r and "when_gate" in r for r in reports):
        when_placement(reports, FIGURES / f"{prefix}when_placement.png")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK})
    FIGURES.mkdir(parents=True, exist_ok=True)
    order = {"O4": 0, "N3": 1, "T": 2}
    registry = sorted(RESULTS.glob("registry_*.json"), key=lambda p: order.get(p.stem.split("_")[1], 9))
    draw([json.loads(p.read_text()) for p in registry], "")
    print("wrote", ", ".join(sorted(p.name for p in FIGURES.glob("*.png"))))


if __name__ == "__main__":
    main()

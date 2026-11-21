"""Draws the charts in docs/figures from the evaluation reports in docs/results.

    python tools/plot_report.py

registry_*.json  -> who_curves.png, who_ranking.png

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


def who_ranking(reports: list[dict], path: Path) -> None:
    """Ranking quality (ROC AUC) in each walk-forward test year, with the 95 % interval of the last one."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    years = [y["test_cutoff"][:7] for y in reports[0]["who_by_year"]]
    x = np.arange(len(years))
    ends = []
    for i, (r, colour) in enumerate(zip(reports, COLOURS, strict=False)):
        auc = [y["roc_auc"] for y in r["who_by_year"]]
        ax.plot(x, auc, color=colour, linewidth=2, label=r["name"])
        ax.plot(x, auc, "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        low, high = r["who"]["roc_auc_ci95"]
        bar_x = x[-1] + 0.12 * (i + 1)  # beside the last point, so overlapping intervals stay readable
        ax.plot([bar_x, bar_x], [low, high], color=colour, linewidth=2.5, solid_capstyle="round")
        ends.append([auc[-1], f"{r['name']}  {auc[-1]:.0%}"])
    # Spread the end labels so series that finish at the same value do not collide.
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.025)
    for y_label, text in ends:
        ax.text(x[-1] + 0.12 * (len(reports) + 1), y_label, text, va="center", fontsize=9, color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(x[0], 0.505, "random ranking", fontsize=8.5, color=MUTED, va="bottom")
    ax.set_xticks(x, years)
    ax.set_xlim(-0.3, len(years) - 1 + 2.1)
    ax.set_ylim(0.45, 0.9)
    ax.set_xlabel("test year (cutoff)", color=SECONDARY, fontsize=9)
    ax.set_ylabel("buyer ranked above a non-buyer", color=SECONDARY, fontsize=9)
    title(
        ax,
        "Ranking in each test year, each a separate model fitted on earlier years\n"
        "(bars beside the last year = its 95 % interval)",
    )
    legend(ax, loc="lower left", ncols=3)
    style(ax, "y")
    percent_axis(ax, x=False)
    ax.tick_params(axis="x", colors=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def draw(reports: list[dict], prefix: str) -> None:
    reports = [r for r in reports if "who" in r]
    if not reports:
        return
    who_curves(reports, FIGURES / f"{prefix}who_curves.png")
    who_ranking(reports, FIGURES / f"{prefix}who_ranking.png")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK})
    FIGURES.mkdir(parents=True, exist_ok=True)
    order = {"O4": 0, "N3": 1, "T": 2}
    registry = sorted(RESULTS.glob("registry_*.json"), key=lambda p: order.get(p.stem.split("_")[1], 9))
    draw([json.loads(p.read_text()) for p in registry], "")
    print("wrote", ", ".join(sorted(p.name for p in FIGURES.glob("*.png"))))


if __name__ == "__main__":
    main()

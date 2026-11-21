"""Full evaluation of both models on a purchase history, saved as JSON for the charts.

    python examples/evaluate.py purchases_O4.parquet O4 "O4 semi-trailers" --data-end 2026-09-01
    python tools/plot_report.py            # draws docs/figures/*.png from docs/results/registry_*.json

The input file (parquet or csv) has one row per purchased vehicle with the columns
firm (int), purchase_date (date), brand (str), year_built (int), weight (float).
The reports in docs/results were made this way from the registry, which is not in
this repository.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from flotio_core import evaluate, when, who

RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results"
COLUMNS = {"firm": pl.Int64, "purchase_date": pl.Date, "brand": pl.Utf8, "year_built": pl.Int32, "weight": pl.Float64}


def load(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path, try_parse_dates=True)
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing the columns {sorted(missing)}")
    return frame.select([pl.col(c).cast(t) for c, t in COLUMNS.items()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("purchases", type=Path, help="parquet or csv, one row per purchased vehicle")
    parser.add_argument("category", help="short code used in the file name, e.g. O4")
    parser.add_argument("name", help="label in the charts, e.g. 'O4 semi-trailers'")
    parser.add_argument("--data-end", type=date.fromisoformat, required=True, help="first day after the data")
    args = parser.parse_args()

    purchases = load(args.purchases).filter(pl.col("purchase_date") < args.data_end)
    data_end = args.data_end
    test = who.add_months(data_end, -12)
    cutoff = who.add_months(data_end, -24)

    report = {
        "name": args.name,
        "data_end": str(data_end),
        "who": evaluate.evaluate_who(purchases, test),
        # the test year above is the last of these; the earlier years each get their own model
        "who_by_year": [
            {k: r[k] for k in ("test_cutoff", "firms", "base_rate", "roc_auc", "pr_auc", "brier_skill")}
            | {k: r["threshold"][k] for k in ("precision", "recall", "false_positive_rate")}
            for r in evaluate.walk_forward(purchases, [who.add_months(test, -12 * i) for i in (4, 3, 2, 1, 0)])
        ],
        "when_gate": when.backtest(purchases, cutoff, data_end),
        "when": evaluate.month_report(purchases, cutoff, data_end),
    }
    out = RESULTS / f"registry_{args.category}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")

    w, t = report["who"], report["who"]["threshold"]
    print(
        f"WHO   buyer ranked higher {w['roc_auc']:.1%}  "
        f"average precision {w['pr_auc']:.3f} (base rate {w['base_rate']:.3f})  "
        f"Brier skill {w['brier_skill']:+.3f}"
    )
    print(
        f"      at the threshold: accuracy {t['accuracy']:.3f}  precision {t['precision']:.3f}  "
        f"recall {t['recall']:.3f}"
    )
    m = report["when"]
    print(f"WHEN  monthly probability: purchase month ranked higher {m['monthly_probability']['roc_auc']:.1%}")
    for name in ("model", "hand_rule", "random"):
        p = m["placement"][name]
        print(f"      {name:<17s} top month right {p['top1']:.1%}   one of top three right {p['top3']:.1%}")
    print("wrote", out.relative_to(out.parents[2]))


if __name__ == "__main__":
    main()

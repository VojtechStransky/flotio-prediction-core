# Forward check: September and October 2026

The results in [RESULTS.md](RESULTS.md) come from models that were trained on
the registry release of 2026-09-01 and judged on older, held-out years. The
release of 2026-11-01 adds two months that no model had seen: September and
October 2026. This page checks the two models on those two months.

- **Who buys** holds up. Buyers in the two new months were ranked above
  non-buyers about as often as in the same check a year earlier: 84 % for
  trailers, 85 % for trucks and 72 % for tractors.
- **The monthly probabilities** are close to what happened for trailers and
  trucks, within about 10 % in both years. For tractors they are too high,
  by 13 % this year and 32 % the year before.
- **Tractors** have a coverage gap. Half of the tractor buyers had never bought
  before, so neither model could score them.
- **Not tested:** timing. Two months cannot show whether the model picks the
  right month, so the finding in RESULTS.md that it places the month barely
  better than chance still stands.

Run on 2026-11-21. The reports are in [`results/forward_*.json`](results/),
aggregates only. The registry data and the script that loads it are not in the
repository.

Contents: [How it was run](#how-it-was-run) · [Is the new data complete](#is-the-new-data-complete) · [Who buys](#who-buys) · [Monthly probability](#monthly-probability) · [What this does not show](#what-this-does-not-show)

## How it was run

**Forward.** Both models were built with the code in this repository, from
the 2026-09-01 release only:

- `year_prediction.fit()` with validation cutoff 2025-09, scoring every firm at
  2026-09-01;
- `month_prediction.train()` with `data_end` 2026-09-01, giving every modelled
  firm a probability for September (h = 0) and October (h = 1).

The outcomes come only from the 2026-11-01 release: which firms bought in
[2026-09-01, 2026-11-01). The models never saw that release.

**A year earlier, for comparison.** The same code was run one year earlier on
the old release: cutoff 2025-09-01, outcomes September and October 2025.

A two-month result cannot be compared with the 12-month figures in
RESULTS.md. Even a perfect 12-month model cannot tell which of its buyers will
buy in the first two months. Read each forward number against the year-earlier
column next to it.

## Is the new data complete

The purchases were built again from the raw files of the new release, with the
same rules as the production loader. Two checks:

| Vehicle category | Purchases 2016–2025, old / new release | Largest difference in a month | August 2026, old / new | Sep + Oct 2025 | Sep + Oct 2026 |
|---|---:|---:|---:|---:|---:|
| O4 semi-trailers | 53,815 / 53,815 | 0.4 % | 445 / 442 | 963 | 962 |
| N3 heavy trucks | 97,096 / 97,095 | 0.2 % | 619 / 614 | 1,776 | 1,819 |
| T tractors | 41,829 / 41,836 | 0.7 % | 293 / 295 | 590 | 617 |

- **The rebuild matches.** The two releases agree on ten years of history, so
  any difference in the new months is real, not a side effect of how the data
  was built.
- **The newest months are complete.** August was already complete in the old
  release, and September–October 2026 have as many purchases as a year
  earlier. The code allows for up to three months of lag in the source data,
  but none shows up between these two releases. This is one pair of releases,
  so it is a reason not to discount October, not proof that the lag never
  happens.

## Who buys

The 12-month model scored every firm with an earlier purchase on 2026-09-01.
The question is whether the firms it ranked high are the ones that bought in
the next one or two months.

How often a buyer was ranked above a non-buyer, with the 95 % range:

| Vehicle category | September only | Year earlier | September + October | Year earlier |
|---|---:|---:|---:|---:|
| O4 semi-trailers | 84 % (81 to 88 %) | 84 % (80 to 88 %) | 84 % (81 to 86 %) | 81 % (79 to 84 %) |
| N3 heavy trucks | 86 % (83 to 88 %) | 87 % (84 to 89 %) | 85 % (83 to 87 %) | 85 % (83 to 87 %) |
| T tractors | 79 % (74 to 83 %) | 75 % (70 to 81 %) | 72 % (68 to 75 %) | 72 % (68 to 77 %) |

All the forward numbers fall within the year-earlier ranges: the model held,
and did not get better or worse.

The yes/no call over September + October, at the threshold in use.
Each cell is **forward / a year earlier**:

| Vehicle category | Firms called | Recall | Precision | Lift | Buyers in the top 10 % of firms |
|---|---:|---:|---:|---:|---:|
| O4 semi-trailers | 26 % | 74 % / 70 % | 8.4 % / 8.9 % | 2.9× / 2.7× | 58 % / 55 % |
| N3 heavy trucks | 24 % | 73 % / 79 % | 8.1 % / 9.8 % | 3.0× / 3.2× | 61 % / 60 % |
| T tractors | 20 % | 50 % / 56 % | 2.2 % / 2.4 % | 2.5× / 2.9× | 40 % / 45 % |

Precision is low because only about 3 % of trailer and truck firms buy in a
given two months. It is not comparable to the 30 % in RESULTS.md, which counts
buyers over a full year. Lift compares like with like: the called firms bought
about three times as often as an average firm, as in RESULTS.md.

### Buyers the model could not score

A firm with no earlier purchase has no history to score.

| Vehicle category | Buyers, Sep + Oct 2026 | Not scored | A year earlier |
|---|---:|---:|---:|
| O4 semi-trailers | 337 | 82 (24 %) | 65 of 341 (19 %) |
| N3 heavy trucks | 505 | 109 (22 %) | 103 of 538 (19 %) |
| T tractors | 481 | 247 (51 %) | 246 of 455 (54 %) |

For tractors, half of the buyers are first-time buyers in the registry. This
limits the tractor model more than its ranking does.

## Monthly probability

The month model gives each firm a probability of buying in each month. This is
the first check of those probabilities on months after the training data.

Average predicted monthly probability against the share of firm-months with a
purchase, and expected against actual buyers over the two months:

| Vehicle category | Predicted / observed | Year earlier | Expected / actual buyers | Year earlier |
|---|---:|---:|---:|---:|
| O4 semi-trailers | 3.55 % / 3.49 % | 3.33 % / 3.65 % | 251 / 233 | 233 / 244 |
| N3 heavy trucks | 3.12 % / 2.98 % | 3.09 % / 3.38 % | 382 / 354 | 370 / 389 |
| T tractors | 0.85 % / 0.75 % | 0.90 % / 0.68 % | 201 / 172 | 212 / 153 |

- **Trailers and trucks:** within about 10 % in both years, and too high
  (2026) as often as too low (2025). The close match in 2026 is partly luck.
- **Tractors:** too high in both years, by 13 % and 32 %. This is a consistent
  bias. Tractor firms buy rarely, and the model expects more purchases than
  happen.

The model ranks firm-months well: 82 % for trailers, 85 % for trucks and 82 %
for tractors, the same as a year earlier. That ranking mostly reflects *who*
buys, not *when* (see below).

The month model covers only firms with a purchase in the last five years. Of
the September buyers it covered 68 % for trailers, 76 % for trucks and 39 % for
tractors. The predicted and observed rates above are for those firms only.

## What this does not show

- **The 12-month result.** The full 12 months after 2026-09-01 will not be
  known until September 2027. Until then, two months show that the ranking
  held up, not that the 12-month probabilities are right.
- **Which month.** The ranking of firm-months above mostly measures which firms
  buy. Whether the model picks the right month among a buyer's months was not
  tested, and two months are too few to test it well. The answer in
  RESULTS.md (barely better than chance) stands.
- **The hand-written rule.** The rule gives each firm a share of *its own*
  months, not a probability that can be compared across firms, so it was not
  put in the tables. The model against the rule is compared in the gate in
  RESULTS.md.
- **A single pair of releases.** One forward window. Rerunning the check with
  each new release would show whether this holds.

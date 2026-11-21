"""month_shares(): what callers use, each month's share of a firm's purchase chance."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from .history import month_index, purchase_months
from .model import HazardModel


def month_shares(
    hm: HazardModel, purchases: pl.DataFrame, months: list[date], origin: date | None = None
) -> dict[int, np.ndarray]:
    """Per firm, the share of each of `months` in its purchase chance (sums to 1).

    Scored at `origin`, the first month without data (default hm.data_end). Pass the
    current end of the data when scoring newer data than the model was trained on, or
    firms that bought since then still look overdue. The lag of the source data is a gap in
    months-ahead rather than a hole in the inputs. Firms outside the
    model (no purchase in ACTIVE_MONTHS) are absent; callers fall back for them.
    """
    pm = purchase_months(purchases)
    origin = origin or hm.data_end
    firms, ph = hm.predict(pm, origin)
    hs = [month_index(m) - month_index(origin) for m in months]
    if any(h < 0 or h >= hm.horizon for h in hs):
        raise ValueError(f"months {months[0]}..{months[-1]} fall outside the model horizon from {origin}")
    p = ph[:, hs]
    w = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    return dict(zip(firms.to_list(), w, strict=True))

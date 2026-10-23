"""Service analytics across every recorded run: the history behind the live board.

Each sitting becomes one row with its clock times and its timings in minutes,
so questions like "how long do tables turn on Fridays?" or "when does the
kitchen fall behind?" are a group-by away. Rows are built from the same SQLite
store the pipeline writes; nothing extra needs recording.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import date, datetime, timedelta

import pandas as pd

from tablewatch.state.metrics import sitting_metrics, table_summary
from tablewatch.state.table_state import Sitting, Visit
from tablewatch.storage.db import EventStore

# A sitting that starts at 01:00 still belongs to the previous evening's service.
SERVICE_DAY_START_HOUR = 4

SITTING_COLUMNS = [
    "site", "run_id", "table_id", "sitting", "seated", "left", "service_date", "weekday", "hour",
    "guests", "open", "turn_time_min", "first_visit_min", "visits", "mean_gap_min", "max_gap_min",
    "drinks_min", "food_min", "order_to_food_min", "reset_min", "alerts",
]
OCCUPANCY_COLUMNS = ["site", "run_id", "service_date", "table_id", "occupied_min", "idle_min"]


def service_date(moment: datetime) -> date:
    return (moment - timedelta(hours=SERVICE_DAY_START_HOUR)).date()


def _minutes(seconds: float | None) -> float | None:
    return None if seconds is None else seconds / 60.0


def reset_times(sittings: Iterable[Sitting], visits: Iterable[Visit]) -> dict[str, float]:
    """Sitting key -> seconds from the guests leaving to staff starting to clear the table.

    The clearing visit is the first staff visit at the empty table before the
    next guests sit down there; sittings with no such visit are left out.
    """
    sittings = sorted(sittings, key=lambda s: s.seated_at)
    clearing = sorted((v for v in visits if v.sitting_key is None), key=lambda v: v.start)
    out = {}
    for s in sittings:
        if s.ended_at is None:
            continue
        next_seat = min((n.seated_at for n in sittings if n.table_id == s.table_id and n.seated_at > s.seated_at), default=None)
        for v in clearing:
            if v.table_id == s.table_id and v.start >= s.ended_at and (next_seat is None or v.start < next_seat):
                out[s.key] = v.start - s.ended_at
                break
    return out


def _in_range(df: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    if start is not None:
        df = df[df["service_date"] >= start]
    if end is not None:
        df = df[df["service_date"] <= end]
    return df.reset_index(drop=True)


def sittings_frame(
    store: EventStore, sites: Iterable[str] | None = None, start: date | None = None, end: date | None = None
) -> pd.DataFrame:
    """One row per sitting across all runs (optionally limited to sites and service dates)."""
    sites = set(sites) if sites else None
    rows = []
    for run in store.runs():
        if sites and run.site not in sites:
            continue
        sittings, visits = store.sittings(run.run_id), store.visits(run.run_id)
        alerts = Counter(a["sitting_key"] for a in store.alerts(run.run_id))
        resets = reset_times(sittings, visits)
        for s in sittings:
            m = sitting_metrics(s, visits, run.last_ts)
            seated = datetime.fromtimestamp(run.media_start + s.seated_at)
            rows.append(
                {
                    "site": run.site,
                    "run_id": run.run_id,
                    "table_id": s.table_id,
                    "sitting": s.key,
                    "seated": seated,
                    "left": datetime.fromtimestamp(run.media_start + s.ended_at) if s.ended_at is not None else None,
                    "service_date": service_date(seated),
                    "weekday": seated.strftime("%a"),
                    "hour": seated.hour,
                    "guests": s.guests_max or None,
                    "open": m.is_open,
                    # An open sitting was cut off (still seated, or the video ended): no turn time.
                    "turn_time_min": None if m.is_open else _minutes(m.duration_s),
                    "first_visit_min": _minutes(m.time_to_first_visit_s),
                    "visits": m.visit_count,
                    "mean_gap_min": _minutes(m.mean_gap_s),
                    "max_gap_min": _minutes(m.max_gap_s),
                    "drinks_min": _minutes(m.time_to_drinks_s),
                    "food_min": _minutes(m.time_to_food_s),
                    "order_to_food_min": _minutes(m.order_to_food_s),
                    "reset_min": _minutes(resets.get(s.key)),
                    "alerts": alerts.get(s.key, 0),
                }
            )
    df = pd.DataFrame(rows, columns=SITTING_COLUMNS)
    numeric = [c for c in SITTING_COLUMNS if c.endswith("_min")] + ["guests"]
    df[numeric] = df[numeric].astype(float)
    return _in_range(df, start, end)


def occupancy_frame(
    store: EventStore, sites: Iterable[str] | None = None, start: date | None = None, end: date | None = None
) -> pd.DataFrame:
    """Occupied vs idle minutes per table per run (for utilisation)."""
    sites = set(sites) if sites else None
    rows = []
    for run in store.runs():
        if (sites and run.site not in sites) or run.last_ts <= 0:
            continue
        sittings, visits = store.sittings(run.run_id), store.visits(run.run_id)
        day = service_date(datetime.fromtimestamp(run.media_start))
        for t in run.tables:
            s = table_summary(t["table_id"], sittings, visits, 0.0, run.last_ts)
            rows.append(
                {"site": run.site, "run_id": run.run_id, "service_date": day, "table_id": s.table_id,
                 "occupied_min": s.occupied_s / 60.0, "idle_min": s.idle_s / 60.0}
            )
    return _in_range(pd.DataFrame(rows, columns=OCCUPANCY_COLUMNS), start, end)


def _mean(series: pd.Series) -> float | None:
    series = series.dropna()
    return float(series.mean()) if len(series) else None


def summary(sittings: pd.DataFrame, occupancy: pd.DataFrame | None = None) -> dict[str, float | int | None]:
    """Headline numbers for a set of sittings (averages skip sittings without that event)."""
    occupied = occupancy["occupied_min"].sum() if occupancy is not None and len(occupancy) else 0.0
    observed = occupied + (occupancy["idle_min"].sum() if occupancy is not None and len(occupancy) else 0.0)
    return {
        "sittings": len(sittings),
        "services": int(sittings["service_date"].nunique()) if len(sittings) else 0,
        "avg_turn_time_min": _mean(sittings["turn_time_min"]),
        "avg_first_visit_min": _mean(sittings["first_visit_min"]),
        "avg_drinks_min": _mean(sittings["drinks_min"]),
        "avg_food_min": _mean(sittings["food_min"]),
        "avg_order_to_food_min": _mean(sittings["order_to_food_min"]),
        "avg_gap_min": _mean(sittings["mean_gap_min"]),
        "avg_reset_min": _mean(sittings["reset_min"]),
        "sittings_with_alert_pct": float((sittings["alerts"] > 0).mean() * 100) if len(sittings) else None,
        "utilisation_pct": float(occupied / observed * 100) if observed else None,
    }


_AGG = {
    "sittings": ("sitting", "count"),
    "avg_turn_time_min": ("turn_time_min", "mean"),
    "avg_first_visit_min": ("first_visit_min", "mean"),
    "avg_drinks_min": ("drinks_min", "mean"),
    "avg_food_min": ("food_min", "mean"),
    "alerts": ("alerts", "sum"),
}


def by_day(sittings: pd.DataFrame) -> pd.DataFrame:
    return sittings.groupby("service_date", as_index=False).agg(**_AGG)


def by_hour(sittings: pd.DataFrame) -> pd.DataFrame:
    """By the hour guests sat down: when it gets busy, and whether service keeps up."""
    return sittings.groupby("hour", as_index=False).agg(**_AGG)


def by_table(sittings: pd.DataFrame, occupancy: pd.DataFrame | None = None) -> pd.DataFrame:
    keys = ["site", "table_id"]
    out = sittings.groupby(keys, as_index=False).agg(**_AGG)
    if occupancy is not None and len(occupancy):
        occ = occupancy.groupby(keys, as_index=False)[["occupied_min", "idle_min"]].sum()
        occ["utilisation_pct"] = occ["occupied_min"] / (occ["occupied_min"] + occ["idle_min"]) * 100
        out = out.merge(occ[[*keys, "utilisation_pct"]], on=keys, how="left")
    return out

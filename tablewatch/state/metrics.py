"""Service metrics computed from sittings and visits.

Pure functions over `Sitting` / `Visit` records so the same code serves the
live pipeline, the dashboard (records loaded from SQLite) and the tests.
Open sittings (still in progress, or cut off by the end of a video) are
measured up to `now` and flagged `is_open`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from statistics import mean

from tablewatch.state.table_state import Sitting, TableSnapshot, TableState, Visit


@dataclass
class SittingMetrics:
    table_id: str
    sitting_key: str
    seated_at: float
    ended_at: float | None
    is_open: bool
    duration_s: float  # table turn time (so far, if open)
    visit_count: int
    time_to_first_visit_s: float | None
    visit_gaps_s: list[float] = field(default_factory=list)  # end of one visit -> start of the next
    time_to_drinks_s: float | None = None  # seated -> drinks on the table (estimated)
    time_to_food_s: float | None = None  # seated -> food on the table (estimated)
    order_to_food_s: float | None = None  # first staff visit -> food on the table

    @property
    def max_gap_s(self) -> float | None:
        return max(self.visit_gaps_s) if self.visit_gaps_s else None

    @property
    def mean_gap_s(self) -> float | None:
        return mean(self.visit_gaps_s) if self.visit_gaps_s else None


@dataclass
class IdlePeriod:
    table_id: str
    start: float
    end: float
    is_open: bool  # still empty at `now`

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TableSummary:
    table_id: str
    sittings: int
    completed_sittings: int
    visits: int
    avg_time_to_first_visit_s: float | None
    max_time_to_first_visit_s: float | None
    avg_visit_gap_s: float | None
    max_visit_gap_s: float | None
    avg_visits_per_sitting: float | None
    avg_turn_time_s: float | None  # completed sittings only
    idle_s: float
    occupied_s: float

    @property
    def utilisation(self) -> float | None:
        total = self.idle_s + self.occupied_s
        return self.occupied_s / total if total > 0 else None


def _visits_for(sitting: Sitting, visits: Iterable[Visit]) -> list[Visit]:
    return sorted((v for v in visits if v.sitting_key == sitting.key), key=lambda v: v.start)


def sitting_metrics(sitting: Sitting, visits: Iterable[Visit], now: float) -> SittingMetrics:
    own = _visits_for(sitting, visits)
    end = sitting.ended_at if sitting.ended_at is not None else now
    gaps = []
    for prev, cur in pairwise(own):
        if prev.end is not None:
            gaps.append(max(0.0, cur.start - prev.end))
    return SittingMetrics(
        table_id=sitting.table_id,
        sitting_key=sitting.key,
        seated_at=sitting.seated_at,
        ended_at=sitting.ended_at,
        is_open=sitting.ended_at is None,
        duration_s=max(0.0, end - sitting.seated_at),
        visit_count=len(own),
        time_to_first_visit_s=max(0.0, own[0].start - sitting.seated_at) if own else None,
        visit_gaps_s=gaps,
        time_to_drinks_s=_since(sitting.seated_at, sitting.drinks_served_at),
        time_to_food_s=_since(sitting.seated_at, sitting.food_served_at),
        order_to_food_s=_since(own[0].start, sitting.food_served_at) if own else None,
    )


def _since(start: float, end: float | None) -> float | None:
    return None if end is None else max(0.0, end - start)


def idle_periods(table_id: str, sittings: Iterable[Sitting], run_start: float, now: float) -> list[IdlePeriod]:
    """Stretches where the table had no sitting, from the start of the run to `now`."""
    periods = []
    cursor: float | None = run_start
    for s in sorted((s for s in sittings if s.table_id == table_id), key=lambda s: s.seated_at):
        if cursor is not None and s.seated_at > cursor:
            periods.append(IdlePeriod(table_id, cursor, s.seated_at, is_open=False))
        cursor = s.ended_at  # None while the sitting is still open
    if cursor is not None and now > cursor:
        periods.append(IdlePeriod(table_id, cursor, now, is_open=True))
    return periods


def _avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def table_summary(
    table_id: str, sittings: Iterable[Sitting], visits: Iterable[Visit], run_start: float, now: float
) -> TableSummary:
    sittings = [s for s in sittings if s.table_id == table_id]
    visits = [v for v in visits if v.table_id == table_id]
    per = [sitting_metrics(s, visits, now) for s in sittings]
    firsts = [m.time_to_first_visit_s for m in per if m.time_to_first_visit_s is not None]
    gaps = [g for m in per for g in m.visit_gaps_s]
    done = [m for m in per if not m.is_open]
    idle = sum(p.duration_s for p in idle_periods(table_id, sittings, run_start, now))
    return TableSummary(
        table_id=table_id,
        sittings=len(per),
        completed_sittings=len(done),
        visits=sum(m.visit_count for m in per),
        avg_time_to_first_visit_s=_avg(firsts),
        max_time_to_first_visit_s=max(firsts) if firsts else None,
        avg_visit_gap_s=_avg(gaps),
        max_visit_gap_s=max(gaps) if gaps else None,
        avg_visits_per_sitting=_avg([m.visit_count for m in per]),
        avg_turn_time_s=_avg([m.duration_s for m in done]),
        idle_s=idle,
        occupied_s=sum(m.duration_s for m in per),
    )


def state_at(table_id: str, t: float, sittings: Iterable[Sitting], visits: Iterable[Visit]) -> tuple[TableState, float | None]:
    """Reconstruct a table's state at time `t` from stored records (for replay).

    Returns (state, since) where `since` is when that state began, or None if
    the table has been empty since before the first record.
    """
    sitting = None
    last_end = None
    for s in sittings:
        if s.table_id != table_id:
            continue
        if s.seated_at <= t and (s.ended_at is None or t < s.ended_at):
            sitting = s
        elif s.ended_at is not None and s.ended_at <= t:
            last_end = s.ended_at if last_end is None else max(last_end, s.ended_at)
    if sitting is None:
        return TableState.EMPTY, last_end
    own = [v for v in _visits_for(sitting, visits) if v.start <= t]
    if not own:
        return TableState.SEATED, sitting.seated_at
    latest = own[-1]
    if latest.end is None or t < latest.end:
        return TableState.ATTENDED, latest.start
    return TableState.WAITING, latest.end


def clip_records(sittings: Iterable[Sitting], visits: Iterable[Visit], t: float) -> tuple[list[Sitting], list[Visit]]:
    """The records as they would have looked at time `t` (for replay)."""
    clipped_sittings = [
        Sitting(
            s.key, s.table_id, s.seated_at,
            ended_at=s.ended_at if s.ended_at is not None and s.ended_at <= t else None,
            guests_max=s.guests_max,
            drinks_served_at=s.drinks_served_at if s.drinks_served_at is not None and s.drinks_served_at <= t else None,
            food_served_at=s.food_served_at if s.food_served_at is not None and s.food_served_at <= t else None,
        )
        for s in sittings
        if s.seated_at <= t
    ]
    clipped_visits = [
        Visit(v.key, v.table_id, v.start, v.end if v.end is not None and v.end <= t else None, v.sitting_key)
        for v in visits
        if v.start <= t
    ]
    return clipped_sittings, clipped_visits


def snapshot_at(table_id: str, t: float, sittings: Iterable[Sitting], visits: Iterable[Visit]) -> TableSnapshot:
    """Rebuild a table snapshot at time `t` from stored records (head-counts are not stored)."""
    sittings, visits = clip_records(sittings, visits, t)
    state, since = state_at(table_id, t, sittings, visits)
    sitting = next((s for s in sittings if s.table_id == table_id and s.ended_at is None), None)
    own = _visits_for(sitting, visits) if sitting else []
    ends = [v.end for v in own if v.end is not None]
    return TableSnapshot(
        table_id=table_id,
        ts=t,
        state=state,
        state_since=since if since is not None else 0.0,
        sitting_key=sitting.key if sitting else None,
        seated_at=sitting.seated_at if sitting else None,
        visit_count=len(own),
        visit_started_at=own[-1].start if state == TableState.ATTENDED else None,
        last_visit_end=max(ends) if ends else None,
        empty_since=(since if since is not None else 0.0) if state == TableState.EMPTY else None,
        drinks_served_at=sitting.drinks_served_at if sitting else None,
        food_served_at=sitting.food_served_at if sitting else None,
    )


@dataclass
class Segment:
    table_id: str
    state: TableState
    start: float
    end: float
    sitting_key: str | None = None


def state_segments(
    table_ids: Iterable[str], sittings: Iterable[Sitting], visits: Iterable[Visit], run_start: float, now: float
) -> list[Segment]:
    """Split each table's timeline into contiguous state segments up to `now`."""
    sittings, visits = clip_records(sittings, visits, now)
    segments: list[Segment] = []
    for table_id in table_ids:
        for idle in idle_periods(table_id, sittings, run_start, now):
            segments.append(Segment(table_id, TableState.EMPTY, idle.start, idle.end))
        for s in (s for s in sittings if s.table_id == table_id):
            end = s.ended_at if s.ended_at is not None else now
            cursor, waiting_state = s.seated_at, TableState.SEATED
            for v in _visits_for(s, visits):
                v_end = min(v.end if v.end is not None else now, end)
                if v.start > cursor:
                    segments.append(Segment(table_id, waiting_state, cursor, v.start, s.key))
                segments.append(Segment(table_id, TableState.ATTENDED, max(v.start, cursor), v_end, s.key))
                cursor, waiting_state = max(cursor, v_end), TableState.WAITING
            if end > cursor:
                segments.append(Segment(table_id, waiting_state, cursor, end, s.key))
    return [seg for seg in segments if seg.end > seg.start]

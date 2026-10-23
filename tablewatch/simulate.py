"""Synthetic restaurant service: exercises the dashboard, analytics and alerts without video.

Generates plausible guest and staff presence per table (arrivals, orders,
drinks and food deliveries, check-backs, slow tables, clearing up) plus the
drink and food items a camera would see on each table, and feeds it through
the real state machine, served-item detection, alert engine and SQLite store,
exactly as the vision pipeline would.

    python -m tablewatch simulate --hours 3                    # one evening, instant, for replay
    python -m tablewatch simulate --days 7                     # a week of evenings for analytics
    python -m tablewatch simulate --hours 3 --live --speed 30   # watch it unfold live
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tablewatch.alerts.notifiers import NotificationDispatcher
from tablewatch.config import AppConfig
from tablewatch.pipeline import ServiceMonitor
from tablewatch.state.served import DRINKS, FOOD
from tablewatch.storage.db import EventStore
from tablewatch.zones.zones import TableZone, ZoneMap

log = logging.getLogger(__name__)
MIN = 60.0
BUSY_DAYS = {4, 5}  # Friday and Saturday (Monday = 0): fuller rooms, slower service


@dataclass
class TablePlan:
    guests: list[tuple[float, float, int]] = field(default_factory=list)  # (start, end, count)
    staff: list[tuple[float, float]] = field(default_factory=list)
    items: list[tuple[float, float, str, int]] = field(default_factory=list)  # (start, end, category, count)

    def counts(self, ts: float) -> tuple[int, int, dict[str, int]]:
        g = next((n for s, e, n in self.guests if s <= ts < e), 0)
        st = sum(1 for s, e in self.staff if s <= ts < e)
        items: dict[str, int] = {}
        for s, e, category, n in self.items:
            if s <= ts < e:
                items[category] = items.get(category, 0) + n
        return g, st, items

    def visit(self, rng: random.Random, at: float, low: float = 30, high: float = 90) -> float:
        end = at + rng.uniform(low, high)
        self.staff.append((at, end))
        return end


def plan_table(rng: random.Random, seats: int, duration_s: float, busy: bool = False) -> TablePlan:
    plan = TablePlan()
    t = rng.uniform(0, 15 * MIN)
    while t < duration_s:
        party = rng.randint(1, seats)
        seat = t
        leave = seat + rng.uniform(45, 110 if busy else 95) * MIN
        plan.guests.append((seat, leave, party))

        slow_start = rng.random() < (0.25 if busy else 0.1)  # waits too long for the first visit
        order = seat + (rng.uniform(5.5, 9) if slow_start else rng.uniform(1, 4)) * MIN
        plan.visit(rng, order, 60, 150)  # take the order

        drinks = order + rng.uniform(3, 7) * MIN
        plan.visit(rng, drinks)
        clear = leave + rng.uniform(0.3, 1.5 if busy else 2.5) * MIN
        plan.items.append((drinks + 15, clear + 20, DRINKS, party))

        slow_kitchen = rng.random() < (0.3 if busy else 0.12)
        food = order + (rng.uniform(26, 38) if slow_kitchen else rng.uniform(12, 24)) * MIN
        if food < leave - 15 * MIN:
            check_backs(plan, rng, drinks, food - 3 * MIN, busy)  # while the kitchen works
            plan.visit(rng, food)
            plates_cleared = min(leave - rng.uniform(5, 12) * MIN, food + rng.uniform(25, 45) * MIN)
            plan.items.append((food + 15, plates_cleared + 20, FOOD, party))
            check_backs(plan, rng, food, plates_cleared - 3 * MIN, busy)
            plan.visit(rng, plates_cleared)
            check_backs(plan, rng, plates_cleared, leave - 3 * MIN, busy)
        else:
            check_backs(plan, rng, drinks, leave - 3 * MIN, busy)  # drinks only
        plan.visit(rng, clear, 40, 90)  # clear and reset the table
        t = leave + rng.uniform(2, 12 if busy else 30) * MIN
    return plan


def check_backs(plan: TablePlan, rng: random.Random, after: float, until: float, busy: bool) -> None:
    """Staff look in every 4 to 8 minutes; now and then (more on busy nights) a gap runs long enough to alert."""
    long_gap = 0.15 if busy else 0.04
    v = after + rng.uniform(4, 8) * MIN
    while v < until:
        end = plan.visit(rng, v)
        v = end + (rng.uniform(10.5, 15) if rng.random() < long_gap else rng.uniform(4, 8)) * MIN


def default_zones(n_tables: int) -> ZoneMap:
    """Placeholder geometry: the simulator never looks at pixels."""
    zones = [
        TableZone(f"T{i + 1}", [(i * 10, 0), (i * 10 + 9, 0), (i * 10 + 9, 9), (i * 10, 9)], seats=4 if i % 3 else 2)
        for i in range(n_tables)
    ]
    return ZoneMap(zones, (n_tables * 10, 10))


def simulate_service(
    cfg: AppConfig,
    store: EventStore,
    start: datetime,
    tables: int = 8,
    hours: float = 3.0,
    seed: int = 7,
    live: bool = False,
    speed: float = 30.0,
    notify: bool = False,
    step_s: float = 1.0,
) -> int | None:
    """One evening's service as one run. Returns its run id.

    New guests are seated for `hours`; the run then carries on until the last
    table has been cleared, as a real service would.
    """
    rng = random.Random(seed)
    zone_map = default_zones(tables)
    busy = start.weekday() in BUSY_DAYS
    plans = {z.table_id: plan_table(rng, z.seats, hours * 3600, busy) for z in zone_map.zones}
    duration = max((end for p in plans.values() for _, end in p.staff), default=hours * 3600) + 60
    dispatcher = NotificationDispatcher.from_config(cfg.alerts.notifiers) if notify else None
    monitor = ServiceMonitor(cfg, zone_map, store, dispatcher)
    media_start = time.time() if live else start.timestamp()
    monitor.start_run(f"simulation(seed={seed}, tables={tables})", is_live=live, fps=1 / step_s, media_start=media_start)
    began = time.time()
    ts = 0.0
    try:
        while ts <= duration:
            counts = {tid: plan.counts(ts) for tid, plan in plans.items()}
            monitor.update(
                ts,
                {tid: c[0] for tid, c in counts.items()},
                {tid: c[1] for tid, c in counts.items()},
                item_counts={tid: c[2] for tid, c in counts.items()},
                wall_time=media_start + ts,
            )
            ts += step_s
            if live:
                delay = began + ts / speed - time.time()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        monitor.finish(min(ts, duration))
    return monitor.run_id


def simulate(
    cfg: AppConfig,
    tables: int = 8,
    hours: float = 3.0,
    seed: int = 7,
    live: bool = False,
    speed: float = 30.0,
    start_time: datetime | None = None,
    notify: bool = False,
    days: int = 1,
) -> list[int | None]:
    """Simulate `days` consecutive evenings (one run each), the last one today unless `start_time` is given."""
    if live and days != 1:
        raise ValueError("--live simulates a single service")
    first = start_time or datetime.now().replace(hour=18, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
    store = EventStore(cfg.storage.db_path)
    run_ids = []
    for d in range(days):
        day = first + timedelta(days=d)
        run_ids.append(simulate_service(cfg, store, day, tables, hours, seed + d, live, speed, notify))
        if days > 1:
            log.warning("Simulated %s (%s)", day.strftime("%a %d %b"), "busy" if day.weekday() in BUSY_DAYS else "quiet")
    return run_ids

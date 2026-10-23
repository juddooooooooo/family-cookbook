"""Synthetic restaurant service: exercises the dashboard and alerts without video.

Generates plausible guest and staff presence per table (arrivals, service
visits, slow tables, clearing up) and feeds it through the real state machine,
alert engine and SQLite store, exactly as the vision pipeline would.

    python -m tablewatch simulate --hours 3                    # instant, for replay
    python -m tablewatch simulate --hours 3 --live --speed 30   # watch it unfold live
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime

from tablewatch.alerts.notifiers import NotificationDispatcher
from tablewatch.config import AppConfig
from tablewatch.pipeline import ServiceMonitor
from tablewatch.storage.db import EventStore
from tablewatch.zones.zones import TableZone, ZoneMap

log = logging.getLogger(__name__)
MIN = 60.0


@dataclass
class TablePlan:
    guests: list[tuple[float, float, int]] = field(default_factory=list)  # (start, end, count)
    staff: list[tuple[float, float]] = field(default_factory=list)

    def counts(self, ts: float) -> tuple[int, int]:
        g = next((n for s, e, n in self.guests if s <= ts < e), 0)
        st = sum(1 for s, e in self.staff if s <= ts < e)
        return g, st


def plan_table(rng: random.Random, seats: int, duration_s: float) -> TablePlan:
    plan = TablePlan()
    t = rng.uniform(0, 15 * MIN)
    while t < duration_s:
        seat, leave = t, t + rng.uniform(35, 95) * MIN
        plan.guests.append((seat, leave, rng.randint(1, seats)))
        slow_start = rng.random() < 0.2  # ~1 in 5 tables waits too long for the first visit
        v = seat + (rng.uniform(5.5, 9) if slow_start else rng.uniform(1, 4)) * MIN
        while v < leave - 2 * MIN:
            dur = rng.uniform(30, 150)
            plan.staff.append((v, v + dur))
            long_gap = rng.random() < 0.15
            v += dur + (rng.uniform(10.5, 16) if long_gap else rng.uniform(3, 9)) * MIN
        clear = leave + rng.uniform(0.5, 2.5) * MIN
        plan.staff.append((clear, clear + rng.uniform(40, 90)))
        t = leave + rng.uniform(3, 25) * MIN
    return plan


def default_zones(n_tables: int) -> ZoneMap:
    """Placeholder geometry: the simulator never looks at pixels."""
    zones = [
        TableZone(f"T{i + 1}", [(i * 10, 0), (i * 10 + 9, 0), (i * 10 + 9, 9), (i * 10, 9)], seats=4 if i % 3 else 2)
        for i in range(n_tables)
    ]
    return ZoneMap(zones, (n_tables * 10, 10))


def simulate(
    cfg: AppConfig,
    tables: int = 8,
    hours: float = 3.0,
    seed: int = 7,
    live: bool = False,
    speed: float = 30.0,
    start_time: datetime | None = None,
    notify: bool = False,
    step_s: float = 1.0,
) -> int | None:
    rng = random.Random(seed)
    zone_map = default_zones(tables)
    duration = hours * 3600
    plans = {z.table_id: plan_table(rng, z.seats, duration) for z in zone_map.zones}
    dispatcher = NotificationDispatcher.from_config(cfg.alerts.notifiers) if notify else None
    monitor = ServiceMonitor(cfg, zone_map, EventStore(cfg.storage.db_path), dispatcher)
    media_start = time.time() if live else (start_time or datetime.now().replace(hour=18, minute=0, second=0)).timestamp()
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

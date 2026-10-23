import sqlite3
from datetime import date, datetime

import pytest

from tablewatch.analytics.history import (
    by_day,
    by_hour,
    by_table,
    occupancy_frame,
    reset_times,
    service_date,
    sittings_frame,
    summary,
)
from tablewatch.config import AppConfig, ServedConfig
from tablewatch.pipeline import ServiceMonitor
from tablewatch.simulate import TablePlan, default_zones, simulate_service
from tablewatch.state.served import DRINKS, FOOD, ItemMonitor
from tablewatch.state.table_state import EventType, Sitting, Visit
from tablewatch.storage.db import EventStore

FRI_6PM = datetime(2026, 10, 23, 18, 0)


def drive(monitor: ServiceMonitor, plans: dict[str, TablePlan], until: float) -> list:
    events = []
    ts = 0.0
    while ts <= until:
        counts = {tid: p.counts(ts) for tid, p in plans.items()}
        ev, _, _ = monitor.update(
            ts, {t: c[0] for t, c in counts.items()}, {t: c[1] for t, c in counts.items()},
            item_counts={t: c[2] for t, c in counts.items()},
        )
        events += ev
        ts += 1.0
    monitor.finish(until)
    return events


def one_scripted_sitting() -> TablePlan:
    """Seated 60 s in for an hour: order at 4 min, drinks at 10, food at 25, cleared 41 s after leaving."""
    return TablePlan(
        guests=[(60, 3660, 2)],
        staff=[(240, 300), (600, 660), (1500, 1560), (3700, 3760)],
        items=[(615, 3700, DRINKS, 2), (1515, 3000, FOOD, 2)],
    )


@pytest.fixture
def scripted_store(tmp_path):
    store = EventStore(tmp_path / "t.db")
    cfg = AppConfig(site="Bistro")
    monitor = ServiceMonitor(cfg, default_zones(1), store)
    monitor.start_run("script", is_live=False, fps=1.0, media_start=FRI_6PM.timestamp())
    events = drive(monitor, {"T1": one_scripted_sitting()}, 3900)
    return store, events


def test_drinks_and_food_detected_after_delivery_visits(scripted_store):
    _, events = scripted_store
    served = {e.type: e.ts for e in events if e.type in (EventType.DRINKS_SERVED, EventType.FOOD_SERVED)}
    assert 615 <= served[EventType.DRINKS_SERVED] <= 625
    assert 1515 <= served[EventType.FOOD_SERVED] <= 1525


def test_sittings_frame_end_to_end(scripted_store):
    store, _ = scripted_store
    df = sittings_frame(store)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["site"] == "Bistro" and row["table_id"] == "T1"
    assert row["seated"] == datetime(2026, 10, 23, 18, 1)
    assert (row["service_date"], row["weekday"], row["hour"]) == (date(2026, 10, 23), "Fri", 18)
    assert not row["open"]
    assert row["turn_time_min"] == pytest.approx(60, abs=0.1)
    assert row["first_visit_min"] == pytest.approx(3, abs=0.05)
    assert row["drinks_min"] == pytest.approx(9.3, abs=0.2)
    assert row["food_min"] == pytest.approx(24.3, abs=0.2)
    assert row["order_to_food_min"] == pytest.approx(21.3, abs=0.2)
    assert row["reset_min"] == pytest.approx(41 / 60, abs=0.05)
    assert row["visits"] == 3
    assert row["alerts"] == 2  # 14 min from drinks to food, then 35 min from food until they leave


def test_summary_and_rollups(scripted_store):
    store, _ = scripted_store
    df, occ = sittings_frame(store), occupancy_frame(store)
    s = summary(df, occ)
    assert s["sittings"] == 1 and s["services"] == 1
    assert s["sittings_with_alert_pct"] == 100
    assert s["utilisation_pct"] == pytest.approx(3599 / 3900 * 100, abs=0.5)
    assert by_day(df).iloc[0]["sittings"] == 1
    assert list(by_hour(df)["hour"]) == [18]
    t = by_table(df, occ).iloc[0]
    assert t["utilisation_pct"] == pytest.approx(s["utilisation_pct"])


def test_filters_by_site_and_date(scripted_store):
    store, _ = scripted_store
    assert len(sittings_frame(store, sites=["Elsewhere"])) == 0
    assert len(sittings_frame(store, start=date(2026, 10, 24))) == 0
    assert len(sittings_frame(store, start=date(2026, 10, 23), end=date(2026, 10, 23))) == 1


def test_service_date_rolls_late_sittings_into_the_evening():
    assert service_date(datetime(2026, 10, 24, 1, 30)) == date(2026, 10, 23)
    assert service_date(datetime(2026, 10, 24, 11, 0)) == date(2026, 10, 24)


def test_reset_time_uses_first_clearing_visit_before_next_guests():
    sittings = [Sitting("T1-S1", "T1", 0, ended_at=1000), Sitting("T1-S2", "T1", 1500, ended_at=None)]
    visits = [
        Visit("T1-V1", "T1", 100, 150, "T1-S1"),
        Visit("T1-V2", "T1", 1060, 1100, None),  # clearing after S1
        Visit("T1-V3", "T1", 1200, 1250, None),
        Visit("T2-V1", "T2", 1010, 1050, None),  # another table
    ]
    assert reset_times(sittings, visits) == {"T1-S1": 60}


def test_item_monitor_needs_a_recent_visit_and_fires_once():
    cfg = ServedConfig(window_s=4, visit_window_s=60)
    m = ItemMonitor(cfg)
    for t in range(10):
        assert not m.update(t, 1, "S1", visit_count=0, last_staff_ts=None)  # a glass already on the table
    # Items appear 5 minutes after the last visit: not attributed to service.
    assert not any(m.update(t, 3, "S1", visit_count=1, last_staff_ts=10) for t in range(400, 410))
    m2 = ItemMonitor(cfg)
    for t in range(10):
        m2.update(t, 1, "S1", visit_count=0, last_staff_ts=None)
    fired = [t for t in range(20, 40) if m2.update(t, 2, "S1", visit_count=1, last_staff_ts=t)]
    assert len(fired) == 1
    assert m2.update(60, 2, "S2", visit_count=0, last_staff_ts=None) is False  # new sitting resets


def test_simulated_week_day_has_drinks_and_food(tmp_path):
    store = EventStore(tmp_path / "sim.db")
    simulate_service(AppConfig(), store, FRI_6PM, tables=3, hours=2.0, seed=3)
    df = sittings_frame(store)
    closed = df[~df["open"]]  # a sitting cut off by the end of the evening may not have been served yet
    assert len(closed) >= 3
    assert closed["drinks_min"].notna().all()
    assert closed["food_min"].notna().mean() > 0.5
    assert (closed["food_min"].dropna() > closed["drinks_min"].dropna().min()).all()


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:  # the first release's tables, without site / drinks columns
        conn.executescript(
            """
            CREATE TABLE runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, is_live INTEGER NOT NULL,
                fps REAL, started_wall REAL NOT NULL, media_start REAL NOT NULL, last_ts REAL DEFAULT 0, ended_wall REAL,
                tables_json TEXT, config_json TEXT);
            CREATE TABLE sittings (run_id INTEGER NOT NULL, sitting_key TEXT NOT NULL, table_id TEXT NOT NULL,
                seated_at REAL NOT NULL, ended_at REAL, guests_max INTEGER, food_served_at REAL,
                PRIMARY KEY (run_id, sitting_key));
            INSERT INTO runs (source, is_live, started_wall, media_start, tables_json) VALUES ('old.mp4', 0, 0, 0, '[]');
            INSERT INTO sittings VALUES (1, 'T1-S1', 'T1', 5, 60, 2, NULL);
            """
        )
    ro = EventStore(path, read_only=True)  # read-only: not migrated, but readable
    assert ro.runs()[0].site == "default"
    assert ro.sittings(1)[0].drinks_served_at is None
    ro.close()
    EventStore(path).close()
    with sqlite3.connect(path) as conn:
        assert "site" in {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        assert "drinks_served_at" in {r[1] for r in conn.execute("PRAGMA table_info(sittings)")}

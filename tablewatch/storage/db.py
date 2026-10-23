"""SQLite persistence for events, sittings, visits, alerts and live table status.

Only derived events and metrics are stored: no frames, crops or faces. The
database runs in WAL mode so the dashboard can read while the pipeline writes.
All `ts` columns are seconds since the run started; `runs.media_start` maps
them to wall-clock time.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tablewatch.alerts.rules import Alert
from tablewatch.state.table_state import EventType, Sitting, TableEvent, TableSnapshot, Visit

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,
    is_live      INTEGER NOT NULL,
    fps          REAL,
    started_wall REAL NOT NULL,   -- epoch seconds when processing started
    media_start  REAL NOT NULL,   -- epoch seconds corresponding to ts = 0
    last_ts      REAL DEFAULT 0,  -- latest processed timestamp
    ended_wall   REAL,
    tables_json  TEXT,            -- [{"table_id": ..., "seats": ...}]
    config_json  TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(run_id),
    ts          REAL NOT NULL,
    table_id    TEXT NOT NULL,
    type        TEXT NOT NULL,
    sitting_key TEXT,
    visit_key   TEXT,
    data_json   TEXT
);
CREATE TABLE IF NOT EXISTS sittings (
    run_id         INTEGER NOT NULL REFERENCES runs(run_id),
    sitting_key    TEXT NOT NULL,
    table_id       TEXT NOT NULL,
    seated_at      REAL NOT NULL,
    ended_at       REAL,
    guests_max     INTEGER,
    food_served_at REAL,
    PRIMARY KEY (run_id, sitting_key)
);
CREATE TABLE IF NOT EXISTS visits (
    run_id      INTEGER NOT NULL REFERENCES runs(run_id),
    visit_key   TEXT NOT NULL,
    table_id    TEXT NOT NULL,
    sitting_key TEXT,
    start_ts    REAL NOT NULL,
    end_ts      REAL,
    PRIMARY KEY (run_id, visit_key)
);
CREATE TABLE IF NOT EXISTS alerts (
    run_id       INTEGER NOT NULL REFERENCES runs(run_id),
    alert_key    TEXT NOT NULL,
    rule         TEXT NOT NULL,
    table_id     TEXT NOT NULL,
    sitting_key  TEXT,
    severity     TEXT,
    message      TEXT,
    ts           REAL NOT NULL,
    value_s      REAL,
    threshold_s  REAL,
    resolved_ts  REAL,
    notify_count INTEGER DEFAULT 1,
    PRIMARY KEY (run_id, alert_key)
);
CREATE TABLE IF NOT EXISTS table_status (
    run_id            INTEGER NOT NULL REFERENCES runs(run_id),
    table_id          TEXT NOT NULL,
    ts                REAL NOT NULL,
    state             TEXT NOT NULL,
    state_since       REAL,
    guests            INTEGER,
    staff             INTEGER,
    sitting_key       TEXT,
    seated_at         REAL,
    visit_count       INTEGER,
    visit_started_at  REAL,
    last_visit_end    REAL,
    empty_since       REAL,
    food_served_at    REAL,
    updated_wall      REAL,
    PRIMARY KEY (run_id, table_id)
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_alerts_run ON alerts(run_id, ts);
"""


@dataclass
class RunInfo:
    run_id: int
    source: str
    is_live: bool
    fps: float | None
    started_wall: float
    media_start: float
    last_ts: float
    ended_wall: float | None
    tables: list[dict[str, Any]]

    @property
    def is_running(self) -> bool:
        return self.ended_wall is None


class EventStore:
    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = str(path)
        if read_only:
            self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    # -- writing (pipeline) --------------------------------------------------

    def start_run(
        self,
        source: str,
        is_live: bool,
        fps: float | None,
        tables: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
        media_start: float | None = None,
    ) -> int:
        now = time.time()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO runs (source, is_live, fps, started_wall, media_start, tables_json, config_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (source, int(is_live), fps, now, media_start or now, json.dumps(tables), json.dumps(config or {})),
            )
        return int(cur.lastrowid)

    def end_run(self, run_id: int, last_ts: float) -> None:
        with self.conn:
            self.conn.execute("UPDATE runs SET ended_wall = ?, last_ts = ? WHERE run_id = ?", (time.time(), last_ts, run_id))

    def record_events(self, run_id: int, events: Iterable[TableEvent]) -> None:
        with self.conn:
            for e in events:
                self._record_event(run_id, e)

    def _record_event(self, run_id: int, e: TableEvent) -> None:
        c = self.conn
        c.execute(
            "INSERT INTO events (run_id, ts, table_id, type, sitting_key, visit_key, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, e.ts, e.table_id, e.type.value, e.sitting_key, e.visit_key, json.dumps(e.data) if e.data else None),
        )
        if e.type == EventType.SITTING_STARTED:
            c.execute(
                "INSERT OR REPLACE INTO sittings (run_id, sitting_key, table_id, seated_at, guests_max) VALUES (?, ?, ?, ?, ?)",
                (run_id, e.sitting_key, e.table_id, e.ts, e.data.get("guests")),
            )
        elif e.type == EventType.SITTING_ENDED:
            c.execute(
                "UPDATE sittings SET ended_at = ?, guests_max = ? WHERE run_id = ? AND sitting_key = ?",
                (e.ts, e.data.get("guests_max"), run_id, e.sitting_key),
            )
        elif e.type == EventType.FOOD_SERVED:
            c.execute(
                "UPDATE sittings SET food_served_at = ? WHERE run_id = ? AND sitting_key = ?",
                (e.ts, run_id, e.sitting_key),
            )
        elif e.type == EventType.VISIT_STARTED:
            c.execute(
                "INSERT OR REPLACE INTO visits (run_id, visit_key, table_id, sitting_key, start_ts) VALUES (?, ?, ?, ?, ?)",
                (run_id, e.visit_key, e.table_id, e.sitting_key, e.ts),
            )
        elif e.type == EventType.VISIT_ENDED:
            c.execute(
                "UPDATE visits SET end_ts = ?, sitting_key = ? WHERE run_id = ? AND visit_key = ?",
                (e.ts, e.sitting_key, run_id, e.visit_key),
            )
        elif e.type == EventType.VISIT_REASSIGNED:
            c.execute(
                "UPDATE visits SET sitting_key = ? WHERE run_id = ? AND visit_key = ?",
                (e.sitting_key, run_id, e.visit_key),
            )

    def record_alert(self, run_id: int, alert: Alert) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO alerts (run_id, alert_key, rule, table_id, sitting_key, severity, message, ts,
                                       value_s, threshold_s, resolved_ts, notify_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (run_id, alert_key) DO UPDATE SET
                       message = excluded.message, value_s = excluded.value_s,
                       resolved_ts = excluded.resolved_ts, notify_count = excluded.notify_count""",
                (
                    run_id, alert.key, alert.rule, alert.table_id, alert.sitting_key, alert.severity, alert.message,
                    alert.ts, alert.value_s, alert.threshold_s, alert.resolved_ts, alert.notify_count,
                ),
            )

    def write_status(self, run_id: int, snapshots: Iterable[TableSnapshot], last_ts: float) -> None:
        now = time.time()
        with self.conn:
            for s in snapshots:
                self.conn.execute(
                    """INSERT OR REPLACE INTO table_status
                       (run_id, table_id, ts, state, state_since, guests, staff, sitting_key, seated_at, visit_count,
                        visit_started_at, last_visit_end, empty_since, food_served_at, updated_wall)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id, s.table_id, s.ts, s.state.value, s.state_since, s.guests, s.staff, s.sitting_key,
                        s.seated_at, s.visit_count, s.visit_started_at, s.last_visit_end, s.empty_since,
                        s.food_served_at, now,
                    ),
                )
            self.conn.execute("UPDATE runs SET last_ts = ? WHERE run_id = ?", (last_ts, run_id))

    # -- reading (dashboard, reports) ------------------------------------------

    def runs(self) -> list[RunInfo]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY run_id DESC").fetchall()
        return [
            RunInfo(
                run_id=r["run_id"], source=r["source"], is_live=bool(r["is_live"]), fps=r["fps"],
                started_wall=r["started_wall"], media_start=r["media_start"], last_ts=r["last_ts"] or 0.0,
                ended_wall=r["ended_wall"], tables=json.loads(r["tables_json"] or "[]"),
            )
            for r in rows
        ]

    def run(self, run_id: int) -> RunInfo | None:
        return next((r for r in self.runs() if r.run_id == run_id), None)

    def sittings(self, run_id: int) -> list[Sitting]:
        rows = self.conn.execute("SELECT * FROM sittings WHERE run_id = ? ORDER BY seated_at", (run_id,)).fetchall()
        return [
            Sitting(
                key=r["sitting_key"], table_id=r["table_id"], seated_at=r["seated_at"], ended_at=r["ended_at"],
                guests_max=r["guests_max"] or 0, food_served_at=r["food_served_at"],
            )
            for r in rows
        ]

    def visits(self, run_id: int) -> list[Visit]:
        rows = self.conn.execute("SELECT * FROM visits WHERE run_id = ? ORDER BY start_ts", (run_id,)).fetchall()
        return [
            Visit(key=r["visit_key"], table_id=r["table_id"], start=r["start_ts"], end=r["end_ts"], sitting_key=r["sitting_key"])
            for r in rows
        ]

    def alerts(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM alerts WHERE run_id = ? ORDER BY ts DESC", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def table_status(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM table_status WHERE run_id = ? ORDER BY table_id", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def events(self, run_id: int, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events WHERE run_id = ? ORDER BY ts DESC, id DESC"
        args: tuple = (run_id,)
        if limit:
            sql += " LIMIT ?"
            args += (limit,)
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

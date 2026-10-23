"""Per-table service state machine.

Two independent lifecycles are tracked per table and the visible state is
derived from them:

* a **sitting** starts when guests have been present for `seat_confirm_s` and
  ends when they have been gone for `empty_confirm_s`;
* a **visit** starts when staff have been present for `visit_min_s` and ends
  when they have been gone for `visit_end_grace_s`.

    EMPTY ──guests settle──▶ SEATED ──staff arrive──▶ ATTENDED ──staff leave──▶ WAITING
      ▲                                                  ▲                        │
      │                                                  └──────staff arrive──────┘
      └────────────────── guests gone (from any non-empty state) ─────────────────┘

Start and end times are backdated to when presence actually began/ended, so
debounce delays don't distort the metrics. A visit counts toward a sitting
only if it started at or after the guests sat down (a host escorting guests to
the table is not their first service visit).

The machine is pure Python and driven only by timestamps, so it behaves the
same on recorded video, live streams and in unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tablewatch.config import StateConfig


class TableState(str, Enum):
    EMPTY = "EMPTY"
    SEATED = "SEATED"  # guests seated, no staff visit yet
    ATTENDED = "ATTENDED"  # staff at the table
    WAITING = "WAITING"  # visited before, staff currently away


class EventType(str, Enum):
    STATE_CHANGED = "state_changed"
    SITTING_STARTED = "sitting_started"
    SITTING_ENDED = "sitting_ended"
    VISIT_STARTED = "visit_started"
    VISIT_ENDED = "visit_ended"
    VISIT_REASSIGNED = "visit_reassigned"  # a visit turned out not to belong to the sitting
    FOOD_SERVED = "food_served"


@dataclass
class TableEvent:
    type: EventType
    table_id: str
    ts: float
    sitting_key: str | None = None
    visit_key: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Visit:
    key: str
    table_id: str
    start: float
    end: float | None = None
    sitting_key: str | None = None  # None: staff at an empty table (reset / bussing)

    def duration(self, now: float | None = None) -> float | None:
        end = self.end if self.end is not None else now
        return None if end is None else max(0.0, end - self.start)


@dataclass
class Sitting:
    key: str
    table_id: str
    seated_at: float
    ended_at: float | None = None
    guests_max: int = 0
    food_served_at: float | None = None
    visits: list[Visit] = field(default_factory=list)

    @property
    def first_visit_at(self) -> float | None:
        return self.visits[0].start if self.visits else None

    @property
    def last_visit_end(self) -> float | None:
        ends = [v.end for v in self.visits if v.end is not None]
        return max(ends) if ends else None


@dataclass
class TableSnapshot:
    """Point-in-time view of one table, for overlays, alerts and the dashboard."""

    table_id: str
    ts: float
    state: TableState
    state_since: float
    guests: int = 0
    staff: int = 0
    sitting_key: str | None = None
    seated_at: float | None = None
    visit_count: int = 0
    visit_started_at: float | None = None
    last_visit_end: float | None = None
    empty_since: float | None = None
    food_served_at: float | None = None

    @property
    def time_in_state(self) -> float:
        return max(0.0, self.ts - self.state_since)

    @property
    def sitting_duration(self) -> float | None:
        return None if self.seated_at is None else max(0.0, self.ts - self.seated_at)

    @property
    def waiting_for_staff_s(self) -> float | None:
        """How long the guests have been without staff (None when not applicable)."""
        if self.state == TableState.SEATED and self.seated_at is not None:
            return max(0.0, self.ts - self.seated_at)
        if self.state == TableState.WAITING and self.last_visit_end is not None:
            return max(0.0, self.ts - self.last_visit_end)
        return None


class TableStateMachine:
    def __init__(self, table_id: str, cfg: StateConfig | None = None, start_ts: float = 0.0):
        self.table_id = table_id
        self.cfg = cfg or StateConfig()
        self.state = TableState.EMPTY
        self.state_since = start_ts
        self.empty_since: float | None = start_ts
        self.sitting: Sitting | None = None
        self.visit: Visit | None = None
        self.guests = 0
        self.staff = 0
        self._guest_first: float | None = None
        self._guest_last: float | None = None
        self._staff_first: float | None = None
        self._staff_last: float | None = None
        self._last_visit_end: float | None = None
        self._n_sittings = 0
        self._n_visits = 0

    # -- public API ---------------------------------------------------------

    def update(self, ts: float, guests: int, staff: int = 0, staff_since: float | None = None) -> list[TableEvent]:
        """Feed the current head-count for this table.

        `staff_since` is the earliest time any staff member currently at the
        table entered the zone; it backdates a visit when a person is only
        recognised as staff after they arrived.
        """
        cfg = self.cfg
        events: list[TableEvent] = []
        self.guests, self.staff = guests, staff

        if guests > 0:
            if self._guest_last is None or ts - self._guest_last > cfg.guest_gap_tolerance_s:
                self._guest_first = ts
            self._guest_last = ts
            if self.sitting:
                self.sitting.guests_max = max(self.sitting.guests_max, guests)

        if staff > 0:
            if self._staff_first is None or ts - self._staff_last > cfg.visit_end_grace_s:
                first = ts if staff_since is None else min(ts, staff_since)
                if self._last_visit_end is not None:
                    first = max(first, self._last_visit_end)  # visits never overlap
                self._staff_first = first
            self._staff_last = ts

        if self.sitting is None and guests > 0 and ts - self._guest_first >= cfg.seat_confirm_s:
            self._start_sitting(self._guest_first, ts, events)

        if self.visit is None and staff > 0 and ts - self._staff_first >= cfg.visit_min_s:
            self._start_visit(self._staff_first, events)

        if self.visit is not None and staff == 0 and ts - self._staff_last >= cfg.visit_end_grace_s:
            self._end_visit(self._staff_last, events)

        if self.sitting is not None and ts - self._guest_last >= cfg.empty_confirm_s:
            self._close_sitting(staff_present=staff > 0, events=events)

        return events

    def mark_food_served(self, ts: float) -> list[TableEvent]:
        if self.sitting is None or self.sitting.food_served_at is not None:
            return []
        self.sitting.food_served_at = ts
        return [TableEvent(EventType.FOOD_SERVED, self.table_id, ts, self.sitting.key)]

    def snapshot(self, ts: float) -> TableSnapshot:
        s, v = self.sitting, self.visit
        attached_visit = v if (v and s and v.sitting_key == s.key) else None
        return TableSnapshot(
            table_id=self.table_id,
            ts=ts,
            state=self.state,
            state_since=self.state_since,
            guests=self.guests,
            staff=self.staff,
            sitting_key=s.key if s else None,
            seated_at=s.seated_at if s else None,
            visit_count=len(s.visits) if s else 0,
            visit_started_at=attached_visit.start if attached_visit else None,
            last_visit_end=s.last_visit_end if s else None,
            empty_since=self.empty_since if not s else None,
            food_served_at=s.food_served_at if s else None,
        )

    # -- transitions ---------------------------------------------------------

    def _start_sitting(self, seated_at: float, confirmed_at: float, events: list[TableEvent]) -> None:
        self._n_sittings += 1
        sitting = Sitting(f"{self.table_id}-S{self._n_sittings}", self.table_id, seated_at, guests_max=self.guests)
        self.sitting = sitting
        idle = seated_at - self.empty_since if self.empty_since is not None else None
        self.empty_since = None
        events.append(
            TableEvent(
                EventType.SITTING_STARTED, self.table_id, seated_at, sitting.key,
                data={"confirmed_at": confirmed_at, "guests": self.guests, "idle_before_s": idle},
            )
        )
        if self.visit is not None and self.visit.sitting_key is None and self.visit.start >= seated_at:
            self.visit.sitting_key = sitting.key  # staff arrived together with the guests
            sitting.visits.append(self.visit)
            events.append(
                TableEvent(
                    EventType.VISIT_REASSIGNED, self.table_id, self.visit.start, sitting.key, self.visit.key,
                    data={"from_sitting": None},
                )
            )
        self._sync_state(seated_at, events)

    def _close_sitting(self, staff_present: bool, events: list[TableEvent]) -> None:
        """Guests have gone: end the sitting and split off any visit still in progress."""
        left_at = self._guest_last
        visit = self.visit
        restart_visit = False
        if visit is not None and visit.sitting_key == self.sitting.key:
            if visit.start > left_at:
                # Staff arrived after the guests had left: clearing up, not service.
                self.sitting.visits.remove(visit)
                visit.sitting_key = None
                events.append(
                    TableEvent(
                        EventType.VISIT_REASSIGNED, self.table_id, visit.start, None, visit.key,
                        data={"from_sitting": self.sitting.key},
                    )
                )
            else:
                # Service visit ends when the guests leave; anything after is clearing up.
                self._end_visit(min(left_at, self._staff_last), events, sync=False)
                restart_visit = staff_present
        self._end_sitting(left_at, events)
        if restart_visit:
            self._start_visit(left_at, events)

    def _end_sitting(self, ended_at: float, events: list[TableEvent]) -> None:
        sitting = self.sitting
        sitting.ended_at = ended_at
        self.sitting = None
        self.empty_since = ended_at
        events.append(
            TableEvent(
                EventType.SITTING_ENDED, self.table_id, ended_at, sitting.key,
                data={
                    "duration_s": ended_at - sitting.seated_at,
                    "visit_count": len(sitting.visits),
                    "guests_max": sitting.guests_max,
                },
            )
        )
        self._sync_state(ended_at, events)

    def _start_visit(self, start: float, events: list[TableEvent]) -> None:
        self._n_visits += 1
        sitting = self.sitting if (self.sitting and start >= self.sitting.seated_at) else None
        visit = Visit(f"{self.table_id}-V{self._n_visits}", self.table_id, start, sitting_key=sitting.key if sitting else None)
        self.visit = visit
        if sitting:
            sitting.visits.append(visit)
        events.append(TableEvent(EventType.VISIT_STARTED, self.table_id, start, visit.sitting_key, visit.key))
        self._sync_state(start, events)

    def _end_visit(self, end: float, events: list[TableEvent], sync: bool = True) -> None:
        visit = self.visit
        visit.end = max(end, visit.start)
        self.visit = None
        self._last_visit_end = visit.end
        events.append(
            TableEvent(
                EventType.VISIT_ENDED, self.table_id, visit.end, visit.sitting_key, visit.key,
                data={"duration_s": visit.end - visit.start},
            )
        )
        if sync:
            self._sync_state(visit.end, events)

    def _derive_state(self) -> TableState:
        if self.sitting is None:
            return TableState.EMPTY
        if self.visit is not None and self.visit.sitting_key == self.sitting.key:
            return TableState.ATTENDED
        return TableState.WAITING if self.sitting.visits else TableState.SEATED

    def _sync_state(self, ts: float, events: list[TableEvent]) -> None:
        new = self._derive_state()
        if new == self.state:
            return
        old, self.state, self.state_since = self.state, new, ts
        events.append(
            TableEvent(
                EventType.STATE_CHANGED, self.table_id, ts,
                self.sitting.key if self.sitting else None,
                data={"from": old.value, "to": new.value},
            )
        )

"""Timeline helpers: feed a state machine scripted head-counts at a fixed frame rate."""

from __future__ import annotations

from tablewatch.config import StateConfig
from tablewatch.state.table_state import TableEvent, TableStateMachine

STEP = 0.5  # seconds between simulated frames (2 fps)

FAST = StateConfig(
    seat_confirm_s=10.0,
    empty_confirm_s=30.0,
    guest_gap_tolerance_s=2.0,
    visit_min_s=2.0,
    visit_end_grace_s=3.0,
)


def run_timeline(
    sm: TableStateMachine,
    segments: list[tuple[float, float, int, int]],
    step: float = STEP,
    staff_since: float | None = None,
) -> list[TableEvent]:
    """Each segment is (start_s, end_s, guests, staff); end is exclusive."""
    events: list[TableEvent] = []
    for start, end, guests, staff in segments:
        t = start
        while t < end - 1e-9:
            events.extend(sm.update(t, guests, staff, staff_since if staff else None))
            t = round(t + step, 6)
    return events


def of_type(events: list[TableEvent], type_) -> list[TableEvent]:
    return [e for e in events if e.type == type_]

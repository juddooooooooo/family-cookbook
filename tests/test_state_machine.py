import pytest
from helpers import FAST, of_type, run_timeline

from tablewatch.state.table_state import EventType, TableState, TableStateMachine


def make() -> TableStateMachine:
    return TableStateMachine("T1", FAST, start_ts=0.0)


def test_starts_empty():
    sm = make()
    assert sm.state == TableState.EMPTY
    assert sm.snapshot(0).empty_since == 0.0


def test_brief_guest_presence_does_not_seat():
    sm = make()
    run_timeline(sm, [(0, 8, 2, 0), (8, 40, 0, 0)])  # 8 s < seat_confirm_s
    assert sm.state == TableState.EMPTY


def test_seated_after_confirm_and_backdated():
    sm = make()
    events = run_timeline(sm, [(5, 20, 2, 0)])
    assert sm.state == TableState.SEATED
    (started,) = of_type(events, EventType.SITTING_STARTED)
    assert started.ts == 5.0  # when guests arrived, not when confirmed
    assert started.data["confirmed_at"] == pytest.approx(15.0)
    assert started.data["idle_before_s"] == pytest.approx(5.0)
    assert sm.snapshot(20).seated_at == 5.0


def test_detection_dropouts_shorter_than_tolerance_do_not_reset_seating():
    sm = make()
    run_timeline(sm, [(0, 4, 2, 0), (4, 5, 0, 0), (5, 11, 2, 0)])  # 1 s dropout
    assert sm.state == TableState.SEATED
    assert sm.sitting.seated_at == 0.0


def test_full_service_cycle():
    sm = make()
    events = run_timeline(
        sm,
        [
            (0, 60, 2, 0),  # guests sit
            (60, 90, 2, 1),  # first visit 60-90
            (90, 300, 2, 0),  # wait
            (300, 320, 2, 1),  # second visit 300-320
            (320, 400, 2, 0),
            (400, 450, 0, 0),  # guests leave at 400
        ],
    )
    transitions = [(e.data["from"], e.data["to"]) for e in of_type(events, EventType.STATE_CHANGED)]
    assert transitions == [
        ("EMPTY", "SEATED"),
        ("SEATED", "ATTENDED"),
        ("ATTENDED", "WAITING"),
        ("WAITING", "ATTENDED"),
        ("ATTENDED", "WAITING"),
        ("WAITING", "EMPTY"),
    ]
    visits = of_type(events, EventType.VISIT_ENDED)
    assert [(v.ts, v.data["duration_s"]) for v in visits] == [
        (pytest.approx(89.5), pytest.approx(29.5)),
        (pytest.approx(319.5), pytest.approx(19.5)),
    ]
    assert all(v.sitting_key == "T1-S1" for v in visits)
    (ended,) = of_type(events, EventType.SITTING_ENDED)
    assert ended.ts == pytest.approx(399.5)  # last time guests were seen
    assert ended.data["visit_count"] == 2
    assert sm.state == TableState.EMPTY
    assert sm.snapshot(450).empty_since == pytest.approx(399.5)


def test_staff_pass_by_shorter_than_visit_min_is_ignored():
    sm = make()
    events = run_timeline(sm, [(0, 30, 2, 0), (30, 31, 2, 1), (31, 60, 2, 0)])
    assert of_type(events, EventType.VISIT_STARTED) == []
    assert sm.state == TableState.SEATED


def test_brief_staff_dropout_keeps_single_visit():
    sm = make()
    events = run_timeline(sm, [(0, 20, 2, 0), (20, 30, 2, 1), (30, 31.5, 2, 0), (31.5, 40, 2, 1), (40, 60, 2, 0)])
    assert len(of_type(events, EventType.VISIT_STARTED)) == 1
    (end,) = of_type(events, EventType.VISIT_ENDED)
    assert end.data["duration_s"] == pytest.approx(19.5)


def test_staff_recognised_late_backdates_visit_start():
    sm = make()
    run_timeline(sm, [(0, 30, 2, 0)])
    # A person who walked up at t=30 is only classified as staff from t=40.
    events = run_timeline(sm, [(40, 50, 2, 1)], staff_since=30.0)
    (start,) = of_type(events, EventType.VISIT_STARTED)
    assert start.ts == 30.0
    assert sm.state == TableState.ATTENDED


def test_backdating_never_overlaps_previous_visit():
    sm = make()
    run_timeline(sm, [(0, 20, 2, 0), (20, 30, 2, 1), (30, 60, 2, 0)])
    events = run_timeline(sm, [(60, 70, 2, 1)], staff_since=10.0)
    (start,) = of_type(events, EventType.VISIT_STARTED)
    assert start.ts == pytest.approx(29.5)  # end of the previous visit, not 10


def test_guests_occluded_by_staff_do_not_end_sitting():
    sm = make()
    run_timeline(sm, [(0, 20, 2, 0), (20, 45, 0, 1), (45, 60, 2, 0)])  # 25 s < empty_confirm_s
    assert sm.state == TableState.WAITING
    assert sm.sitting.key == "T1-S1"


def test_staff_at_empty_table_is_not_a_sitting_visit():
    sm = make()
    events = run_timeline(sm, [(0, 20, 0, 1), (20, 40, 0, 0)])
    (visit,) = of_type(events, EventType.VISIT_ENDED)
    assert visit.sitting_key is None
    assert sm.state == TableState.EMPTY
    assert of_type(events, EventType.STATE_CHANGED) == []


def test_host_escort_before_seating_is_not_first_visit():
    sm = make()
    # Host walks guests over and leaves; guests counted from t=2, host from t=0.
    events = run_timeline(sm, [(0, 2, 0, 1), (2, 6, 2, 1), (6, 60, 2, 0)])
    assert sm.state == TableState.SEATED
    assert sm.sitting.visits == []
    assert of_type(events, EventType.VISIT_ENDED)[0].sitting_key is None


def test_staff_arriving_with_guests_counts_once_seated():
    sm = make()
    # Guests from t=0, staff arrive at t=4 (after guests) before the sitting is confirmed at t=10.
    events = run_timeline(sm, [(0, 4, 2, 0), (4, 20, 2, 1), (20, 40, 2, 0)])
    assert len(sm.sitting.visits) == 1
    assert sm.sitting.visits[0].start == 4.0
    reassigned = of_type(events, EventType.VISIT_REASSIGNED)
    assert reassigned and reassigned[0].sitting_key == "T1-S1"
    assert sm.state == TableState.WAITING


def test_clearing_after_guests_leave_is_detached_from_sitting():
    sm = make()
    events = run_timeline(
        sm,
        [
            (0, 20, 2, 0),
            (20, 30, 2, 1),  # service visit
            (30, 100, 2, 0),
            (100, 110, 0, 0),  # guests leave at ~100
            (110, 150, 0, 1),  # staff clear the table from 110
        ],
    )
    (ended,) = of_type(events, EventType.SITTING_ENDED)
    assert ended.data["visit_count"] == 1
    reassigned = of_type(events, EventType.VISIT_REASSIGNED)
    assert reassigned and reassigned[0].sitting_key is None
    assert sm.state == TableState.EMPTY
    assert sm.visit is not None and sm.visit.sitting_key is None


def test_visit_spanning_departure_is_split_at_departure():
    sm = make()
    events = run_timeline(sm, [(0, 20, 2, 0), (20, 50, 2, 1), (50, 120, 0, 1), (120, 140, 0, 0)])
    ends = of_type(events, EventType.VISIT_ENDED)
    assert ends[0].sitting_key == "T1-S1"
    assert ends[0].ts == pytest.approx(49.5)  # service visit ends when guests left
    assert ends[1].sitting_key is None  # remainder is clearing up
    assert ends[1].ts == pytest.approx(119.5)


def test_new_sitting_after_empty_gets_new_key_and_idle_time():
    sm = make()
    events = run_timeline(sm, [(0, 20, 2, 0), (20, 60, 0, 0), (60, 80, 3, 0)])
    starts = of_type(events, EventType.SITTING_STARTED)
    assert [s.sitting_key for s in starts] == ["T1-S1", "T1-S2"]
    assert starts[1].data["idle_before_s"] == pytest.approx(60 - 19.5)
    assert sm.sitting.guests_max == 3


def test_snapshot_timers():
    sm = make()
    run_timeline(sm, [(0, 20, 2, 0), (20, 30, 2, 1), (30, 100, 2, 0)])
    snap = sm.snapshot(100)
    assert snap.state == TableState.WAITING
    assert snap.visit_count == 1
    assert snap.last_visit_end == pytest.approx(29.5)
    assert snap.waiting_for_staff_s == pytest.approx(70.5)
    assert snap.sitting_duration == pytest.approx(100)
    assert snap.time_in_state == pytest.approx(70.5)


def test_served_milestones_marked_once_per_sitting():
    sm = make()
    run_timeline(sm, [(0, 20, 2, 0)])
    (drinks,) = sm.mark_served("drinks", 22)
    assert drinks.type == EventType.DRINKS_SERVED and drinks.sitting_key == "T1-S1"
    assert len(sm.mark_served("food", 25)) == 1
    assert sm.mark_served("food", 30) == []
    snap = sm.snapshot(30)
    assert (snap.drinks_served_at, snap.food_served_at) == (22, 25)


def test_served_ignored_without_a_sitting():
    assert make().mark_served("food", 5) == []

import pytest

from tablewatch.state.metrics import idle_periods, sitting_metrics, state_at, table_summary
from tablewatch.state.table_state import Sitting, TableState, TableStateMachine, Visit

from helpers import FAST, run_timeline


def records():
    sittings = [
        Sitting("T1-S1", "T1", seated_at=100, ended_at=1000),
        Sitting("T1-S2", "T1", seated_at=1300, ended_at=None),  # still open
    ]
    visits = [
        Visit("T1-V1", "T1", 160, 200, "T1-S1"),
        Visit("T1-V2", "T1", 500, 530, "T1-S1"),
        Visit("T1-V3", "T1", 800, 820, "T1-S1"),
        Visit("T1-V4", "T1", 1010, 1060, None),  # clearing, not service
        Visit("T1-V5", "T1", 1400, None, "T1-S2"),  # in progress
    ]
    return sittings, visits


def test_sitting_metrics_closed():
    sittings, visits = records()
    m = sitting_metrics(sittings[0], visits, now=1500)
    assert m.duration_s == 900
    assert m.visit_count == 3
    assert m.time_to_first_visit_s == 60
    assert m.visit_gaps_s == [300, 270]
    assert m.max_gap_s == 300
    assert m.mean_gap_s == 285
    assert not m.is_open


def test_sitting_metrics_open_measured_to_now():
    sittings, visits = records()
    m = sitting_metrics(sittings[1], visits, now=1500)
    assert m.is_open
    assert m.duration_s == 200
    assert m.time_to_first_visit_s == 100
    assert m.visit_gaps_s == []


def test_idle_periods():
    sittings, _ = records()
    periods = idle_periods("T1", sittings, run_start=0, now=1500)
    assert [(p.start, p.end, p.is_open) for p in periods] == [(0, 100, False), (1000, 1300, False)]


def test_idle_period_open_when_table_empty_now():
    periods = idle_periods("T1", [Sitting("T1-S1", "T1", 10, 50)], run_start=0, now=80)
    assert [(p.start, p.end, p.is_open) for p in periods] == [(0, 10, False), (50, 80, True)]


def test_table_summary():
    sittings, visits = records()
    s = table_summary("T1", sittings, visits, run_start=0, now=1500)
    assert s.sittings == 2 and s.completed_sittings == 1
    assert s.visits == 4  # the clearing visit is not counted
    assert s.avg_time_to_first_visit_s == 80
    assert s.max_visit_gap_s == 300
    assert s.avg_turn_time_s == 900
    assert s.idle_s == 400
    assert s.occupied_s == 1100
    assert s.utilisation == pytest.approx(1100 / 1500)


@pytest.mark.parametrize(
    "t, expected, since",
    [
        (50, TableState.EMPTY, None),
        (120, TableState.SEATED, 100),
        (180, TableState.ATTENDED, 160),
        (300, TableState.WAITING, 200),
        (1100, TableState.EMPTY, 1000),
        (1450, TableState.ATTENDED, 1400),
    ],
)
def test_state_at_replay(t, expected, since):
    sittings, visits = records()
    assert state_at("T1", t, sittings, visits) == (expected, since)


def test_replay_matches_live_state_machine():
    sm = TableStateMachine("T1", FAST)
    segments = [(0, 60, 2, 0), (60, 90, 2, 1), (90, 300, 2, 0), (300, 320, 2, 1), (320, 400, 2, 0), (400, 450, 0, 0)]
    sittings, visits, live = {}, {}, []
    for start, end, g, st in segments:
        t = start
        while t < end:
            sm.update(t, g, st)
            if sm.sitting:
                sittings[sm.sitting.key] = sm.sitting
                for v in sm.sitting.visits:
                    visits[v.key] = v
            live.append((t, sm.state))
            t += 0.5
    # Live state lags behind the backdated records by the debounce windows, so
    # only compare where the machine had settled (away from each transition).
    unsettled = [(0, 12), (58, 64), (88, 95), (298, 304), (318, 325), (395, 435)]
    checked = 0
    for t, state in live:
        if not any(a <= t <= b for a, b in unsettled):
            assert state_at("T1", t, list(sittings.values()), list(visits.values()))[0] == state, t
            checked += 1
    assert checked > 700

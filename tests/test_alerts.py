import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tablewatch.alerts.notifiers import (
    RAISED,
    RESOLVED,
    NotificationDispatcher,
    Notifier,
    SlackNotifier,
    WebhookNotifier,
    build_notifier,
)
from tablewatch.alerts.rules import AlertEngine, LongSittingRule, NoFirstVisitRule, VisitGapRule, build_rules
from tablewatch.config import AlertConfig, NotifierConfig
from tablewatch.state.table_state import TableSnapshot, TableState, TableStateMachine

from helpers import FAST

NO_VISIT_S = 300
GAP_S = 600


def engine(repeat=None) -> AlertEngine:
    return AlertEngine([NoFirstVisitRule(NO_VISIT_S), VisitGapRule(GAP_S)], repeat_every_s=repeat)


def simulate(segments, eng, step=1.0, tables=("T1",)):
    """Drive one state machine per table and the alert engine; return alerts by time."""
    machines = {t: TableStateMachine(t, FAST) for t in tables}
    log = []
    for start, end, counts in segments:
        t = start
        while t < end:
            for table, sm in machines.items():
                g, s = counts.get(table, (0, 0))
                sm.update(t, g, s)
            upd = eng.evaluate([sm.snapshot(t) for sm in machines.values()], t)
            # Copy: the engine keeps updating active alerts in place.
            log += [(t, "raised", dataclasses.replace(a)) for a in upd.raised]
            log += [(t, "repeated", dataclasses.replace(a)) for a in upd.repeated]
            log += [(t, "resolved", dataclasses.replace(a)) for a in upd.resolved]
            t += step
    return log


def test_no_first_visit_fires_once_at_threshold():
    log = simulate([(0, 900, {"T1": (2, 0)})], engine())
    raised = [(t, a) for t, kind, a in log if kind == "raised"]
    assert len(raised) == 1
    t, alert = raised[0]
    assert alert.rule == "no_first_visit" and alert.table_id == "T1"
    assert t == NO_VISIT_S  # seated_at=0 is backdated, so it fires at exactly 300 s
    assert alert.sitting_key == "T1-S1"
    assert "no staff visit" in alert.message


def test_no_alert_when_visited_in_time():
    log = simulate([(0, 200, {"T1": (2, 0)}), (200, 230, {"T1": (2, 1)}), (230, 500, {"T1": (2, 0)})], engine())
    assert log == []


def test_first_visit_resolves_alert():
    log = simulate([(0, 400, {"T1": (2, 0)}), (400, 420, {"T1": (2, 1)})], engine())
    kinds = [(kind, a.rule) for _, kind, a in log]
    assert kinds == [("raised", "no_first_visit"), ("resolved", "no_first_visit")]
    resolved = log[-1][2]
    assert resolved.resolved_ts == pytest.approx(402)  # visit confirmed after visit_min_s


def test_visit_gap_fires_per_gap():
    log = simulate(
        [
            (0, 60, {"T1": (2, 0)}),
            (60, 80, {"T1": (2, 1)}),  # visit 1
            (80, 800, {"T1": (2, 0)}),  # gap > 600 -> alert
            (800, 820, {"T1": (2, 1)}),  # visit 2 resolves it
            (820, 1500, {"T1": (2, 0)}),  # second long gap -> a new alert
        ],
        engine(),
    )
    raised = [(t, a.key) for t, kind, a in log if kind == "raised"]
    assert [k for _, k in raised] == ["visit_gap:T1-S1:1", "visit_gap:T1-S1:2"]
    assert raised[0][0] == pytest.approx(79 + GAP_S)  # last visit ended at 79


def test_alerts_are_per_table():
    log = simulate([(0, 400, {"T1": (2, 0), "T2": (0, 0)})], engine(), tables=("T1", "T2"))
    assert {a.table_id for _, kind, a in log if kind == "raised"} == {"T1"}


def test_emptied_table_resolves_alert():
    log = simulate([(0, 350, {"T1": (2, 0)}), (350, 420, {"T1": (0, 0)})], engine())
    assert [kind for _, kind, _ in log] == ["raised", "resolved"]


def test_repeat_notifications():
    log = simulate([(0, 700, {"T1": (2, 0)})], engine(repeat=120))
    repeats = [(t, a.notify_count) for t, kind, a in log if kind == "repeated"]
    assert repeats == [(420, 2), (540, 3), (660, 4)]


def test_long_sitting_rule():
    snap = TableSnapshot("T1", ts=4000, state=TableState.WAITING, state_since=3000, sitting_key="T1-S1", seated_at=0)
    alert = LongSittingRule(3600).check(snap)
    assert alert is not None and alert.severity == "info"
    assert LongSittingRule(5000).check(snap) is None


def test_build_rules_respects_disabled_thresholds():
    rules = build_rules(AlertConfig(no_first_visit_s=None, visit_gap_s=600, long_sitting_s=None))
    assert [r.name for r in rules] == ["visit_gap"]


class Recorder(Notifier):
    def __init__(self, cfg=None):
        super().__init__(cfg or NotifierConfig())
        self.sent = []

    def send(self, alert, event):
        self.sent.append((alert.key, event))


class Broken(Notifier):
    def send(self, alert, event):
        raise RuntimeError("boom")


def test_dispatcher_isolates_failing_notifier():
    rec = Recorder()
    disp = NotificationDispatcher([Broken(NotifierConfig()), rec])
    log = simulate([(0, 310, {"T1": (2, 0)})], engine())
    disp.dispatch(log[0][2], RAISED)
    disp.close()
    assert rec.sent == [("no_first_visit:T1-S1", RAISED)]


def test_custom_notifier_by_import_path():
    n = build_notifier(NotifierConfig(type="tests.test_alerts:Recorder"))
    assert isinstance(n, Recorder)
    with pytest.raises(ValueError):
        build_notifier(NotifierConfig(type="carrier-pigeon"))


@pytest.fixture
def http_sink():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/hook", received
    server.shutdown()


def test_webhook_and_slack_payloads(http_sink):
    url, received = http_sink
    alert = simulate([(0, 310, {"T1": (2, 0)})], engine())[0][2]
    WebhookNotifier(NotifierConfig(type="webhook", url=url)).send(alert, RAISED)
    SlackNotifier(NotifierConfig(type="slack", url=url)).send(alert, RESOLVED)
    assert received[0]["event"] == "raised"
    assert received[0]["alert"]["table_id"] == "T1"
    assert received[1]["text"].startswith(":white_check_mark: RESOLVED T1")

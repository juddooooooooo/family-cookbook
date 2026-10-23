"""Alert rules and the engine that raises, repeats and resolves alerts.

A rule looks at one table snapshot and returns an `Alert` while its condition
holds. Each alert has a stable key (e.g. one per sitting, or one per gap
between visits) so it is raised once, optionally re-notified every
`repeat_every_s`, and resolved automatically when the condition clears.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

from tablewatch.config import AlertConfig
from tablewatch.core import fmt_duration
from tablewatch.state.table_state import TableSnapshot, TableState


@dataclass
class Alert:
    key: str  # dedupe key: one alert per key, ever
    rule: str
    table_id: str
    sitting_key: str | None
    ts: float  # when raised (seconds since run start)
    severity: str
    message: str
    value_s: float  # how long the condition has held
    threshold_s: float
    wall_time: float | None = None
    resolved_ts: float | None = None
    notify_count: int = 0
    last_notified_ts: float | None = None


@dataclass
class AlertUpdate:
    raised: list[Alert] = field(default_factory=list)
    repeated: list[Alert] = field(default_factory=list)
    resolved: list[Alert] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.raised or self.repeated or self.resolved)


class AlertRule(ABC):
    name: str = "rule"

    def __init__(self, threshold_s: float, severity: str = "warning"):
        self.threshold_s = threshold_s
        self.severity = severity

    @abstractmethod
    def check(self, snap: TableSnapshot) -> Alert | None:
        """Return an alert while the condition holds for this table, else None."""

    def _alert(self, snap: TableSnapshot, key_suffix: str, value_s: float, message: str) -> Alert:
        return Alert(
            key=f"{self.name}:{key_suffix}",
            rule=self.name,
            table_id=snap.table_id,
            sitting_key=snap.sitting_key,
            ts=snap.ts,
            severity=self.severity,
            message=message,
            value_s=value_s,
            threshold_s=self.threshold_s,
        )


class NoFirstVisitRule(AlertRule):
    """Guests seated for too long without any staff visit."""

    name = "no_first_visit"

    def check(self, snap: TableSnapshot) -> Alert | None:
        waited = snap.waiting_for_staff_s
        if snap.state != TableState.SEATED or waited is None or waited < self.threshold_s:
            return None
        return self._alert(
            snap, snap.sitting_key, waited,
            f"Table {snap.table_id}: seated {fmt_duration(waited)} with no staff visit",
        )


class VisitGapRule(AlertRule):
    """Too long since staff last visited a seated table."""

    name = "visit_gap"

    def check(self, snap: TableSnapshot) -> Alert | None:
        waited = snap.waiting_for_staff_s
        if snap.state != TableState.WAITING or waited is None or waited < self.threshold_s:
            return None
        return self._alert(
            snap, f"{snap.sitting_key}:{snap.visit_count}", waited,
            f"Table {snap.table_id}: {fmt_duration(waited)} since last staff visit",
        )


class LongSittingRule(AlertRule):
    """A sitting running longer than expected (e.g. to prompt the bill)."""

    name = "long_sitting"

    def __init__(self, threshold_s: float, severity: str = "info"):
        super().__init__(threshold_s, severity)

    def check(self, snap: TableSnapshot) -> Alert | None:
        duration = snap.sitting_duration
        if snap.state == TableState.EMPTY or duration is None or duration < self.threshold_s:
            return None
        return self._alert(
            snap, snap.sitting_key, duration,
            f"Table {snap.table_id}: sitting has lasted {fmt_duration(duration)}",
        )


def build_rules(cfg: AlertConfig) -> list[AlertRule]:
    rules: list[AlertRule] = []
    if cfg.no_first_visit_s:
        rules.append(NoFirstVisitRule(cfg.no_first_visit_s))
    if cfg.visit_gap_s:
        rules.append(VisitGapRule(cfg.visit_gap_s))
    if cfg.long_sitting_s:
        rules.append(LongSittingRule(cfg.long_sitting_s))
    return rules


class AlertEngine:
    def __init__(self, rules: list[AlertRule], repeat_every_s: float | None = None):
        self.rules = rules
        self.repeat_every_s = repeat_every_s
        self.active: dict[str, Alert] = {}
        self._finished: set[str] = set()

    def evaluate(self, snapshots: Iterable[TableSnapshot], ts: float, wall_time: float | None = None) -> AlertUpdate:
        update = AlertUpdate()
        firing: dict[str, Alert] = {}
        for snap in snapshots:
            for rule in self.rules:
                alert = rule.check(snap)
                if alert is not None:
                    firing[alert.key] = alert

        for key, alert in firing.items():
            current = self.active.get(key)
            if current is None:
                if key in self._finished:
                    continue
                alert.wall_time = wall_time
                alert.notify_count, alert.last_notified_ts = 1, ts
                self.active[key] = alert
                update.raised.append(alert)
            else:
                current.value_s, current.message = alert.value_s, alert.message
                if self.repeat_every_s and ts - current.last_notified_ts >= self.repeat_every_s:
                    current.notify_count += 1
                    current.last_notified_ts = ts
                    update.repeated.append(current)

        for key in [k for k in self.active if k not in firing]:
            alert = self.active.pop(key)
            alert.resolved_ts = ts
            self._finished.add(key)
            update.resolved.append(alert)
        return update

    def active_tables(self) -> set[str]:
        return {a.table_id for a in self.active.values()}

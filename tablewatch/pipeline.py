"""Runs the whole chain on a video source:

    source -> detect -> track -> zone -> staff/guest -> table state -> alerts
                                                          |              |
                                                       SQLite  <---------+--> notifiers
                                                          |
                                                 annotated video (opt-in)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from tablewatch.alerts.notifiers import RAISED, REPEATED, RESOLVED, NotificationDispatcher
from tablewatch.alerts.rules import AlertEngine, AlertUpdate, build_rules
from tablewatch.classification.roles import RoleClassifier, build_role_classifier
from tablewatch.config import AppConfig
from tablewatch.core import Detection, Role, Track, fmt_duration
from tablewatch.detection.detector import Detector, UltralyticsDetector
from tablewatch.input.source import Frame, open_source
from tablewatch.output.annotator import Annotator, VideoWriter
from tablewatch.state.served import CATEGORIES, ItemMonitor, class_categories
from tablewatch.state.table_state import EventType, TableEvent, TableSnapshot, TableStateMachine
from tablewatch.storage.db import EventStore
from tablewatch.tracking.tracker import ByteTrackTracker, Tracker
from tablewatch.zones.assigner import ZoneAssigner
from tablewatch.zones.zones import ZoneMap

log = logging.getLogger(__name__)

LOGGED_EVENTS = {
    EventType.STATE_CHANGED,
    EventType.VISIT_ENDED,
    EventType.DRINKS_SERVED,
    EventType.FOOD_SERVED,
}


@dataclass
class FrameResult:
    frame: Frame
    tracks: list[Track]
    objects: list[Detection]
    snapshots: dict[str, TableSnapshot]
    events: list[TableEvent] = field(default_factory=list)
    alerts: AlertUpdate = field(default_factory=AlertUpdate)


class ServiceMonitor:
    """Table-level logic, independent of vision: state machines, drinks/food
    served signals, alerts, persistence and notifications. Fed with per-table
    head-counts (and, optionally, per-table counts of drink and food items).
    """

    def __init__(
        self,
        cfg: AppConfig,
        zone_map: ZoneMap,
        store: EventStore | None = None,
        dispatcher: NotificationDispatcher | None = None,
    ):
        self.cfg = cfg
        self.zone_map = zone_map
        self.machines = {z.table_id: TableStateMachine(z.table_id, cfg.state) for z in zone_map.zones}
        self.served = (
            {(z.table_id, c): ItemMonitor(cfg.served) for z in zone_map.zones for c in CATEGORIES}
            if cfg.served.enabled
            else {}
        )
        self.alerts = AlertEngine(build_rules(cfg.alerts), cfg.alerts.repeat_every_s)
        self.store = store
        self.dispatcher = dispatcher
        self.run_id: int | None = None
        self._last_status_ts = -float("inf")

    def start_run(self, source: str, is_live: bool, fps: float | None, media_start: float | None = None) -> None:
        if self.store is None:
            return
        tables = [{"table_id": z.table_id, "seats": z.seats} for z in self.zone_map.zones]
        self.run_id = self.store.start_run(
            source, is_live, fps, tables, self.cfg.to_dict(), media_start, site=self.cfg.site
        )
        log.info("Run %d started (db: %s)", self.run_id, self.store.path)

    def update(
        self,
        ts: float,
        guests: dict[str, int],
        staff: dict[str, int],
        staff_since: dict[str, float] | None = None,
        item_counts: dict[str, dict[str, int]] | None = None,
        wall_time: float | None = None,
    ) -> tuple[list[TableEvent], AlertUpdate, dict[str, TableSnapshot]]:
        staff_since = staff_since or {}
        events: list[TableEvent] = []
        for table_id, sm in self.machines.items():
            events += sm.update(ts, guests.get(table_id, 0), staff.get(table_id, 0), staff_since.get(table_id))
        if self.served and item_counts is not None:
            events += self._update_served(item_counts, ts)
        snapshots = {tid: sm.snapshot(ts) for tid, sm in self.machines.items()}
        alert_update = self.alerts.evaluate(snapshots.values(), ts, wall_time)
        self._persist(events, alert_update, snapshots, ts)
        for e in events:
            if e.type in LOGGED_EVENTS:
                log.info("[%s] %s", fmt_duration(e.ts), _describe_event(e))
        return events, alert_update, snapshots

    def _update_served(self, counts: dict[str, dict[str, int]], ts: float) -> list[TableEvent]:
        """`counts[table_id][category]` = drink or food items seen in that zone this frame."""
        events = []
        for (table_id, category), monitor in self.served.items():
            sm = self.machines[table_id]
            snap = sm.snapshot(ts)
            last_staff = ts if snap.staff > 0 else snap.last_visit_end
            n = counts.get(table_id, {}).get(category, 0)
            if monitor.update(ts, n, snap.sitting_key, snap.visit_count, last_staff):
                events += sm.mark_served(category, ts)
        return events

    def _persist(self, events: list[TableEvent], alerts: AlertUpdate, snapshots: dict[str, TableSnapshot], ts: float) -> None:
        if self.store is not None and self.run_id is not None:
            if events:
                self.store.record_events(self.run_id, events)
            for alert in alerts.raised + alerts.repeated + alerts.resolved:
                self.store.record_alert(self.run_id, alert)
            if ts - self._last_status_ts >= self.cfg.storage.snapshot_interval_s:
                self.store.write_status(self.run_id, snapshots.values(), ts)
                self._last_status_ts = ts
        if self.dispatcher is not None:
            for kind, items in ((RAISED, alerts.raised), (REPEATED, alerts.repeated), (RESOLVED, alerts.resolved)):
                for alert in items:
                    self.dispatcher.dispatch(alert, kind)

    def finish(self, last_ts: float) -> None:
        if self.store is not None and self.run_id is not None:
            snapshots = [sm.snapshot(last_ts) for sm in self.machines.values()]
            self.store.write_status(self.run_id, snapshots, last_ts)
            self.store.end_run(self.run_id, last_ts)
        if self.dispatcher is not None:
            self.dispatcher.close()


class Pipeline:
    """Vision front-end: turns frames into per-table head-counts for the ServiceMonitor."""

    def __init__(
        self,
        cfg: AppConfig,
        zone_map: ZoneMap,
        detector: Detector,
        tracker: Tracker,
        classifier: RoleClassifier,
        monitor: ServiceMonitor,
    ):
        self.cfg = cfg
        self.zone_map = zone_map
        self.detector = detector
        self.tracker = tracker
        self.classifier = classifier
        self.assigner = ZoneAssigner(zone_map, forget_after_s=cfg.roles.forget_after_s)
        self.monitor = monitor
        self.item_category = class_categories(cfg.served)

    def step(self, frame: Frame) -> FrameResult:
        ts = frame.ts
        found = self.detector.detect(frame.image)
        tracks = self.tracker.update(found.people, ts)
        self.assigner.update(tracks, ts)
        self.classifier.update(tracks, frame.image, ts)

        guests: dict[str, int] = {}
        staff: dict[str, int] = {}
        staff_since: dict[str, float] = {}
        for t in tracks:
            if t.zone_id is None:
                continue
            if t.role == Role.STAFF:
                staff[t.zone_id] = staff.get(t.zone_id, 0) + 1
                staff_since[t.zone_id] = min(staff_since.get(t.zone_id, ts), t.zone_entered_ts or ts)
            else:
                guests[t.zone_id] = guests.get(t.zone_id, 0) + 1

        item_counts = None
        if self.monitor.served:
            item_counts = {}
            for det in found.objects:
                category = self.item_category.get(det.class_name)
                x1, y1, x2, y2 = det.xyxy
                table_id = self.zone_map.assign(((x1 + x2) / 2, (y1 + y2) / 2))
                if category and table_id is not None:
                    per_table = item_counts.setdefault(table_id, {})
                    per_table[category] = per_table.get(category, 0) + 1

        events, alerts, snapshots = self.monitor.update(ts, guests, staff, staff_since, item_counts, frame.wall_time)
        return FrameResult(frame, tracks, found.objects, snapshots, events, alerts)


def _describe_event(e: TableEvent) -> str:
    if e.type == EventType.STATE_CHANGED:
        return f"{e.table_id}: {e.data['from']} -> {e.data['to']}"
    if e.type == EventType.VISIT_ENDED:
        kind = "staff visit" if e.sitting_key else "staff at empty table"
        return f"{e.table_id}: {kind} ended, {fmt_duration(e.data['duration_s'])}"
    return f"{e.table_id}: {e.type.value}"


def build_pipeline(cfg: AppConfig, zone_map: ZoneMap, fps: float, with_store: bool = True) -> Pipeline:
    det = cfg.detection
    detector = UltralyticsDetector(
        det.model, det.device, det.confidence, det.iou, det.image_size,
        list(class_categories(cfg.served)) if cfg.served.enabled else [], det.object_confidence,
    )
    trk = cfg.tracking
    tracker = ByteTrackTracker(
        frame_rate=fps,
        activation_threshold=trk.activation_threshold,
        high_confidence_threshold=trk.high_confidence_threshold,
        lost_track_buffer_s=trk.lost_track_buffer_s,
        min_iou=trk.min_iou,
        min_consecutive_frames=trk.min_consecutive_frames,
    )
    store = EventStore(cfg.storage.db_path) if with_store else None
    monitor = ServiceMonitor(cfg, zone_map, store, NotificationDispatcher.from_config(cfg.alerts.notifiers))
    return Pipeline(cfg, zone_map, detector, tracker, build_role_classifier(cfg.roles), monitor)


def run(
    cfg: AppConfig,
    zone_map: ZoneMap,
    max_seconds: float | None = None,
    media_start: float | None = None,
) -> int | None:
    """Process `cfg.input.source` until it ends (or `max_seconds`). Returns the run id."""
    inp = cfg.input
    source = open_source(inp.source, inp.target_fps, inp.loop, inp.realtime, inp.reconnect_delay_s)
    writer = None
    last_ts = 0.0
    with source:
        zones = zone_map.for_frame_size(*source.frame_size)
        log.info("Source %s: %dx%d, processing at %.1f fps", inp.source, *source.frame_size, source.fps)
        pipeline = build_pipeline(cfg, zones, source.fps)
        pipeline.monitor.start_run(source.uri, source.is_live, source.fps, media_start)
        annotator = Annotator(zones, cfg.output.blur_heads) if (cfg.output.video_path or cfg.output.show_window) else None
        if cfg.output.video_path:
            writer = VideoWriter(cfg.output.video_path, source.fps)
        started, frames = time.time(), 0
        try:
            for frame in source:
                result = pipeline.step(frame)
                last_ts = frame.ts
                frames += 1
                if annotator is not None:
                    image = annotator.draw(
                        frame.image, result.tracks, result.snapshots, frame.ts,
                        pipeline.monitor.alerts.active_tables(), result.objects,
                    )
                    if writer:
                        writer.write(image)
                    if cfg.output.show_window and _show(image):
                        break
                if frames % 100 == 0:
                    elapsed = time.time() - started
                    log.info("t=%s  %d frames, %.1f frames/s", fmt_duration(frame.ts), frames, frames / elapsed)
                if max_seconds is not None and frame.ts >= max_seconds:
                    break
        except KeyboardInterrupt:
            log.info("Interrupted")
        finally:
            pipeline.monitor.finish(last_ts)
            if writer:
                writer.close()
                log.info("Annotated video written to %s", cfg.output.video_path)
    return pipeline.monitor.run_id


def _show(image: np.ndarray) -> bool:
    import cv2

    cv2.imshow("TableWatch", image)
    return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)

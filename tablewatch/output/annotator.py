"""Draws zones coloured by table state, per-table timers and staff/guest boxes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tablewatch.core import Detection, Role, Track, fmt_duration
from tablewatch.state.table_state import TableSnapshot, TableState
from tablewatch.zones.zones import ZoneMap

# BGR
STATE_COLOURS = {
    TableState.EMPTY: (150, 150, 150),
    TableState.SEATED: (0, 190, 255),  # amber: waiting for first visit
    TableState.ATTENDED: (90, 200, 60),  # green: staff at table
    TableState.WAITING: (230, 140, 40),  # blue: visited, waiting for next
}
ALERT_COLOUR = (40, 40, 230)  # red
ROLE_COLOURS = {Role.GUEST: (255, 200, 80), Role.STAFF: (200, 70, 220)}
OBJECT_COLOUR = (120, 220, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5, colour=(255, 255, 255), bg=None) -> None:
    (w, h), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)


def blur_heads(image: np.ndarray, tracks: list[Track]) -> None:
    """Blur the top of each person box in place so faces never reach the output."""
    ih, iw = image.shape[:2]
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.xyxy)
        x1, x2 = max(0, x1), min(iw, x2)
        head_h = int(max(t.height * 0.22, t.width * 0.45))
        hy1, hy2 = max(0, y1), min(ih, y1 + head_h)
        if x2 - x1 < 2 or hy2 - hy1 < 2:
            continue
        roi = image[hy1:hy2, x1:x2]
        k = max(3, (min(roi.shape[:2]) // 2) | 1)
        image[hy1:hy2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def timer_lines(snap: TableSnapshot) -> list[str]:
    """Human-readable timers for one table."""
    if snap.state == TableState.EMPTY:
        idle = snap.ts - snap.empty_since if snap.empty_since is not None else None
        return [f"idle {fmt_duration(idle)}"]
    lines = []
    if snap.state == TableState.SEATED:
        lines.append(f"no visit {fmt_duration(snap.waiting_for_staff_s)}")
    elif snap.state == TableState.ATTENDED:
        lines.append(f"visit {fmt_duration(snap.ts - (snap.visit_started_at or snap.ts))}")
    elif snap.state == TableState.WAITING:
        lines.append(f"since visit {fmt_duration(snap.waiting_for_staff_s)}")
    lines.append(f"seated {fmt_duration(snap.sitting_duration)} | visits {snap.visit_count}")
    return lines


class Annotator:
    def __init__(self, zone_map: ZoneMap, blur: bool = True):
        self.zone_map = zone_map
        self.blur = blur

    def draw(
        self,
        image: np.ndarray,
        tracks: list[Track],
        snapshots: dict[str, TableSnapshot] | None = None,
        ts: float = 0.0,
        alert_tables: set[str] | None = None,
        objects: list[Detection] | None = None,
    ) -> np.ndarray:
        out = image.copy()
        if self.blur:
            blur_heads(out, tracks)
        snapshots = snapshots or {}
        alert_tables = alert_tables or set()

        overlay = out.copy()
        for zone in self.zone_map.zones:
            snap = snapshots.get(zone.table_id)
            colour = STATE_COLOURS[snap.state] if snap else STATE_COLOURS[TableState.EMPTY]
            cv2.fillPoly(overlay, [np.array(zone.polygon, dtype=np.int32)], colour)
        out = cv2.addWeighted(overlay, 0.28, out, 0.72, 0)

        for zone in self.zone_map.zones:
            snap = snapshots.get(zone.table_id)
            alerting = zone.table_id in alert_tables
            colour = ALERT_COLOUR if alerting else (STATE_COLOURS[snap.state] if snap else STATE_COLOURS[TableState.EMPTY])
            pts = np.array(zone.polygon, dtype=np.int32)
            cv2.polylines(out, [pts], True, colour, 3 if alerting else 2, cv2.LINE_AA)
            x, y = int(min(p[0] for p in zone.polygon)), int(min(p[1] for p in zone.polygon))
            title = f"{zone.table_id} {snap.state.value if snap else ''}" + ("  ALERT" if alerting else "")
            _text(out, title, (x + 4, max(14, y + 16)), 0.55, bg=colour)
            if snap:
                for i, line in enumerate(timer_lines(snap)):
                    _text(out, line, (x + 4, max(14, y + 16) + 18 * (i + 1)), 0.45, bg=(30, 30, 30))

        for det in objects or []:
            x1, y1, x2, y2 = (int(v) for v in det.xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), OBJECT_COLOUR, 1)

        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t.xyxy)
            colour = ROLE_COLOURS[t.role]
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"{t.role.value} #{t.track_id}"
            _text(out, label, (x1 + 2, max(12, y1 - 4)), 0.45, colour=(0, 0, 0), bg=colour)
            ax, ay = (int(v) for v in t.anchor)
            cv2.circle(out, (ax, ay), 4, colour, -1)

        staff = sum(t.role == Role.STAFF for t in tracks)
        _text(out, f"t={fmt_duration(ts)}  guests {len(tracks) - staff}  staff {staff}", (8, 20), 0.6, bg=(0, 0, 0))
        return out


class VideoWriter:
    """Thin wrapper over cv2.VideoWriter that opens lazily on the first frame."""

    def __init__(self, path: str, fps: float):
        self.path = path
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {self.path}")
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

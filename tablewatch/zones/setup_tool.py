"""Interactive zone drawing tool.

Shows one frame from the video (or a live camera) and lets you click a polygon
around each table. Draw the polygon over the floor area covered by the table
and its chairs: people are assigned to a table by the bottom-centre of their
bounding box, which for a seated guest lands on the chair or under the table.

    python -m tablewatch zones --source data/sample.mp4 --out config/zones.yaml

Controls (shown on screen):
    left click   add a point            right click / u   undo last point
    Enter / n    finish this table      d                 delete last table
    c            clear current points   s                 save zones.yaml
    q / Esc      quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tablewatch.core import Point
from tablewatch.zones.zones import TableZone, ZoneMap, load_zones, save_zones

MAX_DISPLAY_WIDTH = 1280
HELP = "click: add point | right-click/u: undo | Enter/n: finish table | d: delete last | s: save | q: quit"


class ZoneEditor:
    """Editing state, independent of any GUI so it can be unit tested."""

    def __init__(self, zones: list[TableZone] | None = None, default_seats: int = 4):
        self.zones: list[TableZone] = list(zones or [])
        self.current: list[Point] = []
        self.default_seats = default_seats
        self.dirty = False

    def add_point(self, x: float, y: float) -> None:
        self.current.append((float(x), float(y)))

    def undo_point(self) -> None:
        if self.current:
            self.current.pop()

    def clear_current(self) -> None:
        self.current = []

    def next_table_id(self) -> str:
        taken = {z.table_id for z in self.zones}
        n = len(self.zones) + 1
        while f"T{n}" in taken:
            n += 1
        return f"T{n}"

    def can_finish(self) -> bool:
        return len(self.current) >= 3

    def finish(self, table_id: str | None = None, seats: int | None = None) -> TableZone:
        if not self.can_finish():
            raise ValueError("A table zone needs at least 3 points")
        table_id = (table_id or "").strip() or self.next_table_id()
        if any(z.table_id == table_id for z in self.zones):
            raise ValueError(f"Table id {table_id!r} already exists")
        zone = TableZone(table_id, list(self.current), seats or self.default_seats)
        self.zones.append(zone)
        self.current = []
        self.dirty = True
        return zone

    def delete_last(self) -> TableZone | None:
        if not self.zones:
            return None
        self.dirty = True
        return self.zones.pop()

    def zone_map(self, frame_size: tuple[int, int]) -> ZoneMap:
        return ZoneMap(list(self.zones), frame_size)


def read_frame(source: str, at_s: float = 0.0) -> np.ndarray:
    """Grab one frame from a file (optionally at a time offset) or a live stream."""
    from tablewatch.input.source import FileSource, open_source

    src = open_source(source)
    with src:
        if isinstance(src, FileSource) and at_s > 0:
            src._cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000.0)
        return src.first_frame()


def render(frame: np.ndarray, editor: ZoneEditor, scale: float) -> np.ndarray:
    canvas = frame.copy()
    overlay = canvas.copy()
    for zone in editor.zones:
        pts = np.array(zone.polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (80, 160, 80))
        cv2.polylines(canvas, [pts], True, (60, 220, 60), 2)
    canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)
    for zone in editor.zones:
        cx, cy = (int(v) for v in zone.centroid)
        label = f"{zone.table_id} ({zone.seats} seats)"
        cv2.putText(canvas, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if editor.current:
        pts = np.array(editor.current, dtype=np.int32)
        cv2.polylines(canvas, [pts], len(pts) >= 3, (0, 200, 255), 2)
        for p in pts:
            cv2.circle(canvas, tuple(int(v) for v in p), 4, (0, 200, 255), -1)
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(canvas, HELP, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _prompt(editor: ZoneEditor, ask: bool) -> tuple[str | None, int | None]:
    if not ask:
        return None, None
    suggested = editor.next_table_id()
    table_id = input(f"Table id [{suggested}]: ").strip() or suggested
    seats_raw = input(f"Seats [{editor.default_seats}]: ").strip()
    return table_id, int(seats_raw) if seats_raw.isdigit() else None


def run_editor(source: str, out: str, at_s: float = 0.0, default_seats: int = 4, ask: bool = True) -> None:
    frame = read_frame(source, at_s)
    height, width = frame.shape[:2]
    existing = load_zones(out).for_frame_size(width, height).zones if Path(out).exists() else []
    editor = ZoneEditor(existing, default_seats)
    if existing:
        print(f"Loaded {len(existing)} existing zone(s) from {out}")
    scale = min(1.0, MAX_DISPLAY_WIDTH / width)
    window = "TableWatch - draw table zones"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.add_point(x / scale, y / scale)
        elif event == cv2.EVENT_RBUTTONDOWN:
            editor.undo_point()

    cv2.setMouseCallback(window, on_mouse)
    quit_armed = False
    while True:
        cv2.imshow(window, render(frame, editor, scale))
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        if key in (13, 10, ord("n")):
            if editor.can_finish():
                try:
                    zone = editor.finish(*_prompt(editor, ask))
                    print(f"Added {zone.table_id} ({zone.seats} seats, {len(zone.polygon)} points)")
                except ValueError as exc:
                    print(f"Not added: {exc}")
            else:
                print("Click at least 3 points first")
        elif key == ord("u"):
            editor.undo_point()
        elif key == ord("c"):
            editor.clear_current()
        elif key == ord("d"):
            removed = editor.delete_last()
            print(f"Deleted {removed.table_id}" if removed else "No tables to delete")
        elif key == ord("s"):
            save_zones(editor.zone_map((width, height)), out)
            editor.dirty = False
            print(f"Saved {len(editor.zones)} zone(s) to {out}")
        elif key in (ord("q"), 27):
            if editor.dirty and not quit_armed:
                print("Unsaved changes: press s to save, or q again to quit without saving")
                quit_armed = True
                continue
            break
        quit_armed = False
    cv2.destroyAllWindows()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, help="video file, RTSP URL or webcam index")
    parser.add_argument("--out", default="config/zones.yaml", help="zones file to create or edit")
    parser.add_argument("--at", type=float, default=0.0, help="use the frame at this many seconds (files only)")
    parser.add_argument("--seats", type=int, default=4, help="default seats per table")
    parser.add_argument("--no-prompt", action="store_true", help="auto-name tables T1, T2, ... without asking")


def run_from_args(args: argparse.Namespace) -> None:
    run_editor(args.source, args.out, args.at, args.seats, ask=not args.no_prompt)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    main()

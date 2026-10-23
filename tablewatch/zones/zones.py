"""Table zones: one polygon per table, stored in zones.yaml.

Polygons are saved together with the frame size they were drawn on, so they
are rescaled automatically if the camera later streams at another resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tablewatch.core import Point


@dataclass
class TableZone:
    table_id: str
    polygon: list[Point]
    seats: int = 4

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"Zone {self.table_id!r} needs at least 3 points")
        self.polygon = [(float(x), float(y)) for x, y in self.polygon]

    def contains(self, point: Point) -> bool:
        """Ray-casting point-in-polygon test (points on an edge may go either way)."""
        x, y = point
        inside = False
        pts = self.polygon
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside

    @property
    def centroid(self) -> Point:
        xs, ys = zip(*self.polygon)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def scaled(self, sx: float, sy: float) -> TableZone:
        return TableZone(self.table_id, [(x * sx, y * sy) for x, y in self.polygon], self.seats)


@dataclass
class ZoneMap:
    zones: list[TableZone] = field(default_factory=list)
    frame_size: tuple[int, int] | None = None  # (width, height) the polygons were drawn on

    def __post_init__(self) -> None:
        ids = [z.table_id for z in self.zones]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate table_id(s) in zones: {sorted(dupes)}")

    @property
    def table_ids(self) -> list[str]:
        return [z.table_id for z in self.zones]

    def get(self, table_id: str) -> TableZone:
        for z in self.zones:
            if z.table_id == table_id:
                return z
        raise KeyError(table_id)

    def assign(self, point: Point) -> str | None:
        """Table whose polygon contains `point`; nearest centroid wins on overlap."""
        hits = [z for z in self.zones if z.contains(point)]
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0].table_id
        px, py = point
        return min(hits, key=lambda z: (z.centroid[0] - px) ** 2 + (z.centroid[1] - py) ** 2).table_id

    def for_frame_size(self, width: int, height: int) -> ZoneMap:
        """Rescale polygons if the video resolution differs from the drawing resolution."""
        if not self.frame_size or tuple(self.frame_size) == (width, height):
            return ZoneMap(list(self.zones), (width, height))
        sx, sy = width / self.frame_size[0], height / self.frame_size[1]
        return ZoneMap([z.scaled(sx, sy) for z in self.zones], (width, height))


def load_zones(path: str | Path) -> ZoneMap:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    zones = [
        TableZone(
            table_id=str(t["table_id"]),
            polygon=[tuple(p) for p in t["polygon"]],
            seats=int(t.get("seats", 4)),
        )
        for t in data.get("tables", [])
    ]
    size = data.get("frame_size")
    return ZoneMap(zones, tuple(size) if size else None)


def save_zones(zone_map: ZoneMap, path: str | Path) -> None:
    data = {
        "frame_size": list(zone_map.frame_size) if zone_map.frame_size else None,
        "tables": [
            {
                "table_id": z.table_id,
                "seats": z.seats,
                "polygon": [[int(round(x)), int(round(y))] for x, y in z.polygon],
            }
            for z in zone_map.zones
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=None)

"""Shared data types passed between pipeline stages.

Everything here is plain Python so the state machine, alert logic and tests
never need OpenCV or a model loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
Point = tuple[float, float]


class Role(str, Enum):
    GUEST = "guest"
    STAFF = "staff"


@dataclass(frozen=True)
class Detection:
    """One object found by the detector in a single frame."""

    xyxy: BBox
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Track:
    """A person followed across frames, enriched as it moves down the pipeline."""

    track_id: int
    xyxy: BBox
    confidence: float
    zone_id: str | None = None
    zone_entered_ts: float | None = None
    role: Role = Role.GUEST
    role_reason: str = ""

    @property
    def anchor(self) -> Point:
        """Bottom-centre of the box: where the person meets the floor or chair."""
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return max(1.0, self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return max(1.0, self.xyxy[3] - self.xyxy[1])

    @property
    def aspect_ratio(self) -> float:
        """Height / width. Standing people are tall and narrow, seated people squat."""
        return self.height / self.width


@dataclass
class FrameDetections:
    """Detector output for one frame, split into people and other objects."""

    people: list[Detection] = field(default_factory=list)
    objects: list[Detection] = field(default_factory=list)


def fmt_duration(seconds: float | None) -> str:
    """Format seconds as m:ss (or h:mm:ss); '-' for None."""
    if seconds is None:
        return "-"
    seconds = max(0, round(float(seconds)))  # float(): numpy float32 would round to a float
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

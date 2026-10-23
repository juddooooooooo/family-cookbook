"""Typed configuration loaded from YAML.

Every section maps to a dataclass so typos in config.yaml fail loudly at start-up
instead of silently falling back to defaults.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InputConfig:
    source: str = "data/sample.mp4"  # file path, rtsp://... URL, or webcam index ("0")
    target_fps: float = 10.0  # process at most this many frames/s (0 = every frame)
    loop: bool = False  # restart a file when it ends (handy for dashboard demos)
    realtime: bool = False  # pace file playback at real speed, like a live camera
    reconnect_delay_s: float = 5.0  # live streams: wait before reconnecting


@dataclass
class DetectionConfig:
    model: str = "yolo11n.pt"  # any Ultralytics detection weights (yolov8n.pt, yolo11s.pt, ...)
    device: str | None = None  # None = auto, or "cpu", "cuda:0", "mps"
    confidence: float = 0.25  # low on purpose: ByteTrack uses weak boxes to bridge occlusions
    iou: float = 0.5
    image_size: int = 640
    object_confidence: float = 0.25  # for the drink/food classes counted in `served`


@dataclass
class TrackingConfig:
    activation_threshold: float = 0.5  # min confidence to start a new track
    high_confidence_threshold: float = 0.5  # ByteTrack's first-stage association cut-off
    lost_track_buffer_s: float = 3.0  # keep a lost track alive this long (occlusions)
    min_iou: float = 0.1  # min box overlap to associate a detection with a track
    min_consecutive_frames: int = 2  # frames before a new track is reported


@dataclass
class UniformConfig:
    enabled: bool = False
    # One or more HSV ranges (OpenCV scale: H 0-179, S/V 0-255). Default: black/very dark.
    hsv_ranges: list[list[list[int]]] = field(default_factory=lambda: [[[0, 0, 0], [179, 255, 60]]])
    min_fraction: float = 0.45  # share of torso pixels inside the range(s)
    min_observations: int = 5  # frames of evidence before the colour rule can fire
    smoothing: float = 0.2  # EMA weight of the newest observation


@dataclass
class RoleConfig:
    # "heuristic" or "package.module:ClassName" for a custom RoleClassifier.
    classifier: str = "heuristic"
    min_zones_for_staff: int | None = 2  # distinct tables visited
    zone_min_dwell_s: float = 1.5  # time inside a zone before it counts as visited
    standing_aspect_ratio: float | None = 2.0  # box height/width above this = standing
    standing_staff_s: float | None = 30.0  # cumulative standing time that marks staff
    travel_staff_body_heights: float | None = 12.0  # path length in body heights
    sticky_staff: bool = True  # once staff, always staff for that track
    forget_after_s: float = 30.0  # drop history of tracks not seen for this long
    uniform: UniformConfig = field(default_factory=UniformConfig)


@dataclass
class StateConfig:
    seat_confirm_s: float = 10.0  # guests present this long -> SEATED
    empty_confirm_s: float = 45.0  # guests gone this long -> EMPTY
    guest_gap_tolerance_s: float = 3.0  # detection dropouts shorter than this are ignored
    visit_min_s: float = 2.0  # staff present this long counts as a visit
    visit_end_grace_s: float = 4.0  # staff absent this long ends the visit


@dataclass
class ServedConfig:
    """Estimate when drinks and food reach each table from COCO object detections."""

    enabled: bool = True
    drink_classes: list[str] = field(default_factory=lambda: ["cup", "wine glass", "bottle"])
    # COCO has no "plate": bowls and the food classes themselves stand in for it.
    food_classes: list[str] = field(
        default_factory=lambda: [
            "bowl", "sandwich", "pizza", "cake", "donut", "hot dog",
            "broccoli", "carrot", "apple", "orange", "banana",
        ]
    )
    window_s: float = 6.0  # rolling window for item counts
    min_increase: int = 1  # items above the pre-service level that count as served
    visit_window_s: float = 90.0  # the increase must follow a staff visit within this time


@dataclass
class NotifierConfig:
    type: str = "console"  # console | webhook | slack | whatsapp | package.module:ClassName
    enabled: bool = True
    url: str | None = None
    to: str | None = None
    timeout_s: float = 5.0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    no_first_visit_s: float | None = 300.0  # seated with no staff visit for 5 min
    visit_gap_s: float | None = 600.0  # >10 min since the last visit ended
    long_sitting_s: float | None = None  # optional: sitting longer than this
    repeat_every_s: float | None = None  # re-notify while an alert stays active
    notifiers: list[NotifierConfig] = field(default_factory=lambda: [NotifierConfig()])


@dataclass
class StorageConfig:
    db_path: str = "data/tablewatch.db"
    snapshot_interval_s: float = 1.0  # how often live table status is written


@dataclass
class OutputConfig:
    video_path: str | None = None  # annotated video; None = don't save any frames
    show_window: bool = False
    blur_heads: bool = True  # blur the head region of every person in the output video


@dataclass
class AppConfig:
    site: str = "default"  # venue or camera name; keeps analytics from different cameras apart
    zones_path: str = "config/zones.yaml"
    input: InputConfig = field(default_factory=InputConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    roles: RoleConfig = field(default_factory=RoleConfig)
    state: StateConfig = field(default_factory=StateConfig)
    served: ServedConfig = field(default_factory=ServedConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ConfigError(ValueError):
    pass


def _build(cls: type, data: Any, path: str) -> Any:
    """Recursively turn a dict into dataclass `cls`, rejecting unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{path or 'config'}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints[name], value, f"{path}.{name}" if path else name)
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(hint)
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, path)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        inner = [a for a in typing.get_args(hint) if a is not type(None)]
        return _coerce(inner[0], value, path) if len(inner) == 1 else value
    if origin is list:
        (item,) = typing.get_args(hint) or (Any,)
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list")
        return [_coerce(item, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint is str and isinstance(value, int) and not isinstance(value, bool):
        return str(value)  # e.g. webcam index written as 0
    return value


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load config.yaml (or defaults when path is None) and apply dotted-key overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    for dotted, value in (overrides or {}).items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return _build(AppConfig, data, "")

"""Staff vs guest classification.

`RoleClassifier` is the extension point: a trained model (e.g. a small CNN on
person crops, or a re-ID embedding matched against enrolled staff) can replace
`HeuristicRoleClassifier` by implementing `update()` and being named in
config.yaml as `roles.classifier: "my_package.module:MyClassifier"`.

The heuristic marks a track as staff when any enabled rule fires:
  * it lingered in `min_zones_for_staff` different table zones,
  * it spent `standing_staff_s` seconds standing (tall, narrow box),
  * it walked `travel_staff_body_heights` body heights in total,
  * (optional) its torso colour matches the configured uniform HSV range.
Everyone else is a guest.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from tablewatch.config import RoleConfig, UniformConfig
from tablewatch.core import Point, Role, Track

MAX_STEP_S = 1.0  # never credit more than this per update (protects against frame gaps)
TRAVEL_STEP_BODY_HEIGHTS = 0.3  # movement below this is treated as box jitter


class RoleClassifier(ABC):
    @abstractmethod
    def update(self, tracks: list[Track], image: np.ndarray | None, ts: float) -> None:
        """Set `role` and `role_reason` on every track, in place."""


@dataclass
class _History:
    first_ts: float
    last_ts: float
    standing_s: float = 0.0
    travel: float = 0.0  # body heights
    travel_ref: Point | None = None
    zone: str | None = None
    zone_since: float = 0.0
    zones_visited: set[str] = field(default_factory=set)
    uniform_score: float = 0.0
    uniform_obs: int = 0
    staff_reason: str | None = None


def torso_crop(image: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """Central chest region of a person box: avoids the face, background and table."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    ih, iw = image.shape[:2]
    cx1, cx2 = int(max(0, x1 + 0.25 * w)), int(min(iw, x2 - 0.25 * w))
    cy1, cy2 = int(max(0, y1 + 0.2 * h)), int(min(ih, y1 + 0.5 * h))
    return image[cy1:cy2, cx1:cx2]


def uniform_fraction(image: np.ndarray, xyxy, hsv_ranges: list[list[list[int]]]) -> float | None:
    """Share of torso pixels inside any of the HSV ranges; None if the crop is empty."""
    import cv2

    crop = torso_crop(image, xyxy)
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return float(np.count_nonzero(mask)) / mask.size


class HeuristicRoleClassifier(RoleClassifier):
    def __init__(self, cfg: RoleConfig | None = None):
        self.cfg = cfg or RoleConfig()
        self._hist: dict[int, _History] = {}

    def update(self, tracks: list[Track], image: np.ndarray | None, ts: float) -> None:
        for track in tracks:
            hist = self._hist.get(track.track_id)
            if hist is None:
                hist = self._hist[track.track_id] = _History(first_ts=ts, last_ts=ts, zone_since=ts)
            self._observe(hist, track, image, ts)
            reason = self._staff_reason(hist)
            if reason and not hist.staff_reason:
                hist.staff_reason = reason
            elif not reason and not self.cfg.sticky_staff:
                hist.staff_reason = None
            track.role = Role.STAFF if hist.staff_reason else Role.GUEST
            track.role_reason = hist.staff_reason or ""
        cutoff = ts - self.cfg.forget_after_s
        for tid in [tid for tid, h in self._hist.items() if h.last_ts < cutoff]:
            del self._hist[tid]

    def _observe(self, hist: _History, track: Track, image: np.ndarray | None, ts: float) -> None:
        cfg = self.cfg
        dt = min(max(0.0, ts - hist.last_ts), MAX_STEP_S)
        hist.last_ts = ts

        if cfg.standing_aspect_ratio and track.aspect_ratio >= cfg.standing_aspect_ratio:
            hist.standing_s += dt

        anchor = track.anchor
        if hist.travel_ref is None:
            hist.travel_ref = anchor
        else:
            moved = np.hypot(anchor[0] - hist.travel_ref[0], anchor[1] - hist.travel_ref[1]) / track.height
            if moved >= TRAVEL_STEP_BODY_HEIGHTS:
                hist.travel += moved
                hist.travel_ref = anchor

        if track.zone_id != hist.zone:
            hist.zone, hist.zone_since = track.zone_id, ts
        elif hist.zone and ts - hist.zone_since >= cfg.zone_min_dwell_s:
            hist.zones_visited.add(hist.zone)

        if cfg.uniform.enabled and image is not None:
            self._observe_uniform(hist, track, image, cfg.uniform)

    @staticmethod
    def _observe_uniform(hist: _History, track: Track, image: np.ndarray, ucfg: UniformConfig) -> None:
        frac = uniform_fraction(image, track.xyxy, ucfg.hsv_ranges)
        if frac is None:
            return
        if hist.uniform_obs == 0:
            hist.uniform_score = frac
        else:
            hist.uniform_score = (1 - ucfg.smoothing) * hist.uniform_score + ucfg.smoothing * frac
        hist.uniform_obs += 1

    def _staff_reason(self, hist: _History) -> str | None:
        cfg = self.cfg
        if cfg.uniform.enabled and hist.uniform_obs >= cfg.uniform.min_observations:
            if hist.uniform_score >= cfg.uniform.min_fraction:
                return "uniform"
        if cfg.min_zones_for_staff and len(hist.zones_visited) >= cfg.min_zones_for_staff:
            return "multi-zone"
        if cfg.standing_staff_s and hist.standing_s >= cfg.standing_staff_s:
            return "standing"
        if cfg.travel_staff_body_heights and hist.travel >= cfg.travel_staff_body_heights:
            return "travel"
        return None


def build_role_classifier(cfg: RoleConfig) -> RoleClassifier:
    """`heuristic`, or `package.module:ClassName` taking the RoleConfig as its only argument."""
    if cfg.classifier == "heuristic":
        return HeuristicRoleClassifier(cfg)
    module_name, _, class_name = cfg.classifier.partition(":")
    if not class_name:
        raise ValueError(f"roles.classifier must be 'heuristic' or 'module:Class', got {cfg.classifier!r}")
    cls = getattr(importlib.import_module(module_name), class_name)
    instance = cls(cfg)
    if not isinstance(instance, RoleClassifier):
        raise TypeError(f"{cfg.classifier} is not a RoleClassifier")
    return instance

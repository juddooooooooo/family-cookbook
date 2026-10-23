"""Estimate when drinks and food reach each table, from object detections.

COCO has no "plate" class, so drinks are counted from cups, glasses and
bottles, and food from bowls and the food classes themselves. When the
smoothed count of a category inside a table zone rises above its level before
service, shortly after a staff visit, that category is marked served for the
sitting. Small objects seen from a ceiling camera are detected unreliably, so
these times are estimates: good for trends, not for timing one plate.
"""

from __future__ import annotations

from collections import deque
from statistics import median

from tablewatch.config import ServedConfig

DRINKS, FOOD = "drinks", "food"
CATEGORIES = (DRINKS, FOOD)


def class_categories(cfg: ServedConfig) -> dict[str, str]:
    """COCO class name -> "drinks" | "food"."""
    mapping = dict.fromkeys(cfg.drink_classes, DRINKS)
    mapping.update(dict.fromkeys(cfg.food_classes, FOOD))
    return mapping


class ItemMonitor:
    """Watches one category (drinks or food) on one table."""

    def __init__(self, cfg: ServedConfig | None = None):
        self.cfg = cfg or ServedConfig()
        self._counts: deque[tuple[float, int]] = deque()
        self._sitting_key: str | None = None
        self._baseline: float | None = None
        self._served = False

    def update(
        self,
        ts: float,
        item_count: int,
        sitting_key: str | None,
        visit_count: int,
        last_staff_ts: float | None,
    ) -> bool:
        """Return True exactly once per sitting, when the items are judged to have arrived.

        `last_staff_ts` is `ts` while staff are at the table, otherwise the
        end of the most recent visit (None if there has been none).
        """
        self._counts.append((ts, item_count))
        while self._counts and self._counts[0][0] < ts - self.cfg.window_s:
            self._counts.popleft()
        level = median(c for _, c in self._counts)

        if sitting_key != self._sitting_key:
            self._sitting_key, self._baseline, self._served = sitting_key, None, False
        if sitting_key is None or self._served:
            return False
        if visit_count == 0:
            # Before anyone has served the table, track the lowest settled level.
            self._baseline = level if self._baseline is None else min(self._baseline, level)
            return False
        if self._baseline is None:
            self._baseline = 0.0
        recent_visit = last_staff_ts is not None and ts - last_staff_ts <= self.cfg.visit_window_s
        if recent_visit and level >= self._baseline + self.cfg.min_increase:
            self._served = True
            return True
        return False

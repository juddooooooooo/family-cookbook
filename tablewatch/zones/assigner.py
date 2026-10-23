"""Assign tracked people to table zones and remember when they entered."""

from __future__ import annotations

from dataclasses import dataclass

from tablewatch.core import Track
from tablewatch.zones.zones import ZoneMap


@dataclass
class _Membership:
    zone_id: str | None
    entered_ts: float
    outside_since: float | None = None  # when the anchor last left the zone


class ZoneAssigner:
    """Sets `track.zone_id` / `track.zone_entered_ts` from the box's bottom-centre.

    A short `exit_grace_s` keeps a seated guest assigned to their table when
    box jitter briefly pushes the anchor just outside the polygon.
    """

    def __init__(self, zone_map: ZoneMap, exit_grace_s: float = 1.0, forget_after_s: float = 30.0):
        self.zone_map = zone_map
        self.exit_grace_s = exit_grace_s
        self.forget_after_s = forget_after_s
        self._members: dict[int, _Membership] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, tracks: list[Track], ts: float) -> None:
        for track in tracks:
            zone_id = self.zone_map.assign(track.anchor)
            m = self._members.get(track.track_id)
            if m is None:
                m = self._members[track.track_id] = _Membership(zone_id, ts)
            elif zone_id == m.zone_id:
                m.outside_since = None
            elif zone_id is None and m.zone_id is not None:
                if m.outside_since is None:
                    m.outside_since = ts
                if ts - m.outside_since > self.exit_grace_s:
                    m.zone_id, m.entered_ts, m.outside_since = None, ts, None
            else:
                m.zone_id, m.entered_ts, m.outside_since = zone_id, ts, None
            track.zone_id = m.zone_id
            track.zone_entered_ts = m.entered_ts if m.zone_id else None
            self._last_seen[track.track_id] = ts
        cutoff = ts - self.forget_after_s
        for tid in [tid for tid, seen in self._last_seen.items() if seen < cutoff]:
            del self._last_seen[tid]
            self._members.pop(tid, None)

"""Multi-object tracking: give each detected person a stable ID across frames."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tablewatch.core import Detection, Track


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], ts: float) -> list[Track]:
        """Associate this frame's person detections with existing tracks."""


class ByteTrackTracker(Tracker):
    """ByteTrack from Roboflow's `trackers` package, driven by frame timestamps.

    Timestamps (rather than frame counts) keep track lifetimes correct when a
    live stream delivers frames at an uneven rate.
    """

    def __init__(
        self,
        frame_rate: float = 10.0,
        activation_threshold: float = 0.5,
        high_confidence_threshold: float = 0.5,
        lost_track_buffer_s: float = 3.0,
        min_iou: float = 0.1,
        min_consecutive_frames: int = 2,
    ):
        import supervision as sv
        from trackers import ByteTrackTracker as _ByteTrack

        self._sv = sv
        self._tracker = _ByteTrack(
            # The library expresses the buffer as a number of 30 fps frames.
            lost_track_buffer=max(1, round(lost_track_buffer_s * 30)),
            frame_rate=frame_rate,
            track_activation_threshold=activation_threshold,
            high_conf_det_threshold=high_confidence_threshold,
            minimum_consecutive_frames=min_consecutive_frames,
            minimum_iou_threshold=min_iou,
        )

    def update(self, detections: list[Detection], ts: float) -> list[Track]:
        if detections:
            dets = self._sv.Detections(
                xyxy=np.array([d.xyxy for d in detections], dtype=np.float32),
                confidence=np.array([d.confidence for d in detections], dtype=np.float32),
                class_id=np.array([d.class_id for d in detections], dtype=int),
            )
        else:
            dets = self._sv.Detections.empty()
        tracked = self._tracker.update(dets, timestamp=ts)
        tracks = []
        if tracked.tracker_id is None:
            return tracks
        for box, conf, tid in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
            if tid < 0:  # detection not (yet) part of a confirmed track
                continue
            tracks.append(Track(track_id=int(tid), xyxy=tuple(float(v) for v in box), confidence=float(conf)))
        return tracks

"""Video input abstraction: recorded files and live streams behind one interface.

The rest of the pipeline only sees `Frame` objects carrying a timestamp in
seconds since the run started. For files that is media time (frame position /
fps), for live streams it is wall-clock time, so all downstream logic is
identical whether the input is a phone recording or an RTSP camera.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

STREAM_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


@dataclass
class Frame:
    image: np.ndarray  # BGR, as produced by OpenCV
    ts: float  # seconds since the start of the run
    index: int  # sequence number of processed frames
    wall_time: float  # epoch seconds when the frame was produced/processed


class VideoSource(ABC):
    """Iterable of frames. Use as a context manager so devices are released."""

    is_live: bool = False

    def __init__(self, uri: str, target_fps: float = 0.0):
        self.uri = uri
        self.target_fps = target_fps

    @property
    @abstractmethod
    def fps(self) -> float:
        """Rate at which frames are yielded (after any frame skipping)."""

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(width, height) in pixels."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]: ...

    def first_frame(self) -> np.ndarray:
        for frame in self:
            return frame.image
        raise RuntimeError(f"No frames could be read from {self.uri}")

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FileSource(VideoSource):
    """A recorded video file, optionally sub-sampled, looped or paced in real time."""

    def __init__(self, uri: str, target_fps: float = 0.0, loop: bool = False, realtime: bool = False):
        super().__init__(uri, target_fps)
        if not Path(uri).exists():
            raise FileNotFoundError(f"Video file not found: {uri}")
        self.loop = loop
        self.realtime = realtime
        self._cap: cv2.VideoCapture | None = None
        self._native_fps = 0.0
        self._stride = 1
        self._size = (0, 0)

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.uri)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.uri}")
        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not 1.0 <= self._native_fps <= 240.0:
            log.warning("File reports fps=%.2f; assuming 25", self._native_fps)
            self._native_fps = 25.0
        self._stride = max(1, round(self._native_fps / self.target_fps)) if self.target_fps > 0 else 1
        self._size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._native_fps / self._stride

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._size

    @property
    def duration_s(self) -> float:
        frames = self._cap.get(cv2.CAP_PROP_FRAME_COUNT) if self._cap else 0
        return frames / self._native_fps if self._native_fps else 0.0

    def __iter__(self) -> Iterator[Frame]:
        if self._cap is None:
            self.open()
        start_wall = time.time()
        offset = 0.0  # media time carried over between loops
        position = 0  # native frame index within the current pass
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                if not self.loop or position == 0:
                    return
                offset += position / self._native_fps
                position = 0
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            ts = offset + position / self._native_fps
            position += 1
            for _ in range(self._stride - 1):  # skip frames cheaply
                if not self._cap.grab():
                    break
                position += 1
            if self.realtime:
                delay = start_wall + ts - time.time()
                if delay > 0:
                    time.sleep(delay)
            yield Frame(image=image, ts=ts, index=index, wall_time=time.time())
            index += 1


class StreamSource(VideoSource):
    """A live RTSP/HTTP stream or local webcam.

    A background thread drains the stream continuously and keeps only the
    newest frame, so slow processing drops frames instead of building up lag.
    Dropped connections are retried forever until `close()`.
    """

    is_live = True

    def __init__(self, uri: str, target_fps: float = 0.0, reconnect_delay_s: float = 5.0):
        super().__init__(uri, target_fps)
        self.reconnect_delay_s = reconnect_delay_s
        self._device: str | int = int(uri) if uri.isdigit() else uri
        self._cap: cv2.VideoCapture | None = None
        self._cond = threading.Condition()
        self._latest: tuple[np.ndarray, float] | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream_fps = 0.0
        self._size = (0, 0)
        self._t0 = 0.0

    def open(self) -> None:
        self._stop.clear()
        self._t0 = time.monotonic()
        self._connect()
        if self._cap is None:
            raise RuntimeError(f"Could not open stream: {self.uri}")
        self._thread = threading.Thread(target=self._reader, name="tablewatch-stream", daemon=True)
        self._thread.start()

    def _connect(self) -> None:
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            cap.release()
            self._cap = None
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._stream_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._cap = cap

    def _reader(self) -> None:
        while not self._stop.is_set():
            if self._cap is None:
                log.warning("Stream %s unavailable, retrying in %.0fs", self.uri, self.reconnect_delay_s)
                self._stop.wait(self.reconnect_delay_s)
                self._connect()
                continue
            ok, image = self._cap.read()
            if not ok:
                log.warning("Lost stream %s", self.uri)
                self._cap.release()
                self._cap = None
                continue
            with self._cond:
                self._latest = (image, time.monotonic())
                self._seq += 1
                self._cond.notify_all()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        if self.target_fps > 0:
            return self.target_fps
        return self._stream_fps if 1.0 <= self._stream_fps <= 120.0 else 25.0

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._size

    def __iter__(self) -> Iterator[Frame]:
        if self._thread is None:
            self.open()
        min_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        last_seq = 0
        last_yield = -float("inf")
        index = 0
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda: self._seq != last_seq or self._stop.is_set(), timeout=1.0)
                if self._seq == last_seq or self._latest is None:
                    continue
                image, captured = self._latest
                last_seq = self._seq
            wait = last_yield + min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
                continue  # grab whatever is newest after the pause
            last_yield = time.monotonic()
            yield Frame(image=image, ts=captured - self._t0, index=index, wall_time=time.time())
            index += 1


def is_stream_uri(uri: str) -> bool:
    return uri.lower().startswith(STREAM_PREFIXES) or uri.isdigit()


def open_source(
    uri: str,
    target_fps: float = 0.0,
    loop: bool = False,
    realtime: bool = False,
    reconnect_delay_s: float = 5.0,
) -> VideoSource:
    """Pick the right source for a file path, stream URL or webcam index."""
    uri = str(uri)
    if is_stream_uri(uri):
        return StreamSource(uri, target_fps=target_fps, reconnect_delay_s=reconnect_delay_s)
    return FileSource(uri, target_fps=target_fps, loop=loop, realtime=realtime)

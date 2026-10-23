"""Person (and tableware) detection.

`Detector` is the interface the pipeline depends on; `UltralyticsDetector` is
the default YOLO implementation. Swap in another model by implementing
`detect()`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from tablewatch.core import Detection, FrameDetections

log = logging.getLogger(__name__)

PERSON_CLASS = "person"


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> FrameDetections:
        """Return people and (optionally) objects found in a BGR image."""


class UltralyticsDetector(Detector):
    """YOLOv8 / YOLO11 via Ultralytics. Weights download automatically on first use."""

    def __init__(
        self,
        model: str = "yolo11n.pt",
        device: str | None = None,
        confidence: float = 0.35,
        iou: float = 0.5,
        image_size: int = 640,
        object_classes: list[str] | None = None,
        object_confidence: float = 0.25,
    ):
        from ultralytics import YOLO  # imported lazily: heavy, and not needed by tests

        self.model = YOLO(model)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.object_confidence = object_confidence
        names: dict[int, str] = self.model.names
        by_name = {name: idx for idx, name in names.items()}
        if PERSON_CLASS not in by_name:
            raise ValueError(f"Model {model} has no '{PERSON_CLASS}' class")
        self.names = names
        self.person_id = by_name[PERSON_CLASS]
        self.object_ids: set[int] = set()
        for name in object_classes or []:
            if name in by_name:
                self.object_ids.add(by_name[name])
            else:
                log.warning("Model has no class %r; ignoring it", name)
        self._classes = sorted({self.person_id, *self.object_ids})

    def detect(self, image: np.ndarray) -> FrameDetections:
        result = self.model.predict(
            image,
            conf=min(self.confidence, self.object_confidence) if self.object_ids else self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            classes=self._classes,
            device=self.device,
            verbose=False,
        )[0]
        out = FrameDetections()
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, conf, cls):
            det = Detection(tuple(float(v) for v in box), float(c), int(k), self.names[int(k)])
            if k == self.person_id:
                if c >= self.confidence:
                    out.people.append(det)
            elif c >= self.object_confidence:
                out.objects.append(det)
        return out

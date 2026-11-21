"""
perception/object_detection.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Object Detection Module (YOLOv11 / Ultralytics)
────────────────────────────────────────────────
Detects: cars, trucks, buses, motorcycles, pedestrians,
         bicycles, traffic lights, stop signs.

Outputs a list of `Detection` dataclasses with class label,
confidence, bounding box, estimated distance and threat level.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[ObjectDetector] ultralytics not found – running in mock mode.")

DEFAULT_YOLO_MODEL = (
    Path(__file__).resolve().parents[2] / "models" / "yolov8n.pt"
)


# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────

# COCO class IDs for traffic-relevant objects
TRAFFIC_CLASS_IDS: Dict[int, str] = {
    0:  "person",        # pedestrian
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic_light",
    11: "stop_sign",
}

# Threat levels for downstream decision making
class ThreatLevel(Enum):
    NONE     = 0
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


THREAT_RULES: Dict[str, ThreatLevel] = {
    "person":        ThreatLevel.CRITICAL,
    "bicycle":       ThreatLevel.HIGH,
    "motorcycle":    ThreatLevel.HIGH,
    "car":           ThreatLevel.MEDIUM,
    "bus":           ThreatLevel.MEDIUM,
    "truck":         ThreatLevel.MEDIUM,
    "traffic_light": ThreatLevel.HIGH,
    "stop_sign":     ThreatLevel.HIGH,
}

# Colour map for bounding boxes
CLASS_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "person":        (0, 0, 255),
    "bicycle":       (255, 165, 0),
    "motorcycle":    (255, 100, 0),
    "car":           (0, 255, 0),
    "bus":           (0, 200, 255),
    "truck":         (100, 0, 255),
    "traffic_light": (255, 255, 0),
    "stop_sign":     (0, 0, 200),
}


# ────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class Detection:
    """A single object detection result."""
    class_name: str
    class_id: int
    confidence: float
    bbox: BBox
    distance_m: float = -1.0         # Estimated distance (metres); -1 = unknown
    threat_level: ThreatLevel = ThreatLevel.NONE
    track_id: Optional[int] = None   # Populated when tracker is active

    def to_dict(self) -> dict:
        return {
            "class":      self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox":       list(self.bbox.as_tuple()),
            "distance_m": round(self.distance_m, 2),
            "threat":     self.threat_level.name,
            "track_id":   self.track_id,
        }


@dataclass
class DetectionFrame:
    """All detections for a single camera frame."""
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    frame_id: int = 0
    annotated_frame: Optional[np.ndarray] = None

    # ── Convenience filters ─────────────────────────────────

    def pedestrians(self) -> List[Detection]:
        return [d for d in self.detections if d.class_name == "person"]

    def vehicles(self) -> List[Detection]:
        return [d for d in self.detections
                if d.class_name in ("car", "truck", "bus", "motorcycle")]

    def traffic_controls(self) -> List[Detection]:
        return [d for d in self.detections
                if d.class_name in ("traffic_light", "stop_sign")]

    def nearest(self) -> Optional[Detection]:
        valid = [d for d in self.detections if d.distance_m > 0]
        return min(valid, key=lambda d: d.distance_m) if valid else None

    def critical_threats(self) -> List[Detection]:
        return [d for d in self.detections
                if d.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)]


# ────────────────────────────────────────────────────────────
# Object Detector
# ────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    YOLOv11-based object detector wrapping Ultralytics API.

    Parameters
    ----------
    model_path        : Path to `.pt` model file (YOLOv11 weights).
    confidence        : Detection confidence threshold (0–1).
    iou_threshold     : NMS IoU threshold.
    img_size          : Input resolution passed to YOLO.
    device            : "cuda" or "cpu".
    track             : Enable ByteTrack multi-object tracking.
    camera_fov_deg    : Horizontal FOV for distance estimation.
    frame_width       : Expected frame width for distance maths.
    known_car_width_m : Reference object width (metres).
    """

    def __init__(
        self,
        model_path: str = str(DEFAULT_YOLO_MODEL),
        confidence: float = 0.45,
        iou_threshold: float = 0.50,
        img_size: int = 640,
        device: str = "cpu",
        track: bool = False,
        camera_fov_deg: float = 110.0,
        frame_width: int = 1280,
        known_car_width_m: float = 2.0,
    ) -> None:
        self.confidence     = confidence
        self.iou_threshold  = iou_threshold
        self.img_size       = img_size
        self.device         = device
        self.track          = track
        self.camera_fov_deg = camera_fov_deg
        self.frame_width    = frame_width
        self.focal_length   = self._compute_focal(frame_width, camera_fov_deg)
        self.known_car_width_m = known_car_width_m

        self._model = None
        self._frame_counter = 0

        if YOLO_AVAILABLE:
            try:
                self._model = YOLO(model_path)
                print(f"[ObjectDetector] Loaded model: {model_path}")
            except Exception as exc:
                print(f"[ObjectDetector] Could not load {model_path}: {exc}")
                print(f"[ObjectDetector] Attempting to load {DEFAULT_YOLO_MODEL} …")
                try:
                    self._model = YOLO(str(DEFAULT_YOLO_MODEL))
                except Exception:
                    print("[ObjectDetector] Falling back to mock mode.")

    # ── Public API ───────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectionFrame:
        """
        Run inference on a BGR frame.
        Returns a `DetectionFrame` with all detections and an annotated image.
        """
        self._frame_counter += 1
        t0 = time.perf_counter()

        if self._model is None:
            return self._mock_detect(frame, t0)

        # Run YOLO inference
        if self.track:
            results = self._model.track(
                frame,
                persist=True,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )
        else:
            results = self._model.predict(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        detections = self._parse_results(results, frame.shape[1])

        df = DetectionFrame(
            detections=detections,
            inference_ms=elapsed_ms,
            frame_id=self._frame_counter,
        )
        df.annotated_frame = self._annotate(frame.copy(), df)
        return df

    # ── Internal Helpers ─────────────────────────────────────

    def _parse_results(self, results, frame_width: int) -> List[Detection]:
        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])

                # Filter to traffic-relevant classes only
                if cls_id not in TRAFFIC_CLASS_IDS:
                    continue

                class_name = TRAFFIC_CLASS_IDS[cls_id]
                conf       = float(box.conf[0])
                xyxy       = box.xyxy[0].cpu().numpy().astype(int)
                bbox       = BBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3])

                track_id: Optional[int] = None
                if self.track and boxes.id is not None:
                    track_id = int(boxes.id[i])

                dist = self._estimate_distance(bbox, class_name, frame_width)
                threat = self._assess_threat(class_name, dist)

                detections.append(Detection(
                    class_name=class_name,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=bbox,
                    distance_m=dist,
                    threat_level=threat,
                    track_id=track_id,
                ))

        # Sort by distance (closest first)
        detections.sort(key=lambda d: d.distance_m if d.distance_m > 0 else 9999)
        return detections

    def _estimate_distance(
        self, bbox: BBox, class_name: str, frame_width: int
    ) -> float:
        """
        Mono-camera distance estimate using apparent width.
        distance = (real_width × focal_length) / apparent_width_px
        """
        known_widths = {
            "car":        2.0,
            "truck":      2.5,
            "bus":        2.5,
            "motorcycle": 0.8,
            "bicycle":    0.6,
            "person":     0.5,
        }
        real_w = known_widths.get(class_name, 1.5)
        apparent_w = max(bbox.width, 1)
        dist = (real_w * self.focal_length) / apparent_w
        return round(min(dist, 120.0), 2)

    @staticmethod
    def _assess_threat(class_name: str, distance_m: float) -> ThreatLevel:
        base = THREAT_RULES.get(class_name, ThreatLevel.LOW)
        if distance_m <= 0:
            return base
        # Escalate based on proximity
        if distance_m < 5.0:
            return ThreatLevel.CRITICAL
        if distance_m < 10.0 and base.value >= ThreatLevel.MEDIUM.value:
            return ThreatLevel.CRITICAL
        if distance_m < 20.0 and base == ThreatLevel.MEDIUM:
            return ThreatLevel.HIGH
        return base

    @staticmethod
    def _compute_focal(width_px: int, fov_deg: float) -> float:
        import math
        return (width_px / 2.0) / math.tan(math.radians(fov_deg / 2.0))

    # ── Annotation ───────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, df: DetectionFrame) -> np.ndarray:
        for det in df.detections:
            colour = CLASS_COLOURS.get(det.class_name, (200, 200, 200))
            x1, y1, x2, y2 = det.bbox.as_tuple()

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Label background
            label = (
                f"{det.class_name} {det.confidence:.2f}"
                f"  {det.distance_m:.1f}m"
                + (f" #{det.track_id}" if det.track_id is not None else "")
            )
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), colour, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Threat indicator dot
            cx, cy = det.bbox.center
            threat_colours = {
                ThreatLevel.NONE:     (200, 200, 200),
                ThreatLevel.LOW:      (0, 255, 0),
                ThreatLevel.MEDIUM:   (0, 165, 255),
                ThreatLevel.HIGH:     (0, 50, 255),
                ThreatLevel.CRITICAL: (0, 0, 255),
            }
            cv2.circle(frame, (cx, cy), 5,
                       threat_colours.get(det.threat_level, (255, 255, 255)), -1)

        # HUD summary
        cv2.putText(frame,
                    f"Objects: {len(df.detections)}  "
                    f"Inf: {df.inference_ms:.1f}ms",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return frame

    # ── Mock Mode ────────────────────────────────────────────

    def _mock_detect(self, frame: np.ndarray, t0: float) -> DetectionFrame:
        """Return a plausible synthetic detection for testing without YOLO."""
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        h, w = frame.shape[:2]
        mock_det = Detection(
            class_name="car",
            class_id=2,
            confidence=0.91,
            bbox=BBox(w // 2 - 80, h // 2 - 40, w // 2 + 80, h // 2 + 40),
            distance_m=18.5,
            threat_level=ThreatLevel.MEDIUM,
        )
        df = DetectionFrame(
            detections=[mock_det],
            inference_ms=elapsed_ms,
            frame_id=self._frame_counter,
        )
        df.annotated_frame = self._annotate(frame.copy(), df)
        return df


# ────────────────────────────────────────────────────────────
# Quick Test
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    detector = ObjectDetector(model_path=str(DEFAULT_YOLO_MODEL), device="cpu")
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    df = detector.detect(blank)
    print("=" * 50)
    print("Object Detection Test")
    for d in df.detections:
        print(f"  {d.to_dict()}")
    print(f"  Inference time: {df.inference_ms:.1f} ms")
    print("=" * 50)

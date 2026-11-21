"""
perception/traffic_sign_detection.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Traffic Sign Detection & Classification
────────────────────────────────────────
Two-stage pipeline:
  Stage 1 – YOLO detects sign ROIs (stop_sign, traffic_light)
  Stage 2 – CNN classifier assigns fine-grained sign type
             (STOP / SPEED_LIMIT / TRAFFIC_LIGHT_RED|GREEN|YELLOW /
              TURN_LEFT / TURN_RIGHT / YIELD / NO_ENTRY)

When no classifier weights are available the module falls back to
YOLO-only detection with heuristic colour-based traffic-light parsing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

DEFAULT_YOLO_MODEL = (
    Path(__file__).resolve().parents[2] / "models" / "yolov8n.pt"
)


# ────────────────────────────────────────────────────────────
# Sign Classes & Semantics
# ────────────────────────────────────────────────────────────

class SignType(Enum):
    UNKNOWN           = auto()
    STOP              = auto()
    SPEED_LIMIT_30    = auto()
    SPEED_LIMIT_50    = auto()
    SPEED_LIMIT_60    = auto()
    SPEED_LIMIT_80    = auto()
    SPEED_LIMIT_100   = auto()
    TRAFFIC_LIGHT_RED = auto()
    TRAFFIC_LIGHT_YELLOW = auto()
    TRAFFIC_LIGHT_GREEN = auto()
    TURN_LEFT         = auto()
    TURN_RIGHT        = auto()
    GO_STRAIGHT       = auto()
    YIELD             = auto()
    NO_ENTRY          = auto()
    PEDESTRIAN_CROSSING = auto()


# Map sign types to required vehicle actions
SIGN_ACTIONS: Dict[SignType, str] = {
    SignType.STOP:                   "STOP",
    SignType.TRAFFIC_LIGHT_RED:      "STOP",
    SignType.TRAFFIC_LIGHT_YELLOW:   "SLOW_DOWN",
    SignType.TRAFFIC_LIGHT_GREEN:    "GO",
    SignType.TURN_LEFT:              "TURN_LEFT",
    SignType.TURN_RIGHT:             "TURN_RIGHT",
    SignType.GO_STRAIGHT:            "GO_STRAIGHT",
    SignType.YIELD:                  "YIELD",
    SignType.NO_ENTRY:               "STOP",
    SignType.PEDESTRIAN_CROSSING:    "SLOW_DOWN",
    SignType.SPEED_LIMIT_30:         "SET_SPEED_30",
    SignType.SPEED_LIMIT_50:         "SET_SPEED_50",
    SignType.SPEED_LIMIT_60:         "SET_SPEED_60",
    SignType.SPEED_LIMIT_80:         "SET_SPEED_80",
    SignType.SPEED_LIMIT_100:        "SET_SPEED_100",
}

SIGN_COLOURS: Dict[SignType, Tuple[int, int, int]] = {
    SignType.STOP:                   (0, 0, 200),
    SignType.TRAFFIC_LIGHT_RED:      (0, 0, 255),
    SignType.TRAFFIC_LIGHT_YELLOW:   (0, 200, 255),
    SignType.TRAFFIC_LIGHT_GREEN:    (0, 200, 0),
    SignType.TURN_LEFT:              (255, 100, 0),
    SignType.TURN_RIGHT:             (255, 100, 0),
    SignType.YIELD:                  (0, 140, 255),
    SignType.NO_ENTRY:               (0, 0, 180),
}


# ────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────

@dataclass
class SignDetection:
    sign_type: SignType
    confidence: float
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    required_action: str = ""
    distance_m: float = -1.0

    def __post_init__(self) -> None:
        self.required_action = SIGN_ACTIONS.get(self.sign_type, "NONE")

    def to_dict(self) -> dict:
        return {
            "sign":            self.sign_type.name,
            "confidence":      round(self.confidence, 4),
            "bbox":            list(self.bbox),
            "required_action": self.required_action,
            "distance_m":      round(self.distance_m, 2),
        }


@dataclass
class TrafficSignFrame:
    signs: List[SignDetection] = field(default_factory=list)
    inference_ms: float = 0.0
    annotated_frame: Optional[np.ndarray] = None

    def get_action(self) -> str:
        """Highest-priority required action from all detected signs."""
        priority = [
            "STOP", "SLOW_DOWN", "YIELD",
            "TURN_LEFT", "TURN_RIGHT", "GO_STRAIGHT",
            "SET_SPEED_30", "SET_SPEED_50",
            "GO", "NONE",
        ]
        actions = {s.required_action for s in self.signs}
        for act in priority:
            if act in actions:
                return act
        return "NONE"

    def is_red_light(self) -> bool:
        return any(s.sign_type == SignType.TRAFFIC_LIGHT_RED for s in self.signs)

    def is_green_light(self) -> bool:
        return any(s.sign_type == SignType.TRAFFIC_LIGHT_GREEN for s in self.signs)

    def is_stop_sign(self) -> bool:
        return any(s.sign_type == SignType.STOP for s in self.signs)


# ────────────────────────────────────────────────────────────
# Lightweight CNN Classifier (fallback / fine-grained)
# ────────────────────────────────────────────────────────────

class SignClassifierCNN(nn.Module if TORCH_AVAILABLE else object):
    """
    Tiny MobileNetV2-style classifier for 48×48 sign crops.
    Used in Stage 2 to refine YOLO detections.
    """
    NUM_CLASSES = len(SignType)

    def __init__(self) -> None:
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ────────────────────────────────────────────────────────────
# Traffic Sign Detector
# ────────────────────────────────────────────────────────────

class TrafficSignDetector:
    """
    Two-stage traffic-sign detector.

    Parameters
    ----------
    yolo_model_path      : YOLOv8/11 model path (for stage-1 detection).
    classifier_path      : Path to trained CNN weights (stage-2). Optional.
    confidence           : YOLO confidence threshold.
    device               : "cuda" or "cpu".
    """

    IMG_SIZE = 48  # CNN input size

    def __init__(
        self,
        yolo_model_path: str = str(DEFAULT_YOLO_MODEL),
        classifier_path: Optional[str] = None,
        confidence: float = 0.40,
        device: str = "cpu",
    ) -> None:
        self.confidence = confidence
        self.device = device

        # Stage 1: YOLO for sign ROI detection
        self._yolo = None
        if YOLO_AVAILABLE:
            try:
                self._yolo = YOLO(yolo_model_path)
            except Exception as exc:
                print(f"[TrafficSignDetector] YOLO load failed: {exc}")

        # Stage 2: CNN classifier
        self._classifier = None
        self._transform = None
        if TORCH_AVAILABLE and classifier_path:
            try:
                model = SignClassifierCNN()
                model.load_state_dict(torch.load(classifier_path, map_location=device))
                model.eval()
                self._classifier = model.to(device)
                self._transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((self.IMG_SIZE, self.IMG_SIZE)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                ])
                print("[TrafficSignDetector] CNN classifier loaded.")
            except Exception as exc:
                print(f"[TrafficSignDetector] Classifier load failed: {exc}")

    # ── Public API ───────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> TrafficSignFrame:
        """Detect and classify traffic signs in a BGR frame."""
        t0 = time.perf_counter()

        if self._yolo is None:
            return self._heuristic_detect(frame, t0)

        results = self._yolo.predict(
            frame,
            conf=self.confidence,
            verbose=False,
            device=self.device,
        )
        signs = self._parse_yolo(results, frame)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        tsf = TrafficSignFrame(signs=signs, inference_ms=elapsed_ms)
        tsf.annotated_frame = self._annotate(frame.copy(), tsf)
        return tsf

    # ── Internal ─────────────────────────────────────────────

    def _parse_yolo(self, results, frame: np.ndarray) -> List[SignDetection]:
        SIGN_CLASS_IDS = {9: "traffic_light", 11: "stop_sign"}
        signs: List[SignDetection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in SIGN_CLASS_IDS:
                    continue
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                bbox = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                crop = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

                sign_type = self._classify_crop(crop, cls_id)
                dist = self._estimate_sign_distance(xyxy[3] - xyxy[1])
                signs.append(SignDetection(
                    sign_type=sign_type,
                    confidence=conf,
                    bbox=bbox,
                    distance_m=dist,
                ))

        return signs

    def _classify_crop(self, crop: np.ndarray, yolo_cls_id: int) -> SignType:
        """Use CNN classifier if available, else use heuristic colour check."""
        if crop.size == 0:
            return SignType.UNKNOWN

        # CNN stage-2 classification
        if self._classifier is not None and self._transform is not None:
            try:
                tensor = self._transform(crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self._classifier(tensor)
                    idx = logits.argmax(dim=1).item()
                return list(SignType)[idx]
            except Exception:
                pass

        # Heuristic fallback
        if yolo_cls_id == 11:  # stop_sign
            return SignType.STOP
        if yolo_cls_id == 9:   # traffic_light
            return self._classify_traffic_light_colour(crop)

        return SignType.UNKNOWN

    def _classify_traffic_light_colour(self, crop: np.ndarray) -> SignType:
        """
        Determine traffic-light state by HSV colour segmentation.
        Crop is split into top/mid/bottom thirds; the dominant
        coloured region determines the active light.
        """
        if crop.size == 0:
            return SignType.UNKNOWN

        h_total = crop.shape[0]
        top    = crop[:h_total // 3]
        mid    = crop[h_total // 3: 2 * h_total // 3]
        bottom = crop[2 * h_total // 3:]

        def dominant_colour(region: np.ndarray) -> str:
            if region.size == 0:
                return "none"
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            # Red (hue wraps around)
            red1 = cv2.inRange(hsv, (0, 100, 100),   (10, 255, 255))
            red2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
            red  = cv2.bitwise_or(red1, red2)
            # Yellow
            yel  = cv2.inRange(hsv, (20, 100, 100),  (35, 255, 255))
            # Green
            grn  = cv2.inRange(hsv, (40, 60, 60),    (90, 255, 255))

            counts = {"red": red.sum(), "yellow": yel.sum(), "green": grn.sum()}
            best = max(counts, key=counts.get)
            return best if counts[best] > 500 else "none"

        top_col    = dominant_colour(top)
        mid_col    = dominant_colour(mid)
        bottom_col = dominant_colour(bottom)

        if top_col == "red":
            return SignType.TRAFFIC_LIGHT_RED
        if mid_col == "yellow":
            return SignType.TRAFFIC_LIGHT_YELLOW
        if bottom_col == "green":
            return SignType.TRAFFIC_LIGHT_GREEN

        return SignType.TRAFFIC_LIGHT_RED  # conservative default

    @staticmethod
    def _estimate_sign_distance(height_px: int) -> float:
        """
        Rough distance from apparent sign height.
        Standard stop-sign real height ≈ 0.9 m; focal ≈ 800 px.
        """
        if height_px == 0:
            return -1.0
        return round(min((0.9 * 800) / height_px, 100.0), 1)

    # ── Heuristic-only mode ──────────────────────────────────

    def _heuristic_detect(
        self, frame: np.ndarray, t0: float
    ) -> TrafficSignFrame:
        """
        No YOLO available – scan frame quadrants for colour blobs
        that suggest traffic lights or stop signs.
        """
        signs: List[SignDetection] = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Look for large red blobs (stop sign / red light)
        red1  = cv2.inRange(hsv, (0, 120, 70),   (10, 255, 255))
        red2  = cv2.inRange(hsv, (160, 120, 70), (180, 255, 255))
        red   = cv2.bitwise_or(red1, red2)
        cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            signs.append(SignDetection(
                sign_type=SignType.TRAFFIC_LIGHT_RED,
                confidence=0.55,
                bbox=(x, y, x + w, y + h),
                distance_m=self._estimate_sign_distance(h),
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        tsf = TrafficSignFrame(signs=signs[:3], inference_ms=elapsed_ms)
        tsf.annotated_frame = self._annotate(frame.copy(), tsf)
        return tsf

    # ── Annotation ───────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, tsf: TrafficSignFrame) -> np.ndarray:
        for sign in tsf.signs:
            colour = SIGN_COLOURS.get(sign.sign_type, (200, 200, 200))
            x1, y1, x2, y2 = sign.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
            label = (
                f"{sign.sign_type.name.replace('_', ' ')} "
                f"{sign.confidence:.2f}  {sign.distance_m:.1f}m"
            )
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        # Show current required action
        action = tsf.get_action()
        if action != "NONE":
            cv2.putText(frame, f"ACTION: {action}",
                        (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return frame


# ────────────────────────────────────────────────────────────
# Quick Test
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    detector = TrafficSignDetector(device="cpu")
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Paint a red blob to simulate red light
    cv2.circle(blank, (640, 200), 50, (0, 0, 200), -1)
    tsf = detector.detect(blank)
    print("=" * 50)
    print("Traffic Sign Detection Test")
    for s in tsf.signs:
        print(f"  {s.to_dict()}")
    print(f"  Required action : {tsf.get_action()}")
    print(f"  Inference time  : {tsf.inference_ms:.1f} ms")
    print("=" * 50)

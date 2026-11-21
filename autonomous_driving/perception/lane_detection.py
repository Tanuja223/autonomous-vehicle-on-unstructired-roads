"""
perception/lane_detection.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lane Detection Module
─────────────────────
Implements a classical computer-vision pipeline:
  1. Undistort & colour-space transform
  2. ROI masking
  3. Canny edge detection
  4. Probabilistic Hough line detection
  5. Lane-line fitting (1st-order polynomial)
  6. Lane-centre & heading-angle computation

All public methods return a `LaneResult` dataclass so
downstream modules consume a clean, typed interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────

@dataclass
class Line:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def slope(self) -> float:
        dx = self.x2 - self.x1
        return (self.y2 - self.y1) / dx if dx != 0 else float("inf")

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class LaneResult:
    """Output contract for the lane-detection pipeline."""
    lane_center_x: float = 0.0          # Pixel x of estimated lane centre
    lane_angle: float = 0.0             # Steering angle (degrees from vertical)
    lateral_offset: float = 0.0        # Metres from lane centre (+ = right)
    left_line: Optional[Line] = None
    right_line: Optional[Line] = None
    confidence: float = 0.0            # 0–1 detection quality
    annotated_frame: Optional[np.ndarray] = None
    lane_mask: Optional[np.ndarray] = None


# ────────────────────────────────────────────────────────────
# Lane Detector
# ────────────────────────────────────────────────────────────

class LaneDetector:
    """
    Classical Hough-transform lane detector.

    Parameters
    ----------
    canny_low, canny_high   : Canny edge-detection thresholds
    hough_threshold         : Minimum Hough votes
    hough_min_line_len      : Minimum pixel length of a Hough line
    hough_max_line_gap      : Maximum gap to bridge in a Hough line
    roi_height_ratio        : Top of the ROI as a fraction of frame height
                              (0.5 = lower half of image)
    lane_width_meters       : Assumed lane width for lateral-offset calc
    camera_fov_deg          : Horizontal FOV of camera
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 40,
        hough_min_line_len: int = 80,
        hough_max_line_gap: int = 20,
        roi_height_ratio: float = 0.55,
        lane_width_meters: float = 3.7,
        camera_fov_deg: float = 110.0,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap
        self.roi_height_ratio = roi_height_ratio
        self.lane_width_meters = lane_width_meters
        self.camera_fov_deg = camera_fov_deg

        # Temporal smoothing buffers
        self._left_history: List[Line] = []
        self._right_history: List[Line] = []
        self._history_len = 8

    # ── Public API ───────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> LaneResult:
        """
        Run the full lane-detection pipeline on a BGR frame.

        Returns a `LaneResult` with all computed fields populated.
        """
        h, w = frame.shape[:2]

        # 1. Pre-process
        gray = self._preprocess(frame)

        # 2. Region-of-interest mask
        roi, roi_mask = self._apply_roi(gray, h, w)

        # 3. Edge detection
        edges = cv2.Canny(roi, self.canny_low, self.canny_high)

        # 4. Hough lines
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.hough_max_line_gap,
        )

        # 5. Classify & fit
        left, right = self._classify_lines(raw_lines, h, w)
        left_avg = self._smooth(left, self._left_history, "left")
        right_avg = self._smooth(right, self._right_history, "right")

        # 6. Compute driving info
        result = self._compute_lane_info(left_avg, right_avg, h, w)

        # 7. Annotate
        result.annotated_frame = self._annotate(
            frame.copy(), left_avg, right_avg, result, h, w
        )
        result.lane_mask = self._build_lane_mask(left_avg, right_avg, h, w)
        return result

    # ── Internal Pipeline Steps ──────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale with bilateral smoothing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    def _apply_roi(
        self, gray: np.ndarray, h: int, w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mask everything outside the trapezoidal driving region."""
        roi_top = int(h * self.roi_height_ratio)
        vertices = np.array(
            [[
                (int(w * 0.05), h),
                (int(w * 0.42), roi_top),
                (int(w * 0.58), roi_top),
                (int(w * 0.95), h),
            ]],
            dtype=np.int32,
        )
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(gray, gray, mask=mask), mask

    def _classify_lines(
        self,
        raw_lines: Optional[np.ndarray],
        h: int,
        w: int,
    ) -> Tuple[List[Line], List[Line]]:
        """Split Hough lines into left / right groups by slope & position."""
        left, right = [], []
        if raw_lines is None:
            return left, right

        mid_x = w / 2
        for seg in raw_lines:
            if len(seg.shape) == 1:
                     x1, y1, x2, y2 = seg
            else:
                    x1, y1, x2, y2 = seg[0]
            ln = Line(x1, y1, x2, y2)

            # Filter near-horizontal noise
            if abs(ln.slope) < 0.4:
                continue
            # Filter near-vertical noise
            if abs(ln.slope) > 8.0:
                continue

            if ln.slope < 0 and x1 < mid_x and x2 < mid_x:
                left.append(ln)
            elif ln.slope > 0 and x1 > mid_x and x2 > mid_x:
                right.append(ln)

        return left, right

    def _fit_line(self, lines: List[Line], h: int, bottom_y: int, top_y: int) -> Optional[Line]:
        """Fit a single representative line by averaging slope & intercept."""
        if not lines:
            return None

        slopes = [ln.slope for ln in lines]
        intercepts = [ln.y1 - ln.slope * ln.x1 for ln in lines]

        # Weight by line length
        weights = [ln.length for ln in lines]
        total_w = sum(weights)
        if total_w == 0:
            return None

        avg_slope = sum(s * w for s, w in zip(slopes, weights)) / total_w
        avg_intercept = sum(i * w for i, w in zip(intercepts, weights)) / total_w

        if avg_slope == 0:
            return None

        x_bottom = int((bottom_y - avg_intercept) / avg_slope)
        x_top = int((top_y - avg_intercept) / avg_slope)
        return Line(x_bottom, bottom_y, x_top, top_y)

    def _smooth(
        self,
        candidates: List[Line],
        history: List[Line],
        side: str,
    ) -> Optional[Line]:
        """Temporally smooth the detected line using a rolling buffer."""
        h_placeholder = 720  # Placeholder; actual frame dims used upstream
        fitted = self._fit_line(candidates, h_placeholder, h_placeholder, int(h_placeholder * self.roi_height_ratio))
        if fitted is not None:
            history.append(fitted)
        if len(history) > self._history_len:
            history.pop(0)
        if not history:
            return None
        # Average across history
        avg_x1 = int(np.mean([l.x1 for l in history]))
        avg_y1 = int(np.mean([l.y1 for l in history]))
        avg_x2 = int(np.mean([l.x2 for l in history]))
        avg_y2 = int(np.mean([l.y2 for l in history]))
        return Line(avg_x1, avg_y1, avg_x2, avg_y2)

    def _compute_lane_info(
        self,
        left: Optional[Line],
        right: Optional[Line],
        h: int,
        w: int,
    ) -> LaneResult:
        """Compute lane-centre x, heading angle, lateral offset, and confidence."""
        result = LaneResult()

        bottom_y = h
        mid_x = w / 2

        if left is not None and right is not None:
            # Bottom intersections with image bottom
            left_x_bottom = left.x1
            right_x_bottom = right.x1
            result.lane_center_x = (left_x_bottom + right_x_bottom) / 2.0
            result.confidence = 1.0

        elif left is not None:
            result.lane_center_x = left.x1 + self._lane_width_px(w) / 2
            result.confidence = 0.5

        elif right is not None:
            result.lane_center_x = right.x1 - self._lane_width_px(w) / 2
            result.confidence = 0.5

        else:
            result.lane_center_x = mid_x
            result.confidence = 0.0

        # Heading angle: angle of lane-centre vector from image bottom-centre
        centre_offset_px = result.lane_center_x - mid_x
        # Approximate: 1 px → (fov/w) degrees
        px_per_deg = w / self.camera_fov_deg
        result.lane_angle = centre_offset_px / px_per_deg

        # Lateral offset in metres
        px_per_m = self._lane_width_px(w) / self.lane_width_meters
        result.lateral_offset = centre_offset_px / px_per_m if px_per_m > 0 else 0.0

        result.left_line = left
        result.right_line = right
        return result

    def _lane_width_px(self, frame_width: int) -> float:
        """Estimate lane width in pixels based on FOV assumption."""
        return frame_width * 0.4

    # ── Visualisation ────────────────────────────────────────

    def _annotate(
        self,
        frame: np.ndarray,
        left: Optional[Line],
        right: Optional[Line],
        result: LaneResult,
        h: int,
        w: int,
    ) -> np.ndarray:
        overlay = frame.copy()

        # Draw lane lines
        if left is not None:
            cv2.line(overlay, (left.x1, left.y1), (left.x2, left.y2), (0, 255, 0), 4)
        if right is not None:
            cv2.line(overlay, (right.x1, right.y1), (right.x2, right.y2), (0, 255, 0), 4)

        # Fill lane polygon
        if left is not None and right is not None:
            pts = np.array(
                [[left.x1, left.y1], [left.x2, left.y2],
                 [right.x2, right.y2], [right.x1, right.y1]],
                np.int32,
            ).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # Lane-centre indicator
        cx = int(result.lane_center_x)
        cv2.circle(frame, (cx, h - 30), 8, (0, 0, 255), -1)
        cv2.line(frame, (int(w / 2), h - 30), (cx, h - 30), (255, 255, 0), 2)

        # HUD text
        cv2.putText(frame, f"Angle: {result.lane_angle:.2f} deg",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Offset: {result.lateral_offset:.2f} m",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {result.confidence:.0%}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame

    def _build_lane_mask(
        self,
        left: Optional[Line],
        right: Optional[Line],
        h: int,
        w: int,
    ) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        if left is not None and right is not None:
            pts = np.array(
                [[left.x1, left.y1], [left.x2, left.y2],
                 [right.x2, right.y2], [right.x1, right.y1]],
                np.int32,
            ).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        return mask


# ────────────────────────────────────────────────────────────
# Quick Test
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    detector = LaneDetector()

    # Test on a synthetic blank road image
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw fake lane lines on blank frame
    cv2.line(test_frame, (400, 720), (560, 400), (200, 200, 200), 10)
    cv2.line(test_frame, (880, 720), (720, 400), (200, 200, 200), 10)
    # Add road colour
    test_frame[400:] = (50, 50, 50)

    result = detector.detect(test_frame)
    print("=" * 50)
    print("Lane Detection Test")
    print(f"  Lane centre X : {result.lane_center_x:.1f} px")
    print(f"  Heading angle : {result.lane_angle:.2f}°")
    print(f"  Lateral offset: {result.lateral_offset:.2f} m")
    print(f"  Confidence    : {result.confidence:.0%}")
    print("=" * 50)

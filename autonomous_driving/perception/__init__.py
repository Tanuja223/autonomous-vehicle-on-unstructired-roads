"""
perception/depth_estimation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Monocular Depth Estimation
───────────────────────────
Primary backend  : Intel MiDaS (DPT_Large / MiDaS_small)
Fallback backend : Disparity-from-focus heuristic (no DNN needed)

Outputs a metric depth map and per-object distance estimates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[DepthEstimator] PyTorch not found – using heuristic depth.")

try:
    from perception.object_detection import BBox, Detection
except ImportError:
    BBox = None  # type: ignore
    Detection = None  # type: ignore


# ────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────

@dataclass
class DepthResult:
    depth_map: np.ndarray           # HxW float32 array (relative or metric)
    depth_colormap: np.ndarray      # HxW×3 uint8 for visualisation
    min_distance_m: float           # Closest point in the ROI (metres)
    object_distances: Dict[str, float]  # {class_name: distance_m}
    inference_ms: float

    def distance_at(self, x: int, y: int) -> float:
        """Return metric distance at pixel (x, y)."""
        h, w = self.depth_map.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        return float(self.depth_map[y, x])


# ────────────────────────────────────────────────────────────
# Depth Estimator
# ────────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Monocular depth estimator backed by MiDaS or a heuristic fallback.

    Parameters
    ----------
    model_type  : "DPT_Large", "DPT_Hybrid", or "MiDaS_small"
    device      : "cuda" or "cpu"
    max_dist_m  : Clip depth to this maximum (metres)
    use_metric  : If True, scale output to approximate metric depth
                  using a calibration factor (suitable for CARLA where
                  depth camera provides ground-truth scale reference).
    """

    # Rough scale factor from relative MiDaS output to metres.
    # Calibrate against CARLA depth camera or LiDAR.
    SCALE_FACTOR = 8.0

    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: str = "cpu",
        max_dist_m: float = 80.0,
        use_metric: bool = True,
    ) -> None:
        self.device     = device
        self.max_dist_m = max_dist_m
        self.use_metric = use_metric

        self._model       = None
        self._transform   = None
        self._mode        = "heuristic"

        if TORCH_AVAILABLE:
            self._load_midas(model_type)

    # ── Public API ───────────────────────────────────────────

    def estimate(
        self,
        frame: np.ndarray,
        detections: Optional[List] = None,  # List[Detection]
    ) -> DepthResult:
        """
        Estimate depth from a BGR frame.

        Parameters
        ----------
        frame       : BGR image from camera.
        detections  : Optional list of `Detection` objects.
                      If provided, the mean depth inside each bounding
                      box is computed and stored in `object_distances`.
        """
        t0 = time.perf_counter()

        if self._model is not None:
            depth_raw = self._midas_infer(frame)
        else:
            depth_raw = self._heuristic_depth(frame)

        # Convert to approximate metric depth
        depth_m = self._to_metric(depth_raw)
        depth_m = np.clip(depth_m, 0.3, self.max_dist_m).astype(np.float32)

        # Closest obstacle in central ROI
        h, w = depth_m.shape
        roi = depth_m[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        min_dist = float(roi.min())

        # Per-object distances
        obj_distances: Dict[str, float] = {}
        if detections:
            for det in detections:
                dist = self._depth_in_bbox(depth_m, det.bbox)
                key  = f"{det.class_name}_{det.track_id or id(det)}"
                obj_distances[key] = dist
                # Back-fill detection distance
                det.distance_m = dist

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return DepthResult(
            depth_map       = depth_m,
            depth_colormap  = self._colormap(depth_m),
            min_distance_m  = min_dist,
            object_distances= obj_distances,
            inference_ms    = elapsed_ms,
        )

    def estimate_object_distance(
        self, frame: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]
    ) -> float:
        """Quick single-object distance estimate without full depth map."""
        result = self.estimate(frame)
        x1, y1, x2, y2 = bbox_xyxy
        h, w = result.depth_map.shape
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        roi = result.depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return -1.0
        # Use 10th percentile to get closest (not mean)
        return float(np.percentile(roi, 10))

    # ── MiDaS Backend ────────────────────────────────────────

    def _load_midas(self, model_type: str) -> None:
        try:
            print(f"[DepthEstimator] Loading MiDaS ({model_type}) …")
            self._model = torch.hub.load(
                "intel-isl/MiDaS", model_type, trust_repo=True
            )
            self._model.to(self.device).eval()

            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            if model_type in ("DPT_Large", "DPT_Hybrid"):
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform

            self._mode = "midas"
            print(f"[DepthEstimator] MiDaS ({model_type}) loaded on {self.device}.")
        except Exception as exc:
            print(f"[DepthEstimator] MiDaS load failed ({exc}) – using heuristic.")

    def _midas_infer(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self.device)
        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy()
        # MiDaS outputs inverse depth – flip so near=small, far=large
        depth = 1.0 / (depth + 1e-6)
        return depth.astype(np.float32)

    # ── Heuristic Fallback ───────────────────────────────────

    def _heuristic_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Cheap depth proxy using vertical position + edge density.
        Objects higher in the image are assumed further away.
        """
        h, w = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blur  = cv2.GaussianBlur(gray, (21, 21), 0)
        edges = cv2.Canny(blur.astype(np.uint8), 30, 80).astype(np.float32)

        # Vertical gradient (row index ∝ closeness)
        y_idx = np.linspace(0.2, 1.0, h, dtype=np.float32)[:, None]
        v_grad = np.broadcast_to(y_idx, (h, w)).copy()

        # Blend: areas with many edges are treated as objects (closer)
        edge_norm = edges / (edges.max() + 1e-6)
        depth_proxy = v_grad + 0.3 * edge_norm
        return depth_proxy

    # ── Metric Conversion ────────────────────────────────────

    def _to_metric(self, depth_raw: np.ndarray) -> np.ndarray:
        if not self.use_metric:
            return depth_raw
        d_min, d_max = depth_raw.min(), depth_raw.max()
        if d_max - d_min < 1e-6:
            return np.full_like(depth_raw, self.max_dist_m / 2)
        # Normalize to [0, 1] then scale
        norm = (depth_raw - d_min) / (d_max - d_min)
        return (norm * self.max_dist_m).astype(np.float32)

    # ── Utilities ────────────────────────────────────────────

    @staticmethod
    def _depth_in_bbox(depth_map: np.ndarray, bbox) -> float:
        """Mean depth inside a bounding-box region."""
        if bbox is None:
            return -1.0
        if hasattr(bbox, "x1"):
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        else:
            x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return -1.0
        return float(np.percentile(roi, 20))

    @staticmethod
    def _colormap(depth_m: np.ndarray) -> np.ndarray:
        """Create PLASMA colour-map for depth visualisation."""
        d_min, d_max = depth_m.min(), depth_m.max()
        if d_max - d_min < 1e-6:
            norm = np.zeros_like(depth_m, dtype=np.uint8)
        else:
            norm = ((depth_m - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)


# ────────────────────────────────────────────────────────────
# Quick Test
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    estimator = DepthEstimator(model_type="MiDaS_small", device="cpu")
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    result = estimator.estimate(test_frame)
    print("=" * 50)
    print("Depth Estimation Test")
    print(f"  Mode             : {estimator._mode}")
    print(f"  Depth map shape  : {result.depth_map.shape}")
    print(f"  Min distance     : {result.min_distance_m:.2f} m")
    print(f"  Inference time   : {result.inference_ms:.1f} ms")
    print("=" * 50)

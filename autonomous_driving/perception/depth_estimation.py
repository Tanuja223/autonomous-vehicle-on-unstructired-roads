"""
perception/depth_estimation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Monocular Depth Estimation — MiDaS + heuristic fallback.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DepthResult:
    depth_map: np.ndarray
    depth_colormap: np.ndarray
    min_distance_m: float
    object_distances: Dict[str, float]
    inference_ms: float

    def distance_at(self, x: int, y: int) -> float:
        h, w = self.depth_map.shape[:2]
        return float(self.depth_map[max(0,min(y,h-1)), max(0,min(x,w-1))])


class DepthEstimator:
    SCALE_FACTOR = 8.0

    def __init__(self, model_type="MiDaS_small", device="cpu",
                 max_dist_m=80.0, use_metric=True):
        self.device = device
        self.max_dist_m = max_dist_m
        self.use_metric = use_metric
        self._model = None
        self._transform = None
        self._mode = "heuristic"
        if TORCH_AVAILABLE:
            self._load_midas(model_type)

    def estimate(self, frame: np.ndarray, detections=None) -> DepthResult:
        t0 = time.perf_counter()
        depth_raw = self._midas_infer(frame) if self._model else self._heuristic_depth(frame)
        depth_m = np.clip(self._to_metric(depth_raw), 0.3, self.max_dist_m).astype(np.float32)
        h, w = depth_m.shape
        roi = depth_m[h//4:3*h//4, w//4:3*w//4]
        min_dist = float(roi.min())
        obj_distances: Dict[str, float] = {}
        if detections:
            for det in detections:
                dist = self._depth_in_bbox(depth_m, det.bbox)
                obj_distances[f"{det.class_name}_{id(det)}"] = dist
                det.distance_m = dist
        return DepthResult(
            depth_map=depth_m,
            depth_colormap=self._colormap(depth_m),
            min_distance_m=min_dist,
            object_distances=obj_distances,
            inference_ms=(time.perf_counter()-t0)*1000,
        )

    def _load_midas(self, model_type):
        try:
            self._model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            self._model.to(self.device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._transform = transforms.dpt_transform if "DPT" in model_type else transforms.small_transform
            self._mode = "midas"
            print(f"[DepthEstimator] MiDaS ({model_type}) loaded.")
        except Exception as e:
            print(f"[DepthEstimator] MiDaS failed: {e} — using heuristic.")

    def _midas_infer(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch = self._transform(rgb).to(self.device)
        with torch.no_grad():
            pred = self._model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        d = pred.cpu().numpy()
        return (1.0 / (d + 1e-6)).astype(np.float32)

    def _heuristic_depth(self, frame):
        h, w = frame.shape[:2]
        y_idx = np.linspace(0.2, 1.0, h, dtype=np.float32)[:, None]
        return np.broadcast_to(y_idx, (h, w)).copy()

    def _to_metric(self, d):
        lo, hi = d.min(), d.max()
        if hi - lo < 1e-6:
            return np.full_like(d, self.max_dist_m / 2)
        return ((d - lo) / (hi - lo) * self.max_dist_m).astype(np.float32)

    @staticmethod
    def _depth_in_bbox(depth_map, bbox):
        if bbox is None:
            return -1.0
        x1,y1,x2,y2 = (bbox.x1,bbox.y1,bbox.x2,bbox.y2) if hasattr(bbox,"x1") else bbox
        h,w = depth_map.shape
        roi = depth_map[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        return float(np.percentile(roi, 20)) if roi.size else -1.0

    @staticmethod
    def _colormap(depth_m):
        lo,hi = depth_m.min(), depth_m.max()
        norm = ((depth_m-lo)/(hi-lo+1e-6)*255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)


if __name__ == "__main__":
    est = DepthEstimator()
    r = est.estimate(np.random.randint(0,255,(480,640,3),dtype=np.uint8))
    print(f"Mode: {est._mode} | Min dist: {r.min_distance_m:.2f}m | {r.inference_ms:.1f}ms")

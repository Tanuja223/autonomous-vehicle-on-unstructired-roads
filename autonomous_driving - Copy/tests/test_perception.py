"""
tests/test_perception.py — Unit tests for perception modules
Run: pytest tests/ -v
"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from perception.lane_detection import LaneDetector
from perception.object_detection import ObjectDetector, BBox
from perception.depth_estimation import DepthEstimator


def make_frame(h=360, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestLaneDetector:
    def setup_method(self):
        self.det = LaneDetector()

    def test_returns_result(self):
        result = self.det.detect(make_frame())
        assert result is not None

    def test_lane_center_within_frame(self):
        frame = make_frame()
        result = self.det.detect(frame)
        assert 0 <= result.lane_center_x <= 640

    def test_confidence_range(self):
        result = self.det.detect(make_frame())
        assert 0.0 <= result.confidence <= 1.0

    def test_annotated_frame_shape(self):
        frame = make_frame()
        result = self.det.detect(frame)
        assert result.annotated_frame is not None
        assert result.annotated_frame.shape == frame.shape


class TestObjectDetector:
    def setup_method(self):
        self.det = ObjectDetector(device="cpu")

    def test_returns_detection_frame(self):
        df = self.det.detect(make_frame())
        assert df is not None
        assert hasattr(df, "detections")

    def test_inference_time_positive(self):
        df = self.det.detect(make_frame())
        assert df.inference_ms >= 0

    def test_annotated_frame_exists(self):
        df = self.det.detect(make_frame())
        assert df.annotated_frame is not None


class TestDepthEstimator:
    def setup_method(self):
        self.est = DepthEstimator(device="cpu")

    def test_depth_map_shape(self):
        frame = make_frame()
        result = self.est.estimate(frame)
        assert result.depth_map.shape[:2] == frame.shape[:2]

    def test_min_distance_positive(self):
        result = self.est.estimate(make_frame())
        assert result.min_distance_m >= 0

    def test_colormap_3_channels(self):
        result = self.est.estimate(make_frame())
        assert result.depth_colormap.ndim == 3
        assert result.depth_colormap.shape[2] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

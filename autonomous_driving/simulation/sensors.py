"""
simulation/sensors.py
━━━━━━━━━━━━━━━━━━━━━
CARLA Sensor Manager
─────────────────────
Manages all sensor actors attached to the ego vehicle:
  • RGB Camera (front, left, right, rear)
  • Depth Camera
  • Semantic Segmentation Camera
  • LiDAR
  • GNSS / IMU
  • Collision & Lane-Invasion event sensors

Each sensor writes to a thread-safe queue; consumers poll
`get_latest()` for the most recent reading.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


# ────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────

@dataclass
class CameraData:
    image:     np.ndarray        # H×W×3 uint8 BGR
    frame:     int  = 0
    timestamp: float = 0.0
    width:     int  = 0
    height:    int  = 0


@dataclass
class DepthData:
    depth_m:   np.ndarray        # H×W float32 metres
    frame:     int  = 0
    timestamp: float = 0.0


@dataclass
class SemanticData:
    label_map: np.ndarray        # H×W uint8 (CityScapes class IDs)
    coloured:  np.ndarray        # H×W×3 uint8 coloured overlay
    frame:     int  = 0
    timestamp: float = 0.0


@dataclass
class LidarData:
    points:    np.ndarray        # N×4 float32 [x, y, z, intensity]
    frame:     int  = 0
    timestamp: float = 0.0


@dataclass
class GNSSData:
    latitude:  float = 0.0
    longitude: float = 0.0
    altitude:  float = 0.0
    timestamp: float = 0.0


@dataclass
class IMUData:
    accel_x:  float = 0.0
    accel_y:  float = 0.0
    accel_z:  float = 0.0
    gyro_x:   float = 0.0
    gyro_y:   float = 0.0
    gyro_z:   float = 0.0
    compass:  float = 0.0
    timestamp: float = 0.0


@dataclass
class CollisionEvent:
    actor_id:    int    = -1
    actor_type:  str    = ""
    impulse_ms2: float  = 0.0
    timestamp:   float  = 0.0


# ────────────────────────────────────────────────────────────
# Semantic Segmentation Colour Map (CityScapes)
# ────────────────────────────────────────────────────────────

SEMANTIC_COLOURS: Dict[int, Tuple[int, int, int]] = {
    0:  (0,   0,   0),    # Unlabelled
    1:  (70,  70,  70),   # Building
    2:  (100, 40,  40),   # Fence
    3:  (55,  90,  80),   # Other
    4:  (220, 20,  60),   # Pedestrian
    5:  (153, 153, 153),  # Pole
    6:  (157, 234, 50),   # Road Line
    7:  (128, 64,  128),  # Road
    8:  (244, 35,  232),  # Sidewalk
    9:  (107, 142, 35),   # Vegetation
    10: (0,   0,   192),  # Vehicles
    11: (102, 102, 156),  # Wall
    12: (220, 220, 0),    # Traffic Sign
}


def label_to_colour(label_map: np.ndarray) -> np.ndarray:
    """Convert semantic label map (H×W) → coloured image (H×W×3)."""
    h, w = label_map.shape
    coloured = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, colour in SEMANTIC_COLOURS.items():
        mask = label_map == cls_id
        coloured[mask] = colour
    return coloured


# ────────────────────────────────────────────────────────────
# Thread-Safe Sensor Buffer
# ────────────────────────────────────────────────────────────

class SensorBuffer:
    """Single-slot thread-safe buffer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Any = None

    def put(self, data: Any) -> None:
        with self._lock:
            self._data = data

    def get(self) -> Any:
        with self._lock:
            return self._data

    def clear(self) -> None:
        with self._lock:
            self._data = None


# ────────────────────────────────────────────────────────────
# Sensor Manager
# ────────────────────────────────────────────────────────────

class SensorManager:
    """
    Attaches and manages all CARLA sensors for the ego vehicle.

    Usage
    -----
    mgr = SensorManager(world, ego_vehicle, config)
    mgr.setup()
    ...
    cam  = mgr.get_camera()
    lidar= mgr.get_lidar()
    ...
    mgr.destroy()
    """

    def __init__(
        self,
        world,
        ego,
        img_width:  int   = 1280,
        img_height: int   = 720,
        cam_fov:    float = 110.0,
        lidar_range:float = 85.0,
        enable_lidar:    bool = True,
        enable_semantic: bool = True,
        enable_depth:    bool = True,
        enable_imu_gnss: bool = True,
    ) -> None:
        self._world = world
        self._ego   = ego
        self.img_w  = img_width
        self.img_h  = img_height
        self.fov    = cam_fov
        self.lidar_range = lidar_range

        self._enable_lidar    = enable_lidar
        self._enable_semantic = enable_semantic
        self._enable_depth    = enable_depth
        self._enable_imu_gnss = enable_imu_gnss

        self._actors: List[Any] = []

        # Buffers
        self._rgb_buf       = SensorBuffer()
        self._depth_buf     = SensorBuffer()
        self._semantic_buf  = SensorBuffer()
        self._lidar_buf     = SensorBuffer()
        self._gnss_buf      = SensorBuffer()
        self._imu_buf       = SensorBuffer()
        self._collision_buf: deque = deque(maxlen=10)
        self._lane_inv_buf:  deque = deque(maxlen=10)

    # ── Setup ────────────────────────────────────────────────

    def setup(self) -> None:
        """Spawn and attach all configured sensors."""
        if not CARLA_AVAILABLE:
            print("[SensorManager] CARLA not available – mock mode.")
            return

        bp_lib = self._world.get_blueprint_library()

        # ── Front RGB Camera ─────────────────────────────────
        self._attach_camera(bp_lib, carla.Transform(
            carla.Location(x=1.5, z=2.4)), self._on_rgb_front)

        # ── Depth Camera ─────────────────────────────────────
        if self._enable_depth:
            self._attach_depth_cam(bp_lib, carla.Transform(
                carla.Location(x=1.5, z=2.4)))

        # ── Semantic Segmentation ────────────────────────────
        if self._enable_semantic:
            self._attach_semantic_cam(bp_lib, carla.Transform(
                carla.Location(x=1.5, z=2.4)))

        # ── LiDAR ────────────────────────────────────────────
        if self._enable_lidar:
            self._attach_lidar(bp_lib, carla.Transform(
                carla.Location(x=0, z=2.8)))

        # ── GNSS + IMU ───────────────────────────────────────
        if self._enable_imu_gnss:
            self._attach_gnss(bp_lib)
            self._attach_imu(bp_lib)

        # ── Event sensors ────────────────────────────────────
        self._attach_collision(bp_lib)
        self._attach_lane_invasion(bp_lib)

        print(f"[SensorManager] {len(self._actors)} sensors attached.")

    # ── Camera Builders ──────────────────────────────────────

    def _attach_camera(self, bp_lib, transform, callback) -> None:
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.img_w))
        bp.set_attribute("image_size_y", str(self.img_h))
        bp.set_attribute("fov",          str(self.fov))
        bp.set_attribute("sensor_tick",  "0.033")
        actor = self._world.spawn_actor(bp, transform, attach_to=self._ego)
        actor.listen(callback)
        self._actors.append(actor)

    def _attach_depth_cam(self, bp_lib, transform) -> None:
        bp = bp_lib.find("sensor.camera.depth")
        bp.set_attribute("image_size_x", str(self.img_w // 2))
        bp.set_attribute("image_size_y", str(self.img_h // 2))
        bp.set_attribute("fov",          "90")
        actor = self._world.spawn_actor(bp, transform, attach_to=self._ego)
        actor.listen(self._on_depth)
        self._actors.append(actor)

    def _attach_semantic_cam(self, bp_lib, transform) -> None:
        bp = bp_lib.find("sensor.camera.semantic_segmentation")
        bp.set_attribute("image_size_x", str(self.img_w // 2))
        bp.set_attribute("image_size_y", str(self.img_h // 2))
        bp.set_attribute("fov",          str(self.fov))
        actor = self._world.spawn_actor(bp, transform, attach_to=self._ego)
        actor.listen(self._on_semantic)
        self._actors.append(actor)

    def _attach_lidar(self, bp_lib, transform) -> None:
        bp = bp_lib.find("sensor.lidar.ray_cast")
        bp.set_attribute("channels",             "64")
        bp.set_attribute("range",                str(self.lidar_range))
        bp.set_attribute("points_per_second",    "1120000")
        bp.set_attribute("rotation_frequency",   "20")
        bp.set_attribute("upper_fov",            "15")
        bp.set_attribute("lower_fov",            "-25")
        actor = self._world.spawn_actor(bp, transform, attach_to=self._ego)
        actor.listen(self._on_lidar)
        self._actors.append(actor)

    def _attach_gnss(self, bp_lib) -> None:
        bp = bp_lib.find("sensor.other.gnss")
        bp.set_attribute("sensor_tick", "0.1")
        actor = self._world.spawn_actor(
            bp, carla.Transform(), attach_to=self._ego)
        actor.listen(self._on_gnss)
        self._actors.append(actor)

    def _attach_imu(self, bp_lib) -> None:
        bp = bp_lib.find("sensor.other.imu")
        bp.set_attribute("sensor_tick", "0.02")
        actor = self._world.spawn_actor(
            bp, carla.Transform(), attach_to=self._ego)
        actor.listen(self._on_imu)
        self._actors.append(actor)

    def _attach_collision(self, bp_lib) -> None:
        bp    = bp_lib.find("sensor.other.collision")
        actor = self._world.spawn_actor(
            bp, carla.Transform(), attach_to=self._ego)
        actor.listen(self._on_collision)
        self._actors.append(actor)

    def _attach_lane_invasion(self, bp_lib) -> None:
        bp    = bp_lib.find("sensor.other.lane_invasion")
        actor = self._world.spawn_actor(
            bp, carla.Transform(), attach_to=self._ego)
        actor.listen(self._on_lane_invasion)
        self._actors.append(actor)

    # ── Sensor Callbacks ─────────────────────────────────────

    def _on_rgb_front(self, image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        bgr = arr[:, :, :3][:, :, ::-1]   # RGBA → BGR
        self._rgb_buf.put(CameraData(
            image=bgr,
            frame=image.frame,
            timestamp=image.timestamp,
            width=image.width,
            height=image.height,
        ))

    def _on_depth(self, image) -> None:
        # CARLA logarithmic depth decoding
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4)).astype(np.float32)
        normalised = (arr[:, :, 0] + arr[:, :, 1] * 256
                      + arr[:, :, 2] * 256 * 256) / (256 ** 3 - 1)
        depth_m = 1000.0 * normalised    # In metres
        self._depth_buf.put(DepthData(
            depth_m=depth_m.astype(np.float32),
            frame=image.frame,
            timestamp=image.timestamp,
        ))

    def _on_semantic(self, image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        labels = arr[:, :, 2]   # Red channel = semantic class in CARLA
        coloured = label_to_colour(labels)
        self._semantic_buf.put(SemanticData(
            label_map=labels,
            coloured=coloured,
            frame=image.frame,
            timestamp=image.timestamp,
        ))

    def _on_lidar(self, point_cloud) -> None:
        points = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4))   # x, y, z, intensity
        self._lidar_buf.put(LidarData(
            points=points,
            frame=point_cloud.frame,
            timestamp=point_cloud.timestamp,
        ))

    def _on_gnss(self, gnss) -> None:
        self._gnss_buf.put(GNSSData(
            latitude=gnss.latitude,
            longitude=gnss.longitude,
            altitude=gnss.altitude,
            timestamp=gnss.timestamp,
        ))

    def _on_imu(self, imu) -> None:
        self._imu_buf.put(IMUData(
            accel_x=imu.accelerometer.x,
            accel_y=imu.accelerometer.y,
            accel_z=imu.accelerometer.z,
            gyro_x=imu.gyroscope.x,
            gyro_y=imu.gyroscope.y,
            gyro_z=imu.gyroscope.z,
            compass=imu.compass,
            timestamp=imu.timestamp,
        ))

    def _on_collision(self, event) -> None:
        other = event.other_actor
        impulse = event.normal_impulse
        magnitude = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._collision_buf.append(CollisionEvent(
            actor_id=other.id,
            actor_type=other.type_id,
            impulse_ms2=magnitude,
            timestamp=event.timestamp,
        ))

    def _on_lane_invasion(self, event) -> None:
        self._lane_inv_buf.append({
            "markings": [str(m.type) for m in event.crossed_lane_markings],
            "timestamp": event.timestamp,
        })

    # ── Public Getters ───────────────────────────────────────

    def get_camera(self)    -> Optional[CameraData]:   return self._rgb_buf.get()
    def get_depth(self)     -> Optional[DepthData]:    return self._depth_buf.get()
    def get_semantic(self)  -> Optional[SemanticData]: return self._semantic_buf.get()
    def get_lidar(self)     -> Optional[LidarData]:    return self._lidar_buf.get()
    def get_gnss(self)      -> Optional[GNSSData]:     return self._gnss_buf.get()
    def get_imu(self)       -> Optional[IMUData]:      return self._imu_buf.get()

    def pop_collisions(self) -> List[CollisionEvent]:
        events = list(self._collision_buf)
        self._collision_buf.clear()
        return events

    def pop_lane_invasions(self) -> List[dict]:
        events = list(self._lane_inv_buf)
        self._lane_inv_buf.clear()
        return events

    def has_collision(self) -> bool:
        return len(self._collision_buf) > 0

    # ── Cleanup ──────────────────────────────────────────────

    def destroy(self) -> None:
        for actor in self._actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self._actors.clear()
        print("[SensorManager] All sensors destroyed.")


# ────────────────────────────────────────────────────────────
# LiDAR Processing Utilities
# ────────────────────────────────────────────────────────────

class LidarProcessor:
    """Utility methods for processing raw LiDAR point clouds."""

    @staticmethod
    def to_bird_eye_view(
        points: np.ndarray,
        x_range: Tuple[float, float] = (-40, 40),
        y_range: Tuple[float, float] = (-40, 40),
        resolution: float = 0.2,
    ) -> np.ndarray:
        """Project 3-D point cloud to a 2-D bird's-eye-view image."""
        x_size = int((x_range[1] - x_range[0]) / resolution)
        y_size = int((y_range[1] - y_range[0]) / resolution)
        bev = np.zeros((y_size, x_size), dtype=np.float32)

        mask = (
            (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) &
            (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])
        )
        filtered = points[mask]

        xi = ((filtered[:, 0] - x_range[0]) / resolution).astype(int)
        yi = ((filtered[:, 1] - y_range[0]) / resolution).astype(int)
        xi = np.clip(xi, 0, x_size - 1)
        yi = np.clip(yi, 0, y_size - 1)

        # Height encoding
        bev[yi, xi] = np.maximum(bev[yi, xi], filtered[:, 2])

        # Normalise and convert to uint8
        if bev.max() > bev.min():
            bev = (bev - bev.min()) / (bev.max() - bev.min())
        return (bev * 255).astype(np.uint8)

    @staticmethod
    def nearest_obstacle(points: np.ndarray, forward_only: bool = True) -> float:
        """
        Find distance to nearest point cloud cluster.
        If forward_only, restrict to x > 0 (in front of vehicle).
        """
        if points.shape[0] == 0:
            return 999.0
        pts = points[:, :3]
        if forward_only:
            pts = pts[pts[:, 0] > 0.5]
        if pts.shape[0] == 0:
            return 999.0
        dists = np.linalg.norm(pts, axis=1)
        return float(dists.min())


# ────────────────────────────────────────────────────────────
# Quick Test
# ────────────────────────────────────────────────────────────

from typing import Tuple

if __name__ == "__main__":
    print("=" * 50)
    print("Sensor Utilities Test (no CARLA required)")

    # Test LiDAR BEV
    pts = np.random.uniform(-30, 30, (5000, 4)).astype(np.float32)
    pts[:, 2] = np.random.uniform(-2, 3, 5000).astype(np.float32)
    bev = LidarProcessor.to_bird_eye_view(pts)
    print(f"  BEV shape   : {bev.shape}")
    print(f"  BEV max px  : {bev.max()}")
    print(f"  Nearest obs : {LidarProcessor.nearest_obstacle(pts):.2f} m")

    # Semantic colour test
    labels = np.random.randint(0, 13, (360, 640), dtype=np.uint8)
    coloured = label_to_colour(labels)
    print(f"  Semantic map: {coloured.shape}")
    print("=" * 50)

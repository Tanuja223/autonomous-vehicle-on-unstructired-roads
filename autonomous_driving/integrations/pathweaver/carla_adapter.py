"""Convert CARLA world coordinates to PathWeaver's forward/left local frame."""

from __future__ import annotations

import math
from typing import Any, Iterable


EGO_LOCAL_X_M = 5.0
POSITION_VARIANCE_M2 = 0.04
MAX_AGENT_DISTANCE_M = 45.0


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _local_position(
    x: float, y: float, ego_x: float, ego_y: float, ego_yaw: float
) -> tuple[float, float]:
    dx = x - ego_x
    dy = y - ego_y
    forward = dx * math.cos(ego_yaw) + dy * math.sin(ego_yaw)
    left = dx * math.sin(ego_yaw) - dy * math.cos(ego_yaw)
    return EGO_LOCAL_X_M + forward, left


def _local_vector(x: float, y: float, ego_yaw: float) -> tuple[float, float]:
    forward = x * math.cos(ego_yaw) + y * math.sin(ego_yaw)
    left = x * math.sin(ego_yaw) - y * math.cos(ego_yaw)
    return forward, left


def _classify_actor(type_id: str) -> str | None:
    normalized = type_id.lower()
    if normalized.startswith("walker.pedestrian"):
        return "pedestrian"
    if normalized.startswith("vehicle.") and any(
        term in normalized
        for term in (
            "motorcycle",
            "bicycle",
            "harley",
            "kawasaki",
            "vespa",
            "yamaha",
        )
    ):
        return "two_wheeler"
    if normalized.startswith("vehicle."):
        return "car"
    return None


def _actor_box_radius(actor: Any) -> float:
    extent = actor.bounding_box.extent
    radius = math.hypot(float(extent.x), float(extent.y))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"Actor {actor.id} has invalid bounding-box extents.")
    return radius


def build_request(
    ego: Any,
    actors: Iterable[Any],
    *,
    timestamp_s: float,
    road_half_width_m: float,
    route_speed_mps: float,
    collision: bool = False,
    reset: bool = False,
) -> dict[str, Any]:
    """Build one JSON-ready PathWeaver request from CARLA ground-truth actors."""
    timestamp = _finite(timestamp_s, "timestamp_s")
    half_width = _finite(road_half_width_m, "road_half_width_m")
    route_speed = _finite(route_speed_mps, "route_speed_mps")
    if timestamp < 0.0:
        raise ValueError("timestamp_s must be nonnegative.")
    if half_width <= 1.5:
        raise ValueError("road_half_width_m must exceed 1.5 m.")
    if route_speed <= 0.0:
        raise ValueError("route_speed_mps must be positive.")

    ego_transform = ego.get_transform()
    ego_location = ego_transform.location
    ego_yaw = math.radians(float(ego_transform.rotation.yaw))
    ego_ground_z = float(ego_location.z)
    ego_velocity = ego.get_velocity()
    ego_acceleration = ego.get_acceleration()
    ego_control = ego.get_control()

    ego_speed = math.sqrt(
        float(ego_velocity.x) ** 2
        + float(ego_velocity.y) ** 2
        + float(ego_velocity.z) ** 2
    )
    acceleration_forward, _ = _local_vector(
        float(ego_acceleration.x),
        float(ego_acceleration.y),
        ego_yaw,
    )
    ego_state = {
        "positionWorldM": [EGO_LOCAL_X_M, 0.0],
        "headingRad": 0.0,
        "speedMps": _finite(ego_speed, "ego speed"),
        "accelerationMps2": _finite(acceleration_forward, "ego acceleration"),
        "steeringRad": _finite(float(ego_control.steer), "ego steering")
        * math.radians(30.0),
        "timestampS": timestamp,
    }

    dynamic_agents: list[dict[str, Any]] = []
    static_obstacles: list[tuple[float, float, float, float]] = []
    for actor in actors:
        if actor.id == ego.id or not getattr(actor, "is_alive", True):
            continue

        transform = actor.get_transform()
        location = transform.location
        bounding_box_location = getattr(actor.bounding_box, "location", None)
        if bounding_box_location is not None and hasattr(transform, "transform"):
            location = transform.transform(bounding_box_location)
        if abs(float(location.z) - ego_ground_z) > 3.5:
            continue
        local_position = _local_position(
            float(location.x),
            float(location.y),
            float(ego_location.x),
            float(ego_location.y),
            ego_yaw,
        )
        forward_distance = local_position[0] - EGO_LOCAL_X_M
        lateral_distance = local_position[1]
        distance = math.hypot(forward_distance, lateral_distance)
        type_id = str(actor.type_id).lower()
        actor_radius = _actor_box_radius(actor)

        actor_class = _classify_actor(type_id)
        if actor_class is not None and distance <= MAX_AGENT_DISTANCE_M:
            velocity = actor.get_velocity()
            local_velocity = _local_vector(
                float(velocity.x), float(velocity.y), ego_yaw
            )
            relative_heading = -math.radians(
                float(transform.rotation.yaw) - float(ego_transform.rotation.yaw)
            )
            relative_heading = (relative_heading + math.pi) % (2.0 * math.pi) - math.pi
            dynamic_agents.append(
                {
                    "id": int(actor.id),
                    "class": actor_class,
                    "positionWorldM": list(local_position),
                    "velocityWorldMps": list(local_velocity),
                    "headingRad": relative_heading,
                    "collisionRadiusM": actor_radius,
                    "positionCovarianceWorldM2": [
                        [POSITION_VARIANCE_M2, 0.0],
                        [0.0, POSITION_VARIANCE_M2],
                    ],
                    "timestampS": timestamp,
                }
            )
        elif (
            type_id.startswith("static.")
            and distance <= MAX_AGENT_DISTANCE_M
            and forward_distance >= -5.0
            and abs(lateral_distance) <= half_width + 4.0
        ):
            static_obstacles.append(
                (distance, local_position[0], local_position[1], actor_radius)
            )

    static_obstacles.sort(key=lambda obstacle: obstacle[0])
    if static_obstacles:
        _, obstacle_x, obstacle_y, obstacle_radius = static_obstacles[0]
    else:
        obstacle_x, obstacle_y, obstacle_radius = 10_000.0, 10_000.0, 0.1

    return {
        "timestampS": timestamp,
        "ego": ego_state,
        "agents": dynamic_agents,
        "roadHalfWidthM": half_width,
        "roadLengthM": max(80.0, EGO_LOCAL_X_M + route_speed * 3.0 + 40.0),
        "routeSpeedMps": route_speed,
        "collision": bool(collision),
        "reset": bool(reset),
        "staticObstacle": {
            "positionWorldM": [obstacle_x, obstacle_y],
            "radiusM": obstacle_radius,
        },
    }


def pathweaver_to_carla_world(
    local_position: tuple[float, float] | list[float],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw_rad: float,
) -> tuple[float, float]:
    """Map a PathWeaver point back to CARLA XY, accounting for opposite Y axes."""
    x_local = float(local_position[0]) - EGO_LOCAL_X_M
    y_local = float(local_position[1])
    x_world = (
        float(ego_x)
        + x_local * math.cos(ego_yaw_rad)
        + y_local * math.sin(ego_yaw_rad)
    )
    y_world = (
        float(ego_y)
        + x_local * math.sin(ego_yaw_rad)
        - y_local * math.cos(ego_yaw_rad)
    )
    if not math.isfinite(x_world) or not math.isfinite(y_world):
        raise ValueError("PathWeaver returned a non-finite target position.")
    return x_world, y_world

"""
unstructured_autonomous.py
CARLA 0.9.15

GOAL
----
Keep Town10HD buildings/trees/terrain and create an unstructured-driving
scenario around its roads.

IMPORTANT LIMITATION
--------------------
Town10HD's lane lines are part of the road assets/materials, so CARLA's
normal Python API cannot erase them. This script therefore:
    * keeps the Town10HD environment
    * is intended to run on a generated OpenDRIVE road
    * uses actor-aware risk assessment and obstacle avoidance
    * spawns moving Traffic Manager vehicles on the selected route
    * disables stationary route obstacles, props, parked vehicles and walkers
    * uses OSM direction and searches CARLA's Driving-waypoint graph
    * does not claim camera/YOLO perception; hazards are read from CARLA actors
    * provides a conservative actor-aware speed/steering controller

For truly ZERO visible lane markings, replace the Town10HD road material
in Unreal Engine. The controller does not use lane markings or GPU-based
perception. OSM coordinates provide a desired compass direction only; CARLA
start, destination, and route are selected from its driving waypoints.

This is a CARLA simulation demo, not a real-vehicle controller. CARLA itself
still needs a supported graphics setup; this script does not initialize CUDA,
neural networks, or camera sensors.
"""

import heapq
import itertools
import carla
import random
import math
import time
import threading
import json
import os
from autonomous_driving.integrations.pathweaver import (
    PathWeaverBridge,
    PathWeaverBridgeError,
)
from autonomous_driving.integrations.pathweaver.carla_adapter import (
    build_request as build_pathweaver_request,
    pathweaver_to_carla_world,
)

# ============================================================
# CONFIG
# ============================================================

HOST = os.environ.get("CARLA_HOST", "127.0.0.1")
PORT = int(os.environ.get("CARLA_PORT", "2000"))

MAP_NAME = os.environ.get("CARLA_MAP_NAME", "ANY")

RUN_DRIVING = True

NUM_ROCKS = 10
NUM_DEBRIS = 10
NUM_CONES = 0
NUM_BARRIERS = 0
NUM_PARKED = 10
NUM_PARKED_PER_ROUTE_END = 6
PARKED_END_ZONE_METERS = 180.0
NUM_POTHOLE_MARKERS = 0  # OpenDRIVE generator creates physical full-width dips; tire props are disabled
NUM_ROUTE_OBSTACLES = 4
NUM_MOVING_SCOOTERS = 3
NUM_MOVING_VANS = 3
NUM_FALLBACK_MOVING_CARS = 3
NUM_PEDESTRIANS = 5
SCENARIO_HAZARD_LENGTH_METERS = 2_000.0
TRAFFIC_MANAGER_PORT = 8000

CRUISE_SPEED = 15.0       # km/h; reduced for the rough-road demonstration
STOP_DISTANCE = 3.0       # metres of clearance to an obstacle
MIN_STANDOFF_DISTANCE = 2.0
SAFE_DECELERATION = 3.0   # m/s^2 used for distance-based speed planning
REACTION_TIME = 0.5       # seconds
AVOID_DETECT_DISTANCE = 30.0
AVOIDANCE_EMERGENCY_BYPASS_DISTANCE = 5.0
AVOID_RETRY_SECONDS = 1.0
LANE_CHANGE_DISTANCE = 12.0
BYPASS_MIN_CLEARANCE = 0.75
PEDESTRIAN_LOOKAHEAD = 35.0
PEDESTRIAN_SIDE_MARGIN = 1.0
PEDESTRIAN_PREDICTION_HORIZON_SECONDS = 8.0
PEDESTRIAN_PREDICTION_STEP_SECONDS = 0.25
PEDESTRIAN_CROSSING_TRIGGER_M = 35.0
PEDESTRIAN_STOP_BUFFER_M = 18.0
PARKING_SHOULDER_WIDTH_M = 3.5
ROAD_LATERAL_SCAN_M = 24.0
ROAD_LATERAL_SCAN_STEP_M = 0.5
AVOIDANCE_SPEED = 6.0     # km/h
AVOIDANCE_TIMEOUT = 15.0  # seconds without measurable bypass progress
COLLISION_BRAKE_HOLD_SECONDS = 0.5
COLLISION_REPEAT_DISTANCE_METERS = 2.0
MAX_OSM_BODY_BYTES = 1_000_000
MAX_OSM_ROUTE_POINTS = 20_000
ROUTE_STEP_METERS = 4.0
ROUTE_SEARCH_MAX_EXPANSIONS = 50_000
DESTINATION_REACHED_METERS = 4.0
MIN_DIRECTIONAL_ROUTE_METERS = 40.0
TARGET_DIRECTIONAL_ROUTE_METERS = 160.0
MAX_DIRECTIONAL_ROUTE_METERS = 350.0
TARGET_DRIVE_ROUTE_METERS = 2_000.0
DETOUR_SEARCH_MAX_EXPANSIONS = 12_000
DETOUR_MAX_DISTANCE_METERS = 75.0
ROUTE_RECOVERY_MAX_DISTANCE_METERS = 200.0
ROUTE_RECOVERY_MAX_EXPANSIONS = 8_000
NO_PROGRESS_TIMEOUT_SECONDS = 120.0
MAX_DIRECTIONAL_START_CANDIDATES = 16
# PathWeaver is disabled for this CARLA demo because its bridge currently
# returns infeasible plans and raises PathWeaverBridgeError. The existing
# CARLA Python controller remains responsible for steering and speed control.
PATHWEAVER_ENABLED = False
PATHWEAVER_REPLAN_INTERVAL_SECONDS = 0.15
PATHWEAVER_TRIGGER_DISTANCE_METERS = 40.0

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ============================================================
# GLOBALS
# ============================================================

client = None
world = None
ego = None
traffic_manager = None

environment_actors = []
sensor_actors = []
moving_traffic_actors = []

destination_location = None
road_pedestrian_states = []
ego_spawn_route_index = 0
static_hazard_map_name = None
static_hazard_cells = {}
static_hazards_by_id = {}
pathweaver_bridge = None
pathweaver_original_world_settings = None
STATIC_HAZARD_CELL_SIZE = 40.0
STATIC_HAZARD_LABELS = (
    "Poles",
    "TrafficLight",
    "TrafficSigns",
    "Fences",
    "Other",
    "Walls",
    "Buildings",
    "Vegetation",
    "GuardRail",
)


# ============================================================
# OSM DESTINATION RECEIVER
# ============================================================

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OSM_HOST="127.0.0.1"
OSM_PORT=5001
_osm_destination=None
_osm_lock=threading.Lock()


def _coordinate(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a numeric latitude/longitude.")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric latitude/longitude.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    limit = 90.0 if "lat" in name else 180.0
    if not -limit <= result <= limit:
        raise ValueError(f"{name} is outside valid geographic bounds.")
    return result


def validate_osm_destination(data):
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")

    required = (
        "start_lat",
        "start_lon",
        "destination_lat",
        "destination_lon",
    )
    coordinates = {
        key: _coordinate(data.get(key), key)
        for key in required
    }

    route = data.get("route")
    if not isinstance(route, list) or not 2 <= len(route) <= MAX_OSM_ROUTE_POINTS:
        raise ValueError(
            f"route must contain between 2 and {MAX_OSM_ROUTE_POINTS} points."
        )

    normalized_route = []
    for index, point in enumerate(route):
        if not isinstance(point, dict):
            raise ValueError(f"route point {index} must be an object.")
        normalized_route.append({
            "lat": _coordinate(point.get("lat"), f"route[{index}].lat"),
            "lon": _coordinate(point.get("lon"), f"route[{index}].lon"),
        })

    normalized_route[0] = {
        "lat": coordinates["start_lat"],
        "lon": coordinates["start_lon"],
    }
    normalized_route[-1] = {
        "lat": coordinates["destination_lat"],
        "lon": coordinates["destination_lon"],
    }
    return {
        **coordinates,
        "route": normalized_route,
        "display_name": str(data.get("display_name", ""))[:256],
    }


class OSMDestinationHandler(BaseHTTPRequestHandler):
    def _send_json(self,status,payload):
        body=json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type","application/json")
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.send_header("Content-Length",str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def do_OPTIONS(self): self._send_json(200,{"status":"ok"})
    def do_GET(self):
        with _osm_lock: value=dict(_osm_destination) if _osm_destination else None
        self._send_json(200,{"status":"ok","destination":value})
    def do_POST(self):
        global _osm_destination
        if self.path!="/destination":
            self._send_json(404,{"status":"not_found"}); return
        try:
            n=int(self.headers.get("Content-Length","0"))
            if n <= 0:
                raise ValueError("Request body is empty.")
            if n > MAX_OSM_BODY_BYTES:
                self._send_json(413,{"status":"error","message":"Request body is too large."})
                return
            data=json.loads(self.rfile.read(n).decode("utf-8"))
            destination=validate_osm_destination(data)
            with _osm_lock:
                _osm_destination=destination
            print("\nOSM DESTINATION RECEIVED:",_osm_destination)
            self._send_json(200,{"status":"success"})
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self._send_json(400,{"status":"error","message":str(exc)})
    def log_message(self,*_): return

def start_osm_server():
    server=ThreadingHTTPServer((OSM_HOST,OSM_PORT),OSMDestinationHandler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f"OSM -> CARLA server running at http://{OSM_HOST}:{OSM_PORT}")
    return server

def wait_for_osm_destination(timeout=300.0):
    print("Waiting for destination from the OSM webpage...")
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        with _osm_lock:
            if _osm_destination is not None: return dict(_osm_destination)
        time.sleep(0.2)
    raise TimeoutError(f"No OSM destination received within {timeout:.0f} seconds.")



# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)


def distance(a, b):
    return math.sqrt(
        (a.x-b.x)**2 +
        (a.y-b.y)**2 +
        (a.z-b.z)**2
    )


def right_vector(transform):
    f = transform.get_forward_vector()
    return carla.Vector3D(-f.y, f.x, 0.0)


def is_physical_right_lane_change(source_waypoint, target_waypoint):
    lateral_offset = (
        target_waypoint.transform.location
        - source_waypoint.transform.location
    )
    right = right_vector(source_waypoint.transform)
    return lateral_offset.x * right.x + lateral_offset.y * right.y > 0.0


def forward_vector(transform):
    return transform.get_forward_vector()


def pathweaver_road_half_width_m():
    """Measure a conservative local road half-width from CARLA lane geometry."""
    override = os.environ.get("PATHWEAVER_ROAD_HALF_WIDTH_M")
    if override is not None:
        half_width = float(override)
        if not math.isfinite(half_width) or half_width <= 1.5:
            raise ValueError(
                "PATHWEAVER_ROAD_HALF_WIDTH_M must be finite and greater than 1.5."
            )
        return half_width

    map_obj = world.get_map()
    waypoint = map_obj.get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        raise RuntimeError(
            "Cannot measure the local CARLA road width. Set "
            "PATHWEAVER_ROAD_HALF_WIDTH_M explicitly for this map."
        )

    vehicle_transform = ego.get_transform()
    vehicle_location = vehicle_transform.location
    right = right_vector(vehicle_transform)
    forward = vehicle_transform.get_forward_vector()
    left_x, left_y = -right.x, -right.y
    left_edges = []
    right_edges = []

    def add_lane_edges(lane):
        lane_location = lane.transform.location
        offset = (
            (lane_location.x - vehicle_location.x) * left_x
            + (lane_location.y - vehicle_location.y) * left_y
        )
        half_lane_width = float(lane.lane_width) / 2.0
        left_edges.append(offset + half_lane_width)
        right_edges.append(offset - half_lane_width)

    add_lane_edges(waypoint)
    for method_name in ("get_left_lane", "get_right_lane"):
        lane = waypoint
        for _ in range(8):
            adjacent = getattr(lane, method_name)()
            adjacent_forward = (
                adjacent.transform.get_forward_vector()
                if adjacent is not None
                else None
            )
            if (
                adjacent is None
                or adjacent.road_id != waypoint.road_id
                or adjacent.section_id != waypoint.section_id
                or adjacent.lane_type != carla.LaneType.Driving
                or (
                    adjacent_forward.x * forward.x
                    + adjacent_forward.y * forward.y
                )
                < 0.5
            ):
                break
            add_lane_edges(adjacent)
            lane = adjacent

    half_width = min(max(left_edges), -min(right_edges))
    if not math.isfinite(half_width) or half_width <= 1.5:
        raise RuntimeError(
            f"Measured road half-width {half_width:.2f} m is too narrow or invalid."
        )
    return half_width


def pathweaver_route_is_straight(route, route_index, ego_transform):
    """Check the local route against PathWeaver's straight-road assumption."""
    if not route:
        return False
    forward = ego_transform.get_forward_vector()
    previous_forward_distance = 0.0
    checked = 0
    for waypoint in route[max(0, route_index):]:
        location = waypoint.transform.location
        dx = location.x - ego_transform.location.x
        dy = location.y - ego_transform.location.y
        forward_distance = dx * forward.x + dy * forward.y
        lateral_distance = dx * -forward.y + dy * forward.x
        if forward_distance < -2.0:
            continue
        if forward_distance > 30.0:
            break
        if (
            forward_distance + 1.0 < previous_forward_distance
            or abs(lateral_distance) > 0.75
        ):
            return False
        waypoint_forward = waypoint.transform.get_forward_vector()
        direction_dot = (
            forward.x * waypoint_forward.x
            + forward.y * waypoint_forward.y
        )
        if direction_dot < math.cos(math.radians(10.0)):
            return False
        previous_forward_distance = forward_distance
        checked += 1
    return checked >= 3 and previous_forward_distance >= 12.0


def pathweaver_control(plan, anchor_transform, timestamp_s):
    """Track a PathWeaver target with the existing CARLA steering/speed helpers."""
    max_steering = math.radians(30.0)
    max_steering_change = math.radians(50.0) * 0.05 / max_steering
    previous_steering = float(ego.get_control().steer)
    if (
        not plan.feasible
        or plan.emergency
        or plan.behavior_state == "EMERGENCY_BRAKE"
    ):
        steering = clamp(
            0.0,
            previous_steering - max_steering_change,
            previous_steering + max_steering_change,
        )
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=steering)

    target_x, target_y = pathweaver_to_carla_world(
        plan.target_position_local_m,
        ego_x=anchor_transform.location.x,
        ego_y=anchor_transform.location.y,
        ego_yaw_rad=math.radians(anchor_transform.rotation.yaw),
    )
    target = carla.Location(
        x=target_x,
        y=target_y,
        z=anchor_transform.location.z,
    )
    steering = calculate_steering(target)
    steering = clamp(
        steering,
        max(-0.45, previous_steering - max_steering_change),
        min(0.45, previous_steering + max_steering_change),
    )
    target_speed_kmh = plan.speed_at(timestamp_s + 0.05) * 3.6
    throttle, brake = speed_control(speed_kmh(ego), target_speed_kmh)
    return carla.VehicleControl(
        throttle=throttle,
        brake=brake,
        steer=steering,
    )


def pathweaver_emergency_stop():
    ego.apply_control(
        carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
    )
    world.tick()


# ============================================================
# CONNECT
# ============================================================

def connect():
    global client, world, traffic_manager

    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    print("Connecting to CARLA...")

    world = client.get_world()
    traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    traffic_manager.set_global_distance_to_leading_vehicle(10.0)
    traffic_manager.set_synchronous_mode(False)

    print("CARLA:", client.get_server_version())
    print("MAP  :", world.get_map().name)

    if MAP_NAME.upper() != "ANY" and MAP_NAME not in world.get_map().name:
        raise RuntimeError(
            f"Expected CARLA map '{MAP_NAME}', got '{world.get_map().name}'. "
            "Set CARLA_MAP_NAME to the CARLA map you intend to use."
        )

    print("Connected to map:", world.get_map().name)


# ============================================================
# WORLD SETTINGS
# ============================================================

def configure_world():

    # Keep Town10HD environment.
    # Remove decals only; this does NOT remove baked road markings.
    try:
        world.unload_map_layer(carla.MapLayer.Decals)
        print("Decals layer unloaded.")
    except Exception as exc:
        print("Decals layer could not be unloaded:", exc)

    weather = carla.WeatherParameters(
        cloudiness=15.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=8.0,
        sun_altitude_angle=55.0,
        sun_azimuth_angle=120.0,
        fog_density=0.0
    )

    world.set_weather(weather)


# ============================================================
# FIND ROAD
# ============================================================

def osm_bearing(destination):
    """Calculate the OSM start-to-destination bearing without using map XY."""
    lat1 = math.radians(destination["start_lat"])
    lat2 = math.radians(destination["destination_lat"])
    delta_lon = math.radians(
        destination["destination_lon"] - destination["start_lon"]
    )
    east = math.sin(delta_lon) * math.cos(lat2)
    north = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    )
    if math.hypot(east, north) < 1e-12:
        raise ValueError("OSM start and destination must be different points.")
    return math.atan2(east, north)


def heading_error(heading, desired_heading):
    return abs(math.atan2(
        math.sin(heading - desired_heading),
        math.cos(heading - desired_heading),
    ))


def is_driving_waypoint(waypoint):
    if waypoint is None:
        return False
    try:
        lane_type = int(waypoint.lane_type)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(lane_type & int(carla.LaneType.Driving))


def is_driving_segment(map_obj, start, end, sample_spacing=0.75):
    """Keep the vehicle center and both body edges on CARLA Driving lanes."""
    segment_length = distance(start, end)
    sample_count = max(1, math.ceil(segment_length / sample_spacing))
    if segment_length > 1e-6:
        right_x = -(end.y - start.y) / segment_length
        right_y = (end.x - start.x) / segment_length
    else:
        right_x = 0.0
        right_y = 0.0
    half_width = (
        ego.bounding_box.extent.y + 0.15
        if ego is not None
        else 1.1
    )
    for index in range(sample_count + 1):
        fraction = index / sample_count
        point = carla.Location(
            x=start.x + (end.x - start.x) * fraction,
            y=start.y + (end.y - start.y) * fraction,
            z=start.z + (end.z - start.z) * fraction,
        )
        for lateral_offset in (-half_width, 0.0, half_width):
            footprint_point = carla.Location(
                x=point.x + right_x * lateral_offset,
                y=point.y + right_y * lateral_offset,
                z=point.z,
            )
            waypoint = map_obj.get_waypoint(
                footprint_point,
                project_to_road=False,
                lane_type=carla.LaneType.Driving,
            )
            if not is_driving_waypoint(waypoint):
                waypoint = map_obj.get_waypoint(
                    footprint_point,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if not is_driving_waypoint(waypoint) or distance(footprint_point, waypoint.transform.location) > (waypoint.lane_width / 2.0 + 0.2):
                    return False
    return True


def waypoint_key(waypoint):
    return (
        waypoint.road_id,
        waypoint.section_id,
        waypoint.lane_id,
        int(round(waypoint.s / (ROUTE_STEP_METERS / 2.0))),
    )


def choose_start(map_obj, osm_destination):
    candidates = driving_start_candidates(map_obj, osm_destination)
    if not candidates:
        raise RuntimeError("CARLA map has no valid LaneType.Driving spawn waypoint.")
    return candidates[0][1]


def driving_start_candidates(map_obj, osm_destination):
    spawns = map_obj.get_spawn_points()
    if not spawns:
        spawns = [
            waypoint.transform
            for waypoint in map_obj.generate_waypoints(30.0)
            if is_driving_waypoint(waypoint)
        ]
    if not spawns:
        raise RuntimeError("CARLA map has no vehicle spawn points.")

    desired_heading = osm_bearing(osm_destination)
    candidates = {}
    for transform in spawns:
        waypoint = map_obj.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if not is_driving_waypoint(waypoint):
            continue
        successors = [
            candidate
            for candidate in waypoint.next(ROUTE_STEP_METERS)
            if is_driving_waypoint(candidate)
            and is_driving_segment(
                map_obj,
                waypoint.transform.location,
                candidate.transform.location,
            )
        ]
        headings = successors or [waypoint]
        score = min(
            heading_error(
                math.atan2(
                    candidate.transform.get_forward_vector().y,
                    candidate.transform.get_forward_vector().x,
                ),
                desired_heading,
            )
            for candidate in headings
        )
        key = waypoint_key(waypoint)
        if score < candidates.get(key, (float("inf"), None))[0]:
            candidates[key] = (score, waypoint)

    return sorted(candidates.values(), key=lambda candidate: candidate[0])


def find_directional_destination(map_obj, start_waypoint, desired_heading):
    """Search reachable driving lanes and select a forward goal by OSM bearing."""
    start_key = waypoint_key(start_waypoint)
    frontier_order = itertools.count()
    frontier = [(0.0, next(frontier_order), 0.0, start_key)]
    waypoints = {start_key: start_waypoint}
    cost_so_far = {start_key: 0.0}
    best_goal = None
    best_score = float("inf")
    expanded = 0
    start_location = start_waypoint.transform.location

    while frontier and expanded < ROUTE_SEARCH_MAX_EXPANSIONS:
        _, _, queued_cost, current_key = heapq.heappop(frontier)
        if queued_cost != cost_so_far.get(current_key):
            continue
        current = waypoints[current_key]
        current_location = current.transform.location
        expanded += 1

        dx = current_location.x - start_location.x
        dy = current_location.y - start_location.y
        displacement = math.hypot(dx, dy)
        if (
            MIN_DIRECTIONAL_ROUTE_METERS <= queued_cost
            <= MAX_DIRECTIONAL_ROUTE_METERS
            and displacement > 1.0
        ):
            displacement_heading = math.atan2(dy, dx)
            displacement_error = heading_error(
                displacement_heading,
                desired_heading,
            )
            if displacement_error <= math.pi / 2.0:
                forward = current.transform.get_forward_vector()
                endpoint_error = heading_error(
                    math.atan2(forward.y, forward.x),
                    desired_heading,
                )
                distance_penalty = (
                    abs(queued_cost - TARGET_DIRECTIONAL_ROUTE_METERS)
                    / TARGET_DIRECTIONAL_ROUTE_METERS
                )
                score = (
                    displacement_error
                    + 0.25 * endpoint_error
                    + 0.10 * distance_penalty
                )
                if score < best_score:
                    best_goal = current
                    best_score = score

        if queued_cost >= MAX_DIRECTIONAL_ROUTE_METERS:
            continue

        for next_waypoint in current.next(ROUTE_STEP_METERS):
            if not is_driving_waypoint(next_waypoint):
                continue
            if not is_driving_segment(
                map_obj,
                current_location,
                next_waypoint.transform.location,
            ):
                continue
            next_key = waypoint_key(next_waypoint)
            step_cost = distance(
                current_location,
                next_waypoint.transform.location,
            )
            if step_cost <= 0.1:
                continue
            new_cost = queued_cost + step_cost
            if new_cost >= cost_so_far.get(next_key, float("inf")):
                continue
            cost_so_far[next_key] = new_cost
            waypoints[next_key] = next_waypoint
            heapq.heappush(
                frontier,
                (new_cost, next(frontier_order), new_cost, next_key),
            )

    if best_goal is None:
        raise RuntimeError(
            "Could not find a reachable CARLA LaneType.Driving destination "
            "in the OSM-requested direction."
        )
    return best_goal


def choose_destination(map_obj, start_waypoint, osm_destination):
    return find_directional_destination(
        map_obj,
        start_waypoint,
        osm_bearing(osm_destination),
    )


def build_map_route(map_obj, start_waypoint, destination_waypoint):
    if not is_driving_waypoint(start_waypoint):
        raise RuntimeError("Route start is not a CARLA LaneType.Driving waypoint.")
    if not is_driving_waypoint(destination_waypoint):
        raise RuntimeError(
            "Route destination is not a CARLA LaneType.Driving waypoint."
        )

    start_key = waypoint_key(start_waypoint)
    frontier_order = itertools.count()
    frontier = [(0.0, next(frontier_order), 0.0, start_key)]
    waypoints = {start_key: start_waypoint}
    came_from = {}
    cost_so_far = {start_key: 0.0}
    destination_key = waypoint_key(destination_waypoint)
    destination = destination_waypoint.transform.location
    expanded = 0
    reached_key = None

    while frontier and expanded < ROUTE_SEARCH_MAX_EXPANSIONS:
        _, _, queued_cost, current_key = heapq.heappop(frontier)
        if queued_cost != cost_so_far.get(current_key):
            continue
        current = waypoints[current_key]
        expanded += 1

        if current_key == destination_key:
            reached_key = current_key
            break

        for next_waypoint in current.next(ROUTE_STEP_METERS):
            if not is_driving_waypoint(next_waypoint):
                continue
            if not is_driving_segment(
                map_obj,
                current.transform.location,
                next_waypoint.transform.location,
            ):
                continue
            next_key = waypoint_key(next_waypoint)
            step_cost = distance(
                current.transform.location,
                next_waypoint.transform.location,
            )
            new_cost = cost_so_far[current_key] + step_cost
            if new_cost >= cost_so_far.get(next_key, float("inf")):
                continue

            cost_so_far[next_key] = new_cost
            came_from[next_key] = current_key
            waypoints[next_key] = next_waypoint
            remaining = distance(next_waypoint.transform.location, destination)
            heapq.heappush(
                frontier,
                (
                    new_cost + remaining,
                    next(frontier_order),
                    new_cost,
                    next_key,
                ),
            )

    if reached_key is None:
        raise RuntimeError(
            "No connected CARLA LaneType.Driving route was found to the "
            f"selected directional goal after {expanded} waypoint expansions."
        )

    route_keys = [reached_key]
    while route_keys[-1] != start_key:
        parent = came_from.get(route_keys[-1])
        if parent is None:
            raise RuntimeError("CARLA route reconstruction failed.")
        route_keys.append(parent)
    route_keys.reverse()
    route = [waypoints[key] for key in route_keys]
    if len(route) < 2:
        raise RuntimeError("Selected CARLA destination is too close to the start.")
    for start, end in zip(route, route[1:]):
        if not is_driving_waypoint(start) or not is_driving_waypoint(end):
            raise RuntimeError("A* route contains a non-driving waypoint.")
        if not is_driving_segment(
            map_obj,
            start.transform.location,
            end.transform.location,
        ):
            raise RuntimeError("A* route contains a non-drivable segment.")
    return route


def extend_route_with_driving_laps(map_obj, route, target_distance):
    """Repeat a verified Driving-lane loop until the requested route length."""
    if len(route) < 2:
        return route

    try:
        return_path = build_map_route(map_obj, route[-1], route[0])
    except RuntimeError as exc:
        print(
            "Could not close the selected route into a Driving-lane loop; "
            f"searching for the longest connected continuation ({exc})"
        )
        continuation = find_verified_route_continuation(map_obj, route[-1])
        if continuation is None:
            return route
        extended = route + continuation[1:]
        extended_distance = sum(
            distance(first.transform.location, second.transform.location)
            for first, second in zip(extended, extended[1:])
        )
        while extended_distance < target_distance:
            previous = extended[-1]
            for waypoint in continuation:
                if not is_driving_segment(
                    map_obj,
                    previous.transform.location,
                    waypoint.transform.location,
                ):
                    raise RuntimeError(
                        "Verified Driving-lane continuation became "
                        "disconnected while extending the route."
                    )
                extended_distance += distance(
                    previous.transform.location,
                    waypoint.transform.location,
                )
                extended.append(waypoint)
                previous = waypoint
                if extended_distance >= target_distance:
                    break
        return extended

    loop = route + return_path[1:]
    loop_distance = sum(
        distance(first.transform.location, second.transform.location)
        for first, second in zip(loop, loop[1:])
    )
    if (
        not loop
        or waypoint_key(loop[0]) != waypoint_key(loop[-1])
        or loop_distance <= 0.0
    ):
        print(
            "Could not verify a closed Driving-lane loop; "
            "keeping the selected OSM-directional route."
        )
        return route

    extended = list(loop)
    extended_distance = loop_distance
    while extended_distance < target_distance:
        previous = extended[-1]
        for waypoint in loop[1:]:
            if (
                not is_driving_waypoint(previous)
                or not is_driving_waypoint(waypoint)
                or not is_driving_segment(
                    map_obj,
                    previous.transform.location,
                    waypoint.transform.location,
                )
            ):
                raise RuntimeError(
                    "Driving-lane loop became disconnected while extending "
                    "the route."
                )
            extended_distance += distance(
                previous.transform.location,
                waypoint.transform.location,
            )
            extended.append(waypoint)
            previous = waypoint
            if extended_distance >= target_distance:
                break

    return extended


def find_farthest_driving_extension(map_obj, start_waypoint):
    """Return the longest shortest-path continuation reachable on Driving lanes."""
    start_key = waypoint_key(start_waypoint)
    frontier_order = itertools.count()
    frontier = [(0.0, next(frontier_order), start_key)]
    waypoints = {start_key: start_waypoint}
    parents = {}
    cost_so_far = {start_key: 0.0}
    farthest_key = start_key
    expanded = 0

    while frontier and expanded < ROUTE_SEARCH_MAX_EXPANSIONS:
        queued_cost, _, current_key = heapq.heappop(frontier)
        if queued_cost != cost_so_far.get(current_key):
            continue
        current = waypoints[current_key]
        expanded += 1
        if queued_cost > cost_so_far[farthest_key]:
            farthest_key = current_key

        for next_waypoint in current.next(ROUTE_STEP_METERS):
            if not is_driving_waypoint(next_waypoint):
                continue
            if not is_driving_segment(
                map_obj,
                current.transform.location,
                next_waypoint.transform.location,
            ):
                continue
            next_key = waypoint_key(next_waypoint)
            next_cost = queued_cost + distance(
                current.transform.location,
                next_waypoint.transform.location,
            )
            if next_cost >= cost_so_far.get(next_key, float("inf")):
                continue
            cost_so_far[next_key] = next_cost
            waypoints[next_key] = next_waypoint
            parents[next_key] = current_key
            heapq.heappush(
                frontier,
                (next_cost, next(frontier_order), next_key),
            )

    if farthest_key == start_key:
        return [start_waypoint]

    path_keys = [farthest_key]
    while path_keys[-1] != start_key:
        parent_key = parents.get(path_keys[-1])
        if parent_key is None:
            raise RuntimeError(
                "Could not reconstruct the farthest Driving-lane continuation."
            )
        path_keys.append(parent_key)
    path_keys.reverse()
    path = [waypoints[key] for key in path_keys]
    for first, second in zip(path, path[1:]):
        if (
            not is_driving_waypoint(first)
            or not is_driving_waypoint(second)
            or not is_driving_segment(
                map_obj,
                first.transform.location,
                second.transform.location,
            )
        ):
            raise RuntimeError(
                "Farthest Driving-lane continuation contains an invalid segment."
            )
    return path


def find_verified_route_continuation(map_obj, start_waypoint):
    """Find a forward Driving-lane cycle that reconnects to its start."""
    continuation = find_farthest_driving_extension(map_obj, start_waypoint)
    if (
        len(continuation) < 2
        or distance(
            continuation[0].transform.location,
            start_waypoint.transform.location,
        ) > 1.0
    ):
        return None

    start = continuation[0].transform.location
    end = continuation[-1].transform.location
    return_waypoints = [
        waypoint
        for waypoint in continuation[-1].next(ROUTE_STEP_METERS)
        if is_driving_waypoint(waypoint)
        and distance(waypoint.transform.location, start)
        <= ROUTE_STEP_METERS * 1.25
        and is_driving_segment(map_obj, end, waypoint.transform.location)
    ]
    return continuation if return_waypoints else None


def plan_osm_directional_drive(map_obj, osm_destination):
    """Choose a compass-aligned spawn that supports the longest safe route."""
    candidates = driving_start_candidates(map_obj, osm_destination)
    if not candidates:
        raise RuntimeError("CARLA map has no valid LaneType.Driving spawn waypoint.")

    best_heading = candidates[0][0]
    heading_limit = best_heading + math.radians(5.0)
    best_plan = None
    best_distance = -1.0
    failures = []
    evaluated = 0

    for heading_error_value, start_waypoint in candidates:
        if (
            heading_error_value > heading_limit
            or evaluated >= MAX_DIRECTIONAL_START_CANDIDATES
        ):
            break
        evaluated += 1
        try:
            destination_waypoint = choose_destination(
                map_obj,
                start_waypoint,
                osm_destination,
            )
            route = build_map_route(
                map_obj,
                start_waypoint,
                destination_waypoint,
            )
            route = extend_route_with_driving_laps(
                map_obj,
                route,
                TARGET_DRIVE_ROUTE_METERS,
            )
        except RuntimeError as exc:
            failures.append(str(exc))
            continue

        route_distance = sum(
            distance(first.transform.location, second.transform.location)
            for first, second in zip(route, route[1:])
        )
        if route_distance > best_distance:
            best_plan = (start_waypoint, destination_waypoint, route)
            best_distance = route_distance
        if route_distance >= TARGET_DRIVE_ROUTE_METERS:
            break

    if best_plan is None:
        failure_detail = failures[-1] if failures else "no directional candidate"
        raise RuntimeError(
            "Could not build a connected CARLA Driving-lane route for the "
            f"OSM direction: {failure_detail}"
        )
    return best_plan


def select_safe_route_target(map_obj, current_location, route, route_index):
    """Choose the nearest valid route waypoint to avoid cutting corners."""
    if not route or route_index < 0 or route_index >= len(route):
        return None

    ego_forward = ego.get_transform().get_forward_vector()
    last_index = min(route_index + 3, len(route) - 1)
    for index in range(route_index, last_index + 1):
        waypoint = route[index]
        offset = waypoint.transform.location - current_location
        forward_distance = (
            offset.x * ego_forward.x
            + offset.y * ego_forward.y
        )
        if (
            forward_distance > 0.5
            and is_driving_waypoint(waypoint)
            and is_driving_segment(
                map_obj,
                current_location,
                waypoint.transform.location,
            )
        ):
            return waypoint
    return None


route_recovery_active = False
route_recovery_points = []
route_recovery_point_index = 0
route_recovery_merge_index = None


def start_route_recovery(map_obj, current_location, route, route_index):
    """Find a connected Driving-lane path from the car to a future route point."""
    global route_recovery_active
    global route_recovery_points
    global route_recovery_point_index
    global route_recovery_merge_index

    start_waypoint = map_obj.get_waypoint(
        current_location,
        project_to_road=False,
        lane_type=carla.LaneType.Driving,
    )
    if not is_driving_waypoint(start_waypoint):
        projected_waypoint = map_obj.get_waypoint(
            current_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if (
            not is_driving_waypoint(projected_waypoint)
            or distance(
                current_location,
                projected_waypoint.transform.location,
            ) > min(2.0, projected_waypoint.lane_width * 0.45)
            or not is_driving_segment(
                map_obj,
                current_location,
                projected_waypoint.transform.location,
            )
        ):
            return False
        start_waypoint = projected_waypoint

    start_location = start_waypoint.transform.location
    first_goal_index = min(len(route), route_index + 2)
    route_goals = {}
    route_distance = 0.0
    previous_location = current_location
    for index in range(route_index, len(route)):
        waypoint = route[index]
        location = waypoint.transform.location
        route_distance += distance(previous_location, location)
        previous_location = location
        if (
            index >= first_goal_index
            and route_distance >= 6.0
            and is_driving_waypoint(waypoint)
        ):
            route_goals.setdefault(waypoint_key(waypoint), index)
    if not route_goals:
        return start_forward_route_recovery(
            map_obj,
            current_location,
            start_waypoint,
            route,
            route_index,
        )

    start_key = waypoint_key(start_waypoint)
    frontier_order = itertools.count()
    frontier = [(0.0, next(frontier_order), start_key)]
    cost_so_far = {start_key: 0.0}
    waypoint_by_key = {start_key: start_waypoint}
    parent = {}
    reached_key = None
    expanded = 0

    while frontier and expanded < ROUTE_RECOVERY_MAX_EXPANSIONS:
        current_cost, _, current_key = heapq.heappop(frontier)
        if current_cost != cost_so_far.get(current_key):
            continue
        if current_key in route_goals:
            reached_key = current_key
            break
        expanded += 1

        current = waypoint_by_key[current_key]
        current_location_on_lane = current.transform.location
        current_forward = current.transform.get_forward_vector()
        neighbors = [
            waypoint
            for waypoint in current.next(ROUTE_STEP_METERS)
            if is_driving_waypoint(waypoint)
        ]
        for lane_method in ("get_left_lane", "get_right_lane"):
            neighbor = getattr(current, lane_method)()
            if is_driving_waypoint(neighbor):
                neighbors.append(neighbor)

        seen = set()
        for neighbor in neighbors:
            neighbor_key = waypoint_key(neighbor)
            if neighbor_key == current_key or neighbor_key in seen:
                continue
            seen.add(neighbor_key)
            neighbor_forward = neighbor.transform.get_forward_vector()
            if (
                current_forward.x * neighbor_forward.x
                + current_forward.y * neighbor_forward.y
            ) < 0.5:
                continue
            neighbor_location = neighbor.transform.location
            if (
                distance(start_location, neighbor_location)
                > ROUTE_RECOVERY_MAX_DISTANCE_METERS
                or not is_driving_segment(
                    map_obj,
                    current_location_on_lane,
                    neighbor_location,
                )
            ):
                continue
            new_cost = current_cost + distance(
                current_location_on_lane,
                neighbor_location,
            )
            if (
                new_cost > ROUTE_RECOVERY_MAX_DISTANCE_METERS
                or new_cost >= cost_so_far.get(neighbor_key, float("inf"))
            ):
                continue
            cost_so_far[neighbor_key] = new_cost
            waypoint_by_key[neighbor_key] = neighbor
            parent[neighbor_key] = current_key
            heapq.heappush(
                frontier,
                (new_cost, next(frontier_order), neighbor_key),
            )

    if reached_key is None:
        return start_forward_route_recovery(
            map_obj,
            current_location,
            start_waypoint,
            route,
            route_index,
        )

    path_keys = [reached_key]
    while path_keys[-1] != start_key:
        parent_key = parent.get(path_keys[-1])
        if parent_key is None:
            return start_forward_route_recovery(
                map_obj,
                current_location,
                start_waypoint,
                route,
                route_index,
            )
        path_keys.append(parent_key)
    path_keys.reverse()
    path = [waypoint_by_key[key] for key in path_keys]
    if len(path) < 2:
        return start_forward_route_recovery(
            map_obj,
            current_location,
            start_waypoint,
            route,
            route_index,
        )

    if not is_driving_segment(
        map_obj,
        current_location,
        path[0].transform.location,
    ):
        return start_forward_route_recovery(
            map_obj,
            current_location,
            start_waypoint,
            route,
            route_index,
        )
    for first, second in zip(path, path[1:]):
        if (
            not is_driving_waypoint(first)
            or not is_driving_waypoint(second)
            or not is_driving_segment(
                map_obj,
                first.transform.location,
                second.transform.location,
            )
        ):
            return start_forward_route_recovery(
                map_obj,
                current_location,
                start_waypoint,
                route,
                route_index,
            )

    route_recovery_points = path
    route_recovery_point_index = 1
    route_recovery_merge_index = route_goals[reached_key]
    route_recovery_active = True
    print(
        "Route tracking recovery: following connected Driving waypoints "
        f"to original route index {route_recovery_merge_index}."
    )
    return True


def start_forward_route_recovery(
    map_obj,
    current_location,
    start_waypoint,
    route,
    route_index,
):
    """Follow connected forward Driving waypoints while searching for a merge."""
    global route_recovery_active
    global route_recovery_points
    global route_recovery_point_index
    global route_recovery_merge_index

    if (
        not is_driving_waypoint(start_waypoint)
        or not is_driving_segment(
            map_obj,
            current_location,
            start_waypoint.transform.location,
        )
    ):
        return False

    path = [start_waypoint]
    current = start_waypoint
    walked = distance(current_location, current.transform.location)
    visited = {waypoint_key(current)}
    merge_index = None

    while walked < 36.0:
        current_location_on_lane = current.transform.location
        next_waypoints = [
            waypoint
            for waypoint in current.next(ROUTE_STEP_METERS)
            if is_driving_waypoint(waypoint)
            and waypoint_key(waypoint) not in visited
            and is_driving_segment(
                map_obj,
                current_location_on_lane,
                waypoint.transform.location,
            )
        ]
        if not next_waypoints:
            break

        current_forward = current.transform.get_forward_vector()
        next_waypoint = max(
            next_waypoints,
            key=lambda waypoint: (
                current_forward.x
                * waypoint.transform.get_forward_vector().x
                + current_forward.y
                * waypoint.transform.get_forward_vector().y
            ),
        )
        walked += distance(
            current_location_on_lane,
            next_waypoint.transform.location,
        )
        path.append(next_waypoint)
        current = next_waypoint
        visited.add(waypoint_key(current))

        for candidate_index in range(
            min(route_index + 2, len(route)),
            len(route),
        ):
            candidate = route[candidate_index]
            if (
                distance(
                    current.transform.location,
                    candidate.transform.location,
                ) <= 1.5
                and is_driving_segment(
                    map_obj,
                    current.transform.location,
                    candidate.transform.location,
                )
            ):
                path.append(candidate)
                merge_index = candidate_index
                break
        if merge_index is not None:
            break

    if len(path) < 2:
        return False

    for first, second in zip(path, path[1:]):
        if (
            not is_driving_waypoint(first)
            or not is_driving_waypoint(second)
            or not is_driving_segment(
                map_obj,
                first.transform.location,
                second.transform.location,
            )
        ):
            return False

    route_recovery_points = path
    route_recovery_point_index = 1
    route_recovery_merge_index = merge_index
    route_recovery_active = True
    if merge_index is None:
        print(
            "Route recovery: following a short connected Driving-lane "
            "continuation and will search for a route merge again."
        )
    else:
        print(
            "Route recovery: following connected Driving waypoints to "
            f"original route index {merge_index}."
        )
    return True


# ============================================================
# EGO
# ============================================================

def spawn_ego(start_wp, route=None):
    global ego_spawn_route_index
    library = world.get_blueprint_library()
    blueprints = library.filter("vehicle.*")
    if not blueprints:
        raise RuntimeError("CARLA has no vehicle blueprints for the ego vehicle.")

    bp = None
    for name in ("vehicle.tesla.model3", "vehicle.tesla.modely"):
        matches = library.filter(name)
        if matches:
            bp = matches[0]
            break
    if bp is None:
        bp = random.choice(blueprints)
    candidates = [start_wp] if route is None else route[: min(len(route), 16)]
    vehicle = None
    selected_index = 0
    for route_index, waypoint in enumerate(candidates):
        waypoint_transform = waypoint.transform
        transform = carla.Transform(
            carla.Location(
                x=waypoint_transform.location.x,
                y=waypoint_transform.location.y,
                z=waypoint_transform.location.z + 0.5,
            ),
            waypoint_transform.rotation,
        )
        vehicle = world.try_spawn_actor(bp, transform)
        if vehicle is not None:
            selected_index = route_index
            break

    if vehicle is None:
        attempted = len(candidates)
        raise RuntimeError(
            f"Could not spawn ego vehicle at any of {attempted} initial "
            "route waypoints. Check for existing actors blocking the road "
            "near the selected CARLA start."
        )

    vehicle.set_autopilot(False)

    environment_actors.append(vehicle)
    ego_spawn_route_index = selected_index

    print(
        f"Ego vehicle: {vehicle.id} "
        f"(route waypoint {ego_spawn_route_index})"
    )

    return vehicle


# ============================================================
# GENERIC PROP SPAWNING
# ============================================================

def spawn_prop(bp, location, rotation=None):
    if ego is not None and distance(location, ego.get_location()) < 20.0:
        return None

    if rotation is None:
        rotation = carla.Rotation(
            yaw=random.uniform(0, 360)
        )

    actor = world.try_spawn_actor(
        bp,
        carla.Transform(location, rotation)
    )

    if actor:
        environment_actors.append(actor)

    return actor


def driving_lanes_across_road(waypoint):
    """Find this road section's driving lanes on both sides of a waypoint."""
    transform = waypoint.transform
    right = right_vector(transform)
    map_obj = world.get_map()
    lanes = []
    seen_lane_ids = set()
    scan_steps = int(ROAD_LATERAL_SCAN_M / ROAD_LATERAL_SCAN_STEP_M)

    for step in range(-scan_steps, scan_steps + 1):
        offset = step * ROAD_LATERAL_SCAN_STEP_M
        probe = transform.location + right * offset
        lane = map_obj.get_waypoint(
            probe,
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        if (
            not is_driving_waypoint(lane)
            or lane.road_id != waypoint.road_id
            or lane.section_id != waypoint.section_id
            or lane.lane_id in seen_lane_ids
        ):
            continue

        seen_lane_ids.add(lane.lane_id)
        lane_location = lane.transform.location
        lateral = (
            (lane_location.x - transform.location.x) * right.x
            + (lane_location.y - transform.location.y) * right.y
        )
        if abs(lateral) > 1e-6:
            lanes.append((lateral, lane))

    return lanes


def road_crossing_endpoints(waypoint):
    """Return shoulder-center endpoints from the road's left to right side."""
    transform = waypoint.transform
    right = right_vector(transform)
    side_lanes = {1.0: [], -1.0: []}
    for lateral, lane in driving_lanes_across_road(waypoint):
        if lateral > 0.0:
            side_lanes[1.0].append((lateral, lane))
        elif lateral < 0.0:
            side_lanes[-1.0].append((lateral, lane))

    if not side_lanes[1.0] or not side_lanes[-1.0]:
        return None

    right_edge = max(
        lateral + lane.lane_width * 0.5
        for lateral, lane in side_lanes[1.0]
    )
    left_edge = min(
        lateral - lane.lane_width * 0.5
        for lateral, lane in side_lanes[-1.0]
    )
    start_offset = right_edge + PARKING_SHOULDER_WIDTH_M * 0.5
    end_offset = left_edge - PARKING_SHOULDER_WIDTH_M * 0.5
    return (
        transform.location + right * end_offset,
        transform.location + right * start_offset,
    )


# ============================================================
# UNSTRUCTURED ENVIRONMENT
# ============================================================

def add_rocks(route):

    library = world.get_blueprint_library()
    props = library.filter("static.prop.*")

    if not props:
        return

    created = 0

    for _ in range(NUM_ROCKS):

        wp = random.choice(route)
        t = wp.transform

        right = right_vector(t)
        forward = forward_vector(t)

        lateral = random.choice([
            random.uniform(-8.0, -5.5),
            random.uniform(5.5, 8.0)
        ])

        location = (
            t.location
            + right*lateral
            + forward*random.uniform(-3, 3)
        )

        location.z += 0.15

        if spawn_prop(random.choice(props), location):
            created += 1

    print("Rocks/roadside props:", created)


def add_debris(route):

    library = world.get_blueprint_library()

    patterns = [
        "static.prop.barrel",
        "static.prop.box",
        "static.prop.pallet",
        "static.prop.*"
    ]

    props = []

    for pattern in patterns:
        props.extend(library.filter(pattern))

    if not props:
        return

    created = 0

    for _ in range(NUM_DEBRIS):

        wp = random.choice(route)
        t = wp.transform

        right = right_vector(t)
        forward = forward_vector(t)

        lateral = random.choice([
            random.uniform(-7.0, -5.0),
            random.uniform(5.0, 7.0)
        ])

        location = (
            t.location
            + right*lateral
            + forward*random.uniform(-2, 2)
        )

        location.z += 0.15

        if spawn_prop(random.choice(props), location):
            created += 1

    print("Road debris:", created)


def add_cones(route):

    library = world.get_blueprint_library()

    props = library.filter("static.prop.trafficcone*")

    if not props:
        print("Traffic cones unavailable.")
        return

    created = 0

    for _ in range(NUM_CONES):

        wp = random.choice(route)
        t = wp.transform

        right = right_vector(t)

        lateral = random.choice([
            random.uniform(-6, -3),
            random.uniform(3, 6)
        ])

        location = t.location + right*lateral
        location.z += 0.1

        if spawn_prop(random.choice(props), location):
            created += 1

    print("Traffic cones:", created)


def add_barriers(route):

    library = world.get_blueprint_library()

    props = []

    for pattern in [
        "static.prop.streetbarrier",
        "static.prop.warningconstruction",
        "static.prop.construction",
        "static.prop.*"
    ]:
        props.extend(library.filter(pattern))

    if not props:
        return

    created = 0

    for _ in range(NUM_BARRIERS):

        wp = random.choice(route)
        t = wp.transform

        right = right_vector(t)

        lateral = random.choice([
            random.uniform(-7, -4),
            random.uniform(4, 7)
        ])

        location = t.location + right*lateral
        location.z += 0.1

        if spawn_prop(
            random.choice(props),
            location,
            carla.Rotation(
                yaw=t.rotation.yaw + random.uniform(-20, 20)
            )
        ):
            created += 1

    print("Construction objects:", created)


def add_parked_vehicles(route):
    library = world.get_blueprint_library()
    vehicles = library.filter("vehicle.*")
    if not vehicles:
        return

    created_by_side = {"right": 0, "left": 0}
    locations = []
    right_count = (NUM_PARKED + 1) // 2
    side_plan = [1.0] * right_count + [-1.0] * (NUM_PARKED - right_count)
    random.shuffle(side_plan)
    route_length = sum(
        distance(first.transform.location, second.transform.location)
        for first, second in zip(route, route[1:])
    )
    end_parked_count = min(NUM_PARKED_PER_ROUTE_END, NUM_PARKED // 2)

    for index, side in enumerate(side_plan):
        route_distance_range = None
        if route_length > 2.0 * PARKED_END_ZONE_METERS:
            if index < end_parked_count:
                route_distance_range = (35.0, PARKED_END_ZONE_METERS)
            elif index < 2 * end_parked_count:
                route_distance_range = (
                    route_length - PARKED_END_ZONE_METERS,
                    route_length - 35.0,
                )

        for _ in range(20):
            if route_distance_range is None:
                waypoint = random.choice(route)
            else:
                target_distance = random.uniform(*route_distance_range)
                waypoint = route_waypoint_at_distance(route, target_distance)
                if waypoint is None:
                    continue
            right = right_vector(waypoint.transform)
            side_lanes = [
                (lateral, lane)
                for lateral, lane in driving_lanes_across_road(waypoint)
                if side * lateral > 0.0
            ]
            if not side_lanes:
                continue
            outer_lane = max(side_lanes, key=lambda item: side * item[0])[1]

            outer_location = outer_lane.transform.location
            shoulder_offset = (
                outer_lane.lane_width * 0.5
                + PARKING_SHOULDER_WIDTH_M * 0.5
            )
            location = carla.Location(
                x=outer_location.x + right.x * side * shoulder_offset,
                y=outer_location.y + right.y * side * shoulder_offset,
                z=outer_location.z + 0.4,
            )
            if ego and distance(location, ego.get_location()) < 20.0:
                continue
            if any(distance(location, old) < 8.0 for old in locations):
                continue

            actor = spawn_prop(
                random.choice(vehicles),
                location,
                outer_lane.transform.rotation,
            )
            if actor is not None:
                locations.append(location)
                created_by_side["right" if side > 0 else "left"] += 1
                break

    print(
        "Parked vehicles on shoulders: "
        f"{created_by_side['right']} right, {created_by_side['left']} left."
    )


def route_waypoint_at_distance(route, target_distance):
    """Return the route Driving waypoint nearest a distance from its start."""
    if not route or target_distance < 0.0:
        return None

    travelled = 0.0
    closest = route[0]
    closest_error = abs(target_distance)
    for first, second in zip(route, route[1:]):
        travelled += distance(
            first.transform.location,
            second.transform.location,
        )
        error = abs(travelled - target_distance)
        if error < closest_error:
            closest = second
            closest_error = error
        if travelled >= target_distance:
            break

    return closest if is_driving_waypoint(closest) else None


def scenario_route_distances(route, count, start=65.0):
    """Spread scenario actors along the available first kilometre of route."""
    route_length = sum(
        distance(first.transform.location, second.transform.location)
        for first, second in zip(route, route[1:])
    )
    end = min(SCENARIO_HAZARD_LENGTH_METERS - 35.0, route_length - 35.0)
    if count <= 0 or end < start:
        return []
    if count == 1:
        return [start]
    return [
        start + (end - start) * index / (count - 1)
        for index in range(count)
    ]


def bypassable_obstacle_waypoint(route, route_distance):
    """Choose a non-junction route point with a same-direction Driving lane."""
    offsets = [0.0] + [
        sign * step
        for step in range(10, 101, 10)
        for sign in (1.0, -1.0)
    ]
    for offset in offsets:
        candidate_distance = route_distance + offset
        if candidate_distance < 60.0 or any(
            abs(candidate_distance - used_distance) < 40.0
            for used_distance in obstacle_route_distances
        ):
            continue

        waypoint = route_waypoint_at_distance(route, candidate_distance)
        if waypoint is None or waypoint.is_junction:
            continue
        ahead = waypoint.next(30.0)
        behind = waypoint.previous(15.0)
        if (
            not ahead
            or not behind
            or ahead[0].is_junction
            or behind[0].is_junction
        ):
            continue

        forward = waypoint.transform.get_forward_vector()
        has_same_direction_lane = False
        for side in (waypoint.get_left_lane(), waypoint.get_right_lane()):
            if not is_driving_waypoint(side) or side.is_junction:
                continue
            side_forward = side.transform.get_forward_vector()
            if (
                forward.x * side_forward.x
                + forward.y * side_forward.y
            ) >= 0.5:
                has_same_direction_lane = True
                break
        if not has_same_direction_lane:
            continue

        obstacle_route_distances.append(candidate_distance)
        return waypoint
    return None


def add_route_obstacles(route):
    """Place vehicle hazards where a same-direction Driving lane is nearby."""
    if NUM_ROUTE_OBSTACLES == 0:
        print("Stationary route vehicle hazards: disabled.")
        return

    vehicles = world.get_blueprint_library().filter("vehicle.*")
    if not vehicles:
        print("Route vehicle obstacles unavailable; no hazards spawned.")
        return

    distances = scenario_route_distances(route, NUM_ROUTE_OBSTACLES)
    created = 0
    for route_distance in distances:
        waypoint = bypassable_obstacle_waypoint(route, route_distance)
        if waypoint is None:
            print(
                "Skipped route obstacle at "
                f"{route_distance:.0f} m: no Driving waypoint."
            )
            continue
        transform = waypoint.transform
        location = carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z + 0.25,
        )
        actor = world.try_spawn_actor(
            random.choice(vehicles),
            carla.Transform(location, transform.rotation),
        )
        if actor is None:
            print(
                "Could not spawn route obstacle at "
                f"{route_distance:.0f} m."
            )
            continue
        actor.set_simulate_physics(False)
        environment_actors.append(actor)
        created += 1

    print(
        f"Stationary route obstacles: {created} across "
        f"{SCENARIO_HAZARD_LENGTH_METERS:.0f} m."
    )


def moving_vehicle_blueprints(blueprints, vehicle_class):
    """Select CARLA motorcycle/scooter or van blueprints by their asset IDs."""
    tokens = {
        "scooter": (
            "vespa",
            "scooter",
            "motorcycle",
            "ninja",
            "harley-davidson",
        ),
        "van": ("van", "volkswagen.t2", "sprinter", "transit"),
    }
    if vehicle_class not in tokens:
        raise ValueError(f"Unsupported moving vehicle class: {vehicle_class}")

    return [
        blueprint
        for blueprint in blueprints
        if any(
            token in blueprint.id.lower()
            for token in tokens[vehicle_class]
        )
    ]


def moving_vehicle_waypoint(route, route_distance):
    """Find a non-junction route waypoint with clear space around it."""
    offsets = [0.0] + [
        sign * step
        for step in range(10, 51, 10)
        for sign in (1.0, -1.0)
    ]
    for offset in offsets:
        candidate_distance = route_distance + offset
        if candidate_distance < 45.0 or any(
            abs(candidate_distance - used_distance) < 25.0
            for used_distance in obstacle_route_distances
        ):
            continue

        waypoint = route_waypoint_at_distance(route, candidate_distance)
        if waypoint is None or waypoint.is_junction:
            continue
        ahead = waypoint.next(15.0)
        if not ahead or ahead[0].is_junction:
            continue
        if (
            waypoint.lane_id != ahead[0].lane_id
            or not is_driving_segment(
                world.get_map(),
                waypoint.transform.location,
                ahead[0].transform.location,
            )
        ):
            continue

        obstacle_route_distances.append(candidate_distance)
        return waypoint
    return None


def add_moving_traffic(route):
    """Spawn traffic on the ego route, with CARLA TM collision avoidance."""
    if traffic_manager is None:
        raise RuntimeError(
            "CARLA Traffic Manager is unavailable; moving traffic cannot start."
        )

    library = world.get_blueprint_library()
    vehicles = library.filter("vehicle.*")
    scooter_blueprints = moving_vehicle_blueprints(vehicles, "scooter")
    van_blueprints = moving_vehicle_blueprints(vehicles, "van")
    fallback_cars = [
        blueprint
        for blueprint in vehicles
        if not any(
            token in blueprint.id.lower()
            for token in (
                "van",
                "volkswagen.t2",
                "sprinter",
                "transit",
                "bus",
                "truck",
                "firetruck",
                "ambulance",
                "vespa",
                "scooter",
                "motorcycle",
                "ninja",
                "harley-davidson",
            )
        )
    ]
    use_car_fallback = not scooter_blueprints
    use_van_fallback = not van_blueprints
    traffic_classes = (
        [
            (
                "moving car (scooter model unavailable)"
                if use_car_fallback
                else "scooter",
                fallback_cars if use_car_fallback else scooter_blueprints,
            )
        ] * (NUM_FALLBACK_MOVING_CARS if use_car_fallback else NUM_MOVING_SCOOTERS)
        + [
            (
                "moving car (van model unavailable)"
                if use_van_fallback
                else "van",
                fallback_cars if use_van_fallback else van_blueprints,
            )
        ] * NUM_MOVING_VANS
    )
    route_distances = scenario_route_distances(
        route,
        len(traffic_classes),
        start=70.0,
    )
    created = {"scooter": 0, "van": 0, "fallback car": 0}
    moving_traffic_actors.clear()

    for (vehicle_class, blueprints), route_distance in zip(
        traffic_classes,
        route_distances,
    ):
        if not blueprints:
            continue
        waypoint = moving_vehicle_waypoint(route, route_distance)
        if waypoint is None:
            print(
                f"Skipped moving {vehicle_class}: no clear "
                "non-junction Driving waypoint."
            )
            continue

        transform = waypoint.transform
        location = carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z + 0.5,
        )
        actor = world.try_spawn_actor(
            random.choice(blueprints),
            carla.Transform(location, transform.rotation),
        )
        if actor is None:
            print(
                f"Could not spawn moving {vehicle_class} "
                f"at route distance {route_distance:.0f} m."
            )
            continue

        environment_actors.append(actor)
        actor.set_autopilot(True, TRAFFIC_MANAGER_PORT)
        route_index = min(
            range(len(route)),
            key=lambda index: distance(
                route[index].transform.location,
                transform.location,
            ),
        )
        traffic_manager.set_path(
            actor,
            [
                waypoint.transform.location
                for waypoint in route[route_index:]
            ],
        )
        traffic_manager.set_desired_speed(actor, CRUISE_SPEED)
        traffic_manager.auto_lane_change(actor, False)
        moving_traffic_actors.append(actor)
        for other in [ego, *moving_traffic_actors[:-1]]:
            traffic_manager.collision_detection(actor, other, True)
            traffic_manager.collision_detection(other, actor, True)
        if vehicle_class.startswith("moving car"):
            created["fallback car"] += 1
        else:
            created[vehicle_class] += 1

    if not scooter_blueprints:
        print(
            "No scooter/motorcycle blueprint; substituting moving car "
            f"blueprints when available ({NUM_FALLBACK_MOVING_CARS} requested)."
        )
    if not van_blueprints:
        print(
            "No van blueprint; substituting moving car blueprints "
            "when available."
        )
    print(
        "Moving traffic: "
        f"{created['scooter']} scooter/motorcycle(s), "
        f"{created['van']} van(s), "
        f"{created['fallback car']} fallback car(s); following the CARLA "
        "route with Traffic Manager collision avoidance enabled."
    )


def update_moving_traffic_speed():
    """Match traffic target speed to ego speed; TM handles safe headways."""
    if ego is None or traffic_manager is None:
        return

    # Keep NPC traffic moving even while ego briefly brakes for a crossing;
    # Traffic Manager collision avoidance remains enabled for every NPC.
    target_speed = CRUISE_SPEED
    for actor in moving_traffic_actors:
        if actor.is_alive:
            traffic_manager.set_desired_speed(actor, target_speed)


def add_pothole_markers(route):
    """Add flat tire props as visible, actor-detectable pothole proxies."""
    if NUM_POTHOLE_MARKERS == 0:
        print("Pothole markers: disabled.")
        return

    tire_blueprints = world.get_blueprint_library().filter("static.prop.tire")
    if not tire_blueprints:
        print("Pothole markers unavailable (static.prop.tire not found).")
        return

    created = 0
    for _ in range(NUM_POTHOLE_MARKERS):
        waypoint = random.choice(route)
        transform = waypoint.transform
        right = right_vector(transform)
        location = (
            transform.location
            + right * random.choice((
                random.uniform(-3.0, -2.5),
                random.uniform(2.5, 3.0),
            ))
            + forward_vector(transform) * random.uniform(-2.0, 2.0)
        )
        location.z += 0.03
        marker = spawn_prop(
            random.choice(tire_blueprints),
            location,
            carla.Rotation(pitch=90, yaw=random.uniform(0, 360)),
        )
        if marker:
            marker.set_simulate_physics(False)
            created += 1

    print("Visual tire pothole proxies:", created)


def add_road_pedestrians(route):
    """Spawn pedestrians to cross the complete road from left to right."""
    if NUM_PEDESTRIANS == 0:
        print("Road pedestrians: disabled.")
        return

    road_pedestrian_states.clear()
    library = world.get_blueprint_library()
    walkers = library.filter("walker.pedestrian.*")
    if not walkers:
        print("Pedestrian blueprints unavailable; no road walkers spawned.")
        return

    # Walkers are manually controlled along shoulder-to-shoulder endpoints;
    # this does not require a CARLA pedestrian navigation mesh.

    pedestrian_distances = scenario_route_distances(
        route,
        NUM_PEDESTRIANS,
        start=90.0,
    )
    created = 0
    failed = 0
    for route_distance in pedestrian_distances:
        spawned = False
        for offset in (0.0, -6.0, 6.0, -12.0, 12.0, -18.0, 18.0):
            crossing_distance = route_distance + offset
            if crossing_distance < 25.0:
                continue
            if any(
                abs(crossing_distance - used_distance) < 20.0
                for used_distance in obstacle_route_distances
            ):
                continue

            waypoint = route_waypoint_at_distance(route, crossing_distance)
            if waypoint is None or waypoint.is_junction:
                continue
            if distance(waypoint.transform.location, ego.get_location()) < 25.0:
                continue

            transform = waypoint.transform
            endpoints = road_crossing_endpoints(waypoint)
            if endpoints is None:
                continue
            start_location, end_location = endpoints
            end_location.z += 0.1
            walker_blueprint = random.choice(walkers)
            walker = None
            for spawn_height in (0.5, 1.0):
                spawn_location = carla.Location(
                    x=start_location.x,
                    y=start_location.y,
                    z=start_location.z + spawn_height,
                )
                walker = world.try_spawn_actor(
                    walker_blueprint,
                    carla.Transform(
                        spawn_location,
                        carla.Rotation(yaw=transform.rotation.yaw),
                    ),
                )
                if walker is not None:
                    break
            if walker is None or not walker.is_alive:
                continue

            environment_actors.append(walker)
            road_pedestrian_states.append({
                "actor": walker,
                "crossing_end": end_location,
                "started": False,
                "completed": False,
                "speed": random.uniform(0.8, 1.0),
            })
            obstacle_route_distances.append(crossing_distance)
            created += 1
            spawned = True
            break

        if not spawned:
            failed += 1

    print(
        f"Left-to-right pedestrian crossings: {created}/{len(pedestrian_distances)} "
        f"spawned; {failed} could not find a clear shoulder spot."
    )


def update_road_pedestrians():
    """Start crossings as ego approaches and remove walkers after they clear."""
    for pedestrian in list(road_pedestrian_states):
        actor = pedestrian["actor"]
        if not actor.is_alive:
            road_pedestrian_states.remove(pedestrian)
            if actor in environment_actors:
                environment_actors.remove(actor)
            continue

        location = actor.get_location()
        if pedestrian["completed"]:
            if actor.destroy():
                road_pedestrian_states.remove(pedestrian)
                if actor in environment_actors:
                    environment_actors.remove(actor)
            else:
                actor.apply_control(
                    carla.WalkerControl(
                        direction=carla.Vector3D(),
                        speed=0.0,
                        jump=False,
                    )
                )
            continue

        if (
            not pedestrian["started"]
            and distance(ego.get_location(), location)
            <= PEDESTRIAN_CROSSING_TRIGGER_M
        ):
            pedestrian["started"] = True

        if not pedestrian["started"]:
            actor.apply_control(
                carla.WalkerControl(
                    direction=carla.Vector3D(),
                    speed=0.0,
                    jump=False,
                )
            )
            continue

        target = pedestrian["crossing_end"]
        walker_half_width = max(
            actor.bounding_box.extent.x,
            actor.bounding_box.extent.y,
        )
        if distance(location, target) <= max(0.5, walker_half_width + 0.25):
            pedestrian["completed"] = True
            if actor.destroy():
                road_pedestrian_states.remove(pedestrian)
                if actor in environment_actors:
                    environment_actors.remove(actor)
                print(
                    f"Pedestrian {actor.id} cleared the right shoulder "
                    "and was removed."
                )
            else:
                pedestrian["completed"] = True
                actor.apply_control(
                    carla.WalkerControl(
                        direction=carla.Vector3D(),
                        speed=0.0,
                        jump=False,
                    )
                )
            continue

        delta = target - location
        length = math.hypot(delta.x, delta.y)
        if length < 1e-6:
            continue
        actor.apply_control(
            carla.WalkerControl(
                direction=carla.Vector3D(
                    delta.x / length,
                    delta.y / length,
                    0.0,
                ),
                speed=pedestrian["speed"],
                jump=False,
            )
        )

    road_pedestrian_states[:] = [
        pedestrian
        for pedestrian in road_pedestrian_states
        if pedestrian["actor"].is_alive
    ]
    environment_actors[:] = [
        actor for actor in environment_actors if actor.is_alive
    ]


def road_pedestrian_crossing_requires_stop():
    """
    Keep the ego vehicle fully stopped from the moment a crossing walker is
    triggered until that walker reaches the opposite shoulder and is marked
    completed. The pedestrian state is updated every simulation loop.
    """
    if ego is None or not ego.is_alive:
        return False

    ego_location = ego.get_location()
    for pedestrian in road_pedestrian_states:
        actor = pedestrian["actor"]
        if not actor.is_alive or pedestrian["completed"]:
            continue

        distance_to_walker = distance(ego_location, actor.get_location())
        crossing_triggered = (
            pedestrian["started"]
            or distance_to_walker <= PEDESTRIAN_CROSSING_TRIGGER_M
        )
        if crossing_triggered:
            return True

    return False


# ============================================================
# COLLISION SENSOR
# ============================================================

collision_state = {
    "hit": False,
    "last_time": 0.0,
    "last_actor_id": None,
    "last_actor_type": None,
    "last_location": None,
}


def attach_collision_sensor():

    library = world.get_blueprint_library()

    bp = library.find("sensor.other.collision")

    sensor = world.spawn_actor(
        bp,
        carla.Transform(),
        attach_to=ego
    )

    sensor_actors.append(sensor)

    def callback(event):
        other_actor = event.other_actor
        actor_id = other_actor.id if other_actor else None
        actor_type = other_actor.type_id if other_actor else "unknown actor"
        if ego is None or not ego.is_alive:
            return
        ego_location = ego.get_location()
        previous_location = collision_state["last_location"]
        same_actor_episode = (
            actor_type == collision_state["last_actor_type"]
            and (
                actor_id == collision_state["last_actor_id"]
                or actor_type == "static.unknown"
            )
            and previous_location is not None
            and distance(ego_location, previous_location)
            < COLLISION_REPEAT_DISTANCE_METERS
        )
        if same_actor_episode:
            return

        collision_state["hit"] = True
        collision_state["last_time"] = time.monotonic()
        collision_state["last_actor_id"] = actor_id
        collision_state["last_actor_type"] = actor_type
        collision_state["last_location"] = carla.Location(
            x=ego_location.x,
            y=ego_location.y,
            z=ego_location.z,
        )

        print(
            "[COLLISION]",
            actor_type
        )

    sensor.listen(callback)

    print("Collision sensor attached.")

    return sensor


# ============================================================
# FORWARD OBSTACLE CHECK
# ============================================================

def find_forward_obstacle(max_distance=30.0):
    """Return the nearest actor's estimated front-bumper clearance in metres."""
    ego_tf = ego.get_transform()
    origin = ego_tf.location
    ego_box_center = ego_tf.transform(ego.bounding_box.location)
    forward = ego_tf.get_forward_vector()
    right = right_vector(ego_tf)
    ego_half_width = ego.bounding_box.extent.y
    ego_half_length = ego.bounding_box.extent.x
    best_clearance = float("inf")
    best_actor = None
    route_segments = []
    if avoidance_active and avoidance_points:
        corridor_route = avoidance_points
        route_index = avoidance_point_index
    elif route_recovery_active and route_recovery_points:
        corridor_route = route_recovery_points
        route_index = route_recovery_point_index
    else:
        corridor_route = active_route
        route_index = active_route_index
    route_index = max(0, min(route_index, len(corridor_route) - 1))
    if corridor_route and distance(
        origin,
        corridor_route[route_index].transform.location,
    ) <= 6.0:
        progress = 0.0
        previous = origin
        first_route_location = corridor_route[route_index].transform.location
        first_length = distance(previous, first_route_location)
        if first_length > 0.1 and is_driving_segment(
            world.get_map(),
            previous,
            first_route_location,
        ):
            route_segments.append(
                (previous, first_route_location, progress, first_length)
            )
            progress += first_length
        previous = first_route_location
        for waypoint in corridor_route[route_index + 1:]:
            location = waypoint.transform.location
            segment_length = distance(previous, location)
            if segment_length > 0.1:
                route_segments.append(
                    (previous, location, progress, segment_length)
                )
                progress += segment_length
            previous = location
            if progress >= max_distance + 8.0:
                break

    candidates = list(world.get_actors())
    candidates.extend(
        static_hazards_near(origin, max_distance + 12.0)
    )
    for actor in candidates:
        if not blocking_actor(actor):
            continue

        actor_transform = actor.get_transform()
        local_box_center = actor.bounding_box.location
        loc = actor_transform.transform(
            carla.Location(
                x=local_box_center.x,
                y=local_box_center.y,
                z=local_box_center.z,
            )
        )
        if abs(loc.z - ego_box_center.z) > (
            ego.bounding_box.extent.z
            + actor.bounding_box.extent.z
            + 0.1
        ):
            continue
        rel = loc - origin
        longitudinal = rel.x * forward.x + rel.y * forward.y
        lateral = abs(rel.x * right.x + rel.y * right.y)

        extent = actor.bounding_box.extent
        actor_forward = actor_transform.get_forward_vector()
        actor_forward_length = math.hypot(actor_forward.x, actor_forward.y)
        if actor_forward_length < 1e-6:
            continue
        actor_forward = carla.Vector3D(
            actor_forward.x / actor_forward_length,
            actor_forward.y / actor_forward_length,
            0.0,
        )
        actor_right = carla.Vector3D(
            -actor_forward.y,
            actor_forward.x,
            0.0,
        )
        actor_longitudinal_extent = (
            extent.x * abs(
                actor_forward.x * forward.x
                + actor_forward.y * forward.y
            )
            + extent.y * abs(
                actor_right.x * forward.x
                + actor_right.y * forward.y
            )
        )
        actor_lateral_extent = (
            extent.x * abs(
                actor_forward.x * right.x
                + actor_forward.y * right.y
            )
            + extent.y * abs(
                actor_right.x * right.x
                + actor_right.y * right.y
            )
        )
        if actor.type_id.lower().startswith("static.prop") and longitudinal > 18.0:
            continue
        clearance = float("inf")
        if longitudinal > 0.0:
            lateral_clearance = (
                lateral - ego_half_width - actor_lateral_extent
            )
            if lateral_clearance <= 0.35:
                clearance = max(
                    0.0,
                    longitudinal
                    - ego_half_length
                    - actor_longitudinal_extent,
                )

        for segment_start, segment_end, path_progress, segment_length in route_segments:
            segment_x = segment_end.x - segment_start.x
            segment_y = segment_end.y - segment_start.y
            fraction = clamp(
                (
                    (loc.x - segment_start.x) * segment_x
                    + (loc.y - segment_start.y) * segment_y
                )
                / (segment_length * segment_length),
                0.0,
                1.0,
            )
            projected_x = segment_start.x + fraction * segment_x
            projected_y = segment_start.y + fraction * segment_y
            lateral_to_route = math.hypot(
                loc.x - projected_x,
                loc.y - projected_y,
            )
            if lateral_to_route > (
                ego_half_width + actor_lateral_extent + 0.5
            ):
                continue
            route_clearance = max(
                0.0,
                path_progress
                + fraction * segment_length
                - ego_half_length
                - actor_longitudinal_extent,
            )
            clearance = min(clearance, route_clearance)

        if clearance > max_distance:
            continue
        if clearance < best_clearance:
            best_clearance = clearance
            best_actor = actor

    return best_actor, best_clearance


# ============================================================
# DESTINATION
# ============================================================

def set_destination(location):
    """Store the CARLA waypoint selected using the OSM desired direction."""
    global destination_location

    destination_location = location

    print(
        "DESTINATION SET:",
        location
    )


# ============================================================
# ============================================================
# LOCAL OBSTACLE AVOIDANCE + STEERING
# ============================================================

# A detour remains active until its merge point is reached or a safety check
# fails. It never expires back into blind route following.
avoidance_active = False
avoidance_points = []
avoidance_point_index = 0
avoidance_obstacle_id = None
avoidance_last_progress_time = 0.0
avoidance_last_progress_location = None
blocked_obstacle_id = None
blocked_obstacle_retry_attempted = False
avoidance_search_obstacle_id = None
avoidance_search_stop_retry_attempted = False
avoidance_blocked_replan_time = 0.0
active_route = []
active_route_index = 0
avoidance_merge_route_index = None
bypass_last_search_time = 0.0
obstacle_route_distances = []
chosen_avoidance_direction = "NONE"
current_risk_level = "LOW"
current_target_speed = CRUISE_SPEED



def blocking_actor(actor):
    """Return vehicles, pedestrians and physical props as obstacles."""
    if actor is None or not actor.is_alive:
        return False

    if ego is not None and actor.id == ego.id:
        return False

    tid = actor.type_id.lower()

    # Dynamic objects that can block the driving path.
    if (
        tid.startswith("vehicle.")
        or tid.startswith("walker.pedestrian")
        or tid.startswith("static.prop")
    ):
        return True

    # Do not treat every static map object, especially vegetation,
    # as a driving obstacle. Static-map hazards need separate filtering.
    return False



class StaticMapHazard:
    """Actor-like view of a static CARLA environment object."""

    def __init__(self, object_id, name, label, transform, bounding_box):
        self.id = f"static:{object_id}"
        self.type_id = f"static.map.{label.lower()}.{name}"
        self.is_alive = True
        self._transform = transform
        self.bounding_box = bounding_box

    def get_transform(self):
        return self._transform

    def get_location(self):
        return self._transform.location


def _static_hazard_cell(x, y):
    return (
        math.floor(x / STATIC_HAZARD_CELL_SIZE),
        math.floor(y / STATIC_HAZARD_CELL_SIZE),
    )


def _safe_environment_objects(label):
    try:
        return world.get_environment_objects(label)
    except RuntimeError:
        return []


def _index_static_map_hazards():
    """Load physical map objects once and index them by CARLA-world position."""
    global static_hazard_map_name
    global static_hazard_cells
    global static_hazards_by_id

    map_name = world.get_map().name
    if static_hazard_map_name == map_name:
        return

    cells = {}
    hazards_by_id = {}
    for label_name in STATIC_HAZARD_LABELS:
        label = getattr(carla.CityObjectLabel, label_name)
        for environment_object in _safe_environment_objects(label):
            object_name = str(environment_object.name)
            hazard_id = f"static:{environment_object.id}"
            if hazard_id in hazards_by_id:
                continue

            source_box = environment_object.bounding_box
            extent = source_box.extent
            if max(extent.x, extent.y, extent.z) <= 0.05:
                continue
            if (
                label_name == "Vegetation"
                and (
                    max(extent.x, extent.y) < 0.25
                    or extent.z < 0.25
                )
            ):
                continue

            source_transform = environment_object.transform
            center = source_transform.transform(source_box.location)
            transform = carla.Transform(
                center,
                carla.Rotation(
                    pitch=0.0,
                    yaw=(
                        source_transform.rotation.yaw
                        + source_box.rotation.yaw
                    ),
                    roll=0.0,
                ),
            )
            box = carla.BoundingBox(
                carla.Location(),
                carla.Vector3D(extent.x, extent.y, extent.z),
            )
            hazard = StaticMapHazard(
                environment_object.id,
                object_name,
                label_name,
                transform,
                box,
            )
            hazards_by_id[hazard.id] = hazard

            radius = math.hypot(extent.x, extent.y) + 2.0
            min_cell = _static_hazard_cell(
                center.x - radius,
                center.y - radius,
            )
            max_cell = _static_hazard_cell(
                center.x + radius,
                center.y + radius,
            )
            for cell_x in range(min_cell[0], max_cell[0] + 1):
                for cell_y in range(min_cell[1], max_cell[1] + 1):
                    cells.setdefault((cell_x, cell_y), []).append(hazard)

    static_hazard_map_name = map_name
    static_hazard_cells = cells
    static_hazards_by_id = hazards_by_id
    print(
        f"Indexed {len(hazards_by_id)} static CARLA hazards "
        f"on {map_name}."
    )


def static_hazards_near(location, radius):
    """Return unique indexed static hazards whose bounds are nearby."""
    _index_static_map_hazards()
    min_cell = _static_hazard_cell(
        location.x - radius,
        location.y - radius,
    )
    max_cell = _static_hazard_cell(
        location.x + radius,
        location.y + radius,
    )
    candidates = {}
    for cell_x in range(min_cell[0], max_cell[0] + 1):
        for cell_y in range(min_cell[1], max_cell[1] + 1):
            for hazard in static_hazard_cells.get((cell_x, cell_y), ()):
                candidates[hazard.id] = hazard

    return [
        hazard
        for hazard in candidates.values()
        if distance(location, hazard.get_location())
        <= radius + math.hypot(
            hazard.bounding_box.extent.x,
            hazard.bounding_box.extent.y,
        )
    ]


def nearby_blocking_objects(location, radius):
    """Return dynamic actors and indexed static objects in a local region."""
    nearby = [
        actor
        for actor in world.get_actors()
        if blocking_actor(actor)
        and distance(location, actor.get_location()) <= radius
    ]
    nearby.extend(static_hazards_near(location, radius))
    return nearby


def point_clearance(point, actors=None, forward=None):
    """Estimate oriented-box clearance between the ego and nearby actors."""
    best = float("inf")
    if forward is None:
        forward = ego.get_transform().get_forward_vector()
    forward_length = math.hypot(forward.x, forward.y)
    if forward_length < 1e-6:
        return float("-inf")
    ego_forward = carla.Vector3D(
        forward.x / forward_length,
        forward.y / forward_length,
        0.0,
    )
    ego_right = carla.Vector3D(-ego_forward.y, ego_forward.x, 0.0)
    ego_extent = ego.bounding_box.extent

    if actors is None:
        actors = nearby_blocking_objects(point, 40.0)
    for actor in actors:
        if not blocking_actor(actor):
            continue
        loc = actor.get_location()
        actor_extent = actor.bounding_box.extent
        actor_transform = actor.get_transform()
        actor_forward = actor_transform.get_forward_vector()
        actor_forward_length = math.hypot(
            actor_forward.x,
            actor_forward.y,
        )
        if actor_forward_length < 1e-6:
            actor_forward = ego_forward
        else:
            actor_forward = carla.Vector3D(
                actor_forward.x / actor_forward_length,
                actor_forward.y / actor_forward_length,
                0.0,
            )
        actor_right = carla.Vector3D(
            -actor_forward.y,
            actor_forward.x,
            0.0,
        )
        offset_x = loc.x - point.x
        offset_y = loc.y - point.y
        clearance = float("-inf")
        for axis in (
            ego_forward,
            ego_right,
            actor_forward,
            actor_right,
        ):
            ego_radius = (
                ego_extent.x * abs(
                    ego_forward.x * axis.x + ego_forward.y * axis.y
                )
                + ego_extent.y * abs(
                    ego_right.x * axis.x + ego_right.y * axis.y
                )
            )
            actor_radius = (
                actor_extent.x * abs(
                    actor_forward.x * axis.x
                    + actor_forward.y * axis.y
                )
                + actor_extent.y * abs(
                    actor_right.x * axis.x
                    + actor_right.y * axis.y
                )
            )
            axis_clearance = (
                abs(offset_x * axis.x + offset_y * axis.y)
                - ego_radius
                - actor_radius
            )
            clearance = max(clearance, axis_clearance)
        clearance -= 0.5
        best = min(best, clearance)

    return best


def detour_segment_clearance(start, end, actors=None):
    map_obj = world.get_map()
    if not is_driving_segment(map_obj, start, end):
        return None

    segment_length = distance(start, end)
    sample_count = max(1, math.ceil(segment_length))
    if segment_length > 1e-6:
        forward = carla.Vector3D(
            (end.x - start.x) / segment_length,
            (end.y - start.y) / segment_length,
            0.0,
        )
    else:
        forward = ego.get_transform().get_forward_vector()
    minimum_clearance = float("inf")
    for index in range(sample_count + 1):
        fraction = index / sample_count
        point = carla.Location(
            x=start.x + (end.x - start.x) * fraction,
            y=start.y + (end.y - start.y) * fraction,
            z=start.z + (end.z - start.z) * fraction,
        )
        clearance = point_clearance(
            point,
            actors=actors,
            forward=forward,
        )
        if clearance < 0.75:
            return None
        minimum_clearance = min(minimum_clearance, clearance)
    return minimum_clearance


def advance_driving_lane(map_obj, start_waypoint, distance_m):
    """Advance along one Driving lane, refusing lane changes or non-driving links."""
    if not is_driving_waypoint(start_waypoint):
        return None

    current = start_waypoint
    remaining = distance_m
    while remaining > 0.5:
        step = min(ROUTE_STEP_METERS, remaining)
        choices = [
            candidate
            for candidate in current.next(step)
            if is_driving_waypoint(candidate)
            and candidate.lane_id == current.lane_id
            and is_driving_segment(
                map_obj,
                current.transform.location,
                candidate.transform.location,
            )
        ]
        if not choices:
            return None
        current_forward = current.transform.get_forward_vector()
        current = max(
            choices,
            key=lambda candidate: (
                current_forward.x * candidate.transform.get_forward_vector().x
                + current_forward.y * candidate.transform.get_forward_vector().y
            ),
        )
        remaining -= step
    return current


def find_adjacent_lane_bypass(
    map_obj,
    current_waypoint,
    origin,
    route,
    first_route_index,
    minimum_merge_progress,
    nearby_actors,
):
    """Find a clear adjacent Driving-lane path that rejoins after the obstacle."""
    origin_to_waypoint = distance(
        origin,
        current_waypoint.transform.location,
    )
    route_progress = {}
    progress = origin_to_waypoint + distance(
        current_waypoint.transform.location,
        route[first_route_index].transform.location,
    )
    previous_location = current_waypoint.transform.location
    for route_index in range(first_route_index, len(route)):
        route_waypoint = route[route_index]
        route_location = route_waypoint.transform.location
        if route_index > first_route_index:
            progress += distance(previous_location, route_location)
        route_progress[route_index] = progress
        previous_location = route_location

    best_candidates = {"LEFT": None, "RIGHT": None}
    forward = current_waypoint.transform.get_forward_vector()
    for lane_method in ("get_left_lane", "get_right_lane"):
        adjacent_start = getattr(current_waypoint, lane_method)()
        if (
            not is_driving_waypoint(adjacent_start)
            or adjacent_start.is_junction
            or adjacent_start.road_id != current_waypoint.road_id
            or adjacent_start.section_id != current_waypoint.section_id
        ):
            continue
        # Either physical side is allowed, but only same-direction Driving lanes.
        direction = (
            "RIGHT"
            if is_physical_right_lane_change(current_waypoint, adjacent_start)
            else "LEFT"
        )

        adjacent_forward = adjacent_start.transform.get_forward_vector()
        if (
            forward.x * adjacent_forward.x
            + forward.y * adjacent_forward.y
            < 0.5
        ):
            continue
        if detour_segment_clearance(
            current_waypoint.transform.location,
            adjacent_start.transform.location,
            actors=nearby_actors,
        ) is None:
            continue

        path = [current_waypoint, adjacent_start]
        current = adjacent_start
        path_length = origin_to_waypoint + distance(
            current_waypoint.transform.location,
            adjacent_start.transform.location,
        )
        path_clearance = detour_segment_clearance(
            current_waypoint.transform.location,
            adjacent_start.transform.location,
            actors=nearby_actors,
        )

        while path_length <= DETOUR_MAX_DISTANCE_METERS:
            current_location = current.transform.location
            for lane_method in ("get_left_lane", "get_right_lane"):
                merge_waypoint = getattr(current, lane_method)()
                if not is_driving_waypoint(merge_waypoint):
                    continue
                if merge_waypoint.lane_id != current_waypoint.lane_id:
                    continue
                connector_clearance = detour_segment_clearance(
                    current_location,
                    merge_waypoint.transform.location,
                    actors=nearby_actors,
                )
                if connector_clearance is None:
                    continue

                for route_index in range(
                    first_route_index + 1,
                    len(route),
                ):
                    route_waypoint = route[route_index]
                    if route_progress[route_index] < minimum_merge_progress:
                        continue
                    if distance(
                        merge_waypoint.transform.location,
                        route_waypoint.transform.location,
                    ) > 1.5:
                        continue
                    route_connector_clearance = detour_segment_clearance(
                        merge_waypoint.transform.location,
                        route_waypoint.transform.location,
                        actors=nearby_actors,
                    )
                    if route_connector_clearance is None:
                        continue
                    combined_clearance = min(
                        path_clearance,
                        connector_clearance,
                        route_connector_clearance,
                    )
                    candidate = (
                        combined_clearance,
                        path_length
                        + distance(
                            current_location,
                            merge_waypoint.transform.location,
                        )
                        + distance(
                            merge_waypoint.transform.location,
                            route_waypoint.transform.location,
                        ),
                        route_index,
                        path + [merge_waypoint, route_waypoint],
                        direction,
                    )
                    candidate_length = candidate[1]
                    if candidate_length > DETOUR_MAX_DISTANCE_METERS:
                        continue
                    best_candidate = best_candidates[direction]
                    if best_candidate is None or (
                        candidate[0],
                        -candidate_length,
                    ) > (
                        best_candidate[0],
                        -best_candidate[1],
                    ):
                        best_candidates[direction] = candidate

            next_waypoints = [
                candidate
                for candidate in current.next(ROUTE_STEP_METERS)
                if is_driving_waypoint(candidate)
                and candidate.lane_id == current.lane_id
                and is_driving_segment(
                    map_obj,
                    current_location,
                    candidate.transform.location,
                )
            ]
            if not next_waypoints:
                break
            current_forward = current.transform.get_forward_vector()
            next_waypoint = max(
                next_waypoints,
                key=lambda candidate: (
                    current_forward.x
                    * candidate.transform.get_forward_vector().x
                    + current_forward.y
                    * candidate.transform.get_forward_vector().y
                ),
            )
            next_location = next_waypoint.transform.location
            edge_clearance = detour_segment_clearance(
                current_location,
                next_location,
                actors=nearby_actors,
            )
            if edge_clearance is None:
                break
            path_length += distance(current_location, next_location)
            path_clearance = min(path_clearance, edge_clearance)
            path.append(next_waypoint)
            current = next_waypoint

    available_candidates = [
        candidate
        for candidate in best_candidates.values()
        if candidate is not None
    ]
    best_candidate = max(
        available_candidates,
        key=lambda candidate: (candidate[0], -candidate[1]),
        default=None,
    )
    if best_candidate is None:
        return None

    return best_candidate[2], best_candidate[3], best_candidate[4]


def walk_driving_lane(map_obj, start_waypoint, max_distance):
    """Follow one non-junction Driving lane without entering oncoming traffic."""
    points = [start_waypoint]
    current = start_waypoint
    walked = 0.0
    while walked < max_distance:
        current_forward = current.transform.get_forward_vector()
        choices = [
            candidate
            for candidate in current.next(ROUTE_STEP_METERS)
            if is_driving_waypoint(candidate)
            and candidate.road_id == current.road_id
            and candidate.lane_id == current.lane_id
            and not candidate.is_junction
            and is_driving_segment(
                map_obj,
                current.transform.location,
                candidate.transform.location,
            )
            and (
                current_forward.x
                * candidate.transform.get_forward_vector().x
                + current_forward.y
                * candidate.transform.get_forward_vector().y
            ) >= 0.5
        ]
        if not choices:
            break

        next_waypoint = max(
            choices,
            key=lambda candidate: (
                current_forward.x
                * candidate.transform.get_forward_vector().x
                + current_forward.y
                * candidate.transform.get_forward_vector().y
            ),
        )
        walked += distance(
            current.transform.location,
            next_waypoint.transform.location,
        )
        points.append(next_waypoint)
        current = next_waypoint
    return points


def walk_opposing_driving_lane(map_obj, start_waypoint, travel_forward, max_distance):
    """Follow an opposing lane backward in its direction, matching ego travel."""
    points = [start_waypoint]
    current = start_waypoint
    walked = 0.0
    while walked < max_distance:
        current_location = current.transform.location
        choices = [
            candidate
            for candidate in current.previous(ROUTE_STEP_METERS)
            if is_driving_waypoint(candidate)
            and candidate.road_id == current.road_id
            and candidate.lane_id == current.lane_id
            and not candidate.is_junction
            and is_driving_segment(
                map_obj,
                current_location,
                candidate.transform.location,
            )
            and (
                (candidate.transform.location.x - current_location.x)
                * travel_forward.x
                + (candidate.transform.location.y - current_location.y)
                * travel_forward.y
            ) >= ROUTE_STEP_METERS * 0.5
        ]
        if not choices:
            break
        next_waypoint = max(
            choices,
            key=lambda candidate: (
                (candidate.transform.location.x - current_location.x)
                * travel_forward.x
                + (candidate.transform.location.y - current_location.y)
                * travel_forward.y
            ),
        )
        walked += distance(
            current_location,
            next_waypoint.transform.location,
        )
        points.append(next_waypoint)
        current = next_waypoint
    return points


def opposing_lane_path_is_clear(path_points, traffic_predictions):
    """Reject opposing-lane passes that conflict with predicted traffic."""
    if not path_points:
        return False

    ego_extent = ego.bounding_box.extent
    planned_speed = max(AVOIDANCE_SPEED / 3.6, 1.0)
    path_progress = 0.0
    for first, second in zip(path_points, path_points[1:]):
        start = first.transform.location
        end = second.transform.location
        segment_length = distance(start, end)
        if segment_length <= 1e-6:
            continue
        ego_forward = carla.Vector3D(
            (end.x - start.x) / segment_length,
            (end.y - start.y) / segment_length,
            0.0,
        )
        ego_right = carla.Vector3D(-ego_forward.y, ego_forward.x, 0.0)
        sample_count = max(1, math.ceil(segment_length / 1.5))
        for index in range(sample_count + 1):
            fraction = index / sample_count
            point = carla.Location(
                x=start.x + (end.x - start.x) * fraction,
                y=start.y + (end.y - start.y) * fraction,
                z=start.z + (end.z - start.z) * fraction,
            )
            eta = min(
                (path_progress + segment_length * fraction) / planned_speed,
                25.0,
            )
            for location, velocity, actor_extent_x, actor_extent_y, actor_forward in traffic_predictions:
                predicted_x = location.x + velocity.x * eta
                predicted_y = location.y + velocity.y * eta
                delta_x = point.x - predicted_x
                delta_y = point.y - predicted_y
                actor_right = carla.Vector3D(
                    -actor_forward.y,
                    actor_forward.x,
                    0.0,
                )
                separated = False
                for axis in (ego_forward, ego_right, actor_forward, actor_right):
                    projected_distance = abs(
                        delta_x * axis.x + delta_y * axis.y
                    )
                    ego_support = (
                        ego_extent.x
                        * abs(ego_forward.x * axis.x + ego_forward.y * axis.y)
                        + ego_extent.y
                        * abs(ego_right.x * axis.x + ego_right.y * axis.y)
                    )
                    actor_support = (
                        actor_extent_x
                        * abs(actor_forward.x * axis.x + actor_forward.y * axis.y)
                        + actor_extent_y
                        * abs(actor_right.x * axis.x + actor_right.y * axis.y)
                    )
                    if projected_distance > ego_support + actor_support + 1.0:
                        separated = True
                        break
                if not separated:
                    return False
        path_progress += segment_length
    return True


def find_smooth_lane_change_bypass(
    map_obj,
    current_waypoint,
    origin,
    route,
    first_route_index,
    minimum_merge_progress,
    nearby_actors,
    obstacle_distance=None,
):
    """Search a same-direction adjacent lane and validate both lane changes."""
    if current_waypoint is None or current_waypoint.is_junction:
        return None

    current_location = current_waypoint.transform.location
    origin_offset = distance(origin, current_location)
    route_progress = {}
    progress = origin_offset
    previous_location = current_location
    for route_index in range(first_route_index, len(route)):
        route_location = route[route_index].transform.location
        progress += distance(previous_location, route_location)
        previous_location = route_location
        route_progress[route_index] = progress
        if progress > DETOUR_MAX_DISTANCE_METERS:
            break

    best_candidates = {
        "LEFT": None,
        "RIGHT": None,
        "LEFT-ONCOMING": None,
    }
    traffic_predictions = []
    for actor in world.get_actors():
        if (
            actor.id == ego.id
            or not (
                actor.type_id.lower().startswith("vehicle.")
                or actor.type_id.lower().startswith("walker.pedestrian")
            )
        ):
            continue
        location = actor.get_location()
        if distance(origin, location) > DETOUR_MAX_DISTANCE_METERS + 60.0:
            continue
        extent = actor.bounding_box.extent
        traffic_predictions.append((
            location,
            actor.get_velocity(),
            extent.x,
            extent.y,
            actor.get_transform().get_forward_vector(),
        ))

    current_forward = current_waypoint.transform.get_forward_vector()
    opposing_lane = False  # Never use an oncoming lane for this safety-first controller.
    for lane_method in ("get_left_lane", "get_right_lane"):
        adjacent_start = getattr(current_waypoint, lane_method)()
        if (
            not is_driving_waypoint(adjacent_start)
            or adjacent_start.is_junction
            or adjacent_start.road_id != current_waypoint.road_id
            or adjacent_start.section_id != current_waypoint.section_id
        ):
            continue

        adjacent_forward = adjacent_start.transform.get_forward_vector()
        alignment = (
            current_forward.x * adjacent_forward.x
            + current_forward.y * adjacent_forward.y
        )
        # Reject opposing/poorly aligned lanes; allow either safe side.
        if alignment < 0.5:
            continue
        direction = (
            "RIGHT"
            if is_physical_right_lane_change(current_waypoint, adjacent_start)
            else "LEFT"
        )
        if opposing_lane:
            lane_points = walk_opposing_driving_lane(
                map_obj,
                adjacent_start,
                current_forward,
                DETOUR_MAX_DISTANCE_METERS,
            )
        else:
            lane_points = walk_driving_lane(
                map_obj,
                adjacent_start,
                DETOUR_MAX_DISTANCE_METERS,
            )
        if len(lane_points) < 4:
            continue

        progress_ahead = [origin_offset]
        for first, second in zip(lane_points, lane_points[1:]):
            progress_ahead.append(
                progress_ahead[-1]
                + distance(
                    first.transform.location,
                    second.transform.location,
                )
            )
        target_entry_dist = 0.0 if opposing_lane else LANE_CHANGE_DISTANCE
        if obstacle_distance is not None and not opposing_lane:
            target_entry_dist = min(LANE_CHANGE_DISTANCE, max(4.0, obstacle_distance * 0.6))
        entry_index = (
            0
            if opposing_lane
            else next(
                (
                    index
                    for index in range(1, len(lane_points))
                    if progress_ahead[index] - progress_ahead[0]
                    >= target_entry_dist
                ),
                None,
            )
        )
        if entry_index is None:
            continue

        entry_clearance = detour_segment_clearance(
            current_location,
            lane_points[entry_index].transform.location,
            actors=nearby_actors,
        )
        if (
            entry_clearance is None
            or entry_clearance < BYPASS_MIN_CLEARANCE
        ):
            continue

        path_clearance = entry_clearance
        for lane_index in range(entry_index, len(lane_points)):
            if lane_index > entry_index:
                edge_clearance = detour_segment_clearance(
                    lane_points[lane_index - 1].transform.location,
                    lane_points[lane_index].transform.location,
                    actors=nearby_actors,
                )
                if (
                    edge_clearance is None
                    or edge_clearance < BYPASS_MIN_CLEARANCE
                ):
                    break
                path_clearance = min(path_clearance, edge_clearance)

            if progress_ahead[lane_index] < minimum_merge_progress:
                continue

            merge_progress = (
                progress_ahead[lane_index] + target_entry_dist
            )
            for route_index, route_distance in route_progress.items():
                if not (merge_progress <= route_distance <= merge_progress + 3.0 * ROUTE_STEP_METERS):
                    continue
                route_waypoint = route[route_index]
                if (
                    not is_driving_waypoint(route_waypoint)
                    or route_waypoint.is_junction
                    or route_waypoint.lane_id != current_waypoint.lane_id
                ):
                    continue

                merge_clearance = detour_segment_clearance(
                    lane_points[lane_index].transform.location,
                    route_waypoint.transform.location,
                    actors=nearby_actors,
                )
                if (
                    merge_clearance is None
                    or merge_clearance < BYPASS_MIN_CLEARANCE
                ):
                    continue

                path_points = (
                    [current_waypoint]
                    + lane_points[entry_index:lane_index + 1]
                    + [route_waypoint]
                )
                if opposing_lane and not opposing_lane_path_is_clear(
                    path_points,
                    traffic_predictions,
                ):
                    continue

                total_clearance = min(path_clearance, merge_clearance)
                candidate_length = route_distance
                candidate = (
                    (
                        round(min(total_clearance, 3.0) * 2.0) / 2.0,
                        -candidate_length,
                    ),
                    route_index,
                    path_points,
                    direction,
                )
                best_candidate = best_candidates[direction]
                if (
                    candidate_length <= DETOUR_MAX_DISTANCE_METERS
                    and (
                        best_candidate is None
                        or candidate[0] > best_candidate[0]
                    )
                ):
                    best_candidates[direction] = candidate
                break

    same_direction_candidates = [
        candidate
        for candidate in (best_candidates["LEFT"], best_candidates["RIGHT"])
        if candidate is not None
    ]
    best_candidate = max(
        same_direction_candidates,
        key=lambda candidate: candidate[0],
        default=None,
    )
    if best_candidate is None:
        return None
    return best_candidate[1], best_candidate[2], best_candidate[3]


def make_local_detour(obstacle):
    """Search reachable Driving waypoints and merge onto the active route."""
    global avoidance_merge_route_index
    global chosen_avoidance_direction

    map_obj = world.get_map()
    current_waypoint = map_obj.get_waypoint(
        ego.get_location(),
        project_to_road=False,
        lane_type=carla.LaneType.Driving,
    )
    if not is_driving_waypoint(current_waypoint):
        current_waypoint = map_obj.get_waypoint(
            ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if not is_driving_waypoint(current_waypoint):
            return False

    forward = current_waypoint.transform.get_forward_vector()
    origin = ego.get_location()
    obstacle_loc = obstacle.get_location()
    relative = obstacle_loc - origin
    obstacle_longitudinal = (
        relative.x * forward.x + relative.y * forward.y
    )
    if obstacle_longitudinal <= 0.0:
        return False

    if not active_route:
        return False

    obstacle_extent = obstacle.bounding_box.extent
    minimum_merge_progress = (
        obstacle_longitudinal
        + ego.bounding_box.extent.x
        + math.hypot(obstacle_extent.x, obstacle_extent.y)
        + MIN_STANDOFF_DISTANCE
    )
    route_goals = {}
    route_distance = 0.0
    previous_location = origin
    first_route_index = max(0, min(active_route_index, len(active_route) - 1))
    for route_index in range(first_route_index, len(active_route)):
        route_waypoint = active_route[route_index]
        route_location = route_waypoint.transform.location
        route_distance += distance(previous_location, route_location)
        previous_location = route_location
        if (
            minimum_merge_progress
            <= route_distance
            <= DETOUR_MAX_DISTANCE_METERS
            and is_driving_waypoint(route_waypoint)
        ):
            route_goals[waypoint_key(route_waypoint)] = (
                route_distance,
                route_index,
                route_waypoint,
            )
    if not route_goals:
        return False

    nearby_actors = nearby_blocking_objects(
        origin,
        DETOUR_MAX_DISTANCE_METERS,
    )
    start_clearance = detour_segment_clearance(
        origin,
        current_waypoint.transform.location,
        actors=nearby_actors,
    )
    if start_clearance is None:
        return False

    smooth_bypass = find_smooth_lane_change_bypass(
        map_obj,
        current_waypoint,
        origin,
        active_route,
        first_route_index,
        minimum_merge_progress,
        nearby_actors,
        obstacle_distance=obstacle_longitudinal,
    )
    if smooth_bypass is not None:
        avoidance_merge_route_index, path_points, chosen_dir = smooth_bypass
        chosen_avoidance_direction = chosen_dir
        avoidance_points.clear()
        avoidance_points.extend(path_points)
        return True

    adjacent_bypass = find_adjacent_lane_bypass(
        map_obj,
        current_waypoint,
        origin,
        active_route,
        first_route_index,
        minimum_merge_progress,
        nearby_actors,
    )
    if adjacent_bypass is not None:
        avoidance_merge_route_index, path_points, chosen_dir = adjacent_bypass
        chosen_avoidance_direction = chosen_dir
        avoidance_points.clear()
        avoidance_points.extend(path_points)
        return True

    start_key = waypoint_key(current_waypoint)
    clearance_to = {start_key: start_clearance}
    distance_to = {start_key: 0.0}
    waypoint_by_key = {start_key: current_waypoint}
    parent = {}
    order = itertools.count()
    frontier = [(-start_clearance, 0.0, next(order), start_key)]
    reachable_goals = set()
    expansions = 0

    while frontier and expansions < DETOUR_SEARCH_MAX_EXPANSIONS:
        negative_clearance, path_length, _, current_key = heapq.heappop(frontier)
        current_clearance = -negative_clearance
        if (
            current_clearance < clearance_to.get(current_key, float("-inf"))
            or path_length > distance_to.get(current_key, float("inf"))
        ):
            continue
        if current_key in route_goals:
            reachable_goals.add(current_key)
        expansions += 1

        current = waypoint_by_key[current_key]
        current_location = current.transform.location
        neighbors = [
            candidate
            for candidate in current.next(ROUTE_STEP_METERS)
            if is_driving_waypoint(candidate)
        ]
        for lane_method in ("get_left_lane", "get_right_lane"):
            adjacent = getattr(current, lane_method)()
            if is_driving_waypoint(adjacent):
                neighbors.append(adjacent)

        seen_neighbors = set()
        current_forward = current.transform.get_forward_vector()
        for candidate in neighbors:
            candidate_key = waypoint_key(candidate)
            if candidate_key == current_key or candidate_key in seen_neighbors:
                continue
            seen_neighbors.add(candidate_key)
            candidate_forward = candidate.transform.get_forward_vector()
            candidate_dot = (
                current_forward.x * candidate_forward.x
                + current_forward.y * candidate_forward.y
            )
            if candidate_dot < 0.5:
                continue
            candidate_location = candidate.transform.location
            if (
                distance(origin, candidate_location)
                > DETOUR_MAX_DISTANCE_METERS
            ):
                continue
            edge_clearance = detour_segment_clearance(
                current_location,
                candidate_location,
                actors=nearby_actors,
            )
            if edge_clearance is None:
                continue

            candidate_length = path_length + distance(
                current_location,
                candidate_location,
            )
            if candidate_length > DETOUR_MAX_DISTANCE_METERS:
                continue
            candidate_clearance = min(current_clearance, edge_clearance)
            previous_clearance = clearance_to.get(
                candidate_key,
                float("-inf"),
            )
            previous_length = distance_to.get(
                candidate_key,
                float("inf"),
            )
            if (
                candidate_clearance < previous_clearance
                or (
                    candidate_clearance == previous_clearance
                    and candidate_length >= previous_length
                )
            ):
                continue
            clearance_to[candidate_key] = candidate_clearance
            distance_to[candidate_key] = candidate_length
            waypoint_by_key[candidate_key] = candidate
            parent[candidate_key] = current_key
            heapq.heappush(
                frontier,
                (
                    -candidate_clearance,
                    candidate_length,
                    next(order),
                    candidate_key,
                ),
            )

    if not reachable_goals:
        return False

    reached_key = None
    path_points = None
    ordered_goals = sorted(
        reachable_goals,
        key=lambda key: (
            route_goals[key][0],
            -clearance_to[key],
            distance_to[key],
        ),
    )
    for candidate_goal in ordered_goals:
        candidate_path_keys = [candidate_goal]
        while candidate_path_keys[-1] != start_key:
            parent_key = parent.get(candidate_path_keys[-1])
            if parent_key is None:
                break
            candidate_path_keys.append(parent_key)
        if candidate_path_keys[-1] != start_key:
            continue
        candidate_path_keys.reverse()
        candidate_path = [
            waypoint_by_key[key] for key in candidate_path_keys
        ]
        lane_change_index = next(
            (
                index
                for index, waypoint in enumerate(candidate_path[1:], start=1)
                if waypoint.lane_id != current_waypoint.lane_id
            ),
            None,
        )
        if lane_change_index is None:
            continue
        lane_change_source = candidate_path[lane_change_index - 1]
        lane_change_target = candidate_path[lane_change_index]
        reached_key = candidate_goal
        path_points = candidate_path
        break
    if reached_key is None or path_points is None:
        return False

    lane_change_index = next(
        (
            index
            for index, waypoint in enumerate(path_points[1:], start=1)
            if waypoint.lane_id != current_waypoint.lane_id
        ),
        None,
    )
    if lane_change_index is not None:
        lane_change_waypoint = path_points[lane_change_index]
        adjacent_start = None
        for lane_method in ("get_left_lane", "get_right_lane"):
            adjacent = getattr(current_waypoint, lane_method)()
            if (
                is_driving_waypoint(adjacent)
                and adjacent.lane_id == lane_change_waypoint.lane_id
            ):
                adjacent_start = adjacent
                break
        if adjacent_start is not None:
            shortcut_points = [current_waypoint, adjacent_start]
            if distance(
                adjacent_start.transform.location,
                lane_change_waypoint.transform.location,
            ) >= 1.0:
                shortcut_points.append(lane_change_waypoint)
            shortcut_points.extend(path_points[lane_change_index + 1:])
            shortcut_is_safe = all(
                is_driving_segment(
                    map_obj,
                    first.transform.location,
                    second.transform.location,
                )
                and detour_segment_clearance(
                    first.transform.location,
                    second.transform.location,
                    actors=nearby_actors,
                ) is not None
                for first, second in zip(
                    shortcut_points,
                    shortcut_points[1:],
                )
            )
            if shortcut_is_safe:
                path_points = shortcut_points

    for first, second in zip(path_points, path_points[1:]):
        if (
            not is_driving_waypoint(first)
            or not is_driving_waypoint(second)
            or not is_driving_segment(
                map_obj,
                first.transform.location,
                second.transform.location,
            )
        ):
            return False

    avoidance_merge_route_index = route_goals[reached_key][1]
    avoidance_points.clear()
    avoidance_points.extend(path_points)
    chosen_avoidance_direction = (
        "RIGHT"
        if is_physical_right_lane_change(
            lane_change_source, lane_change_target
        )
        else "LEFT"
    )
    return True


def start_obstacle_avoidance(obstacle, obstacle_distance):
    """Start a temporary bypass instead of stopping in front of the object."""
    global avoidance_active
    global avoidance_point_index
    global avoidance_obstacle_id
    global avoidance_last_progress_time
    global avoidance_last_progress_location
    global avoidance_blocked_replan_time
    global blocked_obstacle_id
    global blocked_obstacle_retry_attempted
    global chosen_avoidance_direction

    if obstacle is None:
        return False

    if make_local_detour(obstacle):
        blocked_obstacle_id = None
        blocked_obstacle_retry_attempted = False
        avoidance_active = True
        avoidance_point_index = 0
        avoidance_obstacle_id = obstacle.id
        avoidance_last_progress_time = time.monotonic()
        avoidance_last_progress_location = ego.get_location()
        avoidance_blocked_replan_time = 0.0

        print(
            f"\n[AVOIDANCE] Obstacle detected at {obstacle_distance:.1f} m "
            f"-> Steering {chosen_avoidance_direction} into adjacent driving lane"
        )
        return True

    print(
        f"\n[AVOIDANCE] Obstacle at {obstacle_distance:.1f} m "
        "-> no safe Driving-lane bypass found yet"
    )
    return False


def avoidance_path_status(current, points, point_index, nearby_actors):
    """Validate remaining bypass lane segments and report blocked geometry."""
    path_start = current
    path_blocked = False
    for point in points[point_index:]:
        if not is_driving_waypoint(point):
            return "AVOIDANCE_TARGET_NOT_DRIVING", path_blocked
        segment_end = point.transform.location
        if not is_driving_segment(
            world.get_map(),
            path_start,
            segment_end,
        ):
            return "AVOIDANCE_NON_DRIVING_SEGMENT", path_blocked
        if detour_segment_clearance(
            path_start,
            segment_end,
            actors=nearby_actors,
        ) is None:
            path_blocked = True
        path_start = segment_end
    return None, path_blocked


def obstacle_is_directly_ahead(obstacle):
    """Distinguish a forward-lane hazard from an object beside the vehicle."""
    if obstacle is None:
        return False

    ego_transform = ego.get_transform()
    forward = ego_transform.get_forward_vector()
    right = right_vector(ego_transform)
    actor_transform = obstacle.get_transform()
    box_location = obstacle.bounding_box.location
    center = actor_transform.transform(
        carla.Location(
            x=box_location.x,
            y=box_location.y,
            z=box_location.z,
        )
    )
    relative = center - ego_transform.location
    longitudinal = relative.x * forward.x + relative.y * forward.y
    if longitudinal <= 0.0:
        return False

    actor_forward = actor_transform.get_forward_vector()
    actor_forward_length = math.hypot(actor_forward.x, actor_forward.y)
    if actor_forward_length < 1e-6:
        return False
    actor_forward = carla.Vector3D(
        actor_forward.x / actor_forward_length,
        actor_forward.y / actor_forward_length,
        0.0,
    )
    actor_right = carla.Vector3D(-actor_forward.y, actor_forward.x, 0.0)
    actor_extent = obstacle.bounding_box.extent
    actor_lateral_extent = (
        actor_extent.x * abs(
            actor_forward.x * right.x + actor_forward.y * right.y
        )
        + actor_extent.y * abs(
            actor_right.x * right.x + actor_right.y * right.y
        )
    )
    lateral = abs(relative.x * right.x + relative.y * right.y)
    lateral_clearance = (
        lateral
        - ego.bounding_box.extent.y
        - actor_lateral_extent
    )
    return lateral_clearance <= 0.35


def obstacle_needs_detour(obstacle, directly_ahead):
    """Plan a bypass when an ahead obstacle enters the vehicle's safety margin."""
    if directly_ahead or obstacle is None:
        return directly_ahead

    ego_transform = ego.get_transform()
    forward = ego_transform.get_forward_vector()
    right = right_vector(ego_transform)
    actor_transform = obstacle.get_transform()
    actor_box = obstacle.bounding_box
    center = actor_transform.transform(
        carla.Location(
            x=actor_box.location.x,
            y=actor_box.location.y,
            z=actor_box.location.z,
        )
    )
    relative = center - ego_transform.location
    longitudinal = relative.x * forward.x + relative.y * forward.y
    if longitudinal <= 0.0:
        return False

    actor_forward = actor_transform.get_forward_vector()
    actor_forward_length = math.hypot(actor_forward.x, actor_forward.y)
    if actor_forward_length < 1e-6:
        return False
    actor_forward = carla.Vector3D(
        actor_forward.x / actor_forward_length,
        actor_forward.y / actor_forward_length,
        0.0,
    )
    actor_right = carla.Vector3D(-actor_forward.y, actor_forward.x, 0.0)
    lateral = abs(
        relative.x * right.x + relative.y * right.y
    )
    actor_lateral_extent = (
        actor_box.extent.x * abs(
            actor_forward.x * right.x + actor_forward.y * right.y
        )
        + actor_box.extent.y * abs(
            actor_right.x * right.x + actor_right.y * right.y
        )
    )
    lateral_clearance = (
        lateral
        - ego.bounding_box.extent.y
        - actor_lateral_extent
    )
    return lateral_clearance <= 1.25


def nearest_pedestrian_clearance():
    """Estimate clearance to a pedestrian in or approaching the ego corridor."""
    if ego is None:
        return None

    transform = ego.get_transform()
    origin = transform.location
    forward = transform.get_forward_vector()
    right = right_vector(transform)
    ego_half_width = ego.bounding_box.extent.y
    ego_half_length = ego.bounding_box.extent.x
    best_clearance = None
    for walker in world.get_actors().filter("walker.pedestrian.*"):
        location = walker.get_location()
        relative_x = location.x - origin.x
        relative_y = location.y - origin.y
        longitudinal = relative_x * forward.x + relative_y * forward.y
        if longitudinal <= -ego_half_length or longitudinal > PEDESTRIAN_LOOKAHEAD:
            continue

        lateral = relative_x * right.x + relative_y * right.y
        velocity = walker.get_velocity()
        walker_half_width = max(
            walker.bounding_box.extent.x,
            walker.bounding_box.extent.y,
        )
        walker_half_length = walker_half_width
        corridor_half_width = (
            ego_half_width + walker_half_width + PEDESTRIAN_SIDE_MARGIN
        )
        lateral_speed = velocity.x * right.x + velocity.y * right.y
        longitudinal_speed = velocity.x * forward.x + velocity.y * forward.y
        prediction_steps = math.ceil(
            PEDESTRIAN_PREDICTION_HORIZON_SECONDS
            / PEDESTRIAN_PREDICTION_STEP_SECONDS
        )
        for step in range(prediction_steps + 1):
            elapsed = step * PEDESTRIAN_PREDICTION_STEP_SECONDS
            predicted_lateral = lateral + lateral_speed * elapsed
            predicted_longitudinal = (
                longitudinal + longitudinal_speed * elapsed
            )
            if (
                abs(predicted_lateral) > corridor_half_width
                or predicted_longitudinal <= -ego_half_length
            ):
                continue

            clearance = max(
                0.0,
                predicted_longitudinal
                - ego_half_length
                - walker_half_length
                - 0.5,
            )
            if best_clearance is None or clearance < best_clearance:
                best_clearance = clearance
            break
    return best_clearance


def pedestrian_stop_distance(current_speed):
    """Include a fixed crossing buffer and speed-dependent stopping distance."""
    speed_mps = max(0.0, current_speed) / 3.6
    return max(
        PEDESTRIAN_STOP_BUFFER_M,
        STOP_DISTANCE
        + speed_mps * REACTION_TIME
        + speed_mps * speed_mps / (2.0 * SAFE_DECELERATION),
    )


def avoidance_steering(target):
    """Track opposing-lane pass points in travel direction, not lane heading."""
    if chosen_avoidance_direction != "LEFT-ONCOMING":
        # Keep steering gentle so the vehicle tracks the adjacent lane
        # instead of sharply turning off the road.
        return clamp(calculate_steering(target, gain=1.0), -0.22, 0.22)

    lookahead_distance = 12.0
    previous = ego.get_location()
    for waypoint in avoidance_points[avoidance_point_index:]:
        next_location = waypoint.transform.location
        segment_length = distance(previous, next_location)
        if segment_length <= 1e-6:
            continue
        if segment_length >= lookahead_distance:
            fraction = lookahead_distance / segment_length
            lookahead = carla.Location(
                x=previous.x + (next_location.x - previous.x) * fraction,
                y=previous.y + (next_location.y - previous.y) * fraction,
                z=previous.z + (next_location.z - previous.z) * fraction,
            )
            return calculate_steering(lookahead, gain=0.6)
        lookahead_distance -= segment_length
        previous = next_location

    return calculate_steering(previous, gain=0.6)


def avoidance_control():
    """Follow a validated bypass; any lost clearance causes a full stop."""
    global avoidance_active
    global avoidance_point_index
    global avoidance_obstacle_id
    global active_route_index
    global avoidance_merge_route_index
    global avoidance_last_progress_time
    global avoidance_last_progress_location
    global avoidance_blocked_replan_time
    global chosen_avoidance_direction

    if not avoidance_active or not avoidance_points:
        return None

    if avoidance_point_index >= len(avoidance_points):
        avoidance_active = False
        avoidance_points.clear()
        if avoidance_merge_route_index is not None:
            active_route_index = max(
                active_route_index,
                avoidance_merge_route_index,
            )
        avoidance_merge_route_index = None
        chosen_avoidance_direction = "NONE"
        print("Obstacle passed -> MERGING BACK TO MAIN ROUTE")
        return None

    target = avoidance_points[min(
        avoidance_point_index,
        len(avoidance_points) - 1
    )]
    if not is_driving_waypoint(target):
        return carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0,
        ), "AVOIDANCE_TARGET_NOT_DRIVING"

    current = ego.get_location()
    now = time.monotonic()
    if (
        avoidance_last_progress_location is None
        or distance(current, avoidance_last_progress_location) >= 1.0
    ):
        avoidance_last_progress_location = current
        avoidance_last_progress_time = now

    if now - avoidance_last_progress_time > AVOIDANCE_TIMEOUT:
        obstacle, obstacle_distance = find_forward_obstacle(
            max_distance=AVOID_DETECT_DISTANCE * 2.0,
        )
        if obstacle is not None and make_local_detour(obstacle):
            avoidance_point_index = 0
            avoidance_obstacle_id = obstacle.id
            avoidance_last_progress_time = now
            avoidance_last_progress_location = current
            print(
                "Obstacle avoidance made no progress; replanned a new "
                f"Driving-lane bypass for actor {obstacle.id} "
                f"at {obstacle_distance:.1f} m."
            )
            target = avoidance_points[avoidance_point_index]
        else:
            if obstacle is not None:
                print(
                    "Obstacle avoidance made no progress and no alternate "
                    f"Driving-lane bypass is safe (actor {obstacle.id}, "
                    f"{obstacle_distance:.1f} m); applying safety stop."
                )
            else:
                print(
                    "Obstacle avoidance made no progress and no blocking "
                    "actor remains; returning to route following."
                )
            avoidance_active = False
            avoidance_points.clear()
            avoidance_obstacle_id = None
            avoidance_merge_route_index = None
            chosen_avoidance_direction = "NONE"
            return None

    d = distance(current, target.transform.location)

    if d < 2.5:
        avoidance_point_index += 1
        avoidance_last_progress_time = now
        avoidance_last_progress_location = current

        if avoidance_point_index >= len(avoidance_points):
            avoidance_active = False
            avoidance_points.clear()
            if avoidance_merge_route_index is not None:
                active_route_index = max(
                    active_route_index,
                    avoidance_merge_route_index,
                )
            avoidance_merge_route_index = None
            chosen_avoidance_direction = "NONE"
            print("Obstacle passed -> MERGING BACK TO MAIN ROUTE")
            return None

        target = avoidance_points[avoidance_point_index]

    nearby_actors = nearby_blocking_objects(
        current,
        DETOUR_MAX_DISTANCE_METERS,
    )
    path_error, path_blocked_by_actor = avoidance_path_status(
        current,
        avoidance_points,
        avoidance_point_index,
        nearby_actors,
    )
    if path_error is not None:
        return carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0,
        ), path_error

    obstacle, obstacle_distance = find_forward_obstacle(max_distance=30.0)
    obstacle_ahead = obstacle_is_directly_ahead(obstacle)
    if (
        path_blocked_by_actor
        and obstacle is not None
        and now - avoidance_blocked_replan_time >= AVOID_RETRY_SECONDS
    ):
        avoidance_blocked_replan_time = now
        blocked_actor_id = obstacle.id
        if make_local_detour(obstacle):
            avoidance_point_index = 0
            avoidance_obstacle_id = obstacle.id
            avoidance_last_progress_time = now
            avoidance_last_progress_location = current
            target = avoidance_points[avoidance_point_index]
            nearby_actors = nearby_blocking_objects(
                current,
                DETOUR_MAX_DISTANCE_METERS,
            )
            path_error, path_blocked_by_actor = avoidance_path_status(
                current,
                avoidance_points,
                avoidance_point_index,
                nearby_actors,
            )
            obstacle, obstacle_distance = find_forward_obstacle(
                max_distance=30.0,
            )
            obstacle_ahead = obstacle_is_directly_ahead(obstacle)
            print(
                "Blocked bypass detected; replanned around actor "
                f"{blocked_actor_id}."
            )
            if path_error is not None:
                return carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    steer=0.0,
                ), path_error
        else:
            print(
                "Blocked bypass has no clear alternate Driving-lane path; "
                "holding position until one is available."
            )

    # The first waypoint can still be on the ego's current lane. Aim at the
    # first validated waypoint in the adjacent lane so the lane-change begins
    # with a non-zero steering request instead of following the obstacle lane.
    try:
        ego_location_for_lane = ego.get_location()
        current_wp = world.get_map().get_waypoint(
            ego_location_for_lane,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if current_wp is not None:
            for candidate in avoidance_points[avoidance_point_index:]:
                if (
                    is_driving_waypoint(candidate)
                    and candidate.lane_id != current_wp.lane_id
                ):
                    target = candidate
                    break
    except (RuntimeError, AttributeError, TypeError):
        pass

    steer = avoidance_steering(target)
    current_speed = speed_kmh(ego)
    pedestrian_clearance = nearest_pedestrian_clearance()
    if (
        pedestrian_clearance is not None
        and pedestrian_clearance <= pedestrian_stop_distance(current_speed)
    ):
        return carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0,
        ), "PEDESTRIAN_STOP"

    bypassing_the_detected_actor = (
        avoidance_active
        and bool(avoidance_points)
        and obstacle is not None
        and obstacle.id == avoidance_obstacle_id
        and chosen_avoidance_direction in ("LEFT", "RIGHT", "LEFT-ONCOMING")
    )

    if (
        obstacle is not None
        and obstacle_ahead
        and obstacle_distance <= emergency_stop_clearance_m(current_speed)
        and not bypassing_the_detected_actor
    ):
        return carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0,
        ), "AVOIDANCE_EMERGENCY_STOP"

    if path_blocked_by_actor:
        # The planned lane-change path can briefly share the original lane
        # corridor while the ego starts moving laterally. Permit a slow start
        # only for the exact actor being bypassed and only with a validated
        # Driving waypoint and meaningful steering. Other blockage still stops.
        if (
            bypassing_the_detected_actor
            and obstacle_ahead
            and obstacle_distance > max(2.5, STOP_DISTANCE)
            and is_driving_waypoint(target)
            and abs(steer) >= 0.05
        ):
            return carla.VehicleControl(
                throttle=0.12,
                brake=0.0,
                steer=max(-0.45, min(0.45, steer)),
            ), "AVOIDANCE_LANE_CHANGE_INIT"
        return carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0,
        ), "AVOIDANCE_PATH_BLOCKED_WAITING_FOR_CLEARANCE"

    target_speed = AVOIDANCE_SPEED
    if pedestrian_clearance is not None:
        target_speed = min(
            target_speed,
            obstacle_target_speed(pedestrian_clearance),
        )
    if obstacle is not None and not obstacle_ahead:
        throttle, brake = speed_control_without_braking(
            current_speed,
            target_speed,
        )
    else:
        throttle, brake = speed_control(current_speed, target_speed)

    return carla.VehicleControl(
        throttle=throttle,
        brake=brake,
        steer=steer
    ), "AVOIDING_OBSTACLE"

def calculate_steering(target_wp, gain=1.0):
    """Steer toward a route waypoint or a local detour point."""

    vehicle_tf = ego.get_transform()
    vehicle_loc = vehicle_tf.location

    vehicle_forward = vehicle_tf.get_forward_vector()
    if hasattr(target_wp, "transform"):
        wp_tf = target_wp.transform
        wp_loc = wp_tf.location
        dx = wp_loc.x - vehicle_loc.x
        dy = wp_loc.y - vehicle_loc.y
        target_distance = math.hypot(dx, dy)
        if target_distance < 0.1:
            return 0.0
        dot = vehicle_forward.x * dx + vehicle_forward.y * dy
        if dot <= 0.5:
            return 0.0
        cross = vehicle_forward.x * dy - vehicle_forward.y * dx
        target_angle = math.atan2(cross, dot)
        steering = math.atan2(
            2.7 * math.sin(target_angle),
            target_distance,
        )
    else:
        dx = target_wp.x - vehicle_loc.x
        dy = target_wp.y - vehicle_loc.y
        target_length = math.hypot(dx, dy)
        if target_length < 0.1:
            return 0.0
        dot = vehicle_forward.x * dx + vehicle_forward.y * dy
        cross = vehicle_forward.x * dy - vehicle_forward.y * dx
        steering = math.atan2(cross, dot)

    return clamp(steering * gain, -0.35, 0.35)


def stopping_distance_m(speed):
    speed_mps = max(0.0, speed) / 3.6
    return (
        speed_mps * REACTION_TIME
        + speed_mps * speed_mps / (2.0 * SAFE_DECELERATION)
    )


def emergency_stop_clearance_m(speed):
    return max(
        STOP_DISTANCE,
        stopping_distance_m(speed) + MIN_STANDOFF_DISTANCE,
    )


def obstacle_target_speed(clearance):
    usable_clearance = clearance - MIN_STANDOFF_DISTANCE
    if usable_clearance <= 0.0:
        return 0.0

    deceleration = max(0.1, SAFE_DECELERATION)
    reaction_time = max(0.0, REACTION_TIME)
    speed_mps = (
        math.sqrt(
            (deceleration * reaction_time) ** 2
            + 2.0 * deceleration * usable_clearance
        )
        - deceleration * reaction_time
    )
    return min(CRUISE_SPEED, max(0.0, speed_mps * 3.6))


def speed_control(current_speed, target_speed):
    if target_speed <= 0.1:
        return 0.0, 0.0 if current_speed <= 0.5 else 0.35
    if current_speed < target_speed - 1.0:
        throttle = clamp((target_speed - current_speed) * 0.025, 0.08, 0.35)
        return throttle, 0.0
    if current_speed > target_speed + 1.0:
        brake = clamp((current_speed - target_speed) * 0.06, 0.12, 0.7)
        return 0.0, brake
    return 0.08, 0.0


def speed_control_without_braking(current_speed, target_speed):
    """Reduce speed by coasting for beside-the-vehicle hazards."""
    if current_speed > target_speed + 1.0:
        return 0.0, 0.0
    return speed_control(current_speed, target_speed)


def estimate_risk(obstacle, obstacle_distance, current_speed, obstacle_ahead=True):
    """
    Estimate driving risk level based on obstacle distance, relative position,
    and vehicle speed.

    Risk levels:
      - LOW: continue at normal cruising speed.
      - MEDIUM: reduce throttle and slow down smoothly.
      - HIGH: slow down significantly and prepare to avoid obstacle.
      - CRITICAL: emergency braking (collision imminent or no safe path at stop distance).
    """
    if collision_state.get("hit", False):
        if (
            time.monotonic() - collision_state["last_time"]
            < COLLISION_BRAKE_HOLD_SECONDS
        ):
            return "CRITICAL", 0.0

    pedestrian_clearance = nearest_pedestrian_clearance()
    if (
        pedestrian_clearance is not None
        and pedestrian_clearance <= pedestrian_stop_distance(current_speed)
    ):
        return "CRITICAL", 0.0

    if obstacle is None or obstacle_distance >= float("inf"):
        return "LOW", CRUISE_SPEED

    if not obstacle_ahead:
        if obstacle_distance <= 4.0:
            return "MEDIUM", (CRUISE_SPEED + AVOIDANCE_SPEED) / 2.0
        return "LOW", CRUISE_SPEED

    speed_mps = max(0.0, current_speed) / 3.6
    ttc = (obstacle_distance / speed_mps) if speed_mps > 0.5 else 999.0

    # CRITICAL: within the conservative stopping margin or closing too quickly
    if (
        obstacle_distance <= emergency_stop_clearance_m(current_speed)
        or (ttc < 1.5 and obstacle_distance < 8.0)
    ):
        return "CRITICAL", 0.0

    # HIGH: Obstacle ahead within 15.0m or (TTC < 3.0s and obstacle_distance <= 20.0)
    if obstacle_distance <= 15.0 or (ttc < 3.0 and obstacle_distance <= 20.0):
        target_speed = min(AVOIDANCE_SPEED, obstacle_target_speed(obstacle_distance))
        return "HIGH", target_speed

    # MEDIUM: Obstacle ahead within 25.0m or (TTC < 4.5s and obstacle_distance <= 30.0)
    if obstacle_distance <= 25.0 or (ttc < 4.5 and obstacle_distance <= 30.0):
        target_speed = min(
            (CRUISE_SPEED + AVOIDANCE_SPEED) / 2.0,
            obstacle_target_speed(obstacle_distance),
        )
        return "MEDIUM", target_speed

    # LOW: Obstacle is far away (> 25.0m)
    return "LOW", CRUISE_SPEED


def calculate_control(target):
    global avoidance_active
    global avoidance_obstacle_id
    global avoidance_points
    global route_recovery_active
    global blocked_obstacle_id
    global blocked_obstacle_retry_attempted
    global avoidance_search_obstacle_id
    global avoidance_search_stop_retry_attempted
    global avoidance_blocked_replan_time
    global bypass_last_search_time
    global chosen_avoidance_direction
    global current_risk_level
    global current_target_speed

    current_speed = speed_kmh(ego)
    obstacle, obstacle_distance = find_forward_obstacle(max_distance=30.0)
    obstacle_ahead = obstacle_is_directly_ahead(obstacle)
    obstacle_needs_bypass = obstacle_needs_detour(
        obstacle,
        obstacle_ahead,
    )

    # 1. Collision emergency brake check
    if collision_state["hit"]:
        if (
            time.monotonic() - collision_state["last_time"]
            < COLLISION_BRAKE_HOLD_SECONDS
        ):
            is_pedestrian = (
                obstacle is not None
                and obstacle.type_id.lower().startswith("walker.pedestrian")
            )
            if (
                not avoidance_active
                and obstacle is not None
                and obstacle_needs_bypass
                and not is_pedestrian
                and obstacle_distance <= AVOID_DETECT_DISTANCE
            ):
                start_obstacle_avoidance(obstacle, obstacle_distance)

            current_risk_level = "CRITICAL"
            current_target_speed = 0.0
            steering_target = (
                avoidance_points[avoidance_point_index]
                if avoidance_active
                and avoidance_points
                and avoidance_point_index < len(avoidance_points)
                else target
            )
            steer = (
                avoidance_steering(steering_target)
                if avoidance_active and is_driving_waypoint(steering_target)
                else calculate_steering(steering_target, gain=1.15)
                if is_driving_waypoint(steering_target)
                else 0.0
            )
            return carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=steer,
            ), "COLLISION_BRAKE"

        collision_state["hit"] = False
        print(
            "Collision brake hold elapsed; rechecking the route and "
            "obstacles before continuing."
        )

    # 2. Pedestrian emergency stop check
    pedestrian_clearance = nearest_pedestrian_clearance()
    if (
        pedestrian_clearance is not None
        and pedestrian_clearance <= pedestrian_stop_distance(current_speed)
    ):
        current_risk_level = "CRITICAL"
        current_target_speed = 0.0
        return carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0,
        ), "PEDESTRIAN_STOP"

    # 3. Active avoidance execution
    if avoidance_active:
        if (
            obstacle is not None
            and obstacle.id != avoidance_obstacle_id
            and obstacle_distance <= STOP_DISTANCE
            and obstacle_ahead
        ):
            avoidance_active = False
            avoidance_points.clear()
            avoidance_obstacle_id = None
            chosen_avoidance_direction = "NONE"
        else:
            result = avoidance_control()
            if result is not None:
                current_risk_level = "HIGH"
                current_target_speed = AVOIDANCE_SPEED
                return result
            # Detour ended -> merged back to main route
            avoidance_active = False
            chosen_avoidance_direction = "NONE"

    # 4. Route target driving lane validation
    if not is_driving_waypoint(target):
        current_risk_level = "CRITICAL"
        current_target_speed = 0.0
        return carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0,
        ), (
            "ROUTE_RECOVERY_NOT_FOUND"
            if target is None
            else "ROUTE_TARGET_NOT_DRIVING"
        )

    # 5. Calculate Risk Level and Target Speed
    risk_level, risk_target_speed = estimate_risk(
        obstacle,
        obstacle_distance,
        current_speed,
        obstacle_ahead=obstacle_ahead,
    )
    current_risk_level = risk_level
    current_target_speed = risk_target_speed

    is_pedestrian = (
        obstacle is not None
        and obstacle.type_id.lower().startswith("walker.pedestrian")
    )

    # 6. Plan a validated bypass before a nearby obstacle becomes critical.
    now = time.monotonic()
    if (
        obstacle is not None
        and obstacle_needs_bypass
        and not is_pedestrian
        and obstacle_distance <= AVOID_DETECT_DISTANCE
        and (
            risk_level == "HIGH"
            or obstacle_distance <= AVOIDANCE_EMERGENCY_BYPASS_DISTANCE
        )
        and (
            (now - bypass_last_search_time >= AVOID_RETRY_SECONDS)
            or (obstacle.id != avoidance_search_obstacle_id)
            or obstacle_distance <= AVOIDANCE_EMERGENCY_BYPASS_DISTANCE
        )
    ):
        bypass_last_search_time = now
        avoidance_search_obstacle_id = obstacle.id
        print(
            f"[AVOIDANCE] Planning bypass for actor {obstacle.id} "
            f"at {obstacle_distance:.1f} m (Risk={risk_level})"
        )
        if start_obstacle_avoidance(obstacle, obstacle_distance):
            blocked_obstacle_id = None
            blocked_obstacle_retry_attempted = False
            result = avoidance_control()
            if result is not None:
                return result

    # 7. Imminent collision or close obstacle without bypass -> emergency braking
    if (
        risk_level == "CRITICAL"
        or (
            obstacle is not None
            and obstacle_ahead
            and obstacle_distance
            <= max(STOP_DISTANCE, AVOIDANCE_EMERGENCY_BYPASS_DISTANCE)
        )
    ):
        if obstacle is not None:
            blocked_obstacle_id = obstacle.id
        return carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0,
        ), "WAITING_FOR_BLOCKED_OBSTACLE"

    # Clear blocked obstacle tracking when path ahead is clear
    if (
        blocked_obstacle_id is not None
        and (
            obstacle is None
            or obstacle.id != blocked_obstacle_id
            or obstacle_distance > STOP_DISTANCE
            or not obstacle_ahead
        )
    ):
        blocked_obstacle_id = None
        blocked_obstacle_retry_attempted = False

    # 8. Adjust speed based on risk
    target_speed = risk_target_speed
    if route_recovery_active:
        target_speed = min(target_speed, AVOIDANCE_SPEED)

    if obstacle is not None and not obstacle_ahead:
        throttle, brake = speed_control_without_braking(
            current_speed,
            target_speed,
        )
        state = "OBSTACLE_SLOWDOWN_BESIDE_NO_BRAKE"
    else:
        throttle, brake = speed_control(current_speed, target_speed)
        if risk_level == "HIGH":
            state = "PREPARE_AVOIDANCE"
        elif risk_level == "MEDIUM":
            state = "OBSTACLE_SLOWDOWN"
        elif route_recovery_active:
            state = "ROUTE_RECOVERY"
        else:
            state = "CRUISING"

    steer = calculate_steering(target, gain=1.15)

    return carla.VehicleControl(
        throttle=throttle,
        brake=brake,
        steer=steer
    ), state

# ============================================================
# SPECTATOR
# ============================================================

def follow_vehicle():

    spectator = world.get_spectator()

    transform = ego.get_transform()
    forward = transform.get_forward_vector()

    camera_location = (
        transform.location
        - forward*10
        + carla.Location(z=5)
    )

    spectator.set_transform(
        carla.Transform(
            camera_location,
            carla.Rotation(
                pitch=-15,
                yaw=transform.rotation.yaw,
                roll=0
            )
        )
    )


# ============================================================
# ENVIRONMENT CREATION
# ============================================================

def create_environment(route):
    global ego
    collision_state["hit"] = False
    ego = spawn_ego(route[0], route)
    add_rocks(route)
    add_debris(route)
    add_cones(route)
    add_barriers(route)
    add_parked_vehicles(route)
    add_route_obstacles(route)
    add_moving_traffic(route)
    add_pothole_markers(route)
    add_road_pedestrians(route)
    attach_collision_sensor()
    follow_vehicle()
    print("\n"+"="*70)
    print("OSM-DIRECTIONAL UNSTRUCTURED CARLA ROUTE")
    print("="*70)
    print("Generated OpenDRIVE road    : YES (load before starting controller)")
    print("Physical rough-road dips     : generated in XODR elevation profile")
    print("Isolated pothole cavities   : NOT provided by XODR elevation; requires custom road mesh")
    print("Buildings / trees / terrain : depends on generated-map content")
    print("Roadside rocks/debris       : disabled by default")
    print("Construction objects        : disabled by default")
    print("Stationary route hazards     : disabled")
    print(
        f"Moving traffic request       : {NUM_MOVING_SCOOTERS} scooters/"
        f"fallback cars, {NUM_MOVING_VANS} vans"
    )
    print(
        f"Road pedestrians             : up to {NUM_PEDESTRIANS} "
        "crossing left-to-right across the full road"
    )
    print(f"Parked vehicles              : {NUM_PARKED} requested, both shoulders")
    print(f"Obstacle bypass trigger      : {AVOIDANCE_EMERGENCY_BYPASS_DISTANCE:.1f} m")
    print("Obstacle bypass            : safe same-direction LEFT or RIGHT lane")
    print("Lane following              : CARLA Driving-waypoint graph")
    print("Potholes                     : visual tire proxies only; not road depressions")
    print("RGB / depth / LiDAR          : disabled; this controller uses CARLA actor queries")
    print("Collision sensor             : YES (lightweight CARLA sensor)")
    print("OSM GPS is used only for compass direction; CARLA XY is not GPS-matched.")
    print("Route and avoidance targets are CARLA Driving-lane waypoints.")
    print("NOTE: Town10HD baked lane markings are not removed by Python.")
    print("="*70)

# ============================================================
# AUTONOMOUS LOOP
# ============================================================

def autonomous_loop(route):
    global avoidance_active
    global avoidance_points
    global destination_location
    global active_route
    global active_route_index
    global blocked_obstacle_id
    global blocked_obstacle_retry_attempted
    global avoidance_search_obstacle_id
    global avoidance_search_stop_retry_attempted
    global bypass_last_search_time
    global avoidance_merge_route_index
    global route_recovery_active
    global route_recovery_points
    global route_recovery_point_index
    global route_recovery_merge_index
    global pathweaver_bridge
    global pathweaver_original_world_settings
    active_route = route
    active_route_index = ego_spawn_route_index
    avoidance_active = False
    avoidance_points.clear()
    blocked_obstacle_id = None
    blocked_obstacle_retry_attempted = False
    avoidance_search_obstacle_id = None
    avoidance_search_stop_retry_attempted = False
    avoidance_blocked_replan_time = 0.0
    bypass_last_search_time = 0.0
    avoidance_merge_route_index = None
    route_recovery_active = False
    route_recovery_points.clear()
    route_recovery_point_index = 0
    route_recovery_merge_index = None
    destination_location = route[-1].transform.location
    travelled = 0.0
    previous = ego.get_location()
    route_index = ego_spawn_route_index
    best_route_index = route_index
    route_extension_attempted_for_end = None
    step = 0
    destination_distance = distance(previous, destination_location)
    last_progress_time = time.monotonic()
    last_progress_location = previous
    drive_started = time.monotonic()
    pathweaver_plan = None
    pathweaver_anchor_transform = None

    if PATHWEAVER_ENABLED:
        repository_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "integrations",
            "PathWeaver",
        )
        pathweaver_bridge = PathWeaverBridge(repository_path)
        pathweaver_bridge.start()
        pathweaver_original_world_settings = world.get_settings()
        synchronous_settings = world.get_settings()
        synchronous_settings.synchronous_mode = True
        synchronous_settings.fixed_delta_seconds = 0.05
        world.apply_settings(synchronous_settings)
        traffic_manager.set_synchronous_mode(True)
        print(
            "PathWeaver enabled: MATLAB Engine active; CARLA is stepping "
            "synchronously at 0.05 s."
        )

    while ego.is_alive:
        current = ego.get_location()
        destination_distance = distance(current, destination_location)
        route_index = max(route_index, active_route_index)
        while (
            route_index < len(route) - 1
            and distance(current, route[route_index].transform.location) < 4.0
        ):
            route_index += 1
        active_route_index = route_index
        if (
            len(route) - 1 - route_index
            <= max(1, math.ceil(100.0 / ROUTE_STEP_METERS))
            and route_extension_attempted_for_end != len(route)
        ):
            route_extension_attempted_for_end = len(route)
            continuation = find_verified_route_continuation(
                world.get_map(),
                route[-1],
            )
            if continuation is not None:
                route.extend(continuation[1:])
                destination_location = route[-1].transform.location
                print(
                    "Route segment extended before its endpoint using a "
                    "verified connected Driving-lane cycle "
                    f"({len(continuation) - 1} new waypoints)."
                )
        if (
            route_index >= len(route) - 1
            and destination_distance <= DESTINATION_REACHED_METERS
        ):
            print(
                "Route endpoint reached without a verified closed "
                "Driving-lane continuation; applying safety stop."
            )
            break
        if time.monotonic() - drive_started > 1800.0:
            print("Drive timeout -> applying safety stop.")
            break
        segment = distance(current, previous)
        if segment < 15.0:
            travelled += segment
        previous = current
        if distance(current, last_progress_location) >= 2.0:
            last_progress_location = current
            last_progress_time = time.monotonic()
        if route_index > best_route_index:
            best_route_index = route_index
            last_progress_time = time.monotonic()
        elif (
            time.monotonic() - last_progress_time
            > NO_PROGRESS_TIMEOUT_SECONDS
        ):
            print(
                f"No route progress for {NO_PROGRESS_TIMEOUT_SECONDS:.0f} "
                "seconds -> applying safety stop."
            )
            break
        if route_recovery_active:
            if (
                route_recovery_point_index < len(route_recovery_points)
                and distance(
                    current,
                    route_recovery_points[
                        route_recovery_point_index
                    ].transform.location,
                ) < 2.5
            ):
                route_recovery_point_index += 1
            if route_recovery_point_index >= len(route_recovery_points):
                if route_recovery_merge_index is not None:
                    route_index = max(route_index, route_recovery_merge_index)
                    active_route_index = route_index
                    route_recovery_active = False
                    route_recovery_points.clear()
                    route_recovery_point_index = 0
                    route_recovery_merge_index = None
                    print("Route tracking recovered; following original route.")
                else:
                    route_recovery_active = False
                    route_recovery_points.clear()
                    route_recovery_point_index = 0
                    if not start_route_recovery(
                        world.get_map(),
                        current,
                        route,
                        route_index,
                    ):
                        print(
                            "Route recovery continuation ended without a "
                            "safe Driving-lane extension; applying safety stop."
                        )

        if route_recovery_active:
            target = route_recovery_points[route_recovery_point_index]
        else:
            target = select_safe_route_target(
                world.get_map(),
                current,
                route,
                route_index,
            )
            if target is None and start_route_recovery(
                world.get_map(),
                current,
                route,
                route_index,
            ):
                target = route_recovery_points[route_recovery_point_index]

        if pathweaver_bridge is not None:
            obstacle, obstacle_distance = find_forward_obstacle(
                PATHWEAVER_TRIGGER_DISTANCE_METERS
            )
            use_pathweaver = (
                collision_state["hit"]
                or (
                    obstacle is not None
                    and obstacle_distance <= PATHWEAVER_TRIGGER_DISTANCE_METERS
                )
            ) and not route_recovery_active and not avoidance_active
            if use_pathweaver:
                use_pathweaver = pathweaver_route_is_straight(
                    route,
                    route_index,
                    ego.get_transform(),
                )
            snapshot = world.get_snapshot()
            timestamp_s = float(snapshot.timestamp.elapsed_seconds)
            if use_pathweaver:
                if (
                    pathweaver_plan is None
                    or timestamp_s - pathweaver_plan.timestamp_s
                    >= PATHWEAVER_REPLAN_INTERVAL_SECONDS - 1e-9
                ):
                    try:
                        request = build_pathweaver_request(
                            ego,
                            nearby_blocking_objects(
                                current,
                                PATHWEAVER_TRIGGER_DISTANCE_METERS,
                            ),
                            timestamp_s=timestamp_s,
                            road_half_width_m=pathweaver_road_half_width_m(),
                            route_speed_mps=CRUISE_SPEED / 3.6,
                            collision=collision_state["hit"],
                        )
                        pathweaver_plan = pathweaver_bridge.plan(request)
                        pathweaver_anchor_transform = ego.get_transform()
                    except (
                        PathWeaverBridgeError,
                        RuntimeError,
                        ValueError,
                        TypeError,
                        AttributeError,
                    ):
                        pathweaver_emergency_stop()
                        raise

                try:
                    control = pathweaver_control(
                        pathweaver_plan,
                        pathweaver_anchor_transform,
                        timestamp_s,
                    )
                except (PathWeaverBridgeError, ValueError, TypeError, RuntimeError):
                    pathweaver_emergency_stop()
                    raise
                state = (
                    "PATHWEAVER/"
                    f"{pathweaver_plan.behavior_state if pathweaver_plan.feasible else 'NO_FEASIBLE_TRAJECTORY_BRAKING'} "
                    f"risk={pathweaver_plan.risk_score:.3f} "
                    f"TTC={pathweaver_plan.minimum_ttc_s} "
                    f"MATLAB={pathweaver_plan.matlab_planning_latency_s * 1000:.1f}ms "
                    f"roundtrip={pathweaver_plan.round_trip_latency_s * 1000:.1f}ms"
                )
            else:
                pathweaver_plan = None
                pathweaver_anchor_transform = None
                control, state = calculate_control(target)
        else:
            obstacle = None
            obstacle_distance = float("inf")
            control, state = calculate_control(target)

        if road_pedestrian_crossing_requires_stop():
            control = carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=0.0,
            )
            state = "PEDESTRIAN_CROSSING_WAIT"
            current_risk_level = "CRITICAL"
            current_target_speed = 0.0

        ego.apply_control(control)
        update_moving_traffic_speed()
        update_road_pedestrians()
        if pathweaver_bridge is not None:
            world.tick()
        follow_vehicle()
        if step % 20 == 0:
            if obstacle is None:
                obstacle, obstacle_distance = find_forward_obstacle(30.0)
            route_remaining = (
                distance(
                    current,
                    route[route_index].transform.location,
                )
                + sum(
                    distance(
                        first.transform.location,
                        second.transform.location,
                    )
                    for first, second in zip(
                        route[route_index:],
                        route[route_index + 1:],
                    )
                )
            )
            obstacle_label = (
                f"{obstacle.type_id}#{obstacle.id} "
                f"{obstacle_distance:5.1f} m"
                if obstacle is not None
                else " clear"
            )
            if (
                obstacle is not None
                and obstacle.type_id.startswith("static.map.vegetation.")
            ):
                extent = obstacle.bounding_box.extent
                obstacle_location = obstacle.get_location()
                ego_location = ego.get_location()
                obstacle_label += (
                    f" bbox-half=({extent.x:.1f},{extent.y:.1f},{extent.z:.1f})m"
                    f" object=({obstacle_location.x:.1f},{obstacle_location.y:.1f})"
                    f" ego=({ego_location.x:.1f},{ego_location.y:.1f})"
                )
            obs_dist_str = f"{obstacle_distance:.1f} m" if (obstacle is not None and obstacle_distance < float('inf')) else "N/A"
            print(
                f"Step {step:04d} | State={state:<25} | Risk={current_risk_level:<8} "
                f"| Dist={obs_dist_str:<8} | Speed={speed_kmh(ego):4.1f} km/h | Target={current_target_speed:4.1f} km/h "
                f"| AvoidDir={chosen_avoidance_direction:<5} | Steer={control.steer:+.2f} | DistTravelled={travelled:6.1f} m"
            )
        step += 1
        if pathweaver_bridge is None:
            time.sleep(0.05)

    if ego.is_alive:
        ego.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        )
        if pathweaver_bridge is not None:
            world.tick()
    print("\n"+"="*70)
    if (
        route_index >= len(route) - 1
        and destination_distance <= DESTINATION_REACHED_METERS
    ):
        print(f"DESTINATION REACHED after {travelled:.1f} m.")
    else:
        print(
            f"Journey stopped before destination "
            f"({destination_distance:.1f} m remaining)."
        )
    print("="*70)

# ============================================================
# CLEANUP
# ============================================================

def cleanup():
    global pathweaver_bridge
    global pathweaver_original_world_settings

    print()
    print("Cleaning up...")

    for sensor in sensor_actors:

        try:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        except RuntimeError as exc:
            print(f"Sensor cleanup warning: {exc}")

    for actor in reversed(environment_actors):

        try:
            if actor.is_alive:
                if actor.type_id.startswith("controller.ai.walker"):
                    actor.stop()
                actor.destroy()
        except RuntimeError as exc:
            print(f"Actor cleanup warning: {exc}")

    road_pedestrian_states.clear()
    moving_traffic_actors.clear()

    if pathweaver_bridge is not None:
        try:
            pathweaver_bridge.close()
        except PathWeaverBridgeError as exc:
            print(f"MATLAB Engine shutdown warning: {exc}")
        finally:
            pathweaver_bridge = None
    if pathweaver_original_world_settings is not None:
        world.apply_settings(pathweaver_original_world_settings)
        traffic_manager.set_synchronous_mode(False)
        pathweaver_original_world_settings = None

    print("Cleanup complete.")


# ============================================================
# MAIN
# ============================================================

def main():
    osm_server=None
    print("PathWeaver: DISABLED; using the existing CARLA Python controller.")
    print("Traffic Manager port:", TRAFFIC_MANAGER_PORT)
    try:
        connect()
        existing_vehicle_count = len(
            list(world.get_actors().filter("vehicle.*"))
        )
        expected_vehicle_count = (
            NUM_PARKED
            + max(NUM_MOVING_SCOOTERS, NUM_FALLBACK_MOVING_CARS)
            + NUM_MOVING_VANS
            + 1
        )
        if existing_vehicle_count > max(50, expected_vehicle_count * 2):
            print(
                f"WARNING: CARLA already contains {existing_vehicle_count} "
                "vehicles, well above this scenario's expected count. "
                "Restart CARLA to remove actors left by earlier runs; "
                "unowned actors are not deleted automatically."
            )
        configure_world()
        osm_server = start_osm_server()
        destination = wait_for_osm_destination()
        map_obj = world.get_map()
        _, _, route = plan_osm_directional_drive(
            map_obj,
            destination,
        )
        route_distance = sum(
            distance(first.transform.location, second.transform.location)
            for first, second in zip(route, route[1:])
        )
        set_destination(route[-1].transform.location)
        requested_bearing = math.degrees(osm_bearing(destination)) % 360.0
        print(
            f"OSM desired bearing: {requested_bearing:.1f} degrees. "
            f"Selected a {route_distance:.1f} m CARLA Driving-lane route "
            f"with {len(route)} waypoints "
            f"on '{map_obj.name}'. OSM coordinates were not mapped to CARLA XY."
        )
        create_environment(route)
        time.sleep(1.0)
        follow_vehicle()
        if RUN_DRIVING:
            print("\nCARLA driving-waypoint route selected. Starting simulation.\n")
            autonomous_loop(route)
        else:
            print("Environment created; RUN_DRIVING is disabled.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if osm_server is not None:
            try: osm_server.shutdown(); osm_server.server_close()
            except OSError as exc:
                print(f"OSM server shutdown warning: {exc}")
        cleanup()


if __name__ == "__main__":
    main()

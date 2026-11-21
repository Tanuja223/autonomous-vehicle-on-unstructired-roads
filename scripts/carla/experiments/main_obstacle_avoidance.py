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
    * unloads decals
    * adds irregular roadside/road obstacles
    * adds parked vehicles, rocks, debris and cones
    * disables lane-following and A*
    * provides a collision-safe speed/steering controller
    * is ready for a destination supplied later

For truly ZERO visible lane markings, replace the Town10HD road material
in Unreal Engine. The driving controller below does NOT depend on lane lines.

Run:
    python unstructured_autonomous.py

The script starts in ENVIRONMENT_ONLY mode.
Later change RUN_DRIVING = True and connect your destination source.
"""

import carla
import random
import math
import time
import threading
import json

# ============================================================
# CONFIG
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

MAP_NAME = "Town10HD_Opt"

RUN_DRIVING = True       # False = only build environment
                         # True  = run simple autonomous controller

ROAD_LENGTH = 1000.0
NUM_ROCKS = 15
NUM_DEBRIS = 10
NUM_CONES = 5
NUM_BARRIERS = 3
NUM_PARKED = 3

CRUISE_SPEED = 25.0       # km/h
CAUTION_SPEED = 12.0      # km/h
SLOW_SPEED = 6.0          # km/h
STOP_DISTANCE = 4.0       # metres
SLOW_DISTANCE = 12.0      # metres
SAFE_DISTANCE = 20.0      # metres

# Local obstacle avoidance
AVOID_DETECT_DISTANCE = 18.0
AVOID_LATERAL_OFFSET = 4.5
AVOID_FORWARD_OFFSET = 7.0
AVOID_REJOIN_DISTANCE = 12.0
AVOID_MAX_TIME = 8.0

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ============================================================
# GLOBALS
# ============================================================

client = None
world = None
ego = None

environment_actors = []
sensor_actors = []

latest_rgb = None
rgb_lock = threading.Lock()

destination_location = None

# Obstacle avoidance state
avoidance_active = False
avoidance_route = []
avoidance_index = 0
avoidance_side = 0
avoidance_started_at = 0.0
avoidance_obstacle_id = None


# ============================================================
# OSM DESTINATION RECEIVER
# ============================================================

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OSM_HOST="127.0.0.1"
OSM_PORT=5001
_osm_destination=None
_osm_lock=threading.Lock()

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
            data=json.loads(self.rfile.read(n).decode("utf-8"))
            _osm_destination={
                "start_lat":data.get("start_lat"),
                "start_lon":data.get("start_lon"),
                "destination_lat":float(data["destination_lat"]),
                "destination_lon":float(data["destination_lon"]),
                "display_name":data.get("display_name","")}
            print("\\nOSM DESTINATION RECEIVED:",_osm_destination)
            self._send_json(200,{"status":"success"})
        except Exception as exc:
            self._send_json(400,{"status":"error","message":str(exc)})
    def log_message(self,*_): return

def start_osm_server():
    server=ThreadingHTTPServer((OSM_HOST,OSM_PORT),OSMDestinationHandler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f"OSM -> CARLA server running at http://{OSM_HOST}:{OSM_PORT}")
    return server

def wait_for_osm_destination():
    print("Waiting for destination from the OSM webpage...")
    while True:
        with _osm_lock:
            if _osm_destination is not None: return dict(_osm_destination)
        time.sleep(0.2)

def osm_bearing(destination):
    try:
        dlat=float(destination["destination_lat"])-float(destination["start_lat"])
        dlon=float(destination["destination_lon"])-float(destination["start_lon"])
        return math.atan2(dlon,dlat)
    except Exception: return 0.0



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


def forward_vector(transform):
    return transform.get_forward_vector()


# ============================================================
# CONNECT
# ============================================================

def connect():
    global client, world

    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    print("Connecting to CARLA...")

    world = client.get_world()

    print("CARLA:", client.get_server_version())
    print("MAP  :", world.get_map().name)

    if MAP_NAME not in world.get_map().name:
        raise RuntimeError(
            "Start CARLA with Town10HD_Opt before running this script."
        )

    print("Town10HD connected.")


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

def choose_start():

    map_obj = world.get_map()
    spawns = map_obj.get_spawn_points()

    if not spawns:
        raise RuntimeError("Town10HD has no spawn points.")

    # Try several points and select one that is a driving waypoint.
    candidates = []

    for transform in spawns:
        wp = map_obj.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if wp:
            candidates.append(wp)

    if not candidates:
        raise RuntimeError("No driving waypoint found.")

    return candidates[min(20, len(candidates)-1)]


def build_road_segment(start_wp, length=450.0):

    route = [start_wp]
    current = start_wp
    travelled = 0.0

    while travelled < length:

        choices = current.next(5.0)

        if not choices:
            break

        if len(choices) == 1:
            nxt = choices[0]

        else:
            current_forward = current.transform.get_forward_vector()

            best = choices[0]
            best_score = -999.0

            for candidate in choices:

                f = candidate.transform.get_forward_vector()

                score = (
                    current_forward.x*f.x +
                    current_forward.y*f.y
                )

                if score > best_score:
                    best_score = score
                    best = candidate

            nxt = best

        travelled += distance(
            current.transform.location,
            nxt.transform.location
        )

        route.append(nxt)
        current = nxt

    return route


# ============================================================
# EGO
# ============================================================

def spawn_ego(start_wp):

    library = world.get_blueprint_library()

    bp = None

    for name in [
        "vehicle.tesla.model3",
        "vehicle.tesla.modely"
    ]:
        matches = library.filter(name)
        if matches:
            bp = matches[0]
            break

    if bp is None:
        bp = random.choice(library.filter("vehicle.*"))

    t = start_wp.transform

    transform = carla.Transform(
        carla.Location(
            x=t.location.x,
            y=t.location.y,
            z=t.location.z + 0.5
        ),
        t.rotation
    )

    vehicle = world.try_spawn_actor(bp, transform)

    if vehicle is None:
        raise RuntimeError("Could not spawn ego vehicle.")

    vehicle.set_autopilot(False)

    environment_actors.append(vehicle)

    print("Ego vehicle:", vehicle.id)

    return vehicle


# ============================================================
# GENERIC PROP SPAWNING
# ============================================================

def spawn_prop(bp, location, rotation=None):

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

        # Mostly roadside, with a few near the driving area.
        lateral = random.choice([
            random.uniform(-7.5, -3.0),
            random.uniform(3.0, 7.5)
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

        # Some debris deliberately lies close to the travel area.
        lateral = random.uniform(-5.0, 5.0)

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

    created = 0
    locations = []

    for _ in range(NUM_PARKED):

        for _attempt in range(20):

            wp = random.choice(route)
            t = wp.transform
            right = right_vector(t)

            lateral = random.choice([
                random.uniform(-9, -5.5),
                random.uniform(5.5, 9)
            ])

            location = t.location + right*lateral
            location.z += 0.4

            if ego and distance(
                location,
                ego.get_location()
            ) < 20:
                continue

            if any(
                distance(location, old) < 8
                for old in locations
            ):
                continue

            actor = spawn_prop(
                random.choice(vehicles),
                location,
                carla.Rotation(
                    yaw=t.rotation.yaw + random.choice([-90, 90])
                )
            )

            if actor:
                locations.append(location)
                created += 1

            break

    print("Parked vehicles:", created)


# ============================================================
# CAMERA
# ============================================================

def attach_rgb_camera():

    global latest_rgb

    library = world.get_blueprint_library()

    camera_bp = library.find("sensor.camera.rgb")

    camera_bp.set_attribute("image_size_x", "640")
    camera_bp.set_attribute("image_size_y", "360")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "0.10")

    transform = carla.Transform(
        carla.Location(
            x=1.8,
            z=1.6
        ),
        carla.Rotation(
            pitch=0
        )
    )

    camera = world.spawn_actor(
        camera_bp,
        transform,
        attach_to=ego
    )

    sensor_actors.append(camera)

    def callback(image):

        global latest_rgb

        with rgb_lock:
            latest_rgb = image

    camera.listen(callback)

    print("RGB camera attached.")

    return camera


# ============================================================
# COLLISION SENSOR
# ============================================================

collision_state = {
    "hit": False,
    "last_time": 0.0
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

        collision_state["hit"] = True
        collision_state["last_time"] = time.time()

        print(
            "[COLLISION]",
            event.other_actor.type_id
        )

    sensor.listen(callback)

    print("Collision sensor attached.")

    return sensor


# ============================================================
# FORWARD OBSTACLE CHECK
# ============================================================

def find_forward_obstacle(max_distance=30.0):

    ego_transform = ego.get_transform()
    origin = ego_transform.location + carla.Location(z=1.0)

    forward = ego_transform.get_forward_vector()

    actors = world.get_actors()

    best_distance = float("inf")
    best_actor = None

    for actor in actors:

        if actor.id == ego.id:
            continue

        if not actor.is_alive:
            continue

        type_id = actor.type_id.lower()

        # We only care about vehicles, walkers and large props.
        relevant = (
            "vehicle." in type_id
            or "walker.pedestrian" in type_id
            or "static.prop" in type_id
        )

        if not relevant:
            continue

        loc = actor.get_location()

        dx = loc.x - origin.x
        dy = loc.y - origin.y

        longitudinal = dx*forward.x + dy*forward.y

        if longitudinal <= 0:
            continue

        if longitudinal > max_distance:
            continue

        lateral = abs(
            dx*(-forward.y) +
            dy*forward.x
        )

        # Approximate vehicle travel corridor.
        if lateral > 2.5:
            continue

        d = math.sqrt(dx*dx + dy*dy)

        if d < best_distance:
            best_distance = d
            best_actor = actor

    return best_actor, best_distance


# ============================================================
# DESTINATION
# ============================================================

def set_destination(location):

    """
    Later your OSM destination code can call:

        set_destination(carla.Location(x, y, z))

    This keeps the environment/controller separate from
    your destination-selection UI.
    """

    global destination_location

    destination_location = location

    print(
        "DESTINATION SET:",
        location
    )


# ============================================================
# OBSTACLE DETECTION + LOCAL PATH REPLANNING
# ============================================================

def actor_is_blocking(actor):
    if actor is None or not actor.is_alive or actor.id == ego.id:
        return False
    tid = actor.type_id.lower()
    return (
        "vehicle." in tid
        or "walker.pedestrian" in tid
        or "static.prop" in tid
    )


def find_forward_obstacle(max_distance=30.0):
    """Find the closest object in the ego vehicle's forward corridor."""
    ego_transform = ego.get_transform()
    origin = ego_transform.location + carla.Location(z=0.8)
    forward = ego_transform.get_forward_vector()
    right = carla.Vector3D(-forward.y, forward.x, 0.0)

    best_distance = float("inf")
    best_actor = None

    for actor in world.get_actors():
        if not actor_is_blocking(actor):
            continue

        loc = actor.get_location()
        rel = loc - origin
        longitudinal = rel.x * forward.x + rel.y * forward.y
        lateral = abs(rel.x * right.x + rel.y * right.y)

        if longitudinal <= 0.0 or longitudinal > max_distance:
            continue

        # Wider than the old 2.5 m corridor so an obstacle is detected
        # before the vehicle gets close enough to stop behind it.
        if lateral > 3.2:
            continue

        d = math.sqrt(rel.x * rel.x + rel.y * rel.y)
        if d < best_distance:
            best_distance = d
            best_actor = actor

    return best_actor, best_distance


def point_clear_of_actors(point, clearance=3.0, ignore_id=None):
    """Check a candidate detour point against nearby vehicles/props."""
    for actor in world.get_actors():
        if actor.id == ego.id or actor.id == ignore_id or not actor_is_blocking(actor):
            continue
        if distance(point, actor.get_location()) < clearance:
            return False
    return True


def build_local_avoidance_route(obstacle):
    """Create a short left/right CARLA-waypoint detour around an obstacle.

    The vehicle is not simply steered around the object. A temporary route is
    built from CARLA waypoints, then the normal 1-km route is resumed.
    """
    global avoidance_route, avoidance_index, avoidance_side

    current_wp = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if current_wp is None:
        return False

    # Prefer a lane/road on either side. We score candidates by clearance
    # from the obstacle and by how well their heading continues forward.
    candidates = []
    left = current_wp.get_left_lane()
    right = current_wp.get_right_lane()

    for side, candidate in ((1, left), (-1, right)):
        if candidate is None or candidate.lane_type != carla.LaneType.Driving:
            continue

        p = candidate.transform.location
        f = candidate.transform.get_forward_vector()
        ego_f = current_wp.transform.get_forward_vector()
        heading_score = ego_f.x * f.x + ego_f.y * f.y
        clearance = distance(p, obstacle.get_location())

        # Reject a candidate if the obstacle is still close to that side.
        if clearance < 3.0:
            continue

        # Follow the side for several waypoints, then merge back.
        detour = [candidate]
        cur = candidate
        for _ in range(10):
            nxt = cur.next(3.0)
            if not nxt:
                break
            # Continue in the same lane and avoid reversing direction.
            same_lane = [w for w in nxt if w.lane_id == candidate.lane_id]
            pool = same_lane or nxt
            best = max(
                pool,
                key=lambda w: (
                    w.transform.get_forward_vector().x * f.x
                    + w.transform.get_forward_vector().y * f.y
                ),
            )
            detour.append(best)
            cur = best

        if len(detour) >= 5:
            candidates.append((clearance + 5.0 * heading_score, side, detour))

    if not candidates:
        return False

    _, avoidance_side, avoidance_route = max(candidates, key=lambda x: x[0])
    avoidance_index = 0
    return True


def start_obstacle_avoidance(obstacle):
    global avoidance_active, avoidance_started_at, avoidance_obstacle_id

    if obstacle is None:
        return False

    if avoidance_active and avoidance_obstacle_id == obstacle.id:
        return True

    if not build_local_avoidance_route(obstacle):
        return False

    avoidance_active = True
    avoidance_started_at = time.time()
    avoidance_obstacle_id = obstacle.id

    side_name = "LEFT" if avoidance_side > 0 else "RIGHT"
    print(
        f"OBSTACLE DETECTED at {distance(ego.get_location(), obstacle.get_location()):.1f} m "
        f"-> REPLANNING {side_name}"
    )
    return True


def avoidance_target():
    global avoidance_active, avoidance_index

    if not avoidance_active or not avoidance_route:
        return None

    # Find the closest detour waypoint and look a few points ahead.
    current = ego.get_location()
    while (
        avoidance_index < len(avoidance_route) - 1
        and distance(current, avoidance_route[avoidance_index].transform.location) < 3.0
    ):
        avoidance_index += 1

    if avoidance_index >= len(avoidance_route):
        avoidance_active = False
        return None

    if time.time() - avoidance_started_at > AVOID_MAX_TIME:
        avoidance_active = False
        return None

    return avoidance_route[min(avoidance_index + 2, len(avoidance_route) - 1)].transform.location


def calculate_steering(target):
    transform = ego.get_transform()
    loc = transform.location
    forward = transform.get_forward_vector()

    dx = target.x - loc.x
    dy = target.y - loc.y
    length = max(math.sqrt(dx*dx + dy*dy), 0.001)

    tx = dx / length
    ty = dy / length

    cross = forward.x*ty - forward.y*tx
    dot = clamp(forward.x*tx + forward.y*ty, -1.0, 1.0)
    angle = math.atan2(cross, dot)

    return clamp(angle * 1.8, -1.0, 1.0)


def calculate_control(target):
    current_speed = speed_kmh(ego)
    obstacle, obstacle_distance = find_forward_obstacle(30.0)

    if collision_state["hit"]:
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0), "COLLISION_BRAKE"

    # Start the detour early. Do not wait until STOP_DISTANCE.
    if obstacle is not None and obstacle_distance < AVOID_DETECT_DISTANCE:
        if not avoidance_active:
            if start_obstacle_avoidance(obstacle):
                pass

    detour_target = avoidance_target()
    if detour_target is not None:
        # Slow but KEEP MOVING around the obstacle.
        steer = calculate_steering(detour_target)
        target_speed = 8.0
        throttle = 0.30 if current_speed < target_speed else 0.05
        brake = 0.0 if current_speed <= target_speed + 1.5 else 0.15
        return carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer,
            reverse=False,
            hand_brake=False,
        ), "PATH_AVOIDANCE"

    # If no detour could be built, stop only when genuinely blocked.
    if obstacle is not None and obstacle_distance < STOP_DISTANCE:
        return carla.VehicleControl(
            throttle=0.0, brake=1.0, steer=0.0
        ), "BLOCKED_NO_SAFE_PATH"

    if obstacle_distance < SLOW_DISTANCE:
        steer = calculate_steering(target)
        target_speed = SLOW_SPEED
        throttle = 0.25 if current_speed < target_speed else 0.0
        brake = 0.2 if current_speed > target_speed + 2 else 0.0
        return carla.VehicleControl(
            throttle=throttle, brake=brake, steer=steer
        ), "OBSTACLE_SLOW"

    if obstacle_distance < SAFE_DISTANCE:
        steer = calculate_steering(target)
        target_speed = CAUTION_SPEED
        throttle = 0.35 if current_speed < target_speed else 0.0
        brake = 0.15 if current_speed > target_speed + 2 else 0.0
        return carla.VehicleControl(
            throttle=throttle, brake=brake, steer=steer
        ), "OBSTACLE_CAUTION"

    steer = calculate_steering(target)
    if current_speed < CRUISE_SPEED:
        throttle, brake = 0.45, 0.0
    else:
        throttle, brake = 0.05, 0.1

    return carla.VehicleControl(
        throttle=throttle,
        brake=brake,
        steer=steer,
        reverse=False,
        hand_brake=False,
    ), "CRUISING"

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
    ego=spawn_ego(route[0])
    add_rocks(route); add_debris(route); add_cones(route); add_barriers(route); add_parked_vehicles(route)
    # RGB and collision sensors intentionally disabled for D3D-safe testing.
    follow_vehicle()
    print("\\n"+"="*70)
    print("TOWN10HD UNSTRUCTURED 1 KM ENVIRONMENT")
    print("="*70)
    print("Buildings / trees / terrain : YES")
    print("Roadside rocks/debris       : YES")
    print("Construction objects        : YES")
    print("Parked vehicles             : YES")
    print("Moving NPC vehicles         : NO")
    print("Moving pedestrians          : NO")
    print("Artificial front obstacle   : NO")
    print("Artificial rear obstacle    : NO")
    print("Obstacle avoidance           : LEFT/RIGHT LOCAL REPLAN")
    print("Lane following              : NO")
    print("RGB camera                  : NO (D3D-safe mode)")
    print("Collision sensor            : NO")
    print("NOTE: Town10HD baked lane markings are not removed by Python.")
    print("="*70)

# ============================================================
# DESTINATION DEMO
# ============================================================

def destination_demo(route, osm_destination):
    desired=osm_bearing(osm_destination)
    best_index=min(25,len(route)-1); best_score=float("inf")
    for i,wp in enumerate(route[:80]):
        f=wp.transform.get_forward_vector(); heading=math.atan2(f.y,f.x)
        error=abs(math.atan2(math.sin(heading-desired),math.cos(heading-desired)))
        score=error+abs(i-25)*0.005
        if score<best_score: best_score=score; best_index=i
    goal=route[best_index].transform.location
    set_destination(carla.Location(x=goal.x,y=goal.y,z=goal.z))

# ============================================================
# AUTONOMOUS LOOP
# ============================================================

def autonomous_loop(route, osm_destination):
    destination_demo(route, osm_destination)
    travelled=0.0; previous=ego.get_location(); route_index=0; step=0
    while ego.is_alive and travelled<ROAD_LENGTH:
        current=ego.get_location(); segment=distance(current,previous)
        if segment<15.0: travelled+=segment
        previous=current
        while route_index<len(route)-1 and distance(current,route[route_index].transform.location)<5.0:
            route_index+=1
        if route_index>=len(route)-1:
            wp=world.get_map().get_waypoint(current,project_to_road=True,lane_type=carla.LaneType.Driving)
            if wp is None: break
            nxt=wp.next(5.0)
            if not nxt: break
            f=wp.transform.get_forward_vector()
            best=max(nxt,key=lambda x:f.x*x.transform.get_forward_vector().x+f.y*x.transform.get_forward_vector().y)
            route.append(best); route_index=len(route)-2
        target=route[min(route_index+4,len(route)-1)].transform.location

        # Once the obstacle is behind us, end the temporary detour and merge
        # back to the main 1-km route.
        if avoidance_active and avoidance_obstacle_id is not None:
            try:
                actor = world.get_actor(avoidance_obstacle_id)
                if actor is None or not actor.is_alive:
                    globals()["avoidance_active"] = False
                else:
                    rel = actor.get_location() - ego.get_location()
                    f = ego.get_transform().get_forward_vector()
                    passed = rel.x*f.x + rel.y*f.y < -2.0
                    if passed:
                        globals()["avoidance_active"] = False
                        globals()["avoidance_route"] = []
                        globals()["avoidance_index"] = 0
                        print("Obstacle passed -> MERGING BACK TO MAIN ROUTE")
            except Exception:
                pass

        control,state=calculate_control(target)
        ego.apply_control(control); follow_vehicle()
        if step%20==0:
            _,od=find_forward_obstacle(30.0)
            print(f"Step {step:04d} | distance={travelled:7.1f} m | speed={speed_kmh(ego):5.1f} km/h | obstacle={od:5.1f} m | steer={control.steer:+.2f} | state={state}")
        step+=1; time.sleep(0.05)
    if ego.is_alive:
        ego.apply_control(carla.VehicleControl(throttle=0.0,brake=1.0,steer=0.0))
    print("\\n"+"="*70)
    print(f"SUCCESS: 1 KM JOURNEY COMPLETED ({travelled:.1f} m)" if travelled>=ROAD_LENGTH else f"Journey stopped at {travelled:.1f} m")
    print("="*70)

# ============================================================
# CLEANUP
# ============================================================

def cleanup():

    print()
    print("Cleaning up...")

    for sensor in sensor_actors:

        try:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        except Exception:
            pass

    for actor in reversed(environment_actors):

        try:
            if actor.is_alive:
                actor.destroy()
        except Exception:
            pass

    print("Cleanup complete.")


# ============================================================
# MAIN
# ============================================================

def main():
    osm_server=None
    try:
        connect(); configure_world()
        osm_server=start_osm_server(); destination=wait_for_osm_destination()
        start_wp=choose_start(); route=build_road_segment(start_wp,ROAD_LENGTH)
        if len(route)<2: raise RuntimeError("Could not create a connected 1 km road route.")
        create_environment(route)
        time.sleep(2.0); follow_vehicle()
        print("\\nOSM destination accepted. Starting continuous 1 km drive.")
        print("No moving traffic in front/behind. No artificial front/rear obstacle.\\n")
        autonomous_loop(route,destination)
    except KeyboardInterrupt:
        print("\\nStopped by user.")
    except Exception as exc:
        print("\\nERROR:"); print(exc)
    finally:
        if osm_server is not None:
            try: osm_server.shutdown(); osm_server.server_close()
            except Exception: pass
        cleanup()


if __name__ == "__main__":
    main()

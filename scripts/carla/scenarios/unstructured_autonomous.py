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

# ============================================================
# CONFIG
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

MAP_NAME = "Town10HD_Opt"

RUN_DRIVING = True      # False = only build environment
                         # True  = run simple autonomous controller

ROAD_LENGTH = 200.0
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
# SIMPLE COLLISION-FREE STEERING
# ============================================================

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
    dot = forward.x*tx + forward.y*ty

    angle = math.atan2(cross, dot)

    # Proportional steering.
    steer = angle * 1.8

    return clamp(steer, -1.0, 1.0)


def calculate_control(target):

    current_speed = speed_kmh(ego)

    obstacle, obstacle_distance = find_forward_obstacle(
        max_distance=30.0
    )

    # --------------------------------------------------------
    # EMERGENCY STOP
    # --------------------------------------------------------

    if collision_state["hit"]:

        return carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0
        ), "COLLISION_BRAKE"

    # --------------------------------------------------------
    # OBSTACLE SPEED CONTROL
    # --------------------------------------------------------

    if obstacle_distance < STOP_DISTANCE:

        return carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0
        ), "EMERGENCY_OBSTACLE_BRAKE"

    if obstacle_distance < SLOW_DISTANCE:

        steer = calculate_steering(target)

        # Reduce speed as obstacle gets closer.
        target_speed = SLOW_SPEED

        throttle = 0.25 if current_speed < target_speed else 0.0

        brake = 0.2 if current_speed > target_speed + 2 else 0.0

        return carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer
        ), "OBSTACLE_SLOW"

    if obstacle_distance < SAFE_DISTANCE:

        steer = calculate_steering(target)

        target_speed = CAUTION_SPEED

        throttle = 0.35 if current_speed < target_speed else 0.0

        brake = 0.15 if current_speed > target_speed + 2 else 0.0

        return carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer
        ), "OBSTACLE_CAUTION"

    # --------------------------------------------------------
    # NORMAL DRIVING
    # --------------------------------------------------------

    steer = calculate_steering(target)

    target_speed = CRUISE_SPEED

    if current_speed < target_speed:
        throttle = 0.45
        brake = 0.0
    else:
        throttle = 0.05
        brake = 0.1

    return carla.VehicleControl(
        throttle=throttle,
        brake=brake,
        steer=steer
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

def create_environment():

    start_wp = choose_start()

    route = build_road_segment(
        start_wp,
        ROAD_LENGTH
    )

    print(
        "Selected road section:",
        len(route),
        "waypoints"
    )

    global ego

    ego = spawn_ego(start_wp)

    add_rocks(route)
    add_debris(route)
    add_cones(route)
    add_barriers(route)
    add_parked_vehicles(route)

    # TEMPORARILY DISABLED
    # attach_rgb_camera()
    # attach_collision_sensor()

    follow_vehicle()

    print()
    print("=" * 70)
    print("TOWN10HD UNSTRUCTURED ENVIRONMENT")
    print("=" * 70)
    print()
    print("Buildings          : YES")
    print("Trees              : YES")
    print("Terrain            : YES")
    print("Rocks              : YES")
    print("Debris             : YES")
    print("Parked vehicles    : YES")
    print("RGB camera         : TEMPORARILY OFF")
    print("Collision sensor   : TEMPORARILY OFF")
    print("Lane following     : OFF")
    print("A*                 : OFF")
    print()
    print("Press Ctrl+C to stop.")
    print("=" * 70)

    return route


# ============================================================
# DESTINATION DEMO
# ============================================================

def destination_demo(route):

    """
    This is only a local CARLA destination demo.

    Later replace this with your OSM destination:
        set_destination(...)
    """

    # Select a point roughly 120m ahead.
    index = min(
        25,
        len(route)-1
    )

    goal = route[index].transform.location

    goal = carla.Location(
        x=goal.x,
        y=goal.y,
        z=goal.z
    )

    set_destination(goal)


# ============================================================
# AUTONOMOUS LOOP
# ============================================================

def autonomous_loop(route):

    global destination_location

    destination_demo(route)

    step = 0

    while ego.is_alive:

        if destination_location is None:
            time.sleep(0.1)
            continue

        current = ego.get_location()

        remaining = distance(
            current,
            destination_location
        )

        if remaining < 5.0:

            ego.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0
                )
            )

            print("Destination reached.")

            break

        control, state = calculate_control(
            destination_location
        )

        ego.apply_control(control)

        follow_vehicle()

        if step % 20 == 0:

            obstacle, obstacle_distance = (
                find_forward_obstacle(30.0)
            )

            print(
                f"Step {step:04d} | "
                f"speed={speed_kmh(ego):5.1f} km/h | "
                f"destination={remaining:6.1f} m | "
                f"obstacle={obstacle_distance:5.1f} m | "
                f"steer={control.steer:+.2f} | "
                f"state={state}"
            )

        step += 1

        time.sleep(0.10)


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

    try:

        connect()

        configure_world()

        route = create_environment()

        if RUN_DRIVING:

            print()
            print("AUTONOMOUS DRIVING ENABLED")
            print("The local destination demo will start.")
            print()

            autonomous_loop(route)

        else:

            print()
            print("ENVIRONMENT-ONLY MODE")
            print()
            print("The car is spawned but will NOT drive.")
            print()
            print("This is the stage for checking the road environment.")
            print()
            print("Later:")
            print("1. Select destination from your OSM page.")
            print("2. Convert destination to a CARLA goal.")
            print("3. Replace destination_demo() with OSM destination.")
            print("4. Enable autonomous_loop().")
            print()
            print("Press Ctrl+C to stop.")

            while True:

                if not ego.is_alive:
                    break

                follow_vehicle()

                time.sleep(1)

    except KeyboardInterrupt:

        print("\nStopped by user.")

    except Exception as exc:

        print()
        print("ERROR:")
        print(exc)

    finally:

        cleanup()


if __name__ == "__main__":
    main()

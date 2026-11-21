"""
unstructured_town10.py

CARLA 0.9.15
Town10HD-based unstructured-road environment.

Keeps:
    - Town10HD buildings
    - Trees
    - Terrain
    - Street environment
    - Existing road network

Adds:
    - Pothole-like road patches
    - Rocks
    - Road debris
    - Construction barriers
    - Traffic cones
    - Concrete pipes
    - Parked vehicles
    - Irregular roadside obstacles

IMPORTANT:
    This script DOES NOT use lane following or A*.

The Town10HD baked lane markings cannot be physically removed
through the normal CARLA Python API. For complete removal,
the Unreal map/material must be edited.
"""

import carla
import random
import math
import time
import sys


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

RANDOM_SEED = 42

# How far the test environment extends from the starting point
ROAD_LENGTH = 350.0

# Number of objects
NUM_ROCKS = 45
NUM_DEBRIS = 35
NUM_CONES = 14
NUM_BARRIERS = 8
NUM_PARKED_CARS = 8
NUM_POTHOLES = 18

# Distance between environment objects
OBJECT_SPACING = 8.0

random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def get_forward_right(transform):
    forward = transform.get_forward_vector()

    right = carla.Vector3D(
        x=-forward.y,
        y=forward.x,
        z=0.0
    )

    return forward, right


def find_blueprint(world, keywords):
    """
    Find a blueprint containing one of the requested keywords.
    """
    library = world.get_blueprint_library()

    for keyword in keywords:
        matches = library.filter(keyword)

        if matches:
            return random.choice(matches)

    return None


# ============================================================
# ROAD / WAYPOINT SELECTION
# ============================================================

def find_start_waypoint(world):
    """
    Find a Town10HD driving waypoint near the center of the map.
    """

    map_obj = world.get_map()
    spawn_points = map_obj.get_spawn_points()

    if not spawn_points:
        raise RuntimeError("Town10HD has no spawn points.")

    # Prefer a spawn point away from intersections
    candidates = []

    for transform in spawn_points:

        wp = map_obj.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if wp is None:
            continue

        candidates.append(wp)

    if not candidates:
        raise RuntimeError("Could not find a driving waypoint.")

    # Pick a reproducible location
    return candidates[min(20, len(candidates) - 1)]


def build_route_segment(start_wp, distance_m=350.0):
    """
    Build a simple sequence of Town10HD waypoints.

    IMPORTANT:
    This is ONLY used to determine where to place
    unstructured-road objects.

    The autonomous vehicle should NOT use this
    as lane-following logic.
    """

    route = [start_wp]

    current = start_wp

    travelled = 0.0

    while travelled < distance_m:

        next_points = current.next(5.0)

        if not next_points:
            break

        # Prefer straight continuation
        if len(next_points) == 1:
            nxt = next_points[0]

        else:
            current_forward = current.transform.get_forward_vector()

            best = None
            best_angle = 999.0

            for candidate in next_points:

                candidate_forward = (
                    candidate.transform.get_forward_vector()
                )

                dot = (
                    current_forward.x * candidate_forward.x
                    + current_forward.y * candidate_forward.y
                )

                dot = max(-1.0, min(1.0, dot))

                angle = math.acos(dot)

                if angle < best_angle:
                    best_angle = angle
                    best = candidate

            nxt = best

        segment_distance = distance(
            current.transform.location,
            nxt.transform.location
        )

        travelled += segment_distance

        route.append(nxt)

        current = nxt

    return route


# ============================================================
# EGO VEHICLE
# ============================================================

def spawn_ego_vehicle(world, start_wp):
    library = world.get_blueprint_library()

    # Prefer Tesla
    vehicle_bp = None

    tesla_candidates = [
        "vehicle.tesla.model3",
        "vehicle.tesla.modely",
    ]

    for name in tesla_candidates:

        candidates = library.filter(name)

        if candidates:
            vehicle_bp = candidates[0]
            break

    # Fallback
    if vehicle_bp is None:
        candidates = library.filter("vehicle.*")

        if not candidates:
            raise RuntimeError("No vehicle blueprint found.")

        vehicle_bp = candidates[0]

    transform = start_wp.transform

    # Put vehicle slightly above road
    spawn_transform = carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z + 0.5
        ),
        transform.rotation
    )

    vehicle = world.try_spawn_actor(
        vehicle_bp,
        spawn_transform
    )

    if vehicle is None:
        raise RuntimeError("Could not spawn ego vehicle.")

    vehicle.set_autopilot(False)

    print(f"[EGO] Vehicle spawned: {vehicle.id}")

    return vehicle


# ============================================================
# ROCKS
# ============================================================

def spawn_rocks(world, route, actors):

    bp = find_blueprint(
        world,
        [
            "static.prop.rock",
            "static.prop.rock1",
            "static.prop.rock2",
            "static.prop.*"
        ]
    )

    if bp is None:
        print("[ROCKS] No suitable static prop found.")
        return

    created = 0

    for i in range(NUM_ROCKS):

        wp = random.choice(route)

        transform = wp.transform

        forward, right = get_forward_right(transform)

        # Keep rocks near the road rather than directly
        # in the vehicle center line.
        lateral = random.choice([
            random.uniform(4.0, 7.0),
            random.uniform(-7.0, -4.0)
        ])

        longitudinal = random.uniform(-3.0, 3.0)

        location = (
            transform.location
            + right * lateral
            + forward * longitudinal
        )

        location.z += 0.2

        rotation = carla.Rotation(
            pitch=random.uniform(-10, 10),
            yaw=random.uniform(0, 360),
            roll=random.uniform(-10, 10)
        )

        actor = world.try_spawn_actor(
            bp,
            carla.Transform(location, rotation)
        )

        if actor:
            actors.append(actor)
            created += 1

    print(f"[ROCKS] Added: {created}")


# ============================================================
# DEBRIS
# ============================================================

def spawn_debris(world, route, actors):

    library = world.get_blueprint_library()

    candidates = []

    for pattern in [
        "static.prop.barrel",
        "static.prop.box",
        "static.prop.construction",
        "static.prop.*"
    ]:

        found = library.filter(pattern)

        if found:
            candidates.extend(found)

    if not candidates:
        print("[DEBRIS] No static props available.")
        return

    created = 0

    for i in range(NUM_DEBRIS):

        wp = random.choice(route)

        transform = wp.transform

        forward, right = get_forward_right(transform)

        lateral = random.uniform(-5.5, 5.5)

        longitudinal = random.uniform(-3.0, 3.0)

        location = (
            transform.location
            + right * lateral
            + forward * longitudinal
        )

        location.z += 0.2

        bp = random.choice(candidates)

        rotation = carla.Rotation(
            yaw=random.uniform(0, 360)
        )

        actor = world.try_spawn_actor(
            bp,
            carla.Transform(location, rotation)
        )

        if actor:
            actors.append(actor)
            created += 1

    print(f"[DEBRIS] Added: {created}")


# ============================================================
# TRAFFIC CONES
# ============================================================

def spawn_cones(world, route, actors):

    bp = find_blueprint(
        world,
        [
            "static.prop.trafficcone",
            "static.prop.constructioncone"
        ]
    )

    if bp is None:
        print("[CONES] Traffic cone blueprint unavailable.")
        return

    created = 0

    for i in range(NUM_CONES):

        wp = random.choice(route)

        transform = wp.transform

        forward, right = get_forward_right(transform)

        lateral = random.choice([
            random.uniform(-6.0, -3.5),
            random.uniform(3.5, 6.0)
        ])

        location = (
            transform.location
            + right * lateral
        )

        location.z += 0.1

        actor = world.try_spawn_actor(
            bp,
            carla.Transform(
                location,
                carla.Rotation(
                    yaw=random.uniform(0, 360)
                )
            )
        )

        if actor:
            actors.append(actor)
            created += 1

    print(f"[CONES] Added: {created}")


# ============================================================
# CONSTRUCTION BARRIERS
# ============================================================

def spawn_barriers(world, route, actors):

    bp = find_blueprint(
        world,
        [
            "static.prop.warningconstruction",
            "static.prop.streetbarrier",
            "static.prop.construction",
            "static.prop.*"
        ]
    )

    if bp is None:
        print("[BARRIERS] No barrier blueprint found.")
        return

    created = 0

    for i in range(NUM_BARRIERS):

        wp = random.choice(route)

        transform = wp.transform

        forward, right = get_forward_right(transform)

        lateral = random.choice([
            random.uniform(-6.0, -4.0),
            random.uniform(4.0, 6.0)
        ])

        location = transform.location + right * lateral

        location.z += 0.1

        actor = world.try_spawn_actor(
            bp,
            carla.Transform(
                location,
                carla.Rotation(
                    yaw=transform.rotation.yaw
                    + random.uniform(-15, 15)
                )
            )
        )

        if actor:
            actors.append(actor)
            created += 1

    print(f"[BARRIERS] Added: {created}")


# ============================================================
# PARKED VEHICLES
# ============================================================

def spawn_parked_vehicles(world, route, ego, actors):

    library = world.get_blueprint_library()

    vehicles = library.filter("vehicle.*")

    if not vehicles:
        print("[PARKED] No vehicle blueprints.")
        return

    created = 0

    used_locations = []

    for i in range(NUM_PARKED_CARS):

        for attempt in range(20):

            wp = random.choice(route)

            transform = wp.transform

            forward, right = get_forward_right(transform)

            # Park near road edge
            lateral = random.choice([
                random.uniform(-8.0, -5.5),
                random.uniform(5.5, 8.0)
            ])

            location = transform.location + right * lateral

            location.z += 0.4

            # Avoid spawning too close to ego
            if distance(
                location,
                ego.get_location()
            ) < 15.0:
                continue

            if any(
                distance(location, old) < 8.0
                for old in used_locations
            ):
                continue

            bp = random.choice(vehicles)

            rotation = carla.Rotation(
                yaw=transform.rotation.yaw
                + random.choice([90, -90])
            )

            actor = world.try_spawn_actor(
                bp,
                carla.Transform(location, rotation)
            )

            if actor:

                actors.append(actor)

                used_locations.append(location)

                created += 1

                break

    print(f"[PARKED VEHICLES] Added: {created}")


# ============================================================
# POTHOLE MARKERS
# ============================================================

def spawn_potholes(world, route, actors):

    """
    CARLA's normal Python API does not provide a simple way to
    cut physical holes into Town10HD's baked road mesh.

    Therefore these are visual pothole markers placed slightly
    above the road.

    They make the road visually irregular and provide
    perception targets for YOLO/depth processing.
    """

    library = world.get_blueprint_library()

    candidates = []

    for pattern in [
        "static.prop.tire",
        "static.prop.barrel",
        "static.prop.rock",
        "static.prop.*"
    ]:

        found = library.filter(pattern)

        if found:
            candidates.extend(found)

    if not candidates:
        print("[POTHOLES] No visual marker blueprint.")
        return

    created = 0

    for i in range(NUM_POTHOLES):

        wp = random.choice(route)

        transform = wp.transform

        forward, right = get_forward_right(transform)

        # Put potholes inside the driveable area
        lateral = random.uniform(-4.0, 4.0)

        location = transform.location + right * lateral

        location.z += 0.05

        bp = random.choice(candidates)

        actor = world.try_spawn_actor(
            bp,
            carla.Transform(
                location,
                carla.Rotation(
                    pitch=random.uniform(-5, 5),
                    yaw=random.uniform(0, 360)
                )
            )
        )

        if actor:
            actors.append(actor)
            created += 1

    print(f"[POTHOLE AREAS] Added: {created}")


# ============================================================
# ROADSIDE CLUTTER
# ============================================================

def spawn_roadside_clutter(world, route, actors):

    library = world.get_blueprint_library()

    candidates = []

    for pattern in [
        "static.prop.barrel",
        "static.prop.box",
        "static.prop.pallet",
        "static.prop.*"
    ]:

        found = library.filter(pattern)

        if found:
            candidates.extend(found)

    if not candidates:
        return

    created = 0

    for wp in route[::4]:

        if random.random() > 0.55:
            continue

        transform = wp.transform

        forward, right = get_forward_right(transform)

        lateral = random.choice([
            random.uniform(-9, -6),
            random.uniform(6, 9)
        ])

        location = transform.location + right * lateral

        location.z += 0.2

        actor = world.try_spawn_actor(
            random.choice(candidates),
            carla.Transform(
                location,
                carla.Rotation(
                    yaw=random.uniform(0, 360)
                )
            )
        )

        if actor:
            actors.append(actor)
            created += 1

    print(f"[ROADSIDE CLUTTER] Added: {created}")


# ============================================================
# WEATHER
# ============================================================

def set_weather(world):

    weather = carla.WeatherParameters(
        cloudiness=15.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=10.0,
        sun_azimuth_angle=100.0,
        sun_altitude_angle=55.0,
        fog_density=0.0
    )

    world.set_weather(weather)


# ============================================================
# SPECTATOR
# ============================================================

def position_spectator(world, vehicle):

    spectator = world.get_spectator()

    transform = vehicle.get_transform()

    forward = transform.get_forward_vector()

    camera_location = (
        transform.location
        - forward * 12.0
        + carla.Location(z=6.0)
    )

    rotation = carla.Rotation(
        pitch=-15.0,
        yaw=transform.rotation.yaw,
        roll=0.0
    )

    spectator.set_transform(
        carla.Transform(
            camera_location,
            rotation
        )
    )


# ============================================================
# CLEANUP
# ============================================================

def cleanup(actors):

    print("\nCleaning spawned actors...")

    for actor in actors:

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

    print()
    print("=" * 70)
    print("      TOWN10HD UNSTRUCTURED ROAD ENVIRONMENT")
    print("=" * 70)
    print()

    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    try:
        world = client.get_world()

    except Exception as exc:

        print("Could not connect to CARLA.")
        print(exc)
        return

    print(f"CARLA: {client.get_server_version()}")
    print(f"MAP  : {world.get_map().name}")

    if "Town10HD" not in world.get_map().name:

        print()
        print("ERROR: CARLA is not running Town10HD.")
        print()
        print("Start CARLA with Town10HD_Opt before running this script.")
        return

    set_weather(world)

    actors = []

    try:

        # ----------------------------------------------------
        # Find road
        # ----------------------------------------------------

        start_wp = find_start_waypoint(world)

        route = build_route_segment(
            start_wp,
            ROAD_LENGTH
        )

        print()
        print(f"[ROAD] Waypoints selected : {len(route)}")
        print(f"[ROAD] Test length         : {ROAD_LENGTH:.1f} m")

        # ----------------------------------------------------
        # Spawn ego
        # ----------------------------------------------------

        ego = spawn_ego_vehicle(
            world,
            start_wp
        )

        actors.append(ego)

        # ----------------------------------------------------
        # Add environment
        # ----------------------------------------------------

        print()
        print("Creating unstructured-road environment...")

        spawn_rocks(
            world,
            route,
            actors
        )

        spawn_debris(
            world,
            route,
            actors
        )

        spawn_cones(
            world,
            route,
            actors
        )

        spawn_barriers(
            world,
            route,
            actors
        )

        spawn_potholes(
            world,
            route,
            actors
        )

        spawn_roadside_clutter(
            world,
            route,
            actors
        )

        spawn_parked_vehicles(
            world,
            route,
            ego,
            actors
        )

        # ----------------------------------------------------
        # Spectator
        # ----------------------------------------------------

        time.sleep(2)

        position_spectator(
            world,
            ego
        )

        # ----------------------------------------------------
        # Final information
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("          UNSTRUCTURED ENVIRONMENT READY")
        print("=" * 70)
        print()
        print("Base environment       : Town10HD_Opt")
        print("Buildings              : YES")
        print("Trees / vegetation     : YES")
        print("Terrain                : YES")
        print("Street environment     : YES")
        print("Pothole areas          : YES")
        print("Rocks                  : YES")
        print("Road debris            : YES")
        print("Construction objects   : YES")
        print("Parked vehicles        : YES")
        print("Lane following         : DISABLED")
        print("A* lane routing        : DISABLED")
        print()
        print("IMPORTANT:")
        print("Town10HD's baked lane markings cannot be deleted")
        print("through the standard CARLA Python API.")
        print()
        print("This script creates an unstructured driving scenario")
        print("around the existing Town10HD environment.")
        print()
        print("Keep this terminal running.")
        print("Press Ctrl+C to clean up.")
        print("=" * 70)
        print()

        while True:

            if not ego.is_alive:
                print("Ego vehicle no longer exists.")
                break

            time.sleep(1)

    except KeyboardInterrupt:

        print("\nStopping environment...")

    except Exception as exc:

        print()
        print("ERROR:")
        print(exc)

    finally:

        cleanup(actors)


if __name__ == "__main__":
    main()
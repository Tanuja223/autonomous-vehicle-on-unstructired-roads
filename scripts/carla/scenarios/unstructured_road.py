"""
unstructured_road.py
CARLA 0.9.15 - lightweight unstructured Indian-road test environment.

This does NOT modify Town10HD's baked lane markings. Instead it creates a
separate test area using CARLA's existing OpenDRIVE/map road surface plus
irregular roadside/obstacle props. For a truly texture-free/no-marking road,
use the supplied OpenDRIVE file with a custom CARLA map package.

Run:
    python unstructured_road.py

Then connect your main.py to the returned/spawned vehicle at CARLA port 2000.
"""

import random
import math
import time
import carla

HOST = "127.0.0.1"
PORT = 2000
SEED = 42

ROAD_LENGTH_M = 220.0
POTHOLES = 18
ROCKS = 25
DEBRIS = 18
PARKED_VEHICLES = 5

random.seed(SEED)


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def make_location(base, forward, lateral, z=0.05):
    # CARLA forward/right vectors from yaw.
    yaw = math.radians(base.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    rx, ry = -math.sin(yaw), math.cos(yaw)
    return carla.Location(
        x=base.location.x + fx * forward + rx * lateral,
        y=base.location.y + fy * forward + ry * lateral,
        z=base.location.z + z,
    )


def spawn_prop(world, bp, transform, scale=None):
    try:
        actor = world.try_spawn_actor(bp, transform)
        return actor
    except Exception:
        return None


def find_blueprint(bp_lib, patterns):
    for pattern in patterns:
        bps = bp_lib.filter(pattern)
        if bps:
            return random.choice(bps)
    return None


def create_unstructured_environment(world, ego):
    bp = world.get_blueprint_library()
    actors = []

    # Use a broad local area around the ego vehicle. This keeps generation
    # independent of CARLA's lane graph and avoids A*.
    base = ego.get_transform()

    # ------------------------------------------------------------------
    # 1. IRREGULAR ROAD-SIDE BOUNDARIES
    # ------------------------------------------------------------------
    # These are visual/physical roadside markers rather than lane borders.
    # They intentionally have random spacing and lateral displacement.
    rock_bp = find_blueprint(bp, [
        "static.prop.rock",
        "static.prop.*",
    ])

    if rock_bp:
        for i in range(35):
            forward = random.uniform(5, ROAD_LENGTH_M)
            side = random.choice([-1, 1])
            lateral = side * random.uniform(5.5, 9.5)
            loc = make_location(base, forward, lateral, random.uniform(0.0, 0.25))
            rot = carla.Rotation(
                pitch=random.uniform(-8, 8),
                yaw=random.uniform(0, 360),
                roll=random.uniform(-8, 8),
            )
            a = spawn_prop(world, rock_bp, carla.Transform(loc, rot))
            if a:
                actors.append(a)

    # ------------------------------------------------------------------
    # 2. POTHOLES
    # ------------------------------------------------------------------
    # CARLA's standard Python API does not provide a generic "dig a hole"
    # operation in an existing map. We represent potholes visually/physically
    # using shallow dark/flat props. The controller can later use their
    # locations as ground-truth pothole objects for validation.
    pothole_bp = find_blueprint(bp, [
        "static.prop.tire",
        "static.prop.roadwork",
        "static.prop.*",
    ])

    pothole_locations = []
    for i in range(POTHOLES):
        forward = random.uniform(12, ROAD_LENGTH_M - 5)
        lateral = random.uniform(-4.0, 4.0)

        # Keep a small minimum distance between potholes.
        loc = make_location(base, forward, lateral, 0.01)
        if any(dist(loc, p) < 5.0 for p in pothole_locations):
            continue

        pothole_locations.append(loc)

        if pothole_bp:
            scale = random.uniform(0.7, 1.6)
            a = spawn_prop(
                world,
                pothole_bp,
                carla.Transform(
                    loc,
                    carla.Rotation(yaw=random.uniform(0, 360)),
                ),
            )
            if a:
                try:
                    a.set_simulate_physics(False)
                except Exception:
                    pass
                actors.append(a)

    # Store ground truth for later perception evaluation.
    world._unstructured_potholes = pothole_locations

    # ------------------------------------------------------------------
    # 3. ROAD DEBRIS
    # ------------------------------------------------------------------
    debris_bp = find_blueprint(bp, [
        "static.prop.barrel",
        "static.prop.box",
        "static.prop.constructioncone",
        "static.prop.*",
    ])

    if debris_bp:
        for i in range(DEBRIS):
            forward = random.uniform(15, ROAD_LENGTH_M)
            lateral = random.uniform(-4.5, 4.5)
            loc = make_location(base, forward, lateral, random.uniform(0.05, 0.25))
            rot = carla.Rotation(yaw=random.uniform(0, 360))
            a = spawn_prop(world, debris_bp, carla.Transform(loc, rot))
            if a:
                actors.append(a)

    # ------------------------------------------------------------------
    # 4. IRREGULAR PARKED / SLOW VEHICLES
    # ------------------------------------------------------------------
    vehicle_bps = bp.filter("vehicle.*")
    usable = [v for v in vehicle_bps if not v.id.startswith("vehicle.carlamotors")]
    random.shuffle(usable)

    spawn_points = []
    for i in range(PARKED_VEHICLES):
        forward = random.uniform(30, ROAD_LENGTH_M - 10)
        lateral = random.choice([-1, 1]) * random.uniform(2.5, 5.0)
        loc = make_location(base, forward, lateral, 0.2)
        yaw = base.rotation.yaw + random.uniform(-25, 25)
        spawn_points.append(carla.Transform(loc, carla.Rotation(yaw=yaw)))

    for transform, vbp in zip(spawn_points, usable[:PARKED_VEHICLES]):
        a = spawn_prop(world, vbp, transform)
        if a:
            try:
                a.set_simulate_physics(False)
            except Exception:
                pass
            actors.append(a)

    print()
    print("=" * 65)
    print("UNSTRUCTURED ROAD ENVIRONMENT CREATED")
    print("=" * 65)
    print(f"Road test length      : {ROAD_LENGTH_M:.0f} m")
    print(f"Pothole locations     : {len(pothole_locations)}")
    print(f"Rocks/roadside props  : {len(actors)} (combined)")
    print(f"Parked vehicles       : {PARKED_VEHICLES}")
    print("Lane following        : DISABLED")
    print("A* route following    : DISABLED")
    print("=" * 65)
    print()

    return actors, pothole_locations


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)
    world = client.get_world()

    print("CARLA:", client.get_server_version())
    print("MAP  :", world.get_map().name)

    # Spawn ego in a known road location.
    bp_lib = world.get_blueprint_library()
    teslas = bp_lib.filter("vehicle.tesla.model3")
    if not teslas:
        raise RuntimeError("Tesla Model 3 blueprint not found.")

    ego_bp = teslas[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No CARLA spawn points available.")

    ego = None
    # Try several points to avoid a blocked spawn.
    for sp in spawn_points[:30]:
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego:
            break

    if ego is None:
        raise RuntimeError("Could not spawn ego vehicle.")

    print(f"Ego vehicle spawned: {ego.id}")

    actors, potholes = create_unstructured_environment(world, ego)

    print("Environment is ready.")
    print("Keep this terminal running while testing main.py.")
    print("Press Ctrl+C to clean up.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ego.destroy()
        except Exception:
            pass
        for actor in actors:
            try:
                actor.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()

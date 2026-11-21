"""
load_unstructured_world.py

CARLA 0.9.15
----------------
Loads the custom OpenDRIVE road at runtime.

This is the important Step 1 change:
    Town10HD_Opt
        X
        ↓
    Custom OpenDRIVE road
        ✓ wide
        ✓ single unstructured driving surface
        ✓ NO roadMark/lane-marking records

CARLA's OpenDRIVE standalone generator creates the road mesh directly from
the XODR. It does not use Town10HD's baked road markings.

After this script starts, leave this terminal running.
"""

from pathlib import Path
import random
import time
import math
import carla

HOST = "127.0.0.1"
PORT = 2000
XODR_FILE = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "maps"
    / "unstructured_indian_road.xodr"
)

random.seed(42)

ROAD_LENGTH = 220.0
ROAD_WIDTH = 12.0


def find_bp(bp_lib, patterns):
    for pattern in patterns:
        bps = bp_lib.filter(pattern)
        if bps:
            return random.choice(bps)
    return None


def spawn(world, bp, transform):
    if bp is None:
        return None
    try:
        return world.try_spawn_actor(bp, transform)
    except Exception:
        return None


def add_environment(world, ego):
    bp_lib = world.get_blueprint_library()
    actors = []

    # Road center is y=0; driving surface is approximately y=-12..0.
    # Put props irregularly near edges and on the road.
    rock_bp = find_bp(bp_lib, [
        "static.prop.rock",
        "static.prop.barrel",
        "static.prop.box",
    ])

    cone_bp = find_bp(bp_lib, [
        "static.prop.constructioncone",
        "static.prop.trafficcone",
    ])

    tire_bp = find_bp(bp_lib, [
        "static.prop.tire",
    ])

    # Rocks / irregular roadside objects.
    if rock_bp:
        for _ in range(28):
            x = random.uniform(8, ROAD_LENGTH - 5)
            y = random.choice([
                random.uniform(-1.0, 1.0),
                random.uniform(-15.0, -12.0),
            ])
            z = 0.05
            a = spawn(
                world,
                rock_bp,
                carla.Transform(
                    carla.Location(x=x, y=y, z=z),
                    carla.Rotation(yaw=random.uniform(0, 360)),
                ),
            )
            if a:
                try:
                    a.set_simulate_physics(False)
                except Exception:
                    pass
                actors.append(a)

    # Debris / construction objects.
    if cone_bp:
        for _ in range(12):
            x = random.uniform(15, ROAD_LENGTH - 5)
            y = random.uniform(-11.0, -1.0)
            a = spawn(
                world,
                cone_bp,
                carla.Transform(
                    carla.Location(x=x, y=y, z=0.05),
                    carla.Rotation(yaw=random.uniform(0, 360)),
                ),
            )
            if a:
                actors.append(a)

    # Pothole visual markers.
    # These are deliberately flat and dark-looking props where available.
    # True mesh depressions require a custom road mesh/material.
    if tire_bp:
        potholes = []
        for _ in range(12):
            x = random.uniform(15, ROAD_LENGTH - 8)
            y = random.uniform(-10.5, -1.5)

            if any(abs(x-px) < 7 and abs(y-py) < 2 for px, py in potholes):
                continue

            potholes.append((x, y))

            a = spawn(
                world,
                tire_bp,
                carla.Transform(
                    carla.Location(x=x, y=y, z=0.02),
                    carla.Rotation(pitch=90, yaw=random.uniform(0, 360)),
                ),
            )
            if a:
                try:
                    a.set_simulate_physics(False)
                except Exception:
                    pass
                actors.append(a)

    print()
    print("=" * 70)
    print("CUSTOM UNSTRUCTURED ROAD LOADED")
    print("=" * 70)
    print("Road type             : Wide rural / unstructured")
    print("Road width            : 12 m")
    print("Lane markings         : NONE")
    print("Town10HD road         : NOT USED")
    print("A*                    : NOT USED")
    print("Lane following        : NOT USED")
    print("Rocks/debris          : added")
    print("Pothole markers       : added")
    print("=" * 70)
    print()
    return actors


def main():
    if not XODR_FILE.exists():
        raise FileNotFoundError(f"Missing XODR file: {XODR_FILE}")

    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)

    print("CARLA server:", client.get_server_version())
    print("Loading custom OpenDRIVE road...")

    xodr_xml = XODR_FILE.read_text(encoding="utf-8")

    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=50.0,
        wall_height=0.0,
        additional_width=0.6,
        smooth_junctions=True,
        enable_mesh_visibility=True,
    )

    world = client.generate_opendrive_world(
        xodr_xml,
        params,
        reset_settings=True,
    )

    print("CUSTOM MAP GENERATED")
    print("Map:", world.get_map().name)

    # Spawn directly on the generated road.
    bp_lib = world.get_blueprint_library()
    teslas = bp_lib.filter("vehicle.tesla.model3")
    if not teslas:
        raise RuntimeError("Tesla Model 3 blueprint not found.")

    ego_bp = teslas[0]

    # Reference road runs along +X. The 12 m lane is on the right side,
    # therefore y around -6 m is near its center.
    spawn_transform = carla.Transform(
        carla.Location(x=8.0, y=-6.0, z=0.6),
        carla.Rotation(yaw=0.0),
    )

    ego = world.try_spawn_actor(ego_bp, spawn_transform)

    if ego is None:
        # Try a few nearby heights if mesh collision needs adjustment.
        for z in (1.0, 1.5, 2.0):
            spawn_transform.location.z = z
            ego = world.try_spawn_actor(ego_bp, spawn_transform)
            if ego:
                break

    if ego is None:
        raise RuntimeError(
            "Could not spawn Tesla on the generated road. "
            "The XODR was loaded, but the spawn position needs adjustment."
        )

    actors = add_environment(world, ego)

    print(f"Ego vehicle: {ego.id}")
    print("Environment ready.")
    print("KEEP THIS TERMINAL RUNNING.")
    print("Open the CARLA window and inspect the road.")
    print("Press Ctrl+C here to clean up.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nCleaning up...")

    for a in actors:
        try:
            a.destroy()
        except Exception:
            pass

    try:
        ego.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()

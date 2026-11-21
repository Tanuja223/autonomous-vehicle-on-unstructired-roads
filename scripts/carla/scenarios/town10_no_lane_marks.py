import carla
import random
import time
import math

HOST = "127.0.0.1"
PORT = 2000

random.seed(42)

# ============================================================
# CONNECT TO CARLA
# ============================================================

client = carla.Client(HOST, PORT)
client.set_timeout(30.0)

print("Connecting to CARLA...")

# Load Town10HD layered map
world = client.load_world(
    "Town10HD_Opt",
    carla.MapLayer.All
)

print("Town10HD loaded.")

time.sleep(5)

print("CARLA version:", client.get_server_version())
print("Map:", world.get_map().name)


# ============================================================
# REMOVE DECALS
# ============================================================

print()
print("Removing Town10HD Decals...")

try:
    world.unload_map_layer(carla.MapLayer.Decals)
    print("SUCCESS: Decals layer unloaded.")
except Exception as e:
    print("Could not unload Decals:", e)


# ============================================================
# KEEP IMPORTANT ENVIRONMENT LAYERS
# ============================================================

print()
print("Town10HD environment:")
print("  Buildings   : ON")
print("  Foliage     : ON")
print("  Ground      : ON")
print("  Props       : ON")
print("  StreetLights: ON")
print("  Walls       : ON")
print("  ParkedCars  : ON")
print("  Decals      : OFF")
print()


# ============================================================
# WEATHER
# ============================================================

weather = carla.WeatherParameters(
    cloudiness=10.0,
    precipitation=0.0,
    precipitation_deposits=0.0,
    wind_intensity=8.0,
    sun_altitude_angle=55.0,
    sun_azimuth_angle=120.0,
    fog_density=0.0
)

world.set_weather(weather)


# ============================================================
# GET WAYPOINTS
# ============================================================

carla_map = world.get_map()

spawn_points = carla_map.get_spawn_points()

if not spawn_points:
    raise RuntimeError("No spawn points found.")

start_transform = random.choice(spawn_points)

start_wp = carla_map.get_waypoint(
    start_transform.location,
    project_to_road=True,
    lane_type=carla.LaneType.Driving
)

print("Starting waypoint:", start_wp.transform.location)


# ============================================================
# SPAWN EGO VEHICLE
# ============================================================

blueprints = world.get_blueprint_library()

vehicle_bp = None

for name in [
    "vehicle.tesla.model3",
    "vehicle.tesla.modely"
]:

    candidates = blueprints.filter(name)

    if candidates:
        vehicle_bp = candidates[0]
        break


if vehicle_bp is None:
    vehicle_bp = random.choice(
        blueprints.filter("vehicle.*")
    )


vehicle_transform = carla.Transform(
    carla.Location(
        x=start_wp.transform.location.x,
        y=start_wp.transform.location.y,
        z=start_wp.transform.location.z + 0.5
    ),
    start_wp.transform.rotation
)

ego = world.try_spawn_actor(
    vehicle_bp,
    vehicle_transform
)

if ego is None:
    raise RuntimeError(
        "Could not spawn ego vehicle."
    )

ego.set_autopilot(False)

print("Ego vehicle:", ego.id)


# ============================================================
# FIND ROAD SECTION
# ============================================================

route = []

current = start_wp

for i in range(70):

    route.append(current)

    next_points = current.next(5.0)

    if not next_points:
        break

    # Choose the waypoint that changes direction the least
    if len(next_points) == 1:
        current = next_points[0]

    else:

        forward = current.transform.get_forward_vector()

        best = None
        best_score = -999

        for candidate in next_points:

            candidate_forward = (
                candidate.transform.get_forward_vector()
            )

            score = (
                forward.x * candidate_forward.x
                + forward.y * candidate_forward.y
            )

            if score > best_score:
                best_score = score
                best = candidate

        current = best


print("Road section length:", len(route), "waypoints")


# ============================================================
# SPAWN IRREGULAR ROAD OBJECTS
# ============================================================

environment_actors = [ego]


def get_right_vector(transform):

    forward = transform.get_forward_vector()

    return carla.Vector3D(
        x=-forward.y,
        y=forward.x,
        z=0
    )


def spawn_prop(
    blueprint,
    location,
    rotation
):

    actor = world.try_spawn_actor(
        blueprint,
        carla.Transform(
            location,
            rotation
        )
    )

    if actor:
        environment_actors.append(actor)

    return actor


# ============================================================
# ROCKS / DEBRIS
# ============================================================

static_props = (
    blueprints.filter("static.prop.*")
)

print(
    "Available static props:",
    len(static_props)
)


for i in range(35):

    wp = random.choice(route)

    transform = wp.transform

    right = get_right_vector(transform)

    # Keep some objects beside the driving area
    lateral = random.choice([
        random.uniform(-6.0, -3.5),
        random.uniform(3.5, 6.0)
    ])

    forward_offset = random.uniform(-3, 3)

    location = (
        transform.location
        + right * lateral
        + transform.get_forward_vector() * forward_offset
    )

    location.z += 0.2

    if static_props:

        bp = random.choice(static_props)

        spawn_prop(
            bp,
            location,
            carla.Rotation(
                yaw=random.uniform(0, 360)
            )
        )


# ============================================================
# PARKED VEHICLES
# ============================================================

vehicle_bps = blueprints.filter("vehicle.*")

parked_count = 0

for i in range(6):

    wp = random.choice(route)

    transform = wp.transform

    right = get_right_vector(transform)

    lateral = random.choice([
        random.uniform(-8.0, -5.5),
        random.uniform(5.5, 8.0)
    ])

    location = (
        transform.location
        + right * lateral
    )

    location.z += 0.5

    if (
        location.distance(ego.get_location())
        < 15
    ):
        continue

    bp = random.choice(vehicle_bps)

    rotation = carla.Rotation(
        yaw=transform.rotation.yaw
        + random.choice([-90, 90])
    )

    actor = spawn_prop(
        bp,
        location,
        rotation
    )

    if actor:
        parked_count += 1


print(
    "Parked vehicles added:",
    parked_count
)


# ============================================================
# TRAFFIC CONES
# ============================================================

cone_bps = (
    blueprints.filter(
        "static.prop.trafficcone*"
    )
)

if cone_bps:

    for i in range(12):

        wp = random.choice(route)

        transform = wp.transform

        right = get_right_vector(transform)

        lateral = random.choice([
            random.uniform(-6, -4),
            random.uniform(4, 6)
        ])

        location = (
            transform.location
            + right * lateral
        )

        location.z += 0.1

        spawn_prop(
            random.choice(cone_bps),
            location,
            carla.Rotation(
                yaw=random.uniform(0, 360)
            )
        )


# ============================================================
# POSITION SPECTATOR
# ============================================================

time.sleep(2)

spectator = world.get_spectator()

ego_transform = ego.get_transform()

forward = ego_transform.get_forward_vector()

camera_location = (
    ego_transform.location
    - forward * 12
    + carla.Location(z=6)
)

camera_rotation = carla.Rotation(
    pitch=-15,
    yaw=ego_transform.rotation.yaw,
    roll=0
)

spectator.set_transform(
    carla.Transform(
        camera_location,
        camera_rotation
    )
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("       TOWN10HD UNSTRUCTURED ENVIRONMENT")
print("=" * 70)
print()
print("Map                 : Town10HD_Opt")
print("Buildings           : ON")
print("Trees / foliage     : ON")
print("Terrain             : ON")
print("Street lights       : ON")
print("Parked vehicles     : ON")
print("Road debris         : ON")
print("Rocks               : ON")
print("Cones               : ON")
print("Lane following      : OFF")
print("A* navigation       : OFF")
print("Town10HD Decals     : OFF")
print()
print("CHECK THE CARLA WINDOW.")
print()
print("If the lane markings disappeared,")
print("we have solved it without modifying Unreal.")
print()
print("Press Ctrl+C to stop.")
print("=" * 70)


# ============================================================
# KEEP RUNNING
# ============================================================

try:

    while True:

        if not ego.is_alive:
            print("Ego vehicle destroyed.")
            break

        time.sleep(1)

except KeyboardInterrupt:

    print()
    print("Stopping...")


# ============================================================
# CLEANUP
# ============================================================

print("Cleaning spawned actors...")

for actor in environment_actors:

    try:
        if actor.is_alive:
            actor.destroy()
    except:
        pass

print("Done.")

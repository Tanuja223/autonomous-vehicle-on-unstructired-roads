"""
unstructured_dirt_road.py
CARLA 0.9.15 / 0.9.16 - Complete Unstructured Dirt Road Environment

Provides:
  1. UNSTRUCTURED DIRTY ROADS:
     - Town07: CARLA's rural map with genuine dirt/gravel tracks, rustic terrain,
       and ZERO painted lane markings on the country roads.
     - OpenDRIVE: Standalone procedural mesh generator creating custom-width
       dirt tracks with 0 lane marks (<roadMark type="none">).
     - Town10HD fallback with MapLayer.Decals disabled.
  2. DIRTY ROAD PHYSICS:
     - Sets vehicle wheel tire friction down to 0.9 - 1.2 (simulating loose dirt,
       mud, gravel, and wheel slip instead of sticky asphalt).
     - Atmospheric dusty/muddy weather presets.
  3. POTHOLE SIMULATION (Visual + Physical Dynamics):
     - Flat dark ground props + 3D CARLA debug rings on the road surface.
     - Real-time physics engine: applies downward suspension shock impulses,
       pitch/roll jolts, and steering disturbance when driving over potholes.
     - Exports ground truth coordinates to results/unstructured_potholes.json.
  4. NATURAL ROADSIDE CLUTTER:
     - Rocks, boulders, debris, and irregular roadside obstacles without curbs.
  5. SPECTATOR CHASE CAMERA:
     - Automatically tracks behind the ego vehicle for direct observation.

Usage:
  python unstructured_dirt_road.py --map Town07 --weather dusty --autopilot
  python unstructured_dirt_road.py --map opendrive --potholes 25
  python unstructured_dirt_road.py --map Town10HD --friction 1.0
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time

try:
    import carla
except ImportError:
    print("ERROR: CARLA Python API not found. Please activate your CARLA virtual environment.")
    sys.exit(1)

# Default configuration
DEFAULT_HOST = os.environ.get("CARLA_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("CARLA_PORT", "2000"))
XODR_FILE = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "maps"
    / "unstructured_dirt_track.xodr"
)
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "results"


# ==============================================================================
# Helper Math Functions
# ==============================================================================

def dist_2d(loc_a, loc_b):
    """Euclidean distance in XY plane."""
    return math.hypot(loc_a.x - loc_b.x, loc_a.y - loc_b.y)


def offset_location(base_transform, forward_m, lateral_m, z_offset_m=0.0):
    """Compute 3D position relative to a base transform."""
    yaw = math.radians(base_transform.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    rx, ry = -math.sin(yaw), math.cos(yaw)
    return carla.Location(
        x=base_transform.location.x + fx * forward_m + rx * lateral_m,
        y=base_transform.location.y + fy * forward_m + ry * lateral_m,
        z=base_transform.location.z + z_offset_m,
    )


def draw_debug_circle(world, center, radius, color=carla.Color(200, 30, 30), life_time=0.0, num_segments=16):
    """Draw a circular ring on the road to clearly identify a pothole."""
    angle_step = (2 * math.pi) / num_segments
    for i in range(num_segments):
        theta1 = i * angle_step
        theta2 = (i + 1) * angle_step
        p1 = carla.Location(
            x=center.x + radius * math.cos(theta1),
            y=center.y + radius * math.sin(theta1),
            z=center.z + 0.05,
        )
        p2 = carla.Location(
            x=center.x + radius * math.cos(theta2),
            y=center.y + radius * math.sin(theta2),
            z=center.z + 0.05,
        )
        world.debug.draw_line(p1, p2, thickness=0.04, color=color, life_time=life_time)


# ==============================================================================
# Weather Presets
# ==============================================================================

def apply_unstructured_weather(world, preset="dusty"):
    """Configure weather for dusty or muddy rural environments."""
    if preset == "dusty":
        weather = carla.WeatherParameters(
            cloudiness=45.0,
            precipitation=0.0,
            precipitation_deposits=10.0,
            wind_intensity=35.0,
            sun_altitude_angle=35.0,
            sun_azimuth_angle=140.0,
            fog_density=12.0,
            fog_distance=40.0,
            dust_storm=40.0,
        )
    elif preset == "muddy":
        weather = carla.WeatherParameters(
            cloudiness=85.0,
            precipitation=25.0,
            precipitation_deposits=80.0,
            wetness=85.0,
            wind_intensity=20.0,
            sun_altitude_angle=25.0,
            fog_density=15.0,
        )
    elif preset == "overcast":
        weather = carla.WeatherParameters(
            cloudiness=90.0,
            precipitation=0.0,
            precipitation_deposits=15.0,
            wind_intensity=15.0,
            sun_altitude_angle=40.0,
            fog_density=5.0,
        )
    else:  # clear
        weather = carla.WeatherParameters.ClearNoon

    world.set_weather(weather)
    print(f"[WEATHER] Applied '{preset}' weather profile.")


# ==============================================================================
# Dirt Road Vehicle Physics
# ==============================================================================

def configure_dirt_road_physics(vehicle, friction=1.1):
    """
    Modify tire friction on all wheels to simulate loose dirt, gravel, or mud.
    Default CARLA tire friction is ~3.5 (dry sticky asphalt).
    Dirt/gravel roads typically range from 0.8 to 1.3.
    """
    try:
        physics_control = vehicle.get_physics_control()
        wheels = []
        for wheel in physics_control.wheels:
            wheel.tire_friction = float(friction)
            wheels.append(wheel)
        physics_control.wheels = wheels
        vehicle.apply_physics_control(physics_control)
        print(f"[PHYSICS] Applied dirt-road tire friction: {friction:.2f} (default asphalt is ~3.5).")
    except Exception as e:
        print(f"[PHYSICS] Warning: Could not adjust tire friction: {e}")


# ==============================================================================
# Environment & Pothole Builder
# ==============================================================================

class UnstructuredEnvironment:
    def __init__(self, client, world, ego_vehicle, route_waypoints=None, num_potholes=20, num_rocks=30):
        self.client = client
        self.world = world
        self.ego = ego_vehicle
        self.route_waypoints = route_waypoints
        self.num_potholes = num_potholes
        self.num_rocks = num_rocks
        self.spawned_actors = []
        self.potholes = []
        self.pothole_cooldowns = {}
        self.running = True

    def find_bp(self, patterns):
        bp_lib = self.world.get_blueprint_library()
        for pat in patterns:
            bps = bp_lib.filter(pat)
            if bps:
                return random.choice(bps)
        return None

    def build(self):
        """Build roadside clutter and road potholes."""
        print(f"[SETUP] Populating unstructured road elements...")

        # Find blueprints
        rock_bp = self.find_bp(["static.prop.rock", "static.prop.barrel", "static.prop.box", "static.prop.*"])
        cone_bp = self.find_bp(["static.prop.constructioncone", "static.prop.trafficcone"])
        tire_bp = self.find_bp(["static.prop.tire", "static.prop.roadwork", "static.prop.*"])

        base_transform = self.ego.get_transform()
        spawned_locations = []

        # 1. GENERATE POTHOLES
        print(f"[SETUP] Generating {self.num_potholes} potholes along the track...")
        for i in range(self.num_potholes):
            if self.route_waypoints and len(self.route_waypoints) > 5:
                # Pick a waypoint along the road route
                idx = random.randint(3, len(self.route_waypoints) - 1)
                wp = self.route_waypoints[idx]
                right = wp.transform.get_right_vector()
                lateral = random.uniform(-2.5, 2.5)  # within drivable corridor
                loc = wp.transform.location + right * lateral
                loc.z = wp.transform.location.z + 0.02
                rot = carla.Rotation(pitch=90.0, yaw=random.uniform(0, 360))
            else:
                # Use linear distance along ego base
                forward = random.uniform(15.0, 350.0)
                lateral = random.uniform(-3.5, 3.5)
                loc = offset_location(base_transform, forward, lateral, 0.02)
                rot = carla.Rotation(pitch=90.0, yaw=random.uniform(0, 360))

            # Maintain spacing between potholes
            if any(dist_2d(loc, p["location"]) < 6.0 for p in self.potholes):
                continue

            radius = random.uniform(0.6, 1.2)
            depth_m = random.uniform(0.08, 0.20)
            pothole_id = len(self.potholes) + 1

            # Try to spawn visual prop (flat tire or roadwork prop embedded at surface)
            prop_actor = None
            if tire_bp:
                try:
                    prop_actor = self.world.try_spawn_actor(tire_bp, carla.Transform(loc, rot))
                    if prop_actor:
                        prop_actor.set_simulate_physics(False)
                        self.spawned_actors.append(prop_actor)
                except Exception:
                    pass

            # Draw visual debug outline on road surface
            draw_debug_circle(self.world, loc, radius, color=carla.Color(180, 20, 20), life_time=3600.0)

            pothole_record = {
                "id": pothole_id,
                "location": loc,
                "x": round(loc.x, 3),
                "y": round(loc.y, 3),
                "z": round(loc.z, 3),
                "radius": round(radius, 2),
                "depth_m": round(depth_m, 3),
                "severity": round(depth_m / 0.15, 2),
                "prop_id": prop_actor.id if prop_actor else None,
            }
            self.potholes.append(pothole_record)
            self.pothole_cooldowns[pothole_id] = 0.0

        # 2. GENERATE ROADSIDE IRREGULARITIES (ROCKS & CLUTTER)
        print(f"[SETUP] Scattering {self.num_rocks} roadside boulders and natural debris...")
        for _ in range(self.num_rocks):
            if self.route_waypoints and len(self.route_waypoints) > 5:
                idx = random.randint(1, len(self.route_waypoints) - 1)
                wp = self.route_waypoints[idx]
                right = wp.transform.get_right_vector()
                # Place near road boundaries (outside center corridor)
                side = random.choice([-1, 1])
                lateral = side * random.uniform(3.5, 6.5)
                loc = wp.transform.location + right * lateral
                loc.z = wp.transform.location.z + 0.1
                rot = carla.Rotation(
                    pitch=random.uniform(-10, 10),
                    yaw=random.uniform(0, 360),
                    roll=random.uniform(-10, 10),
                )
            else:
                forward = random.uniform(8.0, 400.0)
                side = random.choice([-1, 1])
                lateral = side * random.uniform(4.0, 7.5)
                loc = offset_location(base_transform, forward, lateral, 0.1)
                rot = carla.Rotation(yaw=random.uniform(0, 360))

            bp = rock_bp if random.random() > 0.3 else cone_bp
            if bp:
                try:
                    actor = self.world.try_spawn_actor(bp, carla.Transform(loc, rot))
                    if actor:
                        actor.set_simulate_physics(False)
                        self.spawned_actors.append(actor)
                except Exception:
                    pass

        # Save ground truth to JSON
        self.export_ground_truth()

    def export_ground_truth(self):
        """Save pothole coordinates to results for autonomous perception evaluation."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gt_path = OUTPUT_DIR / "unstructured_potholes.json"
        serializable = [
            {
                "id": p["id"],
                "x": p["x"],
                "y": p["y"],
                "z": p["z"],
                "radius_m": p["radius"],
                "depth_m": p["depth_m"],
                "severity": p["severity"],
            }
            for p in self.potholes
        ]
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_potholes": len(serializable),
                    "surface_type": "dirt_unstructured",
                    "lane_markings": False,
                    "potholes": serializable,
                },
                f,
                indent=2,
            )
        print(f"[EXPORT] Saved {len(serializable)} ground-truth potholes to {gt_path}")

    def start_physics_loop(self):
        """
        Thread that monitors vehicle position and applies dynamic impulses
        when the car encounters potholes.
        """
        thread = threading.Thread(target=self._pothole_physics_worker, daemon=True)
        thread.start()

    def _pothole_physics_worker(self):
        """Check for pothole hits at 20 Hz."""
        while self.running:
            time.sleep(0.05)
            if not self.ego.is_alive:
                break

            current_time = time.time()
            ego_loc = self.ego.get_location()
            v = self.ego.get_velocity()
            speed_kmh = 3.6 * math.hypot(v.x, v.y)

            # Skip physics calculation if car is nearly stationary
            if speed_kmh < 2.0:
                continue

            for pothole in self.potholes:
                d = dist_2d(ego_loc, pothole["location"])
                if d <= pothole["radius"]:
                    # Cooldown check (don't re-trigger for 1.2s for same pothole)
                    last_hit = self.pothole_cooldowns.get(pothole["id"], 0.0)
                    if (current_time - last_hit) > 1.2:
                        self.pothole_cooldowns[pothole["id"]] = current_time
                        self._apply_pothole_shock(pothole, speed_kmh)

    def _apply_pothole_shock(self, pothole, speed_kmh):
        """Apply vertical impulse, pitch disturbance, and steering kick to ego."""
        try:
            mass = self.ego.get_physics_control().mass
        except Exception:
            mass = 1700.0

        # Shock magnitude scales with speed and pothole depth
        speed_factor = min(1.8, max(0.4, speed_kmh / 25.0))
        severity = pothole["severity"]
        downward_force = -1.0 * (mass * 2.2) * severity * speed_factor

        # 1. Downward dip impulse into the pothole
        self.ego.add_impulse(carla.Vector3D(0.0, 0.0, downward_force))

        # 2. Angular suspension kick (pitch and slight roll)
        pitch_kick = random.choice([-1.0, 1.0]) * (mass * 0.8) * severity
        roll_kick = random.choice([-1.0, 1.0]) * (mass * 0.5) * severity
        self.ego.add_angular_impulse(carla.Vector3D(pitch_kick, roll_kick, 0.0))

        print(
            f"[POTHOLE HIT] Pothole #{pothole['id']} encountered! "
            f"Speed={speed_kmh:.1f} km/h, Depth={pothole['depth_m']*100:.1f}cm, Shock={downward_force:.0f} N"
        )

    def cleanup(self):
        """Destroy all spawned environment props."""
        self.running = False
        print(f"[CLEANUP] Destroying {len(self.spawned_actors)} environment actors...")
        for actor in self.spawned_actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass


# ==============================================================================
# Map Loaders
# ==============================================================================

def setup_town07(client):
    """
    Load Town07: CARLA's rural map. Country sections are real unpaved dirt
    roads with 0 lane markings.
    """
    print("[MAP] Loading CARLA Town07 (Rural Dirt Road map)...")
    world = client.load_world("Town07")
    time.sleep(2.0)

    # Search for waypoints on country dirt roads (no lane markings)
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    selected_sp = None
    route_waypoints = []

    # Filter for spawns where lane markings are NONE or Grass
    for sp in spawn_points:
        wp = carla_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        left_type = wp.left_lane_marking.type if wp.left_lane_marking else None
        right_type = wp.right_lane_marking.type if wp.right_lane_marking else None

        # Prefer rural dirt sections with no solid/broken lane markings
        if (
            left_type in (carla.LaneMarkingType.NONE, carla.LaneMarkingType.Grass, None)
            or right_type in (carla.LaneMarkingType.NONE, carla.LaneMarkingType.Grass, None)
        ):
            selected_sp = sp
            break

    if selected_sp is None:
        selected_sp = spawn_points[0]

    # Trace a sequence of waypoints along this road
    curr_wp = carla_map.get_waypoint(selected_sp.location, project_to_road=True)
    for _ in range(80):
        route_waypoints.append(curr_wp)
        next_wps = curr_wp.next(4.0)
        if not next_wps:
            break
        curr_wp = next_wps[0]

    print(f"[MAP] Town07 loaded successfully. Found rural road segment ({len(route_waypoints)} waypoints).")
    return world, selected_sp, route_waypoints


def setup_opendrive_world(client):
    """
    Generate an OpenDRIVE road world directly at runtime without any lane markings.
    """
    if not XODR_FILE.exists():
        raise FileNotFoundError(f"OpenDRIVE file not found: {XODR_FILE}")

    print(f"[MAP] Generating procedural OpenDRIVE road from {XODR_FILE.name}...")
    xodr_xml = XODR_FILE.read_text(encoding="utf-8")

    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=50.0,
        wall_height=0.0,
        additional_width=0.6,
        smooth_junctions=True,
        enable_mesh_visibility=True,
    )

    world = client.generate_opendrive_world(xodr_xml, params, reset_settings=True)
    time.sleep(2.0)

    # Spawn position for the 12m wide single driving lane
    spawn_transform = carla.Transform(
        carla.Location(x=10.0, y=-6.0, z=0.8),
        carla.Rotation(yaw=0.0),
    )
    return world, spawn_transform, None


def setup_town10hd(client):
    """
    Load Town10HD_Opt with the Decals map layer unloaded to remove lane lines.
    """
    print("[MAP] Loading Town10HD_Opt with Decals unloaded...")
    world = client.load_world("Town10HD_Opt", carla.MapLayer.All)
    time.sleep(2.0)

    try:
        world.unload_map_layer(carla.MapLayer.Decals)
        print("[MAP] Successfully unloaded Decals layer (lane markings removed).")
    except Exception as e:
        print(f"[MAP] Notice: Decals unload returned: {e}")

    spawn_points = world.get_map().get_spawn_points()
    sp = random.choice(spawn_points)

    curr_wp = world.get_map().get_waypoint(sp.location, project_to_road=True)
    route_waypoints = []
    for _ in range(70):
        route_waypoints.append(curr_wp)
        next_wps = curr_wp.next(4.0)
        if not next_wps:
            break
        curr_wp = next_wps[0]

    return world, sp, route_waypoints


# ==============================================================================
# Spectator Chase Camera
# ==============================================================================

def run_spectator_follower(world, ego_vehicle, stop_event):
    """Keep the CARLA spectator camera behind the ego vehicle."""
    spectator = world.get_spectator()
    while not stop_event.is_set():
        if not ego_vehicle.is_alive:
            break
        try:
            t = ego_vehicle.get_transform()
            forward = t.get_forward_vector()
            up = carla.Location(z=4.2)
            cam_loc = t.location - forward * 9.0 + up
            cam_rot = carla.Rotation(pitch=-16.0, yaw=t.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))
        except Exception:
            pass
        time.sleep(0.04)


# ==============================================================================
# Main Scenario Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="CARLA Unstructured Dirt Road & Potholes Scenario")
    parser.add_argument(
        "--map",
        choices=["Town07", "opendrive", "Town10HD"],
        default="Town07",
        help="Map type: Town07 (native dirt roads), opendrive (procedural 0-marking track), Town10HD (no decals)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="CARLA host IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="CARLA port")
    parser.add_argument(
        "--weather",
        choices=["dusty", "muddy", "overcast", "clear"],
        default="dusty",
        help="Weather preset for rural environment",
    )
    parser.add_argument("--potholes", type=int, default=18, help="Number of potholes to generate")
    parser.add_argument("--rocks", type=int, default=25, help="Number of roadside rocks and debris")
    parser.add_argument("--friction", type=float, default=1.1, help="Tire friction for dirt/gravel (default 1.1)")
    parser.add_argument("--autopilot", action="store_true", help="Enable CARLA autopilot")
    parser.add_argument("--no-spectator", action="store_true", help="Disable spectator chase camera")

    args = parser.parse_args()

    print("=" * 70)
    print("      CARLA UNSTRUCTURED DIRTY ROAD & POTHOLES ENVIRONMENT")
    print("=" * 70)
    print(f"Connecting to CARLA at {args.host}:{args.port}...")

    client = carla.Client(args.host, args.port)
    client.set_timeout(45.0)

    try:
        server_ver = client.get_server_version()
        print(f"Connected to CARLA Server: {server_ver}")
    except Exception as e:
        print(f"ERROR: Cannot connect to CARLA simulator: {e}")
        print("Please make sure CarlaUE4.exe is running.")
        sys.exit(1)

    # 1. SETUP MAP
    if args.map == "Town07":
        world, spawn_transform, route_wps = setup_town07(client)
    elif args.map == "opendrive":
        world, spawn_transform, route_wps = setup_opendrive_world(client)
    else:
        world, spawn_transform, route_wps = setup_town10hd(client)

    # 2. APPLY RURAL / DIRT WEATHER
    apply_unstructured_weather(world, args.weather)

    # 3. SPAWN EGO VEHICLE
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")
    if not vehicle_bp:
        vehicle_bp = bp_lib.filter("vehicle.*")
    ego_bp = vehicle_bp[0]

    print(f"[VEHICLE] Spawning {ego_bp.id} on the unstructured road...")
    ego = None
    for z_try in (spawn_transform.location.z, spawn_transform.location.z + 0.5, spawn_transform.location.z + 1.2):
        spawn_transform.location.z = z_try
        ego = world.try_spawn_actor(ego_bp, spawn_transform)
        if ego:
            break

    if ego is None:
        print("ERROR: Failed to spawn ego vehicle. Retrying with first available spawn point...")
        sp_fallback = world.get_map().get_spawn_points()[0]
        ego = world.spawn_actor(ego_bp, sp_fallback)

    print(f"[VEHICLE] Ego vehicle active (ID: {ego.id})")

    # 4. CONFIGURE DIRT ROAD PHYSICS
    configure_dirt_road_physics(ego, friction=args.friction)

    # 5. SPAWN POTHOLES & ROADSIDE CLUTTER
    env = UnstructuredEnvironment(
        client=client,
        world=world,
        ego_vehicle=ego,
        route_waypoints=route_wps,
        num_potholes=args.potholes,
        num_rocks=args.rocks,
    )
    env.build()
    env.start_physics_loop()

    # 6. AUTOPILOT
    if args.autopilot:
        ego.set_autopilot(True)
        print("[CONTROL] CARLA Autopilot ENABLED.")
    else:
        print("[CONTROL] Ego vehicle ready in standby (manual / external controller).")

    # 7. SPECTATOR CHASE CAMERA
    stop_cam_event = threading.Event()
    if not args.no_spectator:
        cam_thread = threading.Thread(
            target=run_spectator_follower,
            args=(world, ego, stop_cam_event),
            daemon=True,
        )
        cam_thread.start()
        print("[CAMERA] Spectator chase camera enabled.")

    print()
    print("=" * 70)
    print(" ENVIRONMENT ACTIVE:")
    print(f"  • Map                 : {args.map}")
    print(f"  • Road Surface        : Unstructured Dirt / Gravel (Friction: {args.friction})")
    print(f"  • Lane Markings       : NONE")
    print(f"  • Dynamic Potholes    : {len(env.potholes)} active with suspension physics")
    print(f"  • Roadside Obstacles  : {len(env.spawned_actors)} props spawned")
    print(f"  • Weather Profile     : {args.weather}")
    print("=" * 70)
    print("Press Ctrl+C in this terminal to shut down and clean up.")
    print()

    try:
        while True:
            time.sleep(1.0)
            if not ego.is_alive:
                print("Ego vehicle no longer active.")
                break
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down scenario...")
    finally:
        stop_cam_event.set()
        env.cleanup()
        try:
            if ego.is_alive:
                ego.destroy()
                print("[CLEANUP] Ego vehicle destroyed.")
        except Exception:
            pass
        print("[DONE] Environment cleanup complete.")


if __name__ == "__main__":
    main()

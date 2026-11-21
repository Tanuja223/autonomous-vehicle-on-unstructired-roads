
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import carla


# ============================================================
# CARLA SETTINGS
# ============================================================

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000

vehicle = None
world = None


# ============================================================
# CONNECT TO CARLA
# ============================================================

def connect_carla():

    global world

    print("\nConnecting to CARLA...")

    client = carla.Client(
        CARLA_HOST,
        CARLA_PORT
    )

    client.set_timeout(30.0)

    world = client.get_world()

    print("CONNECTED TO CARLA")
    print("Map:", world.get_map().name)

    return client


# ============================================================
# SPAWN VEHICLE
# ============================================================
def spawn_vehicle():

    global vehicle

    blueprint_library = world.get_blueprint_library()

    vehicles = blueprint_library.filter(
        "vehicle.tesla.model3"
    )

    if not vehicles:
        print("Tesla Model 3 not found")
        return None

    vehicle_bp = vehicles[0]

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        print("No spawn points found")
        return None

    spawn_point = spawn_points[0]

    vehicle = world.try_spawn_actor(
        vehicle_bp,
        spawn_point
    )

    if vehicle:

        print(
            f"Vehicle spawned: {vehicle.id}"
        )

        vehicle.set_autopilot(False)

        # Enable physics
        vehicle.set_simulate_physics(True)

        print("Vehicle ready for waypoint navigation.")

    else:

        print("Vehicle spawn failed")

    return vehicle


# ============================================================
# DRIVE VEHICLE TO CARLA DESTINATION
# ============================================================
def drive_to_carla_destination():

    if vehicle is None:
        print("Vehicle does not exist")
        return

    print("\n========================================")
    print("STARTING CARLA WAYPOINT NAVIGATION")
    print("========================================")

    map_obj = world.get_map()

    spawn_points = map_obj.get_spawn_points()

    if len(spawn_points) < 2:
        print("Not enough spawn points")
        return

    # -------------------------------------------------------
    # Current vehicle location
    # -------------------------------------------------------

    start_location = vehicle.get_location()

    print(
        f"Vehicle start: "
        f"X={start_location.x:.2f}, "
        f"Y={start_location.y:.2f}"
    )

    # -------------------------------------------------------
    # CARLA destination
    # -------------------------------------------------------
    # For now this is a CARLA waypoint.
    # Later we will map the OSM GPS destination to CARLA.
    # -------------------------------------------------------

    destination = spawn_points[10]

    destination_location = destination.location

    print(
        f"CARLA destination: "
        f"X={destination_location.x:.2f}, "
        f"Y={destination_location.y:.2f}"
    )

    # -------------------------------------------------------
    # BasicAgent
    # -------------------------------------------------------

    agent = BasicAgent(
        vehicle,
        target_speed=25
    )

    # Set destination
    agent.set_destination(destination_location)

    print("\nWaypoint route created.")
    print("Vehicle following CARLA waypoints...")
    print("Target speed: 25 km/h")

    # -------------------------------------------------------
    # Drive
    # -------------------------------------------------------

    while True:

        # Check vehicle still exists
        if not vehicle.is_alive:
            print("Vehicle no longer exists")
            break

        # Current location
        current_location = vehicle.get_location()

        # Distance
        distance = current_location.distance(
            destination_location
        )

        # Current speed
        velocity = vehicle.get_velocity()

        speed = (
            3.6 *
            (
                velocity.x ** 2 +
                velocity.y ** 2 +
                velocity.z ** 2
            ) ** 0.5
        )

        print(
            f"\rDistance: {distance:6.2f} m | "
            f"Speed: {speed:5.1f} km/h",
            end=""
        )

        # ---------------------------------------------------
        # Destination reached
        # ---------------------------------------------------

        if agent.done() or distance < 5:

            print("\n\nDESTINATION REACHED!")

            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0
                )
            )

            break

        # ---------------------------------------------------
        # Get waypoint-based control
        # ---------------------------------------------------

        control = agent.run_step()

        # Keep CARLA from using manual gear shifting
        control.manual_gear_shift = False

        # Apply control
        vehicle.apply_control(control)

        time.sleep(0.05)

def setup_collision_sensor():
    collision_bp = world.get_blueprint_library().find(
        "sensor.other.collision"
    )

    sensor_transform = carla.Transform(
        carla.Location(x=2.0, z=1.0)
    )

    sensor = world.spawn_actor(
        collision_bp,
        sensor_transform,
        attach_to=vehicle
    )

    def collision_callback(event):
        print("\n⚠️ COLLISION DETECTED!")

        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0
            )
        )

    sensor.listen(collision_callback)

    return sensor

# ============================================================
# RECEIVE OSM DESTINATION
# ============================================================
def follow_vehicle():

    spectator = world.get_spectator()

    while vehicle is not None:

        try:

            transform = vehicle.get_transform()

            # Camera behind and above the vehicle
            camera_location = (
                transform.location
                + carla.Location(
                    x=-8,
                    z=5
                )
            )

            # Look in the same direction as vehicle
            camera_rotation = transform.rotation

            spectator.set_transform(
                carla.Transform(
                    camera_location,
                    camera_rotation
                )
            )

            time.sleep(0.05)

        except Exception:
            break
class DestinationHandler(
    BaseHTTPRequestHandler
):


    def do_OPTIONS(self):

        self.send_response(200)

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, OPTIONS"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )

        self.end_headers()


    def do_POST(self):

        if self.path != "/destination":

            self.send_response(404)

            self.end_headers()

            return


        content_length = int(
            self.headers.get(
                "Content-Length",
                0
            )
        )


        body = self.rfile.read(
            content_length
        )


        try:

            data = json.loads(
                body.decode("utf-8")
            )


            print("\n")
            print("=" * 50)
            print("OSM DESTINATION RECEIVED")
            print("=" * 50)


            print(
                "Start latitude:",
                data.get("start_lat")
            )

            print(
                "Start longitude:",
                data.get("start_lon")
            )

            print(
                "Destination latitude:",
                data.get("destination_lat")
            )

            print(
                "Destination longitude:",
                data.get("destination_lon")
            )


            print(
                "Distance/route data received."
            )


            # Start CARLA driving
            threading.Thread(
                target=drive_to_carla_destination,
                daemon=True
            ).start()


            response = {

                "status": "success",

                "message":
                    "Destination received. "
                    "CARLA vehicle started."

            }


            response_bytes = json.dumps(
                response
            ).encode("utf-8")


            self.send_response(200)

            self.send_header(
                "Content-Type",
                "application/json"
            )

            self.send_header(
                "Access-Control-Allow-Origin",
                "*"
            )

            self.send_header(
                "Content-Length",
                str(len(response_bytes))
            )

            self.end_headers()

            self.wfile.write(
                response_bytes
            )


        except Exception as e:

            print(
                "ERROR:",
                e
            )

            self.send_response(500)

            self.send_header(
                "Access-Control-Allow-Origin",
                "*"
            )

            self.end_headers()


    def log_message(
        self,
        format,
        *args
    ):

        return


# ============================================================
# START SERVER
# ============================================================

def start_server():

    server = HTTPServer(
        ("127.0.0.1", 5001),
        DestinationHandler
    )


    print(
        "\nOSM → CARLA server running"
    )

    print(
        "http://127.0.0.1:5001"
    )


    server.serve_forever()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        connect_carla()

        spawn_vehicle()

        start_server()

    except KeyboardInterrupt:

        print("\nStopping...")

        if vehicle:

            vehicle.destroy()

    except Exception as e:

        print(
            "\nCARLA ERROR:",
            e
        )

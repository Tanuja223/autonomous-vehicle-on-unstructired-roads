import carla
import time

HOST = "127.0.0.1"
PORT = 2000

print("Connecting to CARLA...")

client = carla.Client(HOST, PORT)
client.set_timeout(10.0)

world = client.get_world()

print("Connected:", world.get_map().name)

# Get Tesla
blueprints = world.get_blueprint_library()
vehicle_bp = blueprints.find("vehicle.tesla.model3")

# Get spawn points
spawn_points = world.get_map().get_spawn_points()

vehicle = world.try_spawn_actor(
    vehicle_bp,
    spawn_points[0]
)

if vehicle is None:
    print("Vehicle spawn failed.")
    exit()

print("Vehicle spawned:", vehicle.id)

# Enable physics
vehicle.set_simulate_physics(True)

# Disable autopilot
vehicle.set_autopilot(False)

print("Starting vehicle...")

# Drive forward
for i in range(200):

    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.35,
            steer=0.0,
            brake=0.0
        )
    )

    velocity = vehicle.get_velocity()

    speed = (
        velocity.x ** 2 +
        velocity.y ** 2 +
        velocity.z ** 2
    ) ** 0.5

    speed_kmh = speed * 3.6

    print(
        f"\rSpeed: {speed_kmh:.1f} km/h",
        end=""
    )

    time.sleep(0.05)

# Stop
vehicle.apply_control(
    carla.VehicleControl(
        throttle=0.0,
        steer=0.0,
        brake=1.0
    )
)

print("\nVehicle stopped.")

time.sleep(2)

vehicle.destroy()

print("Test completed.")
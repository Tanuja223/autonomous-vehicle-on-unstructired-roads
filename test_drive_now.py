
import carla
import time
import math

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
world = client.get_world()

vehicles = world.get_actors().filter("vehicle.*")
if not vehicles:
    raise RuntimeError("No vehicle found in CARLA.")

# Find the Tesla created by your scenario.
vehicle = next(
    (v for v in vehicles if "tesla" in v.type_id.lower()),
    vehicles[0]
)

print("Vehicle:", vehicle.id, vehicle.type_id)
print("Before:", vehicle.get_location())
print("Control before:", vehicle.get_control())

vehicle.set_autopilot(False)
vehicle.apply_control(
    carla.VehicleControl(
        throttle=0.7,
        steer=0.0,
        brake=0.0,
        hand_brake=False,
        reverse=False
    )
)

for i in range(30):
    time.sleep(0.2)
    velocity = vehicle.get_velocity()
    speed = 3.6 * math.sqrt(
        velocity.x**2 + velocity.y**2 + velocity.z**2
    )
    print(f"{i:02d}: speed={speed:.2f} km/h, "
          f"location={vehicle.get_location()}")

vehicle.apply_control(carla.VehicleControl(brake=1.0))
print("Movement test finished.")

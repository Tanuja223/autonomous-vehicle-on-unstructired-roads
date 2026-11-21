
import carla
import time

client = carla.Client("localhost", 2000)
client.set_timeout(10)

world = client.get_world()
vehicles = world.get_actors().filter("vehicle.tesla.model3")

print("Tesla count:", len(vehicles))

if not vehicles:
    print("Tesla not found. Spawn the vehicle first.")
else:
    vehicle = vehicles[0]

    vehicle.set_autopilot(False)

    print("Moving Tesla forward...")

    try:
        for i in range(100):
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.3,
                    steer=0.0,
                    brake=0.0
                )
            )
            time.sleep(0.1)

    finally:
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0
            )
        )

    print("Movement test complete.")

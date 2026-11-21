
import carla
import math

client = carla.Client("localhost", 2000)
client.set_timeout(10)

world = client.get_world()
vehicles = world.get_actors().filter("vehicle.tesla.model3")

if not vehicles:
    print("No Tesla found. Spawn the vehicle first.")
else:
    ego = vehicles[0]
    transform = ego.get_transform()
    velocity = ego.get_velocity()

    speed = math.sqrt(
        velocity.x**2 + velocity.y**2 + velocity.z**2
    )

    print("\n--- CARLA EGO STATE ---")
    print("Vehicle ID:", ego.id)
    print("Location (m):", transform.location)
    print("Rotation (degrees):", transform.rotation)
    print("Speed (m/s):", round(speed, 3))
    print("Speed (km/h):", round(speed * 3.6, 2))

    print("\n--- NEARBY VEHICLES ---")
    count = 0

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego.id:
            continue

        distance = transform.location.distance(
            actor.get_location()
        )

        if distance <= 50:
            count += 1
            print(
                "ID:", actor.id,
                "| Type:", actor.type_id,
                "| Distance (m):", round(distance, 2)
            )

    print("Nearby vehicles:", count)

import carla
import time

print("Connecting to CARLA...")

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(30.0)

world = client.get_world()

print("Connected:", world.get_map().name)

# Get a vehicle
blueprints = world.get_blueprint_library()
vehicle_bp = blueprints.find("vehicle.tesla.model3")

spawn_points = world.get_map().get_spawn_points()
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])

if vehicle is None:
    print("Vehicle spawn failed")
    exit()

print("Vehicle spawned:", vehicle.id)

# Create a VERY small RGB camera
camera_bp = blueprints.find("sensor.camera.rgb")

camera_bp.set_attribute("image_size_x", "64")
camera_bp.set_attribute("image_size_y", "48")
camera_bp.set_attribute("fov", "90")
camera_bp.set_attribute("sensor_tick", "0.50")

camera_transform = carla.Transform(
    carla.Location(x=1.5, z=2.0)
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle
)

print("RGB CAMERA SPAWNED:", camera.id)


def image_callback(image):
    print("Camera frame received:", image.frame)


camera.listen(image_callback)

print("Camera test running...")
print("Waiting 10 seconds...")

try:
    time.sleep(10)
finally:
    print("Cleaning up...")

    camera.stop()
    camera.destroy()
    vehicle.destroy()

print("CAMERA TEST FINISHED")
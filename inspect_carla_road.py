
import carla

client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.get_world()

vehicles = world.get_actors().filter("vehicle.tesla.model3")

if not vehicles:
    print("Tesla not found")
else:
    ego = vehicles[0]
    transform = ego.get_transform()
    forward = transform.get_forward_vector()

    print("Vehicle ID:", ego.id)
    print("Forward vector:",
          round(forward.x, 4),
          round(forward.y, 4),
          round(forward.z, 4))

    waypoint = world.get_map().get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if waypoint:
        print("Road waypoint:", waypoint.transform.location)
        print("Road heading:", waypoint.transform.rotation.yaw)
        print("Road ID:", waypoint.road_id)
        print("Lane ID:", waypoint.lane_id)
        print("Lane width:", waypoint.lane_width)
    else:
        print("No driving-road waypoint found nearby")


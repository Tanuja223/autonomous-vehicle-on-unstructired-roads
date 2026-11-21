
import carla

client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.get_world()
road_map = world.get_map()

vehicles = world.get_actors().filter("vehicle.tesla.model3")
if not vehicles:
    raise RuntimeError("Tesla not found")

ego = vehicles[0]
waypoint = road_map.get_waypoint(
    ego.get_location(),
    project_to_road=True,
    lane_type=carla.LaneType.Driving
)

print("Starting road:", waypoint.road_id)
print("Starting lane:", waypoint.lane_id)
print("\nRoute waypoints ahead:")

for i in range(1, 11):
    next_points = waypoint.next(3.0)
    if not next_points:
        print("Route ended at step", i)
        break

    # Choose the continuation closest to the current heading.
    waypoint = min(
        next_points,
        key=lambda p: abs(
            (p.transform.rotation.yaw -
             ego.get_transform().rotation.yaw + 180) % 360 - 180
        )
    )

    loc = waypoint.transform.location
    print(
        i,
        "x=", round(loc.x, 2),
        "y=", round(loc.y, 2),
        "road=", waypoint.road_id,
        "lane=", waypoint.lane_id
    )

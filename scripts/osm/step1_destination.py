import carla
import math
import time
import heapq


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

WAYPOINT_DISTANCE = 3.0
CRUISE_SPEED = 25.0          # km/h
ARRIVAL_DISTANCE = 5.0


# ============================================================
# CARLA CONNECTION
# ============================================================

def connect_to_carla():

    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    world = client.get_world()

    print("\nConnected to CARLA")
    print("Map:", world.get_map().name)

    return client, world


# ============================================================
# DISTANCE
# ============================================================

def distance(a, b):

    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


# ============================================================
# BUILD ROAD GRAPH
# ============================================================

def build_road_graph(world):

    print("\nBuilding CARLA road graph...")

    topology = world.get_map().get_topology()

    graph = {}
    waypoint_map = {}

    for start, end in topology:

        start_id = start.id
        end_id = end.id

        waypoint_map[start_id] = start
        waypoint_map[end_id] = end

        if start_id not in graph:
            graph[start_id] = []

        graph[start_id].append(end_id)

    print("Road graph nodes:", len(graph))

    return graph, waypoint_map


# ============================================================
# HEURISTIC FOR A*
# ============================================================

def heuristic(waypoint, goal):

    return distance(
        waypoint.transform.location,
        goal.transform.location
    )


# ============================================================
# A* SEARCH
# ============================================================

def astar(graph, waypoint_map, start, goal):

    start_id = start.id
    goal_id = goal.id

    open_set = []

    heapq.heappush(
        open_set,
        (0.0, start_id)
    )

    came_from = {}

    g_score = {
        start_id: 0.0
    }

    f_score = {
        start_id: heuristic(start, goal)
    }

    visited = set()

    while open_set:

        _, current_id = heapq.heappop(open_set)

        if current_id in visited:
            continue

        visited.add(current_id)

        if current_id == goal_id:

            path = []

            current = goal_id

            while current in came_from:

                path.append(
                    waypoint_map[current]
                )

                current = came_from[current]

            path.append(
                waypoint_map[start_id]
            )

            path.reverse()

            return path

        for neighbor_id in graph.get(current_id, []):

            current_wp = waypoint_map[current_id]
            neighbor_wp = waypoint_map[neighbor_id]

            edge_cost = distance(
                current_wp.transform.location,
                neighbor_wp.transform.location
            )

            tentative_g = (
                g_score[current_id] +
                edge_cost
            )

            if (
                neighbor_id not in g_score
                or tentative_g < g_score[neighbor_id]
            ):

                came_from[neighbor_id] = current_id

                g_score[neighbor_id] = tentative_g

                f = (
                    tentative_g +
                    heuristic(neighbor_wp, goal)
                )

                f_score[neighbor_id] = f

                heapq.heappush(
                    open_set,
                    (f, neighbor_id)
                )

    return []


# ============================================================
# FIND NEAREST TOPOLOGY WAYPOINT
# ============================================================

def nearest_waypoint(location, waypoint_map):

    best = None
    best_distance = float("inf")

    for waypoint in waypoint_map.values():

        d = distance(
            location,
            waypoint.transform.location
        )

        if d < best_distance:

            best_distance = d
            best = waypoint

    return best


# ============================================================
# CHOOSE DESTINATION
# ============================================================

def choose_destination(world, vehicle, waypoint_map):

    current_location = vehicle.get_location()

    candidates = list(waypoint_map.values())

    # Only consider sufficiently far destinations
    far_candidates = [
        wp
        for wp in candidates
        if distance(
            current_location,
            wp.transform.location
        ) > 100.0
    ]

    if not far_candidates:

        far_candidates = candidates

    # Choose a destination that is far away
    destination = max(
        far_candidates,
        key=lambda wp: distance(
            current_location,
            wp.transform.location
        )
    )

    return destination


# ============================================================
# DRAW DESTINATION
# ============================================================

def draw_destination(world, destination):

    world.debug.draw_point(
        destination.transform.location,
        size=0.5,
        color=carla.Color(255, 0, 0),
        life_time=0.2
    )


# ============================================================
# DRAW ROUTE
# ============================================================

def draw_route(world, route):

    for i in range(len(route) - 1):

        p1 = route[i].transform.location
        p2 = route[i + 1].transform.location

        world.debug.draw_line(
            p1,
            p2,
            thickness=0.08,
            color=carla.Color(0, 255, 0),
            life_time=0.2
        )


# ============================================================
# GET VEHICLE SPEED
# ============================================================

def get_speed(vehicle):

    velocity = vehicle.get_velocity()

    speed = math.sqrt(
        velocity.x ** 2 +
        velocity.y ** 2 +
        velocity.z ** 2
    )

    return speed * 3.6


# ============================================================
# STEERING TOWARD WAYPOINT
# ============================================================

def calculate_steering(vehicle, target):

    vehicle_transform = vehicle.get_transform()

    vehicle_location = vehicle_transform.location

    vehicle_forward = vehicle_transform.get_forward_vector()

    target_location = target.transform.location

    dx = target_location.x - vehicle_location.x
    dy = target_location.y - vehicle_location.y

    target_angle = math.degrees(
        math.atan2(dy, dx)
    )

    vehicle_angle = vehicle_transform.rotation.yaw

    angle = target_angle - vehicle_angle

    while angle > 180:
        angle -= 360

    while angle < -180:
        angle += 360

    # Normalize steering
    steering = angle / 45.0

    steering = max(
        -1.0,
        min(1.0, steering)
    )

    return steering


# ============================================================
# MAIN
# ============================================================

def main():

    client = None
    vehicle = None

    try:

        # ----------------------------------------------------
        # Connect
        # ----------------------------------------------------

        client, world = connect_to_carla()

        # ----------------------------------------------------
        # Build graph
        # ----------------------------------------------------

        graph, waypoint_map = build_road_graph(world)

        # ----------------------------------------------------
        # Spawn vehicle
        # ----------------------------------------------------

        blueprint_library = world.get_blueprint_library()

        vehicle_bp = blueprint_library.filter(
            "vehicle.tesla.model3"
        )[0]

        spawn_points = world.get_map().get_spawn_points()

        vehicle = None

        for spawn in spawn_points:

            vehicle = world.try_spawn_actor(
                vehicle_bp,
                spawn
            )

            if vehicle is not None:
                break

        if vehicle is None:

            raise RuntimeError(
                "Could not spawn Tesla."
            )

        print(
            "\nVehicle spawned:",
            vehicle.id
        )

        # IMPORTANT:
        # We are NOT using CARLA autopilot.
        vehicle.set_autopilot(False)

        # ----------------------------------------------------
        # Start position
        # ----------------------------------------------------

        start = nearest_waypoint(
            vehicle.get_location(),
            waypoint_map
        )

        # ----------------------------------------------------
        # Destination
        # ----------------------------------------------------

        destination = choose_destination(
            world,
            vehicle,
            waypoint_map
        )

        print(
            "\nSOURCE:",
            vehicle.get_location()
        )

        print(
            "\nDESTINATION:",
            destination.transform.location
        )

        # ----------------------------------------------------
        # A*
        # ----------------------------------------------------

        print("\nRunning A*...")

        route = astar(
            graph,
            waypoint_map,
            start,
            destination
        )

        if not route:

            raise RuntimeError(
                "A* could not find a route."
            )

        print(
            "A* route generated."
        )

        print(
            "Route waypoints:",
            len(route)
        )

        # ----------------------------------------------------
        # Destination marker
        # ----------------------------------------------------

        world.debug.draw_point(
            destination.transform.location,
            size=1.0,
            color=carla.Color(255, 0, 0),
            life_time=60.0
        )

        # ----------------------------------------------------
        # Camera follows vehicle
        # ----------------------------------------------------

        spectator = world.get_spectator()

        # ----------------------------------------------------
        # Route following
        # ----------------------------------------------------

        current_index = 0

        print(
            "\nAUTONOMOUS DESTINATION DRIVING STARTED"
        )

        print(
            "Autopilot: OFF"
        )

        print(
            "Controller: Python"
        )

        print(
            "Planner: A*"
        )

        print(
            "\nPress Ctrl+C to stop.\n"
        )

        while True:

            # ------------------------------------------------
            # Check destination
            # ------------------------------------------------

            vehicle_location = vehicle.get_location()

            destination_distance = distance(
                vehicle_location,
                destination.transform.location
            )

            if destination_distance < ARRIVAL_DISTANCE:

                print(
                    "\nDESTINATION REACHED!"
                )

                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        brake=1.0
                    )
                )

                break

            # ------------------------------------------------
            # Select target waypoint
            # ------------------------------------------------

            while (
                current_index < len(route) - 1
                and distance(
                    vehicle_location,
                    route[current_index].transform.location
                ) < 5.0
            ):

                current_index += 1

            target = route[current_index]

            # ------------------------------------------------
            # Steering
            # ------------------------------------------------

            steering = calculate_steering(
                vehicle,
                target
            )

            # ------------------------------------------------
            # Speed
            # ------------------------------------------------

            speed = get_speed(vehicle)

            if speed < CRUISE_SPEED:

                throttle = 0.45

            else:

                throttle = 0.05

            brake = 0.0

            # ------------------------------------------------
            # Apply control
            # ------------------------------------------------

            control = carla.VehicleControl(
                throttle=throttle,
                steer=steering,
                brake=brake
            )

            vehicle.apply_control(control)

            # ------------------------------------------------
            # Draw route
            # ------------------------------------------------

            draw_route(
                world,
                route[
                    max(0, current_index - 1):
                    min(len(route), current_index + 10)
                ]
            )

            draw_destination(
                world,
                destination
            )

            # ------------------------------------------------
            # Spectator
            # ------------------------------------------------

            transform = vehicle.get_transform()

            forward = transform.get_forward_vector()

            camera_location = (
                transform.location
                - forward * 12
            )

            camera_location.z += 6

            spectator.set_transform(
                carla.Transform(
                    camera_location,
                    carla.Rotation(
                        pitch=-15,
                        yaw=transform.rotation.yaw
                    )
                )
            )

            # ------------------------------------------------
            # Console information
            # ------------------------------------------------

            print(
                f"\rSpeed: {speed:5.1f} km/h | "
                f"Waypoint: {current_index:3d}/"
                f"{len(route):3d} | "
                f"Destination: "
                f"{destination_distance:6.1f} m | "
                f"Steer: {steering:+.2f}",
                end=""
            )

            time.sleep(0.05)

    except KeyboardInterrupt:

        print(
            "\n\nStopped by user."
        )

    except Exception as e:

        print(
            "\n\nERROR:",
            e
        )

    finally:

        if vehicle is not None:

            try:

                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        brake=1.0
                    )
                )

                time.sleep(0.5)

                vehicle.destroy()

            except Exception:
                pass

        print(
            "\nVehicle cleaned up."
        )


if __name__ == "__main__":

    main()
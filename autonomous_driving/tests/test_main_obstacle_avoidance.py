from types import SimpleNamespace
import types
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
try:
    import carla
except ImportError:
    carla_stub = types.ModuleType("carla")

    class VehicleControl:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    carla_stub.VehicleControl = VehicleControl
    sys.modules["carla"] = carla_stub

import main


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return FakeLocation(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __add__(self, other):
        return FakeLocation(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __mul__(self, scalar):
        return FakeLocation(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )


class FakeTransform:
    def __init__(self, location):
        self.location = location
        self.rotation = SimpleNamespace()

    def get_forward_vector(self):
        return SimpleNamespace(x=1.0, y=0.0, z=0.0)

    def transform(self, location):
        return FakeLocation(
            self.location.x + location.x,
            self.location.y + location.y,
            self.location.z + location.z,
        )


class FakeWaypoint:
    def __init__(self, location):
        self.transform = SimpleNamespace(
            location=location,
            rotation=SimpleNamespace(),
        )


class FakeVehicle:
    def __init__(self, location):
        self.location = location

    def get_location(self):
        return self.location

    def get_transform(self):
        return FakeTransform(self.location)


class FakeSpawnedVehicle:
    def __init__(self):
        self.autopilot = None

    def set_autopilot(self, enabled, port=None):
        self.autopilot = (enabled, port)


@pytest.fixture
def avoidance_setup(monkeypatch):
    current = FakeLocation()
    target = FakeWaypoint(FakeLocation())
    monkeypatch.setattr(main, "ego", FakeVehicle(current))
    monkeypatch.setattr(main, "avoidance_active", True)
    monkeypatch.setattr(main, "avoidance_points", [target])
    monkeypatch.setattr(main, "avoidance_point_index", 0)
    monkeypatch.setattr(main, "avoidance_obstacle_id", 7)
    monkeypatch.setattr(main, "avoidance_merge_route_index", None)
    monkeypatch.setattr(main, "avoidance_last_progress_location", current)
    monkeypatch.setattr(main, "avoidance_last_progress_time", time.monotonic())
    monkeypatch.setattr(main, "avoidance_blocked_replan_time", 0.0)
    monkeypatch.setattr(main, "active_route_index", 0)
    monkeypatch.setattr(main, "distance", lambda first, second: (
        0.0 if first is second else 5.0
    ))
    monkeypatch.setattr(main, "is_driving_waypoint", lambda waypoint: True)
    monkeypatch.setattr(main, "obstacle_is_directly_ahead", lambda _actor: True)
    monkeypatch.setattr(main, "nearby_blocking_objects", lambda *_: [])
    monkeypatch.setattr(main, "calculate_steering", lambda *_args, **_kwargs: 0.1)
    monkeypatch.setattr(main, "speed_kmh", lambda _vehicle: 0.0)
    monkeypatch.setattr(main, "speed_control", lambda *_: (0.2, 0.0))
    monkeypatch.setattr(main, "nearest_pedestrian_clearance", lambda: None)
    return current


def test_blocked_bypass_replans_and_continues_on_clear_path(
    monkeypatch,
    avoidance_setup,
):
    current = avoidance_setup
    obstacle = SimpleNamespace(id=7)
    new_target = FakeWaypoint(FakeLocation())
    clearance_states = iter(((None, True), (None, False)))
    detected_obstacles = iter(((obstacle, 1.0), (None, float("inf"))))
    monkeypatch.setattr(
        main,
        "avoidance_path_status",
        lambda *_: next(clearance_states),
    )
    monkeypatch.setattr(
        main,
        "find_forward_obstacle",
        lambda **_kwargs: next(detected_obstacles),
    )

    def replan(_obstacle):
        main.avoidance_points[:] = [new_target]
        return True

    monkeypatch.setattr(main, "make_local_detour", replan)

    control, state = main.avoidance_control()

    assert state == "AVOIDING_OBSTACLE"
    assert control.throttle > 0.0
    assert control.brake == 0.0
    assert main.avoidance_points == [new_target]
    assert main.avoidance_last_progress_location is current


def test_blocked_bypass_holds_if_replanning_finds_no_safe_path(
    monkeypatch,
    avoidance_setup,
):
    obstacle = SimpleNamespace(id=7)
    monkeypatch.setattr(main, "avoidance_path_status", lambda *_: (None, True))
    monkeypatch.setattr(
        main,
        "find_forward_obstacle",
        lambda **_kwargs: (obstacle, 1.0),
    )
    monkeypatch.setattr(main, "make_local_detour", lambda _obstacle: False)

    control, state = main.avoidance_control()

    assert state == "AVOIDANCE_PATH_BLOCKED_AT_STOP_DISTANCE"
    assert control.throttle == 0.0
    assert control.brake == 1.0


@pytest.mark.parametrize(
    ("vehicle_class", "blueprint_ids"),
    [
        ("scooter", ["vehicle.harley-davidson.low_rider", "vehicle.tesla.model3"]),
        ("van", ["vehicle.volkswagen.t2", "vehicle.tesla.model3"]),
    ],
)
def test_moving_traffic_selects_matching_carla_vehicle_blueprints(
    vehicle_class,
    blueprint_ids,
):
    blueprints = [
        SimpleNamespace(id=blueprint_id)
        for blueprint_id in blueprint_ids
    ]

    selected = main.moving_vehicle_blueprints(blueprints, vehicle_class)

    assert len(selected) == 1
    assert vehicle_class in selected[0].id.lower() or (
        vehicle_class == "scooter"
        and any(
            token in selected[0].id.lower()
            for token in ("harley", "ninja", "vespa", "motorcycle")
        )
    ) or (vehicle_class == "van" and "t2" in selected[0].id.lower())


def test_moving_traffic_rejects_unsupported_vehicle_category():
    with pytest.raises(ValueError, match="Unsupported moving vehicle class"):
        main.moving_vehicle_blueprints([], "truck")


def test_scenario_traffic_is_distributed_through_first_kilometre(monkeypatch):
    route = [FakeWaypoint(SimpleNamespace(x=index, y=0.0, z=0.0)) for index in range(1_101)]
    monkeypatch.setattr(
        main,
        "distance",
        lambda first, second: abs(first.x - second.x),
    )

    distances = main.scenario_route_distances(
        route,
        15,
    )

    assert len(distances) == 15
    assert distances[0] == 65.0
    assert distances[-1] == 965.0
    assert all(
        first < second
        for first, second in zip(distances, distances[1:])
    )


def test_route_distance_at_index_tracks_ego_progress(monkeypatch):
    route = [
        FakeWaypoint(SimpleNamespace(x=x, y=0.0, z=0.0))
        for x in (0.0, 10.0, 25.0, 40.0)
    ]
    monkeypatch.setattr(
        main,
        "distance",
        lambda first, second: abs(first.x - second.x),
    )

    assert main.route_distance_at_index(route, 2) == 25.0
    assert main.route_distance_at_index(route, 99) == 40.0


def test_stationary_vehicle_and_tire_hazards_are_disabled():
    assert main.NUM_ROCKS == 0
    assert main.NUM_DEBRIS == 0
    assert main.NUM_CONES == 0
    assert main.NUM_BARRIERS == 0
    assert main.NUM_ROUTE_OBSTACLES == 0
    assert main.NUM_PARKED == 8
    assert main.NUM_POTHOLE_MARKERS == 0
    assert main.NUM_MOVING_SCOOTERS == 4
    assert main.NUM_MOVING_VANS == 4
    assert main.NUM_FALLBACK_MOVING_CARS == 4
    assert main.NUM_PEDESTRIANS == 0


def test_spawn_ego_retries_later_route_waypoint_when_start_is_blocked(
    monkeypatch,
):
    blueprint = SimpleNamespace(id="vehicle.tesla.model3")
    attempts = []
    class FakeEgoActor:
        id = 123

        def set_autopilot(self, enabled, port=None):
            self.autopilot = (enabled, port)

    spawned_actor = FakeEgoActor()

    class FakeBlueprintLibrary:
        def filter(self, name):
            return [blueprint] if name in (
                "vehicle.*",
                "vehicle.tesla.model3",
            ) else []

    class FakeWorld:
        def get_blueprint_library(self):
            return FakeBlueprintLibrary()

        def try_spawn_actor(self, _blueprint, transform):
            attempts.append(transform)
            return spawned_actor if len(attempts) == 3 else None

    route = [
        FakeWaypoint(FakeLocation(index * 4.0, 0.0))
        for index in range(5)
    ]
    monkeypatch.setattr(main, "world", FakeWorld())
    monkeypatch.setattr(main, "environment_actors", [])
    monkeypatch.setattr(main, "ego_spawn_route_index", 0)
    monkeypatch.setattr(
        main.carla,
        "Location",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        main.carla,
        "Transform",
        lambda location, rotation: SimpleNamespace(
            location=location,
            rotation=rotation,
        ),
        raising=False,
    )

    result = main.spawn_ego(route[0], route)

    assert result is spawned_actor
    assert len(attempts) == 3
    assert attempts[2].location.x == 8.0
    assert main.ego_spawn_route_index == 2
    assert main.environment_actors == [spawned_actor]
    assert spawned_actor.autopilot == (False, None)


def test_pedestrians_spawn_on_road_and_receive_route_following_control(
    monkeypatch,
):
    class FakeWalker(FakeVehicle):
        def __init__(self, location):
            super().__init__(location)
            self.is_alive = True
            self.controls = []

        def apply_control(self, control):
            self.controls.append(control)

    class FakeWorld:
        def __init__(self, walker):
            self.walker = walker
            self.spawned_transform = None

        def get_blueprint_library(self):
            return SimpleNamespace(filter=lambda _pattern: ["walker-blueprint"])

        def try_spawn_actor(self, _blueprint, transform):
            self.spawned_transform = transform
            self.walker.location = transform.location
            return self.walker

    route = [
        FakeWaypoint(FakeLocation(index * 4.0, 0.0))
        for index in range(11)
    ]
    walker = FakeWalker(FakeLocation())
    fake_world = FakeWorld(walker)
    monkeypatch.setattr(main, "world", fake_world)
    monkeypatch.setattr(main, "ego", FakeVehicle(FakeLocation(-100.0, 0.0)))
    monkeypatch.setattr(main, "environment_actors", [])
    monkeypatch.setattr(main, "road_pedestrian_states", [])
    monkeypatch.setattr(main, "obstacle_route_distances", [])
    monkeypatch.setattr(main, "NUM_PEDESTRIANS", 1)
    monkeypatch.setattr(main, "scenario_route_distances", lambda *_args, **_kwargs: [100.0])
    monkeypatch.setattr(
        main,
        "route_waypoint_at_distance",
        lambda *_args: SimpleNamespace(
            transform=FakeTransform(FakeLocation(8.0, 0.0)),
            lane_width=3.5,
            is_junction=False,
        ),
    )
    monkeypatch.setattr(main.random, "choice", lambda values: values[0])
    monkeypatch.setattr(main.random, "uniform", lambda _low, _high: 1.0)
    monkeypatch.setattr(main.carla, "Vector3D", lambda x, y, z: FakeLocation(x, y, z), raising=False)
    monkeypatch.setattr(
        main.carla,
        "Transform",
        lambda location, rotation: SimpleNamespace(
            location=location,
            rotation=rotation,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        main.carla,
        "WalkerControl",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    main.add_road_pedestrians(route)
    main.update_road_pedestrians()

    assert len(main.road_pedestrian_states) == 1
    assert abs(fake_world.spawned_transform.location.y) < 1.75
    assert walker.controls
    assert walker.controls[-1].speed == 1.0
    assert abs(walker.controls[-1].direction.x) > 0.0


def test_beside_obstacle_slows_by_coasting_without_braking(monkeypatch):
    obstacle = SimpleNamespace(
        id=9,
        type_id="vehicle.van",
    )
    monkeypatch.setattr(main, "ego", FakeVehicle(FakeLocation()))
    monkeypatch.setattr(main, "collision_state", {"hit": False})
    monkeypatch.setattr(main, "speed_kmh", lambda _ego: 20.0)
    monkeypatch.setattr(
        main,
        "find_forward_obstacle",
        lambda **_kwargs: (obstacle, 0.5),
    )
    monkeypatch.setattr(main, "obstacle_is_directly_ahead", lambda _actor: False)
    monkeypatch.setattr(main, "nearest_pedestrian_clearance", lambda: None)
    monkeypatch.setattr(main, "is_driving_waypoint", lambda _waypoint: True)
    monkeypatch.setattr(main, "avoidance_active", False)
    monkeypatch.setattr(main, "blocked_obstacle_id", None)
    monkeypatch.setattr(main, "avoidance_search_obstacle_id", None)
    monkeypatch.setattr(main, "avoidance_search_stop_retry_attempted", False)
    monkeypatch.setattr(main, "bypass_last_search_time", time.monotonic())
    monkeypatch.setattr(main, "obstacle_target_speed", lambda _clearance: 6.0)
    monkeypatch.setattr(main, "calculate_steering", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        main,
        "start_obstacle_avoidance",
        lambda *_args: pytest.fail("beside obstacle must not trigger a bypass"),
    )

    control, state = main.calculate_control(object())

    assert state == "OBSTACLE_SLOWDOWN_BESIDE_NO_BRAKE"
    assert control.throttle == 0.0
    assert control.brake == 0.0


def test_close_forward_obstacle_still_applies_safety_brake(monkeypatch):
    obstacle = SimpleNamespace(
        id=10,
        type_id="vehicle.van",
    )
    monkeypatch.setattr(main, "ego", FakeVehicle(FakeLocation()))
    monkeypatch.setattr(main, "collision_state", {"hit": False})
    monkeypatch.setattr(main, "speed_kmh", lambda _ego: 10.0)
    monkeypatch.setattr(
        main,
        "find_forward_obstacle",
        lambda **_kwargs: (obstacle, 2.0),
    )
    monkeypatch.setattr(main, "obstacle_is_directly_ahead", lambda _actor: True)
    monkeypatch.setattr(main, "nearest_pedestrian_clearance", lambda: None)
    monkeypatch.setattr(main, "is_driving_waypoint", lambda _waypoint: True)
    monkeypatch.setattr(main, "avoidance_active", False)
    monkeypatch.setattr(main, "blocked_obstacle_id", None)
    monkeypatch.setattr(main, "avoidance_search_obstacle_id", None)
    monkeypatch.setattr(main, "avoidance_search_stop_retry_attempted", True)
    monkeypatch.setattr(main, "bypass_last_search_time", time.monotonic())
    monkeypatch.setattr(
        main,
        "start_obstacle_avoidance",
        lambda *_args: False,
    )

    control, state = main.calculate_control(object())

    assert state == "WAITING_FOR_BLOCKED_OBSTACLE"
    assert control.throttle == 0.0
    assert control.brake == 1.0


def test_pedestrian_in_front_of_vehicle_applies_full_brake(monkeypatch):
    monkeypatch.setattr(main, "ego", FakeVehicle(FakeLocation()))
    monkeypatch.setattr(main, "collision_state", {"hit": False})
    monkeypatch.setattr(main, "speed_kmh", lambda _ego: 12.0)
    monkeypatch.setattr(
        main,
        "find_forward_obstacle",
        lambda **_kwargs: (None, float("inf")),
    )
    monkeypatch.setattr(main, "obstacle_is_directly_ahead", lambda _actor: False)
    monkeypatch.setattr(main, "nearest_pedestrian_clearance", lambda: 3.5)
    monkeypatch.setattr(main, "current_risk_level", "LOW")
    monkeypatch.setattr(main, "current_target_speed", main.CRUISE_SPEED)

    control, state = main.calculate_control(object())

    assert state == "PEDESTRIAN_STOP"
    assert control.throttle == 0.0
    assert control.brake == 1.0


def test_pedestrian_outside_stop_distance_does_not_trigger_full_brake(
    monkeypatch,
):
    monkeypatch.setattr(main, "ego", FakeVehicle(FakeLocation()))
    monkeypatch.setattr(main, "collision_state", {"hit": False})
    monkeypatch.setattr(main, "speed_kmh", lambda _ego: 12.0)
    monkeypatch.setattr(
        main,
        "find_forward_obstacle",
        lambda **_kwargs: (None, float("inf")),
    )
    monkeypatch.setattr(main, "obstacle_is_directly_ahead", lambda _actor: False)
    monkeypatch.setattr(main, "nearest_pedestrian_clearance", lambda: 8.0)
    monkeypatch.setattr(main, "is_driving_waypoint", lambda _waypoint: True)
    monkeypatch.setattr(main, "route_recovery_active", False)
    monkeypatch.setattr(main, "calculate_steering", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(main, "current_risk_level", "LOW")
    monkeypatch.setattr(main, "current_target_speed", main.CRUISE_SPEED)

    control, state = main.calculate_control(object())

    assert state == "CRUISING"
    assert control.brake < 1.0


def test_infeasible_pathweaver_fallback_brakes_without_raising(monkeypatch):
    monkeypatch.setattr(
        main,
        "ego",
        SimpleNamespace(
            get_control=lambda: SimpleNamespace(steer=0.2),
        ),
    )
    plan = SimpleNamespace(
        feasible=False,
        emergency=False,
        behavior_state="EMERGENCY_BRAKE",
    )

    control = main.pathweaver_control(plan, anchor_transform=None, timestamp_s=1.0)

    assert control.throttle == 0.0
    assert control.brake == 1.0
    assert control.steer < 0.2


@pytest.mark.parametrize(
    ("obstacle_location", "expected_ahead"),
    [
        (FakeLocation(6.0, 0.0), True),
        (FakeLocation(0.5, 3.5), False),
        (FakeLocation(-0.5, 0.0), False),
    ],
)
def test_obstacle_ahead_classifier_distinguishes_side_objects(
    monkeypatch,
    obstacle_location,
    expected_ahead,
):
    ego = FakeVehicle(FakeLocation())
    ego.bounding_box = SimpleNamespace(extent=SimpleNamespace(y=1.0))
    actor = FakeVehicle(obstacle_location)
    actor.bounding_box = SimpleNamespace(
        location=FakeLocation(),
        extent=SimpleNamespace(x=1.0, y=0.5),
    )
    monkeypatch.setattr(main, "ego", ego)
    monkeypatch.setattr(
        main.carla,
        "Vector3D",
        lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
        raising=False,
    )
    monkeypatch.setattr(
        main.carla,
        "Location",
        lambda **kwargs: FakeLocation(**kwargs),
        raising=False,
    )

    assert main.obstacle_is_directly_ahead(actor) is expected_ahead


def test_add_moving_traffic_spawns_five_manual_route_followers(monkeypatch):
    class FakeBlueprint:
        def __init__(self, blueprint_id):
            self.id = blueprint_id

    class FakeTransform:
        def __init__(self, location, rotation):
            self.location = location
            self.rotation = rotation

    blueprints = [
        FakeBlueprint("vehicle.kawasaki.ninja"),
        FakeBlueprint("vehicle.volkswagen.t2"),
    ]
    spawned = []
    requested_start = []

    class FakeBlueprintLibrary:
        def filter(self, _pattern):
            return blueprints

    class FakeWorld:
        def get_blueprint_library(self):
            return FakeBlueprintLibrary()

        def try_spawn_actor(self, _blueprint, _transform):
            actor = FakeSpawnedVehicle()
            spawned.append(actor)
            return actor

    class FakeTrafficManager:
        pass

    monkeypatch.setattr(main, "world", FakeWorld())
    monkeypatch.setattr(main, "traffic_manager", FakeTrafficManager())
    monkeypatch.setattr(main, "environment_actors", [])
    monkeypatch.setattr(main, "moving_traffic_states", [])
    monkeypatch.setattr(main, "obstacle_route_distances", [])
    monkeypatch.setattr(main, "ego_spawn_route_index", 2)

    def fake_scenario_route_distances(_route, count, start):
        requested_start.append(start)
        return [start + 30 * index for index in range(count)]

    monkeypatch.setattr(main, "scenario_route_distances", fake_scenario_route_distances)
    monkeypatch.setattr(
        main,
        "moving_vehicle_waypoint",
        lambda *_args, **_kwargs: FakeWaypoint(
            SimpleNamespace(x=0.0, y=0.0, z=0.0),
        ),
    )
    monkeypatch.setattr(
        main,
        "moving_vehicle_route_locations",
        lambda *_args: ["route-point-1", "route-point-2"],
    )
    monkeypatch.setattr(main, "is_driving_segment", lambda *_args: True)
    monkeypatch.setattr(
        main.carla,
        "Location",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        main.carla,
        "Transform",
        FakeTransform,
        raising=False,
    )

    route = [
        FakeWaypoint(SimpleNamespace(x=x, y=0.0, z=0.0))
        for x in (0.0, 10.0, 20.0, 30.0)
    ]
    main.add_moving_traffic(route)

    assert len(spawned) == (
        main.NUM_MOVING_SCOOTERS + main.NUM_MOVING_VANS
    )
    assert requested_start == [
        20.0 + main.NPC_MIN_AHEAD_DISTANCE_METERS
    ]
    assert main.NPC_TARGET_SPEED_KMH > main.CRUISE_SPEED
    assert all(actor.autopilot == (False, None) for actor in spawned)
    assert len(main.moving_traffic_states) == len(spawned)
    assert all(
        len(state["route_locations"]) >= 2
        for state in main.moving_traffic_states
    )


def test_add_moving_traffic_uses_moving_cars_when_classes_are_unavailable(
    monkeypatch,
):
    class FakeBlueprint:
        def __init__(self, blueprint_id):
            self.id = blueprint_id

    class FakeTransform:
        def __init__(self, location, rotation):
            self.location = location
            self.rotation = rotation

    sedan = FakeBlueprint("vehicle.tesla.model3")
    spawned_blueprints = []

    class FakeBlueprintLibrary:
        def filter(self, _pattern):
            return [sedan]

    class FakeWorld:
        def get_blueprint_library(self):
            return FakeBlueprintLibrary()

        def try_spawn_actor(self, blueprint, _transform):
            spawned_blueprints.append(blueprint)
            return FakeSpawnedVehicle()

    class FakeTrafficManager:
        pass

    monkeypatch.setattr(main, "world", FakeWorld())
    monkeypatch.setattr(main, "traffic_manager", FakeTrafficManager())
    monkeypatch.setattr(main, "environment_actors", [])
    monkeypatch.setattr(main, "moving_traffic_states", [])
    monkeypatch.setattr(main, "obstacle_route_distances", [])
    monkeypatch.setattr(
        main,
        "scenario_route_distances",
        lambda *_args, **_kwargs: list(
            range(
                70,
                70 + 30 * (
                    main.NUM_FALLBACK_MOVING_CARS + main.NUM_MOVING_VANS
                ),
                30,
            )
        ),
    )
    monkeypatch.setattr(
        main,
        "moving_vehicle_waypoint",
        lambda *_args, **_kwargs: FakeWaypoint(
            SimpleNamespace(x=0.0, y=0.0, z=0.0),
        ),
    )
    monkeypatch.setattr(
        main,
        "moving_vehicle_route_locations",
        lambda *_args: ["route-point-1", "route-point-2"],
    )
    monkeypatch.setattr(main, "is_driving_segment", lambda *_args: True)
    monkeypatch.setattr(
        main.carla,
        "Location",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        main.carla,
        "Transform",
        FakeTransform,
        raising=False,
    )

    main.add_moving_traffic([])

    assert len(spawned_blueprints) == (
        main.NUM_FALLBACK_MOVING_CARS + main.NUM_MOVING_VANS
    )
    assert all(blueprint is sedan for blueprint in spawned_blueprints)


def test_update_moving_traffic_applies_forward_throttle_and_route_steering(
    monkeypatch,
):
    class FakeNpc(FakeVehicle):
        def __init__(self):
            super().__init__(FakeLocation(0.0, 0.0))
            self.is_alive = True
            self.controls = []

        def get_velocity(self):
            return FakeLocation()

        def apply_control(self, control):
            self.controls.append(control)

    actor = FakeNpc()
    monkeypatch.setattr(
        actor,
        "get_transform",
        lambda: SimpleNamespace(
            get_forward_vector=lambda: SimpleNamespace(x=1.0, y=0.0),
        ),
    )
    monkeypatch.setattr(main, "moving_traffic_states", [{
        "actor": actor,
        "route_locations": [
            FakeLocation(0.0, 0.0),
            FakeLocation(10.0, 0.0),
            FakeLocation(20.0, 0.0),
        ],
        "route_index": 1,
    }])
    monkeypatch.setattr(
        main.carla,
        "VehicleControl",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    main.update_moving_traffic()

    assert len(actor.controls) == 1
    assert actor.controls[0].throttle > 0.0
    assert actor.controls[0].brake == 0.0

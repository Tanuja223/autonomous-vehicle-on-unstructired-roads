import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SCRIPTS = (
    "unstructured_autonomous_rough_road.py",
    "unstructured_autonomous_no_pathweaver_road.py",
    "unstructured_autonomous_road.py",
    "unstructured_autonomous_no_pathweaver.py",
    "main.py",
)


class Location:
    def __init__(self, x):
        self.x = x
        self.y = 0.0
        self.z = 0.0


class Waypoint:
    def __init__(self, x):
        self.transform = SimpleNamespace(location=Location(x))


def load_function(source_tree, name, namespace):
    function_node = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[function_node], type_ignores=[])
    )
    exec(compile(module, "<scenario test>", "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("filename", SCENARIO_SCRIPTS)
def test_scenario_requests_two_kilometers_and_crossing_actors(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))

    def configured_value(expression):
        if isinstance(expression, ast.IfExp):
            return max(
                ast.literal_eval(expression.body),
                ast.literal_eval(expression.orelse),
            )
        return ast.literal_eval(expression)

    constants = {
        node.targets[0].id: configured_value(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        for target in node.targets
        if target.id in {
            "TARGET_DRIVE_ROUTE_METERS",
            "CRUISE_SPEED",
            "AVOIDANCE_SPEED",
            "SCENARIO_HAZARD_LENGTH_METERS",
            "NUM_MOVING_SCOOTERS",
            "NUM_MOVING_VANS",
            "NUM_FALLBACK_MOVING_CARS",
            "NUM_PEDESTRIANS",
        }
    }

    assert constants["TARGET_DRIVE_ROUTE_METERS"] == 2_000.0
    assert constants["CRUISE_SPEED"] == 10.0
    assert constants["AVOIDANCE_SPEED"] == 8.0
    assert constants["SCENARIO_HAZARD_LENGTH_METERS"] == 2_000.0
    assert constants["NUM_PEDESTRIANS"] > 0
    assert (
        constants["NUM_MOVING_SCOOTERS"]
        + constants["NUM_MOVING_VANS"]
        + constants["NUM_FALLBACK_MOVING_CARS"]
        > 0
    )

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "add_road_pedestrians" in functions
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "add_road_pedestrians"
        for node in ast.walk(functions["create_environment"])
    )


def test_scripted_pedestrians_advance_when_the_loop_updates_them():
    tree = ast.parse(
        (ROOT / "unstructured_autonomous_rough_road.py").read_text(
            encoding="utf-8"
        )
    )

    class FakeLocation:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        def __sub__(self, other):
            return SimpleNamespace(
                x=self.x - other.x,
                y=self.y - other.y,
                z=self.z - other.z,
            )

    class FakeActor:
        id = 1
        is_alive = True

        def set_transform(self, transform):
            self.transform = transform

    actor = FakeActor()
    pedestrian = {
        "actor": actor,
        "crossing_start": FakeLocation(0.0, 0.0, 0.0),
        "crossing_end": FakeLocation(10.0, 0.0, 0.0),
        "completed": False,
        "last_update": 0.0,
        "progress_m": 0.0,
        "speed": 1.2,
        "yaw": 0.0,
    }
    update_pedestrians = load_function(
        tree,
        "update_road_pedestrians",
        {
            "pedestrian_simulation_time": lambda: 0.1,
            "math": math,
            "road_pedestrian_states": [pedestrian],
            "environment_actors": [actor],
            "carla": SimpleNamespace(
                Location=FakeLocation,
                Rotation=lambda yaw: SimpleNamespace(yaw=yaw),
                Transform=lambda location, rotation: SimpleNamespace(
                    location=location,
                    rotation=rotation,
                ),
            ),
        },
    )

    update_pedestrians()

    assert pedestrian["progress_m"] == pytest.approx(0.12)
    assert actor.transform.location.x > 0.0


@pytest.mark.parametrize("filename", SCENARIO_SCRIPTS)
def test_bypass_is_requested_for_front_obstacle_at_ten_meters(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "OBSTACLE_BYPASS_REQUEST_DISTANCE_M"
    }
    assert constants["OBSTACLE_BYPASS_REQUEST_DISTANCE_M"] == 10.0

    control_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "calculate_control"
    )
    bypass_assignment = next(
        node
        for node in ast.walk(control_function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "should_plan_bypass"
            for target in node.targets
        )
    )
    expression = ast.fix_missing_locations(
        ast.Expression(body=bypass_assignment.value)
    )
    scenario = SimpleNamespace(id=1)
    environment = {
        "obstacle": scenario,
        "obstacle_needs_bypass": False,
        "vehicle_ahead_slow": False,
        "is_pedestrian": False,
        "obstacle_distance": 10.0,
        "AVOID_DETECT_DISTANCE": 30.0,
        "risk_level": "CRITICAL",
        "AVOIDANCE_EMERGENCY_BYPASS_DISTANCE": 6.5,
        "OBSTACLE_BYPASS_REQUEST_DISTANCE_M": 10.0,
        "obstacle_ahead": True,
        "now": 2.0,
        "bypass_last_search_time": 0.0,
        "AVOID_RETRY_SECONDS": 1.0,
        "time": SimpleNamespace(monotonic=lambda: 2.0),
        "avoidance_search_obstacle_id": None,
    }

    assert eval(compile(expression, "<bypass test>", "eval"), environment)


@pytest.mark.parametrize("filename", SCENARIO_SCRIPTS)
def test_smooth_bypass_prefers_left_when_candidate_safety_is_equal(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    choose_candidate = load_function(
        tree,
        "select_smooth_bypass_candidate",
        {},
    )
    left = ((1.5, -24.0), 8, [], "LEFT")
    right = ((1.5, -24.0), 8, [], "RIGHT")
    safer_right = ((2.0, -24.0), 8, [], "RIGHT")
    oncoming_left = ((1.5, -24.0), 8, [], "LEFT-ONCOMING")

    assert choose_candidate([right, left]) is left
    assert choose_candidate([left, safer_right]) is safer_right
    assert choose_candidate([left, oncoming_left]) is left
    assert choose_candidate([]) is None


@pytest.mark.parametrize("filename", SCENARIO_SCRIPTS)
def test_driving_segment_allows_small_lane_edge_sampling_variance(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))

    class FakeLocation:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class FakeWaypoint:
        lane_width = 3.05

        def __init__(self, location):
            self.transform = SimpleNamespace(location=location)

    class FakeMap:
        def get_waypoint(self, location, project_to_road, lane_type):
            if not project_to_road:
                return None
            return FakeWaypoint(
                FakeLocation(location.x + 1.739, location.y, location.z)
            )

    class FakeLaneType:
        Driving = 1

    is_driving_segment = load_function(
        tree,
        "is_driving_segment",
        {
            "math": math,
            "distance": lambda first, second: math.sqrt(
                (first.x - second.x) ** 2
                + (first.y - second.y) ** 2
                + (first.z - second.z) ** 2
            ),
            "carla": SimpleNamespace(
                Location=FakeLocation,
                LaneType=FakeLaneType,
            ),
            "ego": None,
            "is_driving_waypoint": lambda waypoint: waypoint is not None,
        },
    )

    assert is_driving_segment(
        FakeMap(),
        FakeLocation(0.0, 0.0, 0.0),
        FakeLocation(0.0, 0.0, 0.0),
    )


@pytest.mark.parametrize("filename", SCENARIO_SCRIPTS)
def test_is_driving_path_rejects_non_driving_segments(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    points = [
        SimpleNamespace(
            is_driving=True,
            transform=SimpleNamespace(
                location=SimpleNamespace(name=name)
            ),
        )
        for name in ("start", "middle", "end")
    ]
    invalid_segment = (
        points[1].transform.location,
        points[2].transform.location,
    )
    is_driving_path = load_function(
        tree,
        "is_driving_path",
        {
            "is_driving_waypoint": lambda waypoint: waypoint.is_driving,
            "is_driving_segment": lambda _map, start, end: (
                start,
                end,
            ) != invalid_segment,
        },
    )

    assert not is_driving_path("map", points)
    points[2].is_driving = False
    assert not is_driving_path("map", points)


@pytest.mark.parametrize("rear_corridor_clear", [True, False])
def test_close_obstacle_recovery_stops_checks_rear_and_then_plans_bypass(
    rear_corridor_clear,
):
    tree = ast.parse(
        (ROOT / "unstructured_autonomous_rough_road.py").read_text(
            encoding="utf-8"
        )
    )

    class FakeLocation:
        def __init__(self, x, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class FakeVector:
        x = 1.0
        y = 0.0
        z = 0.0

    class FakeControl:
        def __init__(self, **values):
            self.__dict__.update(values)
            self.reverse = values.get("reverse", False)

    ego = SimpleNamespace(
        location=FakeLocation(0.0),
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2.0)),
        get_location=lambda: ego.location,
        get_transform=lambda: SimpleNamespace(
            get_forward_vector=lambda: FakeVector()
        ),
    )
    obstacle = SimpleNamespace(id=42)
    planned_obstacles = []
    namespace = {
        "avoidance_active": True,
        "avoidance_points": [object()],
        "avoidance_point_index": 1,
        "avoidance_obstacle_id": 41,
        "avoidance_merge_route_index": 3,
        "chosen_avoidance_direction": "RIGHT",
        "reverse_recovery_active": False,
        "reverse_recovery_obstacle_id": None,
        "reverse_recovery_start_location": None,
        "reverse_recovery_forward": None,
        "reverse_recovery_target_distance": 0.0,
        "reverse_recovery_started_at": 0.0,
        "current_risk_level": "LOW",
        "current_target_speed": 8.0,
        "ego": ego,
        "world": SimpleNamespace(get_map=lambda: "map"),
        "carla": SimpleNamespace(
            Location=FakeLocation,
            VehicleControl=FakeControl,
        ),
        "is_driving_segment": lambda *_args: True,
        "nearby_blocking_objects": lambda *_args: [],
        "detour_segment_clearance": (
            lambda *_args, **_kwargs: 1.0
            if rear_corridor_clear
            else None
        ),
        "AVOIDANCE_SPEED": 8.0,
        "AVOIDANCE_EMERGENCY_BYPASS_DISTANCE": 6.5,
        "REVERSE_RECOVERY_MIN_DISTANCE_M": 3.0,
        "REVERSE_RECOVERY_MAX_DISTANCE_M": 6.5,
        "REVERSE_RECOVERY_SPEED_KMH": 3.0,
        "REVERSE_RECOVERY_TIMEOUT_SECONDS": 12.0,
        "time": SimpleNamespace(monotonic=lambda: 10.0),
    }
    namespace["reset_reverse_recovery"] = load_function(
        tree,
        "reset_reverse_recovery",
        namespace,
    )
    namespace["start_obstacle_avoidance"] = (
        lambda candidate, _distance: (
            planned_obstacles.append(candidate) or True
        )
    )
    namespace["avoidance_control"] = lambda: (
        FakeControl(throttle=0.2, brake=0.0, steer=-0.1),
        "AVOIDING_OBSTACLE",
    )
    recover = load_function(tree, "reverse_recovery_control", namespace)

    control, state = recover(obstacle, 2.0, 4.0)
    assert state == "REVERSE_RECOVERY_STOPPING"
    assert control.brake == 1.0
    assert not control.reverse

    control, state = recover(obstacle, 2.0, 0.0)
    if not rear_corridor_clear:
        assert state == "REVERSE_RECOVERY_BLOCKED_REAR"
        assert control.brake == 1.0
        assert not control.reverse
        assert not planned_obstacles
        return

    assert state == "REVERSE_RECOVERY"
    assert control.reverse
    assert control.throttle > 0.0
    assert namespace["reverse_recovery_target_distance"] == pytest.approx(4.5)

    ego.location = FakeLocation(-4.5)
    control, state = recover(obstacle, 6.5, 3.0)

    assert state == "AVOIDING_OBSTACLE"
    assert planned_obstacles == [obstacle]


@pytest.mark.parametrize("filename", SCENARIO_SCRIPTS)
def test_avoidance_path_validation_keeps_the_planned_segment_anchor(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    current = SimpleNamespace(name="current")
    points = [
        SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(name=name)))
        for name in ("previous", "target", "next")
    ]
    checked_segments = []

    def is_driving_segment(_map, start, end):
        checked_segments.append((start, end))
        return start is not current

    validate_path = load_function(
        tree,
        "avoidance_path_status",
        {
            "world": SimpleNamespace(get_map=lambda: "map"),
            "is_driving_waypoint": lambda _waypoint: True,
            "is_driving_segment": is_driving_segment,
            "detour_segment_clearance": lambda *_args, **_kwargs: 1.0,
        },
    )

    assert validate_path(current, points, 1, []) == (None, False)
    assert checked_segments == [
        (points[0].transform.location, points[1].transform.location),
        (points[1].transform.location, points[2].transform.location),
    ]


@pytest.mark.parametrize("filename", SCENARIO_SCRIPTS)
def test_avoidance_path_validation_does_not_recheck_the_vehicle_chord(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    current = SimpleNamespace(name="current")
    points = [
        SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(name=name)))
        for name in ("first", "next")
    ]
    checked_segments = []
    checked_clearances = []

    def is_driving_segment(_map, start, end):
        checked_segments.append((start, end))
        return start is not current

    def detour_segment_clearance(start, end, actors):
        checked_clearances.append((start, end, actors))
        return 1.0

    validate_path = load_function(
        tree,
        "avoidance_path_status",
        {
            "world": SimpleNamespace(get_map=lambda: "map"),
            "is_driving_waypoint": lambda _waypoint: True,
            "is_driving_segment": is_driving_segment,
            "detour_segment_clearance": detour_segment_clearance,
        },
    )

    assert validate_path(current, points, 0, []) == (None, False)
    assert checked_segments == [
        (points[0].transform.location, points[1].transform.location)
    ]
    assert checked_clearances[0][:2] == (
        current,
        points[0].transform.location,
    )


@pytest.mark.parametrize("filename", SCENARIO_SCRIPTS)
def test_stall_recovery_only_runs_when_route_is_safe_to_replan(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"NO_PROGRESS_RECOVERY_SECONDS", "STALL_RECOVERY_BLOCKED_STATES"}
    }
    namespace = {
        "NO_PROGRESS_RECOVERY_SECONDS": constants[
            "NO_PROGRESS_RECOVERY_SECONDS"
        ],
        "STALL_RECOVERY_BLOCKED_STATES": constants[
            "STALL_RECOVERY_BLOCKED_STATES"
        ],
    }
    should_recover = load_function(
        tree,
        "should_attempt_route_stall_recovery",
        namespace,
    )

    safe_state = (
        "CRUISING",
        "LOW",
        False,
        False,
        False,
        False,
    )
    assert should_recover(
        constants["NO_PROGRESS_RECOVERY_SECONDS"],
        *safe_state,
    )
    assert not should_recover(
        constants["NO_PROGRESS_RECOVERY_SECONDS"] - 0.1,
        *safe_state,
    )
    assert not should_recover(
        constants["NO_PROGRESS_RECOVERY_SECONDS"],
        "CRUISING",
        "CRITICAL",
        False,
        False,
        False,
        False,
    )
    for state in constants["STALL_RECOVERY_BLOCKED_STATES"]:
        assert not should_recover(
            constants["NO_PROGRESS_RECOVERY_SECONDS"],
            state,
            "HIGH",
            False,
            False,
            False,
            False,
        )
    assert not should_recover(
        constants["NO_PROGRESS_RECOVERY_SECONDS"],
        "CRUISING",
        "LOW",
        True,
        False,
        False,
        False,
    )
    assert not should_recover(
        constants["NO_PROGRESS_RECOVERY_SECONDS"],
        "CRUISING",
        "LOW",
        False,
        True,
        False,
        False,
    )
    assert not should_recover(
        constants["NO_PROGRESS_RECOVERY_SECONDS"],
        "CRUISING",
        "LOW",
        False,
        False,
        True,
        False,
    )
    assert not should_recover(
        constants["NO_PROGRESS_RECOVERY_SECONDS"],
        "CRUISING",
        "LOW",
        False,
        False,
        False,
        True,
    )


@pytest.mark.parametrize("filename", SCENARIO_SCRIPTS)
def test_obstacle_risk_can_reduce_speed_below_ten_kmh(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    namespace = {
        "math": math,
        "CRUISE_SPEED": 10.0,
        "AVOIDANCE_SPEED": 8.0,
        "MIN_STANDOFF_DISTANCE": 2.0,
        "SAFE_DECELERATION": 3.0,
        "REACTION_TIME": 0.5,
    }
    namespace["obstacle_target_speed"] = load_function(
        tree,
        "obstacle_target_speed",
        namespace,
    )
    estimate_risk = load_function(tree, "estimate_risk", namespace)
    namespace.update(
        {
            "collision_state": {"hit": False, "last_time": 0.0},
            "COLLISION_BRAKE_HOLD_SECONDS": 0.5,
            "nearest_pedestrian_clearance": lambda: None,
            "pedestrian_stop_distance": lambda _speed: 0.0,
            "STOP_DISTANCE": 3.0,
            "time": SimpleNamespace(monotonic=lambda: 1.0),
        }
    )

    vehicle = SimpleNamespace(
        type_id="vehicle.tesla.model3",
        get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0),
    )
    prop = SimpleNamespace(type_id="static.prop.box")
    risk_at_six_meters = estimate_risk(vehicle, 6.0, 10.0)
    risk_at_three_and_half_meters = estimate_risk(vehicle, 3.5, 10.0)
    medium_risk = estimate_risk(prop, 20.0, 10.0)

    assert risk_at_six_meters[0] == "HIGH"
    assert risk_at_six_meters[1] == 8.0
    assert risk_at_three_and_half_meters[0] == "HIGH"
    assert 0.0 < risk_at_three_and_half_meters[1] < 8.0
    assert medium_risk[0] == "MEDIUM"
    assert medium_risk[1] == 9.0


@pytest.mark.parametrize("filename", SCENARIO_SCRIPTS)
def test_route_extension_repeats_a_validated_connected_cycle(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    start = Waypoint(0.0)
    middle = Waypoint(1.0)
    cycle = [middle, Waypoint(2.0), middle]

    def find_cycle(_map, _start):
        return cycle

    namespace = {
        "build_map_route": lambda *_args: (_ for _ in ()).throw(RuntimeError("open route")),
        "find_verified_route_continuation": find_cycle,
        "find_farthest_driving_extension": find_cycle,
        "distance": lambda first, second: math.hypot(first.x - second.x, first.y - second.y),
        "is_driving_segment": lambda *_args: True,
        "is_driving_waypoint": lambda _waypoint: True,
        "print": lambda *_args: None,
    }
    extend_route = load_function(tree, "extend_route_with_driving_laps", namespace)
    route = extend_route(object(), [start, middle], 5.0)
    length = sum(
        math.hypot(
            first.transform.location.x - second.transform.location.x,
            first.transform.location.y - second.transform.location.y,
        )
        for first, second in zip(route, route[1:])
    )

    assert length >= 5.0

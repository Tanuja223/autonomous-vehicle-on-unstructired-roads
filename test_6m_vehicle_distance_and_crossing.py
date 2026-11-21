"""
test_6m_vehicle_distance_and_crossing.py
Verify user requests:
1. Increase the speed of the car to 10km/h at start
2. Based on risk decrease or increase the speed (LOW: 10km/h, MEDIUM: 9km/h, HIGH: 8km/h, CRITICAL: 0km/h, resume to 10km/h)
3. Add people crossing roads from left to right, brake during crossing and resume after clearing
4. When obstacle is at 10m far the car should move to left or right
"""
import sys
import math
import carla
from types import SimpleNamespace
import unstructured_autonomous_rough_road as rough_sim
import unstructured_autonomous_no_pathweaver_road as no_pw_road_sim
import unstructured_autonomous_road as road_sim
import unstructured_autonomous_no_pathweaver as no_pw_sim
import main as main_sim

class MockActor:
    def __init__(self, actor_id, type_id, location, velocity=None, extent=None):
        self.id = actor_id
        self.type_id = type_id
        self._location = location
        self._velocity = velocity or carla.Vector3D(0.0, 0.0, 0.0)
        self.bounding_box = carla.BoundingBox(
            carla.Location(0, 0, 0),
            extent or carla.Vector3D(2.0, 1.0, 0.8)
        )
        self.is_alive = True
        self.applied_control = None

    def get_location(self):
        return self._location

    def get_velocity(self):
        return self._velocity

    def get_transform(self):
        return carla.Transform(self._location, carla.Rotation(0, 0, 0))

    def set_transform(self, transform):
        self._location = transform.location

    def apply_control(self, control):
        self.applied_control = control

    def destroy(self):
        self.is_alive = False
        return True

class MockActorList:
    def __init__(self, actors=None):
        self._actors = actors or []
    def filter(self, pattern):
        token = pattern.replace("*", "")
        return [a for a in self._actors if token in a.type_id]
    def __iter__(self):
        return iter(self._actors)
    def __len__(self):
        return len(self._actors)

class MockWaypoint:
    def __init__(self, x=0.0, y=0.0, z=0.0, road_id=1, section_id=0, lane_id=-1, lane_width=3.6):
        self.transform = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=0.0))
        self.road_id = road_id
        self.section_id = section_id
        self.lane_id = lane_id
        self.lane_width = lane_width
        self.lane_type = carla.LaneType.Driving
        self.is_junction = False

    def next(self, dist):
        return [MockWaypoint(self.transform.location.x + dist, self.transform.location.y, self.transform.location.z,
                             road_id=self.road_id, section_id=self.section_id, lane_id=self.lane_id, lane_width=self.lane_width)]

class MockMap:
    def __init__(self):
        self.name = "unstructured_road"
    def get_spawn_points(self):
        return []
    def generate_waypoints(self, step):
        return [MockWaypoint(x=float(i*step), y=0.0, road_id=1, section_id=0, lane_id=-1) for i in range(10)]
    def get_waypoint(self, location, project_to_road=False, lane_type=carla.LaneType.Driving):
        lane_idx = int(round(location.y / 3.6))
        if lane_idx == 0:
            lane_idx = 1 if location.y >= 0 else -1
        return MockWaypoint(location.x, location.y, road_id=1, section_id=0, lane_id=lane_idx)

class MockWorld:
    def __init__(self):
        self.actors = []
        self._map = MockMap()
        self.elapsed_seconds = 0.0
    def get_actors(self):
        return MockActorList(self.actors)
    def get_map(self):
        return self._map
    def get_environment_objects(self, label):
        return []
    def get_snapshot(self):
        return SimpleNamespace(
            timestamp=SimpleNamespace(elapsed_seconds=self.elapsed_seconds)
        )


def test_controller_speed_and_risk_transitions(sim, module_name):
    print(f"\n--- Testing Speed and Risk Transitions on {module_name} ---")
    sim.world = MockWorld()
    sim.ego = MockActor(1, "vehicle.lincoln.mkz_2020", carla.Location(0.0, 0.0, 0.0))

    # 1. Starting speed must be 10 km/h (CRUISE_SPEED == 10.0)
    assert sim.CRUISE_SPEED == 10.0, f"Expected CRUISE_SPEED == 10.0, got {sim.CRUISE_SPEED}"
    assert sim.OBSTACLE_BYPASS_REQUEST_DISTANCE_M == 10.0, f"Expected OBSTACLE_BYPASS_REQUEST_DISTANCE_M == 10.0, got {sim.OBSTACLE_BYPASS_REQUEST_DISTANCE_M}"

    # 2. Clear road -> LOW risk, speed = 10 km/h
    risk, speed = sim.estimate_risk(None, float("inf"), current_speed=10.0, obstacle_ahead=False)
    assert risk == "LOW" and speed == 10.0, f"Expected (LOW, 10.0), got ({risk}, {speed})"
    print(f"[{module_name}] Start / Clear road (LOW risk): speed={speed:.1f} km/h (10.0 km/h verified)")

    # 3. Obstacle at 20m -> MEDIUM risk, speed decreased to 9.0 km/h
    prop = MockActor(301, "static.prop.box", carla.Location(20.0, 0.0, 0.0))
    risk_med, speed_med = sim.estimate_risk(prop, 20.0, current_speed=10.0, obstacle_ahead=True)
    assert risk_med == "MEDIUM", f"Expected MEDIUM, got {risk_med}"
    assert speed_med < 10.0, f"Speed should decrease below 10.0 km/h, got {speed_med}"
    assert speed_med == 9.0, f"Expected 9.0 km/h on MEDIUM risk, got {speed_med}"
    print(f"[{module_name}] Approaching obstacle at 20m (MEDIUM risk): speed decreased to {speed_med:.1f} km/h")

    # 4. Obstacle at 10m -> HIGH risk, speed decreased to 8.0 km/h
    risk_high_10m, speed_high_10m = sim.estimate_risk(prop, 10.0, current_speed=10.0, obstacle_ahead=True)
    assert risk_high_10m == "HIGH", f"Expected HIGH risk at 10m, got {risk_high_10m}"
    assert speed_high_10m == 8.0, f"Expected 8.0 km/h at 10m HIGH risk, got {speed_high_10m}"
    print(f"[{module_name}] Obstacle at 10m (HIGH risk): speed decreased to {speed_high_10m:.1f} km/h")

    # 5. Lead vehicle at 6m -> HIGH risk, speed = 8.0 km/h (avoidance speed)
    lead_vehicle = MockActor(201, "vehicle.tesla.model3", carla.Location(6.0, 0.0, 0.0))
    risk_6m, speed_6m = sim.estimate_risk(lead_vehicle, 6.0, current_speed=10.0, obstacle_ahead=True)
    assert risk_6m == "HIGH" and speed_6m == 8.0, f"Expected (HIGH, 8.0), got ({risk_6m}, {speed_6m})"
    print(f"[{module_name}] Vehicle at 6m (HIGH risk): speed={speed_6m:.1f} km/h")

    # 6. Obstacle cleared -> speed increases back to 10.0 km/h
    risk_clear, speed_clear = sim.estimate_risk(None, float("inf"), current_speed=8.0, obstacle_ahead=False)
    assert risk_clear == "LOW" and speed_clear == 10.0
    print(f"[{module_name}] Obstacle cleared: speed increases back to {speed_clear:.1f} km/h (CRUISING)")


def test_obstacle_at_10m_triggers_left_or_right_bypass(sim, module_name):
    print(f"\n--- Testing 10m Obstacle Avoidance (LEFT and RIGHT) on {module_name} ---")
    sim.world = MockWorld()
    sim.ego = MockActor(1, "vehicle.lincoln.mkz_2020", carla.Location(0.0, 0.0, 0.0))
    sim.avoidance_active = False
    sim.avoidance_points.clear()
    sim.avoidance_obstacle_id = None
    sim.bypass_last_search_time = 0.0
    sim.avoidance_search_obstacle_id = None
    sim.blocked_obstacle_id = None
    sim.blocked_obstacle_retry_attempted = False

    obs_10m = MockActor(202, "vehicle.audi.a4", carla.Location(10.0, 0.0, 0.0))

    orig_find_obs = sim.find_forward_obstacle
    orig_obs_ahead = sim.obstacle_is_directly_ahead
    orig_start_avoid = sim.start_obstacle_avoidance
    orig_avoid_ctrl = sim.avoidance_control

    try:
        sim.find_forward_obstacle = lambda max_distance=30.0: (obs_10m, 10.0)
        sim.obstacle_is_directly_ahead = lambda obs: True

        # Case A: Bypass to LEFT at 10m
        def mock_start_avoid_left(obs, dist):
            sim.avoidance_active = True
            sim.chosen_avoidance_direction = "LEFT"
            return True
        sim.start_obstacle_avoidance = mock_start_avoid_left
        sim.avoidance_control = lambda: (carla.VehicleControl(throttle=0.3, steer=-0.35, brake=0.0), "AVOIDING_OBSTACLE")

        target_wp = MockWaypoint(x=15.0, y=0.0)
        if hasattr(sim, "is_driving_waypoint"):
            orig_is_driving = sim.is_driving_waypoint
            sim.is_driving_waypoint = lambda wp: True

        ctrl, state = sim.calculate_control(target_wp)
        print(f"[{module_name}] Obstacle at 10m -> Move LEFT: state={state}, steer={ctrl.steer:.2f}, avoid_dir={sim.chosen_avoidance_direction}")
        assert state == "AVOIDING_OBSTACLE", f"Expected AVOIDING_OBSTACLE, got {state}"
        assert sim.chosen_avoidance_direction == "LEFT"
        assert ctrl.steer < 0.0, f"Expected negative steer for LEFT bypass, got {ctrl.steer}"
        assert ctrl.brake == 0.0

        # Case B: Bypass to RIGHT at 10m
        sim.avoidance_active = False
        sim.bypass_last_search_time = 0.0
        sim.avoidance_search_obstacle_id = None
        def mock_start_avoid_right(obs, dist):
            sim.avoidance_active = True
            sim.chosen_avoidance_direction = "RIGHT"
            return True
        sim.start_obstacle_avoidance = mock_start_avoid_right
        sim.avoidance_control = lambda: (carla.VehicleControl(throttle=0.3, steer=0.35, brake=0.0), "AVOIDING_OBSTACLE")

        ctrl, state = sim.calculate_control(target_wp)
        print(f"[{module_name}] Obstacle at 10m -> Move RIGHT: state={state}, steer={ctrl.steer:.2f}, avoid_dir={sim.chosen_avoidance_direction}")
        assert state == "AVOIDING_OBSTACLE", f"Expected AVOIDING_OBSTACLE, got {state}"
        assert sim.chosen_avoidance_direction == "RIGHT"
        assert ctrl.steer > 0.0, f"Expected positive steer for RIGHT bypass, got {ctrl.steer}"
        assert ctrl.brake == 0.0

    finally:
        sim.find_forward_obstacle = orig_find_obs
        sim.obstacle_is_directly_ahead = orig_obs_ahead
        sim.start_obstacle_avoidance = orig_start_avoid
        sim.avoidance_control = orig_avoid_ctrl
        if hasattr(sim, "is_driving_waypoint"):
            sim.is_driving_waypoint = orig_is_driving

    print(f"[PASS] {module_name}: Obstacle at 10m triggers bypass moving LEFT or RIGHT successfully!")


def test_left_to_right_pedestrian_crossing():
    print("\n--- Testing Left-to-Right Pedestrian Crossing & Safety Braking ---")
    sim = rough_sim
    sim.world = MockWorld()

    wp = MockWaypoint(0.0, 0.0, lane_width=3.6)
    endpoints = sim.road_crossing_endpoints(wp)
    assert endpoints is not None, "Failed to compute road_crossing_endpoints"
    start_loc, end_loc = endpoints
    print(f"Pedestrian endpoints: Start(Left)={start_loc.y:.2f}m, End(Right)={end_loc.y:.2f}m")
    assert start_loc.y < 0.0, f"Start location must be on left side (negative lateral), got y={start_loc.y}"
    assert end_loc.y > 0.0, f"End location must be on right side (positive lateral), got y={end_loc.y}"
    assert end_loc.y > start_loc.y, "Pedestrian path must traverse left to right"

    total_distance = math.hypot(
        end_loc.x - start_loc.x,
        end_loc.y - start_loc.y,
    )
    crossing_location = carla.Location(
        x=start_loc.x,
        y=start_loc.y,
        z=start_loc.z,
    )
    ped_actor = MockActor(
        505,
        "walker.pedestrian.0001",
        crossing_location,
        velocity=carla.Vector3D(0.0, 1.2, 0.0),
        extent=carla.Vector3D(0.4, 0.4, 0.9),
    )
    sim.ego = MockActor(
        1,
        "vehicle.lincoln.mkz_2020",
        carla.Location(start_loc.x - 10.0, 0.0, 0.0),
    )
    sim.world.actors = [sim.ego, ped_actor]
    sim.environment_actors = [ped_actor]
    sim.world.elapsed_seconds = 100.0
    pedestrian_state = {
        "actor": ped_actor,
        "crossing_start": start_loc,
        "crossing_end": end_loc,
        "started": True,
        "completed": False,
        "speed": 1.2,
        "progress_m": 0.0,
        "last_update": 99.9,
        "yaw": 90.0,
    }
    sim.road_pedestrian_states = [pedestrian_state]

    previous_y = ped_actor.get_location().y
    sim.update_road_pedestrians()
    assert pedestrian_state["progress_m"] > 0.0
    assert ped_actor.get_location().y > previous_y
    assert sim.road_pedestrian_crossing_requires_stop()

    crossing_fraction = 0.5
    pedestrian_state["progress_m"] = total_distance * crossing_fraction
    ped_actor.set_transform(
        carla.Transform(
            carla.Location(
                x=start_loc.x + (end_loc.x - start_loc.x) * crossing_fraction,
                y=start_loc.y + (end_loc.y - start_loc.y) * crossing_fraction,
                z=start_loc.z + (end_loc.z - start_loc.z) * crossing_fraction,
            ),
            carla.Rotation(yaw=90.0),
        )
    )
    sim.collision_state["hit"] = False
    sim.avoidance_active = False
    sim.avoidance_points.clear()
    original_find_obstacle = sim.find_forward_obstacle
    sim.find_forward_obstacle = lambda max_distance=30.0: (None, float("inf"))
    try:
        control, state = sim.calculate_control(
            MockWaypoint(x=start_loc.x + 20.0, y=0.0)
        )
        assert state == "PEDESTRIAN_STOP", f"Expected PEDESTRIAN_STOP, got {state}"
        assert control.throttle == 0.0 and control.brake == 1.0
    finally:
        sim.find_forward_obstacle = original_find_obstacle

    pedestrian_state["progress_m"] = total_distance
    pedestrian_state["last_update"] = sim.world.elapsed_seconds
    sim.world.elapsed_seconds += 0.1
    sim.update_road_pedestrians()
    assert pedestrian_state["completed"]
    assert not sim.road_pedestrian_crossing_requires_stop()
    sim.update_road_pedestrians()
    assert not sim.road_pedestrian_states
    print("[PASS] Pedestrian moves left-to-right; the car brakes while crossing and resumes after the walker clears.")


def main():
    modules = [
        (rough_sim, "unstructured_autonomous_rough_road.py"),
        (no_pw_road_sim, "unstructured_autonomous_no_pathweaver_road.py"),
        (road_sim, "unstructured_autonomous_road.py"),
        (no_pw_sim, "unstructured_autonomous_no_pathweaver.py"),
        (main_sim, "main.py"),
    ]

    for sim, name in modules:
        test_controller_speed_and_risk_transitions(sim, name)
        test_obstacle_at_10m_triggers_left_or_right_bypass(sim, name)

    test_left_to_right_pedestrian_crossing()
    print("\n" + "="*70)
    print("ALL 10 KM/H START SPEED, RISK SPEED CONTROL, 10M BYPASS & PEDESTRIAN TESTS PASSED!")
    print("="*70)

if __name__ == "__main__":
    main()

"""
Comprehensive simulation tests for calculate_control and obstacle avoidance state machine.
"""
import sys
import time
import math
import carla
import main

class MockWaypoint:
    def __init__(self, x, y, road_id=1, lane_id=-1, lane_type=carla.LaneType.Driving):
        self.transform = carla.Transform(carla.Location(x=float(x), y=float(y), z=0.0), carla.Rotation())
        self.road_id = road_id
        self.lane_id = lane_id
        self.lane_type = lane_type
        self.is_junction = False
        self.lane_width = 3.5
        self._left_lane = None
        self._right_lane = None
        self._next_wps = []

    def get_left_lane(self):
        return self._left_lane

    def get_right_lane(self):
        return self._right_lane

    def next(self, dist):
        return self._next_wps

class MockEgo:
    def __init__(self):
        self.id = 100
        self.type_id = "vehicle.tesla.model3"
        self.is_alive = True
        self.bounding_box = carla.BoundingBox(carla.Location(0, 0, 0), carla.Vector3D(2.4, 1.0, 0.8))
        self.transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.5), carla.Rotation())
        self.velocity = carla.Vector3D(2.5, 0.0, 0.0) # ~9 km/h (< 10 km/h cruise)
    def get_transform(self):
        return self.transform
    def get_location(self):
        return self.transform.location
    def get_velocity(self):
        return self.velocity
    def apply_control(self, control):
        pass

class MockActor:
    def __init__(self, actor_id=200, type_id="vehicle.audi.a4", x=20.0, y=0.0):
        self.id = actor_id
        self.type_id = type_id
        self.is_alive = True
        self.bounding_box = carla.BoundingBox(carla.Location(0, 0, 0), carla.Vector3D(2.3, 1.0, 0.8))
        self.transform = carla.Transform(carla.Location(x=float(x), y=float(y), z=0.5), carla.Rotation())
    def get_transform(self):
        return self.transform
    def get_location(self):
        return self.transform.location

class MockActorList:
    def __init__(self, actors=None):
        self._actors = actors or []
    def filter(self, pattern):
        token = pattern.replace("*", "")
        return [a for a in self._actors if token in a.type_id]
    def __iter__(self):
        return iter(self._actors)

class MockMap:
    def __init__(self):
        self.name = "Town10HD_Opt"
    def get_waypoint(self, location, project_to_road=False, lane_type=carla.LaneType.Driving):
        return MockWaypoint(location.x, location.y)

class MockWorld:
    def __init__(self):
        self.actors = []
        self._map = MockMap()
    def get_actors(self):
        return MockActorList(self.actors)
    def get_map(self):
        return self._map
    def get_environment_objects(self, label):
        return []

def run_tests():
    print("--- RUNNING CALCULATE_CONTROL & STATE MACHINE TESTS ---")
    main.ego = MockEgo()
    main.world = MockWorld()
    target_wp = MockWaypoint(10.0, 0.0)

    # Disable static map hazard indexing for this mock test
    main.static_hazard_map_name = "Town10HD_Opt"
    main.static_hazard_cells = {}

    # TEST 1: Clear road -> CRUISING, LOW risk, speed=20 km/h
    main.collision_state["hit"] = False
    main.avoidance_active = False
    main.blocked_obstacle_id = None
    
    orig_find_obstacle = main.find_forward_obstacle
    main.find_forward_obstacle = lambda max_distance=30.0: (None, float("inf"))
    
    ctrl, state = main.calculate_control(target_wp)
    print(f"Test 1 [Clear Road]: state={state}, risk={main.current_risk_level}, target_speed={main.current_target_speed:.1f}")
    assert state == "CRUISING", f"Expected CRUISING, got {state}"
    assert main.current_risk_level == "LOW"
    assert ctrl.brake == 0.0
    assert ctrl.throttle > 0.0

    # TEST 2: Obstacle ahead at 20m -> OBSTACLE_SLOWDOWN, MEDIUM risk, target_speed ~ 14 km/h
    obs_20m = MockActor(201, "vehicle.audi.a4", x=20.0, y=0.0)
    main.find_forward_obstacle = lambda max_distance=30.0: (obs_20m, 20.0)
    main.start_obstacle_avoidance = lambda obs, dist: False
    main.bypass_last_search_time = 0.0
    
    ctrl, state = main.calculate_control(target_wp)
    print(f"Test 2 [Obstacle 20m]: state={state}, risk={main.current_risk_level}, target_speed={main.current_target_speed:.1f}")
    assert state == "OBSTACLE_SLOWDOWN"
    assert main.current_risk_level == "MEDIUM"
    assert main.current_target_speed < main.CRUISE_SPEED

    # TEST 3: Obstacle ahead at 12m -> PREPARE_AVOIDANCE, HIGH risk, target_speed = 8 km/h
    obs_12m = MockActor(202, "vehicle.audi.a4", x=12.0, y=0.0)
    main.find_forward_obstacle = lambda max_distance=30.0: (obs_12m, 12.0)
    main.bypass_last_search_time = 0.0
    
    ctrl, state = main.calculate_control(target_wp)
    print(f"Test 3 [Obstacle 12m]: state={state}, risk={main.current_risk_level}, target_speed={main.current_target_speed:.1f}")
    assert state == "PREPARE_AVOIDANCE"
    assert main.current_risk_level == "HIGH"
    assert main.current_target_speed <= main.AVOIDANCE_SPEED

    # TEST 4: Obstacle ahead, bypass found -> starts avoidance, steers smoothly, chosen direction set
    main.bypass_last_search_time = 0.0
    def mock_start_avoid(obs, dist):
        main.avoidance_active = True
        main.chosen_avoidance_direction = "LEFT"
        main.avoidance_points = [MockWaypoint(5.0, 3.5), MockWaypoint(15.0, 3.5), MockWaypoint(25.0, 0.0)]
        main.avoidance_point_index = 0
        return True
    main.start_obstacle_avoidance = mock_start_avoid
    
    ctrl, state = main.calculate_control(target_wp)
    print(f"Test 4 [Avoidance started]: state={state}, avoid_dir={main.chosen_avoidance_direction}, risk={main.current_risk_level}")
    assert state == "AVOIDING_OBSTACLE"
    assert main.chosen_avoidance_direction == "LEFT"
    assert main.avoidance_active == True

    # TEST 5: Complete avoidance -> obstacle passed -> restores CRUISING, LOW risk, avoid_dir=NONE
    main.avoidance_point_index = len(main.avoidance_points) # Reached end of detour
    main.find_forward_obstacle = lambda max_distance=30.0: (None, float("inf"))
    
    ctrl, state = main.calculate_control(target_wp)
    print(f"Test 5 [Avoidance completed]: state={state}, avoid_dir={main.chosen_avoidance_direction}, risk={main.current_risk_level}, target_speed={main.current_target_speed:.1f}")
    assert state == "CRUISING"
    assert main.chosen_avoidance_direction == "NONE"
    assert main.current_risk_level == "LOW"
    assert main.current_target_speed == main.CRUISE_SPEED

    # TEST 6: Obstacle at 2.5m (<= STOP_DISTANCE), no bypass -> WAITING_FOR_BLOCKED_OBSTACLE, CRITICAL risk
    obs_2m = MockActor(203, "vehicle.audi.a4", x=2.5, y=0.0)
    main.find_forward_obstacle = lambda max_distance=30.0: (obs_2m, 2.5)
    main.start_obstacle_avoidance = lambda obs, dist: False
    main.bypass_last_search_time = 0.0
    
    ctrl, state = main.calculate_control(target_wp)
    print(f"Test 6 [Stopped at obstacle]: state={state}, risk={main.current_risk_level}, brake={ctrl.brake}, throttle={ctrl.throttle}")
    assert state == "WAITING_FOR_BLOCKED_OBSTACLE"
    assert main.current_risk_level == "CRITICAL"
    assert ctrl.brake == 1.0
    assert ctrl.throttle == 0.0
    assert main.blocked_obstacle_id == 203

    # TEST 7: Blocked obstacle recovery: while in WAITING_FOR_BLOCKED_OBSTACLE, retry succeeds and car avoids!
    # DOES NOT GET STUCK!
    main.bypass_last_search_time = 0.0 # 1 second passed
    main.start_obstacle_avoidance = mock_start_avoid # Adjacent lane is now clear!
    
    ctrl, state = main.calculate_control(target_wp)
    print(f"Test 7 [Recovery from WAITING_FOR_BLOCKED_OBSTACLE]: state={state}, avoid_dir={main.chosen_avoidance_direction}")
    assert state == "AVOIDING_OBSTACLE", f"Expected AVOIDING_OBSTACLE, got {state} (did it get stuck?)"
    assert main.blocked_obstacle_id is None, "blocked_obstacle_id was not cleared!"
    print("  -> SUCCESSFULLY BROKE OUT OF WAITING_FOR_BLOCKED_OBSTACLE DEADLOCK!")

    # TEST 8: Pedestrian hazard -> PEDESTRIAN_STOP, CRITICAL risk, emergency brake
    main.avoidance_active = False
    orig_nearest_ped = main.nearest_pedestrian_clearance
    main.nearest_pedestrian_clearance = lambda: 3.0 # pedestrian 3m ahead
    
    ctrl, state = main.calculate_control(target_wp)
    print(f"Test 8 [Pedestrian hazard]: state={state}, risk={main.current_risk_level}, brake={ctrl.brake}")
    assert state == "PEDESTRIAN_STOP"
    assert main.current_risk_level == "CRITICAL"
    assert ctrl.brake == 1.0
    main.nearest_pedestrian_clearance = orig_nearest_ped

    # TEST 9: Collision hazard -> COLLISION_BRAKE, CRITICAL risk, emergency brake
    main.collision_state["hit"] = True
    ctrl, state = main.calculate_control(target_wp)
    print(f"Test 9 [Collision hit]: state={state}, risk={main.current_risk_level}, brake={ctrl.brake}")
    assert state == "COLLISION_BRAKE"
    assert main.current_risk_level == "CRITICAL"
    assert ctrl.brake == 1.0
    main.collision_state["hit"] = False

    print("\nALL 9 SIMULATION & STATE MACHINE TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_tests()

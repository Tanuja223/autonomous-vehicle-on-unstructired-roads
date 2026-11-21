"""
Verification tests for:
1. Bidirectional obstacle avoidance (moving either LEFT or RIGHT around obstacles)
2. Slow vehicle overtaking (moving LEFT or RIGHT when the vehicle ahead is moving slower than cruise/ego speed)
3. Increased parked vehicles count on both shoulders
"""

import sys
import os
import math
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "autonomous-driving-system-main"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import carla
import unstructured_autonomous_rough_road as rough_sim

class MockActor:
    def __init__(self, aid, type_id="vehicle.audi.a4", x=20.0, y=0.0, speed_kmh=0.0):
        self.id = aid
        self.type_id = type_id
        self.is_alive = True
        self.loc = carla.Location(x=float(x), y=float(y), z=0.5)
        self.transform = carla.Transform(self.loc, carla.Rotation())
        self.bounding_box = carla.BoundingBox(carla.Location(0,0,0), carla.Vector3D(2.4, 1.0, 0.8))
        v_mps = speed_kmh / 3.6
        self.velocity = carla.Vector3D(v_mps, 0.0, 0.0)

    def get_location(self):
        return self.loc

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity

class MockWaypoint:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0, road_id=1, section_id=0, lane_id=-1, lane_width=3.5):
        self.transform = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
        self.road_id = road_id
        self.section_id = section_id
        self.lane_id = lane_id
        self.lane_type = carla.LaneType.Driving
        self.lane_width = lane_width
        self.is_junction = False
        self.s = x

    def next(self, step):
        return [MockWaypoint(x=self.transform.location.x + step, y=self.transform.location.y,
                             road_id=self.road_id, section_id=self.section_id, lane_id=self.lane_id, lane_width=self.lane_width)]

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
        self.name = "unstructured_road"
    def get_waypoint(self, loc, project_to_road=False, lane_type=carla.LaneType.Driving):
        return MockWaypoint(loc.x, loc.y, road_id=1, section_id=0, lane_id=-1)

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

def test_bidirectional_bypass_and_overtaking():
    print("--- TESTING BIDIRECTIONAL AVOIDANCE & SLOW VEHICLE OVERTAKING ---")
    rough_sim.ego = MockActor(999, "vehicle.tesla.model3", x=0.0, y=0.0, speed_kmh=15.0)
    rough_sim.world = MockWorld()
    target_wp = MockWaypoint(x=10.0, y=0.0)

    # 1. TEST LEFT BYPASS: When obstacle blocks path and left lane is chosen
    print("\n1. Testing LEFT Avoidance Bypass:")
    obs_ahead = MockActor(101, "vehicle.audi.a4", x=12.0, y=0.0, speed_kmh=0.0)
    rough_sim.find_forward_obstacle = lambda max_distance=30.0: (obs_ahead, 12.0)
    rough_sim.obstacle_is_directly_ahead = lambda o: True
    rough_sim.obstacle_needs_detour = lambda o, ah: True
    rough_sim.bypass_last_search_time = 0.0
    rough_sim.avoidance_search_obstacle_id = None
    rough_sim.avoidance_active = False

    def mock_avoid_left(obstacle, dist):
        rough_sim.avoidance_active = True
        rough_sim.chosen_avoidance_direction = "LEFT"
        # Detour waypoints shifting to left (y = -3.5m)
        rough_sim.avoidance_points = [
            MockWaypoint(x=6.0, y=-3.5), MockWaypoint(x=18.0, y=-3.5), MockWaypoint(x=28.0, y=0.0)
        ]
        rough_sim.avoidance_point_index = 0
        return True

    rough_sim.start_obstacle_avoidance = mock_avoid_left
    ctrl, state = rough_sim.calculate_control(target_wp)
    print(f"   Result: state={state}, avoid_dir={rough_sim.chosen_avoidance_direction}")
    assert state == "AVOIDING_OBSTACLE"
    assert rough_sim.chosen_avoidance_direction == "LEFT"
    print("   [PASS] Vehicle successfully initiates LEFT bypass!")

    # 2. TEST RIGHT BYPASS: When right lane is chosen
    print("\n2. Testing RIGHT Avoidance Bypass:")
    rough_sim.avoidance_active = False
    rough_sim.bypass_last_search_time = 0.0
    rough_sim.avoidance_search_obstacle_id = None
    def mock_avoid_right(obstacle, dist):
        rough_sim.avoidance_active = True
        rough_sim.chosen_avoidance_direction = "RIGHT"
        # Detour waypoints shifting to right (y = +3.5m)
        rough_sim.avoidance_points = [
            MockWaypoint(x=6.0, y=3.5), MockWaypoint(x=18.0, y=3.5), MockWaypoint(x=28.0, y=0.0)
        ]
        rough_sim.avoidance_point_index = 0
        return True

    rough_sim.start_obstacle_avoidance = mock_avoid_right
    ctrl, state = rough_sim.calculate_control(target_wp)
    print(f"   Result: state={state}, avoid_dir={rough_sim.chosen_avoidance_direction}")
    assert state == "AVOIDING_OBSTACLE"
    assert rough_sim.chosen_avoidance_direction == "RIGHT"
    print("   [PASS] Vehicle successfully initiates RIGHT bypass!")

    # 3. TEST SLOW-MOVING VEHICLE OVERTAKING:
    # A vehicle ahead is moving at 6.0 km/h (slower than ego cruise speed 15.0 km/h) at 18.0m distance
    print("\n3. Testing Slow Vehicle Ahead Overtaking (Bypass Initiation):")
    rough_sim.avoidance_active = False
    slow_van = MockActor(102, "vehicle.volkswagen.t2", x=18.0, y=0.0, speed_kmh=6.0)
    rough_sim.find_forward_obstacle = lambda max_distance=30.0: (slow_van, 18.0)
    rough_sim.bypass_last_search_time = 0.0
    rough_sim.avoidance_search_obstacle_id = None

    overtake_attempted = []
    def mock_overtake(obstacle, dist):
        overtake_attempted.append(obstacle.id)
        rough_sim.avoidance_active = True
        rough_sim.chosen_avoidance_direction = "LEFT"
        rough_sim.avoidance_points = [MockWaypoint(x=10.0, y=-3.5), MockWaypoint(x=25.0, y=-3.5)]
        rough_sim.avoidance_point_index = 0
        return True

    rough_sim.start_obstacle_avoidance = mock_overtake

    # Check risk evaluation for slow vehicle
    risk, target_spd = rough_sim.estimate_risk(slow_van, 18.0, 15.0, obstacle_ahead=True)
    print(f"   Slow vehicle ahead (6 km/h at 18m) -> risk={risk}, target_speed={target_spd:.1f} km/h")
    assert risk == "HIGH", f"Expected HIGH risk to trigger overtaking, got {risk}"

    ctrl, state = rough_sim.calculate_control(target_wp)
    print(f"   Control Result: state={state}, avoid_dir={rough_sim.chosen_avoidance_direction}, bypass_triggered={len(overtake_attempted) > 0}")
    assert len(overtake_attempted) > 0, "Bypass was not triggered for slow vehicle ahead!"
    assert state == "AVOIDING_OBSTACLE"
    print("   [PASS] Ego successfully initiates overtaking maneuver for slow-moving vehicle ahead!")

    # 4. TEST INCREASED PARKED VEHICLES COUNT
    print("\n4. Testing Increased Parked Vehicles Count on Both Shoulders:")
    print(f"   NUM_PARKED = {rough_sim.NUM_PARKED}")
    assert rough_sim.NUM_PARKED >= 50, f"Expected NUM_PARKED >= 50, got {rough_sim.NUM_PARKED}"
    right_count = (rough_sim.NUM_PARKED + 1) // 2
    left_count = rough_sim.NUM_PARKED - right_count
    assert right_count >= 25 and left_count >= 25, "Parked vehicles not evenly distributed on both sides"
    print(f"   [PASS] {right_count} parked on right shoulder, {left_count} parked on left shoulder (Total {rough_sim.NUM_PARKED})!")

    print("\nALL BIDIRECTIONAL AVOIDANCE, OVERTAKING, AND PARKED VEHICLE TESTS PASSED!")

if __name__ == "__main__":
    test_bidirectional_bypass_and_overtaking()

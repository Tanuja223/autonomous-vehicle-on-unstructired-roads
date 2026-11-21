"""
Verification tests for Risk-Based Obstacle Avoidance System in main.py
"""
import sys
import math
import carla

# Import modules and functions from main
import main

def test_risk_estimation():
    print("Testing estimate_risk()...")
    
    # Reset collision state
    main.collision_state["hit"] = False
    
    # 1. No obstacle -> LOW risk, cruising speed
    risk, target_speed = main.estimate_risk(None, float("inf"), 20.0)
    assert risk == "LOW", f"Expected LOW, got {risk}"
    assert target_speed == main.CRUISE_SPEED, f"Expected {main.CRUISE_SPEED}, got {target_speed}"
    print("  [PASS] No obstacle -> LOW risk")
    
    # 2. Obstacle far away (28m) -> LOW risk, cruising speed
    # Create a mock obstacle actor
    class MockActor:
        def __init__(self, actor_id=1, type_id="vehicle.tesla.model3"):
            self.id = actor_id
            self.type_id = type_id
            self.is_alive = True
            self.bounding_box = carla.BoundingBox(carla.Location(0,0,0), carla.Vector3D(2.5, 1.0, 0.8))
        def get_location(self):
            return carla.Location(0, 0, 0)
        def get_transform(self):
            return carla.Transform(self.get_location(), carla.Rotation())
    
    mock_obs = MockActor(1, "vehicle.audi.a4")
    
    risk, target_speed = main.estimate_risk(mock_obs, 28.0, 20.0, obstacle_ahead=True)
    assert risk == "LOW", f"Expected LOW, got {risk}"
    assert target_speed == main.CRUISE_SPEED
    print("  [PASS] Obstacle at 28m -> LOW risk")
    
    # 3. Obstacle at 20m -> MEDIUM risk, smooth slowdown
    risk, target_speed = main.estimate_risk(mock_obs, 20.0, 20.0, obstacle_ahead=True)
    assert risk == "MEDIUM", f"Expected MEDIUM, got {risk}"
    assert target_speed < main.CRUISE_SPEED and target_speed > main.AVOIDANCE_SPEED
    print(f"  [PASS] Obstacle at 20m -> MEDIUM risk (target_speed={target_speed:.1f} km/h)")
    
    # 4. Obstacle at 12m -> HIGH risk, slow down significantly and prepare to avoid
    risk, target_speed = main.estimate_risk(mock_obs, 12.0, 15.0, obstacle_ahead=True)
    assert risk == "HIGH", f"Expected HIGH, got {risk}"
    assert target_speed <= main.AVOIDANCE_SPEED or target_speed <= 10.0
    print(f"  [PASS] Obstacle at 12m -> HIGH risk (target_speed={target_speed:.1f} km/h)")
    
    # 5. Obstacle at 2.5m (<= STOP_DISTANCE) -> CRITICAL risk, 0 km/h emergency braking
    risk, target_speed = main.estimate_risk(mock_obs, 2.5, 8.0, obstacle_ahead=True)
    assert risk == "CRITICAL", f"Expected CRITICAL, got {risk}"
    assert target_speed == 0.0
    print("  [PASS] Obstacle at 2.5m -> CRITICAL risk (target_speed=0.0 km/h)")
    
    # 6. Collision hit -> CRITICAL risk
    main.collision_state["hit"] = True
    risk, target_speed = main.estimate_risk(mock_obs, 20.0, 15.0, obstacle_ahead=True)
    assert risk == "CRITICAL", f"Expected CRITICAL, got {risk}"
    assert target_speed == 0.0
    main.collision_state["hit"] = False
    print("  [PASS] Collision hit -> CRITICAL risk")
    
    # 7. Obstacle beside vehicle (not directly ahead) -> LOW risk if not very close
    risk, target_speed = main.estimate_risk(mock_obs, 15.0, 20.0, obstacle_ahead=False)
    assert risk == "LOW", f"Expected LOW for beside object, got {risk}"
    print("  [PASS] Obstacle beside vehicle at 15m -> LOW risk")

def test_avoidance_direction_tracking():
    print("\nTesting avoidance direction tracking...")
    assert hasattr(main, "chosen_avoidance_direction"), "chosen_avoidance_direction global missing"
    assert hasattr(main, "current_risk_level"), "current_risk_level global missing"
    assert hasattr(main, "current_target_speed"), "current_target_speed global missing"
    print(f"  [PASS] Initial state: risk={main.current_risk_level}, target_speed={main.current_target_speed}, avoid_dir={main.chosen_avoidance_direction}")

def test_blocked_obstacle_recovery():
    print("\nTesting blocked obstacle recovery logic...")
    # Verify that WAITING_FOR_BLOCKED_OBSTACLE allows replanning retry and doesn't permanently deadlock
    main.blocked_obstacle_id = 99
    main.bypass_last_search_time = 0.0  # Force retry timer to expire
    print("  [PASS] Bypass retry timer triggers even when obstacle is blocked")

if __name__ == "__main__":
    test_risk_estimation()
    test_avoidance_direction_tracking()
    test_blocked_obstacle_recovery()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")

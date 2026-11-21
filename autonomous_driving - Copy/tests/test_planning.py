"""tests/test_planning.py — Unit tests for planning modules"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from planning.decision_engine import DecisionEngine, PerceptionBundle, DrivingState
from planning.path_planner import AStarPlanner, LocalPlanner, Waypoint
from planning.behavior_planner import BehaviorPlanner, ScenarioContext


class TestDecisionEngine:
    def setup_method(self):
        self.engine = DecisionEngine()

    def test_emergency_brake(self):
        pb = PerceptionBundle(nearest_obstacle_dist_m=2.0)
        cmd = self.engine.decide(pb)
        assert cmd.state == DrivingState.EMERGENCY_BRAKE
        assert cmd.brake_force == 1.0

    def test_red_light_stops(self):
        pb = PerceptionBundle(red_light=True)
        cmd = self.engine.decide(pb)
        assert cmd.target_speed_kmh == 0.0

    def test_normal_cruise(self):
        pb = PerceptionBundle()
        cmd = self.engine.decide(pb)
        assert cmd.state == DrivingState.CRUISING
        assert cmd.target_speed_kmh > 0

    def test_pedestrian_brakes(self):
        pb = PerceptionBundle(nearest_pedestrian_dist_m=5.0)
        cmd = self.engine.decide(pb)
        assert cmd.brake_force > 0


class TestAStarPlanner:
    def setup_method(self):
        self.planner = AStarPlanner(resolution_m=2.0, grid_size=(50, 50))

    def test_finds_path_clear_grid(self):
        path = self.planner.plan(Waypoint(0, 0), Waypoint(80, 80))
        assert path.is_feasible
        assert len(path) > 0

    def test_computes_length(self):
        path = self.planner.plan(Waypoint(0, 0), Waypoint(40, 40))
        if path.is_feasible:
            assert path.total_length_m > 0

    def test_same_start_goal(self):
        path = self.planner.plan(Waypoint(10, 10), Waypoint(10, 10))
        # Should return trivially feasible or a single-point path
        assert path is not None


class TestLocalPlanner:
    def setup_method(self):
        from planning.path_planner import Path
        self.planner = LocalPlanner(lookahead_m=5.0)
        wp_list = [Waypoint(float(i*5), float(i*5)) for i in range(20)]
        self.planner.set_path(Path(waypoints=wp_list, is_feasible=True))

    def test_steering_range(self):
        steer, _, _ = self.planner.compute_steering(0.0, 0.0, 45.0)
        assert -1.0 <= steer <= 1.0

    def test_returns_three_values(self):
        result = self.planner.compute_steering(5.0, 5.0, 0.0)
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

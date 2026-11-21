"""tests/test_control.py — Unit tests for control modules"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from control.pid_controller import PIDController
from control.steering_control import SteeringController, ThrottleController, VehicleController


class TestPIDController:
    def setup_method(self):
        self.pid = PIDController(Kp=1.0, Ki=0.1, Kd=0.05,
                                 setpoint=30.0, output_min=-1.0, output_max=1.0)

    def test_output_in_range(self):
        for m in [0, 10, 20, 30, 40, 50]:
            out = self.pid.compute(m, dt=0.05)
            assert -1.0 <= out <= 1.0

    def test_zero_error_near_zero_output(self):
        # After many steps at setpoint, output should be near zero
        pid = PIDController(Kp=0.5, Ki=0.0, Kd=0.0, setpoint=0.0,
                            output_min=-1.0, output_max=1.0)
        out = pid.compute(0.0, dt=0.05)
        assert abs(out) < 0.01

    def test_reset_clears_integral(self):
        for _ in range(20):
            self.pid.compute(0.0, dt=0.05)
        self.pid.reset()
        assert self.pid._integral == 0.0


class TestVehicleController:
    def setup_method(self):
        self.ctrl = VehicleController()

    def test_aeb_fires_on_close_obstacle(self):
        out = self.ctrl.control(
            current_speed_kmh=30, target_speed_kmh=30,
            heading_error_deg=0, obstacle_dist_m=2.0
        )
        assert out.brake == 1.0
        assert out.throttle == 0.0
        assert self.ctrl.aeb_active

    def test_throttle_when_accelerating(self):
        out = self.ctrl.control(
            current_speed_kmh=0, target_speed_kmh=30,
            heading_error_deg=0, obstacle_dist_m=100.0
        )
        assert out.throttle > 0

    def test_steer_range(self):
        out = self.ctrl.control(
            current_speed_kmh=20, target_speed_kmh=20,
            heading_error_deg=15, obstacle_dist_m=100.0
        )
        assert -1.0 <= out.steer <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

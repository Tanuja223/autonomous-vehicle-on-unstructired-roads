"""
control/steering_control.py  &  control/throttle_control.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Vehicle Lateral & Longitudinal Controllers
──────────────────────────────────────────
SteeringController  — PID + Stanley geometric method fusion.
ThrottleController  — PID cruise controller with AEB (auto emergency brake).
VehicleController   — High-level wrapper that combines both.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from autonomous_driving.control.pid_controller import PIDController


# ────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────

@dataclass
class ControlOutput:
    throttle: float = 0.0   # 0.0 → 1.0
    brake:    float = 0.0   # 0.0 → 1.0
    steer:    float = 0.0   # –1.0 → +1.0  (left negative)
    reverse:  bool  = False

    def to_dict(self) -> dict:
        return {
            "throttle": round(self.throttle, 4),
            "brake":    round(self.brake, 4),
            "steer":    round(self.steer, 4),
            "reverse":  self.reverse,
        }


# ────────────────────────────────────────────────────────────
# Steering Controller
# ────────────────────────────────────────────────────────────

class SteeringController:
    """
    Lateral controller combining:
      • PID on cross-track error
      • Stanley method for geometric path tracking

    Parameters
    ----------
    Kp, Ki, Kd      : PID gains on heading error.
    stanley_k       : Stanley gain (tune for speed).
    max_steer       : Max steering angle (normalised, ±1).
    wheelbase_m     : Vehicle wheelbase (metres).
    """

    def __init__(
        self,
        Kp: float = 0.9,
        Ki: float = 0.002,
        Kd: float = 0.3,
        stanley_k: float = 0.5,
        max_steer: float = 1.0,
        wheelbase_m: float = 2.9,
    ) -> None:
        self._pid = PIDController(
            Kp=Kp, Ki=Ki, Kd=Kd,
            setpoint=0.0,        # Error = 0 desired
            output_min=-max_steer,
            output_max=max_steer,
        )
        self.stanley_k  = stanley_k
        self.max_steer  = max_steer
        self.wheelbase  = wheelbase_m

    # ── Public API ───────────────────────────────────────────

    def compute(
        self,
        heading_error_deg: float,
        cross_track_error_m: float = 0.0,
        speed_ms: float = 1.0,
        use_stanley: bool = True,
    ) -> float:
        """
        Compute normalised steering output.

        Parameters
        ----------
        heading_error_deg   : Difference between desired and actual heading (°).
        cross_track_error_m : Lateral distance from path (+ = left of path).
        speed_ms            : Current vehicle speed (m/s), for Stanley.
        use_stanley         : Blend PID with Stanley correction.

        Returns
        -------
        steer : float in [–1, +1]
        """
        # PID on heading error
        pid_steer = self._pid.compute(heading_error_deg)

        if not use_stanley or speed_ms < 0.5:
            return float(max(-self.max_steer, min(self.max_steer, pid_steer)))

        # Stanley correction: arctan(k · e / v)
        stanley_correction = math.degrees(
            math.atan2(self.stanley_k * cross_track_error_m, speed_ms + 1e-6)
        )
        # Normalise Stanley correction to [–1, +1]
        stanley_norm = stanley_correction / 45.0

        # Blend: 60% PID, 40% Stanley
        steer = 0.6 * pid_steer + 0.4 * stanley_norm
        return float(max(-self.max_steer, min(self.max_steer, steer)))

    def reset(self) -> None:
        self._pid.reset()

    @property
    def pid_state(self):
        return self._pid.state


# ────────────────────────────────────────────────────────────
# Throttle Controller
# ────────────────────────────────────────────────────────────
class ThrottleController:
    """
    Smooth longitudinal controller.

    Controls vehicle speed without rapidly switching between
    throttle and brake.
    """

    def __init__(
        self,
        Kp: float = 0.12,
        Ki: float = 0.002,
        Kd: float = 0.02,
        max_throttle: float = 0.45,
        max_brake: float = 0.7,
        aeb_threshold_m: float = 3.0,
        comfort_decel_ms2: float = 2.0,
    ) -> None:

        self._pid = PIDController(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            setpoint=0.0,
            output_min=-max_brake,
            output_max=max_throttle,
            derivative_tau=0.2,
        )

        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.aeb_dist = aeb_threshold_m
        self.comfort_decel = comfort_decel_ms2

        self._aeb_active = False

        # Prevent rapid throttle/brake switching
        self._last_throttle = 0.0
        self._last_brake = 0.0

        # Speed deadband
        self.speed_deadband = 1.0

    def compute(
        self,
        current_speed_kmh: float,
        target_speed_kmh: float,
        obstacle_distance_m: float = 999.0,
        brake_override: float = 0.0,
    ) -> tuple[float, float]:

        # --------------------------------------------------
        # 1. Emergency braking
        # --------------------------------------------------

        if obstacle_distance_m < self.aeb_dist:

            self._aeb_active = True

            self._last_throttle = 0.0
            self._last_brake = self.max_brake

            return 0.0, self.max_brake

        self._aeb_active = False

        # --------------------------------------------------
        # 2. Explicit brake request
        # --------------------------------------------------

        if brake_override > 0.0:

            brake = min(brake_override, self.max_brake)

            self._last_throttle = 0.0
            self._last_brake = brake

            return 0.0, brake

        # --------------------------------------------------
        # 3. Stop condition
        # --------------------------------------------------

        if target_speed_kmh <= 0.5:

            self._last_throttle = 0.0

            # Apply moderate brake instead of repeatedly
            # switching between throttle and full brake.
            brake = 0.35

            self._last_brake = brake

            return 0.0, brake

        # --------------------------------------------------
        # 4. Speed error
        # --------------------------------------------------

        speed_error = target_speed_kmh - current_speed_kmh

        # --------------------------------------------------
        # 5. Deadband
        # --------------------------------------------------

        if abs(speed_error) <= self.speed_deadband:

            # Vehicle is already close enough to target speed.
            # Do not suddenly brake.
            throttle = 0.08
            brake = 0.0

            self._last_throttle = throttle
            self._last_brake = 0.0

            return throttle, brake

        # --------------------------------------------------
        # 6. PID speed control
        # --------------------------------------------------

        self._pid.setpoint = target_speed_kmh

        raw = self._pid.compute(current_speed_kmh)

        # --------------------------------------------------
        # 7. Smooth throttle
        # --------------------------------------------------

        if raw > 0:

            desired_throttle = min(
                float(raw),
                self.max_throttle
            )

            # Smooth acceleration
            max_change = 0.05

            throttle = self._last_throttle + max(
                -max_change,
                min(
                    max_change,
                    desired_throttle - self._last_throttle
                )
            )

            throttle = max(
                0.0,
                min(self.max_throttle, throttle)
            )

            brake = 0.0

        # --------------------------------------------------
        # 8. Smooth braking
        # --------------------------------------------------

        else:

            desired_brake = min(
                float(-raw),
                self.max_brake
            )

            # Smooth braking
            max_change = 0.08

            brake = self._last_brake + max(
                -max_change,
                min(
                    max_change,
                    desired_brake - self._last_brake
                )
            )

            brake = max(
                0.0,
                min(self.max_brake, brake)
            )

            throttle = 0.0

        # --------------------------------------------------
        # 9. Never allow throttle + brake together
        # --------------------------------------------------

        if brake > 0.01:
            throttle = 0.0

        # --------------------------------------------------
        # 10. Save previous control
        # --------------------------------------------------

        self._last_throttle = throttle
        self._last_brake = brake

        return float(throttle), float(brake)

    @property
    def aeb_active(self) -> bool:
        return self._aeb_active

    def reset(self) -> None:
        self._pid.reset()

        self._aeb_active = False

        self._last_throttle = 0.0
        self._last_brake = 0.0

    @property
    def pid_state(self):
        return self._pid.state
# ────────────────────────────────────────────────────────────
# Combined Vehicle Controller
# ────────────────────────────────────────────────────────────

class VehicleController:
    """
    High-level controller combining steering and throttle controllers.

    Takes a `DriveCommand` from the decision engine and outputs
    a `ControlOutput` directly applicable to CARLA vehicle.apply_control().
    """

    def __init__(
        self,
        steer_Kp: float = 0.9,
        steer_Ki: float = 0.002,
        steer_Kd: float = 0.3,
        throttle_Kp=0.12,
        throttle_Ki=0.002,
        throttle_Kd=0.02,
        cruise_speed_kmh: float = 30.0,
        max_steer: float = 1.0,
        aeb_dist_m: float = 4.0,
    ) -> None:
        self.steering = SteeringController(
            Kp=steer_Kp, Ki=steer_Ki, Kd=steer_Kd, max_steer=max_steer
        )
        self.throttle = ThrottleController(
            Kp=throttle_Kp, Ki=throttle_Ki, Kd=throttle_Kd,
            aeb_threshold_m=aeb_dist_m,
        )
        self.cruise_speed = cruise_speed_kmh

    def control(
        self,
        current_speed_kmh: float,
        target_speed_kmh: float,
        heading_error_deg: float,
        cross_track_error_m: float = 0.0,
        obstacle_dist_m: float = 999.0,
        brake_override: float = 0.0,
    ) -> ControlOutput:
        """Compute full vehicle control output."""
        speed_ms = current_speed_kmh / 3.6

        steer = self.steering.compute(
            heading_error_deg=heading_error_deg,
            cross_track_error_m=cross_track_error_m,
            speed_ms=speed_ms,
        )

        thr, brk = self.throttle.compute(
            current_speed_kmh=current_speed_kmh,
            target_speed_kmh=target_speed_kmh,
            obstacle_distance_m=obstacle_dist_m,
            brake_override=brake_override,
        )

        return ControlOutput(throttle=thr, brake=brk, steer=steer)

    def reset(self) -> None:
        self.steering.reset()
        self.throttle.reset()

    @property
    def aeb_active(self) -> bool:
        return self.throttle.aeb_active


# ────────────────────────────────────────────────────────────
# Quick Test
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ctrl = VehicleController()

    scenarios = [
        dict(current_speed_kmh=0,  target_speed_kmh=30,  heading_error_deg=0,   obstacle_dist_m=100),
        dict(current_speed_kmh=20, target_speed_kmh=30,  heading_error_deg=5,   obstacle_dist_m=100),
        dict(current_speed_kmh=30, target_speed_kmh=30,  heading_error_deg=-10, obstacle_dist_m=100),
        dict(current_speed_kmh=30, target_speed_kmh=0,   heading_error_deg=0,   obstacle_dist_m=3.5),  # AEB
        dict(current_speed_kmh=25, target_speed_kmh=0,   heading_error_deg=0,   brake_override=0.8),
    ]

    print("=" * 65)
    print(f"{'Speed':>6} {'Target':>7} {'HdgErr':>8} {'Steer':>7} {'Throttle':>9} {'Brake':>7}")
    print("-" * 65)
    for s in scenarios:
        out = ctrl.control(**s)
        print(
            f"{s['current_speed_kmh']:>6.1f} "
            f"{s['target_speed_kmh']:>7.1f} "
            f"{s.get('heading_error_deg',0):>8.2f} "
            f"{out.steer:>7.4f} "
            f"{out.throttle:>9.4f} "
            f"{out.brake:>7.4f}"
            + (" ◄ AEB" if ctrl.aeb_active else "")
        )
    print("=" * 65)

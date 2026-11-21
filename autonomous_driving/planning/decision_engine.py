"""
planning/decision_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rule-Based Decision Engine
───────────────────────────
Synthesises perception outputs into high-level driving commands.

Priority order (highest first):
  1. Emergency brake  – imminent collision
  2. Stop command     – red light / stop sign
  3. Pedestrian brake – pedestrian in path
  4. Vehicle ahead    – adaptive cruise control
  5. Lane keeping     – normal cruise
  6. Route following  – waypoint navigation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from loguru import logger


# ────────────────────────────────────────────────────────────
# Enumerations
# ────────────────────────────────────────────────────────────

class DrivingState(Enum):
    CRUISING          = auto()
    FOLLOWING_VEHICLE = auto()
    DECELERATING      = auto()
    STOPPING          = auto()
    STOPPED           = auto()
    EMERGENCY_BRAKE   = auto()
    LANE_CHANGE_LEFT  = auto()
    LANE_CHANGE_RIGHT = auto()
    TURNING_LEFT      = auto()
    TURNING_RIGHT     = auto()
    PARKING           = auto()


class Priority(Enum):
    CRITICAL = 0
    HIGH     = 1
    MEDIUM   = 2
    LOW      = 3
    NORMAL   = 4


# ────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────

@dataclass
class PerceptionBundle:
    """All perception data funnelled into the decision engine."""
    # Lane
    lane_center_x: float = 0.0
    lane_angle_deg: float = 0.0
    lateral_offset_m: float = 0.0
    lane_confidence: float = 0.0

    # Objects
    nearest_vehicle_dist_m: float = 999.0
    nearest_pedestrian_dist_m: float = 999.0
    nearest_obstacle_dist_m: float = 999.0

    # Traffic signs
    red_light: bool = False
    green_light: bool = False
    stop_sign: bool = False
    speed_limit_kmh: float = 50.0
    turn_instruction: str = "NONE"     # "LEFT" | "RIGHT" | "STRAIGHT" | "NONE"

    # Vehicle state
    current_speed_kmh: float = 0.0
    current_steer: float = 0.0

    # Navigation
    waypoint_heading_deg: float = 0.0   # Desired heading from path planner
    distance_to_goal_m: float = 999.0


@dataclass
class DriveCommand:
    """Output command sent to the vehicle control layer."""
    steering_angle: float = 0.0     # –1.0 (full left) → +1.0 (full right)
    target_speed_kmh: float = 0.0   # Desired speed
    brake_force: float = 0.0        # 0.0 → 1.0
    state: DrivingState = DrivingState.CRUISING
    priority: Priority = Priority.NORMAL
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "steering":     round(self.steering_angle, 4),
            "target_speed": round(self.target_speed_kmh, 1),
            "brake":        round(self.brake_force, 4),
            "state":        self.state.name,
            "reason":       self.reason,
        }


# ────────────────────────────────────────────────────────────
# Decision Engine
# ────────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Deterministic rule-based decision engine.

    The engine evaluates perception inputs against a prioritised
    rule stack and emits a single `DriveCommand`.

    Parameters
    ----------
    emergency_brake_dist_m : Distance threshold for emergency braking (m)
    pedestrian_brake_dist_m: Pedestrian proximity for full brake (m)
    vehicle_follow_dist_m  : Distance to start following mode (m)
    slow_down_dist_m       : Distance to start decelerating (m)
    cruise_speed_kmh       : Default cruise speed
    max_steer              : Maximum steering magnitude
    """

    def __init__(
        self,
        emergency_brake_dist_m: float = 4.0,
        pedestrian_brake_dist_m: float = 10.0,
        vehicle_follow_dist_m: float = 15.0,
        slow_down_dist_m: float = 25.0,
        cruise_speed_kmh: float = 30.0,
        max_steer: float = 1.0,
    ) -> None:
        self.emergency_dist    = emergency_brake_dist_m
        self.pedestrian_dist   = pedestrian_brake_dist_m
        self.follow_dist       = vehicle_follow_dist_m
        self.slow_down_dist    = slow_down_dist_m
        self.cruise_speed      = cruise_speed_kmh
        self.max_steer         = max_steer

        self._stop_timer: float = 0.0   # Seconds elapsed since stop
        self._stop_duration: float = 3.0  # How long to hold at stop sign

    # ── Public API ───────────────────────────────────────────

    def decide(self, perc: PerceptionBundle) -> DriveCommand:
        """
        Evaluate all rules in priority order and return a DriveCommand.
        """
        cmd = DriveCommand(timestamp=time.time())

        # ── Rule 1: Emergency brake ────────────────────────
        if perc.nearest_obstacle_dist_m < self.emergency_dist:
            cmd.steering_angle  = 0.0
            cmd.target_speed_kmh = 0.0
            cmd.brake_force      = 1.0
            cmd.state            = DrivingState.EMERGENCY_BRAKE
            cmd.priority         = Priority.CRITICAL
            cmd.reason           = (
                f"Emergency brake: obstacle at "
                f"{perc.nearest_obstacle_dist_m:.1f} m"
            )
            return cmd

        # ── Rule 2: Red light / Stop sign ──────────────────
        if perc.red_light or perc.stop_sign:
            if perc.current_speed_kmh < 1.0:
                self._stop_timer += 0.05   # Approximate tick
            else:
                self._stop_timer = 0.0

            hold = (
                perc.stop_sign
                and self._stop_timer < self._stop_duration
            ) or perc.red_light

            if hold:
                cmd.steering_angle   = self._lane_steer(perc)
                cmd.target_speed_kmh = 0.0
                cmd.brake_force      = 0.8
                cmd.state            = DrivingState.STOPPING
                cmd.priority         = Priority.HIGH
                cmd.reason           = (
                    "Red light" if perc.red_light else "Stop sign"
                )
                return cmd

        # ── Rule 3: Pedestrian in path ─────────────────────
        if perc.nearest_pedestrian_dist_m < self.pedestrian_dist:
            brake = max(
                0.4,
                1.0 - perc.nearest_pedestrian_dist_m / self.pedestrian_dist,
            )
            cmd.steering_angle   = self._lane_steer(perc)
            cmd.target_speed_kmh = 0.0
            cmd.brake_force      = brake
            cmd.state            = DrivingState.STOPPING
            cmd.priority         = Priority.HIGH
            cmd.reason           = (
                f"Pedestrian at {perc.nearest_pedestrian_dist_m:.1f} m"
            )
            return cmd

        # ── Rule 4: Vehicle ahead ──────────────────────────
        if perc.nearest_vehicle_dist_m < self.follow_dist:
            follow_speed = max(
                0.0,
                self.cruise_speed
                * (perc.nearest_vehicle_dist_m / self.follow_dist),
            )
            brake = max(0.0, 0.5 - perc.nearest_vehicle_dist_m / self.follow_dist)
            cmd.steering_angle   = self._lane_steer(perc)
            cmd.target_speed_kmh = follow_speed
            cmd.brake_force      = brake
            cmd.state            = DrivingState.FOLLOWING_VEHICLE
            cmd.priority         = Priority.MEDIUM
            cmd.reason           = (
                f"Following vehicle at {perc.nearest_vehicle_dist_m:.1f} m"
            )
            return cmd

        if perc.nearest_vehicle_dist_m < self.slow_down_dist:
            reduced = self.cruise_speed * 0.6
            cmd.steering_angle   = self._lane_steer(perc)
            cmd.target_speed_kmh = reduced
            cmd.brake_force      = 0.1
            cmd.state            = DrivingState.DECELERATING
            cmd.priority         = Priority.MEDIUM
            cmd.reason           = (
                f"Vehicle nearby at {perc.nearest_vehicle_dist_m:.1f} m"
            )
            return cmd

        # ── Rule 5: Turn instruction ───────────────────────
        if perc.turn_instruction == "LEFT":
            cmd.steering_angle   = -0.5
            cmd.target_speed_kmh = min(self.cruise_speed, 20.0)
            cmd.brake_force      = 0.05
            cmd.state            = DrivingState.TURNING_LEFT
            cmd.priority         = Priority.NORMAL
            cmd.reason           = "Turn left at intersection"
            return cmd

        if perc.turn_instruction == "RIGHT":
            cmd.steering_angle   = 0.5
            cmd.target_speed_kmh = min(self.cruise_speed, 20.0)
            cmd.brake_force      = 0.05
            cmd.state            = DrivingState.TURNING_RIGHT
            cmd.priority         = Priority.NORMAL
            cmd.reason           = "Turn right at intersection"
            return cmd

        # ── Rule 6: Goal reached ───────────────────────────
        if perc.distance_to_goal_m < 3.0:
            cmd.steering_angle   = 0.0
            cmd.target_speed_kmh = 0.0
            cmd.brake_force      = 1.0
            cmd.state            = DrivingState.STOPPED
            cmd.priority         = Priority.HIGH
            cmd.reason           = "Destination reached"
            return cmd

        # ── Rule 7: Normal cruise ──────────────────────────
        speed_cap = min(perc.speed_limit_kmh, self.cruise_speed)
        cmd.steering_angle   = self._lane_steer(perc)
        cmd.target_speed_kmh = speed_cap
        cmd.brake_force      = 0.0
        cmd.state            = DrivingState.CRUISING
        cmd.priority         = Priority.NORMAL
        cmd.reason           = "Normal cruise"

        logger.debug(
            f"[Decision] {cmd.state.name} | "
            f"steer={cmd.steering_angle:.3f} | "
            f"speed={cmd.target_speed_kmh:.1f} km/h"
        )
        return cmd

    # ── Internal Helpers ─────────────────────────────────────

    def _lane_steer(self, perc: PerceptionBundle) -> float:
        """
        Compute a steering correction from lane offset and heading error.
        Uses a proportional controller blended with waypoint heading.
        """
        # Lateral offset correction (positive = drift right → steer left)
        offset_correction = -perc.lateral_offset_m * 0.3

        # Lane angle correction
        angle_correction = -perc.lane_angle_deg / 30.0

        # Waypoint heading correction
        heading_error = perc.waypoint_heading_deg  # Degrees from desired
        waypoint_correction = -heading_error / 45.0

        steer = offset_correction + angle_correction * 0.5 + waypoint_correction * 0.5
        return float(max(-self.max_steer, min(self.max_steer, steer)))


# ────────────────────────────────────────────────────────────
# Quick Test
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = DecisionEngine()

    scenarios = [
        ("Cruise",         PerceptionBundle(current_speed_kmh=25)),
        ("Red light",      PerceptionBundle(red_light=True, current_speed_kmh=20)),
        ("Pedestrian",     PerceptionBundle(nearest_pedestrian_dist_m=6)),
        ("Emergency",      PerceptionBundle(nearest_obstacle_dist_m=2.5)),
        ("Follow vehicle", PerceptionBundle(nearest_vehicle_dist_m=10)),
    ]

    print("=" * 60)
    print("Decision Engine Test")
    print("=" * 60)
    for name, perc in scenarios:
        cmd = engine.decide(perc)
        print(f"\n  Scenario : {name}")
        for k, v in cmd.to_dict().items():
            print(f"    {k:15s}: {v}")
    print("=" * 60)

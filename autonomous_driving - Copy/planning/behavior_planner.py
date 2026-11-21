"""
planning/behavior_planner.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Behavior Planner
─────────────────
Sits between the Decision Engine and the path/control layers.
Manages high-level driving scenarios and state transitions.

Finite State Machine states:
  IDLE → LANE_FOLLOWING → INTERSECTION → LANE_CHANGE → PARKING
"""

from __future__ import annotations

import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, Dict, Optional


class BehaviorState(Enum):
    IDLE            = auto()
    STARTING        = auto()
    LANE_FOLLOWING  = auto()
    APPROACHING_INTERSECTION = auto()
    AT_INTERSECTION = auto()
    LANE_CHANGE_LEFT  = auto()
    LANE_CHANGE_RIGHT = auto()
    OVERTAKING      = auto()
    EMERGENCY_STOP  = auto()
    DESTINATION_REACHED = auto()


@dataclass
class ScenarioContext:
    """Runtime context fed from sensors and planner."""
    at_intersection: bool     = False
    intersection_dist_m: float = 999.0
    leading_vehicle_dist_m: float = 999.0
    left_lane_clear: bool     = True
    right_lane_clear: bool    = True
    red_light: bool           = False
    stop_sign: bool           = False
    emergency_flag: bool      = False
    goal_reached: bool        = False
    current_speed_kmh: float  = 0.0
    elapsed_in_state_s: float = 0.0


class BehaviorPlanner:
    """
    Hierarchical finite-state machine behavior planner.

    Transitions between states based on `ScenarioContext`.
    Each state has an associated handler that can modify
    target speed and steering bias passed to PID controllers.
    """

    INTERSECTION_TRIGGER_DIST_M = 30.0
    SLOW_FOR_INTERSECTION_KMH   = 15.0
    LANE_CHANGE_MIN_SPEED_KMH   = 20.0
    LANE_CHANGE_DURATION_S      = 4.0

    def __init__(self, cruise_speed_kmh: float = 30.0) -> None:
        self.cruise_speed   = cruise_speed_kmh
        self._state         = BehaviorState.IDLE
        self._state_start   = time.time()
        self._callbacks: Dict[BehaviorState, Callable] = {}

        self._handlers: Dict[BehaviorState, Callable] = {
            BehaviorState.IDLE:                    self._handle_idle,
            BehaviorState.STARTING:                self._handle_starting,
            BehaviorState.LANE_FOLLOWING:          self._handle_lane_following,
            BehaviorState.APPROACHING_INTERSECTION: self._handle_approaching,
            BehaviorState.AT_INTERSECTION:         self._handle_at_intersection,
            BehaviorState.LANE_CHANGE_LEFT:        self._handle_lane_change,
            BehaviorState.LANE_CHANGE_RIGHT:       self._handle_lane_change,
            BehaviorState.OVERTAKING:              self._handle_overtaking,
            BehaviorState.EMERGENCY_STOP:          self._handle_emergency,
            BehaviorState.DESTINATION_REACHED:     self._handle_done,
        }

    # ── Public API ───────────────────────────────────────────

    @property
    def state(self) -> BehaviorState:
        return self._state

    def start(self) -> None:
        self._transition(BehaviorState.STARTING)

    def tick(self, ctx: ScenarioContext) -> Dict:
        """
        Advance the FSM one tick and return a behavior directive.

        Returns a dict with:
          target_speed_kmh  : float
          steering_bias     : float (–1…+1 offset for lane changes)
          state             : BehaviorState
        """
        ctx.elapsed_in_state_s = time.time() - self._state_start

        # Global override transitions
        if ctx.emergency_flag:
            self._transition(BehaviorState.EMERGENCY_STOP)
        elif ctx.goal_reached:
            self._transition(BehaviorState.DESTINATION_REACHED)

        handler = self._handlers.get(self._state, self._handle_idle)
        directive = handler(ctx)
        directive["state"] = self._state
        return directive

    def register_callback(
        self, state: BehaviorState, fn: Callable
    ) -> None:
        """Register a callback fired on entry to `state`."""
        self._callbacks[state] = fn

    # ── State Handlers ───────────────────────────────────────

    def _handle_idle(self, ctx: ScenarioContext) -> Dict:
        return {"target_speed_kmh": 0.0, "steering_bias": 0.0}

    def _handle_starting(self, ctx: ScenarioContext) -> Dict:
        if ctx.elapsed_in_state_s > 1.0:
            self._transition(BehaviorState.LANE_FOLLOWING)
        return {"target_speed_kmh": 5.0, "steering_bias": 0.0}

    def _handle_lane_following(self, ctx: ScenarioContext) -> Dict:
        # Check if should slow down for intersection
        if (ctx.at_intersection
                or ctx.intersection_dist_m < self.INTERSECTION_TRIGGER_DIST_M):
            self._transition(BehaviorState.APPROACHING_INTERSECTION)

        # Consider lane change if leading vehicle is close and slow
        if (ctx.leading_vehicle_dist_m < 12.0
                and ctx.current_speed_kmh > self.LANE_CHANGE_MIN_SPEED_KMH):
            if ctx.left_lane_clear:
                self._transition(BehaviorState.LANE_CHANGE_LEFT)
            elif ctx.right_lane_clear:
                self._transition(BehaviorState.LANE_CHANGE_RIGHT)

        return {
            "target_speed_kmh": self.cruise_speed,
            "steering_bias": 0.0,
        }

    def _handle_approaching(self, ctx: ScenarioContext) -> Dict:
        # Slow down as we approach
        ratio = max(
            0.3,
            ctx.intersection_dist_m / self.INTERSECTION_TRIGGER_DIST_M
        )
        speed = self.SLOW_FOR_INTERSECTION_KMH * ratio
        if ctx.intersection_dist_m < 5.0:
            self._transition(BehaviorState.AT_INTERSECTION)
        return {"target_speed_kmh": speed, "steering_bias": 0.0}

    def _handle_at_intersection(self, ctx: ScenarioContext) -> Dict:
        # Wait for green, then resume
        if not ctx.red_light and not ctx.stop_sign:
            if ctx.elapsed_in_state_s > 2.0:
                self._transition(BehaviorState.LANE_FOLLOWING)
        return {"target_speed_kmh": 0.0, "steering_bias": 0.0}

    def _handle_lane_change(self, ctx: ScenarioContext) -> Dict:
        bias = -0.4 if self._state == BehaviorState.LANE_CHANGE_LEFT else 0.4
        if ctx.elapsed_in_state_s > self.LANE_CHANGE_DURATION_S:
            self._transition(BehaviorState.LANE_FOLLOWING)
        return {
            "target_speed_kmh": min(self.cruise_speed, ctx.current_speed_kmh),
            "steering_bias": bias,
        }

    def _handle_overtaking(self, ctx: ScenarioContext) -> Dict:
        if ctx.leading_vehicle_dist_m > 25.0:
            self._transition(BehaviorState.LANE_FOLLOWING)
        return {"target_speed_kmh": self.cruise_speed * 1.1, "steering_bias": -0.3}

    def _handle_emergency(self, ctx: ScenarioContext) -> Dict:
        if not ctx.emergency_flag and ctx.elapsed_in_state_s > 2.0:
            self._transition(BehaviorState.LANE_FOLLOWING)
        return {"target_speed_kmh": 0.0, "steering_bias": 0.0}

    def _handle_done(self, _ctx: ScenarioContext) -> Dict:
        return {"target_speed_kmh": 0.0, "steering_bias": 0.0}

    # ── FSM Transition ───────────────────────────────────────

    def _transition(self, new_state: BehaviorState) -> None:
        if new_state == self._state:
            return
        old = self._state
        self._state      = new_state
        self._state_start = time.time()
        if new_state in self._callbacks:
            self._callbacks[new_state]()
        print(f"[BehaviorPlanner] {old.name} → {new_state.name}")


# ────────────────────────────────────────────────────────────
# __init__ exports
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    planner = BehaviorPlanner(cruise_speed_kmh=30)
    planner.start()

    test_contexts = [
        ScenarioContext(),
        ScenarioContext(intersection_dist_m=20, at_intersection=False),
        ScenarioContext(at_intersection=True, red_light=True),
        ScenarioContext(at_intersection=True, red_light=False),
        ScenarioContext(leading_vehicle_dist_m=8, current_speed_kmh=28, left_lane_clear=True),
        ScenarioContext(goal_reached=True),
    ]
    print("=" * 50)
    for ctx in test_contexts:
        time.sleep(0.05)
        directive = planner.tick(ctx)
        print(f"  [{directive['state'].name:30s}]  "
              f"speed={directive['target_speed_kmh']:5.1f} kmh  "
              f"bias={directive['steering_bias']:+.2f}")
    print("=" * 50)

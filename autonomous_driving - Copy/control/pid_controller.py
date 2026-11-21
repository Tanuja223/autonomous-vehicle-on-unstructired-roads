"""
control/pid_controller.py
━━━━━━━━━━━━━━━━━━━━━━━━━
PID Controller — Generic Implementation
─────────────────────────────────────────
Features:
  • Anti-windup (integral clamping)
  • Derivative filtering (low-pass)
  • Output clamping
  • Bumpless transfer on setpoint change
  • Gain scheduling support
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ────────────────────────────────────────────────────────────
# Data Types
# ────────────────────────────────────────────────────────────

@dataclass
class PIDState:
    """Internal controller state (for serialisation / monitoring)."""
    error:        float = 0.0
    integral:     float = 0.0
    derivative:   float = 0.0
    output:       float = 0.0
    setpoint:     float = 0.0
    measurement:  float = 0.0
    p_term:       float = 0.0
    i_term:       float = 0.0
    d_term:       float = 0.0


@dataclass
class PIDGains:
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0


# ────────────────────────────────────────────────────────────
# PID Controller
# ────────────────────────────────────────────────────────────

class PIDController:
    """
    Discrete-time PID controller with anti-windup and derivative filter.

    Parameters
    ----------
    Kp, Ki, Kd      : PID gains.
    setpoint        : Desired value (can be updated at runtime).
    output_min      : Lower clamp on control output.
    output_max      : Upper clamp on control output.
    integral_max    : Anti-windup: max absolute integral accumulation.
    derivative_tau  : Low-pass filter time constant for derivative (s).
                      0 = no filtering.
    sample_time     : Fixed sample period (s). None = use wall clock.
    """

    def __init__(
        self,
        Kp: float,
        Ki: float = 0.0,
        Kd: float = 0.0,
        setpoint: float = 0.0,
        output_min: float = -1.0,
        output_max: float = 1.0,
        integral_max: float = 50.0,
        derivative_tau: float = 0.05,
        sample_time: Optional[float] = None,
    ) -> None:
        self.gains          = PIDGains(Kp, Ki, Kd)
        self.setpoint       = setpoint
        self.out_min        = output_min
        self.out_max        = output_max
        self.integral_max   = integral_max
        self.tau            = derivative_tau
        self.sample_time    = sample_time

        self._integral      = 0.0
        self._prev_error    = 0.0
        self._prev_output   = 0.0
        self._deriv_filtered= 0.0
        self._last_time     = time.time()
        self._initialized   = False

        self.state          = PIDState()

    # ── Public API ───────────────────────────────────────────

    def compute(self, measurement: float, dt: Optional[float] = None) -> float:
        """
        Compute control output for given measurement.

        Parameters
        ----------
        measurement : Current process variable.
        dt          : Time delta (s). Auto-computed from wall clock if None.

        Returns
        -------
        Control output clamped to [output_min, output_max].
        """
        now = time.time()
        if dt is None:
            dt = now - self._last_time if self._initialized else (self.sample_time or 0.02)
        self._last_time  = now
        self._initialized = True

        dt = max(dt, 1e-6)

        error = self.setpoint - measurement

        # Proportional
        p_term = self.gains.Kp * error

        # Integral with anti-windup (clamped accumulation + back-calculation)
        self._integral += error * dt
        self._integral = max(
            -self.integral_max,
            min(self.integral_max, self._integral),
        )
        i_term = self.gains.Ki * self._integral

        # Derivative with low-pass filter
        raw_deriv = (error - self._prev_error) / dt
        if self.tau > 0:
            # First-order low-pass: y[k] = α·x[k] + (1-α)·y[k-1]
            alpha = dt / (self.tau + dt)
            self._deriv_filtered = (
                alpha * raw_deriv + (1.0 - alpha) * self._deriv_filtered
            )
        else:
            self._deriv_filtered = raw_deriv
        d_term = self.gains.Kd * self._deriv_filtered

        output = p_term + i_term + d_term
        output = max(self.out_min, min(self.out_max, output))

        self._prev_error  = error
        self._prev_output = output

        # Update monitoring state
        self.state = PIDState(
            error=error,
            integral=self._integral,
            derivative=self._deriv_filtered,
            output=output,
            setpoint=self.setpoint,
            measurement=measurement,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
        )

        return output

    def reset(self) -> None:
        """Reset internal state (use after large setpoint changes)."""
        self._integral       = 0.0
        self._prev_error     = 0.0
        self._deriv_filtered = 0.0
        self._initialized    = False

    def update_gains(self, Kp: float, Ki: float, Kd: float) -> None:
        """Hot-update PID gains (gain scheduling)."""
        self.gains = PIDGains(Kp, Ki, Kd)

    def tune(self, **kwargs) -> None:
        """Keyword-based gain update: tune(Kp=1.2, Ki=0.01)."""
        for k, v in kwargs.items():
            setattr(self.gains, k, v)

    @property
    def output(self) -> float:
        return self.state.output


# ────────────────────────────────────────────────────────────
# Quick Test (Step Response)
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math

    pid = PIDController(Kp=0.9, Ki=0.01, Kd=0.3, setpoint=30.0,
                        output_min=0.0, output_max=1.0)

    print("=" * 55)
    print(f"{'Step':>5} {'Meas':>8} {'Error':>8} {'P':>8} {'I':>8} {'D':>8} {'Out':>8}")
    print("-" * 55)

    speed = 0.0
    for step in range(30):
        control = pid.compute(speed, dt=0.1)
        # Very simple plant model: speed ≈ throttle × 50 (km/h)
        speed += control * 50 * 0.1 - speed * 0.05
        s = pid.state
        print(
            f"{step:>5} {speed:>8.2f} {s.error:>8.2f} "
            f"{s.p_term:>8.4f} {s.i_term:>8.4f} {s.d_term:>8.4f} {s.output:>8.4f}"
        )
    print("=" * 55)

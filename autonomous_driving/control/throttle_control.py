"""
control/throttle_control.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Re-exports ThrottleController from steering_control.py
(kept separate for clean module import paths)
"""
from control.steering_control import ThrottleController, VehicleController, ControlOutput

__all__ = ["ThrottleController", "VehicleController", "ControlOutput"]

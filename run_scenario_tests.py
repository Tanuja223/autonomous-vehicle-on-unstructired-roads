import sys
import os
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

# Mock pytest if not installed
mock_pytest = MagicMock()
def mock_decorator(*args, **kwargs):
    def wrapper(fn):
        return fn
    return wrapper
mock_pytest.mark.parametrize = mock_decorator
sys.modules["pytest"] = mock_pytest

# Add current directory to path
sys.path.insert(0, os.getcwd())

import autonomous_driving.tests.test_scenario_route_requirements as sr

test_functions = [
    sr.test_scenario_requests_two_kilometers_and_crossing_actors,
    sr.test_bypass_is_requested_for_front_obstacle_at_ten_meters,
    sr.test_smooth_bypass_prefers_left_when_candidate_safety_is_equal,
    sr.test_driving_segment_allows_small_lane_edge_sampling_variance,
    sr.test_stall_recovery_only_runs_when_route_is_safe_to_replan,
    sr.test_obstacle_risk_can_reduce_speed_below_ten_kmh,
    sr.test_route_extension_repeats_a_validated_connected_cycle,
]

failed = 0
passed = 0

for filename in sr.SCENARIO_SCRIPTS:
    print(f"\nTesting {filename}...")
    for test_fn in test_functions:
        test_name = test_fn.__name__
        try:
            test_fn(filename)
            print(f"  [PASS] {test_name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_name}: {e}")
            failed += 1

print("\n" + "="*60)
print(f"TOTAL: {passed} passed, {failed} failed")
print("="*60)
if failed > 0:
    sys.exit(1)

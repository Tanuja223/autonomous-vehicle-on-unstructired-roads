"""Tests for the optional PathWeaver request and MATLAB bridge contracts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from integrations.pathweaver.bridge import (
    PathWeaverBridge,
    PathWeaverBridgeError,
)
from integrations.pathweaver.carla_adapter import (
    build_request,
    pathweaver_to_carla_world,
)


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeRotation:
    def __init__(self, yaw=0.0):
        self.yaw = yaw


class FakeTransform:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.location = FakeLocation(x, y)
        self.rotation = FakeRotation(yaw)

    def transform(self, location):
        return FakeLocation(
            self.location.x + location.x,
            self.location.y + location.y,
            self.location.z + location.z,
        )


class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeActor:
    def __init__(
        self,
        actor_id,
        type_id,
        *,
        x=0.0,
        y=0.0,
        z=0.0,
        yaw=0.0,
        vx=0.0,
        vy=0.0,
        extent_x=0.5,
        extent_y=0.5,
    ):
        self.id = actor_id
        self.type_id = type_id
        self.is_alive = True
        self._transform = FakeTransform(x, y, yaw)
        self._transform.location.z = z
        self._velocity = FakeVector(vx, vy)
        self.bounding_box = type(
            "BoundingBox",
            (),
            {
                "extent": FakeVector(extent_x, extent_y),
                "location": FakeLocation(),
            },
        )()

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity

    def get_acceleration(self):
        return FakeVector()

    def get_control(self):
        return type("Control", (), {"steer": 0.0})()

    def get_location(self):
        return self._transform.location


class FakeMatlabEngine:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def addpath(self, path, nargout):
        self.calls.append(("addpath", path, nargout))

    def setupPath(self, nargout):
        self.calls.append(("setupPath", nargout))

    def pathweaverCarlaReset(self, nargout):
        self.calls.append(("reset", nargout))

    def pathweaverCarlaStep(self, request_json, nargout):
        self.calls.append(("step", json.loads(request_json), nargout))
        return json.dumps(self.response)

    def quit(self):
        self.calls.append(("quit",))


class FailingMatlabEngine(FakeMatlabEngine):
    def pathweaverCarlaStep(self, request_json, nargout):
        raise RuntimeError("simulated MATLAB failure")


def make_response(**overrides):
    response = {
        "timestampS": 1.0,
        "selectedTrajectory": {
            "positionsWorldM": [[5.0, 0.0], [7.0, 0.0], [10.0, 1.0]],
            "timestampsS": [1.0, 1.2, 1.4],
            "speedsMps": [8.0, 6.0, 4.0],
            "accelerationsMps2": [0.0, -5.0, -5.0],
            "isFeasible": True,
        },
        "riskScore": 0.2,
        "minimumTtcS": -1,
        "behaviorState": "YIELD",
        "emergencyFlag": False,
        "lookaheadM": 3.0,
        "planningLatencyS": 0.008,
    }
    response.update(overrides)
    return response


def test_build_request_uses_forward_left_coordinates_and_conservative_radius():
    ego = FakeActor(1, "vehicle.tesla.model3", x=10.0, y=20.0)
    pedestrian = FakeActor(
        2, "walker.pedestrian.0001", x=20.0, y=17.0, vx=1.0, vy=-2.0
    )
    static_prop = FakeActor(3, "static.prop.trafficcone", x=15.0, y=18.0)

    request = build_request(
        ego,
        [ego, pedestrian, static_prop],
        timestamp_s=1.25,
        road_half_width_m=3.5,
        route_speed_mps=5.0,
    )

    agent = request["agents"][0]
    assert agent["positionWorldM"] == pytest.approx([15.0, 3.0])
    assert agent["velocityWorldMps"] == pytest.approx([1.0, 2.0])
    assert agent["headingRad"] == pytest.approx(0.0)
    assert agent["collisionRadiusM"] == pytest.approx(math.hypot(0.5, 0.5))
    assert request["staticObstacle"]["positionWorldM"] == pytest.approx([10.0, 2.0])
    assert request["roadLengthM"] == 80.0
    assert request["ego"]["positionWorldM"] == [5.0, 0.0]


def test_build_request_ignores_actors_on_a_different_vertical_level():
    ego = FakeActor(1, "vehicle.tesla.model3")
    overhead = FakeActor(2, "vehicle.audi.a2", x=10.0, z=5.0)

    request = build_request(
        ego,
        [overhead],
        timestamp_s=0.5,
        road_half_width_m=3.5,
        route_speed_mps=5.0,
    )

    assert request["agents"] == []


def test_coordinate_round_trip_maps_pathweaver_left_to_carla_negative_y():
    assert pathweaver_to_carla_world(
        [7.0, 2.0], ego_x=10.0, ego_y=20.0, ego_yaw_rad=0.0
    ) == pytest.approx((12.0, 18.0))
    assert pathweaver_to_carla_world(
        [7.0, 2.0], ego_x=10.0, ego_y=20.0, ego_yaw_rad=math.pi / 2
    ) == pytest.approx((12.0, 22.0))


def test_bridge_initializes_once_and_validates_plan():
    repository = Path(__file__).parents[2] / "integrations" / "PathWeaver"
    engine = FakeMatlabEngine(make_response())
    bridge = PathWeaverBridge(repository, engine=engine)

    bridge.start()
    bridge.start()
    plan = bridge.plan({"timestampS": 1.0})

    assert len([call for call in engine.calls if call[0] == "setupPath"]) == 1
    assert plan.target_position_local_m == (10.0, 1.0)
    assert plan.target_speed_mps == pytest.approx(6.5)
    assert plan.minimum_ttc_s is None
    assert plan.speed_at(1.1) == pytest.approx(7.0)
    assert plan.acceleration_at(1.3) == pytest.approx(-5.0)
    assert plan.round_trip_latency_s >= 0.0
    bridge.close()
    assert engine.calls[-1] == ("quit",)


@pytest.mark.parametrize(
    "response",
    [
        [],
        make_response(
            selectedTrajectory={
                "positionsWorldM": [[5.0, 0.0]],
                "timestampsS": [1.0],
                "speedsMps": [0.0],
                "accelerationsMps2": [0.0],
                "isFeasible": True,
            }
        ),
        make_response(
            selectedTrajectory={
                "positionsWorldM": [[5.0, 0.0], [6.0, 0.0]],
                "timestampsS": [1.0, 1.2],
                "speedsMps": [0.0],
                "accelerationsMps2": [0.0, 0.0],
                "isFeasible": True,
            }
        ),
        make_response(emergencyFlag="false"),
    ],
)
def test_bridge_rejects_invalid_plans(response):
    repository = Path(__file__).parents[2] / "integrations" / "PathWeaver"
    bridge = PathWeaverBridge(repository, engine=FakeMatlabEngine(response))
    bridge.start()

    with pytest.raises(PathWeaverBridgeError):
        bridge.plan({"timestampS": 1.0})


def test_bridge_reports_missing_engine_checkout():
    bridge = PathWeaverBridge(Path("missing-pathweaver-checkout"), engine=object())
    with pytest.raises(PathWeaverBridgeError, match="checkout not found"):
        bridge.start()


def test_bridge_rejects_a_response_for_a_different_timestamp():
    repository = Path(__file__).parents[2] / "integrations" / "PathWeaver"
    response = make_response(timestampS=2.0)
    bridge = PathWeaverBridge(repository, engine=FakeMatlabEngine(response))
    bridge.start()

    with pytest.raises(PathWeaverBridgeError, match="timestamp"):
        bridge.plan({"timestampS": 1.0})


def test_bridge_surfaces_matlab_execution_errors():
    repository = Path(__file__).parents[2] / "integrations" / "PathWeaver"
    bridge = PathWeaverBridge(repository, engine=FailingMatlabEngine({}))
    bridge.start()

    with pytest.raises(PathWeaverBridgeError, match="simulated MATLAB failure"):
        bridge.plan({"timestampS": 1.0})


def test_request_rejects_nonfinite_or_too_narrow_inputs():
    ego = FakeActor(1, "vehicle.tesla.model3")

    with pytest.raises(ValueError, match="timestamp_s"):
        build_request(
            ego,
            [],
            timestamp_s=float("nan"),
            road_half_width_m=3.5,
            route_speed_mps=5.0,
        )
    with pytest.raises(ValueError, match="road_half_width_m"):
        build_request(
            ego,
            [],
            timestamp_s=1.0,
            road_half_width_m=1.5,
            route_speed_mps=5.0,
        )

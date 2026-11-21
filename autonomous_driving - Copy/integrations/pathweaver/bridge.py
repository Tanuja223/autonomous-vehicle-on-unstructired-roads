"""Validated synchronous bridge to a persistent MATLAB Engine session."""

from __future__ import annotations

import importlib
import json
import math
import time
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class PathWeaverBridgeError(RuntimeError):
    """Raised when MATLAB is unavailable or returns an invalid plan."""


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PathWeaverBridgeError(f"PathWeaver response field {name} is not numeric.") from exc
    if not math.isfinite(result):
        raise PathWeaverBridgeError(f"PathWeaver response field {name} is not finite.")
    return result


def _interpolate(times: Sequence[float], values: Sequence[float], timestamp: float) -> float:
    if timestamp <= times[0]:
        return values[0]
    if timestamp >= times[-1]:
        return values[-1]
    upper = bisect_left(times, timestamp)
    lower = upper - 1
    ratio = (timestamp - times[lower]) / (times[upper] - times[lower])
    return values[lower] + ratio * (values[upper] - values[lower])


@dataclass(frozen=True)
class PathWeaverPlan:
    timestamp_s: float
    positions_local_m: tuple[tuple[float, float], ...]
    timestamps_s: tuple[float, ...]
    speeds_mps: tuple[float, ...]
    accelerations_mps2: tuple[float, ...]
    target_position_local_m: tuple[float, float]
    target_speed_mps: float
    risk_score: float
    minimum_ttc_s: float | None
    behavior_state: str
    emergency: bool
    feasible: bool
    matlab_planning_latency_s: float
    round_trip_latency_s: float

    def speed_at(self, timestamp_s: float) -> float:
        return _interpolate(self.timestamps_s, self.speeds_mps, timestamp_s)

    def acceleration_at(self, timestamp_s: float) -> float:
        return _interpolate(self.timestamps_s, self.accelerations_mps2, timestamp_s)


class PathWeaverBridge:
    """Starts MATLAB once, exchanges JSON requests, and validates every result."""

    def __init__(
        self,
        repository_path: str | Path,
        *,
        engine: Any | None = None,
    ) -> None:
        self.repository_path = Path(repository_path).resolve()
        self.adapter_path = Path(__file__).resolve().parent / "matlab"
        self._engine = engine
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        if not (self.repository_path / "setupPath.m").is_file():
            raise PathWeaverBridgeError(
                f"PathWeaver checkout not found at {self.repository_path}. "
                "Clone integrations/PathWeaver from the documented repository."
            )
        if not (self.adapter_path / "pathweaverCarlaStep.m").is_file():
            raise PathWeaverBridgeError(
                f"CARLA adapter is missing from {self.adapter_path}."
            )

        if self._engine is None:
            try:
                matlab_engine = importlib.import_module("matlab.engine")
            except ImportError as exc:
                raise PathWeaverBridgeError(
                    "MATLAB Engine for Python is not installed in this interpreter. "
                    "Install it from the MATLAB release's extern/engines/python folder."
                ) from exc
            self._engine = matlab_engine.start_matlab()

        try:
            self._engine.addpath(str(self.repository_path), nargout=0)
            self._engine.setupPath(nargout=0)
            self._engine.addpath(str(self.adapter_path), nargout=0)
            self._engine.pathweaverCarlaReset(nargout=0)
        except Exception as exc:
            raise PathWeaverBridgeError(
                f"Could not initialize PathWeaver in MATLAB: {exc}"
            ) from exc
        self._started = True

    def plan(self, request: Mapping[str, Any]) -> PathWeaverPlan:
        if not self._started or self._engine is None:
            raise PathWeaverBridgeError("Call PathWeaverBridge.start() before plan().")
        try:
            expected_timestamp = _finite_number(
                request["timestampS"], "request.timestampS"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PathWeaverBridgeError(
                "PathWeaver request must contain a finite timestampS."
            ) from exc
        payload = json.dumps(request, separators=(",", ":"), allow_nan=False)
        started_at = time.perf_counter()
        try:
            raw_response = self._engine.pathweaverCarlaStep(payload, nargout=1)
        except Exception as exc:
            raise PathWeaverBridgeError(
                f"MATLAB failed while planning the CARLA request: {exc}"
            ) from exc
        round_trip = time.perf_counter() - started_at
        if isinstance(raw_response, bytes):
            raw_response = raw_response.decode("utf-8")
        if not isinstance(raw_response, str):
            raise PathWeaverBridgeError(
                f"MATLAB returned {type(raw_response).__name__}, expected a JSON string."
            )
        try:
            response = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise PathWeaverBridgeError(
                "MATLAB returned invalid JSON from pathweaverCarlaStep."
            ) from exc
        return self._validate_response(
            response,
            round_trip,
            expected_timestamp=expected_timestamp,
        )

    @staticmethod
    def _validate_response(
        response: Any,
        round_trip: float,
        *,
        expected_timestamp: float,
    ) -> PathWeaverPlan:
        if not isinstance(response, dict):
            raise PathWeaverBridgeError("PathWeaver response must be a JSON object.")
        try:
            timestamp = _finite_number(response["timestampS"], "timestampS")
            trajectory = response["selectedTrajectory"]
            positions = tuple(
                (
                    _finite_number(point[0], "positionsWorldM.x"),
                    _finite_number(point[1], "positionsWorldM.y"),
                )
                for point in trajectory["positionsWorldM"]
            )
            times = tuple(
                _finite_number(value, "timestampsS")
                for value in trajectory["timestampsS"]
            )
            speeds = tuple(
                _finite_number(value, "speedsMps")
                for value in trajectory["speedsMps"]
            )
            accelerations = tuple(
                _finite_number(value, "accelerationsMps2")
                for value in trajectory["accelerationsMps2"]
            )
            feasible = trajectory["isFeasible"]
            emergency = response["emergencyFlag"]
            behavior = response["behaviorState"]
            risk = _finite_number(response["riskScore"], "riskScore")
            matlab_latency = _finite_number(
                response["planningLatencyS"], "planningLatencyS"
            )
            ttc_value = response.get("minimumTtcS")
            ttc = None if ttc_value is None or float(ttc_value) < 0 else _finite_number(
                ttc_value, "minimumTtcS"
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            if isinstance(exc, PathWeaverBridgeError):
                raise
            raise PathWeaverBridgeError(
                f"PathWeaver response is missing or has malformed fields: {exc}"
            ) from exc

        if (
            not isinstance(feasible, bool)
            or not isinstance(emergency, bool)
            or not isinstance(behavior, str)
        ):
            raise PathWeaverBridgeError(
                "PathWeaver feasibility, emergency, or behavior fields have invalid types."
            )
        if abs(timestamp - expected_timestamp) > 1e-8:
            raise PathWeaverBridgeError(
                "PathWeaver response timestamp does not match its request."
            )
        if len(positions) < 2 or not (
            len(positions) == len(times) == len(speeds) == len(accelerations)
        ):
            raise PathWeaverBridgeError(
                "PathWeaver trajectory arrays must have matching lengths of at least two."
            )
        if any(right <= left for left, right in zip(times, times[1:])):
            raise PathWeaverBridgeError("PathWeaver trajectory timestamps are not increasing.")
        if abs(times[0] - timestamp) > 1e-8:
            raise PathWeaverBridgeError(
                "PathWeaver trajectory does not begin at the request timestamp."
            )
        if any(speed < 0.0 for speed in speeds):
            raise PathWeaverBridgeError("PathWeaver returned a negative target speed.")
        if risk < 0.0 or matlab_latency < 0.0:
            raise PathWeaverBridgeError(
                "PathWeaver returned a negative risk score or planning latency."
            )

        lookahead = _finite_number(response.get("lookaheadM", 2.5), "lookaheadM")
        target_index = len(positions) - 1
        for index, (x, y) in enumerate(positions):
            if times[index] < timestamp:
                continue
            if math.hypot(x - 5.0, y) >= lookahead and x > 5.0:
                target_index = index
                break
        return PathWeaverPlan(
            timestamp_s=timestamp,
            positions_local_m=positions,
            timestamps_s=times,
            speeds_mps=speeds,
            accelerations_mps2=accelerations,
            target_position_local_m=positions[target_index],
            target_speed_mps=_interpolate(times, speeds, timestamp + 0.15),
            risk_score=risk,
            minimum_ttc_s=ttc,
            behavior_state=behavior,
            emergency=emergency,
            feasible=feasible,
            matlab_planning_latency_s=matlab_latency,
            round_trip_latency_s=round_trip,
        )

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception as exc:
                raise PathWeaverBridgeError(
                    f"Could not close the MATLAB Engine session: {exc}"
                ) from exc
            finally:
                self._engine = None
                self._started = False

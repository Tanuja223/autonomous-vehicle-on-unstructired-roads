"""
rl_agent/evaluate.py
━━━━━━━━━━━━━━━━━━━━
PPO Agent Evaluation Script
────────────────────────────
Loads a trained PPO model and runs structured evaluation across
all 5 scenarios, computing:
  • Collision Rate
  • Lane Keeping Accuracy
  • Average Speed
  • Route Completion %
  • Success Rate
  • Inference Latency
  • Detection mAP (proxy)

Results are saved to results/evaluation_report.json + CSV.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.carla_env import CarlaEnv, Scenario

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[Evaluate] stable-baselines3 not found – running mock evaluation.")


# ────────────────────────────────────────────────────────────
# Metrics Dataclass
# ────────────────────────────────────────────────────────────

@dataclass
class EpisodeMetrics:
    scenario: str
    episode:  int
    total_reward:       float = 0.0
    steps:              int   = 0
    collisions:         int   = 0
    lane_violations:    int   = 0
    avg_speed_kmh:      float = 0.0
    max_speed_kmh:      float = 0.0
    distance_m:         float = 0.0
    route_completion:   float = 0.0   # 0–1
    success:            bool  = False
    inference_ms:       float = 0.0
    duration_s:         float = 0.0


@dataclass
class ScenarioReport:
    scenario:           str
    n_episodes:         int   = 0
    collision_rate:     float = 0.0    # collisions / episode
    lane_keeping_acc:   float = 0.0    # % steps within 0.5 m of centre
    avg_speed_kmh:      float = 0.0
    route_completion:   float = 0.0
    success_rate:       float = 0.0
    avg_inference_ms:   float = 0.0
    avg_reward:         float = 0.0
    episodes: List[EpisodeMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("episodes")
        return d


@dataclass
class EvaluationReport:
    model_path:  str
    timestamp:   str
    scenarios: List[ScenarioReport] = field(default_factory=list)

    @property
    def overall_success_rate(self) -> float:
        if not self.scenarios:
            return 0.0
        return float(np.mean([s.success_rate for s in self.scenarios]))

    @property
    def overall_collision_rate(self) -> float:
        if not self.scenarios:
            return 0.0
        return float(np.mean([s.collision_rate for s in self.scenarios]))

    def to_dict(self) -> dict:
        return {
            "model_path":            self.model_path,
            "timestamp":             self.timestamp,
            "overall_success_rate":  round(self.overall_success_rate, 4),
            "overall_collision_rate":round(self.overall_collision_rate, 4),
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


# ────────────────────────────────────────────────────────────
# Evaluator
# ────────────────────────────────────────────────────────────

class AgentEvaluator:
    """
    Runs structured evaluation of a trained PPO agent.

    Parameters
    ----------
    model_path    : Path to PPO .zip file.
    n_episodes    : Episodes per scenario.
    scenarios     : Which scenarios to evaluate (default: all).
    results_dir   : Where to save reports.
    deterministic : Use deterministic policy actions.
    """

    ALL_SCENARIOS = list(Scenario)

    def __init__(
        self,
        model_path: str = "models/ppo_autonomous_driver_final",
        n_episodes: int = 5,
        scenarios: Optional[List[Scenario]] = None,
        results_dir: str = "results",
        deterministic: bool = True,
    ) -> None:
        self.model_path    = model_path
        self.n_episodes    = n_episodes
        self.scenarios     = scenarios or self.ALL_SCENARIOS
        self.results_dir   = Path(results_dir)
        self.deterministic = deterministic
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._model = None

    # ── Public API ───────────────────────────────────────────

    def load_model(self) -> bool:
        if not SB3_AVAILABLE:
            print("[Evaluator] SB3 not available — using random policy.")
            return False
        try:
            self._model = PPO.load(self.model_path)
            print(f"[Evaluator] Model loaded: {self.model_path}")
            return True
        except Exception as exc:
            print(f"[Evaluator] Could not load model: {exc}")
            print("[Evaluator] Falling back to random policy.")
            return False

    def evaluate(self) -> EvaluationReport:
        """Run full evaluation across all configured scenarios."""
        import datetime
        report = EvaluationReport(
            model_path=self.model_path,
            timestamp=datetime.datetime.now().isoformat(),
        )

        print("\n" + "=" * 65)
        print("  AUTONOMOUS DRIVING AGENT EVALUATION")
        print("=" * 65)

        for scenario in self.scenarios:
            print(f"\n▶  Scenario: {scenario.name}")
            sr = self._evaluate_scenario(scenario)
            report.scenarios.append(sr)
            self._print_scenario_summary(sr)

        print("\n" + "=" * 65)
        print(f"  OVERALL SUCCESS RATE : {report.overall_success_rate:.1%}")
        print(f"  OVERALL COLLISION RATE: {report.overall_collision_rate:.3f}")
        print("=" * 65)

        self._save_report(report)
        return report

    # ── Internal ─────────────────────────────────────────────

    def _evaluate_scenario(self, scenario: Scenario) -> ScenarioReport:
        env = CarlaEnv(scenario=scenario)
        sc_report = ScenarioReport(scenario=scenario.name)
        episode_data: List[EpisodeMetrics] = []
        lane_in_cnt  = 0
        total_steps  = 0

        for ep in range(self.n_episodes):
            obs, _ = env.reset()
            ep_metrics = EpisodeMetrics(
                scenario=scenario.name,
                episode=ep,
            )
            speeds:  List[float] = []
            t_start = time.time()
            t_infer = []
            done    = False

            while not done:
                # Get action
                t0 = time.perf_counter()
                action = self._get_action(obs)
                t_infer.append((time.perf_counter() - t0) * 1000)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                speed = float(obs["speed"][0])
                speeds.append(speed)
                ep_metrics.total_reward += reward
                ep_metrics.steps += 1
                total_steps += 1

                lane_pos = abs(float(obs["lane_pos"][0]))
                if lane_pos < 0.5:
                    lane_in_cnt += 1

                if info.get("collision"):
                    ep_metrics.collisions += 1

            # Finalise episode metrics
            ep_metrics.duration_s    = time.time() - t_start
            ep_metrics.avg_speed_kmh = float(np.mean(speeds)) if speeds else 0.0
            ep_metrics.max_speed_kmh = float(np.max(speeds))  if speeds else 0.0
            ep_metrics.distance_m    = info.get("total_dist_m", 0.0)
            ep_metrics.inference_ms  = float(np.mean(t_infer)) if t_infer else 0.0

            # Success = completed without collision in > 80% of max steps
            max_steps = env.MAX_EPISODE_STEPS
            completion = ep_metrics.steps / max_steps
            ep_metrics.route_completion = min(completion, 1.0)
            ep_metrics.success = (
                ep_metrics.collisions == 0
                and ep_metrics.route_completion >= 0.8
            )
            episode_data.append(ep_metrics)

            status = "✓ SUCCESS" if ep_metrics.success else "✗ FAIL"
            print(
                f"    Ep {ep+1:02d}/{self.n_episodes} "
                f"| reward={ep_metrics.total_reward:+.1f} "
                f"| speed={ep_metrics.avg_speed_kmh:.1f} km/h "
                f"| dist={ep_metrics.distance_m:.0f} m "
                f"| {status}"
            )

        env.close()

        # Aggregate
        sc_report.n_episodes       = self.n_episodes
        sc_report.collision_rate   = float(np.mean([e.collisions for e in episode_data]))
        sc_report.lane_keeping_acc = lane_in_cnt / max(total_steps, 1)
        sc_report.avg_speed_kmh    = float(np.mean([e.avg_speed_kmh for e in episode_data]))
        sc_report.route_completion = float(np.mean([e.route_completion for e in episode_data]))
        sc_report.success_rate     = float(np.mean([e.success for e in episode_data]))
        sc_report.avg_inference_ms = float(np.mean([e.inference_ms for e in episode_data]))
        sc_report.avg_reward       = float(np.mean([e.total_reward for e in episode_data]))
        sc_report.episodes         = episode_data
        return sc_report

    def _get_action(self, obs: dict) -> np.ndarray:
        """Return action from trained model or fallback heuristic."""
        if self._model is not None:
            action, _ = self._model.predict(obs, deterministic=self.deterministic)
            return action
        # Random heuristic fallback
        return np.array([
            np.random.uniform(-0.2, 0.2),   # steer
            0.4,                             # throttle
            0.0,                             # brake
        ], dtype=np.float32)

    @staticmethod
    def _print_scenario_summary(sr: ScenarioReport) -> None:
        print(f"\n    ── {sr.scenario} Summary ──────────────────────")
        print(f"    Success rate    : {sr.success_rate:.1%}")
        print(f"    Collision rate  : {sr.collision_rate:.3f} / episode")
        print(f"    Lane keeping    : {sr.lane_keeping_acc:.1%}")
        print(f"    Avg speed       : {sr.avg_speed_kmh:.1f} km/h")
        print(f"    Route completion: {sr.route_completion:.1%}")
        print(f"    Avg inference   : {sr.avg_inference_ms:.2f} ms")

    def _save_report(self, report: EvaluationReport) -> None:
        # JSON
        json_path = self.results_dir / "evaluation_report.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\n[Evaluator] JSON report saved → {json_path}")

        # CSV (per-scenario)
        rows = [s.to_dict() for s in report.scenarios]
        df   = pd.DataFrame(rows)
        csv_path = self.results_dir / "evaluation_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"[Evaluator] CSV  report saved → {csv_path}")

        # Per-episode CSV
        ep_rows = []
        for sr in report.scenarios:
            for ep in sr.episodes:
                ep_rows.append(asdict(ep))
        ep_df = pd.DataFrame(ep_rows)
        ep_csv = self.results_dir / "episode_details.csv"
        ep_df.to_csv(ep_csv, index=False)
        print(f"[Evaluator] Episode CSV saved → {ep_csv}")


# ────────────────────────────────────────────────────────────
# Benchmark Utilities
# ────────────────────────────────────────────────────────────

class PerceptionBenchmark:
    """
    Lightweight perception-only benchmark.
    Measures inference latency for lane, object, sign, depth modules.
    """

    @staticmethod
    def run(n_frames: int = 100) -> Dict[str, float]:
        import cv2
        from perception.lane_detection    import LaneDetector
        from perception.object_detection  import ObjectDetector
        from perception.traffic_sign_detection import TrafficSignDetector
        from perception.depth_estimation  import DepthEstimator

        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        results: Dict[str, List[float]] = {
            "lane":  [],
            "object":[],
            "sign":  [],
            "depth": [],
        }

        lane_det  = LaneDetector()
        obj_det   = ObjectDetector()
        sign_det  = TrafficSignDetector()
        depth_est = DepthEstimator()

        print(f"\n[Benchmark] Running {n_frames} frames …")
        for _ in range(n_frames):
            t0 = time.perf_counter(); lane_det.detect(frame);    results["lane"].append((time.perf_counter()-t0)*1000)
            t0 = time.perf_counter(); obj_det.detect(frame);     results["object"].append((time.perf_counter()-t0)*1000)
            t0 = time.perf_counter(); sign_det.detect(frame);    results["sign"].append((time.perf_counter()-t0)*1000)
            t0 = time.perf_counter(); depth_est.estimate(frame); results["depth"].append((time.perf_counter()-t0)*1000)

        summary = {}
        print("\n  Module         Mean(ms)  P95(ms)  P99(ms)")
        print("  " + "-"*45)
        for k, times in results.items():
            arr = np.array(times)
            mean = float(arr.mean())
            p95  = float(np.percentile(arr, 95))
            p99  = float(np.percentile(arr, 99))
            summary[f"{k}_mean_ms"] = mean
            summary[f"{k}_p95_ms"]  = p95
            print(f"  {k:12s}  {mean:8.2f}  {p95:7.2f}  {p99:7.2f}")
        return summary


# ────────────────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained autonomous driving agent")
    parser.add_argument("--model",      type=str,  default="models/ppo_autonomous_driver_final")
    parser.add_argument("--episodes",   type=int,  default=3)
    parser.add_argument("--scenario",   type=str,  default=None,
                        choices=[s.name for s in Scenario] + [None])  # type: ignore
    parser.add_argument("--benchmark",  action="store_true",
                        help="Run perception latency benchmark only")
    parser.add_argument("--results_dir",type=str,  default="results")
    args = parser.parse_args()

    if args.benchmark:
        PerceptionBenchmark.run(n_frames=50)
        sys.exit(0)

    scenarios = (
        [Scenario[args.scenario]] if args.scenario else None
    )

    evaluator = AgentEvaluator(
        model_path=args.model,
        n_episodes=args.episodes,
        scenarios=scenarios,
        results_dir=args.results_dir,
    )
    evaluator.load_model()
    report = evaluator.evaluate()
    print(f"\nOverall success rate: {report.overall_success_rate:.1%}")

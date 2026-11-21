"""
rl_agent/train.py
━━━━━━━━━━━━━━━━━
PPO Reinforcement Learning Agent — Training Script
────────────────────────────────────────────────────
Uses Stable-Baselines3 (PPO) with a custom CNN feature extractor
tuned for the ego-vehicle camera observation.

Training loop
  1. Initialise CarlaEnv (or mock env for CI/testing)
  2. Wrap with VecEnv for vectorised training
  3. Build PPO agent with CnnPolicy
  4. Train for N timesteps with periodic saves & eval
  5. Log to TensorBoard and optionally W&B
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ── Graceful import guards ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[Train] stable-baselines3 not found – demo mode only.")

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.carla_env import CarlaEnv, Scenario


# ────────────────────────────────────────────────────────────────────────────
# Custom CNN Feature Extractor
# ────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE and SB3_AVAILABLE:
    class AutonomousDrivingCNN(BaseFeaturesExtractor):
        """
        Lightweight CNN that processes the (H,W,3) camera image
        plus scalar telemetry (speed, steer, lane_pos) into a
        flat feature vector for the PPO policy head.

        Architecture
        ─────────────
        3-block ConvNet → GlobalAvgPool → Concat(telem) → FC(512)
        """

        def __init__(self, observation_space: spaces.Dict, features_dim: int = 512) -> None:
            super().__init__(observation_space, features_dim)

            n_input_channels = 3   # RGB
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            cnn_out_dim = 128 * 4 * 4   # 2048

            # Scalar telemetry (speed + steer + lane_pos = 3 values)
            self.telem_fc = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
            )

            self.fc_out = nn.Sequential(
                nn.Linear(cnn_out_dim + 32, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations: Dict[str, Any]) -> Any:
            img    = observations["camera"].float() / 255.0
            # SB3 passes images as (B, H, W, C) → need (B, C, H, W)
            if img.dim() == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)
            cnn_feats = self.cnn(img)

            speed    = observations["speed"]
            steer    = observations["steer"]
            lane_pos = observations["lane_pos"]
            telem    = torch.cat([speed, steer, lane_pos], dim=-1).float()
            telem_feats = self.telem_fc(telem)

            combined = torch.cat([cnn_feats, telem_feats], dim=-1)
            return self.fc_out(combined)


# ────────────────────────────────────────────────────────────────────────────
# Custom Callbacks
# ────────────────────────────────────────────────────────────────────────────

if SB3_AVAILABLE:
    class TrainingMetricsCallback(BaseCallback):
        """Logs custom metrics (collision rate, lane keeping) to TensorBoard."""

        def __init__(self, verbose: int = 0) -> None:
            super().__init__(verbose)
            self._episode_rewards: list = []
            self._collisions: int = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "collision" in info and info["collision"]:
                    self._collisions += 1
                if "episode" in info:
                    self._episode_rewards.append(info["episode"]["r"])
                    self.logger.record(
                        "custom/episode_reward",
                        info["episode"]["r"]
                    )
                    self.logger.record(
                        "custom/episode_length",
                        info["episode"]["l"]
                    )
            if self.n_calls % 1000 == 0 and self._episode_rewards:
                avg_rew = np.mean(self._episode_rewards[-50:])
                self.logger.record("custom/avg_reward_50ep", avg_rew)
                self.logger.record("custom/total_collisions", self._collisions)
                if self.verbose:
                    print(f"[Callback] Step {self.num_timesteps:,} | "
                          f"Avg reward (50ep): {avg_rew:.2f} | "
                          f"Collisions: {self._collisions}")
            return True


# ────────────────────────────────────────────────────────────────────────────
# Training Configuration
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    # Environment
    "carla_host":        "localhost",
    "carla_port":        2000,
    "scenario":          Scenario.STRAIGHT_ROAD,
    "n_envs":            1,        # Parallel CARLA instances (RAM intensive!)

    # PPO Hyperparameters
    "total_timesteps":   500_000,
    "n_steps":           2048,
    "batch_size":        64,
    "n_epochs":          10,
    "gamma":             0.99,
    "gae_lambda":        0.95,
    "clip_range":        0.2,
    "ent_coef":          0.01,
    "learning_rate":     3e-4,
    "max_grad_norm":     0.5,
    "vf_coef":           0.5,

    # Paths
    "log_dir":           "results/rl_logs",
    "save_dir":          "models",
    "model_name":        "ppo_autonomous_driver",
    "save_freq":         10_000,
    "eval_freq":         20_000,
    "eval_episodes":     5,
}


# ────────────────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────────────────

class RLTrainer:
    """
    Orchestrates PPO training on the CARLA environment.

    Parameters
    ----------
    config : dict overriding DEFAULT_CONFIG entries.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        Path(self.cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.cfg["save_dir"]).mkdir(parents=True, exist_ok=True)

        self._model: Optional[Any] = None

    # ── Public API ───────────────────────────────────────────

    def train(self) -> None:
        if not SB3_AVAILABLE:
            print("[Trainer] stable-baselines3 not installed. Cannot train.")
            self._demo_loop()
            return

        print("=" * 60)
        print("  Autonomous Driving PPO Trainer")
        print(f"  Timesteps : {self.cfg['total_timesteps']:,}")
        print(f"  Scenario  : {self.cfg['scenario'].name}")
        print("=" * 60)

        # Build environment(s)
        train_env = self._build_env(self.cfg["scenario"])
        eval_env  = self._build_env(self.cfg["scenario"])

        # Policy kwargs with custom CNN
        policy_kwargs: Dict[str, Any] = {}
        if TORCH_AVAILABLE:
            policy_kwargs = {
                "features_extractor_class":  AutonomousDrivingCNN,
                "features_extractor_kwargs": {"features_dim": 512},
                "net_arch": [256, 256],
            }

        # Build PPO
        self._model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log=self.cfg["log_dir"],
            n_steps=self.cfg["n_steps"],
            batch_size=self.cfg["batch_size"],
            n_epochs=self.cfg["n_epochs"],
            gamma=self.cfg["gamma"],
            gae_lambda=self.cfg["gae_lambda"],
            clip_range=self.cfg["clip_range"],
            ent_coef=self.cfg["ent_coef"],
            learning_rate=self.cfg["learning_rate"],
            max_grad_norm=self.cfg["max_grad_norm"],
            vf_coef=self.cfg["vf_coef"],
            policy_kwargs=policy_kwargs or None,
        )

        # Callbacks
        callbacks = [
            TrainingMetricsCallback(verbose=1),
            CheckpointCallback(
                save_freq=self.cfg["save_freq"],
                save_path=self.cfg["save_dir"],
                name_prefix=self.cfg["model_name"],
            ),
            EvalCallback(
                eval_env,
                n_eval_episodes=self.cfg["eval_episodes"],
                eval_freq=self.cfg["eval_freq"],
                best_model_save_path=self.cfg["save_dir"],
                log_path=self.cfg["log_dir"],
                deterministic=True,
            ),
        ]

        t0 = time.time()
        self._model.learn(
            total_timesteps=self.cfg["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
        )
        elapsed = time.time() - t0

        # Save final model
        save_path = os.path.join(
            self.cfg["save_dir"], self.cfg["model_name"] + "_final"
        )
        self._model.save(save_path)
        print(f"\n[Trainer] Training complete in {elapsed:.0f}s")
        print(f"[Trainer] Model saved → {save_path}.zip")

        train_env.close()
        eval_env.close()

    def resume(self, checkpoint_path: str) -> None:
        """Resume training from a saved checkpoint."""
        if not SB3_AVAILABLE:
            return
        train_env = self._build_env(self.cfg["scenario"])
        self._model = PPO.load(checkpoint_path, env=train_env)
        print(f"[Trainer] Resumed from {checkpoint_path}")
        self.train()

    # ── Internal ─────────────────────────────────────────────

    def _build_env(self, scenario: Scenario):
        def _make():
            env = CarlaEnv(
                host=self.cfg["carla_host"],
                port=self.cfg["carla_port"],
                scenario=scenario,
            )
            return Monitor(env)

        if self.cfg["n_envs"] > 1:
            return DummyVecEnv([_make for _ in range(self.cfg["n_envs"])])
        return DummyVecEnv([_make])

    def _demo_loop(self) -> None:
        """Run a random-action demo loop when SB3 is unavailable."""
        print("[Trainer] Running demo loop (random actions) …")
        env = CarlaEnv(scenario=self.cfg["scenario"])
        obs, _ = env.reset()

        ep_reward = 0.0
        for step in range(200):
            action = np.array([
                np.random.uniform(-0.3, 0.3),   # steer
                0.4,                             # throttle
                0.0,                             # brake
            ])
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if step % 20 == 0:
                print(f"  Step {step:3d} | speed={obs['speed'][0]:.1f} "
                      f"| reward={reward:.3f} | total={ep_reward:.1f}")
            if term or trunc:
                break

        print(f"Demo complete. Episode reward: {ep_reward:.2f}")
        env.close()


# ────────────────────────────────────────────────────────────────────────────
# Reward Shaping Helper
# ────────────────────────────────────────────────────────────────────────────

class RewardShaper:
    """
    Stateful reward-shaping wrapper providing potential-based shaping.
    Wrap around the environment to add rich reward signals.
    """

    WEIGHTS = {
        "lane_keeping":          1.0,
        "destination_progress":  2.0,
        "collision":            -5.0,
        "lane_departure":       -3.0,
        "accident":            -10.0,
        "speed_compliance":      0.5,
        "smooth_driving":        0.2,
    }

    def __init__(self) -> None:
        self._prev_dist_to_goal = 999.0
        self._prev_steer        = 0.0

    def shape(
        self,
        raw_reward: float,
        speed_kmh: float,
        target_speed_kmh: float,
        lane_offset_m: float,
        steer: float,
        collision: bool,
        lane_invaded: bool,
        dist_to_goal: float,
    ) -> float:
        r = raw_reward

        # Lane keeping
        if lane_offset_m < 0.5:
            r += self.WEIGHTS["lane_keeping"]
        else:
            r += self.WEIGHTS["lane_departure"] * min(1.0, lane_offset_m / 2.0)

        # Progress reward
        delta = self._prev_dist_to_goal - dist_to_goal
        if delta > 0:
            r += self.WEIGHTS["destination_progress"] * delta * 0.1
        self._prev_dist_to_goal = dist_to_goal

        # Speed compliance
        speed_err = abs(speed_kmh - target_speed_kmh)
        r += self.WEIGHTS["speed_compliance"] * max(0, 1.0 - speed_err / 10.0)

        # Smooth steering
        steer_jerk = abs(steer - self._prev_steer)
        r += self.WEIGHTS["smooth_driving"] * max(0, 1.0 - steer_jerk / 0.5)
        self._prev_steer = steer

        # Penalties
        if collision:
            r += self.WEIGHTS["collision"]
        if lane_invaded:
            r += self.WEIGHTS["lane_departure"]

        return r


# ────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO autonomous driving agent")
    parser.add_argument("--timesteps",  type=int,   default=500_000)
    parser.add_argument("--scenario",   type=str,   default="STRAIGHT_ROAD",
                        choices=[s.name for s in Scenario])
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint .zip to resume from")
    parser.add_argument("--demo",       action="store_true",
                        help="Run random-action demo without training")
    args = parser.parse_args()

    cfg = {
        "total_timesteps": args.timesteps,
        "scenario":        Scenario[args.scenario],
    }

    trainer = RLTrainer(config=cfg)

    if args.demo:
        trainer._demo_loop()
    elif args.resume:
        trainer.resume(args.resume)
    else:
        trainer.train()

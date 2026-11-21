<div align="center">

# 🚗 End-to-End Autonomous Driving System

**An industry-grade autonomous vehicle simulation built with Python, PyTorch, YOLOv11, CARLA and Streamlit.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-00B5E2?style=flat)](https://carla.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple?style=flat)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-25%20passing-brightgreen?style=flat&logo=pytest)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)

*Internship capstone portfolio project — AI / Autonomous Systems*

</div>

---

## 📖 What This Project Does

This system simulates a self-driving car that can:

- 🛣️ **Detect lanes** — Hough Transform with temporal smoothing
- 🚗 **Detect objects** — YOLOv11 for vehicles, pedestrians, bicycles
- 🚦 **Read traffic signs** — Two-stage YOLO + CNN classifier (stop, lights, speed limits)
- 📏 **Estimate depth** — Intel MiDaS monocular depth estimation
- 🧠 **Make decisions** — Rule-based FSM with 7 priority levels
- 🗺️ **Plan paths** — A* global + Pure Pursuit local planner
- 🎮 **Control the vehicle** — PID + Stanley controller with Auto Emergency Braking
- 🤖 **Learn to drive** — PPO reinforcement learning agent (Stable-Baselines3)
- 📺 **Monitor everything** — Live Streamlit dashboard with gauges, charts, alerts

All inside the **CARLA simulator** with 5 built-in scenarios.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    CARLA Simulator                        │
│  RGB Camera  │  Depth Camera  │  LiDAR  │  GNSS/IMU      │
└──────┬───────────────┬───────────────────────────────────┘
       │               │
       ▼               ▼
┌──────────────────────────────────────────────────────────┐
│                  PERCEPTION LAYER                         │
│  Lane Detection │ Object Detection │ Signs │ Depth        │
│  (Hough+CV2)    │ (YOLOv11)        │ (CNN) │ (MiDaS)     │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                   PLANNING LAYER                          │
│  Decision Engine (rules)  +  Behavior Planner (FSM)      │
│  Path Planner: A* Global  +  Pure Pursuit Local          │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                   CONTROL LAYER                           │
│  Steering (PID + Stanley)  │  Throttle PID  │  AEB        │
└──────────────────────────────────────────────────────────┘
              ↕ (alternatively)
┌──────────────────────────────────────────────────────────┐
│               RL AGENT — PPO (Stable-Baselines3)         │
│  Custom CNN extracts camera + speed + steer + lane_pos   │
└──────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│            STREAMLIT DASHBOARD  :8501                    │
│  Live feed │ Gauges │ Warnings │ Charts │ Eval report    │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
autonomous_driving/
│
├── perception/
│   ├── lane_detection.py          # Hough Transform + temporal smoothing
│   ├── object_detection.py        # YOLOv11 + ByteTrack + threat levels
│   ├── traffic_sign_detection.py  # YOLO + CNN + HSV colour parsing
│   └── depth_estimation.py        # MiDaS monocular depth
│
├── planning/
│   ├── decision_engine.py         # 7-priority rule engine
│   ├── behavior_planner.py        # 9-state hierarchical FSM
│   └── path_planner.py            # A* global + Pure Pursuit local
│
├── control/
│   ├── pid_controller.py          # Generic PID with anti-windup
│   ├── steering_control.py        # PID + Stanley fusion
│   └── throttle_control.py        # Speed PID + Auto Emergency Braking
│
├── simulation/
│   ├── carla_env.py               # Gymnasium-compatible CARLA env
│   └── sensors.py                 # Thread-safe sensor manager
│
├── rl_agent/
│   ├── train.py                   # PPO training pipeline
│   └── evaluate.py                # Multi-scenario evaluation
│
├── dashboard/
│   └── app.py                     # Streamlit live dashboard
│
├── tests/                         # 25 unit tests (all passing ✅)
│   ├── test_perception.py
│   ├── test_planning.py
│   └── test_control.py
│
├── configs/config.yaml            # Central configuration
├── docs/dataset_guide.md
├── main.py                        # CLI orchestrator
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Requirements
- Python **3.11+**
- Git
- NVIDIA GPU (recommended — CPU fallback available)
- CARLA 0.9.15 ([optional — mock mode works without it](#running-without-carla))

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/autonomous-driving-system.git
cd autonomous-driving-system/autonomous_driving
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If `carla` fails to install, skip it. The system runs in **mock mode** without CARLA.

### 4. Verify installation

```bash
python -m pytest tests/ -v
# Expected: 25 passed
```

---

## 🚀 Running the System

### Running Without CARLA (Demo / Mock Mode)

Every module works standalone — no simulator needed:

```bash
# Test individual perception modules
python perception/lane_detection.py
python perception/object_detection.py
python perception/depth_estimation.py

# Test planning
python planning/decision_engine.py
python planning/path_planner.py

# Test control
python control/pid_controller.py
python control/steering_control.py

# Full system in mock mode
python main.py --mode rule
```

### Launch the Dashboard (No CARLA needed)

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser → click **▶ START**.

### Running With CARLA

**Step 1 — Start CARLA server** (separate terminal):

```bash
# Windows
CarlaUE4.exe -carla-server -fps=20

# Linux
./CarlaUE4.sh -carla-server -fps=20
```

**Step 2 — Install CARLA Python client:**

```bash
pip install carla==0.9.15
```

**Step 3 — Run scenarios:**

```bash
# Rule-based agent
python main.py --mode rule --scenario STRAIGHT_ROAD
python main.py --mode rule --scenario INTERSECTION
python main.py --mode rule --scenario PEDESTRIAN_CROSS
python main.py --mode rule --scenario DENSE_URBAN
python main.py --mode rule --scenario NIGHT_DRIVING
```

---

## 🧠 Training the RL Agent

```bash
# Quick training test (50k steps, ~10 min on GPU)
python main.py --mode train --timesteps 50000

# Full training (500k steps)
python main.py --mode train --timesteps 500000 --scenario DENSE_URBAN

# Monitor with TensorBoard
tensorboard --logdir results/rl_logs

# Resume from checkpoint
python rl_agent/train.py --resume models/ppo_autonomous_driver_50000_steps.zip
```

### Reward Function

| Event | Reward |
|-------|--------|
| Lane keeping (offset < 0.5 m) | **+1.0** |
| Progress toward destination | **+2.0** |
| Speed compliance | **+0.5** |
| Smooth steering | **+0.2** |
| Collision | **−5.0** |
| Lane departure | **−3.0** |
| Major accident | **−10.0** |

---

## 📊 Evaluation

```bash
# Evaluate all 5 scenarios (5 episodes each)
python main.py --mode eval --episodes 5

# Single scenario
python rl_agent/evaluate.py --scenario INTERSECTION --episodes 10

# Perception latency benchmark
python main.py --mode benchmark
```

Results saved to `results/evaluation_report.json` and `results/evaluation_metrics.csv`.

---

## 📈 Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Collision Rate | Collisions per episode | < 0.1 |
| Lane Keeping Accuracy | % steps within 0.5 m of centre | > 85% |
| Success Rate | Episodes without collision | > 70% |
| Route Completion | % of route completed | > 80% |
| Inference Latency | Per-module forward pass | < 50 ms |

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

```
tests/test_control.py::TestPIDController::test_output_in_range          PASSED
tests/test_control.py::TestPIDController::test_zero_error_near_zero_output PASSED
tests/test_control.py::TestPIDController::test_reset_clears_integral    PASSED
tests/test_control.py::TestVehicleController::test_aeb_fires_on_close   PASSED
tests/test_control.py::TestVehicleController::test_throttle_accelerating PASSED
tests/test_control.py::TestVehicleController::test_steer_range           PASSED
tests/test_perception.py ...                                             PASSED (10)
tests/test_planning.py ...                                               PASSED (9)

25 passed in 0.97s ✅
```

---

## 🖥️ Dashboard Features

| Panel | What it shows |
|-------|--------------|
| **Live Camera** | Front camera with lane + object overlays |
| **Speed Gauge** | Real-time speedometer (0–120 km/h) |
| **Steering Gauge** | Steering angle (±1.0) |
| **Warnings** | AEB, collision, lane departure, overspeed |
| **Drive State** | Current FSM state |
| **Speed / Steer / Reward Charts** | Rolling time-series graphs |
| **Detection Table** | Live object list with distance & threat |
| **Evaluation Tab** | Scenario success rate bar chart |

---

## 🔧 Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11+ |
| Deep Learning | PyTorch 2.1+ |
| Object Detection | Ultralytics YOLOv11 |
| Depth Estimation | Intel MiDaS |
| Computer Vision | OpenCV |
| Reinforcement Learning | Stable-Baselines3 (PPO) |
| Simulation | CARLA 0.9.15 |
| Dashboard | Streamlit + Plotly |
| Path Planning | A* + Pure Pursuit |
| Testing | pytest (25 tests) |

---

## 🔮 Roadmap

- [ ] Semantic Segmentation (DeepLabV3+)
- [ ] Transformer perception (BEVFormer)
- [ ] Multi-camera 360° fusion
- [ ] LiDAR object detection (PointPillar)
- [ ] HD Map (OpenDRIVE) integration
- [ ] V2X communication simulation
- [ ] ROS2 + Autoware bridge

---

## 📚 References

1. Dosovitskiy et al. (2017) — [CARLA: An Open Urban Driving Simulator](https://arxiv.org/abs/1711.03938)
2. Redmon et al. — [YOLO: Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
3. Ranftl et al. (2021) — [Vision Transformers for Dense Prediction (MiDaS)](https://arxiv.org/abs/2103.13413)
4. Schulman et al. (2017) — [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built as an AI/Autonomous Systems Internship Capstone Portfolio Project.<br>
⭐ Star this repo if it was helpful!
</div>

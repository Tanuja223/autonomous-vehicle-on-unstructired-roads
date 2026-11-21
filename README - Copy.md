# 🚗 End-to-End Autonomous Driving System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/CARLA-0.9.15+-00B5E2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/YOLOv11-Ultralytics-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit" />
</p>

> **Industry-grade autonomous vehicle simulation system** integrating
> deep-learning perception, rule-based + RL decision making, classical
> path planning, PID vehicle control, and a real-time Streamlit dashboard.
> Built as an AI/Autonomous Systems internship capstone portfolio project.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Module Breakdown](#-module-breakdown)
4. [Installation](#-installation)
5. [Dataset Setup](#-dataset-setup)
6. [Running the System](#-running-the-system)
7. [Training the RL Agent](#-training-the-rl-agent)
8. [Evaluation](#-evaluation)
9. [Dashboard Usage](#-dashboard-usage)
10. [Evaluation Metrics](#-evaluation-metrics)
11. [Results](#-results)
12. [Project Structure](#-project-structure)
13. [Future Improvements](#-future-improvements)
14. [References](#-references)

---

## 🎯 Project Overview

This project implements a **complete, modular autonomous driving pipeline**
capable of:

| Capability                | Implementation                         |
|---------------------------|----------------------------------------|
| Lane detection            | Hough Transform + temporal smoothing   |
| Object detection          | YOLOv11 (Ultralytics) + ByteTracker    |
| Traffic sign recognition  | YOLO + fine-grained CNN classifier     |
| Depth estimation          | Intel MiDaS (DPT_Large)               |
| Driving decisions         | Rule-based FSM + Behaviour Planner     |
| Global path planning      | A* search on occupancy grid            |
| Local path planning       | Pure Pursuit + potential fields        |
| Vehicle control           | PID + Stanley controller + AEB         |
| End-to-end RL driving     | PPO (Stable-Baselines3)                |
| Simulation                | CARLA 0.9.15+ (5 scenarios)            |
| Monitoring                | Streamlit real-time dashboard          |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CARLA Simulator                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  RGB Camera  │  │  Depth Cam   │  │  LiDAR / GNSS / IMU  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼─────────────────┼────────────────────  ┼─────────────┘
          │                 │                       │
          ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PERCEPTION LAYER                            │
│  ┌───────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Lane Detection│ │Object Detect │ │Sign Detec│ │  Depth   │ │
│  │ (Hough+CV2)   │ │ (YOLOv11)    │ │(YOLO+CNN)│ │ (MiDaS)  │ │
│  └───────┬───────┘ └──────┬───────┘ └────┬─────┘ └────┬─────┘ │
└──────────┼────────────────┼──────────────┼─────────────┼───────┘
           │                │              │             │
           └────────────────┼──────────────┼─────────────┘
                            ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PLANNING LAYER                              │
│  ┌──────────────────┐      ┌──────────────────────────────────┐ │
│  │  Decision Engine │      │      Behavior Planner (FSM)      │ │
│  │  (Rule-based)    │      │  IDLE→LANE_FOLLOWING→INTERSECT.. │ │
│  └────────┬─────────┘      └──────────────┬───────────────────┘ │
│           │        ┌───────────────────────┘                     │
│           ▼        ▼                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Path Planner: A* Global + Pure Pursuit Local            │   │
│  └────────────────────────────┬─────────────────────────────┘   │
└───────────────────────────────┼─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONTROL LAYER                               │
│  ┌───────────────────┐  ┌──────────────────┐  ┌─────────────┐  │
│  │ Steering (PID+    │  │  Throttle (PID)  │  │  AEB        │  │
│  │ Stanley)          │  │  Speed control   │  │  Emergency  │  │
│  └───────────────────┘  └──────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   RL Agent (PPO)      │ ← Optional end-to-end
                    │   Stable-Baselines3   │
                    └───────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Streamlit Dashboard   │
                    │ (Live monitoring)     │
                    └───────────────────────┘
```

---

## 🔩 Module Breakdown

### Perception (`perception/`)

| File | Description |
|------|-------------|
| `lane_detection.py` | Canny + Hough lane detection, temporal smoothing, lane-centre computation |
| `object_detection.py` | YOLOv11 wrapper, distance estimation, threat classification, ByteTrack |
| `traffic_sign_detection.py` | Two-stage YOLO+CNN, HSV traffic-light colour parsing |
| `depth_estimation.py` | MiDaS monocular depth, per-object distance enrichment |

### Planning (`planning/`)

| File | Description |
|------|-------------|
| `decision_engine.py` | Prioritised rule-stack: emergency brake → red light → pedestrian → follow → cruise |
| `behavior_planner.py` | Hierarchical FSM with 9 states + lane-change logic |
| `path_planner.py` | A* global planner on occupancy grid + Pure Pursuit local planner |

### Control (`control/`)

| File | Description |
|------|-------------|
| `pid_controller.py` | Generic PID with anti-windup, derivative filter, gain scheduling |
| `steering_control.py` | PID + Stanley controller fusion for lateral control |
| `throttle_control.py` | PID longitudinal control + Automatic Emergency Braking (AEB) |

### Simulation (`simulation/`)

| File | Description |
|------|-------------|
| `carla_env.py` | Gymnasium-compatible CARLA environment, 5 scenarios, reward shaping |
| `sensors.py` | Thread-safe sensor manager: RGB, depth, semantic, LiDAR, GNSS/IMU |

### RL Agent (`rl_agent/`)

| File | Description |
|------|-------------|
| `train.py` | PPO trainer with custom CNN feature extractor, EvalCallback, W&B logging |
| `evaluate.py` | Structured evaluation across all scenarios, JSON/CSV report generation |

---

## ⚙️ Installation

### Prerequisites

- Ubuntu 20.04+ or Windows 10/11
- Python 3.11+
- NVIDIA GPU with CUDA 11.8+ (recommended; CPU fallback available)
- CARLA 0.9.15 ([download](https://github.com/carla-simulator/carla/releases))

### Step 1 — Clone & Create Environment

```bash
git clone https://github.com/yourusername/autonomous-driving-system.git
cd autonomous-driving-system

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Step 2 — Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r autonomous_driving/requirements.txt
```

### Step 3 — Install CARLA Python Client

```bash
# For CARLA 0.9.15
pip install carla==0.9.15

# OR install from the CARLA PythonAPI egg (if server is local)
cd /opt/carla/PythonAPI/carla/dist
pip install carla-0.9.15-cp311-cp311-linux_x86_64.whl
```

### Step 4 — Download YOLO Weights

```bash
# Run from the models directory so downloaded weights stay organized
cd models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
cd ..

# Or for YOLOv11:
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

### Step 5 — Verify Installation

```bash
python -m pytest autonomous_driving/tests -q
```

---

## 📦 Dataset Setup

### CARLA — No dataset download needed

The CARLA simulator generates synthetic sensor data in real-time.
Start the server and the system auto-collects training data.

### Optional: Pre-trained Sign Classifier

For enhanced traffic-sign classification, train the CNN on:

| Dataset | Classes | Link |
|---------|---------|------|
| German Traffic Sign (GTSRB) | 43 | [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) |
| LISA Traffic Sign Dataset | 47 | [UCSD](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html) |

```bash
# Prepare dataset
mkdir -p datasets/traffic_signs
# Download and extract to datasets/traffic_signs/
# Expected structure:
#   datasets/traffic_signs/
#     train/  (class_0/, class_1/, ...)
#     val/
#     test/
```

Train the sign classifier:

```bash
python scripts/train_sign_classifier.py \
  --data datasets/traffic_signs \
  --epochs 50 \
  --output models/sign_classifier.pt
```

---

## 🚀 Running the System

### 1. Start CARLA Server (separate terminal)

```bash
# Linux
cd /opt/carla
./CarlaUE4.sh -carla-server -fps=20 -ResX=1280 -ResY=720

# Windows
CarlaUE4.exe -carla-server -fps=20
```

### Unstructured OSM simulation (`main.py`)

`main.py` is a separate CARLA-only simulation demo. It listens for the OSM
web page's route at `http://127.0.0.1:5001/destination`. OSM start and
destination coordinates are used only to calculate the desired compass
bearing; they are never converted to or matched against Town10HD world
coordinates. The controller compares compass-aligned CARLA starts and plans
an approximately 2 km route through connected `LaneType.Driving` waypoints.
It closes and repeats a verified Driving-lane loop when available, or follows
the longest connected Driving-lane continuation when the selected network
has no return path. For a closed network, the controller extends the route
before it reaches each segment endpoint, preventing an artificial route end
from stopping the vehicle. The controller tracks the route occurrence, not just
geographic proximity to the final point, so repeating a loop does not cause
an early arrival. For an obstacle, a bounded search explores reachable nearby
Driving waypoints, ranks paths by their minimum actor clearance, and targets
a nearest future waypoint on the original route after the vehicle and obstacle
can clear one another, then resumes route tracking from that merge point.
Obstacle bypasses may start only with a validated move into the physical
right-side same-direction Driving lane. The broader local graph search also
rejects paths whose first lane change is left. It follows the right-side lane
past the obstacle and merges back only over a validated connection after
clearing it; if no safe right-side path exists, the vehicle slows and brakes
instead of steering left.
Every waypoint and sampled segment must remain on CARLA Driving lanes across
the vehicle-width footprint, not just at the centerline; otherwise the
vehicle safely stops rather than crossing onto a sidewalk. The controller
starts searching for and following a bypass at the 30 m detection range
and retries the bounded search once per second while there is room to
maneuver, reducing speed rather than waiting until it is almost touching the
obstacle. Spawned road-crossing pedestrians begin walking when the ego is
within 35 m; the controller fully brakes until each walker reaches the
opposite shoulder and is removed. Pedestrians are spawned above the road
surface to avoid CARLA rejecting a ground-intersecting spawn point. It first
evaluates gradual right-side same-direction lane changes,
then uses the connected-waypoint search with the same right-only initial
maneuver rule; oncoming lanes are never used. If no
valid bypass is found, it keeps slowing, retries while there is room to
maneuver, and brakes if the bypass remains blocked. The emergency stop margin
uses the current speed's reaction and braking distance plus a 2 m stand-off,
with a 3 m minimum. Pedestrians are not treated as
lane-change targets: their position and short-horizon crossing motion reduce
speed, and every spawned road crossing holds the car at a full stop from
crossing start until the pedestrian clears the road. Obstacle
distance uses oriented bounding boxes for moving
actors and indexed CARLA static environment objects (including poles, street
fixtures, walls, buildings, fences, and vegetation), and checks the upcoming
route curve so hazards around a bend can be considered before they enter the
straight-ahead corridor. Static objects are loaded once per map and spatially
indexed; sidewalk surfaces are not treated as bypass space, and every bypass
waypoint and sampled segment remains restricted to CARLA Driving lanes. The
speed cap is derived from
reaction time, deceleration, and obstacle clearance. If a bypass makes no
progress for 15 seconds, the controller retries route planning; if the
blocking actor has disappeared, it clears stale avoidance and returns to the
main route. Route following first targets the nearest valid route waypoint
and validates the direct segment to it. If route lookahead is lost while the
car remains on a Driving lane, a connected waypoint search tries to rejoin a
future point on the original route; if no merge is immediately reachable, it
follows a short validated forward Driving-lane continuation and searches
again. It brakes if no all-Driving recovery path exists. A lack of
route progress is tolerated for up to 120 seconds, and physical movement
resets the stall timer. Targets behind the vehicle are rejected rather than
steering back toward them; route steering uses a bounded pursuit controller.
When a finite route segment ends, the controller appends a connected,
validated Driving-lane continuation instead of stopping at that artificial
route boundary. It stops if no safe continuation exists or when you stop the
script.
The default map is
`Town10HD_Opt`. Open
`osm_navigation/index.html` in a browser to choose an OSM destination. Since
Town10HD has no real-world alignment to the OSM map, the selected CARLA goal
follows the requested direction, not the real GPS destination. The default
scene spawns eight parked vehicles beside the selected route, spaced apart
and aligned with the road, and requests eight moving traffic vehicles ahead
of the ego across the first kilometre: four scooter/motorcycle actors and
four vans. Parked-vehicle placement is best-effort where CARLA has clear
space beside the road. Moving vehicles start at least
35 m beyond the ego's actual spawn point, and their 24 km/h target is above
the ego's 20 km/h cruise limit to help them remain ahead. If CARLA has no
scooter/motorcycle blueprint, it substitutes four moving cars; missing van
blueprints are also substituted with moving cars
blueprints when available. Every spawned traffic vehicle is assigned to CARLA
manual throttle and steering controls on each simulation step, following its
assigned forward segment of the selected route. This demo traffic controller
does not implement NPC-to-NPC collision avoidance and is not a real-road
driving policy. Moving vehicles participate in the same obstacle detection
and avoidance logic as other vehicles. Rock/debris piles, cones,
and construction barriers remain disabled. A vehicle hazard beside the ego
vehicle is handled as a coast-down speed restriction without brake; braking
and the stop-distance latch are reserved for hazards classified in the
forward vehicle-width corridor. The vehicle fully brakes when a pedestrian is
detected in or predicted to enter its forward corridor within the safety
distance. Collision emergency braking remains active.

No project-spawned pedestrians, roadside props, or stationary route hazards
are enabled in the default scene. CARLA's built-in map scenery remains
present and is not removed.

```bash
python main.py
```

This entry point does not run CUDA, neural-network inference, or RGB/depth/
LiDAR capture. CARLA's renderer still uses the graphics hardware required by
the simulator. This is simulation code only and is not suitable for
controlling a real vehicle.

The root `main.py` is the OSM-to-CARLA simulation entry point; it does not
accept `--mode`, `--scenario`, or `--device` options. The modular CARLA,
perception, planning, control, and RL code lives in `autonomous_driving/`.

### Generated unstructured-road scenario

This separate scenario uses a generated OpenDRIVE loop with a default 400 m
radius, 3.5 m OpenDRIVE shoulder lanes on both sides, and no lane markings.
The shoulder lanes are part of the generated road surface so parked vehicles
do not spawn over the unsupported edge. It requests 60 parked vehicles
split across both shoulders, including 10 near each end of the selected route,
and up to 15 moving Traffic Manager vehicles, plus six pedestrians that
cross the full roadway from left to right, and visual pothole proxies. The
generated map must include pedestrian navigation data for CARLA to keep walker
actors alive; the map generator enables it. If an older map is loaded, rerun
`generate_unstructured_map.py` before starting the controller. The controller
checks for navigation data and warns instead of reporting dead walkers as
spawned. If a parked vehicle blocks a pedestrian's shoulder spawn point, the
driver retries nearby crossing locations and prints the number of walkers it
spawned. Moving traffic consists of up to five scooter/motorcycle actors and
ten vans; car models substitute when those vehicle types are unavailable. The
moving Traffic Manager vehicles follow the ego route and have their target
speed updated to match the ego vehicle; Traffic Manager collision avoidance
and a 10 m global following gap are enabled. Its safety behavior may slow
traffic below the ego speed to preserve clearance. Parked shoulder vehicles
remain stationary on both shoulders. Low risk follows the route at cruise
speed, medium risk reduces speed, and high risk tries a validated right-side
same-direction bypass. It stops rather than using a left or opposing-lane
bypass when a safe right-side path is unavailable. Pedestrians are not
bypassed: their crossing trajectory is sampled for up to 8 seconds, and the
car brakes as a crossing walker approaches, holds while the walker crosses,
then resumes after the walker clears the opposite shoulder and is removed.
Six flat tire props are visual pothole proxies, not physical road depressions.
The controller cannot guarantee collision-free operation; it stops rather
than entering a path that fails its safety checks. Restart CARLA between tests
if the world has accumulated actors from runs that were terminated without
cleanup.
The generated OpenDRIVE road has no terrain beside the road. The
`add_buildings.py` helper places up to 40 building-like structures along the
shoulders by default (or accepts compatible CARLA mesh assets), where they
stand on the empty plane. Potholes are represented by tire props, not physical
road depressions. Collision notifications
apply a short brake hold, then the controller rechecks the route and nearby
actors rather than staying latched in emergency braking indefinitely.

On Windows, start CARLA first, then run these commands from the project root:

The root `unstructured_autonomous.py` file is a legacy compatibility launcher;
use the generated-road controller below for this OSM scenario.

```powershell
.\.venv_carla915\Scripts\python.exe .\generate_unstructured_map.py --preview
.\.venv_carla915\Scripts\python.exe .\add_buildings.py --count 40 --spacing 90
.\.venv_carla915\Scripts\python.exe .\apply_patch3.py
.\.venv_carla915\Scripts\python.exe .\unstructured_autonomous_no_pathweaver_road.py
```

The map-generation command reloads CARLA with pedestrian navigation enabled.
Run it before the controller; it clears actors from the currently loaded world.
`apply_patch3.py` generates the runnable controller from
`unstructured_autonomous_no_pathweaver.py`.

In another terminal, serve the OSM page:

```powershell
& .\.venv_carla915\Scripts\python.exe -m http.server 8080 --bind 127.0.0.1 --directory .\osm_navigation
```

Open `http://127.0.0.1:8080/`, submit an OSM destination, and leave the
driving script running. Regenerating the map overwrites `unstructured_road.xodr`
and replaces the current CARLA world. OSM coordinates select a desired
direction; they are not mapped to CARLA world coordinates.

### Optional MATLAB PathWeaver obstacle planner

The PathWeaver source is checked out under `integrations/PathWeaver` at
upstream revision `f37a3d9696572ffa5832d2a3a8d0b47e8a359460`. The integration
is opt-in; with `PATHWEAVER_ENABLED` unset, the existing CARLA controller and
simulation timing are unchanged. When enabled, the root route-following loop
retains its existing route controller and calls PathWeaver for hazards detected
within 40 m, when the next 30 m of route is approximately straight and the
vehicle is not already in route recovery or bypass mode. Curves and recovery
continue using the existing controller because this PathWeaver adapter uses a
straight local-road model. CARLA is switched to synchronous 0.05 s stepping so
it cannot continue moving while a MATLAB request is being computed.

For a fresh checkout, run these commands from the project root:

```powershell
New-Item -ItemType Directory -Force .\integrations | Out-Null
git clone https://github.com/PathWeaverr/PathWeaver.git .\integrations\PathWeaver
git -C .\integrations\PathWeaver checkout f37a3d9696572ffa5832d2a3a8d0b47e8a359460
```

The bridge uses the MATLAB Engine for Python in the same interpreter that has
CARLA 0.9.15 installed. Install the engine from the matching MATLAB release
using that interpreter's executable (replace the path if needed):

```matlab
cd(fullfile(matlabroot, 'extern', 'engines', 'python'))
system('"C:\autonomous-driving-system-main\.venv_carla0915\Scripts\python.exe" -m pip install .')
```

Verify the engine import, start CARLA and open `osm_navigation/index.html` for
the route, then enable the integration in PowerShell:

```powershell
& .\.venv_carla0915\Scripts\python.exe -c "import matlab.engine; print('MATLAB Engine available')"
$env:PATHWEAVER_ENABLED = "1"
# Optional only when lane geometry cannot provide the width; use a verified value.
# $env:PATHWEAVER_ROAD_HALF_WIDTH_M = "3.5"
& .\.venv_carla0915\Scripts\python.exe .\main.py
```

Unset the variables (or use a fresh terminal) to run the original controller.
When no hazard is in the trigger range, that controller remains in charge. For
a PathWeaver request, the bridge passes CARLA ego state and nearby vehicle /
pedestrian ground-truth actors to MATLAB, maps the returned local trajectory
back to CARLA coordinates, tracks its target point, and uses its target speed
for throttle and braking. If PathWeaver finds no feasible trajectory, the
integration applies full braking and keeps the synchronous simulation running,
replanning as the hazard moves or clears. The vehicle remains stopped while
PathWeaver reports no safe trajectory. MATLAB execution or bridge errors still
apply a brake and stop the run with an explicit error.

The bridge uses CARLA actor state, not YOLO outputs. Vehicle bounding boxes are
conservatively represented by circles; PathWeaver's single static-obstacle
input is populated by the nearest nearby static object. Road width is
approximated from same-direction adjacent CARLA `Driving` lane geometry (or
the explicit half-width setting), while PathWeaver retains its local
road-boundary model.
These approximations are not sensor fusion or a complete collision-safety
proof. The existing YOLO module and model are not connected to the root
controller; the root demo still uses Town10HD and does not use the
`unstructured_indian_road.xodr` asset. There is no native RoadRunner scene or
integration. This remains a simulation prototype and must not control a real
vehicle.

Python contract tests for coordinate conversion and mocked Engine requests:

```powershell
& .\.venv\Scripts\python.exe -m pytest .\autonomous_driving\tests\test_pathweaver_integration.py
```

MATLAB and a running CARLA server are required for an end-to-end integration
run. Neither was available in the implementation environment, so the MATLAB
adapter and live CARLA control path are not claimed as runtime-verified.

## 🧠 Training the RL Agent

```bash
# Quick training
python autonomous_driving/rl_agent/train.py \
  --timesteps 50000 \
  --scenario STRAIGHT_ROAD

# Full training
python autonomous_driving/rl_agent/train.py \
  --timesteps 500000 \
  --scenario INTERSECTION

# Resume from checkpoint
python autonomous_driving/rl_agent/train.py \
  --resume models/ppo_autonomous_driver_100000_steps.zip \
  --timesteps 400000

# Monitor with TensorBoard
tensorboard --logdir results/rl_logs
```

### Reward Function

| Event | Reward |
|-------|--------|
| Lane keeping (offset < 0.5 m) | +1.0 |
| Progress toward destination | +2.0 × delta |
| Speed compliance | +0.5 |
| Smooth steering | +0.2 |
| Collision | **−5.0** + episode end |
| Lane departure | **−3.0** |
| Accident (major) | **−10.0** |

---

## 📊 Evaluation

```bash
# Evaluate single scenario
python autonomous_driving/rl_agent/evaluate.py \
  --model models/ppo_autonomous_driver_final \
  --scenario INTERSECTION \
  --episodes 10

# Run project tests
python -m pytest autonomous_driving/tests -q

# View results
cat results/evaluation_report.json
```

Evaluation output: `results/evaluation_report.json`, `results/evaluation_metrics.csv`,  
`results/episode_details.csv`

---

## 📺 Dashboard Usage

```bash
streamlit run autonomous_driving/dashboard/app.py
# Opens at http://localhost:8501
```

### Dashboard Features

| Panel | Content |
|-------|---------|
| **Live Feed** | Front camera with lane + detection overlays |
| **Speed Gauge** | Real-time speedometer (0–120 km/h) |
| **Steering Gauge** | Steering angle indicator (±1.0) |
| **Warning Panel** | AEB, collision, lane departure, overspeed alerts |
| **Drive State** | Current FSM state (CRUISING / EMERGENCY_BRAKE / ...) |
| **Metrics Tab** | Rolling speed, steer, reward, FPS time-series charts |
| **Detections Tab** | Live detection table + class distribution chart |
| **Evaluation Tab** | Loaded evaluation report with scenario bar charts |
| **System Info Tab** | Architecture overview, quick-start guide |

---

## 📈 Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Collision Rate** | Collisions / episode | < 0.1 |
| **Lane Keeping Accuracy** | Steps within 0.5m / total steps | > 85% |
| **Success Rate** | Episodes without collision & > 80% route | > 70% |
| **Route Completion** | Steps completed / max steps | > 80% |
| **Average Speed** | Mean speed (km/h) during episode | ≈ cruise target |
| **Inference Latency** | Per-module forward pass (ms) | < 50ms/frame |
| **Detection mAP** | YOLO mAP@0.5 on test set | > 0.75 |

---

## 📁 Project Structure

```
.
├── main.py                         # OSM-to-CARLA simulation entry point
├── app.py                          # Compatibility launcher for dashboard
├── autonomous_driving/             # Modular driving system and tests
│   ├── control/
│   ├── dashboard/
│   ├── main.py                     # Existing standalone CARLA entry point
│   ├── perception/
│   ├── planning/
│   ├── rl_agent/
│   ├── simulation/
│   ├── tests/
│   └── requirements.txt
├── osm_navigation/                 # Browser-based OSM destination picker
├── scripts/
│   ├── carla/
│   │   ├── diagnostics/            # Camera and vehicle test scripts
│   │   ├── experiments/            # Experimental controllers
│   │   └── scenarios/              # Unstructured-road/map scenarios
│   └── osm/                        # Legacy OSM helper scripts
├── assets/
│   └── maps/                       # OpenDRIVE map assets
├── models/                         # Model weights (local, Git-ignored)
├── results/                        # Simulation logs and evaluation output
├── README.md
└── [root compatibility launchers]  # Preserve earlier script commands
```

---

## 🔮 Future Improvements

- [ ] **Semantic Segmentation** — Replace Hough lanes with DeepLabV3+/SegFormer drivable-area mask
- [ ] **Transformer Perception** — DETR / BEVFormer for multi-view object detection
- [ ] **Multi-Camera Fusion** — 360° surround-view with BEV projection
- [ ] **LiDAR Integration** — PointPillar / VoxelNet for LiDAR object detection
- [ ] **HD Map Integration** — OpenDRIVE map parsing for precise path planning
- [ ] **V2X Communication** — Vehicle-to-infrastructure signal simulation
- [ ] **Driver Monitoring** — Drowsiness / distraction detection
- [ ] **Imitation Learning** — Pre-training from expert CARLA demonstrations
- [ ] **Uncertainty Estimation** — Bayesian detection confidence for safety
- [ ] **Real-world Deployment** — ROS2 + Autoware.Universe integration

---

## 📚 References

1. Dosovitskiy et al. (2017) — [CARLA: An Open Urban Driving Simulator](https://arxiv.org/abs/1711.03938)
2. Redmon et al. — [YOLO: Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
3. Ranftl et al. (2021) — [Vision Transformers for Dense Prediction (MiDaS)](https://arxiv.org/abs/2103.13413)
4. Schulman et al. (2017) — [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
5. Coulter (1992) — Pure Pursuit Path Tracking
6. Hart, Nilsson, Raphael (1968) — A* Search Algorithm

---

## 👨‍💻 Author

Built as an **AI/Autonomous Systems Internship Capstone Project**.  
Demonstrates mastery of: deep learning · computer vision · robotics · reinforcement learning · simulation · software engineering.

**Tech Stack:** Python · PyTorch · OpenCV · Ultralytics YOLOv11 · MiDaS · Stable-Baselines3 · CARLA · Streamlit · NumPy · Pandas

---

<p align="center">
  ⭐ Star this repo if it helped you! · 🍴 Fork it for your own project
</p>

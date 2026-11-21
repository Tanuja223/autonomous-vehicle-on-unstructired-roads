"""
dashboard/app.py
━━━━━━━━━━━━━━━━
Streamlit Monitoring Dashboard — Fixed Version
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from simulation.carla_env import Scenario
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


# ════════════════════════════════════════════════════════════
# Page Config
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Autonomous Driving — Live Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #1e2130;
    border-radius: 10px;
    padding: 12px 18px;
    border: 1px solid #2d3250;
}
.stMetric label { color: #8892b0 !important; font-size: 0.82rem; }
.section-header {
    font-size: 1rem; font-weight: 700; color: #64ffda;
    border-bottom: 1px solid #2d3250;
    padding-bottom: 4px; margin-bottom: 10px;
}
.state-badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-size: 0.85rem; font-weight: 700;
    background: #0d1117; border: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════

def _hex_to_rgba(hex_colour: str, alpha: float = 0.15) -> str:
    """Convert '#rrggbb' → 'rgba(r,g,b,alpha)' safely for Plotly."""
    h = hex_colour.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(100,100,100,{alpha})"


# ════════════════════════════════════════════════════════════
# Session State
# ════════════════════════════════════════════════════════════

HISTORY_LEN = 300

def _init_state() -> None:
    defaults = {
        "running":          False,
        "frame_count":      0,
        "speed_history":    [],
        "steer_history":    [],
        "reward_history":   [],
        "fps_history":      [],
        "collision_count":  0,
        "lane_violation_count": 0,
        "detections":       [],
        "current_state":    "IDLE",
        "current_speed":    0.0,
        "current_steer":    0.0,
        "total_distance":   0.0,
        "aeb_active":       False,
        "eval_report":      None,
        "scenario":         "STRAIGHT_ROAD",
        "last_frame":       None,
        "t_last":           time.time(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🚗 ADS Control Panel")
    st.divider()

    scenario_name = st.selectbox(
        "Scenario",
        ["STRAIGHT_ROAD", "INTERSECTION", "PEDESTRIAN_CROSS",
         "DENSE_URBAN", "NIGHT_DRIVING"],
    )
    st.session_state["scenario"] = scenario_name
    speed_limit = st.slider("Speed Limit (km/h)", 10, 120, 30, 5)

    st.markdown("**Perception Modules**")
    show_lane  = st.toggle("Lane Detection",         value=True)
    show_obj   = st.toggle("Object Detection",       value=True)
    show_sign  = st.toggle("Traffic Sign Detection", value=True)
    show_depth = st.toggle("Depth Estimation",       value=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ START", use_container_width=True, type="primary")
    with col2:
        stop_btn  = st.button("⏹ STOP",  use_container_width=True)

    if start_btn:
        st.session_state["running"]         = True
        st.session_state["frame_count"]     = 0
        st.session_state["collision_count"] = 0
        st.session_state["total_distance"]  = 0.0
        st.session_state["t_last"]          = time.time()

    if stop_btn:
        st.session_state["running"] = False

    st.divider()
    if st.button("📊 Load Eval Report", use_container_width=True):
        rp = Path("results/evaluation_report.json")
        if rp.exists():
            with open(rp) as f:
                st.session_state["eval_report"] = json.load(f)
            st.success("Report loaded!")
        else:
            st.warning("No report found. Run evaluate.py first.")

    if st.button("🔄 Reset Metrics", use_container_width=True):
        for k in ["speed_history", "steer_history", "reward_history",
                  "fps_history", "collision_count", "lane_violation_count",
                  "frame_count", "total_distance"]:
            st.session_state[k] = [] if "history" in k else 0

    st.divider()
    st.caption(f"Frame : {st.session_state['frame_count']:,}")
    st.caption(f"Scenario : {st.session_state['scenario']}")


# ════════════════════════════════════════════════════════════
# Synthetic Frame Generator
# ════════════════════════════════════════════════════════════

def _gen_synthetic_frame(t: float) -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:180]  = (100, 120, 180)   # sky
    frame[180:]  = (60,  60,  65)    # road
    if CV2_AVAILABLE:
        cx = 320 + int(30 * np.sin(t * 0.5))
        cv2.line(frame, (cx - 120, 360), (cx - 80, 200), (220, 220, 220), 8)
        cv2.line(frame, (cx + 120, 360), (cx + 80, 200), (220, 220, 220), 8)
        for y in range(200, 360, 40):
            cv2.line(frame, (cx, y), (cx, y + 20), (255, 255, 100), 3)
        car_y = 200 + int(20 * np.sin(t * 0.3 + 1))
        cv2.rectangle(frame, (cx - 40, car_y), (cx + 40, car_y + 60), (0, 120, 255), -1)
        cv2.putText(frame, f"t={t:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


# ════════════════════════════════════════════════════════════
# Plotly Chart Helpers
# ════════════════════════════════════════════════════════════

def _speed_gauge(speed: float, max_speed: float = 120.0) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=speed,
        number={"suffix": " km/h", "font": {"size": 26, "color": "#64ffda"}},
        gauge={
            "axis": {"range": [0, max_speed], "tickcolor": "#8892b0"},
            "bar":  {"color": "#64ffda", "thickness": 0.25},
            "bgcolor": "#1e2130",
            "steps": [
                {"range": [0,                  max_speed * 0.4],  "color": "#0d1117"},
                {"range": [max_speed * 0.4,    max_speed * 0.75], "color": "#0d1117"},
                {"range": [max_speed * 0.75,   max_speed],        "color": "#1a0000"},
            ],
            "threshold": {
                "line": {"color": "#ff4b4b", "width": 3},
                "thickness": 0.75,
                "value": max_speed * 0.9,
            },
        },
        title={"text": "Speed", "font": {"color": "#8892b0", "size": 13}},
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#0d1117",
        font_color="#8892b0",
    )
    return fig


def _steer_gauge(steer: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=steer,
        number={"valueformat": ".3f", "font": {"size": 22, "color": "#ffd700"}},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar":  {"color": "#ffd700", "thickness": 0.4},
            "bgcolor": "#1e2130",
            "steps": [
                {"range": [-1,   -0.5], "color": "#1a0a00"},
                {"range": [-0.5,  0.5], "color": "#0d1117"},
                {"range": [0.5,   1.0], "color": "#1a0a00"},
            ],
        },
        title={"text": "Steering", "font": {"color": "#8892b0", "size": 13}},
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#0d1117",
        font_color="#8892b0",
    )
    return fig


def _line_chart(history: List[float], label: str, colour: str) -> go.Figure:
    """
    Fixed: fillcolor now always uses rgba() string — never bare hex+alpha.
    """
    fill_colour = _hex_to_rgba(colour, alpha=0.15)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=list(history[-HISTORY_LEN:]),
        mode="lines",
        line=dict(color=colour, width=2),
        fill="tozeroy",
        fillcolor=fill_colour,          # ← fixed: proper rgba string
        name=label,
    ))
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font_color="#8892b0",
        title=dict(text=label, font=dict(size=12, color="#8892b0")),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2130", zeroline=True,
                   zerolinecolor="#2d3250"),
        showlegend=False,
    )
    return fig


# ════════════════════════════════════════════════════════════
# Main Layout
# ════════════════════════════════════════════════════════════

st.markdown("## 🚗 Autonomous Driving System — Live Monitor")
st.divider()

# KPI row
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("Speed",       f"{st.session_state['current_speed']:.1f} km/h")
kpi2.metric("Steering",    f"{st.session_state['current_steer']:.3f}")
kpi3.metric("Collisions",  st.session_state["collision_count"])
kpi4.metric("Lane Viol.",  st.session_state["lane_violation_count"])
kpi5.metric("Distance",    f"{st.session_state['total_distance']:.0f} m")
kpi6.metric("Frames",      f"{st.session_state['frame_count']:,}")

st.divider()

tab_live, tab_metrics, tab_detections, tab_eval, tab_docs = st.tabs([
    "📷 Live Feed", "📈 Metrics", "🔍 Detections", "📊 Evaluation", "📋 System Info",
])


# ── TAB 1: Live Feed ─────────────────────────────────────────────────────────
with tab_live:
    col_cam, col_gauges = st.columns([3, 1])

    with col_cam:
        st.markdown('<div class="section-header">📷 Camera Feed</div>',
                    unsafe_allow_html=True)
        cam_placeholder = st.empty()

    with col_gauges:
        st.markdown('<div class="section-header">Vehicle State</div>',
                    unsafe_allow_html=True)
        gauge_placeholder = st.empty()
        steer_placeholder = st.empty()

        st.markdown('<div class="section-header">🚨 Warnings</div>',
                    unsafe_allow_html=True)
        warn_placeholder  = st.empty()

        st.markdown('<div class="section-header">🧠 Drive State</div>',
                    unsafe_allow_html=True)
        state_placeholder = st.empty()


# ── TAB 2: Metrics ────────────────────────────────────────────────────────────
with tab_metrics:
    m1, m2, m3 = st.columns(3)
    speed_chart_ph  = m1.empty()
    steer_chart_ph  = m2.empty()
    reward_chart_ph = m3.empty()
    fps_chart_ph    = st.empty()


# ── TAB 3: Detections ─────────────────────────────────────────────────────────
with tab_detections:
    st.markdown('<div class="section-header">🔍 Real-time Object Detections</div>',
                unsafe_allow_html=True)
    det_table_ph = st.empty()
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Detection Class Distribution**")
        det_chart_ph  = st.empty()
    with col_d2:
        st.markdown("**Distance to Nearest Obstacles**")
        dist_chart_ph = st.empty()


# ── TAB 4: Evaluation ─────────────────────────────────────────────────────────
with tab_eval:
    eval_ph = st.empty()


# ── TAB 5: System Info ────────────────────────────────────────────────────────
with tab_docs:
    st.markdown("""
### System Architecture

| Module | Technology | Purpose |
|--------|-----------|---------|
| **Lane Detection** | OpenCV + Hough | Detect lane boundaries & heading |
| **Object Detection** | YOLOv11 (Ultralytics) | Detect vehicles & pedestrians |
| **Sign Detection** | YOLO + CNN classifier | Stop signs, traffic lights |
| **Depth Estimation** | MiDaS DPT | Monocular distance estimation |
| **Decision Engine** | Rule-based FSM | High-level driving commands |
| **Path Planning** | A* + Pure Pursuit | Global & local path following |
| **Vehicle Control** | PID + Stanley | Throttle, brake, steering |
| **RL Agent** | PPO (SB3) | End-to-end learned driving |
| **Simulation** | CARLA 0.9.15+ | Photo-realistic environment |
| **Dashboard** | Streamlit | Live monitoring & evaluation |

### Quick Start
```bash
# 1. Start CARLA server
./CarlaUE4.sh -carla-server

# 2. Rule-based agent
python main.py --mode rule --scenario STRAIGHT_ROAD

# 3. Train RL agent
python main.py --mode train --timesteps 500000

# 4. Evaluate
python main.py --mode eval --episodes 5

# 5. Launch dashboard
streamlit run dashboard/app.py
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Collision Rate** | Collisions per episode |
| **Lane Keeping Acc.** | % timesteps within 0.5 m of lane centre |
| **Success Rate** | Episodes completed without collision |
| **Route Completion** | % of route completed |
| **Inference Latency** | Per-module forward-pass time (ms) |
""")


# ════════════════════════════════════════════════════════════
# Simulation Step
# ════════════════════════════════════════════════════════════

def _append_history(key: str, val: float) -> None:
    st.session_state[key].append(val)
    if len(st.session_state[key]) > HISTORY_LEN:
        st.session_state[key].pop(0)


def _run_simulation_step() -> None:
    t  = time.time()
    fc = st.session_state["frame_count"]

    # Synthetic telemetry
    speed  = 25.0 + 10.0 * float(np.sin(t * 0.3))
    steer  = 0.15 * float(np.sin(t * 0.7))
    reward = 1.0 - abs(steer) * 3.0

    collided  = (fc % 300 == 0) and fc > 0
    lane_viol = (fc % 80  == 0) and fc > 0

    if collided:
        st.session_state["collision_count"] += 1
    if lane_viol:
        st.session_state["lane_violation_count"] += 1

    st.session_state["current_speed"] = speed
    st.session_state["current_steer"] = steer
    st.session_state["total_distance"] += speed / 3.6 * 0.05
    st.session_state["frame_count"]   += 1

    _append_history("speed_history",  speed)
    _append_history("steer_history",  steer)
    _append_history("reward_history", reward)

    dt  = max(t - st.session_state["t_last"], 1e-4)
    st.session_state["t_last"] = t
    _append_history("fps_history", 1.0 / dt)

    # Synthetic detections
    st.session_state["detections"] = [
        {"class": "car",    "confidence": 0.93, "distance_m": 18.5, "threat": "MEDIUM"},
        {"class": "person", "confidence": 0.87, "distance_m": 32.1, "threat": "HIGH"},
        {"class": "truck",  "confidence": 0.78, "distance_m": 45.0, "threat": "LOW"},
    ]

    state = "CRUISING" if abs(steer) < 0.1 else "LANE_CORRECTION"
    if collided:
        state = "EMERGENCY_BRAKE"
    st.session_state["current_state"] = state

    # ── Camera ───────────────────────────────────────────────
    frame = _gen_synthetic_frame(t)
    img   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else frame
    cam_placeholder.image(img, caption="Front Camera (demo)", use_container_width=True)

    # ── Gauges ───────────────────────────────────────────────
    if PLOTLY_AVAILABLE:
        gauge_placeholder.plotly_chart(
            _speed_gauge(speed, max_speed=float(speed_limit)),
            use_container_width=True, key=f"gauge_{fc}",
        )
        steer_placeholder.plotly_chart(
            _steer_gauge(steer),
            use_container_width=True, key=f"steer_{fc}",
        )

    # ── Warnings ─────────────────────────────────────────────
    warns = []
    if collided:
        warns.append("🔴 **COLLISION DETECTED**")
    if abs(steer) > 0.4:
        warns.append("🟠 **HARD STEERING**")
    if lane_viol:
        warns.append("🟡 **LANE DEPARTURE**")
    if speed > speed_limit * 0.9:
        warns.append("🔵 **OVERSPEED**")
    warn_placeholder.markdown(
        "\n\n".join(warns) if warns else "✅ No warnings"
    )

    # ── State badge ───────────────────────────────────────────
    state_colour = {
        "EMERGENCY_BRAKE": "#ff4b4b",
        "LANE_CORRECTION":  "#ffd700",
        "CRUISING":         "#00cc66",
    }.get(state, "#8892b0")
    state_placeholder.markdown(
        f'<span class="state-badge" style="color:{state_colour};'
        f'border-color:{state_colour}">{state}</span>',
        unsafe_allow_html=True,
    )

    # ── Metrics charts ────────────────────────────────────────
    if PLOTLY_AVAILABLE:
        speed_chart_ph.plotly_chart(
            _line_chart(st.session_state["speed_history"],  "Speed (km/h)", "#64ffda"),
            use_container_width=True, key=f"sc_{fc}",
        )
        steer_chart_ph.plotly_chart(
            _line_chart(st.session_state["steer_history"],  "Steering",     "#ffd700"),
            use_container_width=True, key=f"stc_{fc}",
        )
        reward_chart_ph.plotly_chart(
            _line_chart(st.session_state["reward_history"], "Reward",        "#bb86fc"),
            use_container_width=True, key=f"rc_{fc}",
        )
        fps_chart_ph.plotly_chart(
            _line_chart(st.session_state["fps_history"],    "FPS",           "#ff8c00"),
            use_container_width=True, key=f"fps_{fc}",
        )

    # ── Detection table & charts ──────────────────────────────
    dets = st.session_state["detections"]
    if dets:
        df = pd.DataFrame(dets)
        det_table_ph.dataframe(df, use_container_width=True, hide_index=True)

        if PLOTLY_AVAILABLE:
            counts = df["class"].value_counts().reset_index()
            counts.columns = ["class", "count"]
            det_chart_ph.plotly_chart(
                px.bar(counts, x="class", y="count",
                       color_discrete_sequence=["#64ffda"],
                       template="plotly_dark"),
                use_container_width=True, key=f"detc_{fc}",
            )
            dist_chart_ph.plotly_chart(
                px.bar(df, x="class", y="distance_m",
                       color="threat",
                       color_discrete_map={
                           "CRITICAL": "#ff4b4b",
                           "HIGH":     "#ff8c00",
                           "MEDIUM":   "#ffd700",
                           "LOW":      "#00cc66",
                       },
                       template="plotly_dark"),
                use_container_width=True, key=f"distc_{fc}",
            )


# ════════════════════════════════════════════════════════════
# Eval Report Renderer
# ════════════════════════════════════════════════════════════

def _render_eval_report(report: dict) -> None:
    with eval_ph.container():
        st.markdown(f"**Model:** `{report.get('model_path', 'N/A')}`")
        st.markdown(f"**Timestamp:** {report.get('timestamp', '')}")
        c1, c2 = st.columns(2)
        c1.metric("Overall Success Rate",
                  f"{report.get('overall_success_rate', 0):.1%}")
        c2.metric("Overall Collision Rate",
                  f"{report.get('overall_collision_rate', 0):.4f}")
        st.divider()
        scenarios = report.get("scenarios", [])
        if scenarios:
            df = pd.DataFrame(scenarios)
            st.dataframe(df, use_container_width=True, hide_index=True)
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    df, x="scenario", y="success_rate",
                    title="Success Rate by Scenario",
                    color="success_rate",
                    color_continuous_scale="teal",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# Main Loop
# ════════════════════════════════════════════════════════════

if st.session_state["eval_report"]:
    _render_eval_report(st.session_state["eval_report"])

if st.session_state["running"]:
    _run_simulation_step()
    time.sleep(0.05)
    st.rerun()
else:
    cam_placeholder.info("▶ Click **START** in the sidebar to begin the simulation.")

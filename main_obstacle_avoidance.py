"""Compatibility launcher for the older obstacle-avoidance prototype."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "carla" / "experiments" / "main_obstacle_avoidance.py"
    runpy.run_path(str(script), run_name="__main__")

"""Compatibility launcher for the Town10 decal-layer demo."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "carla" / "scenarios" / "town10_no_lane_marks.py"
    runpy.run_path(str(script), run_name="__main__")

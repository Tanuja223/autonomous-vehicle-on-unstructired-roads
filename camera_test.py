"""Compatibility launcher for the CARLA camera diagnostic."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "carla" / "diagnostics" / "camera_test.py"
    runpy.run_path(str(script), run_name="__main__")

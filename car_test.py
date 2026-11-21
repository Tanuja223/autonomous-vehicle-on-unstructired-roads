"""Compatibility launcher for the CARLA vehicle diagnostic."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "carla" / "diagnostics" / "car_test.py"
    runpy.run_path(str(script), run_name="__main__")

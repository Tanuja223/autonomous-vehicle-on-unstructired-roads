"""Compatibility launcher for the legacy unstructured-driving prototype."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "carla" / "scenarios" / "unstructured_autonomous.py"
    runpy.run_path(str(script), run_name="__main__")

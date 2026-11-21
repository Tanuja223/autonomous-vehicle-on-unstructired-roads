"""Compatibility launcher for the custom OpenDRIVE world loader."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "carla" / "scenarios" / "load_unstructured_world.py"
    runpy.run_path(str(script), run_name="__main__")

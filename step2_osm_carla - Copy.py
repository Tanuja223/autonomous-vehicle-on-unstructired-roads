"""Compatibility launcher for the legacy OSM-to-CARLA bridge."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "osm" / "step2_osm_carla.py"
    runpy.run_path(str(script), run_name="__main__")

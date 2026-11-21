"""Compatibility launcher for the legacy OSM destination selector."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "osm" / "step1_destination.py"
    runpy.run_path(str(script), run_name="__main__")

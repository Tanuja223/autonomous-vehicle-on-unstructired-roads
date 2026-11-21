"""Compatibility launcher for the Streamlit dashboard."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    dashboard = (
        Path(__file__).resolve().parent
        / "autonomous_driving"
        / "dashboard"
        / "app.py"
    )
    runpy.run_path(str(dashboard), run_name="__main__")

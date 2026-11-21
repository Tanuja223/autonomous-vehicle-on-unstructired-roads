import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[2]
GENERATORS = (
    "generate_unstructured_map.py",
    "generate_unstructured_rough_road.py",
)


def load_generator(filename):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename", GENERATORS)
def test_generated_roads_include_full_width_shoulders(filename):
    generator = load_generator(filename)
    segments = [
        {
            "p": (0.0, 0.0),
            "h": 0.0,
            "cu": (0.0, 10.0, 0.0, 0.0),
            "cv": (0.0, 0.0, 0.0, 0.0),
            "length": 10.0,
            "end": (10.0, 0.0),
            "end_h": 0.0,
        }
        for _ in range(3)
    ]
    build_options = {}
    if filename == "generate_unstructured_rough_road.py":
        build_options["rough_patches"] = 0

    xodr, _, _ = generator.build_xodr(
        segments,
        roads=3,
        lanes_per_side=2,
        min_width=3.0,
        max_width=4.2,
        seed=42,
        **build_options,
    )
    root = ET.fromstring(xodr)

    for road in root.findall("road"):
        lane_section = road.find("./lanes/laneSection")
        for side in ("left", "right"):
            shoulders = lane_section.findall(f"./{side}/lane[@type='shoulder']")
            assert len(shoulders) == 1
            width = shoulders[0].find("width")
            assert width is not None
            assert float(width.attrib["a"]) == pytest.approx(3.5)


def test_checked_in_road_asset_includes_full_width_shoulders():
    root = ET.parse(ROOT / "unstructured_road.xodr").getroot()

    for road in root.findall("road"):
        lane_section = road.find("./lanes/laneSection")
        for side in ("left", "right"):
            shoulders = lane_section.findall(f"./{side}/lane[@type='shoulder']")
            assert len(shoulders) == 1
            width = shoulders[0].find("width")
            assert width is not None
            assert float(width.attrib["a"]) == pytest.approx(3.5)

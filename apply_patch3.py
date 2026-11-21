"""
Generate the OpenDRIVE-compatible OSM controller from its maintained source.

Fixes for generated maps:
  * map-name check no longer insists on Town10HD (default is now "ANY")
  * generated maps have no spawn points -> start points are taken from road waypoints
  * static-object index is cached (otherwise it was re-read every control step = very slow)
  * people never spawn or cross OFF the road (there is no ground outside the road mesh)

Usage (same folder as your script):
    python apply_patch3.py                          # uses unstructured_autonomous_no_pathweaver.py
    python apply_patch3.py my_file.py
Output: <name>_road.py   (your file is NOT changed)
"""
import py_compile
import sys
from pathlib import Path

DEFAULT_SRC = Path(__file__).resolve().with_name(
    "unstructured_autonomous_no_pathweaver.py"
)
SRC = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else DEFAULT_SRC
)
DST = SRC.with_name(SRC.stem + "_road.py")

EDITS = []  # (old, new, required)

EDITS.append((
    'MAP_NAME = os.environ.get("CARLA_MAP_NAME", "Town10HD_Opt")',
    'MAP_NAME = os.environ.get("CARLA_MAP_NAME", "ANY")',
    True))

EDITS.append((
    "    if MAP_NAME not in world.get_map().name:",
    '    if MAP_NAME.upper() != "ANY" and MAP_NAME not in world.get_map().name:',
    True))

EDITS.append((
    '    print("Town10HD connected.")',
    '    print("Connected to map:", world.get_map().name)',
    False))

EDITS.append((
    """    spawns = map_obj.get_spawn_points()
    if not spawns:
        raise RuntimeError("CARLA map has no vehicle spawn points.")""",
    """    spawns = map_obj.get_spawn_points()
    if not spawns:
        spawns = [
            waypoint.transform
            for waypoint in map_obj.generate_waypoints(30.0)
            if is_driving_waypoint(waypoint)
        ]
    if not spawns:
        raise RuntimeError("CARLA map has no vehicle spawn points.")""",
    True))

EDITS.append((
    "    if static_hazard_map_name == map_name and static_hazard_cells:",
    "    if static_hazard_map_name == map_name:",
    True))

EDITS.append((
    "def _index_static_map_hazards():",
    """def _safe_environment_objects(label):
    try:
        return world.get_environment_objects(label)
    except RuntimeError:
        return []


def _index_static_map_hazards():""",
    True))

EDITS.append((
    "        for environment_object in world.get_environment_objects(label):",
    "        for environment_object in _safe_environment_objects(label):",
    True))

def main():
    if not SRC.exists():
        sys.exit(f"File not found: {SRC}  (put apply_patch3.py next to your script)")
    text = SRC.read_text(encoding="utf-8")
    skipped = []
    for number, (old, new, required) in enumerate(EDITS, start=1):
        count = text.count(old)
        if count != 1:
            if required:
                print(f"PATCH NOT APPLIED: edit {number} - text found {count} times: {old.strip()[:70]!r}")
                print("Your file differs from the one I patched. Send me that part and I will fix it.")
                sys.exit(1)
            skipped.append(number)
            continue
        text = text.replace(old, new)
    DST.write_text(text, encoding="utf-8")
    py_compile.compile(str(DST), doraise=True)
    print(f"OK -> {DST}   (syntax check passed)")
    if skipped:
        print(f"Optional compatibility edits skipped: {skipped}.")
    print(f"Run it:  python {DST.name}")


if __name__ == "__main__":
    main()

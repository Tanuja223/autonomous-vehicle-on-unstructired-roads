#!/usr/bin/env python3
"""
add_buildings.py   (CARLA 0.9.15)

Adds houses / buildings on BOTH sides of your generated unstructured road.
It does NOT change the road. Run it once, after the road is loaded:

    python generate_unstructured_map.py
    python add_buildings.py --count 40 --spacing 90
    python apply_patch3.py
    python unstructured_autonomous_no_pathweaver_road.py

The buildings stay in CARLA until the map is reloaded, so you can start and stop
the driving script as often as you like.

Options
    --count 40            how many houses (default 40)
    --spacing 90          average distance between houses along the road (m)
    --setback 6           free space between the road edge and a house (m)
    --seed 3              different seed = different village
    --no-decor            no plants / lamps / bins in front of the houses
    --carla-dir C:\\CARLA_0.9.15   also try REAL building models from the CARLA package
    --mesh /Game/...      use your own mesh path(s) (can be repeated)
    --remove              delete the buildings made by this script
"""
import argparse
import math
import random
import re
import sys
from pathlib import Path

TAG = "unstructured_building"
DECOR_IDS = ("static.prop.plantpot01", "static.prop.plantpot03", "static.prop.plantpot05",
             "static.prop.gardenlamp", "static.prop.trashcan01", "static.prop.trashcan03",
             "static.prop.bench01", "static.prop.mailbox")
SKIP_WORDS = ("window", "door", "roof", "balcon", "chimney", "fence", "pipe", "lod1", "lod2",
              "collision", "decal", "sign", "wire", "lamp", "light", "antenna", "stair")
SKIP_PREFIX = ("t_", "m_", "mi_", "mm_", "bp_", "tx_", "mat_", "pm_")


# ------------------------------------------------------------------ small maths
def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def rotate(x, y, yaw_rad):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return x * c - y * s, x * s + y * c


# ------------------------------------------------------- real building models
def find_building_meshes(carla_dir):
    """Best effort: list building meshes stored in the CARLA .pak files."""
    paks = sorted(Path(carla_dir).glob("CarlaUE4/Content/Paks/*.pak"))
    if not paks:
        print(f"  no .pak files found under {carla_dir}\\CarlaUE4\\Content\\Paks")
        return []
    found = []
    for pak in paks:
        size = pak.stat().st_size
        with open(pak, "rb") as handle:
            handle.seek(max(0, size - 96 * 1024 * 1024))   # the file index is at the end
            data = handle.read()
        current_dir = ""
        for match in re.finditer(rb"[\x20-\x7e]{4,}", data):
            text = match.group().decode("ascii")
            if text.endswith("/"):
                current_dir = text
                continue
            if not text.lower().endswith(".uasset"):
                continue
            base = text[:-7]
            full = (current_dir + text).lower()
            if not any(word in full for word in ("building", "house", "villa", "residential", "apartment")):
                continue
            if any(word in full for word in SKIP_WORDS) or base.lower().startswith(SKIP_PREFIX):
                continue
            where = current_dir
            at = where.find("Content/")
            where = where[at + 8:] if at >= 0 else where
            found.append(f"/Game/{where}{base}.{base}")
    unique = sorted(set(found))
    print(f"  found {len(unique)} possible building meshes in the CARLA package")
    return unique


# ------------------------------------------------------------------- the road
def collect_sites(carla, carla_map, spacing, rng, min_curve_radius):
    """Spots next to the road, on both sides, every ~spacing metres."""
    inner = [
        w for w in carla_map.generate_waypoints(8.0)
        if int(w.lane_type) & int(carla.LaneType.Driving) and abs(w.lane_id) == 1 and not w.is_junction
    ]
    groups = {}
    for w in inner:
        groups.setdefault((w.road_id, w.lane_id), []).append(w)

    sites = []
    for lane in groups.values():
        lane.sort(key=lambda w: w.s)
        next_s = lane[0].s + rng.uniform(0.2, 1.0) * spacing
        for w in lane:
            if w.s < next_s:
                continue
            next_s = w.s + spacing * rng.uniform(0.7, 1.3)
            ahead, behind = w.next(15.0), w.previous(15.0)
            if not ahead or not behind:
                continue
            turn = abs(wrap(math.radians(ahead[0].transform.rotation.yaw)
                            - math.radians(behind[0].transform.rotation.yaw)))
            if turn > 1e-3 and 30.0 / turn < min_curve_radius:
                continue                                  # too sharp a bend for a house
            outer = w
            for _ in range(8):                            # walk outward to the last lane
                nxt = outer.get_right_lane()
                if nxt is None:
                    break
                outer = nxt
            yaw = math.radians(w.transform.rotation.yaw)
            outward = (-math.sin(yaw), math.cos(yaw))    # right of the lane = away from the road
            loc = outer.transform.location
            half = outer.lane_width / 2.0
            sites.append({
                "edge": (loc.x + outward[0] * half, loc.y + outward[1] * half, loc.z),
                "outward": outward,
                "yaw": yaw,
            })
    rng.shuffle(sites)
    return sites


# -------------------------------------------------------------------- objects
class Prop:
    """A blueprint plus its measured size (taken from a throw-away test spawn)."""

    def __init__(self, world, blueprint):
        self.blueprint = blueprint
        self.ok = False
        probe = world.try_spawn_actor(blueprint, _far_away(world))
        if probe is None:
            return
        box = probe.bounding_box
        self.ex, self.ey, self.ez = box.extent.x, box.extent.y, box.extent.z
        self.ox, self.oy, self.oz = box.location.x, box.location.y, box.location.z
        probe.destroy()
        self.ok = max(self.ex, self.ey) > 0.1 and self.ez > 0.1
        # put the long side along the road
        self.turn = math.pi / 2.0 if self.ey > self.ex else 0.0
        self.long, self.short = max(self.ex, self.ey), min(self.ex, self.ey)


def _far_away(world):
    import carla
    return carla.Transform(carla.Location(x=0.0, y=0.0, z=-400.0))


def make_prop(world, library, blueprint_id, mesh_path=None):
    try:
        blueprint = library.find(blueprint_id)
    except (IndexError, RuntimeError):
        return None
    if mesh_path:
        blueprint.set_attribute("mesh_path", mesh_path)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", TAG)
    prop = Prop(world, blueprint)
    return prop if prop.ok else None


def spawn_part(world, carla, prop, center, base_yaw, extra_turn, ground_z, height_center):
    """Spawn one prop so its bounding-box centre lands on `center` (x, y)."""
    yaw = base_yaw + prop.turn + extra_turn
    off_x, off_y = rotate(prop.ox, prop.oy, yaw)
    origin = carla.Location(x=center[0] - off_x, y=center[1] - off_y,
                            z=ground_z + height_center - prop.oz + 0.02)
    actor = world.try_spawn_actor(
        prop.blueprint, carla.Transform(origin, carla.Rotation(yaw=math.degrees(yaw))))
    if actor is None:
        return None
    try:
        actor.set_simulate_physics(False)
    except RuntimeError:
        pass
    return actor


def build_house(world, carla, recipe, props, site, setback, rng):
    """Spawn one house at a site. Returns the list of spawned actors."""
    yaw = site["yaw"]
    out = site["outward"]
    along = (math.cos(yaw), math.sin(yaw))
    ground = site["edge"][2]
    actors = []

    if recipe == "container_house":
        p = props["container"]
        n_long, n_wide, floors = rng.choice([1, 2, 2, 3]), rng.choice([1, 2]), rng.choice([1, 1, 2])
        half_long, half_short = n_long * p.long, n_wide * p.short
        centre = (site["edge"][0] + out[0] * (setback + half_short),
                  site["edge"][1] + out[1] * (setback + half_short))
        shift = rng.uniform(-0.5, 0.5) * p.long
        for i in range(n_long):
            for j in range(n_wide):
                for k in range(floors):
                    dx = (i - (n_long - 1) / 2.0) * 2 * p.long + (shift if k else 0.0)
                    dy = (j - (n_wide - 1) / 2.0) * 2 * p.short
                    wx, wy = rotate(dx, dy, yaw)
                    actor = spawn_part(world, carla, p, (centre[0] + wx, centre[1] + wy),
                                       yaw, 0.0, ground, k * 2 * p.ez + p.ez)
                    if actor:
                        actors.append(actor)
    else:
        p = props[recipe]
        centre = (site["edge"][0] + out[0] * (setback + p.short),
                  site["edge"][1] + out[1] * (setback + p.short))
        actor = spawn_part(world, carla, p, centre, yaw, 0.0, ground, p.ez)
        if actor:
            actors.append(actor)
        centre = (centre[0], centre[1])
        half_short = p.short

    return actors, centre, half_short, along


def add_decor(world, carla, decor, site, setback, house_centre, half_short, rng):
    out, yaw = site["outward"], site["yaw"]
    along = (math.cos(yaw), math.sin(yaw))
    made = []
    for _ in range(rng.choice([0, 1, 2, 2])):
        prop = rng.choice(decor)
        lateral = setback * rng.uniform(0.25, 0.7)
        shift = rng.uniform(-6.0, 6.0)
        centre = (site["edge"][0] + out[0] * lateral + along[0] * shift,
                  site["edge"][1] + out[1] * lateral + along[1] * shift)
        actor = spawn_part(world, carla, prop, centre, yaw, rng.uniform(-0.5, 0.5),
                           site["edge"][2], prop.ez)
        if actor:
            made.append(actor)
    return made


# ----------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Add houses on both sides of the generated road.")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--spacing", type=float, default=90.0)
    parser.add_argument("--setback", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-decor", action="store_true")
    parser.add_argument("--carla-dir", default=None)
    parser.add_argument("--mesh", action="append", default=[])
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    carla_map = world.get_map()

    if args.remove:
        mine = [a for a in world.get_actors().filter("static.prop.*")
                if a.attributes.get("role_name") == TAG]
        client.apply_batch([carla.command.DestroyActor(a) for a in mine])
        print(f"Removed {len(mine)} building parts.")
        return

    print(f"Map: {carla_map.name}")
    rng = random.Random(args.seed)
    library = world.get_blueprint_library()

    # --- what can we build with?
    props = {}
    container = make_prop(world, library, "static.prop.container")
    if container:
        props["container"] = container
    for key, blueprint_id in (("kiosk", "static.prop.kiosk_01"), ("pergola", "static.prop.pergola"),
                              ("foodcart", "static.prop.foodcart")):
        prop = make_prop(world, library, blueprint_id)
        if prop:
            props[key] = prop

    mesh_props = []
    paths = list(args.mesh)
    if args.carla_dir:
        print("Looking for real building models in the CARLA package ...")
        found = find_building_meshes(args.carla_dir)
        rng.shuffle(found)
        paths += found[:40]
    for path in paths:
        prop = make_prop(world, library, "static.prop.mesh", mesh_path=path)
        if prop and 4.0 <= prop.long * 2 <= 60.0 and 3.0 <= prop.ez * 2 <= 45.0:
            mesh_props.append(prop)
        if len(mesh_props) >= 8:
            break
    if paths:
        print(f"  usable real building models: {len(mesh_props)}")

    if not props and not mesh_props:
        sys.exit("No building objects could be spawned. Is CARLA running with the road loaded?")

    decor = [] if args.no_decor else [
        p for p in (make_prop(world, library, i) for i in DECOR_IDS) if p]

    recipes = []
    if "container" in props:
        recipes += ["container_house"] * 6
    recipes += [key for key in ("kiosk", "pergola", "foodcart") if key in props]
    if mesh_props:
        recipes = ["mesh"] * (len(recipes) * 2 or 6) + recipes

    sites = collect_sites(carla, carla_map, args.spacing, rng, min_curve_radius=35.0)
    if not sites:
        sys.exit("The road has no free spots. Load the road first (generate_unstructured_map.py).")
    print(f"{len(sites)} spots along the road, building {min(args.count, len(sites))} houses ...")

    made, parts = 0, 0
    for site in sites:
        if made >= args.count:
            break
        recipe = rng.choice(recipes)
        if recipe == "mesh":
            prop = rng.choice(mesh_props)
            yaw, out = site["yaw"], site["outward"]
            centre = (site["edge"][0] + out[0] * (args.setback + prop.short),
                      site["edge"][1] + out[1] * (args.setback + prop.short))
            actor = spawn_part(world, carla, prop, centre, yaw, 0.0, site["edge"][2], prop.ez)
            actors, half_short = ([actor] if actor else []), prop.short
        else:
            actors, centre, half_short, _ = build_house(world, carla, recipe, props, site,
                                                        args.setback, rng)
        if not actors:
            continue
        made += 1
        parts += len(actors)
        if decor:
            parts += len(add_decor(world, carla, decor, site, args.setback, centre, half_short, rng))

    print(f"Done: {made} houses ({parts} objects) placed on both sides of the road.")
    print("Remove them later with:  python add_buildings.py --remove")
    print("Note: there is no ground beside the road, so the houses stand on the empty plane.")


if __name__ == "__main__":
    main()

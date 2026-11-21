#!/usr/bin/env python3
"""
generate_unstructured_map.py   (CARLA 0.9.15)

Creates an UNSTRUCTURED road network as an OpenDRIVE (.xodr) file and loads it
into the running CARLA server - no Unreal Editor needed.

What the road looks like
  * one closed, winding loop (random curves, different every --seed)
  * irregular lane widths (they change slowly along the road)
  * NO lane markings at all
  * shallow physical vertical dips in the OpenDRIVE road elevation profile
  * 2 lanes in each direction by default (so the car can go around obstacles)

Usage
    python generate_unstructured_map.py                  # make + load into CARLA
    python generate_unstructured_map.py --seed 7 --radius 300 --irregularity 0.8
    python generate_unstructured_map.py --no-load        # only write the .xodr file
    python generate_unstructured_map.py --preview        # also save a PNG sketch

Start CARLA first (low quality is fine):
    CarlaUE4.exe -quality-level=Low -ResX=640 -ResY=480 -windowed
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

MIN_INNER_EDGE_RADIUS_M = 20.0  # tightest curve at the INNER lane edge (a car at 20 km/h needs ~15 m)
WIDTH_NODE_SPACING_M = 25.0  # lane width changes every ~25 m
SPEED_LIMIT_KMH = 30
SHOULDER_WIDTH_M = 3.5


# ------------------------------------------------------------------ geometry
def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def make_nodes(radius, irregularity, rng, count):
    """Closed star-shaped loop: a circle bent by a few random harmonics."""
    harmonics = [
        (k, rng.uniform(0.5, 1.0) * irregularity * 0.30 / (k - 1) ** 0.8,
         rng.uniform(0.0, 2.0 * math.pi))
        for k in (2, 3, 4, 5, 7, 9)
    ]
    nodes = []
    for i in range(count):
        theta = 2.0 * math.pi * i / count
        factor = 1.0 + sum(a * math.cos(k * theta + ph) for k, a, ph in harmonics)
        factor *= 1.0 + rng.uniform(-1.0, 1.0) * irregularity * 0.008
        nodes.append((radius * factor * math.cos(theta),
                      radius * factor * math.sin(theta)))
    return nodes


def node_headings(nodes):
    count = len(nodes)
    return [
        math.atan2(nodes[(i + 1) % count][1] - nodes[i - 1][1],
                   nodes[(i + 1) % count][0] - nodes[i - 1][0])
        for i in range(count)
    ]


def hermite_segment(p0, h0, p1, h1):
    """Cubic (paramPoly3, normalized p in [0,1]) from p0/h0 to p1/h1."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    cos_h, sin_h = math.cos(-h0), math.sin(-h0)
    ex, ey = dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h  # end point, local frame
    delta = wrap_angle(h1 - h0)
    chord = math.hypot(ex, ey)
    t0 = (chord, 0.0)
    t1 = (chord * math.cos(delta), chord * math.sin(delta))
    coeff_u = (0.0, t0[0], 3 * ex - 2 * t0[0] - t1[0], -2 * ex + t0[0] + t1[0])
    coeff_v = (0.0, t0[1], 3 * ey - 2 * t0[1] - t1[1], -2 * ey + t0[1] + t1[1])
    return coeff_u, coeff_v


def poly(c, t):
    return c[0] + c[1] * t + c[2] * t * t + c[3] * t ** 3


def dpoly(c, t):
    return c[1] + 2 * c[2] * t + 3 * c[3] * t * t


def ddpoly(c, t):
    return 2 * c[2] + 6 * c[3] * t


def segment_stats(cu, cv, samples=2000):
    """Arc length and the tightest curvature of one cubic segment."""
    length = 0.0
    max_curv = 0.0
    prev = None
    for k in range(samples + 1):
        t = k / samples
        speed = math.hypot(dpoly(cu, t), dpoly(cv, t))
        if prev is not None:
            length += 0.5 * (speed + prev) / samples
        prev = speed
        if k % 20 == 0 and speed > 1e-9:
            curv = abs(dpoly(cu, t) * ddpoly(cv, t) - dpoly(cv, t) * ddpoly(cu, t)) / speed ** 3
            max_curv = max(max_curv, curv)
    return length, max_curv


def build_loop(radius, irregularity, seed, roads, half_road_width):
    """Return (segments, min_radius). Retries with a gentler shape if too tight."""
    rng = random.Random(seed)
    circumference = 2.0 * math.pi * radius
    count = max(3 * roads, int(round(circumference / 30.0 / roads)) * roads)
    attempt_irregularity = irregularity
    for _ in range(40):
        nodes = make_nodes(radius, attempt_irregularity, random.Random(seed), count)
        heads = node_headings(nodes)
        segments = []
        worst_curv = 0.0
        for i in range(count):
            j = (i + 1) % count
            cu, cv = hermite_segment(nodes[i], heads[i], nodes[j], heads[j])
            length, curv = segment_stats(cu, cv)
            worst_curv = max(worst_curv, curv)
            segments.append({"p": nodes[i], "h": heads[i], "cu": cu, "cv": cv,
                             "length": length, "end": nodes[j], "end_h": heads[j]})
        min_radius = 1.0 / worst_curv if worst_curv > 0 else float("inf")
        if min_radius - half_road_width >= MIN_INNER_EDGE_RADIUS_M:
            return segments, min_radius, attempt_irregularity
        attempt_irregularity *= 0.85
    raise RuntimeError("Could not build a loop with gentle enough curves; try a bigger --radius.")


def verify_geometry(segments):
    """Check that every segment ends exactly where (and how) the next one starts."""
    worst_pos = worst_hdg = 0.0
    for i, seg in enumerate(segments):
        cu, cv = seg["cu"], seg["cv"]
        u, v = poly(cu, 1.0), poly(cv, 1.0)
        c, s = math.cos(seg["h"]), math.sin(seg["h"])
        ex, ey = seg["p"][0] + u * c - v * s, seg["p"][1] + u * s + v * c
        worst_pos = max(worst_pos, math.hypot(ex - seg["end"][0], ey - seg["end"][1]))
        end_heading = seg["h"] + math.atan2(dpoly(cv, 1.0), dpoly(cu, 1.0))
        worst_hdg = max(worst_hdg, abs(wrap_angle(end_heading - seg["end_h"])))
    return worst_pos, worst_hdg


# ---------------------------------------------------------------- OpenDRIVE
def fmt(value):
    return f"{value:.9f}"



def build_rough_elevation_profile(road_length, seed, patch_count=2,
                                  dip_depth=0.045, dip_length=2.4):
    """
    Create shallow, smooth vertical depressions in the OpenDRIVE road profile.

    IMPORTANT: OpenDRIVE elevation is longitudinal and applies across the
    complete road width. These are physically modelled full-width road dips,
    NOT isolated left/right pothole cavities.
    """
    if patch_count < 0:
        raise ValueError("rough patch count cannot be negative")
    if dip_depth < 0.005 or dip_depth > 0.08:
        raise ValueError("dip depth must be between 0.005 m and 0.08 m")
    if dip_length < 1.0 or dip_length > 8.0:
        raise ValueError("dip length must be between 1.0 m and 8.0 m")
    if patch_count == 0:
        return [(0.0, 0.0, 0.0, 0.0, 0.0)]

    half = dip_length / 2.0
    spacing = road_length / (patch_count + 1)
    if spacing <= dip_length + 2.0:
        raise ValueError(
            "Road segment is too short for the selected rough patches; "
            "reduce --rough-patches or --dip-length."
        )

    rng = random.Random(seed)
    records = [(0.0, 0.0, 0.0, 0.0, 0.0)]
    for index in range(patch_count):
        center = spacing * (index + 1) + rng.uniform(-0.12, 0.12) * spacing
        start = center - half
        end = center + half

        # First half: smooth descent from level road to the bottom.
        # z(x) = -3*d/h^2*x^2 + 2*d/h^3*x^3
        records.append((
            start, 0.0, 0.0,
            -3.0 * dip_depth / (half ** 2),
            2.0 * dip_depth / (half ** 3),
        ))

        # Second half: smooth ascent from the bottom to level road.
        # z(x) = -d + 3*d/h^2*x^2 - 2*d/h^3*x^3
        records.append((
            center, -dip_depth, 0.0,
            3.0 * dip_depth / (half ** 2),
            -2.0 * dip_depth / (half ** 3),
        ))
        records.append((end, 0.0, 0.0, 0.0, 0.0))

    # OpenDRIVE elevation entries must be ordered by increasing s.
    records.sort(key=lambda item: item[0])

    # Remove any duplicate s positions caused by floating point edge cases.
    deduped = []
    for record in records:
        if deduped and abs(record[0] - deduped[-1][0]) < 1e-5:
            deduped[-1] = record
        else:
            deduped.append(record)
    return deduped


def build_xodr(segments, roads, lanes_per_side, min_width, max_width, seed,
              rough_patches=2, dip_depth=0.045, dip_length=2.4):
    rng = random.Random(seed + 1)
    per_road = len(segments) // roads
    total_length = sum(s["length"] for s in segments)

    # lane width profile: random values every ~25 m, periodic around the loop
    nodes_count = max(8, int(round(total_length / WIDTH_NODE_SPACING_M)))
    spacing = total_length / nodes_count
    lane_ids = list(range(lanes_per_side, 0, -1)) + [-i for i in range(1, lanes_per_side + 1)]
    profiles = {}
    for lane_id in lane_ids:
        values = [rng.uniform(min_width, max_width) for _ in range(nodes_count)]
        values.append(values[0])
        profiles[lane_id] = values

    def width_at(lane_id, s):
        values = profiles[lane_id]
        position = min(max(s / spacing, 0.0), nodes_count - 1e-9)
        k = int(position)
        frac = position - k
        return values[k] * (1 - frac) + values[k + 1] * frac

    out = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append("<OpenDRIVE>")
    out.append('  <header revMajor="1" revMinor="4" name="unstructured_loop" version="1.00" '
               f'date="{time.strftime("%Y-%m-%dT%H:%M:%S")}" north="0" south="0" east="0" west="0" '
               'vendor="custom"/>')

    road_start_s = 0.0
    for road_id in range(roads):
        first = road_id * per_road
        group = segments[first:first + per_road]
        road_length = sum(s["length"] for s in group)
        prev_id = (road_id - 1) % roads
        next_id = (road_id + 1) % roads
        out.append(f'  <road name="road_{road_id}" length="{fmt(road_length)}" id="{road_id}" junction="-1" rule="RHT">')
        out.append("    <link>")
        out.append(f'      <predecessor elementType="road" elementId="{prev_id}" contactPoint="end"/>')
        out.append(f'      <successor elementType="road" elementId="{next_id}" contactPoint="start"/>')
        out.append("    </link>")
        out.append(f'    <type s="0" type="rural"><speed max="{SPEED_LIMIT_KMH}" unit="km/h"/></type>')
        out.append("    <planView>")
        s_local = 0.0
        for seg in group:
            out.append(f'      <geometry s="{fmt(s_local)}" x="{fmt(seg["p"][0])}" y="{fmt(seg["p"][1])}" '
                       f'hdg="{fmt(seg["h"])}" length="{fmt(seg["length"])}">')
            cu, cv = seg["cu"], seg["cv"]
            out.append(f'        <paramPoly3 aU="{fmt(cu[0])}" bU="{fmt(cu[1])}" cU="{fmt(cu[2])}" dU="{fmt(cu[3])}" '
                       f'aV="{fmt(cv[0])}" bV="{fmt(cv[1])}" cV="{fmt(cv[2])}" dV="{fmt(cv[3])}" pRange="normalized"/>')
            out.append("      </geometry>")
            s_local += seg["length"]
        out.append("    </planView>")
        elevation_records = build_rough_elevation_profile(
            road_length,
            seed + road_id * 101,
            patch_count=rough_patches,
            dip_depth=dip_depth,
            dip_length=dip_length,
        )
        out.append("    <elevationProfile>")
        for elev_s, a, b, c, d in elevation_records:
            out.append(
                f'      <elevation s="{fmt(elev_s)}" a="{fmt(a)}" '
                f'b="{fmt(b)}" c="{fmt(c)}" d="{fmt(d)}"/>'
            )
        out.append("    </elevationProfile>")
        out.append("    <lateralProfile/>")
        out.append("    <lanes>")
        out.append('      <laneSection s="0">')

        # width breakpoints inside this road
        road_end_s = road_start_s + road_length
        cuts = [road_start_s] + [
            k * spacing for k in range(1, nodes_count)
            if road_start_s + 1e-6 < k * spacing < road_end_s - 1e-6
        ] + [road_end_s]

        def lane_xml(lane_id):
            lines = [f'          <lane id="{lane_id}" type="driving" level="false">']
            lines.append(f'            <link><predecessor id="{lane_id}"/><successor id="{lane_id}"/></link>')
            for a_s, b_s in zip(cuts, cuts[1:]):
                w0, w1 = width_at(lane_id, a_s), width_at(lane_id, b_s)
                slope = (w1 - w0) / (b_s - a_s)
                lines.append(f'            <width sOffset="{fmt(a_s - road_start_s)}" a="{fmt(w0)}" '
                             f'b="{fmt(slope)}" c="0" d="0"/>')
            lines.append('            <roadMark sOffset="0" type="none" weight="standard" color="standard" '
                         'width="0.0" laneChange="none"/>')
            lines.append("          </lane>")
            return lines

        def shoulder_xml(lane_id):
            return [
                f'          <lane id="{lane_id}" type="shoulder" level="false">',
                f'            <link><predecessor id="{lane_id}"/><successor id="{lane_id}"/></link>',
                f'            <width sOffset="0" a="{fmt(SHOULDER_WIDTH_M)}" b="0" c="0" d="0"/>',
                '            <roadMark sOffset="0" type="none" weight="standard" color="standard" '
                'width="0.0" laneChange="none"/>',
                "          </lane>",
            ]

        out.append("        <left>")
        out.extend(shoulder_xml(lanes_per_side + 1))
        for lane_id in range(lanes_per_side, 0, -1):
            out.extend(lane_xml(lane_id))
        out.append("        </left>")
        out.append("        <center>")
        out.append('          <lane id="0" type="none" level="false">')
        out.append('            <roadMark sOffset="0" type="none" weight="standard" color="standard" '
                   'width="0.0" laneChange="none"/>')
        out.append("          </lane>")
        out.append("        </center>")
        out.append("        <right>")
        for lane_id in range(-1, -lanes_per_side - 1, -1):
            out.extend(lane_xml(lane_id))
        out.extend(shoulder_xml(-(lanes_per_side + 1)))
        out.append("        </right>")
        out.append("      </laneSection>")
        out.append("    </lanes>")
        out.append("  </road>")
        road_start_s = road_end_s
    out.append("</OpenDRIVE>")
    return "\n".join(out) + "\n", total_length, profiles


# ------------------------------------------------------------------ preview
def save_preview(segments, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - preview skipped)")
        return
    xs, ys = [], []
    for seg in segments:
        c, s = math.cos(seg["h"]), math.sin(seg["h"])
        for k in range(21):
            t = k / 20.0
            u, v = poly(seg["cu"], t), poly(seg["cv"], t)
            xs.append(seg["p"][0] + u * c - v * s)
            ys.append(seg["p"][1] + u * s + v * c)
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, "k-", linewidth=3)
    plt.axis("equal")
    plt.title("Unstructured road loop (centre line)")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print("Preview saved:", path)


# ------------------------------------------------------------------- CARLA
def load_into_carla(xodr, host, port, walls):
    import carla

    client = carla.Client(host, port)
    client.set_timeout(180.0)
    parameters = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=50.0,
        wall_height=walls,
        additional_width=0.6,
        smooth_junctions=True,
        enable_mesh_visibility=True,
        enable_pedestrian_navigation=True,  # optional; controller crossings do not depend on navmesh
    )
    print("Sending the road to CARLA (this can take up to a minute)...")
    client.generate_opendrive_world(xodr, parameters)
    time.sleep(2.0)
    world = client.get_world()
    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(10.0)
    print(f"Loaded map '{carla_map.name}' with {len(waypoints)} road waypoints "
          f"(spawn points: {len(carla_map.get_spawn_points())}).")
    if waypoints:
        spot = waypoints[0].transform.location
        world.get_spectator().set_transform(
            carla.Transform(carla.Location(x=spot.x, y=spot.y, z=80.0),
                            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)))
        print("Spectator moved above the road start.")
    print(
        "\nNow run the generated-road controller, for example:\n"
        "  python unstructured_autonomous_rough_road_pedestrians.py"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate and load an unstructured CARLA road loop.")
    parser.add_argument("--seed", type=int, default=42, help="different number = different road")
    parser.add_argument("--radius", type=float, default=230.0, help="average loop radius in metres (lap ~ 6.28 x radius)")
    parser.add_argument("--irregularity", type=float, default=0.8, help="0 = circle, 1 = very winding")
    parser.add_argument("--lanes", type=int, default=2, choices=(1, 2, 3), help="lanes per direction")
    parser.add_argument("--min-width", type=float, default=3.0, help="narrowest lane (m)")
    parser.add_argument("--max-width", type=float, default=4.2, help="widest lane (m)")
    parser.add_argument("--out", default="unstructured_road.xodr")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--walls", type=float, default=0.0, help="edge wall height in m (0 = no walls)")
    parser.add_argument("--rough-patches", type=int, default=2, help="smooth physical road dips per OpenDRIVE road segment")
    parser.add_argument("--dip-depth", type=float, default=0.045, help="maximum vertical dip depth in metres (0.045 = 4.5 cm)")
    parser.add_argument("--dip-length", type=float, default=2.4, help="length of each road dip in metres")
    parser.add_argument("--no-load", action="store_true", help="only write the .xodr file")
    parser.add_argument("--preview", action="store_true", help="save a PNG sketch of the loop")
    args = parser.parse_args()

    if args.min_width < 2.6 or args.max_width < args.min_width:
        sys.exit("Lane widths must satisfy 2.6 <= min-width <= max-width (the car is 2.1 m wide).")

    roads = 3
    segments, min_radius, used_irregularity = build_loop(
        args.radius, args.irregularity, args.seed, roads,
        args.lanes * args.max_width + SHOULDER_WIDTH_M)
    pos_err, hdg_err = verify_geometry(segments)
    if pos_err > 1e-6 or hdg_err > 1e-6:
        sys.exit(f"Geometry check failed (position error {pos_err:.2e} m, heading error {hdg_err:.2e} rad)")

    xodr, total_length, _ = build_xodr(
        segments, roads, args.lanes, args.min_width, args.max_width, args.seed,
        rough_patches=args.rough_patches,
        dip_depth=args.dip_depth,
        dip_length=args.dip_length,
    )
    Path(args.out).write_text(xodr, encoding="utf-8")

    print(f"Road written to {args.out}")
    print(f"  lap length        : {total_length:.0f} m")
    print(f"  tightest curve    : radius {min_radius:.0f} m")
    if used_irregularity < args.irregularity - 1e-9:
        print(f"  (irregularity reduced from {args.irregularity:.2f} to {used_irregularity:.2f} to keep curves drivable)")
    print(f"  lanes             : {args.lanes} per direction, width {args.min_width}-{args.max_width} m, no markings")
    print(f"  shoulders         : {SHOULDER_WIDTH_M:.1f} m on both sides")
    print(f"  rough road dips   : {args.rough_patches} per road segment, {args.dip_depth:.3f} m deep, {args.dip_length:.1f} m long")
    print("  NOTE              : dips span the full road width; they are not isolated pothole cavities.")
    print(f"  geometry check    : OK (gaps {pos_err:.1e} m, {hdg_err:.1e} rad)")

    if args.preview:
        save_preview(segments, str(Path(args.out).with_suffix(".png")))
    if args.no_load:
        return
    try:
        load_into_carla(xodr, args.host, args.port, args.walls)
    except ImportError:
        sys.exit("The 'carla' Python package is not installed in this Python. "
                 "The .xodr file was still saved; load it with --no-load later.")
    except RuntimeError as exc:
        sys.exit(f"CARLA error: {exc}\nIs CARLA running? The file {args.out} was saved anyway.")


if __name__ == "__main__":
    main()

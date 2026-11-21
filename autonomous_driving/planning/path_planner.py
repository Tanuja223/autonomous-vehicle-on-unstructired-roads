"""
planning/path_planner.py
━━━━━━━━━━━━━━━━━━━━━━━━
Path Planning Module
─────────────────────
Global Planner  : A* on a grid/graph representation of the road network.
Local Planner   : Pure Pursuit / Stanley controller for waypoint following
                  with obstacle avoidance via potential fields.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────

@dataclass(order=True)
class PriorityEntry:
    priority: float
    node: Tuple[int, int] = field(compare=False)


@dataclass
class Waypoint:
    x: float
    y: float
    z: float = 0.0
    heading_deg: float = 0.0    # Road heading at this point
    speed_limit: float = 50.0   # km/h

    def distance_to(self, other: "Waypoint") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Path:
    waypoints: List[Waypoint] = field(default_factory=list)
    total_length_m: float = 0.0
    is_feasible: bool = False

    def __len__(self) -> int:
        return len(self.waypoints)

    def compute_length(self) -> float:
        if len(self.waypoints) < 2:
            self.total_length_m = 0.0
            return 0.0
        total = sum(
            self.waypoints[i].distance_to(self.waypoints[i + 1])
            for i in range(len(self.waypoints) - 1)
        )
        self.total_length_m = total
        return total


# ────────────────────────────────────────────────────────────
# A* Global Planner
# ────────────────────────────────────────────────────────────

class AStarPlanner:
    """
    A* search on a discrete occupancy grid.

    The grid is derived from the HD map / CARLA road network.
    Each cell represents `resolution` metres of real-world space.

    Parameters
    ----------
    resolution_m  : Metres per grid cell.
    grid_size     : (rows, cols) of the occupancy grid.
    """

    def __init__(
        self,
        resolution_m: float = 1.0,
        grid_size: Tuple[int, int] = (200, 200),
    ) -> None:
        self.res    = resolution_m
        self.rows, self.cols = grid_size
        self.grid   = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.origin = Waypoint(0.0, 0.0)   # World coords of cell (0,0)

    # ── Public API ───────────────────────────────────────────

    def set_obstacles(self, obstacles: List[Tuple[float, float, float]]) -> None:
        """
        Mark obstacle regions on the grid.

        Parameters
        ----------
        obstacles : List of (world_x, world_y, radius_m)
        """
        self.grid.fill(0)
        for wx, wy, radius in obstacles:
            cx, cy = self._world_to_grid(wx, wy)
            r_cells = max(1, int(radius / self.res))
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    if dr ** 2 + dc ** 2 <= r_cells ** 2:
                        rr, cc = cx + dr, cy + dc
                        if 0 <= rr < self.rows and 0 <= cc < self.cols:
                            self.grid[rr, cc] = 1

    def plan(
        self, start: Waypoint, goal: Waypoint
    ) -> Path:
        """Run A* from start to goal. Returns a `Path`."""
        s_cell = self._world_to_grid(start.x, start.y)
        g_cell = self._world_to_grid(goal.x, goal.y)

        if not self._valid(s_cell) or not self._valid(g_cell):
            return Path(is_feasible=False)

        # A* search
        open_heap: List[PriorityEntry] = []
        heapq.heappush(open_heap, PriorityEntry(0.0, s_cell))
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {s_cell: None}
        g_score: Dict[Tuple[int, int], float] = {s_cell: 0.0}

        while open_heap:
            current = heapq.heappop(open_heap).node

            if current == g_cell:
                return self._reconstruct(came_from, current, start, goal)

            for nbr in self._neighbours(current):
                step_cost = math.hypot(
                    nbr[0] - current[0], nbr[1] - current[1]
                ) * self.res
                new_g = g_score[current] + step_cost

                if nbr not in g_score or new_g < g_score[nbr]:
                    g_score[nbr] = new_g
                    f = new_g + self._heuristic(nbr, g_cell)
                    heapq.heappush(open_heap, PriorityEntry(f, nbr))
                    came_from[nbr] = current

        # No path found
        return Path(is_feasible=False)

    # ── Internal ─────────────────────────────────────────────

    def _world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        r = int((wx - self.origin.x) / self.res)
        c = int((wy - self.origin.y) / self.res)
        return max(0, min(r, self.rows - 1)), max(0, min(c, self.cols - 1))

    def _grid_to_world(self, r: int, c: int) -> Tuple[float, float]:
        wx = self.origin.x + r * self.res
        wy = self.origin.y + c * self.res
        return wx, wy

    def _valid(self, cell: Tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0

    def _neighbours(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = cell
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        return [
            (r + dr, c + dc)
            for dr, dc in dirs
            if self._valid((r + dr, c + dc))
        ]

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _reconstruct(
        self,
        came_from: Dict,
        current: Tuple[int, int],
        start: Waypoint,
        goal: Waypoint,
    ) -> Path:
        cells = []
        while current is not None:
            cells.append(current)
            current = came_from[current]
        cells.reverse()

        waypoints: List[Waypoint] = []
        for i, (r, c) in enumerate(cells):
            wx, wy = self._grid_to_world(r, c)
            # Compute heading from consecutive cells
            if i + 1 < len(cells):
                nr, nc = cells[i + 1]
                nwx, nwy = self._grid_to_world(nr, nc)
                heading = math.degrees(math.atan2(nwy - wy, nwx - wx))
            else:
                heading = waypoints[-1].heading_deg if waypoints else 0.0
            waypoints.append(Waypoint(wx, wy, heading_deg=heading))

        path = Path(waypoints=waypoints, is_feasible=True)
        path.compute_length()
        return path


# ────────────────────────────────────────────────────────────
# Local Planner — Pure Pursuit
# ────────────────────────────────────────────────────────────

class LocalPlanner:
    """
    Pure Pursuit local planner.

    Tracks a list of global waypoints and outputs the desired
    steering angle and speed using the pure-pursuit algorithm.
    Combined with an artificial potential field for obstacle avoidance.

    Parameters
    ----------
    lookahead_m      : Lookahead distance (m).
    wheelbase_m      : Vehicle wheelbase (m).
    max_steer_rad    : Max steering angle (rad).
    obstacle_gain    : Repulsive potential field gain.
    """

    def __init__(
        self,
        lookahead_m: float = 8.0,
        wheelbase_m: float = 2.9,
        max_steer_rad: float = 0.7,
        obstacle_gain: float = 2.0,
    ) -> None:
        self.lookahead      = lookahead_m
        self.wheelbase      = wheelbase_m
        self.max_steer      = max_steer_rad
        self.obstacle_gain  = obstacle_gain

        self._waypoints: List[Waypoint] = []
        self._current_idx: int = 0

    # ── Public API ───────────────────────────────────────────

    def set_path(self, path: Path) -> None:
        self._waypoints  = path.waypoints
        self._current_idx = 0

    def compute_steering(
        self,
        ego_x: float,
        ego_y: float,
        ego_heading_deg: float,
        obstacles: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[float, float, bool]:
        """
        Compute desired steering angle and look-ahead heading.

        Returns
        -------
        (steering_normalised, target_heading_deg, goal_reached)
        steering_normalised : –1.0 … +1.0
        """
        if not self._waypoints:
            return 0.0, ego_heading_deg, True

        # Advance index to lookahead waypoint
        target_wp = self._find_lookahead(ego_x, ego_y)
        if target_wp is None:
            return 0.0, ego_heading_deg, True

        # Angle to target in vehicle frame
        dx = target_wp.x - ego_x
        dy = target_wp.y - ego_y
        target_angle_world = math.degrees(math.atan2(dy, dx))
        alpha = math.radians(target_angle_world - ego_heading_deg)

        # Normalise alpha to [-π, π]
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

        # Pure pursuit steering (radians)
        dist = math.hypot(dx, dy)
        if dist < 0.5:
            steer_rad = 0.0
        else:
            steer_rad = math.atan2(
                2.0 * self.wheelbase * math.sin(alpha), dist
            )

        # Obstacle repulsion
        if obstacles:
            steer_rad += self._obstacle_repulsion(ego_x, ego_y, obstacles)

        # Clamp
        steer_rad = max(-self.max_steer, min(self.max_steer, steer_rad))
        steer_norm = steer_rad / self.max_steer

        goal_reached = (
            self._current_idx >= len(self._waypoints) - 1
            and math.hypot(
                self._waypoints[-1].x - ego_x,
                self._waypoints[-1].y - ego_y,
            ) < 3.0
        )

        return steer_norm, target_angle_world, goal_reached

    def distance_to_goal(self, ego_x: float, ego_y: float) -> float:
        if not self._waypoints:
            return 0.0
        g = self._waypoints[-1]
        return math.hypot(g.x - ego_x, g.y - ego_y)

    # ── Internal ─────────────────────────────────────────────

    def _find_lookahead(self, ego_x: float, ego_y: float) -> Optional[Waypoint]:
        """Return the first waypoint beyond the lookahead distance."""
        for i in range(self._current_idx, len(self._waypoints)):
            wp = self._waypoints[i]
            if math.hypot(wp.x - ego_x, wp.y - ego_y) >= self.lookahead:
                self._current_idx = i
                return wp
        # Beyond last waypoint
        self._current_idx = len(self._waypoints) - 1
        return self._waypoints[-1]

    def _obstacle_repulsion(
        self, ex: float, ey: float, obstacles: List[Tuple[float, float]]
    ) -> float:
        """Artificial potential field repulsion (radians)."""
        repulsion = 0.0
        for ox, oy in obstacles:
            dx, dy = ex - ox, ey - oy
            dist = math.hypot(dx, dy)
            if dist < 0.1 or dist > 15.0:
                continue
            force = self.obstacle_gain / (dist ** 2)
            angle = math.atan2(dy, dx)
            repulsion += force * math.sin(angle)
        return repulsion


# ────────────────────────────────────────────────────────────
# Quick Test
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    planner = AStarPlanner(resolution_m=2.0, grid_size=(100, 100))
    planner.set_obstacles([(60, 40, 6), (40, 70, 5)])

    start = Waypoint(0, 0)
    goal  = Waypoint(180, 180)
    path  = planner.plan(start, goal)

    print("=" * 50)
    print("A* Global Planner Test")
    print(f"  Feasible        : {path.is_feasible}")
    print(f"  Waypoints       : {len(path)}")
    print(f"  Total length    : {path.total_length_m:.1f} m")
    print()

    local = LocalPlanner(lookahead_m=8.0)
    local.set_path(path)
    steer, heading, done = local.compute_steering(5.0, 5.0, 45.0)
    print("Local Planner Test")
    print(f"  Steering (norm) : {steer:.4f}")
    print(f"  Target heading  : {heading:.1f}°")
    print(f"  Goal reached    : {done}")
    print("=" * 50)

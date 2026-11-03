"""Lazy Theta* on a grid (any-angle path planning), used by LiDAR-ORCA.

The map is rasterized at `resolution` px; a cell is free if its center is outside the inflated
obstacles (InflatedMap). Theta* connects a cell directly to its grandparent when the straight line
between them is free, so paths are straight lines with few corners instead of 8-direction zigzags.
"Lazy": the line-of-sight check is done once when a cell is expanded, not for every neighbor.

Nash, Koenig, Tovey, "Lazy Theta*: Any-Angle Path Planning and Path Length Analysis in 3D", AAAI 2010.
"""
import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

from .vehicle import Path
from .walls import InflatedMap

_DIRS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def auto_resolution(obstacles, wall_margin, max_cell=20.0):
    """Grid cell size from the narrowest corridor: half the free band (gap - 2 * wall_margin), clamped
    to [5, max_cell]. If cells are larger than the band, no cell center lands in the corridor."""
    boxes = [p.bounds for p in obstacles]
    best = None
    for i in range(len(boxes)):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            bx0, by0, bx1, by1 = boxes[j]
            if min(ax1, bx1) - max(ax0, bx0) > 1:          # overlap in x -> vertical gap
                g = max(by0 - ay1, ay0 - by1)
                if g > 1 and (best is None or g < best):
                    best = g
            if min(ay1, by1) - max(ay0, by0) > 1:          # overlap in y -> horizontal gap
                g = max(bx0 - ax1, ax0 - bx1)
                if g > 1 and (best is None or g < best):
                    best = g
    if best is None:
        return max_cell
    return max(5.0, min(max_cell, (best - 2.0 * wall_margin) * 0.5))


class ThetaStar:
    def __init__(self, env: InflatedMap, resolution: float, max_speed: float):
        self.env = env
        self.res = resolution
        # The heuristic is distance / max_speed while the edge cost is distance, so the heuristic is
        # weaker than plain A*. Kept as is: it decides which of several equal-length paths is returned.
        self.max_speed = max_speed
        minx, miny, maxx, maxy = env.bounds
        self.ox, self.oy = float(minx), float(miny)
        self.nx = int((maxx - minx) / resolution) + 1
        self.ny = int((maxy - miny) / resolution) + 1
        self.free = np.zeros((self.nx, self.ny), dtype=bool)
        for i in range(self.nx):
            for j in range(self.ny):
                self.free[i, j] = env.is_point_free(*self._cell_to_xy(i, j))

    def _cell_to_xy(self, i, j) -> Tuple[float, float]:
        return float(self.ox + i * self.res), float(self.oy + j * self.res)

    def _xy_to_cell(self, x, y) -> Tuple[int, int]:
        i = int(round((x - self.ox) / self.res))
        j = int(round((y - self.oy) / self.res))
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def _nearest_free_cell(self, x, y):
        ci, cj = self._xy_to_cell(x, y)
        if self.free[ci, cj]:
            return ci, cj
        for r in range(1, max(self.nx, self.ny)):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    i, j = ci + di, cj + dj
                    if 0 <= i < self.nx and 0 <= j < self.ny and self.free[i, j]:
                        return i, j
        raise RuntimeError("no free cell (is the whole map blocked?)")

    def _candidate_cells(self, x, y, radius_cells=2):
        """Free cells around the exact point (x, y) with the cost to reach them, so the search itself
        chooses where to enter / leave the grid."""
        ci, cj = self._xy_to_cell(x, y)
        out = []
        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                i, j = ci + di, cj + dj
                if 0 <= i < self.nx and 0 <= j < self.ny and self.free[i, j]:
                    cx, cy = self._cell_to_xy(i, j)
                    out.append(((i, j), float(math.hypot(cx - x, cy - y))))
        if out:
            return out
        cell = self._nearest_free_cell(x, y)
        cx, cy = self._cell_to_xy(*cell)
        return [(cell, float(math.hypot(cx - x, cy - y)))]

    def _neighbors(self, cell):
        ci, cj = cell
        result = []
        for di, dj in _DIRS_8:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < self.nx and 0 <= nj < self.ny) or not self.free[ni, nj]:
                continue
            # A diagonal move needs both side cells free (no corner cutting).
            if di != 0 and dj != 0 and (not self.free[ci + di, cj] or not self.free[ci, cj + dj]):
                continue
            result.append((ni, nj))
        return result

    def _edge_cost(self, a, b) -> float:
        ax, ay = self._cell_to_xy(*a)
        bx, by = self._cell_to_xy(*b)
        return float(math.hypot(bx - ax, by - ay))

    def _line_of_sight(self, a, b) -> bool:
        return self.env.is_line_free(self._cell_to_xy(*a), self._cell_to_xy(*b))

    def _search(self, start_candidates, goal_candidates) -> Optional[List[Tuple[int, int]]]:
        goal_extra = dict(goal_candidates)
        goal_set = set(goal_extra)
        goal_xy = [self._cell_to_xy(*g) for g in goal_set]

        def heuristic(c):
            ax, ay = self._cell_to_xy(*c)
            return min(float(math.hypot(bx - ax, by - ay)) for bx, by in goal_xy) / self.max_speed

        g_score, parent, open_heap = {}, {}, []
        for cell, cost in start_candidates:
            if cell not in g_score or cost < g_score[cell]:
                g_score[cell] = cost
                parent[cell] = cell              # a root: its own parent
                heapq.heappush(open_heap, (cost + heuristic(cell), cell))
        closed = set()
        best_total, best_cell = float("inf"), None

        while open_heap:
            f, s = heapq.heappop(open_heap)
            if s in closed:
                continue
            if f >= best_total:                  # no remaining cell can beat the best goal found
                break
            # SetVertex: verify the optimistic parent link now. If the line is blocked, fall back to
            # the cheapest closed neighbor with a verified line of sight.
            p = parent[s]
            if p != s and not self._line_of_sight(p, s):
                best_g, best_p = float("inf"), None
                for nb in self._neighbors(s):
                    if nb not in g_score or not self._line_of_sight(nb, s):
                        continue
                    cand = g_score[nb] + self._edge_cost(nb, s)
                    if cand < best_g:
                        best_g, best_p = cand, nb
                if best_p is None:
                    continue
                parent[s] = best_p
                g_score[s] = best_g

            closed.add(s)
            if s in goal_set:
                total = g_score[s] + goal_extra[s]
                if total < best_total:
                    best_total, best_cell = total, s
                continue

            p_s = parent[s]
            for s2 in self._neighbors(s):
                if s2 in closed:
                    continue
                if s2 not in g_score:
                    g_score[s2] = float("inf")
                # Optimistically link s2 to the grandparent (any-angle); fall back to s if cheaper.
                cost_via_parent = g_score[p_s] + self._edge_cost(p_s, s2)
                cost_via_s = g_score[s] + self._edge_cost(s, s2)
                if cost_via_parent <= cost_via_s:
                    cand_g, cand_p = cost_via_parent, p_s
                else:
                    cand_g, cand_p = cost_via_s, s
                if cand_g < g_score[s2] - 1e-9:
                    g_score[s2] = cand_g
                    parent[s2] = cand_p
                    heapq.heappush(open_heap, (cand_g + heuristic(s2), s2))

        if best_cell is None:
            return None
        path = [best_cell]
        while path[-1] != parent[path[-1]]:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    def _simplify(self, waypoints):
        """String pulling: drop intermediate points while the straight line stays free."""
        if len(waypoints) <= 2:
            return waypoints
        simplified = [waypoints[0]]
        i = 0
        while i < len(waypoints) - 1:
            j = len(waypoints) - 1
            while j > i + 1 and not self.env.is_line_free(waypoints[i], waypoints[j]):
                j -= 1
            simplified.append(waypoints[j])
            i = j
        return simplified

    def plan(self, start, goal) -> Optional[Path]:
        if self.env.is_line_free(start, goal):
            return Path([tuple(start), tuple(goal)])
        cells = self._search(self._candidate_cells(*start), self._candidate_cells(*goal))
        if cells is None:
            return None
        waypoints = [tuple(start)] + [self._cell_to_xy(i, j) for i, j in cells] + [tuple(goal)]
        return Path(self._simplify(waypoints))

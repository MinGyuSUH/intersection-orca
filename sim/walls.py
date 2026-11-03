"""Map loading and wall geometry.

Two views of the same walls:
  WallGeometry  - the real walls, used for collision checks, wall distances, ray casts and LiDAR
                  occlusion. The map boundary is treated as a wall too.
  InflatedMap   - walls grown by (body radius + margin), used only by the grid planner (Theta*)
                  so that planned paths keep a safety distance from the walls.

Distances follow one convention everywhere: "gap" = distance from the body surface (a circle of
radius R around the vehicle center) to the nearest wall. gap < 0 means the body overlaps a wall.
"""
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numba import njit
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree


@dataclass
class Map:
    name: str
    bounds: Tuple[float, float, float, float]   # minx, miny, maxx, maxy
    obstacles: List[Polygon]
    grid: dict                                   # goal grid: cell size and cell counts


def load_map(path) -> Map:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return Map(name=d["map_name"], bounds=tuple(d["workspace_bounds"]),
               obstacles=[Polygon(o["exterior"]) for o in d["obstacles"]], grid=d["grid"])


@njit(cache=True)
def _nearest_point(px, py, ax, ay, bx, by, own, poly_idx):
    """Nearest point on the segments (a -> b) to (px, py). If poly_idx >= 0, only segments owned by
    that polygon are considered. Returns (distance, qx, qy), or (-1, 0, 0) if there are none."""
    best = 1e18
    bqx = 0.0
    bqy = 0.0
    n = ax.shape[0]
    for k in range(n):
        if poly_idx >= 0 and own[k] != poly_idx:
            continue
        ex = bx[k] - ax[k]
        ey = by[k] - ay[k]
        den = ex * ex + ey * ey
        if den > 1e-18:
            t = ((px - ax[k]) * ex + (py - ay[k]) * ey) / den
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        else:
            t = 0.0
        qx = ax[k] + t * ex
        qy = ay[k] + t * ey
        ddx = qx - px
        ddy = qy - py
        d2 = ddx * ddx + ddy * ddy
        if d2 < best:
            best = d2
            bqx = qx
            bqy = qy
    if best > 1e17:
        return -1.0, 0.0, 0.0
    return np.sqrt(best), bqx, bqy


@njit(cache=True)
def _ray_first_hit(px, py, dx, dy, ax, ay, bx, by):
    """First intersection of the ray p + t*d (0 <= t <= 1) with the segments.
    Returns (t, segment index), or (-1, -1) if nothing is hit. Parallel segments are skipped."""
    best_t = 1e18
    best_k = -1
    n = ax.shape[0]
    for k in range(n):
        sx = bx[k] - ax[k]
        sy = by[k] - ay[k]
        den = dx * sy - dy * sx
        if den > -1e-12 and den < 1e-12:
            continue
        qx = ax[k] - px
        qy = ay[k] - py
        t = (qx * sy - qy * sx) / den
        if t < 0.0 or t > 1.0:
            continue
        u = (qx * dy - qy * dx) / den
        if u < 0.0 or u > 1.0:
            continue
        if t < best_t:
            best_t = t
            best_k = k
    if best_k < 0:
        return -1.0, -1
    return best_t, best_k


def _segments(polys):
    """All rings of the polygons as segment arrays: (ax, ay, bx, by, owner polygon index)."""
    A, B, own = [], [], []
    for i, poly in enumerate(polys):
        for ring in [poly.exterior] + list(poly.interiors):
            c = np.asarray(ring.coords, dtype=float)
            if len(c) < 2:
                continue
            A.append(c[:-1])
            B.append(c[1:])
            own.append(np.full(len(c) - 1, i, dtype=np.int64))
    A = np.concatenate(A)
    B = np.concatenate(B)
    return (np.ascontiguousarray(A[:, 0]), np.ascontiguousarray(A[:, 1]),
            np.ascontiguousarray(B[:, 0]), np.ascontiguousarray(B[:, 1]), np.concatenate(own))


class WallGeometry:
    """The real walls (no margin) for a body of the given radius."""

    def __init__(self, m: Map, radius: float):
        self.radius = radius
        self.bounds = m.bounds
        self.raw_polys: List[Polygon] = list(m.obstacles)
        self.buffered_polys: List[Polygon] = [p.buffer(radius) for p in self.raw_polys]
        # The map boundary is a wall too: a thick frame polygon whose hole is the workspace.
        minx, miny, maxx, maxy = m.bounds
        bw = max(maxx - minx, maxy - miny)
        outer = [(minx - bw, miny - bw), (maxx + bw, miny - bw), (maxx + bw, maxy + bw), (minx - bw, maxy + bw)]
        inner = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        frame = Polygon(outer, [inner])
        self.raw_polys.append(frame)
        self.buffered_polys.append(frame.buffer(radius))

        # Exact wall segments (raw walls) and the body-inflated outline used for ray casts.
        self.ax, self.ay, self.bx, self.by, self.owner = _segments(self.raw_polys)
        self.segs = np.c_[self.ax, self.ay, self.bx, self.by]        # (n, 4)
        self._ray = _segments(self.buffered_polys)
        self._build_distance_field()

    def _build_distance_field(self):
        """1 px distance field over the whole map for O(1) approximate gap lookups."""
        minx, miny, maxx, maxy = self.bounds
        w, h = int(maxx - minx) + 2, int(maxy - miny) + 2
        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        draw.rectangle([1, 1, w - 2, h - 2], fill=255)       # 1 px black border = map boundary
        for poly in self.raw_polys[:-1]:                      # obstacles (not the boundary frame)
            pts = [(x - minx, y - miny) for x, y in poly.exterior.coords]
            if len(pts) >= 3:
                draw.polygon(pts, fill=0)
        free = (np.array(img, dtype=np.uint8) > 0).astype(np.float32)
        dist = distance_transform_edt(free).astype(np.float32)
        self.dist_field = dist - self.radius                   # gap, same convention as nearest_wall
        self.df_minx = float(minx)
        self.df_miny = float(miny)
        self._df_h, self._df_w = dist.shape

    def nearest_wall_fast(self, p) -> float:
        """Approximate gap from the distance field (error up to about 1 px)."""
        col = int(float(p[0]) - self.df_minx)
        row = int(float(p[1]) - self.df_miny)
        col = max(0, min(col, self._df_w - 1))
        row = max(0, min(row, self._df_h - 1))
        return self.dist_field.item(row, col)

    def _nearest_raw(self, p, poly_idx=-1):
        d, qx, qy = _nearest_point(float(p[0]), float(p[1]), self.ax, self.ay, self.bx, self.by,
                                   self.owner, int(poly_idx))
        if d < 0.0:
            return None, None
        return d, np.array([qx, qy])

    def nearest_wall(self, p) -> Tuple[float, Optional[np.ndarray]]:
        """Exact gap to the nearest wall and the wall's unit normal pointing toward p."""
        dist, q = self._nearest_raw(p)
        if dist < 1e-9:
            return -self.radius, None
        vec = np.asarray(p, dtype=float) - q
        return dist - self.radius, vec / dist

    def raycast(self, p0, heading, max_dist: float = 1e5):
        """Cast a ray of the body center along heading. Returns (hit point, wall normal) where the
        body would first touch a wall, or (end point, None) if nothing is hit within max_dist."""
        p1 = p0 + heading * max_dist
        p0 = np.asarray(p0, dtype=float)
        d = np.asarray(p1, dtype=float) - p0
        ax, ay, bx, by, owner = self._ray
        t, k = _ray_first_hit(float(p0[0]), float(p0[1]), float(d[0]), float(d[1]), ax, ay, bx, by)
        if k < 0:
            return p1, None
        pt = p0 + d * t
        # Normal: from the nearest point on the raw polygon that was hit (buffer() can change ring topology).
        _, q = self._nearest_raw(pt, poly_idx=int(owner[k]))
        if q is None:
            return p1, None
        nv = pt - q
        nn = math.sqrt(nv[0] * nv[0] + nv[1] * nv[1])
        if nn < 1e-9:
            return p1, None
        return pt, nv / nn


class InflatedMap:
    """Obstacles grown by `margin`, for grid path planning."""

    def __init__(self, m: Map, margin: float):
        self.bounds = m.bounds
        self.polys = [p.buffer(margin) for p in m.obstacles]
        self._tree = STRtree(self.polys)

    def in_bounds(self, x, y) -> bool:
        minx, miny, maxx, maxy = self.bounds
        return minx <= x <= maxx and miny <= y <= maxy

    def is_point_free(self, x, y) -> bool:
        if not self.in_bounds(x, y):
            return False
        pt = Point(x, y)
        return not any(self.polys[i].contains(pt) for i in self._tree.query(pt))

    def is_line_free(self, p1, p2) -> bool:
        p1 = (float(p1[0]), float(p1[1]))
        p2 = (float(p2[0]), float(p2[1]))
        if not (self.in_bounds(*p1) and self.in_bounds(*p2)):
            return False
        if p1 == p2:
            return self.is_point_free(*p1)
        line = LineString([p1, p2])
        return not any(self.polys[i].intersects(line) for i in self._tree.query(line))

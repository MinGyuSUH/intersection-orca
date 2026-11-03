r"""Hybrid-A* global path planner for one AMR (the global planner of LiDAR-ORCA-HybFF).

Search state (x, y, theta). One expansion = an arc of length STEP with one of a few curvatures
(straight, and left/right at 4 radii), forward only. Turning in place at the start is also allowed
(differential drive).

Cost = driving time: arc length / the speed the vehicle can hold on that curvature,
    v(kappa) = min(v_max, w_max / |kappa|, sqrt(a_max / |kappa|), (12 alpha TOL_LAT / |kappa|)^(1/3)),
plus the time lost to slow down and speed up again when the curvature changes. So the planner
trades sharp turns against detours by itself; no turning radius is hand-tuned.

Walls: a pose is free if the gap (body surface to wall) is at least min_gap. The gap is read from a
1 px distance field; where that lookup is ambiguous (within its sqrt(2) px error) the exact segment
distance is used. A soft cost map additionally makes poses close to walls more expensive.

Heuristic: shortest 2D path from the goal (8-connected, around walls), divided by v_max, weighted by
h_weight (weighted A*: cost at most h_weight x optimal). Near the goal, every few expansions the
search tries an analytic Dubins connection that ends exactly on the goal point.

Dolgov, Thrun, Montemerlo, Diebel, "Path Planning for Autonomous Vehicles in Unknown Semi-structured
Environments", IJRR 2010.

The search itself runs in numba (_hybrid_astar_nb.py).
"""
import math

import numpy as np

from . import _hybrid_astar_nb as NB

STEP = 3.0                      # arc length per expansion [px]
TH_BINS = 72                    # heading bins (5 deg)
XY_CELL = 2.0                   # duplicate-detection grid [px]
H_CELL = 3.0                    # heuristic grid [px]
GOAL_TOL = 3.0
SWITCH = 0.02                   # tie-break cost for changing curvature [s] (avoids zigzag)
WALL_EPS = math.sqrt(2.0)       # max error of the 1 px distance-field lookup
TOL_LAT = 2.0                   # allowed lateral error in the angular-acceleration speed limit [px]


def kappas(r_min):
    """Curvature set: straight, and radii r_min, 2 r_min, 4 r_min, 8 r_min to both sides."""
    ks = [0.0]
    for r in (r_min, 2 * r_min, 4 * r_min, 8 * r_min):
        ks += [1.0 / r, -1.0 / r]
    return ks


def speed_of(k, v_max, w_max, a_max, alpha=None):
    """Speed the vehicle can hold on curvature k (same limits the path follower applies)."""
    k = abs(k)
    if k < 1e-9:
        return v_max
    v = min(v_max, w_max / k, math.sqrt(a_max / k))
    if alpha:
        v = min(v, (12.0 * alpha * TOL_LAT / k) ** (1.0 / 3.0))
    return v


class HybridAStar:
    SHOT_DIST = 80.0            # try the analytic connection when the goal is this close

    def __init__(self, field, segs, radius, v_max, w_max, a_max, alpha, bounds, min_gap, h_weight=1.0):
        """field: (distance field, minx, miny) of WallGeometry. segs: (n, 4) exact wall segments.
        min_gap: required gap from the body surface to walls [px], measured on the distance field."""
        self.field = field
        self.segs = np.ascontiguousarray(segs, dtype=float)
        self.radius = float(radius)
        self.min_gap = float(min_gap)
        self.h_weight = float(h_weight)
        self.v_max, self.w_max, self.a_max, self.alpha = v_max, w_max, a_max, alpha
        self.bounds = bounds
        # Sharpest arc: the radius at which the angular velocity limit still allows v_max / 4.
        # Anything sharper is effectively turning in place.
        self.r_min = (v_max / 4.0) / w_max
        self.K = kappas(self.r_min)
        self.vk = [speed_of(k, v_max, w_max, a_max, alpha) for k in self.K]
        self.cost = [STEP / v for v in self.vk]
        self.shot_every = 4     # try the analytic connection every this many expansions
        # Analytic connection radii: r_min .. 300 px, 60 log-spaced values.
        R_HI, N_R = 300.0, 60
        self.shot_ratio = (R_HI / self.r_min) ** (1.0 / (N_R - 1))
        self.shot_grid = [self.r_min * self.shot_ratio ** k for k in range(N_R)]
        self.expansions = 0

    def is_free(self, x, y) -> bool:
        F, fx, fy = self.field
        return bool(NB.free(F, float(fx), float(fy), self.segs, self.radius, self.min_gap, float(x), float(y)))

    def _h_gap(self):
        # The heuristic map also opens cells that might be free once measured exactly (keeps it admissible).
        return self.min_gap - 2 * WALL_EPS

    def _grid_shape(self):
        x0, y0, x1, y1 = self.bounds
        return x0, y0, int((x1 - x0) / H_CELL) + 1, int((y1 - y0) / H_CELL) + 1

    def distance_field(self, goal):
        """Shortest 2D distance to the goal around walls, on the H_CELL grid -> (dist, x0, y0)."""
        F, fx, fy = self.field
        x0, y0, W, H = self._grid_shape()
        return (NB.hfield(F, float(fx), float(fy), float(x0), float(y0), W, H, H_CELL, self._h_gap(),
                          float(goal[0]), float(goal[1])), x0, y0)

    def pivot_seeds(self, th0, v_now=0.0, tol=0.05):
        """Turn-in-place start options: (headings, start costs). Cost = time to stop from v_now plus the
        time to rotate (trapezoidal profile). Headings within tol need no pivot and are skipped."""
        tb = 2 * math.pi / TH_BINS
        ths, gs = [], []
        stop = v_now / (2.0 * self.a_max) if v_now > 1e-9 else 0.0
        for b in range(TH_BINS):
            th = b * tb
            d = abs(math.atan2(math.sin(th - th0), math.cos(th - th0)))
            if d < tol:
                continue
            w, al = self.w_max, self.alpha
            t = d / w + w / al if d >= w * w / al else 2.0 * math.sqrt(d / al)
            ths.append(th)
            gs.append(stop + t)
        return np.asarray(ths, float), np.asarray(gs, float)

    def plan(self, start, goal, cost_map, v_now=0.0, relax=0.0, brake=0.0, max_expand=400000):
        """start (x, y, theta), goal (x, y). The path ends exactly on the goal point; the arrival heading
        is free. cost_map = (C, x0, y0, cell): cost multiplier grid (time x (1 + C)).
        v_now: current speed. brake: current braking distance - a moving vehicle stops going straight
        before it can turn in place, so pivot starts are placed brake px ahead (and dropped if that
        stretch is not clear). relax > 0: the start is already too close to a wall; within relax px of
        the start, poses are allowed as long as the gap does not get 0.5 px worse.
        Returns (path (N, 3), estimated time [s]) or None."""
        F, fx, fy = self.field
        x0, y0, W, H = self._grid_shape()
        C, cx0, cy0, ccell = cost_map
        hd = NB.hfield_hat(F, float(fx), float(fy), float(x0), float(y0), W, H, H_CELL, self._h_gap(),
                           float(goal[0]), float(goal[1]), C, float(cx0), float(cy0), float(ccell))
        seed_th, seed_g = self.pivot_seeds(float(start[2]), v_now)
        bx, by = float(start[0]), float(start[1])
        if brake > 1e-9:
            c, s = math.cos(float(start[2])), math.sin(float(start[2]))
            need = min(self.min_gap, NB.gap(F, float(fx), float(fy), bx, by) - 0.5)
            ok = all(NB.gap(F, float(fx), float(fy), bx + c * u, by + s * u) >= need
                     for u in np.linspace(0.0, brake, max(2, int(math.ceil(brake)) + 1)))
            if not ok:
                seed_th, seed_g = np.empty(0), np.empty(0)
            bx, by = bx + c * brake, by + s * brake
        P, self.expansions, g = NB.search(
            F, float(fx), float(fy), hd, float(x0), float(y0), H_CELL, float(self.v_max),
            float(start[0]), float(start[1]), float(start[2]), float(goal[0]), float(goal[1]),
            np.asarray(self.K, float), np.asarray(self.vk, float), np.asarray(self.cost, float), float(self.a_max),
            SWITCH, self.min_gap, XY_CELL, TH_BINS, STEP, GOAL_TOL, self.h_weight, int(max_expand),
            0.0, False, True, 0.0, int(self.shot_every), float(self.SHOT_DIST),
            np.asarray(self.shot_grid, float), float(self.shot_ratio), float(self.w_max), float(self.alpha),
            True, seed_th, seed_g, C, float(cx0), float(cy0), float(ccell), float(relax), bx, by,
            self.segs, self.radius)
        return (P, g) if len(P) else None

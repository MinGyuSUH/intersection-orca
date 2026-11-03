"""The vehicle: a differential-drive (unicycle) AMR that follows a global path.

State is (x, y, theta, v, omega). Each step the controller picks a desired (v, omega) and the motor
model clamps it to the vehicle limits (acceleration, angular acceleration, max speed, max angular
velocity) and integrates. One step is split in two so that every vehicle plans from the same
snapshot of the world: plan_step() computes the next state, commit_step() applies it.

Steering, two modes:
  pure pursuit (default)  turn toward a look-ahead point on the path (or toward the avoidance target).
  feed-forward (steer_ff) turn at the path curvature (v * kappa) plus a heading correction toward the
                          path (Stanley-like). Used by LiDAR-ORCA-HybFF, whose Hybrid-A* paths are smooth.
"""
import math
from typing import List, Optional, Tuple

import numpy as np
from numba import njit

Point = Tuple[float, float]


class Path:
    """A polyline the vehicle follows. Treated as immutable once created."""

    def __init__(self, waypoints: List[Point]):
        self.waypoints = waypoints
        self.pts = np.asarray(waypoints, dtype=np.float64)          # (n, 2)
        self._cum = None

    @property
    def cum_length(self) -> np.ndarray:
        """Arc length at each waypoint."""
        if self._cum is None:
            self._cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(self.pts, axis=0).T))]
        return self._cum


@njit(cache=True)
def _path_progress(px, py, seg_idx, wpts, lookahead_dist):
    """Project (px, py) onto the path, searching only from segment seg_idx onward (progress never goes
    back). Returns (look-ahead point x, y, remaining distance, index of the nearest segment).
    The remaining distance includes the lateral offset from the path."""
    n = len(wpts)
    if n < 2:
        return wpts[0, 0], wpts[0, 1], 0.0, seg_idx
    best_idx = seg_idx
    bx = wpts[best_idx, 0]
    by = wpts[best_idx, 1]
    best_dist = 1e18
    for i in range(seg_idx, n - 1):
        x0, y0 = wpts[i, 0], wpts[i, 1]
        x1, y1 = wpts[i + 1, 0], wpts[i + 1, 1]
        sx, sy = x1 - x0, y1 - y0
        seg_len2 = sx * sx + sy * sy
        if seg_len2 < 1e-12:
            continue
        t = ((px - x0) * sx + (py - y0) * sy) / seg_len2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        qx, qy = x0 + sx * t, y0 + sy * t
        dx, dy = px - qx, py - qy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            bx, by = qx, qy
    ddx, ddy = px - bx, py - by
    remaining = math.sqrt(ddx * ddx + ddy * ddy)
    lx, ly = bx, by
    budget = lookahead_dist
    reached = False
    cx, cy = bx, by
    idx = best_idx
    while idx < n - 1:
        nx, ny = wpts[idx + 1, 0], wpts[idx + 1, 1]
        sx, sy = nx - cx, ny - cy
        seg_len = math.sqrt(sx * sx + sy * sy)
        if not reached:
            if seg_len >= budget:
                if seg_len > 1e-9:
                    lx = cx + sx / seg_len * budget
                    ly = cy + sy / seg_len * budget
                else:
                    lx, ly = cx, cy
                reached = True
            else:
                budget -= seg_len
        remaining += seg_len
        cx, cy = nx, ny
        idx += 1
    if not reached:
        lx, ly = wpts[-1, 0], wpts[-1, 1]
    return lx, ly, remaining, best_idx


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _angle_diff(a: float, b: float) -> float:
    """a - b wrapped to [-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


class AMR:
    # A new path whose first direction differs from the current heading by more than this is
    # started by stopping (straight) and turning in place, like a real differential-drive AMR.
    PIVOT_ALIGN_TOL = 0.05
    # Feed-forward steering gains.
    FF_K_TH = 2.0       # heading error gain [1/s]
    FF_K_Y = 1.0        # lateral error gain
    FF_DS = 4.0         # arc-length window for path tangent and curvature [px]

    def __init__(self, agent_id: int, pos, heading: float, max_speed: float, length: float, width: float,
                 accel: float, max_angular_velocity: float, max_angular_accel: float, lidar_range: float):
        self.id = agent_id
        self.pos = np.array(pos, dtype=float)
        self.theta = float(heading)
        self.v = 0.0
        self.omega = 0.0
        self.max_speed = max_speed
        self.length = length
        self.width = width
        self.a_max = accel
        self.decel = accel
        self.max_angular_velocity = max_angular_velocity
        self.alpha_max = max_angular_accel
        # Collisions are checked with the circumscribed circle (covers any heading).
        self.radius = math.hypot(length, width) / 2.0
        self.lookahead_min = self.radius * 1.5

        self.path: Optional[Path] = None
        self.goal = None                      # current goal (set by the simulation)
        self.steer_ff = False
        self.distance_traveled = 0.0
        self._seg_idx = 0                     # path progress index (never decreases)
        self._pp_cache = None
        self._pivot_required = False
        self._pending = None                  # next state from plan_step

        # Set by the simulation every step.
        self.steer_target: Optional[np.ndarray] = None     # avoidance direction (point to steer toward)
        self.orca_speed_cap: Optional[float] = None
        self.path_speed_cap: Optional[float] = None        # curvature speed limit of the path (HybFF)
        self.path_caps = None                               # (path, per-waypoint speed limits) (HybFF)

        # LiDAR: neighbors are observed as positions only; their velocity is the difference to the
        # previous observation. Neighbors that are out of sight are forgotten.
        self.lidar_range = lidar_range
        self.seen_pos = {}                    # neighbor id -> position observed last step
        self.visible_ids = set()

    @property
    def heading(self) -> np.ndarray:
        if getattr(self, "_heading_theta", None) != self.theta:
            self._heading_theta = self.theta
            self._heading_cache = np.array([math.cos(self.theta), math.sin(self.theta)])
        return self._heading_cache

    @property
    def braking_distance(self) -> float:
        return (self.v ** 2) / (2 * self.decel)

    def assign_path(self, path: Path):
        self.path = path
        self._seg_idx = 0
        self._pivot_required = False
        if len(path.waypoints) >= 2:
            seg = np.asarray(path.waypoints[1], dtype=float) - np.asarray(path.waypoints[0], dtype=float)
            if float(np.dot(seg, seg)) > 1e-9:
                start_heading = math.atan2(seg[1], seg[0])
                self._pivot_required = abs(_angle_diff(start_heading, self.theta)) >= self.PIVOT_ALIGN_TOL

    def path_progress(self, lookahead_dist: float):
        """-> (look-ahead point, remaining path length). Also advances the progress index."""
        pos_key = (float(self.pos[0]), float(self.pos[1]))
        c = self._pp_cache
        if c is not None and c[0] == lookahead_dist and c[1] == pos_key and c[2] is self.path:
            return c[3]
        lx, ly, remaining, best_idx = _path_progress(self.pos[0], self.pos[1], self._seg_idx,
                                                     self.path.pts, lookahead_dist)
        self._seg_idx = best_idx
        result = (np.array([lx, ly]), remaining)
        self._pp_cache = (lookahead_dist, pos_key, self.path, result)
        return result

    def ff_reference(self):
        """Feed-forward reference -> (desired heading, path curvature), or None if the path is too short.

        Desired heading = path tangent + atan(k * e_y / v): the tangent when on the path, turning back
        toward it (at most 90 deg) when off it. Tangent and curvature are measured over an arc-length
        window (FF_DS), not over single waypoints, so tiny kinks in the path do not fool them."""
        if self.path is None or len(self.path.waypoints) < 2:
            return None
        P = self.path.pts
        cum = self.path.cum_length
        total = float(cum[-1])
        D = self.FF_DS
        if total < D:
            return None
        # Projection of the current position onto the path near the progress index (arc length s0).
        k0 = min(max(self._seg_idx - 1, 0), len(P) - 2)
        k1 = min(k0 + 32, len(P) - 1)
        A, B = P[k0:k1], P[k0 + 1:k1 + 1]
        AB = B - A
        L2 = np.einsum("ij,ij->i", AB, AB)
        u = np.where(L2 > 1e-18, np.einsum("ij,ij->i", self.pos - A, AB) / np.maximum(L2, 1e-18), 0.0)
        u = np.clip(u, 0.0, 1.0)
        Q = A + u[:, None] * AB
        j = int(np.argmin(np.einsum("ij,ij->i", Q - self.pos, Q - self.pos)))
        s0 = float(cum[k0 + j] + u[j] * math.sqrt(L2[j]))

        def at(sv):
            sv = min(max(sv, 0.0), total)
            i = min(int(np.searchsorted(cum, sv, side="right")) - 1, len(P) - 2)
            seg = cum[i + 1] - cum[i]
            f = (sv - cum[i]) / seg if seg > 1e-12 else 0.0
            return P[i] + f * (P[i + 1] - P[i])

        sa = min(s0, total - D)
        pa, pb = at(sa), at(sa + D)
        th_p = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
        # Past the end of the path: head for the end point instead of following the end tangent.
        end = P[-1]
        if s0 >= total - 1e-6 and ((self.pos[0] - end[0]) * math.cos(th_p)
                                   + (self.pos[1] - end[1]) * math.sin(th_p)) > 0.0:
            return math.atan2(end[1] - self.pos[1], end[0] - self.pos[0]), 0.0
        kap = 0.0
        if sa + 2.0 * D <= total:
            pc = at(sa + 2.0 * D)
            kap = _angle_diff(math.atan2(pc[1] - pb[1], pc[0] - pb[0]), th_p) / D
        q = Q[j]
        tx, ty = math.cos(th_p), math.sin(th_p)
        e_y = (q[0] - self.pos[0]) * (-ty) + (q[1] - self.pos[1]) * tx    # + if the path is to the left
        th_d = th_p + math.atan2((self.FF_K_Y / self.FF_K_TH) * e_y, max(self.v, 1.0))
        return th_d, kap

    def _ff_omega(self, steer_target):
        """Feed-forward angular velocity: v * kappa + K * (heading error). The heading to follow is the
        avoidance direction if there is one, else the path reference heading."""
        ref = self.ff_reference()
        if ref is None:
            return None
        th_cmd, kap = ref
        if steer_target is not None:
            dx, dy = steer_target[0] - self.pos[0], steer_target[1] - self.pos[1]
            if dx * dx + dy * dy > 1e-12:
                th_cmd = math.atan2(dy, dx)
        return self.v * kap + self.FF_K_TH * _angle_diff(th_cmd, self.theta)

    def motor_step(self, v_desired, w_desired, dt):
        """Clamp the command to the vehicle limits and integrate one step (heading first, then move)."""
        d_omega = _clamp(w_desired - self.omega, -self.alpha_max * dt, self.alpha_max * dt)
        decel_rate = self.decel if v_desired < self.v else self.a_max
        d_v = _clamp(v_desired - self.v, -decel_rate * dt, self.a_max * dt)
        new_v = _clamp(self.v + d_v, 0.0, self.max_speed)
        new_omega = _clamp(self.omega + d_omega, -self.max_angular_velocity, self.max_angular_velocity)
        new_theta = self.theta + new_omega * dt
        new_x = self.pos[0] + new_v * math.cos(new_theta) * dt
        new_y = self.pos[1] + new_v * math.sin(new_theta) * dt
        return new_x, new_y, new_theta, new_v, new_omega

    def plan_step(self, dt: float, obstacle_gap: float, speed_cap: Optional[float], steer_target=None):
        """Compute the next state without applying it.

        obstacle_gap: free distance to the vehicle ahead (brake so as to stop before it).
        speed_cap:    upper speed limit from the avoidance layer (ORCA, wall, path curvature).
        steer_target: point to steer toward instead of the path look-ahead point (avoidance direction)."""
        if self.path is None:
            self._pending = (self.pos[0], self.pos[1], self.theta, self.v, self.omega)
            return
        obstacle_gap = max(0.0, obstacle_gap)
        lookahead_dist = max(3.0 * self.v, self.lookahead_min)
        lookahead_point, remaining = self.path_progress(lookahead_dist)
        if steer_target is not None:
            lookahead_point = steer_target
        dx = float(lookahead_point[0] - self.pos[0])
        dy = float(lookahead_point[1] - self.pos[1])
        desired_heading = self.theta if dx * dx + dy * dy < 1e-6 else math.atan2(dy, dx)
        alpha = _angle_diff(desired_heading, self.theta)
        # Proportional heading control, independent of v (so it also turns when standing still).
        heading_gain = self.max_angular_velocity / (math.pi / 2.0)
        omega_desired = _clamp(heading_gain * alpha, -self.max_angular_velocity, self.max_angular_velocity)
        if self.steer_ff:
            ff = self._ff_omega(steer_target)
            if ff is not None:
                omega_desired = _clamp(ff, -self.max_angular_velocity, self.max_angular_velocity)

        if self._pivot_required and abs(alpha) >= self.PIVOT_ALIGN_TOL:
            # Turn in place: first stop going straight, then rotate.
            v_desired = 0.0
            if abs(self.v) > 1e-9:
                omega_desired = 0.0
        else:
            self._pivot_required = False
            v_desired = self.max_speed if speed_cap is None else speed_cap
            v_desired = min(v_desired, math.sqrt(2.0 * self.a_max * max(0.0, remaining)))   # stop at the path end
            v_desired = min(v_desired, math.sqrt(2.0 * self.a_max * max(0.0, obstacle_gap)))  # stop before the car ahead

        # Reachability limit: a target inside the turning circle (radius v / w_max) can never be reached
        # and the vehicle would circle forever. Required: v <= w_max * L / (2 sin|alpha|).
        if v_desired > 0.0:
            L = math.hypot(float(lookahead_point[0]) - self.pos[0], float(lookahead_point[1]) - self.pos[1])
            s = abs(math.sin(alpha))
            if L > 1e-9 and s > 1e-6:
                v_desired = min(v_desired, self.max_angular_velocity * L / (2.0 * s))

        # Turning in place uses the time-optimal (trapezoidal) profile: brake the rotation so that it
        # stops exactly at the target angle.
        if v_desired <= 1e-9 and self.v <= 1e-9:
            omega_desired = math.copysign(
                min(self.max_angular_velocity, math.sqrt(2.0 * self.alpha_max * abs(alpha))), alpha)

        self._pending = self.motor_step(v_desired, omega_desired, dt)

    def commit_step(self):
        x, y, theta, v, omega = self._pending
        new_pos = np.array([x, y])
        mx, my = new_pos[0] - self.pos[0], new_pos[1] - self.pos[1]
        self.distance_traveled += math.sqrt(mx * mx + my * my)
        self.pos = new_pos
        self.theta = theta
        self.v = v
        self.omega = omega
        if self.path is not None:
            self.path_progress(0.0)

    def update_lidar_history(self, agents, visible):
        """Remember where each visible neighbor within LiDAR range is now (for next step's velocity estimate)."""
        in_range = set()
        for other in agents:
            if other.id == self.id or other.id not in visible:
                continue
            dx = other.pos[0] - self.pos[0]
            dy = other.pos[1] - self.pos[1]
            if math.sqrt(dx * dx + dy * dy) < self.lidar_range:
                self.seen_pos[other.id] = other.pos.copy()
                in_range.add(other.id)
        for oid in list(self.seen_pos):
            if oid not in in_range:
                del self.seen_pos[oid]

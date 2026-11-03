"""The two simulations.

Simulation (LiDAR-ORCA)
    global path : Lazy Theta* on a grid (straight segments)
    tracking    : pure pursuit
    avoidance   : ORCA against walls and LiDAR-visible neighbors
HybFFSimulation (LiDAR-ORCA-HybFF)
    global path : Hybrid-A* (heading, curvature and acceleration aware)
    tracking    : feed-forward steering + curvature speed limit
    avoidance   : ORCA, plus a braking guard (can I still stop before walls / cars?)
    replanning  : if pushed off the path and not making progress for 10 s

Task: every vehicle drives to a random goal; once it arrives it immediately gets a new one. Goals are
the centers of the map's free grid cells. Arrival = the whole body is inside the goal cell.

One step (dt):
  1. avoidance: ORCA picks a velocity per vehicle -> steering target + speed cap
  2. every vehicle computes its next state from the same snapshot (plan_step)
  3. all states are applied (commit_step), arrivals get new goals
  4. collision check: any body overlap with a wall or another body sets collision_stop
     (nothing is corrected - overlaps are reported, not hidden)

Units: px, s (1 px = 55.7 mm).
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit

from . import hybrid_astar as HYB
from .config import Config
from .lidar import update_visibility
from .orca import brake_turn_slack, lp2, lp3, orca_line, seg_dist
from .theta_star import ThetaStar, auto_resolution
from .vehicle import AMR, Path
from .walls import InflatedMap, WallGeometry, load_map


@dataclass(eq=False)
class Goal:
    x: float
    y: float


@njit(cache=True)
def _contact_brake_gap(px, py, hx, hy, svx, svy, brake_d, accel, o_pos, o_v, o_hx, o_hy, o_id, cr2, self_id):
    """Free distance to the nearest vehicle ahead, as used by the low-level "brake before you touch"
    rule (plan_step stops within this distance). For each neighbor, the smaller of
      - static gap: distance along my heading until my circle (combined radius sqrt(cr2)) touches it,
        minus my braking distance if it is in front of me, and
      - closing gap: the distance a braking vehicle would need given the time to contact at the
        current relative velocity (accel * t^2 / 2).
    A vehicle driving in the same direction ahead of its neighbor (the leader) does not brake for it."""
    gap = np.inf
    n = o_pos.shape[0]
    for k in range(n):
        ox = o_pos[k, 0]; oy = o_pos[k, 1]
        dx = ox - px; dy = oy - py
        dist = math.sqrt(dx * dx + dy * dy)
        cos_angle = (hx * dx + hy * dy) / dist

        ohx = o_hx[k]; ohy = o_hy[k]
        if hx * ohx + hy * ohy > 0.5:                      # same direction: the leader ignores the follower
            shx = hx + ohx; shy = hy + ohy
            along = -dx * shx - dy * shy
            if abs(along) > 0.5:
                is_leader = along > 0.5
            else:
                is_leader = self_id < o_id[k]
            if is_leader:
                continue

        oc_x = px - ox; oc_y = py - oy                     # ray (my heading) vs. neighbor circle
        b_rc = hx * oc_x + hy * oc_y
        c_rc = oc_x * oc_x + oc_y * oc_y - cr2
        if c_rc <= 0.0:
            physical_gap = 0.0 if b_rc <= 0.0 else np.inf
        else:
            disc = b_rc * b_rc - c_rc
            if disc < 0.0:
                physical_gap = np.inf
            else:
                t_rc = -b_rc - math.sqrt(disc)
                physical_gap = t_rc if t_rc >= 0.0 else np.inf
        if physical_gap == np.inf:
            static_gap = np.inf
        else:
            static_gap = physical_gap - (brake_d if cos_angle > 0.0 else 0.0)
            if static_gap < 0.0:
                static_gap = 0.0

        rvx = svx - o_v[k, 0]; rvy = svy - o_v[k, 1]       # time to contact
        rel_px = px - ox; rel_py = py - oy
        a_ttc = rvx * rvx + rvy * rvy
        b_ttc = rel_px * rvx + rel_py * rvy
        c_ttc = rel_px * rel_px + rel_py * rel_py - cr2
        if c_ttc <= 0.0:
            t_contact = 0.0 if b_ttc < 0.0 else np.inf
        elif a_ttc < 1e-9:
            t_contact = np.inf
        else:
            disc2 = b_ttc * b_ttc - a_ttc * c_ttc
            if disc2 < 0.0:
                t_contact = np.inf
            else:
                t_val = (-b_ttc - math.sqrt(disc2)) / a_ttc
                t_contact = t_val if t_val >= 0.0 else np.inf
        closing_gap = np.inf if t_contact == np.inf else accel * t_contact * t_contact / 2.0

        g = static_gap if static_gap < closing_gap else closing_gap
        if g < gap:
            gap = g
    return gap


class Simulation:
    """LiDAR-ORCA: Lazy Theta* + pure pursuit + ORCA."""
    name = "LiDAR-ORCA"

    LIDAR_RANGE = 20.0           # LiDAR range = this x body radius (141 px)
    ORCA_TAU = 3.0               # time horizon for neighbors [s]
    ORCA_TAU_OBST = 2.0          # time horizon for walls [s]
    ORCA_LOOKAHEAD = 30.0        # preferred velocity points at the path this far ahead [px]
    # Keep this far from walls [px]. ORCA's wall constraint is exact for a holonomic robot; ours is a
    # unicycle that can only approximate the chosen velocity, so it needs a small execution budget.
    ORCA_OBST_MARGIN = 0.25
    ORCA_SYM_BREAK = 0.03        # tiny random perturbation of the preferred velocity (breaks perfect symmetry)
    CONTACT_MARGIN = 1.5         # extra distance kept by the low-level brake-before-contact rule [px]
    STEER_FF = False             # feed-forward steering (HybFF) instead of pure pursuit
    ORCA_ZERO_STOP = False       # stop if ORCA's answer is zero velocity
    ORCA_STOP_GUARD = False      # braking guard (HybFF)

    def __init__(self, config: Config):
        self.config = config
        self.dt = config.dt
        self.t = 0.0
        self.map = load_map(config.map_path)
        self.radius = math.hypot(config.amr_length, config.amr_width) / 2.0
        self.margin = 0.5 * self.radius        # extra distance between vehicles (ORCA combined radius)
        self.walls = WallGeometry(self.map, self.radius)
        wall_margin = self.radius + self.margin
        self.plan_map = InflatedMap(self.map, wall_margin)
        self.theta_star = ThetaStar(self.plan_map, auto_resolution(self.map.obstacles, wall_margin),
                                    config.max_speed)
        # Goal candidates: centers of the goal grid cells whose center is free.
        g = self.map.grid
        self.cell = float(g["cell"])
        pts = [((i + 0.5) * self.cell, (j + 0.5) * self.cell) for j in range(int(g["ny"])) for i in range(int(g["nx"]))]
        self.goal_pool = np.array([p for p in pts if float(self.walls.nearest_wall_fast(p)) >= 0.0], float)

        self.rng = np.random.default_rng(config.seed)
        self.agents = []
        self.arrivals = 0
        self.collision_stop = False
        self.collision_stop_reason = ""
        for i in range(config.num_amrs):
            self.add_agent(i)

    # ------------------------------------------------------------------ vehicles and goals

    def add_agent(self, agent_id: Optional[int] = None) -> AMR:
        """Spawn a vehicle on a random goal cell, clear of the others, with a random heading."""
        if agent_id is None:
            agent_id = max((a.id for a in self.agents), default=-1) + 1
        sep = 2.0 * self.radius + self.margin + 2.0
        for _ in range(5000):
            x, y = self.goal_pool[int(self.rng.integers(0, len(self.goal_pool)))]
            if all(math.hypot(a.pos[0] - x, a.pos[1] - y) >= sep for a in self.agents):
                break
        else:
            raise RuntimeError("no free spawn spot")
        heading = float(self.rng.uniform(-math.pi, math.pi))
        goal = self._random_goal()
        c = self.config
        amr = AMR(agent_id, (float(x), float(y)), heading, c.max_speed, c.amr_length, c.amr_width, c.accel,
                  c.max_angular_velocity, c.max_angular_accel, lidar_range=self.radius * self.LIDAR_RANGE)
        amr.steer_ff = self.STEER_FF
        self._plan_to(amr, goal)
        self.agents.append(amr)
        return amr

    def remove_agent(self):
        if self.agents:
            self.agents.pop()

    def _random_goal(self, exclude=None) -> Goal:
        """A random goal cell center; never the cell that contains `exclude` (the current position)."""
        x, y = self.goal_pool[int(self.rng.integers(0, len(self.goal_pool)))]
        while exclude is not None and abs(x - exclude[0]) <= self.cell / 2 and abs(y - exclude[1]) <= self.cell / 2:
            x, y = self.goal_pool[int(self.rng.integers(0, len(self.goal_pool)))]
        return Goal(float(x), float(y))

    def _in_goal_cell(self, amr) -> bool:
        """The whole body (circumscribed circle) is inside the goal cell."""
        if amr.goal is None:
            return False
        h = self.cell / 2.0 - amr.radius
        return abs(amr.pos[0] - amr.goal.x) <= h and abs(amr.pos[1] - amr.goal.y) <= h

    def _plan_to(self, amr, goal: Goal, exclude=None):
        """Plan a global path to `goal`. Five fallback goals are drawn up front and tried in order if the
        goal cannot be reached from here."""
        candidates = [goal] + [self._random_goal(exclude) for _ in range(5)]
        for cand in candidates:
            got = self._global_path(amr, cand)
            if got is None:
                continue
            path, speed_caps = got
            amr.goal = cand
            smooth = self._smooth_kinks(path)
            amr.assign_path(smooth)
            # Curvature speed limits belong to the unmodified path only.
            amr.path_caps = (smooth, speed_caps) if speed_caps is not None and smooth.waypoints == path.waypoints else None
            return
        amr.goal = None

    def _global_path(self, amr, goal: Goal):
        """-> (Path, per-waypoint speed caps or None) or None."""
        path = self.theta_star.plan(tuple(amr.pos), (goal.x, goal.y))
        return None if path is None else (path, None)

    def _smooth_kinks(self, path: Path) -> Path:
        """Drop a waypoint where the path turns by more than 60 deg within a short stretch (< 4.25 R),
        if the shortcut is free."""
        min_runway = 4.25 * self.radius
        wpts = list(path.waypoints)
        changed = True
        while changed and len(wpts) >= 3:
            changed = False
            for i in range(1, len(wpts) - 1):
                p0, p1, p2 = (np.asarray(wpts[k], dtype=float) for k in (i - 1, i, i + 1))
                n01 = float(np.linalg.norm(p1 - p0))
                n12 = float(np.linalg.norm(p2 - p1))
                if n01 < 1e-6 or n12 < 1e-6:
                    continue
                d1, d2 = p1 - p0, p2 - p1
                cosang = float(np.dot(d1, d2) / (math.hypot(d1[0], d1[1]) * math.hypot(d2[0], d2[1])))
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
                if angle <= 60.0 or min(n01, n12) >= min_runway:
                    continue
                if self.plan_map.is_line_free(tuple(p0), tuple(p2)):
                    del wpts[i]
                    changed = True
                    break
        return Path(wpts)

    # ------------------------------------------------------------------ step

    def step(self):
        arrived = [self._in_goal_cell(a) for a in self.agents]    # judged at the start of the step
        self._before_step()
        update_visibility(self.agents, self.walls)
        self._orca_step()
        for amr in self.agents:
            amr.plan_step(self.dt, obstacle_gap=self._contact_gap(amr), speed_cap=self._speed_cap(amr),
                          steer_target=amr.steer_target)
        for amr, done in zip(self.agents, arrived):
            amr.commit_step()
            if done or amr.goal is None:
                self._on_arrival(amr, done)
        self._check_collision()
        self.t += self.dt

    def _before_step(self):
        pass

    def _on_arrival(self, amr, done=True):
        if done:
            self.arrivals += 1
        here = (float(amr.pos[0]), float(amr.pos[1]))
        self._plan_to(amr, self._random_goal(here), exclude=here)

    # ------------------------------------------------------------------ ORCA

    def _orca_step(self):
        """ORCA velocity for every vehicle -> amr.steer_target (direction) and amr.orca_speed_cap (speed)."""
        dt = self.dt
        inv_tau = 1.0 / self.ORCA_TAU
        inv_dt = 1.0 / dt
        P = np.array([np.asarray(a.pos, dtype=float) for a in self.agents])
        for amr in self.agents:
            amr.steer_target = None
            amr.orca_speed_cap = None
            if amr.path is None or amr._pivot_required:
                continue
            pos = np.asarray(amr.pos, dtype=float)
            self_vel = amr.v * np.asarray(amr.heading, dtype=float)

            # Preferred velocity: full speed toward the path look-ahead point (or the feed-forward heading).
            look, _ = amr.path_progress(max(self.ORCA_LOOKAHEAD, 3.0 * amr.v))
            gdir = np.asarray(look, dtype=float) - pos
            gn = float(np.hypot(gdir[0], gdir[1]))
            v_pref = (gdir / gn) * amr.max_speed if gn > 1e-6 else np.zeros(2)
            if amr.steer_ff:
                ref = amr.ff_reference()
                if ref is not None:
                    v_pref = np.array([math.cos(ref[0]), math.sin(ref[0])]) * amr.max_speed
            v_pref = v_pref + (self.rng.random(2) - 0.5) * amr.max_speed * self.ORCA_SYM_BREAK

            # Wall lines first (never relaxed), then one line per visible neighbor.
            lines = self._wall_lines(amr, pos)
            num_walls = len(lines)
            rel_all = P - pos
            d_all = np.hypot(rel_all[:, 0], rel_all[:, 1])
            for k, o in enumerate(self.agents):
                if o.id == amr.id or o.id not in amr.visible_ids:
                    continue
                d = float(d_all[k])
                if d > amr.lidar_range or d < 1e-6:
                    continue
                # LiDAR sees positions only: velocity = change since the last observation (0 if just appeared).
                o_prev = amr.seen_pos.get(o.id, o.pos)
                o_vel = (np.asarray(o.pos, dtype=float) - np.asarray(o_prev, dtype=float)) / dt
                comb_r = amr.radius + o.radius + self.margin
                lines.append(orca_line(rel_all[k], self_vel - o_vel, comb_r, inv_tau, self_vel, inv_dt))

            idx, new_v = lp2(lines, amr.max_speed, v_pref, False)
            if idx < len(lines):
                new_v = lp3(lines, num_walls, max(idx, num_walls), amr.max_speed, new_v)
            speed = float(np.hypot(new_v[0], new_v[1]))
            if speed <= 1e-6 and self.ORCA_ZERO_STOP:
                amr.orca_speed_cap = 0.0
            if speed > 1e-6:
                new_dir = new_v / speed
                if self.ORCA_STOP_GUARD and num_walls:
                    new_dir = self._stop_guard(amr, new_dir)
                amr.steer_target = pos + new_dir * self.ORCA_LOOKAHEAD
                amr.orca_speed_cap = self._exec_consistent_cap(amr, pos, speed)
                if self.ORCA_STOP_GUARD and num_walls:
                    amr.orca_speed_cap = self._stop_guard_speed(amr, new_dir, amr.orca_speed_cap)

        for amr in self.agents:
            amr.update_lidar_history(self.agents, amr.visible_ids)

    def _wall_lines(self, amr, pos):
        """Walls as ORCA lines: "do not move toward this wall faster than (gap - margin) / tau_obst".
        Three walls are used: the nearest one, the one opposite to it, and the one straight ahead."""
        inv_tau = 1.0 / self.ORCA_TAU_OBST
        reach = amr.max_speed * self.ORCA_TAU_OBST
        r = amr.radius
        p = (float(pos[0]), float(pos[1]))
        lines = []

        def add(dist, wn):
            wn = np.asarray(wn, dtype=float)
            c = -max(dist - self.ORCA_OBST_MARGIN, 0.0) * inv_tau
            lines.append((wn * c, np.array([wn[1], -wn[0]])))

        wd, wn = self.walls.nearest_wall(p)          # gap and normal pointing away from the wall
        if wn is None:
            return lines
        wn = np.asarray(wn, dtype=float)
        if wd < reach:
            add(wd, wn)

        def ray_add(direction):
            hit, nrm = self.walls.raycast(p, direction, reach + 2.0 * r + 5.0)
            if nrm is None:
                return
            dd = float(np.hypot(hit[0] - pos[0], hit[1] - pos[1]))    # = body-surface gap (ray hits the inflated wall)
            if dd >= reach:
                return
            n = np.asarray(nrm, dtype=float)
            if float(n[0] * direction[0] + n[1] * direction[1]) > 0.0:
                n = -n
            if float(n[0] * wn[0] + n[1] * wn[1]) > 0.95 and abs(dd - wd) < 2.0:
                return                                       # same wall as the nearest one
            add(dd, n)

        ray_add(wn)                                          # opposite wall
        if amr.v > 0.3:
            ray_add(np.asarray(amr.heading, dtype=float))    # wall ahead
        return lines

    def _exec_consistent_cap(self, amr, pos, cap):
        """Apply ORCA's nearest-wall constraint to the velocity that is actually executed. ORCA constrains
        a holonomic velocity vector, but the unicycle moves along its heading; slow down by how much the
        heading points into the wall (no limit when driving parallel to it)."""
        gap, wn = self.walls.nearest_wall((float(pos[0]), float(pos[1])))
        if wn is None:
            return cap
        inward = -float(np.asarray(amr.heading, dtype=float) @ np.asarray(wn, dtype=float))
        if inward <= 1e-6:
            return cap
        budget = max(gap - self.ORCA_OBST_MARGIN, 0.0) / self.ORCA_TAU_OBST
        return min(cap, budget / inward)

    # ------------------------------------------------------------------ speed limits

    def _wall_speed_cap(self, amr) -> Optional[float]:
        """Braking-distance limit toward the nearest wall: the part of the braking distance that points
        into the wall must fit in the gap (minus margin): v <= sqrt(2 a (gap - margin) / inward)."""
        v_cmd = amr.max_speed
        gap, wn = self.walls.nearest_wall((float(amr.pos[0]), float(amr.pos[1])))
        if wn is None:
            return None
        heading = amr.heading
        inward = -(heading[0] * wn[0] + heading[1] * wn[1])
        if inward <= 1e-6:
            return v_cmd
        h = gap - self.ORCA_OBST_MARGIN
        if h <= 0.0:
            return 0.0
        lim = math.sqrt(2.0 * amr.decel * h / inward)
        return v_cmd if lim >= v_cmd else lim

    def _speed_cap(self, amr) -> Optional[float]:
        cap = self._wall_speed_cap(amr)
        for extra in (amr.path_speed_cap, amr.orca_speed_cap):
            if extra is not None:
                cap = extra if cap is None else min(cap, extra)
        return cap

    def _contact_gap(self, amr) -> float:
        others = [o for o in self.agents if o.id != amr.id]
        if not others:
            return float("inf")
        o_pos = np.array([o.pos for o in others], dtype=np.float64)
        o_h = np.array([o.heading for o in others], dtype=np.float64)
        o_v = o_h * np.array([[o.v] for o in others], dtype=np.float64)
        diffs = o_pos - amr.pos
        dists = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        if dists[int(np.argmin(dists))] < 1e-6:
            return 0.0
        combined = amr.radius + self.radius + self.CONTACT_MARGIN
        h = amr.heading
        sv = amr.v * h
        return _contact_brake_gap(float(amr.pos[0]), float(amr.pos[1]), float(h[0]), float(h[1]),
                                  float(sv[0]), float(sv[1]), float(amr.braking_distance), float(self.config.accel),
                                  o_pos, o_v, np.ascontiguousarray(o_h[:, 0]), np.ascontiguousarray(o_h[:, 1]),
                                  np.array([o.id for o in others], dtype=np.int64), combined * combined, int(amr.id))

    # ------------------------------------------------------------------ braking guard (HybFF)

    def _brake_slack_fn(self, amr, cars=False):
        """-> brake_slack(phi, v1): the smallest margin while turning toward heading phi with speed command
        v1 for one step and then braking to a stop (>= 0 is safe). Walls must keep min(margin, current gap).
        With cars=True, visible neighbors within 5 combined radii count too: an approaching car must stay
        half the current gap away (both keep half), a standing or crossing car min(margin, gap); cars
        driving away are ignored. Neighbors are taken at their current position."""
        p = amr.pos
        S = self.walls.segs
        R = float(amr.radius)
        thr = min(self.ORCA_OBST_MARGIN, seg_dist(S, float(p[0]), float(p[1])) - R)
        ff = amr.ff_reference() if amr.steer_ff else None
        kap, K = (float(ff[1]), float(amr.FF_K_TH)) if ff is not None else (0.0, amr.max_angular_velocity / (math.pi / 2.0))
        state = (float(amr.theta), abs(float(amr.v)), float(amr.omega))
        lim = (kap, K, float(amr.max_angular_velocity), float(amr.alpha_max), float(amr.a_max), float(amr.decel),
               float(self.dt), R)
        O, req = [], []
        if cars:
            v = abs(float(amr.v))
            for o in self.agents:
                if o.id == amr.id or o.id not in amr.visible_ids:
                    continue
                rel = np.asarray(o.pos, dtype=float) - np.asarray(p, dtype=float)
                dist = float(np.hypot(rel[0], rel[1]))
                if dist < 1e-6 or dist > 5.0 * (R + float(o.radius)):
                    continue
                toward = -float(o.v) * float(np.asarray(o.heading, dtype=float) @ rel) / dist
                if toward < -0.5 * v:
                    continue
                gap0 = dist - R - float(o.radius)
                keep = 0.5 * gap0 if toward > 1e-6 else min(self.ORCA_OBST_MARGIN, gap0)
                O.append((float(o.pos[0]), float(o.pos[1])))
                req.append(R + float(o.radius) + keep)
        O = np.asarray(O, dtype=float).reshape(-1, 2)
        req = np.asarray(req, dtype=float)

        def brake_slack(phi, v1):
            return brake_turn_slack(S, float(p[0]), float(p[1]), state[0], state[1], state[2], phi, *lim, v1, thr, O, req)
        return brake_slack

    def _stop_guard(self, amr, new_dir):
        """Direction part of the braking guard. Candidate directions lie on the arc the vehicle would
        actually turn through (current heading -> ORCA direction). Walls must stay stoppable along the
        whole turn; among those, prefer the one that violates neighbor clearance least, then the one
        closest to ORCA's direction."""
        v = abs(float(amr.v))
        if v < 1e-3:
            return new_dir
        k = 2.0 * float(amr.a_max) / (v * v)
        p = np.asarray(amr.pos, dtype=float)
        brake_slack = self._brake_slack_fn(amr)

        def wall_over(d):
            return max(-brake_slack(math.atan2(d[1], d[0]), 0.0), 0.0)

        planes = []          # (outward normal, allowed inward component) per approaching visible neighbor
        for o in self.agents:
            if o.id == amr.id or o.id not in amr.visible_ids:
                continue
            rel = np.asarray(o.pos, dtype=float) - p
            dist = float(np.hypot(rel[0], rel[1]))
            if dist < 1e-6:
                continue
            u = rel / dist
            if float(o.v) * float(np.asarray(o.heading, dtype=float) @ u) > 0.5 * v:
                continue                                     # driving away
            lim = k * max(dist - amr.radius - o.radius - self.ORCA_OBST_MARGIN, 0.0) * 0.5
            if lim < 1.0:
                planes.append((-u, lim))

        def car_over(d):
            return max((max(-(d[0] * n[0] + d[1] * n[1]) - lim, 0.0) for n, lim in planes), default=0.0)

        if wall_over(new_dir) == 0.0 and car_over(new_dir) == 0.0:
            return new_dir
        h = np.asarray(amr.heading, dtype=float)
        a_h = math.atan2(h[1], h[0])
        a_n = math.atan2(new_dir[1], new_dir[0])
        delta = (a_n - a_h + math.pi) % (2 * math.pi) - math.pi
        arc = [a_h + delta * i / 20.0 for i in range(21)]
        ok_arc = []
        for ang in arc:                                      # stop at the first direction that breaks a wall
            d = np.array([math.cos(ang), math.sin(ang)])
            if wall_over(d) > 1e-9:
                break
            ok_arc.append((ang, d))
        if ok_arc:
            best = None
            for ang, d in ok_arc:
                kk = (car_over(d), abs((ang - a_n + math.pi) % (2 * math.pi) - math.pi))
                if best is None or kk < best[0]:
                    best = (kk, d)
            return best[1]
        # Already heading into a wall: least wall violation near the heading and along the turn.
        best = None
        for ang in arc + [a_h + math.radians(5.0 * i) for i in range(-6, 7)]:
            d = np.array([math.cos(ang), math.sin(ang)])
            kk = (wall_over(d), car_over(d), abs((ang - a_n + math.pi) % (2 * math.pi) - math.pi))
            if best is None or kk < best[0]:
                best = (kk, d)
        return best[1]

    def _stop_guard_speed(self, amr, new_dir, cap):
        """Speed part of the braking guard: the fastest command (accelerate, else hold) after which the
        vehicle can still turn to new_dir and stop safely (walls and nearby cars). Else 0."""
        v = abs(float(amr.v))
        v_up = min(cap, v + float(amr.a_max) * self.dt)
        brake_slack = self._brake_slack_fn(amr, cars=True)
        phi = math.atan2(new_dir[1], new_dir[0])
        for v1 in (v_up, min(v, cap)):
            if brake_slack(phi, v1) >= 0.0:
                return cap if v1 == v_up else v1
        return 0.0

    # ------------------------------------------------------------------ collision check

    def _check_collision(self):
        """Report (do not fix) any body overlap: sets collision_stop and the reason."""
        for amr in self.agents:
            p = (float(amr.pos[0]), float(amr.pos[1]))
            if self.walls.nearest_wall_fast(p) >= 3.0:       # far from walls (lookup error is ~1 px)
                continue
            gap = self.walls.nearest_wall(p)[0]
            if gap < -1e-6:
                self.collision_stop = True
                self.collision_stop_reason = f"AMR{amr.id} hit wall ({gap:.2f}px)"
                return
        for i, a in enumerate(self.agents):
            for b in self.agents[i + 1:]:
                dist = math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1])
                if dist < a.radius + b.radius - 1e-6:
                    self.collision_stop = True
                    self.collision_stop_reason = f"AMR{a.id}-AMR{b.id} crash ({dist - a.radius - b.radius:.2f}px)"
                    return


def _path_speed_caps(P, w_max, a_max, v_max):
    """Speed limit at each path point: v <= w_max / kappa, v <= sqrt(a_max / kappa), and braking ahead
    of slower points: v_i <= sqrt(v_{i+1}^2 + 2 a d)."""
    n = len(P)
    seg = np.r_[np.hypot(*np.diff(P, axis=0).T), 0.0]
    v = np.full(n, float(v_max))
    for i in range(1, n - 1):
        a, b = P[i] - P[i - 1], P[i + 1] - P[i]
        la, lb = float(np.hypot(*a)), float(np.hypot(*b))
        if la < 1e-9 or lb < 1e-9:
            continue
        kap = abs(math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))) / (0.5 * (la + lb))
        if kap > 1e-6:
            v[i] = min(v_max, w_max / kap, math.sqrt(a_max / kap))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_max * seg[i]))
    return v


class HybFFSimulation(Simulation):
    """LiDAR-ORCA-HybFF: Hybrid-A* + feed-forward steering + curvature speed limit + ORCA + braking guard.
    The path is planned once per goal; it is replanned only if the vehicle has been pushed off it and
    has not made progress for REPLAN_T seconds."""
    name = "LiDAR-ORCA-HybFF"
    STEER_FF = True
    ORCA_ZERO_STOP = True
    ORCA_STOP_GUARD = True

    H_WEIGHT = 1.1               # weighted A*: path cost at most 1.1 x optimal
    MAX_EXPAND = 400000
    SOFT_K = 2.0                 # wall-proximity cost: x(1 + (k - 1)(1 - gap / D0)^2) for gap < D0
    SOFT_D0 = 4.9                # [px]
    REPLAN_T = 10.0              # off the path without progress for this long -> replan [s]
    REPLAN_PROG = 5.0            # progress = remaining distance shrank by this much [px]

    def __init__(self, config: Config):
        self._hyb = None             # built on first use (needs self.walls)
        self._off_since = {}         # agent id -> (time, remaining distance, goal) when it left its path
        super().__init__(config)

    def _planner(self) -> HYB.HybridAStar:
        if self._hyb is None:
            w, c = self.walls, self.config
            # Hard wall limit = ORCA's wall margin (the planner must not allow poses the executor refuses);
            # the soft cost map keeps paths away from walls where there is room.
            self._hyb = HYB.HybridAStar((w.dist_field, w.df_minx, w.df_miny), w.segs, w.radius, c.max_speed,
                                        c.max_angular_velocity, c.accel, c.max_angular_accel, self.map.bounds,
                                        min_gap=self.ORCA_OBST_MARGIN + HYB.WALL_EPS, h_weight=self.H_WEIGHT)
            F, D0 = w.dist_field, self.SOFT_D0
            C = np.where(F < D0, (self.SOFT_K - 1.0) * (1.0 - np.clip(F, 0.0, D0) / D0) ** 2, 0.0)
            self._cost_map = (C, float(w.df_minx), float(w.df_miny), 1.0)
        return self._hyb

    def _global_path(self, amr, goal: Goal):
        """Hybrid-A* from the current pose; Theta* if it fails."""
        pl = self._planner()
        g = (float(goal.x), float(goal.y))
        if pl.is_free(*g):
            start = (float(amr.pos[0]), float(amr.pos[1]), float(amr.theta))
            brake = amr.braking_distance
            # Start already too close to a wall (pushed there by avoidance): allow leaving it.
            relax = 0.0 if pl.is_free(start[0], start[1]) else 2.0 * amr.radius + brake
            got = pl.plan(start, g, self._cost_map, v_now=float(abs(amr.v)), relax=relax, brake=brake,
                          max_expand=self.MAX_EXPAND)
            if got is not None:
                # Resample at 3 px (for curvature and progress) and end exactly on the goal.
                P = got[0][:, :2]
                s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
                ss = np.arange(0.0, s[-1], HYB.STEP)
                Q = np.vstack([np.c_[np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])], g])
                caps = _path_speed_caps(Q, amr.max_angular_velocity, amr.a_max, amr.max_speed)
                return Path([(float(x), float(y)) for x, y in Q]), caps
        return super()._global_path(amr, goal)

    def _before_step(self):
        # Curvature speed limit at the current path position.
        for a in self.agents:
            got = a.path_caps
            if got is None or got[0] is not a.path:
                a.path_speed_cap = None
                continue
            vv = got[1]
            k = min(a._seg_idx, len(vv) - 1)
            cap = float(min(vv[k:k + 2]))
            a.path_speed_cap = cap if cap < a.max_speed - 1e-6 else None
        self._replan_off_path()

    def _replan_off_path(self):
        """Replan from the current pose if the vehicle is more than 2R off its path and its remaining
        distance to the goal (around walls) has not shrunk by REPLAN_PROG within REPLAN_T seconds.
        A vehicle stuck on its path (e.g. waiting for others) is not off the path and is left alone."""
        for a in self.agents:
            p, goal = a.path, a.goal
            if p is None or goal is None or a._pivot_required:
                self._off_since.pop(a.id, None)
                continue
            P = p.pts
            k0 = max(int(a._seg_idx) - 1, 0)
            if len(P) - k0 < 2:
                self._off_since.pop(a.id, None)
                continue
            A, B = P[k0:-1], P[k0 + 1:]
            D = B - A
            L2 = np.maximum((D * D).sum(axis=1), 1e-12)
            q = np.asarray(a.pos, dtype=float)
            t = np.clip(((q - A) * D).sum(axis=1) / L2, 0.0, 1.0)
            if float(np.sqrt((((A + D * t[:, None]) - q) ** 2).sum(axis=1).min())) <= 2.0 * a.radius:
                self._off_since.pop(a.id, None)
                continue
            rem = self._remaining_distance(a, goal)
            st = self._off_since.get(a.id)
            if st is None or st[2] is not goal or rem < st[1] - self.REPLAN_PROG:
                self._off_since[a.id] = (self.t, rem, goal)
                continue
            if self.t - st[0] < self.REPLAN_T:
                continue
            del self._off_since[a.id]
            self._plan_to(a, goal)

    def _remaining_distance(self, a, goal) -> float:
        """Distance to the goal around walls, from the planner's heuristic map (one per goal)."""
        cached = getattr(a, "_goal_field", None)
        if cached is None or cached[0] is not goal:
            key = (round(float(goal.x), 1), round(float(goal.y), 1))
            cached = a._goal_field = (goal, self._planner().distance_field(key))
        dist, x0, y0 = cached[1]
        i, j = int(round((a.pos[0] - x0) / HYB.H_CELL)), int(round((a.pos[1] - y0) / HYB.H_CELL))
        if 0 <= i < dist.shape[0] and 0 <= j < dist.shape[1] and np.isfinite(dist[i, j]):
            return float(dist[i, j])
        return float(np.hypot(a.pos[0] - goal.x, a.pos[1] - goal.y))


ALGORITHMS = {cls.name: cls for cls in (HybFFSimulation, Simulation)}

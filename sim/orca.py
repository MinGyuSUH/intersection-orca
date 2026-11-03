"""ORCA (Optimal Reciprocal Collision Avoidance), ported from RVO2.

van den Berg, Guy, Lin, Manocha, "Reciprocal n-Body Collision Avoidance", ISRR 2011.

Every neighbor (and every nearby wall) becomes a half-plane constraint in velocity space:
  - neighbor: the velocities that would hit it within tau seconds (a truncated cone, the velocity
    obstacle) are cut off by a line; each side takes half of the avoidance effort (reciprocal).
  - wall: "do not move toward this wall faster than (gap - margin) / tau_obst".
A 2D linear program then picks the velocity closest to the preferred velocity that satisfies all
half-planes and the speed limit. If no such velocity exists (dense traffic), the neighbor
constraints are relaxed to minimize the worst penetration (RVO2 linearProgram3). Wall lines are
never relaxed.

A line is (point, direction); the allowed side is to the left of the direction.

This module also holds _brake_turn_slack, used by the braking guard of LiDAR-ORCA-HybFF
(see Simulation._stop_guard in simulation.py).
"""
import math

import numpy as np
from numba import njit


def _det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def orca_line(rel_pos, rel_vel, comb_r, inv_tau, self_vel, inv_dt, share=0.5):
    """ORCA half-plane for one neighbor (RVO2 Agent::computeNewVelocity).
    rel_pos = neighbor - me, rel_vel = my velocity - neighbor velocity, comb_r = sum of radii (+ margin).
    share = my part of the avoidance (0.5 = reciprocal)."""
    dist_sq = float(rel_pos[0] ** 2 + rel_pos[1] ** 2)
    r_sq = comb_r * comb_r
    if dist_sq > r_sq:
        # Not colliding yet: velocity obstacle truncated at time horizon tau.
        w = rel_vel - inv_tau * rel_pos
        w_len_sq = float(w[0] ** 2 + w[1] ** 2)
        dot1 = float(w[0] * rel_pos[0] + w[1] * rel_pos[1])
        if dot1 < 0.0 and dot1 * dot1 > r_sq * w_len_sq:
            # Project on the cut-off circle.
            w_len = math.sqrt(w_len_sq)
            unit_w = w / w_len if w_len > 1e-9 else np.array([0.0, 0.0])
            direction = np.array([unit_w[1], -unit_w[0]])
            u = (comb_r * inv_tau - w_len) * unit_w
        else:
            # Project on a leg of the cone.
            leg = math.sqrt(max(dist_sq - r_sq, 0.0))
            if _det(rel_pos, w) > 0.0:      # left leg
                direction = np.array([rel_pos[0] * leg - rel_pos[1] * comb_r,
                                      rel_pos[0] * comb_r + rel_pos[1] * leg]) / dist_sq
            else:                           # right leg
                direction = -np.array([rel_pos[0] * leg + rel_pos[1] * comb_r,
                                       -rel_pos[0] * comb_r + rel_pos[1] * leg]) / dist_sq
            dot2 = float(rel_vel[0] * direction[0] + rel_vel[1] * direction[1])
            u = dot2 * direction - rel_vel
    else:
        # Already overlapping: resolve within one time step.
        w = rel_vel - inv_dt * rel_pos
        w_len = float(math.hypot(w[0], w[1]))
        unit_w = w / w_len if w_len > 1e-9 else np.array([0.0, 0.0])
        direction = np.array([unit_w[1], -unit_w[0]])
        u = (comb_r * inv_dt - w_len) * unit_w
    point = self_vel + share * u
    return point, direction


def _lp1(lines, i, max_speed, opt_vel, dir_opt, result):
    """Best velocity on line i subject to lines 0..i-1 and the speed circle. (ok, velocity)."""
    line_pt, line_dir = lines[i]
    dot = float(line_pt[0] * line_dir[0] + line_pt[1] * line_dir[1])
    disc = dot * dot + max_speed * max_speed - float(line_pt[0] ** 2 + line_pt[1] ** 2)
    if disc < 0.0:
        return False, result
    sq = math.sqrt(disc)
    t_left = -dot - sq
    t_right = -dot + sq
    for j in range(i):
        pj, dj = lines[j]
        denom = _det(line_dir, dj)
        numer = _det(dj, line_pt - pj)
        if abs(denom) <= 1e-9:
            if numer < 0.0:
                return False, result
            continue
        t = numer / denom
        if denom >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, result
    if dir_opt:
        t = t_right if float(opt_vel[0] * line_dir[0] + opt_vel[1] * line_dir[1]) > 0.0 else t_left
        res = line_pt + t * line_dir
    else:
        t = float((opt_vel - line_pt)[0] * line_dir[0] + (opt_vel - line_pt)[1] * line_dir[1])
        t = min(max(t, t_left), t_right)
        res = line_pt + t * line_dir
    return True, res


def lp2(lines, max_speed, opt_vel, dir_opt):
    """Velocity closest to opt_vel inside the speed circle and all half-planes.
    Returns (len(lines), velocity) on success, or (index of the failing line, best so far)."""
    if dir_opt:
        result = opt_vel * max_speed
    elif float(opt_vel[0] ** 2 + opt_vel[1] ** 2) > max_speed * max_speed:
        result = opt_vel / math.sqrt(opt_vel[0] * opt_vel[0] + opt_vel[1] * opt_vel[1]) * max_speed
    else:
        result = opt_vel.copy()
    for i in range(len(lines)):
        pt, dr = lines[i]
        if _det(dr, pt - result) > 0.0:      # result is outside this half-plane
            ok, res = _lp1(lines, i, max_speed, opt_vel, dir_opt, result)
            if not ok:
                return i, result
            result = res
    return len(lines), result


def lp3(lines, num_fixed, begin, max_speed, result):
    """Infeasible case: minimize the largest violation of lines[num_fixed:] while keeping
    lines[:num_fixed] (walls) satisfied (RVO2 linearProgram3)."""
    distance = 0.0
    for i in range(begin, len(lines)):
        pi, di = lines[i]
        if _det(di, pi - result) > distance:
            proj_lines = list(lines[:num_fixed])
            for j in range(num_fixed, i):
                pj, dj = lines[j]
                denom = _det(di, dj)
                if abs(denom) <= 1e-9:
                    if float(di[0] * dj[0] + di[1] * dj[1]) > 0.0:
                        continue
                    line_pt = 0.5 * (pi + pj)
                else:
                    line_pt = pi + (_det(dj, pi - pj) / denom) * di
                d = dj - di
                n = math.sqrt(d[0] * d[0] + d[1] * d[1])
                line_dir = d / n if n > 1e-9 else np.array([0.0, 0.0])
                proj_lines.append((line_pt, line_dir))
            opt = np.array([-di[1], di[0]])
            idx, new_res = lp2(proj_lines, max_speed, opt, True)
            if idx == len(proj_lines):
                result = new_res
            distance = _det(di, pi - result)
    return result


@njit(cache=True)
def seg_dist(S, x, y):
    """Exact distance from (x, y) to the nearest wall segment (S: (n, 4) = ax, ay, bx, by)."""
    best = 1e18
    for k in range(S.shape[0]):
        ax, ay, bx, by = S[k, 0], S[k, 1], S[k, 2], S[k, 3]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-12 else ((x - ax) * dx + (y - ay) * dy) / L2
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        qx, qy = ax + t * dx - x, ay + t * dy - y
        d = qx * qx + qy * qy
        if d < best:
            best = d
    return math.sqrt(best)


@njit(cache=True)
def brake_turn_slack(S, x, y, th, v, w, phi, kap, K, wmax, amax, acc, decel, dt, R, v1, thr, O, req):
    """Simulate turning toward heading phi with speed command v1 for the first step, then braking at
    full deceleration until stopped, with the real motor model (angular accel -> heading -> speed -> move).
    Returns the smallest margin along the way:
        min(body-wall gap - thr, distance to neighbor j - req[j]).
    >= 0 means the vehicle can still stop safely. Steering = v * kap + K * (heading error)."""
    worst = 1e18
    first = True
    while first or v > 0.0:
        al = (phi - th + math.pi) % (2.0 * math.pi) - math.pi
        wd = v * kap + K * al
        wd = -wmax if wd < -wmax else (wmax if wd > wmax else wd)
        dw = wd - w
        lim = amax * dt
        dw = -lim if dw < -lim else (lim if dw > lim else dw)
        w = w + dw
        w = -wmax if w < -wmax else (wmax if w > wmax else w)
        th = th + w * dt
        if first and v1 > v:
            v = min(v + acc * dt, v1)
        else:
            v = max(v - decel * dt, v1 if first else 0.0)
        first = False
        x = x + v * math.cos(th) * dt
        y = y + v * math.sin(th) * dt
        g = seg_dist(S, x, y) - R - thr
        for j in range(O.shape[0]):
            dx, dy = x - O[j, 0], y - O[j, 1]
            gj = math.sqrt(dx * dx + dy * dy) - req[j]
            if gj < g:
                g = gj
        if g < worst:
            worst = g
    return worst

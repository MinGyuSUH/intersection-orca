"""LiDAR visibility (occlusion).

A real LiDAR only sees what its beams hit first: a neighbor hidden behind a wall or behind another
vehicle is not seen, even if it is within range. A neighbor counts as visible if the line of sight
to its center or to any of its 4 corners is free. Vehicles are rotated rectangles (length x width),
not circles, so they do not hide more than their real body.
"""
import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _seg_hit(p0x, p0y, p1x, p1y, q0x, q0y, q1x, q1y):
    """Do segments p0-p1 and q0-q1 intersect? Parallel segments count as no."""
    rx = p1x - p0x
    ry = p1y - p0y
    sx = q1x - q0x
    sy = q1y - q0y
    den = rx * sy - ry * sx
    if den == 0.0:
        return False
    cx = q0x - p0x
    cy = q0y - p0y
    t = (cx * sy - cy * sx) / den
    if t < 0.0 or t > 1.0:
        return False
    u = (cx * ry - cy * rx) / den
    return 0.0 <= u <= 1.0


@njit(cache=True)
def _visible_from(sx, sy, si, px, py, corn, wax, way, wbx, wby, rng):
    """Which vehicles sensor si (at sx, sy) can see. corn: (n, 4, 2) rectangle corners.
    wax..wby: wall segments. Returns (n,) bool."""
    n = px.shape[0]
    out = np.zeros(n, np.bool_)
    nw = wax.shape[0]
    for j in range(n):
        if j == si:
            continue
        dx = px[j] - sx
        dy = py[j] - sy
        if dx * dx + dy * dy > rng * rng:
            continue
        for m in range(5):
            if m == 0:
                tx = px[j]
                ty = py[j]
            else:
                tx = corn[j, m - 1, 0]
                ty = corn[j, m - 1, 1]
            blocked = False
            # (1) other vehicles first: few of them, and they block most often
            for k in range(n):
                if k == si or k == j:
                    continue
                for e in range(4):
                    e2 = 0 if e == 3 else e + 1
                    if _seg_hit(sx, sy, tx, ty,
                                corn[k, e, 0], corn[k, e, 1],
                                corn[k, e2, 0], corn[k, e2, 1]):
                        blocked = True
                        break
                if blocked:
                    break
            # (2) walls, with a bounding-box prefilter
            if not blocked:
                lox = sx if sx < tx else tx
                hix = tx if sx < tx else sx
                loy = sy if sy < ty else ty
                hiy = ty if sy < ty else sy
                for w in range(nw):
                    wa_x = wax[w]
                    wb_x = wbx[w]
                    if (wa_x < lox and wb_x < lox) or (wa_x > hix and wb_x > hix):
                        continue
                    wa_y = way[w]
                    wb_y = wby[w]
                    if (wa_y < loy and wb_y < loy) or (wa_y > hiy and wb_y > hiy):
                        continue
                    if _seg_hit(sx, sy, tx, ty, wa_x, wa_y, wb_x, wb_y):
                        blocked = True
                        break
            if not blocked:
                out[j] = True
                break
    return out


@njit(cache=True)
def _rect_corners(px, py, ct, st, half_l, half_w):
    """Positions, headings (cos, sin) and half sizes -> (n, 4, 2) rectangle corners."""
    n = px.shape[0]
    out = np.empty((n, 4, 2), np.float64)
    for i in range(n):
        c = ct[i]
        s = st[i]
        fx = c * half_l[i]
        fy = s * half_l[i]
        lx = -s * half_w[i]
        ly = c * half_w[i]
        out[i, 0, 0] = px[i] + fx + lx
        out[i, 0, 1] = py[i] + fy + ly
        out[i, 1, 0] = px[i] + fx - lx
        out[i, 1, 1] = py[i] + fy - ly
        out[i, 2, 0] = px[i] - fx - lx
        out[i, 2, 1] = py[i] - fy - ly
        out[i, 3, 0] = px[i] - fx + lx
        out[i, 3, 1] = py[i] - fy + ly
    return out


def update_visibility(agents, walls):
    """Set agent.visible_ids for every agent (ids of the neighbors its LiDAR sees right now)."""
    n = len(agents)
    px = np.array([a.pos[0] for a in agents], dtype=np.float64)
    py = np.array([a.pos[1] for a in agents], dtype=np.float64)
    ct, st = np.empty(n), np.empty(n)
    hl, hw = np.empty(n), np.empty(n)
    for i, a in enumerate(agents):
        h = a.heading
        ct[i] = h[0]
        st[i] = h[1]
        hl[i] = a.length * 0.5
        hw[i] = a.width * 0.5
    corn = _rect_corners(px, py, ct, st, hl, hw)
    masks = [_visible_from(px[i], py[i], i, px, py, corn, walls.ax, walls.ay, walls.bx, walls.by,
                           float(a.lidar_range)) for i, a in enumerate(agents)]
    for i, a in enumerate(agents):
        a.visible_ids = {o.id for k, o in enumerate(agents) if masks[i][k]}

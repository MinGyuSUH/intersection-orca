"""numba kernels for hybrid_astar.py: wall checks, heuristic maps, analytic Dubins connection, search."""
import heapq
import math

import numpy as np
from numba import njit, types
from numba.typed import Dict

_OFF = 1 << 20
_BIG = 1 << 21
_EPS = math.sqrt(2.0)          # max error of the 1 px distance-field lookup


@njit(cache=True)
def gap(F, fminx, fminy, x, y):
    """Approximate body-to-wall gap from the distance field F (same rule as WallGeometry.nearest_wall_fast)."""
    h, w = F.shape
    col = int(x - fminx)
    row = int(y - fminy)
    col = max(0, min(col, w - 1))
    row = max(0, min(row, h - 1))
    return float(F[row, col])


@njit(cache=True)
def exact_gap(S, R, x, y):
    """Exact body-to-wall gap: distance to the nearest wall segment (S: (n, 4) ax, ay, bx, by) minus R."""
    best = np.inf
    for i in range(S.shape[0]):
        ax, ay, bx, by = S[i, 0], S[i, 1], S[i, 2], S[i, 3]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
        ex, ey = ax + t * dx - x, ay + t * dy - y
        d2 = ex * ex + ey * ey
        if d2 < best:
            best = d2
    return math.sqrt(best) - R


@njit(cache=True)
def free(F, fminx, fminy, S, R, min_gap, x, y):
    """Is the pose free? Lookup >= min_gap passes (guarantees real gap >= min_gap - sqrt2). Where the
    lookup is ambiguous (>= min_gap - 2 sqrt2) the exact distance decides with the same guarantee."""
    g = gap(F, fminx, fminy, x, y)
    if g >= min_gap:
        return True
    if S.shape[0] == 0 or g < min_gap - 2.0 * _EPS:
        return False
    return exact_gap(S, R, x, y) >= min_gap - _EPS


@njit(cache=True)
def hfield(F, fminx, fminy, x0, y0, W, H, hcell, min_gap, gx, gy):
    """8-connected Dijkstra from the goal (gx, gy) over cells with gap >= min_gap -> distance grid (W, H)."""
    freec = np.zeros((W, H), np.bool_)
    for i in range(W):
        for j in range(H):
            freec[i, j] = gap(F, fminx, fminy, x0 + i * hcell, y0 + j * hcell) >= min_gap
    dist = np.full((W, H), np.inf)
    gi = int(round((gx - x0) / hcell))
    gj = int(round((gy - y0) / hcell))
    dist[gi, gj] = 0.0
    s2 = math.sqrt(2.0)
    di = np.array([1, -1, 0, 0, 1, 1, -1, -1])
    dj = np.array([0, 0, 1, -1, 1, -1, 1, -1])
    cc = np.array([1.0, 1.0, 1.0, 1.0, s2, s2, s2, s2])
    pq = [(0.0, gi, gj)]
    while len(pq) > 0:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for t in range(8):
            a = i + di[t]
            b = j + dj[t]
            if 0 <= a < W and 0 <= b < H and freec[a, b]:
                nd = d + cc[t] * hcell
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    heapq.heappush(pq, (nd, a, b))
    return dist


@njit(cache=True)
def _hat(C, cx0, cy0, ccell, x, y):
    """Extra cost multiplier at (x, y) from the cost map C (0 outside it or if C is empty)."""
    if C.shape[0] == 0:
        return 0.0
    i = int((y - cy0) / ccell)
    j = int((x - cx0) / ccell)
    if 0 <= i < C.shape[0] and 0 <= j < C.shape[1]:
        return float(C[i, j])
    return 0.0


@njit(cache=True)
def hfield_hat(F, fminx, fminy, x0, y0, W, H, hcell, min_gap, gx, gy, C, cx0, cy0, ccell):
    """Like hfield, but each step costs distance x (1 + cost map), averaged over the two cells. Still a
    lower bound of the search cost, so it stays a valid heuristic."""
    freec = np.zeros((W, H), np.bool_)
    mul = np.ones((W, H))
    for i in range(W):
        for j in range(H):
            px, py = x0 + i * hcell, y0 + j * hcell
            freec[i, j] = gap(F, fminx, fminy, px, py) >= min_gap
            mul[i, j] = 1.0 + _hat(C, cx0, cy0, ccell, px, py)
    dist = np.full((W, H), np.inf)
    gi = int(round((gx - x0) / hcell))
    gj = int(round((gy - y0) / hcell))
    dist[gi, gj] = 0.0
    s2 = math.sqrt(2.0)
    di = np.array([1, -1, 0, 0, 1, 1, -1, -1])
    dj = np.array([0, 0, 1, -1, 1, -1, 1, -1])
    cc = np.array([1.0, 1.0, 1.0, 1.0, s2, s2, s2, s2])
    pq = [(0.0, gi, gj)]
    while len(pq) > 0:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for t in range(8):
            a = i + di[t]
            b = j + dj[t]
            if 0 <= a < W and 0 <= b < H and freec[a, b]:
                nd = d + cc[t] * hcell * 0.5 * (mul[i, j] + mul[a, b])
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    heapq.heappush(pq, (nd, a, b))
    return dist


@njit(cache=True)
def _h(hd, hx0, hy0, hcell, v_max, x, y):
    i = int(round((x - hx0) / hcell))
    j = int(round((y - hy0) / hcell))
    if 0 <= i < hd.shape[0] and 0 <= j < hd.shape[1] and np.isfinite(hd[i, j]):
        return hd[i, j] / v_max
    return np.inf


@njit(cache=True)
def _key(x, y, th, xy_cell, tb, th_bins):
    ix = int(math.floor(x / xy_cell))
    iy = int(math.floor(y / xy_cell))
    it = int(round(th / tb)) % th_bins
    return ((ix + _OFF) * _BIG + (iy + _OFF)) * th_bins + it


# ---------- Analytic Dubins connection ----------
_TW = 2 * math.pi
# Words: LSL RSR LSR RSL RLR LRL. +1 = left, -1 = right, 0 = straight.
_WORDS = np.array([[1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [-1, 1, -1], [1, -1, 1]], dtype=np.int64)


@njit(cache=True)
def _mod(a):
    return a % _TW


@njit(cache=True)
def _dubins_words(x0, y0, t0, x1, y1, t1, r):
    """All Dubins paths from (x0, y0, t0) to (x1, y1, t1) with radius r -> (count, word ids (6,), segment lengths (6, 3))."""
    codes = np.empty(6, np.int64)
    segs = np.empty((6, 3))
    n = 0
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy) / r
    th = math.atan2(dy, dx)
    a, b = _mod(t0 - th), _mod(t1 - th)
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    tmp = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
    if tmp >= 0:
        at = math.atan2(cb - ca, d + sa - sb)
        codes[n] = 0
        segs[n, 0], segs[n, 1], segs[n, 2] = _mod(-a + at), math.sqrt(tmp), _mod(b - at)
        n += 1
    tmp = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
    if tmp >= 0:
        at = math.atan2(ca - cb, d - sa + sb)
        codes[n] = 1
        segs[n, 0], segs[n, 1], segs[n, 2] = _mod(a - at), math.sqrt(tmp), _mod(-b + at)
        n += 1
    tmp = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
    if tmp >= 0:
        pp = math.sqrt(tmp)
        at = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, pp)
        codes[n] = 2
        segs[n, 0], segs[n, 1], segs[n, 2] = _mod(-a + at), pp, _mod(-_mod(b) + at)
        n += 1
    tmp = -2 + d * d + 2 * cab - 2 * d * (sa + sb)
    if tmp >= 0:
        pp = math.sqrt(tmp)
        at = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, pp)
        codes[n] = 3
        segs[n, 0], segs[n, 1], segs[n, 2] = _mod(a - at), pp, _mod(b - at)
        n += 1
    tmp = (6.0 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8.0
    if abs(tmp) <= 1:
        pp = _mod(2 * math.pi - math.acos(tmp))
        t = _mod(a - math.atan2(ca - cb, d - sa + sb) + pp / 2.0)
        codes[n] = 4
        segs[n, 0], segs[n, 1], segs[n, 2] = t, pp, _mod(a - b - t + pp)
        n += 1
    tmp = (6.0 - d * d + 2 * cab + 2 * d * (-sa + sb)) / 8.0
    if abs(tmp) <= 1:
        pp = _mod(2 * math.pi - math.acos(tmp))
        t = _mod(-a + math.atan2(-ca + cb, d + sa - sb) + pp / 2.0)
        codes[n] = 5
        segs[n, 0], segs[n, 1], segs[n, 2] = t, pp, _mod(b - a - t + pp)
        n += 1
    return n, codes, segs


@njit(cache=True)
def _dubins_sample(x, y, th, code, s0, s1, s2, r, step):
    seg = (s0, s1, s2)
    m_all = 0
    for i in range(3):
        m_all += max(1, int(math.ceil(seg[i] * r / step)))
    out = np.empty((m_all, 3))
    q = 0
    for i in range(3):
        c = _WORDS[code, i]
        length = seg[i] * r
        m = max(1, int(math.ceil(length / step)))
        ds = length / m
        for _ in range(m):
            if c == 0:
                x, y = x + ds * math.cos(th), y + ds * math.sin(th)
            else:
                k = (1.0 if c == 1 else -1.0) / r
                th2 = th + k * ds
                x, y = x + (math.sin(th2) - math.sin(th)) / k, y - (math.cos(th2) - math.cos(th)) / k
                th = th2
            out[q, 0], out[q, 1], out[q, 2] = x, y, th
            q += 1
    return out


@njit(cache=True)
def _speed_of(k, v_max, w_max, a_max, alpha):
    k = abs(k)
    if k < 1e-9:
        return v_max
    v = min(v_max, w_max / k, math.sqrt(a_max / k))
    if alpha:
        v = min(v, (12.0 * alpha * 2.0 / k) ** (1.0 / 3.0))     # 2.0 = hybrid_astar.TOL_LAT
    return v


@njit(cache=True)
def _accel_loss(v1, v2, a_max):
    hi, lo = max(v1, v2), min(v1, v2)
    return 0.0 if hi < 1e-9 else (hi - lo) * (hi - lo) / (2.0 * a_max * hi)


@njit(cache=True)
def _shot_cost(code, s0, s1, s2, r, v_max, w_max, a_max, alpha, v_in, has_v_in):
    """Driving time of a Dubins path including speed-change losses."""
    vk = _speed_of(1.0 / r, v_max, w_max, a_max, alpha)
    seg = (s0, s1, s2)
    # Compensated (Neumaier) summation, same as Python's built-in sum() for floats since 3.12.
    f, cc = 0.0, 0.0
    for i in range(3):
        xv = (seg[i] * r) / (v_max if _WORDS[code, i] == 0 else vk)
        t = f + xv
        if abs(f) >= abs(xv):
            cc += (f - t) + xv
        else:
            cc += (xv - t) + f
        f = t
    if cc != 0.0 and math.isfinite(cc):
        f += cc
    tc = f
    first = True
    prev = v_in
    for i in range(3):
        if seg[i] > 1e-9:
            v = v_max if _WORDS[code, i] == 0 else vk
            if first:
                if not has_v_in:
                    prev = v
                first = False
            tc += _accel_loss(prev, v, a_max)
            prev = v
    return tc


@njit(cache=True)
def _shot_ok(F, fminx, fminy, min_gap, x0, y0, t0, code, s0, s1, s2, r, step, S, R):
    """Walk the Dubins path point by point and stop at the first blocked point (most candidates are
    blocked). Returns the sampled path if it is free, else an empty array."""
    x, y, th = x0, y0, t0
    seg = (s0, s1, s2)
    for i in range(3):
        c = _WORDS[code, i]
        length = seg[i] * r
        m = max(1, int(math.ceil(length / step)))
        ds = length / m
        for _ in range(m):
            if c == 0:
                x, y = x + ds * math.cos(th), y + ds * math.sin(th)
            else:
                k = (1.0 if c == 1 else -1.0) / r
                th2 = th + k * ds
                x, y = x + (math.sin(th2) - math.sin(th)) / k, y - (math.cos(th2) - math.cos(th)) / k
                th = th2
            if not free(F, fminx, fminy, S, R, min_gap, x, y):
                return np.empty((0, 3))
    return _dubins_sample(x0, y0, t0, code, s0, s1, s2, r, step)


@njit(cache=True)
def _shot(F, fminx, fminy, min_gap, x0, y0, t0, x1, y1, t1, grid, ratio, step, v_max, w_max, a_max, alpha,
          v_in, has_v_in, S, R):
    """Cheapest free Dubins connection over the radius grid -> (points (m, 3) or (0, 3), time).
    Candidates are tried cheapest first; the first free one is refined by bisecting its radius."""
    ng = grid.shape[0]
    cost = np.empty(ng * 6)
    cr = np.empty(ng * 6)
    ccode = np.empty(ng * 6, np.int64)
    cseg = np.empty((ng * 6, 3))
    nc = 0
    for gi in range(ng):
        r = grid[gi]
        n, codes, segs = _dubins_words(x0, y0, t0, x1, y1, t1, r)
        for j in range(n):
            cost[nc] = _shot_cost(codes[j], segs[j, 0], segs[j, 1], segs[j, 2], r, v_max, w_max, a_max, alpha,
                                  v_in, has_v_in)
            cr[nc] = r
            ccode[nc] = codes[j]
            cseg[nc, 0], cseg[nc, 1], cseg[nc, 2] = segs[j, 0], segs[j, 1], segs[j, 2]
            nc += 1
    order = np.argsort(cost[:nc], kind="mergesort")              # stable sort
    for oi in range(nc):
        q = order[oi]
        tc, r, code = cost[q], cr[q], ccode[q]
        pts = _shot_ok(F, fminx, fminy, min_gap, x0, y0, t0, code, cseg[q, 0], cseg[q, 1], cseg[q, 2], r, step, S, R)
        if pts.shape[0] == 0:
            continue
        lo, hi = r, r * ratio
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            n, codes, segs = _dubins_words(x0, y0, t0, x1, y1, t1, mid)
            found = -1
            for j in range(n):
                if codes[j] == code:
                    found = j
                    break
            pm = np.empty((0, 3))
            if found >= 0:
                pm = _shot_ok(F, fminx, fminy, min_gap, x0, y0, t0, code, segs[found, 0], segs[found, 1],
                              segs[found, 2], mid, step, S, R)
            if pm.shape[0] > 0:
                cm = _shot_cost(code, segs[found, 0], segs[found, 1], segs[found, 2], mid, v_max, w_max, a_max,
                                alpha, v_in, has_v_in)
                if cm < tc:
                    lo, tc, pts = mid, cm, pm
                    continue
            hi = mid
        return pts, tc
    return np.empty((0, 3)), np.inf


@njit(cache=True)
def search(F, fminx, fminy, hd, hx0, hy0, hcell, v_max, sx, sy, sth, gx, gy, K, vk, cost, a_max, switch,
           min_gap, xy_cell, th_bins, step, goal_tol, w, max_expand, v0, has_v0,
           pose=False, gth=0.0, shot_every=1, shot_dist=80.0, grid=np.empty(0), ratio=1.0, w_max=1.0, alpha=0.0,
           free_gth=False, seed_th=np.empty(0), seed_g=np.empty(0),
           C=np.zeros((0, 0)), cx0=0.0, cy0=0.0, ccell=1.0, relax=0.0, bx=np.nan, by=np.nan, S=np.zeros((0, 4)), R=0.0):
    """Weighted A* over (x, y, theta) -> (path (m, 3) or (0, 3), expansions, cost).
    pose=True: finish only through an analytic connection to (gx, gy, gth). free_gth=True: the arrival
    heading is free - each connection attempt uses the heading from the node toward the goal.
    seed_th / seed_g: turn-in-place starts at (bx, by), with their start costs."""
    tb = 2.0 * math.pi / th_bins
    # relax > 0: the start is too close to a wall. Within relax px of the start, poses are allowed if the
    # gap does not get more than 0.5 px worse; a blocked heuristic there falls back to straight-line distance.
    g_start = gap(F, fminx, fminy, sx, sy)
    cap = 1 << 18
    NX = np.empty(cap)
    NY = np.empty(cap)
    NT = np.empty(cap)
    NG = np.empty(cap)
    NP = np.empty(cap, np.int64)
    NK = np.empty(cap, np.int64)
    NX[0], NY[0], NT[0], NG[0], NP[0], NK[0] = sx, sy, sth, 0.0, -1, -1
    nn = 1
    best = Dict.empty(key_type=types.int64, value_type=types.float64)
    best[_key(sx, sy, sth, xy_cell, tb, th_bins)] = 0.0
    pq = [(w * _h(hd, hx0, hy0, hcell, v_max, sx, sy), 0)]
    px = sx if math.isnan(bx) else bx
    py = sy if math.isnan(by) else by
    for j in range(seed_th.shape[0]):
        kj = _key(px, py, seed_th[j], xy_cell, tb, th_bins)
        if seed_g[j] < best.get(kj, np.inf) - 1e-9:
            best[kj] = seed_g[j]
            NX[nn], NY[nn], NT[nn], NG[nn], NP[nn], NK[nn] = px, py, seed_th[j], seed_g[j], -1, -1
            heapq.heappush(pq, (seed_g[j] + w * _h(hd, hx0, hy0, hcell, v_max, px, py), nn))
            nn += 1
    exp = 0
    nk = K.shape[0]
    # Keep the cheapest analytic connection; stop only when no open node can beat it.
    best_tot = np.inf
    best_out = np.empty((0, 3))
    while len(pq) > 0 and exp < max_expand:
        _f, n = heapq.heappop(pq)
        if _f >= best_tot:
            break
        x, y, th, g, kp = NX[n], NY[n], NT[n], NG[n], NK[n]
        if g > best.get(_key(x, y, th, xy_cell, tb, th_bins), np.inf) + 1e-9:
            continue
        exp += 1
        if pose and exp % shot_every == 0 and math.hypot(x - gx, y - gy) <= shot_dist:
            if kp >= 0:
                vin, hv = vk[kp], True
            else:
                vin, hv = v0, has_v0
            g_th = math.atan2(gy - y, gx - x) if free_gth else gth
            pts, tc = _shot(F, fminx, fminy, min_gap, x, y, th, gx, gy, g_th, grid, ratio, step, v_max, w_max,
                            a_max, alpha, vin, hv, S, R)
            if pts.shape[0] > 0 and C.shape[0] > 0:
                hsum = 0.0                                   # cost map along the curve (lower bound of time)
                for pi in range(pts.shape[0]):
                    hsum += _hat(C, cx0, cy0, ccell, pts[pi, 0], pts[pi, 1])
                tc += hsum * step / v_max
            if pts.shape[0] > 0:
                m = 0
                q = n
                while q >= 0:
                    m += 1
                    q = NP[q]
                out = np.empty((m + pts.shape[0], 3))
                q = n
                for t in range(m - 1, -1, -1):
                    out[t, 0], out[t, 1], out[t, 2] = NX[q], NY[q], NT[q]
                    q = NP[q]
                out[m:] = pts
                if g + tc < best_tot:
                    best_tot = g + tc
                    best_out = out
        if (not pose) and math.hypot(x - gx, y - gy) <= goal_tol:
            m = 0
            q = n
            while q >= 0:
                m += 1
                q = NP[q]
            out = np.empty((m, 3))
            q = n
            for t in range(m - 1, -1, -1):
                out[t, 0], out[t, 1], out[t, 2] = NX[q], NY[q], NT[q]
                q = NP[q]
            return out, exp, g
        for ki in range(nk):
            k = K[ki]
            if abs(k) < 1e-9:
                x2 = x + step * math.cos(th)
                y2 = y + step * math.sin(th)
                th2 = th
            else:
                th2 = th + k * step
                r = 1.0 / k
                x2 = x + r * (math.sin(th2) - math.sin(th))
                y2 = y - r * (math.cos(th2) - math.cos(th))
            near = relax > 0.0 and math.hypot(x2 - sx, y2 - sy) < relax
            if near:
                if min(gap(F, fminx, fminy, x2, y2), gap(F, fminx, fminy, (x + x2) / 2, (y + y2) / 2)) < \
                        min(min_gap, g_start - 0.5):
                    continue
            elif not free(F, fminx, fminy, S, R, min_gap, x2, y2) or \
                    not free(F, fminx, fminy, S, R, min_gap, (x + x2) / 2, (y + y2) / 2):
                continue
            if kp >= 0:
                vprev = vk[kp]
            elif has_v0:
                vprev = v0
            else:
                vprev = vk[ki]
            hi = max(vprev, vk[ki])
            lo = min(vprev, vk[ki])
            al = 0.0 if hi < 1e-9 else (hi - lo) * (hi - lo) / (2.0 * a_max * hi)
            g2 = g + cost[ki] * (1.0 + _hat(C, cx0, cy0, ccell, x2, y2)) + al + (switch if (kp >= 0 and ki != kp) else 0.0)
            d = _key(x2, y2, th2, xy_cell, tb, th_bins)
            if g2 < best.get(d, np.inf) - 1e-9:
                best[d] = g2
                h2 = _h(hd, hx0, hy0, hcell, v_max, x2, y2)
                if near and not np.isfinite(h2):
                    h2 = math.hypot(x2 - gx, y2 - gy) / v_max
                if not np.isfinite(h2):
                    continue
                if nn >= NX.shape[0]:
                    c2 = 2 * NX.shape[0]
                    NX2, NY2, NT2, NG2 = np.empty(c2), np.empty(c2), np.empty(c2), np.empty(c2)
                    NP2, NK2 = np.empty(c2, np.int64), np.empty(c2, np.int64)
                    NX2[:nn], NY2[:nn], NT2[:nn], NG2[:nn] = NX[:nn], NY[:nn], NT[:nn], NG[:nn]
                    NP2[:nn], NK2[:nn] = NP[:nn], NK[:nn]
                    NX, NY, NT, NG, NP, NK = NX2, NY2, NT2, NG2, NP2, NK2
                NX[nn], NY[nn], NT[nn], NG[nn], NP[nn], NK[nn] = x2, y2, th2, g2, n, ki
                nn += 1
                heapq.heappush(pq, (g2 + w * h2, nn - 1))
    if best_out.shape[0] > 0:
        return best_out, exp, best_tot
    return np.empty((0, 3)), exp, np.inf

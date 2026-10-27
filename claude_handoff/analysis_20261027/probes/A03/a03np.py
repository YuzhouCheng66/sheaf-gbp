"""a03np.py (A03): numpy / scipy-only versions of the probe helpers (no torch import in pool workers: the shared Windows box ran out of
commit memory when 30 torch processes loaded their CUDA DLLs). Same arithmetic as mapping_3d_storey_data_v420.observe and the
rate_baselines_lib metric (6-neighbourhood dilation = scipy cross structuring element); both checked against the torch versions in p3/p4."""
import json, math, os
import numpy as np
from scipy import ndimage
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = (128, 128, 32)
SENSOR_RANGE = 5.0
BEAMS = 65536
CROSS = ndimage.generate_binary_structure(3, 1)


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


RAYS = fibonacci_sphere(BEAMS)


def cell_of(L):
    return (L / 128, L / 128, 3.2 / 32)


def observe_np(occ, cell, poses, rays=RAYS):
    occf = np.asarray(occ, bool).reshape(-1)
    X, Y, Z = DIMS; V = X * Y * Z
    cellv = np.asarray(cell, np.float64); shape = np.array([X, Y, Z])
    ds = 0.5 * float(cellv.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    P = len(poses); R = rays.shape[0]
    org = np.repeat(np.asarray(poses, np.float64), R, 0); dirs = np.tile(rays, (P, 1))
    known = np.zeros(V, bool); hit = np.zeros(V, bool)
    for s in range(1, nsteps + 1):
        if len(org) == 0:
            break
        idx = np.floor((org + dirs * (s * ds)) / cellv).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape), 1)
        if not inside.all():
            org, dirs, idx = org[inside], dirs[inside], idx[inside]
        lin = (idx[:, 0] * Y + idx[:, 1]) * Z + idx[:, 2]
        h = occf[lin]
        known[lin] = True; hit[lin[h]] = True
        if h.any():
            keep = ~h; org, dirs = org[keep], dirs[keep]
    out = np.full(V, -1, np.int8); out[known] = 0; out[hit] = 1
    return out.reshape(DIMS)


def dilate6(m):
    return ndimage.binary_dilation(m, CROSS)


def _f1(a, b):
    tp = int((a & b).sum()); fp = int((a & ~b).sum()); fn = int((~a & b).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None


def score_np(p, t, slab, obj, struct):
    """= rate_baselines_lib.metrics for one map (target-known scope)"""
    s = (t == 1) & dilate6(t == 0); region = ~slab & ~dilate6(struct)
    known = t >= 0
    po = (p == 1) & known; to = (t == 1) & known; ps = po & ~((t == 1) & ~s)
    return dict(surface_f1=_f1(ps, s), var_occ_f1=_f1(po & ~slab, to & ~slab), furn_f1=_f1(po & region, to & region))


def fpr(p, t):
    u = t < 0
    return float((p[u] >= 0).mean()) if u.any() else None


NB2 = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, 1), (1, 2)]


def ctx2d_bytes(img, k):
    s = np.asarray(img).astype(np.int64); X, Y = s.shape; c = np.zeros_like(s)
    for dx, dy in NB2:
        sh = np.zeros_like(s)
        if dy >= 0:
            sh[dx:, dy:] = s[:X - dx, :Y - dy]
        else:
            sh[dx:, :dy] = s[:X - dx, -dy:]
        c = c * k + sh
    idx = (c * k + s).reshape(-1)
    cnt = np.bincount(idx, minlength=k ** (len(NB2) + 1)).reshape(-1, k).astype(np.float64)
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    a = 0.5
    bits = ((gammaln(N + k * a) - gammaln(k * a)) - (gammaln(cnt + a) - gammaln(a)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2


def targets(own, reached):
    r = np.asarray(reached, bool)
    return np.where(r[:, :, None, None, None], own[None], np.int8(-1)).max(1)


def _psum(m):
    p = np.zeros(tuple(d + 1 for d in m.shape), np.int64)
    p[1:, 1:, 1:] = m.astype(np.int64).cumsum(0).cumsum(1).cumsum(2)
    return p


def _bsum(p, x0, x1, y0, y1, z0, z1):
    return (p[x1, y1, z1] - p[x0, y1, z1] - p[x1, y0, z1] - p[x1, y1, z0] + p[x0, y0, z1] + p[x0, y1, z0] + p[x1, y0, z0] - p[x0, y0, z0])


def box_cover(must, forbid):
    X, Y, Z = must.shape
    pm = _psum(must); pf = _psum(forbid)
    cov = np.zeros_like(must)
    boxes = []
    for z, x, y in np.argwhere(must.transpose(2, 0, 1)):
        if cov[x, y, z]:
            continue
        b = [x, x + 1, y, y + 1, z, z + 1]
        grown = True
        while grown:
            grown = False
            for ax, sgn in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
                lo, hi = 2 * ax, 2 * ax + 1; lim = (X, Y, Z)[ax]
                nb = list(b)
                if sgn > 0:
                    if b[hi] >= lim:
                        continue
                    nb[lo], nb[hi] = b[hi], b[hi] + 1
                else:
                    if b[lo] <= 0:
                        continue
                    nb[lo], nb[hi] = b[lo] - 1, b[lo]
                if _bsum(pf, *nb) == 0 and _bsum(pm, *nb) > 0:
                    if sgn > 0:
                        b[hi] += 1
                    else:
                        b[lo] -= 1
                    grown = True
        cov[b[0]:b[1], b[2]:b[3], b[4]:b[5]] |= must[b[0]:b[1], b[2]:b[3], b[4]:b[5]]
        boxes.append(tuple(int(v) for v in b))
    return boxes


def paint(boxes, shape=DIMS):
    g = np.zeros(shape, bool)
    for b in boxes:
        x0, x1, y0, y1, z0, z1 = b[:6]
        g[x0:x1, y0:y1, z0:z1] = True
    return g


def load_fx(name):
    z = np.load(os.path.join(HERE, 'fx', name + '.npz'))
    meta = json.load(open(os.path.join(HERE, 'fx', name + '.json')))
    return {k: z[k] for k in z.files}, meta


def names(demo_only=False):
    out = sorted(f[:-4] for f in os.listdir(os.path.join(HERE, 'fx')) if f.endswith('.npz'))
    if demo_only:
        out = [n for n in out if n.endswith('_i0') or n.endswith('_i1')]
    return out


NB3 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]


def ctx_bytes_np(m, k=3, offset=1):
    """= rate_baselines_lib.ctx_bytes (k = 3, values -1/0/1) or the binary rate_split coder (k = 2, offset 0), numpy + gammaln"""
    s = np.asarray(m).astype(np.int64) + offset
    X, Y, Z = s.shape; c = np.zeros_like(s)
    for dx, dy, dz in NB3:
        sh = np.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * k + sh
    idx = (c * k + s).reshape(-1)
    cnt = np.bincount(idx, minlength=k ** (len(NB3) + 1)).reshape(-1, k).astype(np.float64)
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    a = 0.5
    bits = ((gammaln(N + k * a) - gammaln(k * a)) - (gammaln(cnt + a) - gammaln(a)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2

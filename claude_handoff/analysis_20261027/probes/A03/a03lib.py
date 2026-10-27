"""a03lib.py (A03, 2026-10-27): shared helpers for the structured / render-decoder probes. CPU only.
- observe_t: torch re-implementation of mapping_3d_storey_data_v420.observe (same float64 arithmetic; checked bit-for-bit in p1)
- ctx coders: ternary / binary 3D (as rate_baselines_lib / rate_split), n-ary 2D causal context coder for plan maps
- box cover: greedy maximal axis-aligned boxes that cover a must-cover voxel set without entering forbidden voxels
- fixture loader for probes/A03/fx (regenerated VAL houses: true poses, object specs)"""
import json, math, os, sys
import numpy as np, torch
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from rate_baselines_lib import ctx_bytes, metrics, targets, NB, LG  # noqa: E402,F401
import rate_baselines_lib as RBL  # noqa: E402

DIMS = (128, 128, 32)
SENSOR_RANGE = 5.0
BEAMS = 65536


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


RAYS = torch.from_numpy(fibonacci_sphere(BEAMS))


def cell_of(L):
    return (L / 128, L / 128, 3.2 / 32)


def observe_t(occ, cell, poses, rays=RAYS):
    """occ bool [X,Y,Z] (np or torch); poses [P,3] metres; returns int8 np [X,Y,Z] ternary (-1 unknown, 0 free, 1 occupied)."""
    occ = torch.as_tensor(np.asarray(occ)).bool().reshape(-1)
    X, Y, Z = DIMS; V = X * Y * Z
    cellt = torch.tensor(np.asarray(cell, np.float64)); shape = torch.tensor([X, Y, Z])
    ds = 0.5 * float(np.min(cell)); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    P = len(poses); R = rays.shape[0]
    org = torch.as_tensor(np.asarray(poses, np.float64)).repeat_interleave(R, 0); dirs = rays.repeat(P, 1)
    alive = torch.arange(P * R)
    known = torch.zeros(V, dtype=torch.bool); hit = torch.zeros(V, dtype=torch.bool)
    for s in range(1, nsteps + 1):
        if alive.numel() == 0:
            break
        idx = torch.floor((org[alive] + dirs[alive] * (s * ds)) / cellt).long()
        inside = ((idx >= 0) & (idx < shape)).all(1)
        alive = alive[inside]; idx = idx[inside]
        lin = (idx[:, 0] * Y + idx[:, 1]) * Z + idx[:, 2]
        h = occ[lin]
        known[lin] = True; hit[lin[h]] = True
        alive = alive[~h]
    out = torch.full((V,), -1, dtype=torch.int8); out[known] = 0; out[hit] = 1
    return out.reshape(DIMS).numpy()


def render_union(occ, cell, pose_sets):
    """lattice join (max) of observe_t over several pose sets (one per agent)."""
    out = None
    for ps in pose_sets:
        o = observe_t(occ, cell, ps)
        out = o if out is None else np.maximum(out, o)
    return out


def ctx_bytes_binary(mask):
    s = torch.from_numpy(np.asarray(mask).astype(np.int64)); X, Y, Z = s.shape; c = torch.zeros_like(s)
    for dx, dy, dz in NB:
        sh = torch.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * 2 + sh
    idx = (c * 2 + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=2 ** (len(NB) + 1)).reshape(-1, 2).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    bits = ((LG(N + 1.) - LG(1.)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2


NB2 = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, 1), (1, 2)]


def ctx2d_bytes(img, k):
    """exact adaptive KT code length of a k-ary 2D image (values 0..k-1) in raster order, 8 causal neighbours, + 2 B"""
    s = torch.from_numpy(np.asarray(img).astype(np.int64)); X, Y = s.shape; c = torch.zeros_like(s)
    for dx, dy in NB2:
        sh = torch.zeros_like(s)
        if dy >= 0:
            sh[dx:, dy:] = s[:X - dx, :Y - dy]
        else:
            sh[dx:, :dy] = s[:X - dx, -dy:]
        c = c * k + sh
    idx = (c * k + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=k ** (len(NB2) + 1)).reshape(-1, k).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    a = 0.5
    bits = ((LG(N + k * a) - LG(k * a)) - (LG(cnt + a) - LG(a)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2


def _psum(m):
    p = np.zeros(tuple(d + 1 for d in m.shape), np.int64)
    p[1:, 1:, 1:] = m.astype(np.int64).cumsum(0).cumsum(1).cumsum(2)
    return p


def _bsum(p, x0, x1, y0, y1, z0, z1):
    return (p[x1, y1, z1] - p[x0, y1, z1] - p[x1, y0, z1] - p[x1, y1, z0] + p[x0, y0, z1] + p[x0, y1, z0] + p[x1, y0, z0] - p[x0, y0, z0])


def box_cover(must, forbid):
    """greedy maximal boxes: seed at the first uncovered must-voxel (z, x, y order), grow one layer at a time in the 6 directions while the
    new layer holds no forbidden voxel and at least one must-voxel; returns list of (x0, x1, y0, y1, z0, z1) half-open voxel boxes."""
    X, Y, Z = must.shape
    pm = _psum(must); pf = _psum(forbid)
    cov = np.zeros_like(must)
    boxes = []
    order = np.argwhere(must.transpose(2, 0, 1))                        # (z, x, y) order
    for z, x, y in order:
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
    for x0, x1, y0, y1, z0, z1 in boxes:
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


def score(pred, target, slab, obj, struct):
    """metrics() of rate_baselines_lib on single maps -> dict"""
    t = lambda a: torch.from_numpy(np.asarray(a).reshape(1, -1))
    return metrics(t(pred), t(target), t(slab)[0], t(obj)[0], t(struct)[0])[0]

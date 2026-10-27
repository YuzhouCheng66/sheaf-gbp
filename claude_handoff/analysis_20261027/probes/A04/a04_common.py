"""A04 probes, shared helpers (Claude, 2026-10-27). CPU only. Reads the 6 VAL demo houses through rate_baselines_lib.
Adds: binary / conditional adaptive context coders (same KT estimator as rate_baselines_lib.ctx_bytes), the generator's first-hit ray
marcher (a copy of mapping_3d_storey_data_v420.observe, with occupancy given as a boolean grid), tree helpers."""
import glob, math, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
sys.path.insert(0, ANALYSIS)
from rate_baselines_lib import load, targets, ctx_bytes, metrics, down_up, NB, LG, DIMS  # noqa: E402,F401

DATA = os.path.join(ROOT, 'demo_web3d', 'data')
SENSOR_RANGE = 5.0


def houses():
    return sorted(glob.glob(os.path.join(DATA, 'house_*.json')))


def _code_len(idx, nctx, nsym):
    cnt = torch.bincount(idx, minlength=nctx * nsym).reshape(-1, nsym).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    a = 0.5; A = a * nsym
    bits = ((LG(N + A) - LG(A)) - (LG(cnt + a) - LG(a)).sum(1)).sum() / math.log(2)
    return float(bits)


def causal_ctx(s, base):
    """s: int64 tensor [X, Y, Z] with values in 0..base-1 -> context id from the 9 causal neighbours of rate_baselines_lib.NB"""
    X, Y, Z = s.shape; c = torch.zeros_like(s)
    for dx, dy, dz in NB:
        sh = torch.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * base + sh
    return c, base ** len(NB)


def ctx_bits_cond(y, base, side=(), mask=None):
    """exact adaptive KT code length (bits) of grid y (values 0..base-1) in raster order with the 9-voxel causal context of y itself
    and extra decoder-known context grids `side` (list of (int array, cardinality)). mask: only these voxels are coded (the decoder must
    know the mask). Returns bits (no header)."""
    s = torch.from_numpy(np.ascontiguousarray(y).astype(np.int64))
    c, n = causal_ctx(s, base)
    for g, card in side:
        c = c * card + torch.from_numpy(np.ascontiguousarray(g).astype(np.int64)); n *= card
    idx = (c * base + s)
    if mask is not None:
        idx = idx[torch.from_numpy(np.ascontiguousarray(mask))]
    return _code_len(idx.reshape(-1), n, base)


def ctx_bytes_binary(m):
    return ctx_bits_cond(m.astype(np.int64), 2) / 8 + 2


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


def observe(occ, cell, poses, rays):
    """copy of mapping_3d_storey_data_v420.observe: first-hit ray marching, occ bool/uint8 [X, Y, Z], poses [P, 3] metres."""
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float64)
    out = np.full(occ.shape, -1, np.int8)
    ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    for p in poses:
        alive = np.ones(len(rays), bool)
        for s in range(1, nsteps + 1):
            idx_all = np.floor((p[None] + rays * (s * ds)) / cell).astype(np.int64)
            inside = np.all((idx_all >= 0) & (idx_all < shape), 1) & alive
            if not inside.any():
                break
            ii = idx_all[inside]
            hit = occ[ii[:, 0], ii[:, 1], ii[:, 2]] == 1
            fr = ii[~hit]; out[fr[:, 0], fr[:, 1], fr[:, 2]] = np.maximum(out[fr[:, 0], fr[:, 1], fr[:, 2]], 0)
            hh = ii[hit]; out[hh[:, 0], hh[:, 1], hh[:, 2]] = 1
            ai = np.nonzero(inside)[0]; alive[ai[hit]] = False
            alive &= inside | ~alive
    return out


def tree_sides(n, edges):
    """directed tree edges (s, r) -> sorted sender-side node list"""
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    out = {}
    for i, j in edges:
        for s, r in ((i, j), (j, i)):
            side, stack = {s}, [s]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v != r and v not in side:
                        side.add(v); stack.append(v)
            out[(s, r)] = sorted(side)
    return out


def gt_masks(gt):
    slab = torch.from_numpy(gt['slab'].reshape(-1)); obj = torch.from_numpy(gt['obj'].reshape(-1))
    struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(-1))
    return slab, obj, struct


def retry(fn, *args, tries=40, wait=15, **kw):
    """the shared machine's commit charge sometimes hits its limit (other probes); small allocations then fail. Retry the unit of work."""
    import gc, time as _t
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except MemoryError as e:
            gc.collect(); print(f'  [retry {i + 1}: {type(e).__name__}]', flush=True); _t.sleep(wait)
    raise MemoryError('gave up')

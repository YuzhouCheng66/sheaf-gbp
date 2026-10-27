"""a07_lib.py (A07, 2026-10-27): shared helpers for the A07 evaluation probes. CPU only.
- ray casting copied from mapping_3d_storey_data_v420.observe (bit-exact clean path) + a noisy log-odds integrator;
- binary / ternary / conditional adaptive context coders (same KT estimator and 9-voxel causal context as rate_baselines_lib.ctx_bytes);
- demo loader re-exported from ../../rate_baselines_lib.py."""
import math, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from rate_baselines_lib import load, targets, metrics, ctx_bytes, down_up, NB, LG, gz, xz, DIMS  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))), 'demo_web3d', 'data')
SENSOR_RANGE = 5.0
BEAMS = 65536
POSE_BYTES = 5.5          # world_dl: 44 bit per agent (centre at the R1 cell + 3 jitters at 1 cm)


def fibonacci_sphere(n=BEAMS):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


RAYS = fibonacci_sphere()


def observe(occ, cell, poses, rays=RAYS):
    """bit-exact copy of v420 observe: first-hit marching, occupied wins; returns int8 ternary map."""
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float64)
    out = np.full(occ.shape, -1, np.int8)
    ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    for p in poses:
        p = np.asarray(p, np.float64)
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


def trace(occ, cell, p, rays=RAYS):
    """per ray: true first-hit range (m, sample distance s*ds) or nan, and the free extent (m) for rays without a hit
    (where the ray left the grid or the range)."""
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float64); p = np.asarray(p, np.float64)
    ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    hit_r = np.full(len(rays), np.nan); exit_r = np.full(len(rays), (nsteps + 0.5) * ds)
    alive = np.ones(len(rays), bool)
    for s in range(1, nsteps + 1):
        ai = np.nonzero(alive)[0]
        if not len(ai):
            break
        idx = np.floor((p[None] + rays[ai] * (s * ds)) / cell).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape), 1)
        out_i = ai[~inside]; exit_r[out_i] = (s - 0.5) * ds; alive[out_i] = False
        ai2 = ai[inside]; idx = idx[inside]
        hit = occ[idx[:, 0], idx[:, 1], idx[:, 2]] == 1
        hit_r[ai2[hit]] = s * ds; alive[ai2[hit]] = False
    return hit_r, exit_r, ds


def rotz(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def integrate(shape, cell, ds, meas, l_hit=0.85, l_miss=-0.4):
    """log-odds integration of measured rays. meas: list of (origin [3], dirs [R,3], rng [R] (nan = no hit), free_ext [R]) in the
    MAP frame. A voxel counts once per ray (consecutive samples in one voxel are merged); the endpoint voxel is not also a miss.
    Returns ternary int8 (unknown if never touched; occupied if log-odds > 0 else free) and the raw counts."""
    V = int(np.prod(shape)); shape_a = np.array(shape); cell = np.asarray(cell, np.float64)
    nh = np.zeros(V, np.int64); nm = np.zeros(V, np.int64)
    for o, d, rng, fext in meas:
        hit = ~np.isnan(rng)
        reach = np.where(hit, rng, fext)                                   # free samples strictly before the endpoint
        end_idx = np.full(len(d), -1, np.int64)
        if hit.any():
            e = np.floor((o[None] + d[hit] * rng[hit, None]) / cell).astype(np.int64)
            ok = np.all((e >= 0) & (e < shape_a), 1)
            ef = np.ravel_multi_index(e[ok].T, shape)
            hi = np.nonzero(hit)[0][ok]; end_idx[hi] = ef
            nh += np.bincount(ef, minlength=V)
        nsteps = int(np.ceil(np.nanmax(reach) / ds)) if len(reach) else 0
        last = np.full(len(d), -1, np.int64); acc = []
        for s in range(1, nsteps + 1):
            act = np.nonzero(s * ds < reach - 1e-9)[0]
            if not len(act):
                break
            idx = np.floor((o[None] + d[act] * (s * ds)) / cell).astype(np.int64)
            inside = np.all((idx >= 0) & (idx < shape_a), 1)
            act = act[inside]; idx = idx[inside]
            f = (idx[:, 0] * shape[1] + idx[:, 1]) * shape[2] + idx[:, 2]
            new = (f != last[act]) & (f != end_idx[act])
            acc.append(f[new])
            last[act] = f
        if acc:
            nm += np.bincount(np.concatenate(acc), minlength=V)
    L = l_hit * nh + l_miss * nm
    out = np.where((nh + nm) == 0, -1, np.where(L > 0, 1, 0)).astype(np.int8).reshape(shape)
    return out, nh.reshape(shape), nm.reshape(shape)


# ---------------------------------------------------------------- context coders (KT, exact adaptive code length)
def _ctx_index(s, base):
    X, Y, Z = s.shape; c = torch.zeros_like(s)
    for dx, dy, dz in NB:
        sh = torch.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * base + sh
    return c


def _kt_bits(idx, nsym, nctx):
    cnt = torch.bincount(idx, minlength=nctx * nsym).reshape(-1, nsym).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    a = 0.5
    return float(((LG(N + a * nsym) - LG(a * nsym)) - (LG(cnt + a) - LG(a)).sum(1)).sum() / math.log(2))


def ctx_bin(mask):
    s = torch.from_numpy(np.asarray(mask).astype(np.int64))
    c = _ctx_index(s, 2)
    return _kt_bits((c * 2 + s).reshape(-1), 2, 2 ** len(NB)) / 8 + 2


def ctx_bin_cond(mask, side):
    """binary map coded with its 9 causal neighbours AND the co-located voxel of a side map known to both ends (3 states: -1/0/1 or 0/1)."""
    s = torch.from_numpy(np.asarray(mask).astype(np.int64)); si = torch.from_numpy(np.asarray(side).astype(np.int64) + 1)
    c = _ctx_index(s, 2) * 3 + si
    return _kt_bits((c * 2 + s).reshape(-1), 2, 2 ** len(NB) * 3) / 8 + 2


def ctx_ter_cond(m, side):
    """ternary map (-1/0/1) coded with 9 causal neighbours AND the co-located voxel of a ternary side map known to both ends."""
    s = torch.from_numpy(np.asarray(m).astype(np.int64) + 1); si = torch.from_numpy(np.asarray(side).astype(np.int64) + 1)
    c = _ctx_index(s, 3) * 3 + si
    return _kt_bits((c * 3 + s).reshape(-1), 3, 3 ** len(NB) * 3) / 8 + 2


def ctx_ter(m):
    return ctx_bytes(m)


# ---------------------------------------------------------------- metrics helpers
def score(pred, target, gt):
    """metrics() rows for stacked int8 [n, X, Y, Z] pred / target and the scene's gt planes."""
    slab = torch.from_numpy(gt['slab'].reshape(-1)); obj = torch.from_numpy(gt['obj'].reshape(-1))
    struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(-1))
    n = pred.shape[0]
    return metrics(torch.from_numpy(pred.reshape(n, -1)), torch.from_numpy(target.reshape(n, -1)), slab, obj, struct)


def mean_rows(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def demo_paths():
    import glob
    return sorted(glob.glob(os.path.join(DATA, 'house_*.json')))

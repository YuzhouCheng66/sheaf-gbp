"""A08 debate probe R (CPU, one process) = probe N + an INTER-AGENT FRAME ERROR: each agent writes its marks in its own slightly
rotated / shifted frame (rigid SE(2) error about its centre, 5 cm RMS / 1 deg), i.e. its map is misregistered against the public frame.
Original probe N docstring follows.
: under independent RANGE noise, which fusion is right on the radio forest -- the max-join
(current lattice rule; 'occupied wins') or the sum of per-agent log-odds (information-form GBP, exact on a tree for independent
evidence)?  Truth = the CLEAN map a noise-free sensor at the same poses would produce (A07's definition), so noise is not rewarded.
Worlds = demo houses' GT rasters; agents at stored centres + our own +-0.1 m jitter (probe 1 law); clean maps via the verbatim v420
observe(); noisy maps via a two-pass version of the same ray march: pass 1 finds the true first hit, the range gets noise, pass 2 marks
free voxels before the noisy return and a hit at it (dropouts mark nothing; no-hit rays carve free to range as in v420).
Per agent: hit / free counts over its 3 poses -> log-odds L = 0.85 hits - 0.41 frees (OctoMap-like); own noisy ternary = sign(L).
R8 (hop-8 reached sets of the demo forest): max = per-voxel max of the agents' noisy ternary maps; sum = sign(sum of their L).
Scored with the training metrics against the clean R8 union (and K0 against the clean own map); also ctx bytes noisy vs clean."""
import glob, json, math, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a08_common import observe, metrics, ctx_bytes, RAYS, SENSOR_RANGE  # noqa: E402
from a08_sim_lossy import load_gt  # noqa: E402

DATA = sys.argv[1]; OUT = sys.argv[2]; NH = int(sys.argv[3]) if len(sys.argv) > 3 else 3
LEVELS = {'reg': dict(sa=0.0, sr=0.0, drop=0.0, out=0.0, reg=True), 'cam+reg': dict(sa=0.0, sr=0.015, drop=0.10, out=0.005, reg=True)}
L_HIT, L_FREE = 0.85, 0.41


def observe_noisy(occ, cell, poses, rng, sa, sr, drop, out, reg=False, rays=RAYS):
    occ = occ.astype(np.uint8); shape = np.array(occ.shape); cell = np.asarray(cell, np.float64); V = int(np.prod(shape))
    ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds)); R = len(rays)
    hits = np.zeros(V, np.int32); frees = np.zeros(V, np.int32)
    strides = np.array([shape[1] * shape[2], shape[2], 1])
    c0 = poses.mean(0); th = np.deg2rad(rng.normal(0., 1.)) if reg else 0.; tr = rng.normal(0., 0.05 / np.sqrt(2), 2) if reg else np.zeros(2)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for p in poses:
        alive = np.ones(R, bool); s_hit = np.full(R, -1)
        for s in range(1, nsteps + 1):                                   # pass 1: true first hit (same march as v420)
            idx = np.floor((p[None] + rays * (s * ds)) / cell).astype(np.int64)
            inside = np.all((idx >= 0) & (idx < shape), 1) & alive
            if not inside.any():
                break
            ii = idx[inside]; h = occ[ii[:, 0], ii[:, 1], ii[:, 2]] == 1
            ai = np.nonzero(inside)[0]; s_hit[ai[h]] = s; alive[ai[h]] = False
            alive &= inside | ~alive
        hit_ray = s_hit > 0
        r_true = np.where(hit_ray, s_hit * ds, 0.)
        r_noisy = r_true + rng.normal(0., 1., R) * (sa + sr * r_true)
        is_out = hit_ray & (rng.random(R) < out)
        r_noisy = np.where(is_out, rng.uniform(0.3, np.maximum(r_true, 0.31)), r_noisy)
        s_stop = np.where(hit_ray, np.clip(np.round(r_noisy / ds), 1, nsteps + 1), nsteps + 1).astype(np.int64)
        dropped = rng.random(R) < drop
        live = ~dropped
        fidx, hidx = [], []
        for s in range(1, nsteps + 1):                                   # pass 2: carve free before the noisy return, hit at it
            pts = p[None] + rays * (s * ds)
            if reg:                                                      # misregistered frame: rigid SE(2) about the agent centre
                xy = (pts[:, :2] - c0[None, :2]) @ Rm.T + c0[None, :2] + tr[None]; pts = np.concatenate((xy, pts[:, 2:]), 1)
            idx = np.floor(pts / cell).astype(np.int64)
            inside = np.all((idx >= 0) & (idx < shape), 1) & live
            live &= inside | (s_stop <= s)                               # a ray that leaves the grid stops (as in v420)
            fr = inside & (s < s_stop); hh = inside & (s == s_stop) & hit_ray
            if fr.any(): fidx.append(idx[fr] @ strides)
            if hh.any(): hidx.append(idx[hh] @ strides)
            if not (inside & (s <= s_stop)).any():
                break
        if fidx: frees += np.bincount(np.concatenate(fidx), minlength=V).astype(np.int32)
        if hidx: hits += np.bincount(np.concatenate(hidx), minlength=V).astype(np.int32)
    return hits.reshape(occ.shape), frees.reshape(occ.shape)


def tern(L):
    return np.where(L > 0, np.int8(1), np.where(L < 0, np.int8(0), np.int8(-1))).astype(np.int8)


def score(P, T, gt):
    slab = torch.from_numpy(gt['slab'].reshape(-1)); obj = torch.from_numpy(gt['obj'].reshape(-1))
    struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(-1))
    rows = metrics(torch.from_numpy(P.reshape(len(P), -1)), torch.from_numpy(T.reshape(len(T), -1)), slab, obj, struct)
    for r, p, t in zip(rows, P, T):
        r['fpr'] = float(((p >= 0) & (t < 0)).sum()) / float((t < 0).sum())
        r['free_recall'] = float(((p == 0) & (t == 0)).sum()) / max(float((t == 0).sum()), 1.)
    return rows


if __name__ == '__main__':
    t0 = time.time(); paths = sorted(glob.glob(os.path.join(DATA, 'house_*.json')))[:NH]
    res = {lv: {'K0': [], 'R8max': [], 'R8sum': [], 'ctx_noisy': [], 'ctx_clean': []} for lv in LEVELS}
    for hi, path in enumerate(paths):
        m, gt = load_gt(path); cell = tuple(m['cell_m']); R8 = np.asarray(m['reached'][m['hops'].index(8)], bool)
        clean, Ls = [], {lv: [] for lv in LEVELS}
        for a in range(8):
            rng = np.random.default_rng(1000 * hi + a)
            c = np.asarray(m['positions_m'][a], float)
            poses = np.stack([c + np.r_[rng.uniform(-.1, .1, 2), 0.] for _ in range(3)])
            clean.append(observe(gt['occ'], cell, poses))
            for li, (lv, prm) in enumerate(LEVELS.items()):
                h, f = observe_noisy(gt['occ'], cell, poses, np.random.default_rng(777 + 100 * li + 1000 * hi + a), **prm)
                Ls[lv].append((L_HIT * h - L_FREE * f).astype(np.float32))
        clean = np.stack(clean); T8 = np.stack([clean[R8[a]].max(0) for a in range(8)])
        for lv in LEVELS:
            L = np.stack(Ls[lv]); own = tern(L)
            res[lv]['K0'] += score(own, clean, gt)
            res[lv]['R8max'] += score(np.stack([own[R8[a]].max(0) for a in range(8)]), T8, gt)
            res[lv]['R8sum'] += score(np.stack([tern(L[R8[a]].sum(0)) for a in range(8)]), T8, gt)
            res[lv]['ctx_noisy'] += [ctx_bytes(own[a]) for a in range(8)]
            res[lv]['ctx_clean'] += [ctx_bytes(clean[a]) for a in range(8)]
        print(f"{m['house']} done {time.time() - t0:.0f}s", flush=True)
    agg = {}
    for lv in LEVELS:
        d = {'ctx_noisy_med': float(np.median(res[lv]['ctx_noisy'])), 'ctx_clean_med': float(np.median(res[lv]['ctx_clean']))}
        for k in ('K0', 'R8max', 'R8sum'):
            for mk in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr', 'free_recall'):
                v = [r[mk] for r in res[lv][k] if r[mk] is not None]; d[f'{k}_{mk}'] = float(np.mean(v))
        agg[lv] = d
        print(lv, json.dumps({k: round(x, 4) for k, x in d.items()}))
    json.dump(agg, open(OUT, 'w'), indent=1); print(f'total {time.time() - t0:.0f}s')

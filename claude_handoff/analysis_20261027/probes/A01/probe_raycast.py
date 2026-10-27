"""probe_raycast.py (A01, 2026-10-27): what does a LOSSLESS map cost if the receiver re-derives visibility by ray casting?
Claim: own map = observe(O, poses) exactly, where O = observed-occupied voxels (own == 1), because every ray of the agent stops at
its first world-occupied voxel, which is in O, and every voxel it crosses before that is world-free, hence not in O. Same for the
round-8 union: target = max_a observe(O_union, poses_a) (other agents' surfaces never lie on agent a's free path).
So a lossless codec only needs ctx(O) + poses. This script
 (a) measures ctx(full ternary), ctx(known mask), ctx(O) with the 9-neighbour context of rate_baselines and a 13-neighbour context,
     for own maps, round-8 unions and directed-edge unions of the 6 VAL demo houses;
 (b) checks the principle exactly on synthetic re-observations (gt world of the demo house, demo agent centres, our own jitter),
     and prices pose precision (float64 / 1 mm / 1 cm) by the number of voxels that change when poses are quantised.
CPU only. Usage: python probe_raycast.py DATA_DIR OUT_JSON"""
import glob, json, math, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from rate_baselines_lib import load, targets, ctx_bytes, LG  # noqa: E402

torch.set_num_threads(4)
DATA = sys.argv[1]; OUT = sys.argv[2]
NB9 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]
NB13 = [(0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, -1), (0, 2, 0), (1, 0, 0), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0), (2, 0, 0), (1, 1, 1)]


def shift(s, dx, dy, dz):
    """value of the neighbour at (x-dx, y-dy, z-dz), zero outside"""
    out = torch.zeros_like(s); X, Y, Z = s.shape
    xs = slice(max(dx, 0), X + min(dx, 0)); xd = slice(max(-dx, 0), X - max(dx, 0))
    ys = slice(max(dy, 0), Y + min(dy, 0)); yd = slice(max(-dy, 0), Y - max(dy, 0))
    zs = slice(max(dz, 0), Z + min(dz, 0)); zd = slice(max(-dz, 0), Z - max(dz, 0))
    out[xs, ys, zs] = s[xd, yd, zd]
    return out


def ctx_binary(mask, nb):
    s = torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64)); c = torch.zeros_like(s)
    for d in nb:
        c = c * 2 + shift(s, *d)
    idx = (c * 2 + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=2 ** (len(nb) + 1)).reshape(-1, 2).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    bits = ((LG(N + 1.) - LG(1.)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2


# ---- the generator's sensor (copied from mapping_3d_storey_data_v420.observe / fibonacci_sphere, numpy only)
SENSOR_RANGE = 5.0


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


def observe(occ, cell, poses, rays):
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


def sizes(m):
    return dict(full=ctx_bytes(m), known=ctx_binary(m >= 0, NB9), O9=ctx_binary(m == 1, NB9), O13=ctx_binary(m == 1, NB13),
                n_occ=int((m == 1).sum()), n_known=int((m >= 0).sum()))


res = dict(own=[], union8=[], edge=[], principle=[])
t0 = time.time()
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]
    for a in range(n):
        res['own'].append(dict(house=m['house'], agent=a, **sizes(own[a])))
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    for a in range(n):
        res['union8'].append(dict(house=m['house'], agent=a, **sizes(T8[a])))
    adj = {i: set() for i in range(n)}
    for i, j in m['edges']:
        adj[i].add(j); adj[j].add(i)
    for i, j in m['edges']:
        for s, r in ((i, j), (j, i)):
            side, stack = {s}, [s]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v != r and v not in side:
                        side.add(v); stack.append(v)
            U = own[sorted(side)].max(0)
            res['edge'].append(dict(house=m['house'], sender=s, side=len(side), **sizes(U)))
    print(m['house'], 'sizes done', f'{time.time()-t0:.0f}s', flush=True)

# ---- (b) principle check on synthetic re-observations of the first demo house, 2 agents
m, own, pred, gt = load(sorted(glob.glob(os.path.join(DATA, 'house_*.json')))[0])
cell = tuple(m['cell_m']); occ = gt['occ'].astype(np.uint8); rays = fibonacci_sphere(65536)
rng = np.random.default_rng(20261027)
for a in range(2):
    centre = np.asarray(m['positions_m'][a], np.float64)
    jit = rng.uniform(-.1, .1, (3, 3)); jit[:, 2] = 0.
    poses = centre[None] + jit
    t1 = time.time(); own_s = observe(occ, cell, poses, rays); t_obs = time.time() - t1
    O = (own_s == 1).astype(np.uint8)
    replay = observe(O, cell, poses, rays)
    row = dict(agent=a, seconds_per_observe=t_obs, mismatch_exact=int((replay != own_s).sum()), n_known=int((own_s >= 0).sum()), n_occ=int(O.sum()),
               **{f'synthetic_{k}': v for k, v in sizes(own_s).items()})
    for q in (1e-3, 1e-2):                                             # poses quantised to q metres
        pq = np.round(poses / q) * q
        rq = observe(O, cell, pq, rays)
        row[f'mismatch_pose_{q:g}m'] = int((rq != own_s).sum())
        row[f'mismatch_pose_{q:g}m_occ_to_other'] = int(((own_s == 1) & (rq != 1)).sum())
    res['principle'].append(row); print('principle', row, flush=True)


def med(v):
    return float(np.median(v))


S = {}
for g in ('own', 'union8', 'edge'):
    r = res[g]
    S[g] = {k: med([x[k] for x in r]) for k in ('full', 'known', 'O9', 'O13', 'n_occ', 'n_known')}
    S[g]['O13_over_full_median'] = med([x['O13'] / x['full'] for x in r])
    S[g]['full_over_O13_median'] = med([x['full'] / x['O13'] for x in r])
    S[g]['mean'] = {k: float(np.mean([x[k] for x in r])) for k in ('full', 'known', 'O9', 'O13')}
json.dump(dict(summary=S, rows=res), open(OUT, 'w'), indent=1)
print(json.dumps(S, indent=1))

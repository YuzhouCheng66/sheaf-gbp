"""p6_noise.py (A03): what does RANGE NOISE do to lossless voxel costs vs a structured description? (6 demo VAL houses, 8 agents)
Noisy sensor: the same rays; a ray that hits at distance d reports d' = d + N(0, sigma); its endpoint voxel (at d') is occupied and the
voxels it passes at distances < d' - 0.10 m are free (a 0.1 m clearing margin, as occupancy mappers do); max-range rays unchanged;
occupied wins over free. sigma = 0, 0.02, 0.05 m (sigma = 0 with the margin is the control).
Per own map: ctx ternary (lossless map), hits-only binary ctx + poses, the label-free 'shell' box codec (exact cover of the noisy hits),
and the F1 that the CLEAN-world own map (= what a perfect structured / denoising codec would transmit) scores against the noisy target.
Usage: python -B p6_noise.py"""
import json, math, os, sys, time, zlib
from collections import Counter
import numpy as np
from a03np import cell_of, load_fx, HERE, box_cover, ctx_bytes_np, score_np, RAYS, DIMS, SENSOR_RANGE

SIGMAS = (0.0, 0.02, 0.05)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_p6'); os.makedirs(CACHE, exist_ok=True)
MARGIN = 0.10


def observe_noisy(occ, cell, poses, sigma, rng, rays=RAYS):
    occf = np.asarray(occ, bool).reshape(-1); X, Y, Z = DIMS; V = X * Y * Z
    cellv = np.asarray(cell, np.float64); shape = np.array([X, Y, Z])
    ds = 0.5 * float(cellv.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    known = np.zeros(V, bool); hit = np.zeros(V, bool)
    for p in poses:
        R = rays.shape[0]; org = np.asarray(p, np.float64)
        dist = np.full(R, np.inf)                                        # first-hit distance per ray (clean)
        alive = np.arange(R); last = np.full(R, -1)
        for s in range(1, nsteps + 1):
            if len(alive) == 0:
                break
            idx = np.floor((org + rays[alive] * (s * ds)) / cellv).astype(np.int64)
            inside = np.all((idx >= 0) & (idx < shape), 1)
            last[alive[~inside]] = s - 1; alive = alive[inside]; idx = idx[inside]
            lin = (idx[:, 0] * Y + idx[:, 1]) * Z + idx[:, 2]
            h = occf[lin]
            dist[alive[h]] = s * ds; alive = alive[~h]
        last[alive] = nsteps                                             # max-range rays
        hr = np.isfinite(dist)
        dn = dist.copy(); dn[hr] = np.clip(dist[hr] + rng.normal(0, sigma, hr.sum()) if sigma > 0 else dist[hr], ds, SENSOR_RANGE)
        for s in range(1, nsteps + 1):                                   # free voxels: clean rays up to their end, hit rays up to d' - margin
            lim = np.where(hr, dn - MARGIN, (last + 0.5) * ds)
            sel = np.nonzero(s * ds < lim)[0]
            if len(sel) == 0:
                break
            idx = np.floor((org + rays[sel] * (s * ds)) / cellv).astype(np.int64)
            ok = np.all((idx >= 0) & (idx < shape), 1); idx = idx[ok]
            known[(idx[:, 0] * Y + idx[:, 1]) * Z + idx[:, 2]] = True
        e = np.floor((org + rays[hr] * dn[hr][:, None]) / cellv).astype(np.int64)
        e = np.clip(e, 0, shape - 1); lin = (e[:, 0] * Y + e[:, 1]) * Z + e[:, 2]
        known[lin] = True; hit[lin] = True
    out = np.full(V, -1, np.int8); out[known] = 0; out[hit] = 1
    return out.reshape(DIMS)


def run(args):
    name, sigma = args
    t0 = time.time()
    fx, meta = load_fx(name); cell = cell_of(meta['chart_m']); traj = fx['trajectory']
    occ = fx['occ'].astype(bool); slab = fx['slab'].astype(bool); obj = fx['obj'].astype(bool); struct = occ & ~obj & ~slab
    rng = np.random.default_rng(zlib.crc32(f'{name}:{sigma}'.encode()))
    rows = []
    for a in range(traj.shape[1]):
        noisy = observe_noisy(occ, cell, traj[:, a], sigma, rng)
        clean = observe_noisy(occ, cell, traj[:, a], 0.0, rng) if sigma > 0 else noisy
        hits = noisy == 1
        sb = box_cover(hits & ~slab, ~(hits & ~slab))
        rows.append(dict(house=name, sigma=sigma, agent=a, ctx=ctx_bytes_np(noisy), hits_ctx=ctx_bytes_np(hits, 2, 0), shell_boxes=[list(b) for b in sb],
                         n_hits=int(hits.sum()), clean_vs_noisy=score_np(clean, noisy, slab, obj, struct)))
    print(name, sigma, f'{time.time()-t0:.0f}s', flush=True)
    json.dump(rows, open(os.path.join(CACHE, f'{name}_{sigma}.json'), 'w'))
    return 1


if __name__ == '__main__':
    import multiprocessing as mp
    houses = ['r1_i0', 'r1_i1', 'r2_i0', 'r2_i1', 'r3_i0', 'r3_i1']
    todo = [(h, s) for s in SIGMAS for h in houses if not os.path.exists(os.path.join(CACHE, f'{h}_{s}.json'))]
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(run, todo):
            pass
    R = [r for s in SIGMAS for h in houses for r in json.load(open(os.path.join(CACHE, f'{h}_{s}.json')))]
    S = {}
    for sg in SIGMAS:
        rr = [r for r in R if r['sigma'] == sg]
        tab = [Counter(), Counter(), Counter(), Counter()]
        for r in rr:
            for b in r['shell_boxes']:
                tab[0][b[4]] += 1; tab[1][b[1] - b[0]] += 1; tab[2][b[3] - b[2]] += 1; tab[3][b[5] - b[4]] += 1
        sizes = (32, 128, 128, 32)
        for r in rr:
            tot = 8. + 60
            for b in r['shell_boxes']:
                tot += 14
                for f, v in enumerate((b[4], b[1] - b[0], b[3] - b[2], b[5] - b[4])):
                    cc = tab[f]; N = sum(cc.values()); tot += -math.log2((cc.get(v, 0) + 0.5) / (N + 0.5 * sizes[f]))
            r['shell_bytes'] = tot / 8
        med = lambda f: float(np.median([f(r) for r in rr]))
        S[str(sg)] = dict(n=len(rr), ctx=med(lambda r: r['ctx']), hits_plus_poses=med(lambda r: r['hits_ctx'] + 5.5), shell_bytes=med(lambda r: r['shell_bytes']),
                          shell_boxes=med(lambda r: len(r['shell_boxes'])), n_hits=med(lambda r: r['n_hits']),
                          clean_vs_noisy_surface=float(np.mean([r['clean_vs_noisy']['surface_f1'] for r in rr])),
                          clean_vs_noisy_furn=float(np.mean([r['clean_vs_noisy']['furn_f1'] for r in rr if r['clean_vs_noisy']['furn_f1'] is not None])))
        print(sg, json.dumps(S[str(sg)]))
    json.dump(dict(summary=S, rows=[{k: v for k, v in r.items() if k != 'shell_boxes'} for r in R]), open(os.path.join(HERE, 'p6_noise.json'), 'w'), indent=1)

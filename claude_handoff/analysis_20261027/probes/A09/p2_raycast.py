"""A09 probe 2 (Claude, 2026-10-27, CPU only): is free space really ~free given S and the poses?

observe() below is a verbatim copy of mapping_3d_storey_data_v420.observe (first-hit ray marching, 65,536 Fibonacci beams, 5 m range,
step = half the smallest cell). The demo export has each agent's CENTRE pose (positions_m) but not the three +-0.1 m jittered poses, so:
  (a) centre-pose ray cast through S (the agent's own observed-occupied set) vs through the full GT occupancy: share of the own map's known
      voxels reproduced (recall), share of ray-cast-known voxels that are target-unknown (FPR, the guard's quantity), XOR with the own map;
  (b) the same through OUR K0 prediction's occupied voxels (3S-X + calibrated rule): what a 'ray-cast known' decoder would report with
      today's imperfect geometry, vs the learned known head (FPR and free recall of the exported prediction);
  (c) for two agents: a coarse-to-fine search for the three jittered poses (1 cm grid) -> does observe(S, 3 recovered poses) reproduce the own
      map voxel for voxel? (the claim own_map == observe(S, poses) holds by construction for the exact float poses).
Output: probes/A09/p2_raycast.json
"""
import glob, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a09lib import load, DIMS   # noqa: E402

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
SENSOR_RANGE = 5.0
BEAMS = 65536


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


def observe(occ, cell, poses, rays):
    """verbatim mapping_3d_storey_data_v420.observe"""
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


_observe_raw = observe


def observe(occ, cell, poses, rays, tries=60):
    """retry wrapper: the machine's commit charge is shared with other agents' probes (MemoryError is transient)"""
    for t in range(tries):
        try:
            return _observe_raw(occ, cell, poses, rays)
        except MemoryError:
            time.sleep(10)
    raise MemoryError('observe: no memory after retries')


def compare(rc, own):
    kr, ko = rc >= 0, own >= 0
    return dict(known_recall=float((kr & ko).sum() / max(ko.sum(), 1)), fpr=float((kr & ~ko).sum() / max((~ko).sum(), 1)),
                xor_share=float((rc != own).sum() / max(ko.sum(), 1)), occ_recall=float(((rc == 1) & (own == 1)).sum() / max((own == 1).sum(), 1)))


rays = fibonacci_sphere(BEAMS)
res = dict(centre_S=[], centre_GT=[], centre_pred=[], learned_pred=[], jitter=[])
t0 = time.time()
paths = sorted(glob.glob(os.path.join(DATA, 'house_*.json')))
for path in paths:
    m, own, pred, gt = load(path); n = own.shape[0]
    cell = np.array(m['cell_m'], np.float64)
    centres = []
    for a in range(n):                                         # snap the exported (3-decimal) centre back onto its R1 cell centre
        x, y, z = m['positions_m'][a]
        centres.append(np.array([(math.floor(x / cell[0]) + .5) * cell[0], (math.floor(y / cell[1]) + .5) * cell[1], z]))
    for a in range(n):
        S = (own[a] == 1).astype(np.uint8)
        rcS = observe(S, cell, [centres[a]], rays)
        rcG = observe(gt['occ'].astype(np.uint8), cell, [centres[a]], rays)
        P = (pred[0][a] == 1).astype(np.uint8)
        rcP = observe(P, cell, [centres[a]], rays)
        res['centre_S'].append(dict(house=m['house'], agent=a, **compare(rcS, own[a])))
        res['centre_GT'].append(dict(house=m['house'], agent=a, **compare(rcG, own[a])))
        # a ray-cast-known decoder: occupied = our predicted occupied voxels (all of them), free = ray-cast free through them
        rp = np.where(P == 1, np.int8(1), rcP)
        ko = own[a] >= 0; tf = own[a] == 0
        res['centre_pred'].append(dict(house=m['house'], agent=a, fpr=float(((rp >= 0) & ~ko).sum() / max((~ko).sum(), 1)),
                                       free_recall=float(((rp == 0) & tf).sum() / max(tf.sum(), 1))))
        lp = pred[0][a]
        res['learned_pred'].append(dict(house=m['house'], agent=a, fpr=float(((lp >= 0) & ~ko).sum() / max((~ko).sum(), 1)),
                                        free_recall=float(((lp == 0) & tf).sum() / max(tf.sum(), 1))))
    print(m['house'], f'{time.time() - t0:.0f}s', flush=True)

# (c) jitter recovery for two agents of the first house: coarse 5 cm grid, then 1 cm around the best coarse candidates
m, own, pred, gt = load(paths[0]); cell = np.array(m['cell_m'], np.float64)
for a in (0, 1):
    x, y, z = m['positions_m'][a]
    c = np.array([(math.floor(x / cell[0]) + .5) * cell[0], (math.floor(y / cell[1]) + .5) * cell[1], z])
    S = (own[a] == 1).astype(np.uint8); ko = own[a] >= 0

    def viol(off):
        rc = observe(S, cell, [c + np.array([off[0], off[1], 0.])], rays)
        return int(((rc >= 0) & ~ko).sum()), rc
    coarse = [(dx, dy) for dx in np.linspace(-.1, .1, 5) for dy in np.linspace(-.1, .1, 5)]
    vs = sorted(((viol(o)[0], o) for o in coarse), key=lambda t: t[0])
    cand = {}
    for v0, o in vs[:4]:
        for dx in np.arange(-.03, .031, .01):
            for dy in np.arange(-.03, .031, .01):
                oo = (round(float(np.clip(o[0] + dx, -.1, .1)), 2), round(float(np.clip(o[1] + dy, -.1, .1)), 2))
                if oo not in cand:
                    cand[oo] = viol(oo)
    ranked = sorted(cand.items(), key=lambda kv: kv[1][0])
    chosen, union = [], np.full(own[a].shape, -1, np.int8)                 # greedy: low-violation poses that add the most own-known coverage
    for _ in range(3):
        best = None
        for oo, (v, rc) in ranked[:40]:
            if oo in chosen:
                continue
            gain = int(((np.maximum(union, rc) >= 0) & ko).sum()) - int(((union >= 0) & ko).sum()) - 10 * v
            if best is None or gain > best[0]:
                best = (gain, oo, rc)
        chosen.append(best[1]); union = np.maximum(union, best[2])
    res['jitter'].append(dict(house=m['house'], agent=a, poses_cm=[[round(100 * o[0]), round(100 * o[1])] for o in chosen],
                              min_violation=ranked[0][1][0], **compare(union, own[a]), xor_voxels=int((union != own[a]).sum()),
                              known_voxels=int(ko.sum())))
    print('jitter', a, res['jitter'][-1], f'{time.time() - t0:.0f}s', flush=True)


def mean(rows, k):
    return float(np.mean([r[k] for r in rows]))


summary = {name: {k: mean(rows, k) for k in rows[0] if k not in ('house', 'agent')} for name, rows in res.items() if name != 'jitter'}
summary['jitter'] = res['jitter']
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, rows=res), open(os.path.join(HERE, 'p2_raycast.json'), 'w'), indent=1)

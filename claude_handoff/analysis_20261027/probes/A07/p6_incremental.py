"""p6_incremental.py (A07): what does CONTINUOUS mapping cost per update? For each demo house, two agents (0 and 4) walk a random path
of 12 poses, 0.4 m apart, in free floor with >= 0.35 m clearance (single 65,536-beam scan per pose, v420 sensor, occupied wins).
Per update t >= 2 (map M_t = union of scans 1..t):
  full_ctx      : re-send the whole map, ternary context code;
  delta_ctx     : the new map coded with the previous map as side information (both ends hold M_{t-1});
  full_H / delta_H : sensor-aware: (new) hit voxels coded with the previous hit set as side information + 4 B per new pose; the decoder
                  re-casts all poses through H_t (exact, see p1);
  ours_fixed    : a fixed 2,048 B message per update (32x32x1 @ 8+8);
  ours_event    : only the 32x32 code columns whose ternary content changed, 8 bit each + the index set log2 C(1024, k) (proxy)."""
import json, math, os, sys, time
import numpy as np
from scipy import ndimage
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a07_lib import observe, ctx_ter, ctx_ter_cond, ctx_bin, ctx_bin_cond

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, 'sim_cache')
STEPS, STEP_M, CLEAR_M = 12, 0.4, 0.35


def lg2binom(n, k):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def path(occ, cell, start, rng):
    free2 = ~occ[:, :, 1:17].any(2); clear = ndimage.distance_transform_edt(free2) * cell[0]
    def ok(p):
        i, j = int(p[0] // cell[0]), int(p[1] // cell[1])
        return 0 <= i < occ.shape[0] and 0 <= j < occ.shape[1] and clear[i, j] >= CLEAR_M
    pts = [np.array(start, float)]; h = rng.uniform(0, 2 * math.pi)
    while len(pts) < STEPS:
        for k in range(80):
            hh = h + (rng.normal(0, math.radians(25)) if k < 40 else rng.uniform(0, 2 * math.pi))
            q = pts[-1] + STEP_M * np.array([math.cos(hh), math.sin(hh), 0.])
            if ok(q) and ok((q + pts[-1]) / 2):
                pts.append(q); h = hh; break
        else:
            break
    return np.stack(pts)


def house(name):
    t0 = time.time(); z = np.load(os.path.join(CACHE, name + '.npz')); occ = z['gt_occ'].astype(np.uint8); cell = z['cell']
    rng = np.random.default_rng(sum(map(ord, name))); out = dict(house=name, walks=[])
    for a in (0, 4):
        P = path(occ, cell, z['pos'][a], rng); M = None; rows = []
        for t, p in enumerate(P):
            scan = observe(occ, cell, p[None]); Mn = scan if M is None else np.maximum(M, scan)
            if M is not None:
                Hn, Ho = Mn == 1, M == 1
                ch = (Mn != M).reshape(32, 4, 32, 4, 32).any((1, 3, 4)).sum()
                rows.append(dict(t=t + 1, full_ctx=ctx_ter(Mn), delta_ctx=ctx_ter_cond(Mn, M), full_H=ctx_bin(Hn) + 4 * (t + 1),
                                 delta_H=ctx_bin_cond(Hn & ~Ho, Ho.astype(np.int8)) + 4, ours_fixed=2048., ours_event=(int(ch) * 8 + lg2binom(1024, int(ch))) / 8,
                                 changed_cols=int(ch), known=float((Mn >= 0).mean())))
            M = Mn
        out['walks'].append(dict(agent=a, steps=len(P), rows=rows))
    print(name, [w['steps'] for w in out['walks']], f'{time.time() - t0:.0f}s', flush=True)
    return out


if __name__ == '__main__':
    names = sorted(f[:-4] for f in os.listdir(CACHE) if f.endswith('.npz'))
    with Pool(2) as pool:
        res = pool.map(house, names)
    rows = [r for h in res for w in h['walks'] for r in w['rows']]
    S = {k: float(np.median([r[k] for r in rows])) for k in ('full_ctx', 'delta_ctx', 'full_H', 'delta_H', 'ours_fixed', 'ours_event', 'changed_cols', 'known')}
    last = [w['rows'][-1] for h in res for w in h['walks'] if w['rows']]
    S['final_full_ctx'] = float(np.median([r['full_ctx'] for r in last])); S['final_full_H'] = float(np.median([r['full_H'] for r in last]))
    S['sum_delta_H_per_walk'] = float(np.median([sum(r['delta_H'] for r in w['rows']) for h in res for w in h['walks']]))
    S['sum_delta_ctx_per_walk'] = float(np.median([sum(r['delta_ctx'] for r in w['rows']) for h in res for w in h['walks']]))
    S['n_updates'] = len(rows)
    json.dump(dict(summary=S, houses=res), open(os.path.join(HERE, 'p6_incremental.json'), 'w'), indent=1)
    print(json.dumps(S, indent=1))

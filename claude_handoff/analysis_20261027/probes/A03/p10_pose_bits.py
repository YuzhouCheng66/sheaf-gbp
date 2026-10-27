"""p10_pose_bits.py (A03): how many pose bits does the render decoder need? The generator's agent centres sit on the R1 raster
(14 bit); the 3 jitters are float64 uniform in +-0.1 m (x, y). Render own = observe(own hits, poses) with the jitters rounded to
1 cm (world_dl's 44 bit), 2 mm and 0.5 mm, and count voxels that differ from the true own map (6 demo houses, 48 own maps).
Usage: python -B p10_pose_bits.py [workers]"""
import json, os, sys, time
import numpy as np
from a03np import observe_np, cell_of, load_fx, HERE, score_np

STEPS = (0.01, 0.002, 0.0005)


def house(name):
    t0 = time.time()
    fx, meta = load_fx(name); own = fx['private']; cell = cell_of(meta['chart_m']); traj = fx['trajectory']; cx = cell[0]
    slab = fx['slab'].astype(bool); obj = fx['obj'].astype(bool); struct = fx['occ'].astype(bool) & ~obj & ~slab
    rows = []
    for a in range(own.shape[0]):
        P = traj[:, a]
        centre = fx['positions'][a, :2]                                  # the R1 cell centre the jitters are added to (stored by the generator)
        for q in STEPS:
            Q = P.copy(); Q[:, :2] = centre + np.round((P[:, :2] - centre) / q) * q
            r = observe_np(own[a] == 1, cell, Q)
            rows.append(dict(house=name, agent=a, step=q, diff=int((r != own[a]).sum()), known=int((own[a] >= 0).sum()), **score_np(r, own[a], slab, obj, struct)))
    print(name, f'{time.time()-t0:.0f}s', flush=True)
    return rows


if __name__ == '__main__':
    import multiprocessing as mp
    hs = ['r1_i0', 'r1_i1', 'r2_i0', 'r2_i1', 'r3_i0', 'r3_i1']
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        R = [r for rr in pool.map(house, hs) for r in rr]
    S = {}
    for q in STEPS:
        rr = [r for r in R if r['step'] == q]
        S[str(q)] = dict(n=len(rr), exact=sum(r['diff'] == 0 for r in rr), diff_median=float(np.median([r['diff'] for r in rr])), diff_frac_known=float(np.mean([r['diff'] / r['known'] for r in rr])),
                         surface_f1=float(np.mean([r['surface_f1'] for r in rr])), furn_f1=float(np.mean([r['furn_f1'] for r in rr if r['furn_f1'] is not None])),
                         bits_per_agent=14 + 6 * float(np.ceil(np.log2(0.2 / q + 1))))
        print(q, json.dumps(S[str(q)]))
    json.dump(dict(summary=S, rows=R), open(os.path.join(HERE, 'p10_pose_bits.json'), 'w'), indent=1)

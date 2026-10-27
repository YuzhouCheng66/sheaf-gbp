"""A08 probe 4 (CPU, simulation): does the BOX decoder (which fills unknown space inside each component's bounding box) stay accurate
when the receiver only has poses rounded to 1 cm (5 B per agent)?  Probe 1 measured this for re-casting through O itself; boxes can be
hit by slightly shifted rays inside the sender's unknown region, so it is re-measured here.  Same worlds / jitter law / seeds as probe 3.
Per agent: target M (exact poses); decodes RayCast(D, round(P, 1 cm)) for D = plan + O_furn voxels and D = plan + greedy boxes; K0 and
R8 (per-source casts joined by max) scored with the training metrics + unknown->known FPR."""
import glob, json, os, sys, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a08_common import observe, gt_layers, plan_split, box_cover, render_boxes, render_structure  # noqa: E402
from a08_sim_lossy import load_gt, pack, unpack, score  # noqa: E402

DATA = sys.argv[1]; OUT = sys.argv[2]


def job(args):
    path, a, seed = args
    m, gt = load_gt(path); lay = gt_layers(gt); cell = tuple(m['cell_m']); rng = np.random.default_rng(seed)
    c = np.asarray(m['positions_m'][a], float)
    poses = np.stack([c + np.r_[rng.uniform(-.1, .1, 2), 0.] for _ in range(3)])
    M = observe(gt['occ'], cell, poses); O = M == 1; F = M == 0
    plan, Of = plan_split(O, gt, lay); S = render_structure(plan, gt, lay)
    Pq = np.round(poses / 0.01) * 0.01
    out = dict(house=m['house'], agent=a, target=pack(M))
    out['vox_q1'] = pack(observe(S | Of, cell, Pq))
    out['box_q1'] = pack(observe(S | render_boxes(box_cover(Of, ~F)), cell, Pq))
    return out


if __name__ == '__main__':
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(DATA, 'house_*.json')))
    jobs = [(p, a, 1000 * i + a) for i, p in enumerate(paths) for a in range(8)]
    with Pool(6, maxtasksperchild=2) as pool:
        res = pool.map(job, jobs, chunksize=1)
    agg = {}
    for v in ('vox_q1', 'box_q1'):
        K, R = [], []
        for p in paths:
            m, gt = load_gt(p); R8 = np.asarray(m['reached'][m['hops'].index(8)], bool)
            rr = sorted([r for r in res if r['house'] == m['house']], key=lambda r: r['agent'])
            T = np.stack([unpack(*r['target']) for r in rr]); P = np.stack([unpack(*r[v]) for r in rr])
            K += score(P, T, gt); R += score(np.stack([P[R8[a]].max(0) for a in range(8)]), np.stack([T[R8[a]].max(0) for a in range(8)]), gt)
        agg[v] = {f'{h}_{k}': float(np.mean([r[k] for r in rows if r[k] is not None])) for h, rows in (('K0', K), ('R8', R)) for k in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr')}
        print(v, json.dumps({k: round(x, 4) for k, x in agg[v].items()}))
    json.dump(agg, open(OUT, 'w'), indent=1); print(f'{time.time() - t0:.0f}s')

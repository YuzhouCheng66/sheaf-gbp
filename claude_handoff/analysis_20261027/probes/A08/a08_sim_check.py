"""A08 probe 1 (CPU): is an agent's ternary map a deterministic function of (its observed-occupied voxels O, its 3 poses)?
Simulation on the demo houses' GT raster (the real jitter is not stored, so poses = stored centres + our own +-0.1 m jitter, same law).
For each agent: M = observe(gt_occ, poses); O = (M == 1); M1 = observe(O, poses) -> count voxels where M1 != M (theorem: 0).
Also: re-cast with poses rounded to 1 cm / 2 cm / 5 cm -> mismatch counts and F1 vs M (what pose quantisation costs)."""
import json, os, sys, time
import numpy as np, torch
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a08_common import load, observe, metrics, DIMS, gt_layers  # noqa: E402

DATA = sys.argv[1]; OUT = sys.argv[2]; NH = int(sys.argv[3]) if len(sys.argv) > 3 else 6


def one(args):
    path, a, seed = args
    m, own, pred, gt = load(path)
    cell = tuple(m['cell_m']); rng = np.random.default_rng(seed)
    c = np.asarray(m['positions_m'][a], float)
    poses = np.stack([c + np.r_[rng.uniform(-.1, .1, 2), 0.] for _ in range(3)])
    t0 = time.time(); M = observe(gt['occ'], cell, poses); t_obs = time.time() - t0
    O = M == 1
    M1 = observe(O, cell, poses)
    res = dict(house=m['house'], agent=a, known=int((M >= 0).sum()), n_O=int(O.sum()), exact_mismatch=int((M1 != M).sum()), t_observe=t_obs)
    for q in (0.01, 0.02, 0.05):
        Mq = observe(O, cell, np.round(poses / q) * q)
        res[f'mismatch_q{int(q*100)}cm'] = int((Mq != M).sum())
        slab = torch.from_numpy(gt['slab'].reshape(-1)); obj = torch.from_numpy(gt['obj'].reshape(-1))
        struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(-1))
        r = metrics(torch.from_numpy(Mq.reshape(1, -1)), torch.from_numpy(M.reshape(1, -1)), slab, obj, struct)[0]
        fpr = float(((Mq >= 0) & (M < 0)).sum()) / float((M < 0).sum())
        res[f'q{int(q*100)}cm'] = dict(surface_f1=r['surface_f1'], furn_f1=r['furn_f1'], var_occ_f1=r['var_occ_f1'], fpr=fpr)
    return res


if __name__ == '__main__':
    paths = sorted(p for p in __import__('glob').glob(os.path.join(DATA, 'house_*.json')))[:NH]
    jobs = [(p, a, 1000 * i + a) for i, p in enumerate(paths) for a in range(8)]
    with Pool(min(24, len(jobs))) as pool:
        rows = pool.map(one, jobs)
    json.dump(rows, open(OUT, 'w'), indent=1)
    print('agents', len(rows), 'exact mismatches total', sum(r['exact_mismatch'] for r in rows), 'median n_O', np.median([r['n_O'] for r in rows]), 'median known', np.median([r['known'] for r in rows]))
    for q in (1, 2, 5):
        print(f'pose rounded to {q} cm: mismatch median {np.median([r[f"mismatch_q{q}cm"] for r in rows]):.0f} voxels; '
              f'surface F1 {np.mean([r[f"q{q}cm"]["surface_f1"] for r in rows]):.4f} furn F1 {np.mean([r[f"q{q}cm"]["furn_f1"] for r in rows if r[f"q{q}cm"]["furn_f1"] is not None]):.4f} '
              f'FPR {np.mean([r[f"q{q}cm"]["fpr"] for r in rows]):.5f}')
    print('observe seconds median', np.median([r['t_observe'] for r in rows]))

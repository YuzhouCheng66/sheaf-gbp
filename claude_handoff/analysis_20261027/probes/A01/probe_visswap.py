"""probe_visswap.py (A01, 2026-10-27): what if visibility were derived by ray casting instead of predicted by the known head?
No retraining: the demo's 3S-X predictions (calibrated rule) give the decoded occupied set S = (pred == 1). The receiver casts the public
ray set from the exact poses (poses_demo.json, verified bit-exact) of the agents it has heard from (K0: itself; R8: all reached) through
S and labels voxels: first hit = occupied, crossed before = free, never reached = unknown. Scored with the training metric copy against
the same targets. Variants:
  pred      : the model's own ternary output (reference; equals the export's metrics)
  rc        : ray casting through S
  rc_or_pred: rc, plus voxels the model calls occupied that no ray reached (only matters where the target is known)
  oracle_rc : ray casting through the TRUE observed surface O of the target (must be exact: F1 = 1, FPR = 0)
Usage: python probe_visswap.py DATA_DIR POSES_JSON OUT_JSON [processes]"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from probe_visorder import fibonacci_sphere, first_visits  # noqa: E402

VARIANTS = ('pred', 'rc', 'rc_or_pred', 'oracle_rc')


def raycast(S, cell, poses_list, rays, dims):
    out = np.full(dims, -1, np.int8)
    for poses in poses_list:
        o, _ = first_visits(S.astype(np.uint8), cell, np.asarray(poses), rays)
        out = np.maximum(out, o)
    return out


def house_job(args):
    path, poses_json = args
    cache = os.path.join(HERE, 'visswap_rows_' + os.path.basename(path))
    if os.path.exists(cache):
        return json.load(open(cache))
    import torch
    torch.set_num_threads(1)
    from rate_baselines_lib import load, targets, metrics, DIMS
    rays = fibonacci_sphere(65536)
    m, own, pred, gt = load(path); n = own.shape[0]; cell = tuple(m['cell_m'])
    traj = np.asarray(json.load(open(poses_json))[m['house']]['trajectory_m'])
    slab = torch.from_numpy(gt['slab'].reshape(-1)); obj = torch.from_numpy(gt['obj'].reshape(-1))
    struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(-1))
    rows = []
    for hop in (0, 8):
        R = np.asarray(m['reached'][m['hops'].index(hop)], bool)
        T = targets(own, R)
        for a in range(n):
            plist = [traj[:, s] for s in np.nonzero(R[a])[0]]
            S = pred[hop][a] == 1
            rc = raycast(S, cell, plist, rays, DIMS)
            variants = dict(pred=pred[hop][a], rc=rc, rc_or_pred=np.where((rc < 0) & S, np.int8(1), rc))
            if hop == 0 or a == 0:                                     # oracle exactness check: every K0 map, one R8 map per house (time)
                variants['oracle_rc'] = raycast(T[a] == 1, cell, plist, rays, DIMS)
            t = torch.from_numpy(T[a].reshape(1, -1)); unk = T[a] < 0
            for name, pv in variants.items():
                r = metrics(torch.from_numpy(pv.reshape(1, -1)), t, slab, obj, struct)[0]
                r.update(house=m['house'], hop=hop, agent=a, variant=name, fpr=float((pv[unk] >= 0).mean()) if unk.any() else None,
                         known_recall=float(((pv >= 0) & (T[a] >= 0)).sum() / max((T[a] >= 0).sum(), 1)))
                rows.append(r)
        print(m['house'], 'hop', hop, 'done', flush=True)
    json.dump(rows, open(cache, 'w'))
    return rows


if __name__ == '__main__':
    import multiprocessing
    DATA, POSES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; procs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    paths = sorted(glob.glob(os.path.join(DATA, 'house_*.json')))
    with multiprocessing.get_context('spawn').Pool(procs) as pool:
        rows = [r for rs in pool.map(house_job, [(p, POSES) for p in paths]) for r in rs]
    S_ = {}
    for hop in (0, 8):
        for name in VARIANTS:
            R_ = [r for r in rows if r['hop'] == hop and r['variant'] == name]
            S_[f'K{hop}_{name}'] = {k: float(np.mean([r[k] for r in R_ if r[k] is not None])) for k in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr', 'known_recall')}
    json.dump(dict(summary=S_, rows=rows), open(OUT, 'w'), indent=1)
    print(json.dumps(S_, indent=1))

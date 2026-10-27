"""p7_utility.py (A07): is furniture-region F1 the right yardstick for 'practical use'? Same 6 VAL demo houses, our 3S-X predictions
(calibrated rule) at K0 and R8 vs the classical ternary 2x2x1 tau .5 down-up (rate_baselines) on the SAME demo targets.
Metrics (all restricted to target-known voxels, like metrics3d):
  furn_exact      : furniture-region F1 (metrics3d);
  furn_tol1       : same region, a prediction counts if a target-occupied region voxel lies within 1 voxel (6-nbhd), recall likewise;
  obst2d / tol1   : 2-D obstacle map for a ground robot: a column is an obstacle if any known voxel in 0.2-1.9 m is occupied; scope =
                    columns with a known voxel in the band; furniture columns only (columns away from the dilated wall footprint);
  obj_recall      : 26-connected components of target-occupied furniture-region voxels with >= 8 voxels count as observed object
                    parts; detected if >= 50 % of their voxels have a predicted-occupied voxel within 1 voxel;
  ghost_parts     : predicted-occupied components (>= 8 voxels) in the furniture region with no target-occupied voxel within 1 voxel.
"""
import json, os, sys
import numpy as np, torch
import torch.nn.functional as F
from scipy import ndimage
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a07_lib import load, targets, down_up, demo_paths, DIMS

HERE = os.path.dirname(os.path.abspath(__file__))
S6 = ndimage.generate_binary_structure(3, 1)


def dil(m, it=1, st=S6):
    return ndimage.binary_dilation(m, st, iterations=it)


def f1(tp_p, np_, tp_r, nr):
    if np_ + nr == 0:
        return None
    p = tp_p / np_ if np_ else 1.; r = tp_r / nr if nr else 1.
    return 2 * p * r / max(p + r, 1e-12)


def utility(p, t, region, wallcols):
    known = t >= 0; po = (p == 1) & known & region; to = (t == 1) & known & region
    out = {}
    tp = (po & to).sum(); out['furn_exact'] = f1(tp, po.sum(), tp, to.sum())
    out['furn_tol1'] = f1((po & dil(to)).sum(), po.sum(), (to & dil(po)).sum(), to.sum())
    band = slice(2, 19)
    kb = known[:, :, band].any(2) & ~wallcols
    ot = ((t[:, :, band] == 1) & known[:, :, band] & region[:, :, band]).any(2) & kb
    op = ((p[:, :, band] == 1) & known[:, :, band] & region[:, :, band]).any(2) & kb
    out['obst2d'] = f1((ot & op).sum(), op.sum(), (ot & op).sum(), ot.sum())
    S8 = np.ones((3, 3), bool)
    out['obst2d_tol1'] = f1((op & ndimage.binary_dilation(ot, S8)).sum(), op.sum(), (ot & ndimage.binary_dilation(op, S8)).sum(), ot.sum())
    lab, nl = ndimage.label(to, np.ones((3, 3, 3), bool)); dp = dil(po)
    det = tot = 0
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == k
        if comp.sum() < 8:
            continue
        tot += 1; det += (dp[sl][comp].mean() >= .5)
    out['obj_recall'] = det / tot if tot else None; out['obj_parts'] = tot
    labp, npl = ndimage.label(po, np.ones((3, 3, 3), bool)); dt = dil(to); ghosts = 0
    for k, sl in enumerate(ndimage.find_objects(labp), 1):
        comp = labp[sl] == k
        if comp.sum() >= 8 and not dt[sl][comp].any():
            ghosts += 1
    out['ghost_parts'] = ghosts
    return out


def house(path):
    meta, own, pred, gt = load(path); n = own.shape[0]
    struct = gt['occ'] & ~gt['obj'] & ~gt['slab']
    region = ~gt['slab'] & ~dil(struct)
    wallcols = ndimage.binary_dilation(struct[:, :, 2:19].any(2), np.ones((3, 3), bool))
    T8 = targets(own, meta['reached'][meta['hops'].index(8)])
    res = dict(house=meta['house'], rows=[])
    for hop, T in ((0, own), (8, T8)):
        P = pred[hop]
        for a in range(n):
            r_ours = utility(P[a], T[a], region, wallcols)
            _, u = down_up(T[a], (2, 2, 1), .5); r_221 = utility(u, T[a], region, wallcols)
            res['rows'].append(dict(hop=hop, agent=a, ours=r_ours, classic221=r_221))
    print(meta['house'], flush=True)
    return res


if __name__ == '__main__':
    with Pool(6) as pool:
        res = pool.map(house, demo_paths())
    S = {}
    for hop in (0, 8):
        for who in ('ours', 'classic221'):
            rows = [r[who] for h in res for r in h['rows'] if r['hop'] == hop]
            S[f'K{hop}_{who}'] = {k: float(np.mean([x[k] for x in rows if x[k] is not None])) for k in rows[0]}
    json.dump(dict(summary=S, houses=res), open(os.path.join(HERE, 'p7_utility.json'), 'w'), indent=1)
    for k, v in S.items():
        print(f'{k:16s} ' + '  '.join(f'{kk} {vv:.3f}' for kk, vv in v.items()))

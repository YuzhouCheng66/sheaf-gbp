"""A04 probe P8 (Claude, 2026-10-27, debate round): the training metric scores occupied voxels inside the target-known region only.
Does a prediction that is just the observed OCCUPIED set S (no poses, no free space: every other voxel reported unknown) reach F1 = 1?
If so every "F1 = 1 at X bytes" claim needs a free-space guard. Reports, for K0 own maps and R8 unions: surface / variable / furniture F1
(metrics copy of rate_baselines_lib), unknown->known FPR, 3-way accuracy on target-known voxels, and free-space recall (target-free
voxels predicted free), plus the occupied-set code length (ctx of the binary occupied mask, from p1)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, targets, metrics, gt_masks, HERE, retry  # noqa: E402


def extra(pred, tgt):
    known = tgt >= 0
    return dict(fpr=float((pred[~known] >= 0).mean()) if (~known).any() else 0., acc=float((pred[known] == tgt[known]).mean()),
                free_recall=float((pred[tgt == 0] == 0).mean()) if (tgt == 0).any() else None)


def one(path):
    m, own, pred, gt = load(path); n = own.shape[0]
    slab, obj, struct = gt_masks(gt)
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    out = {}
    for tag, T in (('K0', own), ('R8', T8)):
        P = np.where(T == 1, np.int8(1), np.int8(-1))
        rows = metrics(torch.from_numpy(P.reshape(n, -1)), torch.from_numpy(T.reshape(n, -1)), slab, obj, struct)
        ours = pred[0 if tag == 'K0' else 8]
        rows_ours = metrics(torch.from_numpy(ours.reshape(n, -1)), torch.from_numpy(T.reshape(n, -1)), slab, obj, struct)
        out[tag] = [dict(**r, **extra(P[a], T[a]), ours=dict(**ro, **extra(ours[a], T[a]))) for a, (r, ro) in enumerate(zip(rows, rows_ours))]
    return out


agg = {'K0': [], 'R8': []}
for path in houses():
    r = retry(one, path)
    for k in agg:
        agg[k] += r[k]
S = {}
for k, rows in agg.items():
    mean = lambda key, sub=None: float(np.mean([(x[sub][key] if sub else x[key]) for x in rows if (x[sub][key] if sub else x[key]) is not None]))
    S[k] = dict(S_only={q: mean(q) for q in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr', 'acc', 'free_recall')},
                ours_3SX={q: mean(q, 'ours') for q in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr', 'acc', 'free_recall')})
p1 = json.load(open(os.path.join(HERE, 'p1_visibility.json')))['summary']
S['occupied_set_ctx_median_B'] = dict(K0=p1['own']['ctx_occ_median'], R8=p1['union8']['ctx_occ_median'])
print(json.dumps(S, indent=1))
json.dump(S, open(os.path.join(HERE, 'p8_free_space_loophole.json'), 'w'), indent=1)

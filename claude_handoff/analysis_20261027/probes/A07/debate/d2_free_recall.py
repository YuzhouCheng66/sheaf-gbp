"""d2_free_recall.py (A07, debate round): the metric hole A09 pointed at. metrics3d's four F1s and the unknown->known FPR never
reward predicting target-known FREE voxels as free, so 'send only the occupied set S, call everything else unknown' scores F1 = 1,
FPR = 0. We measure free-space recall (target-free voxels predicted free) and 3-state accuracy on target-known voxels for: our 3S-X
predictions (calibrated rule), the classical ternary 2x2x1 down-up, and the S-only scheme, on the 6 VAL demo houses (K0 and R8)."""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from a07_lib import load, targets, down_up, demo_paths, score, mean_rows

def extra(p, t):
    k = t >= 0; f = t == 0
    return dict(free_recall=float(((p == 0) & f).sum() / max(f.sum(), 1)), acc=float((p[k] == t[k]).mean()))

rows = []
for path in demo_paths():
    m, own, pred, gt = load(path); T8 = targets(own, m['reached'][m['hops'].index(8)])
    for hop, T in ((0, own), (8, T8)):
        n = T.shape[0]
        S_only = np.where(T == 1, 1, -1).astype(np.int8)
        du = np.stack([down_up(T[a], (2, 2, 1), .5)[1] for a in range(n)])
        for name, P in (('ours', pred[hop]), ('classic221', du), ('S_only', S_only)):
            sc = score(P, T, gt)
            for a in range(n):
                rows.append(dict(house=m['house'], hop=hop, scheme=name, agent=a, **extra(P[a], T[a]), surface_f1=sc[a]['surface_f1'], furn_f1=sc[a]['furn_f1'],
                                 fpr=float(((P[a] >= 0) & (T[a] < 0)).sum() / max((T[a] < 0).sum(), 1))))
    print(m['house'], flush=True)
S = {}
for hop in (0, 8):
    for name in ('ours', 'classic221', 'S_only'):
        rr = [r for r in rows if r['hop'] == hop and r['scheme'] == name]
        S[f'K{hop}_{name}'] = {k: float(np.mean([r[k] for r in rr if r[k] is not None])) for k in ('surface_f1', 'furn_f1', 'fpr', 'free_recall', 'acc')}
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'd2_free_recall.json'), 'w'), indent=1)
for k, v in S.items():
    print(f'{k:16s} ' + '  '.join(f'{kk} {vv:.3f}' for kk, vv in v.items()))

"""A02 probe P4b (2026-10-27, CPU only): how much lossless information is the current 2 KB message worth?
Static (leave-one-house-out) context coder of the ternary target with the 9 causal neighbours of ctx, with and without the receiver's
decoded prediction (3S-X + calibrated rule, exported planes) as an extra context symbol at the voxel.  The saving = the information the
prediction carries about the exact map.  K0 (own maps) and R8 (round-8 unions), 6 demo VAL houses x 8 agents.
Output: probes/A02/p4b_pred_value.json"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from rate_baselines_lib import load, targets  # noqa: E402
NB9 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]


def ctx9(T):
    s = (T.astype(np.int64) + 1); X, Y, Z = s.shape
    P = np.zeros((X + 2, Y + 2, Z + 2), np.int64); P[2:, 2:, 2:] = s
    c = np.zeros_like(s)
    for dx, dy, dz in NB9:
        c = c * 3 + P[2 - dx:2 - dx + X, 2 - dy:2 - dy + Y, 2 - dz:2 - dz + Z]
    return c, s


houses = []
for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
    m, own, pred, gt = load(path)
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    items = []
    for hop, Tk in ((0, own), (8, T8)):
        for a in range(own.shape[0]):
            c, s = ctx9(Tk[a]); items.append(dict(hop=hop, agent=a, c=c.ravel().astype(np.int32), s=s.ravel().astype(np.int8), p=(pred[hop][a].ravel() + 1).astype(np.int8)))
    houses.append(dict(name=m['house'], items=items)); print('loaded', m['house'], flush=True)
NC = 3 ** 9
rows = []
for hold in range(len(houses)):
    for hop in (0, 8):
        n0 = np.zeros((NC, 3)); n1 = np.zeros((NC * 3, 3))
        for h_, H in enumerate(houses):
            if h_ == hold:
                continue
            for it in H['items']:
                if it['hop'] == hop:
                    c = it['c'].astype(np.int64); s = it['s'].astype(np.int64); p = it['p'].astype(np.int64)
                    n0 += np.bincount(c * 3 + s, minlength=NC * 3).reshape(NC, 3)
                    n1 += np.bincount((c * 3 + p) * 3 + s, minlength=NC * 9).reshape(NC * 3, 3)
        for it in houses[hold]['items']:
            if it['hop'] != hop:
                continue
            c = it['c'].astype(np.int64); s = it['s'].astype(np.int64); p = it['p'].astype(np.int64)
            q0 = (n0[c, s] + 2 / 3) / (n0[c].sum(1) + 2)                       # backoff-free KT-smoothed static model
            q1 = (n1[c * 3 + p, s] + 2 * q0) / (n1[c * 3 + p].sum(1) + 2)      # with the prediction, backing off to q0
            rows.append(dict(house=houses[hold]['name'], hop=hop, agent=it['agent'], static_no_pred=float(-np.log2(q0).sum() / 8), static_with_pred=float(-np.log2(q1).sum() / 8)))
            print(rows[-1], flush=True)
S = {}
for hop in (0, 8):
    rr = [r for r in rows if r['hop'] == hop]
    S[f'hop{hop}'] = dict(no_pred=float(np.median([r['static_no_pred'] for r in rr])), with_pred=float(np.median([r['static_with_pred'] for r in rr])),
                          saving_median=float(np.median([r['static_no_pred'] - r['static_with_pred'] for r in rr])),
                          saving_ratio_median=float(np.median([r['static_with_pred'] / r['static_no_pred'] for r in rr])))
print(json.dumps(S, indent=1))
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p4b_pred_value.json'), 'w'), indent=1)

"""a10_stalk_cd.py (A10): stalk counts of the converged / scheduled protocols for the coverage-keyed max join (see a10_stalk.py).
Per directed edge s->r: ev = stalks with evidence on the sender side; inn = stalks where the sender side's best strictly beats the
receiver side's best (the converged innovation; the receiver cannot need anything else). Collect-distribute schedule (best root):
child->parent sends ev of the child's subtree (no side information yet), parent->child sends inn over the child's subtree (both
ends know the child's subtree message). Also the per-agent own evidence and the global evidence (one copy of the fused section)."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import load, adjacency, side_of, hops_from  # noqa


def coverage(own, L):                                                # = a10_stalk.coverage
    k = (own >= 0).reshape(own.shape[0], L, 128 // L, L, 128 // L, 32).mean((2, 4, 5))
    return np.ceil(k * 255).astype(np.int64).reshape(own.shape[0], -1)


DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
res = {}
for L in (16, 32, 64, 128):
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
        m, own, pred, gt = load(path); n = own.shape[0]; adj = adjacency(m['edges'], n)
        x = coverage(own, L); E = {}
        for i, j in m['edges']:
            for s, r in ((i, j), (j, i)):
                S = x[side_of(adj, s, r)].max(0); R = x[side_of(adj, r, s)].max(0)
                E[(s, r)] = dict(ev=int((S > 0).sum()), inn=int((S > R).sum()))
        best = None
        for root in range(n):
            d = hops_from(adj, root); tot = 0
            for i, j in m['edges']:
                c, p = (i, j) if d[i] > d[j] else (j, i)
                tot += E[(c, p)]['ev'] + E[(p, c)]['inn']
            best = tot if best is None else min(best, tot)
        rows.append(dict(house=m['house'], cd_best=best, cut=sum(v['inn'] for v in E.values()), relay_ev=sum(v['ev'] for v in E.values()),
                         own_ev=float((x > 0).sum(1).mean()), global_ev=int((x.max(0) > 0).sum())))
    res[L] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'house'}
    print(L, {k: round(v) for k, v in res[L].items()})
json.dump(res, open(os.path.join(HERE, 'a10_stalk_cd.json'), 'w'), indent=1)

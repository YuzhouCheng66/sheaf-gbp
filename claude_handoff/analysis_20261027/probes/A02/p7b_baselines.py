"""A02 probe P7b (2026-10-27, CPU only): matched classical baselines (adaptive ctx and gzip-9 of the ternary map, rate_baselines_lib)
for exactly the maps P7 / P8 code: own maps, round-8 union targets and directed-edge sender-side unions of the 6 demo VAL houses.
Output: probes/A02/p7b_baselines.json (rows tagged by house so P7 / P8 rows can be matched)."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from rate_baselines_lib import load, targets, ctx_bytes, gz  # noqa: E402
rows = []
for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    for a in range(n):
        rows.append(dict(house=m['house'], kind='own', agent=a, ctx=ctx_bytes(own[a]), gzip=gz(own[a])))
        rows.append(dict(house=m['house'], kind='union8', agent=a, ctx=ctx_bytes(T8[a]), gzip=gz(T8[a])))
    adj = {i: set() for i in range(n)}
    for i, j in m['edges']:
        adj[i].add(j); adj[j].add(i)
    for i, j in m['edges']:
        for s, r in ((i, j), (j, i)):
            S, st = {s}, [s]
            while st:
                u = st.pop()
                for v in adj[u]:
                    if v != r and v not in S:
                        S.add(v); st.append(v)
            U = own[sorted(S)].max(0)
            rows.append(dict(house=m['house'], kind='edge', s=s, r=r, side=len(S), ctx=ctx_bytes(U), gzip=gz(U)))
    print(m['house'], flush=True)
S = {}
for kind in ('own', 'union8', 'edge'):
    for sub, hs in (('all6', None), ('held3', ('r1_i1', 'r2_i0', 'r3_i1'))):
        rr = [r for r in rows if r['kind'] == kind and (hs is None or r['house'] in hs)]
        S[f'{kind}_{sub}'] = {k: dict(median=float(np.median([r[k] for r in rr])), mean=float(np.mean([r[k] for r in rr]))) for k in ('ctx', 'gzip')}
print(json.dumps(S, indent=1))
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p7b_baselines.json'), 'w'), indent=1)

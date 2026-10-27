"""a10_struct_net.py (A10): network totals of the structure-aware exact code (a10_struct) per house: union relay (every directed edge
sends its sender-side structured code once), collect-distribute (best root; collect = sender-side structured code, distribute =
structured innovation over the child's side), cut-set sum of innovations. Re-uses a10_struct.code_struct (its probe body is guarded)."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import load, adjacency, side_of, union, reachable, hops_from  # noqa
import importlib.util
spec = importlib.util.spec_from_file_location('a10s', os.path.join(HERE, 'a10_struct.py'))
src = open(os.path.join(HERE, 'a10_struct.py'), encoding='utf-8').read().split('\nrows = []')[0]   # definitions only
mod = {'__file__': os.path.join(HERE, 'a10_struct.py'), '__name__': 'a10_struct_defs'}; exec(compile(src, 'a10_struct_defs', 'exec'), mod)
code_struct = mod['code_struct']; POSE_B = mod['POSE_B']

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
out = []
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]; adj = adjacency(m['edges'], n); pos = np.asarray(m['positions_m']); cell = m['cell_m']
    slab = gt['slab']; obj = gt['obj']; struct = gt['occ'] & ~gt['obj'] & ~gt['slab']; U = own.max(0)
    E = {}
    for i, j in m['edges']:
        for s_, r_ in ((i, j), (j, i)):
            Ss = side_of(adj, s_, r_); S = union(own, Ss); R = union(own, side_of(adj, r_, s_))
            reach = np.zeros(R.shape, bool)
            for a in Ss:
                reach |= reachable(R == 1, cell, pos[a])
            full = code_struct(S == 1, obj, struct, slab)['total'] + POSE_B * len(Ss)
            inn = code_struct(U == 1, obj, struct, slab, R, reach)['total'] + 2 + POSE_B * len(Ss)
            E[(s_, r_)] = (full, inn)
    best = min(sum(E[(c, p)][0] + E[(p, c)][1] for c, p in (((i, j) if hops_from(adj, root)[i] > hops_from(adj, root)[j] else (j, i)) for i, j in m['edges'])) for root in range(n))
    h = dict(house=m['house'], relay_struct=sum(v[0] for v in E.values()), cd_struct=best, cut_struct=sum(v[1] for v in E.values()),
             edge_full_median=float(np.median([v[0] for v in E.values()])), edge_inn_median=float(np.median([v[1] for v in E.values()])))
    out.append(h); print(h, flush=True)
json.dump(out, open(os.path.join(HERE, 'a10_struct_net.json'), 'w'), indent=1)
print({k: round(float(np.mean([h[k] for h in out]))) for k in out[0] if k != 'house'})

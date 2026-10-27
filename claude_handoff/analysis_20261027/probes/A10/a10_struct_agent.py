"""a10_struct_agent.py (A10): the structure-aware exact code WITHOUT ground-truth labels -- the decomposition an agent can make from its
own ternary map alone. A column is extruded as a full-height wall (W2) only when the agent saw it OCCUPIED both below 2.0 m (under any
lintel: door_h >= 2.0 m) and above 2.25 m (over any furniture: the tallest library part ends at 2.2 m), ceiling layers excluded.
Everything else the agent hit outside the floor / ceiling layers is sent as explicit hits (V_x). Exactness holds for ANY decomposition
in which every extruded voxel is truly occupied (true here: such a column cannot be furniture or a lintel-only door cell) and every
first hit is in O. Slab layers are the floor layer and the layers from the ceiling height up (house constants, 16 bit). Also the
innovation version per directed edge (receiver side R), as in a10_struct. Poses float32 (24 B / agent)."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import load, causal, kt_bits, adjacency, side_of, union, reachable, fwd_flags, hops_from  # noqa
src = open(os.path.join(HERE, 'a10_struct.py'), encoding='utf-8').read().split('\nrows = []')[0]
mod = {'__file__': os.path.join(HERE, 'a10_struct.py'), '__name__': 'defs'}; exec(compile(src, 'defs', 'exec'), mod)
ctx2d = mod['ctx2d']; POSE_B = mod['POSE_B']

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
ZC = (np.arange(32) + .5) * 0.1 - 0.1                                    # height above the floor top


def decompose(t, slab_layers):
    V = t == 1
    low = V[:, :, (ZC < 2.0) & ~slab_layers].any(2); high = V[:, :, (ZC > 2.25) & ~slab_layers].any(2)
    W2 = low & high
    Vx = V & ~np.broadcast_to(W2[:, :, None], V.shape) & ~np.broadcast_to(slab_layers[None, None], V.shape)
    return W2, Vx


def code(t, slab_layers, R=None, reach=None):
    W2, Vx = decompose(t, slab_layers)
    sl = np.broadcast_to(slab_layers[None, None], t.shape); wc = np.broadcast_to(W2[:, :, None], t.shape)
    v = Vx.astype(np.int64)
    if R is None:
        bW = kt_bits(W2.astype(np.int64), ctx2d(W2.astype(np.int64)), 2)[0]
        sel = ~wc & ~sl
        bV = kt_bits(v[sel], causal(v, 2)[sel], 2)[0]
        return (bW + bV) / 8 + 4
    W2R, _ = decompose(R, slab_layers); Rk = R >= 0
    cand = ~W2R; colknown = Rk.any(2)
    bW = kt_bits(W2[cand].astype(np.int64), (ctx2d(W2.astype(np.int64)) * 2 + colknown)[cand], 2)[0]
    sel = ~wc & ~sl & ~Rk
    bV = kt_bits(v[sel], ((causal(v, 2) * 8 + fwd_flags(Rk)) * 2 + reach.astype(np.int64))[sel], 2)[0]
    return (bW + bV) / 8 + 4


rows = []; houses = []
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]; adj = adjacency(m['edges'], n); pos = np.asarray(m['positions_m']); cell = m['cell_m']
    slab_layers = gt['slab'].any((0, 1))                                   # the house constants (floor layer, ceiling layers)
    U = own.max(0)
    # exactness precondition: every extruded voxel truly occupied
    for t in list(own) + [U]:
        W2, _ = decompose(t, slab_layers)
        assert gt['occ'][W2].all(), 'an extruded column is not fully occupied'
    for a in range(n):
        rows.append(dict(kind='own', house=m['house'], total=code(own[a], slab_layers) + POSE_B))
    rows.append(dict(kind='union', house=m['house'], total=code(U, slab_layers) + POSE_B * n))
    E = {}
    for i, j in m['edges']:
        for s_, r_ in ((i, j), (j, i)):
            Ss = side_of(adj, s_, r_); S = union(own, Ss); R = union(own, side_of(adj, r_, s_))
            reach = np.zeros(R.shape, bool)
            for a in Ss:
                reach |= reachable(R == 1, cell, pos[a])
            full = code(S, slab_layers) + POSE_B * len(Ss); inn = code(U, slab_layers, R, reach) + POSE_B * len(Ss)
            E[(s_, r_)] = (full, inn); rows.append(dict(kind='edge', house=m['house'], nS=len(Ss), full=full, total=inn))
    cd = min(sum(E[(c, p)][0] + E[(p, c)][1] for c, p in (((i, j) if hops_from(adj, root)[i] > hops_from(adj, root)[j] else (j, i)) for i, j in m['edges'])) for root in range(n))
    houses.append(dict(house=m['house'], cd=cd, cut=sum(v[1] for v in E.values()), relay=sum(v[0] for v in E.values())))
    print(houses[-1], flush=True)
json.dump(dict(rows=rows, houses=houses), open(os.path.join(HERE, 'a10_struct_agent.json'), 'w'), indent=1)
for kind in ('own', 'union', 'edge'):
    r = [x['total'] for x in rows if x['kind'] == kind]
    print(kind, 'median', round(float(np.median(r))), 'mean', round(float(np.mean(r))), 'p90', round(float(np.percentile(r, 90))))
print('houses mean', {k: round(float(np.mean([h[k] for h in houses]))) for k in ('cd', 'cut', 'relay')})

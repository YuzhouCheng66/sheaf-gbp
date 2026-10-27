"""a10_struct.py (A10): a STRUCTURE-AWARE lossless code of what an agent saw, still voxel-exact, no learning.
Facts of the v420 world used (mapping_3d_storey_data_v420.rasterize): the floor slab and the ceiling slab are full planes (heights from
wall_h: 16 bit per house, DL convention); wall columns are occupied over the WHOLE height; door cells carry a lintel from 0.1 + door_h
up; furniture never shares a wall column. For first-hit rays (observe()), the agent's whole ternary map is then EXACTLY
raycast(occupied set O, poses) with O = slab planes + every SEEN wall column (full height, lintel if a door cell) + the furniture
voxels the agent hit (V_f = own == 1 & obj): every true first hit lies in O, and every voxel of O is truly occupied, so the
decoder's marching stops exactly where the sensor's did. Code = 16 bit + 2-D map W2 of seen wall columns + door flag per W2 cell +
3-D furniture hits V_f (coded only outside slab layers and W2 columns, the decoder knows where those are) + poses (float32, 24 B/agent).
Also the INNOVATION version for every directed tree edge (receiver side R): new W2 cells (only cells with W2_R = 0; context: R knows
any voxel of the column), their door flags, new furniture hits where R is unknown (reach flag as in a10_probe).
Exactness of the rendering needs the exact poses (float32: +24 B / agent); the demo json has only pose centres, so the rendering
itself is NOT re-run here -- the argument above is the proof; a fixture-based assertion is part of the proposed experiment."""
import glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import load, causal, kt_bits, NB, adjacency, side_of, union, reachable, fwd_flags  # noqa

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
POSE_B = 24.                                                                  # float32 (x, y) of 3 poses: the rendering must be bit-exact
NB2 = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, 1), (1, 2)]


def ctx2d(g):
    X, Y = g.shape; c = np.zeros(g.shape, np.int64)
    for dx, dy in NB2:
        sh = np.zeros(g.shape, np.int64)
        if dy >= 0:
            sh[dx:, dy:] = g[:X - dx, :Y - dy]
        else:
            sh[dx:, :dy] = g[:X - dx, -dy:]
        c = c * 2 + sh
    return c


def code_struct(V, obj, struct, slab, R=None, reach=None):
    """bytes of (W2, door flags, V_f) given an optional receiver-side ternary map R (innovation) and reach flags"""
    W2 = (V & struct).any(2)
    nonslab = ~slab.any((0, 1))                                               # z layers outside the floor / ceiling slabs
    full = struct[:, :, nonslab].all(2)                                       # full-height wall column vs lintel-only (door) column
    door = W2 & ~full
    Vf = V & obj
    wc = np.broadcast_to(W2[:, :, None], V.shape)
    sl = np.broadcast_to(slab.any((0, 1))[None, None], V.shape)
    if R is None:
        bW = kt_bits(W2.astype(np.int64), ctx2d(W2.astype(np.int64)), 2)[0]
        bD = kt_bits(door[W2].astype(np.int64), np.zeros(int(W2.sum()), np.int64), 2)[0]
        sel = ~wc & ~sl
        v = Vf.astype(np.int64)
        c = causal(v, 2) * 2 + np.broadcast_to(np.pad(W2, 1)[:-2, 1:-1][:, :, None] | np.pad(W2, 1)[1:-1, :-2][:, :, None], V.shape).astype(np.int64)
        bV = kt_bits(v[sel], c[sel], 2)[0]
    else:
        Rk = R >= 0
        W2R = ((R == 1) & struct).any(2)
        cand = ~W2R                                                           # cells the receiver side has not seen as wall
        colknown = Rk.any(2)
        bW = kt_bits(W2[cand].astype(np.int64), (ctx2d(W2.astype(np.int64)) * 2 + colknown)[cand], 2)[0]
        newW = W2 & ~W2R
        bD = kt_bits(door[newW].astype(np.int64), np.zeros(int(newW.sum()), np.int64), 2)[0]
        sel = ~wc & ~sl & ~Rk
        v = Vf.astype(np.int64)
        c = (causal(v, 2) * 8 + fwd_flags(Rk)) * 2 + reach.astype(np.int64)
        bV = kt_bits(v[sel], c[sel], 2)[0]
    return dict(W2=bW / 8, door=bD / 8, Vf=bV / 8, total=(bW + bD + bV) / 8 + 2 + (2 if R is None else 0))


rows = []
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]; adj = adjacency(m['edges'], n); pos = np.asarray(m['positions_m']); cell = m['cell_m']
    slab = gt['slab']; obj = gt['obj']; struct = gt['occ'] & ~gt['obj'] & ~gt['slab']
    U = own.max(0)
    for a in range(n):
        r = code_struct(own[a] == 1, obj, struct, slab); r.update(kind='own', house=m['house'], total=r['total'] + POSE_B); rows.append(r)
    r = code_struct(U == 1, obj, struct, slab); r.update(kind='union', house=m['house'], total=r['total'] + POSE_B * n); rows.append(r)
    for i, j in m['edges']:
        for s_, r_ in ((i, j), (j, i)):
            Ss = side_of(adj, s_, r_); R = union(own, side_of(adj, r_, s_))
            reach = np.zeros(R.shape, bool)
            for a in Ss:
                reach |= reachable(R == 1, cell, pos[a])
            r = code_struct(U == 1, obj, struct, slab, R, reach); r.update(kind='edge', house=m['house'], nS=len(Ss), total=r['total'] + 2 + POSE_B * len(Ss)); rows.append(r)
    print(m['house'], flush=True)
json.dump(rows, open(os.path.join(HERE, 'a10_struct.json'), 'w'), indent=1)
for kind in ('own', 'union', 'edge'):
    r = [x for x in rows if x['kind'] == kind]
    print(kind, {k: round(float(np.median([x[k] for x in r])), 1) for k in ('W2', 'door', 'Vf', 'total')}, 'mean total', round(float(np.mean([x['total'] for x in r]))))

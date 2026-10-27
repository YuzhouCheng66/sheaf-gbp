"""a10_vctx.py (A10): is the 9-neighbour KT coder the floor for the visible-surface set V? Cheap hand-made context upgrades on the
48 own maps and 6 union maps (exact adaptive KT code lengths, + pose bytes) -- a lower bound on what a LEARNED prior would add.
  nb9        : 9 causal neighbours (the a10 baseline)
  nb13       : + 4 more causal neighbours (x-1,y+1), (x-1,z+1), (y-1,z+1), (x-1,y-1,z-1)
  nb9_z      : nb9 x the voxel's height level (32) -- floor / ceiling / furniture-top heights are shared across the house
  nb9_d      : nb9 x distance-to-nearest-own-pose bucket (public poses: < 1.5, < 3, < 4.5, < 5.2 m, beyond range)
  nb13_z_d   : all of it
  col2       : two-pass column code: first the 2-D map of 'column has any V voxel' (9-nbr 2-D context), then the z-profiles of the
               non-empty columns only (nb9_z context); the decoder skips empty columns."""
import glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import load, causal, kt_bits, NB, targets  # noqa

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
NB13 = NB + [(1, -1, 0), (1, 0, -1), (0, 1, -1), (1, 1, 1)]
POSE_B = 44 / 8


def causal_any(g, base, nb):
    """causal context with possibly negative offsets (still earlier in C raster order); out-of-grid reads 0"""
    X, Y, Z = g.shape
    c = np.zeros(g.shape, np.int64)
    for dx, dy, dz in nb:
        sh = np.zeros(g.shape, np.int64)
        xs = slice(dx, X) if dx >= 0 else slice(0, X + dx); xd = slice(0, X - dx) if dx >= 0 else slice(-dx, X)
        ys = slice(dy, Y) if dy >= 0 else slice(0, Y + dy); yd = slice(0, Y - dy) if dy >= 0 else slice(-dy, Y)
        zs = slice(dz, Z) if dz >= 0 else slice(0, Z + dz); zd = slice(0, Z - dz) if dz >= 0 else slice(-dz, Z)
        sh[xs, ys, zs] = g[xd, yd, zd]
        c = c * base + sh
    return c


def dist_bucket(shape, cell, poses):
    X, Y, Z = shape
    xs = (np.arange(X) + .5) * cell[0]; ys = (np.arange(Y) + .5) * cell[1]; zs = (np.arange(Z) + .5) * cell[2]
    P = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1)
    d = np.full(shape, np.inf)
    for p in poses:
        d = np.minimum(d, np.linalg.norm(P - np.asarray(p)[None, None, None], axis=-1))
    return np.digitize(d, [1.5, 3.0, 4.5, 5.2]).astype(np.int64)


def codes(V, cell, poses):
    v = V.astype(np.int64); Z = V.shape[2]
    zlev = np.broadcast_to(np.arange(Z)[None, None], V.shape).astype(np.int64)
    db = dist_bucket(V.shape, cell, poses)
    c9 = causal(v, 2); c13 = causal_any(v, 2, NB13)
    out = dict(nb9=kt_bits(v, c9, 2)[0], nb13=kt_bits(v, c13, 2)[0], nb9_z=kt_bits(v, c9 * Z + zlev, 2)[0],
               nb9_d=kt_bits(v, c9 * 5 + db, 2)[0], nb13_z_d=kt_bits(v, (c13 * Z + zlev) * 5 + db, 2)[0])
    col = V.any(2).astype(np.int64)                                     # pass 1: 2-D occupancy of columns
    c2 = np.zeros_like(col)
    for dx, dy in ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)):
        sh = np.zeros_like(col)
        if dy >= 0:
            sh[dx:, dy:] = col[:col.shape[0] - dx, :col.shape[1] - dy]
        else:
            sh[dx:, :dy] = col[:col.shape[0] - dx, -dy:]
        c2 = c2 * 2 + sh
    b1 = kt_bits(col, c2, 2)[0]
    sel = np.broadcast_to(col[:, :, None].astype(bool), V.shape)          # pass 2: profiles of non-empty columns only
    b2 = kt_bits(v[sel], (c9 * Z + zlev)[sel], 2)[0]
    out['col2'] = b1 + b2
    return {k: b / 8 + 2 + POSE_B * len(poses) for k, b in out.items()}


if __name__ == '__main__':
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
        m, own, pred, gt = load(path); n = own.shape[0]; cell = m['cell_m']; pos = m['positions_m']
        for a in range(n):
            rows.append(dict(kind='own', house=m['house'], agent=a, **codes(own[a] == 1, cell, [pos[a]])))
        U = own.max(0)
        rows.append(dict(kind='union', house=m['house'], **codes(U == 1, cell, pos)))
        print(m['house'], {k: round(v) for k, v in rows[-1].items() if k not in ('kind', 'house')}, flush=True)
    json.dump(rows, open(os.path.join(HERE, 'a10_vctx.json'), 'w'), indent=1)
    for kind in ('own', 'union'):
        r = [x for x in rows if x['kind'] == kind]
        print(kind, {k: round(float(np.median([x[k] for x in r]))) for k in ('nb9', 'nb13', 'nb9_z', 'nb9_d', 'nb13_z_d', 'col2')})

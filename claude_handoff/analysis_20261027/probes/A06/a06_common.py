"""A06 common helpers (Claude A06, 2026-10-27): load demo houses, build targets, furniture region, part-thickness classes.
CPU only. Uses ../../rate_baselines_lib.py (load, targets, metrics, ctx_bytes, down_up) unchanged."""
import glob, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, LIB)
import rate_baselines_lib as RB  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(LIB))
DATA = os.path.join(ROOT, 'demo_web3d', 'data')
DIMS = RB.DIMS


def dil6(mask):
    """3D 6-neighbourhood dilation of a bool [X, Y, Z] array (same as metrics3d's dilate6)."""
    return RB.dilate6(torch.from_numpy(mask.reshape(1, -1)), DIMS)[0].numpy().reshape(DIMS)


def opening_box(m, k):
    """binary opening of bool m with an all-ones box of size k = (kx, ky, kz): voxels covered by some fitting box."""
    X, Y, Z = m.shape; kx, ky, kz = k
    fit = np.ones((X - kx + 1, Y - ky + 1, Z - kz + 1), bool)
    for dx in range(kx):
        for dy in range(ky):
            for dz in range(kz):
                fit &= m[dx:dx + X - kx + 1, dy:dy + Y - ky + 1, dz:dz + Z - kz + 1]
    out = np.zeros_like(m)
    for dx in range(kx):
        for dy in range(ky):
            for dz in range(kz):
                out[dx:dx + X - kx + 1, dy:dy + Y - ky + 1, dz:dz + Z - kz + 1] |= fit
    return out


def houses():
    return sorted(glob.glob(os.path.join(DATA, 'house_*.json')))


def scene(path):
    """everything one house needs: own maps, preds per hop, targets per hop, gt masks, region, part classes."""
    m, own, pred, gt = RB.load(path)
    occ, slab, obj = gt['occ'], gt['slab'], gt['obj']
    struct = occ & ~obj & ~slab
    region = ~slab & ~dil6(struct)
    bulk = opening_box(obj, (2, 2, 2))
    thin_xy = obj & ~opening_box(obj, (2, 2, 1))          # thinner than 2 voxels horizontally: legs, panels, poles, chair backs, headboards
    thin_z = obj & ~opening_box(obj, (1, 1, 2))           # one voxel thick vertically: table tops, shelves, seats, lamp bases
    tgt = {k: RB.targets(own, m['reached'][m['hops'].index(k)]) for k in m['hops']}
    return dict(m=m, own=own, pred=pred, tgt=tgt, occ=occ, slab=slab, obj=obj, struct=struct, region=region,
                bulk=bulk, thin=obj & ~bulk, thin_xy=thin_xy, thin_z=thin_z & ~thin_xy)


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def furn_f1_map(p, t, region):
    """furniture-region F1 of one map (same as metrics3d.furn_f1): p, t int8 [X, Y, Z]."""
    known = t >= 0
    po = (p == 1) & known & region; to = (t == 1) & known & region
    tp = int((po & to).sum()); fp = int((po & ~to).sum()); fn = int((~po & to).sum())
    return (f1(tp, fp, fn) if tp + fp + fn else None), tp, fp, fn


def metrics_rows(preds, tgts, S):
    """official metric rows (surface, var, furn) via rate_baselines_lib.metrics for stacked int8 [n, X, Y, Z]."""
    n = preds.shape[0]
    slab = torch.from_numpy(S['slab'].reshape(-1)); obj = torch.from_numpy(S['obj'].reshape(-1)); st = torch.from_numpy(S['struct'].reshape(-1))
    return RB.metrics(torch.from_numpy(preds.reshape(n, -1)), torch.from_numpy(tgts.reshape(n, -1)), slab, obj, st)


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None

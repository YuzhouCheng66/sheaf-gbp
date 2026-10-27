"""A04 probe P2 (Claude, 2026-10-27): what does per-stalk SELECTION (max-key lattice join, one source wins a whole stalk) lose against
the exact UNION of observations (what a join-homomorphic code would carry), with PERFECT per-source codes?
For each R8 target (union of reached own maps) and each stalk layout (block of bx x by x bz voxels), every block takes the content of the
reached source with the largest known-voxel count in it (coverage key; the learned q of the 2D line had Spearman .91 with coverage);
the composed map is scored against the true union with the training metric (surface / variable / furniture-region F1)."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, targets, metrics, gt_masks, HERE  # noqa: E402

LAYOUTS = {'16x16x1': (8, 8, 32), '32x32x1': (4, 4, 32), '64x64x1': (2, 2, 32), '32x32x4': (4, 4, 8), '64x64x4': (2, 2, 8), '128x128x1': (1, 1, 32)}
res = {k: [] for k in LAYOUTS}; lost = {k: [] for k in LAYOUTS}
t0 = time.time()
for path in houses():
    m, own, pred, gt = load(path); n = own.shape[0]
    slab, obj, struct = gt_masks(gt)
    R8 = np.asarray(m['reached'][m['hops'].index(8)]).astype(bool)
    T8 = targets(own, R8)
    X, Y, Z = own.shape[1:]
    for name, (bx, by, bz) in LAYOUTS.items():
        S = np.empty_like(T8)
        for a in range(n):
            src = own[R8[a]]                                                    # [r, X, Y, Z]
            r = src.shape[0]
            blk = src.reshape(r, X // bx, bx, Y // by, by, Z // bz, bz)
            cov = (blk >= 0).sum((2, 4, 6))                                     # [r, nbx, nby, nbz]
            win = np.argmax(cov - np.arange(r)[:, None, None, None] * 1e-6, 0)  # ties -> lowest index (a lattice with a fixed tie order)
            up = win.repeat(bx, 0).repeat(by, 1).repeat(bz, 2)
            S[a] = np.take_along_axis(src, up[None], 0)[0]
            t = T8[a]; known = t >= 0
            lost[name].append(dict(known_lost=float(((S[a] < 0) & known).sum() / max(known.sum(), 1)),
                                   occ_lost=float(((S[a] != 1) & (t == 1)).sum() / max((t == 1).sum(), 1)),
                                   furn_occ_lost=float(((S[a] != 1) & (t == 1) & gt['obj']).sum() / max(((t == 1) & gt['obj']).sum(), 1))))
        res[name] += metrics(torch.from_numpy(S.reshape(n, -1)), torch.from_numpy(T8.reshape(n, -1)), slab, obj, struct)
    print(m['house'], f'{time.time() - t0:.0f}s', flush=True)


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


S = {name: dict(**{k: mean([r[k] for r in res[name]]) for k in ('surface_f1', 'var_occ_f1', 'furn_f1')},
                **{k: mean([r[k] for r in lost[name]]) for k in ('known_lost', 'occ_lost', 'furn_occ_lost')}) for name in LAYOUTS}
print(json.dumps(S, indent=1))
json.dump(S, open(os.path.join(HERE, 'p2_selection_vs_union.json'), 'w'), indent=1)

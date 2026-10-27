"""a10_bitsplit.py (A10): where do the lossless bits of the visible-surface code V go? Static per-voxel cost of the KT-9 V coder
(-log2 of the final-count KT estimate in the voxel's context; sums to the adaptive length minus the learning term) split by region:
slab (floor + ceiling), structure (walls / lintels, dilated by 1 as in the metric), furniture region (the rest), and inside the
furniture region by height band (0-0.3, 0.3-0.9, 0.9-1.5, > 1.5 m) and by 'thin' voxels (furniture V voxels with <= 1 occupied
4-neighbour in the same z slice: legs, shelf edges, poles). Own maps (48) and the 6 unions. Also the share of V voxels per region."""
import glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import load, causal, dilate6  # noqa

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))


def static_costs(V):
    v = V.astype(np.int64); c = causal(v, 2).reshape(-1); s = v.reshape(-1)
    u, inv = np.unique(c, return_inverse=True)
    cnt = np.bincount(inv * 2 + s, minlength=len(u) * 2).reshape(-1, 2).astype(np.float64)
    p = (cnt + .5) / (cnt.sum(1, keepdims=True) + 1.)
    return (-np.log2(p[inv, s])).reshape(V.shape)


rows = []
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]
    slab = gt['slab']; struct = gt['occ'] & ~gt['obj'] & ~gt['slab']; sd = dilate6(struct); furn = ~slab & ~sd
    zc = (np.arange(32) + .5) * 0.1 - 0.1                              # height above the floor top (m)
    band = np.digitize(zc, [0.3, 0.9, 1.5])                           # 0: < .3, 1: .3-.9, 2: .9-1.5, 3: > 1.5
    B = np.broadcast_to(band[None, None], slab.shape)
    for kind, maps in (('own', [own[a] for a in range(n)]), ('union', [own.max(0)])):
        for t in maps:
            V = t == 1; cst = static_costs(V); tot = cst.sum()
            o = V.astype(np.int64); nb = np.zeros_like(o)
            nb[1:] += o[:-1]; nb[:-1] += o[1:]; nb[:, 1:] += o[:, :-1]; nb[:, :-1] += o[:, 1:]
            thin = V & furn & (nb <= 1)
            r = dict(kind=kind, house=m['house'], total_B=tot / 8,
                     share_slab=cst[slab].sum() / tot, share_struct=cst[sd & ~slab].sum() / tot, share_furn=cst[furn].sum() / tot,
                     vox_share_slab=float((V & slab).sum() / V.sum()), vox_share_struct=float((V & sd & ~slab).sum() / V.sum()), vox_share_furn=float((V & furn).sum() / V.sum()),
                     share_thin=cst[thin].sum() / tot, vox_share_thin=float(thin.sum() / V.sum()))
            for b in range(4):
                r[f'furn_band{b}'] = cst[furn & (B == b)].sum() / tot
            rows.append(r)
json.dump(rows, open(os.path.join(HERE, 'a10_bitsplit.json'), 'w'), indent=1)
for kind in ('own', 'union'):
    r = [x for x in rows if x['kind'] == kind]
    print(kind, {k: round(float(np.mean([x[k] for x in r])), 3) for k in r[0] if k not in ('kind', 'house')})

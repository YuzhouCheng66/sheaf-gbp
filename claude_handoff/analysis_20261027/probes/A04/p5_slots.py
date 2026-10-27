"""A04 probe P5 (Claude, 2026-10-27): object-slot stalks on a PUBLIC anchor grid (spatial hashing instead of learned association).
Instances = 26-connected components of the GT furniture mask (touching objects merge; good enough for rates).  Per instance and per
agent that sees >= 5 of its voxels (own map occupied): anchor cell of the observed part on a G x G public grid by three rules
(centroid, footprint-bbox centre, footprint min corner).  Reports (1) how often all viewers agree on the anchor (else the lattice keeps
duplicates at different indices), (2) best single view / union of views per instance (what per-slot max-key selection keeps),
(3) shape oracles on the R8 targets in the furniture region: every instance visible in the target replaced by the 3-D box of its
observed union (box), by per-column [z_min, z_max] fill (interval), or kept exact (reference) -> how much shape a slot payload must carry."""
import json, os, sys, time
import numpy as np, torch
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, targets, metrics, gt_masks, HERE  # noqa: E402

GRIDS = (16, 32, 64)
agree = {(g, r): [] for g in GRIDS for r in ('centroid', 'bbox_centre', 'min_corner')}
ndist = {(g, r): [] for g in GRIDS for r in ('centroid', 'bbox_centre', 'min_corner')}
cover, nvis_own, nvis_u = [], [], []
oracle = {k: [] for k in ('box', 'interval')}
t0 = time.time()
for path in houses():
    m, own, pred, gt = load(path); n = own.shape[0]
    slab, obj, struct = gt_masks(gt)
    R8 = np.asarray(m['reached'][m['hops'].index(8)]).astype(bool); T8 = targets(own, R8)
    lab, K = ndimage.label(gt['obj'], structure=np.ones((3, 3, 3), bool))
    objs = ndimage.find_objects(lab)
    seen = np.zeros((n, K + 1), np.int64)
    for a in range(n):
        seen[a] = np.bincount(lab[own[a] == 1], minlength=K + 1)
    nvis_own += [int((seen[a, 1:] >= 5).sum()) for a in range(n)]
    for k in range(1, K + 1):
        sl = objs[k - 1]; sub = lab[sl] == k
        viewers = [a for a in range(n) if seen[a, k] >= 5]
        anchors = {key: [] for key in agree}
        for a in viewers:
            pts = np.argwhere(sub & (own[a][sl] == 1)) + np.array([s.start for s in sl])
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean(); x0, y0 = pts[:, 0].min(), pts[:, 1].min(); x1, y1 = pts[:, 0].max(), pts[:, 1].max()
            for g in GRIDS:
                s = 128 // g
                anchors[(g, 'centroid')].append((int(cx // s), int(cy // s)))
                anchors[(g, 'bbox_centre')].append((int(((x0 + x1) / 2) // s), int(((y0 + y1) / 2) // s)))
                anchors[(g, 'min_corner')].append((int(x0 // s), int(y0 // s)))
        if len(viewers) >= 2:
            for key, v in anchors.items():
                agree[key].append(len(set(v)) == 1); ndist[key].append(len(set(v)))
    for a in range(n):
        reached = np.nonzero(R8[a])[0]
        vis = [k for k in range(1, K + 1) if seen[reached, k].max() >= 1]
        nvis_u.append(sum(1 for k in vis if seen[reached, k].max() >= 5))
        box = T8[a].copy(); itv = T8[a].copy()
        U = T8[a] == 1
        for k in vis:
            sl = objs[k - 1]; sub = (lab[sl] == k) & U[sl]
            if not sub.any():
                continue
            un = int(sub.sum()); best = int(seen[reached, k].max())
            cover.append((best, un))
            pts = np.argwhere(sub)
            lo = pts.min(0); hi = pts.max(0) + 1
            o = np.array([s.start for s in sl])
            box[o[0] + lo[0]:o[0] + hi[0], o[1] + lo[1]:o[1] + hi[1], o[2] + lo[2]:o[2] + hi[2]] = 1
            col = sub.any(2)
            zz = np.arange(sub.shape[2])
            zmin = np.where(sub, zz[None, None], 999).min(2); zmax = np.where(sub, zz[None, None], -1).max(2)
            fill = col[:, :, None] & (zz[None, None] >= zmin[:, :, None]) & (zz[None, None] <= zmax[:, :, None])
            itv[sl][fill] = 1
        for name, P in (('box', box), ('interval', itv)):
            oracle[name] += metrics(torch.from_numpy(P.reshape(1, -1)), torch.from_numpy(T8[a].reshape(1, -1)), slab, obj, struct)
    print(m['house'], K, 'instances', f'{time.time() - t0:.0f}s', flush=True)

S = dict(instances_visible_per_own_median=float(np.median(nvis_own)), instances_visible_per_union8_median=float(np.median(nvis_u)),
         best_single_view_over_union_voxel_weighted=float(sum(b for b, u in cover) / max(sum(u for b, u in cover), 1)),
         best_single_view_over_union_instance_median=float(np.median([b / u for b, u in cover])),
         anchor_full_agreement={f'{g}_{r}': float(np.mean(agree[(g, r)])) for g, r in agree},
         anchor_distinct_mean={f'{g}_{r}': float(np.mean(ndist[(g, r)])) for g, r in ndist},
         multi_view_instances=len(agree[(32, 'centroid')]),
         shape_oracles_R8={k: {kk: float(np.mean([r[kk] for r in v if r[kk] is not None])) for kk in ('surface_f1', 'var_occ_f1', 'furn_f1')} for k, v in oracle.items()})
print(json.dumps(S, indent=1))
json.dump(S, open(os.path.join(HERE, 'p5_slots.json'), 'w'), indent=1)

"""A05 debate probe 8 (CPU, demo data, exact poses): what does A04's shape oracle look like at the resolution ΔV-Sheaf actually sends?
A04 P5 (R8, furniture region, oracle known mask): 'box' = one AABB per 26-connected furniture INSTANCE (touching objects merge, e.g. a
dining table with its chairs) -> furn .53; 'interval' = per 1x1 VOXEL column [z_min, z_max] of the instance's observed hits -> .93.
ΔV-Sheaf sends 64x64 columns (2x2 voxels). Here, on the same targets:
  box_inst   : A04's per-instance AABB (replication)
  box_prim   : per-PRIMITIVE boxes (visible boxes of the greedy decomposition of gt_obj, p4) -- the box granularity that is exact
  itv_1x1    : A04's interval oracle (replication)
  itv_2x2    : the same interval per 2x2 column, filled over all 4 sub-columns (what a 64^2 stalk with one interval can draw)
  itv_2x2_fp : per 2x2 column one interval + the exact 4-bit sub-column footprint (the extra information a 64^2 stalk would need)
each scored two ways: 'direct' (A04's protocol: oracle known mask, predicted occupancy = target with the fill) and 'raycast' (exact
structure layer + fill, ray-cast from the true poses, as every exact-codec design decodes)."""
import glob, json, os, sys, time
import numpy as np, torch
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from rate_baselines_lib import load, targets, metrics, DIMS  # noqa: E402
from p2b_raycast_decode import ray_table, cast, struct_parts, fpr  # noqa: E402
from p4_box_tokens import greedy_boxes, raster  # noqa: E402


def interval_fill(Hf_inst, b):
    """per b x b column block: [zmin, zmax] of the hits, filled over the whole block (b = 1: per voxel column)"""
    X, Y, Z = Hf_inst.shape
    col = Hf_inst.reshape(X // b, b, Y // b, b, Z).any((1, 3))                       # [X/b, Y/b, Z]
    zz = np.arange(Z)
    has = col.any(2)
    zmin = np.where(col, zz[None, None], 999).min(2); zmax = np.where(col, zz[None, None], -1).max(2)
    fill = has[:, :, None] & (zz[None, None] >= zmin[:, :, None]) & (zz[None, None] <= zmax[:, :, None])
    return fill.repeat(b, 0).repeat(b, 1)


def interval_fill_fp(Hf_inst, b):
    """one interval per b x b block, drawn only in the sub-columns that contain a hit (exact footprint)"""
    full = interval_fill(Hf_inst, b)
    return full & Hf_inst.any(2)[:, :, None]


if __name__ == '__main__':
    poses = json.load(open(os.path.join(HERE, 'poses.json')))
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
        t0 = time.time()
        m, own, pred, gt = load(path); n = own.shape[0]; key = m['house']
        traj = np.asarray(poses[key]['trajectory_m']); cell = tuple(poses[key]['cell']); lk = poses[key]['lintel_k']
        slab = gt['slab']; obj = gt['obj']; occ = gt['occ']; struct = occ & ~obj & ~slab
        slab_t = torch.from_numpy(slab.reshape(-1)); obj_t = torch.from_numpy(obj.reshape(-1)); struct_t = torch.from_numpy(struct.reshape(-1))
        lab, K = ndimage.label(obj, structure=np.ones((3, 3, 3), bool))
        world_boxes = greedy_boxes(obj).reshape(-1, 6)
        R8 = np.asarray(m['reached'][m['hops'].index(8)], bool); T8 = targets(own, R8)
        jobs = [(0, a, own[a], [a]) for a in range(n)] + [(8, 0, T8[0], list(np.nonzero(R8[0])[0]))]     # one component per demo house
        for hop, a, t, agents in jobs:
            tabs = [ray_table(traj[k, b_], cell) for b_ in agents for k in range(traj.shape[0])]
            Hf = (t == 1) & obj
            Gs, _, _ = struct_parts(t == 1, gt, lk)
            fills = {}
            # per-instance AABB (A04 'box') and per-instance intervals
            bi = np.zeros(DIMS, bool); i1 = np.zeros(DIMS, bool); i2 = np.zeros(DIMS, bool); i2f = np.zeros(DIMS, bool)
            for k in np.unique(lab[Hf]):
                if k == 0:
                    continue
                sub = Hf & (lab == k)
                pts = np.argwhere(sub); lo = pts.min(0); hi = pts.max(0) + 1
                bi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
                i1 |= interval_fill(sub, 1); i2 |= interval_fill(sub, 2); i2f |= interval_fill_fp(sub, 2)
            vis = np.array([Hf[x0:x1, y0:y1, z0:z1].any() for x0, x1, y0, y1, z0, z1 in world_boxes], bool)
            fills = dict(box_inst=bi, box_prim=raster(world_boxes[vis], DIMS), itv_1x1=i1, itv_2x2=i2, itv_2x2_fp=i2f)
            for name, F in fills.items():
                F = F & ~struct & ~slab
                P = t.copy(); P[F] = 1                                                   # A04 protocol: oracle known, fill drawn as occupied
                md = metrics(torch.from_numpy(P.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), slab_t, obj_t, struct_t)[0]
                Pr = cast(Gs | F, tabs)
                mr = metrics(torch.from_numpy(Pr.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), slab_t, obj_t, struct_t)[0]
                rows.append(dict(house=key, hop=hop, agent=a, oracle=name, direct_furn=md['furn_f1'], direct_surf=md['surface_f1'],
                                 ray_furn=mr['furn_f1'], ray_surf=mr['surface_f1'], ray_fpr=fpr(Pr, t)))
            del tabs
        print(key, f'{time.time() - t0:.0f}s', flush=True)
    S = {}
    for hop in (0, 8):
        for name in ('box_inst', 'box_prim', 'itv_1x1', 'itv_2x2', 'itv_2x2_fp'):
            r = [x for x in rows if x['hop'] == hop and x['oracle'] == name]
            S[f"{'K0' if hop == 0 else 'R8'} {name}"] = {k: float(np.mean([x[k] for x in r if x[k] is not None])) for k in ('direct_furn', 'direct_surf', 'ray_furn', 'ray_surf', 'ray_fpr')} | dict(n=len(r))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p8_interval_oracle.json'), 'w'), indent=1)
    for k, v in S.items():
        print(f'{k:16s} ' + ' '.join(f'{kk} {vv:.3f}' if isinstance(vv, float) else f'{kk} {vv}' for kk, vv in v.items()))

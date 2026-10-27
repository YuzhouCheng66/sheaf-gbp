"""p4_rotation.py (A07): how much does the synthetic world's AXIS ALIGNMENT flatter voxel coders? Rotate each demo house (occupancy,
furniture, agents and their pose jitters, all from sim_cache) about the chart centre by theta (nearest-neighbour resampling, outside the
source chart = solid, as the v420 margin is), re-simulate the own maps with the v420 sensor, and measure the lossless costs (ctx, H + poses,
gzip) and the classical ternary 2x2x1 tau .5 down-up (bytes, surface / furniture-region F1). The world content and its parametric
description length are unchanged by the rotation; only the grid alignment changes. Agents whose rotated pose voxel is not free are dropped
(the same agents at every angle)."""
import json, math, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a07_lib import observe, ctx_ter, ctx_bin, gz, down_up, score, mean_rows, POSE_BYTES

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, 'sim_cache')
ANGLES = (0, 15, 30, 45)


def rotate(z, theta):
    occ, obj, slab = z['gt_occ'], z['gt_obj'], z['gt_slab']; X, Y, Zd = occ.shape; c = float(z['cell'][0]); L = X * c; ctr = L / 2
    xs = (np.arange(X) + .5) * c - ctr; xx, yy = np.meshgrid(xs, xs, indexing='ij'); t = math.radians(theta)
    sx = math.cos(t) * xx + math.sin(t) * yy; sy = -math.sin(t) * xx + math.cos(t) * yy
    i = np.floor((sx + ctr) / c).astype(int); j = np.floor((sy + ctr) / c).astype(int); inb = (i >= 0) & (i < X) & (j >= 0) & (j < Y)
    ic, jc = np.clip(i, 0, X - 1), np.clip(j, 0, Y - 1)
    occ_r = np.where(inb[:, :, None], occ[ic, jc], True); obj_r = np.where(inb[:, :, None], obj[ic, jc], False)
    R = np.array([[math.cos(t), -math.sin(t), 0.], [math.sin(t), math.cos(t), 0.], [0., 0., 1.]]); centre = np.array([ctr, ctr, 0.])
    poses = (z['poses'] - centre) @ R.T + centre
    return occ_r, obj_r, slab, poses


def house(name):
    t0 = time.time(); z = np.load(os.path.join(CACHE, name + '.npz')); cell = z['cell']; n = z['own'].shape[0]
    rot = {th: rotate(z, th) for th in ANGLES}
    ok = []
    for a in range(n):
        good = True
        for th in ANGLES:
            occ_r, _, _, P = rot[th]
            for p in P[a]:
                ijk = np.floor(p / cell).astype(int)
                if not (np.all(ijk >= 0) and np.all(ijk < occ_r.shape)) or occ_r[tuple(ijk)]:
                    good = False
        if good:
            ok.append(a)
    out = dict(house=name, agents=ok, angles={})
    for th in ANGLES:
        occ_r, obj_r, slab, P = rot[th]
        own = np.stack([observe(occ_r.astype(np.uint8), cell, P[a]) for a in ok])
        gt = dict(occ=occ_r, slab=slab, obj=obj_r & ~slab)
        rows = [dict(ctx=ctx_ter(m), H=ctx_bin(m == 1) + POSE_BYTES, gzip=gz(m), known=float((m >= 0).mean())) for m in own]
        pd, bd = [], []
        for m in own:
            cc, uu = down_up(m, (2, 2, 1), .5); pd.append(uu); bd.append(ctx_ter(cc))
        sc = score(np.stack(pd), own, gt)
        out['angles'][str(th)] = dict(rows=rows, du221_bytes=bd, du221_rows=sc)
    print(name, ok, f'{time.time() - t0:.0f}s', flush=True)
    return out


if __name__ == '__main__':
    names = sorted(f[:-4] for f in os.listdir(CACHE) if f.endswith('.npz'))
    with Pool(2) as pool:
        res = pool.map(house, names)
    S = {}
    for th in ANGLES:
        rows = [r for h in res for r in h['angles'][str(th)]['rows']]; sc = [r for h in res for r in h['angles'][str(th)]['du221_rows']]
        S[str(th)] = dict(n=len(rows), ctx=float(np.median([r['ctx'] for r in rows])), H_poses=float(np.median([r['H'] for r in rows])),
                          gzip=float(np.median([r['gzip'] for r in rows])), known=float(np.mean([r['known'] for r in rows])),
                          du221_bytes=float(np.median([b for h in res for b in h['angles'][str(th)]['du221_bytes']])),
                          du221_surface=mean_rows(sc, 'surface_f1'), du221_furn=mean_rows(sc, 'furn_f1'))
    json.dump(dict(summary=S, houses=res), open(os.path.join(HERE, 'p4_rotation.json'), 'w'), indent=1)
    for th, s in S.items():
        print(f"theta {th:>2}: n {s['n']} ctx {s['ctx']:6.0f} H+p {s['H_poses']:6.0f} gzip {s['gzip']:6.0f} known {s['known']:.3f} | du221 {s['du221_bytes']:5.0f} B surf {s['du221_surface']:.3f} furn {s['du221_furn']:.3f}")

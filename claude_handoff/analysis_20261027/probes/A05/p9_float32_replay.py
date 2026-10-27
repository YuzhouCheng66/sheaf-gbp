"""A05 debate probe 9 (CPU, demo data, exact poses): is 'F1 = 1 by exact replay' robust to how the RECEIVER marches its rays?
Every exact codec (A01 LL, A02 GenFact, A03 RAY-SHEAF, A06 ProgSheaf, A07 ELS, A08 RC, A10 EVS, my structured lossless) decodes by
re-running the generator's first-hit march: idx = floor((p + r * s * ds) / cell) in float64. A GPU decoder would naturally do this in
float32. Here the receiver replays the agent's own hit set H (which is exact in float64, p2a) with (a) float64 arithmetic and float32
poses, (b) float32 arithmetic throughout; mismatching voxels vs the exported map and the resulting F1 / FPR are counted."""
import glob, json, math, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from rate_baselines_lib import load, metrics, DIMS  # noqa: E402
from p2b_raycast_decode import RAYS, cast, fpr  # noqa: E402


def ray_table_dtype(p, cell, dt, shape=DIMS):
    cellv = np.asarray(cell, np.float64); shp = np.asarray(shape)
    ds = 0.5 * float(cellv.min()); S = int(math.ceil(5.0 / ds))
    steps = (np.arange(1, S + 1, dtype=np.float64) * ds).astype(dt)
    rays = RAYS.astype(dt); pp = np.asarray(p).astype(dt); cc = cellv.astype(dt)
    out = np.empty((len(RAYS), S), np.int32)
    for c0 in range(0, len(RAYS), 8192):
        r = rays[c0:c0 + 8192]
        idx = np.floor((pp[None, None, :] + r[:, None, :] * steps[None, :, None]) / cc).astype(np.int64)
        inb = np.all((idx >= 0) & (idx < shp), 2)
        lin = (idx[..., 0] * shp[1] + idx[..., 1]) * shp[2] + idx[..., 2]
        out[c0:c0 + 8192] = np.where(inb, lin, -1)
    return out


if __name__ == '__main__':
    poses = json.load(open(os.path.join(HERE, 'poses.json')))
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json')))[:3]:
        t0 = time.time()
        m, own, pred, gt = load(path); n = own.shape[0]; key = m['house']
        traj = np.asarray(poses[key]['trajectory_m']); cell = tuple(poses[key]['cell'])
        slab = gt['slab']; obj = gt['obj']; occ = gt['occ']; struct = occ & ~obj & ~slab
        slab_t = torch.from_numpy(slab.reshape(-1)); obj_t = torch.from_numpy(obj.reshape(-1)); struct_t = torch.from_numpy(struct.reshape(-1))
        for a in range(n):
            t = own[a]; H = t == 1
            for mode in ('f64', 'f64_pose32', 'f32'):
                tabs = []
                for k in range(traj.shape[0]):
                    p = traj[k, a]
                    if mode == 'f64':
                        tabs.append(ray_table_dtype(p, cell, np.float64))
                    elif mode == 'f64_pose32':
                        tabs.append(ray_table_dtype(np.asarray(p, np.float32).astype(np.float64), cell, np.float64))
                    else:
                        tabs.append(ray_table_dtype(p, cell, np.float32))
                Pr = cast(H, tabs)
                mt = metrics(torch.from_numpy(Pr.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), slab_t, obj_t, struct_t)[0]
                rows.append(dict(house=key, agent=a, mode=mode, mismatch=int((Pr != t).sum()), mismatch_occ=int(((Pr == 1) != (t == 1)).sum()),
                                 known=int((t >= 0).sum()), fpr=fpr(Pr, t), **mt))
        print(key, f'{time.time() - t0:.0f}s', flush=True)
    S = {}
    for mode in ('f64', 'f64_pose32', 'f32'):
        r = [x for x in rows if x['mode'] == mode]
        S[mode] = dict(maps_exact=f"{sum(x['mismatch'] == 0 for x in r)}/{len(r)}", mismatch_median=float(np.median([x['mismatch'] for x in r])),
                       mismatch_max=int(max(x['mismatch'] for x in r)), mismatch_occ_median=float(np.median([x['mismatch_occ'] for x in r])),
                       surface_f1=float(np.mean([x['surface_f1'] for x in r])), furn_f1=float(np.mean([x['furn_f1'] for x in r if x['furn_f1'] is not None])),
                       fpr=float(np.mean([x['fpr'] for x in r])))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p9_float32_replay.json'), 'w'), indent=1)
    print(json.dumps(S, indent=1))

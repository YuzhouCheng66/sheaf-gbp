"""d1_float32_replay.py (A07, debate round): does 'lossless = hits + poses + public rays' survive a receiver that marches rays in
float32 (what a GPU kernel would do) instead of the generator's float64? Same re-simulated maps as sim_cache (float64 truth).
For each own map we replay its hit set H with (a) the exact float64 marcher (control: must be 0 mismatches), (b) the same marcher with
every array in float32 (poses, rays, step, cell), (c) float64 marching but poses stored as float32, and count mismatched voxels and
mismatched occupied voxels. One process, ~1 GB."""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from a07_lib import observe, fibonacci_sphere, SENSOR_RANGE

def observe_f32(occ, cell, poses, rays):
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float32); rays = rays.astype(np.float32)
    out = np.full(occ.shape, -1, np.int8)
    ds = np.float32(0.5) * np.float32(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / float(ds)))
    for p in poses:
        p = np.asarray(p, np.float32); alive = np.ones(len(rays), bool)
        for s in range(1, nsteps + 1):
            idx_all = np.floor((p[None] + rays * (np.float32(s) * ds)) / cell).astype(np.int64)
            inside = np.all((idx_all >= 0) & (idx_all < shape), 1) & alive
            if not inside.any():
                break
            ii = idx_all[inside]; hit = occ[ii[:, 0], ii[:, 1], ii[:, 2]] == 1
            fr = ii[~hit]; out[fr[:, 0], fr[:, 1], fr[:, 2]] = np.maximum(out[fr[:, 0], fr[:, 1], fr[:, 2]], 0)
            hh = ii[hit]; out[hh[:, 0], hh[:, 1], hh[:, 2]] = 1
            ai = np.nonzero(inside)[0]; alive[ai[hit]] = False; alive &= inside | ~alive
    return out

rays = fibonacci_sphere(65536); rows = []
for name in ('r1_i0', 'r2_i1', 'r3_i0'):
    z = np.load(os.path.join(os.path.dirname(HERE), 'sim_cache', name + '.npz')); own, poses, cell = z['own'], z['poses'], z['cell']
    for a in range(0, 8, 2):
        t0 = time.time(); H = (own[a] == 1).astype(np.uint8)
        r64 = observe(H, cell, poses[a], rays)
        r32 = observe_f32(H, cell, poses[a], rays)
        rp32 = observe(H, cell, poses[a].astype(np.float32).astype(np.float64), rays)
        rows.append(dict(house=name, agent=a, known=int((own[a] >= 0).sum()), f64_mismatch=int((r64 != own[a]).sum()),
                         f32_mismatch=int((r32 != own[a]).sum()), f32_occ_mismatch=int(((r32 == 1) != (own[a] == 1)).sum()),
                         pose32_mismatch=int((rp32 != own[a]).sum()), sec=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
S = {k: dict(median=float(np.median([r[k] for r in rows])), max=int(max(r[k] for r in rows))) for k in ('f64_mismatch', 'f32_mismatch', 'f32_occ_mismatch', 'pose32_mismatch', 'known')}
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'd1_float32_replay.json'), 'w'), indent=1); print(json.dumps(S, indent=1))

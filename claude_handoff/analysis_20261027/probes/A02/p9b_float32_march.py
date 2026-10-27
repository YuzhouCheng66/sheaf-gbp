"""A02 debate probe P9b (2026-10-27, CPU only, one process): is the 'exact' render decoder exact on float32 hardware?
The generator marches rays in float64 (floor((p + r * s * ds) / cell)).  A GPU decoder in float32 (or with float32 poses / rays) samples
slightly different points, so voxels near cell boundaries can flip.  For 2 agents in each of the 6 demo VAL houses (P1 poses) we
simulate the own map in float64 (the reference) and re-march the SAME hit set with (i) float32 poses only, (ii) everything in float32
(poses, ray directions, step, cell), and count voxels that differ.  Output: probes/A02/p9b_float32_march.json"""
import glob, json, math, os, sys, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from rate_baselines_lib import load  # noqa: E402
from p1_render_hits import observe, fibonacci_sphere  # noqa: E402


def observe_dtype(occ, cell, poses, rays, dt):
    """observe() with all arithmetic in dtype dt"""
    shape = np.array(occ.shape); cell = np.asarray(cell, dt); rays = rays.astype(dt); poses = np.asarray(poses, dt)
    out = np.full(occ.shape, -1, np.int8)
    ds = dt(0.5) * cell.min(); nsteps = int(math.ceil(5.0 / float(ds)))
    for p in poses:
        alive = np.ones(len(rays), bool)
        for s in range(1, nsteps + 1):
            idx = np.floor((p[None] + rays * (dt(s) * ds)) / cell).astype(np.int64)
            inside = np.all((idx >= 0) & (idx < shape), 1) & alive
            if not inside.any():
                break
            ii = idx[inside]
            hit = occ[ii[:, 0], ii[:, 1], ii[:, 2]] == 1
            fr = ii[~hit]; out[fr[:, 0], fr[:, 1], fr[:, 2]] = np.maximum(out[fr[:, 0], fr[:, 1], fr[:, 2]], 0)
            hh = ii[hit]; out[hh[:, 0], hh[:, 1], hh[:, 2]] = 1
            ai = np.nonzero(inside)[0]; alive[ai[hit]] = False
            alive &= inside | ~alive
    return out


if __name__ == '__main__':
    rays = fibonacci_sphere(65536); rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
        m, own, pred, gt = load(path); del pred, own
        occ = gt['occ'].astype(np.uint8); cell = tuple(m['cell_m'])
        for a in (0, 1):
            rng = np.random.default_rng(zlib.crc32(m['house'].encode()) * 16 + a)
            centre = np.asarray(m['positions_m'][a], np.float64); jit = rng.uniform(-.1, .1, (3, 3)); jit[:, 2] = 0.
            poses = centre[None] + jit
            ref = observe(occ, cell, poses, rays); H = (ref == 1).astype(np.uint8)
            r_pose32 = observe(H, cell, poses.astype(np.float32).astype(np.float64), rays)
            r_all32 = observe_dtype(H, cell, poses, rays, np.float32)
            row = dict(house=m['house'], agent=a, known=int((ref >= 0).sum()), occ=int(H.sum()),
                       mismatch_pose32=int((r_pose32 != ref).sum()), mismatch_all32=int((r_all32 != ref).sum()),
                       mismatch_all32_occ=int(((r_all32 == 1) != (ref == 1)).sum()))
            rows.append(row); print(row, flush=True)
    S = {k: dict(median=float(np.median([r[k] for r in rows])), max=float(np.max([r[k] for r in rows]))) for k in ('mismatch_pose32', 'mismatch_all32', 'mismatch_all32_occ', 'known', 'occ')}
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p9b_float32_march.json'), 'w'), indent=1)

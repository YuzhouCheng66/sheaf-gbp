"""A02 probe P1b (2026-10-27, CPU only; slim re-run of P1 after a memory error on the shared machine): pose precision needed for an exact
render.  agents 2 and 3 of houses r2_i1, r3_i0, r3_i1 (6; P1 covered the others): synthetic own map from 3 jittered poses (P1 sensor), then render(hits, poses with the jitter rounded to
1 mm / 0.1 mm / float32) and count the voxels that differ; the residual cost is the adaptive binary ctx code of the difference mask.
Output: probes/A02/p1b_pose_quant.json"""
import glob, json, os, sys, zlib
import numpy as np
from multiprocessing import get_context
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)


def job(args):
    import torch; torch.set_num_threads(1)
    from rate_baselines_lib import load
    from p1_render_hits import observe, fibonacci_sphere, ctx_bytes_binary
    path, a = args
    m, own, pred, gt = load(path); occ = gt['occ'].astype(np.uint8); cell = tuple(m['cell_m']); rays = fibonacci_sphere(65536)
    del own, pred
    rng = np.random.default_rng(zlib.crc32(m['house'].encode()) * 16 + a)
    centre = np.asarray(m['positions_m'][a], np.float64); jit = rng.uniform(-.1, .1, (3, 3)); jit[:, 2] = 0.
    syn = observe(occ, cell, centre[None] + jit, rays); hits = (syn == 1).astype(np.uint8)
    row = dict(house=m['house'], agent=a, identity=int((observe(hits, cell, centre[None] + jit, rays) != syn).sum()))
    for name, pq in (('1mm', centre[None] + np.round(jit / 1e-3) * 1e-3), ('0.1mm', centre[None] + np.round(jit / 1e-4) * 1e-4),
                     ('float32', (centre[None] + jit).astype(np.float32).astype(np.float64))):
        d = observe(hits, cell, pq, rays) != syn
        row[f'mismatch_{name}'] = int(d.sum()); row[f'resid_bytes_{name}'] = (ctx_bytes_binary(d) - 2) if d.any() else 0.
    print(row, flush=True)
    return row


if __name__ == '__main__':
    paths = sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json')))
    # sequential (the shared machine ran out of commit memory for worker processes); houses / agents not covered by P1
    rows = [job((p, a)) for p in paths if any(h in p for h in ('r2_i1', 'r3_i0', 'r3_i1')) for a in (2, 3)]
    S = {k: dict(median=float(np.median([r[k] for r in rows])), max=float(np.max([r[k] for r in rows]))) for k in rows[0] if k not in ('house', 'agent')}
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p1b_pose_quant.json'), 'w'), indent=1)

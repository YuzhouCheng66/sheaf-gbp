"""p0_sim.py (A07): re-simulate the 6 VAL demo houses' own maps with KNOWN poses (the demo does not carry the private jitter), so that
decoders that re-cast rays from poses can be scored without a pose mismatch. Same sensor as v420 (65,536 beams, 3 poses jittered
+-0.1 m around the demo agent centres, 5 m, occupied wins). Same seeds as p1_lossless (exactness verified there).
Writes sim_cache/<house>.npz: own [8,128,128,32] int8, poses [8,3,3], cell, gt_occ/slab/obj, reached8 [8,8], edges, chart_m."""
import os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a07_lib import load, observe, demo_paths

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim_cache')


def build(path):
    t0 = time.time()
    meta, own, pred, gt = load(path); n = own.shape[0]
    occ = gt['occ'].astype(np.uint8); cell = np.asarray(meta['cell_m'], np.float64)
    rng = np.random.default_rng(7000 + sum(map(ord, meta['house'])))
    pos = np.asarray(meta['positions_m'], np.float64)
    sims, poses = [], []
    for a in range(n):
        jit = rng.uniform(-.1, .1, (3, 3)); jit[:, 2] = 0.
        p = pos[a][None] + jit; poses.append(p); sims.append(observe(occ, cell, p))
    hi8 = meta['hops'].index(8)
    np.savez_compressed(os.path.join(CACHE, meta['house'] + '.npz'), own=np.stack(sims), poses=np.stack(poses), cell=cell, gt_occ=gt['occ'], gt_slab=gt['slab'],
                        gt_obj=gt['obj'], reached8=np.asarray(meta['reached'][hi8], bool), edges=np.asarray(meta['edges']), chart_m=meta['chart_m'], pos=pos)
    print(meta['house'], f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    os.makedirs(CACHE, exist_ok=True)
    with Pool(6) as pool:
        pool.map(build, demo_paths())

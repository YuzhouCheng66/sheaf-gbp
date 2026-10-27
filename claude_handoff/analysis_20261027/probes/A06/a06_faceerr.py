"""A06 debate probe (Claude A06, 2026-10-27): what per-face accuracy must a LEARNED world decoder reach for furniture F1 >= .85 when the
map is decoded by ray casting (RAYFIELD, W-Sheaf, PV-Sheaf, RayHead), versus a decoder that reads known/unknown directly?
World = gt_occ of a demo house; own maps re-simulated with the generator's observe() at the demo centres + the a06_raycast jitters (exact
poses). A 'learned decoder' is emulated by perturbing the FURNITURE of the world: for every furniture component (26-connected gt_obj) and
each of its 4 vertical faces (+-x, +-y) independently, the face moves one voxel OUTWARD with prob p_out (a slab of new voxels in front of
the face, only where the world is free) or one voxel INWARD with prob p_in (the face layer is removed; one-voxel-thin parts vanish).
Decode A (ray cast): observe(perturbed world, exact poses). Decode B (direct, ideal known head): target with its occupancy replaced by the
perturbed world on target-known voxels. Score: training metric (furniture-region F1, surface F1) + unknown->known FPR, K0.
One process, CPU, < 1 GB. Usage: a06_faceerr.py [n_houses]"""
import json, os, sys, time
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a06_common import houses, scene, metrics_rows, mean  # noqa: E402
from a06_raycast import observe, fibonacci_sphere, BEAMS, POSES  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
NH = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CONDS = [(0, 0), (.05, 0), (.1, 0), (.2, 0), (.4, 0), (0, .05), (0, .1), (0, .2), (0, .4), (.05, .05), (.1, .1), (.2, .2)]
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def perturb(world, obj, struct_slab, lab, nl, p_out, p_in, rng):
    W = world.copy()
    if p_out == 0 and p_in == 0:
        return W
    for d, (dx, dy) in enumerate(DIRS):
        u = rng.random(nl + 1)
        out_c = u < p_out; in_c = (u >= p_out) & (u < p_out + p_in)
        nb = np.roll(np.roll(obj, -dx, 0), -dy, 1)                        # neighbour in direction d is furniture?
        face = obj & ~nb                                                   # voxels whose d-side is not furniture
        if out_c.any():
            f = face & out_c[lab]
            grow = np.roll(np.roll(f, dx, 0), dy, 1) & ~world               # one voxel in front of the face, only into free space
            W |= grow
        if in_c.any():
            f = face & in_c[lab]
            W &= ~(f & ~struct_slab)
    return W


rays = fibonacci_sphere(BEAMS)
res = {f'{po}_{pi}': dict(ray=dict(furn=[], surf=[], fpr=[]), direct=dict(furn=[], surf=[])) for po, pi in CONDS}
for path in houses()[:NH]:
    t0 = time.time()
    S = scene(path); m = S['m']; n = m['agents']; cell = np.asarray(m['cell_m'], np.float64)
    world = S['occ'].astype(bool); obj = S['obj']; struct_slab = S['struct'] | S['slab']
    lab, nl = ndimage.label(obj, structure=np.ones((3, 3, 3)))
    rng0 = np.random.default_rng(int.from_bytes(m['house'].encode(), 'little') % (2 ** 32))   # same jitters as a06_raycast.py
    centres = np.asarray(m['positions_m'], np.float64); jit = rng0.uniform(-.1, .1, (POSES, n, 3)); jit[:, :, 2] = 0.; traj = centres[None] + jit
    T = np.stack([observe(world.astype(np.uint8), cell, traj[:, a], rays) for a in range(n)])
    rng = np.random.default_rng(342101)
    for po, pi in CONDS:
        Wp = perturb(world, obj, struct_slab, lab, nl, po, pi, rng)
        P = np.stack([observe(Wp.astype(np.uint8), cell, traj[:, a], rays) for a in range(n)])
        D = np.where(T >= 0, Wp[None].astype(np.int8), np.int8(-1))      # ideal known head: target-known voxels take the perturbed world
        r1 = metrics_rows(P, T, S); r2 = metrics_rows(D, T, S)
        k = f'{po}_{pi}'
        res[k]['ray']['furn'] += [r['furn_f1'] for r in r1]; res[k]['ray']['surf'] += [r['surface_f1'] for r in r1]
        res[k]['ray']['fpr'] += [float(((P[a] >= 0) & (T[a] < 0)).sum() / max((T[a] < 0).sum(), 1)) for a in range(n)]
        res[k]['direct']['furn'] += [r['furn_f1'] for r in r2]; res[k]['direct']['surf'] += [r['surface_f1'] for r in r2]
    print(m['house'], f'{time.time() - t0:.0f}s', flush=True)
summ = {k: dict(ray={kk: mean(v) for kk, v in d['ray'].items()}, direct={kk: mean(v) for kk, v in d['direct'].items()}) for k, d in res.items()}
json.dump(dict(houses=NH, summary=summ, raw=res), open(os.path.join(OUT, 'faceerr.json'), 'w'), indent=1)
print('p_out p_in | ray-cast decode: furn surf fpr | direct (ideal known head): furn surf')
for k, d in summ.items():
    print(f"{k:10s} | {d['ray']['furn']:.3f} {d['ray']['surf']:.3f} {d['ray']['fpr']:.4f} | {d['direct']['furn']:.3f} {d['direct']['surf']:.3f}")

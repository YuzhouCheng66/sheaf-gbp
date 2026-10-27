"""a10d_box_carve.py (A10, debate round): is 'one box per furniture instance gives furniture F1 .53' (A04 P5 shape oracle) a property of
box codes, or of painting boxes into OBSERVED FREE space? CPU, numpy only, one process.
Data: A07's re-simulated demo houses (probes/A07/sim_cache, read-only; own maps are exactly observe(gt, poses)). R8 union U of all 8
agents (all houses are single trees); O = observed occupied, F = observed free. Instances = 26-connected components of gt_obj (A04's
definition); each visible instance -> the 3-D bounding box of its observed voxels (A04's 'box' oracle).
  (a) paint      : U with every box painted occupied (A04 P5) -> scored as A04 did (target-known scope) + unknown->known FPR;
  (b) render     : decoder = the simulator's own first-hit marcher from every agent's 3 poses through W = O u boxes;
  (c) render-carved: same through W = O u (boxes \\ F) -- the consistent-set rule O <= W <= not F (A03 / A06 / A08 / A10).
(c) must reproduce U voxel for voxel; (b) shows what an uncarved box does to visibility."""
import glob, json, math, os, sys, time
import numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import dilate6, metrics1  # noqa

CACHE = os.path.abspath(os.path.join(HERE, '..', 'A07', 'sim_cache'))
SENSOR_RANGE = 5.0


def fib(n=65536):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); th = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)), 1)


RAYS = fib()


def observe(occ, cell, poses, rays=RAYS):                             # = v420 observe (bit-exact copy, as in A07 a07_lib)
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float64)
    out = np.full(occ.shape, -1, np.int8)
    ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    for p in poses:
        p = np.asarray(p, np.float64)
        alive = np.ones(len(rays), bool)
        for s in range(1, nsteps + 1):
            idx_all = np.floor((p[None] + rays * (s * ds)) / cell).astype(np.int64)
            inside = np.all((idx_all >= 0) & (idx_all < shape), 1) & alive
            if not inside.any():
                break
            ii = idx_all[inside]
            hit = occ[ii[:, 0], ii[:, 1], ii[:, 2]] == 1
            fr = ii[~hit]; out[fr[:, 0], fr[:, 1], fr[:, 2]] = np.maximum(out[fr[:, 0], fr[:, 1], fr[:, 2]], 0)
            hh = ii[hit]; out[hh[:, 0], hh[:, 1], hh[:, 2]] = 1
            ai = np.nonzero(inside)[0]; alive[ai[hit]] = False
            alive &= inside | ~alive
    return out


def fpr(pred, tgt):
    unk = tgt < 0
    return float(((pred >= 0) & unk).sum() / max(unk.sum(), 1))


rows = []
t0 = time.time()
for path in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
    name = os.path.basename(path)[:-4]
    z = np.load(path)
    own, poses, cell = z['own'], z['poses'], z['cell']; n = own.shape[0]
    slab = z['gt_slab']; struct = z['gt_occ'] & ~z['gt_obj'] & ~z['gt_slab']
    U = own.max(0); O = U == 1; F = U == 0
    lab, K = ndimage.label(z['gt_obj'], structure=np.ones((3, 3, 3), bool))
    B = np.zeros(U.shape, bool); nvis = 0
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        sub = (lab[sl] == k) & O[sl]
        if not sub.any():
            continue
        nvis += 1
        pts = np.argwhere(sub); lo = pts.min(0); hi = pts.max(0) + 1; o = np.array([s.start for s in sl])
        B[o[0] + lo[0]:o[0] + hi[0], o[1] + lo[1]:o[1] + hi[1], o[2] + lo[2]:o[2] + hi[2]] = True
    paint = U.copy(); paint[B] = 1
    Wb = (O | B).astype(np.uint8); Wc = (O | (B & ~F)).astype(np.uint8)
    rb = np.full(U.shape, -1, np.int8); rc = np.full(U.shape, -1, np.int8)
    for a in range(n):
        rb = np.maximum(rb, observe(Wb, cell, poses[a])); rc = np.maximum(rc, observe(Wc, cell, poses[a]))
    r = dict(house=name, instances_visible=nvis, box_vox=int(B.sum()), box_in_F=float((B & F).sum() / max(B.sum(), 1)),
             box_in_unknown=float((B & (U < 0)).sum() / max(B.sum(), 1)),
             paint=metrics1(paint, U, slab, struct), paint_fpr=fpr(paint, U),
             render=metrics1(rb, U, slab, struct), render_fpr=fpr(rb, U), render_mismatch=int((rb != U).sum()),
             carved=metrics1(rc, U, slab, struct), carved_fpr=fpr(rc, U), carved_mismatch=int((rc != U).sum()))
    rows.append(r)
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ('paint', 'render', 'carved')}),
          'paint', {k: round(v, 3) for k, v in r['paint'].items()}, 'render', {k: round(v, 3) for k, v in r['render'].items()},
          'carved', {k: round(v, 3) for k, v in r['carved'].items()}, f'{time.time() - t0:.0f}s', flush=True)
S = {v: {k: float(np.mean([r[v][k] for r in rows])) for k in ('surface_f1', 'var_occ_f1', 'furn_f1')} for v in ('paint', 'render', 'carved')}
S.update(paint_fpr=float(np.mean([r['paint_fpr'] for r in rows])), render_fpr=float(np.mean([r['render_fpr'] for r in rows])),
         carved_fpr=float(np.mean([r['carved_fpr'] for r in rows])), carved_mismatch_total=int(sum(r['carved_mismatch'] for r in rows)),
         render_mismatch_total=int(sum(r['render_mismatch'] for r in rows)), box_in_F_mean=float(np.mean([r['box_in_F'] for r in rows])),
         box_in_unknown_mean=float(np.mean([r['box_in_unknown'] for r in rows])))
print(json.dumps(S, indent=1))
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'a10d_box_carve.json'), 'w'), indent=1)

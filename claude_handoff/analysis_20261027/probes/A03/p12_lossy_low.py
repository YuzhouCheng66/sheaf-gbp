"""p12_lossy_low.py (A03, debate round): the low-rate end of the NON-LEARNED lossy box codec (p3 'plan+box', oracle split), K0 on the
24 VAL houses (192 own maps, 1 process): keep only boxes covering >= m observed furniture hits, m = 32, 64 (p3 has m = 1, 4, 16),
render from the own poses, score. Bytes = p3's coder (plan context code + leave-one-house-out field prior per box + 16 bit heights +
44 bit poses + 8 bit count), priors from cache_p3 (same boxes: the cover is deterministic)."""
import json, os, time
import numpy as np
import p3_box_codec as P
from a03np import observe_np, cell_of, load_fx, names, HERE, score_np, fpr

MS = (32, 64)
H = []
for h in names():
    x = json.load(open(os.path.join(P.CACHE, h + '.json')))
    H.append(x)
rows = []
for h in names():
    t0 = time.time()
    tabs = P.ent_tables(H, h)
    fx, meta = load_fx(h); own = fx['private']; cell = cell_of(meta['chart_m']); traj = fx['trajectory']
    occ, slab, obj, struct, wall2d, lint2d, lz = P.labels(fx)
    cached = {r['agent']: r for x in H if x['house'] == h for r in x['k0'] if r['variant'] == 'plan+box'}
    for a in range(own.shape[0]):
        d = P.describe(own[a], slab, obj, struct, wall2d, lint2d, 'plan+box')
        boxes = [list(b) + [int(d['must'][b[0]:b[1], b[2]:b[3], b[4]:b[5]].sum())] for b in d['boxes']]
        assert boxes == cached[a]['boxes'], (h, a)                       # same deterministic cover as p3
        for m in MS:
            w, keep = P.world_of(d, slab, lz, m)
            r = observe_np(w, cell, traj[:, a])
            rows.append(dict(house=h, agent=a, m=m, bytes=cached[a]['plan_bytes'] + (P.box_bits(boxes, tabs['plan+box'], m) + 16 + 44) / 8,
                             n_boxes=len(keep), fpr=fpr(r, own[a]), **score_np(r, own[a], slab, obj, struct)))
    print(h, f'{time.time()-t0:.0f}s', flush=True)
S = {}
for m in MS:
    rr = [r for r in rows if r['m'] == m]
    S[m] = dict(bytes_median=float(np.median([r['bytes'] for r in rr])), boxes_median=float(np.median([r['n_boxes'] for r in rr])),
                **{k: float(np.mean([r[k] for r in rr if r[k] is not None])) for k in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr')})
    print(m, S[m])
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p12_lossy_low.json'), 'w'), indent=1)

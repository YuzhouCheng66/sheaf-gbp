"""p8_label_free.py (A03): the 'plan+box' codec WITHOUT oracle labels (24 VAL houses, 192 own maps).
Heuristic split, from the agent's own map only: a column is STRUCTURE when it holds an observed hit in the band z index 23-24
(2.3-2.5 m: above every furniture top of the generator, below the lowest ceiling); it is a LINTEL when the same column also holds an
observed FREE voxel below z index 20 (2.0 m), else a WALL. Everything else that is hit (and not slab) is covered by greedy filled boxes
(forbidden = observed free). The plan is coded by the adaptive 2-D context coder; boxes by p7's 'joint' static prior (leave-one-house-out).
Checks the render exactness on every map (the decoder paints slabs, wall columns full height, lintel columns from the lowest observed
lintel voxel up, and the boxes, then ray-casts from the own poses). Usage: python -B p8_label_free.py [workers]"""
import json, math, os, sys, time
from collections import Counter
import numpy as np
from a03np import observe_np, cell_of, load_fx, names, HERE, box_cover, paint, ctx2d_bytes, score_np
from p7_box_prior import tokens, H


def house(name):
    t0 = time.time()
    fx, meta = load_fx(name); own = fx['private']; cell = cell_of(meta['chart_m']); traj = fx['trajectory']
    slab = fx['slab'].astype(bool); obj = fx['obj'].astype(bool); struct = fx['occ'].astype(bool) & ~obj & ~slab
    out = []
    for a in range(own.shape[0]):
        M = own[a]; hits = (M == 1) & ~slab; free = M == 0
        upper = hits[:, :, 23:25].any(2)
        lint = upper & free[:, :, 1:20].any(2); wall = upper & ~lint
        plan = np.where(wall, 1, np.where(lint, 2, 0)).astype(np.int8)
        lz = int(np.argmax(hits[lint].any(0))) if lint.any() else 21
        rest = hits & ~upper[:, :, None]
        boxes = box_cover(rest, free)
        g = slab | wall[:, :, None] | paint(boxes)
        zz = np.zeros(32, bool); zz[lz:] = True; g |= lint[:, :, None] & zz[None, None, :]
        r = observe_np(g, cell, traj[:, a])
        out.append(dict(house=name, agent=a, plan_bytes=ctx2d_bytes(plan, 3), boxes=[list(b) for b in boxes], exact=bool((r == M).all()),
                        wall_cols=int(wall.sum()), lintel_cols=int(lint.sum()), true_struct_cols_seen=int(((M == 1) & struct).any(2).sum()),
                        struct_hits_in_boxes=int((rest & struct).sum()), score=score_np(r, M, slab, obj, struct)))
    print(name, 'exact', sum(o['exact'] for o in out), '/', len(out), f'{time.time()-t0:.0f}s', flush=True)
    return out


if __name__ == '__main__':
    import multiprocessing as mp
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        R = [r for rr in pool.map(house, names()) for r in rr]
    for r in R:
        other = [q for q in R if q['house'] != r['house']]
        j = dict(dpx=Counter(), dpy=Counter(), z=Counter(), s=Counter())
        for q in other:
            for t in tokens(q['boxes']):
                for k in j:
                    j[k][t[k]] += 1
        r['box_bytes'] = (8 + sum(H(j['dpx'], t['dpx'], 256) + H(j['dpy'], t['dpy'], 256) + H(j['z'], t['z'], 32 * 32) + H(j['s'], t['s'], 128 * 128) for t in tokens(r['boxes']))) / 8
        r['total_bytes'] = r['plan_bytes'] + r['box_bytes'] + (16 + 44) / 8
    med = lambda k: float(np.median([r[k] for r in R]))
    S = dict(n=len(R), exact=sum(r['exact'] for r in R), total_bytes_median=med('total_bytes'), total_bytes_p10=float(np.percentile([r['total_bytes'] for r in R], 10)),
             total_bytes_p90=float(np.percentile([r['total_bytes'] for r in R], 90)), plan_bytes_median=med('plan_bytes'), box_bytes_median=med('box_bytes'),
             boxes_median=float(np.median([len(r['boxes']) for r in R])), struct_hits_in_boxes_median=med('struct_hits_in_boxes'),
             wall_cols_median=med('wall_cols'), true_struct_cols_seen_median=med('true_struct_cols_seen'))
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, rows=[{k: v for k, v in r.items() if k != 'boxes'} for r in R]), open(os.path.join(HERE, 'p8_label_free.json'), 'w'), indent=1)

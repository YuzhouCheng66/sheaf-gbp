"""p3_box_codec.py (A03): a NON-LEARNED, LIBRARY-FREE structured codec with a ray-casting decoder, on the 24 VAL houses.
Description of a ternary map M seen from pose set(s) P:
  slabs: wall height H_w and door height h_d (16 bit, as world_dl);  poses: 44 bit per agent;
  variant 'plan+box' (oracle wall/furniture split of the OBSERVED voxels): structure = 128 x 128 ternary plan (0 none, 1 wall column,
      2 lintel column) of the observed structure columns, adaptive 2-D context code; furniture = greedy boxes (a03lib.box_cover) that cover
      every observed furniture hit and never enter a voxel M marks free;
  variant 'box-only' (no labels at all): every observed non-slab hit is covered by greedy boxes (walls become thin boxes).
Decoder: paint slabs + plan columns (walls full height, lintels from h_d up) + boxes, then ray-cast (observe_t) from the poses, join (max).
Box bits = 14 (x0, y0 uniform) + leave-one-house-out entropies of z0, dx, dy, dz (a 'learned' static prior) + 8 bit count.
Maps: K0 = own map (own poses);  R8 central = fit on the union target T (all reached poses);  R8 relay = every reached agent's own
description, each rendered from ITS OWN poses, then joined (owner-indexed, conflict-free, redundant);  R8 naive = all agents' worlds
painted together and rendered from all reached poses (shows the box over-fill conflicts).  Lossy: drop boxes covering < m hits.
Usage: python -B p3_box_codec.py [all|demo]"""
import json, math, os, sys, time
from collections import Counter
import numpy as np
from a03np import observe_np as observe_t, cell_of, targets, load_fx, names, HERE, box_cover, paint, ctx2d_bytes, score_np as score

MS = (1, 4, 16)                                                       # lossy: keep boxes covering >= m must-voxels (plan+box only)
VARIANTS = ('plan+box', 'box-only', 'shell')
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_p3'); os.makedirs(CACHE, exist_ok=True)


def labels(fx):
    occ = fx['occ'].astype(bool); slab = fx['slab'].astype(bool); obj = fx['obj'].astype(bool)
    struct = occ & ~obj & ~slab
    wall2d = struct[:, :, 5]; lint2d = struct[:, :, 23] & ~wall2d
    lintel_z0 = int(np.argmax(struct[lint2d].any(0))) if lint2d.any() else 21
    return occ, slab, obj, struct, wall2d, lint2d, lintel_z0


def describe(M, slab, obj, struct, wall2d, lint2d, variant):
    hits = M == 1; free = M == 0
    if variant == 'box-only':
        must = hits & ~slab
        return dict(plan=None, boxes=box_cover(must, free), must=must)
    if variant == 'shell':                                              # boxes may cover ONLY observed hits: every box is true geometry
        must = hits & ~slab
        return dict(plan=None, boxes=box_cover(must, ~must), must=must)
    sc = (hits & struct).any(2)
    plan = np.where(sc & wall2d, 1, np.where(sc & lint2d, 2, np.where(sc, 1, 0))).astype(np.int8)
    must = hits & obj
    return dict(plan=plan, boxes=box_cover(must, free), must=must)


def world_of(d, slab, lintel_z0, m=1):
    g = slab.copy()
    if d['plan'] is not None:
        g |= (d['plan'] == 1)[:, :, None]
        lz = np.zeros(g.shape[2], bool); lz[lintel_z0:] = True
        g |= (d['plan'] == 2)[:, :, None] & lz[None, None, :]
    keep = [b for b in d['boxes'] if d['must'][b[0]:b[1], b[2]:b[3], b[4]:b[5]].sum() >= m]
    return g | paint(keep), keep


def house(name):
    t0 = time.time()
    fx, meta = load_fx(name); own = fx['private']; n = own.shape[0]; cell = cell_of(meta['chart_m']); traj = fx['trajectory']
    occ, slab, obj, struct, wall2d, lint2d, lintel_z0 = labels(fx)
    out = dict(house=name, k0=[], r8=[])
    descs = {}
    for tag in VARIANTS:
        ms = MS if tag == 'plan+box' else (1,)
        for a in range(n):
            d = describe(own[a], slab, obj, struct, wall2d, lint2d, tag); descs[(tag, a)] = d
            row = dict(agent=a, variant=tag, plan_bytes=(ctx2d_bytes(d['plan'], 3) if d['plan'] is not None else 0.),
                       boxes=[list(b) + [int(d['must'][b[0]:b[1], b[2]:b[3], b[4]:b[5]].sum())] for b in d['boxes']], scores={})
            for m in ms:
                if tag != 'plan+box':
                    row['scores'][m] = dict(surface_f1=None, var_occ_f1=None, furn_f1=None); row['exact'] = None
                    continue
                w, keep = world_of(d, slab, lintel_z0, m)
                r = observe_t(w, cell, traj[:, a]); row['scores'][m] = score(r, own[a], slab, obj, struct)
                if m == 1:
                    row['exact'] = bool((r == own[a]).all())
            out['k0'].append(row)
        R = fx['reached'][4]; T8 = targets(own, R); done = set()
        for a in range(n):
            key = tuple(int(v) for v in np.nonzero(R[a])[0])
            if key in done:
                continue
            done.add(key); T = T8[a]
            d = describe(T, slab, obj, struct, wall2d, lint2d, tag)
            row = dict(agents=list(key), variant=tag, plan_bytes=(ctx2d_bytes(d['plan'], 3) if d['plan'] is not None else 0.),
                       boxes=[list(b) + [int(d['must'][b[0]:b[1], b[2]:b[3], b[4]:b[5]].sum())] for b in d['boxes']], scores={},
                       relay_plan_bytes=float(sum(ctx2d_bytes(descs[(tag, b)]['plan'], 3) if descs[(tag, b)]['plan'] is not None else 0. for b in key)),
                       relay_boxes=int(sum(len(descs[(tag, b)]['boxes']) for b in key)))
            if tag != 'plan+box':
                for m in ms:
                    row['scores'][m] = dict(surface_f1=None, var_occ_f1=None, furn_f1=None)
                row['exact'] = None; row['naive'] = dict(surface_f1=None, var_occ_f1=None, furn_f1=None); row['naive_exact'] = None
                out['r8'].append(row)
                continue
            for m in ms:
                w, keep = world_of(d, slab, lintel_z0, m)
                U = None
                for b in key:
                    o = observe_t(w, cell, traj[:, b]); U = o if U is None else np.maximum(U, o)
                row['scores'][m] = score(U, T, slab, obj, struct)
                if m == 1:
                    row['exact'] = bool((U == T).all())
            wn = slab.copy()                                            # naive union of the agents' painted worlds
            for b in key:
                wn |= world_of(descs[(tag, b)], slab, lintel_z0, 1)[0]
            U = None
            for b in key:
                o = observe_t(wn, cell, traj[:, b]); U = o if U is None else np.maximum(U, o)
            row['naive'] = score(U, T, slab, obj, struct); row['naive_exact'] = bool((U == T).all())
            out['r8'].append(row)
    print(name, 'k0 exact', sum(bool(r['exact']) for r in out['k0']), 'r8 exact', sum(bool(r['exact']) for r in out['r8']),
          'naive exact', sum(bool(r['naive_exact']) for r in out['r8']), f'{time.time()-t0:.0f}s', flush=True)
    json.dump(out, open(os.path.join(CACHE, name + '.json'), 'w'))
    return name


def ent_tables(houses, skip):
    """leave-one-house-out static distributions of z0, dx, dy, dz (add-0.5), per variant"""
    t = {}
    for h in houses:
        if h['house'] == skip:
            continue
        for r in h['k0']:
            c = t.setdefault(r['variant'], [Counter(), Counter(), Counter(), Counter()])
            for b in r['boxes']:
                c[0][b[4]] += 1; c[1][b[1] - b[0]] += 1; c[2][b[3] - b[2]] += 1; c[3][b[5] - b[4]] += 1
    return t


def box_bits(boxes, tab, m):
    if tab is None:
        return 0.
    sizes = (32, 128, 128, 32)
    tot = 8.
    for b in boxes:
        if b[6] < m:
            continue
        tot += 14
        for f, v in enumerate((b[4], b[1] - b[0], b[3] - b[2], b[5] - b[4])):
            c = tab[f]; N = sum(c.values())
            tot += -math.log2((c.get(v, 0) + 0.5) / (N + 0.5 * sizes[f]))
    return tot


if __name__ == '__main__':
    import multiprocessing as mp
    which = sys.argv[1] if len(sys.argv) > 1 else 'demo'
    todo = [h for h in names(demo_only=(which == 'demo')) if not os.path.exists(os.path.join(CACHE, h + '.json'))]
    with mp.Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(house, todo):
            pass
    H = []
    for h in names(demo_only=(which == 'demo')):
        x = json.load(open(os.path.join(CACHE, h + '.json')))
        for r in x['k0'] + x['r8']:
            r['scores'] = {int(k): v for k, v in r['scores'].items()}
        H.append(x)
    rows = []
    for h in H:
        tabs = ent_tables(H, h['house'])
        for r in h['k0']:
            for m in (MS if r['variant'] == 'plan+box' else (1,)):
                rows.append(dict(map='K0', house=h['house'], variant=r['variant'], m=m, bytes=r['plan_bytes'] + (box_bits(r['boxes'], tabs.get(r['variant']), m) + 16 + 44) / 8,
                                 n_boxes=sum(1 for b in r['boxes'] if b[6] >= m), exact=r['exact'] if m == 1 else None, **r['scores'][m]))
        for r in h['r8']:
          for _rep in r['agents']:                                    # one row per receiving agent (as the evidence pack's union8 rows)
            for m in (MS if r['variant'] == 'plan+box' else (1,)):
                rows.append(dict(map='R8', house=h['house'], variant=r['variant'], m=m, agents=len(r['agents']),
                                 bytes=r['plan_bytes'] + (box_bits(r['boxes'], tabs.get(r['variant']), m) + 16 + 44 * len(r['agents'])) / 8,
                                 n_boxes=sum(1 for b in r['boxes'] if b[6] >= m), exact=r['exact'] if m == 1 else None, **r['scores'][m]))
            relay = r['relay_plan_bytes'] + sum((box_bits(rr['boxes'], tabs.get(r['variant']), 1) + 16 + 44) / 8 for rr in h['k0'] if rr['variant'] == r['variant'] and rr['agent'] in r['agents'])
            rows.append(dict(map='R8relay', house=h['house'], variant=r['variant'], m=1, agents=len(r['agents']), bytes=relay, exact=True, n_boxes=r['relay_boxes'],
                             surface_f1=1., var_occ_f1=1., furn_f1=1.))
            rows.append(dict(map='R8naive', house=h['house'], variant=r['variant'], m=1, agents=len(r['agents']), bytes=relay, exact=r['naive_exact'], n_boxes=r['relay_boxes'], **r['naive']))
    S = {}
    for key in sorted({(r['map'], r['variant'], r['m']) for r in rows}):
        rr = [r for r in rows if (r['map'], r['variant'], r['m']) == key]
        mean = lambda k: float(np.mean([r[k] for r in rr if r.get(k) is not None])) if any(r.get(k) is not None for r in rr) else None
        S['%s|%s|m%d' % key] = dict(n=len(rr), bytes_median=float(np.median([r['bytes'] for r in rr])), bytes_p10=float(np.percentile([r['bytes'] for r in rr], 10)),
                                    bytes_p90=float(np.percentile([r['bytes'] for r in rr], 90)), boxes_median=float(np.median([r['n_boxes'] for r in rr])),
                                    exact=(sum(1 for r in rr if r['exact']) if key[2] == 1 else None), exact_checked=sum(1 for r in rr if r['exact'] is not None), surface_f1=mean('surface_f1'), var_occ_f1=mean('var_occ_f1'), furn_f1=mean('furn_f1'))
    for k, v in S.items():
        print(f"{k:28s} n {v['n']:3d} bytes med {v['bytes_median']:7.0f} [{v['bytes_p10']:6.0f},{v['bytes_p90']:6.0f}] boxes {v['boxes_median']:5.0f} exact {v['exact']} "
              f"surf {v['surface_f1']} var {v['var_occ_f1']} furn {v['furn_f1']}")
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, f'p3_box_codec_{which}.json'), 'w'), indent=1)

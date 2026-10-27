"""p9_message_bytes.py (A03): per-MESSAGE bytes of the label-free RAY-SHEAF description (p8's codec) on the 24 VAL houses, next to the
lossless baselines measured the same way (gzip-9 of the 2-bit planes, ctx = 9-neighbour adaptive context code), for
  own   = each agent's own map (what K0 carries),
  edge  = each directed forest edge's sender-side union (what one converged message stands for; the cavity join),
  union = each agent's round-8 target (what R8 decodes).
Description = 2-D plan (context-coded) + greedy filled boxes (p7 'joint' static prior, leave-one-house-out on the own-map boxes)
+ 16 bit heights + 44 bit per pose triple of every agent in the map + 8 bit count. Bytes only (exactness: p3 central / p8).
Also the protocol total per house: current (8 rounds x every directed edge x 2,048 B) vs innovation-only totals.
Usage: python -B p9_message_bytes.py [workers]"""
import gzip, json, math, os, sys, time
from collections import Counter
import numpy as np
from a03np import load_fx, names, HERE, box_cover, ctx2d_bytes, ctx_bytes_np, targets
from p7_box_prior import tokens, H

CACHE = os.path.join(HERE, 'cache_p9'); os.makedirs(CACHE, exist_ok=True)


def gz(m):
    return len(gzip.compress(np.packbits(np.stack([m >= 0, m == 1]).reshape(-1)).tobytes(), 9))


def describe(M, slab):
    hits = (M == 1) & ~slab; free = M == 0
    upper = hits[:, :, 23:25].any(2)
    lint = upper & free[:, :, 1:20].any(2); wall = upper & ~lint
    plan = np.where(wall, 1, np.where(lint, 2, 0)).astype(np.int8)
    return ctx2d_bytes(plan, 3), [list(b) for b in box_cover(hits & ~upper[:, :, None], free)]


def house(name):
    if os.path.exists(os.path.join(CACHE, name + '.json')):
        return name
    t0 = time.time()
    fx, meta = load_fx(name); own = fx['private']; n = own.shape[0]; slab = fx['slab'].astype(bool)
    rows = []
    for a in range(n):
        pb, bx = describe(own[a], slab)
        rows.append(dict(kind='own', agents=1, gzip=gz(own[a]), ctx=ctx_bytes_np(own[a]), plan=pb, boxes=bx))
    R = fx['reached'][4]; T8 = targets(own, R)
    for a in range(n):
        pb, bx = describe(T8[a], slab)
        rows.append(dict(kind='union', agents=int(R[a].sum()), gzip=gz(T8[a]), ctx=ctx_bytes_np(T8[a]), plan=pb, boxes=bx))
    E = np.asarray(fx['edges']).reshape(-1, 3)[:, :2].astype(int)
    adj = {i: set() for i in range(n)}
    for i, j in E:
        adj[i].add(j); adj[j].add(i)
    for i, j in E:
        for s, r in ((i, j), (j, i)):
            side, stack = {s}, [s]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v != r and v not in side:
                        side.add(v); stack.append(v)
            U = own[sorted(side)].max(0)
            pb, bx = describe(U, slab)
            rows.append(dict(kind='edge', agents=len(side), gzip=gz(U), ctx=ctx_bytes_np(U), plan=pb, boxes=bx))
    json.dump(dict(house=name, n_directed=2 * len(E), rows=rows), open(os.path.join(CACHE, name + '.json'), 'w'))
    print(name, f'{time.time()-t0:.0f}s', flush=True)
    return name


if __name__ == '__main__':
    import multiprocessing as mp
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(house, names()):
            pass
    Hs = [json.load(open(os.path.join(CACHE, h + '.json'))) for h in names()]
    for h in Hs:                                                         # leave-one-house-out joint prior from the OTHER houses' own maps
        j = dict(dpx=Counter(), dpy=Counter(), z=Counter(), s=Counter())
        for g in Hs:
            if g['house'] == h['house']:
                continue
            for r in g['rows']:
                if r['kind'] == 'own':
                    for t in tokens(r['boxes']):
                        for k in j:
                            j[k][t[k]] += 1
        for r in h['rows']:
            bits = 8 + sum(H(j['dpx'], t['dpx'], 256) + H(j['dpy'], t['dpy'], 256) + H(j['z'], t['z'], 32 * 32) + H(j['s'], t['s'], 128 * 128) for t in tokens(r['boxes']))
            r['desc'] = r['plan'] + bits / 8 + (16 + 44 * r['agents']) / 8
            r['n_boxes'] = len(r['boxes'])
    S = {}
    for kind in ('own', 'edge', 'union'):
        rr = [r for h in Hs for r in h['rows'] if r['kind'] == kind]
        med = lambda k: float(np.median([r[k] for r in rr]))
        S[kind] = dict(n=len(rr), desc=med('desc'), desc_p90=float(np.percentile([r['desc'] for r in rr], 90)), ctx=med('ctx'), gzip=med('gzip'), boxes=med('n_boxes'),
                       ctx_over_desc=float(np.median([r['ctx'] / r['desc'] for r in rr])), gzip_over_desc=float(np.median([r['gzip'] / r['desc'] for r in rr])),
                       raw_over_desc=float(np.median([131072 / r['desc'] for r in rr])), ours_over_desc=float(np.median([2048 / r['desc'] for r in rr])))
    tot = []
    for h in Hs:
        e = [r for r in h['rows'] if r['kind'] == 'edge']
        tot.append(dict(current=h['n_directed'] * 8 * 2048, ctx_edges=sum(r['ctx'] for r in e), desc_edges=sum(r['desc'] for r in e)))
    S['protocol_total_per_house'] = {k: float(np.median([t[k] for t in tot])) for k in ('current', 'ctx_edges', 'desc_edges')}
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, rows=[{k: v for k, v in r.items() if k != 'boxes'} | dict(house=h['house']) for h in Hs for r in h['rows']]), open(os.path.join(HERE, 'p9_message_bytes.json'), 'w'), indent=1)

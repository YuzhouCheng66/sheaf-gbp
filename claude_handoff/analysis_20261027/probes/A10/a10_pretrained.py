"""a10_pretrained.py (A10): proxy for a PRE-TRAINED prior. A context model whose counts come from the OTHER 5 demo houses (leave-one-
house-out) is shared by encoder and decoder; each map is then coded adaptively starting from those counts (Dirichlet prior alpha = 0.5
+ w * mean counts per training map, w in map-equivalents). This removes the per-map learning cost that made richer contexts useless in
a10_vctx, and is an exact, implementable code. With 5 training houses it is a weak stand-in for a network trained on 4,708 houses.
Targets: own-map V (48), union V (6), edge innovation V over the receiver side (84 directed edges; only positions where the receiver
side is unknown; + reach flag as in a10_probe). Pose bytes (44 bit per agent) are added as in a10_probe."""
import gc, glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import load, causal, NB, adjacency, side_of, union, reachable, _lg  # noqa
from a10_vctx import causal_any, dist_bucket, NB13  # noqa  (a10_vctx's probe body is under __main__)

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
POSE_B = 44 / 8
DESIGNS = ('nb9', 'nb13', 'nb13_z', 'nb13_z_d')
WS = (0.25, 1.0, 4.0)
ZLEV = np.broadcast_to(np.arange(32)[None, None], (128, 128, 32)).astype(np.int64)


def ctx_of(design, v, db):
    if design == 'nb9':
        return causal(v, 2)
    c13 = causal_any(v, 2, NB13)
    if design == 'nb13':
        return c13
    if design == 'nb13_z':
        return c13 * 32 + ZLEV
    return (c13 * 32 + ZLEV) * 5 + db


def counts(sym, ctx):
    return np.unique(ctx.astype(np.int64) * 2 + sym, return_counts=True)


def merge(prior_list):
    keys = np.concatenate([k for k, _ in prior_list]); vals = np.concatenate([v for _, v in prior_list])
    u, inv = np.unique(keys, return_inverse=True)
    return u, np.bincount(inv, weights=vals)


def lookup(prior, q):
    keys, vals = prior
    idx = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
    return np.where(keys[idx] == q, vals[idx], 0.)


def code_bits(sym, ctx, prior, w):
    u, inv = np.unique(ctx, return_inverse=True)
    cnt = np.bincount(inv * 2 + sym, minlength=len(u) * 2).reshape(-1, 2).astype(np.float64)
    al = np.full((len(u), 2), 0.5)
    if w > 0:
        for s in (0, 1):
            al[:, s] += w * lookup(prior, u.astype(np.int64) * 2 + s)
    A = al.sum(1); N = cnt.sum(1)
    return float(((_lg(A + N) - _lg(A)).sum() - (_lg(al + cnt) - _lg(al)).sum()) / math.log(2))


# ---- raw material once per house (maps, reach masks, distance buckets), contexts built per design
raw = []
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]; cell = m['cell_m']; pos = np.asarray(m['positions_m']); adj = adjacency(m['edges'], n)
    U = own.max(0)
    for a in range(n):
        raw.append(dict(house=m['house'], kind='own', n_pose=1, v=(own[a] == 1).astype(np.int8), sel=None, reach=None,
                        db=dist_bucket(own.shape[1:], cell, [pos[a]]).astype(np.int8)))
    raw.append(dict(house=m['house'], kind='union', n_pose=n, v=(U == 1).astype(np.int8), sel=None, reach=None, db=dist_bucket(U.shape, cell, pos).astype(np.int8)))
    for i, j in m['edges']:
        for s_, r_ in ((i, j), (j, i)):
            Ss = side_of(adj, s_, r_); R = union(own, side_of(adj, r_, s_))
            reach = np.zeros(R.shape, bool)
            for a in Ss:
                reach |= reachable(R == 1, cell, pos[a])
            raw.append(dict(house=m['house'], kind='edge', n_pose=len(Ss), v=(U == 1).astype(np.int8), sel=(R < 0), reach=reach,
                            db=dist_bucket(U.shape, cell, pos[Ss]).astype(np.int8)))
    print(m['house'], 'raw built', flush=True)

houses = sorted(set(x['house'] for x in raw))
res = []
for d in DESIGNS:
    items = []
    for x in raw:
        v = x['v'].astype(np.int64); c = ctx_of(d, v, x['db'].astype(np.int64))
        if x['kind'] == 'edge':
            c = c * 2 + x['reach']
            items.append(dict(house=x['house'], kind='edge', n_pose=x['n_pose'], sym=v[x['sel']].astype(np.int64), ctx=c[x['sel']]))
        else:
            items.append(dict(house=x['house'], kind=x['kind'], n_pose=x['n_pose'], sym=v.reshape(-1), ctx=c.reshape(-1)))
    for h in houses:
        for kind in ('own', 'union', 'edge'):
            train = [x for x in items if x['house'] != h and x['kind'] == kind]
            prior = merge([counts(x['sym'], x['ctx']) for x in train]); ntr = len(train)
            for x in (x for x in items if x['house'] == h and x['kind'] == kind):
                r = dict(house=h, kind=kind, design=d, adaptive=code_bits(x['sym'], x['ctx'], prior, 0.) / 8 + 2 + POSE_B * x['n_pose'])
                for w in WS:
                    r[f'pre_w{w}'] = code_bits(x['sym'], x['ctx'], prior, w / ntr) / 8 + 2 + POSE_B * x['n_pose']
                res.append(r)
    del items; gc.collect()
    print(d, 'coded', flush=True)
json.dump(res, open(os.path.join(HERE, 'a10_pretrained.json'), 'w'), indent=1)
for kind in ('own', 'union', 'edge'):
    for d in DESIGNS:
        r = [x for x in res if x['kind'] == kind and x['design'] == d]
        print(f'{kind:5s} {d:9s} adaptive {np.median([x["adaptive"] for x in r]):7.0f}  ' + '  '.join(f'pre(w={w}) {np.median([x[f"pre_w{w}"] for x in r]):7.0f}' for w in WS))

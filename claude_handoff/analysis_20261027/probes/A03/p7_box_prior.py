"""p7_box_prior.py (A03): how much could a learned prior save on the box tokens? (192 own maps of the 24 VAL houses; no rendering:
the covers are p3's, exact by p3). Three static priors, all leave-one-house-out (a stand-in for 'trained on other houses'):
  field   = p3's coder: x0, y0 uniform (14 bit) + independent per-field entropies of z0, dx, dy, dz;
  joint   = boxes sorted by (x0, y0, z0); position as the delta to the previous box (dx0 in -128..127, y0 delta), joint (z0, dz) and
            joint (dx, dy) tables -- a crude 'autoregressive' context;
  library = the world_dl.py parametric description of the same own maps (generator-aware upper bound on what a perfect prior needs).
Usage: python -B p7_box_prior.py"""
import json, math, os, sys
from collections import Counter
import numpy as np
from a03np import load_fx, names, HERE, box_cover


def covers(name):
    fx, meta = load_fx(name); own = fx['private']
    occ = fx['occ'].astype(bool); slab = fx['slab'].astype(bool); obj = fx['obj'].astype(bool)
    out = []
    for a in range(own.shape[0]):
        hits = own[a] == 1; free = own[a] == 0
        pb = box_cover(hits & obj, free)
        sb = box_cover(hits & ~slab, ~(hits & ~slab))
        out.append(dict(house=name, agent=a, plan_box=pb, shell=sb))
    return out


def H(c, v, size):
    N = sum(c.values())
    return -math.log2((c.get(v, 0) + 0.5) / (N + 0.5 * size))


def tokens(boxes):
    bs = sorted(boxes, key=lambda b: (b[0], b[2], b[4]))
    prev = (0, 0)
    out = []
    for b in bs:
        x0, x1, y0, y1, z0, z1 = b
        out.append(dict(dpx=x0 - prev[0], dpy=y0 - prev[1] if x0 == prev[0] else ('abs', y0), z=(z0, z1 - z0), s=(x1 - x0, y1 - y0)))
        prev = (x0, y0)
    return out


if __name__ == '__main__':
    import multiprocessing as mp
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        R = [r for rr in pool.map(covers, names()) for r in rr]
    res = {}
    for kind in ('plan_box', 'shell'):
        rows = []
        for r in R:
            other = [q for q in R if q['house'] != r['house']]
            f = [Counter(), Counter(), Counter(), Counter()]; j = dict(dpx=Counter(), dpy=Counter(), z=Counter(), s=Counter())
            for q in other:
                for b in q[kind]:
                    f[0][b[4]] += 1; f[1][b[1] - b[0]] += 1; f[2][b[3] - b[2]] += 1; f[3][b[5] - b[4]] += 1
                for t in tokens(q[kind]):
                    for k in j:
                        j[k][t[k]] += 1
            field = 8 + sum(14 + H(f[0], b[4], 32) + H(f[1], b[1] - b[0], 128) + H(f[2], b[3] - b[2], 128) + H(f[3], b[5] - b[4], 32) for b in r[kind])
            joint = 8 + sum(H(j['dpx'], t['dpx'], 256) + H(j['dpy'], t['dpy'], 256) + H(j['z'], t['z'], 32 * 32) + H(j['s'], t['s'], 128 * 128) for t in tokens(r[kind]))
            rows.append(dict(n=len(r[kind]), field=field / 8, joint=joint / 8))
        res[kind] = dict(boxes_median=float(np.median([x['n'] for x in rows])), field_bytes_median=float(np.median([x['field'] for x in rows])),
                         joint_bytes_median=float(np.median([x['joint'] for x in rows])), bits_per_box_field=float(np.sum([x['field'] for x in rows]) * 8 / np.sum([x['n'] for x in rows])),
                         bits_per_box_joint=float(np.sum([x['joint'] for x in rows]) * 8 / np.sum([x['n'] for x in rows])))
    dl = json.load(open(os.path.join(HERE, '..', '..', 'world_dl.json')))['summary']
    res['library_dl_own_median'] = dl['own']['dl_median']
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, 'p7_box_prior.json'), 'w'), indent=1)

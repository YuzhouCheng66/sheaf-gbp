"""A04 probe P4b (Claude, 2026-10-27): the same rate-F1 frontier as rate_baselines 3.3, but with the KNOWN MASK given to the decoder for
free (it is a function of geometry + poses; P1 tests that), i.e. what a codec with pose stalks + a ray-cast known head must still send.
Per map T (K0 own, R8 union): known K = T >= 0 is side information; the code is the block-binary occupancy O_c = [occupied fraction among
known voxels >= .5] on blocks with any known voxel, coded by the binary context coder with the block's known flag as side context;
decoded map = O_c broadcast inside the TRUE known mask (free elsewhere-known).  Also the occupancy-only two-level scheme (refine a
4x4 column only where the 2x2 OCCUPANCY decode differs from the 4x4 one inside K) with its refined share."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, targets, ctx_bits_cond, metrics, gt_masks, HERE, retry  # noqa: E402
PARTS = os.path.join(HERE, 'parts'); os.makedirs(PARTS, exist_ok=True)


def occ_blocks(T, f):
    X, Y, Z = T.shape; bx, by, bz = f
    k = (T >= 0).reshape(X // bx, bx, Y // by, by, Z // bz, bz); o = (T == 1).reshape(X // bx, bx, Y // by, by, Z // bz, bz)
    nk = k.sum((1, 3, 5)); no = o.sum((1, 3, 5))
    kc = nk > 0; oc = kc & (no >= np.maximum(.5 * nk, 1))
    return kc, oc


def up(g, f):
    return g.repeat(f[0], 0).repeat(f[1], 1).repeat(f[2], 2)


def single(T, f):
    kc, oc = occ_blocks(T, f)
    bits = ctx_bits_cond(oc.astype(np.int64), 2, side=[(kc.astype(np.int64), 2)], mask=kc)
    dec = np.where(T >= 0, up(oc, f).astype(np.int8), np.int8(-1))
    return dec, bits / 8 + 2


def two_level(T, cf, ff):
    kc, oc = occ_blocks(T, cf); kf, of = occ_blocks(T, ff)
    K = T >= 0
    dc = up(oc, cf) & K; df = up(of, ff) & K
    X, Y, Z = T.shape
    flags = (dc != df).reshape(X // cf[0], cf[0], Y // cf[1], cf[1], Z).any((1, 3, 4))
    obs = kc.any(2)
    rx, ry = cf[0] // ff[0], cf[1] // ff[1]
    flag_f = flags.repeat(rx, 0).repeat(ry, 1)
    parent = oc.repeat(rx, 0).repeat(ry, 1)
    held = np.where(flag_f[:, :, None], of, parent)
    b = ctx_bits_cond(oc.astype(np.int64), 2, side=[(kc.astype(np.int64), 2)], mask=kc) + 16
    b += ctx_bits_cond(flags[:, :, None].astype(np.int64), 2, side=[(obs[:, :, None].astype(np.int64), 2)], mask=obs[:, :, None]) + 16
    m = np.broadcast_to(flag_f[:, :, None], held.shape) & kf
    if m.any():
        b += ctx_bits_cond(held.astype(np.int64), 2, side=[(parent.astype(np.int64), 2), (kf.astype(np.int64), 2)], mask=m) + 16
    dec = np.where(K, np.where(flag_f.repeat(ff[0], 0).repeat(ff[1], 1)[:, :, None], df, dc).astype(np.int8), np.int8(-1))
    return dec, b / 8, float(flags[obs].mean()) if obs.any() else 0.


out = {}
t0 = time.time()


def one_house(path):
    out = {}
    m, own, pred, gt = load(path); n = own.shape[0]
    slab, obj, struct = gt_masks(gt)
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    for tag, T in (('K0', own), ('R8', T8)):
        cfgs = [('1x1x1', lambda t: single(t, (1, 1, 1)) + (None,)), ('2x2x1', lambda t: single(t, (2, 2, 1)) + (None,)),
                ('4x4x1', lambda t: single(t, (4, 4, 1)) + (None,)), ('2x2x2', lambda t: single(t, (2, 2, 2)) + (None,)),
                ('4x4->2x2', lambda t: two_level(t, (4, 4, 1), (2, 2, 1))), ('4x4->1x1', lambda t: two_level(t, (4, 4, 1), (1, 1, 1)))]
        for name, fn in cfgs:
            decs, bs, sh = [], [], []
            for a in range(n):
                dec, b, s = fn(T[a]); decs.append(dec); bs.append(b); sh.append(s)
            rows = metrics(torch.from_numpy(np.stack(decs).reshape(n, -1)), torch.from_numpy(T.reshape(n, -1)), slab, obj, struct)
            d = out.setdefault(f'{tag}_{name}', dict(bytes=[], share=[], surface_f1=[], var_occ_f1=[], furn_f1=[]))
            d['bytes'] += bs; d['share'] += [x for x in sh if x is not None]
            for r in rows:
                for k in ('surface_f1', 'var_occ_f1', 'furn_f1'):
                    d[k].append(r[k])
    print(m['house'], f'{time.time() - t0:.0f}s', flush=True)
    return out


for path in houses():
    part = os.path.join(PARTS, 'p4b_' + os.path.basename(path))
    if not os.path.exists(part):
        json.dump(retry(one_house, path), open(part, 'w'))
    for k, v in json.load(open(part)).items():
        d = out.setdefault(k, dict(bytes=[], share=[], surface_f1=[], var_occ_f1=[], furn_f1=[]))
        for kk in d:
            d[kk] += v[kk]


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


S = {k: dict(bytes_mean=mean(v['bytes']), bytes_median=float(np.median(v['bytes'])), refined_share=mean(v['share']),
             surface_f1=mean(v['surface_f1']), var_occ_f1=mean(v['var_occ_f1']), furn_f1=mean(v['furn_f1'])) for k, v in out.items()}
for k, v in S.items():
    rs = 'n/a' if v['refined_share'] is None else f"{v['refined_share']:.2f}"
    print(f"{k:16s} bytes {v['bytes_mean']:7.0f} (med {v['bytes_median']:6.0f})  refined {rs}  surf {v['surface_f1']:.3f}  var {v['var_occ_f1']:.3f}  furn {v['furn_f1']:.3f}")
json.dump(S, open(os.path.join(HERE, 'p4b_occ_given_known.json'), 'w'), indent=1)

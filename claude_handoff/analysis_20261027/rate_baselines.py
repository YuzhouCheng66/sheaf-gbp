"""rate_baselines.py (Claude, 2026-10-27): how many bytes do CLASSICAL codecs need for the maps our 2 KB / 8 KB messages carry?
Data: demo_web3d/data/house_*.json (6 VAL houses x 8 agents; current best 3S-X + calibrated rule; bit planes over 128 x 128 x 32).
Lossless coders on a ternary map (-1 unknown, 0 free, 1 occupied):
  gzip9 / bz2 / lzma(extreme) of the 2-bit packed planes (gzip9 = the demo's own_map_gzip_bytes);
  ctx = exact length of an adaptive arithmetic code (Krichevsky-Trofimov, 3 symbols) with a 9-voxel causal 3D context, + 2 B.
Maps measured: each agent's own map (what K0 must carry); each round-8 target (union of all reached own maps = what R8 decodes);
each directed tree edge's union of the sender side (what one converged message stands for).
Lossy classical baseline: block down-sampling (bx, by, bz) of the target, ternary per block (known if any voxel known; occupied if the
occupied fraction among known voxels >= tau), up-sampled back and scored with the SAME metrics3d as training (copied from train_3d_v420);
its size is the ctx code of the coarse grid. Our own prediction planes are scored too, to check the metric copy against the export."""
import base64, bz2, glob, gzip, json, lzma, math, os, sys
import numpy as np, torch
import torch.nn.functional as F

DATA = sys.argv[1] if len(sys.argv) > 1 else 'demo_web3d/data'
DIMS = (128, 128, 32)


# ---- metric (same logic as train_3d_v420.metrics3d / surface_of / dilate6)
def _pool6(m, dims):
    m = m.reshape(-1, 1, *dims).float()
    return ((F.max_pool3d(m, (3, 1, 1), stride=1, padding=(1, 0, 0)) + F.max_pool3d(m, (1, 3, 1), stride=1, padding=(0, 1, 0)) + F.max_pool3d(m, (1, 1, 3), stride=1, padding=(0, 0, 1))) > 0)


def dilate6(mask_flat, dims):
    return _pool6(mask_flat, dims).reshape(mask_flat.shape[0], -1)


def surface_of(t, dims):
    return (t == 1) & dilate6(t == 0, dims)


def _f1(a, b):
    tp = (a & b).sum().item(); fp = (a & ~b).sum().item(); fn = (~a & b).sum().item()
    return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None


def metrics(pred, target, slab, obj, struct):
    """pred, target: int8 [n, V]. surface_f1, var_occ_f1, furn_f1 per row (target-known scope, as in training)"""
    surf = surface_of(target, DIMS); region = ~slab & ~dilate6(struct[None], DIMS)[0]; rows = []
    for i in range(pred.shape[0]):
        p, t, s = pred[i], target[i], surf[i]; known = t >= 0
        po = (p == 1) & known; to = (t == 1) & known; ps = po & ~((t == 1) & ~s)
        rows.append(dict(surface_f1=_f1(ps, s), var_occ_f1=_f1(po & ~slab, to & ~slab), furn_f1=_f1(po & region, to & region)))
    return rows


# ---- lossless coders
def planes(m):
    return np.packbits(np.stack([m >= 0, m == 1]).reshape(-1)).tobytes()


def gz(m):
    return len(gzip.compress(planes(m), 9))


def bz(m):
    return len(bz2.compress(planes(m), 9))


def xz(m):
    return len(lzma.compress(planes(m), preset=9 | lzma.PRESET_EXTREME))


NB = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]


def LG(x):
    return torch.lgamma(torch.as_tensor(x, dtype=torch.float64))


def ctx_bytes(m):
    """exact adaptive KT code length (bits -> bytes) of a ternary 3D grid in C raster order, causal context of 9 earlier voxels, + 2 B"""
    s = torch.from_numpy(m.astype(np.int64) + 1)                       # 0 unknown, 1 free, 2 occ
    X, Y, Z = s.shape; c = torch.zeros_like(s)
    for dx, dy, dz in NB:                                              # out-of-grid neighbours read as 'unknown'
        sh = torch.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * 3 + sh
    idx = (c * 3 + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=3 ** (len(NB) + 1)).reshape(-1, 3).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    bits = ((LG(N + 1.5) - LG(1.5)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2


def down_up(t, f, tau):
    """block-ternary down-sample of t [X, Y, Z] with factors f, then broadcast back"""
    X, Y, Z = t.shape; bx, by, bz_ = f
    k = (t >= 0).reshape(X // bx, bx, Y // by, by, Z // bz_, bz_); o = (t == 1).reshape(X // bx, bx, Y // by, by, Z // bz_, bz_)
    nk = k.sum((1, 3, 5)); no = o.sum((1, 3, 5))
    coarse = np.where(nk > 0, (no >= np.maximum(tau * nk, 1)).astype(np.int8), np.int8(-1)).astype(np.int8)
    up = coarse.repeat(bx, 0).repeat(by, 1).repeat(bz_, 2)
    return coarse, up


def load(path):
    m = json.load(open(path)); raw = gzip.decompress(base64.b64decode(m['bin_gz_b64']))

    def plane(name):
        o, n = m['offsets'][name]
        return np.unpackbits(np.frombuffer(raw[o:o + n], dtype=np.uint8))[:int(np.prod(DIMS))].astype(bool).reshape(DIMS)
    n = m['agents']
    own = np.stack([np.where(plane(f'own_known_{a}'), plane(f'own_occ_{a}').astype(np.int8), np.int8(-1)) for a in range(n)])
    pred = {k: np.stack([np.where(plane(f'pred_{k}_known_{a}'), plane(f'pred_{k}_occ_{a}').astype(np.int8), np.int8(-1)) for a in range(n)]) for k in m['hops']}
    gt = {g: plane(f'gt_{g}') for g in ('occ', 'slab', 'obj')}
    return m, own, pred, gt


def targets(own, reached):
    """reached [n, n] (row i receives agent j) -> amax over reached own maps (= stream_targets)"""
    r = np.asarray(reached, bool)
    return np.where(r[:, :, None, None, None], own[None], np.int8(-1)).max(1)


FACTORS = [(1, 1, 2), (1, 1, 4), (2, 2, 1), (2, 2, 2), (2, 2, 4), (4, 4, 1), (4, 4, 2), (4, 4, 4), (8, 8, 2), (8, 8, 4)]
TAUS = [0.01, 0.25, 0.5]
out = dict(houses=[], own=[], union8=[], edge=[], lossy={}, ours={})
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]
    slab = torch.from_numpy(gt['slab'].reshape(-1)); obj = torch.from_numpy(gt['obj'].reshape(-1))
    struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(-1))
    for a in range(n):                                                 # own maps
        out['own'].append(dict(house=m['house'], agent=a, known=float((own[a] >= 0).mean()), gzip=gz(own[a]), bz2=bz(own[a]), lzma=xz(own[a]), ctx=ctx_bytes(own[a])))
    assert out['own'][-n]['gzip'] == m['own_map_gzip_bytes'], (out['own'][-n]['gzip'], m['own_map_gzip_bytes'])
    hi8 = m['hops'].index(8); T8 = targets(own, m['reached'][hi8]); T0 = own
    for a in range(n):                                                 # round-8 targets
        out['union8'].append(dict(house=m['house'], agent=a, reached=int(np.asarray(m['reached'][hi8])[a].sum()), known=float((T8[a] >= 0).mean()), gzip=gz(T8[a]), lzma=xz(T8[a]), ctx=ctx_bytes(T8[a])))
    adj = {i: set() for i in range(n)}
    for i, j in m['edges']:
        adj[i].add(j); adj[j].add(i)
    for i, j in m['edges']:                                            # directed tree edges: union of the sender side
        for s, r in ((i, j), (j, i)):
            side, stack = {s}, [s]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v != r and v not in side:
                        side.add(v); stack.append(v)
            U = own[sorted(side)].max(0)
            out['edge'].append(dict(house=m['house'], sender=s, side=len(side), gzip=gz(U), ctx=ctx_bytes(U)))
    for k, Tk in ((0, T0), (8, T8)):                                   # our predictions, metric copy checked against the export
        rows = metrics(torch.from_numpy(pred[k].reshape(n, -1)), torch.from_numpy(Tk.reshape(n, -1)), slab, obj, struct)
        for a in range(n):
            ref = m['metrics'][str(k)][a]
            for key in ('surface_f1', 'var_occ_f1', 'furn_f1'):
                if ref[key] is not None:
                    assert abs(ref[key] - rows[a][key]) < 1e-3, (m['house'], k, a, key, ref[key], rows[a][key])
        out['ours'].setdefault(str(k), []).extend(rows)
    for k, Tk in ((0, T0), (8, T8)):                                   # classical lossy: block down-sampling of the targets
        for f in FACTORS:
            for tau in TAUS:
                ups, sizes = [], []
                for a in range(n):
                    coarse, up = down_up(Tk[a], f, tau); ups.append(up); sizes.append(ctx_bytes(coarse))
                rows = metrics(torch.from_numpy(np.stack(ups).reshape(n, -1)), torch.from_numpy(Tk.reshape(n, -1)), slab, obj, struct)
                d = out['lossy'].setdefault(f'K{k}_{f[0]}x{f[1]}x{f[2]}_tau{tau}', dict(bytes=[], surface_f1=[], var_occ_f1=[], furn_f1=[]))
                d['bytes'] += sizes
                for r in rows:
                    for kk in ('surface_f1', 'var_occ_f1', 'furn_f1'):
                        d[kk].append(r[kk])
    out['houses'].append(m['house']); print(m['house'], 'done', flush=True)


def mean(v):
    return float(np.mean([x for x in v if x is not None]))


S = {}
for grp, cols in (('own', ('gzip', 'bz2', 'lzma', 'ctx')), ('union8', ('gzip', 'lzma', 'ctx')), ('edge', ('gzip', 'ctx'))):
    S[grp] = {c: dict(median=float(np.median([r[c] for r in out[grp]])), mean=mean([r[c] for r in out[grp]]), min=float(min(r[c] for r in out[grp])), max=float(max(r[c] for r in out[grp]))) for c in cols}
    if grp != 'edge':
        S[grp]['known'] = mean([r['known'] for r in out[grp]])
S['ours'] = {k: {kk: mean([r[kk] for r in v]) for kk in ('surface_f1', 'var_occ_f1', 'furn_f1')} for k, v in out['ours'].items()}
S['lossy'] = {k: dict(bytes_mean=mean(v['bytes']), bytes_median=float(np.median(v['bytes'])), **{kk: mean(v[kk]) for kk in ('surface_f1', 'var_occ_f1', 'furn_f1')}) for k, v in out['lossy'].items()}
json.dump(dict(summary=S, raw=out), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rate_baselines.json'), 'w'), indent=1)
print(json.dumps({k: S[k] for k in ('own', 'union8', 'edge', 'ours')}, indent=1))
for k, v in sorted(S['lossy'].items()):
    print(f"{k:24s} bytes {v['bytes_mean']:8.0f}  surf {v['surface_f1']:.3f}  var {v['var_occ_f1']:.3f}  furn {v['furn_f1']:.3f}")

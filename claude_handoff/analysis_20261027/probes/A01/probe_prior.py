"""probe_prior.py (A01, 2026-10-27): how much does a PRETRAINED prior lower the lossless cost of the observed surface set O?
O = own == 1 (observed occupied voxels); with poses the receiver re-derives the whole ternary map by ray casting (probe_raycast: exact).
Coders (binary, raster order [x, y, z], causal neighbour contexts):
  adaptive13 : per-map adaptive KT, causal half of the 3x3x3 cube (13 neighbours) -- = probe_raycast O13
  loo{k}     : static model from the OTHER 5 demo houses (40 own maps or 40 union maps), k = 13 / 18 / 22 neighbours, hierarchical
               back-off p = (n_s(c_k) + a p_13(s)) / (n(c_k) + a), p_13 itself KT-smoothed on the pooled counts
  loo{k}+ad  : the same static counts plus the map's own causal counts (pretrained + adaptive)
A learned neural prior should do at least as well as the best of these (it sees more context); this bounds what 'learn the prior,
keep the code lossless' buys. Usage: python probe_prior.py DATA_DIR OUT_JSON"""
import glob, json, math, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from rate_baselines_lib import load, targets  # noqa: E402

torch.set_num_threads(4)
DATA = sys.argv[1]; OUT = sys.argv[2]
NB13 = [(1, a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)] + [(0, 1, -1), (0, 1, 0), (0, 1, 1), (0, 0, 1)]
NB18 = NB13 + [(0, 0, 2), (0, 2, 0), (2, 0, 0), (0, 0, 3), (0, 2, 1)]
NB22 = NB18 + [(2, 1, 0), (2, -1, 0), (1, 2, 0), (1, -2, 0)]
NBS = {13: NB13, 18: NB18, 22: NB22}


def shift(s, dx, dy, dz):
    out = torch.zeros_like(s); X, Y, Z = s.shape
    xs = slice(max(dx, 0), X + min(dx, 0)); xd = slice(max(-dx, 0), X - max(dx, 0))
    ys = slice(max(dy, 0), Y + min(dy, 0)); yd = slice(max(-dy, 0), Y - max(dy, 0))
    zs = slice(max(dz, 0), Z + min(dz, 0)); zd = slice(max(-dz, 0), Z - max(dz, 0))
    out[xs, ys, zs] = s[xd, yd, zd]
    return out


def contexts(mask, nb):
    s = torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64)); c = torch.zeros_like(s)
    for d in nb:
        c = c * 2 + shift(s, *d)
    return c.reshape(-1), s.reshape(-1)


def counts(maps, k):
    tab = torch.zeros(2 ** k, 2, dtype=torch.float64)
    for mk in maps:
        c, s = contexts(mk, NBS[k])
        tab += torch.bincount(c * 2 + s, minlength=2 ** (k + 1)).reshape(-1, 2).double()
    return tab


def causal_counts(c, s):
    """for every position t: number of earlier positions with the same context and symbol 0 / 1 (raster order)"""
    order = torch.argsort(c * 2, stable=True)                       # stable: keeps raster order inside a context
    cs = c[order]; ss = s[order]
    one = ss.double(); zero = 1 - one
    c1 = torch.cumsum(one, 0) - one; c0 = torch.cumsum(zero, 0) - zero
    start = torch.ones_like(cs, dtype=torch.bool); start[1:] = cs[1:] != cs[:-1]
    grp = torch.cumsum(start.long(), 0) - 1
    first = torch.nonzero(start).flatten()
    c1 = c1 - (torch.cumsum(one, 0) - one)[first][grp]; c0 = c0 - (torch.cumsum(zero, 0) - zero)[first][grp]
    out0 = torch.empty_like(c0); out1 = torch.empty_like(c1); out0[order] = c0; out1[order] = c1
    return out0, out1


def code_bits(mask, tab13, tabk, k, adaptive, a=2.):
    c13, s = contexts(mask, NB13); ck, _ = contexts(mask, NBS[k])
    n13 = tab13[c13]; p13 = (n13 + 0.5) / (n13.sum(1, keepdim=True) + 1.)
    nk = tabk[ck].clone()
    if adaptive:
        h0, h1 = causal_counts(ck, s); nk[:, 0] += h0; nk[:, 1] += h1
    p = (nk + a * p13) / (nk.sum(1, keepdim=True) + a)
    ps = p[torch.arange(len(s)), s]
    return float(-torch.log2(ps).sum()) / 8 + 2


def adaptive13(mask):
    c, s = contexts(mask, NB13); h0, h1 = causal_counts(c, s)
    n = torch.stack((h0, h1), 1); p = (n + 0.5) / (n.sum(1, keepdim=True) + 1.)
    return float(-torch.log2(p[torch.arange(len(s)), s]).sum()) / 8 + 2


houses = []
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path)
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    houses.append(dict(house=m['house'], own=[(own[a] == 1) for a in range(own.shape[0])], union8=[(T8[a] == 1) for a in range(own.shape[0])]))
rows = []
for kind in ('own', 'union8'):
    for h in range(len(houses)):
        train = [mk for g in range(len(houses)) if g != h for mk in houses[g][kind]]
        tabs = {k: counts(train, k) for k in (13, 18, 22)}
        for a, mk in enumerate(houses[h][kind]):
            r = dict(kind=kind, house=houses[h]['house'], agent=a, adaptive13=adaptive13(mk))
            for k in (13, 18, 22):
                r[f'loo{k}'] = code_bits(mk, tabs[13], tabs[k], k, False)
                r[f'loo{k}+ad'] = code_bits(mk, tabs[13], tabs[k], k, True)
            rows.append(r)
        print(kind, houses[h]['house'], {k: round(float(np.median([x[k] for x in rows if x['kind'] == kind and x['house'] == houses[h]['house']])), 1) for k in rows[-1] if k not in ('kind', 'house', 'agent')}, flush=True)
S = {kind: {k: float(np.median([x[k] for x in rows if x['kind'] == kind])) for k in rows[0] if k not in ('kind', 'house', 'agent')} for kind in ('own', 'union8')}
json.dump(dict(summary=S, rows=rows), open(OUT, 'w'), indent=1)
print(json.dumps(S, indent=1))

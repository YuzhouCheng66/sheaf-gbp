"""A02 probe P2 (2026-10-27, CPU only): how much does a PRIOR SHARED BY ENCODER AND DECODER save over a per-map adaptive coder?
Leave-one-house-out over the 6 demo VAL houses (train counts on 5 houses = 40 maps, code the 8 maps of the held-out house with STATIC
probabilities = the "decoder-side prior"), against the per-map adaptive KT coder of rate_baselines_lib (ctx).  Payloads:
  full  = the ternary map (unknown / free / occupied), the thing ctx codes today;
  hits  = the observed-occupied voxel set alone (binary), which with the poses determines the full map (probe P1).
Contexts: nb9 = the 9 causal neighbours of ctx; z = absolute layer (32); rng = distance bucket from the nearest reached agent centre
(<1, <2, <3, <4, <5.2, beyond: beyond 5 m + jitter nothing can be observed); nb13 = nb9 + 4 more causal neighbours.
Static model probability = (n_s + a * p_backoff) / (n + a), backoff chain nb13+z+rng -> nb9+z+rng -> nb9 -> uniform, a = 2.
Maps: own (K0 payload, 48) and the round-8 union targets (48).  Output: probes/A02/p2_prior_ctx.json"""
import glob, json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from rate_baselines_lib import load, targets, ctx_bytes  # noqa: E402
torch.set_num_threads(16)

NB9 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]
NB13 = NB9 + [(1, -1, 0), (1, 0, -1), (0, 1, -1), (1, 1, 1)]          # (dx, dy, dz): neighbour (x-dx, y-dy, z-dz), all earlier in C order


def shift(s, d):
    """value of neighbour (x-dx, y-dy, z-dz) at every voxel; outside the grid reads 0"""
    out = torch.zeros_like(s); X, Y, Z = s.shape
    sx = slice(max(d[0], 0), X + min(d[0], 0)); tx = slice(max(-d[0], 0), X - max(d[0], 0))
    sy = slice(max(d[1], 0), Y + min(d[1], 0)); ty = slice(max(-d[1], 0), Y - max(d[1], 0))
    sz = slice(max(d[2], 0), Z + min(d[2], 0)); tz = slice(max(-d[2], 0), Z - max(d[2], 0))
    out[sx, sy, sz] = s[tx, ty, tz]
    return out


def range_bucket(centres, cell, shape):
    X, Y, Z = shape
    xs = (np.arange(X) + .5) * cell[0]; ys = (np.arange(Y) + .5) * cell[1]; zs = (np.arange(Z) + .5) * cell[2]
    g = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1)
    d = np.min([np.linalg.norm(g - np.asarray(c)[None, None, None], axis=-1) for c in centres], axis=0)
    return torch.from_numpy(np.digitize(d, [1., 2., 3., 4., 5.2]).astype(np.int64))         # 0..5


def contexts(sym, k, rb, nb):
    """sym: int64 [X,Y,Z] in 0..k-1; returns dict of context-index tensors for the backoff chain"""
    c = torch.zeros_like(sym)
    for d in nb:
        c = c * k + shift(sym, d)
    z = torch.arange(sym.shape[2]).view(1, 1, -1).expand_as(sym)
    return c, z, rb


class Static:
    """static (pre-trained) context model with dense counts and a 3-level backoff chain"""
    def __init__(self, k, nb):
        self.k, self.nb = k, nb

    def keys(self, sym, rb):
        c, z, _ = contexts(sym, self.k, rb, self.nb); c9, _, _ = contexts(sym, self.k, rb, NB9)
        n_c, n_c9 = self.k ** len(self.nb), self.k ** len(NB9)
        return [(c * 192 + z * 6 + rb, n_c * 192), (c9 * 192 + z * 6 + rb, n_c9 * 192), (c9, n_c9)]

    def fit(self, samples):
        self.cnt = None
        for sym, rb in samples:
            ks = self.keys(sym, rb)
            if self.cnt is None:
                self.cnt = [torch.zeros(size * self.k, dtype=torch.float64) for _, size in ks]
            for lvl, (key, size) in enumerate(ks):
                self.cnt[lvl] += torch.bincount((key * self.k + sym).reshape(-1), minlength=size * self.k).double()
        return self

    def bits(self, sym, rb, alpha=2.):
        ks = self.keys(sym, rb); s = sym.reshape(-1)
        p = torch.full(s.shape, 1. / self.k, dtype=torch.float64)
        for lvl in reversed(range(len(ks))):
            key = ks[lvl][0].reshape(-1); C = self.cnt[lvl].view(-1, self.k)
            n = C[key].sum(1); ns = C[key, s]
            p = (ns + alpha * p) / (n + alpha)
        return float(-torch.log2(p).sum()) / 8


def binary_ctx_adaptive(mask):
    from rate_baselines_lib import NB, LG
    s = torch.from_numpy(mask.astype(np.int64)); X, Y, Z = s.shape; c = torch.zeros_like(s)
    for dx, dy, dz in NB:
        sh = torch.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * 2 + sh
    idx = (c * 2 + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=2 ** (len(NB) + 1)).reshape(-1, 2).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    bits = ((LG(N + 1.) - LG(1.)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2


if __name__ == '__main__':
    data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'demo_web3d', 'data')
    houses = []
    for path in sorted(glob.glob(os.path.join(data, 'house_*.json'))):
        m, own, pred, gt = load(path); n = own.shape[0]
        R8 = np.asarray(m['reached'][m['hops'].index(8)], bool); T8 = targets(own, R8)
        pos = [np.asarray(p, float) for p in m['positions_m']]
        maps = dict(own=[(own[a], range_bucket([pos[a]], m['cell_m'], own.shape[1:])) for a in range(n)],
                    union8=[(T8[a], range_bucket([pos[j] for j in np.nonzero(R8[a])[0]], m['cell_m'], own.shape[1:])) for a in range(n)])
        houses.append(dict(name=m['house'], maps=maps))
        print('loaded', m['house'], flush=True)
    rows = []
    for hold in range(len(houses)):
        for kind in ('own', 'union8'):
            train = [mp for h_, H in enumerate(houses) if h_ != hold for mp in H['maps'][kind]]
            full_models = {nm: Static(3, nb).fit([(torch.from_numpy(t.astype(np.int64) + 1), rb) for t, rb in train]) for nm, nb in (('nb9', NB9),)}
            hit_models = {nm: Static(2, nb).fit([(torch.from_numpy((t == 1).astype(np.int64)), rb) for t, rb in train]) for nm, nb in (('nb9', NB9), ('nb13', NB13))}
            for a, (t, rb) in enumerate(houses[hold]['maps'][kind]):
                full = torch.from_numpy(t.astype(np.int64) + 1); hits = torch.from_numpy((t == 1).astype(np.int64))
                r = dict(house=houses[hold]['name'], kind=kind, agent=a, n_hits=int(hits.sum()), known=float((t >= 0).mean()),
                         full_adaptive=ctx_bytes(t), hits_adaptive=binary_ctx_adaptive(t == 1))
                for nm, M in full_models.items():
                    r[f'full_static_{nm}'] = M.bits(full, rb)
                for nm, M in hit_models.items():
                    r[f'hits_static_{nm}'] = M.bits(hits, rb)
                rows.append(r)
                print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    S = {}
    for kind in ('own', 'union8'):
        rr = [r for r in rows if r['kind'] == kind]
        S[kind] = {k: dict(median=float(np.median([r[k] for r in rr])), mean=float(np.mean([r[k] for r in rr])))
                   for k in rr[0] if k not in ('house', 'kind', 'agent')}
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p2_prior_ctx.json'), 'w'), indent=1)

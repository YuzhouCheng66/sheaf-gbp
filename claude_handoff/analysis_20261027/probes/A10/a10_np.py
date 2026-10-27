"""a10_np.py (A10, 2026-10-27): NUMPY-ONLY twin of rate_baselines_lib (load / ctx / metrics) plus conditional and innovation context
coders, written because the shared CPU box was out of commit memory (44 python processes) and a torch import costs ~1.6 GB.
Every coder is an EXACT adaptive arithmetic-code length (Krichevsky-Trofimov, Dirichlet 1/2 per context, + 2 B): an implementable
two-sided code whose learning cost is included -- the same family as rate_baselines_lib.ctx_bytes (equality asserted in a10_probe).
Side information is used only where the DECODER has it."""
import base64, gzip, json, math
import numpy as np

DIMS = (128, 128, 32)
NB = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]
NB6 = NB[:6]
FWD = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
_lg = np.vectorize(math.lgamma, otypes=[np.float64])


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
    r = np.asarray(reached, bool)
    return np.where(r[:, :, None, None, None], own[None], np.int8(-1)).max(1)


def kt_bits(sym, ctx, nsym):
    """adaptive KT code length (bits) of the symbol stream `sym` (values 0..nsym-1) under per-context counts"""
    sym = np.asarray(sym).reshape(-1); ctx = np.asarray(ctx).reshape(-1)
    if sym.size == 0:
        return 0., 0
    u, inv = np.unique(ctx, return_inverse=True)
    cnt = np.bincount(inv * nsym + sym, minlength=len(u) * nsym).reshape(-1, nsym).astype(np.float64)
    N = cnt.sum(1); a = 0.5
    bits = ((_lg(N + a * nsym) - math.lgamma(a * nsym)).sum() - (_lg(cnt + a) - math.lgamma(a)).sum()) / math.log(2)
    return float(bits), len(u)


def causal(g, base, nb=NB):
    X, Y, Z = g.shape
    c = np.zeros(g.shape, np.int64)
    for dx, dy, dz in nb:
        sh = np.zeros(g.shape, np.int64)
        sh[dx:, dy:, dz:] = g[:X - dx, :Y - dy, :Z - dz]
        c = c * base + sh
    return c


def fwd_flags(mask):
    X, Y, Z = mask.shape
    f = np.zeros(mask.shape, np.int64)
    for dx, dy, dz in FWD:
        sh = np.zeros(mask.shape, bool)
        sh[:X - dx, :Y - dy, :Z - dz] = mask[dx:, dy:, dz:]
        f = f * 2 + sh
    return f


def T(m):
    return m.astype(np.int64) + 1                        # ternary -1/0/1 -> 0/1/2


def tern_bytes(m):
    t = T(m)
    return kt_bits(t, causal(t, 3), 3)[0] / 8 + 2


def bin_bytes(v, nb=NB):
    b = v.astype(np.int64)
    return kt_bits(b, causal(b, 2, nb), 2)[0] / 8 + 2


def down_up(t, f, tau):
    """= rate_baselines_lib.down_up: block-ternary down-sample of t [X, Y, Z] with factors f, then broadcast back"""
    X, Y, Z = t.shape; bx, by, bz_ = f
    k = (t >= 0).reshape(X // bx, bx, Y // by, by, Z // bz_, bz_); o = (t == 1).reshape(X // bx, bx, Y // by, by, Z // bz_, bz_)
    nk = k.sum((1, 3, 5)); no = o.sum((1, 3, 5))
    coarse = np.where(nk > 0, (no >= np.maximum(tau * nk, 1)).astype(np.int8), np.int8(-1)).astype(np.int8)
    up = coarse.repeat(bx, 0).repeat(by, 1).repeat(bz_, 2)
    return coarse, up


def dilate6(m):
    """6-neighbourhood dilation (incl. centre) of a bool grid, same as rate_baselines_lib._pool6"""
    o = m.copy()
    o[1:] |= m[:-1]; o[:-1] |= m[1:]
    o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
    o[:, :, 1:] |= m[:, :, :-1]; o[:, :, :-1] |= m[:, :, 1:]
    return o


def surface_of(t):
    return (t == 1) & dilate6(t == 0)


def _f1(a, b):
    tp = int((a & b).sum()); fp = int((a & ~b).sum()); fn = int((~a & b).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None


def metrics1(p, t, slab, struct):
    """surface / variable / furniture-region F1 of one prediction p against target t (target-known scope), = rate_baselines_lib.metrics"""
    s = surface_of(t); region = ~slab & ~dilate6(struct); known = t >= 0
    po = (p == 1) & known; to = (t == 1) & known; ps = po & ~((t == 1) & ~s)
    return dict(surface_f1=_f1(ps, s), var_occ_f1=_f1(po & ~slab, to & ~slab), furn_f1=_f1(po & region, to & region))


# ---------------------------------------------------------------- tree utilities
def adjacency(edges, n):
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    return adj


def side_of(adj, s, r):
    side, stack = {s}, [s]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v != r and v not in side:
                side.add(v); stack.append(v)
    return sorted(side)


def hops_from(adj, src):
    import collections
    d = {src: 0}; q = collections.deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


def union(own, idx):
    return own[sorted(idx)].max(0)


# ---------------------------------------------------------------- physics side information
def fib(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); th = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)), 1)


def reachable(blocker, cell, pos, rays=16384, rng_m=5.2):
    """voxels a sensor at `pos` (public centre position, metres) could observe if `blocker` (the RECEIVER's known-occupied voxels)
    were the only obstacles; receiver-unknown voxels are transparent. 1-voxel dilated. Used as a context flag only (codes stay
    lossless whatever it says)."""
    shape = np.array(blocker.shape); cell = np.asarray(cell, np.float64)
    out = np.zeros(blocker.shape, bool)
    R = fib(rays); ds = 0.5 * float(cell.min()); n = int(math.ceil(rng_m / ds))
    alive = np.ones(len(R), bool)
    for s in range(1, n + 1):
        idx = np.floor((pos[None] + R * (s * ds)) / cell).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape), 1) & alive
        if not inside.any():
            break
        ii = idx[inside]
        out[ii[:, 0], ii[:, 1], ii[:, 2]] = True
        hit = blocker[ii[:, 0], ii[:, 1], ii[:, 2]]
        ai = np.nonzero(inside)[0]; alive[ai[hit]] = False
        alive &= inside
    return dilate6(out)

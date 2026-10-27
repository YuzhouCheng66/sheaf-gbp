"""p9_entropy_model.py (A07): how far below the adaptive context code could a PRE-TRAINED (learned) entropy model take the exact,
sensor-aware lossless sheaf (hit set H + poses, decoder re-casts rays)? Static context models are trained on 5 houses and code the 6th
(leave-one-house-out, re-simulated own maps and round-8 unions of sim_cache), so no per-map learning cost is paid -- the regime of a
learned prior. Contexts (binary H, raster order x, y, z):
  A  : the 9 causal neighbours of rate_baselines (adaptive KT on the map itself = ctx_bin; the evidence-pack coder);
  B  : A, static (pre-trained counts, add-0.5);
  C  : 13 causal neighbours of the 3x3x3 half-neighbourhood + the z index (32) + in-range of a pose (<= 5 m) -- static;
  D  : C + the coarse parent: the 2x2x2 block's 'any hit' bit, sent first (its own static 13-neighbour code) -- two-level, static.
Bytes include the coarse level for D and 5.5 B per agent of poses for all."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a07_lib import ctx_bin, POSE_BYTES

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, 'sim_cache')
NB9 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]
NB13 = [(1, dy, dz) for dy in (-1, 0, 1) for dz in (-1, 0, 1)] + [(0, 1, dz) for dz in (-1, 0, 1)] + [(0, 0, 1)]


def shifted(m, d):
    """value of m at (x - dx, y - dy, z - dz) (0 outside)."""
    X, Y, Z = m.shape; out = np.zeros_like(m); dx, dy, dz = d
    xs = slice(dx, X) if dx >= 0 else slice(0, X + dx); xd = slice(0, X - dx) if dx >= 0 else slice(-dx, X)
    ys = slice(dy, Y) if dy >= 0 else slice(0, Y + dy); yd = slice(0, Y - dy) if dy >= 0 else slice(-dy, Y)
    zs = slice(dz, Z) if dz >= 0 else slice(0, Z + dz); zd = slice(0, Z - dz) if dz >= 0 else slice(-dz, Z)
    out[xs, ys, zs] = m[xd, yd, zd]
    return out


def causal_ctx(m, nbs):
    c = np.zeros(m.shape, np.int64)
    for d in nbs:
        sh = shifted(m.astype(np.int64), d)
        if d[0] == 0 and d[1] == 0 and d[2] < 0:
            sh[:] = 0
        c = c * 2 + sh
    return c


def in_range(shape, cell, poses):
    X, Y, Z = shape; g = np.stack(np.meshgrid(*[(np.arange(n) + .5) * c for n, c in zip(shape, cell)], indexing='ij'), -1)
    d = np.min([np.linalg.norm(g - p, axis=-1) for p in poses.reshape(-1, 3)], 0)
    return (d <= 5.05).astype(np.int64)


def features(H, cell, poses, mode):
    z = np.broadcast_to(np.arange(H.shape[2])[None, None], H.shape)
    if mode == 'B':
        return causal_ctx(H, NB9)
    c13 = causal_ctx(H, NB13); rng = in_range(H.shape, cell, poses)
    base = (c13 * 32 + z) * 2 + rng
    if mode == 'C':
        return base
    X, Y, Z = H.shape; coarse = H.reshape(X // 2, 2, Y // 2, 2, Z // 2, 2).any((1, 3, 5))
    par = coarse.repeat(2, 0).repeat(2, 1).repeat(2, 2).astype(np.int64)
    return base * 2 + par


def coarse_feats(H, cell, poses):
    X, Y, Z = H.shape; coarse = H.reshape(X // 2, 2, Y // 2, 2, Z // 2, 2).any((1, 3, 5))
    zc = np.broadcast_to(np.arange(Z // 2)[None, None], coarse.shape)
    rng = in_range(coarse.shape, np.asarray(cell) * 2, poses)
    return coarse, (causal_ctx(coarse, NB13) * 16 + zc) * 2 + rng


NCTX = dict(B=2 ** 9, C=2 ** 13 * 32 * 2, D=2 ** 13 * 32 * 2 * 2, coarse=2 ** 13 * 16 * 2)


def map_feats(H, cell, P):
    f = {m: (features(H, cell, P, m).reshape(-1), H.reshape(-1).astype(np.int64)) for m in ('B', 'C', 'D')}
    co, cf = coarse_feats(H, cell, P); f['coarse'] = (cf.reshape(-1), co.reshape(-1).astype(np.int64))
    return f


def counts_of(c, s, nctx):
    return np.stack([np.bincount(c[s == 0], minlength=nctx), np.bincount(c[s == 1], minlength=nctx)], 1).astype(np.float64)


def code_bits(cnt, c, s, alpha=0.5):
    p = (cnt[c, s] + alpha) / (cnt[c].sum(1) + 2 * alpha)
    return float(-np.log2(p).sum())


def main():
    names = sorted(f[:-4] for f in os.listdir(CACHE) if f.endswith('.npz'))
    maps = {}
    for nm in names:
        z = np.load(os.path.join(CACHE, nm + '.npz')); own, R8 = z['own'], z['reached8']
        T8 = np.where(R8[:, :, None, None, None], own[None], np.int8(-1)).max(1)
        L = [('own', own[a] == 1, z['poses'][a]) for a in range(own.shape[0])]
        seen = set()
        for a in range(own.shape[0]):
            key = tuple(np.nonzero(R8[a])[0])
            if key not in seen:
                seen.add(key); L.append(('union8', T8[a] == 1, z['poses'][list(key)]))
        maps[nm] = (z['cell'], L)
    hc = {}
    for nm in names:
        cell, L = maps[nm]; acc = {m: np.zeros((n, 2)) for m, n in NCTX.items()}
        for kind, H, P in L:
            f = map_feats(H, cell, P)
            for m in NCTX:
                acc[m] += counts_of(*f[m], NCTX[m])
        hc[nm] = acc; print(nm, len(L), flush=True)
    tot = {m: sum(hc[nm][m] for nm in names) for m in NCTX}
    rows = []
    for nm in names:
        cell, L = maps[nm]; cnt = {m: tot[m] - hc[nm][m] for m in NCTX}
        for kind, H, P in L:
            f = map_feats(H, cell, P); ag = len(P) if kind == 'union8' else 1; pb = POSE_BYTES * ag
            r = dict(house=nm, kind=kind, agents=ag, A_adaptive9=ctx_bin(H) + pb)
            for m in ('B', 'C'):
                r[f'{m}_static'] = code_bits(cnt[m], *f[m]) / 8 + pb
            r['D_static_2level'] = (code_bits(cnt['coarse'], *f['coarse']) + code_bits(cnt['D'], *f['D'])) / 8 + pb
            rows.append(r)
    S = {}
    for kind in ('own', 'union8'):
        rr = [r for r in rows if r['kind'] == kind]
        S[kind] = {k: float(np.median([r[k] for r in rr])) for k in ('A_adaptive9', 'B_static', 'C_static', 'D_static_2level')}; S[kind]['n'] = len(rr)
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p9_entropy_model.json'), 'w'), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()

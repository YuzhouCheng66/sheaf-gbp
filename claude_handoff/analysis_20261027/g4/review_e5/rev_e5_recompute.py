"""Independent review recompute for G4-E5 (D8-52), reviewer, 2026-10-27. CPU only, seed 1 VAL only (continuity seed; seeds 3 / 4 are
never read: the script refuses any file whose name does not end in _s1.npz).

Own code: quantiser (DP over c-order, E1c value formula), components / sender sides from the edge list, direct sums, features,
logit binning, furniture F1 / free recall, the level-count coder's ideal code length (written from the docstring spec of
g4_quant_wire.py, not from its code). Shared pieces: UNet3 weights loading (E.load_ckpt), G.int_ctx_bytes (E3-R's coder, the control's
byte definition), G.score as a cross-check only.

  panel : all 24 VAL houses, seed 1, cam_nf, Qinf with both checkpoints -> R8 NET furniture (component-expanded mean) and the median
          per-house S / k message bytes; compared with e3r/summary.json (NET_342101 / NET_342202).
  house : val_r1_i0_s1, val_r1_i1_s1: Qinf / Q3 per-house NET furniture, per-edge Sk / lev bytes, B; compared with the implementer's
          smoke job files e5_smoke/eval/jobs/*.json.
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
sys.dont_write_bytecode = True
import json, math, time, glob, multiprocessing
import numpy as np
from math import lgamma

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
AN = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
INP = os.path.join(AN, 'review_E3R', 'inputs')
EXP = os.path.join(INP, 'export')
FIT = os.path.join(AN, 'g4', 'e1c', 'fit.json')
SMOKE = os.path.join(AN, 'g4', 'e5_smoke', 'eval', 'jobs')
OUT = os.path.join(HERE, 'out')
LMIN, LMAX, UNK = -40, 70, -128
NL = LMAX - LMIN + 1
CK = {'342101': 'A', '342202': 'B'}
NB9 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]


def jl(p):
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------------ quantiser (own)
def e1c_c(no, nf):
    return np.log((no + .5) / (no.sum() + .5 * NL)) - np.log((nf + .5) / (nf.sum() + .5 * NL))


def hcost(o, f):
    n = o + f
    return 0. if n <= 0 else -sum(x * math.log(x / n) for x in (o, f) if x > 0)


def dp(no, nf, q):
    c = e1c_c(no, nf)
    order = np.argsort(c, kind='stable'); a = np.r_[0., np.cumsum(no[order])]; b = np.r_[0., np.cumsum(nf[order])]; n = NL
    best = {(0, 0): (0., None)}
    for k in range(1, q + 1):
        for j in range(k, n + 1):
            cand = [(best[(k - 1, i)][0] + hcost(a[j] - a[i], b[j] - b[i]), i) for i in range(k - 1, j) if (k - 1, i) in best]
            if cand:
                best[(k, j)] = min(cand)
    cuts = []; j = n
    for k in range(q, 0, -1):
        i = best[(k, j)][1]; cuts.append((i, j)); j = i
    lab = np.zeros(n, np.int64)
    for bi, (i, j) in enumerate(cuts[::-1]):
        lab[order[i:j]] = bi
    vals = []
    for l in range(q):
        po, pf = no[lab == l].sum(), nf[lab == l].sum()
        v = (math.log((po + .5) / (no.sum() + .5 * NL)) - math.log((pf + .5) / (nf.sum() + .5 * NL))) / 0.05
        vals.append(int(np.rint(v)))
    return lab, np.array(vals, np.int64), best[(q, n)][0]


def tables():
    fit = jl(FIT); out = {}
    for h in 'AB':
        t = fit['halves'][h]['tiers']['cam_nf']; no = np.array(t['n_occ'], float); nf = np.array(t['n_free'], float)
        lab, vals, cost = dp(no, nf, 3)
        out[h] = dict(Qinf=np.array(t['cq'], np.int64), Q3=vals[lab], Q3_lab=lab, Q3_vals=vals, theta=int(t['theta']), cost=cost)
    return out


# ------------------------------------------------------------------ forest (own)
def comps_sides(edges, n):
    und = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j, _ in np.asarray(edges).reshape(-1, 3)})
    nb = {a: [] for a in range(n)}
    for i, j in und:
        nb[i].append(j); nb[j].append(i)
    seen = set(); comps = []
    for a in range(n):
        if a in seen:
            continue
        st = [a]; c = {a}
        while st:
            u = st.pop()
            for w in nb[u]:
                if w not in c:
                    c.add(w); st.append(w)
        seen |= c; comps.append(sorted(c))
    comps.sort()
    sides = {}
    for i, j in und:
        for s, r in ((i, j), (j, i)):
            st = [s]; c = {s}
            while st:
                u = st.pop()
                for w in nb[u]:
                    if w != r and w not in c:
                        c.add(w); st.append(w)
            sides[(s, r)] = sorted(c)
    return comps, sides


# ------------------------------------------------------------------ scoring (own)
def dil6(m):
    o = m.copy()
    o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]; o[:, :, 1:] |= m[:, :, :-1]; o[:, :, :-1] |= m[:, :, 1:]
    return o


def furn_free(pred, tgt, occ, slab, obj):
    struct = (occ == 1) & ~obj & ~slab
    region = ~slab & ~dil6(struct)
    known = tgt >= 0; tr = (tgt == 1) & region
    pr = (pred == 1) & known & region
    tp = int((pr & tr).sum()); fp = int((pr & ~tr).sum()); fn = int((~pr & tr).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None
    fr = tgt == 0
    return f1, (float(((pred == 0) & fr).sum()) / fr.sum()) if fr.any() else None


# ------------------------------------------------------------------ NET (weights via E.load_ckpt; features / binning own)
def net_logit(model, S, k):
    import torch
    Z = S.shape[2]
    f = np.empty((1, 3) + S.shape, np.float32)
    f[0, 0] = S.astype(np.float32) * np.float32(0.05 / 4.0)
    f[0, 1] = np.log1p(k.astype(np.float32))
    f[0, 2] = ((np.arange(Z) + 0.5) / Z * 2.0 - 1.0).astype(np.float32)[None, None, :]
    with torch.no_grad():
        return model(torch.from_numpy(f)).float().numpy()[0]


def readout(lg, k, j):
    b = np.clip(np.floor((lg.astype(np.float64) + 40.0) / 0.02), 0, 3999).astype(np.int64)
    return np.where(k > 0, (b >= j).astype(np.int8), np.int8(-1))


# ------------------------------------------------------------------ level-count coder ideal length (own, from the spec)
def shift(a, d, fill):
    dx, dy, dz = d; b = np.full(a.shape, fill, a.dtype); X, Y, Z = a.shape
    b[dx:, dy:, dz:] = a[:X - dx, :Y - dy, :Z - dz]; return b


def kt_bits_from(keys, bits):
    """keys: int64 context ids of every decision, bits: bool; KT(1/2) adaptive code length in bits"""
    if keys.size == 0:
        return 0.
    u, inv = np.unique(keys, return_inverse=True)
    n1 = np.bincount(inv, weights=bits.astype(np.float64)); n = np.bincount(inv).astype(np.float64); n0 = n - n1
    tot = 0.
    for a, b in zip(n0, n1):
        tot += lgamma(a + b + 1) - lgamma(a + .5) - lgamma(b + .5) + 2 * lgamma(.5)
    return tot / math.log(2)


def lev_ideal_bytes(N):
    """N [q, X, Y, Z] counts. pass 1 known bits (9 causal nbs); pass 2 unary k-1; pass 3 per non-dropped level unary given remainder"""
    N = np.asarray(N, np.int64); q = N.shape[0]; k = N.sum(0); K = k > 0
    tot = N.reshape(q, -1).sum(1); drop = int(np.argmax(tot)); kmax = int(k.max()) if K.any() else 0
    keys, bits = [], []
    # pass 1
    c = np.zeros(k.shape, np.int64)
    for i, d in enumerate(NB9):
        c += shift(K.astype(np.int64), d, 0) << i
    keys.append(c.ravel() + 0); bits.append(K.ravel())
    # pass 2 (tag 1 << 40)
    kz = np.minimum(shift(k, (0, 0, 1), 0), 4); ky = shift(k, (0, 1, 0), 0); kx = shift(k, (1, 0, 0), 0); kyx = np.minimum(np.maximum(ky, kx), 4)
    m = k - 1
    for i in range(max(kmax - 1, 0)):
        sel = K & (m >= i)
        keys.append((1 << 40) + (min(i, 3) * 100 + kz[sel] * 10 + kyx[sel])); bits.append(m[sel] > i)
    # pass 3 (tag 2 << 40): context (slot, min(r, 3), min(i, 2), class of z-1, y-1, x-1 neighbours)
    cls = np.argmax(N, 0); cls = np.where(K, cls, q)
    cz = shift(cls, (0, 0, 1), q); cy = shift(cls, (0, 1, 0), q); cx = shift(cls, (1, 0, 0), q)
    nbk = (cz * 16 + cy) * 16 + cx
    r = k.copy(); slot = 0
    for l in range(q):
        if l == drop:
            continue
        n_l = N[l]
        for i in range(int(r.max()) if K.any() else 0):
            sel = K & (r > i) & (n_l >= i)
            if not sel.any():
                break
            key = (2 << 40) + (((slot * 4 + np.minimum(r[sel], 3)) * 4 + min(i, 2)) * 4096 + nbk[sel])
            keys.append(key); bits.append(n_l[sel] > i)
        r = r - n_l; slot += 1
    assert np.array_equal(r, N[drop])
    keys = np.concatenate([np.asarray(x, np.int64).ravel() for x in keys]); bits = np.concatenate([np.asarray(x, bool).ravel() for x in bits])
    return kt_bits_from(keys, bits) / 8 + 4.0


# ------------------------------------------------------------------ jobs
def house_job(args):
    fname, mode = args
    assert fname.endswith('_s1.npz') and fname.startswith('val_'), 'reviewer recompute: seed 1 VAL only'
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
    import torch
    torch.set_num_threads(1)
    import g4_e3r as E
    G = E.G
    t0 = time.time()
    TB = tables()
    z = np.load(os.path.join(EXP, fname))
    occ, slab, obj, priv = z['occ'], z['slab'], z['obj'], z['private']; n = priv.shape[0]
    comps, sides = comps_sides(z['edges'], n)
    L = z['L_k0_cam_nf']
    known = [L[a] != UNK for a in range(n)]
    kk = [x.astype(np.int32) for x in known]
    H = G.House(occ, slab, obj, z['cell'])
    out = dict(file=fname, comps=comps, ck={})
    for sd, h in CK.items():
        ck, model = E.load_ckpt(os.path.join(INP, 's' + sd, 'final.pt'), 'cpu')
        j = int(jl(os.path.join(INP, 's' + sd, 'threshold.json'))['j'])
        arms = ('Qinf',) if mode == 'panel' else ('Qinf', 'Q3')
        R = {}
        for arm in arms:
            tab = TB[h][arm]
            e = [np.where(known[a], tab[np.clip(L[a].astype(np.int64) - LMIN, 0, NL - 1)], 0).astype(np.int32) for a in range(n)]
            rows = []
            for cm in comps:
                S = sum(e[a] for a in cm); kc = sum(kk[a] for a in cm)
                tgt = priv[cm].max(0)
                p = readout(net_logit(model, S, kc), kc, j)
                f1, fr = furn_free(p, tgt, occ, slab, obj)
                gs = G.score(p, G.Target(tgt, H))
                assert (f1 is None and gs['furn_f1'] is None) or abs(f1 - gs['furn_f1']) < 1e-12, 'own F1 != G.score'
                rows.append(dict(size=len(cm), furn=f1, free=fr))
            edges = {}
            for (s, r), sd_ in sides.items():
                S = sum(e[a] for a in sd_); kc = sum(kk[a] for a in sd_); Km = kc > 0
                bS = G.int_ctx_bytes(S, Km); bk = G.int_ctx_bytes(kc, Km)
                ent = [bS, bk]
                if arm != 'Qinf':
                    lab = TB[h]['Q3_lab']; vals = TB[h]['Q3_vals']
                    Nl = np.zeros((3,) + S.shape, np.int64)
                    for a in sd_:
                        li = lab[np.clip(L[a].astype(np.int64) - LMIN, 0, NL - 1)]
                        for l in range(3):
                            Nl[l] += known[a] & (li == l)
                    assert np.array_equal(np.tensordot(vals, Nl, axes=1), S) and np.array_equal(Nl.sum(0), kc)
                    ent.append(lev_ideal_bytes(Nl))
                edges[f'{s}>{r}'] = ent
            R[arm] = dict(rows=rows, edges=edges, S_sum=float(sum(v[0] for v in edges.values())), k_sum=float(sum(v[1] for v in edges.values())))
            if arm != 'Qinf':
                Sk = sum(v[0] + v[1] for v in edges.values()); lev = sum(v[2] for v in edges.values())
                R[arm].update(Sk=float(Sk), lev=float(lev), B=float(min(Sk, lev) + 1 / 8))
            else:
                R[arm].update(Sk=float(R[arm]['S_sum'] + R[arm]['k_sum']))
        out['ck'][sd] = R
    out['seconds'] = time.time() - t0
    return out


def main():
    mode = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    if mode == 'quant':
        TB = tables()
        crit = jl(os.path.join(AN, 'g4', 'e5_quant', 'criteria_quantizer.json'))['quantizer']
        for h in 'AB':
            same = TB[h]['Q3'].tolist() == crit[h]['Q3']['table']
            print(f"half {h}: own Q3 values {TB[h]['Q3_vals'].tolist()} sizes {np.bincount(TB[h]['Q3_lab']).tolist()} cost {TB[h]['cost']:.3f} "
                  f"impl {crit[h]['Q3']['values']} cost {crit[h]['Q3']['cost_nats']:.3f}  table identical: {same}")
        return
    if mode == 'panel':
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(EXP, 'val_*_s1.npz')))
    else:
        files = ['val_r1_i0_s1.npz', 'val_r1_i1_s1.npz']
    assert all(f.endswith('_s1.npz') for f in files)
    with multiprocessing.get_context('spawn').Pool(4) as pool:
        res = pool.map(house_job, [(f, mode) for f in files])
    with open(os.path.join(OUT, f'{mode}.json'), 'w') as fh:
        json.dump(res, fh)
    print(f'{mode}: {len(res)} houses, max {max(r["seconds"] for r in res):.0f}s')


if __name__ == '__main__':
    main()

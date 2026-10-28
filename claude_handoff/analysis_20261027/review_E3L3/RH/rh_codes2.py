"""rh_codes2.py (R-H, 2026-10-28): rich causal contexts + a FIXED (learnt offline) probability table = the "learnt entropy model"
estimate for the frozen E3-L3 messages (latents and accuracy unchanged; lossless).
Per message (k, d = k - s) [+ side info (k', d') at the voxel]:
  K ctx = (m, a = k[z-1], relation (<, =, >) of k at (y-1), (x-1), (y-1, z-1), (x-1, z-1), (x-1, y-1) to a) [x side]; symbol k in 0..m
  D ctx = (k, [k[z-1] == k], class d[z-1], class d[y-1], class d[x-1]) with class 0, 1, 2, 3, 4+ / n/a [x side]; symbol d in 0..2k (on k > 0)
  side  = (min(k', 4), class d') of the reverse message (= the receiver's partial belief; both ends hold it)
Static table: leave-one-house-out within the (checkpoint, noise seed) cell (23 other VAL houses), estimator (n + alpha) / (N + M alpha),
alpha = 0.5; the table is part of the codec (fitted offline, e.g. on TRAIN in a real run), not sent. Adaptive KT with the same
contexts (state reset per message, no table) is reported as well.
Sets per house: every directed message (plain and, for second messages, | reverse), the total T once, the self maps, and the
floor estimate sum_a cost(T - own_a | own_a) (= what every agent must at least receive: H(T | X_a)).
usage: python -B rh_codes2.py --jobs J1,J2 --cache C1,C2 --out O.json [--workers 8]
"""
import os
import sys

sys.dont_write_bytecode = True
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, glob, json, math, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402
from scipy.special import gammaln  # noqa: E402

LN2 = math.log(2.0)
ALPHA = 0.5
MA = 17                                    # max alphabet (d <= 2 x 8)
NREL = [(0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]   # (dx, dy, dz) causal offsets besides z-1


def nb(v, dx, dy, dz, fill):
    o = np.full_like(v, fill); X, Y, Z = v.shape
    o[dx:, dy:, dz:] = v[:X - dx, :Y - dy, :Z - dz]
    return o


def dcl(d, k):
    return np.where(k > 0, np.minimum(d, 4), 5)


def side_ctx(k2, d2):
    return np.minimum(k2, 4) * 6 + dcl(d2, k2)               # 30 values


def keys_k(k, m, side=None):
    a = nb(k, 0, 0, 1, 0)
    rel = np.zeros(k.shape, np.int64)
    for dx, dy, dz in NREL:
        rel = rel * 3 + (np.sign(nb(k, dx, dy, dz, 0) - a) + 1)
    ctx = (m * 9 + a) * 243 + rel
    if side is not None:
        ctx = ctx * 30 + side
    return ctx.ravel(), k.ravel().astype(np.int64), np.full(k.size, m + 1, np.int64)


def keys_d(k, d, side=None):
    sup = k > 0
    ka = nb(k, 0, 0, 1, 0)
    ca = dcl(nb(d, 0, 0, 1, 0), ka); cb = dcl(nb(d, 0, 1, 0, 0), nb(k, 0, 1, 0, 0)); cc = dcl(nb(d, 1, 0, 0, 0), nb(k, 1, 0, 0, 0))
    ctx = ((((k.astype(np.int64) * 2 + (ka == k)) * 6 + ca) * 6 + cb) * 6 + cc)
    if side is not None:
        ctx = ctx * 30 + side
    return ctx[sup], d[sup].astype(np.int64), (2 * k[sup] + 1).astype(np.int64)


def sparse(ctx, sym, M):
    key = ctx * MA + sym
    u, c = np.unique(key, return_counts=True)
    Mu = np.zeros(len(u), np.int64)
    _, first = np.unique(key, return_index=True)
    Mu = M[first]
    return u, c, Mu


def kt_adapt(u, c, Mu):
    """adaptive KT bits from the sparse counts of ONE message (state reset per message)"""
    if len(u) == 0:
        return 0.0
    ctx = u // MA
    uc, ci = np.unique(ctx, return_inverse=True)
    N = np.bincount(ci, weights=c); Mc = np.zeros(len(uc)); Mc[ci] = Mu
    b = (gammaln(N + Mc * ALPHA) - gammaln(Mc * ALPHA)).sum() - (gammaln(c + ALPHA) - gammaln(ALPHA)).sum()
    return float(b / LN2)


def msg(k, d, m, side=None):
    ck, sk, Mk = keys_k(k, m, side); cd, sd, Md = keys_d(k, d, side)
    return dict(K=sparse(ck, sk, Mk), D=sparse(cd, sd, Md))


def house(job):
    jpath, cache = job
    rec = dict(job=os.path.basename(jpath))
    try:
        t0 = time.time()
        J = json.load(open(jpath)); tag = J['file'][:-4]; sd = J['seed']
        code = np.load(os.path.join(cache, f'{tag}_{sd}.npz'))['code']; n = code.shape[0]
        tau = (code > 0).astype(np.int64); dz = np.where(code > 0, 3 - code, 0).astype(np.int64)
        M = {}
        for e in J['edges']:
            M[(e['s'], e['r'])] = (tau[e['side']].sum(0), dz[e['side']].sum(0), len(e['side']))
        order = sorted(M, key=lambda q: (M[q][2], q)); pos = {q: i for i, q in enumerate(order)}
        items = []                                                   # (set name, msg id, sparse dict)
        for q in order:
            k, d, m = M[q]; rv = (q[1], q[0]); second = rv in M and pos[rv] < pos[q]
            items.append(('plain', f'{q[0]}>{q[1]}|{int(second)}', msg(k, d, m)))
            if second:
                items.append(('rev', f'{q[0]}>{q[1]}', msg(k, d, m, side_ctx(M[rv][0], M[rv][1]))))
        K = tau.sum(0); D = dz.sum(0)
        items.append(('T', 'T', msg(K, D, n)))
        for a in range(n):
            items.append(('self', f'self{a}', msg(tau[a], dz[a], 1)))
            items.append(('floor', f'floor{a}', msg(K - tau[a], D - dz[a], n - 1, side_ctx(tau[a], dz[a]))))
        rec.update(file=J['file'], seed=sd, n=n, items=[(s, i, {ch: [x.tolist() for x in v[ch]] for ch in ('K', 'D')}) for s, i, v in items],
                   seconds=time.time() - t0)
    except Exception as ex:
        rec['failed'] = True; rec['error'] = repr(ex)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def static_eval(recs):
    """leave-one-house-out static tables per (set family: plain/T/self share one table 'P'; rev and floor share 'S'), per channel"""
    fam = dict(plain='P', T='P', self='P', rev='S', floor='S')
    pool = {}
    for r in recs:
        for s, i, v in r['items']:
            for ch in ('K', 'D'):
                u, c, Mu = (np.asarray(x, np.int64) for x in v[ch])
                pool.setdefault((fam[s], ch), []).append((u, c, Mu))
    tabs = {}
    for key, lst in pool.items():
        u = np.concatenate([x[0] for x in lst]); c = np.concatenate([x[1] for x in lst]); Mu = np.concatenate([x[2] for x in lst])
        uu, inv = np.unique(u, return_inverse=True)
        tc = np.bincount(inv, weights=c); tm = np.zeros(len(uu), np.int64); tm[inv] = Mu
        cx = uu // MA; ux, ix = np.unique(cx, return_inverse=True); tn = np.bincount(ix, weights=tc)
        tabs[key] = (uu, tc, tm, ux, tn)
    out = []
    for r in recs:
        own = {}                                                     # this house's own counts per table (to leave out)
        for s, i, v in r['items']:
            for ch in ('K', 'D'):
                u, c, _ = (np.asarray(x, np.int64) for x in v[ch])
                own.setdefault((fam[s], ch), []).append((u, c))
        ownt = {}
        for key, lst in own.items():
            u = np.concatenate([x[0] for x in lst]); c = np.concatenate([x[1] for x in lst])
            uu, inv = np.unique(u, return_inverse=True); ownt[key] = (uu, np.bincount(inv, weights=c))
        res = []
        for s, i, v in r['items']:
            row = dict(set=s, id=i)
            for ch in ('K', 'D'):
                u, c, Mu = (np.asarray(x, np.int64) for x in v[ch])
                row[f'{ch}_adapt'] = kt_adapt(u, c, Mu) / 8
                if len(u) == 0:
                    row[f'{ch}_static'] = 0.0; continue
                uu, tc, tm, ux, tn = tabs[(fam[s], ch)]; ou, oc = ownt[(fam[s], ch)]
                n_sym = tc[np.searchsorted(uu, u)] - oc[np.searchsorted(ou, u)]
                cx = u // MA
                n_ctx = tn[np.searchsorted(ux, cx)]
                o_cx = ou // MA; oux, oix = np.unique(o_cx, return_inverse=True); on = np.bincount(oix, weights=oc)
                n_ctx = n_ctx - on[np.searchsorted(oux, cx)]
                p = (n_sym + ALPHA) / (n_ctx + Mu * ALPHA)
                row[f'{ch}_static'] = float(-(c * np.log2(p)).sum() / 8)
            res.append(row)
        out.append(dict(file=r['file'], seed=r['seed'], n=r['n'], rows=res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True); ap.add_argument('--cache', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); assert a.workers <= 8
    cells = {}
    import multiprocessing as mp
    t0 = time.time()
    for jd, cd in zip(a.jobs.split(','), a.cache.split(',')):
        jobs = [(j, cd) for j in sorted(glob.glob(os.path.join(jd, '*.json')))]
        with mp.get_context('spawn').Pool(a.workers) as pool:
            recs = list(pool.imap_unordered(house, jobs))
        bad = [r for r in recs if r.get('failed')]
        assert not bad, bad[0]
        noise = os.path.basename(os.path.dirname(jd.rstrip('/\\'))).split('_s')[-1] if 'extract_s' in jd else jd
        for ck in sorted({r['seed'] for r in recs}):
            cells[f'{ck}|{noise}'] = static_eval([r for r in recs if r['seed'] == ck])
        print(jd, f'{time.time() - t0:.0f}s', flush=True)
    json.dump(dict(alpha=ALPHA, cells=cells), open(a.out, 'w'))


if __name__ == '__main__':
    main()

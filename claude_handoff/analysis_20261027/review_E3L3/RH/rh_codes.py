"""rh_codes.py (R-H, 2026-10-28): lossless code lengths of the E3-L3 GaBP messages (pure numpy, CPU).
Input: rh_extract.py outputs (jobs/*.json with the O7 bytes and the directed edges / sides; cached code arrays in the scratchpad).
Every message on edge s -> r with sender side A is (k, s) = (sum_A tau_a, sum_A z_a) per voxel (asserted exact in rh_extract);
d = k - s = sum_A (1 - z_a) in [0, 2k].
All coders are sequential adaptive KT (Dirichlet alpha) estimators with CAUSAL contexts in C raster order (z fastest), whose
exact code length is order-independent per context = sum_c [lnG(N_c + M_c a) - lnG(M_c a) - sum_j (lnG(n_cj + a) - lnG(a))] / ln 2.
A context only uses (i) already decoded voxels of the same message (causal neighbours), (ii) the k channel of the same voxel
(k is coded first, then d), (iii) side information the DECODER holds; for 'practical' coders also the ENCODER holds it.
State is reset per message (no learnt prior). + 4 B header per message per coder.
Coders (per message):
  o7        : the frozen O7 bytes (from rh_extract; L.wire_bytes)
  k_o7      : O7 mask + k values only;  s_o7: O7 s values only
  K1        : k on the whole grid, ctx = (k[z-1], k[y-1], k[x-1]) clipped to 0..m (m = |A|), alphabet m + 1
  D1        : d given k on k > 0, ctx = (min(k, 4), class d[z-1], class d[y-1]), class in {0, 1, 2, 3+, n/a}; alphabet 2k + 1
  K1|own, D1|own : + receiver's own (tau_r, d_r) at the voxel (decoder-only side info -> Slepian-Wolf estimate)
  K1|rev, D1|rev : + the reverse message on the same edge (k', class d') at the voxel (both ends hold it: practical)
  K1|bel, D1|bel : + receiver's partial belief before this message (own + earlier incoming), (k', class d') (decoder-only)
Schedule = L.gabp order (sender side size ascending): on each edge the smaller-side message is 'first', the other 'second';
for the second message the reverse message = the receiver's whole partial belief (sums), held by both ends.
Also: total (K, S) once (broadcast reference), per-agent own (tau, z) with K1 / D1 (sum of self maps).
usage: python -B rh_codes.py --jobs J1,J2 --cache C1,C2 --out O.json [--workers 8] [--alpha 0.5]
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
HDR = 4.0
ALPHA = 0.5


def kt_bits(sym, ctx, M, alpha=None):
    """sym int >= 0 [N], ctx int >= 0 [N], M per-voxel alphabet size [N] or int (must be constant within a context)"""
    a = ALPHA if alpha is None else alpha
    sym = np.asarray(sym, np.int64).ravel(); ctx = np.asarray(ctx, np.int64).ravel()
    if sym.size == 0:
        return 0.0
    Mv = np.broadcast_to(np.asarray(M, np.int64), sym.shape).ravel()
    assert (sym < Mv).all() and (sym >= 0).all()
    uc, ci = np.unique(ctx, return_inverse=True)
    Mc = np.zeros(len(uc), np.int64); Mc[ci] = Mv
    assert np.array_equal(Mc[ci], Mv), 'alphabet not constant within a context'
    Mx = int(Mv.max())
    cnt = np.bincount(ci * Mx + sym, minlength=len(uc) * Mx).reshape(len(uc), Mx).astype(np.float64)
    N = cnt.sum(1)
    b = (gammaln(N + Mc * a) - gammaln(Mc * a)) - (gammaln(cnt + a) - gammaln(a)).sum(1)
    return float(b.sum() / LN2)


def shift(v, ax, fill):
    """value of the causal neighbour at index - 1 along axis ax (out of grid = fill)"""
    o = np.full_like(v, fill)
    if ax == 0:
        o[1:] = v[:-1]
    elif ax == 1:
        o[:, 1:] = v[:, :-1]
    else:
        o[:, :, 1:] = v[:, :, :-1]
    return o


def dclass(d, k):
    """0, 1, 2, 3+ where k > 0; 4 = n/a (k == 0 or out of grid)"""
    return np.where(k > 0, np.minimum(d, 3), 4)


def code_k(k, m, side=None):
    """K1 (+ side info ctx [g] in 0..S-1): whole grid; returns bits"""
    kc = np.minimum(k, m)
    a, b, c = (shift(kc, ax, 0) for ax in (2, 1, 0))
    ctx = (a * (m + 1) + b) * (m + 1) + c
    if side is not None:
        ctx = ctx * (int(side.max()) + 1) + side
    return kt_bits(k.ravel(), ctx.ravel(), m + 1)


def code_d(k, d, side=None):
    """D1 (+ side info ctx): on k > 0, alphabet 2k + 1; ctx = (k, class d[z-1], class d[y-1]) (+ side)"""
    sup = k > 0
    if not sup.any():
        return 0.0
    ca = shift(dclass(d, k), 2, 4); cb = shift(dclass(d, k), 1, 4)
    kk = k.astype(np.int64)                                        # k is part of the context: alphabet 2k + 1 constant per ctx
    ctx = (kk * 5 + ca) * 5 + cb
    if side is not None:
        ctx = ctx * (int(side.max()) + 1) + side
    return kt_bits(d[sup], ctx[sup], (2 * kk + 1)[sup])


def msg_bits(k, d, m, sk=None, sd=None):
    """bytes of one message: K1 + D1 (+ side ctx), + header"""
    kb = code_k(k, m, sk); db = code_d(k, d, sd)
    return dict(k=kb / 8 + HDR / 2, d=db / 8 + HDR / 2, tot=(kb + db) / 8 + HDR)


def side_ctx(k2, d2, kmax=None):
    """side info at the voxel: (min(k2, kmax), class d2) packed; kmax small to limit contexts"""
    kc = np.minimum(k2, kmax) if kmax is not None else k2
    return kc * 5 + dclass(d2, k2)


def house(job):
    jpath, cache, alpha = job
    global ALPHA
    ALPHA = alpha
    rec = dict(job=os.path.basename(jpath))
    try:
        t0 = time.time()
        J = json.load(open(jpath))
        tag = J['file'][:-4]; sd = J['seed']
        code = np.load(os.path.join(cache, f'{tag}_{sd}.npz'))['code']
        n = code.shape[0]
        tau = (code > 0).astype(np.int16); dz = np.where(code > 0, 3 - code, 0).astype(np.int16)    # 1 - z: +1 -> 0, 0 -> 1, -1 -> 2
        rec.update(file=J['file'], seed=sd, n=n, o7=J['o7_bytes'])
        msgs = {}
        for e in J['edges']:
            sdn = e['side']
            k = tau[sdn].sum(0); d = dz[sdn].sum(0)
            msgs[(e['s'], e['r'])] = (k, d, len(sdn), e)
        # schedule = sender side size ascending (L.gabp order); ties by (s, r)
        order = sorted(msgs, key=lambda q: (msgs[q][2], q))
        pos = {q: i for i, q in enumerate(order)}
        rows = []
        for q in order:
            k, d, m, e = msgs[q]; s, r = q
            rv = msgs.get((r, s)); second = rv is not None and pos[(r, s)] < pos[q]
            row = dict(s=s, r=r, m=m, second=bool(second), o7=e['wire'], k_o7=e['mask'] + e['k'], s_o7=e['s_ch'], support=e['support'])
            p = msg_bits(k, d, m); row.update(K1=p['k'], D1=p['d'], KD=p['tot'])
            own_k = tau[r]; own_d = dz[r]
            p = msg_bits(k, d, m, sk=side_ctx(own_k, own_d), sd=side_ctx(own_k, own_d)); row.update(KD_own=p['tot'], K_own=p['k'], D_own=p['d'])
            # receiver's partial belief before q: own + incoming earlier in the order
            bk = own_k.copy(); bd = own_d.copy()
            for q2 in order[:pos[q]]:
                if q2[1] == r:
                    bk = bk + msgs[q2][0]; bd = bd + msgs[q2][1]
            p = msg_bits(k, d, m, sk=side_ctx(bk, bd, 4), sd=side_ctx(bk, bd, 4)); row.update(KD_bel=p['tot'], K_bel=p['k'], D_bel=p['d'])
            if second:
                k2, d2 = rv[0], rv[1]
                assert np.array_equal(k2, bk) and np.array_equal(d2, bd), 'reverse message != receiver partial belief'
                p = msg_bits(k, d, m, sk=side_ctx(k2, d2, 4), sd=side_ctx(k2, d2, 4)); row.update(KD_rev=p['tot'], K_rev=p['k'], D_rev=p['d'])
                # same info as 'send the total, receiver subtracts': code (K, D) of the total given the reverse message
                Kt = k + k2; Dt = d + d2
                p2 = msg_bits(Kt, Dt, n, sk=side_ctx(k2, d2, 4), sd=side_ctx(k2, d2, 4)); row.update(KD_tot_rev=p2['tot'])
            rows.append(row)
        rec['msgs'] = rows
        Kt = tau.sum(0); Dt = dz.sum(0)
        p = msg_bits(Kt, Dt, n); rec['total_once'] = p['tot']; rec['total_once_k'] = p['k']
        rec['self'] = [msg_bits(tau[a], dz[a], 1)['tot'] for a in range(n)]
        rec['self_o7'] = J['self_bytes']
        rec['zfrac'] = [int((code == v).sum()) for v in (1, 2, 3)]
        rec['seconds'] = time.time() - t0
    except Exception as ex:
        rec['failed'] = True; rec['error'] = repr(ex)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True); ap.add_argument('--cache', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--alpha', type=float, default=0.5); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    assert a.workers <= 8
    jobs = []
    for jd, cd in zip(a.jobs.split(','), a.cache.split(',')):
        js = sorted(glob.glob(os.path.join(jd, '*.json')))
        if a.limit:
            js = js[:a.limit]
        jobs += [(j, cd, a.alpha) for j in js]
    t0 = time.time(); out = []
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for r in pool.imap_unordered(house, jobs):
            out.append(r)
            print(r['job'], 'FAILED ' + r['error'] if r.get('failed') else f"{r['seconds']:.0f}s", f'{time.time() - t0:.0f}s', flush=True)
    with open(a.out, 'w') as fh:
        json.dump(dict(alpha=a.alpha, recs=out), fh, indent=1)


if __name__ == '__main__':
    main()

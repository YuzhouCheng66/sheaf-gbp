"""rh_tctx.py (R-H): how far does context richness lower the code length of the component total T = (K, D) (K = sum tau, D = sum (1 - z))?
Static (fixed-table) causal-context coders of increasing richness, leave-one-house-out within each (ckpt, noise) cell, alpha = 0.5;
plus the floor version sum_a cost(T - own_a | own_a) for the best design. Code length per voxel = -log2 (n_LOO + a) / (N_LOO + M a).
usage: python -B rh_tctx.py --jobs J1,J2 --cache C1,C2 --out O.json [--workers 8]"""
import os
import sys

sys.dont_write_bytecode = True
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, glob, json, time  # noqa: E401,E402
import numpy as np  # noqa: E402

AL = 0.5
MA = 17


def nb(v, dx, dy, dz, fill=0):
    o = np.full_like(v, fill); X, Y, Z = v.shape
    o[dx:, dy:, dz:] = v[:X - dx, :Y - dy, :Z - dz]
    return o


def rel(u, a):
    return np.sign(u - a) + 1


def dcl(d, k):
    return np.where(k > 0, np.minimum(d, 4), 5)


def k_designs(K, n, side=None):
    a, b, c = nb(K, 0, 0, 1), nb(K, 0, 1, 0), nb(K, 1, 0, 0)
    A = (a * 9 + b) * 9 + c
    r4 = ((rel(nb(K, 0, 1, 1), a) * 3 + rel(nb(K, 1, 0, 1), a)) * 3 + rel(nb(K, 1, 1, 0), a)) * 3 + rel(nb(K, 0, 0, 2), a)
    Bc = A * 81 + r4
    r3 = (rel(nb(K, 0, 2, 0), b) * 3 + rel(nb(K, 2, 0, 0), c)) * 3 + rel(nb(K, 1, 1, 1), a)
    Cc = Bc * 27 + r3
    out = dict(KA=A, KB=Bc, KC=Cc)
    if side is not None:
        out = {k + '|own': v * 30 + side for k, v in out.items()}
    return {k: (v.ravel(), K.ravel(), n + 1) for k, v in out.items()}


def d_designs(K, D, side=None):
    sup = (K > 0).ravel()
    ka, kb, kc = nb(K, 0, 0, 1), nb(K, 0, 1, 0), nb(K, 1, 0, 0)
    da, db, dc = dcl(nb(D, 0, 0, 1), ka), dcl(nb(D, 0, 1, 0), kb), dcl(nb(D, 1, 0, 0), kc)
    A = (((K * 2 + (ka == K)) * 6 + da) * 6 + db) * 6 + dc
    Bc = ((A * 2 + (kb == K)) * 2 + (kc == K)) * 36 + dcl(nb(D, 0, 1, 1), nb(K, 0, 1, 1)) * 6 + dcl(nb(D, 1, 0, 1), nb(K, 1, 0, 1))
    Cc = (Bc * 6 + dcl(nb(D, 1, 1, 0), nb(K, 1, 1, 0))) * 6 + dcl(nb(D, 0, 0, 2), nb(K, 0, 0, 2))
    out = dict(DA=A, DB=Bc, DC=Cc)
    if side is not None:
        out = {k + '|own': v * 30 + side for k, v in out.items()}
    M = (2 * K + 1).ravel()
    return {k: (v.ravel()[sup], D.ravel()[sup], M[sup]) for k, v in out.items()}


def sp(ctx, sym, M):
    key = ctx.astype(np.int64) * MA + sym
    u, first, c = np.unique(key, return_index=True, return_counts=True)
    Mv = np.broadcast_to(np.asarray(M), sym.shape)
    return u, c, np.asarray(Mv)[first]


def house(job):
    jpath, cache = job
    J = json.load(open(jpath)); tag = J['file'][:-4]; sd = J['seed']
    code = np.load(os.path.join(cache, f'{tag}_{sd}.npz'))['code']; n = code.shape[0]
    tau = (code > 0).astype(np.int64); dz = np.where(code > 0, 3 - code, 0).astype(np.int64)
    K = tau.sum(0); D = dz.sum(0)
    items = {}
    for k, v in {**k_designs(K, n), **d_designs(K, D)}.items():
        items[('T', k)] = sp(*v)
    for a in range(n):                                              # floor: T - own_a | own_a (own = (tau_a, class d_a))
        side = np.minimum(tau[a], 4) * 6 + dcl(dz[a], tau[a])
        Ka = K - tau[a]; Da = D - dz[a]
        for k, v in {**k_designs(Ka, n - 1, side), **d_designs(Ka, Da, side)}.items():
            if k.startswith(('KC', 'DC', 'KA', 'DA', 'KB', 'DB')):
                items[(f'F{a}', k)] = sp(*v)
    return dict(file=J['file'], seed=sd, n=n, items=items)


def loo(recs):
    pools = {}
    for r in recs:
        for (s, d), (u, c, M) in r['items'].items():
            pools.setdefault(d, []).append((u, c, M))
    tabs = {}
    for d, lst in pools.items():
        u = np.concatenate([x[0] for x in lst]); c = np.concatenate([x[1] for x in lst]); M = np.concatenate([x[2] for x in lst])
        uu, inv = np.unique(u, return_inverse=True); tc = np.bincount(inv, weights=c)
        cx = uu // MA; ux, ix = np.unique(cx, return_inverse=True); tn = np.bincount(ix, weights=tc)
        tabs[d] = (uu, tc, ux, tn)
    out = []
    for r in recs:
        own = {}
        for (s, d), (u, c, M) in r['items'].items():
            own.setdefault(d, []).append((u, c))
        ownt = {}
        for d, lst in own.items():
            u = np.concatenate([x[0] for x in lst]); c = np.concatenate([x[1] for x in lst])
            uu, inv = np.unique(u, return_inverse=True); oc = np.bincount(inv, weights=c)
            ox, oi = np.unique(uu // MA, return_inverse=True); on = np.bincount(oi, weights=oc)
            ownt[d] = (uu, oc, ox, on)
        res = {}
        for (s, d), (u, c, M) in r['items'].items():
            uu, tc, ux, tn = tabs[d]; ou, oc, ox, on = ownt[d]
            ns = tc[np.searchsorted(uu, u)] - oc[np.searchsorted(ou, u)]
            cx = u // MA
            nc = tn[np.searchsorted(ux, cx)] - on[np.searchsorted(ox, cx)]
            res[f'{s}:{d}'] = float(-(c * np.log2((ns + AL) / (nc + M * AL))).sum() / 8)
        out.append(dict(file=r['file'], seed=r['seed'], n=r['n'], bytes=res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True); ap.add_argument('--cache', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); assert a.workers <= 8
    import multiprocessing as mp
    cells = {}; t0 = time.time()
    for jd, cd in zip(a.jobs.split(','), a.cache.split(',')):
        noise = os.path.basename(os.path.dirname(jd.rstrip('/\\'))).split('_s')[-1]
        with mp.get_context('spawn').Pool(a.workers) as pool:
            recs = list(pool.imap_unordered(house, [(j, cd) for j in sorted(glob.glob(os.path.join(jd, '*.json')))]))
        for ck in sorted({r['seed'] for r in recs}):
            cells[f'{ck}|{noise}'] = loo([r for r in recs if r['seed'] == ck])
        print(jd, f'{time.time() - t0:.0f}s', flush=True)
    json.dump(cells, open(a.out, 'w'), indent=1)
    S = {}
    for cell, hs in cells.items():
        keys = sorted(hs[0]['bytes'])
        tk = [k for k in keys if k.startswith('T:')]
        row = {k: float(np.median([h['bytes'][k] for h in hs])) for k in tk}
        for dk, kk in (('KA', 'DA'), ('KB', 'DB'), ('KC', 'DC')):
            row[f'floor {kk}+{dk}'] = float(np.median([sum(h['bytes'][f'F{i}:{dk}|own'] + h['bytes'][f'F{i}:{kk}|own'] + 4 for i in range(h['n'])) for h in hs]))
        S[cell] = row
        print(cell, {k: round(v) for k, v in row.items()})


if __name__ == '__main__':
    main()

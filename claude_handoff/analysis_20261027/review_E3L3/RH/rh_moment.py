"""rh_moment.py (R-H, REPORT ONLY, a model change, not lossless for the frozen decoder): code length of the belief in MOMENT form
T' = (known mask K > 0, belief mean x = S / K) vs the information form T = (K, S). Tells how much of T's cost is K's interior values,
which the frozen decoder consumes only through log1p(K). Static LOO tables as rh_tctx.py.
  mask: binary, ctx = 9 causal neighbours of the mask (rate_baselines NB)
  x   : index of the reduced fraction (K - D) / K on the mask, ctx = (class x[z-1], class x[y-1], class x[x-1], class x[y-1, z-1]),
        class = 8 bins of x (=1, (.5, 1), .5, (0, .5), 0, (-.5, 0), <= -.5) + n/a
usage: python -B rh_moment.py --jobs J1,J2 --cache C1,C2 --out O.json"""
import os
import sys

sys.dont_write_bytecode = True
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, glob, json, time  # noqa: E401,E402
from fractions import Fraction  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rh_tctx as T  # noqa: E402

NB9 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]
FR = sorted({Fraction(s, k) for k in range(1, 9) for s in range(-k, k + 1)})
FIDX = {f: i for i, f in enumerate(FR)}
LUT = np.full((9, 17), -1, np.int64)
for k in range(1, 9):
    for d in range(0, 2 * k + 1):
        LUT[k, d] = FIDX[Fraction(k - d, k)]
T.MA = max(len(FR) + 1, 17)


def xclass(K, D):
    x = np.where(K > 0, (K - D) / np.maximum(K, 1), 9.0)
    c = np.select([x > 1.5, x == 1, x > .5, x == .5, x > 0, x == 0, x > -.5], [8, 0, 1, 2, 3, 4, 5], 6)
    return c


def house(job):
    jpath, cache = job
    J = json.load(open(jpath)); tag = J['file'][:-4]; sd = J['seed']
    code = np.load(os.path.join(cache, f'{tag}_{sd}.npz'))['code']; n = code.shape[0]
    tau = (code > 0).astype(np.int64); dz = np.where(code > 0, 3 - code, 0).astype(np.int64)
    K = tau.sum(0); D = dz.sum(0); m = (K > 0).astype(np.int64)
    ctx = np.zeros(K.shape, np.int64)
    for dx, dy, dz_ in NB9:
        ctx = ctx * 2 + T.nb(m, dx, dy, dz_, 0)
    items = {('T', 'mask'): T.sp(ctx.ravel(), m.ravel(), 2)}
    xc = xclass(K, D)
    cx = ((T.nb(xc, 0, 0, 1, 8) * 9 + T.nb(xc, 0, 1, 0, 8)) * 9 + T.nb(xc, 1, 0, 0, 8)) * 9 + T.nb(xc, 0, 1, 1, 8)
    sup = K > 0
    xi = LUT[K[sup], D[sup]]
    assert (xi >= 0).all()
    items[('T', 'x')] = T.sp(cx[sup], xi, len(FR))
    return dict(file=J['file'], seed=sd, n=n, items=items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True); ap.add_argument('--cache', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import multiprocessing as mp
    out = {}
    for jd, cd in zip(a.jobs.split(','), a.cache.split(',')):
        noise = os.path.basename(os.path.dirname(jd.rstrip('/\\'))).split('_s')[-1]
        with mp.get_context('spawn').Pool(8) as pool:
            recs = list(pool.imap_unordered(house, [(j, cd) for j in sorted(glob.glob(os.path.join(jd, '*.json')))]))
        for ck in sorted({r['seed'] for r in recs}):
            hs = T.loo([r for r in recs if r['seed'] == ck])
            row = {k: float(np.median([h['bytes'][k] for h in hs])) for k in hs[0]['bytes']}
            row['T_moment'] = float(np.median([h['bytes']['T:mask'] + h['bytes']['T:x'] + 4 for h in hs]))
            out[f'{ck}|{noise}'] = row
            print(ck, noise, {k: round(v) for k, v in row.items()}, flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

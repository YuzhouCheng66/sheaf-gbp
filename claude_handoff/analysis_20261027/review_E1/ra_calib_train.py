"""R-A (E1 review) stage 2, TRAIN houses only (npz from ra_probe_train.py, cam_nf tier, noise seed 7).
Zero-learning per-agent calibration table B6: c(l) = log P(L = l | occ) / P(L = l | free), fitted leave-one-house-out on the other TRAIN
houses; fused l = l0 + sum_a c(L_a) (prior counted once); threshold fitted on the other houses (max pooled furniture F1 subject to free
recall >= the raw sum's). Also: calibration slope of the fused log-odds by noisy multiplicity k (conditional-independence check),
and the V1 integrator under thresholds / the same table (is V1's .32 a read-out threshold problem or lost information?).
usage: python -B ra_calib_train.py NPZ_DIR > out.json"""
import os, sys, glob, json, collections
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'scripts', 'claude_v342'))
import numpy as np
import g4_noise_panels as G

LV = np.arange(-40, 71)                    # clamped V3 / V1 alphabet
THR = np.arange(-6.0, 6.01, 0.5)           # threshold grid in natural log-odds units


def load(d, tier='cam_nf'):
    H = []
    for f in sorted(glob.glob(os.path.join(d, f'*_{tier}.npz'))):
        z = np.load(f)
        priv = z['private']; comp_of = z['comps']; n = priv.shape[0]
        comps = [list(np.nonzero(comp_of == c)[0]) for c in sorted(set(comp_of.tolist()))]
        Ho = G.House(z['occ'], z['slab'], z['obj'], z['cell'])
        R = []
        for cm in comps:
            T8 = priv[cm].max(0)
            T = G.Target(T8, Ho, mult=(priv[cm] >= 0).sum(0))
            R.append((cm, T))
        H.append(dict(name=os.path.basename(f), L=z['L'], Lv1=z['Lv1'], occ=z['occ'].astype(bool), priv=priv, comps=R, Ho=Ho))
    return H


def hist(h, key):
    """per-agent counts of (l, occ) and (l, free) over voxels the agent knows inside its component's scoring scope"""
    no = np.zeros(len(LV)); nf = np.zeros(len(LV))
    for cm, T in h['comps']:
        for a in cm:
            L = h[key][a]; k = (L != G.UNK) & T.known
            y = h['occ'][k]; l = L[k].astype(np.int64) + 40
            no += np.bincount(l[y], minlength=len(LV)); nf += np.bincount(l[~y], minlength=len(LV))
    return no, nf


def table(hs, key):
    no = sum(hist(h, key)[0] for h in hs); nf = sum(hist(h, key)[1] for h in hs)
    c = np.log((no + .5) / (no.sum() + .5 * len(LV))) - np.log((nf + .5) / (nf.sum() + .5 * len(LV)))
    return c, float(np.log(no.sum() / nf.sum()))


def fused(h, cm, key, c=None, l0=0.):
    S = np.zeros(h['occ'].shape, np.float64); K = np.zeros(h['occ'].shape, bool); kcount = np.zeros(h['occ'].shape, np.int64)
    for a in cm:
        L = h[key][a]; k = L != G.UNK
        S[k] += (c[L[k].astype(np.int64) + 40] if c is not None else L[k] * G.LO_UNIT); K |= k; kcount += k
    return S + (l0 if c is not None else 0.), K, kcount


def counts(h, key, c, l0, thr):
    """per threshold: pooled furniture tp/fp/fn and free hits / free total, R8 (component sums)"""
    out = np.zeros((len(thr), 5))
    rows = {t: [] for t in range(len(thr))}
    for cm, T in h['comps']:
        S, K, _ = fused(h, cm, key, c, l0)
        for i, t in enumerate(thr):
            p = np.where(K, (S > t).astype(np.int8), np.int8(-1))
            r = G.score(p, T)
            out[i] += [r['furn_tp'] * len(cm), r['furn_fp'] * len(cm), r['furn_fn'] * len(cm),
                       ((p == 0) & T.free).sum() * len(cm), T.free.sum() * len(cm)]
            rows[i].append((r, len(cm)))
    return out, rows


def f1(x):
    return 2 * x[0] / max(2 * x[0] + x[1] + x[2], 1)


def slope(lg, y):
    """1-D logistic regression y ~ sigmoid(a + b * lg), L-BFGS on the mean log-loss (robust to near-separation); returns (a, b)"""
    from scipy.optimize import minimize
    lg = np.clip(lg, -20, 20)
    def f(w):
        z = w[0] + w[1] * lg; p = 1 / (1 + np.exp(-z))
        ll = np.mean(np.logaddexp(0, z) - y * z)
        g = np.array([np.mean(p - y), np.mean((p - y) * lg)])
        return ll, g
    r = minimize(f, np.array([0., 1.]), jac=True, method='L-BFGS-B')
    return float(r.x[0]), float(r.x[1])


def main():
    hs = load(sys.argv[1])
    res = dict(houses=[h['name'] for h in hs])
    for key in ('L', 'Lv1'):
        thr_raw = THR; acc = collections.defaultdict(lambda: np.zeros(5)); cal_lg = collections.defaultdict(list); cal_y = collections.defaultdict(list)
        mean_rows = collections.defaultdict(list); chosen = []
        pre = []
        for i, h in enumerate(hs):
            others = [x for j, x in enumerate(hs) if j != i]
            c, l0 = table(others, key)
            pre.append((c, l0))
        # per-house threshold curves (raw and calibrated, LOO table)
        curves = [(counts(h, key, None, 0., thr_raw), counts(h, key, *pre[i], THR)) for i, h in enumerate(hs)]
        for i, h in enumerate(hs):
            (raw_c, raw_rows), (cal_c, cal_rows) = curves[i]
            oth = [j for j in range(len(hs)) if j != i]
            raw_o = sum(curves[j][0][0] for j in oth); cal_o = sum(curves[j][1][0] for j in oth)
            i0 = int(np.argmin(np.abs(thr_raw - 0.0)))                               # raw sum > 0 (E1 B4 read-out)
            free_ref = raw_o[i0, 3] / raw_o[i0, 4]
            def pick(o):
                ok = [t for t in range(len(o)) if o[t, 3] / o[t, 4] >= free_ref - 1e-12]
                return max(ok, key=lambda t: f1(o[t]))
            ir, ic = pick(raw_o), pick(cal_o)
            chosen.append(dict(house=h['name'], raw_thr=float(thr_raw[ir]), cal_thr=float(THR[ic]), l0=pre[i][1]))
            for tag, cc, rr, t in (('raw0', raw_c, raw_rows, i0), ('raw_tuned', raw_c, raw_rows, ir), ('cal0', cal_c, cal_rows, int(np.argmin(np.abs(THR)))),
                                   ('cal_tuned', cal_c, cal_rows, ic)):
                acc[tag] += cc[t]; mean_rows[tag] += [r['furn_f1'] for r, k in rr[t] for _ in range(k)]
                mean_rows[tag + '_free'] += [r['free_recall'] for r, k in rr[t] for _ in range(k)]
            # calibration by noisy multiplicity k, in the scoring scope (all voxels, and furniture region only)
            for cm, T in h['comps']:
                S, K, kc = fused(h, cm, key, *pre[i]); m = K & T.known
                for kk in range(1, 9):
                    sel = m & (kc == kk)
                    if sel.sum():
                        cal_lg[('all', min(kk, 5))].append(S[sel]); cal_y[('all', min(kk, 5))].append(h['occ'][sel])
                    sel2 = sel & h['Ho'].region
                    if sel2.sum():
                        cal_lg[('furnreg', min(kk, 5))].append(S[sel2]); cal_y[('furnreg', min(kk, 5))].append(h['occ'][sel2])
                Sr, _, _ = fused(h, cm, key)
                for kk in range(1, 9):
                    sel = m & (kc == kk)
                    if sel.sum():
                        cal_lg[('raw_all', min(kk, 5))].append(Sr[sel]); cal_y[('raw_all', min(kk, 5))].append(h['occ'][sel])
        out = {}
        for tag, x in acc.items():
            out[tag] = dict(f1_pooled=f1(x), prec=x[0] / max(x[0] + x[1], 1), rec=x[0] / max(x[0] + x[2], 1), free_pooled=x[3] / x[4],
                            f1_mean=float(np.mean(mean_rows[tag])), free_mean=float(np.mean(mean_rows[tag + '_free'])))
        out['chosen'] = chosen
        out['table_all_houses'] = {int(l): round(float(v), 3) for l, v in zip(LV, table(hs, key)[0]) if abs(v) > 0 and l % 1 == 0}
        sl = {}
        for (sc, kk), v in sorted(cal_lg.items()):
            lg = np.concatenate(v); y = np.concatenate(cal_y[(sc, kk)]).astype(np.float64)
            a, b = slope(lg, y); sl[f'{sc}|k{kk}'] = dict(n=int(len(y)), occ_rate=float(y.mean()), intercept=a, slope=b)
        out['calib_slope_by_k'] = sl
        res[key] = out
    json.dump(res, sys.stdout, indent=1)


if __name__ == '__main__':
    main()

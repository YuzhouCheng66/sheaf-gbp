"""A02 probes P4 + P5 (2026-10-27, CPU only), 6 demo VAL houses x 8 agents, targets K0 (own map) and R8 (round-8 union).
P4 (verified decoding): the sender can run the receiver's decoder (current best 3S-X + calibrated rule = the exported prediction planes)
and send corrections.  Cost of the correction streams, adaptive arithmetic code with a context the receiver knows (prediction state at the
voxel, number of predicted-occupied 6-neighbours, 3 causal neighbours of the correction stream):
  e_all  = occupancy errors on target-known voxels ((pred == occ) != (target == occ)): after correcting, every F1 in the scope is 1;
  e_furn = the same restricted to the furniture region (neither slab nor dilated structure): furniture-region F1 becomes 1;
  lossless = the whole ternary target given the prediction (context: prediction state x 3 causal target neighbours).
P5 (sampling vs thresholding): a crude calibrated decoder from a block code (2x2x1 and 4x4x1 block statistics of the target): every
fine voxel of a block with any known voxel is predicted known with p_occ = occupied / known in the block.  Furniture-region and surface
F1 for the threshold rule p >= tau (tau .2-.6) and for posterior SAMPLING (Bernoulli(p) per voxel, mean of 2 draws), agents 0-3 of each house (CPU shared).
Output: probes/A02/p45_residual_sampling.json"""
import glob, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from rate_baselines_lib import load, targets, metrics, dilate6, DIMS, LG  # noqa: E402
torch.set_num_threads(8)


def kt_bytes(ctx, sym, k):
    """exact adaptive KT code length of symbols sym (0..k-1) under integer contexts ctx (sequential order irrelevant for KT totals)"""
    idx = ctx.reshape(-1) * k + sym.reshape(-1)
    cnt = torch.bincount(idx, minlength=int(ctx.max()) * k + k).reshape(-1, k).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    bits = ((LG(N + k / 2) - LG(k / 2)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2)
    return float(bits) / 8


def causal3(s):
    out = []
    for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        sh = torch.zeros_like(s); sh[d[0]:, d[1]:, d[2]:] = s[:s.shape[0] - d[0], :s.shape[1] - d[1], :s.shape[2] - d[2]]; out.append(sh)
    return out


def nb6_count(m):
    x = m.float()[None, None]
    k = torch.zeros(1, 1, 3, 3, 3); k[0, 0, 1, 1, 0] = k[0, 0, 1, 1, 2] = k[0, 0, 1, 0, 1] = k[0, 0, 1, 2, 1] = k[0, 0, 0, 1, 1] = k[0, 0, 2, 1, 1] = 1
    return F.conv3d(x, k, padding=1)[0, 0].round().long()


def correction_bytes(e, P):
    pstate = torch.from_numpy(P.astype(np.int64) + 1); pn = nb6_count(torch.from_numpy(P == 1))
    c = causal3(e)
    ctx = ((pstate * 7 + pn) * 2 + c[0]) * 4 + c[1] * 2 + c[2]
    return kt_bytes(ctx, e, 2)


def lossless_given_pred(T, P):
    t = torch.from_numpy(T.astype(np.int64) + 1); pstate = torch.from_numpy(P.astype(np.int64) + 1)
    c = causal3(t)
    ctx = ((pstate * 3 + c[0]) * 3 + c[1]) * 3 + c[2]
    return kt_bytes(ctx, t, 3)


def fast_metrics(P, T, region, slab, surf):
    """same definitions as rate_baselines_lib.metrics (target-known scope), numpy, with region / surface precomputed"""
    known = T >= 0; po = (P == 1) & known; to = (T == 1) & known; ps = po & ~((T == 1) & ~surf)
    def f1(a, b):
        tp = int((a & b).sum()); fp = int((a & ~b).sum()); fn = int((~a & b).sum())
        return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None
    return dict(surface_f1=f1(ps, surf), furn_f1=f1(po & region, to & region))


def block_probs(T, f):
    X, Y, Z = T.shape; bx, by, bz = f
    k = (T >= 0).reshape(X // bx, bx, Y // by, by, Z // bz, bz); o = (T == 1).reshape(X // bx, bx, Y // by, by, Z // bz, bz)
    nk = k.sum((1, 3, 5)); no = o.sum((1, 3, 5))
    p = np.where(nk > 0, no / np.maximum(nk, 1), 0.)
    kn = (nk > 0)
    up = lambda a: a.repeat(bx, 0).repeat(by, 1).repeat(bz, 2)
    return up(p), up(kn)


if __name__ == '__main__':
    data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'demo_web3d', 'data')
    rows = []; samp = []
    TAUS = [0.2, 0.3, 0.4, 0.5, 0.6]
    for path in sorted(glob.glob(os.path.join(data, 'house_*.json'))):
        m, own, pred, gt = load(path); n = own.shape[0]
        slab = torch.from_numpy(gt['slab'].reshape(-1)); obj = torch.from_numpy(gt['obj'].reshape(-1))
        struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(-1))
        region = (~slab & ~dilate6(struct[None], DIMS)[0]).reshape(DIMS).numpy()
        T8 = targets(own, m['reached'][m['hops'].index(8)])
        for k, Tk in ((0, own), (8, T8)):
            for a in range(n):
                T = Tk[a]; P = pred[k][a]
                e_all = ((P == 1) != (T == 1)) & (T >= 0); e_f = e_all & region
                r = dict(house=m['house'], hop=k, agent=a, n_err=int(e_all.sum()), n_err_furn=int(e_f.sum()),
                         n_occ_known=int((T == 1).sum()), n_occ_furn=int(((T == 1) & region).sum()),
                         bytes_e_all=correction_bytes(torch.from_numpy(e_all.astype(np.int64)), P),
                         bytes_e_furn=correction_bytes(torch.from_numpy(e_f.astype(np.int64)), P),
                         bytes_lossless_given_pred=lossless_given_pred(T, P))
                rows.append(r)
                # P5: sampling vs thresholding on block-statistics decoders (agents 0-3 only: shared CPU)
                surf = ((T == 1) & dilate6(torch.from_numpy((T == 0).reshape(1, -1)), DIMS)[0].reshape(DIMS).numpy())
                for f in ((2, 2, 1), (4, 4, 1)) if a < 4 else ():
                    p, kn = block_probs(T, f)
                    rec = dict(house=m['house'], hop=k, agent=a, factor='x'.join(map(str, f)))
                    for tau in TAUS:
                        Pt = np.where(kn, (p >= tau).astype(np.int8), np.int8(-1))
                        mt = fast_metrics(Pt, T, region, None, surf)
                        rec[f'thr{tau}_surf'] = mt['surface_f1']; rec[f'thr{tau}_furn'] = mt['furn_f1']
                    g = np.random.default_rng(7 * a + k)
                    ss, sf = [], []
                    for _ in range(2):
                        Ps = np.where(kn, (g.random(p.shape) < p).astype(np.int8), np.int8(-1))
                        ms = fast_metrics(Ps, T, region, None, surf)
                        ss.append(ms['surface_f1']); sf.append(ms['furn_f1'] if ms['furn_f1'] is not None else np.nan)
                    rec['sample_surf'] = float(np.mean(ss)); rec['sample_furn'] = float(np.nanmean(sf)) if not all(np.isnan(sf)) else None
                    samp.append(rec)
                print(m['house'], k, a, {kk: (round(v, 1) if isinstance(v, float) else v) for kk, v in r.items() if kk.startswith(('bytes', 'n_'))}, flush=True)
    S = {}
    for k in (0, 8):
        rr = [r for r in rows if r['hop'] == k]
        S[f'P4_hop{k}'] = {kk: dict(median=float(np.median([r[kk] for r in rr])), mean=float(np.mean([r[kk] for r in rr])))
                          for kk in rr[0] if kk not in ('house', 'hop', 'agent')}
        for f in ('2x2x1', '4x4x1'):
            ss = [s for s in samp if s['hop'] == k and s['factor'] == f]
            S[f'P5_hop{k}_{f}'] = {kk: float(np.mean([s[kk] for s in ss if s[kk] is not None])) for kk in ss[0] if kk not in ('house', 'hop', 'agent', 'factor')}
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, rows=rows, sampling=samp), open(os.path.join(HERE, 'p45_residual_sampling.json'), 'w'), indent=1)

"""probe_C_analyze (analyst C, 2026-10-26): offline scoring of probe_C_hist.py outputs. Decision rules are CHOSEN ON TRAIN scenes and
REPORTED ON VAL (the VAL-oracle optimum is printed only as an upper bound). Per-agent F1, mean over agents, mean over scenes (= metrics3d / agg).
python probe_C_analyze.py --dir h-s342101 [--dir h-s342202] --out summary.json
"""
import argparse, glob, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--dir', action='append', required=True); ap.add_argument('--out', required=True)
args = ap.parse_args()
NB, LO, BW = 64, -8., .25
EDGES = LO + BW * np.arange(NB + 1)                     # threshold index a -> 'logit >= EDGES[a]' (a = 64: never)


def suffix2(A):
    S = np.zeros(A.shape[:-2] + (NB + 1, NB + 1), np.float64)
    S[..., :NB, :NB] = A[..., ::-1, ::-1].cumsum(-1).cumsum(-2)[..., ::-1, ::-1]
    return S


def cats(t, locs=(0, 1, 2)):
    return [ti * 3 + l for ti in t for l in locs]


def f1grid(Hh, pos, neg):
    """Hh [n, 12, NB, NB] -> per agent F1 over the (a, b) threshold grid [n, 65, 65] (nan = None)."""
    P = Hh[:, pos].sum(1); N = Hh[:, neg].sum(1)
    TP = suffix2(P); FP = suffix2(N); FN = P.sum((-1, -2))[:, None, None] - TP
    den = 2 * TP + FP + FN
    known = Hh[:, cats((1, 2, 3))].sum((1, 2, 3))
    with np.errstate(invalid='ignore', divide='ignore'):
        f = np.where(den > 0, 2 * TP / np.maximum(den, 1e-12), np.nan)
    f[known == 0] = np.nan
    return f


def prodrule(Hh, pos, neg, taus):
    c = LO + BW * (np.arange(NB) + .5); sig = 1 / (1 + np.exp(-c)); prob = sig[:, None] * sig[None, :]
    P = Hh[:, pos].sum(1); N = Hh[:, neg].sum(1); U = Hh[:, cats((0,))].sum(1); out = []; fpr = []
    kpos = (np.arange(NB) >= NB // 2)[:, None]
    for t in taus:
        m = prob >= t
        TP = (P * m).sum((-1, -2)); FP = (N * m).sum((-1, -2)); FN = P.sum((-1, -2)) - TP
        den = 2 * TP + FP + FN
        out.append(np.where(den > 0, 2 * TP / np.maximum(den, 1e-12), np.nan))
        fpr.append((U * (m | kpos)).sum((-1, -2)) / np.maximum(U.sum((-1, -2)), 1))
    return np.stack(out, -1), np.stack(fpr, -1)                        # [n, ntau] x 2


METRICS = dict(surface=(cats((2,)), cats((1,))), var=(cats((2, 3), (1, 2)), cats((1,), (1, 2))), furn=(cats((2, 3), (2,)), cats((1,), (2,))), occ=(cats((2, 3)), cats((1,))))
TAUS = np.round(np.arange(.10, .91, .025), 3)


def load(d, split):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, f'{split}-*.npz'))):
        z = np.load(f); rows.append((z['H'].astype(np.float64), json.loads(str(z['res']))))
    return rows


def grids(rows):
    """mean over scenes of the agent-mean F1 grids, per hop and metric; FPR per known threshold; product rule."""
    out = {}
    for hi, k in enumerate((0, 8)):
        for m, (pos, neg) in METRICS.items():
            out[(k, m)] = np.nanmean(np.stack([np.nanmean(f1grid(H[hi], pos, neg), 0) for H, _ in rows]), 0)
            out[(k, m, 'prod')] = np.nanmean(np.stack([np.nanmean(prodrule(H[hi], pos, neg, TAUS)[0], 0) for H, _ in rows]), 0)
        out[(k, 'fpr_prod')] = np.mean(np.stack([prodrule(H[hi], *METRICS['surface'], TAUS)[1].mean(0) for H, _ in rows]), 0)
        fpr = []
        for H, _ in rows:
            U = H[hi][:, cats((0,))].sum((1, 3))                       # [n, NB] over known-logit bins
            s = np.concatenate([U[:, ::-1].cumsum(-1)[:, ::-1], np.zeros((U.shape[0], 1))], -1) / np.maximum(U.sum(-1, keepdims=True), 1)
            fpr.append(s.mean(0))
        out[(k, 'fpr')] = np.mean(fpr, 0)
        fs = []
        for H, _ in rows:
            U2 = H[hi][:, cats((0,))].sum(1); S = suffix2(U2); tot = np.maximum(U2.sum((-1, -2)), 1)[:, None, None]
            base = S[:, NB // 2, 0][:, None, None]
            extra = np.where((np.arange(NB + 1) < NB // 2)[None, :, None], S - S[:, NB // 2, :][:, None, :], 0.)
            fs.append(((base + extra) / tot).mean(0))
        out[(k, 'fpr_split')] = np.mean(fs, 0)
    return out


def at(g, a, b):
    return float(g[a, b])


def main():
    summary = {}
    D0 = int(np.argmin(np.abs(EDGES - 0)))
    for d in args.dir:
        tr, va = load(d, 'train'), load(d, 'val')
        gt, gv = grids(tr), grids(va)
        s = dict(n_train=len(tr), n_val=len(va))
        # sanity: default rule from histograms vs metrics3d
        s['check_val_surface_K0_hist_vs_metrics3d'] = [at(gv[(0, 'surface')], D0, D0), float(np.mean([r['metrics_K0']['surface_f1'] for _, r in va]))]
        s['check_val_surface_K8_hist_vs_metrics3d'] = [at(gv[(8, 'surface')], D0, D0), float(np.mean([r['metrics_K8']['surface_f1'] for _, r in va]))]
        def report(a, b, g):
            return {f'K{k}_{m}': round(at(g[(k, m)], a, b), 4) for k in (0, 8) for m in METRICS} | {f'K{k}_fpr': round(float(g[(k, 'fpr')][a]), 4) for k in (0, 8)}
        s['default'] = dict(train=report(D0, D0, gt), val=report(D0, D0, gv))
        rules = {}
        for name, obj in (('R8', gt[(8, 'surface')]), ('K0', gt[(0, 'surface')]), ('mean', (gt[(0, 'surface')] + gt[(8, 'surface')]) / 2)):
            a, b = np.unravel_index(np.nanargmax(obj), obj.shape)
            rules[name] = dict(known_thr=float(EDGES[a]), occ_thr=float(EDGES[b]), train=report(a, b, gt), val=report(a, b, gv))
        # occupancy threshold only (known threshold fixed at 0)
        b = int(np.nanargmax((gt[(0, 'surface')][D0] + gt[(8, 'surface')][D0]) / 2))
        rules['occ_only_mean'] = dict(known_thr=0., occ_thr=float(EDGES[b]), train=report(D0, b, gt), val=report(D0, b, gv))
        a = int(np.nanargmax((gt[(0, 'surface')][:, D0] + gt[(8, 'surface')][:, D0]) / 2))
        rules['known_only_mean'] = dict(known_thr=float(EDGES[a]), occ_thr=0., train=report(a, D0, gt), val=report(a, D0, gv))
        rules['no_veto'] = dict(known_thr=float(EDGES[0]), occ_thr=0., val=report(0, D0, gv))
        # VAL oracle (upper bound only, not a decision)
        obj = (gv[(0, 'surface')] + gv[(8, 'surface')]) / 2; a, b = np.unravel_index(np.nanargmax(obj), obj.shape)
        rules['VAL_ORACLE_mean'] = dict(known_thr=float(EDGES[a]), occ_thr=float(EDGES[b]), val=report(a, b, gv))
        # product rule sigma(k) sigma(o) >= tau, chosen on TRAIN (bin-centre approximation)
        t = int(np.nanargmax((gt[(0, 'surface', 'prod')] + gt[(8, 'surface', 'prod')]) / 2))
        rules['product_mean'] = dict(tau=float(TAUS[t]), val={f'K{k}_{m}': round(float(gv[(k, m, 'prod')][t]), 4) for k in (0, 8) for m in METRICS},
                                     val_at_0p5={f'K{k}_surface': round(float(gv[(k, 'surface', 'prod')][int(np.argmin(np.abs(TAUS - .5)))]), 4) for k in (0, 8)})
        # FPR-constrained (guard: TRAIN FPR <= default + 0.003 at K0 and R8)
        f0, f8 = gt[(0, 'fpr')][D0] + .003, gt[(8, 'fpr')][D0] + .003
        obj = (gt[(0, 'surface')] + gt[(8, 'surface')]) / 2
        ok = (gt[(0, 'fpr')] <= f0) & (gt[(8, 'fpr')] <= f8)
        o2 = np.where(ok[:, None], obj, -1); a, b = np.unravel_index(np.nanargmax(o2), o2.shape)
        rules['single_known_thr_fpr_guard'] = dict(known_thr=float(EDGES[a]), occ_thr=float(EDGES[b]), train=report(a, b, gt), val=report(a, b, gv))
        ok2 = (gt[(0, 'fpr_split')] <= f0) & (gt[(8, 'fpr_split')] <= f8)
        o3 = np.where(ok2, obj, -1); a, b = np.unravel_index(np.nanargmax(o3), o3.shape)
        rules['split_rule_fpr_guard'] = dict(occ_needs_known_thr=float(EDGES[a]), occ_thr=float(EDGES[b]), train=report(a, b, gt), val=report(a, b, gv),
                                             val_fpr_split=dict(K0=round(float(gv[(0, 'fpr_split')][a, b]), 4), K8=round(float(gv[(8, 'fpr_split')][a, b]), 4)),
                                             train_fpr_split=dict(K0=round(float(gt[(0, 'fpr_split')][a, b]), 4), K8=round(float(gt[(8, 'fpr_split')][a, b]), 4)))
        okp = (gt[(0, 'fpr_prod')] <= f0) & (gt[(8, 'fpr_prod')] <= f8)
        op = np.where(okp, (gt[(0, 'surface', 'prod')] + gt[(8, 'surface', 'prod')]) / 2, -1); t = int(np.nanargmax(op))
        rules['product_fpr_guard'] = dict(tau=float(TAUS[t]), val={f'K{k}_{m}': round(float(gv[(k, m, 'prod')][t]), 4) for k in (0, 8) for m in METRICS} | {f'K{k}_fpr': round(float(gv[(k, 'fpr_prod')][t]), 4) for k in (0, 8)})
        s['rules'] = rules
        b15 = int(np.argmin(np.abs(EDGES - 1.5)))                       # split-rule sweep of the occupied-output known threshold at o >= +1.5
        s['split_sweep'] = {f'{EDGES[a]:+.2f}': {f'{sp}_K{k}_{m}': round(float(g[(k, m)][a, b15]), 4) for sp, g in (('train', gt), ('val', gv)) for k in (0, 8) for m in ('surface', 'var')}
                            | {f'{sp}_K{k}_fpr': round(float(g[(k, 'fpr_split')][a, b15]), 4) for sp, g in (('train', gt), ('val', gv)) for k in (0, 8)} for a in range(D0 - 6, D0 + 1)}
        # 1-D sweep of the occ threshold at known 0 (VAL and TRAIN, surface), for the memo table
        s['occ_sweep'] = {f'{EDGES[b]:+.2f}': dict(train_K0=round(at(gt[(0, 'surface')], D0, b), 4), train_K8=round(at(gt[(8, 'surface')], D0, b), 4),
                                                   val_K0=round(at(gv[(0, 'surface')], D0, b), 4), val_K8=round(at(gv[(8, 'surface')], D0, b), 4)) for b in range(D0 - 8, D0 + 13, 2)}
        s['known_sweep'] = {f'{EDGES[a]:+.2f}': dict(val_K0=round(at(gv[(0, 'surface')], a, D0), 4), val_K8=round(at(gv[(8, 'surface')], a, D0), 4), val_K0_fpr=round(float(gv[(0, 'fpr')][a]), 4), val_K8_fpr=round(float(gv[(8, 'fpr')][a]), 4),
                                                     val_K0_furn=round(at(gv[(0, 'furn')], a, D0), 4)) for a in range(D0 - 16, D0 + 5, 2)}
        # veto vs occupancy-head decomposition (VAL, default rule): target SURFACE voxels and all target-occupied voxels by location
        dec = {}
        for hi, k in enumerate((0, 8)):
            Hs = sum(H[hi].sum(0) for H, _ in va)                     # [12, NB, NB]
            for lname, l in (('slab', 0), ('structzone', 1), ('region', 2)):
                for tname, tc in (('surface', (2,)), ('occupied', (2, 3))):
                    A = sum(Hs[ti * 3 + l] for ti in tc); tot = A.sum()
                    dec[f'K{k}_{tname}_{lname}'] = dict(n=int(tot), recall=round(A[D0:, D0:].sum() / tot, 4), vetoed=round(A[:D0].sum() / tot, 4),
                                                       vetoed_occ_says_occ=round(A[:D0, D0:].sum() / tot, 4), occ_miss_when_known=round(A[D0:, :D0].sum() / tot, 4),
                                                       occ_head_alone_recall=round(A[:, D0:].sum() / tot, 4))
                Fr = Hs[1 * 3 + l]; dec[f'K{k}_fp_{lname}'] = int(Fr[D0:, D0:].sum())
                dec[f'K{k}_free_{lname}_known_but_occ_head_off'] = int(Fr[D0:, :D0].sum())
        s['veto_decomposition_val'] = dec
        # fusion decomposition at R8 (VAL): classes S / O / M
        fus = {}
        for key in ('fusion_K0', 'fusion_K8', 'copywinner_K8', 'fusion_covkey_K8'):
            if not any(key in r for _, r in va):
                continue
            C = sum(np.array(r[key]).reshape(-1, 3, 4, 3).sum(0) for _, r in va if key in r)   # [fcls, tcls, pred]
            tot_surf = C[:, 2].sum()
            fus[key] = {cl: dict(share_surface=round(C[c, 2].sum() / tot_surf, 4), surface_recall=round(C[c, 2, 2] / max(C[c, 2].sum(), 1), 4),
                                 surface_vetoed=round(C[c, 2, 0] / max(C[c, 2].sum(), 1), 4), fp=int(C[c, 1, 2]), share_fp=round(C[c, 1, 2] / max(C[:, 1, 2].sum(), 1), 4),
                                 precision=round(C[c, 2, 2] / max(C[c, 2, 2] + C[c, 1, 2], 1), 4), share_known=round(C[c, 1:].sum() / max(C[:, 1:].sum(), 1), 4))
                        for c, cl in enumerate(('S', 'O', 'M'))}
            # per-agent surface F1 restricted to the voxels the winner observed (S + O), mean over agents and scenes
            vals = []
            for _, r in va:
                if key not in r:
                    continue
                A = np.array(r[key]).reshape(-1, 3, 4, 3)[:, :2].sum(1)    # [n, tcls, pred]
                tp = A[:, 2, 2]; fp = A[:, 1, 2]; fn = A[:, 2, 0] + A[:, 2, 1]
                den = 2 * tp + fp + fn
                vals.append(np.nanmean(np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan)))
            fus[key]['surface_f1_observed_by_winner_only'] = round(float(np.mean(vals)), 4)
        fus['metrics_val'] = {k: round(float(np.mean([r[k]['surface_f1'] for _, r in va if k in r])), 4) for k in ('metrics_K0', 'metrics_K8', 'metrics_copywinner_K8', 'metrics_covkey_K8')}
        fus['metrics_val_var'] = {k: round(float(np.mean([r[k]['var_occ_f1'] for _, r in va if k in r])), 4) for k in ('metrics_K0', 'metrics_K8', 'metrics_copywinner_K8', 'metrics_covkey_K8')}
        fus['metrics_val_tol1'] = {k: round(float(np.mean([r[k]['surface_f1_tol1'] for _, r in va if k in r])), 4) for k in ('metrics_K0', 'metrics_K8', 'metrics_copywinner_K8', 'metrics_covkey_K8')}
        fus['metrics_train'] = {k: round(float(np.mean([r[k]['surface_f1'] for _, r in tr if k in r])), 4) for k in ('metrics_K0', 'metrics_K8', 'metrics_covkey_K8')}
        s['fusion_val'] = fus
        summary[os.path.basename(d.rstrip('/'))] = s
    json.dump(summary, open(args.out, 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

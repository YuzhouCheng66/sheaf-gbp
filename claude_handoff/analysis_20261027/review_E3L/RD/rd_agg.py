"""aggregate rd_probe outputs (rd_out/q1) -> rd_out/q1_summary.json + printed tables"""
import os, sys, json, glob
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, 'rd_out', 'q1')
SUMMARY = os.path.abspath(os.path.join(HERE, '..', '..', 'g4', 'e3l', 'summary.json'))
LO, STEP, NBIN = -40.0, 0.02, 4000

recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(Q, '*.json')))]
jobs = [r['job'] for r in recs]
print(len(recs), 'jobs')
arms = list(recs[0]['arms'].keys())
KEYS = ('furn_f1', 'furn_f1_tol1', 'surface_f1', 'free_recall', 'unknown_known_fpr', 'obj_recall', 'known_frac')


def mean_key(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return float(np.mean(v)) if v else None


out = dict(jobs=jobs, arms={})
for a in arms:
    rows = [x for r in recs for x in r['arms'][a]['rows']]
    d = {k: mean_key(rows, k) for k in KEYS}
    by = [r['arms'][a].get('bytes') for r in recs if r['arms'][a].get('bytes') is not None]
    d['bytes_median'] = float(np.median(by)) if by else None
    # pooled strata
    for sk in ('k_furn', 'h_furn'):
        acc = None
        for x in rows:
            s = (x.get('strata') or {}).get(sk)
            if s is None:
                continue
            s = np.asarray(s, np.int64)
            acc = s if acc is None else acc + s
        if acc is not None:
            d[sk] = [float(2 * t / max(2 * t + f + n, 1)) if (t + f + n) else None for t, f, n in acc]
            d[sk + '_ntarget'] = [int(t + n) for t, f, n in acc]
    # error decomposition (pooled over rows)
    er = [x['err'] for x in rows if 'err' in x]
    if er:
        S = lambda k: float(sum(e.get(k, 0) for e in er))
        fn = S('fn'); tp = S('tp'); fp = S('fp')
        d['err'] = dict(pooled_f1=2 * tp / max(2 * tp + fp + fn, 1), recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                        fn_near_frac=S('fn_near') / max(fn, 1), fn_far_frac=S('fn_far') / max(fn, 1), fp_near_frac=S('fp_near') / max(fp, 1),
                        obj_missed_frac=S('obj_missed') / max(S('n_obj'), 1), obj_missed_vox_frac=S('obj_missed_vox') / max(S('obj_vox'), 1),
                        fn_in_missed_obj_frac=S('fn_in_missed_obj') / max(fn, 1), far_in_cc_ge32_frac=S('far_cc_vox_ge32') / max(S('fn_far'), 1),
                        f1_if_near_fixed=2 * (tp + S('fn_near')) / max(2 * (tp + S('fn_near')) + fp + S('fn_far'), 1),
                        f1_if_far_fixed=2 * (tp + S('fn_far')) / max(2 * (tp + S('fn_far')) + fp + S('fn_near'), 1))
        if any('tr_nobelief' in e for e in er):
            tr = tp + fn
            d['err'].update(tr_nobelief_frac=S('tr_nobelief') / max(tr, 1), fn_nobelief_frac=S('fn_nobelief') / max(fn, 1),
                            fn_far_nobelief_frac=S('fn_far_nobelief') / max(S('fn_far'), 1), fp_nobelief_frac=S('fp_nobelief') / max(fp, 1),
                            recall_in_nobelief=1 - S('fn_nobelief') / max(S('tr_nobelief'), 1),
                            recall_in_belief=1 - (fn - S('fn_nobelief')) / max(tr - S('tr_nobelief'), 1))
    out['arms'][a] = d

# pooled threshold curves from histograms (R8 arms with Hist)
curves = {}
for a in arms:
    A = B = C = None; tot = np.zeros(2)
    for j in jobs:
        z = np.load(os.path.join(Q, j + '.npz'))
        if f'{a}_A' not in z.files:
            break
        A = z[f'{a}_A'] if A is None else A + z[f'{a}_A']; B = z[f'{a}_B'] if B is None else B + z[f'{a}_B']
        C = z[f'{a}_C'] if C is None else C + z[f'{a}_C']; tot += z[f'{a}_tot']
    else:
        sa = np.cumsum(A[::-1])[::-1]; sb = np.cumsum(B[::-1])[::-1]; pc = np.r_[0, np.cumsum(C)[:-1]]
        f1 = 2 * sa / np.maximum(2 * sa + sb + (tot[0] - sa), 1); fr = pc / max(tot[1], 1)
        curves[a] = (f1, fr)
for a, (f1, fr) in curves.items():
    sd = a.rsplit('_', 1)[1]
    thr = json.load(open(os.path.join(HERE, '..', '..', 'review_E3R', 'inputs', f'e3l_s{sd}', 'threshold.json')))
    jo = thr['R8']['j_occ']
    fbase = curves.get(f'formal_{sd}', (None, None))[1]
    free_ref = fbase[jo] if fbase is not None else None
    ok = fr >= free_ref - 1e-12 if free_ref is not None else np.ones_like(fr, bool)
    jm = int(np.flatnonzero(ok)[np.argmax(f1[ok])]) if ok.any() else None
    jb = int(np.argmax(f1))
    out['arms'][a]['pooled'] = dict(f1_fixed=float(f1[jo]), free_fixed=float(fr[jo]), f1_oracle=float(f1[jb]), t_oracle=LO + jb * STEP,
                                    free_oracle=float(fr[jb]), f1_at_formal_free=(float(f1[jm]) if jm is not None else None),
                                    t_at_formal_free=(LO + jm * STEP if jm is not None else None))

# consistency with the formal summary (per-house R8 furniture of the formal arm vs summary)
S = json.load(open(SUMMARY))
v = S['panels']['val']['seeds']['1']
chk = {}
for sd in ('342101', '342202'):
    tier = v['tiers']['cam_nf'][f'E3L_{sd}']
    chk[sd] = dict(summary_R8_furn_all24=tier['R8']['furn_f1'], probe_formal=out['arms'][f'formal_{sd}']['furn_f1'],
                   summary_bytes_median=tier['bytes_median'], probe_bytes_median=out['arms'][f'formal_{sd}']['bytes_median'])
out['consistency'] = chk
# formal stats
out['encoder_stats'] = {sd: [r['arms'][f'formal_{sd}']['stats'] for r in recs] for sd in ('342101', '342202')}
st = {}
for sd in ('342101', '342202'):
    L = out['encoder_stats'][sd]
    st[sd] = dict(touched=float(np.median([x['touched'] for x in L])), support=float(np.median([x['formal_support'] for x in L])),
                  support_frac=float(np.sum([x['formal_support'] for x in L]) / np.sum([x['touched'] for x in L])),
                  frac_below_floor=float(np.mean([x['frac_touched_below_floor'] for x in L])),
                  frac_lam_lt_half=float(np.mean([x['frac_touched_lam_lt_half'] for x in L])),
                  a_raw_median_q=np.median([x['a_raw_touched_q'] for x in L], 0).tolist(),
                  lam_supported_q=np.median([x['lam_supported_q'] for x in L if x['lam_supported_q']], 0).tolist(),
                  z_absmax_q=np.median([x['z_absmax_q'] for x in L], 0).tolist())
out['encoder_stats_brief'] = st
with open(os.path.join(HERE, 'rd_out', 'q1_summary.json'), 'w') as fh:
    json.dump(out, fh, indent=1)

print(json.dumps(chk, indent=1))
print(json.dumps(st, indent=1))
fmt = lambda x: '   -  ' if x is None else f'{x:.4f}'
print(f"{'arm':22s} {'furn':>7s} {'tol1':>7s} {'surf':>7s} {'free':>7s} {'fpr':>7s} {'known':>7s} {'bytes':>9s} | pooled fixed / oracle / @formal-free")
for a in arms:
    d = out['arms'][a]; p = d.get('pooled', {})
    print(f"{a:22s} {fmt(d['furn_f1'])} {fmt(d['furn_f1_tol1'])} {fmt(d['surface_f1'])} {fmt(d['free_recall'])} {fmt(d['unknown_known_fpr'])} {fmt(d['known_frac'])} "
          f"{(d['bytes_median'] or 0):9.0f} | {fmt(p.get('f1_fixed'))} / {fmt(p.get('f1_oracle'))} (t {p.get('t_oracle', 0):.2f}) / {fmt(p.get('f1_at_formal_free'))}")
print('\nstrata (pooled F1): k_furn = k 0,1,2,3,4,>=5 ; h_furn = 0-0.3, 0.3-0.9, >0.9 m')
for a in arms:
    d = out['arms'][a]
    if 'k_furn' in d:
        print(f"{a:22s} k {[None if x is None else round(x, 3) for x in d['k_furn']]}  h {[None if x is None else round(x, 3) for x in d.get('h_furn', [])]}")
print('\nerror decomposition')
for a in arms:
    if 'err' in out['arms'][a]:
        print(a, json.dumps({k: round(v, 4) for k, v in out['arms'][a]['err'].items()}))

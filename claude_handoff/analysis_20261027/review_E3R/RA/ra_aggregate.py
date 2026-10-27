"""R-A: aggregate out_recompute / out_regnet / out_tiers and compare with the frozen summaries. Writes RA/aggregate.json."""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
S = json.load(open(os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'g4', 'e3r', 'summary.json')))
out = {}

def exp_mean(rows):
    v = [r[0] for r in rows for _ in range(r[1]) if r[0] is not None]
    return float(np.mean(v)) if v else None

# ---- recompute
R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, 'out_recompute', 'val_*.json')))]
rc = {}
maxdiff = 0.0
for seed in ('1', '2'):
    X = [r for r in R if r['house'].endswith('_s' + seed)]
    for tn in ('cam_nf', 'cam'):
        for sd, half in (('342101', 'A'), ('342202', 'B')):
            net = [row for r in X for row in r['tiers'][tn][sd]['R8_net']]
            lin = [row for r in X for row in r['tiers'][tn][sd]['R8_lin']]
            k0n = [v for r in X for v in r['tiers'][tn][sd]['K0_net_sep']]; k0l = [v for r in X for v in r['tiers'][tn][sd]['K0_lin_sep']]
            for r in X:
                for xc in r['tiers'][tn][sd]['xcheck']:
                    maxdiff = max(maxdiff, abs(xc['own'] - xc['G']), abs(xc['own_free'] - xc['G_free']))
            ref = S['panels']['val']['seeds'][seed]['tiers'][tn]
            d = dict(n_houses=len(X), net=exp_mean(net), lin=exp_mean(lin), k0_net_sep=float(np.mean([v for v in k0n if v is not None])),
                     k0_lin_sep=float(np.mean([v for v in k0l if v is not None])),
                     free_net=exp_mean([[r_[2], r_[1]] for r_ in net]),
                     ref_net=ref[f'NET_{sd}']['R8']['furn_f1'], ref_lin=ref[f'LIN_{sd}']['R8']['furn_f1'], ref_b6=ref[f'B6_{half}']['R8']['furn_f1'],
                     ref_k0_net_sep=ref[f'NET_{sd}'].get('K0_sep', {}).get('furn_f1'), ref_k0_lin_sep=ref[f'LIN_{sd}'].get('K0_sep', {}).get('furn_f1'),
                     ref_free_net=ref[f'NET_{sd}']['R8']['free_recall'])
            d['delta_net_minus_lin'] = d['net'] - d['lin']
            # per-house component-expanded readings (for the report table)
            d['per_house'] = {r['house']: [exp_mean(r['tiers'][tn][sd]['R8_net']), exp_mean(r['tiers'][tn][sd]['R8_lin'])] for r in X}
            rc[f'{tn}|{sd}|s{seed}'] = d
out['recompute'] = rc; out['recompute_xcheck_maxdiff_vs_Gscore'] = maxdiff; out['recompute_houses'] = len(R)

# ---- registration x NET
Q = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, 'out_regnet2', 'val_*.json')))]
Q = [q for q in Q if not q.get('failed')]
arms = ('noreg', 'reg', 'ceiling_islands', 'oracle_trans', 'half', 'truepose')
rg = dict(n_jobs=len(Q), jobs=[q['job'] for q in Q], mean_islands=float(np.mean([q['n_islands'] for q in Q])) if Q else None)
for seed in ('1', '2', 'all'):
    X = [q for q in Q if seed == 'all' or q['job'].endswith('_s' + seed)]
    if not X:
        continue
    tab = {}
    for arm in arms:
        tab[arm] = {tag: exp_mean([row for q in X for row in q['arms'][arm][tag]]) for tag in ('B4', 'B6', 'NET')}
        tab[arm]['NET_free'] = exp_mean([[row[2], row[1]] for q in X for row in q['arms'][arm]['NET']])
    for tag in ('B4', 'B6', 'NET'):
        lo, hi = tab['noreg'][tag], tab['truepose'][tag]
        for arm in arms:
            tab[arm][tag + '_recovery'] = (tab[arm][tag] - lo) / (hi - lo) if hi != lo else None
    # E4 aligned B4 rows (matcher gauge) next to ours (oracle gauge), for reference
    e4 = {}
    for arm in ('noreg', 'reg', 'ceiling_islands', 'truepose'):
        e4[arm] = exp_mean([[v, 1] for q in X for v in q['e4_aligned_b4'].get(arm, [])])
    tab['e4_aligned_b4_mean'] = e4
    tab['resid'] = {arm: dict(yaw_rms_deg=float(np.mean([q['resid'][arm]['yaw_rms_deg'] for q in X])),
                              disp_rms_cm=float(np.mean([q['resid'][arm]['disp_rms_cm'] for q in X]))) for arm in arms if arm != 'truepose'}
    tab['e4_asis_b4_maxdiff'] = None
    tab['n_jobs'] = len(X)
    rg['s' + seed] = tab
out['regnet'] = rg
# asis NET (E3-R panel row) of the same jobs, for the gauge effect
if Q:
    asis = []
    for q in Q:
        s = q['job'][-1]; v = rc.get(f'cam|342101|s{s}', {}).get('per_house', {}).get(q['job'])
        if v:
            asis.append(v[0])
    rg['asis_net_same_jobs_mean_of_houses'] = float(np.mean(asis)) if asis else None
    rg['gauge_net_same_jobs_mean_of_houses'] = float(np.mean([exp_mean(q['arms']['noreg']['NET']) for q in Q]))

for d, key, arms_ in (('out_regal', 'regal', ('noreg', 'reg', 'ceiling_islands')), ('out_regnf', 'regnf', ('nf_noreg', 'nf_reg_asis', 'nf_reg_gauge'))):
    X = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, d, 'val_*.json')))]
    X = [x for x in X if not x.get('failed')]
    t = dict(n_jobs=len(X), jobs=[x['job'] for x in X])
    if X:
        for arm in arms_:
            t[arm] = {tag: exp_mean([row for x in X for row in x['arms'][arm][tag]]) for tag in ('B4', 'B6', 'NET')}
        if key == 'regal':
            for tag in ('B4', 'B6', 'NET'):
                t['reg_minus_noreg_' + tag] = t['reg'][tag] - t['noreg'][tag]
                t['ci_minus_noreg_' + tag] = t['ceiling_islands'][tag] - t['noreg'][tag]
            dif = []
            for x in X:
                for arm in arms_:
                    dif += [abs(r[0] - e) for r, e in zip(x['arms'][arm]['B4'], x['e4_aligned_b4'][arm]) if r[0] is not None and e is not None]
            t['e4_aligned_b4_maxdiff'] = max(dif) if dif else None
            t['per_job_net'] = {x['job']: {arm: exp_mean(x['arms'][arm]['NET']) for arm in arms_} for x in X}
        else:
            t['k0_check_all'] = all(x.get('k0_check') for x in X)
            t['per_job_net'] = {x['job']: {arm: exp_mean(x['arms'][arm]['NET']) for arm in arms_} for x in X}
    out[key] = t
    print(key, json.dumps({k: v for k, v in t.items() if k not in ('per_job_net',)}))

# ---- lidar / harsh
Z = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, 'out_tiers', 'val_*.json')))]
Z = [z for z in Z if not z.get('failed')]
tt = dict(n_jobs=len(Z), jobs=[z['job'] for z in Z])
for tn in ('lidar', 'harsh'):
    if not Z:
        break
    d = {}
    for fr in ('pub', 'nf'):
        d['R8_' + fr] = {k: exp_mean([row for z in Z for row in z['tiers'][tn]['R8_' + fr][k]]) for k in ('B4', 'B6own', 'NETcamnf', 'NETown')}
    d['K0'] = {k: float(np.mean([v for z in Z for v in z['tiers'][tn]['K0'][k] if v is not None])) for k in ('B4', 'B6own', 'NETcamnf')}
    tt[tn] = d
out['tiers'] = tt

# ---- per-kind and flips
K = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, 'out_kinds', 'val_*.json')))]
K = [k for k in K if not k.get('failed')]
kk = dict(n_jobs=len(K))
acc = {}; fl = {}
for k in K:
    for cls, d in k['kinds'].items():
        for tag, v in d.items():
            a = acc.setdefault(cls, {}).setdefault(tag, [0, 0, 0]); a[0] += v[0]; a[1] += v[1]; a[2] += v[2]
    for f, v in k['flips'].items():
        b = fl.setdefault(f, [0, 0]); b[0] += v[0]; b[1] += v[1]
def f1(v):
    return 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1)
kk['per_class'] = {cls: dict(NET=f1(d['NET']), B6=f1(d['B6']), n_target=d['NET'][0] + d['NET'][2], delta=f1(d['NET']) - f1(d['B6'])) for cls, d in acc.items() if d['NET'][0] + d['NET'][2] > 0}
kk['flips'] = {f: dict(correct=v[0], wrong=v[1], precision=v[0] / max(v[0] + v[1], 1)) for f, v in fl.items()}
out['kinds'] = kk
print(json.dumps(kk, indent=0)[:2500])

# ---- per-island gauge of E4's edges (oracle-gauge row, same jobs as regnet2)
I_ = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, 'out_regisl', 'val_*.json')))]
I_ = [x for x in I_ if not x.get('failed')]
if I_:
    jobs = [x['job'] for x in I_]
    Qd = {q['job']: q for q in Q}
    both = [jb for jb in jobs if jb in Qd]
    t = dict(n_jobs=len(both), jobs=both)
    for tag in ('B4', 'B6', 'NET'):
        t['reg_island_' + tag] = exp_mean([row for x in I_ if x['job'] in Qd for row in x['arms']['reg_island'][tag]])
        for arm in ('noreg', 'reg', 'ceiling_islands', 'truepose'):
            t[arm + '_' + tag] = exp_mean([row for jb in both for row in Qd[jb]['arms'][arm][tag]])
        lo, hi = t['noreg_' + tag], t['truepose_' + tag]
        t['reg_island_recovery_' + tag] = (t['reg_island_' + tag] - lo) / (hi - lo)
    t['resid'] = dict(yaw_rms_deg=float(np.mean([x['resid']['reg_island']['yaw_rms_deg'] for x in I_ if x['job'] in Qd])),
                      disp_rms_cm=float(np.mean([x['resid']['reg_island']['disp_rms_cm'] for x in I_ if x['job'] in Qd])))
    t['per_job_net'] = {x['job']: dict(reg_island=exp_mean(x['arms']['reg_island']['NET']), noreg=exp_mean(Qd[x['job']]['arms']['noreg']['NET']),
                                       reg=exp_mean(Qd[x['job']]['arms']['reg']['NET']), ci=exp_mean(Qd[x['job']]['arms']['ceiling_islands']['NET']))
                        for x in I_ if x['job'] in Qd}
    t['acc_islands_mean'] = float(np.mean([len(x['acc_islands']) for x in I_]))
    t['acc_islands_equal_obs_islands'] = sum(1 for x in I_ if sorted(map(sorted, x['acc_islands'])) == sorted(map(sorted, x['islands'])))
    out['regisl'] = t
    print('regisl', json.dumps({k: v for k, v in t.items() if k != 'per_job_net'}))
json.dump(out, open(os.path.join(HERE, 'aggregate.json'), 'w'), indent=1)
for k, v in rc.items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'per_house'})
print('xcheck maxdiff', maxdiff, 'houses', len(R))
for k in ('s1', 's2', 'sall'):
    if k in rg:
        print(k, rg[k]['n_jobs'], json.dumps({a: {t: round(x, 4) for t, x in rg[k][a].items() if x is not None} for a in arms}), 'e4diff', rg[k]['e4_asis_b4_maxdiff'])
print(json.dumps(tt))

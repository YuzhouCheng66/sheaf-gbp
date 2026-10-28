"""ri_aggregate.py (R-I): tables from RI/out/*.json (probe only). Writes RI/ri_summary.txt and RI/ri_summary.json."""
import os, sys, json, glob  # noqa: E401
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
E6J = os.path.abspath(os.path.join(HERE, '..', '..', 'g4', 'e6', 'jobs'))
SEEDS = ('342101', '342202')
PAIR = {'342101': 'A', '342202': 'B'}
lines = []
J = {}


def P(*a):
    s = ' '.join(str(x) for x in a); lines.append(s); print(s)


def mean_f1(rows, key='furn_f1'):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else float('nan')


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else float('nan')


def addc(a, b):
    if a is None:
        return json.loads(json.dumps(b))
    if isinstance(b, dict):
        for k, v in b.items():
            a[k] = addc(a.get(k), v)
        return a
    return (np.asarray(a, np.float64) + np.asarray(b, np.float64)).tolist()


recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, '*.json')))]
fails = [r for r in recs if r.get('failed')]
P(f'records {len(recs)}, failed {len(fails)}, skipped {sum(1 for r in recs if r.get("skipped"))}')
for r in fails:
    P('FAILED', r.get('job'), r['failed'][-400:])
arms6 = ('B6_A', 'B6_B', 'NET_342101', 'NET_342202', 'E3L3_342101', 'E3L3_342202')

# ======================================================== Q1: VAL seed 1 strata
V = [r for r in recs if r.get('mode') == 'val' and not r.get('failed')]
P(f'\n=== Q1 VAL seed 1: {len(V)} houses ===')
for tn in ('cam_nf', 'cam'):
    for wh in (('R8', 'K0') if tn == 'cam_nf' else ('R8',)):
        P(f'\n-- {tn} {wh}: mean furn F1 (agent rows), pooled F1, free recall')
        C = {}
        for arm in (arms6 if wh == 'R8' else ('B6_A', 'B6_B', 'NET_342101', 'NET_342202', 'E3L3_342101', 'E3L3_342202')):
            rows = [x for r in V for x in r['tiers'][tn][wh][arm]['rows']]
            c = None
            for r in V:
                c = addc(c, r['tiers'][tn][wh][arm]['counts'])
            C[arm] = c
            b = c['base']
            P(f'  {arm:12s} mean {mean_f1(rows):.4f}  pooled {f1(*b[:3]):.4f}  free {mean_f1(rows, "free_recall"):.4f}  tol1 {mean_f1(rows, "furn_f1_tol1"):.4f}  '
              f'tp {b[0]:.0f} fp {b[1]:.0f} fn {b[2]:.0f}')
        J.setdefault('Q1', {}).setdefault(tn, {})[wh] = C
        for ts in SEEDS:
            n, e = C[f'NET_{ts}'], C[f'E3L3_{ts}']
            P(f'  [{ts}] E3L3 - NET: dTP {e["base"][0]-n["base"][0]:+.0f} dFP {e["base"][1]-n["base"][1]:+.0f} dFN {e["base"][2]-n["base"][2]:+.0f}')
            P('    by k (0,1,2,3,4,>=5): pooled F1 NET / E3L3 / d(pp) ; dFP ; dFN ; n_true')
            for k in range(6):
                nk, ek = n['k'][k], e['k'][k]
                P(f'      k={k}: {f1(*nk):.4f} / {f1(*ek):.4f} / {100*(f1(*ek)-f1(*nk)):+.2f} ; dFP {ek[1]-nk[1]:+.0f} ; dFN {ek[2]-nk[2]:+.0f} ; true {nk[0]+nk[2]:.0f}')
            P('    by height [0,.3) [.3,.9) >=.9 m: F1 NET / E3L3 / d ; dFP ; dFN ; n_true')
            for hb in range(3):
                nk, ek = n['h'][hb], e['h'][hb]
                P(f'      h{hb}: {f1(*nk):.4f} / {f1(*ek):.4f} / {100*(f1(*ek)-f1(*nk)):+.2f} ; dFP {ek[1]-nk[1]:+.0f} ; dFN {ek[2]-nk[2]:+.0f} ; true {nk[0]+nk[2]:.0f}')
            P(f'    FN surface/interior NET {n["fn_split"]} E3L3 {e["fn_split"]} ; FP adjacent/far NET {n["fp_split"]} E3L3 {e["fp_split"]}')
            if 'whole' in n:
                P(f'    whole-object miss: FN vox in whole-missed objects NET {n["whole"][0]:.0f}/{n["base"][2]:.0f}, E3L3 {e["whole"][0]:.0f}/{e["base"][2]:.0f}; '
                  f'objects missed NET {n["whole"][1]:.0f} E3L3 {e["whole"][1]:.0f} of {n["whole"][2]:.0f}; size bins (1-4,5-19,20-99,100+) NET {n["whole_size"]} E3L3 {e["whole_size"]}')
            cats = sorted(n['cat'], key=lambda c: -(n['cat'][c][0] + n['cat'][c][2]))
            P('    by kind: true vox ; F1 NET / E3L3 / d(pp) ; dFP dFN')
            for c in cats:
                nc, ec = n['cat'][c], e['cat'][c]
                if nc[0] + nc[2] < 50:
                    continue
                P(f'      {c:14s} {nc[0]+nc[2]:8.0f} ; {f1(*nc):.4f} / {f1(*ec):.4f} / {100*(f1(*ec)-f1(*nc)):+.2f} ; {ec[1]-nc[1]:+.0f} {ec[2]-nc[2]:+.0f}')
        # paired diffs
        P(f'  paired voxel diff E3L3 vs NET ({tn} {wh}): lose = E3L3 wrong & NET right; win = reverse')
        for ts in SEEDS:
            key = ts if wh == 'R8' else f'K0_{ts}'
            d = None
            for r in V:
                d = addc(d, r['tiers'][tn]['diff'][key])
            J['Q1'][tn].setdefault('diff', {})[key] = d
            lo, wi = d['lose'], d['win']
            P(f'    [{ts}] FN surf/int lose {lo["fn"]} win {wi["fn"]} ; FP adj/far lose {lo["fp"]} win {wi["fp"]}')
            P(f'          lose by k FN {[int(x[0]) for x in lo["k"]]} FP {[int(x[1]) for x in lo["k"]]} ; win by k FN {[int(x[0]) for x in wi["k"]]} FP {[int(x[1]) for x in wi["k"]]}')
            P(f'          lose by h FN {[int(x[0]) for x in lo["h"]]} FP {[int(x[1]) for x in lo["h"]]} ; win by h FN {[int(x[0]) for x in wi["h"]]} FP {[int(x[1]) for x in wi["h"]]}')
    if tn == 'cam_nf':
        zd = {ts: np.sum([r['tiers'][tn]['zdist'][ts] for r in V], 0) for ts in SEEDS}
        P(f'  z distribution (-1, 0, +1) on footprint: ' + '; '.join(f'{ts}: {np.round(v / v.sum(), 3).tolist()}' for ts, v in zd.items()))
# per-house deltas (spread)
P('\n  per-house R8 cam_nf furn F1 delta E3L3 - NET (pp):')
for ts in SEEDS:
    ds = []
    for r in V:
        T = r['tiers']['cam_nf']['R8']
        ds.append(100 * (mean_f1(T[f'E3L3_{ts}']['rows']) - mean_f1(T[f'NET_{ts}']['rows'])))
    P(f'    {ts}: ' + ' '.join(f'{x:+.1f}' for x in ds) + f' | mean {np.nanmean(ds):+.2f} min {np.nanmin(ds):+.2f}')

# ======================================================== Q2: OOD on TRAIN holdout houses
O = [r for r in recs if r.get('mode') == 'ood' and not r.get('failed') and not r.get('skipped')]
P(f'\n=== Q2 OOD (TRAIN holdout houses, seed 203, cam_nf): {len(O)} jobs ===')
Q2 = {}
for v in ('ctrl', 'novel', 'floating'):
    X = [r for r in O if r['variant'] == v]
    if not X:
        continue
    hn = ['r%d_i%d' % (r['rooms'], r['index']) for r in X]
    P(f'\n-- variant {v}: {len(X)} houses {hn}')
    for ts in SEEDS:
        h = PAIR[ts]
        def zone_sum(getter):
            c = np.zeros(9)
            for r in X:
                z = getter(r)
                if z is not None:
                    c += np.asarray(z['zone']['c'], np.float64)
            return c
        zn = zone_sum(lambda r: r['op']['R8'][f'NET_{ts}']['zones'])
        zb = zone_sum(lambda r: r['op']['R8'][f'B6_{h}']['zones'])
        ze = zone_sum(lambda r: r['e3'][f'E3L3_{ts}']['zones'])
        rows_n = [row for r in X for row, w in r['op']['R8'][f'NET_{ts}']['rows'] for _ in range(w)]
        rows_b = [row for r in X for row, w in r['op']['R8'][f'B6_{h}']['rows'] for _ in range(w)]
        rows_e = [row for r in X for row in r['e3'][f'E3L3_{ts}']['rows']]
        fl_n = np.sum([r['op']['flips'][f'NET_{ts}'][f'B6_{h}']['le_-0'] for r in X], 0)
        fl_e = np.sum([r['e3'][f'E3L3_{ts}']['flips']['le_-0'] for r in X], 0)
        fl_n2 = np.sum([r['op']['flips'][f'NET_{ts}'][f'B6_{h}']['le_-2'] for r in X], 0)
        fl_e2 = np.sum([r['e3'][f'E3L3_{ts}']['flips']['le_-2'] for r in X], 0)
        ph_n = np.mean([r['op']['R8'][f'NET_{ts}']['phantoms'][0] for r in X]); ph_b = np.mean([r['op']['R8'][f'B6_{h}']['phantoms'][0] for r in X])
        ph_e = np.mean([r['e3'][f'E3L3_{ts}']['phantoms'][0] for r in X])
        zf = lambda c: f1(c[0], c[1], c[2])
        pr = lambda f: f[0] / (f[0] + f[1]) if (f[0] + f[1]) else float('nan')
        d = dict(zone=dict(NET=zf(zn), B6=zf(zb), E3L3=zf(ze)), overall=dict(NET=mean_f1(rows_n), B6=mean_f1(rows_b), E3L3=mean_f1(rows_e)),
                 free=dict(NET=mean_f1(rows_n, 'free_recall'), B6=mean_f1(rows_b, 'free_recall'), E3L3=mean_f1(rows_e, 'free_recall')),
                 flip=dict(NET=pr(fl_n), E3L3=pr(fl_e), NET_n=fl_n[:2].tolist(), E3L3_n=fl_e[:2].tolist(), NET_le2=pr(fl_n2), E3L3_le2=pr(fl_e2)),
                 phantoms=dict(NET=ph_n, B6=ph_b, E3L3=ph_e))
        fams = {}
        for r in X:
            for fam, c in r['e3'][f'E3L3_{ts}']['families'].items():
                a = fams.setdefault(fam, np.zeros((3, 3)))
                a[0] += c; a[1] += r['op']['R8'][f'NET_{ts}']['families'][fam]; a[2] += r['op']['R8'][f'B6_{h}']['families'][fam]
        d['families'] = {fam: dict(E3L3=f1(*a[0]), NET=f1(*a[1]), B6=f1(*a[2]), n=float(a[2][0] + a[2][2])) for fam, a in fams.items()}
        Q2.setdefault(v, {})[ts] = d
        P(f'  [{ts} / B6_{h}] zone F1 B6 {d["zone"]["B6"]:.4f} NET {d["zone"]["NET"]:.4f} ({100*(d["zone"]["NET"]-d["zone"]["B6"]):+.2f}) '
          f'E3L3 {d["zone"]["E3L3"]:.4f} ({100*(d["zone"]["E3L3"]-d["zone"]["B6"]):+.2f} vs B6, {100*(d["zone"]["E3L3"]-d["zone"]["NET"]):+.2f} vs NET)')
        P(f'      overall B6 {d["overall"]["B6"]:.4f} NET {d["overall"]["NET"]:.4f} E3L3 {d["overall"]["E3L3"]:.4f} | free B6 {d["free"]["B6"]:.4f} NET {d["free"]["NET"]:.4f} E3L3 {d["free"]["E3L3"]:.4f}')
        P(f'      flip precision (S<=0) NET {d["flip"]["NET"]:.3f} {d["flip"]["NET_n"]} E3L3 {d["flip"]["E3L3"]:.3f} {d["flip"]["E3L3_n"]} | S<=-2nat NET {d["flip"]["NET_le2"]:.3f} E3L3 {d["flip"]["E3L3_le2"]:.3f}')
        P(f'      phantoms/house B6 {ph_b:.2f} NET {ph_n:.2f} E3L3 {ph_e:.2f}')
        for fam, q in sorted(d['families'].items(), key=lambda x: -x[1]['n']):
            if q['n'] >= 100:
                P(f'        fam {fam:16s} n {q["n"]:7.0f} B6 {q["B6"]:.3f} NET {q["NET"]:.3f} ({100*(q["NET"]-q["B6"]):+.1f}) E3L3 {q["E3L3"]:.3f} ({100*(q["E3L3"]-q["B6"]):+.1f})')
J['Q2'] = Q2

# ======================================================== Q3: E6 composition with E3-L3
E6 = [r for r in recs if r.get('mode') == 'e6' and not r.get('failed') and not r.get('skipped')]
P(f'\n=== Q3 cam + G4-E6 (VAL seed 1, aligned row): {len(E6)} houses ===')
Vd = {(r['rooms'], r['index']): r for r in V}
Q3 = {}
repro = []
for arm in ('noreg', 'e6'):
    for ro in arms6:
        rows = [x for r in E6 for x in r['arms'][arm]['aligned']['R8'][ro]['rows']]
        Q3.setdefault(arm, {})[ro] = mean_f1(rows)
rows_nf = {ro: [x for r in E6 if (r['rooms'], r['index']) in Vd for x in Vd[(r['rooms'], r['index'])]['tiers']['cam_nf']['R8'][ro]['rows']] for ro in arms6}
rows_asis = {ro: [x for r in E6 if (r['rooms'], r['index']) in Vd for x in Vd[(r['rooms'], r['index'])]['tiers']['cam']['R8'][ro]['rows']] for ro in arms6}
Q3['truepose_cam_nf'] = {ro: mean_f1(v) for ro, v in rows_nf.items()}
Q3['noreg_asis'] = {ro: mean_f1(v) for ro, v in rows_asis.items()}
for r in E6:
    p = os.path.join(E6J, f'val_r{r["rooms"]}_i{r["index"]}_s{r["seed"]}.json')
    if os.path.exists(p):
        e = json.load(open(p))
        for arm in ('noreg', 'e6'):
            for ro in ('NET_342101', 'NET_342202', 'B6_A'):
                theirs = [x['furn_f1'] for x, w in e['conds']['cam']['arms'][arm]['aligned'][ro] for _ in range(w)]
                mine = [x['furn_f1'] for x in r['arms'][arm]['aligned']['R8'][ro]['rows']]
                repro.append(theirs == mine)
P(f'  NET/B6 reproduction vs g4/e6 jobs (row-for-row): {sum(repro)}/{len(repro)}; pub sha ok {sum(r["pub_sha_ok"] for r in E6)}/{len(E6)}')
P('  readout        noreg-asis noreg-aligned e6-aligned  truepose(cam_nf)  e6-noreg(pp)  island loss = truepose - e6 (pp)')
for ro in arms6:
    P(f'  {ro:12s} {Q3["noreg_asis"][ro]:.4f}   {Q3["noreg"][ro]:.4f}     {Q3["e6"][ro]:.4f}     {Q3["truepose_cam_nf"][ro]:.4f}        '
      f'{100*(Q3["e6"][ro]-Q3["noreg"][ro]):+.2f}        {100*(Q3["truepose_cam_nf"][ro]-Q3["e6"][ro]):+.2f}')
for ts in SEEDS:
    P(f'  E3L3 - NET [{ts}]: noreg-aligned {100*(Q3["noreg"]["E3L3_"+ts]-Q3["noreg"]["NET_"+ts]):+.2f}  e6-aligned {100*(Q3["e6"]["E3L3_"+ts]-Q3["e6"]["NET_"+ts]):+.2f}  '
      f'cam_nf {100*(Q3["truepose_cam_nf"]["E3L3_"+ts]-Q3["truepose_cam_nf"]["NET_"+ts]):+.2f}  asis {100*(Q3["noreg_asis"]["E3L3_"+ts]-Q3["noreg_asis"]["NET_"+ts]):+.2f}')
    ds = [100 * (mean_f1(r['arms']['e6']['aligned']['R8']['E3L3_' + ts]['rows']) - mean_f1(r['arms']['e6']['aligned']['R8']['NET_' + ts]['rows'])) for r in E6]
    P(f'    per-house e6 E3L3-NET: ' + ' '.join(f'{x:+.1f}' for x in ds))
    fr = {arm: mean_f1([x for r in E6 for x in r['arms'][arm]['aligned']['R8']['E3L3_' + ts]['rows']], 'free_recall') for arm in ('noreg', 'e6')}
    frn = {arm: mean_f1([x for r in E6 for x in r['arms'][arm]['aligned']['R8']['NET_' + ts]['rows']], 'free_recall') for arm in ('noreg', 'e6')}
    P(f'    free recall E3L3 noreg {fr["noreg"]:.4f} e6 {fr["e6"]:.4f} | NET noreg {frn["noreg"]:.4f} e6 {frn["e6"]:.4f}')
# island loss decomposition (E3L3 342101, e6 vs truepose) by k / FN-FP
for ts in SEEDS:
    ce = None; cn = None
    for r in E6:
        ce = addc(ce, r['arms']['e6']['aligned']['R8']['E3L3_' + ts]['counts'])
    for r in E6:
        if (r['rooms'], r['index']) in Vd:
            cn = addc(cn, Vd[(r['rooms'], r['index'])]['tiers']['cam_nf']['R8']['E3L3_' + ts]['counts'])
    P(f'  [{ts}] E3L3 e6 vs truepose: pooled F1 {f1(*ce["base"][:3]):.4f} vs {f1(*cn["base"][:3]):.4f}; FP {ce["base"][1]:.0f} vs {cn["base"][1]:.0f} (adj/far {ce["fp_split"]} vs {cn["fp_split"]}); '
          f'FN {ce["base"][2]:.0f} vs {cn["base"][2]:.0f} (surf/int {ce["fn_split"]} vs {cn["fn_split"]})')
for arm in ('noreg', 'e6'):
    for ts in SEEDS:
        cn = ce = d = None
        for r in E6:
            A = r['arms'][arm]['aligned']
            cn = addc(cn, A['R8']['NET_' + ts]['counts']); ce = addc(ce, A['R8']['E3L3_' + ts]['counts']); d = addc(d, A['diff'][ts])
        P(f'  {arm} aligned [{ts}] NET vs E3L3: pooled {f1(*cn["base"][:3]):.4f} vs {f1(*ce["base"][:3]):.4f}; TP {cn["base"][0]:.0f}/{ce["base"][0]:.0f} '
          f'FP adj/far {cn["fp_split"]} / {ce["fp_split"]}; FN surf/int {cn["fn_split"]} / {ce["fn_split"]}')
        P(f'      paired diff: lose FN {d["lose"]["fn"]} FP {d["lose"]["fp"]} | win FN {d["win"]["fn"]} FP {d["win"]["fp"]}')
        P(f'      lose FP by k {[int(x[1]) for x in d["lose"]["k"]]} win FP by k {[int(x[1]) for x in d["win"]["k"]]}; lose FN by k {[int(x[0]) for x in d["lose"]["k"]]} win FN by k {[int(x[0]) for x in d["win"]["k"]]}')
# house-level: does the E3L3-NET e6 gap track the island count / registration residual?
hs = []
for r in E6:
    g = [100 * (mean_f1(r['arms']['e6']['aligned']['R8']['E3L3_' + ts]['rows']) - mean_f1(r['arms']['e6']['aligned']['R8']['NET_' + ts]['rows'])) for ts in SEEDS]
    nt = r['arms']['e6']['aligned']['R8']['NET_342101']['counts']['base']
    hs.append((f'r{r["rooms"]}_i{r["index"]}', len(r['islands']), round(float(np.mean(g)), 2), int(nt[0] + nt[2])))
P('  house, islands, mean(E3L3-NET) e6 pp, true furniture vox (NET 342101 tp+fn):', hs)
med = np.median([h[2] for h in hs]); P(f'  median house gap {med:+.2f} pp; mean without the worst house {np.mean(sorted([h[2] for h in hs])[1:]):+.2f}')
J['Q3'] = Q3
isl = [len(r['islands']) for r in E6]
P(f'  islands per house: mean {np.mean(isl):.2f}')

# ======================================================== Q4: tiers on TRAIN holdout houses
T4 = [r for r in recs if r.get('mode') == 'tier' and not r.get('failed')]
P(f'\n=== Q4 tiers (TRAIN holdout, seed 203): {len(T4)} houses ===')
Q4 = {}
for tn in ('cam_nf', 'lidar', 'lidar_nf', 'harsh', 'harsh_nf'):
    s = {}
    for wh in ('R8', 'K0'):
        for ro in arms6:
            rows = [x for r in T4 if tn in r['tiers'] and ro in r['tiers'][tn].get(wh, {}) for x in r['tiers'][tn][wh][ro]['rows']]
            if rows:
                s[f'{wh}_{ro}'] = (mean_f1(rows), mean_f1(rows, 'free_recall'))
    Q4[tn] = s
    for wh in ('R8', 'K0'):
        if f'{wh}_NET_342101' not in s:
            continue
        P(f'  {tn:9s} {wh}: ' + '  '.join(f'{ro.replace("_3421", "_")} {s[f"{wh}_{ro}"][0]:.4f}' for ro in arms6) +
          f' | E3L3-NET {100*(s[f"{wh}_E3L3_342101"][0]-s[f"{wh}_NET_342101"][0]):+.2f}/{100*(s[f"{wh}_E3L3_342202"][0]-s[f"{wh}_NET_342202"][0]):+.2f}'
          f' NET-B6 {100*(s[f"{wh}_NET_342101"][0]-s[f"{wh}_B6_A"][0]):+.2f}/{100*(s[f"{wh}_NET_342202"][0]-s[f"{wh}_B6_B"][0]):+.2f}'
          f' free E3L3 {s[f"{wh}_E3L3_342101"][1]:.4f} NET {s[f"{wh}_NET_342101"][1]:.4f} B6 {s[f"{wh}_B6_A"][1]:.4f}')
for tn in ('lidar', 'harsh'):
    for ts in SEEDS:
        d = None
        for r in T4:
            d = addc(d, r['tiers'][tn]['diff'][f'K0_{ts}'])
        if d:
            P(f'  {tn} K0 diff [{ts}] FN surf/int lose {d["lose"]["fn"]} win {d["win"]["fn"]}; FP adj/far lose {d["lose"]["fp"]} win {d["win"]["fp"]}')
J['Q4'] = Q4

with open(os.path.join(HERE, 'ri_summary.txt'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines) + '\n')
with open(os.path.join(HERE, 'ri_summary.json'), 'w') as fh:
    json.dump(J, fh, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

"""rf_tables.py (R-F): print the report tables from phaseA_train.json / phaseB_val.json (copied back from bigboy). CPU only, read only."""
import json, os, sys
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(D, 'phaseA_train.json')))
B = json.load(open(os.path.join(D, 'phaseB_val.json')))
f = lambda x, n=3: 'na' if x is None else (f'{x:.{n}f}')  # noqa


def short(c):
    return c.replace('E3L2_', '').replace('_final', 'final').replace('E3L_final', 'E3L')


print('== Phase A (TRAIN holdout) ==', A['items'], 'items')
for c, S in A['configs'].items():
    ctl = S['control']; br = A['byte_ratio_median'][c]
    h = S['variants']['head']
    print(f"\n[{c}] ctrl free {ctl['free']:.4f} fpr {ctl['fpr']:.4f} furn {ctl['furn_f1']:.4f} | bytes x ctrl: model {br['model']:.3f} fp {br['fp']:.4f} "
          f"kmask {br['kmask']:.3f} model+fp {br['model_plus_fp']:.3f} model+kmask {br['model_plus_kmask']:.3f}")
    print(f"  head: max free coverage @FPR<=+0.15pp {f(h['max_coverage_fpr15'])} @+0.3pp {f(h['max_coverage_fpr30'])}; official {h.get('at_official_t_known')}")
    for v, x in S['variants'].items():
        rr = ' | '.join(f"{rn[:1]}: F1 {f(y['f1'])} free {f(y['free'], 4)} fpr {f(y['fpr'], 4)}{'' if y['feasible'] else ' INF'}"
                        + (f" tk {y['t_known']:.2f}" if 't_known' in y else '') + f" to {y['t_occ']:.2f}" for rn, y in x['rules'].items())
        extra = ''
        if 'frontier_fpr30' in x:
            extra = ' front30 ' + ' '.join(f"{k}:{f(v2['f1'])}" for k, v2 in x['frontier_fpr30'].items())
            extra += ' | strict ' + json.dumps({k: round(v2, 4) for k, v2 in x['strictest_known'].items()})
        else:
            extra = f" best {f(x['best_any']['f1'])}@free {f(x['best_any']['free'], 4)} front " + ' '.join(f"{k}:{f(v2)}" for k, v2 in x['frontier'].items())
        print(f"  {v:11s} {rr}{extra}")
    lb = S['lb0']
    print('  lb0:', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in lb.items() if k in ('share_target_known_in_lb0', 'share_target_known_in_lb0_fp',
          'share_target_unknown_in_fp_lb0', 'auc_known_vs_unk_in_lb0_fp', 'auc_known_vs_unk_in_lb0', 'auc_known_vs_unk_in_lbpos')}))
    print('  lb0 pred-known:', json.dumps({k: {a: round(b, 3) for a, b in v.items() if b is not None} for k, v in lb['pred_known_rate'].items() if k in ('A_fpr30_maxF1', 'official')}))
    print('  lb0 occ:', json.dumps(lb.get('occ_at_ruleA')), json.dumps(lb['occ_counts']))

print('\n== Phase B (VAL, TRAIN thresholds) ==')
print('control', json.dumps(B['control']))
for c in B['configs']:
    sd = c.split('_')[-2] if c.startswith('E3L2') else c.split('_')[-1]
    for v in ('head', 'fp_lit', 'fp_and', 'fp_and_lit', 'kmask', 'oracle'):
        for rule in ('official', 'A_fpr30_maxF1', 'B_fpr30_free50', 'C_fpr15_free25'):
            out = []
            for ns in ('1', '2'):
                r = B['rows'].get(f'{c}|{v}|{rule}|{ns}')
                if r is None:
                    continue
                ct = B['control'][f'{sd}|{ns}']
                out.append(f"s{ns}: furn {f(r['furn_f1'])} (d {100 * (r['furn_f1'] - ct['furn_f1']):+.1f}) free {f(r['free_recall'], 4)} (d {100 * (r['free_recall'] - ct['free_recall']):+.2f}) "
                           f"fpr {f(r['unknown_known_fpr'], 4)} (d {100 * (r['unknown_known_fpr'] - ct['unknown_known_fpr']):+.2f}) surf {f(r['surface_f1'])}")
            if out:
                print(f"  {c:28s} {v:10s} {rule:15s} " + ' | '.join(out))
    bm = B['bytes_median']
    print(f"  {c} bytes x ctrl: " + ' '.join(f"s{ns}: model {bm[f'{c}|{ns}'] / B['control'][f'{sd}|{ns}']['bytes']:.3f} +fp {bm[f'{c}+fp|{ns}'] / B['control'][f'{sd}|{ns}']['bytes']:.3f} "
                                                 f"+kmask {bm[f'{c}+kmask|{ns}'] / B['control'][f'{sd}|{ns}']['bytes']:.3f}" for ns in ('1', '2')))
    lb = B['configs'][c]['lb0']
    print('   VAL lb0:', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in lb.items() if k.startswith('share') or k.startswith('auc')}))
    print('   VAL lb0 pred-known:', json.dumps({k: {a: round(b, 3) for a, b in v.items() if b is not None} for k, v in lb['pred_known_rate'].items() if k in ('A_fpr30_maxF1', 'official')}))
    print('   VAL lb0 occ:', json.dumps(lb.get('occ_at_ruleA')), json.dumps(lb['occ_counts']))
    vo = B['configs'][c]['variants']
    print('   VAL-tuned (diag) oracle best', f(vo['oracle']['best_any']['f1']), 'oracle_sent best', f(vo['oracle_sent']['best_any']['f1']), 'kmask best', f(vo['kmask']['best_any']['f1']),
          'head A', f(vo['head']['rules']['A_fpr30_maxF1']['f1']))
print('fp bytes x ctrl (VAL):', {ns: B['bytes_median'][f'fp|{ns}'] for ns in ('1', '2')}, 'kmask', {ns: B['bytes_median'][f'kmask|{ns}'] for ns in ('1', '2')})

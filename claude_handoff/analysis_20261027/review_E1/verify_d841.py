"""R-C independent check of D8-41 numbers against g4/e1/summary.json (read-only)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, '..', 'g4', 'e1', 'summary.json')))
S0 = S['s0_rnet_clean']
CONDS = ['clean', 'range_only', 'pose_only', 'inter_only', 'lidar', 'cam_nf', 'cam', 'harsh']


def g(panel, seed, cond, I='V3'):
    return S['panels'][panel]['seeds'][seed]['conds'][cond][I]


def f(x, n=3):
    return 'None' if x is None else f'{x:.{n}f}'


for panel in ('val', 'fresh'):
    print(f'===== {panel} =====')
    print('n_houses', {s: S['panels'][panel]['seeds'][s]['n_houses'] for s in ('1', '2')})
    for cond in CONDS:
        for s in ('1', '2'):
            c = g(panel, s, cond)
            R = c['R8']['sum']; K = c['K0']
            rn = c['rnet_median']
            print(f"{cond:10s} s{s} R8sum furn {f(R['furn_f1'])} tol1 {f(R['furn_f1_tol1'])} K0 {f(K['furn_f1'])} "
                  f"free {f(R['free_recall'])} fpr {f(R['unknown_known_fpr'],4)} B4relay {rn['B4_relay']} ({rn['B4_relay']/S0:.1f}x) "
                  f"int {rn['B4_int_relay']:.0f} ({rn['B4_int_relay']/S0:.1f}x) | maxabs {f(c['R8']['maxabs']['furn_f1'])} maxocc {f(c['R8']['max_occ']['furn_f1'])} "
                  f"| K0free {f(K['free_recall'])} K0fpr {f(K['unknown_known_fpr'],4)}")
    # verdict block
    v = S['panels'][panel]['verdict']['V3']['per_seed']
    for s in ('1', '2'):
        x = v[s]
        print('verdict', s, {k: (round(x[k], 4) if isinstance(x[k], float) else x[k]) for k in
                             ('B4_R8_furn', 'B4_K0_furn', 'B4_rnet_median', 'B4_rnet_over_s0', 'B5_hull_at_quarter_B4', 'B5_best_any',
                              'B5_truepose_hull', 'd_cam_nf_minus_cam')},
              'c', {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in x['c'][k].items()} for k in x['c']})
    # strata (multiplicity) for cam and cam_nf R8 sum, and cam sum_truepose
    for cond, key in (('cam', 'sum'), ('cam', 'sum_truepose'), ('cam_nf', 'sum')):
        for s in ('1', '2'):
            st = g(panel, s, cond)['R8'][key].get('strata')
            if st and 'm_furn' in st:
                print('strata', cond, key, s, [(f(e['f1']), f(e['prec'])) for e in st['m_furn']])
    # B5 tau=0 points (pub_sum) and own bytes
    for s in ('1', '2'):
        c = g(panel, s, 'cam'); B5 = c['B5']
        h = c['B5_hull_pub']
        best = max(((k, e[ms]['R8']['furn_f1'], e[ms]['rnet_median']) for k, e in B5.items() for ms in ('pub_sum', 'pub_max') if ms in e),
                   key=lambda t: t[1])
        print('B5', s, 'best_any', best, 'quarter_B4_bytes', round(h['quarter_B4_bytes']))
        for k in ('cc|0|4|111', 'box|0|4|111'):
            e = B5[k]
            print('   ', k, 'own', round(e['own_bytes_median']), 'pub_sum rnet', round(e['pub_sum']['rnet_median']),
                  f"({e['pub_sum']['rnet_median']/S0:.1f}x)", 'furn', f(e['pub_sum']['R8']['furn_f1']))
        print('    own_bytes_median', {k: round(vv) for k, vv in c['own_bytes_median'].items()})
print('decision', json.dumps(S['decision'], indent=0)[:1500])
# V1/V0/V2 cam R8 furn
for I in ('V1', 'V0', 'V2'):
    for s in ('1', '2'):
        print(I, s, 'cam R8 sum furn', f(g('val', s, 'cam', I)['R8']['sum']['furn_f1']), 'verdict', S['panels']['val']['verdict'][I]['verdict'][:60])

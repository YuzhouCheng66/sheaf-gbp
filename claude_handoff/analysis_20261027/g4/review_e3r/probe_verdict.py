"""reviewer probe of g4_e3r.verdict against D8-44 (synthetic summaries, no data)"""
import os, sys, json, tempfile
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, r'C:\Users\27118\Desktop\sheaf-gbp\experiments\sheaf-gbp-p6-local-skill\scripts\claude_v342')
import g4_e3r as E
G = E.G

def arm(furn, k0=.7, k0sep=None, free=.97, surf=.9):
    R8 = dict(furn_f1=furn, furn_f1_tol1=furn + .05, unknown_known_fpr=.02, free_recall=free, surface_f1=surf, var_occ_f1=.8)
    cal = {f'k{i}': dict(n=10, slope=1.0, ece=.01) for i in range(1, 5)}; cal['k5+'] = dict(n=10, slope=1.0, ece=.01); cal['kappa'] = 1.
    a = dict(R8=R8, K0=dict(furn_f1=k0), calibration_R8=cal)
    if k0sep is not None:
        a['K0_sep'] = dict(furn_f1=k0sep)
    return a

def panel(d_net, net_k0=.80, net_k0sep=.80, lin_k0=.70):
    P = dict(seeds={})
    for ns in ('1', '2'):
        tiers = dict(cam_nf=dict(B4=arm(.85, k0=.66), B6_A=arm(.88, k0=.70), B6_B=arm(.879, k0=.69)), clean=dict())
        for ts in ('342101', '342202'):
            tiers['cam_nf'][f'NET_{ts}'] = arm(.88 + d_net, k0=net_k0, k0sep=net_k0sep)
            tiers['cam_nf'][f'LIN_{ts}'] = arm(.88, k0=lin_k0)
            tiers['clean'][f'NET_{ts}'] = arm(.995, free=.995)
        P['seeds'][ns] = dict(houses=[f'r{r}_i{i}' for r, i in G.panel_specs('val')], tiers=tiers)
    return P

def summ(d_val, d_fresh=None, **kw):
    S = dict(panels=dict(val=panel(d_val, **kw)), failed=[], skipped=[], e1_mismatch=[], agg_reasons=[])
    if d_fresh is not None:
        S['panels']['fresh'] = panel(d_fresh, **kw)
    return S

crit = dict(lever_r8_margin=.03, not_enough_margin=.01, guard_free=.005, guard_surface=.005, k0_threshold='separate', k0_margin=-0.01)
print('1 lever, fresh panel absent      ->', E.verdict(summ(.05), crit)['verdict'])
print('2 not-enough VAL, fresh absent   ->', E.verdict(summ(.0), crit)['verdict'])
S = summ(.05, .05); del S['panels']['fresh']['seeds']['2']
print('3 lever, fresh seed 2 missing    ->', E.verdict(S, crit)['verdict'])
# K0: D8-44 = NET K0 (K0 threshold) >= LIN K0 (K0 threshold) - 1 pp; the code compares with the CONTROL arm's K0 at its R8 theta
v = E.verdict(summ(.05, .05, net_k0sep=.65, lin_k0=.70), crit)
c = list(v['cells'].values())[0]
print('4 NET K0_sep .65 vs LIN K0 .70 (D8-44: fails by 5 pp) ->', v['verdict'], '| code d_K0 =', round(c['d_K0_furn'], 3), 'vs control', c['control'])
# truncated run (>= 2,000 updates) with a NOT_ENOUGH readout
d = tempfile.mkdtemp()
runs = []
for sd in ('342101', '342202'):
    r = os.path.join(d, sd); os.makedirs(r)
    json.dump(dict(seed=int(sd), recipe=dict(evidence=dict(desc=dict(spec='L')))), open(os.path.join(r, 'run.json'), 'w'))
    json.dump(dict(complete=sd != '342202', truncated=sd == '342202', updates_done=5000 if sd == '342202' else 6000, updates_planned=6000), open(os.path.join(r, 'final.json'), 'w'))
    open(os.path.join(r, 'threshold.json'), 'w').write('{}')
    runs.append(r)
print('5 seed 2 truncated at 5,000/6,000, VAL d = 0 ->', E.verdict(summ(.0, .0), crit, runs=runs)['verdict'][:90])
# 6: control K0 low (.66 for B6_A at its R8 theta), the evidence's own LIN K0 .70, NET K0_sep .68 -> D8-44 fails (-2 pp < -1 pp)
S = summ(.05, .05, net_k0sep=.68, lin_k0=.70)
for P in S['panels'].values():
    for ns in P['seeds']:
        P['seeds'][ns]['tiers']['cam_nf']['B6_A']['K0']['furn_f1'] = .66
v = E.verdict(S, crit)
print('6 NET K0_sep .68, LIN K0 .70, control K0 .66 (D8-44: no lever) ->', v['verdict'], '| code d_K0 =', round(list(v['cells'].values())[0]['d_K0_furn'], 3))

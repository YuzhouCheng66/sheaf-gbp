"""analyse_3SZ_rescore.py (Claude, 2026-10-26; registered before 3S-Z's result in D8-30 and D8-31; REPORT ONLY -- 3S-Z's verdict is the
chain's, under the v420 rule, D8-29). Inputs: eval_decision_v431 JSONs of z1-s342101, z1-s342202 (arm) and l1-s342101, l1-s342202 (L1 control).
  1. 3S-Z's reading (Delta R8 var >= +3 and Delta R8 surface >= +1 pp on both seeds = lever; ...) under both decision rules, VAL and fresh.
  2. House-size readout (D8-30): per-house Delta R8 / K0 surface F1 (z1 - L1, same house, same seed) vs the chart side L, under both rules:
     mean Delta for houses <= 13 m vs >= 17 m and the correlation r(L, Delta). B's prediction (D8-32): the gain does not grow with L.
  3. z1's TRAIN-24 optimal occupancy threshold (mechanism of D8-31: within [+1.0, +2.0] at both hops).
python analyse_3SZ_rescore.py z1-s342101.json z1-s342202.json l1-s342101.json l1-s342202.json > rescore.json"""
import json, sys
import numpy as np

z = [json.load(open(f)) for f in sys.argv[1:3]]; l = [json.load(open(f)) for f in sys.argv[3:5]]
assert [r['meta']['seed'] for r in z] == [r['meta']['seed'] for r in l] and len({r['meta']['seed'] for r in z}) == 2, 'seeds'
assert all(r['meta']['blocks'] == [64, 64, 1] for r in z) and all(r['meta']['blocks'] == [32, 32, 1] for r in l), 'layouts'
assert all('sanity_failed' not in r for r in z + l), 'reconciliation'

def summ(r, panel, rule, hop, m):
    return r['panels'][panel][rule]['summary'][hop][m]

def reading(ds, dv):
    if min(dv) >= .03 and min(ds) >= .01: return 'lever'
    if max(ds) <= -.01 or max(dv) <= -.02: return 'worse'
    if max(ds) < .01 and max(dv) < .02: return 'not better'
    return 'ambiguous'

out = dict(inputs=sys.argv[1:5], note='report only; 3S-Z verdict = the chain (v420 rule, VAL)')
for panel in ('val', 'fresh'):
    for rule in ('v420', 'calibrated'):
        ds = [summ(a, panel, rule, '8', 'surface_f1') - summ(c, panel, rule, '8', 'surface_f1') for a, c in zip(z, l)]
        dv = [summ(a, panel, rule, '8', 'var_occ_f1') - summ(c, panel, rule, '8', 'var_occ_f1') for a, c in zip(z, l)]
        dk = [summ(a, panel, rule, '0', 'surface_f1') - summ(c, panel, rule, '0', 'surface_f1') for a, c in zip(z, l)]
        df = [summ(a, panel, rule, '8', 'unknown_known_fpr') - summ(c, panel, rule, '8', 'unknown_known_fpr') for a, c in zip(z, l)]
        out[f'{panel}_{rule}'] = dict(reading=reading(ds, dv), dR8_surface=[round(x, 4) for x in ds], dR8_var=[round(x, 4) for x in dv], dK0_surface=[round(x, 4) for x in dk],
                                     dR8_fpr=[round(x, 4) for x in df], arm_R8_surface=[round(summ(a, panel, rule, '8', 'surface_f1'), 4) for a in z],
                                     arm_K0_surface=[round(summ(a, panel, rule, '0', 'surface_f1'), 4) for a in z], arm_R8_var=[round(summ(a, panel, rule, '8', 'var_occ_f1'), 4) for a in z])
size = {}
for rule in ('v420', 'calibrated'):
    for hop in ('0', '8'):
        rows = []
        for a, c in zip(z, l):
            pa, pc = a['per_scene']['val'], c['per_scene']['val']
            assert [json.dumps(x['spec'], sort_keys=True) for x in pa] == [json.dumps(x['spec'], sort_keys=True) for x in pc], 'VAL houses'
            L = np.array([x['chart_m'] for x in pa]); d = np.array([x[f'{rule}_K{hop}']['surface_f1'] - y[f'{rule}_K{hop}']['surface_f1'] for x, y in zip(pa, pc)])
            rows.append(dict(seed=a['meta']['seed'], r_L_delta=round(float(np.corrcoef(L, d)[0, 1]), 3), small_le13=round(float(d[L <= 13].mean()), 4), large_ge17=round(float(d[L >= 17].mean()), 4),
                             n_small=int((L <= 13).sum()), n_large=int((L >= 17).sum())))
        size[f'{rule}_K{hop}'] = rows
out['house_size_readout'] = size
try:
    out['z1_train_optimal_occ_threshold'] = {r['meta']['seed']: r['panels']['train24']['train_optimal_occ_threshold'] for r in z}
except KeyError as e:
    out['z1_train_optimal_occ_threshold'] = f'n/a ({e})'
print(json.dumps(out, indent=1))

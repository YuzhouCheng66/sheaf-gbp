"""analyse_3SDR.py (Claude, 2026-10-26; review code_review_v431_decision.md S8): the pre-registered verdict of batch 3S-DR (D8-31) from the
eval_decision_v431 JSONs of the two seeds. Delta = calibrated rule - v420 rule on the SAME checkpoint (paired by seed and house).
Primary panel: fresh (never used for any choice). VAL reported only (analyst C saw it).
  lever: Delta R8 var >= +3 pp AND Delta R8 surface >= +1 pp on both seeds; worse: both <= -1 / <= -2; not better: both < +1 and < +2;
  else ambiguous. Guards: R8 unknown->known FPR <= control + 0.3 pp; R8 slab recall >= control - 2 pp; K0 var >= control - 2 pp.
  If only the FPR guard fails, the occ_only fallback (FPR equal to v420's by construction) is reported, without a verdict.
  Mechanism (review S6, fixed before the run): the TRAIN-24 optimal occupancy threshold must lie in [+1.0, +2.0] at BOTH K0 and R8.
python analyse_3SDR.py runs/v431-decision/c2-s342101.json runs/v431-decision/c2-s342202.json > runs/v431-decision/analysis.json"""
import json, sys

def pick(res, panel, rule):
    s = res['panels'][panel][rule]['summary']
    return dict(K0_surface=s['0']['surface_f1'], K0_var=s['0']['var_occ_f1'], R8_surface=s['8']['surface_f1'], R8_var=s['8']['var_occ_f1'],
                R8_furn=s['8']['furn_f1'], K0_furn=s['0']['furn_f1'], R8_fpr=s['8']['unknown_known_fpr'], K0_fpr=s['0']['unknown_known_fpr'],
                R8_tol1=s['8']['surface_f1_tol1'], R8_slab_recall=res['panels'][panel][rule]['R8_slab_recall'])

def verdict(deltas):
    sv = [d['R8_surface'] for d in deltas]; vv = [d['R8_var'] for d in deltas]
    if min(vv) >= .03 and min(sv) >= .01: return 'lever'
    if max(sv) <= -.01 or max(vv) <= -.02: return 'worse'
    if max(sv) < .01 and max(vv) < .02: return 'not better'
    return 'ambiguous'

def guards(pairs):
    return dict(fpr_ok=all(a['R8_fpr'] <= c['R8_fpr'] + .003 for a, c in pairs), slab_ok=all(a['R8_slab_recall'] >= c['R8_slab_recall'] - .02 for a, c in pairs),
                k0var_ok=all(a['K0_var'] >= c['K0_var'] - .02 for a, c in pairs))

runs = [json.load(open(f)) for f in sys.argv[1:3]]
assert all('sanity_failed' not in r for r in runs) and len({r['meta']['seed'] for r in runs}) == 2, 'sanity / seeds'
out = dict(inputs=sys.argv[1:3], constants=runs[0]['constants'], sanity={r['meta']['seed']: {p: (r['panels'][p]['sanity_max_abs_diff_vs_final'], r['panels'][p]['sanity_max_abs_diff_per_scene_surface']) for p in ('val', 'fresh')} for r in runs})
for panel in ('fresh', 'val'):
    block = {}
    for rule in ('calibrated', 'occ_only'):
        pairs = [(pick(r, panel, rule), pick(r, panel, 'v420')) for r in runs]
        deltas = [{k: round(a[k] - c[k], 4) for k in a if a[k] is not None and c[k] is not None} for a, c in pairs]
        block[rule] = dict(verdict=verdict(deltas), guards=guards(pairs), deltas={r['meta']['seed']: d for r, d in zip(runs, deltas)},
                           arm={r['meta']['seed']: {k: round(v, 4) for k, v in a.items() if v is not None} for r, (a, c) in zip(runs, pairs)},
                           control={r['meta']['seed']: {k: round(v, 4) for k, v in c.items() if v is not None} for r, (a, c) in zip(runs, pairs)})
    out[panel] = block
p = out['fresh']['calibrated']
adopt = p['verdict'] == 'lever' and all(p['guards'].values())
out['verdict'] = p['verdict'] if adopt or p['verdict'] != 'lever' else 'lever, but a guard failed: not adopted'
out['adopt_calibrated'] = adopt
if p['verdict'] == 'lever' and not p['guards']['fpr_ok'] and p['guards']['slab_ok'] and p['guards']['k0var_ok']:
    out['fallback_reported'] = dict(rule='occ_only', fresh=out['fresh']['occ_only'], note='reported only, no verdict (D8-31)')
try:
    opt = {r['meta']['seed']: r['panels']['train24']['train_optimal_occ_threshold'] for r in runs}
    out['mechanism'] = dict(train_optimal_occ_threshold=opt, within_1_2=all(1.0 <= float(t) <= 2.0 for o in opt.values() for t in o.values()))
except KeyError as e:
    out['mechanism'] = f'n/a ({e})'
print(json.dumps(out, indent=1))

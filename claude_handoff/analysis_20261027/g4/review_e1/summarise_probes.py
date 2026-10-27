"""summarise_probes.py: tables of the reviewer probes (r1_integrator_probe.json, r1b_perbeam_probe.json, r2_b5_box_probe.json).
Means over houses of per-house agent means (all houses are single 8-agent components here); B5 hull = module's hull_value over the
per-config (median rate, mean furniture) points across houses, read at 1/4 of the median B4 relay, as in g4_noise_panels.aggregate."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', 'scripts', 'claude_v342'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def hull_value(points, x0):                                               # copy of g4_noise_panels.hull_value (no heavy import)
    import itertools
    pts = [(x, y) for x, y in points if x is not None and y is not None]
    best = None
    for x, y in pts:
        if x <= x0:
            best = y if best is None else max(best, y)
    if best is None:
        return None
    for (x1, y1), (x2, y2) in itertools.product(pts, pts):
        if x1 < x0 < x2:
            best = max(best, y1 + (y2 - y1) * (x0 - x1) / (x2 - x1))
    return best


def r(x, n=3):
    return None if x is None else round(float(x), n)


out = {}
p1 = os.path.join(HERE, 'r1_integrator_probe.json')
if os.path.exists(p1):
    R = json.load(open(p1)); out['r1'] = {'houses': [h['house'] for h in R]}
    for cname in R[0]['conds']:
        for v in ('V0_impl', 'V1_octomap', 'V2_probe'):
            rows = [h['conds'][cname][v] for h in R]
            keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)]
            m = {k: r(np.mean([x[k] for x in rows]), 4) for k in keys}
            m['B4_relay_median'] = r(np.median([x['B4_relay'] for x in rows]), 0)
            out['r1'][f'{cname}|{v}'] = m
        if 'B5' in R[0]['conds'][cname]:
            for v in R[0]['conds'][cname]['B5']:
                base = v.replace('_ge', '')
                cfgs = R[0]['conds'][cname]['B5'][v].keys()
                pts = [(float(np.median([h['conds'][cname]['B5'][v][c][0] for h in R])), float(np.mean([h['conds'][cname]['B5'][v][c][1] for h in R]))) for c in cfgs]
                q = 0.25 * float(np.median([h['conds'][cname][base]['B4_relay'] for h in R]))
                b4f = float(np.mean([h['conds'][cname][base]['R8_furn_sum'] for h in R]))
                hv = hull_value(pts, q)
                out['r1'][f'{cname}|B5|{v}'] = dict(q=r(q, 0), hull_at_q=r(hv, 4), best=r(max(y for _, y in pts), 4), B4_R8_furn=r(b4f, 4),
                                                    b5_arm_pass=(hv is None or hv <= b4f + .02),
                                                    per_house=[dict(house=h['house'], hull=r(hull_value(list(h['conds'][cname]['B5'][v].values()), 0.25 * h['conds'][cname][base]['B4_relay']), 4),
                                                                    B4=r(h['conds'][cname][base]['R8_furn_sum'], 4)) for h in R])
p1b = os.path.join(HERE, 'r1b_perbeam_probe.json')
if os.path.exists(p1b):
    R = json.load(open(p1b)); out['r1b'] = {'houses': [h['house'] for h in R]}
    for cname in R[0]['conds']:
        for v in R[0]['conds'][cname]:
            rows = [h['conds'][cname][v] for h in R]
            keys = [k for k in rows[0] if rows[0][k] is not None]
            out['r1b'][f'{cname}|{v}'] = {k: r(np.mean([x[k] for x in rows]), 4) for k in keys}
p2 = os.path.join(HERE, 'r2_b5_box_probe.json')
if os.path.exists(p2):
    R = json.load(open(p2)); out['r2'] = {'houses': [h['house'] for h in R]}
    q = 0.25 * float(np.median([h['B4_relay'] for h in R])); b4f = float(np.mean([h['B4_R8_furn'] for h in R]))
    out['r2']['B4_R8_furn'] = r(b4f, 4); out['r2']['q'] = r(q, 0)
    for fam in ('cc', 'box'):
        cfgs = R[0]['fam'][fam].keys()
        pts = {c: (float(np.median([h['fam'][fam][c][0] for h in R])), float(np.mean([h['fam'][fam][c][1] for h in R]))) for c in cfgs}
        hv = hull_value(list(pts.values()), q)
        best = max(pts.items(), key=lambda kv: kv[1][1])
        out['r2'][fam] = dict(hull_at_q=r(hv, 4), best=r(best[1][1], 4), best_cfg=best[0], best_rate=r(best[1][0], 0),
                              b5_arm_pass=(hv is None or hv <= b4f + .02))
    allp = [(float(np.median([h['fam'][f][c][0] for h in R])), float(np.mean([h['fam'][f][c][1] for h in R]))) for f in ('cc', 'box') for c in R[0]['fam'][f]]
    out['r2']['union_hull_at_q'] = r(hull_value(allp, q), 4)
json.dump(out, open(os.path.join(HERE, 'probe_summary.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))

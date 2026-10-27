"""p3_aggregate.py (A03): aggregate whatever p3 houses are cached (same logic as p3_box_codec.py's main; LOO priors over the cached
houses). Usage: python -B p3_aggregate.py  -> prints the summary, writes p3_box_codec_all.json when all 24 houses are cached."""
import json, os, sys
import numpy as np
import p3_box_codec as P

hs = [h for h in P.names() if os.path.exists(os.path.join(P.CACHE, h + '.json'))]
H = []
for h in hs:
    x = json.load(open(os.path.join(P.CACHE, h + '.json')))
    for r in x['k0'] + x['r8']:
        r['scores'] = {int(k): v for k, v in r['scores'].items()}
    H.append(x)
rows = []
for h in H:
    tabs = P.ent_tables(H, h['house'])
    for r in h['k0']:
        for m in (P.MS if r['variant'] == 'plan+box' else (1,)):
            rows.append(dict(map='K0', house=h['house'], variant=r['variant'], m=m, bytes=r['plan_bytes'] + (P.box_bits(r['boxes'], tabs.get(r['variant']), m) + 16 + 44) / 8,
                             n_boxes=sum(1 for b in r['boxes'] if b[6] >= m), exact=r['exact'] if m == 1 else None, **r['scores'][m]))
    for r in h['r8']:
        for _rep in r['agents']:
            for m in (P.MS if r['variant'] == 'plan+box' else (1,)):
                rows.append(dict(map='R8', house=h['house'], variant=r['variant'], m=m, agents=len(r['agents']),
                                 bytes=r['plan_bytes'] + (P.box_bits(r['boxes'], tabs.get(r['variant']), m) + 16 + 44 * len(r['agents'])) / 8,
                                 n_boxes=sum(1 for b in r['boxes'] if b[6] >= m), exact=r['exact'] if m == 1 else None, **r['scores'][m]))
            relay = r['relay_plan_bytes'] + sum((P.box_bits(rr['boxes'], tabs.get(r['variant']), 1) + 16 + 44) / 8 for rr in h['k0'] if rr['variant'] == r['variant'] and rr['agent'] in r['agents'])
            rows.append(dict(map='R8relay', house=h['house'], variant=r['variant'], m=1, agents=len(r['agents']), bytes=relay, exact=True, n_boxes=r['relay_boxes'], surface_f1=1., var_occ_f1=1., furn_f1=1.))
            rows.append(dict(map='R8naive', house=h['house'], variant=r['variant'], m=1, agents=len(r['agents']), bytes=relay, exact=r['naive_exact'], n_boxes=r['relay_boxes'], **r['naive']))
S = {}
for key in sorted({(r['map'], r['variant'], r['m']) for r in rows}):
    rr = [r for r in rows if (r['map'], r['variant'], r['m']) == key]
    mean = lambda k: (float(np.mean([r[k] for r in rr if r.get(k) is not None])) if any(r.get(k) is not None for r in rr) else None)
    S['%s|%s|m%d' % key] = dict(n=len(rr), bytes_median=float(np.median([r['bytes'] for r in rr])), bytes_p90=float(np.percentile([r['bytes'] for r in rr], 90)),
                                boxes_median=float(np.median([r['n_boxes'] for r in rr])), exact=sum(1 for r in rr if r['exact']), checked=sum(1 for r in rr if r['exact'] is not None),
                                surface_f1=mean('surface_f1'), var_occ_f1=mean('var_occ_f1'), furn_f1=mean('furn_f1'))
print('houses', len(H))
for k, v in S.items():
    f = lambda x: 'n/a' if x is None else f'{x:.4f}'
    print(f"{k:24s} n {v['n']:3d} bytes {v['bytes_median']:7.0f} p90 {v['bytes_p90']:6.0f} boxes {v['boxes_median']:5.0f} exact {v['exact']}/{v['checked']} surf {f(v['surface_f1'])} var {f(v['var_occ_f1'])} furn {f(v['furn_f1'])}")
if len(H) == len(P.names()):
    json.dump(dict(summary=S, rows=rows), open(os.path.join(P.HERE, 'p3_box_codec_all.json'), 'w'), indent=1)

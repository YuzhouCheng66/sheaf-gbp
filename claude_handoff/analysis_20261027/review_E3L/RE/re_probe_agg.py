import json, numpy as np
rows = json.load(open('probe_a.json')) + json.load(open('probe_b.json'))
out = {}
def med(v): return float(np.median(v))
C = np.array([r['Qinf']['CTRL'] for r in rows])
for an in ('Qinf', 'Q3', 'VOTE'):
    d = {}
    d['CTRL_ratio'] = med([r[an]['CTRL'] for r in rows] / C)
    fr = [r[an]['FR1'] for r in rows]
    d['FR1_ratio'] = med(np.array([f['total'] for f in fr]) / C)
    d['FR1_mask_plus_lam_ratio'] = med(np.array([f['mask'] + f['lam'] for f in fr]) / C)
    d['FR1_h_ratio'] = med(np.array([f['h'] for f in fr]) / C)
    d['FR1_src_per_agent_B'] = med([f['src_per_agent'] for f in fr])
    d['FR1_total_over_src'] = med([f['total'] / f['src_per_agent'] for f in fr])
    if 'S2D' in rows[0][an]:
        s = [r[an]['S2D'] for r in rows]; ss = [r[an]['S2D_S'] for r in rows]
        d['S2D_ratio'] = med(np.array([x['total'] for x in s]) / C)
        d['S2D_mask_plus_lam_ratio'] = med(np.array([x['mask'] + x['lam'] for x in s]) / C)
        d['S2D_S_only_ratio'] = med(np.array([x['total'] for x in ss]) / C)
        d['S2D_src_per_agent_B'] = med([x['src_per_agent'] for x in s])
        d['S2D_total_over_src'] = med([x['total'] / x['src_per_agent'] for x in s])
    out[an] = {k: round(v, 4) for k, v in d.items()}
out['houses'] = [r['file'] for r in rows]; out['n_agents'] = [r['n'] for r in rows]; out['directed'] = [r['directed'] for r in rows]
out['CTRL_median_B'] = med(C)
json.dump(out, open('probe_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

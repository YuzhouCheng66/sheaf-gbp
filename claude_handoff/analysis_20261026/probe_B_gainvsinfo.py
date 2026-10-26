"""probe_B_gainvsinfo (analyst B, 2026-10-26, report-only): did the 16x16 -> 32x32 rung help the houses whose own maps carry MORE information
per evidenced coordinate (the information-limit prediction), or the houses with more furniture, or the small / big houses?
Per-house delta (arm - control, same seed) from the existing VAL diag JSONs vs per-house quantities from probe_B_info.py (info.json)."""
import json, sys
import numpy as np
R = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/runs'
info = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'info.json'))['summary']['per_house']
I = {(h['rooms'], h['index']): h for h in info}
out = {}
for name, a, c in (('L1-D2_16k', 'v429-ladder/diag-l1', 'v429-ladder/diag-control'), ('3SX-3ST_2cyc', 'v429-stack/diag-a', 'v429-stack/diag-control'),
                   ('L3-D2_16k', 'v429-ladder/diag-l3', 'v429-ladder/diag-control')):
    for s in ('s342101', 's342202'):
        A = {(r['rooms'], r['index']): r for r in json.load(open(f'{R}/{a}-{s}/diag.json'))['per_scene']}
        C = {(r['rooms'], r['index']): r for r in json.load(open(f'{R}/{c}-{s}/diag.json'))['per_scene']}
        keys = sorted(set(A) & set(C) & set(I))
        bits = np.array([I[k]['bits_per_ev32'] for k in keys]); obj = np.array([I[k]['obj_share'] for k in keys]); L = np.array([I[k]['L'] for k in keys])
        row = {}
        for m in ('K0_surface', 'K8_surface', 'K8_var', 'K8_furn'):
            d = np.array([A[k][m] - C[k][m] for k in keys])
            def pc(x, z):
                Z = np.c_[np.ones_like(z), z]; rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]; rd = d - Z @ np.linalg.lstsq(Z, d, rcond=None)[0]; return float(np.corrcoef(rx, rd)[0, 1])
            hi = bits >= np.median(bits)
            row[m] = dict(r_bits=float(np.corrcoef(bits, d)[0, 1]), r_obj=float(np.corrcoef(obj, d)[0, 1]), r_L=float(np.corrcoef(L, d)[0, 1]),
                          pr_bits_given_L=pc(bits, L), delta_hi_bits=float(d[hi].mean()), delta_lo_bits=float(d[~hi].mean()))
        out[f'{name}_{s}'] = row
        print(name, s, ' '.join('%s: r_bits %+.2f (|L %+.2f) r_obj %+.2f r_L %+.2f d hi/lo bits %+.3f/%+.3f' % (m, v['r_bits'], v['pr_bits_given_L'], v['r_obj'], v['r_L'], v['delta_hi_bits'], v['delta_lo_bits']) for m, v in row.items()))
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else 'gainvsinfo.json', 'w'), indent=1)

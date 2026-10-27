"""rb_s0_demo6.py (reviewer R-B, 2026-10-27): same-panel references on the 6 VAL demo houses (r1_i0 .. r3_i1, a subset of A03's
24 VAL houses, same agent order -- checked against rate_baselines.json gzip/ctx per agent).
  S0  = A03 p9 rows (plan + consistent boxes, label-free), poses re-priced to float32 (206 bit per pose triple, as p9b_exact_pose_bytes);
        per house = sum over directed edges of the sender-side-union description (each directed edge once, NOT cavity-conditioned).
  ray = A01 probe_hyper 'base_pretrained' (ray-order events, 6,561 contexts, counts from the other 5 houses + adaptive) + 206/8 B pose,
        paired per agent with S0 (own maps only).
  gzip / ctx relay and gzip flood per house = A10 a10_probe.out (6-house means, identical to A09 p6).
Usage: python -B rb_s0_demo6.py"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.abspath(os.path.join(HERE, '..', '..'))
DEMO = ['r1_i0', 'r1_i1', 'r2_i0', 'r2_i1', 'r3_i0', 'r3_i1']
B = 206
p9 = json.load(open(os.path.join(AN, 'probes', 'A03', 'p9_message_bytes.json')))['rows']
hy = json.load(open(os.path.join(AN, 'probes', 'A01', 'probe_hyper.json')))['rows']
rb = json.load(open(os.path.join(AN, 'rate_baselines.json')))['raw']['own']
out = {}
for kind in ('own', 'edge', 'union'):
    v = np.array([r['desc'] + r['agents'] * (B - 44) / 8 for r in p9 if r['kind'] == kind and r['house'] in DEMO])
    out[f'S0_{kind}'] = dict(n=int(v.size), median=float(np.median(v)), mean=float(v.mean()))
tot = [sum(r['desc'] + r['agents'] * (B - 44) / 8 for r in p9 if r['house'] == h and r['kind'] == 'edge') for h in DEMO]
out['S0_house'] = dict(per_house=tot, median=float(np.median(tot)), mean=float(np.mean(tot)))
s0own = {h: [r['desc'] + (B - 44) / 8 for r in p9 if r['kind'] == 'own' and r['house'] == h] for h in DEMO}
gz = {h: [r['gzip'] for r in p9 if r['kind'] == 'own' and r['house'] == h] for h in DEMO}
assert all(gz[h] == [r['gzip'] for r in rb if r['house'] == h] for h in DEMO), 'agent order differs'
ray = np.array([r['base_pretrained'] + B / 8 for r in hy if r['kind'] == 'own'])
s0p = np.array([s0own[r['house']][r['agent']] for r in hy if r['kind'] == 'own'])
hyp = np.array([r['hyper_pretrained'] + B / 8 for r in hy if r['kind'] == 'own'])
out['ray_own_f32'] = dict(median=float(np.median(ray)), paired_ratio_ray_over_S0_median=float(np.median(ray / s0p)),
                         share_ray_cheaper=float(np.mean(ray < s0p)))
out['ray_hyper_own_f32'] = dict(median=float(np.median(hyp)), paired_ratio_over_S0_median=float(np.median(hyp / s0p)))
out['S0_bits_per_evidenced_col'] = dict(G32=float(np.median(s0p) * 8 / 184.5), G64=float(np.median(s0p) * 8 / 658.0),
                                        note='median S0 own bytes (float32 pose incl.) x 8 / median evidenced columns (A01 probe_wire)')
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, 'rb_s0_demo6.json'), 'w'), indent=1)

"""rh_agg.py (R-H): per-house totals of the rh_codes.py coders, medians per (ckpt, noise seed), ratios to the control median
(summary.json control_bytes of the same cell), reconciliation of O7 with summary.json bytes_median.
usage: python -B rh_agg.py codes.json summary.json [out.json]"""
import json
import sys

import numpy as np

S = json.load(open(sys.argv[2]))
C = json.load(open(sys.argv[1]))
cells = S['decision']['cells']
out = {}
by = {}
for r in C['recs']:
    assert not r.get('failed'), r
    by.setdefault((r['seed'], r['file'].split('_s')[-1][:-4]), []).append(r)


def tot(r, key, fallback=None):
    v = 0.0
    for m in r['msgs']:
        x = m.get(key)
        if x is None:
            x = m[fallback]
        v += x
    return v


SCHEMES = {
    'O7 (frozen)': lambda r: tot(r, 'o7'),
    'O7 k part (mask + k)': lambda r: tot(r, 'k_o7'),
    'O7 s part': lambda r: tot(r, 's_o7'),
    'P1: K1 + D1 per message (no side info)': lambda r: tot(r, 'KD'),
    'P1 K part': lambda r: tot(r, 'K1'),
    'P1 D part': lambda r: tot(r, 'D1'),
    'P2: P1, second msg | reverse msg (practical)': lambda r: tot(r, 'KD_rev', 'KD'),
    'P2 K part': lambda r: tot(r, 'K_rev', 'K1'),
    'P2 D part': lambda r: tot(r, 'D_rev', 'D1'),
    'P2 first (up) msgs': lambda r: sum(m['KD'] for m in r['msgs'] if not m['second']),
    'P2 second (down) msgs': lambda r: sum(m['KD_rev'] for m in r['msgs'] if m['second']),
    'SW: every msg | receiver partial belief (decoder-only side info)': lambda r: tot(r, 'KD_bel'),
    'SW own: every msg | receiver own map': lambda r: tot(r, 'KD_own'),
    'ref: total T once (K1+D1, m = n)': lambda r: r['total_once'],
    'ref: total T once, K part': lambda r: r['total_once_k'],
    'ref: n x T once': lambda r: r['n'] * r['total_once'],
    'ref: sum of self maps (K1+D1)': lambda r: sum(r['self']),
    'ref: sum of self maps (O7)': lambda r: sum(r['self_o7']),
}
for (ck, seed), rs in sorted(by.items()):          # (checkpoint seed, noise seed)
    cell = cells[f'{ck}|{seed}']
    cb = cell['control_bytes']
    o = dict(houses=len(rs), control_bytes=cb, summary_bytes_median=cell['bytes_median'])
    for name, f in SCHEMES.items():
        v = np.array([f(r) for r in rs])
        o[name] = dict(median=float(np.median(v)), ratio=float(np.median(v) / cb), mean=float(v.mean()))
    o['o7_reconcile_abs'] = abs(o['O7 (frozen)']['median'] - cell['bytes_median'])
    o['n_agents'] = sorted({r['n'] for r in rs})
    zf = np.sum([r['zfrac'] for r in rs], 0); o['z_frac'] = (zf / zf.sum()).tolist()
    out[f'{ck}|{seed}'] = o
names = list(SCHEMES)
print('%-66s' % 'scheme (median per house, bytes; ratio to control)' + ''.join('%22s' % k for k in out))
for nm in names:
    print('%-66s' % nm + ''.join('%12.0f (%.3f)    ' % (out[k][nm]['median'], out[k][nm]['ratio']) for k in out))
print('O7 reconcile |median - summary|:', {k: round(v['o7_reconcile_abs'], 3) for k, v in out.items()})
print('agents per house:', {k: v['n_agents'] for k, v in out.items()}, 'z frac -/0/+:', {k: [round(x, 3) for x in v['z_frac']] for k, v in out.items()})
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], 'w'), indent=1)

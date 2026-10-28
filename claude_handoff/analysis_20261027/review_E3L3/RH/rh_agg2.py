"""rh_agg2.py (R-H): combine rh_codes.py (O7 / P1 / P2 per message), rh_codes2.py (rich ctx, adaptive and static LOO table) and
rh_readout.py (report-only readout downlink) into per-house totals; medians per (ckpt, noise) cell; ratio = median / control median.
usage: python -B rh_agg2.py codes_a05.json codes2.json readout_s1.json,readout_s2.json summary.json out.json"""
import json
import sys

import numpy as np

A = json.load(open(sys.argv[1]))['recs']
B = json.load(open(sys.argv[2]))['cells']
RO = [r for p in sys.argv[3].split(',') for r in json.load(open(p))]
S = json.load(open(sys.argv[4]))['decision']['cells']
FLAG = 1 / 8                                   # one bit per message to pick a coder (best-of)


def noise_of(fname):
    return fname.split('_s')[-1][:-4]


a_by = {(r['seed'], noise_of(r['file']), r['file']): r for r in A}
ro_by = {(r['seed'], noise_of(r['file']), r['file']): r for r in RO}
for r in RO:
    assert not r.get('failed') and r['same_belief'], r.get('error')
out = {}
for cell, houses in sorted(B.items()):
    ck, noise = cell.split('|')
    cb = S[cell]['control_bytes']
    T = {}
    for h in houses:
        a = a_by[(ck, noise, h['file'])]; ro = ro_by[(ck, noise, h['file'])]
        rows = {(x['set'], x['id']): x for x in h['rows']}
        msgs = a['msgs']
        v = {}
        v['O7'] = sum(m['o7'] for m in msgs)
        v['P1 (K1+D1, adaptive)'] = sum(m['KD'] for m in msgs)
        v['P2 (P1, 2nd | reverse)'] = sum(m.get('KD_rev', m['KD']) for m in msgs)
        best1 = 0.0; best_st = 0.0; st_plain = 0.0; st_rev = 0.0; up = 0.0; down = 0.0; up_best = 0.0
        st_k = st_d = 0.0
        for m in msgs:
            pid = (f"plain", f"{m['s']}>{m['r']}|{int(m['second'])}")
            x = rows[pid]
            sp = x['K_static'] + x['D_static'] + 4
            st_plain += sp
            cands = [m['o7'], m['KD'], x['K_adapt'] + x['D_adapt'] + 4]
            if m['second']:
                cands.append(m['KD_rev'])
                xr = rows[('rev', f"{m['s']}>{m['r']}")]
                cands.append(xr['K_adapt'] + xr['D_adapt'] + 4)
                sr = xr['K_static'] + xr['D_static'] + 4
                st_rev += min(sp, sr) + FLAG
                kk, dd = (xr['K_static'], xr['D_static']) if sr < sp else (x['K_static'], x['D_static'])
                down += min(cands) + FLAG
            else:
                st_rev += sp + FLAG
                kk, dd = x['K_static'], x['D_static']
                up += min(cands) + FLAG; up_best += min(cands) + FLAG
            st_k += kk; st_d += dd
            best1 += min(cands) + FLAG
            best_st += min(cands + [sp] + ([sr] if m['second'] else [])) + FLAG
        v['BEST adaptive (per msg min of O7/P1/P2/rich, +1 bit)'] = best1
        v['  of which first (up) msgs'] = up
        v['  of which second (down) msgs'] = down
        v['STATIC LOO table (rich ctx; 2nd msg min(plain, |rev))'] = st_rev
        v['  static K part'] = st_k
        v['  static D part'] = st_d
        v['BEST incl. static table'] = best_st
        tr = rows[('T', 'T')]
        v['ref T once (rich, static)'] = tr['K_static'] + tr['D_static'] + 4
        v['ref T once K part (static)'] = tr['K_static']
        v['ref T once (min adaptive)'] = min(a['total_once'], tr['K_adapt'] + tr['D_adapt'] + 4)
        fl = [rows[('floor', f'floor{i}')] for i in range(h['n'])]
        v['FLOOR sum_a cost(T | own_a), static'] = sum(x['K_static'] + x['D_static'] + 4 for x in fl)
        v['  floor K part'] = sum(x['K_static'] for x in fl)
        v['  floor D part'] = sum(x['D_static'] for x in fl)
        v['FLOOR, adaptive'] = sum(x['K_adapt'] + x['D_adapt'] + 4 for x in fl)
        v['ref self maps (O7)'] = sum(a['self_o7'])
        v['RO (report only, violates C5): best up + readout down (cond)'] = up_best + sum(min(d['plain'], d['cond']) for d in ro['down'])
        v['  readout down only'] = sum(min(d['plain'], d['cond']) for d in ro['down'])
        for k2, x in v.items():
            T.setdefault(k2, []).append(x)
    o = dict(control_bytes=cb, n=len(houses))
    for k2, x in T.items():
        x = np.array(x); o[k2] = dict(median=float(np.median(x)), ratio=float(np.median(x) / cb), min_ratio=float(x.min() / cb), max_ratio=float(x.max() / cb))
    out[cell] = o
keys = [k for k in out[next(iter(out))] if isinstance(out[next(iter(out))][k], dict)]
print('%-62s' % 'per-house median bytes (ratio to control)' + ''.join('%22s' % c for c in out))
for k in keys:
    print('%-62s' % k + ''.join('%11.0f (%.3f)    ' % (out[c][k]['median'], out[c][k]['ratio']) for c in out))
json.dump(out, open(sys.argv[5], 'w'), indent=1)

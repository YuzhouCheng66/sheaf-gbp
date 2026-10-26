"""probe_B_sizegain (analyst B, 2026-10-26): how does the code-ladder gain depend on house size L? Reads existing VAL diag JSONs only.
Pairs (same 24 VAL houses, same seed): 16k ladder  L1 (32x32x1) vs control D2 (16x16x1)   runs/v429-ladder/diag-{l1,control}-s*
                                         2-cycle     3S-X (32x32x1) vs 3S-T D2 (16x16x1)      runs/v429-stack/diag-{a,control}-s*
                                         16k bits    L2 (16x16x1 @16+16), L3 (16x16x2 @4+4) vs control
Also tests whether F1 is a function of the coordinate pitch p = L / B alone (collapse test) and extrapolates to B = 64."""
import json, sys
import numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/runs'
def load(p):
    d = json.load(open(p)); return {r['index'] * 10 + r['rooms']: r for r in d['per_scene']}
def ols(x, y):
    A = np.c_[np.ones_like(x), x]; b, *_ = np.linalg.lstsq(A, y, rcond=None); r = np.corrcoef(x, y)[0, 1]; return b, r
out = {}
pairs = {'ladder16k_L1-D2': ('v429-ladder/diag-l1', 'v429-ladder/diag-control', 32, 16), 'ladder16k_L2-D2': ('v429-ladder/diag-l2', 'v429-ladder/diag-control', 16, 16),
         'ladder16k_L3-D2': ('v429-ladder/diag-l3', 'v429-ladder/diag-control', 16, 16), 'twocycle_3SX-3ST': ('v429-stack/diag-a', 'v429-stack/diag-control', 32, 16),
         'twocycle_3SY-3SX': ('v430-stackdata/diag-a', 'v430-stackdata/diag-control', 32, 32)}
pitch_rows = []
for name, (a, c, Ba, Bc) in pairs.items():
    for s in ('s342101', 's342202'):
        A = load(f'{R}/{a}-{s}/diag.json'); C = load(f'{R}/{c}-{s}/diag.json'); keys = sorted(set(A) & set(C))
        L = np.array([A[k]['chart_m'] for k in keys]); rooms = np.array([A[k]['rooms'] for k in keys])
        res = {}
        for m in ('K0_surface', 'K8_surface', 'K8_var', 'K8_furn'):
            ya = np.array([A[k][m] for k in keys]); yc = np.array([C[k][m] for k in keys]); dlt = ya - yc
            (b0, b1), r = ols(L, dlt)
            small, big = L <= 13, L >= 17
            res[m] = dict(arm=float(ya.mean()), ctl=float(yc.mean()), delta=float(dlt.mean()), slope_delta_per_m=float(b1), r_delta_L=float(r),
                          delta_small_le13=float(dlt[small].mean()), delta_big_ge17=float(dlt[big].mean()),
                          slope_arm_per_m=float(ols(L, ya)[0][1]), slope_ctl_per_m=float(ols(L, yc)[0][1]))
        res['n'] = len(keys); res['n_small'] = int((L <= 13).sum()); res['n_big'] = int((L >= 17).sum())
        out[f'{name}_{s}'] = res
        if name in ('ladder16k_L1-D2',):
            for k in keys:
                pitch_rows.append((A[k]['chart_m'] / Ba, A[k]['K0_surface'], A[k]['K8_surface'], 32, A[k]['rooms'], s))
                pitch_rows.append((C[k]['chart_m'] / Bc, C[k]['K0_surface'], C[k]['K8_surface'], 16, C[k]['rooms'], s))
# collapse test: in the overlapping pitch band, compare F1 of the 16-grid (small houses) and the 32-grid (big houses)
P = np.array([r[0] for r in pitch_rows]); K0 = np.array([r[1] for r in pitch_rows]); K8 = np.array([r[2] for r in pitch_rows]); B = np.array([r[3] for r in pitch_rows])
band = (P >= .45) & (P <= .66)
out['collapse_overlap_band_0.45-0.66m'] = dict(n16=int((band & (B == 16)).sum()), n32=int((band & (B == 32)).sum()),
    K0_16=float(K0[band & (B == 16)].mean()), K0_32=float(K0[band & (B == 32)].mean()), K8_16=float(K8[band & (B == 16)].mean()), K8_32=float(K8[band & (B == 32)].mean()))
(b0, b1), r = ols(P, K0); (c0, c1), r8 = ols(P, K8)
out['pooled_fit_vs_pitch'] = dict(K0_intercept=float(b0), K0_slope_per_m=float(b1), r_K0=float(r), K8_intercept=float(c0), K8_slope_per_m=float(c1), r_K8=float(r8))
# per-house 64 extrapolation (16k recipe): delta_64(L) = F1(L/64) - F1(L/32) under the pooled linear pitch law
L32 = P[B == 32] * 32
pred = dict(K0_gain_mean=float(-b1 * (L32 / 64).mean()), K8_gain_mean=float(-c1 * (L32 / 64).mean()),
            K8_gain_small_le13=float(-c1 * (L32[L32 <= 13] / 64).mean()), K8_gain_big_ge17=float(-c1 * (L32[L32 >= 17] / 64).mean()))
out['naive_pitch_law_prediction_3SZ_16k'] = pred
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else 'probe_B_sizegain.json', 'w'), indent=1)
for k, v in out.items():
    if isinstance(v, dict) and 'K8_surface' in v:
        print(k, 'n', v['n'], 'small/big', v['n_small'], v['n_big'])
        for m in ('K0_surface', 'K8_surface', 'K8_var', 'K8_furn'):
            x = v[m]; print('   %-11s arm %.3f ctl %.3f d %+.4f  d<=13 %+.4f d>=17 %+.4f slope_d %+.4f/m r %.2f | slope arm %+.4f ctl %+.4f /m' % (m, x['arm'], x['ctl'], x['delta'], x['delta_small_le13'], x['delta_big_ge17'], x['slope_delta_per_m'], x['r_delta_L'], x['slope_arm_per_m'], x['slope_ctl_per_m']))
    else:
        print(k, v)

"""A06 furniture-F1 ceilings and sensitivities (Claude A06, 2026-10-27). CPU only; 6 VAL demo houses x 8 agents; K0 (own) and R8 targets.
(1) Perfect-but-reduced targets: drop thin_xy / thin_z / all thin furniture parts (they become free), shift every furniture component by one
voxel (random horizontal direction per component), thicken furniture by one voxel into known free space (6-nbhd / horizontal / vertical).
(2) Classical block down-up of the target (rate_baselines_lib.down_up) at several taus, with bytes (ctx of the coarse grid) and its own
furniture error anatomy (FN/FP, thin/bulk recall). Scored with the training metric (furniture-region F1 = mean over agents)."""
import json, os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a06_common import houses, scene, metrics_rows, mean, dil6, RB  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(342101)


def shift_components(t, obj, region):
    """move every furniture component's target-occupied voxels by one voxel in a random horizontal direction"""
    lab, nl = ndimage.label(obj, structure=np.ones((3, 3, 3)))
    P = t.copy(); furn = (t == 1) & obj & region
    P[furn] = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    choice = rng.integers(0, 4, nl + 1)
    for d in range(4):
        sel = furn & (choice[lab] == d)
        dx, dy = dirs[d]
        moved = np.roll(np.roll(sel, dx, 0), dy, 1)
        P[moved & (t >= 0)] = 1                                  # lands on known voxels: free ones become false positives
    return P


def thicken(t, region, mode):
    to = (t == 1) & region
    if mode == '6':
        g = dil6(to)
    else:
        g = to.copy()
        if mode == 'h':
            g[1:] |= to[:-1]; g[:-1] |= to[1:]; g[:, 1:] |= to[:, :-1]; g[:, :-1] |= to[:, 1:]
        else:
            g[:, :, 1:] |= to[:, :, :-1]; g[:, :, :-1] |= to[:, :, 1:]
    P = t.copy(); P[g & (t == 0) & region] = 1
    return P


VARIANTS = ['drop_thin_xy', 'drop_thin_z', 'drop_thin_all', 'shift1', 'thicken6', 'thicken_h', 'thicken_v']
DOWN = [((2, 2, 1), .25), ((2, 2, 1), .5), ((2, 2, 1), .75), ((1, 1, 2), .5), ((2, 2, 2), .5), ((1, 1, 4), .5), ((4, 4, 1), .5)]
res = {str(h): dict(var={v: {} for v in VARIANTS}, down={}) for h in (0, 8)}
for path in houses():
    S = scene(path); m = S['m']; n = m['agents']; region = S['region']
    for hop in (0, 8):
        T = S['tgt'][hop]; R = res[str(hop)]
        for v in VARIANTS:
            P = []
            for a in range(n):
                t = T[a].copy()
                if v.startswith('drop'):
                    cls = {'drop_thin_xy': S['thin_xy'], 'drop_thin_z': S['thin_z'], 'drop_thin_all': S['thin']}[v]
                    t[(t == 1) & cls & region] = 0; P.append(t)
                elif v == 'shift1':
                    P.append(shift_components(t, S['obj'], region))
                else:
                    P.append(thicken(t, region, v[-1]))
            rows = metrics_rows(np.stack(P), T, S)
            for k in ('surface_f1', 'var_occ_f1', 'furn_f1'):
                R['var'][v].setdefault(k, []).extend([r[k] for r in rows])
        for f, tau in DOWN:
            key = f'{f[0]}x{f[1]}x{f[2]}_tau{tau}'
            ups, by = [], []
            an = dict(tp=0, fp=0, fn=0, thin_xy=[0, 0], thin_z=[0, 0], bulk=[0, 0], fp_adj=0)
            for a in range(n):
                t = T[a]; coarse, up = RB.down_up(t, f, tau); ups.append(up); by.append(RB.ctx_bytes(coarse))
                to = (t == 1) & region; po = (up == 1) & (t >= 0) & region
                an['tp'] += int((to & po).sum()); an['fp'] += int((po & ~to).sum()); an['fn'] += int((to & ~po).sum())
                for nm, cm in (('thin_xy', S['thin_xy']), ('thin_z', S['thin_z']), ('bulk', ~S['thin'])):
                    an[nm][0] += int((to & po & cm).sum()); an[nm][1] += int((to & cm).sum())
                an['fp_adj'] += int((po & ~to & dil6(to)).sum())
            rows = metrics_rows(np.stack(ups), T, S)
            d = R['down'].setdefault(key, dict(bytes=[], anat=dict(tp=0, fp=0, fn=0, thin_xy=[0, 0], thin_z=[0, 0], bulk=[0, 0], fp_adj=0)))
            d['bytes'] += by
            for k in ('surface_f1', 'var_occ_f1', 'furn_f1'):
                d.setdefault(k, []).extend([r[k] for r in rows])
            for k in ('tp', 'fp', 'fn', 'fp_adj'):
                d['anat'][k] += an[k]
            for k in ('thin_xy', 'thin_z', 'bulk'):
                d['anat'][k][0] += an[k][0]; d['anat'][k][1] += an[k][1]
    print(m['house'], 'done', flush=True)

summary = {}
for hop, R in res.items():
    summary[hop] = dict(var={v: {k: mean(x) for k, x in d.items()} for v, d in R['var'].items()},
                        down={k: dict(bytes_median=float(np.median(d['bytes'])), bytes_mean=float(np.mean(d['bytes'])), **{kk: mean(d[kk]) for kk in ('surface_f1', 'var_occ_f1', 'furn_f1')},
                                      recall=d['anat']['tp'] / max(d['anat']['tp'] + d['anat']['fn'], 1), precision=d['anat']['tp'] / max(d['anat']['tp'] + d['anat']['fp'], 1),
                                      fn_over_fp=d['anat']['fn'] / max(d['anat']['fp'], 1), fp_adjacent_share=d['anat']['fp_adj'] / max(d['anat']['fp'], 1),
                                      rec_thin_xy=d['anat']['thin_xy'][0] / max(d['anat']['thin_xy'][1], 1), rec_thin_z=d['anat']['thin_z'][0] / max(d['anat']['thin_z'][1], 1),
                                      rec_bulk=d['anat']['bulk'][0] / max(d['anat']['bulk'][1], 1)) for k, d in R['down'].items()})
json.dump(summary, open(os.path.join(OUT, 'ceilings.json'), 'w'), indent=1)
for hop, s in summary.items():
    print(f'\n=== hop {hop}')
    for v, d in s['var'].items():
        print(f"  {v:14s} surf {d['surface_f1']:.3f} var {d['var_occ_f1']:.3f} furn {d['furn_f1']:.3f}")
    for k, d in s['down'].items():
        print(f"  {k:16s} {d['bytes_median']:6.0f} B  surf {d['surface_f1']:.3f} var {d['var_occ_f1']:.3f} furn {d['furn_f1']:.3f}  R {d['recall']:.3f} P {d['precision']:.3f} FN/FP {d['fn_over_fp']:.2f} FPadj {d['fp_adjacent_share']:.2f}  rec thin_xy {d['rec_thin_xy']:.2f} thin_z {d['rec_thin_z']:.2f} bulk {d['rec_bulk']:.2f}")

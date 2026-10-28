"""Reviewer's independent recompute for G4-E6L (D8-70) on ONE task R-I has already seen: val_r1_i0_s1 (VAL noise seed 1, cam).
Map side: the FROZEN G4-E6 pipeline itself (g4_island_compose.run_task, arms reduced to cam_nf noreg / cam noreg / cam e6, no truth-gauge
row), with OI.readouts wrapped to capture the aligned maps it scores. No code from g4_e6_e3l3 / g4_h1_e3l3 on the reference side.
E3-L3 side: the frozen formal evaluation function g4_e3l3.eval_arm on those aligned maps (learned w, R8 threshold, chunk 2).
Compared with: (1) g4/e6/jobs/val_r1_i0_s1.json aligned rows (B4 / B6 / NET), (2) the reviewed script's smoke record
g4/e6l_smoke/jobs/val_r1_i0_s1.json (E3L3 rows, written by g4_e6_e3l3.py eb083676), (3) the per-house d of the smoke summary.
run: CUDA_VISIBLE_DEVICES= python -B rv_e6l_val.py"""
import os, sys, json, time
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'scripts', 'claude_v342'))
sys.path.insert(0, SCR)
import numpy as np
import g4_island_compose as OI     # frozen G4-E6
import g4_e3l3 as E3               # frozen E3-L3
import torch
torch.set_num_threads(1)
G = OI.G; R = E3.R
NAME = 'val_r1_i0_s1'
A27 = OI.A27
INP = os.path.join(A27, 'review_E3R', 'inputs')
KEYS = ('furn_f1', 'furn_f1_tol1', 'free_recall', 'surface_f1', 'unknown_known_fpr', 'furn_tp', 'furn_fp', 'furn_fn')
out = dict(task=NAME, t0=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


def jr(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


CAP = []
_orig = OI.readouts


def hooked(maps, F, R8T, H, RO, rstats, preds=None):
    res = _orig(maps, F, R8T, H, RO, rstats, preds)
    if preds is None:                                  # eval_arm6: the aligned call has no preds argument
        CAP.append(dict(maps=[np.array(m, copy=True) for m in maps], R8T=R8T, H=H, F=F, rows=res))
    return res


OI.readouts = hooked
OI.ARMS_BY_COND = dict(cam_nf=('noreg',), cam=('noreg', 'e6'))
opts = dict(script_sha256=OI.file_sha256(OI.__file__), threads=1, e4_sha256=OI.pinned_e4_manifest(), conds=('cam_nf', 'cam'), tg_conds=())
t = time.time()
job = OI.run_task(('val', 1, 0, 1, opts))
out['t_frozen_pipeline'] = time.time() - t
order = [('cam_nf', 'noreg'), ('cam', 'noreg'), ('cam', 'e6')]
assert len(CAP) == 3, len(CAP)
e6j = json.load(open(os.path.join(A27, 'g4', 'e6', 'jobs', NAME + '.json')))
sm = json.load(open(os.path.join(A27, 'g4', 'e6l_smoke', 'jobs', NAME + '.json')))
out['smoke_script_sha256'] = sm['script_sha256']
models = []
for ts in ('342101', '342202'):
    dn = f'e3l3_s{ts}'
    ck, model = E3.load_model(os.path.join(INP, dn, 'final.pt'), 'cpu')
    thr = json.load(open(os.path.join(INP, dn, 'threshold.json')))
    tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
    models.append((ts, model, float(model.w.detach()), thr, tb))
res = {}
for (cond, arm), c in zip(order, CAP):
    F = c['F']; n = len(c['maps'])
    K0T = [c['R8T'][F['comp_of'][a]] for a in range(n)]           # K0 rows unused here
    e3 = {}
    for ts, model, w, thr, tb in models:
        a = E3.eval_arm(model, w, thr, tb, c['maps'], c['maps'], F, c['H'], K0T, c['R8T'], 2)
        e3[f'E3L3_{ts}'] = dict(rows=[r for r, _ in a['R8']], known_mismatch=a['known_mismatch'], gbp=a['gbp'])
    ctrl_eq = {ro: jr(c['rows'][ro]) == e6j['conds'][cond]['arms'][arm]['aligned'][ro] for ro in OI.READOUTS}
    mine_e3 = {}
    for k, v in e3.items():
        theirs = [r for r, _ in sm['conds'][cond]['arms'][arm]['aligned'][k]]
        mine_e3[k] = dict(equal=(len(theirs) == len(v['rows']) and all(x[q] == y[q] for x, y in zip(v['rows'], theirs) for q in KEYS)),
                          n_rows=len(v['rows']), known_mismatch=v['known_mismatch'], gbp_cf_max=v['gbp']['cf_max'], gbp_viol=v['gbp']['violations'])
    hm = {}
    for ts in ('342101', '342202'):
        e = [r['furn_f1'] for r in e3[f'E3L3_{ts}']['rows'] if r['furn_f1'] is not None]
        nn = [r['furn_f1'] for r in G._expand(c['rows'][f'NET_{ts}']) if r['furn_f1'] is not None]
        hm[ts] = dict(e3=float(np.mean(e)), net=float(np.mean(nn)), d_pp=100 * (float(np.mean(e)) - float(np.mean(nn))))
    smoke_ctrl_eq = {ro: jr(c['rows'][ro]) == sm['conds'][cond]['arms'][arm]['aligned'][ro] for ro in OI.READOUTS}
    res[f'{cond}/{arm}'] = dict(controls_equal_e6_job=ctrl_eq, controls_equal_smoke=smoke_ctrl_eq, e3l3_equal_smoke=mine_e3, house=hm)
    print(cond, arm, 'ctrl==e6job', ctrl_eq, '| ctrl==smoke', all(smoke_ctrl_eq.values()), '| E3L3==smoke', {k: v['equal'] for k, v in mine_e3.items()},
          '| house', {k: round(v['d_pp'], 4) for k, v in hm.items()}, flush=True)
out['results'] = res
# per-house d of the reviewed smoke summary (cam e6)
S = json.load(open(os.path.join(A27, 'g4', 'e6l_smoke', 'summary.json')))
out['smoke_per_house_pp'] = {ts: S['cells']['val']['1'][ts]['per_house_pp'].get(NAME) for ts in ('342101', '342202')}
out['islands'] = dict(frozen=job['conds']['cam']['e6']['islands'], smoke=sm['conds']['cam']['islands'])
out['t1'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
json.dump(out, open(os.path.join(HERE, 'rv_e6l_val.json'), 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('smoke per-house d', out['smoke_per_house_pp'], '| mine', {ts: res['cam/e6']['house'][ts]['d_pp'] for ts in ('342101', '342202')})
print('islands equal', out['islands']['frozen'] == out['islands']['smoke'])
print('ALL_EQUAL', all(all(v['controls_equal_e6_job'].values()) and all(v['controls_equal_smoke'].values()) and all(x['equal'] for x in v['e3l3_equal_smoke'].values())
                       for v in res.values()))

"""review E1c: end-to-end re-check of the smoke fit on the cached TRAIN maps (no new evaluation, TRAIN only, 1 process).
Recomputes per half: histogram counts, B4 / B6 pooled furniture F1 and pooled free recall at the selected theta via G.score directly,
the range of the component sums (histogram clipping), and whether the free-recall constraint binds."""
import os, sys, json
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import numpy as np
import g4_calib_stalk as C
G = C.G
smoke = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'g4', 'e1c_smoke')
fit = json.load(open(os.path.join(smoke, 'fit.json')))
cache = C.default_cache()
out = {}
for h, house, seed in (('A', 'r1_i60', 7), ('B', 'r2_i61', 7)):
    z, H, comps, R8T = C._load_cache(os.path.join(cache, f'TRAIN_{house}.npz'))
    job = json.load(open(os.path.join(smoke, 'fit_jobs', f'{h}_{house}.json')))
    for tn in ('cam_nf',):
        tb = fit['halves'][h]['tiers'][tn]; cq = np.asarray(tb['cq']); theta = tb['theta']
        maps = z[f'{tn}_{seed}']
        no = np.zeros(C.NL, np.int64); nf = np.zeros(C.NL, np.int64)
        rows6, rows4, smin, smax = [], [], 0, 0
        for ci, cm in enumerate(comps):
            T = R8T[ci]; y = T.t == 1
            for a in cm:
                o, f = C.hist_L(maps[a], T.known, y); no += o; nf += f
            S4, S6, K, kc = C.comp_sums(maps, cm, cq)
            smin = min(smin, int(S6[K].min())); smax = max(smax, int(S6[K].max()))
            r6 = G.score(C.readout(S6, K, theta), T); r4 = G.score(np.where(K, (S4 > 0).astype(np.int8), np.int8(-1)), T)
            rows6 += [r6] * len(cm); rows4 += [r4] * len(cm)
        def pooled(rows):
            tp = sum(r['furn_tp'] for r in rows); fp = sum(r['furn_fp'] for r in rows); fn = sum(r['furn_fn'] for r in rows)
            return 2 * tp / (2 * tp + fp + fn)
        def pfree(rows, Ts):
            return None
        # pooled free recall: hits / total over expanded components
        fh6 = fh4 = ft = 0
        for ci, cm in enumerate(comps):
            T = R8T[ci]; S4, S6, K, kc = C.comp_sums(maps, cm, cq)
            fh6 += len(cm) * int((K & T.free & (S6 <= theta)).sum()); fh4 += len(cm) * int((K & T.free & (S4 <= 0)).sum()); ft += len(cm) * int(T.free.sum())
        out[f'{h}|{tn}'] = dict(hist_equal_fit_job=bool(no.tolist() == job['tiers'][f'{tn}|{seed}']['n_occ'] and nf.tolist() == job['tiers'][f'{tn}|{seed}']['n_free']),
                                B6_pooled_f1_direct=pooled(rows6), B6_fit=tb['train_f1'], B4_pooled_f1_direct=pooled(rows4), B4_fit=tb['train_b4_f1'],
                                B6_free_direct=fh6 / ft, B6_free_fit=tb['train_free'], B4_free_direct=fh4 / ft, B4_free_fit=tb['train_b4_free'],
                                S6_range=[smin, smax], clip_limit=[-C.VOFF, C.VOFF - 1], constraint_binds=bool(abs(tb['train_free'] - tb['train_b4_free']) < 1e-3),
                                theta=theta, l0q=tb['l0q'], tq=tb['tq'])
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rv_fit_e2e_out.json'), 'w'), indent=1)

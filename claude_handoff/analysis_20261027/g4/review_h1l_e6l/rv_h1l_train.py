"""Reviewer's independent recompute for G4-H1L (D8-69) on ONE TRAIN OOD file (R-I scratch export; no VAL file is read).
Reference path: the frozen formal E3-L3 evaluation g4_e3l3.eval_arm (rows) + a hand-written minimal decode loop from frozen primitives
(g4_e3l3.load_model / source, g4_e3l encode_maps / local_forest / gabp / decode_np, g4_e3r net_readout / stalk / linear_readout) +
own counting for zone F1, flips (own pairing, S <= 0), phantom blocks. Nothing from g4_h1_e3l3 is used on the reference side.
Reviewed path: g4_h1_e3l3.evaluate_file on the same file. Also: the O3 gate on a real TRAIN H1 smoke record (same house, other seed).
run: CUDA_VISIBLE_DEVICES= python -B rv_h1l_train.py"""
import os, sys, json, time
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'scripts', 'claude_v342'))
sys.path.insert(0, SCR)
import numpy as np
from scipy import ndimage
import g4_e3l3 as E3          # frozen
L = E3.L; R = E3.R; G = E3.G
import torch
torch.set_num_threads(1)

RI_OOD = r'C:\Users\27118\AppData\Local\Temp\claude\C--Users-27118-AppData-Local-Claude-Data-claude-pzs8sxrjxfjjc-scratch-workspaces-7462828b-8825-4be3-955b-e422393bdef0-4bc58eb2-7206-4518-ad18-b22859b86201-scratch-2026-10-23-91f83a\160d1fa0-6b2e-4733-9167-0e4000fcc081\scratchpad\ri_ood_export'
FNAME = sys.argv[1] if len(sys.argv) > 1 else 'floating_train_r2_i29_s203.npz'
AN = os.path.join(G.ROOT, 'claude_handoff', 'analysis_20261027')
INP = os.path.join(AN, 'review_E3R', 'inputs')
FIT = os.path.join(AN, 'g4', 'e1c', 'fit.json')
PAIR = {'342101': 'A', '342202': 'B'}
KEYS = ('furn_f1', 'furn_f1_tol1', 'free_recall', 'surface_f1', 'unknown_known_fpr', 'furn_tp', 'furn_fp', 'furn_fn')
out = dict(file=FNAME, t0=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))

path = os.path.join(RI_OOD, FNAME)
z = np.load(path)
H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
F = G.forest(R.edges_list(z['edges']), n)
R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
K0T = [G.Target(private[a], H) for a in range(n)]
maps = list(z['L_k0_cam_nf'])
occ = z['occ'] == 1
zone = G.dil6(np.asarray(z['ood_mask'], bool))
near = ndimage.binary_dilation(occ, structure=np.ones((3, 3, 3), bool))
b6ev = {h: R.Evidence.from_spec(f'c:{FIT}:{h}:cam_nf') for h in 'AB'}

# own direct sums (no relay): S_h, k per component
SK = {}
for h in 'AB':
    SK[h] = []
    for cm in F['comps']:
        S = np.zeros(occ.shape, np.int64); k = np.zeros(occ.shape, np.int64)
        for a in cm:
            e, kn = R.stalk(maps[a], b6ev[h].table); S += e; k += kn
        SK[h].append((S, k))

ref = {}
for ts in ('342101', '342202'):
    dn = f'e3l3_s{ts}'
    ck, model = E3.load_model(os.path.join(INP, dn, 'final.pt'), 'cpu')
    thr = json.load(open(os.path.join(INP, dn, 'threshold.json')))
    w = float(model.w.detach()); jo = int(thr['R8']['j_occ'])
    tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
    assert np.array_equal(np.asarray(ck['recipe']['evidence']['table']), b6ev[PAIR[ts]].table)
    t = time.time()
    formal = E3.eval_arm(model, w, thr, tb, maps, maps, F, H, K0T, R8T, 2)        # THE formal E3-L3 evaluation function
    t_formal = time.time() - t
    # minimal hand loop -> per-agent predictions
    with torch.no_grad():
        zp, lp = L.encode_maps(model, maps, tb, 'cpu')
        hp, lq, zq = E3.source(zp, lp, 'round')
        preds = []
        for ci, cm in enumerate(F['comps']):
            Fl = L.local_forest(F, cm)
            hb, lb, _ = L.gabp(hp[cm], lq[cm], Fl, w, 'round')
            lg = L.decode_np(model, hb, lb, 2)
            kc = SK['A'][ci][1]
            for i in range(len(cm)):
                assert np.array_equal(lb[i].numpy() > 0, kc > 0)
                preds.append((ci, R.net_readout(lg[i], kc, jo)))
    rows = [R.scored(p, R8T[ci], H, SK['A'][ci][1]) for ci, p in preds]
    rows_eq = all(a[k] == b[k] for a, (b, _) in zip(rows, formal['R8']) for k in KEYS)
    # own counting
    zc = np.zeros(3, np.int64); fl = np.zeros(2, np.int64); ph = 0.0; ph_list = []
    h = PAIR[ts]
    for ci, p in preds:
        T = R8T[ci]; S, k = SK[h][ci]
        pr = (p == 1) & T.known & H.region; tr = T.tr
        zc += [int((pr & tr & zone).sum()), int((pr & ~tr & zone).sum()), int((tr & ~pr & zone).sum())]
        b6 = np.where(k > 0, (S > b6ev[h].theta).astype(np.int8), np.int8(-1))
        sel = (p == 1) & (b6 == 0) & T.known & H.region & (S <= 0)
        fl += [int((sel & tr).sum()), int((sel & ~tr).sum())]
        m = (p == 1) & (k > 0) & H.region
        lab, nb = ndimage.label(m, structure=np.ones((3, 3, 3), bool))
        nph = int((np.bincount(lab[near & m], minlength=nb + 1)[1:] == 0).sum()) if nb else 0
        ph_list.append(nph); ph += nph / n
    ref[ts] = dict(rows=rows, formal_rows=[r for r, _ in formal['R8']], rows_eq_formal=rows_eq, zone_tp_fp_fn=zc.tolist(),
                   zone_f1=2 * zc[0] / (2 * zc[0] + zc[1] + zc[2]), flips_true_false=fl.tolist(),
                   flip_prec=(fl[0] / fl.sum()) if fl.sum() else None, phantoms=ph, w=w, j_occ=jo, known_mismatch=formal['known_mismatch'],
                   gbp=formal['gbp'], t_formal=t_formal)
    print(ts, 'rows == formal eval_arm:', rows_eq, 'zone', zc.tolist(), 'flips', fl.tolist(), 'phantoms', ph, flush=True)

# reviewed path
import g4_h1_e3l3 as HL
t = time.time()
r = HL.evaluate_file(path, HL.load_models(1), gbp_check=True)
out['t_reviewed'] = time.time() - t
cmp = {}
for ts in ('342101', '342202'):
    arm = f'E3L3_{ts}'; a = ref[ts]
    mine_rows = [x for x, _ in r['R8'][arm]['rows']]
    c = dict(rows_equal=all(x[k] == y[k] for x, y in zip(mine_rows, a['formal_rows']) for k in KEYS) and len(mine_rows) == len(a['formal_rows']),
             zone_equal=r['R8'][arm]['zones']['zone']['c'][:3] == a['zone_tp_fp_fn'],
             flips_equal=r['flips'][arm][f'B6_{PAIR[ts]}']['le_-0'][:2] == a['flips_true_false'],
             phantoms_equal=abs(r['R8'][arm]['phantoms'][0] - a['phantoms']) < 1e-12,
             known_mismatch=r['e3'][arm]['known_mismatch'], gbp_pass=r['e3'][arm]['gbp']['pass'],
             reviewed=dict(zone=r['R8'][arm]['zones']['zone']['c'][:3], flips=r['flips'][arm][f'B6_{PAIR[ts]}']['le_-0'][:2], phantoms=r['R8'][arm]['phantoms'][0]),
             reference=dict(zone=a['zone_tp_fp_fn'], flips=a['flips_true_false'], phantoms=a['phantoms'], zone_f1=a['zone_f1'], flip_prec=a['flip_prec']),
             rows_eq_formal_hand=a['rows_eq_formal'], w=a['w'], j_occ=a['j_occ'],
             mean_furn_f1=float(np.mean([x['furn_f1'] for x in mine_rows if x['furn_f1'] is not None])))
    cmp[arm] = c
    print(arm, {k: v for k, v in c.items() if k not in ('reviewed', 'reference')}, flush=True)
out['compare'] = cmp

# O3 gate on a real TRAIN H1 record: same house, the right seed must match, the other seed must not
recd = os.path.join(AN, 'g4', 'h1_smoke2', 'train', 'jobs', 'ctrl_train_r1_i9.json')
rec = json.load(open(recd))
fp = os.path.join(AN, 'g4', 'h1_export', rec['seeds']['203']['file'])
assert HL.sha(fp) == rec['seeds']['203']['file_sha256']
r2 = HL.evaluate_file(fp, HL.load_models(1), gbp_check=False)
ok_same, bad_same = HL.compare_controls(r2, rec['seeds']['203'])
ok_other, bad_other = HL.compare_controls(r2, rec['seeds']['204'])
out['gate'] = dict(same_seed_equal=ok_same, same_bad=bad_same, other_seed_equal=ok_other, other_bad=bad_other)
print('gate: s203 vs s203 record', ok_same, bad_same, '| s203 vs s204 record', ok_other, bad_other, flush=True)
out['t1'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
json.dump(out, open(os.path.join(HERE, 'rv_h1l_train.json'), 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('ALL_EQUAL', all(c['rows_equal'] and c['zone_equal'] and c['flips_equal'] and c['phantoms_equal'] and c['rows_eq_formal_hand'] for c in cmp.values())
      and ok_same and not ok_other)

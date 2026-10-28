"""Reviewer probe (E3-L4 review, check 2 with a REAL encoder): the frozen E3-L3 checkpoints (non-trivial ternary z) run through the
frozen g4_e3l3.eval_arm (GaBP 'round' on the wire, float GaBP closed-form / energy checks, known mask vs k > 0, O7 bytes) at
w = learned, 8, 3, 1 (cam_nf) on the ONE TRAIN smoke house (e3l2_smoke/export_train, 8 agents, noise seed 203). TRAIN only, CPU only.
Report only: no threshold is selected, no VAL / TEST map is read."""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import json, glob  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import g4_e3l3 as L3  # noqa: E402

L, R, G = L3.L, L3.R, L3.G
torch.set_num_threads(2)
A = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
exp = os.path.join(A, 'g4', 'e3l2_smoke', 'export_train')
rec = json.load(open(glob.glob(os.path.join(exp, 'records', '*.json'))[0]))
assert rec['split'] == 'TRAIN'
z = np.load(os.path.join(exp, rec['file']))
H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
F = G.forest(R.edges_list(z['edges']), n)
K0T = [G.Target(private[a], H) for a in range(n)]
R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
out = {}
for sd in ('342101', '342202'):
    d = os.path.join(A, 'review_E3R', 'inputs', f'e3l3_s{sd}')
    ck, model = L3.load_model(os.path.join(d, 'final.pt'), 'cpu')
    thr = json.load(open(os.path.join(d, 'threshold.json')))
    tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
    wl = float(model.w.detach())
    for w in (wl, 8.0, 3.0, 1.0):                       # reduced (shared 4-process CPU budget): learned, the grid ends, and w = 1
        for tn in ('cam_nf',):
            L_k0 = list(z[f'L_k0_{tn}']); L_pub = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z else L_k0
            a = L3.eval_arm(model, w, thr, tb, L_pub, L_k0, F, H, K0T, R8T, 2)
            r8 = R._summ([x for x in G._expand(a['R8'])])
            row = dict(w=w, tier=tn, gbp_cf_max=a['gbp']['cf_max'], gbp_violations=a['gbp']['violations'], known_mismatch=a['known_mismatch'],
                       bytes=a['bytes'], self_bytes_median=float(np.median([x['self_bytes'] for x in a['rate']])),
                       quant_belief_mean_abs=a['qdiff']['sum_absmax'] / max(a['qdiff']['cells'], 1), quant_belief_max=a['qdiff']['max'],
                       lost_cells=a['qdiff']['lost_cells'], disagree_rms=(float(np.sqrt(a['disagree']['sum_sq'] / a['disagree']['cells'])) if a['disagree']['cells'] else None),
                       z=L3.z_summary(a['z']), R8_furn_train_house=r8['furn_f1'], R8_free=r8['free_recall'])
            out[f'{sd}|{w:.2f}|{tn}'] = row
            print(sd, f"w {w:8.2f} {tn:6s} cf {row['gbp_cf_max']:.1e} viol {row['gbp_violations']} known_mismatch {row['known_mismatch']:6d} bytes {row['bytes']:9.0f} "
                  f"qdiff mean {row['quant_belief_mean_abs']:.3f} max {row['quant_belief_max']:.3f} disagree {row['disagree_rms']} "
                  f"z -/0/+ {row['z']['z_minus']:.3f}/{row['z']['z_zero']:.3f}/{row['z']['z_plus']:.3f} furn {row['R8_furn_train_house']:.4f}", flush=True)
json.dump(out, open(os.path.join(HERE, 'e3l3_ckpt_at_finite_w.json'), 'w'), indent=1)

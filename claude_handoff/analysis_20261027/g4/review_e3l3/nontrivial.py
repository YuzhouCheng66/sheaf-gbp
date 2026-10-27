"""reviewer check (CPU, TRAIN export house only): the smoke checkpoint codes z = 0 everywhere, which makes the GaBP / closed-form checks
vacuous. Here the encoder head is perturbed (as in the unit tests: weight N(0, 2), bias 0.1) to get a non-trivial ternary code at full
resolution; then (1) every directed message is recomputed by an independent integer recursion and compared with the integer sums of the
sender side, (2) eval_job runs the float64 closed form / energy checks and the known-mask counter, (3) bytes vs the z = 0 floor."""
import os
import sys
import copy
import json
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
ROOT = 'C:/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill'
sys.path.insert(0, ROOT + '/scripts/claude_v342')
import numpy as np  # noqa: E402
import torch  # noqa: E402
import g4_e3l3 as M  # noqa: E402

torch.set_num_threads(4)
L, L2, R, G = M.L, M.L2, M.R, M.G
G4 = ROOT + '/claude_handoff/analysis_20261027/g4'
O = G4 + '/review_e3l3/smoke'
EX = G4 + '/e3l2_smoke/export_train'
RUN = O + '/runs/s342101'
G.setup_env(R._HX[0] or G.HOUSEEXPO_JSON_DIR, R._HX[1] or G.HOUSEEXPO_INDEX)
rec_e = json.load(open(EX + '/records/train_r1_i0_s203.json'))
zf = np.load(os.path.join(EX, rec_e['file']))
F = G.forest(R.edges_list(zf['edges']), zf['private'].shape[0])
ck, model = M.load_model(RUN + '/checkpoints/final.pt')
torch.manual_seed(5)
with torch.no_grad():
    model.enc.head.weight.normal_(0, 2.0); model.enc.head.bias.fill_(0.1)
nt = O + '/neg/nontrivial'
shutil.rmtree(nt, ignore_errors=True); os.makedirs(nt + '/checkpoints')
ck2 = copy.deepcopy(ck); ck2['state_dict'] = {k: v.detach().clone() for k, v in model.state_dict().items()}
torch.save(ck2, nt + '/checkpoints/final.pt'); shutil.copy(RUN + '/threshold.json', nt + '/threshold.json')
tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
w = float(model.w.detach())
for tier in ('cam_nf', 'clean'):
    L_pub = list(zf[f'L_k0_{tier}'])
    zh, lam = L.encode_maps(model, L_pub, tb, 'cpu')
    h, lq, zq = M.source(zh, lam, 'round')
    fp = lam > 0
    print(f'[{tier}] z on footprint -1/0/+1 = {[int(((zq == v) & fp).sum()) for v in (-1., 0., 1.)]}, footprint {int(fp.sum())}, '
          f'off-footprint nonzero {int((zq[~fp] != 0).sum())}, saturated {float((zh.abs() > 0.95)[fp].float().mean()):.3f}')
    for cm in F['comps']:
        Fl = L.local_forest(F, cm)
        hb, lb, msgs = L.gabp(h[cm], lq[cm], Fl, w, 'round')
        hz = lq[cm].numpy().astype(np.float64) * zq[cm].numpy().astype(np.float64); ll = lq[cm].numpy().astype(np.float64)
        mine = {}
        for s, r in Fl['order']:
            lt = ll[s].copy(); ht = hz[s].copy()
            for k in Fl['nb'][s]:
                if k != r:
                    lt += mine[(k, s)][1]; ht += mine[(k, s)][0]
            f = w / (w + lt); lm = np.round(lt * f); mine[(s, r)] = (np.round(ht * f) * (lm > 0), lm)
        nd = ns = 0; maxk = 0
        for e in Fl['directed']:
            hi, li = L.msg_ints(*msgs[e])
            assert np.array_equal(hi[0], mine[e][0].astype(np.int64)) and np.array_equal(li, mine[e][1].astype(np.int64)), e
            seen = {e[0], e[1]}; out = [e[0]]; i = 0
            while i < len(out):
                for v in Fl['nb'][out[i]]:
                    if v not in seen:
                        seen.add(v); out.append(v)
                i += 1
            nd += int((li != ll[out].sum(0)).sum()); ns += int((hi[0] != hz[out].sum(0)).sum()); maxk = max(maxk, int(li.max()))
        kc = sum((L_pub[a] != R.UNK).astype(np.int32) for a in cm)
        km = sum(int(((lb[i].numpy() > 0) != (kc > 0)).sum()) for i in range(len(cm)))
        S_ = hz.sum(0); xb = L.belief_mean(hb, lb)[:, 0]; pos = kc > 0
        xe = float(max((xb[i][torch.from_numpy(pos)] - torch.from_numpy(S_ / np.maximum(kc, 1))[torch.from_numpy(pos)]).abs().max() for i in range(len(cm))))
        print(f'  component of {len(cm)}: {len(Fl["directed"])} directed messages = independent recursion (exact); Lam != side k-sum at {nd} cells, '
              f'h != side z-sum at {ns} cells (max side count {maxk}); known mismatch {km}; max |x_b - sum z / k| = {xe:.2e}')
rec = M.eval_job((rec_e, dict(export=EX, ckpts=[nt + '/checkpoints/final.pt'], thr={nt + '/checkpoints/final.pt': nt + '/threshold.json'}, ckpt_shas={},
                              tiers=list(R.EVAL_TIERS), threads=4, chunk=2, script_sha256='x')))
assert not rec.get('failed'), rec.get('traceback')
z0 = json.load(open(O + '/eval/jobs/train_r1_i0_s203.json'))
for tn in R.EVAL_TIERS:
    a = rec['tiers'][tn]['E3L3_342101']; b0 = z0['tiers'][tn]['E3L3_342101']['bytes']
    print(f'[eval {tn}] gbp {a["gbp"]} known_mismatch {a["known_mismatch"]} qdiff max {a["qdiff"]["max"]:.2e} lost {a["qdiff"]["lost_cells"]} '
          f'z {a["z"]} bytes {a["bytes"]:.0f} (z = 0 floor {b0:.0f}; code adds {a["bytes"] - b0:.0f}) upper {a["bytes_upper"]:.0f}')
    assert a['gbp']['violations'] == 0 and a['gbp']['cf_max'] <= L.CF_TOL and a['known_mismatch'] == 0
print('nontrivial ok')

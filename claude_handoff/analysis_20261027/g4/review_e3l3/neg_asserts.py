"""reviewer checks for g4_e3l3.py (CPU, TRAIN house only). Run after run_smoke.sh trained smoke/runs/s342101.
1. full-resolution GaBP on a real 8-agent TRAIN export house with the smoke checkpoint: messages recomputed independently (pure integer
   recursion), = integer sums (sum z, k) of the sender side at w = 1000; float GaBP vs closed form; known = k > 0.
2. negative: the same checkpoint with w = 0.9 -> eval known_mismatch > 0 -> verdict INCOMPLETE; select's assertion fires.
3. negative: a job with a broken GaBP check -> verdict INCOMPLETE.
4. worst-case drift of log w under Adam (sum of the lr schedule) for the chain's update counts."""
import os
import sys
import copy
import json
import math
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
CK = RUN + '/checkpoints/final.pt'
G.setup_env(R._HX[0] or G.HOUSEEXPO_JSON_DIR, R._HX[1] or G.HOUSEEXPO_INDEX)
rec_e = json.load(open(EX + '/records/train_r1_i0_s203.json'))
z = np.load(os.path.join(EX, rec_e['file']))
F = G.forest(R.edges_list(z['edges']), z['private'].shape[0])
ck, model = M.load_model(CK)
tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
w = float(model.w.detach())
print('checkpoint w', w, 'updates', ck['updates_done'])

# ---------------------------------------------------------------- 1. full-resolution GaBP vs independent integer recursion and closed form
L_pub = list(z['L_pub_cam_nf']) if 'L_pub_cam_nf' in z else list(z['L_k0_cam_nf'])
zh, lam = L.encode_maps(model, L_pub, tb, 'cpu')
h, lq, zq = M.source(zh, lam, 'round')
print('z levels', {v: int((zq == v).sum()) for v in (-1., 0., 1.)}, 'footprint', int((lam > 0).sum()), 'grid', tuple(lam.shape))
assert torch.equal(lam, torch.stack([torch.from_numpy(m != R.UNK).float() for m in L_pub])), 'Lam != footprint'
for cm in F['comps']:
    Fl = L.local_forest(F, cm)
    hb, lb, msgs = L.gabp(h[cm], lq[cm], Fl, w, 'round')
    # independent recursion in int64 numpy: m(s->r) = round(w / (w + lt) * (own + incoming)), gate on Lam
    hz = (lq[cm].numpy().astype(np.float64) * zq[cm].numpy().astype(np.float64)); ll = lq[cm].numpy().astype(np.float64)
    mine = {}
    for s, r in Fl['order']:
        lt = ll[s].copy(); ht = hz[s].copy()
        for k in Fl['nb'][s]:
            if k != r:
                lt += mine[(k, s)][1]; ht += mine[(k, s)][0]
        f = w / (w + lt)
        lm = np.round(lt * f); hm = np.round(ht * f) * (lm > 0)
        mine[(s, r)] = (hm, lm)
    nd = 0; nsum = 0
    for e in Fl['directed']:
        hi, li = L.msg_ints(*msgs[e])
        assert np.array_equal(hi[0], mine[e][0].astype(np.int64)) and np.array_equal(li, mine[e][1].astype(np.int64)), e
        # = integer sums of the sender side (w = 1000, m <= 8)
        side = M_side = None
        seen = {e[0], e[1]}; out = [e[0]]; i = 0
        while i < len(out):
            for v in Fl['nb'][out[i]]:
                if v not in seen:
                    seen.add(v); out.append(v)
            i += 1
        S_ = hz[out].sum(0); K_ = ll[out].sum(0)
        nd += int((li != K_).sum()); nsum += int((hi[0] != S_).sum())
    kc = sum((L_pub[a] != R.UNK).astype(np.int32) for a in cm)
    km = sum(int(((lb[i].numpy() > 0) != (kc > 0)).sum()) for i in range(len(cm)))
    c = L.gbp_checks(zq[cm].unsqueeze(1), lq[cm], Fl, w)
    hf, lf, _ = L.gabp(h[cm], lq[cm], Fl, w, None)
    xq = L.belief_mean(hb, lb); xf = L.belief_mean(hf, lf)
    print(f'component {cm}: {len(Fl["directed"])} directed messages recomputed exactly; Lam msgs != k-sums at {nd} cells, h msgs != z-sums at {nsum} cells; '
          f'known mismatch {km}; closed form cf_max {c["cf_max"]:.3e} (cells {c["cf_cells"]}), energy violations {c["violations"]} of {c["steps"]} steps; '
          f'max |x_q - x_float| {float((xq - xf).abs().max()):.4f}; max Lam_b {float(lb.max())}')

# ---------------------------------------------------------------- 2. w = 0.9: the known-mask assertion catches it (eval counter, select assert, verdict)
bad = O + '/neg/w09'
shutil.rmtree(bad, ignore_errors=True); os.makedirs(bad + '/checkpoints')
ck2 = copy.deepcopy(ck); ck2['state_dict']['log_w'] = torch.tensor(math.log(0.9)); torch.save(ck2, bad + '/checkpoints/final.pt')
shutil.copy(RUN + '/run.json', bad + '/run.json'); shutil.copy(RUN + '/threshold.json', bad + '/threshold.json')
r = M.eval_job((rec_e, dict(export=EX, ckpts=[bad + '/checkpoints/final.pt'], thr={bad + '/checkpoints/final.pt': bad + '/threshold.json'}, ckpt_shas={},
                            tiers=list(R.EVAL_TIERS), threads=4, chunk=2, script_sha256='x')))
assert not r.get('failed'), r.get('traceback')
km = {tn: r['tiers'][tn]['E3L3_342101']['known_mismatch'] for tn in r['tiers']}
print('w = 0.9 eval known_mismatch per tier', km)
assert all(v > 0 for v in km.values())
os.makedirs(bad + '/eval/jobs', exist_ok=True); M._jdump(r, bad + '/eval/jobs/train_r1_i0_s203.json')
S = M.aggregate(bad + '/eval')
v = M.verdict(S, [bad], G4 + '/e3r/summary.json', smoke=True)
print('w = 0.9 verdict reasons containing "known":', [x for x in v['incomplete_reasons'] if 'known' in x])
assert v['verdict'].startswith('INCOMPLETE') and any('known mask' in x for x in v['incomplete_reasons'])
os.remove(bad + '/threshold.json')                                  # select on this run is done by run_smoke.sh (CLI; spawn pool)

# ---------------------------------------------------------------- 3. a broken GaBP check -> INCOMPLETE
good = json.load(open(O + '/eval/jobs/train_r1_i0_s203.json'))
brk = O + '/neg/gbp_broken'
shutil.rmtree(brk, ignore_errors=True); os.makedirs(brk + '/jobs')
g2 = copy.deepcopy(good); g2['tiers']['clean']['E3L3_342101']['gbp']['cf_max'] = 2e-5
M._jdump(g2, brk + '/jobs/train_r1_i0_s203.json')
S = M.aggregate(brk); v = M.verdict(S, [RUN], G4 + '/e3r/summary.json', smoke=True)
print('broken GaBP -> gbp pass', S['gbp'], '| reasons:', [x for x in v['incomplete_reasons'] if 'GaBP' in x])
assert any('GaBP assertion failed for E3L3_342101' in x for x in v['incomplete_reasons'])

# ---------------------------------------------------------------- 4. worst-case log w drift (|Adam step| <= lr): sum of the schedule
for U in (8000, 10000, 12000):
    s = sum(R.lr_at(u, U, 1e-3, 300) for u in range(U))
    print(f'updates {U}: sum lr = {s:.3f} -> worst-case w >= 1000 exp(-{s:.3f}) = {1000 * math.exp(-s):.2f} (known mask breaks at w <= 1)')
print('neg asserts ok')

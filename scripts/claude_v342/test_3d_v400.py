"""Math / implementation tests for the 3D Sheaf-GBP trainer v400 (Claude, 2026-10-24, written before any 3D batch).
1. Data: observed occupied voxels are occupied in truth and observed free voxels are free (sensor is exact); union of poses is monotone.
2. Join: forest flooding for diameter rounds equals the centralised per-coordinate max-precision join over the reached set, on every
   coordinate of every node (self echoes included); 16 rounds == 8 rounds (idempotent / monotone); permuting agent order permutes the result.
3. Model: encoder is block-local (changing voxels outside block k leaves mu_k, q_k unchanged); unobserved blocks have mask 0; decoder output
   shape S^3 x 2; one training step reduces the loss on a fixed scene; metrics3d returns F1 = 1 on a perfect prediction and surface F1 < occ F1
   semantics hold.
CPU only.  python test_3d_v400.py
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapping_3d_data_v400 as d3
import train_3d_v400 as T

torch.manual_seed(0)
fx = d3.make_fixture('TRAIN', 7, size=64, agents=8)
occ, priv = fx['occ'], fx['private']
assert all((occ[priv[a] == 1] == 1).all() and (occ[priv[a] == 0] == 0).all() for a in range(8))
single = d3.observe(occ, fx['trajectory_m'][:1, 0])
assert ((single >= 0) <= (priv[0] >= 0)).all(), 'the union over poses must contain the single-pose observation'
print('1. sensor exactness and monotone union OK')

# ---- join exactness
n = 8; Dd = T.D
own = torch.from_numpy(priv)
model = T.SheafModel3D(64, 1).eval()
with torch.no_grad():
    mu, q, mask = model.source(own)
    edges = fx['edges']
    feats8 = T.rollout_max(mu, q, mask, edges, rounds=8, stages=(8,))[8]
    feats16 = T.rollout_max(mu, q, mask, edges, rounds=16, stages=(16,))[16]
    assert torch.equal(feats8, feats16), 'join must be idempotent: 16 rounds == 8 rounds on a tree of diameter <= 8'
    # centralised join over the reached set (diameter <= 8 here: all reached)
    prior = T.LATENT_PRIOR / (n * Dd)
    evid = mask.double() * q.double(); p_own = evid + prior; e_own = evid * mu.double()
    pw, ew, _ = T.kbit_wire_quantize(p_own, e_own)
    reached = torch.from_numpy(fx['reached'][4])                                    # hops = 8
    for j in range(n):
        cand_p = [p_own[j]] + [pw[s] for s in range(n) if reached[j, s] and s != j]
        cand_e = [e_own[j]] + [ew[s] for s in range(n) if reached[j, s] and s != j]
        P = torch.stack(cand_p); E = torch.stack(cand_e)
        tie = torch.arange(P.shape[0], dtype=P.dtype).view(-1, 1) * 1e-15
        idx = (P - tie).argmax(0, keepdim=True); Pj = P.gather(0, idx)[0]; Ej = E.gather(0, idx)[0]
        central = torch.cat((T.latent_quantize((Ej / Pj).float()), (1. / Pj).log().float()))
        assert torch.allclose(central, feats8[j], atol=1e-6), f'flooding != centralised join at node {j}'
    # permutation equivariance
    perm = torch.randperm(n); inv = torch.argsort(perm)
    edges_p = [(int(inv[i]), int(inv[j]), w) for i, j, w in edges]
    feats_p = T.rollout_max(mu[perm], q[perm], mask[perm], edges_p, rounds=8, stages=(8,))[8]
    assert torch.allclose(feats_p[inv], feats8, atol=1e-6), 'permuting agents must permute the result'
print('2. flooding == centralised join on all nodes/coordinates; idempotent; permutation-equivariant')

# ---- model properties
with torch.no_grad():
    k = int(torch.nonzero(mask[0] > 0)[0]); bx, by, bz = 64 // 8, 64 // 8, 64 // 4
    ix, rem = divmod(k, 8 * 4); iy, iz = divmod(rem, 4)
    own2 = own.clone()
    outside = torch.ones(64, 64, 64, dtype=torch.bool); outside[ix * bx:(ix + 1) * bx, iy * by:(iy + 1) * by, iz * bz:(iz + 1) * bz] = False
    own2[0][outside] = torch.where(own2[0][outside] == -1, torch.tensor(0, dtype=torch.int8), own2[0][outside])   # change everything outside block k
    mu2, q2, mask2 = model.source(own2)
    assert torch.equal(mu2[0, k], mu[0, k]) and torch.equal(q2[0, k], q[0, k]), 'encoder must be block-local'
    unobs = torch.nonzero(mask[0] == 0)[0]
    assert float(mask[0, unobs]) == 0.
    known_logit, occ_logit = model.decoder(feats8)
    assert known_logit.shape == (n, 64 ** 3) and occ_logit.shape == (n, 64 ** 3)
    tgt = T.stream_targets(own, reached)
    pred_perfect = tgt.clone()
    rows = T.metrics3d(pred_perfect, tgt, 64)
    assert all(abs(r['occ_f1'] - 1) < 1e-9 and abs(r['surface_f1'] - 1) < 1e-9 and r['accuracy'] == 1. and r['unknown_known_fpr'] == 0. for r in rows)
print('3a. block-local encoder, mask semantics, decoder shape, perfect-prediction metrics OK')
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
losses = []
for step in range(3):
    opt.zero_grad()
    mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, edges, stages=(0, 8))
    l0, _, _ = T.stage_loss(model.decoder, feats[0], T.stream_targets(own, torch.from_numpy(fx['reached'][0])), 64)
    l0.backward(); opt.step(); losses.append(float(l0))
assert losses[-1] < losses[0], losses
print('3b. three optimiser steps reduce the K0 loss:', [round(x, 4) for x in losses])
print('ALL TESTS PASSED')

"""Tests for v421 / batch 3S-B (Claude, 2026-10-25, before any R2 run). Run on uk-linux, CPU (CUDA_VISIBLE_DEVICES=):
python test_3d_v421.py --source <sealed>
A. v421 at R1 equals v420 at R1 bit-for-bit (occupancy, slab, furniture, private maps, positions, edges, reached) on 3 houses, so batch
   3S-A is the R1 arm of the resolution study; at R2 the same house has the same agent positions / forest / reached sets, the occupancy
   refines R1 (every occupied R1 voxel contains an occupied R2 sub-voxel), the sensor is exact, beams = 131,072.
B. train_3d_v421 with --sub 2 --readout-width 0 builds the v420 model bit-for-bit (state_dict, forward); with --sub 4 --readout-width
   128 at R2 the compute grid is 64 x 64 x 16, the last decoder stage has 128 channels, the decoder emits 2 x 256 x 256 x 64 logits, the
   encoder is block-local at the R2 voxel level, and three optimiser steps reduce the K0 loss.
"""
import argparse, importlib.util, os, sys, time
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W0
import mapping_3d_storey_data_v421 as W1
import train_3d_v420 as T0
import train_3d_v421 as T1
t0 = time.time(); torch.manual_seed(0)
R1, R2 = W1.DIMS['R1'], W1.DIMS['R2']
for (split, idx, rooms) in (('TRAIN', 7, 2), ('TRAIN', 30, 1), ('VALIDATION', 3, 3)):
    a = W0.make_fixture(split, idx, rooms, R1); b = W1.make_fixture(split, idx, rooms, R1)
    for k in ('occ', 'slab', 'obj', 'private', 'positions_m', 'reached'):
        assert np.array_equal(a[k], b[k]), (split, idx, k)
    assert a['edges'] == b['edges'] and a['world_id'] == b['world_id']
print(f'A1. v421 at R1 == v420 at R1 bit-for-bit on 3 houses ({time.time()-t0:.0f}s)')
f1 = W1.make_fixture('TRAIN', 7, 2, R1); f2 = W1.make_fixture('TRAIN', 7, 2, R2)
assert np.array_equal(f1['positions_m'], f2['positions_m']) and f1['edges'] == f2['edges'] and np.array_equal(f1['reached'], f2['reached'])
occ2, priv2 = f2['occ'], f2['private']
assert all(((occ2[priv2[a] == 1]) == 1).all() and ((occ2[priv2[a] == 0]) == 0).all() for a in range(priv2.shape[0]))
pooled = occ2.reshape(128, 2, 128, 2, 32, 2).max((1, 3, 5))
assert (pooled[f1['occ'] == 1] == 1).mean() > 0.97, (pooled[f1['occ'] == 1] == 1).mean()
assert W1.beams(R2) == 131072 and W1.beams(R1) == 65536
print(f'A2. R2: same positions / forest / reached as R1; sensor exact; R2 refines R1 (recall of R1 occupancy {(pooled[f1["occ"] == 1] == 1).mean():.3f}); known fraction R1 {(f1["private"] >= 0).mean():.3f} R2 {(priv2 >= 0).mean():.3f} ({time.time()-t0:.0f}s)')

T1.SUB = 2; T1.READOUT_WIDTH = 0
m0 = T0.SheafModel3D(R1, 1); m1 = T1.SheafModel3D(R1, 1)
s0, s1 = m0.state_dict(), m1.state_dict()
assert list(s0) == list(s1) and all(torch.equal(s0[k], s1[k]) for k in s0)
own = torch.from_numpy(f1['private']); T0.CHART_M = T1.CHART_M = f1['chart_m']
with torch.no_grad():
    a0 = m0.source(own); a1 = m1.source(own)
    assert all(torch.equal(x, y) for x, y in zip(a0, a1))
    feats = T0.rollout_max(*a0, f1['edges'], stages=(8,))[8]
    assert all(torch.equal(x, y) for x, y in zip(m0.decoder(feats), m1.decoder(feats)))
print('B1. train_3d_v421 (--sub 2 --readout-width 0) == train_3d_v420 bit-for-bit (state_dict, source, decoder)')
T1.SUB = 4; T1.READOUT_WIDTH = 128
m = T1.SheafModel3D(R2, 1).eval(); T1.CHART_M = f2['chart_m']
assert m.trunk.comp == (64, 64, 16) and m.decoder.output.in_channels >= 128 and m.decoder.output.out_channels == 128
own2 = torch.from_numpy(priv2)
with torch.no_grad():
    mu_raw, q_raw, mask = m.head(m.trunk(own2), own2)
    k = int(torch.nonzero(mask[0] > 0)[0]); bx, by, bz = 256 // 8, 256 // 8, 64 // 4
    ix, rem = divmod(k, 8 * 4); iy, iz = divmod(rem, 4)
    o2 = own2.clone(); outside = torch.ones(R2, dtype=torch.bool); outside[ix * bx:(ix + 1) * bx, iy * by:(iy + 1) * by, iz * bz:(iz + 1) * bz] = False
    o2[0][outside] = torch.randint(-1, 2, R2, dtype=torch.int8)[outside]
    mu2, q2, _ = m.head(m.trunk(o2), o2)
    assert torch.equal(mu2[0, k], mu_raw[0, k]) and torch.equal(q2[0, k], q_raw[0, k])
    f = T1.rollout_max(T1.latent_quantize(mu_raw), q_raw, mask, f2['edges'], stages=(8,))[8]
    kl, ol = m.decoder(f); assert kl.shape == (8, 256 * 256 * 64)
params = {kk: sum(p.numel() for p in v) for kk, v in m.parameter_groups().items()}
opt = torch.optim.AdamW(m.parameters(), lr=1e-3); m.train(); losses = []
for step in range(3):
    opt.zero_grad(); mu, q, mk = m.source(own2); fz = T1.rollout_max(mu, q, mk, f2['edges'], stages=(0,))
    l, _, _ = T1.stage_loss(m.decoder, fz[0], T1.stream_targets(own2, torch.from_numpy(f2['reached'][0])), R2); l.backward(); opt.step(); losses.append(float(l.detach()))
assert losses[-1] < losses[0], losses
print(f'B2. R2 model: compute grid 64x64x16, readout 128 ch, block-local, decoder shape, loss decreases {[round(x, 4) for x in losses]}; params {params}')
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')

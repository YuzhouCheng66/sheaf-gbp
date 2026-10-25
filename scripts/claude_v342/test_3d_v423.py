"""Tests for train_3d_v423 / batch 3S-C (Claude, 2026-10-25, before any v423 run). Run on uk-linux, CPU (CUDA_VISIBLE_DEVICES=):
python test_3d_v423.py --source <sealed>
A. --layout 8x8x4 builds train_3d_v420 bit-for-bit: state_dict for both seeds, source (mu, q, mask), join, decoder logits on a fixture.
B. --layout 16x16x1: D = 256; encoder patch 4 x 4 x 16 compute cells (8 x 8 x 32 voxels: a whole column); the parameter count of every
   group equals v420's (the layout changes only block shapes) and the initial tensors are identical; coordinate k = ix * 16 + iy owns
   column (ix, iy): the coverage mask equals "the column has a known voxel"; the encoder is block-local (randomising every voxel outside
   one column leaves that coordinate's mu and q unchanged); decoder logits 2 x 128 x 128 x 32; three AdamW steps reduce the K0 loss.
C. diag_storey_v423 / diag_interior_v423 take the layout from the checkpoint meta (source text), and the CLI sets the layout before any
   model is built and records it in run.json.
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W
import train_3d_v420 as T0
import train_3d_v423 as T3
t0 = time.time()
R1 = W.DIMS['R1']
fx = W.make_fixture('TRAIN', 7, 2, R1)
own = torch.from_numpy(fx['private']); T0.CHART_M = T3.CHART_M = fx['chart_m']

T3.BX, T3.BY, T3.BZ = T3.LAYOUTS['8x8x4']
for seed in (342101, 342202):
    m0 = T0.SheafModel3D(R1, seed); m1 = T3.SheafModel3D(R1, seed)
    s0, s1 = m0.state_dict(), m1.state_dict()
    assert list(s0) == list(s1) and all(torch.equal(s0[k], s1[k]) for k in s0), seed
with torch.no_grad():
    a0 = m0.source(own); a1 = m1.source(own)
    assert all(torch.equal(x, y) for x, y in zip(a0, a1))
    f0 = T0.rollout_max(*a0, fx['edges'], stages=(0, 8)); f1 = T3.rollout_max(*a1, fx['edges'], stages=(0, 8))
    for k in (0, 8):
        assert torch.equal(f0[k], f1[k]) and all(torch.equal(x, y) for x, y in zip(m0.decoder(f0[k]), m1.decoder(f1[k])))
print(f'A. layout 8x8x4 == train_3d_v420 bit-for-bit (state_dict for both seeds, source, join, decoder logits) ({time.time()-t0:.0f}s)')

T3.BX, T3.BY, T3.BZ = T3.LAYOUTS['16x16x1']
assert T3.BX * T3.BY * T3.BZ == T3.D == 256
m = T3.SheafModel3D(R1, 342202).eval(); s3 = m.state_dict()
assert m.trunk.bs == (4, 4, 16) and m.head.k == (8, 8, 32)
assert list(s3) == list(s0) and all(s3[k].shape == s0[k].shape for k in s0) and all(torch.equal(s3[k], s0[k]) for k in s0)
pg0 = {k: sum(p.numel() for p in v) for k, v in m0.parameter_groups().items()}; pg3 = {k: sum(p.numel() for p in v) for k, v in m.parameter_groups().items()}
assert pg0 == pg3, (pg0, pg3)
with torch.no_grad():
    mu_raw, q_raw, mask = m.head(m.trunk(own), own)
    cov = ((own >= 0).reshape(8, 16, 8, 16, 8, 1, 32).any(-1).any(4).any(2)).reshape(8, 256).float()
    assert torch.equal(mask, cov)
    cand = [int(j) for j in torch.nonzero(mask[0] > 0)[:, 0] if divmod(int(j), 16)[0] != divmod(int(j), 16)[1]]   # review S8: off-diagonal column
    k = cand[0]; ix, iy = divmod(k, 16); assert ix != iy
    o2 = own.clone(); outside = torch.ones(R1, dtype=torch.bool); outside[ix * 8:(ix + 1) * 8, iy * 8:(iy + 1) * 8, :] = False
    o2[0][outside] = torch.randint(-1, 2, R1, dtype=torch.int8)[outside]
    mu2, q2, _ = m.head(m.trunk(o2), o2)
    assert torch.equal(mu2[0, k], mu_raw[0, k]) and torch.equal(q2[0, k], q_raw[0, k])
    others = [j for j in range(256) if j != k]
    assert not torch.equal(mu2[0, others], mu_raw[0, others])            # the perturbation is real
    f = T3.rollout_max(T3.latent_quantize(mu_raw), q_raw, mask, fx['edges'], stages=(8,))[8]
    kl, ol = m.decoder(f); assert kl.shape == ol.shape == (8, 128 * 128 * 32)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3); m.train(); losses = []
for step in range(3):
    opt.zero_grad(); mu, q, mk = m.source(own); fz = T3.rollout_max(mu, q, mk, fx['edges'], stages=(0,))
    l, _, _ = T3.stage_loss(m.decoder, fz[0], T3.stream_targets(own, torch.from_numpy(fx['reached'][0])), R1); l.backward(); opt.step(); losses.append(float(l.detach()))
assert losses[-1] < losses[0], losses
print(f'B. layout 16x16x1: D = 256, patch 4x4x16 cells (whole column), params per group identical {pg3}, initial tensors identical, '
      f'coverage mask = column known, block-local, decoder shape, loss {[round(x, 4) for x in losses]} ({time.time()-t0:.0f}s)')

src = (HERE / 'train_3d_v423.py').read_text()
i_set = src.index("    BX, BY, BZ = LAYOUTS[args.layout]"); i_model = src.index("    model = SheafModel3D(dims, args.seed)"); i_meta = src.index("blocks=[BX, BY, BZ]")
assert "global CHART_M, BX, BY, BZ" in src and i_set < i_model < i_meta
for name in ('diag_storey_v423.py', 'diag_interior_v423.py'):
    d = (HERE / name).read_text()
    assert "T.BX, T.BY, T.BZ = ck['meta']['blocks']" in d and d.index("T.BX, T.BY, T.BZ = ck['meta']['blocks']") < d.index("model = T.SheafModel3D(dims, ck['meta']['seed'])")
print('C. the CLI sets the layout before the model is built and run.json records it; both diag copies take the layout from the checkpoint meta')
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')

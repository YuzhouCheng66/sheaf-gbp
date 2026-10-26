"""Tests for train_3d_v431 / batch 3S-Z (Claude, 2026-10-26, before any run). CPU: python test_3d_v431.py --source <sealed>
A. Defaults are train_3d_v430 bit-for-bit for the stack recipe (32x32x1 @ 8+8, d3 W2): state_dict, source, join, logits, loss, grads.
B. 64x64x1 @ 8+8 (D = 4,096): builds; fold 4 onto the 16 x 16 start (pre input width 32; same schedule / channel plan as the control, only
   pre.0's input width differs); trunk patch 1 x 1 x 16 cells; coverage mask = 2 x 2 x 32-voxel column known; block-local; rollout and
   logits shapes; raw_code_bytes 8,192; parameter counts reported; three AdamW steps reduce the K0 loss.
C. The diag copies (v429) and the probe configure D from the checkpoint meta, so a 64x64x1 checkpoint builds (source text + a build check).
"""
import argparse, os, sys, time
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W
import train_3d_v430 as T0
import train_3d_v431 as T1
t0 = time.time(); R1 = W.DIMS['R1']
fx = W.make_fixture('TRAIN', 7, 2, R1); own = torch.from_numpy(fx['private']); T0.CHART_M = T1.CHART_M = fx['chart_m']
reached0 = torch.from_numpy(fx['reached'][0])

def cfg(M, layout, w=2.0, bits=(8, 8)):
    M.DECODER = 'd3'; M.DECODER_WIDTH = w; M.BX, M.BY, M.BZ = M.LAYOUTS[layout]; M.D = M.BX * M.BY * M.BZ; M.PRECISION = 'learned'; M.KNOWN_WEIGHT = 'uniform'
    M.WIRE_BITS = bits; M._v378.MU_BITS, M._v378.Q_BITS = bits; M.LEVELS = 2 ** (bits[0] - 1) - 1

cfg(T0, '32x32x1'); cfg(T1, '32x32x1')
m0 = T0.SheafModel3D(R1, 342101); m1 = T1.SheafModel3D(R1, 342101)
s0, s1 = m0.state_dict(), m1.state_dict(); assert list(s0) == list(s1) and all(torch.equal(s0[k], s1[k]) for k in s0)
a0 = m0.source(own); a1 = m1.source(own); assert all(torch.equal(x, y) for x, y in zip(a0, a1))
f0 = T0.rollout_max(*a0, fx['edges'], stages=(8,)); f1 = T1.rollout_max(*a1, fx['edges'], stages=(8,)); tgt = T0.stream_targets(own, reached0)
l0, _, _ = T0.stage_loss(m0.decoder, f0[8], tgt, R1); l1, _, _ = T1.stage_loss(m1.decoder, f1[8], tgt, R1); assert torch.equal(l0, l1)
g0 = torch.autograd.grad(l0, list(m0.parameters()), allow_unused=True); g1 = torch.autograd.grad(l1, list(m1.parameters()), allow_unused=True)
assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(g0, g1))
print(f'A. v431 == v430 bit-for-bit for the stack recipe (32x32x1 W2; model, source, join, loss, grads) ({time.time()-t0:.0f}s)')

cfg(T1, '64x64x1'); m = T1.SheafModel3D(R1, 342101).eval()
assert T1.D == 4096 and m.decoder.start == (16, 16, 8) and len(m.decoder.blocks) == 2 and m.decoder.fold == 4 and m.decoder.pre[0].in_channels == 32 and m.trunk.bs == (1, 1, 16) and m.head.k == (2, 2, 32)
sa = m.decoder.state_dict(); sc = m1.decoder.state_dict()
assert list(sa) == list(sc) and all(sa[k].shape == sc[k].shape for k in sa if k != 'pre.0.weight') and sa['pre.0.weight'].shape[1] == 32
with torch.no_grad():
    mu, q, mask = m.source(own); assert mu.shape == (8, 4096)
    cov = ((own >= 0).reshape(8, 64, 2, 64, 2, 1, 32).any(-1).any(4).any(2)).reshape(8, 4096).float(); assert torch.equal(mask, cov)
    k = [int(j) for j in torch.nonzero(mask[0] > 0)[:, 0] if divmod(int(j), 64)[0] != divmod(int(j), 64)[1]][0]; ix, iy = divmod(k, 64)
    o2 = own.clone(); outside = torch.ones(R1, dtype=torch.bool); outside[ix * 2:(ix + 1) * 2, iy * 2:(iy + 1) * 2, :] = False
    o2[0][outside] = torch.randint(-1, 2, R1, dtype=torch.int8)[outside]
    mu2, q2, _ = m.source(o2); assert torch.equal(mu2[0, k], mu[0, k]) and torch.equal(q2[0, k], q[0, k])
    f = T1.rollout_max(mu, q, mask, fx['edges'], stages=(8,))[8]; assert f.shape == (8, 8192)
    kl, ol = m.decoder(f); assert kl.shape == ol.shape == (8, 128 * 128 * 32)
pg = {kk: sum(p.numel() for p in v) for kk, v in m.parameter_groups().items()}
opt = torch.optim.AdamW(m.parameters(), lr=1e-3); m.train(); losses = []
for step in range(3):
    opt.zero_grad(); mu, q, mk = m.source(own); fz = T1.rollout_max(mu, q, mk, fx['edges'], stages=(0,))
    l, _, _ = T1.stage_loss(m.decoder, fz[0], tgt, R1); l.backward(); opt.step(); losses.append(float(l.detach()))
assert losses[-1] < losses[0], losses
print(f'B. 64x64x1 @ 8+8: D 4,096 (8,192 B), fold 4 onto the 16x16 start (pre in 32, same schedule), patch 1x1x16, column mask, block-local, shapes ok; params {pg}; loss {[round(x, 4) for x in losses]} ({time.time()-t0:.0f}s)')
for name in ('diag_storey_v429.py', 'diag_interior_v429.py', 'diag_veto_v429.py', 'probe_wall_v425.py'):
    d = (HERE / name).read_text(); assert 'T.D = T.BX * T.BY * T.BZ' in d, name
print('C. the v429 diag copies and the probe take D from the checkpoint meta')
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')

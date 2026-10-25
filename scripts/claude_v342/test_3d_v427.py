"""Tests for train_3d_v427 (Claude, 2026-10-25, before any run). CPU: python test_3d_v427.py --source <sealed>
A. --known-weight uniform is train_3d_v426 bit-for-bit (d3 W2 8x8x4 learned; plan_column W2 16x16x1 coverage): state_dict, source,
   join, logits, and the stage loss value and gradients on a real fixture.
B. --known-weight boundary: the loss changes; the border mask equals (target surface) | (unknown 6-adjacent to known) and is one voxel thick
   on both sides (checked on a synthetic target: a known cube with a wall); with w_surface = 0 the boundary loss equals the uniform loss;
   three AdamW steps reduce the K0 loss.
C. CLI order and run.json; the v427 diag copies and the probe configure decoder / precision / known-weight from the checkpoint meta.
"""
import argparse, os, sys, time
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W
import train_3d_v426 as T6
import train_3d_v427 as T7
t0 = time.time(); R1 = W.DIMS['R1']
fx = W.make_fixture('TRAIN', 7, 2, R1); own = torch.from_numpy(fx['private']); T6.CHART_M = T7.CHART_M = fx['chart_m']
reached0 = torch.from_numpy(fx['reached'][0])

def cfg(M, dec, w, layout, prec):
    M.DECODER = dec; M.DECODER_WIDTH = w; M.BX, M.BY, M.BZ = M.LAYOUTS[layout]; M.D = M.BX * M.BY * M.BZ; M.PRECISION = prec

def same(ma, mb, Ma, Mb):
    sa, sb = ma.state_dict(), mb.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    a = ma.source(own); b = mb.source(own); assert all(torch.equal(x, y) for x, y in zip(a, b))
    fa = Ma.rollout_max(*a, fx['edges'], stages=(0,)); fb = Mb.rollout_max(*b, fx['edges'], stages=(0,))
    tgt = Ma.stream_targets(own, reached0)
    la, _, _ = Ma.stage_loss(ma.decoder, fa[0], tgt, R1); lb, _, _ = Mb.stage_loss(mb.decoder, fb[0], tgt, R1)
    assert torch.equal(la, lb)
    ga = torch.autograd.grad(la, list(ma.parameters()), allow_unused=True); gb = torch.autograd.grad(lb, list(mb.parameters()), allow_unused=True)
    assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(ga, gb))

T7.KNOWN_WEIGHT = 'uniform'
for dec, w, layout, prec in (('d3', 2.0, '8x8x4', 'learned'), ('plan_column', 2.0, '16x16x1', 'coverage')):
    cfg(T6, dec, w, layout, prec); cfg(T7, dec, w, layout, prec)
    same(T6.SheafModel3D(R1, 342101), T7.SheafModel3D(R1, 342101), T6, T7)
print(f'A. --known-weight uniform == train_3d_v426 bit-for-bit incl. loss value and gradients ({time.time()-t0:.0f}s)')

cfg(T7, 'plan_column', 2.0, '16x16x1', 'learned'); m = T7.SheafModel3D(R1, 342101)
with torch.no_grad():
    mu, q, mk = m.source(own); f0 = T7.rollout_max(mu, q, mk, fx['edges'], stages=(0,))[0]
tgt = T7.stream_targets(own, reached0)
T7.KNOWN_WEIGHT = 'uniform'; lu, ku, ou = T7.stage_loss(m.decoder, f0, tgt, R1)
T7.KNOWN_WEIGHT = 'boundary'; lb, kb, ob = T7.stage_loss(m.decoder, f0, tgt, R1)
assert ou == ob and ku != kb and not torch.equal(lu, lb)
lb0, kb0, _ = T7.stage_loss(m.decoder, f0, tgt, R1, w_surface=0.)
lu0, ku0, _ = (T7.KNOWN_WEIGHT, 'uniform')[0:0] or (None, None, None)
T7.KNOWN_WEIGHT = 'uniform'; lu0, ku0, _ = T7.stage_loss(m.decoder, f0, tgt, R1, w_surface=0.)
assert abs(kb0 - ku0) < 1e-5, (kb0, ku0)
# synthetic border check: a known free cube with one occupied wall slab inside; everything else unknown
dims = (16, 16, 8); t = torch.full((1, 16 * 16 * 8), -1, dtype=torch.int8); v = t.reshape(1, *dims)
v[0, 4:12, 4:12, 2:6] = 0; v[0, 8, 4:12, 2:6] = 1; t = v.reshape(1, -1)
border = (T7.surface_of(t, dims) | (T7.dilate6(t >= 0, dims) & (t < 0))).reshape(dims)
known = (v[0] >= 0)
inner = torch.zeros(dims, dtype=torch.bool); inner[5:11, 5:11, 3:5] = True; inner[7:10, 5:11, 3:5] = False      # known cells not adjacent to unknown nor to the wall
assert not border[inner & known].any()
shell = T7.dilate6((v.reshape(1, -1) >= 0), dims).reshape(dims) & ~known                                        # the unknown shell, one voxel thick
assert border[shell].all() and not border[~known & ~shell].any()
wall_surface = T7.surface_of(t, dims).reshape(dims); assert border[wall_surface].all() and wall_surface[8, 5, 3]
print(f'B. boundary known-weight: loss differs only through the known term; w_surface 0 reproduces uniform; border = surface | unknown shell, one voxel thick ({time.time()-t0:.0f}s)')
T7.KNOWN_WEIGHT = 'boundary'
opt = torch.optim.AdamW(m.parameters(), lr=1e-3); m.train(); losses = []
for step in range(3):
    opt.zero_grad(); mu, q, mk = m.source(own); fz = T7.rollout_max(mu, q, mk, fx['edges'], stages=(0,))
    l, _, _ = T7.stage_loss(m.decoder, fz[0], tgt, R1); l.backward(); opt.step(); losses.append(float(l.detach()))
assert losses[-1] < losses[0], losses
src = (HERE / 'train_3d_v427.py').read_text()
assert src.index("KNOWN_WEIGHT = args.known_weight") < src.index("model = SheafModel3D(dims, args.seed)") and "known_weight=KNOWN_WEIGHT," in src
for name in ('diag_storey_v427.py', 'diag_interior_v427.py', 'probe_wall_v425.py'):
    d = (HERE / name).read_text(); assert "T.KNOWN_WEIGHT = " in d and "T.PRECISION = " in d and "T.DECODER = " in d and d.index("T.DECODER = ") < d.index("model = T.SheafModel3D(dims, ")
print(f'C. CLI order, run.json, and the v427 diag copies + probe configure every knob from the checkpoint meta; loss {[round(x, 4) for x in losses]}')
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')

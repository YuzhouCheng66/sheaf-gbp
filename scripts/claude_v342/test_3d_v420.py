"""Tests for the v420 storey world and trainer (Claude, 2026-10-25, written before any v420 training). Run on uk-linux (CPU):
python test_3d_v420.py --source <sealed>   (needs HOUSEEXPO_JSON_DIR / HOUSEEXPO_INDEX)
A. World: deterministic; TEST closed; TRAIN / VALIDATION houses disjoint (HouseExpo index); sensor exact (observed occupied cells are
   occupied, observed free cells are free); furniture never inside walls; agents stand on free floor away from furniture; the free floor
   at sensor height is one connected component up to 5 %; the furniture footprint share over 12 houses has a median in [0.15, 0.45]
   (Replica: .30); the slab mask is exactly the floor + ceiling layers.
B. Adapters: space_to_depth / depth_to_space inverse for anisotropic grids (index map checked).
C. Join: flooding for 8 rounds equals the centralised per-coordinate max-precision join on every node / coordinate; 16 rounds == 8;
   permutation equivariant.
D. Model: block-local encoder at the voxel level; mask semantics; decoder output 2 x NX NY NZ; the chart-scale plane changes the decoder
   output (it is wired in); perfect prediction scores 1 on every metric; slab-only painting scores ~0 variable-structure F1;
   three optimiser steps reduce the K0 loss.
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W
import train_3d_v420 as T
from scipy import ndimage
t0 = time.time(); torch.manual_seed(0)
dims = W.DIMS['R1']

# ---- A. world
fx = W.make_fixture('TRAIN', 7, 2, dims); fx2 = W.make_fixture('TRAIN', 7, 2, dims)
assert W.array_hash(fx['private']) == W.array_hash(fx2['private']) and np.array_equal(fx['occ'], fx2['occ'])
try:
    W.world_spec('TEST', 0, 1); raise AssertionError('TEST must raise')
except ValueError:
    pass
tr = {W.world_spec('TRAIN', i, s)['house_id'] for s in (1, 2, 3) for i in range(20)}; va = {W.world_spec('VALIDATION', i, s)['house_id'] for s in (1, 2, 3) for i in range(16)}
assert not tr & va
occ, priv = fx['occ'], fx['private']
assert all(((occ[priv[a] == 1]) == 1).all() and ((occ[priv[a] == 0]) == 0).all() for a in range(priv.shape[0])), 'sensor must be exact'
sp = W.world_spec('TRAIN', 7, 2); ras = W.rasterize(sp, dims)
import mapping_houseexpo_data_v371 as v371
walls2d = (v371.rasterize(sp['base_spec'], dims[0])['walls'] != 0).T
assert not (ras['obj'] & walls2d[:, :, None]).any(), 'furniture inside walls'
cx, _, cz = ras['cell']
for p in fx['positions_m']:
    i, j, k = int(p[0] / cx), int(p[1] / cx), int(p[2] / cz)
    assert occ[i, j, int(W.FLOOR_T / cz):k + 1].sum() == 0, 'agent column must be free from the floor top to the sensor'   # review S8: from z cell 1
zc = (np.arange(dims[2]) + .5) * cz
free2 = ~occ[:, :, (zc >= W.FLOOR_T) & (zc < W.FLOOR_T + W.SENSOR_HEIGHT)].astype(bool).any(2)
lab, _ = ndimage.label(free2); sizes = np.bincount(lab.ravel())[1:]
assert sizes.max() >= 0.95 * free2.sum(), (sizes.max(), free2.sum())
slab_expect = np.zeros(dims, bool); slab_expect[:, :, zc < W.FLOOR_T] = True; slab_expect[:, :, zc >= W.FLOOR_T + sp['wall_h']] = True
assert np.array_equal(ras['slab'], slab_expect)
shares = []
for s in (1, 2, 3):
    for i in range(4):
        r = W.rasterize(W.world_spec('TRAIN', 20 + i, s), dims); band = (zc >= 0.25) & (zc < 1.7)
        wl = (r['occ'][:, :, band].astype(bool) & ~r['obj'][:, :, band]).all(2)
        shares.append(r['obj'][:, :, band].any(2).sum() / max(1, (~wl).sum()))
assert 0.15 <= np.median(shares) <= 0.45, shares
# review M3 / S8: no floating furniture: object voxels in 26-connected components touching neither the floor slab nor a wall (wall
# cabinets hang on walls) must stay below 0.5 % over the same 12 houses
float_frac = []
for s in (1, 2, 3):
    for i in range(4):
        r = W.rasterize(W.world_spec('TRAIN', 20 + i, s), dims); support = r['slab'] | (r['occ'].astype(bool) & ~r['obj'] & ~r['slab'])
        lab3, nl = ndimage.label(r['obj'], structure=np.ones((3, 3, 3), bool))
        touch = ndimage.binary_dilation(support, structure=np.ones((3, 3, 3), bool)) & r['obj']
        ok_ids = np.unique(lab3[touch]); fl = r['obj'] & ~np.isin(lab3, ok_ids)
        float_frac.append(fl.sum() / max(1, r['obj'].sum()))
assert max(float_frac) < 0.005, float_frac
print(f'A. world deterministic, TEST closed, splits disjoint, sensor exact, furniture outside walls, agents on free columns, free floor connected, slab mask exact; furniture footprint share median {np.median(shares):.3f}; floating furniture max {max(float_frac):.4f} ({time.time()-t0:.0f}s)')

# ---- B. adapters
x = torch.randn(2, 3, 8, 6, 4)
for s in (1, 2):
    y = T.space_to_depth(x, s); assert torch.equal(T.depth_to_space(y, s), x)
    if s == 2:
        assert torch.equal(y[:, 2 * 8 + 1 * 4 + 0 * 2 + 1, 3, 2, 1], x[:, 2, 7, 4, 3])
print('B. anisotropic space_to_depth / depth_to_space inverse and index map')

# ---- C. join
own = torch.from_numpy(priv); n = own.shape[0]
model = T.SheafModel3D(dims, 1).eval()
T.CHART_M = fx['chart_m']
with torch.no_grad():
    mu_raw, q_raw, mask = model.head(model.trunk(own), own); mu = T.latent_quantize(mu_raw)
    f8 = T.rollout_max(mu, q_raw, mask, fx['edges'], rounds=8, stages=(8,))[8]; f16 = T.rollout_max(mu, q_raw, mask, fx['edges'], rounds=16, stages=(16,))[16]
    assert torch.equal(f8, f16)
    prior = T.LATENT_PRIOR / (n * T.D); ev = mask.double() * q_raw.double(); p_own = ev + prior; e_own = ev * mu.double()
    pw, ew, _ = T.kbit_wire_quantize(p_own, e_own); reached = torch.from_numpy(fx['reached'][4])
    for j in range(n):
        P = torch.stack([p_own[j]] + [pw[s] for s in range(n) if reached[j, s] and s != j]); E = torch.stack([e_own[j]] + [ew[s] for s in range(n) if reached[j, s] and s != j])
        tie = torch.arange(P.shape[0], dtype=P.dtype).view(-1, 1) * 1e-15; idx = (P - tie).argmax(0, keepdim=True)
        Pj, Ej = P.gather(0, idx)[0], E.gather(0, idx)[0]
        assert torch.allclose(torch.cat((T.latent_quantize((Ej / Pj).float()), (1. / Pj).log().float())), f8[j], atol=1e-6)
    perm = torch.randperm(n); inv = torch.argsort(perm)
    fp = T.rollout_max(mu[perm], q_raw[perm], mask[perm], [(int(inv[i]), int(inv[j]), w) for i, j, w in fx['edges']], rounds=8, stages=(8,))[8]
    assert torch.allclose(fp[inv], f8, atol=1e-6)
print('C. flooding == centralised max-precision join on all nodes / coordinates; idempotent; permutation-equivariant')

# ---- D. model
with torch.no_grad():
    k = int(torch.nonzero(mask[0] > 0)[0]); bx, by, bz = dims[0] // 8, dims[1] // 8, dims[2] // 4
    ix, rem = divmod(k, 8 * 4); iy, iz = divmod(rem, 4)
    own2 = own.clone(); outside = torch.ones(dims, dtype=torch.bool); outside[ix * bx:(ix + 1) * bx, iy * by:(iy + 1) * by, iz * bz:(iz + 1) * bz] = False
    own2[0][outside] = torch.randint(-1, 2, dims, dtype=torch.int8)[outside]
    m2, q2, _ = model.head(model.trunk(own2), own2)
    assert torch.equal(m2[0, k], mu_raw[0, k]) and torch.equal(q2[0, k], q_raw[0, k]), 'encoder must be block-local'
    assert (m2[0] != mu_raw[0]).sum() > 100
    kl, ol = model.decoder(f8); assert kl.shape == (n, int(np.prod(dims))) and ol.shape == kl.shape
    T.CHART_M = fx['chart_m'] + 5.; kl2, _ = model.decoder(f8); T.CHART_M = fx['chart_m']
    assert not torch.equal(kl, kl2), 'the chart-scale plane must reach the decoder'
    tgt = T.stream_targets(own, reached); slab = torch.from_numpy(fx['slab'].reshape(-1)); obj = torch.from_numpy(fx['obj'].reshape(-1))
    struct = torch.from_numpy((fx['occ'].astype(bool) & ~fx['obj'] & ~fx['slab']).reshape(-1))
    rows = T.metrics3d(tgt.clone(), tgt, dims, slab, obj, struct)
    assert all(r['furn_f1'] is None or abs(r['furn_f1'] - 1) < 1e-9 for r in rows)
    nofurn = torch.where((tgt == 1) & obj[None], torch.zeros_like(tgt), tgt)           # review M1: a perfect structure-only predictor
    rn = T.metrics3d(nofurn, tgt, dims, slab, obj, struct)
    assert all((r['furn_f1'] or 0.) < 0.05 for r in rn), [r['furn_f1'] for r in rn]
    assert all(abs(r['occ_f1'] - 1) < 1e-9 and abs(r['surface_f1'] - 1) < 1e-9 and abs(r['surface_f1_tol1'] - 1) < 1e-9 and abs(r['var_occ_f1'] - 1) < 1e-9 and r['accuracy'] == 1. and r['unknown_known_fpr'] == 0. for r in rows)
    assert all(r['obj_recall'] is None or abs(r['obj_recall'] - 1) < 1e-9 for r in rows)
    slab_pred = torch.where(tgt >= 0, slab.to(torch.int8)[None].expand_as(tgt), torch.full_like(tgt, -1))
    rs = T.metrics3d(slab_pred, tgt, dims, slab, obj, struct)
    print('    slab-only reference: surface F1 %.3f, variable-structure F1 %.3f' % (np.mean([r['surface_f1'] for r in rs]), np.mean([(r['var_occ_f1'] or 0.) for r in rs])))
    assert all((r['var_occ_f1'] or 0.) < 0.05 for r in rs)
params = {kk: sum(p.numel() for p in v) for kk, v in model.parameter_groups().items()}
opt = torch.optim.AdamW(model.parameters(), lr=1e-3); model.train(); losses = []
for step in range(3):
    opt.zero_grad(); m, q, mk = model.source(own); f = T.rollout_max(m, q, mk, fx['edges'], stages=(0,))
    l, _, _ = T.stage_loss(model.decoder, f[0], T.stream_targets(own, torch.from_numpy(fx['reached'][0])), dims); l.backward(); opt.step(); losses.append(float(l))
assert losses[-1] < losses[0], losses
print(f'D. block-local encoder, decoder shape, chart plane wired, perfect metrics = 1, slab-only var F1 ~ 0, loss decreases {[round(x, 4) for x in losses]}; params {params}')
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')

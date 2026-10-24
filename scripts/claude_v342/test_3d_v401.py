"""Math / implementation tests for the 3D Sheaf-GBP trainer v401 (Claude, 2026-10-24, written before any 128^3 batch).
A. Data equivalence: v401 at 64^3 regenerates the v400 fixture bit-for-bit (private maps, reached sets, positions, edges) -- so batch 3D-A
   is a valid 64^3 reference; at 128^3 the SAME scene has the same positions / edges / reached sets, the sensor is exact, beams = 65,536.
B. Adapters: space_to_depth / depth_to_space are inverses, identity at s = 1, and follow the documented index map (channel c s^3 + dx s^2 +
   dy s + dz at cell (i,j,k) <-> voxel (s i + dx, s j + dy, s k + dz)); pool_ternary equals an explicit numpy ternary-union loop.
C. Model equivalence at s = 1: SheafModel3D(64, seed) has the v400 state_dict key-for-key and bit-for-bit, and the same forward
   (source outputs and decoder logits) on a real fixture -- v401 at 64^3 IS v400.
D. Model at 128^3: the encoder is block-local at the voxel level (changing voxels outside block k, including sub-voxel patterns, leaves
   mu_k, q_k unchanged); unobserved blocks have mask 0; decoder output shape S^3 x 2; the join is idempotent at D = 256; perfect prediction
   scores 1 natively and pooled; skeleton-only painting scores ~0 variable-structure F1; three optimiser steps reduce the K0 loss.
CPU only.  python test_3d_v401.py
"""
import importlib.util, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapping_3d_data_v401 as d4
import train_3d_v401 as T
_s = importlib.util.spec_from_file_location('mapping_3d_data_v400', HERE / 'mapping_3d_data_v400.py'); d3 = importlib.util.module_from_spec(_s); _s.loader.exec_module(d3)
_s = importlib.util.spec_from_file_location('train_3d_v400', HERE / 'train_3d_v400.py'); T0 = importlib.util.module_from_spec(_s); _s.loader.exec_module(T0)

torch.manual_seed(0)
# ---------------------------------------------------------------- A. data
for split, idx in (('TRAIN', 7), ('VALIDATION', 1_000_000)):
    a = d4.make_fixture(split, idx, size=64, agents=8); b = d3.make_fixture(split, idx, size=64, agents=8)
    assert np.array_equal(a['private'], b['private']) and np.array_equal(a['reached'], b['reached']) and np.array_equal(a['positions_m'], b['positions_m']) and a['edges'] == b['edges'] and np.array_equal(a['occ'], b['occ']), (split, idx)
    assert a['world_id'] == b['world_id'] and a['world_id'].startswith('v400:')
print('A1. v401 at 64^3 == v400 at 64^3 bit-for-bit (private, reached, positions, edges, occ) on TRAIN 7 and VALIDATION 1,000,000')
f64 = d4.make_fixture('TRAIN', 7, size=64, agents=8); f128 = d4.make_fixture('TRAIN', 7, size=128, agents=8)
assert d4.beams(64) == 16384 and d4.beams(128) == 65536 and d4.beams(256) == 262144
assert np.array_equal(f128['positions_m'], f64['positions_m']) and np.array_equal(f128['trajectory_m'], f64['trajectory_m']) and f128['edges'] == f64['edges'] and np.array_equal(f128['reached'], f64['reached'])
occ, priv = f128['occ'], f128['private']
assert occ.shape == (128, 128, 128) and priv.shape == (8, 128, 128, 128)
assert all((occ[priv[a] == 1] == 1).all() and (occ[priv[a] == 0] == 0).all() for a in range(8))
# the 128^3 raster refines the 64^3 one: every occupied 64^3 voxel contains an occupied 128^3 sub-voxel (all solids are >= 0.2 m thick),
# while the any-pooled 128^3 raster is THICKER than the native 64^3 raster (solids straddling 0.2 m cell borders) -- hence pooled64 metrics
# are reported but never compared one-to-one with batch 3D-A
pooled_occ = f128['occ'].reshape(64, 2, 64, 2, 64, 2).max((1, 3, 5))
recall = (pooled_occ[f64['occ'] == 1] == 1).mean(); frac64, fracp = f64['occ'].mean(), pooled_occ.mean()
assert recall == 1.0 and fracp > frac64, (recall, frac64, fracp)
kf64 = (f64['private'] >= 0).mean(); kf128 = (priv >= 0).mean()
assert 0.5 < kf128 / kf64 < 1.5, (kf64, kf128)
print(f'A2. 128^3 fixture: same positions / forest / reached as 64^3; sensor exact; 65,536 beams; any-pooled 128^3 contains the 64^3 raster (recall {recall:.3f}; occupied fraction {frac64:.3f} -> {fracp:.3f}); known fraction 64 {kf64:.3f} vs 128 {kf128:.3f}')

# ---------------------------------------------------------------- B. adapters
x = torch.randn(2, 3, 8, 8, 8)
for s in (1, 2, 4):
    y = T.space_to_depth(x, s); assert y.shape == (2, 3 * s ** 3, 8 // s, 8 // s, 8 // s)
    assert torch.equal(T.depth_to_space(y, s), x), s
    if s > 1:
        m = 8 // s
        for (c, dx, dy, dz, i, j, k) in ((0, 1, 0, 1, m - 1, m - 2, 0), (2, s - 1, s - 1, s - 1, 1, 1, 1), (1, 0, 1, 0, 0, 0, m - 1)):
            assert torch.equal(y[:, c * s ** 3 + dx * s ** 2 + dy * s + dz, i, j, k], x[:, c, s * i + dx, s * j + dy, s * k + dz])
assert T.space_to_depth(x, 1) is x
flat = torch.randint(-1, 2, (3, 8 ** 3), dtype=torch.int8)
ref = np.full((3, 4 ** 3), -2, np.int8)
for n in range(3):
    v = flat[n].numpy().reshape(8, 8, 8)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                ref[n, i * 16 + j * 4 + k] = v[2 * i:2 * i + 2, 2 * j:2 * j + 2, 2 * k:2 * k + 2].max()
assert np.array_equal(T.pool_ternary(flat, 8, 2).numpy(), ref)
# tolerance-1 surface F1: a plane wall predicted one voxel off scores tol1 = 1 but surface F1 = 0; two voxels off scores tol1 = 0
S = 32; wall = torch.zeros(S, S, S, dtype=torch.int8); wall[10] = 1; tgt_w = wall.reshape(1, -1); no_skel = torch.zeros(S ** 3, dtype=torch.bool)   # 32 is not a native size: pass an empty skeleton
for shift, want in ((1, 1.), (2, 0.)):
    pw = torch.zeros(S, S, S, dtype=torch.int8); pw[10 + shift] = 1
    r = T.metrics3d(pw.reshape(1, -1), tgt_w, S, no_skel)[0]
    assert abs(r['surface_f1']) < 1e-9 and abs(r['surface_f1_tol1'] - want) < 1e-9, (shift, r)
r = T.metrics3d(tgt_w.clone(), tgt_w, S, no_skel)[0]; assert abs(r['surface_f1_tol1'] - 1) < 1e-9 and abs(r['surface_f1'] - 1) < 1e-9
# review M1: a first-hit-layer target (free in front of the wall, occupied wall layer, unknown shadow behind): a wall decoded one voxel
# INTO the shadow (x = 11, unknown in the target) or towards the sensor (x = 9) scores tol1 = 1; two voxels off or missing scores 0
fh = torch.full((S, S, S), -1, dtype=torch.int8); fh[:10] = 0; fh[10] = 1; tgt_fh = fh.reshape(1, -1)
def pred_wall(x):
    pw = torch.zeros(S, S, S, dtype=torch.int8); pw[x] = 1; return pw.reshape(1, -1)
for x, want in ((11, 1.), (9, 1.), (12, 0.), (10, 1.)):
    r = T.metrics3d(pred_wall(x), tgt_fh, S, no_skel)[0]
    assert abs(r['surface_f1_tol1'] - want) < 1e-9, (x, r)
assert abs(T.metrics3d(pred_wall(10), tgt_fh, S, no_skel)[0]['surface_f1'] - 1) < 1e-9 and abs(T.metrics3d(pred_wall(11), tgt_fh, S, no_skel)[0]['surface_f1']) < 1e-9
r = T.metrics3d(torch.zeros(1, S ** 3, dtype=torch.int8), tgt_fh, S, no_skel)[0]; assert abs(r['surface_f1_tol1']) < 1e-9 and abs(r['surface_f1']) < 1e-9   # missing wall
# a fully known 3-layer slab (interior layer included) shifted by one voxel scores tol1 = 1 (reviewer's case), the exact slab scores 1 at both tolerances
slab = torch.zeros(S, S, S, dtype=torch.int8); slab[:, :, 10:13] = 1; tgt_sl = slab.reshape(1, -1)
sh = torch.zeros(S, S, S, dtype=torch.int8); sh[:, :, 11:14] = 1
assert abs(T.metrics3d(sh.reshape(1, -1), tgt_sl, S, no_skel)[0]['surface_f1_tol1'] - 1) < 1e-9
r = T.metrics3d(tgt_sl.clone(), tgt_sl, S, no_skel)[0]; assert abs(r['surface_f1_tol1'] - 1) < 1e-9 and abs(r['surface_f1'] - 1) < 1e-9
print('B. space_to_depth / depth_to_space inverse + index map; pool_ternary == explicit ternary-union loop; tolerance-1 surface F1 semantics (incl. first-hit-layer target)')

# ---------------------------------------------------------------- C. model equivalence at s = 1
m1 = T.SheafModel3D(64, 1).eval(); m0 = T0.SheafModel3D(64, 1).eval()
sd1, sd0 = m1.state_dict(), m0.state_dict()
assert list(sd1.keys()) == list(sd0.keys()) and all(torch.equal(sd1[k], sd0[k]) for k in sd0), 'v401 at 64^3 must have the v400 weights bit-for-bit'
own64 = torch.from_numpy(f64['private'])
with torch.no_grad():
    mu1, q1, mk1 = m1.source(own64); mu0, q0, mk0 = m0.source(own64)
    assert torch.equal(mu1, mu0) and torch.equal(q1, q0) and torch.equal(mk1, mk0)
    feats = T.rollout_max(mu1, q1, mk1, f64['edges'], stages=(8,))[8]
    feats0 = T0.rollout_max(mu0, q0, mk0, f64['edges'], stages=(8,))[8]
    assert torch.equal(feats, feats0)
    k1, o1 = m1.decoder(feats); k0, o0 = m0.decoder(feats0)
    assert torch.equal(k1, k0) and torch.equal(o1, o0)
print('C. SheafModel3D(64) == v400 bit-for-bit: state_dict, source, join, decoder logits')

# ---------------------------------------------------------------- D. model at 128^3
n = 8; own = torch.from_numpy(priv)
model = T.SheafModel3D(128, 1).eval()
params = {k: sum(p.numel() for p in v) for k, v in model.parameter_groups().items()}
assert model.trunk.net[0].in_channels == 27 and model.decoder.output.out_channels == 16
with torch.no_grad():
    mu, q, mask = model.source(own)
    assert mu.shape == (n, 256) and float(mask.mean()) > 0
    k = int(torch.nonzero(mask[0] > 0)[0]); bx, by, bz = 128 // 8, 128 // 8, 128 // 4
    ix, rem = divmod(k, 8 * 4); iy, iz = divmod(rem, 4)
    own2 = own.clone()
    outside = torch.ones(128, 128, 128, dtype=torch.bool); outside[ix * bx:(ix + 1) * bx, iy * by:(iy + 1) * by, iz * bz:(iz + 1) * bz] = False
    noise = torch.randint(-1, 2, (128, 128, 128), dtype=torch.int8)        # arbitrary sub-voxel patterns outside block k
    own2[0][outside] = noise[outside]
    # locality is checked on the RAW head outputs: at initialisation the hard-8 quantised mean collapses to one level (|mu| ~ 0.05 < 1/127
    # everywhere), so an equality test on quantised mu would be vacuous; raw mu and q are continuous
    mu_raw, q_raw, _ = model.head(model.trunk(own), own); mu2, q2, mask2 = model.head(model.trunk(own2), own2)
    assert torch.equal(mu2[0, k], mu_raw[0, k]) and torch.equal(q2[0, k], q_raw[0, k]), 'encoder must be block-local at 128^3'
    changed = ((mu2[0] != mu_raw[0]).sum().item(), (q2[0] != q_raw[0]).sum().item()); assert min(changed) > 200, changed  # and other blocks do see the change
    unobs = torch.nonzero(mask[0] == 0)
    assert len(unobs) > 0 and float(mask[0, unobs[0]]) == 0.
    edges = f128['edges']
    feats8 = T.rollout_max(mu, q, mask, edges, rounds=8, stages=(8,))[8]
    feats16 = T.rollout_max(mu, q, mask, edges, rounds=16, stages=(16,))[16]
    assert torch.equal(feats8, feats16), 'join must be idempotent at D = 256'
    known_logit, occ_logit = model.decoder(feats8)
    assert known_logit.shape == (n, 128 ** 3) and occ_logit.shape == (n, 128 ** 3)
    # the readout is a true block readout: perturbing one compute cell's features moves exactly its 2x2x2 voxels (1x1 output conv + depth_to_space)
    h = torch.randn(1, model.decoder.output.in_channels, 64, 64, 64); h2 = h.clone(); h2[0, :, 5, 6, 7] += 1.
    o1 = T.depth_to_space(model.decoder.output(h), 2); o2 = T.depth_to_space(model.decoder.output(h2), 2)
    moved = (o1 != o2).any(1)[0]
    assert moved.sum() == 8 and moved[10:12, 12:14, 14:16].all(), int(moved.sum())
    reached = torch.from_numpy(f128['reached'][4])
    tgt = T.stream_targets(own, reached)
    rows = T.metrics3d(tgt.clone(), tgt, 128)
    assert all(abs(r['occ_f1'] - 1) < 1e-9 and abs(r['surface_f1'] - 1) < 1e-9 and abs(r['surface_f1_tol1'] - 1) < 1e-9 and abs(r['var_occ_f1'] - 1) < 1e-9 and r['accuracy'] == 1. and r['unknown_known_fpr'] == 0. for r in rows)
    skel128 = T.skeleton_flat(128, tgt.device); skel_pooled = T.pool_ternary(skel128.to(torch.int8)[None], 128, 2)[0] == 1
    assert skel128.shape == (128 ** 3,) and 0.1 < skel128.float().mean() < 0.3
    pt = T.pool_ternary(tgt, 128, 2)
    rows_p = T.metrics3d(pt.clone(), pt, 64, skel_pooled)
    assert all(abs(r['occ_f1'] - 1) < 1e-9 and abs(r['surface_f1'] - 1) < 1e-9 and abs(r['var_occ_f1'] - 1) < 1e-9 for r in rows_p)
    skel_pred = torch.where(tgt >= 0, skel128.to(torch.int8)[None].expand_as(tgt), torch.full_like(tgt, -1))
    rs = T.metrics3d(skel_pred, tgt, 128)
    print('    skeleton-only reference at 128^3 on this scene: surface F1 %.3f, variable F1 %.3f' % (np.mean([r['surface_f1'] for r in rs]), np.mean([r['var_occ_f1'] for r in rs])))
    assert all(r['var_occ_f1'] < 0.05 for r in rs)
print(f"D1. 128^3: block-local encoder (sub-voxel level), mask semantics, idempotent join, decoder shape, perfect metrics native + pooled; params {params}")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train(); losses = []
for step in range(3):
    opt.zero_grad()
    mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, edges, stages=(0,))
    l0, _, _ = T.stage_loss(model.decoder, feats[0], T.stream_targets(own, torch.from_numpy(f128['reached'][0])), 128)
    l0.backward(); opt.step(); losses.append(float(l0))
assert losses[-1] < losses[0], losses
print('D2. three optimiser steps reduce the K0 loss at 128^3:', [round(x, 4) for x in losses])
print('ALL TESTS PASSED')

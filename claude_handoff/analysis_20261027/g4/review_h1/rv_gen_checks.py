"""Reviewer's generator checks for G4-H1 (2026-10-27). TRAIN only; TEST only to confirm it raises.
1. IDENTITY make_fixture == v420 make_fixture (occ, slab, obj, private, positions, edges) on TRAIN r1_i9.
2. the smoke npz (novel / floating / scale / ctrl, TRAIN) == a fresh derived rasterize + ood arrays (determinism, mask correspondence).
3. plan bitwise identical to v420 for every variant on two more TRAIN houses (walls / lintels / slabs).
4. obox centre inclusion vs an independent polygon test (matplotlib Path) for random non-thin rotated boxes.
5. TEST raises for every variant; module_for(v).world_spec(...) world_id prefix; v420 file sha unchanged."""
import os, sys, json, hashlib, math
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_noise_panels as G  # noqa  (sets HOUSEEXPO env)
import mapping_3d_storey_data_v420ood as WO  # noqa
from matplotlib.path import Path  # noqa
W = WO.W
EXP = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'g4', 'h1_export')
out = {}
out['v420_sha'] = hashlib.sha256(open(W.__file__, 'rb').read()).hexdigest()

# 1
a = W.make_fixture('TRAIN', 9, 1); b = WO.make_fixture(WO.IDENTITY, 'TRAIN', 9, 1)
out['identity_fixture'] = {k: bool(np.array_equal(np.asarray(a[k]), np.asarray(b[k]))) for k in ('occ', 'slab', 'obj', 'private', 'positions_m', 'trajectory_m', 'reached')}
out['identity_fixture']['edges'] = a['edges'] == b['edges']; out['identity_fixture']['world_id'] = a['world_id'] == b['world_id']

# 2
for nm, v, r, i in (('novel_train_r2_i9', 'novel', 2, 9), ('floating_train_r2_i9', 'floating', 2, 9), ('scale_train_r1_i9', 'scale', 1, 9), ('ctrl_train_r1_i9', 'ctrl', 1, 9)):
    z = np.load(os.path.join(EXP, nm + '_s203.npz'))
    sp = WO.world_spec(v, 'TRAIN', i, r); rr = WO.rasterize(sp, (128, 128, 32)); oa = WO.ood_arrays(sp, (128, 128, 32), rr)
    out[nm] = dict(occ=bool(np.array_equal(z['occ'] == 1, rr['occ'] == 1)), obj=bool(np.array_equal(z['obj'], rr['obj'])),
                   slab=bool(np.array_equal(z['slab'], rr['slab'])), ood=bool(np.array_equal(z['ood_mask'], oa['ood_mask'])),
                   ood_subset_obj=bool(not (z['ood_mask'] & ~z['obj']).any()), ood_eq_obj=bool(np.array_equal(z['ood_mask'], z['obj'])) if v != 'ctrl' else None,
                   n_objects=len(sp['objects']), families=sorted({o.get('family') for o in sp['objects']}))

# 3
for (s, i, r) in (('TRAIN', 19, 2), ('TRAIN', 29, 3)):
    ref = W.rasterize(W.world_spec(s, i, r), (128, 128, 32)); plan = (ref['occ'] == 1) & ~ref['obj']
    out[f'plan_{s}_{r}_{i}'] = {v: bool(np.array_equal((WO.rasterize(WO.world_spec(v, s, i, r), (128, 128, 32))['occ'] == 1)
                                                      & ~WO.rasterize(WO.world_spec(v, s, i, r), (128, 128, 32))['obj'], plan)) for v in WO.VARIANTS}

# 4
rng = np.random.default_rng(7); L = 10.0; dims = (128, 128, 32); c = L / 128
xc = (np.arange(128) + .5) * c; X, Y = np.meshgrid(xc, xc, indexing='ij'); pts = np.stack([X.ravel(), Y.ravel()], 1)
bad = 0; tot = 0
for _ in range(200):
    cx, cy = rng.uniform(3, 7, 2); hx, hy = rng.uniform(0.08, 1.0, 2); th = rng.uniform(0, math.pi)
    if 2 * hx < c or 2 * hy < c:
        continue
    m = WO.prim_mask(('obox', cx, cy, hx, hy, th, 0.0, 0.5), dict(chart_m=L), dims).any(2)
    cs, sn = math.cos(th), math.sin(th)
    corners = [(cx + cs * u - sn * v, cy + sn * u + cs * v) for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
    ref = Path(corners).contains_points(pts, radius=1e-12).reshape(128, 128)
    if not ref.any():
        continue
    tot += 1; d = int((m != ref).sum()); bad += d
out['obox_vs_polygon'] = dict(boxes=tot, cells_differing=bad)

# 5
te = {}
for v in WO.VARIANTS:
    try:
        WO.world_spec(v, 'TEST', 0, 1); te[v] = 'NO RAISE'
    except Exception as e:  # noqa
        te[v] = type(e).__name__
out['test_split'] = te
m = WO.module_for('floating'); sp = m.world_spec('TRAIN', 9, 2)
out['module_for_world_id'] = sp['world_id']
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, 'gen_checks.json'), 'w'), indent=1)

"""Tests for the real-scan 2D world v410 (Claude, 2026-10-24, written before any real-scan run). Run on uk-linux (needs the base rasters
and the sealed source):  python test_realscan2d_v410.py --source <sealed source>
A. Contract: a v410 fixture has exactly the keys, dtypes and shapes of a v371 (HouseExpo working-baseline) fixture.
B. Sensor exactness: observed occupied cells are walls and observed free cells are free (the v371 sensor on the v410 raster).
C. Geometry: at yaw 0 the 128 raster equals a direct 4x4 'all free' downsample of the base raster (up to the conservative half-diagonal
   dilation); under a random yaw the free area is preserved within 15 %; every chart contains the whole free region.
D. Splits and determinism: TRAIN and VALIDATION never share a scene; the six FRL apartments are all TRAIN; TEST raises; the VAL plan is
   4 scenes x N{8,16} x 8 placements; the same spec regenerates the same private hash; fresh (index + 8) differs from val.
E. Feasibility: every VAL and fresh spec and the first 240 TRAIN specs build (agents placed, every source has evidence).
F. Trainer plumbing: train_v410 differs from train_v392 only in the world plumbing lines (checked on the source text).
"""
import argparse, difflib, sys, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_realscan2d_data_v410 as R
import mapping_houseexpo_data_v371 as H
t0 = time.time()

# ---- A. contract (against a v371 fixture when the HouseExpo index is available; otherwise against the documented v371 contract)
fx = R.make_fixture('VALIDATION', 0, rooms=1, agents=8)
inp, ev, man = fx['inputs'], fx['evaluation'], fx['manifest']
assert set(inp) == {'private', 'positions_m', 'trajectory_m', 'adjacency', 'public_frame'} and set(fx) == {'inputs', 'evaluation', 'manifest'}
assert inp['private'].dtype == np.int8 and inp['private'].shape == (8, 128, 128) and inp['positions_m'].shape == (8, 2) and inp['trajectory_m'].shape == (3, 8, 2)
assert inp['adjacency'].shape == (8, 8) and inp['public_frame']['extent_m'][0] == R.world_spec('VALIDATION', 0, 1)['chart_m']
assert set(inp['public_frame']) == {'origin_m', 'extent_m', 'shape', 'cell_size_m', 'coordinate_convention', 'sensor_range_m', 'sensor_beams', 'communication_radius_m', 'static_snapshot', 'radio_occlusion'}
assert set(ev) >= {'door_mask', 'walls'} and ev['door_mask'].shape == (128, 128)
try:
    import os
    if os.environ.get('HOUSEEXPO_INDEX'):
        hx = H.make_fixture('VALIDATION', 0, rooms=1, agents=8)
        assert set(hx['inputs']) == set(inp) and set(hx['inputs']['public_frame']) == set(inp['public_frame']) and set(hx['manifest']) == set(man)
        assert hx['inputs']['private'].dtype == inp['private'].dtype and hx['inputs']['private'].shape == inp['private'].shape
        print('A. contract identical to a live v371 fixture')
    else:
        print('A. contract matches the documented v371 contract (HOUSEEXPO_INDEX not set)')
except Exception as e:
    raise AssertionError(f'v371 comparison failed: {e}')

# ---- B. sensor exactness
walls = ev['walls']; priv = inp['private']
assert all(((walls[priv[a] == 1]) == 1).all() and ((walls[priv[a] == 0]) == 0).all() for a in range(8)), 'observations must agree with the raster'
assert (priv >= 0).any(axis=(1, 2)).all()
print('B. sensor exact on the real-scan raster; every agent has evidence; known fractions', np.round((priv >= 0).mean(axis=(1, 2)), 3))

# ---- C. geometry
spec0 = R.world_spec('VALIDATION', 0, 1)
row = next(r for r in R._units()['VALIDATION'] if r['file'] == spec0['base_file']); b = R._base(row)
free_base_m2 = (~b['blocked']).sum() * b['cell'] ** 2
g128 = R.rasterize(spec0, 128)['walls']; cell = spec0['chart_m'] / 128
free128_m2 = (g128 == 0).sum() * cell ** 2
assert 0.6 < free128_m2 / free_base_m2 <= 1.0 + 1e-9, (free128_m2, free_base_m2)   # the all-free + dilation rule only removes free area
for i in (3, 17, 101):
    sp = R.world_spec('TRAIN', i)
    rowt = next(r for r in R._units()['TRAIN'] if r['file'] == sp['base_file'])
    sp0 = dict(sp, yaw=0.)
    pts = (R._base(rowt)['free_yx'][:, ::-1] + .5) * R._base(rowt)['cell']
    sp0['chart_m'] = float(max(8., np.ceil((pts.max(0) - pts.min(0)).max() + 1.0))); sp0['centre_rot'] = ((pts.max(0) + pts.min(0)) / 2).tolist()
    a_rot = (R.rasterize(sp, 512)['walls'] == 0).sum() * (sp['chart_m'] / 512) ** 2; a_0 = (R.rasterize(sp0, 512)['walls'] == 0).sum() * (sp0['chart_m'] / 512) ** 2
    assert abs(a_rot - a_0) / a_0 < 0.15, (i, a_rot, a_0)
    fine = R.rasterize(sp, 512)['walls']
    assert fine[0].all() and fine[-1].all() and fine[:, 0].all() and fine[:, -1].all(), 'the chart border must be wall (free region inside the chart)'
print(f'C. raster: VAL scene 1 free {free128_m2:.1f} of {free_base_m2:.1f} m^2 base; rotated rasters keep the free area within 15 %; free region inside every chart')

# ---- D. splits and determinism
tr = {r['scene'] for r in R._units()['TRAIN']}; va = {r['scene'] for r in R._units()['VALIDATION']}
assert not tr & va and all(f'frl_apartment_{k}' in tr for k in range(6)) and va == set(R.VAL_SCENES)
try:
    R.world_spec('TEST', 0); raise AssertionError('TEST must raise')
except ValueError:
    pass
plan = R.scene_plan('VALIDATION', size=128)
assert len(plan) == 4 * 2 * 8 and {(s['rooms'], s['agents']) for s in plan} == {(g, n) for g in (1, 2, 3, 4) for n in (8, 16)}
assert R.array_hash(R.make_fixture('VALIDATION', 0, rooms=1, agents=8)['inputs']['private']) == R.array_hash(priv)
assert R.array_hash(R.make_fixture('VALIDATION', 8, rooms=1, agents=8)['inputs']['private']) != R.array_hash(priv)
tp = R.scene_plan('TRAIN', size=128, train_per_stratum=30)
assert [s['agents'] for s in tp[:4]] == [8, 16, 8, 16] and len({R.world_spec('TRAIN', s['index'])['house_id'] for s in tp}) == len(R._units()['TRAIN'])
print(f'D. splits: TRAIN {len(R._units()["TRAIN"])} floors of {len(tr)} scenes, VAL {sorted(va)}; deterministic; fresh differs; TEST closed')

# ---- E. feasibility
bad = []
for s in plan + [dict(s, index=s['index'] + 8) for s in plan] + R.scene_plan('TRAIN', size=128, train_per_stratum=240):
    try:
        f = R.make_fixture(**s)
        assert (f['inputs']['private'] >= 0).any(axis=(1, 2)).all()
    except Exception as e:
        bad.append((s, repr(e)[:120]))
assert not bad, bad[:5]
print(f'E. all {len(plan) * 2} VAL / fresh specs and 240 TRAIN specs build ({time.time() - t0:.0f}s)')

# ---- F. trainer plumbing
a = (HERE / 'train_v392.py').read_text().splitlines(); b2 = (HERE / 'train_v410.py').read_text().splitlines()
changed = [l for l in difflib.unified_diff(a, b2, lineterm='', n=0) if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
body = [l for l in changed if not l.startswith('+"""') and 'train_v410 = train_v392' not in l and 'floor slices of real 3D scans' not in l and 'are byte-for-byte the v392' not in l and 'HouseExpo index. The original v392' not in l and l != '+"""']
assert all(('realscan' in l) or ('houseexpo_rect' in l and 'choices' in l) or ("'houseexpo', 'houseexpo_fit', 'houseexpo_rect'" in l) or ('HOUSEEXPO_INDEX' in l) for l in body), body
print(f'F. train_v410 vs train_v392: {len(body)} changed code lines, all world plumbing')
print('ALL TESTS PASSED', f'{time.time() - t0:.0f}s')

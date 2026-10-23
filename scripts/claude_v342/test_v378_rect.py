"""Tests for the rectangular house-fitted chart v378 (Claude, 2026-10-23). CPU; TRAIN/VALIDATION only.
1. make_rays_aniso(5 m, 256, c, c) == v121 make_rays(5 / c, 256) exactly for isotropic cells (several c).
2. narrow_mask(walls, c, c) == v365.narrow_mask(walls, c) exactly.
3. square=True: every fixture field equals v371 exactly (private, trajectory, adjacency, walls, door mask) on TRAIN and VAL scenes.
4. rectangular chart (+ 6. paired poses, review M1): house inside the chart, aspect <= 2, bbox coverage higher than the square chart, poses on free cells with the
   0.45 m metric clearance, every agent observes something, every observed cell centre lies within 5 m (metric) of one of the agent's
   poses, and a ray along +x / +y covers 5/cx resp. 5/cy cells.
"""
import os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get('SEALED_SOURCE', HERE.parent / 'handoff_src/claude_handoff/reference_source_v310')).resolve()
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(SRC / 'exploration_grid')); sys.path.insert(0, str(HERE))
from scripts.prepare_houseexpo_pilot_v121 import make_rays
import mapping_houseexpo_data_v365 as v365
import mapping_houseexpo_data_v371 as v371
import mapping_houseexpo_data_v378 as v378

for c in (20 / 128, 11 / 128, 14 / 128, 0.1, 9 / 128):
    a = v378.make_rays_aniso(5.0, 256, c, c); b = make_rays(5.0 / c, 256)
    assert a.shape == b.shape and np.array_equal(a, b), c
print('1 rays: isotropic equivalence OK')

specs = [('TRAIN', 0, 1, 8), ('TRAIN', 7, 2, 16), ('TRAIN', 13, 3, 8), ('VALIDATION', 0, 1, 16), ('VALIDATION', 3, 2, 8), ('VALIDATION', 5, 3, 16)]
for split, idx, rooms, n in specs:
    f1 = v371.make_fixture(split, idx, 128, rooms, n); f2 = v378.make_fixture(split, idx, 128, rooms, n, square=True)
    for key in ('private', 'trajectory_m', 'adjacency', 'positions_m'):
        assert np.array_equal(f1['inputs'][key], f2['inputs'][key]), (split, idx, key)
    for key in ('walls', 'door_mask'):
        assert np.array_equal(f1['evaluation'][key], f2['evaluation'][key]), (split, idx, key)
    w = f1['evaluation']['walls']; c = f1['inputs']['public_frame']['cell_size_m']
    assert np.array_equal(v378.narrow_mask(w, c, c), v365.narrow_mask(w, c))
print('2/3 square mode == v371 (and narrow_mask == v365) OK')

cov_sq, cov_rect = [], []
for split, idx, rooms, n in specs + [('TRAIN', i, r, 8 if i % 2 == 0 else 16) for i in range(20, 26) for r in (1, 2, 3)]:
    fx = v378.make_fixture(split, idx, 128, rooms, n); sp = v378.world_spec(split, idx, rooms)
    lx, ly = sp['chart_xy_m']; v = np.asarray(sp['verts_m'])
    assert (v >= 0).all() and (v[:, 0] <= lx).all() and (v[:, 1] <= ly).all() and max(lx, ly) <= 2 * min(lx, ly)
    ext = np.ptp(v, 0); L = max(8., np.ceil(ext.max() + 1)); cov_sq.append(ext.prod() / L ** 2); cov_rect.append(ext.prod() / (lx * ly))
    cx, cy = lx / 128, ly / 128
    priv = fx['inputs']['private']; traj = fx['inputs']['trajectory_m']; walls = fx['evaluation']['walls']
    rc = np.stack([np.floor(traj[..., 1] / cy), np.floor(traj[..., 0] / cx)], -1).astype(int)
    assert (walls[rc[..., 0], rc[..., 1]] == 0).all()
    assert (priv >= 0).any(axis=(1, 2)).all()
    yy, xx = np.mgrid[0:128, 0:128]; X = (xx + .5) * cx; Y = (yy + .5) * cy
    for a in range(priv.shape[0]):
        d = np.min([(X - traj[s, a, 0]) ** 2 + (Y - traj[s, a, 1]) ** 2 for s in range(3)], 0)
        assert (d[priv[a] >= 0] <= 25 + 1e-9).all()
    assert fx['inputs']['public_frame']['extent_m'] == [lx, ly]
print(f'4 rectangular charts OK; bbox coverage median square {np.median(cov_sq):.2f} -> rect {np.median(cov_rect):.2f}')

# 6 (review M1) paired poses: rect positions == v371 positions translated, same adjacency (radio graph); houses whose chart does not
#   change get bit-identical private maps (placebo)
import mapping_houseexpo_data_v371 as v371b
from scripts import mapping_native_bank_v306 as bank
same = 0
for split, idx, rooms, n in specs + [('VALIDATION', i, r, nn) for i in range(8) for r in (1, 2, 3) for nn in (8, 16)]:
    fr = v378.make_fixture(split, idx, 128, rooms, n); fs = v371b.make_fixture(split, idx, 128, rooms, n)
    sp = v378.world_spec(split, idx, rooms); L = v371b.world_spec(split, idx, rooms)['chart_m']; lx, ly = sp['chart_xy_m']
    assert np.allclose(fr['inputs']['trajectory_m'] - np.array([lx / 2 - L / 2, ly / 2 - L / 2]), fs['inputs']['trajectory_m'], atol=1e-9)
    assert np.array_equal(fr['inputs']['adjacency'], fs['inputs']['adjacency'])
    f1 = bank.select_forest(fr['inputs']['positions_m'], fr['inputs']['adjacency'], 8.)['edges']; f2 = bank.select_forest(fs['inputs']['positions_m'], fs['inputs']['adjacency'], 8.)['edges']
    assert f1 == f2
    if lx == ly == L:
        assert np.array_equal(fr['inputs']['private'], fs['inputs']['private']) and np.array_equal(fr['evaluation']['walls'], fs['evaluation']['walls']); same += 1
print(f'6 paired poses / identical radio graph and forest OK; {same} unchanged-chart scenes bit-identical')

r = v378.make_rays_aniso(5.0, 256, 0.15, 0.08)
beam_x = r[:, 0]; beam_y = r[:, 64]                                  # angle 0 (+x, columns) and pi/2 (+y, rows)
bx = beam_x[beam_x[:, 0] != 32767]; by = beam_y[beam_y[:, 0] != 32767]
assert bx[:, 0].max() == 0 and abs(bx[:, 1].max() - 5 / .08) <= 1, bx[-1]
assert by[:, 1].max() == 0 and abs(by[:, 0].max() - 5 / .15) <= 1, by[-1]
print('5 anisotropic ray lengths OK; ALL OK')

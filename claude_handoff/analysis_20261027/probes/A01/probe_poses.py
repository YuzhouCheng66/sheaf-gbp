"""probe_poses.py (A01, 2026-10-27): regenerate the EXACT sensor poses of the demo agents and check that observe(gt, poses) reproduces
the demo own maps bit for bit. Same RNG stream as mapping_3d_storey_data_v420.make_fixture: rng = default_rng(_seed(world_id + ':poses:8')),
centres = agent_positions(R1 occupancy, ...), jitter = rng.uniform(-.1, .1, (3, 8, 3)) with z = 0. world_id = 'v420:VALIDATION:<house_id>',
house_id from the local HouseExpo index (lists per (split, stratum) in file order, as v365._index). Writes poses_demo.json.
Usage: python probe_poses.py DATA_DIR INDEX_JSON OUT_JSON"""
import glob, hashlib, json, math, os, sys
import numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from rate_baselines_lib import load  # noqa: E402
from probe_visorder import fibonacci_sphere, first_visits  # noqa: E402  (module has no top-level side effects)

FLOOR_T, SENSOR_HEIGHT = 0.10, 1.2


def _seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def agent_positions(occ_r, cell, agents, rng):
    cx, _, cz = cell
    zc = (np.arange(occ_r.shape[2]) + .5) * cz
    band = (zc >= FLOOR_T) & (zc < FLOOR_T + SENSOR_HEIGHT + 0.3)
    free2 = ~occ_r[:, :, band].any(2)
    lab, nlab = ndimage.label(free2)
    comp = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
    clear = ndimage.distance_transform_edt(free2) * cx
    cand = np.argwhere(comp & (clear >= 0.45))
    order = rng.permutation(len(cand))
    for min_sep in (1.0, 0.5):
        pts = []
        for i in order:
            p = np.array([(cand[i][0] + .5) * cx, (cand[i][1] + .5) * cx])
            if all(np.linalg.norm(p - q) >= min_sep for q in pts):
                pts.append(p)
            if len(pts) == agents:
                return np.array([[q[0], q[1], FLOOR_T + SENSOR_HEIGHT] for q in pts])
    raise ValueError('could not place agents')


DATA, INDEX, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
idx = json.load(open(INDEX)); lists = {}
for r in idx['houses']:
    lists.setdefault((r['split'], r['stratum']), []).append(r)
rays = fibonacci_sphere(65536)
res = {}
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path)
    hid = lists[('VALIDATION', int(m['rooms']))][int(m['index'])]['id']
    rng = np.random.default_rng(_seed(f'v420:VALIDATION:{hid}' + ':poses:8'))
    cell = tuple(m['cell_m'])
    centres = agent_positions(gt['occ'].astype(bool), cell, 8, rng)
    jitter = rng.uniform(-.1, .1, (3, 8, 3)); jitter[:, :, 2] = 0.
    traj = centres[None] + jitter
    dpos = float(np.abs(np.round(centres, 3) - np.asarray(m['positions_m'])).max())
    mism = []
    for a in range(8):
        o, _ = first_visits(gt['occ'].astype(np.uint8), cell, traj[:, a], rays)
        mism.append(int((o != own[a]).sum()))
    res[m['house']] = dict(house_id=hid, centre_max_abs_diff=dpos, own_mismatch_voxels=mism, trajectory_m=traj.tolist())
    print(m['house'], hid, 'centre diff', dpos, 'mismatch', mism, flush=True)
json.dump(res, open(OUT, 'w'), indent=1)

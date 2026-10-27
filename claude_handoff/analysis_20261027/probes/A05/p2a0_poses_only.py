"""A05 probe 2a0: the pose part of p2a only (no ray casting), so p2b / p4 can start while p2a verifies; same code path as p2a."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from p2a_poses_exact import W, seed, load, INDEX, ROOT, structured_G  # noqa: E402

idx = json.load(open(INDEX)); lists = {}
for r in idx['houses']:
    lists.setdefault((r['split'], r['stratum']), []).append(r)
out = {}
for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]
    hid = lists[('VALIDATION', int(m['rooms']))][int(m['index'])]['id']
    rng = np.random.default_rng(seed(f'v420:VALIDATION:{hid}' + f':poses:{n}'))
    L = float(m['chart_m']); cell = (L / 128, L / 128, 3.2 / 32)
    centres = W.agent_positions(None, gt['occ'].astype(bool), cell, n, rng)
    jitter = rng.uniform(-.1, .1, (W.POSES, n, 3)); jitter[:, :, 2] = 0.
    assert np.abs(centres - np.asarray(m['positions_m'])).max() < 2e-3
    _, lk = structured_G(own[0] == 1, gt)
    out[m['house']] = dict(house_id=hid, chart_m=L, cell=cell, trajectory_m=(centres[None] + jitter).tolist(), lintel_k=lk)
    print(m['house'], 'ok', flush=True)
json.dump(out, open(os.path.join(HERE, 'poses.json'), 'w'), indent=1)

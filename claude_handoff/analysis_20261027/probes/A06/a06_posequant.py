"""A06: how precisely must the poses be sent for the ray-cast decode to stay exact? (Claude A06, 2026-10-27)
Re-simulated own maps (a06_raycast jitter seed) for 2 houses x 4 agents; decode = ray-cast through the map's own O from poses rounded to
1 cm / 1 mm / 0.1 mm (x, y; z is the fixed sensor height) -> voxels that differ from the map, and ctx bytes of the correction set."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a06_common import houses, scene
from a06_raycast import observe, fibonacci_sphere, ctx_bin, BEAMS, POSES
rays = fibonacci_sphere(BEAMS); out = []
for path in houses()[:2]:
    S = scene(path); m = S['m']; n = m['agents']; W = S['occ'].astype(np.uint8); cell = np.asarray(m['cell_m'], np.float64)
    rng = np.random.default_rng(int.from_bytes(m['house'].encode(), 'little') % (2 ** 32))
    centres = np.asarray(m['positions_m'], np.float64); jit = rng.uniform(-.1, .1, (POSES, n, 3)); jit[:, :, 2] = 0.; traj = centres[None] + jit
    for a in range(4):
        own = observe(W, cell, traj[:, a], rays); O = (own == 1).astype(np.uint8)
        for q in (0.01, 0.001, 0.0001):
            tq = traj[:, a].copy(); tq[:, :2] = np.round(tq[:, :2] / q) * q
            re = observe(O, cell, tq, rays); diff = re != own
            out.append(dict(house=m['house'], agent=a, q=q, diff=int(diff.sum()), known=int((own >= 0).sum()), corr_B=ctx_bin(diff) if diff.any() else 0.))
            print(out[-1], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'posequant.json'), 'w'), indent=1)

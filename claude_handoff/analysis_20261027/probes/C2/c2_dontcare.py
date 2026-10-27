"""c2_dontcare.py (moderator C2, 2026-10-27): independent check of the 'don't-care' generalisation every exact design relies on.
The coordinator's verify_recast tested the smallest consistent world W = O (observed-occupied voxels).  Here the two other extremes:
  W_max  = NOT F_a           (every voxel the agent did not see free, i.e. all unknown space filled solid),
  W_rand = O_a + a random half of the agent's unknown voxels,
  W_uni  = NOT F_union        (the R8 union's free set: one world for the whole component, rendered from each agent's own poses),
each ray-cast with the generator's own observe() from the TRUE float64 poses, compared voxel for voxel with the agent's own map.
Uses A03's locally regenerated fixtures (probes/A03/fx/<house>.npz: private maps, trajectory, reached; bit-identical to the demo maps).
CPU, one process, < 1 GB.  Output: probes/C2/c2_dontcare.json"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); AN = os.path.abspath(os.path.join(HERE, '..', '..'))
ROOT = os.path.abspath(os.path.join(AN, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342')); sys.path.insert(0, os.path.join(ROOT, 'exploration_grid')); sys.path.insert(0, ROOT)
import mapping_3d_storey_data_v420 as W

houses = sys.argv[1:] or ['r1_i1', 'r3_i1']
rays = W.fibonacci_sphere(W.BEAMS)
rng = np.random.default_rng(342101)
rows = []
for h in houses:
    z = np.load(os.path.join(AN, 'probes', 'A03', 'fx', h + '.npz'))
    meta = json.load(open(os.path.join(AN, 'probes', 'A03', 'fx', h + '.json')))
    L = meta['chart_m']; cell = (L / 128, L / 128, W.HEIGHT_M / 32)
    own = z['private']; traj = z['trajectory']; R = z['reached'][4].astype(bool)
    n = own.shape[0]
    for a in range(n):
        t0 = time.time()
        Fa = own[a] == 0; Oa = own[a] == 1; Ua = own[a] == -1
        Fu = (own[R[a]] == 0).any(0)
        assert not (Fu & Oa).any(), 'union free set touches this agent occupied set'
        Wmax = (~Fa).astype(np.uint8)
        Wrand = (Oa | (Ua & (rng.random(Ua.shape) < .5))).astype(np.uint8)
        Wuni = (~Fu).astype(np.uint8)
        r = dict(house=h, agent=a, unknown_share=float(Ua.mean()))
        for name, Wd in (('max', Wmax), ('rand', Wrand), ('union', Wuni)):
            rec = W.observe(Wd, cell, traj[:, a], rays)
            r[f'mismatch_{name}'] = int((rec != own[a]).sum())
        rows.append(r)
        print(h, a, {k: v for k, v in r.items() if k.startswith('mismatch')}, f'{time.time() - t0:.0f}s', flush=True)
S = {k: dict(maps=len(rows), maps_with_mismatch=int(sum(r[f'mismatch_{k}'] > 0 for r in rows)), total_mismatch=int(sum(r[f'mismatch_{k}'] for r in rows)))
     for k in ('max', 'rand', 'union')}
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'c2_dontcare.json'), 'w'), indent=1)
print(json.dumps(S, indent=1))

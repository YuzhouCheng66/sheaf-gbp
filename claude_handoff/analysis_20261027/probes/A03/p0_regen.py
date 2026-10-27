"""p0_regen.py (A03, 2026-10-27): regenerate v420 VAL fixtures LOCALLY (CPU) from 36 copied HouseExpo jsons, keep what the
structured-codec probes need (true poses, object specs, contour, slabs) and check bit-for-bit against the demo own maps.
Run with python -B (no __pycache__ written into scripts/). Output: probes/A03/fx/<house>.npz + <house>.json"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))          # project root
os.environ['HOUSEEXPO_JSON_DIR'] = os.path.join(HERE, 'houseexpo_json')
os.environ['HOUSEEXPO_INDEX'] = os.path.join(ROOT, 'claude_handoff', 'claude_reviews_20261021', 'houseexpo_index_v365.json')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'exploration_grid')); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342')); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import mapping_3d_storey_data_v420 as W
from rate_baselines_lib import load

OUT = os.path.join(HERE, 'fx'); os.makedirs(OUT, exist_ok=True)
per = int(sys.argv[1]) if len(sys.argv) > 1 else 2
rows = []
for rooms in (1, 2, 3):
    taken = 0
    for index in range(12):
        if taken >= per:
            break
        t0 = time.time()
        try:
            fx = W.make_fixture('VALIDATION', index, rooms)
        except ValueError:
            continue
        taken += 1
        spec = W.world_spec('VALIDATION', index, rooms)
        name = f'r{rooms}_i{index}'
        demo = os.path.join(ROOT, 'demo_web3d', 'data', f'house_{name}.json')
        same = None
        if os.path.exists(demo):
            m, own, pred, gt = load(demo)
            same = bool((own == fx['private']).all()) and bool((gt['occ'] == fx['occ'].astype(bool)).all())
        np.savez_compressed(os.path.join(OUT, name + '.npz'), occ=fx['occ'], slab=fx['slab'], obj=fx['obj'], private=fx['private'],
                            trajectory=fx['trajectory_m'], positions=fx['positions_m'], reached=fx['reached'], edges=np.asarray(fx['edges']))
        json.dump(dict(name=name, chart_m=spec['chart_m'], wall_h=spec['wall_h'], door_h=spec['door_h'], verts_m=spec['verts_m'],
                       objects=spec['objects'], room_rects=spec['room_rects'], demo_match=same), open(os.path.join(OUT, name + '.json'), 'w'))
        rows.append((name, same, len(spec['objects']), round(time.time() - t0, 1)))
        print(name, 'demo_match', same, 'objects', len(spec['objects']), f'{time.time() - t0:.1f}s', flush=True)
print(rows)

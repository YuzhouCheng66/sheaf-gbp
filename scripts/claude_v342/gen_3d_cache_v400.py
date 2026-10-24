"""TRAIN cache for the 3D world family v400 (Claude, 2026-10-24): one compressed npz per TRAIN scene (private maps int8 [N,S,S,S],
reached [H,N,N], edges, positions) plus manifest.json with per-row private sha256 and the data module sha256. TRAIN indices only.
python gen_3d_cache_v400.py --out DIR --count 1600 --size 64 --agents 8
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapping_3d_data_v400 as d3
from train_3d_v400 import sha_file

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--count', type=int, default=1600); ap.add_argument('--size', type=int, default=64); ap.add_argument('--agents', type=int, default=8)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rows = []; t0 = time.time()
for spec in d3.scene_plan('TRAIN', args.count, args.agents):
    spec = dict(spec, size=args.size)
    fx = d3.make_fixture(**spec)
    fn = f"i{spec['index']:06d}.npz"
    np.savez_compressed(out / fn, private=fx['private'], reached=fx['reached'], edges=np.array(fx['edges'], dtype=np.float64), positions_m=fx['positions_m'])
    rows.append(dict(file=fn, spec=spec, private_sha256=d3.array_hash(fx['private']), edges=len(fx['edges'])))
    if len(rows) % 100 == 0:
        print(f'{len(rows)}/{args.count} {time.time()-t0:.0f}s', flush=True)
json.dump(dict(version=d3.VERSION, size=args.size, agents=args.agents, scenes=len(rows), hops=[0, 1, 2, 4, 8], data_module_sha256=sha_file(d3.__file__), rows=rows, seconds=time.time() - t0),
          open(out / 'manifest.json', 'w'), indent=1)
print('done', len(rows), f'{time.time()-t0:.0f}s')

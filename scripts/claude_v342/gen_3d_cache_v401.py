"""TRAIN cache for the 3D world family v401 (Claude, 2026-10-24, batch 3D-B): one compressed npz per TRAIN scene (private maps int8
[N,S,S,S], reached [H,N,N], edges, positions) plus manifest.json with per-row private sha256, the data module sha256, hops and beams.
Same layout as gen_3d_cache_v400.py; scenes are generated in a CPU worker pool (numpy only). TRAIN indices only.
python gen_3d_cache_v401.py --out DIR --count 1600 --size 128 --agents 8 --workers 8
"""
import argparse, json, multiprocessing, time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapping_3d_data_v401 as d3
from train_3d_v401 import sha_file, HOPS

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--count', type=int, default=1600); ap.add_argument('--size', type=int, default=128); ap.add_argument('--agents', type=int, default=8)
ap.add_argument('--workers', type=int, default=8)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)


def one(spec):
    fx = d3.make_fixture(**spec)
    assert fx['hops'] == HOPS
    fn = f"i{spec['index']:06d}.npz"
    np.savez_compressed(out / fn, private=fx['private'], reached=fx['reached'], edges=np.array(fx['edges'], dtype=np.float64), positions_m=fx['positions_m'])
    return dict(file=fn, spec=spec, private_sha256=d3.array_hash(fx['private']), edges=len(fx['edges']))


if __name__ == '__main__':
    plan = d3.scene_plan('TRAIN', args.count, args.agents, args.size)
    rows = []; t0 = time.time()
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        for row in pool.imap(one, plan, chunksize=1):                     # imap keeps the plan order
            rows.append(row)
            if len(rows) % 100 == 0:
                print(f'{len(rows)}/{args.count} {time.time()-t0:.0f}s', flush=True)
    assert [r['spec'] for r in rows] == plan
    json.dump(dict(version=d3.VERSION, size=args.size, agents=args.agents, scenes=len(rows), hops=list(HOPS), beams=d3.beams(args.size),
                   data_module_sha256=sha_file(d3.__file__), rows=rows, seconds=time.time() - t0), open(out / 'manifest.json', 'w'), indent=1)
    print('done', len(rows), f'{time.time()-t0:.0f}s')

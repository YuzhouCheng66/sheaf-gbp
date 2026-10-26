"""probe_C_gen (analyst C, 2026-10-26): regenerate fixtures (private maps, reached, edges, slab / obj / struct masks) for
(a) the first N TRAIN houses per room stratum of the training cache manifest (cache3d-v420/train-r1-k1600; the cache lacks the slab /
obj / struct masks, so the houses are regenerated and checked against the cached private-map hash) and (b) the 24-house VAL panel
(scene_plan('VALIDATION', 16) with index < 8, exactly the trainer's VAL panel). CPU only. Writes <out>/{train,val}-NN.npz.
python -B probe_C_gen.py --source <sealed> --scripts <claude-diag scripts dir> --cache <cache dir> --out <dir> --train-per-stratum 8
"""
import argparse, importlib, json, multiprocessing, os, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--scripts', required=True); ap.add_argument('--cache', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--train-per-stratum', type=int, default=8); ap.add_argument('--workers', type=int, default=4)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, args.scripts)
os.environ['SEALED_SOURCE'] = args.source
T = importlib.import_module('train_3d_v431')
import mapping_3d_storey_data_v420 as W

if __name__ == '__main__':
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dims = W.DIMS['R1']
    man = json.load(open(Path(args.cache) / 'manifest.json'))
    rows = []
    for rooms in (1, 2, 3):
        rows += [r for r in man['rows'] if r['spec']['rooms'] == rooms][:args.train_per_stratum]
    vplan = [s for s in W.scene_plan('VALIDATION', 16, dims) if s['index'] < 8]
    specs = [r['spec'] for r in rows] + vplan
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        fxs = pool.map_async(T._fixture, specs, chunksize=1).get(timeout=3600)
    n_ok = {'train': 0, 'val': 0}
    for i, (sp, fx) in enumerate(zip(specs, fxs)):
        split = 'train' if i < len(rows) else 'val'
        if fx is None:
            continue
        if split == 'train':
            assert W.array_hash(fx['private']) == rows[i]['private_sha256'], f'{rows[i]["file"]} does not regenerate'
        np.savez_compressed(out / f'{split}-{n_ok[split]:02d}.npz', private=fx['private'], reached=fx['reached'], edges=np.array(fx['edges'], dtype=np.float64),
                            slab=fx['slab'], obj=fx['obj'], struct=fx['struct'], chart_m=float(fx['chart_m']), spec=json.dumps(sp))
        n_ok[split] += 1
    print(json.dumps(n_ok))

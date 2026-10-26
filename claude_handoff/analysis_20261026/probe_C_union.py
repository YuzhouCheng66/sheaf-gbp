"""probe_C_union (analyst C, 2026-10-26; report-only, CPU): the 'ideal homomorphism' oracle of analyst B round 1 (4.1) on the 3S-X
checkpoints. Every agent's R8 UNION target map is fed to the encoder as if it were its own observation, its own code is decoded as at
K0 (no join), and scored against the R8 target. fusion loss = oracle R8 - fused R8; target difficulty = K0 - oracle R8.
python -B probe_C_union.py --source <sealed> --scripts <dir> --run <run dir> --fx <fixture dir> --out FILE.json
"""
import argparse, glob, importlib, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.dont_write_bytecode = True
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--scripts', required=True); ap.add_argument('--run', required=True)
ap.add_argument('--fx', required=True); ap.add_argument('--out', required=True); ap.add_argument('--split', default='val'); ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, args.scripts)
os.environ['SEALED_SOURCE'] = args.source
torch.set_num_threads(args.threads)
T = importlib.import_module('train_3d_v431')

if __name__ == '__main__':
    ck = torch.load(Path(args.run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False); meta = ck['meta']
    T.DECODER = meta.get('decoder', 'd3'); T.PRECISION = meta.get('precision', 'learned'); T.KNOWN_WEIGHT = meta.get('known_weight', 'uniform')
    T.BX, T.BY, T.BZ = meta['blocks']; T.D = T.BX * T.BY * T.BZ
    T.WIRE_BITS = tuple(meta.get('wire_bits', [8, 8])); T._v378.MU_BITS, T._v378.Q_BITS = T.WIRE_BITS; T.LEVELS = 2 ** (T.WIRE_BITS[0] - 1) - 1
    T.DECODER_WIDTH = meta.get('decoder_width', 1.0)
    dims = tuple(meta['dims'])
    model = T.SheafModel3D(dims, meta['seed']); model.load_state_dict(ck['state_dict']); model.eval()
    rows = []
    for f in sorted(glob.glob(os.path.join(args.fx, f'{args.split}-*.npz'))):
        z = np.load(f); T.CHART_M = float(z['chart_m'])
        own = torch.from_numpy(z['private']); reached = torch.from_numpy(z['reached']); edges = [(int(i), int(j), float(w)) for i, j, w in z['edges']]
        slab = torch.from_numpy(z['slab'].reshape(-1)); obj = torch.from_numpy(z['obj'].reshape(-1)); struct = torch.from_numpy(z['struct'].reshape(-1))
        with torch.no_grad():
            tgt8 = T.stream_targets(own, reached[T.HOPS.index(8)])
            union = tgt8.reshape(own.shape).to(own.dtype)
            mu, q, mask = model.source(union)
            feats = T.rollout_max(mu, q, mask, edges, stages=(0,))
            pred = T.predict(model.decoder, feats[0])
            m = T.agg(T.metrics3d(pred, tgt8, dims, slab, obj, struct))
        rows.append(dict(file=Path(f).stem, oracle_R8=m))
        print(Path(f).stem, round(m['surface_f1'], 4), flush=True)
    summ = {k: float(np.mean([r['oracle_R8'][k] for r in rows if r['oracle_R8'][k] is not None])) for k in T.METRIC_KEYS}
    json.dump(dict(summary=summ, rows=rows), open(args.out, 'w'), indent=1)
    print(json.dumps(summ))

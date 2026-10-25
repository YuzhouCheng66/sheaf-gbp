"""TRAIN-vs-VAL gap of a finished 3D storey checkpoint (Claude, 2026-10-25; report-only, D8-6 problem localisation).
Scores the same model on 24 TRAIN houses it was trained on (indices 0-7 of each stratum of the v420 TRAIN cache; regenerated
deterministically, so slab / furniture / structure masks exist) and on the 24-house VAL panel, K0 and R8, with train_3d_v420.metrics3d.
  TRAIN ~ VAL  (gap within seed noise)  => the model UNDER-FITS: capacity / optimisation / representation, not data quantity;
  TRAIN >> VAL                           => generalisation: data quantity / diversity (as the 2D line's D4-7, +15 pp R8 gap at 600 houses).
Model configuration (decoder width, block layout) comes from the checkpoint meta via --trainer (train_3d_v420 / v422 / v423). CPU-safe.
python diag_trainval_v42x.py --source <sealed> --run runs/v420-storey/a-s342101 --trainer train_3d_v420 --device cpu --out FILE.json
"""
import argparse, importlib, json, multiprocessing, os, sys, time
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--trainer', default='train_3d_v420'); ap.add_argument('--device', default='cpu'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['SEALED_SOURCE'] = args.source
torch.set_num_threads(args.threads)
T = importlib.import_module(args.trainer)
import mapping_3d_storey_data_v420 as W

if __name__ == '__main__':
    t0 = time.time(); dims = W.DIMS['R1']; dev = torch.device(args.device)
    ck = torch.load(Path(args.run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False); meta = ck['meta']
    if hasattr(T, 'DECODER_WIDTH'):
        T.DECODER_WIDTH = meta.get('decoder_width', 1.0)
    if hasattr(T, 'LAYOUTS'):
        T.BX, T.BY, T.BZ = meta['blocks']
    assert list(meta.get('blocks', [8, 8, 4])) == [getattr(T, 'BX', 8), getattr(T, 'BY', 8), getattr(T, 'BZ', 4)] and meta.get('decoder_width', 1.0) == getattr(T, 'DECODER_WIDTH', 1.0)
    man = json.load(open(Path(meta['cache']['dir']) / 'manifest.json'))
    trained = {(r['spec']['rooms'], r['spec']['index']) for r in man['rows']}
    train_specs = [dict(split='TRAIN', index=i, rooms=s, dims=list(dims), agents=W.AGENTS) for s in (1, 2, 3) for i in range(8) if (s, i) in trained]
    val_specs = [s for s in W.scene_plan('VALIDATION', 16, dims) if s['index'] < 8]
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        fxs = pool.map_async(T._fixture, train_specs + val_specs, chunksize=1).get(timeout=3600)
    model = T.SheafModel3D(dims, meta['seed']); model.load_state_dict(ck['state_dict']); model.to(dev).eval()
    out = dict(run=args.run, seed=meta['seed'], decoder_width=meta.get('decoder_width', 1.0), blocks=meta.get('blocks', [8, 8, 4]), updates=ck.get('updates'))
    with torch.no_grad():
        for name, sel in (('train', fxs[:len(train_specs)]), ('val', fxs[len(train_specs):])):
            acc = {k: [] for k in (0, 8)}; n = 0
            for fx in sel:
                if fx is None:
                    continue
                n += 1; T.CHART_M = fx['chart_m']
                own = torch.from_numpy(fx['private']).to(dev); reached = torch.from_numpy(fx['reached']).to(dev)
                slab = torch.from_numpy(fx['slab'].reshape(-1)).to(dev); obj = torch.from_numpy(fx['obj'].reshape(-1)).to(dev); struct = torch.from_numpy(fx['struct'].reshape(-1)).to(dev)
                mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, fx['edges'], stages=(0, 8))
                for k in (0, 8):
                    tgt = T.stream_targets(own, reached[T.HOPS.index(k)]); pred = T.predict(model.decoder, feats[k])
                    acc[k].append(T.agg(T.metrics3d(pred, tgt, dims, slab, obj, struct)))
            out[name] = {f'K{k}': {m: float(np.mean([r[m] for r in acc[k] if r[m] is not None])) for m in ('surface_f1', 'var_occ_f1', 'furn_f1', 'occ_f1', 'accuracy')} for k in (0, 8)}
            out[name]['scenes'] = n
    out['gap_train_minus_val'] = {f'K{k}': {m: round(out['train'][f'K{k}'][m] - out['val'][f'K{k}'][m], 4) for m in ('surface_f1', 'var_occ_f1', 'furn_f1')} for k in (0, 8)}
    out['seconds'] = round(time.time() - t0)
    json.dump(out, open(args.out, 'w'), indent=1)
    for name in ('train', 'val'):
        print(name, out[name]['scenes'], 'scenes', {k: {m: round(v, 3) for m, v in d.items()} for k, d in out[name].items() if k != 'scenes'})
    print('gap train - val', out['gap_train_minus_val'], f"{out['seconds']}s")

"""v427 copy (decoder / precision / known-weight from the checkpoint meta too) of diag_3sa_v420.py (Claude, 2026-10-25): identical computation, but the model configuration (decoder width, block layout) is
taken from the checkpoint meta, so the same script scores v420 / v422 / v423 checkpoints. Pass --trainer train_3d_v423.
"""
"""Post-hoc diagnosis of batch 3S-A (Claude, 2026-10-25; NOT a pre-registered criterion, report-only). Loads a finished 3S-A checkpoint,
re-runs the 24-scene VAL panel and breaks the reconstruction down by what the target voxel is:
  recall of target-occupied voxels that are slab / wall+lintel (struct) / furniture (obj), overall and per height band;
  where the false positives sit: next to structure (within 1 voxel of struct or slab: boundary jitter), in the furniture region
  (~slab & ~dilate6(struct): invented furniture), elsewhere;
  per-scene R8 surface F1 against the chart side L and the room stratum (does the error grow with the content per block?).
Writes JSON and slice data (npz) to --out. GPU for the model, CPU spawn pool for the fixtures.
python diag_3sa_v420.py --source <sealed> --run runs/v420-storey/a-s342101 --out runs/v420-storey/diag-a-s342101
For batch 3S-W add --trainer train_3d_v422 --decoder-width 2.0 (defaults reproduce the 3S-A diagnosis exactly).
"""
import argparse, json, multiprocessing, os, sys
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--workers', type=int, default=4); ap.add_argument('--device', default='cuda')
ap.add_argument('--trainer', default='train_3d_v420'); ap.add_argument('--decoder-width', type=float, default=None)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['SEALED_SOURCE'] = args.source
import importlib
T = importlib.import_module(args.trainer)
if args.decoder_width is not None:
    T.DECODER_WIDTH = args.decoder_width
import mapping_3d_storey_data_v420 as W

if __name__ == '__main__':
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    dims = W.DIMS['R1']; dev = torch.device(args.device)
    plan = [s for s in W.scene_plan('VALIDATION', 16, dims) if s['index'] < 8]
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        fxs = pool.map_async(T._fixture, plan, chunksize=1).get(timeout=3600)
    ck = torch.load(Path(args.run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False)
    if hasattr(T, 'DECODER'):                                          # v427: decoder / precision / known-weight from the checkpoint
        T.DECODER = ck['meta'].get('decoder', 'd3'); T.PRECISION = ck['meta'].get('precision', 'learned'); T.KNOWN_WEIGHT = ck['meta'].get('known_weight', 'uniform')
    if hasattr(T, 'LAYOUTS'):                                          # v423: block layout from the checkpoint
        T.BX, T.BY, T.BZ = ck['meta']['blocks']; assert T.BX * T.BY * T.BZ == T.D
    if hasattr(T, 'DECODER_WIDTH') and args.decoder_width is None:
        T.DECODER_WIDTH = ck['meta'].get('decoder_width', 1.0)
    assert ck['meta'].get('decoder_width', 1.0) == getattr(T, 'DECODER_WIDTH', 1.0), (ck['meta'].get('decoder_width'), getattr(T, 'DECODER_WIDTH', None))
    assert list(ck['meta'].get('blocks', [8, 8, 4])) == [getattr(T, 'BX', 8), getattr(T, 'BY', 8), getattr(T, 'BZ', 4)], 'trainer / checkpoint layout mismatch (review v423 S1)'
    model = T.SheafModel3D(dims, ck['meta']['seed']); model.load_state_dict(ck['state_dict']); model.to(dev).eval()
    NZ = dims[2]; bands = {'0.0-0.3m': (0, 3), '0.3-0.9m': (3, 9), '0.9-1.5m': (9, 15), '1.5-2.1m': (15, 21), '2.1-3.2m': (21, 32)}
    acc = {k: {} for k in (0, 8)}; scenes = []

    def add(d, key, num, den):
        a = d.setdefault(key, [0., 0.]); a[0] += float(num); a[1] += float(den)

    with torch.no_grad():
        for si, (sp, fx) in enumerate(zip(plan, fxs)):
            if fx is None:
                continue
            T.CHART_M = fx['chart_m']
            own = torch.from_numpy(fx['private']).to(dev); reached = torch.from_numpy(fx['reached']).to(dev)
            slab = torch.from_numpy(fx['slab'].reshape(-1)).to(dev); obj = torch.from_numpy(fx['obj'].reshape(-1)).to(dev); struct = torch.from_numpy(fx['struct'].reshape(-1)).to(dev)
            near = T.dilate6((struct | slab)[None], dims)[0]; region = ~slab & ~T.dilate6(struct[None], dims)[0]
            zidx = torch.arange(int(np.prod(dims)), device=dev) % NZ
            mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, fx['edges'])
            row = dict(spec=sp, chart_m=float(fx['chart_m']))
            for k in (0, 8):
                tgt = T.stream_targets(own, reached[T.HOPS.index(k)]); pred = T.predict(model.decoder, feats[k])
                known = tgt >= 0; to = (tgt == 1) & known; tf = (tgt == 0) & known; po = (pred == 1) & known
                surf = T.surface_of(tgt, dims)
                for name, cat in (('slab', slab), ('struct', struct), ('obj', obj)):
                    add(acc[k], f'recall_{name}', (po & to & cat).sum(), (to & cat).sum())
                    add(acc[k], f'surface_recall_{name}', (po & surf & cat).sum(), (surf & cat).sum())
                    for b, (z0, z1) in bands.items():
                        zb = (zidx >= z0) & (zidx < z1)
                        add(acc[k], f'recall_{name}_{b}', (po & to & cat & zb).sum(), (to & cat & zb).sum())
                fp = po & tf
                add(acc[k], 'fp_near_structure', (fp & near).sum(), fp.sum()); add(acc[k], 'fp_furniture_region', (fp & region & ~near).sum(), fp.sum())
                add(acc[k], 'fp_other', (fp & ~near & ~region).sum(), fp.sum())
                add(acc[k], 'precision_occ', (po & to).sum(), po.sum())
                add(acc[k], 'pred_occ_over_target_occ', po.sum(), to.sum())
                row[f'K{k}'] = T.agg(T.metrics3d(pred, tgt, dims, slab, obj, struct))
                if si < 3:
                    np.savez_compressed(out / f'slices-{si}-K{k}.npz', target=tgt.reshape(-1, *dims)[:2].cpu().numpy(), pred=pred.reshape(-1, *dims)[:2].cpu().numpy(),
                                        slab=fx['slab'], obj=fx['obj'], struct=fx['struct'].reshape(dims), chart_m=fx['chart_m'])
            row['mask_fraction'] = float(mask.mean()); row['q_mean'] = float((q * mask).sum() / mask.sum().clamp_min(1)); row['mu_abs_mean'] = float((mu.abs() * mask).sum() / mask.sum().clamp_min(1))
            scenes.append(row)
    res = {f'K{k}': {key: (v[0] / v[1] if v[1] else None) for key, v in acc[k].items()} for k in (0, 8)}
    L = np.array([r['chart_m'] for r in scenes]); f8 = np.array([r['K8']['surface_f1'] for r in scenes]); rooms = np.array([r['spec']['rooms'] for r in scenes])
    res['per_scene'] = [dict(index=r['spec']['index'], rooms=r['spec']['rooms'], chart_m=r['chart_m'], K0_surface=r['K0']['surface_f1'], K8_surface=r['K8']['surface_f1'],
                             K8_var=r['K8']['var_occ_f1'], K8_furn=r['K8']['furn_f1'], mask_fraction=r['mask_fraction'], q_mean=r['q_mean'], mu_abs_mean=r['mu_abs_mean']) for r in scenes]
    res['corr_K8_surface_vs_chart_m'] = float(np.corrcoef(L, f8)[0, 1]); res['K8_surface_by_rooms'] = {int(r): float(f8[rooms == r].mean()) for r in np.unique(rooms)}
    json.dump(res, open(out / 'diag.json', 'w'), indent=1)
    for k in ('K0', 'K8'):
        print(k, {kk: round(v, 3) for kk, v in res[k].items() if v is not None and not any(b in kk for b in bands)})
        print(k, 'by height', {kk: round(v, 3) for kk, v in res[k].items() if v is not None and any(b in kk for b in bands)})
    print('corr(K8 surface, L) %.3f' % res['corr_K8_surface_vs_chart_m'], 'by rooms', res['K8_surface_by_rooms'])
    for r in res['per_scene']:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

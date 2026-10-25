"""Known-head veto diagnosis (Claude, 2026-10-25; analyst B memo 3.3; report-only mechanism readout for batch 3S-R). On the 24-house VAL
panel, at K0 and R8: the fraction of TARGET-KNOWN OCCUPIED voxels that the model outputs as UNKNOWN (predict(): known_logit < 0 vetoes the
occupancy head), overall and for walls / furniture / slabs; the same fraction when only the occupancy head is consulted would be 0 by
construction, so also reported: the occupancy head's own recall on those vetoed voxels (occ_logit >= 0), i.e. what the veto costs.
Model configuration (layout, width, decoder, precision, known-weight) from the checkpoint meta via --trainer. CPU-safe.
python diag_veto_v427.py --source <sealed> --run DIR --trainer train_3d_v427 --out FILE.json
"""
import argparse, importlib, json, multiprocessing, os, sys
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--trainer', default='train_3d_v427'); ap.add_argument('--device', default='cpu'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['SEALED_SOURCE'] = args.source
torch.set_num_threads(args.threads)
T = importlib.import_module(args.trainer)
import mapping_3d_storey_data_v420 as W

if __name__ == '__main__':
    dims = W.DIMS['R1']; dev = torch.device(args.device)
    ck = torch.load(Path(args.run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False); meta = ck['meta']
    if hasattr(T, 'DECODER'):
        T.DECODER = meta.get('decoder', 'd3'); T.PRECISION = meta.get('precision', 'learned'); T.KNOWN_WEIGHT = meta.get('known_weight', 'uniform')
    if hasattr(T, 'LAYOUTS'):
        T.BX, T.BY, T.BZ = meta['blocks']; T.D = T.BX * T.BY * T.BZ
    if hasattr(T, 'DECODER_WIDTH'):
        T.DECODER_WIDTH = meta.get('decoder_width', 1.0)
    assert list(meta.get('blocks', [8, 8, 4])) == [getattr(T, 'BX', 8), getattr(T, 'BY', 8), getattr(T, 'BZ', 4)], 'layout mismatch'
    plan = [s for s in W.scene_plan('VALIDATION', 16, dims) if s['index'] < 8]
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        fxs = pool.map_async(T._fixture, plan, chunksize=1).get(timeout=3600)
    model = T.SheafModel3D(dims, meta['seed']); model.load_state_dict(ck['state_dict']); model.to(dev).eval()
    acc = {k: {} for k in (0, 8)}

    def add(d, key, num, den):
        a = d.setdefault(key, [0., 0.]); a[0] += float(num); a[1] += float(den)

    with torch.no_grad():
        for fx in fxs:
            if fx is None:
                continue
            T.CHART_M = fx['chart_m']
            own = torch.from_numpy(fx['private']).to(dev); reached = torch.from_numpy(fx['reached']).to(dev)
            slab = torch.from_numpy(fx['slab'].reshape(-1)).to(dev); obj = torch.from_numpy(fx['obj'].reshape(-1)).to(dev); struct = torch.from_numpy(fx['struct'].reshape(-1)).to(dev)
            mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, fx['edges'], stages=(0, 8))
            for k in (0, 8):
                tgt = T.stream_targets(own, reached[T.HOPS.index(k)])
                known_logit, occ_logit = model.decoder(feats[k])
                to = tgt == 1; vet = to & (known_logit < 0)
                add(acc[k], 'vetoed_fraction', vet.sum(), to.sum())
                for name, cat in (('slab', slab), ('struct', struct), ('obj', obj)):
                    add(acc[k], f'vetoed_fraction_{name}', (vet & cat).sum(), (to & cat).sum())
                add(acc[k], 'occ_head_recall_on_vetoed', (vet & (occ_logit >= 0)).sum(), vet.sum())
                add(acc[k], 'occ_head_recall_all', (to & (occ_logit >= 0)).sum(), to.sum())
                add(acc[k], 'known_fpr_unknown_side', ((tgt < 0) & (known_logit >= 0)).sum(), (tgt < 0).sum())
    res = {f'K{k}': {key: (round(v[0] / v[1], 4) if v[1] else None) for key, v in acc[k].items()} for k in (0, 8)}
    res['meta'] = dict(run=args.run, seed=meta['seed'], blocks=meta['blocks'], decoder=meta.get('decoder', 'd3'), decoder_width=meta.get('decoder_width', 1.0), precision=meta.get('precision', 'learned'), known_weight=meta.get('known_weight', 'uniform'))
    json.dump(res, open(args.out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('K0', 'K8')}))

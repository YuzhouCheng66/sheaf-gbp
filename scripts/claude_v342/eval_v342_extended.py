"""Extended read-only evaluation of a v342 checkpoint: ALL sealed node_metrics keys (doors, topology, own P/R, known P/R),
per-scene rows for paired statistics, VAL (idx 0-7) and FRESH (idx 8-15). Usage:
python eval_v342_extended.py --source <sealed> --checkpoint <final.pt> --arm owned|dense --size 128 --out <json>
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--arm', required=True)
ap.add_argument('--size', type=int, default=128); ap.add_argument('--out', required=True); ap.add_argument('--device', default='cuda')
ap.add_argument('--trainer', default=str(Path(__file__).resolve().parent / 'train_v342_owned_v2.py'))
ap.add_argument('--probe', default='none', choices=['none', 'posterior_only', 'owner_priority'], help='counterfactual readout probes (inference only)')
ap.add_argument('--panels', default='val,fresh')
args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid'))
sys.path.insert(0, str(Path(args.trainer).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location('trainer', args.trainer); trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
from scripts import mapping_multiroom_data_v293 as data
from scripts import mapping_native_bank_v306 as bank
from scripts import mapping_multiroom_forest_v295 as forest
from scripts import train_mapping_support_native_v306 as base
from scripts.mapping_recognition_metrics_v277 import node_metrics
from scripts.mapping_implicit_model_v192 import latent_quantize

device = torch.device(args.device); size = args.size; HOPS = (0, 1, 2, 4, 8)
payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
model = trainer.SheafModel(args.arm, size, 0).to(device); model.load_state_dict(payload['state_dict'], strict=True); model.eval()
if hasattr(trainer, 'WALL_HEAD'):
    trainer.WALL_HEAD = payload.get('meta', {}).get('wall_head', 'dt'); print('wall head', trainer.WALL_HEAD)
rollout = trainer.make_rollout(forest, latent_quantize)
val_specs = [s for s in base.scene_plan('VALIDATION') if s['size'] == size]
panels = dict(val=val_specs, fresh=[dict(s, index=s['index'] + 8) for s in val_specs])
panels = {k: v for k, v in panels.items() if k in args.panels.split(',')}
result = dict(checkpoint=args.checkpoint, checkpoint_sha256=trainer.sha_file(args.checkpoint), arm=args.arm, size=size, updates=payload.get('updates'), probe=args.probe, panels={})
t0 = time.time()
with torch.no_grad():
    for label, specs in panels.items():
        rows = []
        for spec_ in specs:
            fx = data.make_fixture(**spec_); inp = fx['inputs']
            targets = {k: bank.forest_targets(inp, k)[0] for k in HOPS}
            edges = bank.select_forest(inp['positions_m'], inp['adjacency'], inp['public_frame']['communication_radius_m'])['edges']
            own = torch.from_numpy(inp['private']).to(device); n = own.shape[0]
            mu, q, mask = model.source(own)
            feats, _ = rollout(mu, q, mask, edges)
            prior = trainer.LATENT_PRIOR / (n * 64)
            if args.probe == 'posterior_only':
                # own planes replaced by the exact 'nothing observed' value (mean 0, log var = -log prior): decoder must read everything from the posterior
                feats = dict(feats); unobs = torch.cat((torch.zeros(n, 64, device=device), torch.full((n, 64), float(-np.log(prior)), device=device)), -1)
                own_feats = unobs
            elif args.probe == 'owner_priority':
                # posterior planes: on own-observed blocks use the own factor (mean, K0 variance); elsewhere the actual posterior
                feats = {k: torch.cat((torch.where(mask > 0, feats[0][:, :64], f[:, :64]), torch.where(mask > 0, feats[0][:, 64:], f[:, 64:])), -1) for k, f in feats.items()}
                own_feats = feats[0]
            else:
                own_feats = feats[0]
            hops = {}
            for k in HOPS:
                pred = trainer.predict(model.decoder, torch.cat((own_feats, feats[k]), -1)).cpu().numpy().reshape(n, size, size)
                m = node_metrics(pred, inp['private'], targets[k], {'door_mask': fx['evaluation']['door_mask']})
                keys = sorted({kk for r in m for kk, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
                agg = {}
                for key in keys:
                    vals = [r[key] for r in m if r.get(key) is not None]
                    agg[key] = float(np.mean(vals)) if vals else None
                hops[str(k)] = agg
            rows.append(dict(spec=spec_, hops=hops))
            print(label, spec_, 'R8 wallF1 %.3f doorfree %s topo %s' % (hops['8']['wall_f1'], hops['8'].get('door_free_recall'), hops['8'].get('pair_topology_accuracy')), f'{time.time()-t0:.0f}s', flush=True)
        summary = {}
        for k in HOPS:
            strata = {}
            for r in rows:
                strata.setdefault((r['spec']['rooms'], r['spec']['agents']), []).append(r['hops'][str(k)])
            keys = list(rows[0]['hops'][str(k)].keys())
            per = {f'rooms{a}/N{b}': {m_: (float(np.mean([x[m_] for x in v if x.get(m_) is not None])) if any(x.get(m_) is not None for x in v) else None) for m_ in keys} for (a, b), v in strata.items()}
            summary[str(k)] = dict(all={m_: (float(np.mean([p[m_] for p in per.values() if p[m_] is not None])) if any(p[m_] is not None for p in per.values()) else None) for m_ in keys}, strata=per)
        result['panels'][label] = dict(rows=rows, summary=summary)
with open(args.out, 'w') as f:
    json.dump(result, f, indent=1)
for label in panels:
    s = result['panels'][label]['summary']
    for k in ('0', '8'):
        a = s[k]['all']
        print(label, 'R' + k, {kk: round(a[kk], 3) for kk in ('wall_f1', 'union_accuracy', 'unknown_known_fpr', 'own_wall_f1', 'own_wall_recall', 'own_wall_precision', 'peer_wall_f1', 'door_free_recall', 'door_false_wall', 'pair_topology_accuracy', 'spurious_pair_connection_rate', 'known_precision', 'known_recall') if a.get(kk) is not None})

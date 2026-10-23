"""Decision-threshold sweep for the DT wall head (read-only, no training): wall := softplus(dt_channel) < tau.
Motivation (error anatomy D3-3): the L1-trained DT head outputs a median; under door-endpoint uncertainty the median erodes door
gaps (90% of far false walls after fusion are door cells). tau is ONE scalar selected on VALIDATION and then read on the fresh panel.
Usage: python eval_threshold_v347.py --source <sealed> --checkpoint <final.pt> --trainer train_v347_maxsel.py --out <json> [--taus 0.5,0.6,0.7,0.8,1.0]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--trainer', required=True)
ap.add_argument('--size', type=int, default=128); ap.add_argument('--out', required=True); ap.add_argument('--device', default='cuda')
ap.add_argument('--taus', default='0.5,0.6,0.7,0.8,1.0'); ap.add_argument('--panels', default='val,fresh'); ap.add_argument('--stages', default='0,8')
ap.add_argument('--probe', default='none', choices=['none', 'owner_priority'], help='owner_priority: on own-observed blocks the posterior planes are replaced by the own factor (counterfactual: separates code replacement from context effects)')
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

device = torch.device(args.device); size = args.size
HOPS = tuple(int(s) for s in args.stages.split(',')); TAUS = [float(t) for t in args.taus.split(',')]
payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False); meta = payload['meta']
for name, key, default in (('READOUT_MODE', 'readout_mode', 'anchored'), ('PRECISION_MODE', 'precision_mode', 'learned'), ('WIRE', 'wire', 'sealed'), ('MU_BITS', 'mu_bits', 8), ('Q_BITS', 'q_bits', 8)):
    if hasattr(trainer, name):
        setattr(trainer, name, meta.get(key, default))
fusion = meta.get('fusion', 'sum')
model = trainer.SheafModel(meta['arm'], size, 0).to(device); model.load_state_dict(payload['state_dict'], strict=True); model.eval()
rollout = trainer.make_rollout(forest, latent_quantize, fusion) if 'fusion' in trainer.make_rollout.__code__.co_varnames else trainer.make_rollout(forest, latent_quantize)
val_specs = [s for s in base.scene_plan('VALIDATION') if s['size'] == size]
panels = dict(val=val_specs, fresh=[dict(s, index=s['index'] + 8) for s in val_specs])
panels = {k: v for k, v in panels.items() if k in args.panels.split(',')}
result = dict(probe=args.probe, checkpoint=args.checkpoint, checkpoint_sha256=trainer.sha_file(args.checkpoint), arm=meta['arm'], fusion=fusion, readout=meta.get('readout_mode'), size=size, updates=payload.get('updates'), taus=TAUS, panels={})
t0 = time.time()
KEYS = ('wall_f1', 'wall_recall', 'wall_precision', 'union_accuracy', 'unknown_known_fpr', 'own_wall_f1', 'peer_wall_f1', 'door_free_recall', 'door_false_wall', 'pair_topology_accuracy', 'spurious_pair_connection_rate', 'known_recall', 'known_precision')
with torch.no_grad():
    for label, specs in panels.items():
        rows = []
        for spec_ in specs:
            fx = data.make_fixture(**spec_); inp = fx['inputs']
            targets = {k: bank.forest_targets(inp, k)[0] for k in HOPS}
            edges = bank.select_forest(inp['positions_m'], inp['adjacency'], inp['public_frame']['communication_radius_m'])['edges']
            own = torch.from_numpy(inp['private']).to(device); n = own.shape[0]
            mu, q, mask = model.source(own)
            feats, _ = rollout(mu, q, mask, edges, stages=(0,) + tuple(k for k in HOPS if k > 0))
            if args.probe == 'owner_priority':
                feats = {k: torch.cat((torch.where(mask > 0, feats[0][:, :64], f[:, :64]), torch.where(mask > 0, feats[0][:, 64:], f[:, 64:])), -1) for k, f in feats.items()}
            hops = {}
            for k in HOPS:
                known_logit, wall_ch = model.decoder(torch.cat((feats[0], feats[k]), -1))
                dt = F.softplus(wall_ch)
                for tau in TAUS:
                    wall = dt < tau
                    pred = torch.where(known_logit >= 0, wall.to(torch.int8), torch.full_like(wall, -1, dtype=torch.int8)).cpu().numpy().reshape(n, size, size)
                    m = node_metrics(pred, inp['private'], targets[k], {'door_mask': fx['evaluation']['door_mask']})
                    agg = {}
                    for key in KEYS:
                        vals = [r[key] for r in m if r.get(key) is not None]
                        agg[key] = float(np.mean(vals)) if vals else None
                    hops[f'{k}@{tau}'] = agg
            rows.append(dict(spec=spec_, hops=hops))
            print(label, spec_['rooms'], spec_['agents'], spec_['index'], ' '.join(f"R8@{t}: F1 {hops[f'8@{t}']['wall_f1']:.3f} door {hops[f'8@{t}']['door_free_recall']:.2f}" for t in TAUS if f'8@{t}' in hops), f'{time.time()-t0:.0f}s', flush=True)
        summary = {}
        for hk in rows[0]['hops']:
            strata = {}
            for r in rows:
                strata.setdefault((r['spec']['rooms'], r['spec']['agents']), []).append(r['hops'][hk])
            per = {f'rooms{a}/N{b}': {m_: (float(np.mean([x[m_] for x in v if x.get(m_) is not None])) if any(x.get(m_) is not None for x in v) else None) for m_ in KEYS} for (a, b), v in strata.items()}
            summary[hk] = dict(all={m_: (float(np.mean([p[m_] for p in per.values() if p[m_] is not None])) if any(p[m_] is not None for p in per.values()) else None) for m_ in KEYS}, strata=per)
        result['panels'][label] = dict(rows=rows, summary=summary)
with open(args.out, 'w') as f:
    json.dump(result, f, indent=1)
for label in panels:
    s = result['panels'][label]['summary']
    for hk in s:
        a = s[hk]['all']
        print(label, hk, {kk: (round(a[kk], 3) if a[kk] is not None else None) for kk in ('wall_f1', 'wall_recall', 'wall_precision', 'union_accuracy', 'unknown_known_fpr', 'own_wall_f1', 'door_free_recall', 'door_false_wall', 'pair_topology_accuracy', 'spurious_pair_connection_rate')})

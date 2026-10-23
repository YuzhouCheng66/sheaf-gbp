"""Item 2, second design: DEMAND-DRIVEN edge stalks (read-only evaluation, no trainer change). Claude 2026-10-22.

Task: each node j wants the map within a region of interest ROI(j) = blocks whose centre lies within rho metres of j (a robot wants
the map around itself). The minimal exact restriction map for this task on the public forest is
    stalk(i -> j) = union of ROI(u) over all nodes u on j's side of the edge (j itself and everything it relays to),
computed from public positions + the public forest only. By construction every block of ROI(j) survives every edge on every path
into j, so the deliverable target for j is exactly the full forest target restricted to ROI(j): no coverage is lost for the task,
and the bytes are the mean stalk fraction. Compared with the midpoint selection (eval_edge_select_v353) this is the demand side
of the same partial-selection family (diagonal-exact by the closure theorem).
Reports per rho and stage: F1/acc/FPR on the ROI-restricted target (sealed node_metrics), the fraction of coordinates per directed
message, and for reference the full-target metrics. rho=0 means 'no demand restriction' (identity stalks, ROI = whole world).
python eval_demand_select_v357.py --source <sealed> --checkpoint <v347 max final.pt> --trainer train_v350_dyn_edge.py --rhos 0,10,8,6,5 --out <json>
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--trainer', required=True)
ap.add_argument('--size', type=int, default=128); ap.add_argument('--out', required=True); ap.add_argument('--device', default='cuda')
ap.add_argument('--rhos', default='0,10,8,6,5'); ap.add_argument('--panel', default='val'); ap.add_argument('--stages', default='0,1,2,4,8')
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

device = torch.device(args.device); size = args.size; p8 = size // 8
HOPS = tuple(int(s) for s in args.stages.split(',')); RHOS = [float(r) for r in args.rhos.split(',')]
payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False); meta = payload['meta']
for name, key, default in (('READOUT_MODE', 'readout_mode', 'anchored'), ('PRECISION_MODE', 'precision_mode', 'learned'), ('WIRE', 'wire', 'sealed'), ('MU_BITS', 'mu_bits', 8), ('Q_BITS', 'q_bits', 8)):
    setattr(trainer, name, meta.get(key, default))
fusion = meta.get('fusion', 'sum')
model = trainer.SheafModel(meta['arm'], size, 0).to(device); model.load_state_dict(payload['state_dict'], strict=True); model.eval()
rollout = trainer.make_rollout(forest, latent_quantize, fusion)
specs = [s for s in base.scene_plan('VALIDATION') if s['size'] == size]
if args.panel == 'fresh':
    specs = [dict(s, index=s['index'] + 8) for s in specs]
KEYS = ('wall_f1', 'wall_recall', 'wall_precision', 'union_accuracy', 'unknown_known_fpr', 'own_wall_f1', 'peer_wall_f1', 'door_free_recall', 'pair_topology_accuracy', 'known_recall')
centres = (np.arange(8) + .5) * 2.5
CY, CX = np.meshgrid(centres, centres, indexing='ij')

def roi(pos, rho):
    if rho <= 0:
        return np.ones(64, bool)
    return (((CX - pos[0]) ** 2 + (CY - pos[1]) ** 2) <= rho ** 2).flatten()

def demand_stalks(positions_m, edges, n, rho):
    """stalk[(i,j)] = union of ROI(u) over u on j's side when edge (i,j) is cut (symmetric in the sense stalk[(i,j)] != stalk[(j,i)])."""
    adj = {i: [] for i in range(n)}
    for i, j, _ in edges:
        adj[i].append(j); adj[j].append(i)
    rois = [roi(positions_m[u], rho) for u in range(n)]
    stalk = {}
    for i, j, _ in edges:
        for a, b in ((i, j), (j, i)):            # message a -> b: everything on b's side (b and what b can relay to)
            seen = {b}; stack = [b]; u = np.zeros(64, bool)
            while stack:
                v = stack.pop(); u |= rois[v]
                for w in adj[v]:
                    if w not in seen and not (v == b and w == a):
                        seen.add(w); stack.append(w)
            stalk[(a, b)] = u
    return stalk, rois

result = dict(checkpoint=args.checkpoint, checkpoint_sha256=trainer.sha_file(args.checkpoint), fusion=fusion, readout=meta.get('readout_mode'), panel=args.panel, rhos=RHOS, rows=[])
t0 = time.time()
with torch.no_grad():
    for spec_ in specs:
        fx = data.make_fixture(**spec_); inp = fx['inputs']
        full = {k: bank.forest_targets(inp, k)[0] for k in HOPS}
        edges = bank.select_forest(inp['positions_m'], inp['adjacency'], inp['public_frame']['communication_radius_m'])['edges']
        own = torch.from_numpy(inp['private']).to(device); n = own.shape[0]
        mu, q, mask = model.source(own)
        own_blocks = (inp['private'] >= 0).reshape(n, 8, p8, 8, p8).any(axis=(2, 4)).reshape(n, 64)
        row = dict(spec=spec_, per_rho={})
        for rho in RHOS:
            stalk, rois = demand_stalks(inp['positions_m'], edges, n, rho)
            trainer.EDGE_SELECT_M = 1. if rho > 0 else 0.          # any positive value routes through the selection path
            trainer.edge_selection = (lambda pos, i, j, dev, _s=stalk: torch.from_numpy(_s[(i, j)].astype(np.float64)).to(dev))
            feats, _ = rollout(mu, q, mask, edges, stages=HOPS, positions_m=inp['positions_m'])
            per = {}
            for k in HOPS:
                pred = trainer.predict(model.decoder, torch.cat((feats[0], feats[k]), -1)).cpu().numpy().reshape(n, size, size)
                # ROI-restricted target: full forest target within ROI(j) blocks; outside ROI the task does not ask for anything
                tgt = np.full_like(full[k], -1); pred_roi = np.full_like(pred, -1)
                for j in range(n):
                    # scoring region = ROI(j) blocks plus the blocks j observed itself (a robot keeps its own map; node_metrics requires own obs in the target)
                    m = np.kron((rois[j] | own_blocks[j]).reshape(8, 8), np.ones((p8, p8), bool))
                    tgt[j] = np.where(m, full[k][j], -1); pred_roi[j] = np.where(m, pred[j], -1)
                m_r = node_metrics(pred_roi, inp['private'], tgt, {'door_mask': fx['evaluation']['door_mask']})
                m_f = node_metrics(pred, inp['private'], full[k], {'door_mask': fx['evaluation']['door_mask']})
                agg = {}
                for key in KEYS:
                    vr = [r[key] for r in m_r if r.get(key) is not None]; vf = [r[key] for r in m_f if r.get(key) is not None]
                    agg['roi_' + key] = float(np.mean(vr)) if vr else None; agg['full_' + key] = float(np.mean(vf)) if vf else None
                agg['roi_fraction_of_full_known'] = float((tgt >= 0).sum() / max((full[k] >= 0).sum(), 1))
                agg['frac_coordinates_sent'] = float(np.mean([stalk[(i, j)].mean() for i, j, _ in edges] + [stalk[(j, i)].mean() for i, j, _ in edges])) if edges else 1.
                per[str(k)] = agg
            row['per_rho'][str(rho)] = per
        result['rows'].append(row)
        print(spec_['rooms'], spec_['agents'], spec_['index'], ' '.join(f"rho{r:g}: roiF1 {row['per_rho'][str(r)]['8']['roi_wall_f1']:.3f} sent {row['per_rho'][str(r)]['8']['frac_coordinates_sent']:.2f}" for r in RHOS), f'{time.time()-t0:.0f}s', flush=True)
summary = {}
for r in RHOS:
    summary[str(r)] = {}
    for k in HOPS:
        strata = {}
        for row in result['rows']:
            strata.setdefault((row['spec']['rooms'], row['spec']['agents']), []).append(row['per_rho'][str(r)][str(k)])
        keys = list(next(iter(strata.values()))[0].keys())
        per = {s: {m_: (float(np.mean([x[m_] for x in v if x.get(m_) is not None])) if any(x.get(m_) is not None for x in v) else None) for m_ in keys} for s, v in strata.items()}
        summary[str(r)][str(k)] = {m_: (float(np.mean([p[m_] for p in per.values() if p[m_] is not None])) if any(p[m_] is not None for p in per.values()) else None) for m_ in keys}
result['summary'] = summary
with open(args.out, 'w') as f:
    json.dump(result, f, indent=1)
for r in RHOS:
    a = summary[str(r)]['8']
    print(f"rho={r:g}: sent {a['frac_coordinates_sent']:.3f} | R8 on ROI target: F1 {a['roi_wall_f1']:.3f} acc {a['roi_union_accuracy']:.3f} FPR {a['roi_unknown_known_fpr']:.3f} own {a['roi_own_wall_f1']:.3f} | ROI known fraction {a['roi_fraction_of_full_known']:.3f} | full-target F1 {a['full_wall_f1']:.3f}")

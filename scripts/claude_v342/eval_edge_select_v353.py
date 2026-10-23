"""Item 2 (edge-stalk selection restriction maps), read-only evaluation with DELIVERABLE targets. Claude 2026-10-22.

With the stalk of edge e = selection S_e (blocks within R m of the edge midpoint), the information of source s that can reach node j
along the tree path s -> j is restricted to the blocks in the intersection of the stalks along that path (relays intersect).
The sealed forest_targets assume everything arrives, so against them any R < world size looks like a loss of accuracy that is
really a loss of COVERAGE by design. This evaluator reports, per R and stage k:
  deliverable target : union over reached sources s of (private_s restricted to the blocks that survive the path s -> j)
  F1 / acc / FPR on the deliverable target (sealed node_metrics)   -- does the decoder render what the sheaf lets through?
  coverage           : known cells of the deliverable target / known cells of the full forest target (what the design gives up)
  F1 on the FULL target (for reference) and the mean fraction of coordinates per directed message (bytes).
R = 0 is the exact identity control (must reproduce the v347 final-val numbers).
python eval_edge_select_v353.py --source <sealed> --checkpoint <v347 max final.pt> --trainer train_v350_dyn_edge.py --radii 0,15,12,10,8,6 --out <json>
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--trainer', required=True)
ap.add_argument('--size', type=int, default=128); ap.add_argument('--out', required=True); ap.add_argument('--device', default='cuda')
ap.add_argument('--radii', default='0,15,12,10,8,6'); ap.add_argument('--panel', default='val'); ap.add_argument('--stages', default='0,1,2,4,8')
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
HOPS = tuple(int(s) for s in args.stages.split(',')); RADII = [float(r) for r in args.radii.split(',')]
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

def deliverable_targets(private, edges, positions_m, R, k):
    """[N,H,H] ternary target per receiver j: union of sources within k hops, each restricted to the blocks surviving the path."""
    n = private.shape[0]
    adj = {i: [] for i in range(n)}
    for i, j, _ in edges:
        adj[i].append(j); adj[j].append(i)
    trainer.EDGE_SELECT_M = R
    stalk = {}
    for i, j, _ in edges:
        s = trainer.edge_selection(positions_m, i, j, 'cpu').numpy().astype(bool)
        stalk[(i, j)] = s; stalk[(j, i)] = s
    out = np.full_like(private, -1); frac_sent = []
    for (i, j, _) in edges:
        frac_sent.append(stalk[(i, j)].mean())
    for j in range(n):
        # BFS from j up to depth k, carrying the intersection of stalks along the path (path edges are traversed from source to j,
        # the stalk is symmetric so the direction does not matter)
        surv = {j: np.ones(64, bool)}; frontier = [j]
        for _ in range(k):
            nxt = []
            for v in frontier:
                for w in adj[v]:
                    if w not in surv:
                        surv[w] = surv[v] & stalk[(w, v)]; nxt.append(w)
            frontier = nxt
        acc = np.full(private.shape[1:], -1, np.int8)
        for s_, blocks in surv.items():
            m = np.kron(blocks.reshape(8, 8), np.ones((p8, p8), bool))
            acc = np.maximum(acc, np.where(m, private[s_], -1))
        out[j] = acc
    return out, float(np.mean(frac_sent)) if frac_sent else 1.

result = dict(checkpoint=args.checkpoint, checkpoint_sha256=trainer.sha_file(args.checkpoint), fusion=fusion, readout=meta.get('readout_mode'), panel=args.panel, radii=RADII, rows=[])
t0 = time.time()
with torch.no_grad():
    for spec_ in specs:
        fx = data.make_fixture(**spec_); inp = fx['inputs']
        full = {k: bank.forest_targets(inp, k)[0] for k in HOPS}
        edges = bank.select_forest(inp['positions_m'], inp['adjacency'], inp['public_frame']['communication_radius_m'])['edges']
        own = torch.from_numpy(inp['private']).to(device); n = own.shape[0]
        mu, q, mask = model.source(own)
        row = dict(spec=spec_, per_R={})
        for R in RADII:
            trainer.EDGE_SELECT_M = R
            feats, _ = rollout(mu, q, mask, edges, stages=HOPS, positions_m=inp['positions_m'])
            per = {}
            for k in HOPS:
                pred = trainer.predict(model.decoder, torch.cat((feats[0], feats[k]), -1)).cpu().numpy().reshape(n, size, size)
                deliv, frac = deliverable_targets(inp['private'], edges, inp['positions_m'], R, k)
                m_d = node_metrics(pred, inp['private'], deliv, {'door_mask': fx['evaluation']['door_mask']})
                m_f = node_metrics(pred, inp['private'], full[k], {'door_mask': fx['evaluation']['door_mask']})
                agg = {}
                for key in KEYS:
                    vd = [r[key] for r in m_d if r.get(key) is not None]; vf = [r[key] for r in m_f if r.get(key) is not None]
                    agg['deliv_' + key] = float(np.mean(vd)) if vd else None; agg['full_' + key] = float(np.mean(vf)) if vf else None
                agg['coverage_known'] = float((deliv >= 0).sum() / max((full[k] >= 0).sum(), 1))
                agg['coverage_wall'] = float((deliv == 1).sum() / max((full[k] == 1).sum(), 1))
                agg['frac_coordinates_sent'] = frac
                per[str(k)] = agg
            row['per_R'][str(R)] = per
        result['rows'].append(row)
        print(spec_['rooms'], spec_['agents'], spec_['index'], ' '.join(f"R{R:g}: dF1 {row['per_R'][str(R)]['8']['deliv_wall_f1']:.3f} cov {row['per_R'][str(R)]['8']['coverage_known']:.2f} fF1 {row['per_R'][str(R)]['8']['full_wall_f1']:.3f}" for R in RADII), f'{time.time()-t0:.0f}s', flush=True)
# equal-weight stratum summary
summary = {}
for R in RADII:
    summary[str(R)] = {}
    for k in HOPS:
        strata = {}
        for r in result['rows']:
            strata.setdefault((r['spec']['rooms'], r['spec']['agents']), []).append(r['per_R'][str(R)][str(k)])
        keys = list(next(iter(strata.values()))[0].keys())
        per = {s: {m_: (float(np.mean([x[m_] for x in v if x.get(m_) is not None])) if any(x.get(m_) is not None for x in v) else None) for m_ in keys} for s, v in strata.items()}
        summary[str(R)][str(k)] = {m_: (float(np.mean([p[m_] for p in per.values() if p[m_] is not None])) if any(p[m_] is not None for p in per.values()) else None) for m_ in keys}
result['summary'] = summary
with open(args.out, 'w') as f:
    json.dump(result, f, indent=1)
for R in RADII:
    a = summary[str(R)]['8']
    print(f"R={R:g}: sent {a['frac_coordinates_sent']:.3f} | R8 deliverable: F1 {a['deliv_wall_f1']:.3f} acc {a['deliv_union_accuracy']:.3f} FPR {a['deliv_unknown_known_fpr']:.3f} own {a['deliv_own_wall_f1']:.3f} | coverage known {a['coverage_known']:.3f} wall {a['coverage_wall']:.3f} | full-target F1 {a['full_wall_f1']:.3f} acc {a['full_union_accuracy']:.3f}")

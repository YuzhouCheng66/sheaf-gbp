"""Dynamic multi-step evaluation (item 3): the 64-D owned code as a multi-agent memory. Claude 2026-10-22.

Protocols compared on the same T-step exploration episodes (dyn_data_v349), same trained encoder/decoder, zero-shot:
  crdt : max-precision selection with persistent per-node memory (join-semilattice / CRDT). Each node keeps the most precise
         factor it has ever heard per coordinate (+ the latest version it knows of every source); at each step it joins its new own
         code (version t), floods R rounds on the current radio forest, and decodes its memory. Diff packets: only coordinates
         whose wire code changed since the last packet on that directed edge are sent (+64-bit change mask + N version bytes).
  sum  : the sealed Gaussian-sum forest GBP, stateless per step (fuse the current versions reachable in the current forest only;
         full 172 B packets every round), which is what sum fusion can do without per-source journaling.
Targets: crdt -> union of the latest version of each source the node has actually heard of (tracked exactly); sum -> union of
current versions within R hops of the current forest. Metrics: sealed node_metrics per step. Bytes: per directed packet.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
import importlib.util

ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--trainer', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--size', type=int, default=128); ap.add_argument('--device', default='cuda')
ap.add_argument('--indices', default='0-7'); ap.add_argument('--rooms', default='6,12,20'); ap.add_argument('--agents', default='8,16')
ap.add_argument('--rounds', type=int, default=8); ap.add_argument('--T', type=int, default=6); ap.add_argument('--split', default='VALIDATION')
ap.add_argument('--radio', type=float, default=8., help='communication radius (m) used to re-derive the per-step radio graph; 8 = v293; smaller => intermittent connectivity')
args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid')); sys.path.insert(0, str(Path(__file__).resolve().parent))
spec_ = importlib.util.spec_from_file_location('trainer', args.trainer); trainer = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(trainer)
import dyn_data_v349 as dyn
from scripts.prepare_houseexpo_pilot_v121 import radius_adjacency
from scripts import mapping_native_bank_v306 as bank
from scripts import mapping_multiroom_forest_v295 as forest
from scripts.mapping_recognition_metrics_v277 import node_metrics
from scripts.mapping_implicit_model_v192 import latent_quantize

device = torch.device(args.device); size = args.size; T = args.T; R = args.rounds
payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
meta = payload['meta']
if hasattr(trainer, 'READOUT_MODE'):
    trainer.READOUT_MODE = meta.get('readout_mode', 'anchored'); trainer.PRECISION_MODE = meta.get('precision_mode', 'learned')
model = trainer.SheafModel(meta['arm'], size, 0).to(device); model.load_state_dict(payload['state_dict']); model.eval()
assert trainer.READOUT_MODE == 'posterior_only', 'decode() feeds the posterior into both halves; only valid for posterior_only checkpoints (review M6)'
rollout_sum = trainer.make_rollout(forest, latent_quantize, 'sum')
rollout_max = trainer.make_rollout(forest, latent_quantize, 'max')      # stateless max on the current forest: the control for 'memory' with the trained fusion rule (review M2)
LATENT_PRIOR = trainer.LATENT_PRIOR
PACKET = 172  # sealed v295 packet bytes (64 mean + 64 precision bytes + header)


def rng_list(text):
    out = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def join(p_list, e_list, v_list):
    """coordinatewise max precision with version tiebreak (newer wins ties). lists of [64] tensors / ints."""
    P = torch.stack(p_list); E = torch.stack(e_list); V = torch.tensor(v_list, dtype=torch.float64, device=P.device)[:, None]
    key = P + V * 1e-9
    idx = key.argmax(0, keepdim=True)
    return P.gather(0, idx)[0], E.gather(0, idx)[0], V.expand_as(P).gather(0, idx)[0]


def agg(rows_metrics, keys=('wall_f1', 'union_accuracy', 'unknown_known_fpr', 'own_wall_f1', 'peer_wall_f1', 'door_free_recall', 'pair_topology_accuracy', 'known_recall')):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows_metrics if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def decode(feats):
    with torch.no_grad():
        return trainer.predict(model.decoder, torch.cat((feats, feats), -1)).cpu().numpy().reshape(-1, size, size)


results = []
t0 = time.time()
with torch.no_grad():
    for rooms in [int(v) for v in args.rooms.split(',')]:
        for agents in [int(v) for v in args.agents.split(',')]:
            for index in rng_list(args.indices):
                fx = dyn.make_dynamic_fixture(args.split, index, size=size, rooms=rooms, agents=agents, T=T)
                maps, paths = fx['maps'], fx['paths_m']; n = agents
                adjs = np.stack([radius_adjacency(paths[t], args.radio) for t in range(T)]) if args.radio != 8. else fx['adjacency']
                public_frame = dict(bank.expected_frame(size)); public_frame['communication_radius_m'] = float(args.radio)
                prior = LATENT_PRIOR / (n * 64)
                # ---- CRDT memory state
                mem_p = [torch.full((64,), prior, dtype=torch.float64, device=device) for _ in range(n)]
                mem_e = [torch.zeros(64, dtype=torch.float64, device=device) for _ in range(n)]
                mem_pv = [torch.zeros(64, dtype=torch.float64, device=device) for _ in range(n)]   # version of the winning factor per coord
                V = -np.ones((n, n), dtype=int)                                                      # V[i][j]: latest version of source j known to i
                last_sent = {}; last_max = {}                                                        # (i,j) -> last packet codes (crdt: + version vector)
                bytes_crdt_diff = 0; bytes_crdt_full = 0; bytes_sum = 0; bytes_max_diff = 0; bytes_max_full = 0; packets_crdt = 0
                per_step = dict(crdt=[], sum=[], max=[], max_on_memory_target=[], own=[])
                for t in range(T):
                    own = torch.from_numpy(maps[t]).to(device)
                    mu, q, mask = model.source(own)
                    evid = mask.double() * q.double(); p_own = evid + prior; e_own = evid * mu.double()
                    pw, ew, _ = forest.quantize(p_own, e_own)
                    fsel = bank.select_forest(paths[t], adjs[t], float(args.radio))
                    graph = fsel['edges']; directed = [(i, j) for i, j, _ in graph] + [(j, i) for i, j, _ in graph]
                    incoming = {j: [a for a, (i2, j2) in enumerate(directed) if j2 == j] for j in range(n)}
                    # --- CRDT: join own (version t) into memory; version bookkeeping
                    for i in range(n):
                        V[i, i] = t
                        mem_p[i], mem_e[i], mem_pv[i] = join([mem_p[i], pw[i]], [mem_e[i], ew[i]], [0, t])
                        # (version tiebreak only matters for ties in p; own memory keeps earlier higher-precision factors otherwise)
                    msg = [(mem_p[i].clone(), mem_e[i].clone(), V[i].copy()) for i, _ in directed]  # initial outgoing = sender memory
                    for r in range(R):
                        new = []
                        for a, (i, j) in enumerate(directed):
                            cand_p = [mem_p[i]] + [msg[b][0] for b in incoming[i] if directed[b][0] != j]
                            cand_e = [mem_e[i]] + [msg[b][1] for b in incoming[i] if directed[b][0] != j]
                            pc, ec, _ = join(cand_p, cand_e, [0] * len(cand_p))
                            vv = V[i].copy()
                            for b in incoming[i]:
                                if directed[b][0] != j:
                                    vv = np.maximum(vv, msg[b][2])
                            new.append((pc, ec, vv))
                            # bytes: diff vs last packet on this directed edge
                            key = (i, j); codes = (pc.cpu().numpy().round(12), ec.cpu().numpy().round(12), vv.copy())
                            changed = int(((codes[0] != last_sent[key][0]) | (codes[1] != last_sent[key][1])).sum()) if key in last_sent else 64
                            vchanged = (key not in last_sent) or bool((vv != last_sent[key][2]).any())
                            packets_crdt += 1
                            bytes_crdt_full += PACKET + n
                            # header(8)+mask(8)+changed coords+version vector; a version-only packet costs header+vector; silent only if nothing changed (review M4)
                            bytes_crdt_diff += (8 + 8 + 2 * changed + n) if changed else ((8 + n) if vchanged else 0)
                            last_sent[key] = codes
                        msg = new
                        # receive: join incoming into memory (idempotent)
                        for j in range(n):
                            if incoming[j]:
                                mem_p[j], mem_e[j], _ = join([mem_p[j]] + [msg[b][0] for b in incoming[j]], [mem_e[j]] + [msg[b][1] for b in incoming[j]], [0] * (1 + len(incoming[j])))
                                for b in incoming[j]:
                                    V[j] = np.maximum(V[j], msg[b][2])
                    # decode CRDT memory
                    P = torch.stack(mem_p); E = torch.stack(mem_e)
                    feats_crdt = torch.cat((latent_quantize((E / P).float()), (1. / P).log().float()), -1)
                    pred_crdt = decode(feats_crdt)
                    target_crdt = np.full_like(maps[t], -1)
                    for i in range(n):
                        for j in range(n):
                            if V[i, j] >= 0:
                                target_crdt[i] = np.maximum(target_crdt[i], maps[V[i, j]][j])
                    m_crdt = node_metrics(pred_crdt, maps[t], target_crdt, {'door_mask': fx['door_mask']})
                    # --- stateless sum on the current forest (sealed reference, billed as full 172 B packets)
                    feats_sum, _ = rollout_sum(mu, q, mask, graph, rounds=R, stages=(0, R))
                    pred_sum = decode(feats_sum[R]); pred_own = decode(feats_sum[0])
                    # --- stateless max on the current forest (trained fusion rule, no memory), billed with the SAME diff/silence rule as the crdt (review M2/M3)
                    feats_max, _ = rollout_max(mu, q, mask, graph, rounds=R, stages=(R,))
                    pred_max = decode(feats_max[R])
                    msg_m = [(pw[i].clone(), ew[i].clone()) for i, _ in directed]
                    for r in range(R):
                        new_m = []
                        for a, (i, j) in enumerate(directed):
                            cand_p = [pw[i]] + [msg_m[b][0] for b in incoming[i] if directed[b][0] != j]
                            cand_e = [ew[i]] + [msg_m[b][1] for b in incoming[i] if directed[b][0] != j]
                            pc, ec, _ = join(cand_p, cand_e, [0] * len(cand_p))
                            new_m.append((pc, ec))
                            key = (i, j); codes = (pc.cpu().numpy().round(12), ec.cpu().numpy().round(12))
                            changed = int(((codes[0] != last_max[key][0]) | (codes[1] != last_max[key][1])).sum()) if key in last_max else 64
                            bytes_max_full += PACKET
                            bytes_max_diff += (8 + 8 + 2 * changed) if changed else 0
                            last_max[key] = codes
                        msg_m = new_m
                    inputs_t = dict(private=maps[t], positions_m=paths[t], trajectory_m=np.stack([paths[max(t - 2, 0)], paths[max(t - 1, 0)], paths[t]]), adjacency=adjs[t], public_frame=public_frame)
                    target_sum, reached = bank.forest_targets(inputs_t, R)
                    m_sum = node_metrics(pred_sum, maps[t], target_sum, {'door_mask': fx['door_mask']})
                    m_max = node_metrics(pred_max, maps[t], target_sum, {'door_mask': fx['door_mask']})
                    m_max_mem = node_metrics(pred_max, maps[t], target_crdt, {'door_mask': fx['door_mask']})   # stateless max scored on the memory target (same target as crdt)
                    m_own = node_metrics(pred_own, maps[t], maps[t], {'door_mask': fx['door_mask']})
                    bytes_sum += 2 * len(graph) * R * PACKET
                    per_step['crdt'].append(dict(step=t, **agg(m_crdt), target_known_cells=float((target_crdt >= 0).mean()), sources_known=float((V >= 0).sum(1).mean())))
                    per_step['sum'].append(dict(step=t, **agg(m_sum), target_known_cells=float((target_sum >= 0).mean()), sources_known=float(reached.sum(1).mean())))
                    per_step['max'].append(dict(step=t, **agg(m_max), target_known_cells=float((target_sum >= 0).mean()), sources_known=float(reached.sum(1).mean())))
                    per_step['max_on_memory_target'].append(dict(step=t, **agg(m_max_mem), target_known_cells=float((target_crdt >= 0).mean()), sources_known=float((V >= 0).sum(1).mean())))
                    per_step['own'].append(dict(step=t, **agg(m_own)))
                results.append(dict(spec=fx['spec'], per_step=per_step, bytes=dict(crdt_diff=bytes_crdt_diff, crdt_full=bytes_crdt_full, sum_full=bytes_sum, max_diff=bytes_max_diff, max_full=bytes_max_full, packets_crdt=packets_crdt)))
                last = per_step['crdt'][-1]; lsum = per_step['sum'][-1]
                print(f"r{rooms} n{agents} i{index} | t={T-1}: crdt F1 {last['wall_f1']:.3f} acc {last['union_accuracy']:.3f} fpr {last['unknown_known_fpr']:.3f} own {last['own_wall_f1']:.3f} srcs {last['sources_known']:.1f} | sum F1 {lsum['wall_f1']:.3f} acc {lsum['union_accuracy']:.3f} srcs {lsum['sources_known']:.1f} | own K0 {per_step['own'][-1]['wall_f1']:.3f} | bytes crdt-diff/sum-full {bytes_crdt_diff/bytes_sum:.3f} | {time.time()-t0:.0f}s", flush=True)

# summaries: equal weight per (rooms,N) stratum, per step and overall
summary = {}
for proto in ('crdt', 'sum', 'max', 'max_on_memory_target', 'own'):
    summary[proto] = {}
    for t in range(T):
        strata = {}
        for r in results:
            strata.setdefault((r['spec']['rooms'], r['spec']['agents']), []).append(r['per_step'][proto][t])
        keys = [k for k in results[0]['per_step'][proto][0] if k != 'step']
        per = {s: {k: (float(np.mean([x[k] for x in v if x.get(k) is not None])) if any(x.get(k) is not None for x in v) else None) for k in keys} for s, v in strata.items()}
        summary[proto][f't{t}'] = {k: (float(np.mean([p[k] for p in per.values() if p[k] is not None])) if any(p[k] is not None for p in per.values()) else None) for k in keys}
tot = {k: sum(r['bytes'][k] for r in results) for k in results[0]['bytes']}
summary['bytes_total'] = tot; summary['bytes_ratio_crdt_diff_over_sum_full'] = tot['crdt_diff'] / tot['sum_full']; summary['bytes_ratio_crdt_full_over_sum_full'] = tot['crdt_full'] / tot['sum_full']
summary['bytes_ratio_crdt_diff_over_max_diff'] = tot['crdt_diff'] / max(tot['max_diff'], 1); summary['bytes_ratio_max_diff_over_sum_full'] = tot['max_diff'] / tot['sum_full']   # like-for-like (same diff rule) and the stateless-max full bill
json.dump(dict(checkpoint=args.checkpoint, checkpoint_sha256=trainer.sha_file(args.checkpoint), meta=dict(arm=meta['arm'], fusion_trained=meta.get('fusion'), readout=meta.get('readout_mode')), args=vars(args), results=results, summary=summary), open(args.out, 'w'), indent=1)
for proto in ('own', 'sum', 'max', 'max_on_memory_target', 'crdt'):
    print(proto, {t: (round(v['wall_f1'], 3), round(v['union_accuracy'], 3), round(v['unknown_known_fpr'], 3), round(v['own_wall_f1'], 3)) for t, v in summary[proto].items()})
print('bytes', tot, 'crdt_diff/sum_full', round(summary['bytes_ratio_crdt_diff_over_sum_full'], 3), 'crdt_diff/max_diff', round(summary['bytes_ratio_crdt_diff_over_max_diff'], 3), 'max_diff/sum_full', round(summary['bytes_ratio_max_diff_over_sum_full'], 3))

"""Patch C from code_review_v350_dynamic_edge.md applied to eval_dynamic_v349.py (2026-10-22):
 - stateless MAX control on the current forest (same fusion rule as the trained decoder) -> per_step['max'];
 - like-for-like bytes: the stateless max protocol is billed with the SAME diff/silence rule as the CRDT (header 8 + mask 8 + 2 B per
   changed coordinate, silent if unchanged) -> bytes 'max_diff' and 'max_full'; crdt version-only packets cost header + vector;
 - assert the checkpoint readout is posterior_only (decode() feeds the posterior into both halves)."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else 'eval_dynamic_v349.py'
s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
rep("model = trainer.SheafModel(meta['arm'], size, 0).to(device); model.load_state_dict(payload['state_dict']); model.eval()\nrollout_sum = trainer.make_rollout(forest, latent_quantize, 'sum')\n",
    "model = trainer.SheafModel(meta['arm'], size, 0).to(device); model.load_state_dict(payload['state_dict']); model.eval()\n"
    "assert trainer.READOUT_MODE == 'posterior_only', 'decode() feeds the posterior into both halves; only valid for posterior_only checkpoints (review M6)'\n"
    "rollout_sum = trainer.make_rollout(forest, latent_quantize, 'sum')\n"
    "rollout_max = trainer.make_rollout(forest, latent_quantize, 'max')      # stateless max on the current forest: the control for 'memory' with the trained fusion rule (review M2)\n")
rep("                last_sent = {}                                                                       # (i,j) -> (p_codes, e_codes) last packet\n                bytes_crdt_diff = 0; bytes_crdt_full = 0; bytes_sum = 0; packets_crdt = 0\n                per_step = dict(crdt=[], sum=[], own=[])\n",
    "                last_sent = {}; last_max = {}                                                        # (i,j) -> last packet codes (crdt: + version vector)\n                bytes_crdt_diff = 0; bytes_crdt_full = 0; bytes_sum = 0; bytes_max_diff = 0; bytes_max_full = 0; packets_crdt = 0\n                per_step = dict(crdt=[], sum=[], max=[], own=[])\n")
rep("                            key = (i, j); codes = (pc.cpu().numpy().round(12), ec.cpu().numpy().round(12))\n                            if key in last_sent:\n                                changed = int(((codes[0] != last_sent[key][0]) | (codes[1] != last_sent[key][1])).sum())\n                            else:\n                                changed = 64\n                            packets_crdt += 1\n                            bytes_crdt_full += PACKET + n\n                            bytes_crdt_diff += (8 + 8 + 2 * changed + n) if changed else 0      # header+mask+changed coords+version vector; silent if nothing changed\n                            last_sent[key] = codes\n",
    "                            key = (i, j); codes = (pc.cpu().numpy().round(12), ec.cpu().numpy().round(12), vv.copy())\n                            changed = int(((codes[0] != last_sent[key][0]) | (codes[1] != last_sent[key][1])).sum()) if key in last_sent else 64\n                            vchanged = (key not in last_sent) or bool((vv != last_sent[key][2]).any())\n                            packets_crdt += 1\n                            bytes_crdt_full += PACKET + n\n                            # header(8)+mask(8)+changed coords+version vector; a version-only packet costs header+vector; silent only if nothing changed (review M4)\n                            bytes_crdt_diff += (8 + 8 + 2 * changed + n) if changed else ((8 + n) if vchanged else 0)\n                            last_sent[key] = codes\n")
rep("                    # --- stateless sum on the current forest\n                    feats_sum, _ = rollout_sum(mu, q, mask, graph, rounds=R, stages=(0, R))\n                    pred_sum = decode(feats_sum[R]); pred_own = decode(feats_sum[0])\n",
    "                    # --- stateless sum on the current forest (sealed reference, billed as full 172 B packets)\n                    feats_sum, _ = rollout_sum(mu, q, mask, graph, rounds=R, stages=(0, R))\n                    pred_sum = decode(feats_sum[R]); pred_own = decode(feats_sum[0])\n"
    "                    # --- stateless max on the current forest (trained fusion rule, no memory), billed with the SAME diff/silence rule as the crdt (review M2/M3)\n"
    "                    feats_max, _ = rollout_max(mu, q, mask, graph, rounds=R, stages=(R,))\n"
    "                    pred_max = decode(feats_max[R])\n"
    "                    msg_m = [(pw[i].clone(), ew[i].clone()) for i, _ in directed]\n"
    "                    for r in range(R):\n"
    "                        new_m = []\n"
    "                        for a, (i, j) in enumerate(directed):\n"
    "                            cand_p = [pw[i]] + [msg_m[b][0] for b in incoming[i] if directed[b][0] != j]\n"
    "                            cand_e = [ew[i]] + [msg_m[b][1] for b in incoming[i] if directed[b][0] != j]\n"
    "                            pc, ec, _ = join(cand_p, cand_e, [0] * len(cand_p))\n"
    "                            new_m.append((pc, ec))\n"
    "                            key = (i, j); codes = (pc.cpu().numpy().round(12), ec.cpu().numpy().round(12))\n"
    "                            changed = int(((codes[0] != last_max[key][0]) | (codes[1] != last_max[key][1])).sum()) if key in last_max else 64\n"
    "                            bytes_max_full += PACKET\n"
    "                            bytes_max_diff += (8 + 8 + 2 * changed) if changed else 0\n"
    "                            last_max[key] = codes\n"
    "                        msg_m = new_m\n")
rep("                    m_sum = node_metrics(pred_sum, maps[t], target_sum, {'door_mask': fx['door_mask']})\n",
    "                    m_sum = node_metrics(pred_sum, maps[t], target_sum, {'door_mask': fx['door_mask']})\n                    m_max = node_metrics(pred_max, maps[t], target_sum, {'door_mask': fx['door_mask']})\n")
rep("                    per_step['sum'].append(dict(step=t, **agg(m_sum), target_known_cells=float((target_sum >= 0).mean()), sources_known=float(reached.sum(1).mean())))\n",
    "                    per_step['sum'].append(dict(step=t, **agg(m_sum), target_known_cells=float((target_sum >= 0).mean()), sources_known=float(reached.sum(1).mean())))\n                    per_step['max'].append(dict(step=t, **agg(m_max), target_known_cells=float((target_sum >= 0).mean()), sources_known=float(reached.sum(1).mean())))\n")
rep("                results.append(dict(spec=fx['spec'], per_step=per_step, bytes=dict(crdt_diff=bytes_crdt_diff, crdt_full=bytes_crdt_full, sum_full=bytes_sum, packets_crdt=packets_crdt)))\n",
    "                results.append(dict(spec=fx['spec'], per_step=per_step, bytes=dict(crdt_diff=bytes_crdt_diff, crdt_full=bytes_crdt_full, sum_full=bytes_sum, max_diff=bytes_max_diff, max_full=bytes_max_full, packets_crdt=packets_crdt)))\n")
rep("for proto in ('crdt', 'sum', 'own'):\n    summary[proto] = {}\n", "for proto in ('crdt', 'sum', 'max', 'own'):\n    summary[proto] = {}\n")
rep("summary['bytes_total'] = tot; summary['bytes_ratio_crdt_diff_over_sum_full'] = tot['crdt_diff'] / tot['sum_full']; summary['bytes_ratio_crdt_full_over_sum_full'] = tot['crdt_full'] / tot['sum_full']\n",
    "summary['bytes_total'] = tot; summary['bytes_ratio_crdt_diff_over_sum_full'] = tot['crdt_diff'] / tot['sum_full']; summary['bytes_ratio_crdt_full_over_sum_full'] = tot['crdt_full'] / tot['sum_full']\n"
    "summary['bytes_ratio_crdt_diff_over_max_diff'] = tot['crdt_diff'] / max(tot['max_diff'], 1); summary['bytes_ratio_max_diff_over_sum_full'] = tot['max_diff'] / tot['sum_full']   # like-for-like (same diff rule) and the stateless-max full bill\n")
rep("for proto in ('own', 'sum', 'crdt'):\n    print(proto,", "for proto in ('own', 'sum', 'max', 'crdt'):\n    print(proto,")
rep("print('bytes', tot, 'crdt_diff/sum_full', round(summary['bytes_ratio_crdt_diff_over_sum_full'], 3))\n",
    "print('bytes', tot, 'crdt_diff/sum_full', round(summary['bytes_ratio_crdt_diff_over_sum_full'], 3), 'crdt_diff/max_diff', round(summary['bytes_ratio_crdt_diff_over_max_diff'], 3), 'max_diff/sum_full', round(summary['bytes_ratio_max_diff_over_sum_full'], 3))\n")
open(p, 'w').write(s)
print('patched', p)

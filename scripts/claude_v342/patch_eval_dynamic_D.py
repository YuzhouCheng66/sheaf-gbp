"""Patch D to eval_dynamic_v349.py (2026-10-22, after Patch C): intermittent-connectivity evaluation of the memory protocol.
 --radio R : re-derive the per-step radio adjacency / forest / targets with communication radius R m (default 8 = v293). With a
             smaller radius the forest fragments at some steps, which is the only regime where memory can matter.
 Extra scoring: the stateless max prediction is also scored against the MEMORY target (everything the node has heard so far),
 'max_on_memory_target', so both protocols are compared on the same target; 'crdt' keeps its own (memory) target and 'max' the
 currently-reachable target as before."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else 'eval_dynamic_v349.py'
s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
rep("ap.add_argument('--rounds', type=int, default=8); ap.add_argument('--T', type=int, default=6); ap.add_argument('--split', default='VALIDATION')\n",
    "ap.add_argument('--rounds', type=int, default=8); ap.add_argument('--T', type=int, default=6); ap.add_argument('--split', default='VALIDATION')\nap.add_argument('--radio', type=float, default=8., help='communication radius (m) used to re-derive the per-step radio graph; 8 = v293; smaller => intermittent connectivity')\n")
rep("import dyn_data_v349 as dyn\n", "import dyn_data_v349 as dyn\nfrom scripts.prepare_houseexpo_pilot_v121 import radius_adjacency\n")
rep("                maps, paths, adjs = fx['maps'], fx['paths_m'], fx['adjacency']; n = agents\n",
    "                maps, paths = fx['maps'], fx['paths_m']; n = agents\n                adjs = np.stack([radius_adjacency(paths[t], args.radio) for t in range(T)]) if args.radio != 8. else fx['adjacency']\n                public_frame = dict(bank.expected_frame(size)); public_frame['communication_radius_m'] = float(args.radio)\n")
rep("                    fsel = bank.select_forest(paths[t], adjs[t], 8.)\n", "                    fsel = bank.select_forest(paths[t], adjs[t], float(args.radio))\n")
rep("adjacency=adjs[t], public_frame=bank.expected_frame(size))\n", "adjacency=adjs[t], public_frame=public_frame)\n")
rep("                    m_max = node_metrics(pred_max, maps[t], target_sum, {'door_mask': fx['door_mask']})\n",
    "                    m_max = node_metrics(pred_max, maps[t], target_sum, {'door_mask': fx['door_mask']})\n                    m_max_mem = node_metrics(pred_max, maps[t], target_crdt, {'door_mask': fx['door_mask']})   # stateless max scored on the memory target (same target as crdt)\n")
rep("                per_step = dict(crdt=[], sum=[], max=[], own=[])\n", "                per_step = dict(crdt=[], sum=[], max=[], max_on_memory_target=[], own=[])\n")
rep("                    per_step['max'].append(dict(step=t, **agg(m_max), target_known_cells=float((target_sum >= 0).mean()), sources_known=float(reached.sum(1).mean())))\n",
    "                    per_step['max'].append(dict(step=t, **agg(m_max), target_known_cells=float((target_sum >= 0).mean()), sources_known=float(reached.sum(1).mean())))\n                    per_step['max_on_memory_target'].append(dict(step=t, **agg(m_max_mem), target_known_cells=float((target_crdt >= 0).mean()), sources_known=float((V >= 0).sum(1).mean())))\n")
rep("for proto in ('crdt', 'sum', 'max', 'own'):\n    summary[proto] = {}\n", "for proto in ('crdt', 'sum', 'max', 'max_on_memory_target', 'own'):\n    summary[proto] = {}\n")
rep("for proto in ('own', 'sum', 'max', 'crdt'):\n    print(proto,", "for proto in ('own', 'sum', 'max', 'max_on_memory_target', 'crdt'):\n    print(proto,")
open(p, 'w').write(s); print('patched', p)

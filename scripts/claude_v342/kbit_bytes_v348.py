"""Exact byte accounting for the k-bit wire under max-selection on the VALIDATION H128 forests (item 1; no model needed).
A directed message i->j at convergence carries, per coordinate, the most precise source factor on i's side of the edge; a coordinate is
'prior-only' iff no source in i's subtree observed that block. Two framings:
  fixed  : 64 coordinates x 2k bits (what train_v348 assumes on the wire)
  masked : 64-bit presence mask + 2k bits per non-prior coordinate (the owned 'empty blocks are free' dividend; the mask is the only
           side information needed because the receiver reconstructs prior-only coordinates exactly)
Reports mean bits per directed message and the mean total bytes per round per world, vs the sealed 172 B profile-4 packet.
python kbit_bytes_v348.py --source <sealed> [--out json]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', default=''); ap.add_argument('--ks', default='8,6,5,4,3,2')
args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid'))
from scripts import mapping_multiroom_data_v293 as data
from scripts import mapping_native_bank_v306 as bank
from scripts import train_mapping_support_native_v306 as base
specs = [s for s in base.scene_plan('VALIDATION') if s['size'] == 128]
ks = [int(k) for k in args.ks.split(',')]
per_msg_nonprior = []; msgs_per_world = []; observed_frac = []
for s in specs:
    fx = data.make_fixture(**s); inp = fx['inputs']
    edges = bank.select_forest(inp['positions_m'], inp['adjacency'], 8.)['edges']
    priv = torch.from_numpy(inp['private']); n = priv.shape[0]
    obs = (torch.nn.functional.avg_pool2d((priv >= 0).float()[:, None], 16).flatten(1) > 0).numpy()   # [N,64] block observed by agent
    observed_frac.append(obs.mean())
    adj = {i: [] for i in range(n)}
    for i, j, _ in edges:
        adj[i].append(j); adj[j].append(i)
    def subtree_union(root, exclude):
        """union of observed blocks over the component containing root when the edge root-exclude is cut."""
        seen = {root}; stack = [root]; u = np.zeros(64, bool)
        while stack:
            v = stack.pop(); u |= obs[v]
            for w in adj[v]:
                if w not in seen and not (v == root and w == exclude):
                    seen.add(w); stack.append(w)
        return u
    cnt = 0
    for i, j, _ in edges:
        for a, b in ((i, j), (j, i)):
            per_msg_nonprior.append(int(subtree_union(a, b).sum())); cnt += 1
    msgs_per_world.append(cnt)
per_msg_nonprior = np.array(per_msg_nonprior)
res = dict(n_scenes=len(specs), mean_directed_messages_per_round=float(np.mean(msgs_per_world)), mean_nonprior_coordinates_per_message=float(per_msg_nonprior.mean()),
           mean_observed_block_fraction_per_agent=float(np.mean(observed_frac)), sealed_packet_bytes=172, schemes={})
for k in ks:
    fixed_bits = 64 * 2 * k
    masked_bits = 64 + 2 * k * per_msg_nonprior
    res['schemes'][f'k{k}'] = dict(bits_per_coordinate=2 * k, fixed_bytes_per_message=fixed_bits / 8, masked_mean_bytes_per_message=float(masked_bits.mean() / 8),
                                   masked_bytes_per_round_per_world=float(masked_bits.mean() / 8 * np.mean(msgs_per_world)),
                                   fixed_over_sealed=fixed_bits / 8 / 172, masked_over_sealed=float(masked_bits.mean() / 8 / 172))
print(json.dumps(res, indent=1))
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)

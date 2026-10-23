"""Byte accounting for edge-stalk selection restriction maps (item 2): mean fraction of the 64 coordinates carried per directed
message for selection radius R, over the VALIDATION H128 forests (public geometry only). python edge_select_bytes.py --source <sealed>"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--radii', default='0,15,12,10,8,6'); ap.add_argument('--out', default='')
args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid')); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts import mapping_multiroom_data_v293 as data
from scripts import mapping_native_bank_v306 as bank
from scripts import train_mapping_support_native_v306 as base
import train_v350_dyn_edge as T
res = {}
specs = [s for s in base.scene_plan('VALIDATION') if s['size'] == 128]
for R in [float(v) for v in args.radii.split(',')]:
    T.EDGE_SELECT_M = R
    fr = []; blocks_obs_cut = []
    for s in specs:
        fx = data.make_fixture(**s); inp = fx['inputs']
        edges = bank.select_forest(inp['positions_m'], inp['adjacency'], 8.)['edges']
        priv = torch.from_numpy(inp['private'])
        obs = (torch.nn.functional.avg_pool2d((priv >= 0).float()[:, None], 16).flatten(1) > 0)
        for i, j, _ in edges:
            for a, b in ((i, j), (j, i)):
                sel = T.edge_selection(inp['positions_m'], a, b, 'cpu')
                fr.append(float(sel.mean()))
                # fraction of the sender's own observed blocks that the stalk drops (direct information loss on that edge)
                own = obs[a]
                if own.any():
                    blocks_obs_cut.append(float(((sel == 0) & own.bool()).sum() / own.sum()))
    res[str(R)] = dict(mean_fraction_coordinates_sent=float(np.mean(fr)), mean_fraction_sender_observed_blocks_cut=float(np.mean(blocks_obs_cut)))
    print(R, res[str(R)])
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)

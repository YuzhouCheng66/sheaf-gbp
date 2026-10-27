"""verify_recast.py (Claude, 2026-10-27): independent check of the claim every round-3 brainstorm relies on -- an agent's own ternary map
equals observe() run on ONLY its observed-occupied voxels (own == 1) from the same 3 poses and the same public ray set, voxel for voxel.
Also: the same with poses rounded to 1 cm and to float32, to see how much pose precision the identity needs.
CPU only; VAL houses, first `--per-stratum` per room stratum that place 8 agents (same rule as the demo export)."""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); ap.add_argument('--per-stratum', type=int, default=1)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mapping_3d_storey_data_v420 as W

rays = W.fibonacci_sphere(W.BEAMS); rows = []
for rooms in (1, 2, 3):
    taken = 0
    for index in range(40):
        if taken >= args.per_stratum:
            break
        try:
            fx = W.make_fixture('VALIDATION', index, rooms)
        except ValueError:
            continue
        taken += 1; t0 = time.time()
        own = fx['private']; cell = fx['cell']; traj = fx['trajectory_m']
        for a in range(own.shape[0]):
            seen = (own[a] == 1).astype(np.uint8)
            row = dict(house=f'r{rooms}_i{index}', agent=a, occupied_seen=int(seen.sum()))
            for name, poses in (('exact', traj[:, a]), ('float32', traj[:, a].astype(np.float32).astype(np.float64)), ('1cm', np.round(traj[:, a], 2))):
                rec = W.observe(seen, cell, poses, rays)
                row[f'mismatch_{name}'] = int((rec != own[a]).sum())
                row[f'mismatch_occ_{name}'] = int(((rec == 1) != (own[a] == 1)).sum())
            rows.append(row)
        print(f'r{rooms}_i{index}', [r['mismatch_exact'] for r in rows[-own.shape[0]:]], [r['mismatch_1cm'] for r in rows[-own.shape[0]:]], f'{time.time() - t0:.0f}s', flush=True)
S = {k: dict(maps=len(rows), maps_with_any_mismatch=sum(r[f'mismatch_{k}'] > 0 for r in rows), median_mismatch=float(np.median([r[f'mismatch_{k}'] for r in rows])),
             median_occ_mismatch=float(np.median([r[f'mismatch_occ_{k}'] for r in rows]))) for k in ('exact', 'float32', '1cm')}
json.dump(dict(summary=S, rows=rows), open(args.out, 'w'), indent=1); print(json.dumps(S, indent=1))

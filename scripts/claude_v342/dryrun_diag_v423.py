"""Dry run of the v423 diag copies on a 16x16x1 checkpoint before 3S-C (Claude, 2026-10-25; review v423 S2). CPU only.
Writes an UNTRAINED 16x16x1 model as runs/v423-dryrun/fake-s342101/checkpoints/final.pt (meta: seed, blocks, D), then the caller runs
diag_storey_v423.py / diag_interior_v423.py on it with --trainer train_3d_v423 --device cpu, and checks that train_3d_v420 is refused.
python dryrun_diag_v423.py --source <sealed> --out runs/v423-dryrun/fake-s342101
"""
import argparse, os, sys
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_3d_v423 as T
T.BX, T.BY, T.BZ = T.LAYOUTS['16x16x1']
m = T.SheafModel3D((128, 128, 32), 342101)
out = Path(args.out); (out / 'checkpoints').mkdir(parents=True, exist_ok=False)
torch.save(dict(state_dict=m.state_dict(), meta=dict(seed=342101, blocks=[T.BX, T.BY, T.BZ], D=T.D, note='untrained dry-run checkpoint (review v423 S2)'), updates=0),
           out / 'checkpoints' / 'final.pt')
print('wrote', out / 'checkpoints' / 'final.pt')

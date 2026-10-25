"""v427 copy (decoder / precision / known-weight from the checkpoint meta too) of diag_interior_v420.py (Claude, 2026-10-25): identical computation, but the model configuration (decoder width, block layout) is
taken from the checkpoint meta, so the same script scores v420 / v422 / v423 checkpoints. Pass --trainer train_3d_v423.
"""
"""Post-hoc (report-only) split of the 3S-A wall recall into INTERIOR and BOUNDARY walls (Claude, 2026-10-25, D8-3).
A target-occupied wall / lintel voxel (struct) is INTERIOR when, in its own horizontal plane, the target has a known-FREE voxel within
W = 4 cells on BOTH sides along x or along y (the wall separates two observed free spaces: only the code's content can place it);
otherwise BOUNDARY (the wall is where the observed region ends: the known / unknown outline already implies it).
Reports recall of each class at K0 and R8 on the 24 VAL storeys, and the share of interior walls.
python diag_interior_v420.py --source <sealed> --run runs/v420-storey/a-s342101 [--trainer train_3d_v422 --decoder-width 2.0]
"""
import argparse, importlib, multiprocessing, os, sys, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--run', required=True); ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--device', default='cuda'); ap.add_argument('--trainer', default='train_3d_v420'); ap.add_argument('--decoder-width', type=float, default=None); ap.add_argument('--w', type=int, default=4)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['SEALED_SOURCE'] = args.source
T = importlib.import_module(args.trainer)
if args.decoder_width is not None:
    T.DECODER_WIDTH = args.decoder_width
import mapping_3d_storey_data_v420 as W


def side_free(free, w, axis, sign):
    """[N, X, Y, Z] bool: a free voxel exists within 1..w cells in direction sign along axis (horizontal axes 1 or 2)."""
    out = torch.zeros_like(free)
    for d in range(1, w + 1):
        out |= torch.roll(free, shifts=sign * d, dims=axis)      # chart border cells are unknown / solid, so wrap-around is harmless here
    return out


if __name__ == '__main__':
    dims = W.DIMS['R1']; dev = torch.device(args.device)
    plan = [s for s in W.scene_plan('VALIDATION', 16, dims) if s['index'] < 8]
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        fxs = pool.map_async(T._fixture, plan, chunksize=1).get(timeout=3600)
    ck = torch.load(Path(args.run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False)
    if hasattr(T, 'DECODER'):                                          # v427: decoder / precision / known-weight from the checkpoint
        T.DECODER = ck['meta'].get('decoder', 'd3'); T.PRECISION = ck['meta'].get('precision', 'learned'); T.KNOWN_WEIGHT = ck['meta'].get('known_weight', 'uniform')
    if hasattr(T, 'LAYOUTS'):                                          # v423: block layout from the checkpoint
        T.BX, T.BY, T.BZ = ck['meta']['blocks']; assert T.BX * T.BY * T.BZ == T.D
    if hasattr(T, 'DECODER_WIDTH') and args.decoder_width is None:
        T.DECODER_WIDTH = ck['meta'].get('decoder_width', 1.0)
    assert ck['meta'].get('decoder_width', 1.0) == getattr(T, 'DECODER_WIDTH', 1.0)
    assert list(ck['meta'].get('blocks', [8, 8, 4])) == [getattr(T, 'BX', 8), getattr(T, 'BY', 8), getattr(T, 'BZ', 4)], 'trainer / checkpoint layout mismatch (review v423 S1)'
    model = T.SheafModel3D(dims, ck['meta']['seed']); model.load_state_dict(ck['state_dict']); model.to(dev).eval()
    acc = {(k, c): [0, 0] for k in (0, 8) for c in ('interior', 'boundary')}
    with torch.no_grad():
        for fx in fxs:
            if fx is None:
                continue
            T.CHART_M = fx['chart_m']
            own = torch.from_numpy(fx['private']).to(dev); reached = torch.from_numpy(fx['reached']).to(dev)
            struct = torch.from_numpy(fx['struct']).to(dev)[None]
            mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, fx['edges'])
            for k in (0, 8):
                tgt = T.stream_targets(own, reached[T.HOPS.index(k)]).reshape(-1, *dims); pred = T.predict(model.decoder, feats[k]).reshape(-1, *dims)
                free = tgt == 0
                inter = (side_free(free, args.w, 1, 1) & side_free(free, args.w, 1, -1)) | (side_free(free, args.w, 2, 1) & side_free(free, args.w, 2, -1))
                wall = (tgt == 1) & struct
                for c, m in (('interior', wall & inter), ('boundary', wall & ~inter)):
                    acc[(k, c)][0] += int((m & (pred == 1)).sum()); acc[(k, c)][1] += int(m.sum())
    res = {f'K{k}_{c}': dict(recall=round(v[0] / max(1, v[1]), 3), voxels=v[1]) for (k, c), v in acc.items()}
    for k in (0, 8):
        res[f'K{k}_interior_share'] = round(acc[(k, 'interior')][1] / max(1, acc[(k, 'interior')][1] + acc[(k, 'boundary')][1]), 3)
    print(json.dumps(res))

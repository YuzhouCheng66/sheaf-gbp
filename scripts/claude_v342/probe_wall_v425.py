"""Wall-relocation probe (Claude, 2026-10-25; analyst C memo section 1.3 made a pinned, deterministic mechanism readout). CPU, seconds.
One agent, chart L = 12.8 m (10 cm cells at R1), a known disk of radius 5 m (floor z = 0 and ceiling z = 28 occupied, z 1-27 free, z > 28
unknown), and a vertical wall plane at voxel column x = j spanning the disk, for every j inside ONE public block along x (block 3: j = 48..63
for the 8x8x4 layout, j = 24..31 for 16x16x1). Two variants: INTERIOR (both sides of the wall known free) and BOUNDARY (everything behind
the wall unknown). The K0 decode (own code only, no fusion) is read per column x: the occupied fraction over the disk's y and z in 3..26; the
wall counts as DRAWN when that fraction is >= .5 within +-1 voxel of j. Reports drawn / total for both variants, the peak column per j and the
codes of the probed block. Model configuration (layout, width, decoder) comes from the checkpoint meta via --trainer.
python probe_wall_v425.py --source <sealed> --run runs/v422-decoder/a-s342101 --trainer train_3d_v422 --out FILE.json
"""
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--trainer', default='train_3d_v425'); ap.add_argument('--block', type=int, default=3); ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(args.threads)
T = importlib.import_module(args.trainer)
ck = torch.load(Path(args.run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False); meta = ck['meta']
if hasattr(T, 'DECODER_WIDTH'): T.DECODER_WIDTH = meta.get('decoder_width', 1.0)
if hasattr(T, 'LAYOUTS'): T.BX, T.BY, T.BZ = meta.get('blocks', [8, 8, 4]); T.D = T.BX * T.BY * T.BZ
if hasattr(T, 'DECODER'): T.DECODER = meta.get('decoder', 'd3')
assert list(meta.get('blocks', [8, 8, 4])) == [getattr(T, 'BX', 8), getattr(T, 'BY', 8), getattr(T, 'BZ', 4)]
dims = (128, 128, 32); NX, NY, NZ = dims; T.CHART_M = 12.8
model = T.SheafModel3D(dims, meta['seed']); model.load_state_dict(ck['state_dict']); model.eval()
cx = cy = 64; r = 50
xx, yy = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij'); disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
bs = NX // T.BX; js = list(range(args.block * bs, (args.block + 1) * bs))

def scene(j, interior):
    m = np.full(dims, -1, dtype=np.int8)
    m[disk, 1:28] = 0; m[disk, 0] = 1; m[disk, 28] = 1
    m[j, disk[j], 1:28] = 1
    if not interior:
        m[j + 1:, :, :] = -1
    return m

res = {}
with torch.no_grad():
    for variant, interior in (('interior', True), ('boundary', False)):
        rows = []
        for j in js:
            own = torch.from_numpy(scene(j, interior))[None]
            mu, q, mask = model.source(own)
            pred = T.predict(model.decoder, T.rollout_max(mu, q, mask, [], stages=(0,))[0]).reshape(dims)
            frac = [float(((pred[x][disk[x]][:, 3:27]) == 1).float().mean()) if disk[x].any() else 0. for x in range(NX)]
            peak = int(np.argmax(frac)); drawn = max(frac[j - 1], frac[j], frac[j + 1]) >= .5
            k = (args.block * T.BY * T.BZ + (cy // (NY // T.BY)) * T.BZ)                       # block (3, 4, z0): coordinate index of its lowest layer
            rows.append(dict(j=j, drawn=bool(drawn), peak=peak, peak_frac=round(frac[peak], 3), frac_at_j=round(frac[j], 3),
                             mu_z0=round(float(mu[0, k]), 3), q_z0=round(float(q[0, k]), 3)))
        res[variant] = dict(drawn=sum(r['drawn'] for r in rows), total=len(rows), rows=rows)
res['meta'] = dict(run=args.run, seed=meta['seed'], blocks=meta.get('blocks', [8, 8, 4]), decoder_width=meta.get('decoder_width', 1.0), decoder=meta.get('decoder', 'd3'), block=args.block, positions=js)
json.dump(res, open(args.out, 'w'), indent=1)
print(f"probe {args.run}: interior {res['interior']['drawn']}/{res['interior']['total']} boundary {res['boundary']['drawn']}/{res['boundary']['total']}")
print(' interior peaks', [(r['j'], r['peak']) for r in res['interior']['rows']])
print(' interior mu_z0', [r['mu_z0'] for r in res['interior']['rows']])

"""Figure of v420 furnished storeys (Claude, 2026-10-25): per house a 3D voxel view (ceiling removed; walls grey, furniture coloured by
height) and the 2D slice 0.15-1.6 m above the floor (white free, black occupied), at R1 = 128 x 128 x 32.
python fig_storey_v420.py --source <sealed> --out FILE --houses TRAIN:3:2 TRAIN:5:3 VALIDATION:1:1"""
import argparse, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); ap.add_argument('--houses', nargs='+', required=True)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, args.source + '/exploration_grid'); sys.path.insert(0, 'scripts')
import mapping_3d_storey_data_v420 as W
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'figure.dpi': 170})
fig = plt.figure(figsize=(11, 3.6 * len(args.houses)))
for r, h in enumerate(args.houses):
    split, idx, rooms = h.split(':')
    sp = W.world_spec(split, int(idx), int(rooms)); ras = W.rasterize(sp, W.DIMS['R1']); occ = ras['occ'].astype(bool); obj = ras['obj']; cx, _, cz = ras['cell']
    zc = (np.arange(occ.shape[2]) + .5) * cz
    keep = (zc >= W.FLOOR_T) & (zc < W.FLOOR_T + sp['wall_h'])                    # hide floor and ceiling slabs
    wall = occ & ~obj; wall[:, :, ~keep] = False
    # walls: only their surface layer (6-neighbour free) to keep the scatter light
    free = ~occ
    nb = np.zeros_like(occ)
    nb[1:] |= free[:-1]; nb[:-1] |= free[1:]; nb[:, 1:] |= free[:, :-1]; nb[:, :-1] |= free[:, 1:]
    ws = np.argwhere(wall & nb); os_ = np.argwhere(obj)
    ax = fig.add_subplot(len(args.houses), 2, 2 * r + 1, projection='3d')
    ax.scatter(ws[:, 0] * cx, ws[:, 1] * cx, ws[:, 2] * cz, s=0.25, c='#9a9a9a', alpha=0.35, linewidths=0, depthshade=False)
    ax.scatter(os_[:, 0] * cx, os_[:, 1] * cx, os_[:, 2] * cz, s=0.6, c=plt.cm.plasma(np.clip(os_[:, 2] * cz / 2.2, 0, 1)), linewidths=0, depthshade=False)
    L = sp['chart_m']; ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, 3.2); ax.set_box_aspect((1, 1, 3.2 / L)); ax.view_init(elev=42, azim=-58); ax.set_axis_off()
    kinds = {}
    for o in sp['objects']:
        kinds[o['kind']] = kinds.get(o['kind'], 0) + 1
    ax.set_title(f"{split} {idx} (stratum {rooms}): L = {L:.0f} m, ceiling {sp['wall_h']:.2f} m, {len(sp['objects'])} objects", pad=0)
    band = (zc >= W.FLOOR_T + 0.15) & (zc < W.FLOOR_T + 1.6)
    sl = occ[:, :, band].any(2)
    ax2 = fig.add_subplot(len(args.houses), 2, 2 * r + 2)
    ax2.imshow(sl.T, cmap='gray_r', origin='lower', interpolation='nearest', extent=(0, L, 0, L)); ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title('2D slice 0.15-1.6 m: ' + ', '.join(f'{k} {v}' for k, v in sorted(kinds.items(), key=lambda t: -t[1])[:7]), pad=3, fontsize=6.5)
fig.tight_layout(); fig.savefig(args.out)
print('saved', args.out)

"""Figures of the batch 3D-A (v400, 64^3) fusion result from a saved validation panel (Claude, 2026-10-24).
Inputs: a final-val-panel-<si>.npz written by train_3d_v400 (own maps, GT occupancy, predictions per hop) + the regenerated fixture
(positions, forest, reached sets; v401 at 64^3 == v400). Outputs (Nature-style, English):
  fig1_<tag>_storeys.png   per storey (rows): ground truth | agent's own map | decoded from own code (K0) | reachable union (R8 target) | decoded after fusion (R8)
  fig2_<tag>_hops.png      one storey, one agent: own map -> K0 -> R1 -> R2 -> R4 -> R8 (how the map fills in with hops)
  fig3_<tag>_3d.png        3D voxel view per storey of the fused (R8) map: hits / false alarms / misses against the reachable union
python fig_3d_v400_panels.py --panel PATH.npz --scene-index 1000000 --out DIR --tag s342202-p0
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapping_3d_data_v401 as d3
import train_3d_v401 as T

ap = argparse.ArgumentParser()
ap.add_argument('--panel', required=True); ap.add_argument('--scene-index', type=int, required=True); ap.add_argument('--out', required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--agent', type=int, default=-1, help='agent to show (-1 = the one with median own coverage)')
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
z = np.load(args.panel); own = z['own']; occ = z['occ']; preds = {k: z[f'pred_k{k}'] for k in (0, 1, 2, 4, 8)}
S = occ.shape[0]; cell = d3.EXTENT / S; n = own.shape[0]
fx = d3.make_fixture('VALIDATION', args.scene_index, size=S, agents=n)
assert np.array_equal(fx['private'], own) and np.array_equal(fx['occ'], occ), 'panel does not match the regenerated fixture'
pos = fx['positions_m']; edges = fx['edges']; reached = fx['reached']            # [H,N,N] for hops (0,1,2,4,8)
hop_index = {0: 0, 1: 1, 2: 2, 4: 3, 8: 4}
cover = (own >= 0).mean((1, 2, 3))
a = int(np.argsort(cover)[n // 2]) if args.agent < 0 else args.agent
storey_of = lambda p: int(p[2] // d3.STOREY)

def union(k, i):                                                                 # stream target for node i at hop k: ternary union of reached own maps
    r = reached[hop_index[k]][i]
    return own[r].max(0)

def f1s(pred, tgt):
    rows = T.metrics3d(torch.from_numpy(pred.reshape(1, -1)), torch.from_numpy(tgt.reshape(1, -1)), S)[0]
    return rows['surface_f1'], rows['var_occ_f1']

# ---- style
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'figure.dpi': 200, 'savefig.dpi': 200, 'axes.linewidth': 0.5})
cmap = ListedColormap(['#e9e9e9', '#ffffff', '#1a1a1a'])                            # unknown, free, occupied  (values -1, 0, 1)
def show(ax, vol, zi, title=None, dots=None, highlight=None):
    ax.imshow(vol[:, :, zi].T, cmap=cmap, vmin=-1, vmax=1, origin='lower', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color('#999999')
    if dots is not None:
        for j, p in enumerate(dots):
            ax.plot(p[0] / cell - .5, p[1] / cell - .5, 'o', ms=3.2 if j == highlight else 2.2, mfc='#d62728' if j == highlight else '#1f77b4', mec='white', mew=0.4)
    if title:
        ax.set_title(title, pad=2)

# ---- fig 1: storeys
zi_of = lambda f: f * (S // d3.FLOORS) + int(round(0.8 / cell)) + 1              # 1.1 m above the slab bottom (0.8 m above the floor top)
t8 = union(8, a); t0 = own[a]
sf0, vf0 = f1s(preds[0][a], t0); sf8, vf8 = f1s(preds[8][a], t8)
fig, axes = plt.subplots(d3.FLOORS, 5, figsize=(7.2, 6.0))
cols = ['Ground truth', f'Own map\n(agent {a})', 'Decoded from\nown code (K0)', 'Reachable union\n(R8 target)', 'Decoded after\nfusion (R8)']
for r, f in enumerate(reversed(range(d3.FLOORS))):
    zi = zi_of(f)
    dots = [p for p in pos if storey_of(p) == f]; hl = next((j for j, (p, i) in enumerate([(p, i) for i, p in enumerate(pos) if storey_of(p) == f]) if i == a), None)
    show(axes[r, 0], occ.astype(np.int8), zi, cols[0] if r == 0 else None, dots=dots, highlight=hl)
    show(axes[r, 1], t0, zi, cols[1] if r == 0 else None)
    show(axes[r, 2], preds[0][a], zi, cols[2] if r == 0 else None)
    show(axes[r, 3], t8, zi, cols[3] if r == 0 else None)
    show(axes[r, 4], preds[8][a], zi, cols[4] if r == 0 else None)
    axes[r, 0].set_ylabel(f'Storey {f + 1}\nz = {zi * cell + cell / 2:.1f} m', rotation=0, ha='right', va='center', labelpad=4)
fig.text(0.5, 0.012, f'64$^3$ voxels (0.2 m), 8 agents, 3 poses each, 5 m range, 512 B per message. Grey = unknown, white = free, black = occupied.\n'
         f'Dots: agent positions on that storey (red = agent {a}). Agent {a}: surface F1 K0 {sf0:.3f}, R8 {sf8:.3f}; variable-structure F1 K0 {vf0:.3f}, R8 {vf8:.3f}.', ha='center', fontsize=6.3, color='#444444', linespacing=1.4)
fig.subplots_adjust(left=0.11, right=0.995, top=0.94, bottom=0.065, wspace=0.05, hspace=0.08)
fig.savefig(out / f'fig1_{args.tag}_storeys.png'); plt.close(fig)

# ---- fig 2: hops on the agent's own storey
f = storey_of(pos[a]); zi = zi_of(f)
fig, axes = plt.subplots(2, 6, figsize=(7.2, 2.75))
labels = ['Own map', 'K0', 'R1', 'R2', 'R4', 'R8']
vols_t = [own[a]] + [union(k, a) for k in (0, 1, 2, 4, 8)]
vols_p = [own[a]] + [preds[k][a] for k in (0, 1, 2, 4, 8)]
for c in range(6):
    show(axes[0, c], vols_t[c], zi, labels[c])
    show(axes[1, c], vols_p[c], zi)
    if c >= 1:
        k = (0, 1, 2, 4, 8)[c - 1]; sfk, _ = f1s(preds[k][a], vols_t[c]); nsrc = int(reached[hop_index[k]][a].sum())
        axes[1, c].set_xlabel(f'{nsrc} src, F1 {sfk:.3f}', fontsize=6.5)
axes[0, 0].set_ylabel('Target\n(reachable union)', rotation=0, ha='right', va='center', labelpad=4); axes[1, 0].set_ylabel('Decoded', rotation=0, ha='right', va='center', labelpad=4)
fig.suptitle(f'Agent {a}, storey {f + 1}, z = {zi * cell + cell / 2:.1f} m: what the agent can know (top) and what it decodes from the fused 512 B codes (bottom)', fontsize=7.5, y=0.995)
fig.subplots_adjust(left=0.13, right=0.995, top=0.88, bottom=0.1, wspace=0.05, hspace=0.12)
fig.savefig(out / f'fig2_{args.tag}_hops.png'); plt.close(fig)

# ---- fig 3: 3D voxel view of the fused (R8) decode against the reachable union; the world-independent skeleton (exterior walls, slabs)
# is removed so that the interior structure (partitions, corridor walls, door edges, furniture) is visible; the whole-building views are
# EXPLODED (storeys separated by GAP metres) and slabs are drawn as outlines
p8 = preds[8][a]
skel = d3.skeleton_mask(S)
tp = (p8 == 1) & (t8 == 1) & ~skel; fp = (p8 == 1) & (t8 == 0) & ~skel; fn = (p8 != 1) & (t8 == 1) & ~skel
gt_int = (occ == 1) & ~skel
COL = dict(hit='#2b5aa6', miss='#ff9f1c', fa='#d62728', gt='#b0b0b0')
SZ = S // d3.FLOORS; GAP = 1.6

def voxels_storey(ax, masks, f, zshift):
    z0, z1 = f * SZ, (f + 1) * SZ
    filled = np.zeros((S, S, SZ), bool); colors = np.empty(filled.shape, object)
    for m, c in masks:
        mm = m[:, :, z0:z1]; filled |= mm; colors[mm] = c
    if filled.any():
        x, y, zz = np.meshgrid(np.arange(S + 1) * cell, np.arange(S + 1) * cell, np.arange(z0, z1 + 1) * cell + zshift, indexing='ij')
        ax.voxels(x, y, zz, filled, facecolors=colors, edgecolor=(0, 0, 0, 0.12), linewidth=0.15)

def frame(ax, floors, zshift_of, zlim):
    for f in floors:
        zt = f * d3.STOREY + zshift_of(f)
        ax.plot([0, d3.EXTENT, d3.EXTENT, 0, 0], [0, 0, d3.EXTENT, d3.EXTENT, 0], [zt] * 5, color='#777777', lw=0.5)
    ax.set_xlim(0, d3.EXTENT); ax.set_ylim(0, d3.EXTENT); ax.set_zlim(*zlim)
    ax.set_xticks([0, 6.4, 12.8]); ax.set_yticks([0, 6.4, 12.8]); ax.set_zticks([]); ax.tick_params(labelsize=6, pad=-2)
    ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0); ax.zaxis.pane.set_alpha(0); ax.grid(False); ax.zaxis.line.set_linewidth(0)
    for axis in (ax.xaxis, ax.yaxis):
        axis.line.set_linewidth(0.5)

fig = plt.figure(figsize=(11, 8.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1.9, 1], wspace=0.0, hspace=0.1, left=0.0, right=1.0, top=0.93, bottom=0.04)
dec_masks = [(tp, COL['hit']), (fn, COL['miss']), (fp, COL['fa'])]
for col, (title, masks) in enumerate((('Ground truth: interior structure, exploded by storey\n(exterior walls and slabs hidden; slab outlines drawn)', [(gt_int, COL['gt'])]),
                                       (f'Agent {a}, decoded after fusion (R8), exploded by storey\nhit / miss / false alarm against the union of the maps it can reach', dec_masks))):
    ax = fig.add_subplot(gs[0, 2 * col:2 * col + 2], projection='3d')
    for f in range(d3.FLOORS):
        voxels_storey(ax, masks, f, f * GAP)
    frame(ax, range(d3.FLOORS), lambda f: f * GAP, (0, d3.EXTENT + (d3.FLOORS - 1) * GAP))
    for j, p in enumerate(pos):
        ax.scatter([p[0]], [p[1]], [p[2] + storey_of(p) * GAP], s=30, c=COL['fa'] if j == a else '#111111', marker='^', depthshade=False, zorder=10)
    ax.set_box_aspect((1, 1, 1.3), zoom=1.15); ax.view_init(elev=22, azim=-50); ax.set_title(title, pad=4, fontsize=7.5)
for f in range(d3.FLOORS):                                                   # per storey, decoded, true scale
    ax = fig.add_subplot(gs[1, f], projection='3d')
    voxels_storey(ax, dec_masks, f, 0.); frame(ax, [f], lambda f: 0., (f * d3.STOREY, (f + 1) * d3.STOREY))
    for j, p in enumerate(pos):
        if storey_of(p) == f:
            ax.scatter([p[0]], [p[1]], [p[2]], s=24, c=COL['fa'] if j == a else '#111111', marker='^', depthshade=False, zorder=10)
    ax.set_box_aspect((1, 1, 0.3), zoom=1.35); ax.view_init(elev=32, azim=-50)
    z0, z1 = f * SZ, (f + 1) * SZ
    ax.set_title(f'Storey {f + 1}: hit {int(tp[:, :, z0:z1].sum()):,}, miss {int(fn[:, :, z0:z1].sum()):,}, false alarm {int(fp[:, :, z0:z1].sum()):,}', pad=2, fontsize=6.8)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=COL['gt'], label='ground truth'), Patch(color=COL['hit'], label='hit (decoded occupied, target occupied)'), Patch(color=COL['miss'], label='miss (target occupied, not decoded)'),
                    Patch(color=COL['fa'], label='false alarm (decoded occupied, target free)')], loc='lower center', ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f'Batch 3D-A, 64$^3$ voxels (0.2 m), 8 agents, 512 B per message. Agent {a} after 8 rounds: surface F1 {sf8:.3f} against the reachable union (only voxels known in that union are scored). Triangles: agents, red = agent {a}.', fontsize=7.2, y=0.992)
fig.savefig(out / f'fig3_{args.tag}_3d.png'); plt.close(fig)
print('agent', a, 'coverage', round(float(cover[a]), 3), 'storey', storey_of(pos[a]), 'K0 surf/var', round(sf0, 3), round(vf0, 3), 'R8 surf/var', round(sf8, 3), round(vf8, 3), 'sources at R8', int(reached[4][a].sum()), 'edges', len(edges))

"""Figure for D8-3 (Claude, 2026-10-25): horizontal slices of one 3S-A VAL storey for one agent, target vs K0 / R8 reconstruction.
Target: slab grey-blue, wall / lintel dark blue, furniture orange, free white, unknown light grey.
Predictions: true positive black, false positive red, missed (target occupied, predicted free / unknown) cyan.
python fig_3sa_diag_v420.py --dir <diag dir with slices-*.npz> --scene 1 --agent 0 --out FILE.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--scene', type=int, default=1); ap.add_argument('--agent', type=int, default=0)
ap.add_argument('--out', required=True); args = ap.parse_args()
k0 = np.load(f'{args.dir}/slices-{args.scene}-K0.npz'); k8 = np.load(f'{args.dir}/slices-{args.scene}-K8.npz')
a = args.agent; L = float(k8['chart_m'])
heights = [(5, '0.5 m'), (12, '1.2 m'), (25, '2.5 m')]

def target_rgb(t, z):
    tt = t[:, :, z]; img = np.full(tt.shape + (3,), .85)
    img[tt == 0] = 1.
    occ = tt == 1
    img[occ & k8['slab'][:, :, z]] = (.55, .62, .75); img[occ & k8['struct'][:, :, z]] = (.10, .20, .45); img[occ & k8['obj'][:, :, z]] = (.95, .55, .10)
    return img

def pred_rgb(p, t, z):
    pp, tt = p[:, :, z], t[:, :, z]; img = np.full(tt.shape + (3,), .85)
    img[tt == 0] = 1.
    img[(pp == 1) & (tt == 1)] = 0.; img[(pp == 1) & (tt == 0)] = (.85, .10, .10); img[(pp != 1) & (tt == 1)] = (.10, .80, .85)
    return img

fig, ax = plt.subplots(len(heights), 3, figsize=(10.5, 3.5 * len(heights)))
for r, (z, name) in enumerate(heights):
    panels = [(target_rgb(k8['target'][a], z), f'target (R8 scope), z = {name}'), (pred_rgb(k0['pred'][a], k0['target'][a], z), f'K0 (own code only), z = {name}'),
              (pred_rgb(k8['pred'][a], k8['target'][a], z), f'R8 (fused), z = {name}')]
    for c, (img, title) in enumerate(panels):
        ax[r, c].imshow(np.transpose(img, (1, 0, 2)), origin='lower', extent=(0, L, 0, L), interpolation='nearest'); ax[r, c].set_title(title, fontsize=9)
        ax[r, c].set_xticks([0, L]); ax[r, c].set_yticks([0, L]); ax[r, c].tick_params(labelsize=7)
fig.suptitle(f'3S-A (seed 342101), VAL storey {args.scene}, agent {a}, chart {L:.0f} m x {L:.0f} m x 3.2 m\n'
             'target: wall dark blue, furniture orange, slab grey-blue | prediction: TP black, FP red, missed cyan, unknown grey', fontsize=9)
fig.tight_layout(); fig.savefig(args.out, dpi=110)
print('wrote', args.out)

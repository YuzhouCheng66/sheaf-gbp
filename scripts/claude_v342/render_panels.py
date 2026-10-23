"""Render fixed, unselected raw panels (own observation | arrived GT | prediction) for stages R0,R1,R8 from the saved npz
validation panels of a v342 run. No post-processing of predictions. Usage: python render_panels.py <run_dir> <label> [<out_png_dir>]"""
import sys, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run = Path(sys.argv[1]); label = sys.argv[2]
outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else run / 'panels_png'
outdir.mkdir(parents=True, exist_ok=True)
cmap = matplotlib.colors.ListedColormap(['#b8bec4', '#f7f5f0', '#1a1a1a'])  # unknown grey, free white, wall black


def show(ax, arr, title):
    ax.imshow(arr, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest'); ax.set_title(title, fontsize=8); ax.axis('off')


for f in sorted(glob.glob(str(run / 'validation' / f'{label}-panel-*.npz'))):
    d = np.load(f); name = Path(f).stem.replace(f'{label}-panel-', '')
    agent = 0
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, k in enumerate((0, 1, 8)):
        show(axes[row, 0], d['own'][agent], f'own sensor (agent {agent})')
        show(axes[row, 1], d[f'target_k{k}'][agent], f'arrived GT R{k}')
        show(axes[row, 2], d[f'pred_k{k}'][agent], f'raw prediction R{k}')
    fig.suptitle(f'{run.name} | {name} | agent {agent} | grey unknown / white free / black wall | no repairs', fontsize=9)
    fig.tight_layout(); fig.savefig(outdir / f'{label}-{name}.png', dpi=110); plt.close(fig)
print('rendered to', outdir)

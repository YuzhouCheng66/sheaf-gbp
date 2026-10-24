"""Montage of the v410 base floor rasters (Claude, 2026-10-24): white = free (interior, no obstacle in 0.15-1.6 m), black = obstacle, grey = not interior."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
base = Path(sys.argv[1]); out = sys.argv[2]
rows = json.load(open(base / 'index.json'))['rows']
cols = 5; r = int(np.ceil(len(rows) / cols))
fig, axes = plt.subplots(r, cols, figsize=(11, 2.3 * r))
for ax in axes.ravel():
    ax.set_axis_off()
for ax, row in zip(axes.ravel(), rows):
    z = np.load(base / row['file']); obs, inte = z['obstacle'], z['interior']
    g = np.where(inte & ~obs, 0, np.where(obs, 1, -1))
    ax.imshow(g, cmap=ListedColormap(['#e6e6e6', '#ffffff', '#1a1a1a']), vmin=-1, vmax=1, origin='lower', interpolation='nearest')
    ax.set_title(f"{row['scene'].replace('_', ' ')} f{row['floor']}: {row['free_m2']:.0f} m$^2$ free", fontsize=7)
fig.suptitle('Real-scan floor slices (Replica, 2 cm): white free, black obstacle within 0.15-1.6 m, grey outside the scanned floor', fontsize=8)
fig.tight_layout(); fig.savefig(out, dpi=170)

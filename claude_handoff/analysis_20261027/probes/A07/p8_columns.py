"""p8_columns.py (A07): what must a pose-rendered-visibility (PRV) code still carry? Per own map / R8 union of the 6 VAL demo houses:
cost of the OBSERVED-COLUMN mask (2-D KT context code, 4 causal neighbours) at 32x32 / 64x64 / 128x128, and the share of observed
columns that hold any occupied voxel other than the floor slab (walls / lintels / furniture) -- only those need content bits."""
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a07_lib import load, targets, demo_paths, LG

def ctx2d(m):
    s = torch.from_numpy(m.astype(np.int64)); X, Y = s.shape; c = torch.zeros_like(s)
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        sh = torch.zeros_like(s)
        if dy >= 0: sh[dx:, dy:] = s[:X - dx, :Y - dy]
        else: sh[dx:, :dy] = s[:X - dx, -dy:]
        c = c * 2 + sh
    idx = (c * 2 + s).reshape(-1); cnt = torch.bincount(idx, minlength=32).reshape(-1, 2).double(); N = cnt.sum(1); u = N > 0
    return float(((LG(N[u] + 1.) - LG(1.)) - (LG(cnt[u] + .5) - LG(.5)).sum(1)).sum() / math.log(2)) / 8 + 1

rows = []
for p in demo_paths():
    meta, own, pred, gt = load(p); T8 = targets(own, meta['reached'][meta['hops'].index(8)])
    slab = gt['slab']
    for name, maps in (('own', own), ('union8', T8)):
        for a in range(own.shape[0]):
            m = maps[a]; r = dict(map=name)
            for g in (32, 64, 128):
                b = 128 // g
                obs = (m >= 0).reshape(g, b, g, b, 32).any((1, 3, 4))
                content = ((m == 1) & ~slab).reshape(g, b, g, b, 32).any((1, 3, 4))
                r[f'obs_{g}'] = int(obs.sum()); r[f'content_{g}'] = int((content & obs).sum()); r[f'mask_bytes_{g}'] = ctx2d(obs)
            rows.append(r)
S = {}
for name in ('own', 'union8'):
    rr = [r for r in rows if r['map'] == name]
    S[name] = {k: float(np.median([r[k] for r in rr])) for k in rr[0] if k != 'map'}
    for g in (32, 64, 128):
        S[name][f'content_share_{g}'] = float(np.median([r[f'content_{g}'] / max(r[f'obs_{g}'], 1) for r in rr]))
json.dump(dict(summary=S, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p8_columns.json'), 'w'), indent=1)
print(json.dumps(S, indent=1))

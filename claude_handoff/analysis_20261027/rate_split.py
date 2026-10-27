"""rate_split.py (Claude, 2026-10-27): where do the lossless bits go, and how many code stalks carry any observation?
(1) ctx code of the known mask alone (binary, same 9-voxel causal context) vs the full ternary map -> share of the lossless cost that is
    the visibility boundary (which a decoder could re-derive by ray casting from geometry + poses);
(2) fraction of code columns (32x32x1 -> 4x4x32 voxels; 64x64x1 -> 2x2x32) that contain any known voxel, for own maps and round-8 targets
    -> the share of a fixed-rate message that carries information at all.
Same 6 VAL demo houses as rate_baselines.py."""
import glob, json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = sys.argv[1] if len(sys.argv) > 1 else 'demo_web3d/data'
from rate_baselines_lib import load, targets, ctx_bytes, NB, LG  # noqa: E402


def ctx_bytes_binary(mask):
    s = torch.from_numpy(mask.astype(np.int64)); X, Y, Z = s.shape; c = torch.zeros_like(s)
    for dx, dy, dz in NB:
        sh = torch.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * 2 + sh
    idx = (c * 2 + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=2 ** (len(NB) + 1)).reshape(-1, 2).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    bits = ((LG(N + 1.) - LG(1.)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2


rows = []
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    for name, maps in (('own', own), ('union8', T8)):
        for a in range(n):
            k = maps[a] >= 0
            full = ctx_bytes(maps[a]); km = ctx_bytes_binary(k)
            col32 = k.reshape(32, 4, 32, 4, 32).any((1, 3, 4)).mean(); col64 = k.reshape(64, 2, 64, 2, 32).any((1, 3, 4)).mean()
            rows.append(dict(map=name, house=m['house'], agent=a, full=full, known_mask=km, cols32=float(col32), cols64=float(col64)))
out = {}
for name in ('own', 'union8'):
    r = [x for x in rows if x['map'] == name]
    out[name] = dict(full_ctx_median=float(np.median([x['full'] for x in r])), known_mask_ctx_median=float(np.median([x['known_mask'] for x in r])),
                     known_mask_share_mean=float(np.mean([x['known_mask'] / x['full'] for x in r])),
                     observed_cols_32x32=float(np.mean([x['cols32'] for x in r])), observed_cols_64x64=float(np.mean([x['cols64'] for x in r])))
print(json.dumps(out, indent=1))
json.dump(dict(summary=out, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rate_split.json'), 'w'), indent=1)

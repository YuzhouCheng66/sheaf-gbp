"""A04 probe P3d (Claude, 2026-10-27): how much of a converged message is INNOVATION for its receiver?  Per directed tree edge s -> r,
Y = U_s v U_r (what r holds afterwards), U_r = what r already holds (the sender knows it: r's own message on the edge).
  voxel level: new known voxels |K(Y) \ K(U_r)| / |K(U_s)|, new occupied voxels |O(Y) \ O(U_r)| / |O(U_s)|;
  stalk level (64x64 and 32x32 columns): sender-observed columns whose CONTENT changes for r (Y != U_r somewhere in the column) --
  what a join-homomorphic stalk must resend -- versus columns whose coverage KEY beats r's (what key selection resends, P3).
Also the ideal pose-aware incremental lossless cost per edge (ctx(Y) - ctx(U_r), from p3b) for comparison."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, tree_sides, HERE, retry  # noqa: E402


def one(path):
    m, own, pred, gt = load(path); n = own.shape[0]
    sides = tree_sides(n, [tuple(e) for e in m['edges']])
    rows = []
    for (s, r), side_s in sides.items():
        Us = own[side_s].max(0); Ur = own[sides[(r, s)]].max(0); Y = np.maximum(Us, Ur)
        Ks, Kr, KY = Us >= 0, Ur >= 0, Y >= 0
        row = dict(house=m['house'], s=s, r=r, side_s=len(side_s), new_known=float((KY & ~Kr).sum() / max(Ks.sum(), 1)),
                   new_occ=float(((Y == 1) & (Ur != 1)).sum() / max((Us == 1).sum(), 1)))
        for g in (32, 64):
            b = 128 // g
            obs = Ks.reshape(g, b, g, b, 32).any((1, 3, 4))
            changed = (Y != Ur).reshape(g, b, g, b, 32).any((1, 3, 4)) & obs
            cs = Ks.reshape(g, b, g, b, 32).sum((1, 3, 4)); cr = Kr.reshape(g, b, g, b, 32).sum((1, 3, 4))
            row[f'changed{g}'] = float(changed.sum() / max(obs.sum(), 1)); row[f'beat{g}'] = float(((cs > cr) & obs).sum() / max(obs.sum(), 1))
        rows.append(row)
    return rows


rows = []
for path in houses():
    rows += retry(one, path)
p3b = [x for x in json.load(open(os.path.join(HERE, 'p3b_delta_static.json')))['rows'] if x['kind'] == 'edge']
S = {k: float(np.mean([x[k] for x in rows])) for k in ('new_known', 'new_occ', 'changed32', 'beat32', 'changed64', 'beat64')}
S['incremental_over_full_adaptive_median'] = float(np.median([x['incremental_adaptive'] / x['adaptive'] for x in p3b]))
S['incremental_median_B'] = float(np.median([x['incremental_adaptive'] for x in p3b]))
print(json.dumps(S, indent=1))
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p3d_innovation.json'), 'w'), indent=1)

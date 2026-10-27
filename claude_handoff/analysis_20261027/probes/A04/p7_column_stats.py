"""A04 probe P7 (Claude, 2026-10-27): what is inside an observed 64x64 code column (2 x 2 x 32 voxels at R1)?  Shares of observed columns
holding structure (wall / lintel), furniture, or only slab + free space; and how many separate occupied z-runs (slab voxels removed, the
2 x 2 voxels OR-ed) a furniture column has -- i.e. how many intervals a per-column payload must carry.  K0 own maps and R8 targets."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, targets, HERE, retry  # noqa: E402


def stats(T, gt):
    occ = (T == 1); known = T >= 0
    s = (occ & ~gt['obj'] & ~gt['slab']).reshape(64, 2, 64, 2, 32).any((1, 3, 4))
    f = (occ & gt['obj']).reshape(64, 2, 64, 2, 32).any((1, 3, 4))
    obs = known.reshape(64, 2, 64, 2, 32).any((1, 3, 4))
    prof = (occ & ~gt['slab']).reshape(64, 2, 64, 2, 32).any((1, 3))            # [64, 64, 32] OR over the 2 x 2 voxels
    runs = (prof[:, :, 1:] & ~prof[:, :, :-1]).sum(2) + prof[:, :, 0]
    fr = runs[f & ~s]
    return dict(observed=int(obs.sum()), struct=int((obs & s).sum()), furniture=int((obs & f & ~s).sum()), plain=int((obs & ~s & ~f).sum()),
                runs_hist=[int((fr == r).sum()) for r in range(1, 6)] + [int((fr >= 6).sum())])


def one(path):
    m, own, pred, gt = load(path); n = own.shape[0]
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    return dict(own=[stats(own[a], gt) for a in range(n)], union8=[stats(T8[a], gt) for a in range(n)])


S = {}
allr = {'own': [], 'union8': []}
for path in houses():
    r = retry(one, path)
    for k in allr:
        allr[k] += r[k]
for k, rows in allr.items():
    obs = sum(x['observed'] for x in rows)
    hist = np.sum([x['runs_hist'] for x in rows], 0)
    S[k] = dict(observed_cols_mean=obs / len(rows), struct_share=sum(x['struct'] for x in rows) / obs, furniture_share=sum(x['furniture'] for x in rows) / obs,
                plain_share=sum(x['plain'] for x in rows) / obs, furniture_runs_share_1to5_6plus=(hist / max(hist.sum(), 1)).round(3).tolist())
print(json.dumps(S, indent=1))
json.dump(S, open(os.path.join(HERE, 'p7_column_stats.json'), 'w'), indent=1)

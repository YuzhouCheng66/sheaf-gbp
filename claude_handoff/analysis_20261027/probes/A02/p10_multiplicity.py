"""A02 debate probe P10 (2026-10-27, CPU only, one process): how many agents observe the same voxel?  In the noise-free world every
observation of a voxel is the same deterministic fact, so a SUM-type (information-adding) fusion of per-agent evidence about it is
over-confident by exactly that multiplicity; a join (max) is not.  Per demo house: for every voxel known in the round-8 union of a
connected component, k = number of agents whose own map knows it; reported for all known voxels, occupied voxels and furniture-region
occupied voxels (6 demo VAL houses).  Output: probes/A02/p10_multiplicity.json"""
import glob, json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from rate_baselines_lib import load, dilate6, DIMS  # noqa: E402
ks = {'known': [], 'occ': [], 'furn_occ': []}
for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
    m, own, pred, gt = load(path); del pred
    R8 = np.asarray(m['reached'][m['hops'].index(8)], bool)
    struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(1, -1))
    region = (~gt['slab'].reshape(-1) & ~dilate6(struct, DIMS)[0].numpy()).reshape(DIMS)
    seen = set()
    for a in range(own.shape[0]):
        comp = tuple(np.nonzero(R8[a])[0])
        if comp in seen:
            continue
        seen.add(comp)
        K = (own[list(comp)] >= 0).sum(0); O = (own[list(comp)] == 1).any(0)
        known = K > 0
        ks['known'] += K[known].tolist(); ks['occ'] += K[O].tolist(); ks['furn_occ'] += K[O & region].tolist()
S = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.percentile(v, 90)), frac_k_ge2=float(np.mean(np.asarray(v) >= 2)), n=len(v)) for k, v in ks.items()}
print(json.dumps(S, indent=1))
json.dump(S, open(os.path.join(HERE, 'p10_multiplicity.json'), 'w'), indent=1)

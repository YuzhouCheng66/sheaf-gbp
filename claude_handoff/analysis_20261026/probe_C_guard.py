"""probe_C_guard (analyst C, 2026-10-26): guard readouts of the decision rules from probe_C_hist.py histograms (VAL): pooled recall of
target-occupied slab / wall-zone / furniture-region voxels, and pooled precision of occupied outputs, under the default rule, the
occupancy-only rule (o >= +1.5) and the split rule (occupied: o >= +1.5 and k >= -0.5; free: k >= 0).
python probe_C_guard.py --dir h-s342101 --dir h-s342202
"""
import argparse, glob, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--dir', action='append', required=True); ap.add_argument('--split', default='val')
args = ap.parse_args()
NB, LO, BW = 64, -8., .25
idx = lambda t: int(round((t - LO) / BW))
RULES = {'default': (idx(0.), idx(0.)), 'occ_only_+1.5': (idx(0.), idx(1.5)), 'split_-0.5_+1.5': (idx(-.5), idx(1.5))}
for d in args.dir:
    Hs = None
    for f in sorted(glob.glob(os.path.join(d, f'{args.split}-*.npz'))):
        H = np.load(f)['H'].astype(np.float64).sum(1)                                     # [hop, 12, NB, NB] pooled over agents
        Hs = H if Hs is None else Hs + H
    out = {}
    for hi, k in enumerate((0, 8)):
        for name, (a, b) in RULES.items():
            r = {}
            for lname, l in (('slab', 0), ('wallzone', 1), ('furniture_region', 2)):
                occ = Hs[hi, 2 * 3 + l] + Hs[hi, 3 * 3 + l]
                r[f'recall_occ_{lname}'] = round(occ[a:, b:].sum() / occ.sum(), 4)
            tp = sum(Hs[hi, t * 3 + l][a:, b:].sum() for t in (2, 3) for l in range(3)); fp = sum(Hs[hi, 1 * 3 + l][a:, b:].sum() for l in range(3))
            r['precision_occ'] = round(tp / (tp + fp), 4)
            out[f'K{k}_{name}'] = r
    print(os.path.basename(d.rstrip('/')), json.dumps(out, indent=1))

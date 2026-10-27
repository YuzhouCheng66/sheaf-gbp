"""probe_debate.py (A01, debate round, 2026-10-27). Two checks on the 6 VAL demo houses, one process, light memory.
(a) METRIC LOOPHOLE: the four F1s of metrics3d count only predicted-occupied voxels inside the target-known region, and FPR only
    penalises predicting 'known' where the target is unknown. So pred = (occupied on the target's occupied set S, unknown everywhere
    else) should score surface = var = furniture F1 = 1 and FPR = 0 while delivering no free space at all. We measure it, together
    with free-space recall (predicted free & target free / target free) and 3-way accuracy, for: S-only, 3S-X calibrated output,
    and the exact ray-cast reconstruction (== target).
(b) OBSERVATION MULTIPLICITY: in each round-8 union, how many reached agents observed each known voxel / each evidenced 32x32 and
    64x64 column. Under information-form (sum) fusion of deterministic, overlapping evidence, the fused precision of a cell is
    over-counted by exactly this multiplicity (the same observation counted k times).
Usage: python probe_debate.py DATA_DIR OUT_JSON"""
import glob, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from rate_baselines_lib import load, targets, metrics  # noqa: E402

torch.set_num_threads(2)
DATA, OUT = sys.argv[1], sys.argv[2]
rows, mult = [], []
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]
    slab = torch.from_numpy(gt['slab'].reshape(-1)); obj = torch.from_numpy(gt['obj'].reshape(-1))
    struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(-1))
    for hop in (0, 8):
        R = np.asarray(m['reached'][m['hops'].index(hop)], bool)
        T = targets(own, R)
        for a in range(n):
            t = T[a]
            variants = dict(S_only=np.where(t == 1, np.int8(1), np.int8(-1)), model=pred[hop][a], exact=t)
            for name, pv in variants.items():
                r = metrics(torch.from_numpy(pv.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), slab, obj, struct)[0]
                known = t >= 0; free = t == 0; unk = ~known
                r.update(house=m['house'], hop=hop, agent=a, variant=name,
                         fpr=float((pv[unk] >= 0).mean()) if unk.any() else None,
                         free_recall=float(((pv == 0) & free).sum() / max(free.sum(), 1)),
                         acc3=float((pv[known] == t[known]).mean()))
                rows.append(r)
        if hop == 8:
            for a in range(n):
                srcs = np.nonzero(R[a])[0]
                K = (own[srcs] >= 0)                                      # [k, X, Y, Z]
                kv = K.sum(0); known = kv > 0
                col = {}
                for G in (32, 64):
                    s = 128 // G
                    ev = K.reshape(len(srcs), G, s, G, s, 32).any((2, 4, 5)).sum(0)
                    col[G] = ev[ev > 0]
                mult.append(dict(house=m['house'], agent=a, reached=int(len(srcs)),
                                 vox_mean=float(kv[known].mean()), vox_ge2=float((kv[known] >= 2).mean()),
                                 col32_mean=float(col[32].mean()), col32_ge2=float((col[32] >= 2).mean()),
                                 col64_mean=float(col[64].mean()), col64_ge2=float((col[64] >= 2).mean())))
    print(m['house'], 'done', flush=True)
S = {}
for hop in (0, 8):
    for name in ('S_only', 'model', 'exact'):
        R_ = [r for r in rows if r['hop'] == hop and r['variant'] == name]
        S[f'K{hop}_{name}'] = {k: float(np.mean([r[k] for r in R_ if r[k] is not None])) for k in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr', 'free_recall', 'acc3')}
S['multiplicity_R8'] = {k: float(np.mean([x[k] for x in mult])) for k in ('reached', 'vox_mean', 'vox_ge2', 'col32_mean', 'col32_ge2', 'col64_mean', 'col64_ge2')}
json.dump(dict(summary=S, rows=rows, multiplicity=mult), open(OUT, 'w'), indent=1)
print(json.dumps(S, indent=1))

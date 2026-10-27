"""A09 probe 1 (Claude, 2026-10-27, CPU only): what does the SCORED content of a map cost, and what is a lossy-S rate / F1 curve?

Facts used (read off train_3d_v431.metrics3d and mapping_3d_storey_data_v420.observe):
  * every pre-registered F1 (occ, surface, var, furniture) counts only predicted-OCCUPIED voxels inside the TARGET-KNOWN region; free-space
    knowledge is unscored except through the FPR guard (over-claiming known). The scored content of a map is S = {observed occupied voxels}.
  * own_map == observe(S, the agent's 3 poses) exactly: a ray stops at its first hit, which is in S, and no voxel of S (all GT-occupied) can lie
    before it on the same ray; so the full ternary map = (S, poses) and the known mask (77 % of the ctx bits) is a function of S + ~44 bit.
Measured on the 6 VAL demo houses x 8 agents:
  (1) ctx code (9-voxel causal context, adaptive KT, + 2 B) of S alone vs the full ternary map, for own maps (K0), R8 unions, and the
      sender-side unions of directed tree edges (what one converged message stands for);
  (2) lossy S: block down-sampling of S with OR (or count >= 2), ctx-coded at the coarse grid; prediction = up-sampled S' as OCCUPIED, all
      else unknown; scored with the same metrics (surface / var / furniture F1) -- an F1 upper bound for 'S-only' codes at that rate, and a
      lower bound for what a learned decoder can do with the same bits.
Output: probes/A09/p1_surface_rate.json
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))                       # claude_handoff/analysis_20261027
sys.path.insert(0, HERE)
from a09lib import load, targets, ctx_bytes, ctx_bytes_binary, metrics, check_export, DIMS   # noqa: E402  (numpy-only copies)

DATA = os.path.abspath(os.path.join(ROOT, '..', '..', 'demo_web3d', 'data'))


def down_or(S, f, m=1):
    X, Y, Z = S.shape; bx, by, bz = f
    cnt = S.reshape(X // bx, bx, Y // by, by, Z // bz, bz).sum((1, 3, 5))
    coarse = cnt >= m
    return coarse, coarse.repeat(bx, 0).repeat(by, 1).repeat(bz, 2)


FACTORS = [(1, 1, 1), (2, 2, 1), (1, 1, 2), (2, 2, 2), (4, 4, 1), (4, 4, 2), (2, 2, 4)]
rows = dict(own=[], union8=[], edge=[]); lossy = {}
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]
    slab, obj = gt['slab'], gt['obj']; struct = gt['occ'] & ~gt['obj'] & ~gt['slab']
    worst = check_export(m, own, pred, gt)
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    for name, maps in (('own', own), ('union8', T8)):
        for a in range(n):
            S = maps[a] == 1
            rows[name].append(dict(house=m['house'], agent=a, S_voxels=int(S.sum()), known_voxels=int((maps[a] >= 0).sum()),
                                   ctx_ternary=ctx_bytes(maps[a]), ctx_S=ctx_bytes_binary(S), ctx_known=ctx_bytes_binary(maps[a] >= 0),
                                   furn_S_voxels=int((S & gt['obj']).sum())))
    adj = {i: set() for i in range(n)}
    for i, j in m['edges']:
        adj[i].add(j); adj[j].add(i)
    for i, j in m['edges']:
        for s, r in ((i, j), (j, i)):
            side, stack = {s}, [s]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v != r and v not in side:
                        side.add(v); stack.append(v)
            U = own[sorted(side)].max(0)
            rows['edge'].append(dict(house=m['house'], sender=s, side=len(side), ctx_ternary=ctx_bytes(U), ctx_S=ctx_bytes_binary(U == 1)))
    for k, Tk in ((0, own), (8, T8)):
        tgt = Tk
        for f in FACTORS:
            for mm in ((1,) if f == (1, 1, 1) else (1, 2)):
                key = f'K{k}_{f[0]}x{f[1]}x{f[2]}_m{mm}'
                preds, byts = [], []
                for a in range(n):
                    coarse, up = down_or(Tk[a] == 1, f, mm)
                    byts.append(ctx_bytes_binary(coarse))
                    preds.append(np.where(up, np.int8(1), np.int8(-1)))
                r = metrics(np.stack(preds), tgt, slab, obj, struct)
                L = lossy.setdefault(key, dict(bytes=[], surface_f1=[], var_occ_f1=[], furn_f1=[]))
                L['bytes'] += byts
                for row in r:
                    for q in ('surface_f1', 'var_occ_f1', 'furn_f1'):
                        if row[q] is not None:
                            L[q].append(row[q])
    print(m['house'], 'done; metric copy max |diff| vs export', round(worst, 6), flush=True)


def med(v):
    return float(np.median(v))


summary = {}
for name in ('own', 'union8', 'edge'):
    r = rows[name]
    summary[name] = dict(n=len(r), ctx_ternary_median=med([x['ctx_ternary'] for x in r]), ctx_S_median=med([x['ctx_S'] for x in r]),
                         S_share_of_ternary_median=med([x['ctx_S'] / x['ctx_ternary'] for x in r]),
                         ctx_S_p10=float(np.percentile([x['ctx_S'] for x in r], 10)), ctx_S_p90=float(np.percentile([x['ctx_S'] for x in r], 90)))
    if name != 'edge':
        summary[name].update(S_voxels_median=med([x['S_voxels'] for x in r]), bits_per_S_voxel_median=med([8 * x['ctx_S'] / max(x['S_voxels'], 1) for x in r]),
                             ctx_known_median=med([x['ctx_known'] for x in r]), furn_share_of_S_median=med([x['furn_S_voxels'] / max(x['S_voxels'], 1) for x in r]))
summary['lossy_S'] = {k: dict(bytes_median=med(v['bytes']), surface_f1=float(np.mean(v['surface_f1'])), var_occ_f1=float(np.mean(v['var_occ_f1'])),
                              furn_f1=float(np.mean(v['furn_f1']))) for k, v in lossy.items()}
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, rows=rows), open(os.path.join(HERE, 'p1_surface_rate.json'), 'w'), indent=1)

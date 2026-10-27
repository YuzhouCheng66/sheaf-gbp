"""A04 probe P1 (Claude, 2026-10-27): is the known mask a function of (occupied first-hit set, sensor poses)?  And what does a
POSE-AWARE lossless coder cost (occupied set coded alone + 44 bit of poses per agent, known mask re-derived by ray casting)?
(a) determinism self-test: simulate one pose on the GT world with the generator's own marcher, then re-cast the same pose through the
    simulated OCCUPIED set only -> must reproduce the ternary map exactly (first-hit model: every surface that stops a ray is in the set).
(b) the real own maps were taken from 3 poses jittered +-0.1 m that the demo export does not store; casting the stored CENTRE through the
    agent's own occupied set gives a lower bound on how much of the known mask is re-derivable without any of the 44 pose bits.
(c) cost: ctx of the ternary map vs ctx of the binary occupied set (+ 5.5 B of poses per agent)."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, targets, ctx_bytes, ctx_bytes_binary, observe, fibonacci_sphere, HERE, retry  # noqa: E402
PARTS = os.path.join(HERE, 'parts'); os.makedirs(PARTS, exist_ok=True)

rays = fibonacci_sphere(65536)
rows, selftest = [], []
t0 = time.time()
def one_house(path):
    rows, selftest = [], []
    m, own, pred, gt = load(path); n = own.shape[0]
    cell = np.asarray(m['cell_m'], np.float64); pos = np.asarray(m['positions_m'], np.float64)
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    # (a) determinism self-test on the GT world, agent 0 centre
    sim = observe(gt['occ'], cell, pos[:1], rays)
    rd = observe(sim == 1, cell, pos[:1], rays)
    selftest.append(dict(house=m['house'], voxels_known=int((sim >= 0).sum()), mismatched=int((rd != sim).sum())))
    for a in range(n):
        o = own[a]; kn = o >= 0
        rc = observe(o == 1, cell, pos[a:a + 1], rays); kr = rc >= 0
        inter = int((kn & kr).sum()); uni = int((kn | kr).sum())
        rows.append(dict(kind='own', house=m['house'], agent=a, ctx3=ctx_bytes(o), ctx_occ=ctx_bytes_binary(o == 1), known=int(kn.sum()),
                         recast_known=int(kr.sum()), known_iou=inter / max(uni, 1), known_recall=inter / max(int(kn.sum()), 1),
                         occ_agree_on_known=float(((rc == 1) == (o == 1))[kn].mean())))
    cache = {}
    for a in range(n):
        R = np.asarray(m['reached'][m['hops'].index(8)])[a].astype(bool); U = T8[a]; kn = U >= 0
        key = tuple(np.nonzero(R)[0])
        if key not in cache:
            kr = np.zeros_like(kn)
            for j in key:
                kr |= observe(U == 1, cell, pos[j:j + 1], rays) >= 0
            cache[key] = kr
        kr = cache[key]
        inter = int((kn & kr).sum()); uni = int((kn | kr).sum())
        rows.append(dict(kind='union8', house=m['house'], agent=a, reached=int(R.sum()), ctx3=ctx_bytes(U), ctx_occ=ctx_bytes_binary(U == 1),
                         known=int(kn.sum()), recast_known=int(kr.sum()), known_iou=inter / max(uni, 1), known_recall=inter / max(int(kn.sum()), 1)))
    print(m['house'], f'{time.time() - t0:.0f}s', selftest[-1], flush=True)
    return rows, selftest


for path in houses():
    part = os.path.join(PARTS, 'p1_' + os.path.basename(path))
    if not os.path.exists(part):
        r_, s_ = retry(one_house, path); json.dump(dict(rows=r_, selftest=s_), open(part, 'w'))
    d = json.load(open(part)); rows += d['rows']; selftest += d['selftest']

S = dict(selftest=selftest)
for k in ('own', 'union8'):
    r = [x for x in rows if x['kind'] == k]
    pose_b = [5.5 * (x.get('reached', 1)) for x in r]
    S[k] = dict(n=len(r), ctx3_median=float(np.median([x['ctx3'] for x in r])), ctx_occ_median=float(np.median([x['ctx_occ'] for x in r])),
                pose_aware_median=float(np.median([x['ctx_occ'] + b for x, b in zip(r, pose_b)])),
                ratio_ctx3_over_pose_aware_median=float(np.median([x['ctx3'] / (x['ctx_occ'] + b) for x, b in zip(r, pose_b)])),
                known_iou_centre_only_mean=float(np.mean([x['known_iou'] for x in r])), known_recall_centre_only_mean=float(np.mean([x['known_recall'] for x in r])))
print(json.dumps(S, indent=1))
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p1_visibility.json'), 'w'), indent=1)

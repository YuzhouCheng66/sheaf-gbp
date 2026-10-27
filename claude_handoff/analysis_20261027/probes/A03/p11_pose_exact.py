"""p11_pose_exact.py (A03, debate round): how many pose bits make the render decoder BIT-EXACT? (6 demo houses x 8 agents, 1 process)
Centres are the generator's R1 cell centres (14 bit); the 3 jitters (x, y in +-0.1 m) are sent as uniform integers on [-0.1, 0.1] with
12 / 16 bit per coordinate, or as float32. Re-render own = observe(own hits, poses) and count voxels that differ from the true own map.
Pose bits per agent = 14 + 6 x b (12 bit: 86, 16 bit: 110 = 13.75 B, float32: 14 + 192 = 206 = 25.75 B)."""
import json, os, time
import numpy as np
from a03np import observe_np, cell_of, load_fx, HERE

rows = []
for name in ('r1_i0', 'r1_i1', 'r2_i0', 'r2_i1', 'r3_i0', 'r3_i1'):
    t0 = time.time()
    fx, meta = load_fx(name); own = fx['private']; cell = cell_of(meta['chart_m']); traj = fx['trajectory']
    for a in range(own.shape[0]):
        P = traj[:, a]; c = fx['positions'][a, :2]
        for tag in ('u12', 'u16', 'f32'):
            Q = P.copy()
            if tag == 'f32':
                Q[:, :2] = c + (P[:, :2] - c).astype(np.float32).astype(np.float64)
            else:
                b = int(tag[1:]); n = 2 ** b - 1
                k = np.round((P[:, :2] - c + 0.1) / 0.2 * n); Q[:, :2] = c + k / n * 0.2 - 0.1
            r = observe_np(own[a] == 1, cell, Q)
            rows.append(dict(house=name, agent=a, q=tag, diff=int((r != own[a]).sum()), occ_diff=int(((r == 1) != (own[a] == 1)).sum())))
    print(name, f'{time.time()-t0:.0f}s', flush=True)
S = {}
for tag, bits in (('u12', 14 + 72), ('u16', 14 + 96), ('f32', 14 + 192)):
    rr = [r for r in rows if r['q'] == tag]
    S[tag] = dict(bits_per_agent=bits, exact=sum(r['diff'] == 0 for r in rr), n=len(rr), diff_median=float(np.median([r['diff'] for r in rr])),
                  diff_max=int(max(r['diff'] for r in rr)), occ_diff_max=int(max(r['occ_diff'] for r in rr)))
    print(tag, S[tag])
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p11_pose_exact.json'), 'w'), indent=1)

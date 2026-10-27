"""A04 probe P3 (Claude, 2026-10-27): how much does a DELTA-STATE message cost when it is coded against the lattice state the receiver
already holds and the sender KNOWS it holds (the receiver's own message on the same edge)?  Idempotent join => omitting / re-coding
what the receiver has is exact.  Lossless voxel level (ctx coder), per directed tree edge s -> r:
  full  = ctx(U_s)                       U_s = union of the sender side (what one converged message stands for; evidence pack 1,752 B)
  delta = ctx(U_s v U_r | U_r)            the receiver only needs the join; context = 9 causal voxels of the join + U_r at the voxel
and a two-sweep schedule on each tree (leaves -> root unconditional, root -> leaves conditional on the child's subtree) vs one full
message per directed edge.  Code level: at 64x64 columns, the fraction of the sender's observed columns whose coverage key beats the
receiver's (the only stalks a key-delta protocol must send)."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, ctx_bytes, ctx_bits_cond, tree_sides, HERE  # noqa: E402

rows, sweeps = [], []
t0 = time.time()
for path in houses():
    m, own, pred, gt = load(path); n = own.shape[0]
    edges = [tuple(e) for e in m['edges']]
    sides = tree_sides(n, edges)
    cov64 = (own >= 0).reshape(n, 64, 2, 64, 2, 32).sum((2, 4, 5))            # [n, 64, 64] known voxels per 2 x 2 x 32 column
    per = {}
    for (s, r), side_s in sides.items():
        side_r = sides[(r, s)]
        Us = own[side_s].max(0); Ur = own[side_r].max(0)
        full = ctx_bytes(Us)
        Y = np.maximum(Us, Ur)
        delta = ctx_bits_cond(Y + 1, 3, side=[(Ur + 1, 3)]) / 8 + 2
        ks = cov64[side_s].max(0); kr = cov64[side_r].max(0)
        obs = ks > 0
        per[(s, r)] = (full, delta)
        rows.append(dict(house=m['house'], s=s, r=r, side_s=len(side_s), side_r=len(side_r), full=full, delta=delta,
                         cols_sender=int(obs.sum()), cols_beat=int((ks > kr).sum()), cols_new=int((obs & (kr == 0)).sum())))
    # two-sweep schedule: root = smallest index of each component
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    seen, up, down = set(), 0., 0.
    for root in range(n):
        if root in seen:
            continue
        stack = [(root, None)]
        while stack:
            u, par = stack.pop(); seen.add(u)
            for v in adj[u]:
                if v != par:
                    up += per[(v, u)][0]; down += per[(u, v)][1]; stack.append((v, u))
    sweeps.append(dict(house=m['house'], edges=len(edges), one_full_per_directed_edge=float(sum(f for f, _ in per.values())),
                       one_delta_per_directed_edge=float(sum(d for _, d in per.values())), two_sweep=up + down))
    print(m['house'], f'{time.time() - t0:.0f}s', sweeps[-1], flush=True)

S = dict(edge_full_median=float(np.median([x['full'] for x in rows])), edge_delta_median=float(np.median([x['delta'] for x in rows])),
         edge_delta_over_full_median=float(np.median([x['delta'] / x['full'] for x in rows])),
         cols_beat_fraction_mean=float(np.mean([x['cols_beat'] / max(x['cols_sender'], 1) for x in rows])),
         cols_new_fraction_mean=float(np.mean([x['cols_new'] / max(x['cols_sender'], 1) for x in rows])),
         house_two_sweep_over_full_mean=float(np.mean([x['two_sweep'] / x['one_full_per_directed_edge'] for x in sweeps])),
         sweeps=sweeps)
print(json.dumps({k: v for k, v in S.items() if k != 'sweeps'}, indent=1))
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p3_delta_edges.json'), 'w'), indent=1)

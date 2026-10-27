"""A06 program codec on the DEMO maps (Claude A06, 2026-10-27). Same greedy box cover as a06_boxes.py (m = 8), applied to the demo's
own maps (K0 content), round-8 union maps, and every directed tree edge's sender-side union (= what one converged message stands for,
evidence pack 3.1). Bytes only (the cover covers O and avoids F by construction; exactness of the ray-cast decode is verified on
re-simulated maps by a06_raycast.py / a06_boxes.py). Box fields entropy-coded with leave-one-house-out static histograms over all
programs of the other houses; residual by the 9-neighbour ctx coder with the rendered boxes as context; poses 44 bit per agent."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a06_common import houses, RB  # noqa: E402
from a06_boxes import cover, render, fields, box_bits, FIELDS  # noqa: E402
from a06_raycast import ctx_bin, POSE_BYTES  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(2)
R = []
for path in houses():
    m, own, pred, gt = RB.load(path); n = own.shape[0]
    reached = np.asarray(m['reached'][m['hops'].index(8)], bool); T8 = RB.targets(own, reached)
    adj = {i: set() for i in range(n)}
    for i, j in m['edges']:
        adj[i].add(j); adj[j].add(i)
    def side_of(s, r):
        side, stack = {s}, [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v != r and v not in side:
                    side.add(v); stack.append(v)
        return sorted(side)
    items = [('own', a, own[a], 1) for a in range(n)] + [('union', a, T8[a], int(reached[a].sum())) for a in range(n)]
    for i, j in m['edges']:
        for s, r in ((i, j), (j, i)):
            sd = side_of(s, r); items.append(('edge', s, own[sd].max(0), len(sd)))
    rows = []
    for kind, a, t, k in items:
        boxes, left = cover(t, 8); wb = render(boxes, t.shape)
        rows.append(dict(kind=kind, agent=int(a), agents=k, n_boxes=len(boxes), fields=fields(boxes), resid_B=ctx_bin(left, side=wb), n_resid=int(left.sum()),
                         poses_B=POSE_BYTES * k, ctx_ternary=RB.ctx_bytes(t), ctx_O=ctx_bin(t == 1), gzip=RB.gz(t)))
    R.append(dict(house=m['house'], rows=rows)); print(m['house'], len(rows), flush=True)
for h in R:
    hist = {name: np.zeros(nn) for name, nn in FIELDS}
    for g in R:
        if g['house'] != h['house']:
            for r in g['rows']:
                for row in r['fields']:
                    for (name, nn), v in zip(FIELDS, row):
                        hist[name][v] += 1
    for r in h['rows']:
        r['box_B'] = box_bits(r['fields'], hist) / 8; r['total_B'] = r['box_B'] + r['resid_B'] + r['poses_B']
json.dump(R, open(os.path.join(OUT, 'boxes_edges.json'), 'w'))
for kind in ('own', 'union', 'edge'):
    rr = [r for h in R for r in h['rows'] if r['kind'] == kind]
    med = lambda k: float(np.median([r[k] for r in rr]))
    print(f"[{kind}] n {len(rr)}: total med {med('total_B'):.0f} B (boxes {med('box_B'):.0f}, resid {med('resid_B'):.0f}, poses {med('poses_B'):.0f}; mean {np.mean([r['total_B'] for r in rr]):.0f}, p90 {np.percentile([r['total_B'] for r in rr], 90):.0f}); "
          f"boxes {med('n_boxes'):.0f}; ctx ternary {med('ctx_ternary'):.0f}; ctx O {med('ctx_O'):.0f}; gzip {med('gzip'):.0f}; "
          f"ratios (median of per-map) ctx/prog {np.median([r['ctx_ternary'] / r['total_B'] for r in rr]):.1f}, gzip/prog {np.median([r['gzip'] / r['total_B'] for r in rr]):.1f}, raw/prog {np.median([131072 / r['total_B'] for r in rr]):.0f}")

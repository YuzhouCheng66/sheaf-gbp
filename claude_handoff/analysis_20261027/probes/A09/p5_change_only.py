"""A09 probe 5 (Claude, 2026-10-27, CPU only): how many stalks does an edge carry over the whole 8-round protocol if it sends only CHANGED
coordinates? rollout_max (train_3d_v431) sends a full D-vector on every directed edge in every round (8 x 2,048 B = 16 KB per edge for 3S-X).
Simulation on the 6 demo houses (their radio forests): the precision key is a proxy, the agent's known-voxel coverage of each 64x64 column
(2 x 2 x 32 voxels) quantised to 8 bit (0 = no evidence; ties broken by agent index, as select() does); messages follow rollout_max exactly
(a message on i->j at round r is the max-key over i's own stalk and the round r-1 messages into i from every neighbour except j).
Counted per directed edge: stalks that change (selected source differs from the previous round's, incl. first appearance) summed over the
8 rounds, versus the non-empty stalks of the converged message; and the round at which the edge's message stops changing.
Output: probes/A09/p5_change_only.json"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a09lib import load   # noqa: E402

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
rows = []
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]
    cov = (own >= 0).reshape(n, 64, 2, 64, 2, 32).mean((2, 4, 5)).reshape(n, -1)          # [n, 4096] known fraction per column
    key = np.where(cov > 0, np.maximum(1, np.round(cov * 255)), 0).astype(np.int64)
    score = key * 16 + (15 - np.arange(n))[:, None]                                       # lexicographic (key, -index): select() breaks ties to the lower index
    score = np.where(key > 0, score, -1)
    directed = [(i, j) for i, j in m['edges']] + [(j, i) for i, j in m['edges']]
    msg = {e: np.full(4096, -1, np.int64) for e in directed}
    changes = {e: 0 for e in directed}; last_change = {e: 0 for e in directed}
    for r in range(1, 9):
        new = {}
        for (i, j) in directed:
            cand = [score[i]] + [msg[(k, i2)] for (k, i2) in directed if i2 == i and k != j]
            new[(i, j)] = np.max(np.stack(cand), 0)
        for e in directed:
            ch = int((new[e] != msg[e]).sum())
            if ch:
                changes[e] += ch; last_change[e] = r
        msg = new
    for e in directed:
        rows.append(dict(house=m['house'], edge=list(e), final_nonempty=int((msg[e] >= 0).sum()), total_changes=changes[e], converged_round=last_change[e]))
fn = np.array([r['final_nonempty'] for r in rows], float); tc = np.array([r['total_changes'] for r in rows], float)
summary = dict(edges=len(rows), final_nonempty_median=float(np.median(fn)), total_changes_median=float(np.median(tc)),
               ratio_changes_over_final_median=float(np.median(tc / np.maximum(fn, 1))), ratio_p90=float(np.percentile(tc / np.maximum(fn, 1), 90)),
               full_vector_sends_over_final_median=float(np.median(8 * 4096 / np.maximum(fn, 1))),
               converged_round_median=float(np.median([r['converged_round'] for r in rows])), converged_round_max=int(max(r['converged_round'] for r in rows)))
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, rows=rows), open(os.path.join(HERE, 'p5_change_only.json'), 'w'), indent=1)

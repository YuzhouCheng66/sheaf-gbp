"""p5_network.py (A07): honest NETWORK accounting of one round-8 fusion (all bytes on all links), 6 VAL demo houses, the demo's forest.
Schemes:
  ours_fixed      : what the trainer simulates -- every directed tree edge carries a full D-coordinate message in each of 8 rounds
                    (2,048 B at 32x32x1 @ 8+8; 8,192 B at 64x64x1);
  ours_stop       : same fixed-length message, but a link stops once its message can no longer change (round h+1, h = depth of the
                    sender's side seen from the sender);
  ours_event      : sparse, event-triggered: a round sends only coordinates whose selected source changed (first round: every evidenced
                    coordinate), 8 bit per coordinate (4+4 matched codebook, analyst B: -0.6 pp) + the index set at its combinatorial
                    entropy log2 C(D, k). Selection key = column coverage (proxy for the learned precision; the code itself is not in the demo).
  flood_*         : every own map is relayed once over every edge of its component (ctx / sensor-aware H + poses);
  union_edge_*    : in-network join with lossless coding: each directed edge carries its sender side's union once;
  twopass_*       : convergecast to the tree centre (child -> parent: subtree union), then broadcast (parent -> child: everything else,
                    coded CONDITIONALLY on the child's subtree union, which the parent already holds) -- a realisable lossless protocol.
Also the per-agent maximum transmit and the number of rounds each scheme needs."""
import json, math, os, sys, collections
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a07_lib import load, ctx_ter, ctx_bin, ctx_bin_cond, ctx_ter_cond, POSE_BYTES, demo_paths

HERE = os.path.dirname(os.path.abspath(__file__))


def lg2binom(n, k):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def coverage(own, layout):
    bx = own.shape[1] // layout; return (own >= 0).reshape(own.shape[0], layout, bx, layout, bx, own.shape[3]).mean((2, 4, 5)).reshape(own.shape[0], -1)


def house(path):
    meta, own, pred, gt = load(path); n = own.shape[0]
    edges = [tuple(e) for e in meta['edges']]
    nb = collections.defaultdict(set)
    for i, j in edges:
        nb[i].add(j); nb[j].add(i)
    def side_of(s, r):
        side, st = {s: 0}, [s]
        while st:
            u = st.pop()
            for v in nb[u]:
                if v != r and v not in side:
                    side[v] = side[u] + 1; st.append(v)
        return side                                                       # agent -> distance from s within the sender side
    directed = [(i, j) for i, j in edges] + [(j, i) for i, j in edges]
    out = dict(house=meta['house'], n_edges=len(edges), directed=len(directed))
    # ---- ours
    for layout, B in ((32, 2048), (64, 8192)):
        D = layout * layout; cov = coverage(own, layout)
        fixed = 8 * len(directed) * B
        stop = 0; ev_bits = 0.; per_agent = collections.Counter(); per_agent_fixed = collections.Counter()
        for s, r in directed:
            side = side_of(s, r); h = max(side.values()); stop += min(h + 1, 8) * B
            per_agent_fixed[s] += 8 * B
            prev = np.full(D, -1)
            for t in range(1, min(h + 1, 8) + 1):
                members = [a for a, d in side.items() if d <= t - 1]
                C = cov[members]; best = np.where(C.max(0) > 0, np.array(members)[np.argmax(C, 0)], -1)
                changed = int((best != prev).sum()); prev = best
                if changed:
                    bits = changed * 8 + lg2binom(D, changed); ev_bits += bits; per_agent[s] += bits / 8
        out[f'ours_fixed_{layout}'] = fixed; out[f'ours_stop_{layout}'] = stop; out[f'ours_event_{layout}'] = ev_bits / 8
        out[f'ours_fixed_{layout}_max_agent'] = max(per_agent_fixed.values()) if per_agent_fixed else 0
        out[f'ours_event_{layout}_max_agent'] = max(per_agent.values()) if per_agent else 0
        out[f'evidence_coords_{layout}_own_median'] = float(np.median((cov > 0).sum(1)))
    # ---- lossless
    sz_ctx = [ctx_ter(own[a]) for a in range(n)]; sz_H = [ctx_bin(own[a] == 1) + POSE_BYTES for a in range(n)]
    comp = {}
    for a in range(n):
        comp[a] = side_of(a, -1)
    out['flood_ctx'] = sum(sz_ctx[a] * (len(comp[a]) - 1) for a in range(n)); out['flood_H'] = sum(sz_H[a] * (len(comp[a]) - 1) for a in range(n))
    ue_ctx = ue_H = 0.
    for s, r in directed:
        S = sorted(side_of(s, r)); U = own[S].max(0)
        ue_ctx += ctx_ter(U); ue_H += ctx_bin(U == 1) + POSE_BYTES * len(S)
    out['union_edge_ctx'] = ue_ctx; out['union_edge_H'] = ue_H
    # two-pass, rooted at each component's centre
    seen = set(); tp_ctx = tp_H = 0.; rounds = 0
    for a in range(n):
        if a in seen:
            continue
        members = sorted(comp[a]); seen |= set(members)
        if len(members) == 1:
            continue
        ecc = {m: max(side_of(m, -1).values()) for m in members}; root = min(members, key=lambda m: (ecc[m], m)); rounds = max(rounds, 2 * ecc[root])
        parent = {root: None}; order = [root]
        for u in order:
            for v in nb[u]:
                if v not in parent:
                    parent[v] = u; order.append(v)
        for c in order[1:]:
            sub = sorted(side_of(c, parent[c])); rest = sorted(set(members) - set(sub))
            Us = own[sub].max(0); Ur = own[rest].max(0)
            tp_ctx += ctx_ter(Us) + ctx_ter_cond(Ur, Us)
            Hs = Us == 1; Hn = (Ur == 1) & ~Hs
            tp_H += ctx_bin(Hs) + POSE_BYTES * len(sub) + ctx_bin_cond(Hn, Hs.astype(np.int8)) + POSE_BYTES * len(rest)
    out['twopass_ctx'] = tp_ctx; out['twopass_H'] = tp_H; out['twopass_rounds'] = rounds
    out['diameter'] = max(max(side_of(a, -1).values()) for a in range(n))
    print(meta['house'], {k: (round(v) if isinstance(v, float) else v) for k, v in out.items() if k != 'house'}, flush=True)
    return out


if __name__ == '__main__':
    with Pool(2) as pool:
        res = pool.map(house, demo_paths())
    keys = [k for k in res[0] if k != 'house']
    S = {k: dict(sum=float(np.sum([r[k] for r in res])), median=float(np.median([r[k] for r in res]))) for k in keys}
    json.dump(dict(summary=S, houses=res), open(os.path.join(HERE, 'p5_network.json'), 'w'), indent=1)
    for k in keys:
        print(f'{k:32s} median {S[k]["median"]:12.0f}  sum6 {S[k]["sum"]:12.0f}')

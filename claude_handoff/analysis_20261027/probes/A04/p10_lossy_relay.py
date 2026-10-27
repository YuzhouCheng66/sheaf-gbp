"""A04 probe P10 (Claude, 2026-10-27, debate round): lossy re-encoding at RELAYS versus lossy encoding once at the SOURCE.
Lossy operator pi_m (a stand-in for 'drop boxes / fragments covering < m hits', the lossy knob of several structured codecs): every
occupied non-slab voxel in a 26-connected occupied component of size < m is set to unknown (deflationary: occ -> bottom). pi_m is NOT a
join-homomorphism: fragments seen by different agents can merge into a component >= m in the union.
  A (source-only): each agent sends pi_m(own) once; relays forward the join of what they received, losslessly.
  B (relay re-fit): every message is pi_m(own_u v joined incoming except from v) -- the relay re-encodes its fused belief lossily.
  C (centralised reference): pi_m(union of the component).
Each node keeps its own map exactly.  Reports: F1 (surface / furniture) of node beliefs against the R8 target, the mismatch with C
outside the node's own known voxels, and the pairwise disagreement between nodes outside both own known regions (0 = a global section)."""
import json, os, sys
import numpy as np, torch
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, targets, metrics, gt_masks, HERE, retry  # noqa: E402

ST = np.ones((3, 3, 3), bool)


def pi(T, slab, m):
    occ = (T == 1) & ~slab
    lab, k = ndimage.label(occ, structure=ST)
    if k == 0:
        return T
    size = np.bincount(lab.ravel(), minlength=k + 1)
    small = (size < m); small[0] = False
    out = T.copy(); out[small[lab]] = -1
    return out


def join(*maps):
    return np.maximum.reduce(maps)


def one(path, m_thr):
    m, own, pred, gt = load(path); n = own.shape[0]
    slabm, obj, struct = gt_masks(gt); slab = gt['slab']
    R8 = np.asarray(m['reached'][m['hops'].index(8)]).astype(bool)
    T8 = targets(own, R8)
    edges = [tuple(e) for e in m['edges']]
    nb = {i: [] for i in range(n)}
    for i, j in edges:
        nb[i].append(j); nb[j].append(i)
    memo = {}

    def msg(u, v):                                                       # protocol B, tree recursion
        if (u, v) not in memo:
            memo[(u, v)] = pi(join(own[u], *[msg(w, u) for w in nb[u] if w != v]), slab, m_thr)
        return memo[(u, v)]
    piown = [pi(own[a], slab, m_thr) for a in range(n)]
    bA, bB, C = [], [], {}
    for v in range(n):
        src = np.nonzero(R8[v])[0]
        bA.append(join(own[v], *[piown[a] for a in src if a != v]))
        bB.append(join(own[v], *[msg(u, v) for u in nb[v]]) if nb[v] else own[v].copy())
        key = tuple(src)
        if key not in C:
            C[key] = pi(join(*[own[a] for a in src]), slab, m_thr)
    res = {}
    for name, B in (('A_source_only', bA), ('B_relay_refit', bB)):
        Bn = np.stack(B)
        mt = metrics(torch.from_numpy(Bn.reshape(n, -1)), torch.from_numpy(T8.reshape(n, -1)), slabm, obj, struct)
        mis = [int(((Bn[v] != C[tuple(np.nonzero(R8[v])[0])]) & (own[v] < 0)).sum()) for v in range(n)]
        pair = []
        for u in range(n):
            for v in range(u + 1, n):
                if R8[u, v]:
                    outside = (own[u] < 0) & (own[v] < 0)
                    pair.append(int(((Bn[u] != Bn[v]) & outside).sum()))
        res[name] = dict(surface_f1=float(np.mean([r['surface_f1'] for r in mt])), furn_f1=float(np.mean([r['furn_f1'] for r in mt if r['furn_f1'] is not None])),
                         mismatch_vs_central=float(np.mean(mis)), pair_disagreement_mean=float(np.mean(pair)) if pair else 0.,
                         pair_disagreement_max=int(max(pair)) if pair else 0)
    return res


S = {}
for m_thr in (8, 32):
    rows = [retry(one, p, m_thr) for p in houses()]
    S[f'm={m_thr}'] = {k: {q: float(np.mean([r[k][q] for r in rows])) for q in rows[0][k]} for k in rows[0]}
    print(m_thr, json.dumps(S[f'm={m_thr}'], indent=1), flush=True)
json.dump(S, open(os.path.join(HERE, 'p10_lossy_relay.json'), 'w'), indent=1)

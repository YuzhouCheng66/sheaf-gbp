"""A04 probe P9 (Claude, 2026-10-27, debate round): with NOISY, conditionally independent per-agent evidence, which member of the
fusion family wins -- max selection (beta = inf, the current rule), sum of log-odds on the Kruskal forest (beta = 1, exact BP), or sum
flooded on the LOOPY radio graph (8 m, 8 synchronous rounds, cavity messages, i.e. what sum-GBP does without a forest) -- and by how much?
Noise model (voxel level, i.i.d. per agent and voxel, deliberately simple so the comparison is about fusion, not about sensing):
every voxel an agent knows is flipped occupied <-> free with probability p, or dropped to unknown with probability 0.1. Evidence is
+1 / -1 / 0 (log-odds in units of L = log((1-p)/p)). Decisions: occupied if fused > 0, free if < 0, unknown if 0. Scored against the
CLEAN union target (the 'true-value' scoring the noisy setting needs) with the training metric, plus FPR, 3-way accuracy and free recall.
max selection: the receiver takes the evidence of the first (lowest-index) reached source that has any (all |l| are equal here, so this
is the current rule's tie order: one noisy observation per voxel).  K0 rows = the agent's own noisy map against its clean own map."""
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, targets, metrics, gt_masks, HERE, retry  # noqa: E402

PS = (0.05, 0.15)
DROP = 0.1


def radio(pos, r=8.0):
    d = np.linalg.norm(pos[:, None, :2] - pos[None, :, :2], axis=-1)
    a = d <= r; np.fill_diagonal(a, False)
    return a


def extra(pred, tgt):
    known = tgt >= 0
    return dict(fpr=float((pred[~known] >= 0).mean()) if (~known).any() else 0., acc=float((pred[known] == tgt[known]).mean()),
                free_recall=float((pred[tgt == 0] == 0).mean()) if (tgt == 0).any() else None)


def decide(s):
    return np.where(s > 0, np.int8(1), np.where(s < 0, np.int8(0), np.int8(-1))).astype(np.int8)


def one(path, p, seed):
    m, own, pred, gt = load(path); n = own.shape[0]
    slab, obj, struct = gt_masks(gt)
    R8 = np.asarray(m['reached'][m['hops'].index(8)]).astype(bool)
    T8 = targets(own, R8)
    rng = np.random.default_rng(seed)
    ev = np.zeros(own.shape, np.int8)
    for a in range(n):
        k = own[a] >= 0
        u = rng.random(own[a].shape)
        val = np.where(own[a] == 1, 1, -1).astype(np.int8)
        val = np.where(u < p, -val, val)                                   # flip
        val = np.where((u >= p) & (u < p + DROP), 0, val)                  # drop to unknown
        ev[a] = np.where(k, val, 0)
    pos = np.asarray(m['positions_m'], np.float64); A = radio(pos)
    loops = int(A.sum() // 2 - (n - 1))
    rows = {}
    # K0: own noisy map vs clean own map
    rows['K0_noisy_own'] = [decide(ev[a].astype(np.int64)) for a in range(n)]
    # R8 rules (receivers reach their forest component; the radio graph spans the same component)
    tree, mx = [], []
    for a in range(n):
        src = np.nonzero(R8[a])[0]
        tree.append(decide(ev[src].astype(np.int64).sum(0)))
        first = np.zeros(own.shape[1:], np.int8); filled = np.zeros(own.shape[1:], bool)
        for s in [a] + [s for s in src if s != a]:                         # equal precisions: own first, then index order (current tie order)
            take = (~filled) & (ev[s] != 0); first[take] = ev[s][take]; filled |= ev[s] != 0
        mx.append(decide(first.astype(np.int64)))
    rows['R8_max'] = mx; rows['R8_tree_sum'] = tree
    # loopy flooding of the sum on the radio graph (8 rounds, cavity messages)
    E = [(i, j) for i in range(n) for j in range(n) if A[i, j]]
    msg = {e: np.zeros(own.shape[1:], np.float64) for e in E}
    for _ in range(8):
        tot = [np.zeros(own.shape[1:], np.float64) for _ in range(n)]
        for (k, l), v in msg.items():
            tot[l] += v
        msg = {(i, j): ev[i] + tot[i] - msg[(j, i)] for (i, j) in E}       # cavity: exclude what j sent to i
        del tot
    loopy = []
    tot = [np.zeros(own.shape[1:], np.float64) for _ in range(n)]
    for (k, l), v in msg.items():
        tot[l] += v
    for a in range(n):
        loopy.append(decide(ev[a] + tot[a]))
    del tot, msg
    rows['R8_loopy_sum'] = loopy
    out = {}
    for name, P in rows.items():
        T = own if name.startswith('K0') else T8
        Pn = np.stack(P)
        mt = metrics(torch.from_numpy(Pn.reshape(n, -1)), torch.from_numpy(T.reshape(n, -1)), slab, obj, struct)
        out[name] = [dict(**r, **extra(Pn[a], T[a])) for a, r in enumerate(mt)]
    disagree = float(np.mean([(np.stack(loopy)[a] != np.stack(tree)[a])[T8[a] >= 0].mean() for a in range(n)]))
    return out, loops, disagree


S = {}
for p in PS:
    agg, loops, dis = {}, [], []
    for hi, path in enumerate(houses()):
        r, lp, d = retry(one, path, p, 342101 + hi)
        loops.append(lp); dis.append(d)
        for k, v in r.items():
            agg.setdefault(k, []).extend(v)
    S[f'p={p}'] = {k: {q: float(np.mean([x[q] for x in v if x[q] is not None])) for q in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr', 'acc', 'free_recall')}
                   for k, v in agg.items()}
    S[f'p={p}']['radio_cycle_rank_mean'] = float(np.mean(loops)); S[f'p={p}']['loopy_vs_tree_decision_disagreement'] = float(np.mean(dis))
    print(p, json.dumps(S[f'p={p}'], indent=1), flush=True)
json.dump(S, open(os.path.join(HERE, 'p9_noise_fusion.json'), 'w'), indent=1)

"""A04 probe P6 (Claude, 2026-10-27): toy numerics for the energy / lattice statements in A04_sheaf.md (no project data).
(a) max-key lattice join on a LOOPY graph, asynchronous random schedule with duplicated and stale messages, floor-quantised precision
    (8-bit log codebook) -> beliefs equal the centralised join; the network free energy Phi = sum_v sum_k -1/2 log q_vk never increases.
(b) sum-GBP flooding on the same loopy graph -> precision overcount after 8 / 16 rounds (the 875-1,582x phenomenon in miniature).
(c) the beta family (fused precision = ||q||_beta): overcount factor after 8 rounds vs beta; beta = inf is the only idempotent member.
(d) join-homomorphic code: enc(S) = max_{o in S} phi(o) (max-pooled ReLU features, floor-quantised) -> elementwise-max consensus is
    bit-exact enc(union); per-block KEY SELECTION of the same codes loses the information of the losers (measured as feature mass lost).
"""
import json, math, os, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(342101); random.seed(342101)
out = {}

# ---- graph: 8 nodes, connected, with cycles
n = 8
while True:
    A = rng.random((n, n)) < 0.45; A = np.triu(A, 1); A = A | A.T
    seen = {0}; st = [0]
    while st:
        u = st.pop()
        for v in np.nonzero(A[u])[0]:
            if v not in seen:
                seen.add(int(v)); st.append(int(v))
    if len(seen) == n and A.sum() // 2 > n:
        break
E = [(i, j) for i in range(n) for j in range(n) if A[i, j]]
out['graph'] = dict(nodes=n, undirected_edges=int(A.sum() // 2), cycles_rank=int(A.sum() // 2 - n + 1))

# ---- (a) max-key join, async, duplicates / stale messages, floor quantiser
K = 200
low, high, NQ = 1e-8, 128., 255
def qfloor(p):
    code = np.floor((np.log(np.clip(p, low, high)) - math.log(low)) / (math.log(high) - math.log(low)) * (NQ - 1))
    return np.where(p > 0, np.exp(code / (NQ - 1) * (math.log(high) - math.log(low)) + math.log(low)), 0.)
obs = rng.random((n, K)) < 0.3
q = np.where(obs, np.exp(rng.uniform(math.log(1e-3), math.log(4.), (n, K))), 0.)
mu = np.where(obs, np.round(rng.uniform(-1, 1, (n, K)) * 127) / 127, 0.)
qq = qfloor(q)                                                     # quantised ONCE at the source; relays forward bit-exact
def key(p, m):                                                      # exact lexicographic order on (precision code, mean code) as one integer
    pc = np.where(p > 0, np.round((np.log(np.maximum(p, low)) - math.log(low)) / (math.log(high) - math.log(low)) * (NQ - 1)), -1)
    return (pc.astype(np.int64) + 1) * 256 + (np.round(m * 127).astype(np.int64) + 127)
bel_q, bel_m = qq.copy(), mu.copy()
def phi(bq):
    return float(np.sum(-0.5 * np.log(np.maximum(bq, 1e-12))))
history = [phi(bel_q)]; mono = True; sent = []
for step in range(4000):
    i, j = E[rng.integers(len(E))]
    if sent and rng.random() < 0.2:                                  # replay a stale / duplicated message
        j, mq, mm = sent[rng.integers(len(sent))]
    else:
        mq, mm = bel_q[i].copy(), bel_m[i].copy()                   # (cavity omitted on purpose: idempotence makes it unnecessary)
        sent.append((j, mq, mm))
    better = key(mq, mm) > key(bel_q[j], bel_m[j])
    bel_q[j] = np.where(better, mq, bel_q[j]); bel_m[j] = np.where(better, mm, bel_m[j])
    history.append(phi(bel_q)); mono &= history[-1] <= history[-2] + 1e-9
best = np.argmax(key(qq, mu), 0)
cent_q = qq[best, np.arange(K)]; cent_m = mu[best, np.arange(K)]
out['a_max_join'] = dict(async_updates=4000, stale_or_duplicate_share=0.2, equals_centralised=bool(np.all(bel_q == cent_q[None]) and np.all(bel_m == cent_m[None])),
                         free_energy_monotone=bool(mono), phi_start=history[0], phi_end=history[-1])

# ---- (b) sum-GBP flooding on the loopy graph (synchronous, cavity = exclude the reverse message)
def flood(beta, rounds):
    """fused precision ||q||_beta, messages i->j = beta-pool of own and incoming except from j; returns mean overcount vs centralised"""
    pw = lambda x: np.power(x, beta) if beta != np.inf else x
    msg = {(i, j): np.zeros(K) for i, j in E}
    for _ in range(rounds):
        new = {}
        for i, j in E:
            parts = [qq[i]] + [msg[(k, i)] for k, _ in E if _ == i and k != j]
            new[(i, j)] = np.max(parts, 0) if beta == np.inf else np.power(np.sum([pw(p) for p in parts], 0), 1 / beta)
        msg = new
    fused = []
    for j in range(n):
        parts = [qq[j]] + [msg[(k, j)] for k, _ in E if _ == j]
        fused.append(np.max(parts, 0) if beta == np.inf else np.power(np.sum([pw(p) for p in parts], 0), 1 / beta))
    fused = np.array(fused)
    cent = qq.max(0) if beta == np.inf else np.power(np.sum(pw(qq), 0), 1 / beta)
    sel = cent > 0
    return float(np.median(fused[:, sel] / cent[None, sel]))
out['b_c_overcount_median'] = {f'beta={b}': {f'R{r}': flood(b, r) for r in (8, 16)} for b in (1, 2, 4, 8, np.inf)}

# ---- (d) join-homomorphic max-pool code vs key selection
U = 400; d = 16; B = 20                                             # universe of observations, feature dim, blocks
blk_of = rng.integers(B, size=U)
W = rng.normal(size=(6, d)); feat = np.maximum(rng.normal(size=(U, 6)) @ W, 0)    # phi(o) >= 0 (ReLU), bottom = 0
featq = np.floor(feat * 8) / 8                                      # monotone floor quantiser commutes with max
S = rng.random((n, U)) < 0.25                                       # agent observation sets (overlapping)
def enc(mask):
    code = np.zeros((B, d))
    for b in range(B):
        sel = mask & (blk_of == b)
        if sel.any():
            code[b] = featq[sel].max(0)
    return code
codes = np.stack([enc(S[i]) for i in range(n)])
fused = codes.max(0)                                                 # elementwise-max consensus (any schedule, any graph)
exact = bool(np.array_equal(fused, enc(S.any(0))))
cov = np.stack([[(S[i] & (blk_of == b)).sum() for b in range(B)] for i in range(n)])
win = np.argmax(cov, 0); sel = codes[win, np.arange(B)]
lost = float((fused - sel).sum() / fused.sum())
out['d_homomorphic'] = dict(fused_equals_enc_of_union=exact, key_selection_feature_mass_lost=lost)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, 'p6_energy_toy.json'), 'w'), indent=1)

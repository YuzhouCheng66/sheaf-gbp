"""probe_wire.py (A01, 2026-10-27): how many stalks does the max-precision GBP actually NEED to put on the wire?
Simulates the deployed protocol (rollout_max: synchronous rounds on the radio forest, message i->j = select(own_i, messages k->i,
k != j), belief j = select(own_j, all messages into j)) on the 6 VAL demo houses, with a PROXY key: the block coverage (fraction of the
column's voxels the agent knows; ties broken by the lower agent index, exactly like select()). Learned q differs from coverage (analyst A:
the coverage key picks the same winner in 37-58 % of contested blocks), so winner identities are a proxy; the COUNTS of evidenced stalks
are exact for any precision that is zero iff the block is unobserved (mask = coverage > 0, as in OwnedHead).
Protocols, counted per directed edge and round (stalk sends):
  fixed      : D stalks every round (current wire: D x 16 bit)
  sparse     : only evidenced stalks of the message (key > 0) + an evidence mask every round
  delta      : only stalks whose (key, source) changed since the last round on that edge (round 1 = all evidenced)
  delta+dom  : delta, and only where the message beats the REVERSE message j->i of the previous round in the select() order; the
               receiver keeps the last value it got. Beliefs must equal the fixed protocol's at every round (asserted).
Bytes: payload b bit per sent stalk (b = 13.63: marginal entropy of the 8+8 symbols measured by analyst B on 3S-X K0; b = 7: 4+4 matched,
entropy-coded estimate; b = 4: rate-trained estimate) + addressing (min of the enumerative bound log2 C(D, k) and a 2-D context code of
the changed-set map). Lossless references per directed edge: ctx of the sender-side union (rate_baselines) -- printed by probe_raycast.
Usage: python probe_wire.py DATA_DIR OUT_JSON"""
import glob, json, math, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from rate_baselines_lib import load, LG  # noqa: E402

DATA = sys.argv[1]; OUT = sys.argv[2]
ROUNDS = 8
BITS = (13.63, 7.0, 4.0)


def ctx2d_bits(mask):
    """adaptive KT code of a binary 2-D map, causal context W, N, NW, NE (+ 2 B header not added here)"""
    s = torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64)); X, Y = s.shape
    def sh(dx, dy):
        o = torch.zeros_like(s)
        xs = slice(max(dx, 0), X + min(dx, 0)); xd = slice(max(-dx, 0), X - max(dx, 0))
        ys = slice(max(dy, 0), Y + min(dy, 0)); yd = slice(max(-dy, 0), Y - max(dy, 0))
        o[xs, ys] = s[xd, yd]; return o
    c = torch.zeros_like(s)
    for d in ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)):
        c = c * 2 + sh(*d)
    idx = (c * 2 + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=2 ** 7).reshape(-1, 2).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    return float(((LG(N + 1.) - LG(1.)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2))


def log2_binom(n, k):
    return float((LG(n + 1.) - LG(k + 1.) - LG(n - k + 1.)) / math.log(2))


def better(ka, sa, kb, sb):
    """select() order: higher key wins; tie -> lower source index wins"""
    return (ka > kb) | ((ka == kb) & (sa < sb))


def run_house(path, G):
    m, own, pred, gt = load(path); n = own.shape[0]; s = 128 // G; D = G * G
    cov = (own >= 0).reshape(n, G, s, G, s, 32).mean((2, 4, 5)).reshape(n, D)          # proxy key, 0 iff unobserved
    edges = [tuple(e) for e in m['edges']]
    directed = [(i, j) for i, j in edges] + [(j, i) for i, j in edges]
    inc = {j: [a for a, (i2, j2) in enumerate(directed) if j2 == j] for j in range(n)}
    rev = {a: directed.index((j, i)) for a, (i, j) in enumerate(directed)}
    NONE_K = np.zeros(D); NONE_S = np.full(D, 10 ** 6)
    own_k = cov.copy(); own_s = np.where(cov > 0, np.arange(n)[:, None], 10 ** 6).astype(np.int64)
    own_k = np.where(cov > 0, cov, 0.)

    def sel(cands):
        K = np.stack([c[0] for c in cands]); S = np.stack([c[1] for c in cands])
        order = np.lexsort((S, -K), axis=0)[0]                        # max key, then min source
        return K[order, np.arange(D)], S[order, np.arange(D)]

    def beliefs(msgs):
        return [sel([(own_k[j], own_s[j])] + [msgs[a] for a in inc[j]]) for j in range(n)]

    full = [(NONE_K.copy(), NONE_S.copy()) for _ in directed]          # fixed protocol state
    rx = [(NONE_K.copy(), NONE_S.copy()) for _ in directed]            # delta+dom: what the receiver holds per edge
    last_sent = [(NONE_K.copy(), NONE_S.copy()) for _ in directed]     # delta: what the sender last put on the edge
    rx_dom_prev = [(NONE_K.copy(), NONE_S.copy()) for _ in directed]
    counts = dict(fixed=[], sparse=[], delta=[], delta_dom=[])
    addr = dict(sparse=[], delta=[], delta_dom=[])
    for r in range(1, ROUNDS + 1):
        new_full = []
        for a, (i, j) in enumerate(directed):
            new_full.append(sel([(own_k[i], own_s[i])] + [full[b] for b in inc[i] if directed[b][0] != j]))
        # delta+dom uses what each node actually holds (rx) to compute its outgoing message
        new_rx = []
        sent_dom = []
        for a, (i, j) in enumerate(directed):
            mk, ms = sel([(own_k[i], own_s[i])] + [rx[b] for b in inc[i] if directed[b][0] != j])
            changed = (mk != last_sent[a][0]) | (ms != last_sent[a][1])
            rk, rs = rx_dom_prev[rev[a]]                                 # reverse message as held by i after round r-1
            useful = better(mk, ms, rk, rs) & (mk > 0)
            send = changed & useful
            hk = np.where(send, mk, rx[a][0]); hs = np.where(send, ms, rx[a][1])
            new_rx.append((hk, hs)); sent_dom.append(int(send.sum()))
            last_sent[a] = (np.where(send, mk, last_sent[a][0]), np.where(send, ms, last_sent[a][1]))
            addr['delta_dom'].append(min(log2_binom(D, int(send.sum())), ctx2d_bits(send.reshape(G, G))) if send.any() else 1.)
        rx_dom_prev = [tuple(x) for x in new_rx]
        for a in range(len(directed)):
            k_new, s_new = new_full[a]; k_old, s_old = full[a]
            ev = k_new > 0
            ch = ((k_new != k_old) | (s_new != s_old)) & ev
            counts['fixed'].append(D); counts['sparse'].append(int(ev.sum())); counts['delta'].append(int(ch.sum()))
            addr['sparse'].append(min(log2_binom(D, int(ev.sum())), ctx2d_bits(ev.reshape(G, G))))
            addr['delta'].append(min(log2_binom(D, int(ch.sum())), ctx2d_bits(ch.reshape(G, G))) if ch.any() else 1.)
        counts['delta_dom'] += sent_dom
        full = new_full; rx = new_rx
        bf = beliefs(full); bd = beliefs(rx)
        for j in range(n):                                               # beliefs identical (winning key; source may tie)
            assert np.array_equal(bf[j][0], bd[j][0]), ('belief key differs', path, G, r, j)
            assert np.array_equal(bf[j][1], bd[j][1]), ('belief source differs', path, G, r, j)
    E = len(directed)
    out = dict(house=m['house'], G=G, D=D, directed_edges=E, k0_evidenced=[int((cov[a] > 0).sum()) for a in range(n)],
               k0_mask_bits=[ctx2d_bits((cov[a] > 0).reshape(G, G)) for a in range(n)])
    for p in counts:
        c = np.asarray(counts[p]).reshape(ROUNDS, E)
        out[f'{p}_per_round_mean'] = c.mean(1).tolist(); out[f'{p}_session_per_edge'] = c.sum(0).tolist()
    for p in addr:
        a_ = np.asarray(addr[p]).reshape(ROUNDS, E)
        out[f'{p}_addr_bits_session_per_edge'] = a_.sum(0).tolist(); out[f'{p}_addr_bits_per_round_mean'] = a_.mean(1).tolist()
    # converged message (round 8) evidenced count per edge
    out['converged_evidenced'] = np.asarray(counts['sparse']).reshape(ROUNDS, E)[-1].tolist()
    out['converged_addr_bits'] = np.asarray(addr['sparse']).reshape(ROUNDS, E)[-1].tolist()
    return out


rows = []
for G in (32, 64):
    for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
        rows.append(run_house(path, G)); print(G, rows[-1]['house'], 'ok', flush=True)

S = {}
for G in (32, 64):
    R = [r for r in rows if r['G'] == G]; D = R[0]['D']
    k0 = np.concatenate([r['k0_evidenced'] for r in R]); k0m = np.concatenate([r['k0_mask_bits'] for r in R])
    conv = np.concatenate([r['converged_evidenced'] for r in R]); conva = np.concatenate([r['converged_addr_bits'] for r in R])
    sess = {p: np.concatenate([r[f'{p}_session_per_edge'] for r in R]) for p in ('fixed', 'sparse', 'delta', 'delta_dom')}
    sess_addr = {p: np.concatenate([r[f'{p}_addr_bits_session_per_edge'] for r in R]) for p in ('sparse', 'delta', 'delta_dom')}
    d = dict(D=D, k0_evidenced_median=float(np.median(k0)), k0_evidenced_frac=float(np.mean(k0) / D), k0_mask_bytes_median=float(np.median(k0m) / 8),
             converged_evidenced_median=float(np.median(conv)), converged_evidenced_frac=float(np.mean(conv) / D), converged_mask_bytes_median=float(np.median(conva) / 8),
             session_stalks_per_edge_median={p: float(np.median(v)) for p, v in sess.items()},
             session_addr_bytes_per_edge_median={p: float(np.median(v) / 8) for p, v in sess_addr.items()})
    for b in BITS:
        d[f'bytes_b{b}'] = dict(
            k0_message_median=float(np.median(k0 * b / 8 + k0m / 8 + 2)),
            converged_message_median=float(np.median(conv * b / 8 + conva / 8 + 2)),
            session_per_edge_fixed=float(np.median(sess['fixed'])) * 16 / 8,
            session_per_edge_sparse_median=float(np.median(sess['sparse'] * b / 8 + sess_addr['sparse'] / 8 + 2 * 8)),
            session_per_edge_delta_median=float(np.median(sess['delta'] * b / 8 + sess_addr['delta'] / 8 + 2 * 8)),
            session_per_edge_delta_dom_median=float(np.median(sess['delta_dom'] * b / 8 + sess_addr['delta_dom'] / 8 + 2 * 8)))
    S[f'G{G}'] = d
json.dump(dict(summary=S, rows=rows), open(OUT, 'w'), indent=1)
print(json.dumps(S, indent=1))

"""a10_stalk.py (A10, 2026-10-27): how many STALKS would the GBP messages carry if they were innovation-coded instead of fixed-length?
Simulates the trainer's synchronous 8-round max-precision join (train_3d_v431.rollout_max: extrinsic messages, own value first on
ties) on the 6 demo trees for code layouts L x L x 1 (L = 16, 32, 64, 128; a stalk = one (128/L) x (128/L) x 32 voxel column).
Precision proxy = the column's known-voxel coverage quantised to 255 levels (evidence iff coverage > 0); the learned q of the trained
model ranks sources differently (D8-22), but the COUNTS of evidence / innovation stalks depend mainly on where the agents looked.
Protocols per directed edge and round:
  fixed   : all D stalks (today: 8 rounds x 14 x D)
  sparse  : only stalks with evidence in the extrinsic belief (no conditioning)
  delta   : only stalks whose extrinsic value rose since this edge last sent them (temporal side information)
  innov   : only stalks whose extrinsic value beats max(last sent on this edge, best heard back from the receiver) (temporal + reverse
            side information; both ends know both). Beliefs of innov are asserted EQUAL to fixed's at every round (exactness).
Support masks are coded with an adaptive binary KT model (context = receiver-reported evidence at k, sender-sent-before at k, left and
up neighbours in the current mask) -- exact code lengths; values are charged b bits per sent stalk (b = 16 raw 8+8, 10, 6)."""
import glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import load, adjacency, kt_bits  # noqa

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
ROUNDS = 8
BVALS = (16, 10, 6)


def coverage(own, L):
    k = (own >= 0).reshape(own.shape[0], L, 128 // L, L, 128 // L, 32).mean((2, 4, 5))
    return np.ceil(k * 255).astype(np.int64).reshape(own.shape[0], -1)          # 0 = no evidence, 1..255


def simulate(x, edges, n):
    adj = adjacency(edges, n); D = x.shape[1]
    directed = [(i, j) for i, j in edges] + [(j, i) for i, j in edges]
    acc_f = {e: np.zeros(D, np.int64) for e in directed}                        # fixed protocol: latest message (monotone)
    acc_i = {e: np.zeros(D, np.int64) for e in directed}                        # innov: max ever received on the edge
    sent_i = {e: np.zeros(D, np.int64) for e in directed}
    sent_d = {e: np.zeros(D, np.int64) for e in directed}
    counts = dict(fixed=0, sparse=0, delta=0, innov=0)
    per_round = {k: [0] * ROUNDS for k in counts}
    masks = []                                                                  # (mask, ctx) per innov message for support coding
    for t in range(ROUNDS):
        new_f, new_i, sends = {}, {}, {}
        for (s, r) in directed:
            ef = x[s].copy()
            for q in adj[s]:
                if q != r:
                    ef = np.maximum(ef, acc_f[(q, s)])
            ei = x[s].copy()
            for q in adj[s]:
                if q != r:
                    ei = np.maximum(ei, acc_i[(q, s)])
            # NB: ei <= ef can differ (a value that came FROM r's side is not echoed back by q), the beliefs below are asserted equal
            known_r = np.maximum(sent_i[(s, r)], acc_i[(r, s)])
            K = ei > known_r
            new_f[(s, r)] = ef; sends[(s, r)] = (K, ei)
            c_fixed = D; c_sparse = int((ef > 0).sum()); c_delta = int((ef > sent_d[(s, r)]).sum()); c_innov = int(K.sum())
            sent_d[(s, r)] = np.maximum(sent_d[(s, r)], ef)
            for k_, v in (('fixed', c_fixed), ('sparse', c_sparse), ('delta', c_delta), ('innov', c_innov)):
                counts[k_] += v; per_round[k_][t] += v
            ctxk = (acc_i[(r, s)] > 0).astype(np.int64) * 2 + (sent_i[(s, r)] > 0)
            masks.append((K.copy(), ctxk))
        for e in directed:                                                      # synchronous update
            acc_f[e] = new_f[e]
            K, v = sends[e]
            acc_i[e] = np.where(K, np.maximum(acc_i[e], v), acc_i[e]); sent_i[e] = np.where(K, np.maximum(sent_i[e], v), sent_i[e])
        for j in range(n):                                                      # exactness: beliefs identical every round
            bf = x[j].copy(); bi = x[j].copy()
            for q in adj[j]:
                bf = np.maximum(bf, acc_f[(q, j)]); bi = np.maximum(bi, acc_i[(q, j)])
            assert (bf == bi).all(), ('belief mismatch', t, j)
    final = [x[j].copy() for j in range(n)]
    for j in range(n):
        for q in adj[j]:
            final[j] = np.maximum(final[j], acc_i[(q, j)])
    gmax = x.max(0)
    assert all((f == gmax).all() for f in final)
    return counts, per_round, masks


def mask_bits(masks, L):
    """adaptive KT over all messages of a house; context = (receiver evidence, sent before) x left x up neighbours in the mask"""
    syms, ctxs = [], []
    for K, c in masks:
        g = K.reshape(L, L).astype(np.int64); left = np.zeros_like(g); up = np.zeros_like(g)
        left[:, 1:] = g[:, :-1]; up[1:, :] = g[:-1, :]
        syms.append(g.reshape(-1)); ctxs.append((c.reshape(L, L) * 4 + left * 2 + up).reshape(-1))
    return kt_bits(np.concatenate(syms), np.concatenate(ctxs), 2)[0]


def own_mask_bits(x, L):
    g = (x > 0).reshape(-1, L, L).astype(np.int64); left = np.zeros_like(g); up = np.zeros_like(g); ul = np.zeros_like(g)
    left[:, :, 1:] = g[:, :, :-1]; up[:, 1:, :] = g[:, :-1, :]; ul[:, 1:, 1:] = g[:, :-1, :-1]
    return kt_bits(g.reshape(-1), (left * 4 + up * 2 + ul).reshape(-1), 2)[0]


res = {}
for L in (16, 32, 64, 128):
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
        m, own, pred, gt = load(path); n = own.shape[0]
        x = coverage(own, L)
        counts, per_round, masks = simulate(x, [tuple(e) for e in m['edges']], n)
        mb = mask_bits(masks, L); ob = own_mask_bits(x, L)
        nmsg = len(masks); D = L * L
        row = dict(house=m['house'], D=D, n_msgs=nmsg, counts=counts, per_round=per_round, mask_bytes_total=mb / 8,
                   own_evidence_mean=float((x > 0).sum(1).mean()), own_mask_bytes_mean=ob / 8 / n,
                   final_evidence_mean=float((x.max(0) > 0).sum()))
        for b in BVALS:
            row[f'innov_bytes_total_b{b}'] = counts['innov'] * b / 8 + mb / 8
            row[f'k0_bytes_b{b}'] = row['own_evidence_mean'] * b / 8 + ob / 8 / n
        rows.append(row)
    agg = dict(D=L * L)
    for k in ('fixed', 'sparse', 'delta', 'innov'):
        agg[f'{k}_stalks_per_house'] = float(np.mean([r['counts'][k] for r in rows]))
    agg['innov_per_round'] = [float(np.mean([r['per_round']['innov'][t] for r in rows])) for t in range(ROUNDS)]
    agg['mask_bytes_per_house'] = float(np.mean([r['mask_bytes_total'] for r in rows]))
    agg['own_evidence_stalks'] = float(np.mean([r['own_evidence_mean'] for r in rows]))
    agg['final_evidence_stalks'] = float(np.mean([r['final_evidence_mean'] for r in rows]))
    agg['own_mask_bytes'] = float(np.mean([r['own_mask_bytes_mean'] for r in rows]))
    for b in BVALS:
        agg[f'innov_bytes_per_house_b{b}'] = float(np.mean([r[f'innov_bytes_total_b{b}'] for r in rows]))
        agg[f'k0_bytes_b{b}'] = float(np.mean([r[f'k0_bytes_b{b}'] for r in rows]))
    agg['fixed_bytes_per_house_8p8'] = agg['fixed_stalks_per_house'] * 2
    res[L] = dict(agg=agg, rows=rows)
    print(f'L={L:3d} D={L*L:5d}  stalks/house: fixed {agg["fixed_stalks_per_house"]:8.0f} sparse {agg["sparse_stalks_per_house"]:7.0f} '
          f'delta {agg["delta_stalks_per_house"]:7.0f} innov {agg["innov_stalks_per_house"]:7.0f} | mask {agg["mask_bytes_per_house"]:6.0f} B | '
          f'innov bytes/house b16 {agg["innov_bytes_per_house_b16"]:7.0f} b10 {agg["innov_bytes_per_house_b10"]:7.0f} b6 {agg["innov_bytes_per_house_b6"]:7.0f} | '
          f'own evid {agg["own_evidence_stalks"]:6.0f} ({agg["own_evidence_stalks"]/(L*L):.2f}) K0 bytes b16 {agg["k0_bytes_b16"]:6.0f} b10 {agg["k0_bytes_b10"]:6.0f} b6 {agg["k0_bytes_b6"]:6.0f} | '
          f'final evid {agg["final_evidence_stalks"]:6.0f}', flush=True)
    print('      innov stalks per round', [round(v) for v in agg['innov_per_round']])
json.dump(res, open(os.path.join(HERE, 'a10_stalk.json'), 'w'), indent=1)

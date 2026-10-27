"""a10_ctxsearch.py: pick compact context designs for the conditional / innovation coders (context-dilution check).
Evaluated on ALL 84 directed tree edges of the 6 demo houses (sender-side union S, receiver-side union R); the chosen design is then
fixed for every later number (choosing 1 of K designs costs log2 K bits, negligible). numpy only."""
import glob, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import *  # noqa

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
NB4 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 2)]


def kt(t, c, sel, n):
    if sel is None:
        return kt_bits(t, c, n)[0] / 8 + 2
    return kt_bits(t[sel], c[sel], n)[0] / 8 + 2


def variants_innov(Um, R, reach):
    t, s = T(Um), T(R); sel = (s == 0); kf = fwd_flags(s > 0); c9 = causal(t, 3); c6 = causal(t, 3, NB6); rr = reach.astype(np.int64)
    return {'i_c9': kt(t, c9, sel, 3), 'i_c9_f': kt(t, c9 * 8 + kf, sel, 3), 'i_c6_f': kt(t, c6 * 8 + kf, sel, 3),
            'i_c9_r': kt(t, c9 * 2 + rr, sel, 3), 'i_c9_f_r': kt(t, (c9 * 8 + kf) * 2 + rr, sel, 3), 'i_c6_f_r': kt(t, (c6 * 8 + kf) * 2 + rr, sel, 3)}


def variants_cond(S, Rown):
    t, s = T(S), T(Rown); kf = fwd_flags(s > 0); c9 = causal(t, 3); c6 = causal(t, 3, NB6); c4 = causal(t, 3, NB4)
    return {'c_c9': kt(t, c9, None, 3), 'c_c9_s': kt(t, c9 * 3 + s, None, 3), 'c_c6_s': kt(t, c6 * 3 + s, None, 3),
            'c_c6_s_f': kt(t, (c6 * 3 + s) * 8 + kf, None, 3), 'c_c4_s_f': kt(t, (c4 * 3 + s) * 8 + kf, None, 3)}


def variants_vinnov(Vm, Rknown, reach):
    v = Vm.astype(np.int64); sel = ~Rknown; kf = fwd_flags(Rknown); c9 = causal(v, 2); rr = reach.astype(np.int64)
    return {'v_c9': kt(v, c9, sel, 2), 'v_c9_f': kt(v, c9 * 8 + kf, sel, 2), 'v_c9_r': kt(v, c9 * 2 + rr, sel, 2), 'v_c9_f_r': kt(v, (c9 * 8 + kf) * 2 + rr, sel, 2)}


tot = {}
t0 = time.time()
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]; adj = adjacency(m['edges'], n)
    pos = np.asarray(m['positions_m']); cell = m['cell_m']
    if path.endswith('r1_i0.json'):
        assert abs(tern_bytes(own[0]) - 1225.675604358198) < 1e-6, tern_bytes(own[0])          # = rate_baselines_lib.ctx_bytes
    for i, j in m['edges']:
        for s_, r_ in ((i, j), (j, i)):
            Sset = side_of(adj, s_, r_); Rset = side_of(adj, r_, s_)
            S = union(own, Sset); R = union(own, Rset); Um = np.maximum(S, R)
            reach = np.zeros(S.shape, bool)
            for a in Sset:
                reach |= reachable(R == 1, cell, pos[a])
            for d in (variants_innov(Um, R, reach), variants_cond(S, own[r_]), variants_vinnov(Um == 1, R >= 0, reach)):
                for k, v in d.items():
                    tot.setdefault(k, []).append(v)
    print(m['house'], f'{time.time() - t0:.0f}s', flush=True)
S = {k: dict(sum=float(np.sum(v)), median=float(np.median(v)), n=len(v)) for k, v in tot.items()}
for k, v in S.items():
    print(f'{k:12s} sum {v["sum"]:10.0f}  median {v["median"]:8.1f}')
json.dump(S, open(os.path.join(HERE, 'a10_ctxsearch.json'), 'w'), indent=1)

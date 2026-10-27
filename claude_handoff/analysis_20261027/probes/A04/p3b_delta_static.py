"""A04 probe P3b (Claude, 2026-10-27): P3's adaptive conditional coder pays a per-message learning cost that grows with the context
alphabet (x3 for the receiver-state context), which hides the delta saving.  A learned entropy model is TRAINED once (shared by all
agents, part of the codec), so the fair estimate uses a pretrained context model: leave-one-house-out counts from the other 5 demo
houses, used as a Dirichlet prior (concentration capped at C per context, shaped by the training frequencies) and adapted on the fly.
Per directed tree edge s -> r:  full = code(U_s) ; delta = code(U_s v U_r | U_r(v)) ; also the incremental proxy ctx(U_s v U_r) - ctx(U_r)
with the adaptive coder.  Same coder family for the unconditional K0 own map and R8 union (to compare with the evidence pack ctx)."""
import json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, targets, ctx_bytes, causal_ctx, tree_sides, LG, HERE  # noqa: E402

CAP = 32.


def ctx_ids(y, side=None):
    s = torch.from_numpy(np.ascontiguousarray(y).astype(np.int64))
    c, n = causal_ctx(s, 3)
    if side is not None:
        c = c * 3 + torch.from_numpy(np.ascontiguousarray(side).astype(np.int64)); n *= 3
    return (c * 3 + s).reshape(-1), n


def counts(idx, nctx):
    return torch.bincount(idx, minlength=nctx * 3).reshape(-1, 3).double()


def bits_with_prior(idx, nctx, prior):
    """Dirichlet-multinomial code length: prior pseudo-counts per context = 0.5 + C * training frequency (C = min(N_train, CAP))"""
    cnt = counts(idx, nctx)
    Nt = prior.sum(1, keepdim=True)
    a = 0.5 + torch.where(Nt > 0, prior / Nt.clamp_min(1) * Nt.clamp(max=CAP), torch.zeros_like(prior))
    used = cnt.sum(1) > 0; cnt, a = cnt[used], a[used]
    A = a.sum(1); N = cnt.sum(1)
    return float(((LG(N + A) - LG(A)) - (LG(cnt + a) - LG(a)).sum(1)).sum() / math.log(2))


data = []
for path in houses():
    m, own, pred, gt = load(path); n = own.shape[0]
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    sides = tree_sides(n, [tuple(e) for e in m['edges']])
    pairs = [(own[ss].max(0), own[sides[(r, s)]].max(0)) for (s, r), ss in sides.items()]
    data.append(dict(house=m['house'], own=own, T8=T8, pairs=pairs))
    print('loaded', m['house'], flush=True)

N0 = 3 ** 9; N1 = 3 ** 10
tabs = []
for d in data:                                                       # per-house training count tables
    u = torch.zeros(N0 * 3, dtype=torch.float64); c = torch.zeros(N1 * 3, dtype=torch.float64)
    for a in range(d['own'].shape[0]):
        u += counts(ctx_ids(d['own'][a] + 1)[0], N0).reshape(-1); u += counts(ctx_ids(d['T8'][a] + 1)[0], N0).reshape(-1)
    for Us, Ur in d['pairs']:
        Y = np.maximum(Us, Ur)
        u += counts(ctx_ids(Us + 1)[0], N0).reshape(-1)
        c += counts(ctx_ids(Y + 1, Ur + 1)[0], N1).reshape(-1)
    tabs.append((u.reshape(-1, 3), c.reshape(-1, 3)))
rows = []
t0 = time.time()
for h, d in enumerate(data):
    pu = sum(t[0] for k, t in enumerate(tabs) if k != h); pc = sum(t[1] for k, t in enumerate(tabs) if k != h)
    for a in range(d['own'].shape[0]):
        rows.append(dict(kind='own', house=d['house'], adaptive=ctx_bytes(d['own'][a]), static=bits_with_prior(ctx_ids(d['own'][a] + 1)[0], N0, pu) / 8 + 2))
        rows.append(dict(kind='union8', house=d['house'], adaptive=ctx_bytes(d['T8'][a]), static=bits_with_prior(ctx_ids(d['T8'][a] + 1)[0], N0, pu) / 8 + 2))
    for Us, Ur in d['pairs']:
        Y = np.maximum(Us, Ur)
        full_s = bits_with_prior(ctx_ids(Us + 1)[0], N0, pu) / 8 + 2
        delta_s = bits_with_prior(ctx_ids(Y + 1, Ur + 1)[0], N1, pc) / 8 + 2
        incr = max(ctx_bytes(Y) - ctx_bytes(Ur), 0.) + 2
        rows.append(dict(kind='edge', house=d['house'], adaptive=ctx_bytes(Us), static=full_s, delta_static=delta_s, incremental_adaptive=incr))
    print(d['house'], f'{time.time() - t0:.0f}s', flush=True)
S = {}
for k in ('own', 'union8', 'edge'):
    r = [x for x in rows if x['kind'] == k]
    S[k] = dict(adaptive_median=float(np.median([x['adaptive'] for x in r])), static_median=float(np.median([x['static'] for x in r])))
    if k == 'edge':
        S[k].update(delta_static_median=float(np.median([x['delta_static'] for x in r])), incremental_adaptive_median=float(np.median([x['incremental_adaptive'] for x in r])),
                    delta_over_full_static_median=float(np.median([x['delta_static'] / x['static'] for x in r])),
                    sum_full_static=float(sum(x['static'] for x in r)), sum_delta_static=float(sum(x['delta_static'] for x in r)),
                    sum_incremental=float(sum(x['incremental_adaptive'] for x in r)))
print(json.dumps(S, indent=1))
json.dump(dict(summary=S, rows=rows, cap=CAP), open(os.path.join(HERE, 'p3b_delta_static.json'), 'w'), indent=1)

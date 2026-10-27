"""R-B probe (a), 2026-10-27: source-side quantisation of each agent's calibrated evidence, exact forest sum, the FROZEN E3-R
network read-out without retraining (a lower bound for a retrained network), and the ctx bytes of the relayed messages.
Imports g4_e3r / g4_noise_panels read-only. CPU only. Writes JSON rows to RB/out_a/. Reads review_E3R/inputs (never copied out).

Quantiser = MI-optimal contiguous partition of the V3 alphabet sorted by the E1c LLR c(L) (TRAIN counts n_occ / n_free of the fit,
table half A or B); each bin's value = the pooled calibrated LLR of the bin, cq_bin = round((log((sum n_occ + .5)/(sum n_free + .5))
- log(N_occ / N_free)) / 0.05) -- the same formula E1c uses per symbol. 'sign' = the one-vote bins {L <= 0}, {L > 0} (tri-state map).
"""
import os, sys, json, time, math
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(v, '1')
sys.dont_write_bytecode = True
ROOT = r'C:\Users\27118\Desktop\sheaf-gbp\experiments\sheaf-gbp-p6-local-skill'
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import numpy as np
import g4_e3r as E
G = E.G
import torch

INP = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'review_E3R', 'inputs')
OUT = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'review_E3R', 'RB', 'out_a')
FIT = os.path.join(INP, 'e1c_fit.json')
LMIN, LMAX = E.LMIN, E.LMAX


def fit_counts(half):
    t = json.load(open(FIT))['halves'][half]['tiers']['cam_nf']
    return np.array(t['n_occ'], float), np.array(t['n_free'], float), np.array(t['cq'], np.int64), int(t['theta'])


def bin_value(no, nf, idx, Lo):
    return int(round((math.log((no[idx].sum() + .5) / (nf[idx].sum() + .5)) - Lo) / 0.05))


def opt_bins(no, nf, c, q):
    """MI-optimal contiguous partition (in c order) into q bins: minimise sum_b N_b h(p_b)"""
    order = np.argsort(c, kind='stable'); a = no[order]; b = nf[order]; n = len(a)
    A = np.concatenate([[0], np.cumsum(a)]); B = np.concatenate([[0], np.cumsum(b)])

    def cost(i, j):
        o = A[j] - A[i]; f = B[j] - B[i]; N = o + f
        if N <= 0:
            return 0.
        r = 0.
        for x in (o, f):
            if x > 0:
                r -= x * math.log(x / N)
        return r
    INF = float('inf'); D = np.full((q + 1, n + 1), INF); P = np.zeros((q + 1, n + 1), int); D[0, 0] = 0.
    for k in range(1, q + 1):
        for j in range(1, n + 1):
            best = INF; bi = 0
            for i in range(k - 1, j):
                v = D[k - 1, i] + cost(i, j)
                if v < best:
                    best, bi = v, i
            D[k, j] = best; P[k, j] = bi
    cuts = []; j = n
    for k in range(q, 0, -1):
        i = P[k, j]; cuts.append((i, j)); j = i
    cuts = cuts[::-1]
    lab = np.zeros(n, int)
    for bidx, (i, j) in enumerate(cuts):
        lab[order[i:j]] = bidx
    return lab


def quant_table(half, kind):
    no, nf, cq, theta = fit_counts(half)
    Lo = math.log(no.sum() / nf.sum())
    c = np.log((no + .5) / (nf + .5)) - Lo
    if kind == 'full':
        return cq, None
    if kind == 'sign':
        Ls = np.arange(LMIN, LMAX + 1); lab = (Ls > 0).astype(int)
    else:
        lab = opt_bins(no, nf, c, int(kind[1:]))
    vals = {int(l): bin_value(no, nf, lab == l, Lo) for l in np.unique(lab)}
    tab = np.array([vals[int(l)] for l in lab], np.int64)
    return tab, lab


def levels_bytes(maps, F, lab, msgs_k):
    """alternative coding of a quantised message: the count map k plus the per-level count maps of all levels but the most frequent
    (k determines the last); ctx bytes of each integer map with the message's known mask"""
    nlev = int(lab.max()) + 1
    own = []
    for L in maps:
        kn = ~G.is_unk(L); li = np.where(kn, lab[np.clip(L.astype(np.int64) - LMIN, 0, len(lab) - 1)], -1)
        own.append(np.stack([(li == l) for l in range(nlev)]).astype(np.int32))
    tot = sum(o.sum((1, 2, 3)) for o in own); drop = int(np.argmax(tot))
    keep = [l for l in range(nlev) if l != drop]
    ownp = [(o[keep], (L != E.UNK)) for o, L in zip(own, maps)]
    msgs, _ = G.relay(ownp, F, G.comb_sum)
    b = 0.
    for e in F['directed']:
        V, K = msgs[e]
        for r in range(V.shape[0]):
            b += G.int_ctx_bytes(V[r], K)
        b += G.int_ctx_bytes(*msgs_k[e])
    return b


def collect_edges(F):
    """directed edges toward the root (smallest id) of each component"""
    up = set()
    for cm in F['comps']:
        root = min(cm); seen = {root}; st = [root]
        while st:
            u = st.pop()
            for w in F['nb'][u]:
                if w not in seen:
                    seen.add(w); st.append(w); up.add((w, u))
    return up


def job(fname):
    t0 = time.time()
    torch.set_num_threads(2)
    z = np.load(os.path.join(INP, 'export', fname))
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(E.edges_list(z['edges']), n)
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    K0T = [G.Target(private[a], H) for a in range(n)]
    maps = list(z['L_k0_cam_nf'])
    up = collect_edges(F)
    nets = {}
    for sd, half in (('342101', 'A'), ('342202', 'B')):
        ck, model = E.load_ckpt(os.path.join(INP, f's{sd}', 'final.pt'), 'cpu')
        thr = json.load(open(os.path.join(INP, f's{sd}', 'threshold.json')))
        nets[sd] = (half, model, int(thr['j']), int(thr['k0']['j_sep']))
    arms = [('342101', 'full'), ('342101', 'sign'), ('342101', 'q2'), ('342101', 'q3'), ('342101', 'q4'), ('342101', 'q8'),
            ('342202', 'full'), ('342202', 'sign'), ('342202', 'q3')]
    rec = dict(file=fname, n=n, comps=F['comps'], arms={})
    for sd, kind in arms:
        half, model, j, jk = nets[sd]
        tab, lab = quant_table(half, kind)
        _, _, _, theta = fit_counts(half)
        sums, msgs, msgn = E.relay_sums(maps, F, tab)
        lg = E.net_logits(model, [s[0] for s in sums], [s[1] for s in sums], 'cpu', 1)
        r8n, r8l, preds = [], [], []
        for ci, cm in enumerate(F['comps']):
            S, kc = sums[ci]; p = E.net_readout(lg[ci], kc, j); preds.append(p)
            r8n.append([G.score(p, R8T[ci]), len(cm)]); r8l.append([G.score(E.linear_readout(S, kc, theta), R8T[ci]), len(cm)])
        a = dict(table=None if kind == 'full' else sorted(set(tab.tolist())), R8_net=r8n, R8_lin=r8l)
        if sd == '342101':
            bs = {e: G.int_ctx_bytes(*msgs[e]) for e in F['directed']}
            bk = {e: G.int_ctx_bytes(*msgn[e]) for e in F['directed']}
            bt = {e: G.ctx_tern_bytes(np.where(msgs[e][1], (msgs[e][0] > 0).astype(np.int8), np.int8(-1))) for e in F['directed']}
            bm = {e: G.ctx_tern_bytes(np.where(msgs[e][1], np.int8(0), np.int8(-1))) for e in F['directed']}
            a['bytes'] = dict(sum=sum(bs.values()), count=sum(bk.values()), tern=sum(bt.values()), mask=sum(bm.values()),
                              up_sum=sum(v for e, v in bs.items() if e in up), up_count=sum(v for e, v in bk.items() if e in up))
            # downlink = the read-out ternary map of the component, once per distribute edge
            dn = 0.
            for e in F['directed']:
                if e not in up:
                    ci = F['comp_of'][e[0]]; dn += G.ctx_tern_bytes(preds[ci])
            a['bytes']['down_readout'] = dn
            if lab is not None and int(lab.max()) + 1 <= 4:
                a['bytes']['levels'] = levels_bytes(maps, F, lab, msgn)
            if kind in ('full', 'sign', 'q3'):
                own = [E.stalk(L, tab) for L in maps]
                a['bytes']['k0_own'] = float(sum(G.int_ctx_bytes(e, k) for e, k in own))
                lk = E.net_logits(model, [e for e, _ in own], [k.astype(np.int32) for _, k in own], 'cpu', 1)
                a['K0_net'] = [G.score(E.net_readout(lk[i], own[i][1].astype(np.int32), jk), K0T[i]) for i in range(n)]
        rec['arms'][f'{sd}:{kind}'] = a
    rec['seconds'] = time.time() - t0
    with open(os.path.join(OUT, fname.replace('.npz', '.json')), 'w') as fh:
        json.dump(rec, fh)
    return fname, rec['seconds']


if __name__ == '__main__':
    import multiprocessing as mp
    files = sys.argv[1].split(',') if len(sys.argv) > 1 and sys.argv[1] != 'val' else \
        sorted(f for f in os.listdir(os.path.join(INP, 'export')) if f.startswith('val_') and f.endswith('.npz'))
    files = [f for f in files if not os.path.exists(os.path.join(OUT, f.replace('.npz', '.json')))]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if workers <= 1:
        for f in files:
            print(job(f), flush=True)
    else:
        with mp.Pool(workers) as pool:
            for r in pool.imap_unordered(job, files):
                print(r, time.strftime('%H:%M:%S', time.gmtime()), flush=True)

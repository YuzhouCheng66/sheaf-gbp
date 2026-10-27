"""R-C probe 4 (report only): source-denoised posteriors quantised to 4 levels before the sum (zero training, no read-out network).
Each agent: d = NET logit on its own stalk - t_k0 (the checkpoint's K0 threshold, chosen on TRAIN hold-out); q = -3 / -1 / +1 / +3 for
d < -tau / [-tau, 0) / [0, tau) / >= tau, on its known voxels; relay sum of q; read-out sum q >= 0 (ties occupied, as VOTE1).
tau in {2, 4} nats, fixed before looking (both reported). Bytes: ctx coder on the q-sum messages and on own q maps."""
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJ, 'scripts', 'claude_v342'))
import numpy as np
import torch
import g4_e3r as E
G = E.G
INP = os.path.join(HERE, '..', 'inputs')
TAUS = (2.0, 4.0)


def run(npz, seed_tag):
    torch.set_num_threads(4)
    ck, model = E.load_ckpt(os.path.join(INP, f's{seed_tag}', 'final.pt'), 'cpu')
    thr = json.load(open(os.path.join(INP, f's{seed_tag}', 'threshold.json')))
    ev = E.Evidence.from_recipe(ck['recipe']['evidence'])
    tk = E.bin_edge(int(thr['k0']['j_sep']))
    z = np.load(npz)
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(E.edges_list(z['edges']), n)
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    L = list(z['L_k0_cam_nf'])
    own = [E.stalk(L[a], ev.table) for a in range(n)]
    lk = E.net_logits(model, [e for e, _ in own], [k.astype(np.int32) for _, k in own], 'cpu', 2)
    shape = L[0].shape
    rec = dict(file=os.path.basename(npz), seed_tag=seed_tag, tk=tk, arms={}, kf={}, bytes={}, own_bytes={})
    for tau in TAUS:
        arm = f'QPOST4_t{int(tau)}'
        qs = []
        for a in range(n):
            k = own[a][1]; d = lk[a] - tk
            q = np.where(d < -tau, -3, np.where(d < 0, -1, np.where(d < tau, 1, 3))).astype(np.int32)
            qs.append((np.where(k, q, 0).astype(np.int32), k))
        qm, _ = G.relay(qs, F, G.comb_sum)
        rows = []; kf = np.zeros((6, 3), np.int64)
        for ci, cm in enumerate(F['comps']):
            V = np.zeros(shape, np.int32); kc = np.zeros(shape, np.int32)
            for a in cm:
                V += qs[a][0]; kc += qs[a][1]
            p = np.where(kc > 0, (V >= 0).astype(np.int8), np.int8(-1)).astype(np.int8)
            r = G.score(p, R8T[ci])
            rows.append(dict(tp=r['furn_tp'], fp=r['furn_fp'], fn=r['furn_fn'], free=r['free_recall'], surf=r['surface_f1'], m=len(cm),
                             nfree=int(R8T[ci].free.sum()), ftol=r['furn_f1_tol1']))
            kf += np.asarray(E.k_strata(p, R8T[ci], H, kc)['k_furn'], np.int64) * len(cm)
        rec['arms'][arm] = rows; rec['kf'][arm] = kf.tolist()
        rec['bytes'][f'q4t{int(tau)}'] = float(sum(G.int_ctx_bytes(*qm[e]) for e in F['directed']))
        rec['own_bytes'][f'q4t{int(tau)}'] = float(np.median([G.int_ctx_bytes(q, k) for q, k in qs]))
    return rec


if __name__ == '__main__':
    seed_tag = sys.argv[1]; files = sys.argv[2:]
    out = os.path.join(HERE, 'out4'); os.makedirs(out, exist_ok=True)
    for f in files:
        t0 = time.time(); p = os.path.join(out, f'{seed_tag}_{os.path.basename(f)[:-4]}.json')
        if os.path.exists(p):
            continue
        rec = run(f, seed_tag); rec['seconds'] = time.time() - t0
        with open(p + '.tmp', 'w') as fh:
            json.dump(rec, fh)
        os.replace(p + '.tmp', p)
        print(f'{os.path.basename(f)} {rec["seconds"]:.1f}s', flush=True)

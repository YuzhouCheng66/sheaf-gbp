"""R-C probe (report only, not a verdict batch): where should the ONE learned prior live, and what happens if it is counted more than once?
cam_nf, R8 components, frozen E3-R checkpoint (no retraining). Imports frozen g4_e3r / g4_noise_panels read-only.
Arms per component (all share the known mask k > 0):
  LIN      B6 linear read-out of the summed calibrated stalks                     (prior once, no structure prior)
  NETR     E3-R: network on the fused (S, k)                                        (structure prior once, at the read-out)
  POSTSUM  sum over agents of the network's single-agent logits (each agent denoised at the source, posteriors summed) >= t
           (prior counted k times)
  EPSITE   sum over agents of (single-agent logit - prior-only logit) + prior-only logit >= t
           ("posterior minus prior" sites; prior once linearly, completions counted per agent)
  VOTE0/1  each agent's own NET map as a hard vote (+1 / -1), sum of votes > 0 (ties free) / >= 0 (ties occupied)
           (decision fusion; source-denoised hard map = what a W3 box code of the denoised map would carry)
  NETVOTE  the network applied to (vote-sum x theta) as if it were S (mismatched input; shows why the read-out must be trained on its stalk)
Bytes (ctx coder, collect-distribute, report only): B6 sum channel, count channel, vote-sum channel; own three-state maps raw vs denoised.
"""
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJ, 'scripts', 'claude_v342'))
import numpy as np
import g4_e3r as E
G = E.G
import torch

INP = os.path.join(HERE, '..', 'inputs')
ARMS = ['LIN', 'NETR', 'POSTSUM', 'EPSITE', 'VOTE0', 'VOTE1', 'NETVOTE']


def run(npz, seed_tag, threads):
    torch.set_num_threads(threads)
    ck, model = E.load_ckpt(os.path.join(INP, f's{seed_tag}', 'final.pt'), 'cpu')
    thr = json.load(open(os.path.join(INP, f's{seed_tag}', 'threshold.json')))
    ev = E.Evidence.from_recipe(ck['recipe']['evidence'])
    j = int(thr['j']); t = E.bin_edge(j); jk = int(thr['k0']['j_sep'])
    z = np.load(npz)
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(E.edges_list(z['edges']), n)
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    tn = 'cam_nf'
    L = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z else list(z[f'L_k0_{tn}'])
    sums, msgs, msgn = E.relay_sums(L, F, ev.table)
    own = [E.stalk(L[a], ev.table) for a in range(n)]
    lk = E.net_logits(model, [e for e, _ in own], [k.astype(np.int32) for _, k in own], 'cpu', 2)
    shape = L[0].shape
    l0 = E.net_logits(model, [np.zeros(shape, np.int32)], [np.zeros(shape, np.int32)], 'cpu', 1)[0]
    lg = E.net_logits(model, [s[0] for s in sums], [s[1] for s in sums], 'cpu', 2)
    votes = []
    for a in range(n):
        e, k = own[a]; ki = k.astype(np.int32)
        p = E.net_readout(lk[a], ki, jk)
        votes.append((np.where(p == 1, 1, np.where(p == 0, -1, 0)).astype(np.int32), k))
    vmsgs, vbel = G.relay(votes, F, G.comb_sum)
    vsum, lgv_in = [], []
    for ci, cm in enumerate(F['comps']):
        V = np.zeros(shape, np.int32)
        for a in cm:
            V += votes[a][0]
        vsum.append(V)
        lgv_in.append(V * int(max(ev.theta, 1)))
    lgv = E.net_logits(model, lgv_in, [s[1] for s in sums], 'cpu', 2)
    rec = dict(file=os.path.basename(npz), seed_tag=seed_tag, n=n, comps=F['comps'], t=t, theta=ev.theta, arms={}, kf={})
    for arm in ARMS:
        rows = []; kf = np.zeros((6, 3), np.int64)
        for ci, cm in enumerate(F['comps']):
            S, kc = sums[ci]
            if arm == 'LIN':
                p = E.linear_readout(S, kc, ev.theta)
            elif arm == 'NETR':
                p = E.net_readout(lg[ci], kc, j)
            elif arm in ('POSTSUM', 'EPSITE'):
                acc = np.zeros(shape, np.float64)
                for a in cm:
                    k = own[a][1]
                    acc[k] += (lk[a][k] if arm == 'POSTSUM' else (lk[a][k] - l0[k]))
                if arm == 'EPSITE':
                    acc = acc + l0
                p = np.where(kc > 0, (acc >= t).astype(np.int8), np.int8(-1)).astype(np.int8)
            elif arm == 'VOTE0':
                p = np.where(kc > 0, (vsum[ci] > 0).astype(np.int8), np.int8(-1)).astype(np.int8)
            elif arm == 'VOTE1':
                p = np.where(kc > 0, (vsum[ci] >= 0).astype(np.int8), np.int8(-1)).astype(np.int8)
            else:
                p = E.net_readout(lgv[ci], kc, j)
            r = G.score(p, R8T[ci])
            rows.append(dict(tp=r['furn_tp'], fp=r['furn_fp'], fn=r['furn_fn'], free=r['free_recall'], surf=r['surface_f1'], m=len(cm),
                             nfree=int(R8T[ci].free.sum()), ftol=r['furn_f1_tol1']))
            kf += np.asarray(E.k_strata(p, R8T[ci], H, kc)['k_furn'], np.int64) * len(cm)
        rec['arms'][arm] = rows; rec['kf'][arm] = kf.tolist()
    D = F['directed']
    rec['bytes'] = dict(sum=float(sum(G.int_ctx_bytes(*msgs[e]) for e in D)), count=float(sum(G.int_ctx_bytes(*msgn[e]) for e in D)),
                        vote=float(sum(G.int_ctx_bytes(*vmsgs[e]) for e in D)), n_directed=len(D))
    raw_t = [G.tern(L[a]) for a in range(n)]
    den_t = [np.where(votes[a][1], (votes[a][0] > 0).astype(np.int8), np.int8(-1)).astype(np.int8) for a in range(n)]
    clean_t = [np.asarray(private[a], np.int8) for a in range(n)]
    rec['own_bytes'] = dict(raw=float(np.median([G.ctx_tern_bytes(x) for x in raw_t])), den=float(np.median([G.ctx_tern_bytes(x) for x in den_t])),
                            clean=float(np.median([G.ctx_tern_bytes(x) for x in clean_t])))
    return rec


if __name__ == '__main__':
    seed_tag = sys.argv[1]; threads = int(sys.argv[2]); files = sys.argv[3:]
    out = os.path.join(HERE, 'out'); os.makedirs(out, exist_ok=True)
    for f in files:
        t0 = time.time()
        p = os.path.join(out, f'{seed_tag}_{os.path.basename(f)[:-4]}.json')
        if os.path.exists(p):
            continue
        rec = run(f, seed_tag, threads); rec['seconds'] = time.time() - t0
        with open(p + '.tmp', 'w') as fh:
            json.dump(rec, fh)
        os.replace(p + '.tmp', p)
        print(f'{os.path.basename(f)} {rec["seconds"]:.1f}s', flush=True)

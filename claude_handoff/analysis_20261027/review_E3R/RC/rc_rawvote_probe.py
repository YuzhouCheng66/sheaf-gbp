"""R-C probe 2 (report only): raw (source-learning-free) votes. Each agent's calibrated stalk e = c(L) is thresholded at the K0 linear
theta of the checkpoint's evidence table (from threshold.json, chosen on TRAIN hold-out), giving a vote +1 / -1 on its known voxels.
Fused by the same exclusion relay (sum of votes), read out by majority (> 0 ties free, >= 0 ties occupied). No network anywhere.
Bytes: ctx coder on vote-sum messages (collect-distribute) and on own vote maps."""
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


def run(npz, seed_tag):
    ck = torch.load(os.path.join(INP, f's{seed_tag}', 'final.pt'), map_location='cpu', weights_only=False)
    thr = json.load(open(os.path.join(INP, f's{seed_tag}', 'threshold.json')))
    ev = E.Evidence.from_recipe(ck['recipe']['evidence'])
    thk = int(thr['k0']['linear_theta_sep'])
    z = np.load(npz)
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(E.edges_list(z['edges']), n)
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    tn = 'cam_nf'
    L = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z else list(z[f'L_k0_{tn}'])
    own = [E.stalk(L[a], ev.table) for a in range(n)]
    ck_, model = E.load_ckpt(os.path.join(INP, f's{seed_tag}', 'final.pt'), 'cpu'); torch.set_num_threads(4)
    j = int(thr['j']); t = E.bin_edge(j); l0 = ev.l0q * E.UNIT
    lk = E.net_logits(model, [e for e, _ in own], [k.astype(np.int32) for _, k in own], 'cpu', 2)
    z0 = E.net_logits(model, [np.zeros(L[0].shape, np.int32)], [np.zeros(L[0].shape, np.int32)], 'cpu', 1)[0]
    votes = [(np.where(k, np.where(e > thk, 1, -1), 0).astype(np.int32), k) for e, k in own]
    vmsgs, _ = G.relay(votes, F, G.comb_sum)
    shape = L[0].shape
    rec = dict(file=os.path.basename(npz), seed_tag=seed_tag, thk=thk, l0=l0, net0_mean=float(np.mean(z0)), net0_p05_p95=[float(np.percentile(z0, 5)), float(np.percentile(z0, 95))], arms={}, kf={})
    for arm in ('RVOTE0', 'RVOTE1', 'EPSITE_L0'):
        rows = []; kf = np.zeros((6, 3), np.int64)
        for ci, cm in enumerate(F['comps']):
            V = np.zeros(shape, np.int32); kc = np.zeros(shape, np.int32)
            for a in cm:
                V += votes[a][0]; kc += votes[a][1]
            if arm == 'EPSITE_L0':
                acc = np.full(shape, l0, np.float64)
                for a in cm:
                    k = own[a][1]; acc[k] += lk[a][k] - l0
                p = np.where(kc > 0, (acc >= t).astype(np.int8), np.int8(-1)).astype(np.int8)
            else:
                p = np.where(kc > 0, ((V > 0) if arm == 'RVOTE0' else (V >= 0)).astype(np.int8), np.int8(-1)).astype(np.int8)
            r = G.score(p, R8T[ci])
            rows.append(dict(tp=r['furn_tp'], fp=r['furn_fp'], fn=r['furn_fn'], free=r['free_recall'], surf=r['surface_f1'], m=len(cm),
                             nfree=int(R8T[ci].free.sum()), ftol=r['furn_f1_tol1']))
            kf += np.asarray(E.k_strata(p, R8T[ci], H, kc)['k_furn'], np.int64) * len(cm)
        rec['arms'][arm] = rows; rec['kf'][arm] = kf.tolist()
    D = F['directed']
    rec['bytes'] = dict(rvote=float(sum(G.int_ctx_bytes(*vmsgs[e]) for e in D)))
    rv_t = [np.where(k, (v > 0).astype(np.int8), np.int8(-1)).astype(np.int8) for v, k in votes]
    rec['own_bytes'] = dict(rvote=float(np.median([G.ctx_tern_bytes(x) for x in rv_t])))
    return rec


if __name__ == '__main__':
    seed_tag = sys.argv[1]; files = sys.argv[2:]
    out = os.path.join(HERE, 'out2'); os.makedirs(out, exist_ok=True)
    for f in files:
        t0 = time.time(); p = os.path.join(out, f'{seed_tag}_{os.path.basename(f)[:-4]}.json')
        if os.path.exists(p):
            continue
        rec = run(f, seed_tag); rec['seconds'] = time.time() - t0
        with open(p + '.tmp', 'w') as fh:
            json.dump(rec, fh)
        os.replace(p + '.tmp', p)
        print(f'{os.path.basename(f)} {rec["seconds"]:.1f}s', flush=True)

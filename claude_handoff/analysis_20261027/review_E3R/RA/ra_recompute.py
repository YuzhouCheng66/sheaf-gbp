"""R-A independent recompute of E3-R (review only; CPU; reads inputs/ only, writes RA/out_recompute/).
Independent pieces (own code): components from edges (union-find), evidence lookup cq[L+40], direct component sums, features,
threshold binning, region / target / furniture F1 / free recall.  Shared pieces: UNet3 architecture (import, to load the weights) and,
as a cross-check only, G.score on the same predictions.
Usage: python -B ra_recompute.py --houses val_r1_i0_s1,... [--tiers cam_nf,cam] [--threads 2]
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
import argparse, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
INP = os.path.join(HERE, '..', 'inputs')
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch  # noqa: E402
import g4_e3r as E  # noqa: E402   (UNet3 + G.score cross-check only)
G = E.G
UNK = -128; LMIN = -40


def comps_from_edges(edges, n):
    par = list(range(n))
    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i, j, _ in np.asarray(edges).reshape(-1, 3):
        par[f(int(i))] = f(int(j))
    groups = {}
    for a in range(n):
        groups.setdefault(f(a), []).append(a)
    return sorted(sorted(g) for g in groups.values())


def dil6(m):
    o = m.copy()
    o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]; o[:, :, 1:] |= m[:, :, :-1]; o[:, :, :-1] |= m[:, :, 1:]
    return o


def evid(L, cq):
    k = L != UNK
    e = np.zeros(L.shape, np.int32); e[k] = np.asarray(cq, np.int64)[L[k].astype(np.int64) - LMIN]
    return e, k


def feats(S, kc):
    Z = S.shape[-1]; z = ((np.arange(Z) + 0.5) / Z * 2 - 1).astype(np.float32)
    f = np.stack([S.astype(np.float32) * np.float32(0.05 / 4.0), np.log1p(kc.astype(np.float32)), np.broadcast_to(z, S.shape).astype(np.float32)])
    return torch.from_numpy(f[None])


@torch.no_grad()
def logits(model, S, kc):
    return model(feats(S, kc))[0].numpy()


def occ_from_logit(lg, kc, j):
    b = np.clip(np.floor((lg.astype(np.float64) + 40.0) / 0.02), 0, 3999).astype(np.int64)
    return (kc > 0) & (b >= j)


def furn(pred_occ, known_pred, t, region):
    known = t >= 0; tr = (t == 1) & region
    pr = pred_occ & known & region
    tp = int((pr & tr).sum()); fp = int((pr & ~tr).sum()); fn = int((~pr & tr).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None
    free = t == 0
    fr = float(((~pred_occ) & known_pred & free).sum()) / max(int(free.sum()), 1)
    return f1, tp, fp, fn, fr


def load_models():
    out = {}
    for sd, half in (('342101', 'A'), ('342202', 'B')):
        ck = torch.load(os.path.join(INP, 's' + sd, 'final.pt'), map_location='cpu', weights_only=False)
        r = ck['recipe']; m = E.UNet3(r['cin'], r['width'], skip=r.get('skip', False), bias0=r.get('bias0', 0.0)); m.load_state_dict(ck['state_dict']); m.eval()
        thr = json.load(open(os.path.join(INP, 's' + sd, 'threshold.json')))
        out[sd] = dict(model=m, j=int(thr['j']), jk=int(thr['k0']['j_sep']), thk=int(thr['k0']['linear_theta_sep']), half=half,
                       recipe_inputs=r.get('inputs'), recipe_table=r['evidence']['table'], recipe_theta=r['evidence']['theta'])
    return out


def run(name, tiers, models, fit, do_k0=True):
    z = np.load(os.path.join(INP, 'export', name + '.npz'))
    occ = z['occ']; slab = z['slab'].astype(bool); obj = z['obj'].astype(bool); private = z['private']; n = private.shape[0]
    region = ~slab & ~dil6((occ == 1) & ~obj & ~slab)
    comps = comps_from_edges(z['edges'], n)
    H = G.House(occ, slab, obj, z['cell'])
    rec = dict(house=name, comps=comps, tiers={})
    for tn in tiers:
        Lk0 = z[f'L_k0_{tn}']; Lpub = z[f'L_pub_{tn}'] if f'L_pub_{tn}' in z.files else Lk0
        T = {}
        for sd, M in models.items():
            tb = fit['halves'][M['half']]['tiers']['cam_nf']; cq = tb['cq']; th = int(tb['theta'])
            assert list(cq) == list(M['recipe_table']) and th == M['recipe_theta'], 'checkpoint evidence table != e1c_fit half'
            rows_net, rows_lin, rows_x = [], [], []
            for cm in comps:
                S = np.zeros(occ.shape, np.int32); kc = np.zeros(occ.shape, np.int32)
                for a in cm:
                    e, k = evid(Lpub[a], cq); S += e; kc += k
                t = private[cm].max(0)
                lg = logits(M['model'], S, kc)
                pn = occ_from_logit(lg, kc, M['j']); pl = (kc > 0) & (S > th)
                fn_ = furn(pn, kc > 0, t, region); fl = furn(pl, kc > 0, t, region)
                # cross-check with the project's scorer on the same prediction
                predn = np.where(kc > 0, pn.astype(np.int8), np.int8(-1)).astype(np.int8)
                gs = G.score(predn, G.Target(t, H, mult=(private[cm] >= 0).sum(0)))
                rows_net.append([fn_[0], len(cm), fn_[4]]); rows_lin.append([fl[0], len(cm), fl[4]])
                rows_x.append(dict(own=fn_[0], G=gs['furn_f1'], own_free=fn_[4], G_free=gs['free_recall']))
            rec_t = dict(R8_net=rows_net, R8_lin=rows_lin, xcheck=rows_x)
            if do_k0 and tn != 'clean':
                k0n, k0l = [], []
                for a in range(n):
                    e, k = evid(Lk0[a], cq); kc = k.astype(np.int32); t = private[a]
                    lg = logits(M['model'], e, kc)
                    k0n.append(furn(occ_from_logit(lg, kc, M['jk']), kc > 0, t, region)[0])
                    k0l.append(furn(k & (e > M['thk']), kc > 0, t, region)[0])
                rec_t['K0_net_sep'] = k0n; rec_t['K0_lin_sep'] = k0l
            T[sd] = rec_t
        rec['tiers'][tn] = T
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--houses', required=True); ap.add_argument('--tiers', default='cam_nf,cam')
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--no-k0', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    fit = json.load(open(os.path.join(INP, 'e1c_fit.json')))
    models = load_models()
    od = os.path.join(HERE, 'out_recompute'); os.makedirs(od, exist_ok=True)
    for h in a.houses.split(','):
        p = os.path.join(od, h + '.json')
        if os.path.exists(p):
            continue
        t0 = time.time(); r = run(h, a.tiers.split(','), models, fit, not a.no_k0); r['seconds'] = time.time() - t0
        json.dump(r, open(p, 'w'))
        print(h, f"{r['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    main()

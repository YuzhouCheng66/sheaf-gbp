"""R-J probe 6 = probe 5 on the CAM tier (L_pub_cam, unregistered, as in E3-L3's cam report), variants WL (learned w, = E3-L3 as run) and W4.
R-J probe 5 = probe 4 with FINITE w instead of clipping: L.gabp(sources, Fl, w_fixed, 'round') with the E3-L3 per-hop
integer quantisation unchanged (step 1), w in {8, 4, 2} (the learned w is ~1240). Exact GaBP of the user's quadratic energy at that w
up to the preregistered message rounding. Frozen decoder / threshold, VAL diagnostic only.
Original probe-4 docstring follows. R-J probe 4 (report only, CPU, VAL diagnostic, NOT a verdict; thresholds are the frozen TRAIN-holdout ones): does the frozen
E3-L3 decoder need the exact footprint COUNT in the GaBP messages, or only the known mask + the mean vote?
Variants (per hop, collect-distribute in Fl['order'] exactly like L.gabp; sources (z_i, 1[tau_i]) unchanged; known mask exact):
  ACT  L.gabp(..., w, 'round')  (E3-L3 as run: integer subtree sums (h, k))
  Cc   message precision clipped: lam_q = min(lam~, c), h_q = round(h~ * lam_q / lam~) (mean-preserving, integer), c in {1, 2, 3}
       (c = 1: the message is the subtree's rounded mean vote on its support -- a 'vote relay')
Readout = frozen decoder on (x = h_b / lam_b, log1p lam_b), frozen threshold j_occ, known = lam_b > 0 (== k > 0 asserted),
R.scored vs the component R8 target, one row per agent (E3-R component-expanded mean). Bytes = L.wire_bytes per directed message."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import json, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch
import g4_e3l3 as E
L = E.L; R = E.R; G = E.G
torch.set_num_threads(4)
INP = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'review_E3R', 'inputs')
UNK = E.UNK


def clip_gabp(h, lam, Fl, c):
    nb = Fl['nb']; msgs = {}
    for s, r in Fl['order']:
        lt = lam[s].clone(); ht = h[s].clone()
        for k in nb[s]:
            if k != r:
                lt = lt + msgs[(k, s)][1]; ht = ht + msgs[(k, s)][0]
        lq = torch.clamp(lt, max=float(c))
        hq = torch.where(lt > 0, torch.round(ht * (lq / torch.where(lt > 0, lt, torch.ones_like(lt))).unsqueeze(0)), torch.zeros_like(ht))
        msgs[(s, r)] = (hq, lq)
    hb, lb = [], []
    for a in range(lam.shape[0]):
        l = lam[a].clone(); hh = h[a].clone()
        for k in nb[a]:
            l = l + msgs[(k, a)][1]; hh = hh + msgs[(k, a)][0]
        hb.append(hh); lb.append(l)
    return torch.stack(hb), torch.stack(lb), msgs


def house(path, model, w, table_t, jo):
    z = np.load(path); private = z['private']; n = private.shape[0]
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell'])
    F = G.forest(R.edges_list(z['edges']), n)
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    Lp = list(z['L_pub_cam'])
    out = {}
    with torch.no_grad():
        zp, lp = L.encode_maps(model, Lp, table_t, 'cpu')
        hp, lpq, zq = E.source(zp, lp, 'round')
        for var in ("WL", "W8", "W4"):
            rows = []; by = 0.0
            for ci, cm in enumerate(F['comps']):
                Fl = L.local_forest(F, cm)
                if var[0] == 'W':
                    hb, lb, msgs = L.gabp(hp[cm], lpq[cm], Fl, (w if var == 'WL' else float(var[1:])), 'round')
                else:
                    hb, lb, msgs = clip_gabp(hp[cm], lpq[cm], Fl, int(var[1]))
                for e in Fl['directed']:
                    hi = msgs[e][0].round().to(torch.int64).numpy(); li = msgs[e][1].round().to(torch.int64).numpy()
                    by += L.wire_bytes(hi, li)
                lg = L.decode_np(model, hb, lb, 2)
                kc = sum((Lp[a] != UNK).astype(np.int32) for a in cm); lbn = lb.numpy()
                for i in range(len(cm)):
                    assert np.array_equal(lbn[i] > 0, kc > 0)
                    p = R.net_readout(lg[i], kc, jo)
                    rows.append(R.scored(p, R8T[ci], H, kc)['furn_f1'])
            out[var] = dict(furn=float(np.mean(rows)), bytes=by)
    return out


def main():
    seed = sys.argv[1]; noise = sys.argv[2]; idx = [int(x) for x in sys.argv[3].split(',')]
    ck, model = E.load_model(os.path.join(INP, f'e3l3_s{seed}', 'final.pt'), 'cpu')
    thr = json.load(open(os.path.join(INP, f'e3l3_s{seed}', 'threshold.json'))); jo = thr['R8']['j_occ']
    w = float(model.w.detach()); table_t = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
    files = sorted(f for f in os.listdir(os.path.join(INP, 'export')) if f.startswith('val_') and f.endswith(f'_s{noise}.npz'))
    files = [files[i] for i in idx]
    res = {}; t0 = time.time(); op = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'rj_wcam2_{seed}_s{noise}.json')
    for f in files:
        res[f] = house(os.path.join(INP, 'export', f), model, w, table_t, jo)
        print(f, json.dumps({k: {a: round(b, 4) for a, b in v.items()} for k, v in res[f].items()}), f'{time.time() - t0:.0f}s', flush=True)
        with open(op, 'w') as fh:
            json.dump(dict(seed=seed, noise=noise, w=w, j_occ=jo, houses=res), fh, indent=1)


if __name__ == '__main__':
    main()

"""rh_readout.py (R-H, 2026-10-28): REPORT-ONLY protocol variant "collect (exact GaBP sums) + distribute the decoded readout".
With w ~ 1,240-1,360 every quantised message is the exact side sum, so every member of a component holds the SAME belief (K, S)
(asserted here), decodes it with the same decoder and the same threshold -> the same ternary readout R8 (= R.net_readout).
Variant: the root (here the second-message senders) sends that readout instead of the belief complement. The scored R8
output at every agent is bit-identical (same function of the same input); the agents no longer hold x = S / K.
Bytes per second (down) message: G.ctx_tern_bytes(readout) (= the O7 mask coder on a ternary map) and a conditional variant
(+ the receiver's own partial-belief mask k' > 0 as one more context bit). CPU only.
usage: python -B rh_readout.py --export X --ckpts A/final.pt,B/final.pt --out O.json [--noise 1] [--workers 6]
"""
import os
import sys

sys.dont_write_bytecode = True
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, glob, json, math, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402
from scipy.special import gammaln  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_e3l3 as E  # noqa: E402  (frozen, import only)

L = E.L; R = E.R; G = E.G
import torch  # noqa: E402


def tern_bits_side(t, side):
    """G.ctx_tern_bits with one more context bit (side, 0 / 1, known to the decoder)"""
    t = np.asarray(t); X, Y, Z = t.shape
    s = (t.astype(np.int16) + 1).astype(np.int64)
    c = np.zeros(t.shape, np.int64)
    for dx, dy, dz in G.NB:
        c *= 3; c[dx:, dy:, dz:] += s[:X - dx, :Y - dy, :Z - dz]
    c = c * 2 + np.asarray(side, np.int64)
    c = c * 3 + s
    cnt = np.bincount(c.ravel()).astype(np.float64)
    cnt = np.pad(cnt, (0, (-len(cnt)) % 3)).reshape(-1, 3)
    N = cnt.sum(1); u = N > 0; cnt, N = cnt[u], N[u]
    return float(((gammaln(N + 1.5) - gammaln(1.5)) - (gammaln(cnt + .5) - gammaln(.5)).sum(1)).sum() / math.log(2.0))


def job(args):
    path, ckpt = args
    rec = dict(file=os.path.basename(path), ckpt=ckpt)
    try:
        torch.set_num_threads(1); t0 = time.time()
        ck, model = E.load_model(ckpt, 'cpu'); w = float(model.w.detach()); rec['seed'] = str(ck['seed'])
        thr = json.load(open(os.path.join(os.path.dirname(ckpt), 'threshold.json'))); jo = thr['R8']['j_occ']
        tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
        z = np.load(path); n = z['private'].shape[0]
        F = G.forest(R.edges_list(z['edges']), n)
        L_pub = list(z['L_k0_cam_nf'])
        with torch.no_grad():
            zp, lp = L.encode_maps(model, L_pub, tb, 'cpu')
            hp, lpq, zq = E.source(zp, lp, 'round')
        rows = []; same = True; occ = 0
        for cm in F['comps']:
            Fl = L.local_forest(F, cm)
            with torch.no_grad():
                hb, lb, msgs = L.gabp(hp[cm], lpq[cm], Fl, w, 'round')
            same &= bool(all(torch.equal(hb[0], hb[i]) and torch.equal(lb[0], lb[i]) for i in range(len(cm))))
            lg = L.decode_np(model, hb[:1], lb[:1], 1)[0]
            kc = sum((L_pub[a] != R.UNK).astype(np.int32) for a in cm)
            p = R.net_readout(lg, kc, jo); occ += int((p == 1).sum())
            b_plain = G.ctx_tern_bytes(p)
            order = sorted(Fl['directed'], key=lambda e: (len(Fl['sides'][e]), e)); pos = {e: i for i, e in enumerate(order)}
            for e in Fl['directed']:
                rv = (e[1], e[0])
                if pos[rv] < pos[e]:                                          # second message on the edge
                    side = [cm[a] for a in Fl['sides'][rv]]                  # the receiver's side
                    kr = sum((L_pub[a] != R.UNK).astype(np.int32) for a in side)
                    rows.append(dict(s=int(cm[e[0]]), r=int(cm[e[1]]), plain=b_plain, cond=tern_bits_side(p, kr > 0) / 8 + 2))
        rec.update(same_belief=same, down=rows, readout_occ=occ, seconds=time.time() - t0)
    except Exception as ex:
        rec['failed'] = True; rec['error'] = repr(ex)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', required=True); ap.add_argument('--ckpts', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--noise', type=int, default=1); ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args(); assert a.workers <= 8
    files = sorted(glob.glob(os.path.join(a.export, f'val_r*_i*_s{a.noise}.npz')))
    jobs = [(f, c) for f in files for c in a.ckpts.split(',')]
    out = []; t0 = time.time()
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r)
            print(r['file'], r.get('seed'), 'FAILED ' + r['error'] if r.get('failed') else f"same {r['same_belief']} {r['seconds']:.0f}s", f'{time.time() - t0:.0f}s', flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

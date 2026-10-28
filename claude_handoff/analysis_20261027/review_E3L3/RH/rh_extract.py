"""rh_extract.py (R-H, 2026-10-28): rebuild every directed E3-L3 GaBP message on VAL noise seed S (cam_nf) for both E3-L3
checkpoints, with the frozen g4_e3l3 / g4_e3l functions (import only), and reproduce the O7 wire bytes per house.
Per (house, ckpt) it writes:
  <cache>/<tag>_<seed>.npz   code int8 [n, X, Y, Z] = tau * (z + 2)  (0 off footprint; 1 / 2 / 3 = z -1 / 0 / +1)   (scratchpad only)
  <out>/jobs/<tag>_<seed>.json  O7 bytes per directed edge, split mask / k values / s values, + exact-sum assertions
CPU only (CUDA_VISIBLE_DEVICES= ), <= 8 processes.
usage: python -B rh_extract.py --export X --ckpts A/final.pt,B/final.pt --cache C --out O [--noise 1] [--houses all|N] [--workers 8]
"""
import os
import sys

sys.dont_write_bytecode = True
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, glob, json, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_e3l3 as E  # noqa: E402  (frozen, import only)

L = E.L; R = E.R; G = E.G
import torch  # noqa: E402


def job(args):
    path, ckpt, cache, out, tier = args
    tag = os.path.basename(path)[:-4]
    rec = dict(file=os.path.basename(path), ckpt=ckpt)
    try:
        torch.set_num_threads(1)
        t0 = time.time()
        ck, model = E.load_model(ckpt, 'cpu'); w = float(model.w.detach()); sd = str(ck['seed'])
        tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
        z = np.load(path); n = z['private'].shape[0]
        F = G.forest(R.edges_list(z['edges']), n)
        L_pub = list(z[f'L_pub_{tier}']) if f'L_pub_{tier}' in z.files else list(z[f'L_k0_{tier}'])
        with torch.no_grad():
            zp, lp = L.encode_maps(model, L_pub, tb, 'cpu')                 # = eval_arm
            hp, lpq, zq = E.source(zp, lp, 'round')
        tau = (lpq.numpy() > 0); zi = np.rint(zq[:, 0].numpy() if zq.dim() == 5 else zq.numpy()).astype(np.int8)
        zi = np.where(tau, zi, 0).astype(np.int8)
        assert np.array_equal(tau, np.stack([m != R.UNK for m in L_pub])), 'footprint != touched'
        code = (tau.astype(np.int8) * (zi + 2)).astype(np.int8)
        np.savez_compressed(os.path.join(cache, f'{tag}_{sd}.npz'), code=code)
        edges = []; tot = 0.0; exact = True
        for ci, cm in enumerate(F['comps']):
            Fl = L.local_forest(F, cm)
            with torch.no_grad():
                hb, lb, msgs = L.gabp(hp[cm], lpq[cm], Fl, w, 'round')
            for e in Fl['directed']:
                hi, li = L.msg_ints(*msgs[e])
                side = [cm[a] for a in Fl['sides'][e]] if 'sides' in Fl else None
                kn = li > 0
                mb = L.mask_bytes(kn); kb = float(L.value_bytes_given_mask(li, kn)); sb = float(L.value_bytes_given_mask(hi[0], kn))
                wb = L.wire_bytes(hi, li); tot += wb
                assert abs(wb - (mb + kb + sb)) < 1e-6
                ks = tau[side].sum(0); ss = zi[side].astype(np.int32).sum(0)
                ok = bool(np.array_equal(li, ks) and np.array_equal(hi[0], ss)); exact &= ok
                edges.append(dict(s=int(cm[e[0]]), r=int(cm[e[1]]), side=[int(a) for a in side], wire=wb, mask=mb, k=kb, s_ch=sb,
                                  support=int(kn.sum()), exact_sum=ok))
        rec.update(seed=sd, w=w, n=n, comps=[list(map(int, c)) for c in F['comps']], und=[list(map(int, e)) for e in F['und']], edges=edges,
                   o7_bytes=tot, exact_sums=exact, footprint=[int(t.sum()) for t in tau],
                   self_bytes=[L.wire_bytes(*L.msg_ints(hp[a], lpq[a])) for a in range(n)], seconds=time.time() - t0)
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    os.makedirs(os.path.join(out, 'jobs'), exist_ok=True)
    with open(os.path.join(out, 'jobs', f"{tag}_{rec.get('seed', 'x')}.json"), 'w') as fh:
        json.dump(rec, fh, indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', required=True); ap.add_argument('--ckpts', required=True); ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--noise', type=int, default=1); ap.add_argument('--tier', default='cam_nf')
    ap.add_argument('--houses', default='all'); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    assert a.workers <= 8
    os.makedirs(a.cache, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.export, f'val_r*_i*_s{a.noise}.npz')))
    assert not any('test' in os.path.basename(f) for f in files)
    if a.houses != 'all':
        files = files[:int(a.houses)]
    cks = a.ckpts.split(',')
    for c in cks:
        thr = json.load(open(os.path.join(os.path.dirname(c), 'threshold.json')))
        sha = G.file_sha256(c)
        print(c, 'sha', sha[:12], 'threshold ckpt sha', thr.get('ckpt_sha256', '')[:12], 'match', sha == thr.get('ckpt_sha256'), flush=True)
    jobs = [(f, c, a.cache, a.out, a.tier) for f in files for c in cks]
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            print(r['file'], r.get('seed'), 'FAILED ' + r['error'] if r.get('failed') else f"o7 {r['o7_bytes']:.0f} exact {r['exact_sums']} {r['seconds']:.0f}s",
                  f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

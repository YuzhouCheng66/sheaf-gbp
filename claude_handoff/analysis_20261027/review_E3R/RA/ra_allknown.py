"""R-A probe (report only): furniture F1 of NET vs B6 scored on EVERY noisy-known voxel (k > 0) in the furniture region against the
generator's occupancy, instead of only inside the clean-known scope (the E1 / E3-R target). Catches occupancy the network adds where
the clean sensor saw nothing (e.g. box interiors, behind surfaces). VAL cam_nf, R8.  Usage: python -B ra_allknown.py --jobs ...
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
import argparse, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
INP = os.path.join(HERE, '..', 'inputs')
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch  # noqa: E402
import g4_e3r as E  # noqa: E402
G = E.G

ap = argparse.ArgumentParser(); ap.add_argument('--jobs', required=True); a = ap.parse_args(); torch.set_num_threads(2)
ck, model = E.load_ckpt(os.path.join(INP, 's342101', 'final.pt'), 'cpu')
ev = E.Evidence.from_recipe(ck['recipe']['evidence']); j = int(json.load(open(os.path.join(INP, 's342101', 'threshold.json')))['j'])
acc = {k: np.zeros(3, np.int64) for k in ('NET_scope', 'B6_scope', 'NET_allk', 'B6_allk', 'NET_outside', 'B6_outside')}
for job in a.jobs.split(','):
    z = np.load(os.path.join(INP, 'export', job + '.npz'))
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(E.edges_list(z['edges']), n); L = list(z['L_k0_cam_nf']); occ = z['occ'] == 1
    for cm in F['comps']:
        S, kc = E.direct_sums(L, cm, ev.table); T = G.Target(private[cm].max(0), H)
        lg = E.net_logits(model, [S], [kc], 'cpu', 1)[0]
        P = dict(NET=E.net_readout(lg, kc, j) == 1, B6=E.linear_readout(S, kc, ev.theta) == 1)
        kn = kc > 0
        zones = dict(scope=T.known & H.region, allk=kn & H.region, outside=kn & ~T.known & H.region)
        for tag, p in P.items():
            for zn, m in zones.items():
                tr = occ & m; pr = p & m
                acc[f'{tag}_{zn}'] += np.array([int((pr & tr).sum()), int((pr & ~tr).sum()), int((~pr & tr).sum())]) * len(cm)
out = {k: dict(tp=int(v[0]), fp=int(v[1]), fn=int(v[2]), f1=float(2 * v[0] / max(2 * v[0] + v[1] + v[2], 1)), prec=float(v[0] / max(v[0] + v[1], 1)),
               rec=float(v[0] / max(v[0] + v[2], 1))) for k, v in acc.items()}
json.dump(dict(jobs=a.jobs.split(','), rows=out), open(os.path.join(HERE, 'allknown.json'), 'w'), indent=1)
print(json.dumps(out, indent=0))

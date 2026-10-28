"""R-J probe 3 (report only, CPU, bytes only, NOT a verdict and NOT an accuracy claim): where is the byte floor of the E3-L3 wire?
Same frozen checkpoint / exports as probe 2. For every directed message the integer pair (h, Lam) of a HYPOTHETICAL source code is
built over the sender side and priced with the verdict coder L.wire_bytes (O7):
  ACT   E3-L3 as run (h = sum z, Lam = k)
  F0    z == 0 (footprint counts only: Lam = k, h = 0)               -> the cost of carrying exact counts n-1 times
  F1    z == 0, Lam = min(k, 2) on the footprint                     -> counts coarsened (mask exact)
  F2    perfect source agreement: every observer sends zbar(v) = round(mean of the component's z) -> h = k zbar, Lam = k
  F2c   F2 with Lam = min(k, 2), h = Lam zbar
  F5    mask + consensus only: Lam = 1[k > 0], h = zbar             -> no count information at all
F2 / F2c / F5 are NOT achievable as such (they use the component consensus at the source); they bound what any entropy-reducing
source code (rate-trained encoder) could buy while the known mask stays exact."""
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


def side_of(nb, s, r):
    seen = {s}; st = [s]
    while st:
        u = st.pop()
        for v in nb[u]:
            if v != r and v not in seen:
                seen.add(v); st.append(v)
    return sorted(seen)


def wb(h, lam):
    return L.wire_bytes(h[None].astype(np.int64), lam.astype(np.int64))


def house(path, model, w, table_t):
    z = np.load(path); private = z['private']; n = private.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    Lp = list(z['L_pub_cam_nf']) if 'L_pub_cam_nf' in z.files else list(z['L_k0_cam_nf'])
    with torch.no_grad():
        zp, lp = L.encode_maps(model, Lp, table_t, 'cpu')
        hp, lpq, zq = E.source(zp, lp, 'round')
    zi = zq.numpy().astype(np.int64).reshape((n,) + tuple(zq.shape[-3:])); fp = (lpq.numpy() > 0).astype(np.int64)
    out = {k: 0.0 for k in ('ACT', 'F0', 'F1', 'F2', 'F2c', 'F5')}; out['n'] = n; out['msgs'] = 0
    for cm in F['comps']:
        if len(cm) < 2:
            continue
        Fl = L.local_forest(F, cm); nb = Fl['nb']
        ktot = fp[cm].sum(0); htot = (zi[cm] * fp[cm]).sum(0)
        zbar = np.where(ktot > 0, np.rint(htot / np.maximum(ktot, 1)), 0).astype(np.int64)
        for (s, r) in Fl['order']:
            sd = [cm[a] for a in side_of(nb, s, r)]
            k = fp[sd].sum(0); h = (zi[sd] * fp[sd]).sum(0); m = k > 0; k2 = np.minimum(k, 2)
            out['ACT'] += wb(h, k); out['F0'] += wb(np.zeros_like(k), k); out['F1'] += wb(np.zeros_like(k), k2)
            out['F2'] += wb(k * zbar, k); out['F2c'] += wb(k2 * zbar, k2); out['F5'] += wb(zbar * m, m.astype(np.int64))
            out['msgs'] += 1
    return out


def main():
    seed = sys.argv[1]; noise = sys.argv[2]; idx = [int(x) for x in sys.argv[3].split(',')]
    ck, model = E.load_model(os.path.join(INP, f'e3l3_s{seed}', 'final.pt'), 'cpu')
    w = float(model.w.detach()); table_t = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
    files = sorted(f for f in os.listdir(os.path.join(INP, 'export')) if f.startswith('val_') and f.endswith(f'_s{noise}.npz'))
    files = [files[i] for i in idx]
    res = {}; t0 = time.time(); op = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'rj_floor_{seed}_s{noise}.json')
    for f in files:
        res[f] = house(os.path.join(INP, 'export', f), model, w, table_t)
        print(f, json.dumps({k: round(v) if isinstance(v, float) else v for k, v in res[f].items()}), f'{time.time() - t0:.0f}s', flush=True)
        with open(op, 'w') as fh:
            json.dump(dict(seed=seed, noise=noise, w=w, houses=res), fh, indent=1)


if __name__ == '__main__':
    main()

import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, torch
sys.path.insert(0, os.path.abspath('../../../../scripts/claude_v342'))
import g4_e3l as L
G = L.G
rng = np.random.default_rng(5)
rec = []
orig = L.quantize
def spy(h, lam, mode):
    rec.append((h.clone(), lam.clone())); return orig(h, lam, mode)
L.quantize = spy
for t in range(200):
    m = int(rng.integers(2, 7)); ed = [(k, int(rng.integers(k))) for k in range(1, m)]
    Fl = G.forest([(i, j, 1.0) for i, j in ed], m)
    lam = rng.gamma(1.2, 2.0, (m, 4, 4, 2)) * (rng.random((m, 4, 4, 2)) > 0.4); z = rng.normal(0, 3, (m, 3, 4, 4, 2)); w = float(rng.choice([0.05, 0.5, 2.0, 30.0]))
    lq0 = np.round(lam); hq0 = np.round(lam[:, None] * z) * (lq0 > 0)[:, None]
    rec.clear()
    hb, lb, mq = L.gabp(torch.from_numpy(hq0), torch.from_numpy(lq0), Fl, w, 'round')
    nb = Fl['nb']; memo = {}
    for n_, (s, r) in enumerate(Fl['order']):
        lt = lq0[s].copy(); ht = hq0[s].copy()
        for k in nb[s]:
            if k != r:
                lt = lt + memo[(k, s)][1]; ht = ht + memo[(k, s)][0]
        f = w / (w + lt); a = ht * f[None]; b = lt * f
        lq = np.round(b); memo[(s, r)] = (np.round(a) * (lq > 0)[None], lq)
        hm, lm = mq[(s, r)]
        dh = hm.numpy() != memo[(s, r)][0]
        if dh.any():
            th, tl = rec[n_]
            print('t', t, 'msg', (s, r), 'np pre', a[dh][:2], 'torch pre', th.numpy()[dh][:2], 'torch out', hm.numpy()[dh][:2], 'np out', memo[(s, r)][0][dh][:2],
                  'lam_q', lq[dh.any(0)][:2], 'sameHt', np.array_equal(th.numpy(), a))
            raise SystemExit

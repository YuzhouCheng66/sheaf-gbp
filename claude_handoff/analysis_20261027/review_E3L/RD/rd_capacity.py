"""R-D capacity probe (Q4, report only). TRAIN houses ONLY (frozen v420 generator via g4_e3r imports; truth maps kept OUTSIDE the
public repo, in --cache, a scratch directory). Question: can a 4 x 4 x 2 block grid with 8 channels represent the clean R8 / K0 target
well enough for furniture F1 ~ .97, given (a) a linear per-block encoder + linear decoder (PCA), (b) a linear per-block encoder +
small context CNN decoder (autoencoder, trained on TRAIN-train, scored on TRAIN-holdout index % 10 == 9), (c) the same with
quantised codes (uniform-noise training, rounding at test) and the factorised empirical entropy of the codes.
Also the lossless ctx code length of the clean own maps (bits / agent) vs the E3-L source budget R* = 20,000 bit.
Usage: gen --cache DIR ; fit --cache DIR --out JSON
"""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import argparse, json, time, glob  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as Fnn  # noqa: E402
import g4_e3r as R  # noqa: E402
G = R.G; W = R.W

HOUSES = [(1, i) for i in (0, 1, 2, 3, 4, 9, 19)] + [(2, i) for i in (0, 1, 2, 3, 4, 9, 19)] + [(3, i) for i in (0, 1, 2, 3, 9)]


def gen(cache):
    os.makedirs(cache, exist_ok=True)
    for rooms, index in HOUSES:
        p = os.path.join(cache, f'TRAIN_r{rooms}_i{index}.npz')
        if os.path.exists(p):
            continue
        t0 = time.time()
        try:
            fx = W.make_fixture('TRAIN', index, rooms)
        except ValueError as e:
            print('skip', rooms, index, repr(e)[:100], flush=True); continue
        n = fx['private'].shape[0]
        F = G.forest(fx['edges'], n)
        comp_of = np.array([F['comp_of'][a] for a in range(n)], np.int64)
        np.savez_compressed(p, occ=np.asarray(fx['occ'], np.int8), slab=np.asarray(fx['slab'], bool), obj=np.asarray(fx['obj'], bool),
                            cell=np.asarray(fx['cell'], np.float64), private=np.asarray(fx['private'], np.int8), comp_of=comp_of)
        print('gen', rooms, index, n, f'{time.time() - t0:.0f}s', flush=True)


def load(cache):
    out = []
    for p in sorted(glob.glob(os.path.join(cache, 'TRAIN_r*_i*.npz'))):
        z = np.load(p); name = os.path.basename(p)[:-4]; index = int(name.split('_i')[1])
        H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); priv = z['private']; comp_of = z['comp_of']
        comps = [list(np.nonzero(comp_of == c)[0]) for c in sorted(set(comp_of.tolist()))]
        r8 = [priv[cm].max(0) for cm in comps]
        out.append(dict(name=name, holdout=(index % 10 == 9), H=H, k0=[priv[a] for a in range(priv.shape[0])], r8=r8, r8n=[len(c) for c in comps]))
    return out


def to_x(t):
    """ternary map -> [2, X, Y, Z] float (known, occupied)"""
    return np.stack([(t >= 0), (t == 1)]).astype(np.float32)


def blocks(x):
    """[2, X, Y, Z] -> [N_blocks, 64] (4 x 4 x 2 blocks)"""
    C, X, Y, Z = x.shape
    b = x.reshape(C, X // 4, 4, Y // 4, 4, Z // 2, 2).transpose(1, 3, 5, 0, 2, 4, 6).reshape(-1, C * 32)
    return b


def unblocks(b, shape):
    X, Y, Z = shape
    return b.reshape(X // 4, Y // 4, Z // 2, 2, 4, 4, 2).transpose(3, 0, 4, 1, 5, 2, 6).reshape(2, X, Y, Z)


def score_rows(maps, preds, H, mult=None):
    rows = []
    for t, (pk, po) in zip(maps, preds):
        p = np.where(pk, po.astype(np.int8), np.int8(-1)).astype(np.int8)
        rows.append(G.score(p, G.Target(t, H)))
    return rows


def summarise(rows, w=None):
    keys = ('furn_f1', 'furn_f1_tol1', 'surface_f1', 'free_recall', 'unknown_known_fpr')
    w = w or [1] * len(rows)
    out = {}
    for k in keys:
        v = [(r[k], ww) for r, ww in zip(rows, w) if r.get(k) is not None]
        out[k] = float(sum(a * b for a, b in v) / max(sum(b for _, b in v), 1)) if v else None
    return out


class AE(nn.Module):
    def __init__(self, D=8, width=32, ctx=True, deep=False):
        super().__init__()
        if deep:                                                            # context encoder (conv at voxel res, strided block conv, conv)
            self.enc = nn.Sequential(nn.Conv3d(2, 16, 3, padding=1), nn.ReLU(), nn.Conv3d(16, 32, (4, 4, 2), stride=(4, 4, 2)), nn.ReLU(),
                                     nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv3d(32, D, 1))
        else:
            self.enc = nn.Conv3d(2, D, (4, 4, 2), stride=(4, 4, 2))       # linear per-block encoder (8 channels)
        k = 3 if ctx else 1; p = 1 if ctx else 0
        self.dec = nn.Sequential(nn.Conv3d(D + 1, width, k, padding=p), nn.ReLU(), nn.Conv3d(width, width, k, padding=p), nn.ReLU(),
                                 nn.Conv3d(width, width, k, padding=p), nn.ReLU(), nn.Conv3d(width, 64, 1))

    def code(self, x):
        return self.enc(x)

    def decode(self, c):
        B, D, X, Y, Z = c.shape
        zz = torch.linspace(-1, 1, Z).view(1, 1, 1, 1, Z).expand(B, 1, X, Y, Z)
        y = self.dec(torch.cat([c, zz], 1))                                 # [B, 64, X, Y, Z] -> [B, 2, 4X, 4Y, 2Z]
        y = y.view(B, 2, 4, 4, 2, X, Y, Z).permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(B, 2, 4 * X, 4 * Y, 2 * Z)
        return y


def train_ae(train_maps, steps, quant=None, ctx=True, seed=0, threads=4, crop=64, deep=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = AE(ctx=ctx, deep=deep); opt = torch.optim.Adam(m.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    X = [torch.from_numpy(to_x(t)) for t in train_maps]
    t0 = time.time()
    for s in range(steps):
        xb = []
        for _ in range(4):
            x = X[int(rng.integers(len(X)))]
            i = int(rng.integers(0, 128 - crop + 1) // 4 * 4); j = int(rng.integers(0, 128 - crop + 1) // 4 * 4)
            x = x[:, i:i + crop, j:j + crop]
            if rng.random() < .5:
                x = x.flip(1)
            if rng.random() < .5:
                x = x.transpose(1, 2)
            xb.append(x)
        xb = torch.stack(xb)
        c = m.code(xb)
        if quant is not None:
            c = quant * torch.tanh(c / quant)                               # bounded code: levels -quant..quant after rounding
            c = c + torch.rand_like(c) - 0.5
        y = m.decode(c)
        lk = Fnn.binary_cross_entropy_with_logits(y[:, 0], xb[:, 0])
        msk = xb[:, 0]
        lo = (Fnn.binary_cross_entropy_with_logits(y[:, 1], xb[:, 1], reduction='none') * msk).sum() / msk.sum().clamp(min=1)
        loss = lk + lo
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 250 == 0 or s == steps - 1:
            print(f'  step {s} loss {float(loss):.4f} ({float(lk):.4f}/{float(lo):.4f}) {time.time() - t0:.0f}s', flush=True)
    return m


@torch.no_grad()
def ae_predict(m, t, quant=None):
    x = torch.from_numpy(to_x(t))[None]
    c = m.code(x)
    hist = None
    if quant is not None:
        c = torch.round(quant * torch.tanh(c / quant))
        kn = Fnn.max_pool3d(x[:, :1], (4, 4, 2), (4, 4, 2))[0, 0] > 0
        hist = [np.bincount((c[0, d][kn].numpy().astype(np.int64) + quant), minlength=2 * quant + 1) for d in range(c.shape[1])]
    y = m.decode(c)[0].numpy()
    return (y[0] > 0, y[1] > 0), hist


def ent_bits(hists):
    """factorised empirical entropy (bits) summed over channels per touched block"""
    tot = 0.0
    for h in hists:
        p = h[h > 0] / h.sum(); tot += float(-(p * np.log2(p)).sum())
    return tot


def fit(cache, out, threads, steps, variants='pca,ae_ctx,ae_noctx,ae_ctx_q2,ae_ctx_q1'):
    variants = variants.split(',')
    torch.set_num_threads(threads)
    hs = load(cache)
    tr = [h for h in hs if not h['holdout']]; ho = [h for h in hs if h['holdout']]
    res = dict(train_houses=[h['name'] for h in tr], holdout_houses=[h['name'] for h in ho])
    print('train', len(tr), 'holdout', len(ho), flush=True)
    # (0) information: lossless ctx code length of clean own maps and R8 unions (bits)
    own_bits = [8 * G.ctx_tern_bytes(t) for h in hs for t in h['k0']]
    r8_bits = [8 * G.ctx_tern_bytes(t) for h in hs for t in h['r8']]
    touched_blocks = [int((blocks(to_x(t))[:, :32].max(1) > 0).sum()) for h in hs for t in h['k0']]
    res['info'] = dict(own_bits_median=float(np.median(own_bits)), own_bits_q=np.quantile(own_bits, [.1, .9]).tolist(),
                       r8_bits_median=float(np.median(r8_bits)), touched_blocks_median=float(np.median(touched_blocks)),
                       bits_per_touched_block_lossless=float(np.median(np.array(own_bits) / np.maximum(touched_blocks, 1))),
                       e3l_budget_bits=20000.0)
    print('info', json.dumps(res['info']), flush=True)
    # (1) PCA-8 linear per block on (known, occ) 64-dim
    if 'pca' in variants:
        Btr = np.concatenate([blocks(to_x(t)) for h in tr for t in h['k0'] + h['r8']])
        mu = Btr.mean(0); U, S, Vt = np.linalg.svd(Btr - mu, full_matrices=False)
        ev = (S ** 2) / (S ** 2).sum()
        res['pca_explained_8'] = float(ev[:8].sum())
        for D in (8, 16, 32):
            P = Vt[:D]
            def pred(t):
                b = blocks(to_x(t)); r = (b - mu) @ P.T @ P + mu
                y = unblocks(r, t.shape); return (y[0] > .5, y[1] > .5)
            rows8 = score_rows([t for h in ho for t in h['r8']], [pred(t) for h in ho for t in h['r8']], None) if False else None
            rr, rk = [], []
            for h in ho:
                rr += [G.score(np.where(pk, po.astype(np.int8), np.int8(-1)).astype(np.int8), G.Target(t, h['H'])) for t in h['r8'] for pk, po in [pred(t)]]
                rk += [G.score(np.where(pk, po.astype(np.int8), np.int8(-1)).astype(np.int8), G.Target(t, h['H'])) for t in h['k0'] for pk, po in [pred(t)]]
            # generous variant: known mask given (oracle), occupancy from the PCA reconstruction
            rro = []
            for h in ho:
                for t in h['r8']:
                    _, po = pred(t); rro.append(G.score(np.where(t >= 0, po.astype(np.int8), np.int8(-1)).astype(np.int8), G.Target(t, h['H'])))
            res[f'pca{D}'] = dict(R8=summarise(rr), K0=summarise(rk), R8_known_oracle=summarise(rro))
            print(f'pca{D}', json.dumps(res[f'pca{D}']), flush=True)
    # (2) AE: linear 8-ch per-block encoder + context CNN decoder; (2b) no-context decoder; (3) quantised codes
    trm = [t for h in tr for t in h['k0'] + h['r8']]
    for name, kw in (('ae_ctx', dict(ctx=True)), ('ae_noctx', dict(ctx=False)), ('ae_ctx_q2', dict(ctx=True, quant=2)), ('ae_ctx_q1', dict(ctx=True, quant=1)), ('ae_deep', dict(ctx=True, deep=True)), ('ae_deep_q2', dict(ctx=True, deep=True, quant=2))):
        if name not in variants:
            continue
        t0 = time.time()
        m = train_ae(trm, steps, threads=threads, **kw)
        rr, rk, hh = [], [], None
        for h in ho:
            for t in h['r8']:
                (pk, po), hi = ae_predict(m, t, kw.get('quant'))
                rr.append(G.score(np.where(pk, po.astype(np.int8), np.int8(-1)).astype(np.int8), G.Target(t, h['H'])))
            for t in h['k0']:
                (pk, po), hi = ae_predict(m, t, kw.get('quant'))
                rk.append(G.score(np.where(pk, po.astype(np.int8), np.int8(-1)).astype(np.int8), G.Target(t, h['H'])))
                if hi is not None:
                    hh = hi if hh is None else [a + b for a, b in zip(hh, hi)]
        res[name] = dict(R8=summarise(rr), K0=summarise(rk), seconds=time.time() - t0, steps=steps,
                         bits_per_touched_block_factorised=(ent_bits(hh) if hh is not None else None),
                         params=sum(p.numel() for p in m.parameters()))
        print(name, json.dumps(res[name]), flush=True)
        with open(out, 'w') as fh:
            json.dump(res, fh, indent=1)
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--cache', required=True); ap.add_argument('--out')
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--steps', type=int, default=1500); ap.add_argument('--variants', default='pca,ae_ctx,ae_noctx,ae_ctx_q2,ae_ctx_q1')
    a = ap.parse_args()
    if a.cmd == 'gen':
        gen(a.cache)
    else:
        fit(a.cache, a.out, a.threads, a.steps, a.variants)


if __name__ == '__main__':
    main()

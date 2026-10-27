"""A06: does generative SAMPLING help or hurt furniture F1? (Claude A06, 2026-10-27). CPU only, 6 VAL demo houses, leave-one-house-out.
Code = the classical 2x2x1 (tau .5) ternary down-sample of the target (bytes fixed: rate_baselines 3.3). Decoder = a learned conditional
generative model p(fine 2x2 ternary pattern | 7 coarse cells: the block and its 6 face neighbours), counted on the 5 other houses
(81 patterns x 2,187 contexts; the stand-in for 'a generative decoder with a prior'). Decoding rules compared on the same p:
broadcast (no prior), MAP pattern, per-voxel marginal threshold theta (F1-optimal rule: theta* = F1*/2), one SAMPLE per block (4 draws,
mean F1), and the average of 16 samples thresholded (Monte-Carlo marginals). Scored with the training metric on K0 and R8 targets."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a06_common import houses, scene, metrics_rows, mean, RB  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
Q = [(0, 0), (0, 1), (1, 0), (1, 1)]
POW3 = 3 ** np.arange(7)


def code_ctx(t):
    c, _ = RB.down_up(t, (2, 2, 1), .5)
    cp = np.pad(c, 1, constant_values=-1)
    nb = [c, cp[:-2, 1:-1, 1:-1], cp[2:, 1:-1, 1:-1], cp[1:-1, :-2, 1:-1], cp[1:-1, 2:, 1:-1], cp[1:-1, 1:-1, :-2], cp[1:-1, 1:-1, 2:]]
    ctx = sum((x.astype(np.int64) + 1) * POW3[q] for q, x in enumerate(nb))
    pat = sum((t[dx::2, dy::2, :].astype(np.int64) + 1) * 3 ** q for q, (dx, dy) in enumerate(Q))
    return c, ctx, pat


def digits(pidx):
    return np.stack([(pidx // 3 ** q) % 3 for q in range(4)], -1) - 1          # [.., 4] ternary


ALL = digits(np.arange(81))                                                      # [81, 4]
cache = {}
for path in houses():
    S = scene(path); h = S['m']['house']
    for hop in (0, 8):
        T = S['tgt'][hop]; cnt = np.zeros((3 ** 7, 81))
        items = []
        for a in range(T.shape[0]):
            c, ctx, pat = code_ctx(T[a]); kn = c >= 0
            np.add.at(cnt, (ctx[kn], pat[kn]), 1); items.append((c, ctx))
        cache[(h, hop)] = dict(cnt=cnt, items=items, T=T)
    cache[(h, 'S')] = dict(slab=S['slab'], obj=S['obj'], struct=S['struct'])
    print(h, 'counted', flush=True)

names = [k[0] for k in cache if k[1] == 'S']
rng = np.random.default_rng(342202)
res = {str(hop): {} for hop in (0, 8)}
for h in names:
    Sx = cache[(h, 'S')]
    for hop in (0, 8):
        cnt = sum(cache[(g, hop)]['cnt'] for g in names if g != h)
        seen = cnt.sum(1) > 0
        p = (cnt + 1e-3) / (cnt + 1e-3).sum(1, keepdims=True)
        p_occ = p @ (ALL == 1).astype(float); p_kn = p @ (ALL >= 0).astype(float)          # [ctx, 4]
        mapi = p.argmax(1); cum = np.cumsum(p, 1).astype(np.float32)
        T = cache[(h, hop)]['T']; preds = {}
        for a, (c, ctx) in enumerate(cache[(h, hop)]['items']):
            kn = c >= 0; unseen = kn & ~seen[ctx]
            def fine(q4):                                                                # q4 [64, 64, 32, 4] ternary -> [128, 128, 32]
                q4 = np.where(kn[..., None], q4, -1); q4 = np.where(unseen[..., None], c[..., None], q4)
                out = np.empty((128, 128, 32), np.int8)
                for q, (dx, dy) in enumerate(Q):
                    out[dx::2, dy::2, :] = q4[..., q]
                return out
            preds.setdefault('broadcast', []).append(fine(np.repeat(c[..., None], 4, -1)))
            preds.setdefault('MAP', []).append(fine(ALL[mapi[ctx]]))
            po, pk = p_occ[ctx], p_kn[ctx]
            for th in (.2, .3, .4, .5, .6):
                preds.setdefault(f'marg_{th}', []).append(fine(np.where(po >= th, 1, np.where(pk >= .5, 0, -1))))
            draws = []
            for s in range(16):
                u = rng.random(ctx.shape)
                pi = (cum[ctx] < u[..., None]).sum(-1).clip(0, 80)
                draws.append(ALL[pi])
                if s < 4:
                    preds.setdefault(f'sample{s}', []).append(fine(ALL[pi]))
            mo = np.mean([d == 1 for d in draws], 0); mk = np.mean([d >= 0 for d in draws], 0)
            for th in (.3, .5):
                preds.setdefault(f'sampavg16_{th}', []).append(fine(np.where(mo >= th, 1, np.where(mk >= .5, 0, -1))))
        for k, P in preds.items():
            Sd = dict(slab=Sx['slab'], obj=Sx['obj'], struct=Sx['struct'])
            rows = metrics_rows(np.stack(P), T, Sd)
            d = res[str(hop)].setdefault(k, dict(surface_f1=[], var_occ_f1=[], furn_f1=[]))
            for r in rows:
                for kk in d:
                    d[kk].append(r[kk])
    print(h, 'decoded', flush=True)

summ = {hop: {k: {kk: mean(v) for kk, v in d.items()} for k, d in R.items()} for hop, R in res.items()}
for hop in summ:
    s = summ[hop]; s['sample_mean4'] = {kk: float(np.mean([s[f'sample{i}'][kk] for i in range(4)])) for kk in ('surface_f1', 'var_occ_f1', 'furn_f1')}
json.dump(summ, open(os.path.join(OUT, 'sampling.json'), 'w'), indent=1)
for hop, s in summ.items():
    print(f'\n=== hop {hop}')
    for k, d in s.items():
        if k.startswith('sample') and k != 'sample_mean4':
            continue
        print(f"  {k:16s} surf {d['surface_f1']:.3f} var {d['var_occ_f1']:.3f} furn {d['furn_f1']:.3f}")

"""a10_residual.py (A10, 2026-10-27): what does it cost to make our decoded map EXACT (everywhere, or only in the furniture region),
when the residual is coded with the decoder's own output as side information known to both ends (closed-loop / Wyner-Ziv with
encoder-side knowledge)? Decoder output = the demo's exported prediction of 3S-X + calibrated rule (hard ternary; soft logits would
code cheaper, so these are upper bounds of a closed-loop residual layer). Compared with a classical base (2x2x1 block-ternary,
tau .5) + the same residual coder, and with coding the furniture surface with no base at all.
Residual codes (exact adaptive KT):
  tern_full : the whole ternary target given pred (context 6 causal target nbrs x pred value x pred-known forward flags)
  V_full    : the whole visible surface V = (target == 1) given pred (binary; the known mask is then rendered from V + poses, 24 B
              per agent for float32 poses -> exact)
  V_furn    : V only inside the furniture region (= ~slab & ~dilate6(structure), the metric's region), hybrid context
After V_furn the furniture-region F1 is 1 by construction; the other F1s are recomputed. Also: V_furn_alone = the furniture
surface with no base (the decoder knows the region)."""
import glob, json, os, sys, time, gc
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import *  # noqa

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
POSE32 = 24.                                                         # 3 poses x (x, y) float32


def tern_given(target, pred):
    t, p = T(target), T(pred)
    c = (causal(t, 3, NB6) * 3 + p) * 8 + fwd_flags(p > 0)
    return kt_bits(t, c, 3)[0] / 8 + 2


def bin_given(target_bin, pred, region=None):
    p = T(pred)
    if region is None:
        h = target_bin.astype(np.int64); sel = np.ones(h.shape, bool)
    else:
        h = np.where(region, target_bin, pred == 1).astype(np.int64); sel = region
    c = causal(h, 2) * 3 + p
    return kt_bits(h[sel], c[sel], 2)[0] / 8 + 2


def bin_alone(target_bin, region):
    h = (target_bin & region).astype(np.int64)
    c = causal(h, 2)
    return kt_bits(h[region], c[region], 2)[0] / 8 + 2


def do_house(path):
    m, own, pred, gt = load(path); n = own.shape[0]
    slab = gt['slab']; struct = gt['occ'] & ~gt['obj'] & ~gt['slab']; region = ~slab & ~dilate6(struct)
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    rows = []
    for hop, Tk in ((0, own), (8, T8)):
        for a in range(n):
            t = Tk[a]; p = pred[hop][a]; V = t == 1
            coarse, up = down_up(t, (2, 2, 1), .5)
            # hybrid prediction after the furniture residual: inside the region occupancy = V exactly (pred's known flag kept elsewhere)
            hyb = p.copy(); inr = region
            hyb[inr & V] = 1
            hyb[inr & ~V & (p == 1)] = 0
            before = metrics1(p, t, slab, struct); after = metrics1(hyb, t, slab, struct)
            cb = metrics1(up, t, slab, struct)
            rows.append(dict(house=m['house'], hop=hop, agent=a, ctx=tern_bytes(t), V=bin_bytes(V) + POSE32,
                             tern_full=tern_given(t, p), V_full=bin_given(V, p) + POSE32, V_furn=bin_given(V, p, region),
                             V_furn_alone=bin_alone(V, region),
                             base2=tern_bytes(coarse), base2_tern_full=tern_given(t, up), base2_V_full=bin_given(V, up) + POSE32, base2_V_furn=bin_given(V, up, region),
                             f1_before=before, f1_after_furnV=after, f1_base2=cb,
                             furn_vox=int((V & region).sum()), surf_vox=int(V.sum())))
    return rows


out = []
t0 = time.time()
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    cache = os.path.join(HERE, 'cache_resid_' + os.path.basename(path))
    for attempt in range(60):
        if os.path.exists(cache):
            break
        try:
            json.dump(do_house(path), open(cache, 'w'))
        except MemoryError as exc:
            print('MemoryError, retry', attempt, flush=True); gc.collect(); time.sleep(20)
    out += json.load(open(cache)); print(os.path.basename(path), f'{time.time() - t0:.0f}s', flush=True)
json.dump(out, open(os.path.join(HERE, 'a10_residual.json'), 'w'), indent=1)
for hop in (0, 8):
    r = [x for x in out if x['hop'] == hop]
    med = lambda k: float(np.median([x[k] for x in r]))
    mf = lambda d, k: float(np.mean([x[d][k] for x in r if x[d][k] is not None]))
    print(f'hop {hop}: median bytes ctx {med("ctx"):.0f}  V+pose {med("V"):.0f} | residual given OUR pred: tern_full {med("tern_full"):.0f} '
          f'V_full {med("V_full"):.0f} V_furn {med("V_furn"):.0f} | furniture V alone {med("V_furn_alone"):.0f} | base 2x2x1 {med("base2"):.0f} '
          f'+ tern {med("base2_tern_full"):.0f} / + V {med("base2_V_full"):.0f} / + V_furn {med("base2_V_furn"):.0f}')
    print(f'        F1 before: surf {mf("f1_before","surface_f1"):.3f} var {mf("f1_before","var_occ_f1"):.3f} furn {mf("f1_before","furn_f1"):.3f} | '
          f'after V_furn: surf {mf("f1_after_furnV","surface_f1"):.3f} var {mf("f1_after_furnV","var_occ_f1"):.3f} furn {mf("f1_after_furnV","furn_f1"):.3f} | '
          f'base2 alone: surf {mf("f1_base2","surface_f1"):.3f} furn {mf("f1_base2","furn_f1"):.3f} | furn vox {med("furn_vox"):.0f} surf vox {med("surf_vox"):.0f}')

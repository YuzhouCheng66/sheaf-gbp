"""A09 probe 4 (Claude, 2026-10-27, CPU only): classical lossy comparators for two gates.
(a) design A (world code, no GT at the encoder): gravity-filled world W = grav_fill(own) (probe 1b), block-OR down-sampled, ctx-coded at the
    coarse grid; prediction = up-sampled W' as occupied (direct scoring on target-known voxels; a sensor-simulation decoder can only be
    as good or worse, so this is an optimistic classical point) -> bytes vs surface / var / furniture F1, K0 and R8;
(b) design B's teacher gate (whole-house GT world, every voxel known): block-OR down-sampling of GT occupancy -> bytes vs F1 on the GT world.
Output: probes/A09/p4_lossy_world.json"""
import glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a09lib import load, targets, ctx_bytes_binary, metrics   # noqa: E402
from p1b_world_interval import grav_fill                        # noqa: E402  (module body guarded below)

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))


def down_or(S, f):
    X, Y, Z = S.shape; bx, by, bz = f
    c = S.reshape(X // bx, bx, Y // by, by, Z // bz, bz).any((1, 3, 5))
    return c, c.repeat(bx, 0).repeat(by, 1).repeat(bz, 2)


FACT = [(1, 1, 1), (2, 2, 1), (1, 1, 2), (2, 2, 2), (4, 4, 1)]
acc = {}; t0 = time.time()
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]
    slab, obj = gt['slab'], gt['obj']; struct = gt['occ'] & ~gt['obj'] & ~gt['slab']
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    for k, Tk in ((0, own), (8, T8)):
        for f in FACT:
            preds, byts = [], []
            for a in range(n):
                c, up = down_or(grav_fill(Tk[a]), f)
                byts.append(ctx_bytes_binary(c)); preds.append(np.where(up, np.int8(1), np.int8(-1)))
            rows = metrics(np.stack(preds), Tk, slab, obj, struct)
            L = acc.setdefault(f'gravW_K{k}_{f[0]}x{f[1]}x{f[2]}', dict(bytes=[], surface_f1=[], var_occ_f1=[], furn_f1=[]))
            L['bytes'] += byts
            for r in rows:
                for q in ('surface_f1', 'var_occ_f1', 'furn_f1'):
                    if r[q] is not None:
                        L[q].append(r[q])
    G = gt['occ'].astype(np.int8)[None]                                # whole-house world, every voxel known
    for f in FACT:
        c, up = down_or(gt['occ'], f)
        r = metrics(np.where(up, np.int8(1), np.int8(0))[None], G, slab, obj, struct)[0]
        L = acc.setdefault(f'GTworld_{f[0]}x{f[1]}x{f[2]}', dict(bytes=[], surface_f1=[], var_occ_f1=[], furn_f1=[]))
        L['bytes'].append(ctx_bytes_binary(c))
        for q in ('surface_f1', 'var_occ_f1', 'furn_f1'):
            L[q].append(r[q])
    print(m['house'], f'{time.time() - t0:.0f}s', flush=True)
summary = {k: dict(bytes_median=float(np.median(v['bytes'])), surface_f1=float(np.mean(v['surface_f1'])), var_occ_f1=float(np.mean(v['var_occ_f1'])),
                   furn_f1=float(np.mean(v['furn_f1']))) for k, v in acc.items()}
for k, v in summary.items():
    print(f"{k:24s} {v['bytes_median']:7.0f} B  surf {v['surface_f1']:.3f}  var {v['var_occ_f1']:.3f}  furn {v['furn_f1']:.3f}")
json.dump(summary, open(os.path.join(HERE, 'p4_lossy_world.json'), 'w'), indent=1)

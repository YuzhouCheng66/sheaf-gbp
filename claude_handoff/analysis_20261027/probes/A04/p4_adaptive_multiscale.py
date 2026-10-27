"""A04 probe P4 (Claude, 2026-10-27): what does ADAPTIVE refinement buy (coarse stalks everywhere, fine stalks only where the coarse one
is wrong)?  Classical oracle on the K0 own maps and the R8 targets, same metric as training:
  coarse = block-ternary down-sample (bx, by, 1), tau = .5 (rate_baselines_lib.down_up); fine = (fx, fy, 1) or lossless (1, 1, 1);
  a coarse column is REFINED iff the fine decode differs from the coarse decode anywhere in it (the encoder knows its own map);
  bytes = ctx(coarse grid) + ctx(refine flags, 2-D) + ctx(fine values inside refined columns | parent coarse value, causal context taken
  from what the decoder actually holds), + 2 B per part.  Reference rows: plain coarse and plain fine (same coder)."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, targets, ctx_bytes, ctx_bits_cond, down_up, metrics, gt_masks, HERE  # noqa: E402

SCHEMES = [((4, 4, 1), (2, 2, 1)), ((4, 4, 1), (1, 1, 1)), ((8, 8, 1), (2, 2, 1)), ((2, 2, 1), (1, 1, 1))]


def two_level(t, cf, ff):
    X, Y, Z = t.shape
    cg, cu = down_up(t, cf, .5)
    if ff == (1, 1, 1):
        fg, fu = t, t
    else:
        fg, fu = down_up(t, ff, .5)
    rx, ry = cf[0] // ff[0], cf[1] // ff[1]                              # fine cells per coarse cell
    diff = (fu != cu).reshape(X // cf[0], cf[0], Y // cf[1], cf[1], Z).any((1, 3, 4))      # [cx, cy] refine flags
    flag_f = diff.repeat(rx, 0).repeat(ry, 1)                            # on the fine grid [fx, fy]
    parent = cg.repeat(rx, 0).repeat(ry, 1)                              # coarse value seen from the fine grid [fx, fy, Z]
    held = np.where(flag_f[:, :, None], fg, parent)                      # what the decoder holds after decoding (causal context source)
    b_coarse = ctx_bytes(cg)
    b_flags = ctx_bits_cond(diff[:, :, None].astype(np.int64), 2) / 8 + 2
    mask = np.broadcast_to(flag_f[:, :, None], held.shape)
    b_fine = (ctx_bits_cond(held + 1, 3, side=[(parent + 1, 3)], mask=mask) / 8 + 2) if mask.any() else 0.
    dec = np.where(flag_f.repeat(ff[0], 0).repeat(ff[1], 1)[:, :, None], fu, cu)
    return dec, b_coarse + b_flags + b_fine, float(diff.mean()), float(diff[(cg >= 0).any(2)].mean()) if (cg >= 0).any() else 0.


out = {}
t0 = time.time()
for path in houses():
    m, own, pred, gt = load(path); n = own.shape[0]
    slab, obj, struct = gt_masks(gt)
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    for tag, T in (('K0', own), ('R8', T8)):
        for cf, ff in SCHEMES:
            name = f'{tag}_{cf[0]}x{cf[1]}->{ff[0]}x{ff[1]}'
            decs, bytes_, share = [], [], []
            for a in range(n):
                dec, b, s_all, s_obs = two_level(T[a], cf, ff); decs.append(dec); bytes_.append(b); share.append(s_obs)
            rows = metrics(torch.from_numpy(np.stack(decs).reshape(n, -1)), torch.from_numpy(T.reshape(n, -1)), slab, obj, struct)
            d = out.setdefault(name, dict(bytes=[], refined_share_of_observed_cols=[], surface_f1=[], var_occ_f1=[], furn_f1=[]))
            d['bytes'] += bytes_; d['refined_share_of_observed_cols'] += share
            for r in rows:
                for k in ('surface_f1', 'var_occ_f1', 'furn_f1'):
                    d[k].append(r[k])
    print(m['house'], f'{time.time() - t0:.0f}s', flush=True)


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


S = {k: dict(bytes_mean=mean(v['bytes']), bytes_median=float(np.median(v['bytes'])), refined_share=mean(v['refined_share_of_observed_cols']),
             surface_f1=mean(v['surface_f1']), var_occ_f1=mean(v['var_occ_f1']), furn_f1=mean(v['furn_f1'])) for k, v in out.items()}
for k, v in S.items():
    print(f"{k:22s} bytes {v['bytes_mean']:7.0f} (med {v['bytes_median']:6.0f})  refined {v['refined_share']:.2f}  surf {v['surface_f1']:.3f}  var {v['var_occ_f1']:.3f}  furn {v['furn_f1']:.3f}")
json.dump(S, open(os.path.join(HERE, 'p4_adaptive_multiscale.json'), 'w'), indent=1)

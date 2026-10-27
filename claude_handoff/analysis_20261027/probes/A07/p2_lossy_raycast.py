"""p2_lossy_raycast.py (A07): the honest CLASSICAL lossy competitor. Send only a block-down-sampled hit set H (occupied voxels) + the
sensor poses; the decoder up-samples H and RE-CASTS the public beams from the poses, which gives known / free / occupied without
spending bits on the visibility boundary (77 % of the ctx bits of an own map). Rate = binary KT context code of the coarse H + 5.5 B
per agent of poses. Scored with the training metrics3d copy against the re-simulated targets (sim_cache, known poses):
K0 = own map, R8 = union of the reached own maps (one decode per forest component). Also the ternary down-up baseline of
rate_baselines (no ray casting) on the same targets, and FPR (target-unknown voxels predicted known)."""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a07_lib import observe, ctx_bin, ctx_ter, down_up, score, mean_rows, POSE_BYTES

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, 'sim_cache')
VARIANTS = [((1, 1, 1), 'any'), ((2, 2, 1), 'any'), ((2, 2, 1), 'ge2'), ((1, 1, 2), 'any'), ((2, 2, 2), 'any'), ((4, 4, 1), 'any')]


def coarse_of(H, f, rule):
    X, Y, Z = H.shape; bx, by, bz = f
    c = H.reshape(X // bx, bx, Y // by, by, Z // bz, bz).sum((1, 3, 5))
    return (c >= (2 if rule == 'ge2' else 1)).astype(np.uint8)


def up(c, f):
    return c.repeat(f[0], 0).repeat(f[1], 1).repeat(f[2], 2)


def fpr(pred, target):
    u = target < 0
    return float(((pred >= 0) & u).sum() / max(u.sum(), 1))


def house(name):
    t0 = time.time()
    z = np.load(os.path.join(CACHE, name + '.npz'))
    own, poses, cell, R8 = z['own'], z['poses'], z['cell'], z['reached8']
    gt = dict(occ=z['gt_occ'], slab=z['gt_slab'], obj=z['gt_obj']); n = own.shape[0]
    T8 = np.where(R8[:, :, None, None, None], own[None], np.int8(-1)).max(1)
    comps = {}
    for a in range(n):
        comps.setdefault(tuple(np.nonzero(R8[a])[0]), []).append(a)
    res = dict(house=name, variants={})
    for f, rule in VARIANTS:
        key = f'{f[0]}x{f[1]}x{f[2]}_{rule}'
        p0, b0 = [], []
        for a in range(n):
            c = coarse_of(own[a] == 1, f, rule); b0.append(ctx_bin(c) + POSE_BYTES)
            p0.append(observe(up(c, f), cell, poses[a]))
        p0 = np.stack(p0)
        p8 = np.zeros_like(own); b8 = [0.] * n
        for members, agents in comps.items():
            U = own[list(members)].max(0); c = coarse_of(U == 1, f, rule); u = up(c, f)
            dec = np.stack([observe(u, cell, poses[m]) for m in members]).max(0)
            for a in agents:
                p8[a] = dec; b8[a] = ctx_bin(c) + POSE_BYTES * len(members)
        r0 = score(p0, own, gt); r8 = score(p8, T8, gt)
        res['variants'][key] = dict(K0=dict(bytes=b0, rows=r0, fpr=[fpr(p0[a], own[a]) for a in range(n)]),
                                    R8=dict(bytes=b8, rows=r8, fpr=[fpr(p8[a], T8[a]) for a in range(n)]))
    for f, tau in (((2, 2, 1), .5), ((1, 1, 2), .5), ((4, 4, 1), .5)):           # rate_baselines ternary down-up, same targets
        key = f'ternary_downup_{f[0]}x{f[1]}x{f[2]}_tau{tau}'
        p0, b0, p8, b8 = [], [], [], []
        for a in range(n):
            c, u = down_up(own[a], f, tau); p0.append(u); b0.append(ctx_ter(c))
            c, u = down_up(T8[a], f, tau); p8.append(u); b8.append(ctx_ter(c))
        p0 = np.stack(p0); p8 = np.stack(p8)
        res['variants'][key] = dict(K0=dict(bytes=b0, rows=score(p0, own, gt), fpr=[fpr(p0[a], own[a]) for a in range(n)]),
                                    R8=dict(bytes=b8, rows=score(p8, T8, gt), fpr=[fpr(p8[a], T8[a]) for a in range(n)]))
    res['lossless'] = dict(K0_ctx=[ctx_ter(own[a]) for a in range(n)], R8_ctx=[ctx_ter(T8[a]) for a in range(n)])
    print(name, f'{time.time() - t0:.0f}s', flush=True)
    return res


if __name__ == '__main__':
    names = sorted(f[:-4] for f in os.listdir(CACHE) if f.endswith('.npz'))
    with Pool(3) as pool:
        res = pool.map(house, names)
    S = {}
    for key in res[0]['variants']:
        S[key] = {}
        for k in ('K0', 'R8'):
            rows = [r for h in res for r in h['variants'][key][k]['rows']]
            S[key][k] = dict(bytes_median=float(np.median([b for h in res for b in h['variants'][key][k]['bytes']])),
                             surface_f1=mean_rows(rows, 'surface_f1'), var_occ_f1=mean_rows(rows, 'var_occ_f1'), furn_f1=mean_rows(rows, 'furn_f1'),
                             fpr=float(np.mean([v for h in res for v in h['variants'][key][k]['fpr']])))
    S['lossless_ctx_median'] = dict(K0=float(np.median([v for h in res for v in h['lossless']['K0_ctx']])), R8=float(np.median([v for h in res for v in h['lossless']['R8_ctx']])))
    json.dump(dict(summary=S, houses=res), open(os.path.join(HERE, 'p2_lossy_raycast.json'), 'w'), indent=1)
    for key, v in S.items():
        if key.startswith('lossless'):
            print(key, v); continue
        print(f"{key:30s} K0 {v['K0']['bytes_median']:6.0f} B surf {v['K0']['surface_f1']:.3f} var {v['K0']['var_occ_f1']:.3f} furn {v['K0']['furn_f1']:.3f} fpr {v['K0']['fpr']:.4f} | "
              f"R8 {v['R8']['bytes_median']:6.0f} B surf {v['R8']['surface_f1']:.3f} var {v['R8']['var_occ_f1']:.3f} furn {v['R8']['furn_f1']:.3f} fpr {v['R8']['fpr']:.4f}")

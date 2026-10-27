"""p3_noise.py (A07): how does the LOSSLESS cost grow, and what does lossless transmission even buy, once the sensor is not perfect?
Same 6 VAL houses and poses as sim_cache. Each beam's TRUE first-hit range is traced in the true world from the TRUE pose; the agent then
integrates a corrupted measurement (range noise sigma0 + k * r, dropped beams, early outlier returns) at its ESTIMATED pose (per-pose
translation / yaw error; for the round-8 union also a per-agent frame error) with OctoMap-style log-odds (+0.85 hit, -0.40 miss, one
miss per voxel per beam, the endpoint voxel is not a miss; occupied if > 0). Clean condition = the v420 map exactly.
Reported per condition (medians over 48 own maps / 48 union rows):
  rates of the NOISY map: ctx (ternary), H_ctx + poses (sensor-aware), gzip;
  fidelity of transmitting the noisy map losslessly, scored against the CLEAN target (what a perfect sensor at the true poses saw):
  surface / var / furniture-region F1 = the 'noise floor' of any lossless scheme;
  two classical denoising-lossy codecs on the same noisy map, scored against the clean target: ternary 2x2x1 tau .5 down-up (ctx of the
  coarse grid) and 2x2x1 'ge2' hit set + poses with decoder ray casting from the estimated poses.
R8 unions are fused two ways: max (occupied wins, = the current stream_targets rule) and log-odds sum (what a real system would do)."""
import json, math, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a07_lib import (trace, integrate, observe, rotz, RAYS, ctx_bin, ctx_ter, gz, down_up, score, mean_rows, POSE_BYTES)

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, 'sim_cache')
# name: (sigma0 m, k (sigma = k r), p_drop, p_out, pose sigma_t m, pose sigma_yaw deg, frame sigma_t m, frame sigma_yaw deg)
CONDS = {
    'clean':    (0.00, 0.000, 0.00, 0.000, 0.00, 0.0, 0.00, 0.0),
    'd2cm':     (0.02, 0.000, 0.00, 0.000, 0.00, 0.0, 0.00, 0.0),
    'dprop1.5': (0.00, 0.015, 0.00, 0.000, 0.00, 0.0, 0.00, 0.0),
    'dprop3':   (0.00, 0.030, 0.00, 0.000, 0.00, 0.0, 0.00, 0.0),
    'pose_s':   (0.00, 0.000, 0.00, 0.000, 0.01, 0.2, 0.00, 0.0),
    'pose_m':   (0.00, 0.000, 0.00, 0.000, 0.03, 1.0, 0.00, 0.0),
    'drop20':   (0.00, 0.000, 0.20, 0.000, 0.00, 0.0, 0.00, 0.0),
    'out1':     (0.00, 0.000, 0.00, 0.010, 0.00, 0.0, 0.00, 0.0),
    'lidar':    (0.02, 0.000, 0.05, 0.000, 0.01, 0.2, 0.03, 0.5),
    'cam':      (0.00, 0.015, 0.10, 0.005, 0.02, 0.5, 0.05, 1.0),
    'harsh':    (0.00, 0.030, 0.20, 0.010, 0.03, 1.0, 0.10, 2.0),
    'cam_nf':   (0.00, 0.015, 0.10, 0.005, 0.02, 0.5, 0.00, 0.0),   # cam without the inter-agent frame error (registration isolated)
    'harsh_nf': (0.00, 0.030, 0.20, 0.010, 0.03, 1.0, 0.00, 0.0),
}


def measure(rng, tr, p, c, frame, centre):
    """one pose: corrupt the true trace and express it in the map frame. frame = (R_a, t_a) per-agent frame error (identity for K0)."""
    s0, k, pdrop, pout, st, syaw, _, _ = c
    hit_r, exit_r = tr
    m = len(RAYS); keep = rng.random(m) >= pdrop
    r = hit_r.copy(); hit = ~np.isnan(r)
    if s0 or k:
        r[hit] = r[hit] + rng.normal(0, 1, hit.sum()) * (s0 + k * r[hit])
    if pout:
        o = hit & (rng.random(m) < pout); r[o] = rng.uniform(0.2, 1., o.sum()) * hit_r[o]
    r[hit] = np.clip(r[hit], 0.05, None)
    dp = np.array([rng.normal(0, st), rng.normal(0, st), rng.normal(0, st / 2)]) if st else np.zeros(3)
    Rp = rotz(rng.normal(0, syaw)) if syaw else np.eye(3)
    Ra, ta = frame
    o_hat = Ra @ (p + dp - centre) + centre + ta
    d_hat = RAYS @ (Ra @ Rp).T
    return (o_hat, d_hat[keep], r[keep], exit_r[keep])


def lossy_raycast(H, cell, poses, f=(2, 2, 1), rule='ge2'):
    X, Y, Z = H.shape
    cnt = H.reshape(X // f[0], f[0], Y // f[1], f[1], Z // f[2], f[2]).sum((1, 3, 5))
    c = (cnt >= (2 if rule == 'ge2' else 1)).astype(np.uint8)
    u = c.repeat(f[0], 0).repeat(f[1], 1).repeat(f[2], 2)
    return np.stack([observe(u, cell, pp) for pp in poses]).max(0), ctx_bin(c)


def despeckle(m):
    """classical denoiser: an occupied voxel with no occupied voxel among its 26 neighbours becomes free."""
    from scipy import ndimage
    occ = (m == 1); nb = ndimage.convolve(occ.astype(np.int16), np.ones((3, 3, 3), np.int16), mode='constant') - occ
    out = m.copy(); out[occ & (nb == 0)] = 0
    return out


def house(args):
    name, conds = args
    t0 = time.time()
    z = np.load(os.path.join(CACHE, name + '.npz'))
    own, poses, cell, R8 = z['own'], z['poses'], z['cell'], z['reached8']
    gt = dict(occ=z['gt_occ'], slab=z['gt_slab'], obj=z['gt_obj']); occ = z['gt_occ'].astype(np.uint8); n = own.shape[0]; shape = own.shape[1:]
    T8 = np.where(R8[:, :, None, None, None], own[None], np.int8(-1)).max(1)
    comps = {}
    for a in range(n):
        comps.setdefault(tuple(np.nonzero(R8[a])[0]), []).append(a)
    traces = [[trace(occ, cell, poses[a, j])[:2] for j in range(3)] for a in range(n)]
    ds = 0.5 * float(np.min(cell))
    out = dict(house=name, conds={})
    for cname in conds:
        c = CONDS[cname]; seed = 1000 * sum(map(ord, name)) + sum(map(ord, cname))
        rng = np.random.default_rng(seed)
        M, Mest_poses, counts = [], [], []
        if cname == 'clean':                                                 # = the v420 maps exactly (verified in a07_lib / p1)
            M = own.copy(); U_max = T8.copy(); U_sum = T8.copy(); Mest_poses = list(poses); u_poses = {a: poses[a] for a in range(n)}
        else:
            for a in range(n):                                               # K0: per-pose noise only
                centre = poses[a].mean(0)
                meas = [measure(rng, traces[a][j], poses[a, j], c, (np.eye(3), np.zeros(3)), centre) for j in range(3)]
                m, nh, nm = integrate(shape, cell, ds, meas); M.append(m); Mest_poses.append(np.stack([mm[0] for mm in meas]))
            M = np.stack(M)
            U_max = np.zeros_like(own); U_sum = np.zeros_like(own); u_poses = {}
            for a in range(n):                                               # R8: per-pose noise + per-agent frame error
                centre = poses[a].mean(0)
                Ra = rotz(rng.normal(0, c[7])) if c[7] else np.eye(3)
                ta = np.array([rng.normal(0, c[6]), rng.normal(0, c[6]), 0.]) if c[6] else np.zeros(3)
                meas = [measure(rng, traces[a][j], poses[a, j], c, (Ra, ta), centre) for j in range(3)]
                m, nh, nm = integrate(shape, cell, ds, meas); counts.append((m, nh, nm)); u_poses[a] = np.stack([mm[0] for mm in meas])
            for members, agents in comps.items():
                um = np.stack([counts[b][0] for b in members]).max(0)
                nh = sum(counts[b][1] for b in members); nm = sum(counts[b][2] for b in members); L = 0.85 * nh - 0.4 * nm
                us = np.where((nh + nm) == 0, -1, np.where(L > 0, 1, 0)).astype(np.int8)
                for a in agents:
                    U_max[a] = um; U_sum[a] = us
        rec = dict()
        rec['K0_rates'] = [dict(ctx=ctx_ter(M[a]), H_ctx=ctx_bin(M[a] == 1), gzip=gz(M[a]), known=float((M[a] >= 0).mean())) for a in range(n)]
        rec['K0_floor'] = score(M, own, gt)
        rec['K0_unknown_known_fpr'] = [float(((M[a] >= 0) & (own[a] < 0)).sum() / max((own[a] < 0).sum(), 1)) for a in range(n)]
        # classical denoising codecs on the noisy map (scored vs the clean target)
        pd, bd = [], []
        for a in range(n):
            cc, uu = down_up(M[a], (2, 2, 1), .5); pd.append(uu); bd.append(ctx_ter(cc))
        rec['K0_downup221'] = dict(bytes=bd, rows=score(np.stack(pd), own, gt))
        pf, bf = [], []
        for a in range(n):                                                   # classical denoise-then-lossless: drop isolated hits
            f = despeckle(M[a]); pf.append(f); bf.append(ctx_ter(f))
        rec['K0_despeckle_lossless'] = dict(bytes=bf, rows=score(np.stack(pf), own, gt))
        for tag, U in (('max', U_max), ('sum', U_sum)):
            rec[f'R8_{tag}_rates'] = [dict(ctx=ctx_ter(U[a]), H_ctx=ctx_bin(U[a] == 1), gzip=gz(U[a])) for a in range(n)]
            rec[f'R8_{tag}_floor'] = score(U, T8, gt)
        pf, bf = [], []
        for a in range(n):
            f = despeckle(U_sum[a]); pf.append(f); bf.append(ctx_ter(f))
        rec['R8_despeckle_lossless'] = dict(bytes=bf, rows=score(np.stack(pf), T8, gt))
        pd, bd = [], []
        for a in range(n):
            cc, uu = down_up(U_sum[a], (2, 2, 1), .5); pd.append(uu); bd.append(ctx_ter(cc))
        rec['R8_downup221'] = dict(bytes=bd, rows=score(np.stack(pd), T8, gt))
        out['conds'][cname] = rec
        print(name, cname, f'{time.time() - t0:.0f}s', flush=True)
    return out


def summarise(res, conds):
    S = {}
    for cname in conds:
        def rows(key):
            return [r for h in res for r in h['conds'][cname][key]]
        def med(key, field):
            return float(np.median([r[field] for r in rows(key)]))
        s = dict(K0_ctx=med('K0_rates', 'ctx'), K0_Hposes=med('K0_rates', 'H_ctx') + POSE_BYTES, K0_gzip=med('K0_rates', 'gzip'),
                 K0_floor={k: mean_rows(rows('K0_floor'), k) for k in ('surface_f1', 'var_occ_f1', 'furn_f1')},
                 K0_fpr=float(np.mean(rows('K0_unknown_known_fpr'))),
                 R8max_ctx=med('R8_max_rates', 'ctx'), R8sum_ctx=med('R8_sum_rates', 'ctx'), R8sum_Hctx=med('R8_sum_rates', 'H_ctx'), R8sum_gzip=med('R8_sum_rates', 'gzip'),
                 R8max_floor={k: mean_rows(rows('R8_max_floor'), k) for k in ('surface_f1', 'var_occ_f1', 'furn_f1')},
                 R8sum_floor={k: mean_rows(rows('R8_sum_floor'), k) for k in ('surface_f1', 'var_occ_f1', 'furn_f1')})
        for key in ('K0_downup221', 'K0_despeckle_lossless', 'R8_despeckle_lossless', 'R8_downup221'):
            rr = [r for h in res for r in h['conds'][cname][key]['rows']]
            s[key] = dict(bytes=float(np.median([b for h in res for b in h['conds'][cname][key]['bytes']])),
                          **{k: mean_rows(rr, k) for k in ('surface_f1', 'var_occ_f1', 'furn_f1')})
        S[cname] = s
    return S


if __name__ == '__main__':
    conds = sys.argv[1].split(',') if len(sys.argv) > 1 else list(CONDS)
    names = sorted(f[:-4] for f in os.listdir(CACHE) if f.endswith('.npz'))
    groups = [conds]
    with Pool(6) as pool:
        parts = pool.map(house, [(nm, g) for nm in names for g in groups])
    res = []
    for nm in names:
        merged = dict(house=nm, conds={})
        for p_ in parts:
            if p_['house'] == nm:
                merged['conds'].update(p_['conds'])
        res.append(merged)
    S = summarise(res, conds)
    tag = sys.argv[2] if len(sys.argv) > 2 else 'all'
    json.dump(dict(summary=S, houses=res, conds={k: CONDS[k] for k in conds}), open(os.path.join(HERE, f'p3_noise_{tag}.json'), 'w'), indent=1)
    for cname, s in S.items():
        f0, f8 = s['K0_floor'], s['R8sum_floor']
        print(f"{cname:9s} K0 ctx {s['K0_ctx']:6.0f} H+p {s['K0_Hposes']:6.0f} gz {s['K0_gzip']:6.0f} floor s/v/f {f0['surface_f1']:.3f}/{f0['var_occ_f1']:.3f}/{f0['furn_f1']:.3f} fpr {s['K0_fpr']:.4f} | "
              f"du221 {s['K0_downup221']['bytes']:5.0f} {s['K0_downup221']['surface_f1']:.3f}/{s['K0_downup221']['furn_f1']:.3f} desp {s['K0_despeckle_lossless']['bytes']:5.0f} {s['K0_despeckle_lossless']['surface_f1']:.3f}/{s['K0_despeckle_lossless']['furn_f1']:.3f}")
        print(f"{'':9s} R8 max ctx {s['R8max_ctx']:6.0f} floor {s['R8max_floor']['surface_f1']:.3f}/{s['R8max_floor']['furn_f1']:.3f} | sum ctx {s['R8sum_ctx']:6.0f} H {s['R8sum_Hctx']:6.0f} gz {s['R8sum_gzip']:6.0f} floor {f8['surface_f1']:.3f}/{f8['var_occ_f1']:.3f}/{f8['furn_f1']:.3f} | "
              f"du221 {s['R8_downup221']['bytes']:5.0f} {s['R8_downup221']['surface_f1']:.3f}/{s['R8_downup221']['furn_f1']:.3f} desp {s['R8_despeckle_lossless']['bytes']:5.0f} {s['R8_despeckle_lossless']['surface_f1']:.3f}/{s['R8_despeckle_lossless']['furn_f1']:.3f}")

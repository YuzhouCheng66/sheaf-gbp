"""c1_noise_seed2.py (C1, 2026-10-27): a SECOND noise seed for the round-3 noise claims, plus a decomposition of which error hurts.
Re-implements probes/A10/a10d_noise_mult.py (A10; logic copied, which itself copies A07's p3_noise tracer / measure / log-odds integrator)
with a different RNG stream (+424242), reading A07's re-simulated demo houses (probes/A07/sim_cache, read only; clean own maps there are
exactly observe(gt, poses)). Nothing is imported from or written into other agents' folders.
Conditions (sigma0, k, p_drop, p_out, pose_t, pose_yaw_deg, frame_t, frame_yaw_deg):
  cam_nf     = A07 'cam' without the inter-agent frame error;  cam = + frame error 5 cm / 1 deg;
  range_only = cam sensor noise (1.5 % r, 10 % drop, 0.5 % outliers), NO pose error;  pose_only = cam per-pose error 2 cm / 0.5 deg only.
Per condition: furniture-region occupancy F1 of R8 fusions (SUM of log-odds = Bayes for independent noise, exact on the tree; MAX =
occupied wins = the current stream_targets join; SINGLE = one clean observer per voxel), pooled and by observation multiplicity m;
precision by m; free recall of the fusions; K0 furniture F1 / free recall of each agent's own noisy map vs its clean own map (only for
conditions without the frame error, i.e. K0 in the agent's frame). Target = clean maps (perfect sensor at the true poses).
CPU only, one process, numpy."""
import glob, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = r'C:/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill/claude_handoff/analysis_20261027/probes/A07/sim_cache'
SENSOR_RANGE = 5.0
SEED_SHIFT = 424242


def dilate6(m):
    o = m.copy()
    o[1:] |= m[:-1]; o[:-1] |= m[1:]
    o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
    o[:, :, 1:] |= m[:, :, :-1]; o[:, :, :-1] |= m[:, :, 1:]
    return o


def fib(n=65536):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); th = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)), 1)


RAYS = fib()
CONDS = {'cam_nf': (0.00, 0.015, 0.10, 0.005, 0.02, 0.5, 0.00, 0.0), 'cam': (0.00, 0.015, 0.10, 0.005, 0.02, 0.5, 0.05, 1.0),
         'range_only': (0.00, 0.015, 0.10, 0.005, 0.00, 0.0, 0.00, 0.0), 'pose_only': (0.00, 0.000, 0.00, 0.000, 0.02, 0.5, 0.00, 0.0)}


def trace(occ, cell, p, rays=RAYS):
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float64); p = np.asarray(p, np.float64)
    ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    hit_r = np.full(len(rays), np.nan); exit_r = np.full(len(rays), (nsteps + 0.5) * ds)
    alive = np.ones(len(rays), bool)
    for s in range(1, nsteps + 1):
        ai = np.nonzero(alive)[0]
        if not len(ai):
            break
        idx = np.floor((p[None] + rays[ai] * (s * ds)) / cell).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape), 1)
        out_i = ai[~inside]; exit_r[out_i] = (s - 0.5) * ds; alive[out_i] = False
        ai2 = ai[inside]; idx = idx[inside]
        hit = occ[idx[:, 0], idx[:, 1], idx[:, 2]] == 1
        hit_r[ai2[hit]] = s * ds; alive[ai2[hit]] = False
    return hit_r, exit_r, ds


def rotz(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def measure(rng, tr, p, c, frame, centre):
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


def integrate(shape, cell, ds, meas):
    V = int(np.prod(shape)); shape_a = np.array(shape); cell = np.asarray(cell, np.float64)
    nh = np.zeros(V, np.int32); nm = np.zeros(V, np.int32)
    for o, d, rng, fext in meas:
        hit = ~np.isnan(rng)
        reach = np.where(hit, rng, fext)
        end_idx = np.full(len(d), -1, np.int64)
        if hit.any():
            e = np.floor((o[None] + d[hit] * rng[hit, None]) / cell).astype(np.int64)
            ok = np.all((e >= 0) & (e < shape_a), 1)
            ef = np.ravel_multi_index(e[ok].T, shape)
            hi = np.nonzero(hit)[0][ok]; end_idx[hi] = ef
            nh += np.bincount(ef, minlength=V).astype(np.int32)
        nsteps = int(np.ceil(np.nanmax(reach) / ds)) if len(reach) else 0
        last = np.full(len(d), -1, np.int64); acc = []
        for s in range(1, nsteps + 1):
            act = np.nonzero(s * ds < reach - 1e-9)[0]
            if not len(act):
                break
            idx = np.floor((o[None] + d[act] * (s * ds)) / cell).astype(np.int64)
            inside = np.all((idx >= 0) & (idx < shape_a), 1)
            act = act[inside]; idx = idx[inside]
            f = (idx[:, 0] * shape[1] + idx[:, 1]) * shape[2] + idx[:, 2]
            new = (f != last[act]) & (f != end_idx[act])
            acc.append(f[new])
            last[act] = f
        if acc:
            nm += np.bincount(np.concatenate(acc), minlength=V).astype(np.int32)
    return nh.reshape(shape), nm.reshape(shape)


def tern(nh, nm):
    L = 0.85 * nh.astype(np.float64) - 0.4 * nm
    return np.where((nh + nm) == 0, -1, np.where(L > 0, 1, 0)).astype(np.int8)


def prf(pred_occ, tgt_occ, scope):
    return int((pred_occ & tgt_occ & scope).sum()), int((pred_occ & ~tgt_occ & scope).sum()), int((~pred_occ & tgt_occ & scope).sum())


f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)
STRATA = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 99)]
out = dict(seed_shift=SEED_SHIFT, conds={})
t0 = time.time()
for cname, c in CONDS.items():
    acc = {f: [[0, 0, 0] for _ in STRATA] for f in ('sum', 'max', 'single')}
    tot = {f: [0, 0, 0] for f in ('sum', 'max', 'single')}
    free = {f: [0, 0] for f in ('sum', 'max')}
    k0 = [0, 0, 0]; k0free = [0, 0]
    for path in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        name = os.path.basename(path)[:-4]
        z = np.load(path)
        own, poses, cell = z['own'], z['poses'], z['cell']; occ = z['gt_occ'].astype(np.uint8); n = own.shape[0]; shape = own.shape[1:]
        slab = z['gt_slab']; struct = z['gt_occ'] & ~z['gt_obj'] & ~z['gt_slab']; region = ~slab & ~dilate6(struct)
        assert z['reached8'].all()
        rng = np.random.default_rng(1000 * sum(map(ord, name)) + sum(map(ord, cname)) + 7 + SEED_SHIFT)
        ds = 0.5 * float(np.min(cell))
        NH, NM = [], []
        for a in range(n):
            centre = poses[a].mean(0)
            Ra = rotz(rng.normal(0, c[7])) if c[7] else np.eye(3)
            ta = np.array([rng.normal(0, c[6]), rng.normal(0, c[6]), 0.]) if c[6] else np.zeros(3)
            meas = [measure(rng, trace(occ, cell, poses[a, j])[:2], poses[a, j], c, (Ra, ta), centre) for j in range(3)]
            nh, nm = integrate(shape, cell, ds, meas); NH.append(nh); NM.append(nm)
        T = own.max(0); known = T >= 0; Tocc = T == 1; Tfree = T == 0
        mult = (own >= 0).sum(0)
        noisy = np.stack([tern(NH[a], NM[a]) for a in range(n)])
        U = dict(sum=tern(sum(NH), sum(NM)), max=noisy.max(0))
        first = np.argmax(own >= 0, axis=0)
        U['single'] = np.take_along_axis(noisy, first[None], 0)[0]
        for s, (lo, hi) in enumerate(STRATA):
            sm = known & (mult >= lo) & (mult <= hi)
            for f in acc:
                r = prf(U[f] == 1, Tocc, sm & region); acc[f][s] = [x + y for x, y in zip(acc[f][s], r)]
        for f in tot:
            r = prf(U[f] == 1, Tocc, known & region); tot[f] = [x + y for x, y in zip(tot[f], r)]
        for f in free:
            free[f][0] += int((Tfree & (U[f] == 0)).sum()); free[f][1] += int(Tfree.sum())
        if c[6] == 0 and c[7] == 0:                                  # K0 in the agent's own (= public) frame
            for a in range(n):
                kn = own[a] >= 0
                r = prf(noisy[a] == 1, own[a] == 1, kn & region); k0 = [x + y for x, y in zip(k0, r)]
                k0free[0] += int(((own[a] == 0) & (noisy[a] == 0)).sum()); k0free[1] += int((own[a] == 0).sum())
        print(cname, name, f'{time.time() - t0:.0f}s', flush=True)
    rec = dict(furn_f1_pooled={f: f1(tot[f]) for f in tot},
               furn_f1_by_m={f: [f1(acc[f][s]) for s in range(len(STRATA))] for f in acc},
               furn_prec_by_m={f: [acc[f][s][0] / max(acc[f][s][0] + acc[f][s][1], 1) for s in range(len(STRATA))] for f in acc},
               free_recall={f: free[f][0] / max(free[f][1], 1) for f in free},
               K0_furn_f1=(f1(k0) if (c[6] == 0 and c[7] == 0) else None), K0_free_recall=(k0free[0] / max(k0free[1], 1) if k0free[1] else None))
    out['conds'][cname] = rec
    print(cname, json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(HERE, 'c1_noise_seed2.json'), 'w'), indent=1)

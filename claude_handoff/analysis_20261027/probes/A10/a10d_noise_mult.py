"""a10d_noise_mult.py (A10, debate round, 2026-10-27): under sensor noise, is multi-agent REDUNDANCY a source of fusion PRECISION
(not only of wasted rate)? CPU only, numpy only, one process.
Data: A07's re-simulated demo houses (probes/A07/sim_cache/*.npz, read-only): clean own maps that are exactly observe(gt, poses) and the
poses. The noise model, tracer and log-odds integrator are COPIED from probes/A07/a07_lib.py + p3_noise.py (A07; logic unchanged) so
that nothing is imported from (or written into) A07's directory. Condition 'cam_nf' (range sigma = 1.5 % r, 10 % dropped beams, 0.5 %
early outliers, per-pose 2 cm / 0.5 deg error, NO inter-agent frame error) and 'cam' (+ per-agent frame error 5 cm / 1 deg).
Per house: every agent's noisy hit / miss counts; fusions over the reached set: SUM of log-odds (Bayes for independent noise, exact on a
tree), MAX (occupied wins = the current stream_targets / join rule), and SINGLE (each voxel read from one agent that saw it cleanly,
the lowest index). Target = the CLEAN union (what perfect sensors at the true poses saw). Scores are stratified by the voxel's
observation multiplicity m = number of agents whose clean map knows it: occupancy F1 in the furniture region and in the variable
(non-slab) region, target-known scope. Rate side: redundancy factor = sum of own-map ctx / ctx of the fused union, clean vs noisy."""
import glob, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import tern_bytes, dilate6  # noqa

CACHE = os.path.abspath(os.path.join(HERE, '..', 'A07', 'sim_cache'))
SENSOR_RANGE = 5.0


def fib(n=65536):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); th = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)), 1)


RAYS = fib()
CONDS = {'cam_nf': (0.00, 0.015, 0.10, 0.005, 0.02, 0.5, 0.00, 0.0), 'cam': (0.00, 0.015, 0.10, 0.005, 0.02, 0.5, 0.05, 1.0)}


def trace(occ, cell, p, rays=RAYS):                                   # = A07 a07_lib.trace
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


def measure(rng, tr, p, c, frame, centre):                            # = A07 p3_noise.measure
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


def integrate(shape, cell, ds, meas):                                  # = A07 a07_lib.integrate (counts only)
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
    tp = int((pred_occ & tgt_occ & scope).sum()); fp = int((pred_occ & ~tgt_occ & scope).sum()); fn = int((~pred_occ & tgt_occ & scope).sum())
    return tp, fp, fn


STRATA = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 99)]
out = dict(conds={}, houses=[])
t0 = time.time()
for cname, c in CONDS.items():
    acc = {f: {s: [0, 0, 0] for s in range(len(STRATA))} for f in ('sum', 'max', 'single')}
    accv = {f: {s: [0, 0, 0] for s in range(len(STRATA))} for f in ('sum', 'max', 'single')}
    tot = {f: [0, 0, 0] for f in ('sum', 'max', 'single')}
    vox_share = np.zeros(len(STRATA)); rates = []
    for path in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        name = os.path.basename(path)[:-4]
        z = np.load(path)
        own, poses, cell = z['own'], z['poses'], z['cell']; occ = z['gt_occ'].astype(np.uint8); n = own.shape[0]; shape = own.shape[1:]
        slab = z['gt_slab']; struct = z['gt_occ'] & ~z['gt_obj'] & ~z['gt_slab']; region = ~slab & ~dilate6(struct)
        R8 = z['reached8']; assert R8.all(), 'demo houses are single trees'
        rng = np.random.default_rng(1000 * sum(map(ord, name)) + sum(map(ord, cname)) + 7)
        ds = 0.5 * float(np.min(cell))
        NH, NM = [], []
        for a in range(n):
            centre = poses[a].mean(0)
            Ra = rotz(rng.normal(0, c[7])) if c[7] else np.eye(3)
            ta = np.array([rng.normal(0, c[6]), rng.normal(0, c[6]), 0.]) if c[6] else np.zeros(3)
            meas = [measure(rng, trace(occ, cell, poses[a, j])[:2], poses[a, j], c, (Ra, ta), centre) for j in range(3)]
            nh, nm = integrate(shape, cell, ds, meas); NH.append(nh); NM.append(nm)
        T = own.max(0); known = T >= 0; Tocc = T == 1
        mult = (own >= 0).sum(0)
        noisy = np.stack([tern(NH[a], NM[a]) for a in range(n)])
        Usum = tern(sum(NH), sum(NM))
        Umax = noisy.max(0)
        first = np.argmax(own >= 0, axis=0)                                  # lowest-index agent that saw the voxel cleanly
        Usingle = np.take_along_axis(noisy, first[None], 0)[0]
        for s, (lo, hi) in enumerate(STRATA):
            sm = known & (mult >= lo) & (mult <= hi)
            vox_share[s] += int((sm & Tocc & region).sum())
            for f, U in (('sum', Usum), ('max', Umax), ('single', Usingle)):
                r = prf(U == 1, Tocc, sm & region); acc[f][s] = [x + y for x, y in zip(acc[f][s], r)]
                r = prf(U == 1, Tocc, sm & ~slab); accv[f][s] = [x + y for x, y in zip(accv[f][s], r)]
        for f, U in (('sum', Usum), ('max', Umax), ('single', Usingle)):
            r = prf(U == 1, Tocc, known & region); tot[f] = [x + y for x, y in zip(tot[f], r)]
        own_clean = sum(tern_bytes(own[a]) for a in range(n)); own_noisy = sum(tern_bytes(noisy[a]) for a in range(n))
        rates.append(dict(house=name, own_clean_sum=own_clean, union_clean=tern_bytes(T), own_noisy_sum=own_noisy, union_noisy_sum=tern_bytes(Usum)))
        print(cname, name, f'{time.time() - t0:.0f}s', flush=True)
    f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)
    rec = dict(strata=[f'{lo}' if lo == hi else f'>={lo}' for lo, hi in STRATA],
               furn_occ_share=(vox_share / vox_share.sum()).tolist(),
               furn_f1={f: [f1(acc[f][s]) for s in range(len(STRATA))] for f in acc},
               var_f1={f: [f1(accv[f][s]) for s in range(len(STRATA))] for f in accv},
               furn_f1_pooled={f: f1(tot[f]) for f in tot},
               furn_prec_rec={f: [(acc[f][s][0] / max(acc[f][s][0] + acc[f][s][1], 1), acc[f][s][0] / max(acc[f][s][0] + acc[f][s][2], 1)) for s in range(len(STRATA))] for f in acc},
               redundancy_clean=float(np.mean([r['own_clean_sum'] / r['union_clean'] for r in rates])),
               redundancy_noisy=float(np.mean([r['own_noisy_sum'] / r['union_noisy_sum'] for r in rates])),
               union_ctx_clean=float(np.median([r['union_clean'] for r in rates])), union_ctx_noisy=float(np.median([r['union_noisy_sum'] for r in rates])),
               rates=rates)
    out['conds'][cname] = rec
    print(cname, json.dumps({k: v for k, v in rec.items() if k != 'rates'}), flush=True)
json.dump(out, open(os.path.join(HERE, 'a10d_noise_mult.json'), 'w'), indent=1)

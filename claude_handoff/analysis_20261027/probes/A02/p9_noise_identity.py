"""A02 debate probe P9 (2026-10-27, CPU only, one process): does the RENDER IDENTITY that nearly every round-1 design rests on
(own map = ray-cast(observed occupied voxels H, poses); join of facts is conflict-free) survive range noise?
3 demo VAL houses (r1_i0, r2_i0, r3_i1) x agents 0-3, 3 jittered poses per agent (P1 RNG), 65,536 beams, the generator's marching rule.
Per beam: the TRUE first-hit sample s_hit is traced in the true world; the measured range d = s_hit * ds + noise; the beam then integrates
'miss' on every voxel it samples before d (once per voxel) and 'hit' on the voxel containing the endpoint. Two integration rules:
  occwins : occupied if any hit, free if only misses (the clean v420 rule);
  logodds : OctoMap-style L = 0.85 hits - 0.40 misses, occupied if L > 0, free if observed and L <= 0 (A07 p3's parameters).
Noise: clean (check: must equal observe()), range sigma 1 cm, 2 cm, and 1.5 % of range (no dropouts, outliers or pose error: range noise only).
Reported: (a) identity residual = voxels where ray-cast(H_noisy, true poses) differs from the noisy own map (fraction of its known voxels,
and of its occupied voxels); (b) inter-agent conflicts in the 4-agent union = voxels some agent calls free and another calls occupied,
as a fraction of the union's occupied voxels (amax = 'occupied wins' turns every conflict into an occupied voxel).
Output: probes/A02/p9_noise_identity.json"""
import json, math, os, sys, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from rate_baselines_lib import load  # noqa: E402
from p1_render_hits import observe, fibonacci_sphere  # noqa: E402

SENSOR_RANGE = 5.0
RAYS = fibonacci_sphere(65536)


def trace(occ, cell, p):
    """true first-hit sample index per beam (0 = no hit) and first sample index that leaves the grid (nsteps + 1 if never)"""
    shape = np.array(occ.shape); ds = 0.5 * float(np.min(cell)); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    m = len(RAYS); s_hit = np.zeros(m, np.int64); s_exit = np.full(m, nsteps + 1, np.int64); alive = np.ones(m, bool)
    for s in range(1, nsteps + 1):
        idx = np.floor((p[None] + RAYS * (s * ds)) / cell).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape), 1)
        newly_out = alive & ~inside; s_exit[newly_out] = s; alive &= inside
        ai = np.nonzero(alive)[0]
        if len(ai) == 0:
            break
        hit = occ[idx[ai, 0], idx[ai, 1], idx[ai, 2]] == 1
        s_hit[ai[hit]] = s; alive[ai[hit]] = False
    return s_hit, s_exit, ds, nsteps


def integrate(shape, cell, p, s_hit, s_exit, ds, nsteps, d_meas, hits, misses):
    """accumulate per-voxel hit / miss counts of one pose; d_meas: measured range per beam (nan = no return)"""
    shape_a = np.array(shape); m = len(RAYS)
    has = ~np.isnan(d_meas)
    limit = np.where(has, d_meas, np.minimum(s_exit, nsteps + 1) * ds)       # samples strictly before this distance are misses
    prev = np.full((m, 3), -1, np.int64)
    for s in range(1, nsteps + 1):
        act = (s * ds < limit) & (s < s_exit)
        if not act.any():
            break
        ai = np.nonzero(act)[0]
        idx = np.floor((p[None] + RAYS[ai] * (s * ds)) / cell).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < shape_a), 1)
        new = ok & np.any(idx != prev[ai], 1)                               # one miss per voxel per beam (straight ray: contiguous runs)
        prev[ai] = idx
        v = idx[new]; np.add.at(misses, (v[:, 0], v[:, 1], v[:, 2]), 1)
    hi = np.nonzero(has)[0]
    e = np.floor((p[None] + RAYS[hi] * d_meas[hi][:, None]) / cell).astype(np.int64)
    ok = np.all((e >= 0) & (e < shape_a), 1); e = e[ok]
    np.add.at(hits, (e[:, 0], e[:, 1], e[:, 2]), 1)


def state(hits, misses, rule):
    out = np.full(hits.shape, -1, np.int8)
    if rule == 'occwins':
        out[misses > 0] = 0; out[hits > 0] = 1
    else:
        L = 0.85 * hits - 0.40 * misses; obs = (hits + misses) > 0
        out[obs] = 0; out[obs & (L > 0)] = 1
    return out


SETTINGS = [('clean', 0.0, 0.0), ('d1cm', 0.01, 0.0), ('d2cm', 0.02, 0.0), ('r1.5pct', 0.0, 0.015)]
RULES = ['occwins', 'logodds']

if __name__ == '__main__':
    data = os.path.join(ROOT, 'demo_web3d', 'data')
    rows, conflicts = [], []
    t0 = time.time()
    for h in ('r1_i0', 'r2_i0', 'r3_i1'):
        m, own, pred, gt = load(os.path.join(data, f'house_{h}.json')); del pred
        occ = gt['occ'].astype(np.uint8); cell = np.asarray(m['cell_m'], np.float64); shape = occ.shape
        maps = {(sn, rule): [] for sn, _, _ in SETTINGS for rule in RULES}
        for a in range(4):
            rng = np.random.default_rng(zlib.crc32(m['house'].encode()) * 16 + a)
            centre = np.asarray(m['positions_m'][a], np.float64); jit = rng.uniform(-.1, .1, (3, 3)); jit[:, 2] = 0.
            poses = centre[None] + jit
            tr = [trace(occ, cell, p) for p in poses]
            clean_ref = observe(occ, cell, poses, RAYS)
            for sn, sig, k in SETTINGS:
                nrng = np.random.default_rng(1000 + 17 * a + len(sn))
                counts = {rule: None for rule in RULES}
                hits = np.zeros(shape, np.int32); misses = np.zeros(shape, np.int32)
                for p, (s_hit, s_exit, ds, nsteps) in zip(poses, tr):
                    d = np.where(s_hit > 0, s_hit * ds, np.nan)
                    hm = ~np.isnan(d)
                    if sig or k:
                        d[hm] = np.clip(d[hm] + nrng.normal(0, 1, hm.sum()) * (sig + k * d[hm]), 0.05, None)
                    integrate(shape, cell, p, s_hit, s_exit, ds, nsteps, d, hits, misses)
                for rule in RULES:
                    M = state(hits, misses, rule)
                    if sn == 'clean':
                        assert (M == clean_ref).all(), (h, a, rule, int((M != clean_ref).sum()))
                    H = (M == 1)
                    R = observe(H.astype(np.uint8), cell, poses, RAYS)
                    diff = R != M; known = M >= 0
                    rows.append(dict(house=h, agent=a, setting=sn, rule=rule, known=int(known.sum()), occ=int(H.sum()),
                                     mismatch=int(diff.sum()), mismatch_frac_known=float(diff.sum() / max(known.sum(), 1)),
                                     mismatch_frac_occ=float(diff.sum() / max(H.sum(), 1)),
                                     known_lost=int((known & (R < 0)).sum()), known_extra=int((~known & (R >= 0)).sum()),
                                     occ_vs_truth_fp=int((H & (occ == 0)).sum()), truth_surface_missed=int(((clean_ref == 1) & ~H).sum())))
                    maps[(sn, rule)].append(M)
                    print(f"{h} a{a} {sn:8s} {rule:8s} known {known.sum():6d} occ {H.sum():5d} identity-mismatch {diff.sum():6d} "
                          f"({100 * diff.sum() / max(known.sum(), 1):.2f} % of known, {100 * diff.sum() / max(H.sum(), 1):.1f} % of occ) {time.time() - t0:.0f}s", flush=True)
        for (sn, rule), Ms in maps.items():
            A = np.stack(Ms); anyfree = (A == 0).any(0); anyocc = (A == 1).any(0)
            U = A.max(0); conf = anyfree & anyocc
            conflicts.append(dict(house=h, setting=sn, rule=rule, union_occ=int((U == 1).sum()), conflicts=int(conf.sum()),
                                  conflict_frac_of_union_occ=float(conf.sum() / max((U == 1).sum(), 1))))
            print(f"{h} union {sn:8s} {rule:8s} conflicts {conf.sum():6d} = {100 * conf.sum() / max((U == 1).sum(), 1):.1f} % of union occupied", flush=True)
        del maps
    S = {}
    for sn, _, _ in SETTINGS:
        for rule in RULES:
            rr = [r for r in rows if r['setting'] == sn and r['rule'] == rule]; cc = [c for c in conflicts if c['setting'] == sn and c['rule'] == rule]
            S[f'{sn}/{rule}'] = dict(identity_mismatch_frac_known_median=float(np.median([r['mismatch_frac_known'] for r in rr])),
                                     identity_mismatch_frac_occ_median=float(np.median([r['mismatch_frac_occ'] for r in rr])),
                                     identity_mismatch_voxels_median=float(np.median([r['mismatch'] for r in rr])),
                                     fp_vs_truth_median=float(np.median([r['occ_vs_truth_fp'] for r in rr])),
                                     union_conflict_frac_median=float(np.median([c['conflict_frac_of_union_occ'] for c in cc])))
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, rows=rows, conflicts=conflicts), open(os.path.join(HERE, 'p9_noise_identity.json'), 'w'), indent=1)
